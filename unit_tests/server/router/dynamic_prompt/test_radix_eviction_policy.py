import pytest
import torch

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.dynamic_prompt.radix_eviction_policy import (
    EVICTION_POLICY_CHOICES,
    ComputeAwareGDSFEvictionPolicy,
    LRUEvictionPolicy,
    get_eviction_policy,
)


class _FakeNode:
    """Minimal stand-in for TreeNode for pure-policy unit tests."""

    def __init__(self, node_prefix_total_len, node_value_len, ref_counter=0, time_id=0):
        self.node_prefix_total_len = node_prefix_total_len
        self.node_value_len = node_value_len
        self.ref_counter = ref_counter
        self.time_id = time_id
        self.children = {}
        self.freq = 0
        self.priority = 0.0

    def update_time(self):
        self.time_id += 1


# --------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------
def test_factory_resolves_names():
    assert isinstance(get_eviction_policy("lru"), LRUEvictionPolicy)
    assert isinstance(get_eviction_policy("gdsf_compute"), ComputeAwareGDSFEvictionPolicy)
    assert isinstance(get_eviction_policy(None), LRUEvictionPolicy)
    assert set(EVICTION_POLICY_CHOICES) == {"lru", "gdsf_compute"}


def test_factory_passthrough_instance():
    policy = ComputeAwareGDSFEvictionPolicy()
    assert get_eviction_policy(policy) is policy


def test_factory_rejects_unknown():
    with pytest.raises(ValueError):
        get_eviction_policy("does_not_exist")


# --------------------------------------------------------------------------------------
# GDSF scoring
# --------------------------------------------------------------------------------------
def test_recompute_cost_is_quadratic_segment():
    policy = ComputeAwareGDSFEvictionPolicy()
    # A node covering prefix span [2, 5): cost = 5**2 - 2**2 = 21
    node = _FakeNode(node_prefix_total_len=5, node_value_len=3)
    assert policy._recompute_cost(node) == pytest.approx(25 - 4)


def test_priority_formula():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.global_age = 10.0
    node = _FakeNode(node_prefix_total_len=8, node_value_len=8)  # cost 64, size 8
    node.freq = 2
    # global_age + freq * cost / size = 10 + 2 * 64 / 8 = 26
    assert policy.compute_priority(node) == pytest.approx(26.0)


def test_record_reuse_increments_freq_and_sets_priority():
    policy = ComputeAwareGDSFEvictionPolicy()
    node = _FakeNode(node_prefix_total_len=4, node_value_len=4)  # cost 16, size 4
    policy.record_reuse(node)
    assert node.freq == 1
    assert node.priority == pytest.approx(4.0)  # 0 + 1 * 16 / 4
    policy.record_reuse(node)
    assert node.freq == 2
    assert node.priority == pytest.approx(8.0)  # frequency reuse raises priority


def test_touch_node_does_not_increment_freq():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.global_age = 7.0
    node = _FakeNode(node_prefix_total_len=4, node_value_len=4)  # cost 16, size 4
    policy.touch_node(node)
    assert node.freq == 0
    # With freq still 0 the compute term vanishes: priority is just the aging floor.
    assert node.priority == pytest.approx(7.0)
    policy.touch_node(node)
    assert node.freq == 0  # repeated bookkeeping never inflates frequency


def test_on_evict_advances_age_monotonically():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.on_evict(_FakeNode(node_prefix_total_len=4, node_value_len=4))  # priority 0 default
    assert policy.global_age == 0.0
    hot = _FakeNode(node_prefix_total_len=4, node_value_len=4)
    hot.priority = 50.0
    policy.on_evict(hot)
    assert policy.global_age == 50.0
    # a later, lower-priority victim must not roll the clock back
    cold = _FakeNode(node_prefix_total_len=4, node_value_len=4)
    cold.priority = 5.0
    policy.on_evict(cold)
    assert policy.global_age == 50.0


def test_compare_key_orders_unreferenced_and_cheap_first():
    policy = ComputeAwareGDSFEvictionPolicy()
    cheap = _FakeNode(node_prefix_total_len=3, node_value_len=3)
    cheap.priority = 3.0
    expensive = _FakeNode(node_prefix_total_len=8, node_value_len=8)
    expensive.priority = 8.0
    referenced = _FakeNode(node_prefix_total_len=3, node_value_len=3, ref_counter=1)
    referenced.priority = 1.0  # low priority, but referenced -> must sort last
    ordered = sorted([expensive, referenced, cheap], key=policy.get_compare_key)
    assert ordered == [cheap, expensive, referenced]


def test_zero_length_node_does_not_divide_by_zero():
    policy = ComputeAwareGDSFEvictionPolicy()
    root_like = _FakeNode(node_prefix_total_len=0, node_value_len=0)
    # size clamps to 1, cost is 0 -> priority == global_age
    assert policy.compute_priority(root_like) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# integration: frequency counts genuine reuse ONLY
# --------------------------------------------------------------------------------------
def _insert(tree, ids):
    return tree.insert(torch.tensor(ids, dtype=torch.int64, device="cpu"))


def _reuse(tree, ids):
    """Simulate a genuine prompt-cache hit: match with refs, then release them."""
    node, _, _ = tree.match_prefix(torch.tensor(ids, dtype=torch.int64, device="cpu"), update_refs=True)
    if node is not None:
        tree.dec_node_ref_counter(node)
    return node


def test_insertion_does_not_count_as_reuse():
    tree = RadixCache("gdsf_insert", 100, 0, eviction_policy="gdsf_compute")
    _, node = _insert(tree, [0, 1, 2])
    assert node.freq == 0  # admitting a node is not reuse
    # Re-inserting an extending sequence walks the shared prefix but must not bump freq.
    _insert(tree, [0, 1, 2, 3, 4])
    assert tree.root_node.children[0].freq == 0


def test_readonly_probe_does_not_count_as_reuse():
    tree = RadixCache("gdsf_probe", 100, 0, eviction_policy="gdsf_compute")
    _, node = _insert(tree, [0, 1, 2])
    tree.match_prefix(torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"), update_refs=False)
    assert node.freq == 0  # a read-only probe is not reuse


def test_ref_bookkeeping_does_not_count_as_reuse():
    tree = RadixCache("gdsf_refbook", 100, 0, eviction_policy="gdsf_compute")
    _, node = _insert(tree, [0, 1, 2])
    tree.add_node_ref_counter(node)
    tree.dec_node_ref_counter(node)
    assert node.freq == 0  # pure reference-counter churn is not reuse


def test_match_with_update_refs_counts_as_reuse():
    tree = RadixCache("gdsf_reuse", 100, 0, eviction_policy="gdsf_compute")
    _, node = _insert(tree, [0, 1, 2])
    assert node.freq == 0

    matched, _, _ = tree.match_prefix(torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"), update_refs=True)
    assert matched is node
    assert node.freq == 1
    tree.dec_node_ref_counter(node)

    tree.match_prefix(torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"), update_refs=True)
    assert node.freq == 2  # each genuine hit increments exactly once
    tree.dec_node_ref_counter(node)


# --------------------------------------------------------------------------------------
# integration: LRU vs GDSF diverge, driven by genuine reuse
# --------------------------------------------------------------------------------------
def test_default_policy_is_lru():
    tree = RadixCache("gdsf_default", 100, 0)
    assert isinstance(tree.eviction_policy, LRUEvictionPolicy)


def test_lru_evicts_oldest_expensive_branch():
    # Insert the long/expensive branch FIRST (oldest), the short/cheap one SECOND.
    tree = RadixCache("lru_case", 100, 0, eviction_policy="lru")
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # branch B: 8 tokens, older
    _insert(tree, [0, 1, 2])  # branch A: 3 tokens, newer
    assert tree.get_tree_total_tokens_num() == 11

    tree.evict(3, lambda mem: None)
    # LRU drops the OLDEST leaf (the expensive branch B), keeping the newer cheap A.
    assert 10 not in tree.root_node.children
    assert 0 in tree.root_node.children
    assert tree.get_tree_total_tokens_num() == 3


def test_gdsf_keeps_expensive_branch_evicts_cheap():
    # Both branches are genuinely reused once; compute-aware GDSF then keeps the
    # branch that is expensive to recompute and drops the cheap one.
    tree = RadixCache("gdsf_case", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # branch B: cost 64/8 = 8
    _insert(tree, [0, 1, 2])  # branch A: cost 9/3 = 3

    _reuse(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # freq_B = 1 -> priority 8
    _reuse(tree, [0, 1, 2])  # freq_A = 1 -> priority 3
    assert tree.get_tree_total_tokens_num() == 11

    tree.evict(3, lambda mem: None)
    assert 0 not in tree.root_node.children  # cheap branch A evicted
    assert 10 in tree.root_node.children  # expensive branch B kept
    assert tree.get_tree_total_tokens_num() == 8


def test_gdsf_frequency_protects_reused_prefix():
    # A cheap prefix reused many times outranks a costlier prefix reused once.
    tree = RadixCache("gdsf_freq", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [0, 1, 2])  # A: per-hit cost/size = 9/3 = 3
    _insert(tree, [10, 11, 12, 13, 14])  # B: per-hit cost/size = 25/5 = 5

    _reuse(tree, [10, 11, 12, 13, 14])  # freq_B = 1 -> priority 5
    for _ in range(3):
        _reuse(tree, [0, 1, 2])  # freq_A = 3 -> priority 9

    tree.evict(3, lambda mem: None)
    assert 0 in tree.root_node.children  # frequently reused cheap prefix survives
    assert 10 not in tree.root_node.children  # cold-ish costly prefix evicted


def test_gdsf_never_reused_nodes_fall_back_to_recency():
    # With no genuine reuse, every node has freq 0 (priority == aging floor), so
    # GDSF orders purely by recency -- like LRU -- and evicts the oldest.
    tree = RadixCache("gdsf_norefuse", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # older, expensive
    _insert(tree, [0, 1, 2])  # newer, cheap
    assert tree.root_node.children[10].freq == 0
    assert tree.root_node.children[0].freq == 0

    tree.evict(3, lambda mem: None)
    assert 10 not in tree.root_node.children  # oldest evicted despite higher cost
    assert 0 in tree.root_node.children


def test_gdsf_existing_radix_semantics_preserved():
    # The structural cache behavior (prefix matching, token accounting, full evict)
    # must be identical under GDSF; only eviction ORDER changes.
    tree = RadixCache("gdsf_semantics", 100, 0, eviction_policy="gdsf_compute")
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int64))
    assert ans == 0
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64))
    assert ans == 5
    ans, _ = tree.insert(torch.tensor([0, 1, 2, 3, 4, 7, 8, 9], dtype=torch.int64))
    assert ans == 8
    assert tree.get_refed_tokens_num() == 0
    assert tree.get_tree_total_tokens_num() == 13

    tree.evict(13, lambda mem: None)
    assert tree.get_tree_total_tokens_num() == 0
    assert tree.get_refed_tokens_num() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
