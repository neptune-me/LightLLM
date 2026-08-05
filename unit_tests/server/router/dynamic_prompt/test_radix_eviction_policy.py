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


def test_update_node_increments_freq_and_sets_priority():
    policy = ComputeAwareGDSFEvictionPolicy()
    node = _FakeNode(node_prefix_total_len=4, node_value_len=4)  # cost 16, size 4
    policy.update_node(node)
    assert node.freq == 1
    assert node.priority == pytest.approx(4.0)  # 0 + 1 * 16 / 4
    policy.update_node(node)
    assert node.freq == 2
    assert node.priority == pytest.approx(8.0)  # frequency reuse raises priority


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
# integration: LRU vs GDSF diverge on the same access pattern
# --------------------------------------------------------------------------------------
def _insert(tree, ids):
    tree.insert(torch.tensor(ids, dtype=torch.int64, device="cpu"))


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
    # Same access pattern as the LRU test, but compute-aware GDSF makes the
    # opposite (better) choice: keep the branch that is expensive to recompute.
    tree = RadixCache("gdsf_case", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # branch B: cost 64/8 = 8
    _insert(tree, [0, 1, 2])  # branch A: cost 9/3 = 3 -> lower priority
    assert tree.get_tree_total_tokens_num() == 11

    tree.evict(3, lambda mem: None)
    # GDSF drops the CHEAP branch A, keeps the expensive branch B.
    assert 0 not in tree.root_node.children
    assert 10 in tree.root_node.children
    assert tree.get_tree_total_tokens_num() == 8


def test_gdsf_frequency_protects_reused_prefix():
    # A short/cheap prefix that is reused many times should outlive a longer prefix
    # that is only touched once, once frequency dominates the cheap per-token cost.
    tree = RadixCache("gdsf_freq", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [0, 1, 2])  # cheap branch A, per-hit cost/size = 3
    _insert(tree, [10, 11, 12, 13, 14])  # branch B, cost 25/5 = 5

    # Reuse A repeatedly: priority_A = age + freq*3 climbs above priority_B = 5.
    for _ in range(3):
        tree.match_prefix(torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"), update_refs=False)

    tree.evict(3, lambda mem: None)
    # Frequently reused cheap prefix A survives; the cold branch B is evicted.
    assert 0 in tree.root_node.children
    assert 10 not in tree.root_node.children


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
