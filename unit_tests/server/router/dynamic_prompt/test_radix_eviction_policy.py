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
        self.age = 0.0
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


def test_priority_uses_frozen_age():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.global_age = 999.0  # current clock must NOT leak into the score
    node = _FakeNode(node_prefix_total_len=8, node_value_len=8)  # cost 64, size 8
    node.freq = 2
    node.age = 10.0  # frozen baseline
    # age + freq * cost / size = 10 + 2 * 64 / 8 = 26
    assert policy.compute_priority(node) == pytest.approx(26.0)


def test_init_node_anchors_age_to_current_clock():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.global_age = 5.0
    node = _FakeNode(node_prefix_total_len=4, node_value_len=4)
    policy.init_node(node)
    assert node.freq == 0  # admission is not reuse
    assert node.age == pytest.approx(5.0)  # anchored at the aging floor
    assert node.priority == pytest.approx(5.0)  # 5 + 0


def test_record_reuse_increments_freq_and_reanchors_age():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.global_age = 5.0
    node = _FakeNode(node_prefix_total_len=4, node_value_len=4)  # cost 16, size 4
    policy.record_reuse(node)
    assert node.freq == 1
    assert node.age == pytest.approx(5.0)
    assert node.priority == pytest.approx(9.0)  # 5 + 1 * 16 / 4


def test_touch_node_preserves_age_freq_and_recency():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.global_age = 7.0
    node = _FakeNode(node_prefix_total_len=4, node_value_len=4, time_id=42)  # cost 16, size 4
    node.age = 3.0  # a frozen baseline from an earlier event
    node.freq = 2
    policy.touch_node(node)
    assert node.freq == 2  # bookkeeping never grows frequency
    assert node.age == pytest.approx(3.0)  # frozen age is NOT re-anchored to 7.0
    assert node.time_id == 42  # a non-reuse touch does NOT refresh recency
    assert node.priority == pytest.approx(3.0 + 2 * 16 / 4)  # rescored from geometry


def test_record_reuse_refreshes_recency():
    policy = ComputeAwareGDSFEvictionPolicy()
    node = _FakeNode(node_prefix_total_len=4, node_value_len=4, time_id=42)
    policy.record_reuse(node)
    assert node.time_id == 43  # a genuine reuse DOES refresh recency


def test_on_split_rescored_both_halves_from_new_geometry():
    # Pre-split node had span [0, 10), age 2, freq 3. Split at 4 into prefix [0, 4)
    # and suffix [4, 10). Both inherit age/freq; each is rescored from its own span.
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.global_age = 100.0  # must not leak into either half
    prefix = _FakeNode(node_prefix_total_len=4, node_value_len=4)  # cost 16, size 4
    prefix.age, prefix.freq = 2.0, 3
    suffix = _FakeNode(node_prefix_total_len=10, node_value_len=6)  # cost 84, size 6
    suffix.age, suffix.freq = 2.0, 3
    policy.on_split(prefix, suffix)
    assert prefix.priority == pytest.approx(2.0 + 3 * 16 / 4)  # 14, not stale, not aged
    assert suffix.priority == pytest.approx(2.0 + 3 * 84 / 6)  # 44


def test_on_evict_advances_age_monotonically():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.on_evict(_FakeNode(node_prefix_total_len=4, node_value_len=4))  # priority 0 default
    assert policy.global_age == 0.0
    hot = _FakeNode(node_prefix_total_len=4, node_value_len=4)
    hot.priority = 50.0
    policy.on_evict(hot)
    assert policy.global_age == 50.0
    cold = _FakeNode(node_prefix_total_len=4, node_value_len=4)
    cold.priority = 5.0
    policy.on_evict(cold)
    assert policy.global_age == 50.0  # a lower-priority victim does not roll it back


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
    # size clamps to 1, cost is 0 -> priority == frozen age (0 here)
    assert policy.compute_priority(root_like) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# integration helpers
# --------------------------------------------------------------------------------------
def _insert(tree, ids):
    return tree.insert(torch.tensor(ids, dtype=torch.int64, device="cpu"))


def _reuse(tree, ids, update_refs=True):
    """Simulate a prompt-cache lookup; with update_refs it is a genuine hit."""
    node, _, _ = tree.match_prefix(torch.tensor(ids, dtype=torch.int64, device="cpu"), update_refs=update_refs)
    if node is not None and update_refs:
        tree.dec_node_ref_counter(node)
    return node


# --------------------------------------------------------------------------------------
# frequency counts genuine reuse ONLY (full and partial hits)
# --------------------------------------------------------------------------------------
def test_insertion_does_not_count_as_reuse():
    tree = RadixCache("gdsf_insert", 100, 0, eviction_policy="gdsf_compute")
    _, node = _insert(tree, [0, 1, 2])
    assert node.freq == 0
    _insert(tree, [0, 1, 2, 3, 4])  # walks the shared prefix but must not bump freq
    assert tree.root_node.children[0].freq == 0


def test_readonly_probe_does_not_count_as_reuse():
    tree = RadixCache("gdsf_probe", 100, 0, eviction_policy="gdsf_compute")
    _, node = _insert(tree, [0, 1, 2])
    tree.match_prefix(torch.tensor([0, 1, 2], dtype=torch.int64, device="cpu"), update_refs=False)
    assert node.freq == 0


def test_ref_bookkeeping_does_not_count_as_reuse():
    tree = RadixCache("gdsf_refbook", 100, 0, eviction_policy="gdsf_compute")
    _, node = _insert(tree, [0, 1, 2])
    tree.add_node_ref_counter(node)
    tree.dec_node_ref_counter(node)
    assert node.freq == 0


def test_full_hit_counts_as_reuse():
    tree = RadixCache("gdsf_full", 100, 0, eviction_policy="gdsf_compute")
    _, node = _insert(tree, [0, 1, 2])
    assert node.freq == 0
    matched = _reuse(tree, [0, 1, 2])
    assert matched is node
    assert node.freq == 1
    _reuse(tree, [0, 1, 2])
    assert node.freq == 2  # each genuine hit increments exactly once


def test_partial_hit_counts_as_reuse():
    # A request matching a proper prefix of a resident node splits it; the matched
    # shared-prefix half is a genuine reuse and must be counted.
    tree = RadixCache("gdsf_partial", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [1, 2, 3, 4, 5])  # single resident node [1,2,3,4,5]
    node = _reuse(tree, [1, 2, 3])  # partial hit -> split into [1,2,3] + [4,5]

    shared_prefix = tree.root_node.children[1]
    assert torch.equal(shared_prefix.token_id_key, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert node is shared_prefix
    assert shared_prefix.freq == 1  # partial hit counted
    # the unmatched suffix was NOT reused
    suffix = shared_prefix.children[4]
    assert torch.equal(suffix.token_id_key, torch.tensor([4, 5], dtype=torch.int64))
    assert suffix.freq == 0


def test_partial_probe_does_not_count_as_reuse():
    tree = RadixCache("gdsf_partial_probe", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [1, 2, 3, 4, 5])
    _reuse(tree, [1, 2, 3], update_refs=False)  # read-only partial probe -> splits
    shared_prefix = tree.root_node.children[1]
    assert torch.equal(shared_prefix.token_id_key, torch.tensor([1, 2, 3], dtype=torch.int64))
    assert shared_prefix.freq == 0  # a probe, even a partial one, is not reuse


# --------------------------------------------------------------------------------------
# non-reuse touches must not improve a node's eviction standing (recency)
# --------------------------------------------------------------------------------------
def test_readonly_probe_does_not_refresh_eviction_order():
    # A and B are equal-priority (freq 0), so the recency tiebreak decides. A read-only
    # probe of the older node A must NOT make it look newer than B.
    tree = RadixCache("gdsf_probe_order", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [0, 1, 2])  # A: older
    _insert(tree, [10, 11, 12])  # B: newer

    _reuse(tree, [0, 1, 2], update_refs=False)  # non-reuse probe of A
    tree.evict(3, lambda mem: None)
    assert 0 not in tree.root_node.children  # A still evicted first (probe did not refresh)
    assert 10 in tree.root_node.children


def test_writeback_does_not_refresh_eviction_order():
    # Re-inserting an already-resident node (KV write-back) must not refresh recency.
    tree = RadixCache("gdsf_wb_order", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [0, 1, 2])  # A: older
    _insert(tree, [10, 11, 12])  # B: newer

    _insert(tree, [0, 1, 2])  # write-back of A -- not a reuse
    tree.evict(3, lambda mem: None)
    assert 0 not in tree.root_node.children  # A still evicted first
    assert 10 in tree.root_node.children


def test_genuine_reuse_does_refresh_eviction_order():
    # Contrast: a genuine reuse of the older node A raises its standing so the
    # never-reused newer node B is evicted first instead.
    tree = RadixCache("gdsf_reuse_order", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [0, 1, 2])  # A: older
    _insert(tree, [10, 11, 12])  # B: newer

    _reuse(tree, [0, 1, 2])  # genuine reuse of A -> priority rises above B's
    tree.evict(3, lambda mem: None)
    assert 0 in tree.root_node.children  # A kept
    assert 10 not in tree.root_node.children  # B evicted


# --------------------------------------------------------------------------------------
# split rescoring: both halves reflect their new geometry, age stays frozen
# --------------------------------------------------------------------------------------
def test_split_rescored_both_halves_preserving_frozen_age():
    tree = RadixCache("gdsf_split_rescore", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8])  # resident node spanning [0, 8)
    # Reuse the whole node once so it has real history (freq=1, age=global_age=0).
    whole = _reuse(tree, [1, 2, 3, 4, 5, 6, 7, 8])
    assert whole.freq == 1
    stale_priority = whole.priority  # cost 64, size 8 -> 0 + 1*8 = 8
    assert stale_priority == pytest.approx(8.0)

    # Now a read-only partial probe splits it at 3 into [1,2,3) and [3..8).
    _reuse(tree, [1, 2, 3], update_refs=False)
    prefix = tree.root_node.children[1]  # span [0, 3): cost 9, size 3
    suffix = prefix.children[4]  # span [3, 8): cost 64-9=55, size 5

    # Both inherit freq=1, age=0, and are rescored from their OWN geometry -- not
    # left holding the pre-split node's stale priority of 8.
    assert prefix.freq == 1 and suffix.freq == 1
    assert prefix.priority == pytest.approx(0.0 + 1 * 9 / 3)  # 3.0
    assert suffix.priority == pytest.approx(0.0 + 1 * 55 / 5)  # 11.0
    assert prefix.priority != pytest.approx(stale_priority)
    assert suffix.priority != pytest.approx(stale_priority)


# --------------------------------------------------------------------------------------
# integration: LRU vs GDSF, driven by genuine reuse
# --------------------------------------------------------------------------------------
def test_default_policy_is_lru():
    tree = RadixCache("gdsf_default", 100, 0)
    assert isinstance(tree.eviction_policy, LRUEvictionPolicy)


def test_lru_evicts_oldest_expensive_branch():
    tree = RadixCache("lru_case", 100, 0, eviction_policy="lru")
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # branch B: 8 tokens, older
    _insert(tree, [0, 1, 2])  # branch A: 3 tokens, newer
    assert tree.get_tree_total_tokens_num() == 11

    tree.evict(3, lambda mem: None)
    assert 10 not in tree.root_node.children  # oldest (expensive) evicted
    assert 0 in tree.root_node.children
    assert tree.get_tree_total_tokens_num() == 3


def test_gdsf_keeps_expensive_branch_evicts_cheap():
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
    tree = RadixCache("gdsf_freq", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [0, 1, 2])  # A: per-hit cost/size = 9/3 = 3
    _insert(tree, [10, 11, 12, 13, 14])  # B: per-hit cost/size = 25/5 = 5

    _reuse(tree, [10, 11, 12, 13, 14])  # freq_B = 1 -> priority 5
    for _ in range(3):
        _reuse(tree, [0, 1, 2])  # freq_A = 3 -> priority 9

    tree.evict(3, lambda mem: None)
    assert 0 in tree.root_node.children  # frequently reused cheap prefix survives
    assert 10 not in tree.root_node.children


def test_gdsf_never_reused_nodes_fall_back_to_recency():
    tree = RadixCache("gdsf_norefuse", 100, 0, eviction_policy="gdsf_compute")
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # older, expensive
    _insert(tree, [0, 1, 2])  # newer, cheap
    assert tree.root_node.children[10].freq == 0
    assert tree.root_node.children[0].freq == 0

    tree.evict(3, lambda mem: None)
    assert 10 not in tree.root_node.children  # oldest evicted despite higher cost
    assert 0 in tree.root_node.children


def test_gdsf_existing_radix_semantics_preserved():
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
