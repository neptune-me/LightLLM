"""Unit tests for the pluggable radix-cache eviction policies.

These test the policy objects in isolation using real ``TreeNode`` instances,
independent of the full ``RadixCache`` (shared-memory) machinery.
"""
import torch
import pytest

from lightllm.server.router.dynamic_prompt.radix_cache import TreeNode
from lightllm.server.router.dynamic_prompt.radix_eviction_policy import (
    LRUEvictionPolicy,
    GDSFEvictionPolicy,
    create_eviction_policy,
)


def _make_node(size: int, ref: int = 0) -> TreeNode:
    node = TreeNode()
    node.token_id_key = torch.zeros((size,), dtype=torch.int64)
    node.token_mem_index_value = torch.zeros((size,), dtype=torch.int64)
    node.node_value_len = size
    node.ref_counter = ref
    return node


# --------------------------------------------------------------------------- #
# LRU policy: must be byte-for-byte compatible with the legacy get_compare_key #
# --------------------------------------------------------------------------- #
def test_lru_compare_key_matches_legacy():
    policy = LRUEvictionPolicy()
    node = _make_node(3)
    assert policy.compare_key(node) == node.get_compare_key()


def test_lru_touch_bumps_time_id_only():
    policy = LRUEvictionPolicy()
    node = _make_node(3)
    old_time = node.time_id
    policy.touch(node)
    assert node.time_id > old_time


# --------------------------------------------------------------------------- #
# GDSF policy                                                                  #
# --------------------------------------------------------------------------- #
def test_gdsf_new_node_initial_priority():
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    node = _make_node(4)
    policy.on_new_node(node)
    # new nodes start at 0 (insertion does not count as reuse)
    assert node._gdsf_freq == 0
    assert policy.priority(node) == pytest.approx(0.0)


def test_gdsf_access_increments_frequency_and_priority():
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    node = _make_node(4)
    policy.on_new_node(node)
    policy.on_access(node)  # genuine reuse: freq -> 1
    assert node._gdsf_freq == 1
    assert policy.priority(node) == pytest.approx(0.25)


def test_gdsf_touch_does_not_change_frequency():
    """touch() is a recency/structural signal (inserts, probes) and must NOT
    inflate the frequency used for size-aware LFU."""
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    node = _make_node(4)
    policy.on_new_node(node)
    freq_before = node._gdsf_freq
    prio_before = policy.priority(node)
    for _ in range(5):
        policy.touch(node)
    assert node._gdsf_freq == freq_before
    assert policy.priority(node) == pytest.approx(prio_before)


def test_gdsf_is_size_aware_larger_node_lower_priority():
    """Equal frequency: the larger node should be evicted first (lower key)."""
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    small = _make_node(2)
    large = _make_node(8)
    policy.on_new_node(small)
    policy.on_new_node(large)
    policy.on_access(small)  # freq 1 each -> equal freq, size decides
    policy.on_access(large)
    assert policy.compare_key(large) < policy.compare_key(small)


def test_gdsf_is_frequency_aware_less_used_lower_priority():
    """Equal size: the less-frequently-used node should be evicted first."""
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    hot = _make_node(3)
    cold = _make_node(3)
    policy.on_new_node(hot)
    policy.on_new_node(cold)
    for _ in range(5):
        policy.on_access(hot)
    assert policy.compare_key(cold) < policy.compare_key(hot)


def test_gdsf_size_exponent_zero_is_pure_lfu():
    """size_exponent=0 removes size-awareness -> equal freq => equal priority term."""
    policy = GDSFEvictionPolicy(size_exponent=0.0)
    small = _make_node(2)
    large = _make_node(8)
    policy.on_new_node(small)
    policy.on_new_node(large)
    assert policy.priority(small) == pytest.approx(policy.priority(large))


def test_gdsf_referenced_nodes_sort_after_unreferenced():
    policy = GDSFEvictionPolicy()
    refed = _make_node(2, ref=1)
    unrefed = _make_node(100)  # huge, tiny priority, but still evictable-first
    policy.on_new_node(refed)
    policy.on_new_node(unrefed)
    # referenced node must never be chosen before an unreferenced one
    assert policy.compare_key(unrefed) < policy.compare_key(refed)


def test_gdsf_on_evict_raises_aging_clock():
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    node = _make_node(4)
    policy.on_new_node(node)
    p = policy.priority(node)
    policy.on_evict(node)
    assert policy.aging_clock == pytest.approx(p)


def test_gdsf_aging_prevents_immediate_reeviction_of_new_node():
    """A node created after an eviction inherits the raised clock, so it is not
    immediately the lowest-priority (oldest-looking) node."""
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    evicted = _make_node(1)
    policy.on_new_node(evicted)
    policy.on_access(evicted)  # freq 1
    policy.on_access(evicted)  # freq 2
    high_priority = policy.priority(evicted)
    policy.on_evict(evicted)  # aging_clock jumps to high_priority
    fresh = _make_node(1)
    policy.on_new_node(fresh)
    policy.on_access(fresh)  # give fresh one reuse so its priority is clock+1/size
    assert policy.priority(fresh) > high_priority


def test_gdsf_access_refreshes_clock_but_touch_does_not():
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    victim = _make_node(1)
    policy.on_new_node(victim)
    policy.on_access(victim)  # need freq to make priority >0
    policy.on_evict(victim)  # raise aging_clock above 0
    assert policy.aging_clock > 0.0

    node = _make_node(4)
    policy.on_new_node(node)
    node._gdsf_clock = 0.0  # pretend it was created long ago
    policy.touch(node)
    assert node._gdsf_clock == 0.0  # touch must not refresh the aging clock
    policy.on_access(node)
    assert node._gdsf_clock == policy.aging_clock  # genuine access refreshes it


def test_gdsf_merge_does_not_overstate_combined_frequency():
    """Merging an ancestor into its child yields one node spanning both. It was
    fully reused only as often as the deeper node was reached, so the combined
    frequency must be the min (deeper node's count), not the max (which would
    count shorter parent-only hits as full-span reuses)."""
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    parent = _make_node(2)
    child = _make_node(3)
    policy.on_new_node(parent)
    policy.on_new_node(child)
    # parent (ancestor) is reached at least as often as the child
    for _ in range(3):
        policy.on_access(parent)  # parent freq -> 3
    policy.on_access(child)  # child freq -> 1

    policy.on_merge(parent, child)  # child survives as the merged node
    assert child._gdsf_freq == 1  # deeper-node count, not max(3, 1)


def test_gdsf_merge_uses_deeper_node_clock():
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    parent = _make_node(2)
    child = _make_node(3)
    policy.on_new_node(parent)
    policy.on_new_node(child)
    child._gdsf_clock = 1.0
    parent._gdsf_clock = 5.0  # parent touched more recently (parent-only access)

    policy.on_merge(parent, child)
    assert child._gdsf_clock == 1.0  # full-span recency, not the parent-only 5.0


def test_gdsf_split_inherits_history():
    """A split node carved from an existing node inherits its freq/clock."""
    policy = GDSFEvictionPolicy()
    original = _make_node(6)
    policy.on_new_node(original)
    policy.on_access(original)
    policy.on_access(original)
    head = _make_node(2)
    policy.on_new_node(head, split_from=original)
    assert head._gdsf_freq == original._gdsf_freq
    assert head._gdsf_clock == original._gdsf_clock


# --------------------------------------------------------------------------- #
# factory                                                                      #
# --------------------------------------------------------------------------- #
def test_factory_defaults_to_lru():
    assert isinstance(create_eviction_policy("lru"), LRUEvictionPolicy)


def test_factory_builds_gdsf_with_kwargs():
    policy = create_eviction_policy("gdsf", size_exponent=0.5)
    assert isinstance(policy, GDSFEvictionPolicy)
    assert policy.size_exponent == 0.5


def test_factory_rejects_unknown_policy():
    with pytest.raises(ValueError):
        create_eviction_policy("nope")


def test_gdsf_has_no_cost_knob():
    """cost was a global scalar that could not change eviction ordering (it
    rescales every priority uniformly), so the knob was removed."""
    with pytest.raises(TypeError):
        GDSFEvictionPolicy(cost=1.0)
    assert not hasattr(GDSFEvictionPolicy(), "cost")


def test_gdsf_ordering_independent_of_would_be_cost_scaling():
    """Guard the property behind the removal: priority ordering depends only on
    freq / size ** e, so any uniform positive rescale leaves argmin unchanged."""
    policy = GDSFEvictionPolicy(size_exponent=1.0)
    a = _make_node(2)  # freq 1 -> 0.5
    b = _make_node(5)  # freq 1 -> 0.2
    policy.on_new_node(a)
    policy.on_new_node(b)
    policy.on_access(a)  # freq 1
    policy.on_access(b)  # freq 1
    keys = {"a": policy.compare_key(a), "b": policy.compare_key(b)}
    # b (larger, lower density) must be the victim regardless of any scalar
    assert min(keys, key=keys.get) == "b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
