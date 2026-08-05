import pytest
import torch

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.dynamic_prompt.radix_eviction_policy import (
    ComputeAwareGDSFEvictionPolicy,
    LRUEvictionPolicy,
)


def _insert(tree, ids):
    return tree.insert(torch.tensor(ids, dtype=torch.int64, device="cpu"))


def _reuse(tree, ids, update_refs=True):
    node, _, _ = tree.match_prefix(torch.tensor(ids, dtype=torch.int64, device="cpu"), update_refs=update_refs)
    if node is not None and update_refs:
        tree.dec_node_ref_counter(node)
    return node


# ----------------------------------------------------------------------
# Default disabled preserves whole-leaf behavior
# ----------------------------------------------------------------------
def test_default_disabled_is_whole_leaf():
    tree = RadixCache("elastic_default", 100, 0)
    assert tree.enable_elastic_eviction is False
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert tree.get_tree_total_tokens_num() == 10
    tree.evict(3, lambda x: None)
    # Without elastic, whole leaf (10) is evicted even though need is 3
    assert tree.get_tree_total_tokens_num() == 0


def test_elastic_disabled_explicit():
    tree = RadixCache("elastic_disabled", 100, 0, enable_elastic_eviction=False)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tree.evict(3, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 0


# ----------------------------------------------------------------------
# Basic elastic retains prefix and is reusable
# ----------------------------------------------------------------------
def test_elastic_retains_prefix_reusable():
    tree = RadixCache("elastic_basic", 100, 0, enable_elastic_eviction=True, elastic_min_retained_tokens=1)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert tree.get_tree_total_tokens_num() == 10
    evicted = []
    tree.evict(3, lambda x: evicted.append(x.clone()))
    assert tree.get_tree_total_tokens_num() == 7
    assert evicted[0].tolist() == [8, 9, 10]
    # Retained prefix must be matchable
    node, size, val = tree.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int64), update_refs=False)
    assert size == 7
    assert val.tolist() == [1, 2, 3, 4, 5, 6, 7]
    # Invariants
    leaf = tree.root_node.children[1]
    assert leaf.node_value_len == 7
    assert leaf.node_prefix_total_len == 7
    assert leaf.is_leaf()
    assert leaf.ref_counter == 0


def test_elastic_reinsertion_after_truncation():
    tree = RadixCache("elastic_reinsert", 100, 0, enable_elastic_eviction=True)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    tree.evict(3, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 7
    # Reinsert same suffix should extend
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert tree.get_tree_total_tokens_num() == 10
    node, size, val = tree.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.int64), update_refs=False)
    assert size == 10


def test_elastic_multiple_truncations_persistent():
    tree = RadixCache("elastic_multi", 100, 0, enable_elastic_eviction=True, elastic_min_retained_tokens=2)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert tree.get_tree_total_tokens_num() == 12
    tree.evict(3, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 9
    tree.evict(2, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 7
    node, size, val = tree.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int64), update_refs=False)
    assert size == 7
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert tree.get_tree_total_tokens_num() == 12


def test_elastic_with_shared_prefix_tree():
    tree = RadixCache("elastic_shared", 100, 0, enable_elastic_eviction=True)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    _insert(tree, [1, 2, 3, 4, 7, 8, 9])
    # Structure: [1,2,3,4] shared, [5,6,7,8,9,10] len6 and [7,8,9] len3 => total 13
    assert tree.get_tree_total_tokens_num() == 13
    tree.evict(2, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 11
    node, size, val = tree.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64), update_refs=False)
    assert size == 8


def test_elastic_min_retained_prevents_over_trim():
    tree = RadixCache("elastic_min", 100, 0, enable_elastic_eviction=True, elastic_min_retained_tokens=8)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # Need 3, but retaining 7 would violate min 8, so whole-leaf fallback
    tree.evict(3, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 0


def test_elastic_need_equals_or_exceeds_leaf_whole():
    tree = RadixCache("elastic_whole", 100, 0, enable_elastic_eviction=True)
    _insert(tree, [1, 2, 3, 4, 5])
    assert tree.get_tree_total_tokens_num() == 5
    tree.evict(5, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 0
    # Need larger than leaf also whole-leaf
    tree2 = RadixCache("elastic_whole2", 100, 0, enable_elastic_eviction=True)
    _insert(tree2, [1, 2, 3, 4, 5])
    _insert(tree2, [10, 11, 12])
    assert tree2.get_tree_total_tokens_num() == 8
    tree2.evict(5, lambda x: None)
    # One leaf 5 whole evicted, total 3 remains
    assert tree2.get_tree_total_tokens_num() == 3


# ----------------------------------------------------------------------
# Per-leaf cap
# ----------------------------------------------------------------------
def test_per_leaf_cap_limits_single_truncation():
    tree = RadixCache("elastic_cap_single", 100, 0, enable_elastic_eviction=True, elastic_max_evict_tokens_per_leaf=2)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8])
    assert tree.get_tree_total_tokens_num() == 8
    # Need 5 with single leaf, cap 2: elastic would be 2 then need 3 more, but other_available 0,
    # so pointless trim (2+6) is avoided and whole leaf 8 is taken directly with 0 clones.
    evicted = []
    tree.evict(5, lambda x: evicted.append(x.tolist()))
    assert tree.get_tree_total_tokens_num() == 0
    # No elastic due to pointless trim avoidance, whole leaf 8 directly
    assert len(evicted) == 1
    assert len(evicted[0]) == 8


def test_per_leaf_cap_persistent_across_calls():
    tree = RadixCache("elastic_cap_persist", 100, 0, enable_elastic_eviction=True, elastic_max_evict_tokens_per_leaf=2)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8])
    ev1 = []
    tree.evict(2, lambda x: ev1.append(x.tolist()))
    assert ev1[0] == [7, 8]
    assert tree.get_tree_total_tokens_num() == 6
    # Per-step cap: second call on same leaf should again be elastic (cap resets per evict call)
    ev2 = []
    tree.evict(2, lambda x: ev2.append(x.tolist()))
    assert ev2[0] == [5, 6]
    assert tree.get_tree_total_tokens_num() == 4


def test_per_leaf_cap_distributes_across_leaves():
    tree = RadixCache("elastic_cap_dist", 100, 0, enable_elastic_eviction=True, elastic_max_evict_tokens_per_leaf=2)
    _insert(tree, [0, 1, 2, 3, 4, 5, 6, 7])
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])
    assert tree.get_tree_total_tokens_num() == 16
    ev = []
    tree.evict(4, lambda x: ev.append(x.tolist()))
    # Need 4, cap 2 per leaf, so should take 2 from each leaf, not 4 from one leaf
    # With deferred reinsertion, cold leaf truncated 2 is not reselected immediately,
    # so next victim is the next coldest leaf.
    assert tree.get_tree_total_tokens_num() == 12
    # Both leaves should have been truncated by cap 2
    assert all(len(x) == 2 for x in ev)
    assert len(ev) == 2


def test_gdsf_cap_repeated_selection_blocked():
    tree = RadixCache(
        "elastic_gdsf_cap", 100, 0, eviction_policy="gdsf_compute", enable_elastic_eviction=True, elastic_max_evict_tokens_per_leaf=2
    )
    _insert(tree, [0, 1, 2, 3, 4, 5, 6, 7])
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])
    # Make B expensive so A is victim
    _reuse(tree, [10, 11, 12, 13, 14, 15, 16, 17])
    # Need 6, with cap 2 per leaf, single leaf would need 3 elastics to reach 6, but cap prevents
    ev = []
    tree.evict(6, lambda x: ev.append(x.tolist()))
    # Should not have any elastic >2
    assert all(len(x) <= 6 for x in ev)  # whole-leaf may be 6
    # Check that no leaf contributed more than cap elastically
    # First leaf elastic should be exactly 2
    assert len(ev[0]) == 2


# ----------------------------------------------------------------------
# LRU recency preserved (cold stays cold)
# ----------------------------------------------------------------------
def test_elastic_lru_preserves_recency():
    tree = RadixCache("elastic_lru_recency", 100, 0, eviction_policy="lru", enable_elastic_eviction=True)
    _insert(tree, [0, 0, 0, 1])
    _insert(tree, [1, 1, 1, 1])
    _insert(tree, [2, 2, 2, 2])
    a = tree.root_node.children[0]
    b = tree.root_node.children[1]
    c = tree.root_node.children[2]
    assert a.time_id < b.time_id < c.time_id
    old_a_time = a.time_id
    tree.evict(2, lambda x: None)
    # Elastic should NOT refresh time
    assert a.time_id == old_a_time
    assert a.time_id < b.time_id < c.time_id


def test_elastic_lru_subsequent_victim_still_cold():
    tree = RadixCache("elastic_lru_order", 100, 0, eviction_policy="lru", enable_elastic_eviction=True, elastic_max_evict_tokens_per_leaf=10)
    _insert(tree, [0, 0, 0, 1])
    _insert(tree, [1, 1, 1, 1])
    _insert(tree, [2, 2, 2, 2])
    tree.evict(2, lambda x: None)
    # A truncated 4->2, still oldest, total 10
    assert tree.get_tree_total_tokens_num() == 10
    # Next evict should still target A (still oldest)
    ev = []
    tree.evict(1, lambda x: ev.append(x.tolist()))
    assert ev[0] == [0]  # tail of A
    assert 0 in tree.root_node.children


def test_elastic_gdsf_rescores_without_recency():
    tree = RadixCache("elastic_gdsf_rescore", 100, 0, eviction_policy="gdsf_compute", enable_elastic_eviction=True)
    _insert(tree, [0, 1, 2, 3, 4, 5, 6, 7])
    _reuse(tree, [0, 1, 2, 3, 4, 5, 6, 7])
    leaf = tree.root_node.children[0]
    old_time = leaf.time_id
    old_pri = leaf.priority
    assert old_pri == 8.0  # 0 +1*64/8
    tree.evict(4, lambda x: None)
    new_time = leaf.time_id
    new_pri = leaf.priority
    assert new_time == old_time  # preserved
    assert new_pri == 4.0  # 0 +1*16/4
    assert leaf.freq == 1  # not counted as reuse


# ----------------------------------------------------------------------
# Cloning expense: at most one clone (2 tensor clones) per leaf per evict
# ----------------------------------------------------------------------
def test_elastic_cloning_at_most_one_per_leaf():
    tree = RadixCache("elastic_clone", 100, 0, enable_elastic_eviction=True, elastic_max_evict_tokens_per_leaf=10)
    _insert(tree, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    orig_clone = torch.Tensor.clone
    cnt = {"n": 0}

    def counting_clone(self, *a, **kw):
        cnt["n"] += 1
        return orig_clone(self, *a, **kw)

    torch.Tensor.clone = counting_clone
    try:
        cnt["n"] = 0
        tree.evict(3, lambda x: None)
        assert cnt["n"] == 2  # key + value once
        # Need 4 with 2 leaves cap2: should be 2 leaves *2 clones =4, not 6
        # Deferred reinsertion ensures each leaf cloned at most once per evict call.
        # Single-leaf cap case would be pointless trim, so use need=2 (within cap) for single leaf
        tree2 = RadixCache("elastic_clone2", 100, 0, enable_elastic_eviction=True, elastic_max_evict_tokens_per_leaf=2)
        _insert(tree2, [0, 1, 2, 3, 4, 5, 6, 7])
        _insert(tree2, [10, 11, 12, 13, 14, 15, 16, 17])
        cnt["n"] = 0
        tree2.evict(4, lambda x: None)
        assert cnt["n"] == 4
    finally:
        torch.Tensor.clone = orig_clone


# ----------------------------------------------------------------------
# Free path, merge, clear
# ----------------------------------------------------------------------
def test_elastic_free_path():
    from unittest.mock import MagicMock

    tree = RadixCache("elastic_free", 100, 0, enable_elastic_eviction=True)
    _insert(tree, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tree.mem_manager = MagicMock()
    tree.mem_manager.allocator.can_use_mem_size = 5
    freed = []

    def fake_free(x):
        freed.append(x.tolist())

    tree.mem_manager.free = fake_free
    tree.free_radix_cache_to_get_enough_token(need_token_num=8)
    assert tree.get_tree_total_tokens_num() == 7
    assert freed[0] == [7, 8, 9]


def test_elastic_merge_after_truncate():
    tree = RadixCache("elastic_merge", 100, 0, enable_elastic_eviction=True)
    _insert(tree, [1, 2, 3])
    _insert(tree, [1, 2, 3, 4, 5])
    assert tree.get_tree_total_tokens_num() == 5
    tree.evict(1, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 4
    tree.merge_unreferenced_nodes()
    assert len(tree.root_node.children) == 1
    merged = list(tree.root_node.children.values())[0]
    assert merged.token_id_key.tolist() == [1, 2, 3, 4]


def test_elastic_clear():
    tree = RadixCache("elastic_clear", 100, 0, enable_elastic_eviction=True)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8])
    _insert(tree, [1, 2, 3, 4, 9, 10])
    assert tree.get_tree_total_tokens_num() == 10
    tree.clear_tree_nodes()
    assert tree.get_tree_total_tokens_num() == 0
    assert len(tree.root_node.children) == 0


# ----------------------------------------------------------------------
# Invariants after truncation
# ----------------------------------------------------------------------
def test_elastic_invariants():
    tree = RadixCache("elastic_inv", 100, 0, enable_elastic_eviction=True)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    tree.evict(5, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 7

    def check(node):
        for child in node.children.values():
            assert child.node_prefix_total_len == child.parent.node_prefix_total_len + child.node_value_len
            check(child)

    check(tree.root_node)


def test_elastic_refed_not_evicted():
    tree = RadixCache("elastic_ref", 100, 0, enable_elastic_eviction=True)
    _insert(tree, [1, 2, 3, 4, 5, 6, 7, 8])
    matched, _, _ = tree.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64), update_refs=True)
    with pytest.raises(AssertionError):
        tree.evict(2, lambda x: None)
    tree.dec_node_ref_counter(matched)
    tree.evict(2, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 6


def test_elastic_deferred_does_not_expose_referenced():
    # Regression: deferred elastic leaves must not cause a referenced leaf to be
    # selected as victim while an unreferenced deferred leaf still exists.
    tree = RadixCache("elastic_deferred_ref", 100, 0, enable_elastic_eviction=True, elastic_max_evict_tokens_per_leaf=2)
    _insert(tree, [0, 1, 2, 3, 4, 5, 6, 7])
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])
    matched, _, _ = tree.match_prefix(torch.tensor([10, 11, 12, 13, 14, 15, 16, 17], dtype=torch.int64), update_refs=True)
    # A unreferenced (0), B referenced (10)
    assert tree.root_node.children[10].ref_counter == 1
    # Evict 2: should trim A elastically, not expose B
    tree.evict(2, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 14
    assert 0 in tree.root_node.children
    assert 10 in tree.root_node.children
    # Evict 2 again: A still unreferenced with 6 tokens, B still referenced.
    # The deferred logic must not pick B as victim even though A is deferred.
    # It should reinsert A before considering B, so A is still victim (elastic or whole).
    # Since A's per-step cap allows another 2, second evict should again trim A, not B.
    tree.evict(2, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 12
    # B must remain untouched and still referenced
    assert tree.root_node.children[10].ref_counter == 1
    assert len(tree.root_node.children[10].token_mem_index_value) == 8
    tree.dec_node_ref_counter(matched)
    # Now both unreferenced, evict should be able to use B
    tree.evict(2, lambda x: None)
    assert tree.get_tree_total_tokens_num() == 10


# ----------------------------------------------------------------------
# Unsupported cache types
# ----------------------------------------------------------------------
def test_linear_att_unsupported():
    from unittest.mock import MagicMock
    from lightllm.server.router.dynamic_prompt.linear_att_radix_cache import LinearAttPagedRadixCache

    mm = MagicMock()
    mm.linear_att_big_page_buffers = MagicMock()
    with pytest.raises(ValueError, match="not supported"):
        LinearAttPagedRadixCache(
            "lin_elastic",
            100,
            0,
            hash_page_size=512,
            big_page_num=1,
            kv_cache_mem_manager=mm,
            linear_att_small_page_buffers=MagicMock(),
            enable_elastic_eviction=True,
        )


def test_startargs_defaults():
    # Verify defaults via file parsing to avoid atomics import on darwin
    with open("lightllm/server/core/objs/start_args_type.py") as f:
        content = f.read()
    assert "enable_elastic_radix_eviction: bool = field(default=False)" in content
    assert "elastic_radix_min_retained_tokens: int = field(default=1)" in content
    assert "elastic_radix_max_evict_tokens_per_leaf" in content
    # Also verify RadixCache defaults
    tree = RadixCache("elastic_defaults_check", 100, 0)
    assert tree.enable_elastic_eviction is False
    assert tree.elastic_min_retained_tokens == 1
    assert tree.elastic_max_evict_tokens_per_leaf is None


def test_startargs_elastic_enabled():
    # Verify wiring via RadixCache directly (avoids StartArgs atomics import)
    tree = RadixCache(
        "elastic_args",
        100,
        0,
        enable_elastic_eviction=True,
        elastic_min_retained_tokens=4,
        elastic_max_evict_tokens_per_leaf=10,
    )
    assert tree.enable_elastic_eviction is True
    assert tree.elastic_min_retained_tokens == 4
    assert tree.elastic_max_evict_tokens_per_leaf == 10
    # Verify CLI contains flags
    with open("lightllm/server/api_cli.py") as f:
        cli = f.read()
    assert "--enable_elastic_radix_eviction" in cli
    assert "--elastic_radix_min_retained_tokens" in cli
    assert "--elastic_radix_max_evict_tokens_per_leaf" in cli
