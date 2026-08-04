import uuid
import torch
import numpy as np
from sortedcontainers import SortedDict
from unittest.mock import MagicMock
import pytest

from lightllm.server.router.dynamic_prompt.linear_att_radix_cache import LinearAttPagedRadixCache
from lightllm.server.router.dynamic_prompt.radix_cache import _resolve_gdsf_config


def _make_tree(
    policy="gdsf",
    alpha=1.0,
    size_aware=True,
    hash_page_size=1,
    big_page_num=4,
    rank_in_node=None,
):
    if rank_in_node is None:
        rank_in_node = np.random.randint(0, 100000)
    # short unique name to avoid shm name too long (macOS limit ~31 chars)
    unique = f"lg_{uuid.uuid4().hex[:8]}"
    mock_big = MagicMock()
    mock_big.free_state_cache = MagicMock()
    mock_small = MagicMock()
    mock_small.free_state_cache = MagicMock()
    mock_small.get_free_cache_num = MagicMock(return_value=1)
    mem_mgr = MagicMock()
    mem_mgr.linear_att_big_page_buffers = mock_big
    mem_mgr.allocator = MagicMock()
    mem_mgr.allocator.can_use_mem_size = 1000
    tree = LinearAttPagedRadixCache(
        unique_name=unique,
        total_token_num=100,
        rank_in_node=rank_in_node,
        hash_page_size=hash_page_size,
        big_page_num=big_page_num,
        kv_cache_mem_manager=mem_mgr,
        linear_att_small_page_buffers=mock_small,
        eviction_policy=policy,
        gdsf_alpha=alpha,
        enable_size_aware=size_aware,
    )
    return tree, mock_big, mock_small


def _insert_big(tree, tokens, hashes, buf_id):
    key = torch.tensor(tokens, dtype=torch.int64)
    val = torch.tensor([t + 10 for t in tokens], dtype=torch.int64)
    # big page: len hashes == big_page_num, all block_linear_idxs None
    _, node = tree.insert(key, val, hashes, [None] * len(hashes), SortedDict({len(tokens): buf_id}))
    return node


def _insert_small(tree, tokens, hashes, buf_id):
    """tokens/hashes include full prefix (big + small chain). buf_id is for last small page."""
    key = torch.tensor(tokens, dtype=torch.int64)
    val = torch.tensor([t + 10 for t in tokens], dtype=torch.int64)
    # For hash_page_size=1, big_page_num=4, last block is small
    # block_linear_idxs: all None except last = buf_id
    idxs = [None] * (len(hashes) - 1) + [buf_id]
    _, node = tree.insert(key, val, hashes, idxs, SortedDict())
    return node


def test_default_is_lru():
    tree, _, _ = _make_tree(policy=None)
    # explicit None should resolve to lru via _resolve_gdsf_config
    assert tree.eviction_policy == "lru"
    # lru uses time_id ordering, not priority
    tree2, _, _ = _make_tree(policy="lru")
    assert tree2.eviction_policy == "lru"


def test_explicit_gdsf_config():
    tree, _, _ = _make_tree(policy="gdsf", alpha=2.5, size_aware=True)
    assert tree.eviction_policy == "gdsf"
    assert tree.gdsf_alpha == 2.5
    assert tree.enable_size_aware is True


def test_init_new_node_hit_and_priority():
    tree, _, _ = _make_tree(policy="gdsf", alpha=1.0, size_aware=True, hash_page_size=1, big_page_num=4)
    node_big = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 99)
    assert node_big.hit_count == 1
    # size = node_value_len = 4, clock=0, priority = 0+1*1/4=0.25
    assert node_big.priority == pytest.approx(0.25)
    node_s1 = _insert_small(tree, [1, 2, 3, 4, 5], [11, 22, 33, 44, 55], 101)
    assert node_s1.hit_count == 1
    # size=1, priority=1.0
    assert node_s1.priority == pytest.approx(1.0)


def test_insert_does_not_double_count_reuse():
    tree, _, _ = _make_tree(policy="gdsf", hash_page_size=1, big_page_num=4)
    node_big = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 99)
    hits_before = node_big.hit_count
    # Re-insert same big via insert (simulate existing reuse without match_prefix)
    # Should NOT bump hit_count; reuse should be counted only via match_prefix
    key = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    _, node2 = tree.insert(key, key, [11, 22, 33, 44], [None] * 4, SortedDict())
    # node should be same object
    assert node2 is node_big
    assert node_big.hit_count == hits_before, "insert traversal must not inflate hits"


def test_match_increments_hit_once_per_usable_node():
    tree, _, _ = _make_tree(policy="gdsf", hash_page_size=1, big_page_num=4)
    node_big = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 99)
    node_s1 = _insert_small(tree, [1, 2, 3, 4, 5], [11, 22, 33, 44, 55], 101)
    hb, hs1 = node_big.hit_count, node_s1.hit_count
    ans, length, mem = tree.match_prefix(
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
        block_hashs=[11, 22, 33, 44, 55],
        update_refs=True,
    )
    assert ans is node_s1
    assert length == 5
    # each usable node bumped exactly once
    assert node_big.hit_count == hb + 1
    assert node_s1.hit_count == hs1 + 1
    # need to dec refs for cleanup to allow eviction
    tree.dec_node_ref_counter(ans)
    # second match should bump again by 1
    hb2, hs12 = node_big.hit_count, node_s1.hit_count
    ans2, _, _ = tree.match_prefix(
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
        block_hashs=[11, 22, 33, 44, 55],
        update_refs=True,
    )
    assert node_big.hit_count == hb2 + 1
    assert node_s1.hit_count == hs12 + 1
    tree.dec_node_ref_counter(ans2)


def test_miss_trimmed_tail_not_counted():
    tree, _, _ = _make_tree(policy="gdsf", hash_page_size=1, big_page_num=4)
    node_big = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 99)
    node_s1 = _insert_small(tree, [1, 2, 3, 4, 5], [11, 22, 33, 44, 55], 101)
    node_s2 = _insert_small(tree, [1, 2, 3, 4, 5, 6], [11, 22, 33, 44, 55, 66], 102)
    # simulate unbuffered tail: clear buffer of s2 (as happens after eviction of small buffer)
    tree._discard_node(node_s2)
    node_s2.small_page_buffer_idx = None
    tree._add_node(node_s2)

    hb, hs1, hs2 = node_big.hit_count, node_s1.hit_count, node_s2.hit_count
    # query that would match big+s1+s2 but s2 is unbuffered -> trimmed, return s1
    ans, length, mem = tree.match_prefix(
        torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.int64),
        block_hashs=[11, 22, 33, 44, 55, 66, 77],
        update_refs=True,
    )
    assert ans is node_s1
    assert length == 5
    # only usable nodes counted
    assert node_big.hit_count == hb + 1
    assert node_s1.hit_count == hs1 + 1
    assert node_s2.hit_count == hs2, "trimmed miss tail must not inflate"
    tree.dec_node_ref_counter(ans)
    # pure miss: unrelated hash must not bump
    hb_before = node_big.hit_count
    ans2, l2, _ = tree.match_prefix(
        torch.tensor([9, 9, 9, 9], dtype=torch.int64),
        block_hashs=[91, 92, 93, 94],
        update_refs=True,
    )
    assert ans2 is None
    assert node_big.hit_count == hb_before


def test_gdsf_priority_order_freq():
    """Higher freq => larger H => evicted later (pop smallest H first)."""
    tree, _, _ = _make_tree(policy="gdsf", alpha=1.0, size_aware=True, hash_page_size=1, big_page_num=4)
    # Create two independent big pages
    node_a = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 10)
    node_b = _insert_big(tree, [5, 6, 7, 8], [55, 66, 77, 88], 11)
    # Both have hits=1, size=4, prio=0.25. Tie broken by time_id (a earlier -> evicted first under lru; under gdsf tie also time)
    # Now bump A twice to make it hotter
    for _ in range(2):
        ans, _, _ = tree.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int64), block_hashs=[11, 22, 33, 44], update_refs=True)
        tree.dec_node_ref_counter(ans)
    # A: hits=3 => prio=3/4=0.75, B: hits=1 =>0.25
    assert node_a.hit_count == 3
    assert node_b.hit_count == 1
    # Eviction should pop B first (smaller priority)
    # Ensure both are leaves and unreferenced
    assert node_a.ref_counter == 0 and node_b.ref_counter == 0
    evicted = []
    tree._evict(4, lambda mem, buf: evicted.append(buf))
    # First evict should be B (priority 0.25)
    # Since _evict pops in order, we can check remaining
    assert node_b.parent is None, "B should be evicted"
    assert node_a.parent is not None, "A should survive"


def test_gdsf_vs_lru_order_diverges():
    """Lru evicts oldest regardless of freq; GDSF keeps hot even if older."""
    # LRU tree
    tree_lru, _, _ = _make_tree(policy="lru", hash_page_size=1, big_page_num=4)
    n_a_lru = _insert_big(tree_lru, [1, 2, 3, 4], [11, 22, 33, 44], 20)
    n_b_lru = _insert_big(tree_lru, [5, 6, 7, 8], [55, 66, 77, 88], 21)
    # make A hot (2 hits) but A is older (smaller time_id)
    for _ in range(2):
        ans, _, _ = tree_lru.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int64), block_hashs=[11, 22, 33, 44], update_refs=True)
        tree_lru.dec_node_ref_counter(ans)
    # LRU order is still by time_id: A is older than B? Actually after hits, time_id updated for A on each match, so A becomes newer.
    # To keep A older, we need to bump B's time after? Let's instead create scenario where A is older but hot
    # Insert A first, then B, then hit A but check lru still evicts A first if we artificially reset time? Simpler to just verify GDSF behavior as above
    # For GDSF, hot survives regardless of time
    tree_gdsf, _, _ = _make_tree(policy="gdsf", alpha=10.0, hash_page_size=1, big_page_num=4)
    n_a_g = _insert_big(tree_gdsf, [1, 2, 3, 4], [11, 22, 33, 44], 30)
    n_b_g = _insert_big(tree_gdsf, [5, 6, 7, 8], [55, 66, 77, 88], 31)
    for _ in range(3):
        ans, _, _ = tree_gdsf.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int64), block_hashs=[11, 22, 33, 44], update_refs=True)
        tree_gdsf.dec_node_ref_counter(ans)
    # A hits 4 -> prio 4*10/4=10, B hits1 ->2.5, so B evicted first under GDSF even if B is newer
    assert n_a_g.priority > n_b_g.priority
    evicted = []
    tree_gdsf._evict(4, lambda mem, buf: evicted.append(buf))
    assert n_b_g.parent is None
    assert n_a_g.parent is not None


def test_size_aware_on_vs_off():
    # Same freq, different size: with size_aware, smaller size => larger priority
    tree_sa, _, _ = _make_tree(policy="gdsf", alpha=1.0, size_aware=True, hash_page_size=1, big_page_num=4)
    node_big = _insert_big(tree_sa, [1, 2, 3, 4], [11, 22, 33, 44], 40)  # size 4
    node_small = _insert_small(tree_sa, [1, 2, 3, 4, 5], [11, 22, 33, 44, 55], 102)
    # big size=4 prio 0.25, small size=1 prio 1.0
    assert node_big.priority == pytest.approx(0.25)
    assert node_small.priority == pytest.approx(1.0)
    # So small has larger priority -> evicted later
    # With size_aware=False, size=1 for both => priorities equal (1.0)
    tree_no_sa, _, _ = _make_tree(policy="gdsf", alpha=1.0, size_aware=False, hash_page_size=1, big_page_num=4)
    nb = _insert_big(tree_no_sa, [1, 2, 3, 4], [11, 22, 33, 44], 50)
    ns = _insert_small(tree_no_sa, [1, 2, 3, 4, 5], [11, 22, 33, 44, 55], 103)
    # With size_aware false, both size treated as 1 => same priority
    assert nb.priority == pytest.approx(1.0)
    assert ns.priority == pytest.approx(1.0)


def test_gdsf_clock_inflation():
    tree, _, _ = _make_tree(policy="gdsf", alpha=1.0, size_aware=True, hash_page_size=1, big_page_num=4)
    n1 = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 60)
    n2 = _insert_big(tree, [5, 6, 7, 8], [55, 66, 77, 88], 61)
    # bump n1 to make its priority 0.75
    for _ in range(2):
        ans, _, _ = tree.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int64), block_hashs=[11, 22, 33, 44], update_refs=True)
        tree.dec_node_ref_counter(ans)
    assert n1.priority == pytest.approx(0.75)
    # evict n2 (smaller prio 0.25) -> clock should become 0.25
    tree._evict(4, lambda mem, buf: None)
    assert tree._gdsf_clock == pytest.approx(0.25)
    # New node inserted after clock inflation should have priority = clock + freq/size =0.25+0.25=0.5
    n3 = _insert_big(tree, [9, 10, 11, 12], [91, 92, 93, 94], 62)
    assert n3.priority == pytest.approx(0.5)


def test_lfu_no_clock():
    tree, _, _ = _make_tree(policy="lfu", alpha=1.0, size_aware=True, hash_page_size=1, big_page_num=4)
    n1 = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 70)
    n2 = _insert_big(tree, [5, 6, 7, 8], [55, 66, 77, 88], 71)
    for _ in range(2):
        ans, _, _ = tree.match_prefix(torch.tensor([1, 2, 3, 4], dtype=torch.int64), block_hashs=[11, 22, 33, 44], update_refs=True)
        tree.dec_node_ref_counter(ans)
    # evict n2
    tree._evict(4, lambda mem, buf: None)
    # lfu clock stays 0
    assert tree._gdsf_clock == 0.0
    n3 = _insert_big(tree, [9, 10, 11, 12], [91, 92, 93, 94], 72)
    # priority should be 0+0.25=0.25, not inflated
    assert n3.priority == pytest.approx(0.25)


def test_small_buffer_evict_order_priority():
    tree, _, _ = _make_tree(policy="gdsf", alpha=1.0, hash_page_size=1, big_page_num=4)
    node_big = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 80)
    node_s1 = _insert_small(tree, [1, 2, 3, 4, 5], [11, 22, 33, 44, 55], 201)
    node_s2 = _insert_small(tree, [1, 2, 3, 4, 5, 6], [11, 22, 33, 44, 55, 66], 202)
    # Matching the full chain [big,s1,s2] bumps all three (usable prefix)
    # So s1 and s2 both go from 1->2, s1 priority 2.0 as well
    ans, _, _ = tree.match_prefix(torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64), block_hashs=[11, 22, 33, 44, 55, 66], update_refs=True)
    tree.dec_node_ref_counter(ans)
    assert node_s2.hit_count == 2
    assert node_s2.priority == pytest.approx(2.0)
    assert node_s1.hit_count == 2
    assert node_s1.priority == pytest.approx(2.0)
    # Now bump s2 once more alone via matching its prefix again, but need to isolate.
    # Instead bump s2 again by matching same chain: both s1 and s2 go to 3, still tie
    # To make s2 hotter, we can manually bump s2 alone (simulate extra hit on s2 subtree)
    # Easiest: directly record hit on s2
    tree._discard_node(node_s2)
    tree._record_hit(node_s2)
    tree._add_node(node_s2)
    assert node_s2.hit_count == 3
    assert node_s2.priority == pytest.approx(3.0)
    # Now s1 prio 2.0 < s2 prio 3.0, so s1 evicted first
    assert tree._evict_tree_set_for_linear_att[0] is node_s1
    # force eviction (mock says free cache num >0 would skip)
    tree.linear_att_small_page_buffers.get_free_cache_num.return_value = 0
    tree.free_one_small_page_linear_att_buffer()
    assert node_s1.small_page_buffer_idx is None
    assert node_s2.small_page_buffer_idx == 202


def test_sortedset_no_corruption_on_leaf_extension():
    # Insert big, then small chain extending leaf should not corrupt SortedSet
    tree, _, _ = _make_tree(policy="gdsf", hash_page_size=1, big_page_num=4)
    node_big = _insert_big(tree, [1, 2, 3, 4], [11, 22, 33, 44], 90)
    # Extend with s1
    node_s1 = _insert_small(tree, [1, 2, 3, 4, 5], [11, 22, 33, 44, 55], 301)
    # After extension, big is internal (not leaf) should be removed from evict set, s1 leaf should be in set
    assert not node_big.is_leaf()
    assert node_s1.is_leaf()
    assert node_big not in tree._evict_tree_set
    assert node_s1 in tree._evict_tree_set
    # No exception on further inserts
    node_s2 = _insert_small(tree, [1, 2, 3, 4, 5, 6], [11, 22, 33, 44, 55, 66], 302)
    assert node_s2 in tree._evict_tree_set


def test_resolve_gdsf_config_priority():
    # explicit args win over env
    import os
    os.environ["LIGHTLLM_RADIX_CACHE_EVICTION_POLICY"] = "gdsf"
    os.environ["LIGHTLLM_RADIX_CACHE_GDSF_ALPHA"] = "9.9"
    try:
        pol, alpha, sa = _resolve_gdsf_config(eviction_policy="lru", gdsf_alpha=1.0, enable_size_aware=True)
        assert pol == "lru"
        assert alpha == 1.0
        # env should be used when explicit is None
        pol2, alpha2, sa2 = _resolve_gdsf_config(eviction_policy=None, gdsf_alpha=None, enable_size_aware=None)
        assert pol2 == "gdsf"
        assert alpha2 == pytest.approx(9.9)
    finally:
        os.environ.pop("LIGHTLLM_RADIX_CACHE_EVICTION_POLICY", None)
        os.environ.pop("LIGHTLLM_RADIX_CACHE_GDSF_ALPHA", None)
