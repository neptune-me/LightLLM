"""Integration tests for GDSF eviction wired through RadixCache.

These exercise the full insert / match_prefix / evict path and assert that the
GDSF policy produces size-aware, frequency-aware eviction that differs from the
default LRU policy.
"""
import torch
import pytest

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache


def _t(*ids):
    return torch.tensor(list(ids), dtype=torch.int64, device="cpu")


def _present(tree, key):
    node, size, _ = tree.match_prefix(key, update_refs=False)
    return node is not None and size == len(key)


def _reuse(tree, key, times=1):
    """Simulate `times` genuine reuses (a request pins the prefix then releases)."""
    for _ in range(times):
        node, _, _ = tree.match_prefix(key, update_refs=True)
        tree.dec_node_ref_counter(node)


# --------------------------------------------------------------------------- #
# Backward compatibility: default is still LRU                                 #
# --------------------------------------------------------------------------- #
def test_default_policy_is_lru():
    tree = RadixCache("gdsf_default", 100, 0)
    assert tree._evict_policy.__class__.__name__ == "LRUEvictionPolicy"


# --------------------------------------------------------------------------- #
# Size awareness: equal frequency, larger cold prefix evicted first           #
# --------------------------------------------------------------------------- #
def test_gdsf_evicts_larger_prefix_first():
    tree = RadixCache("gdsf_size", 100, 0, eviction_policy="gdsf")
    tree.insert(_t(10, 11))  # small: 2 tokens
    tree.insert(_t(20, 21, 22, 23, 24, 25, 26, 27, 28, 29))  # large: 10 tokens

    tree.evict(1, lambda x: x)

    assert _present(tree, _t(10, 11))  # small survives
    assert not _present(tree, _t(20, 21))  # large was evicted


def test_lru_evicts_by_recency_not_size():
    """Same inserts under LRU evict the *oldest* (the small one), proving the
    policy actually changes behavior."""
    tree = RadixCache("lru_size", 100, 0, eviction_policy="lru")
    tree.insert(_t(10, 11))  # inserted first -> oldest
    tree.insert(_t(20, 21, 22, 23, 24, 25, 26, 27, 28, 29))

    tree.evict(1, lambda x: x)

    assert not _present(tree, _t(10, 11))  # oldest evicted
    assert _present(tree, _t(20, 21, 22, 23, 24, 25, 26, 27, 28, 29))


# --------------------------------------------------------------------------- #
# Frequency awareness (LFU): hot prefix survives, cold prefix evicted         #
# --------------------------------------------------------------------------- #
def test_gdsf_keeps_frequently_used_prefix():
    tree = RadixCache("gdsf_freq", 100, 0, eviction_policy="gdsf")
    tree.insert(_t(10, 11, 12))  # A
    tree.insert(_t(20, 21, 22))  # B, same size

    _reuse(tree, _t(10, 11, 12), times=5)  # genuinely reuse A
    _reuse(tree, _t(20, 21, 22), times=1)  # reuse B once

    tree.evict(1, lambda x: x)

    assert _present(tree, _t(10, 11, 12))  # hot A survives
    assert not _present(tree, _t(20, 21))  # cold B evicted


def test_gdsf_frequency_ignores_readonly_probes():
    """Read-only probes (update_refs=False) must NOT raise a node's frequency.

    Discriminating case: B is genuinely reused once (freq 2) and A is only
    probed heavily (freq stays 1). If probes counted, A's frequency would
    dwarf B's and B would be evicted; instead A must be the victim.
    """
    tree = RadixCache("gdsf_probe", 100, 0, eviction_policy="gdsf")
    tree.insert(_t(10, 11, 12))  # A
    tree.insert(_t(20, 21, 22))  # B, same size

    _reuse(tree, _t(20, 21, 22), times=1)  # B: real reuse -> freq 2
    for _ in range(10):
        tree.match_prefix(_t(10, 11, 12), update_refs=False)  # A: probes only

    tree.evict(1, lambda x: x)

    assert not _present(tree, _t(10, 11, 12))  # A evicted despite heavy probing
    assert _present(tree, _t(20, 21, 22))  # genuinely-reused B survives


def test_gdsf_frequency_ignores_reinsertions():
    """Re-inserting the same prefix must NOT inflate its frequency."""
    tree = RadixCache("gdsf_reinsert", 100, 0, eviction_policy="gdsf")
    tree.insert(_t(10, 11, 12))  # A
    tree.insert(_t(20, 21, 22))  # B, same size

    _reuse(tree, _t(20, 21, 22), times=1)  # B: real reuse -> freq 2
    for _ in range(10):
        tree.insert(_t(10, 11, 12))  # A: re-inserted repeatedly

    tree.evict(1, lambda x: x)

    assert not _present(tree, _t(10, 11, 12))  # A evicted despite re-insertions
    assert _present(tree, _t(20, 21, 22))


# --------------------------------------------------------------------------- #
# Referenced nodes are never evicted regardless of policy                      #
# --------------------------------------------------------------------------- #
def test_gdsf_never_evicts_referenced_node():
    tree = RadixCache("gdsf_ref", 100, 0, eviction_policy="gdsf")
    tree.insert(_t(10, 11))
    tree.insert(_t(20, 21, 22, 23, 24, 25, 26, 27, 28, 29))
    # reference the large (would-be first victim) prefix
    tree.match_prefix(_t(20, 21, 22, 23, 24, 25, 26, 27, 28, 29), update_refs=True)

    tree.evict(1, lambda x: x)

    # large is referenced, so the small unreferenced one must go instead
    assert not _present(tree, _t(10, 11))
    assert tree.get_refed_tokens_num() == 10


def test_gdsf_bookkeeping_matches_lru():
    """Token accounting must be identical no matter the policy."""
    tree = RadixCache("gdsf_book", 100, 0, eviction_policy="gdsf")
    tree.insert(_t(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    tree.insert(_t(0, 1, 2, 3, 4, 7, 8, 9))
    assert tree.get_tree_total_tokens_num() == 13
    assert tree.get_refed_tokens_num() == 0
    tree.evict(13, lambda x: x)
    assert tree.get_tree_total_tokens_num() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
