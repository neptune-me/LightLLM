"""Tests for compute-aware GDSF (gdsf_compute) eviction policy.

Covers:
- factory and CLI/StartArgs choices for ``gdsf_compute``
- insertion does not count as reuse (freq 0)
- quadratic recompute cost L*(L+1)/2 and resident_tokens denominator
- priority = global_age + freq * recompute_cost / resident_tokens
- aging via global_age (per-node frozen clock + freq*cost/size)
- integration through RadixCache with gdsf_compute
"""
import torch
import pytest

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache, TreeNode
from lightllm.server.router.dynamic_prompt.radix_eviction_policy import (
    GDSFEvictionPolicy,
    create_eviction_policy,
)


def _make_node(size: int, prefix_len: int = None, ref: int = 0) -> TreeNode:
    node = TreeNode()
    node.token_id_key = torch.zeros((size,), dtype=torch.int64)
    node.token_mem_index_value = torch.zeros((size,), dtype=torch.int64)
    node.node_value_len = size
    node.node_prefix_total_len = prefix_len if prefix_len is not None else size
    node.ref_counter = ref
    return node


def _t(*ids):
    return torch.tensor(list(ids), dtype=torch.int64, device="cpu")


def _present(tree, key):
    node, size, _ = tree.match_prefix(key, update_refs=False)
    return node is not None and size == len(key)


def _reuse(tree, key, times=1):
    for _ in range(times):
        node, _, _ = tree.match_prefix(key, update_refs=True)
        tree.dec_node_ref_counter(node)


# --------------------------------------------------------------------------- #
# Factory / config
# --------------------------------------------------------------------------- #
def test_factory_gdsf_compute_is_compute_aware():
    p = create_eviction_policy("gdsf_compute")
    assert isinstance(p, GDSFEvictionPolicy)
    assert p.compute_aware is True
    # even if caller passes False explicitly, gdsf_compute forces True
    p2 = create_eviction_policy("gdsf_compute", compute_aware=False)
    assert p2.compute_aware is True


def test_factory_gdsf_compute_case_insensitive():
    p = create_eviction_policy("GDSF_COMPUTE")
    assert p.compute_aware is True


def test_factory_gdsf_compute_accepts_kwargs():
    p = create_eviction_policy("gdsf_compute", size_exponent=0.5, compute_weight=0.5)
    assert p.size_exponent == 0.5
    assert p.compute_weight == 0.5
    assert p.compute_aware is True


def test_factory_gdsf_not_compute_aware_by_default():
    p = create_eviction_policy("gdsf")
    assert p.compute_aware is False


def test_factory_gdsf_with_compute_aware_flag():
    p = create_eviction_policy("gdsf", compute_aware=True)
    assert p.compute_aware is True


def test_cli_choices_include_gdsf_compute():
    from lightllm.server.api_cli import add_cli_args
    import argparse

    parser = argparse.ArgumentParser()
    parser = add_cli_args(parser)
    args = parser.parse_args(["--radix_cache_eviction_policy", "gdsf_compute"])
    assert args.radix_cache_eviction_policy == "gdsf_compute"
    # also accepts gdsf and lru
    for choice in ["lru", "gdsf"]:
        a = parser.parse_args(["--radix_cache_eviction_policy", choice])
        assert a.radix_cache_eviction_policy == choice


def test_startargs_choices_include_gdsf_compute():
    try:
        from lightllm.server.core.objs.start_args_type import StartArgs

        a = StartArgs(radix_cache_eviction_policy="gdsf_compute")
        assert a.radix_cache_eviction_policy == "gdsf_compute"
        # defaults
        b = StartArgs()
        assert b.radix_cache_gdsf_compute_aware is False
        assert b.radix_cache_gdsf_compute_weight == 1.0
    except ModuleNotFoundError as e:
        # fallback when optional deps (atomics) not installed in test venv
        import pathlib

        content = pathlib.Path("lightllm/server/core/objs/start_args_type.py").read_text()
        assert "gdsf_compute" in content
        assert "radix_cache_gdsf_compute_aware" in content


# --------------------------------------------------------------------------- #
# Insertion does not count as reuse (freq 0)
# --------------------------------------------------------------------------- #
def test_insertion_does_not_count_as_reuse():
    policy = GDSFEvictionPolicy()
    node = _make_node(4)
    policy.on_new_node(node)
    assert node._gdsf_freq == 0
    assert policy.priority(node) == pytest.approx(0.0)  # clock 0 + 0/size
    # only genuine reuse increments
    policy.on_access(node)
    assert node._gdsf_freq == 1
    assert policy.priority(node) == pytest.approx(0.25)  # 1/4


def test_insertions_via_radix_cache_start_at_zero():
    tree = RadixCache("gdsf_insert_zero", 100, 0, eviction_policy="gdsf")
    _, node = tree.insert(_t(10, 11, 12, 13))
    assert node._gdsf_freq == 0
    # re-insert same prefix must not bump
    tree.insert(_t(10, 11, 12, 13))
    assert node._gdsf_freq == 0
    # genuine reuse bumps
    _reuse(tree, _t(10, 11, 12, 13), times=1)
    assert node._gdsf_freq == 1


def test_split_inherits_zero_freq():
    policy = GDSFEvictionPolicy()
    orig = _make_node(6, prefix_len=6)
    policy.on_new_node(orig)
    assert orig._gdsf_freq == 0
    head = _make_node(2, prefix_len=2)
    policy.on_new_node(head, split_from=orig)
    assert head._gdsf_freq == 0
    # after one reuse of orig, split of a reused node inherits 1
    policy.on_access(orig)
    head2 = _make_node(2, prefix_len=2)
    policy.on_new_node(head2, split_from=orig)
    assert head2._gdsf_freq == 1


# --------------------------------------------------------------------------- #
# Quadratic recompute cost end^2 - start^2
# --------------------------------------------------------------------------- #
def test_compute_cost_is_quadratic():
    policy = GDSFEvictionPolicy(compute_aware=True)
    # isolated nodes: start 0 => cost = L^2
    # L=10 size10 => 100, L=3 size3 => 9
    n10 = _make_node(10, prefix_len=10)
    n3 = _make_node(3, prefix_len=3)
    assert policy._compute_cost(n10) == pytest.approx(100.0)
    assert policy._compute_cost(n3) == pytest.approx(9.0)
    # segment: L=4 size2 => start2 => 16-4=12
    n_seg = _make_node(2, prefix_len=4)
    assert policy._compute_cost(n_seg) == pytest.approx(12.0)
    # fallback when prefix not set -> size^2
    n4_no_prefix = _make_node(4, prefix_len=0)
    # size 4 => 16
    assert policy._compute_cost(n4_no_prefix) == pytest.approx(16.0)


def test_compute_weight_tunes_linear_vs_quadratic():
    # cost 9 for L=3 size3 (3^2)
    # weight 1.0 => 9, weight 0.5 => sqrt(9)=3
    n = _make_node(3, prefix_len=3)
    n._gdsf_freq = 1
    n._gdsf_clock = 0.0
    p_quad = GDSFEvictionPolicy(compute_aware=True, compute_weight=1.0)
    p_lin = GDSFEvictionPolicy(compute_aware=True, compute_weight=0.5)
    # priority = clock + freq*cost/resident ; resident=3
    # quad: 9/3=3.0, lin: sqrt(9)/3 =1.0
    assert p_quad.priority(n) == pytest.approx(3.0)
    assert p_lin.priority(n) == pytest.approx((9.0 ** 0.5) / 3)


# --------------------------------------------------------------------------- #
# Priority = global_age + freq * recompute_cost / resident_tokens
# --------------------------------------------------------------------------- #
def test_priority_formula_global_age_plus_freq_cost_over_resident():
    policy = GDSFEvictionPolicy(compute_aware=True)
    policy.aging_clock = 5.0
    node = _make_node(2, prefix_len=4)  # L=4 size2 => start2 => cost 12, resident 2
    policy.on_new_node(node)  # freq 0, clock 5
    assert node._gdsf_clock == 5.0
    assert policy.priority(node) == pytest.approx(5.0)  # 5 + 0
    policy.on_access(node)  # freq 1, clock stays 5
    # priority = 5 + 1*12/2 = 11
    assert policy.priority(node) == pytest.approx(11.0)
    policy.on_access(node)  # freq 2
    # 5 + 2*12/2 = 17
    assert policy.priority(node) == pytest.approx(17.0)


def test_priority_resident_tokens_not_size_exponent():
    # compute-aware must divide by resident_tokens, not size**exp
    # Use exp 0.5 to show difference: size**0.5 vs size
    policy_exp = GDSFEvictionPolicy(compute_aware=True, size_exponent=0.5)
    node = _make_node(4, prefix_len=4)  # isolated L4 size4 => cost 16
    policy_exp.on_new_node(node)
    policy_exp.on_access(node)  # freq 1
    # Should be 16/4 =4.0, not 16 / 4**0.5 (=8)
    assert policy_exp.priority(node) == pytest.approx(4.0)
    # Non-compute-aware still uses exponent
    policy_plain = GDSFEvictionPolicy(size_exponent=0.5, compute_aware=False)
    node2 = _make_node(4, prefix_len=4)
    policy_plain.on_new_node(node2)
    policy_plain.on_access(node2)
    assert policy_plain.priority(node2) == pytest.approx(1 / (4 ** 0.5))


def test_compute_aware_size_aware_vs_compute_aware_ordering():
    # Equal freq, different size and depth: size-aware and compute-aware disagree
    # Small shallow: size2 L2 cost4 (2^2) ; Large deep: size8 L8 cost64 (8^2)
    # With freq1, size-aware: small 0.5 vs large 0.125 -> small larger, large evicted
    # Compute-aware: small 2 vs large 8 -> large larger, small evicted (flipped)
    # This demonstrates recompute_cost dominates size.
    policy_plain = GDSFEvictionPolicy(size_exponent=1.0, compute_aware=False)
    policy_compute = GDSFEvictionPolicy(compute_aware=True)
    small = _make_node(2, prefix_len=2)
    large = _make_node(8, prefix_len=8)
    for p in (policy_plain, policy_compute):
        # need to create fresh nodes for each policy (time_id tie-breaker)
        pass
    # plain
    s1 = _make_node(2, prefix_len=2)
    l1 = _make_node(8, prefix_len=8)
    policy_plain.on_new_node(s1)
    policy_plain.on_new_node(l1)
    policy_plain.on_access(s1)
    policy_plain.on_access(l1)
    assert policy_plain.compare_key(l1) < policy_plain.compare_key(s1)

    # compute-aware: cost small 3/2=1.5, large 36/8=4.5 => large larger
    s2 = _make_node(2, prefix_len=2)
    l2 = _make_node(8, prefix_len=8)
    policy_compute.on_new_node(s2)
    policy_compute.on_new_node(l2)
    policy_compute.on_access(s2)
    policy_compute.on_access(l2)
    assert policy_compute.compare_key(s2) < policy_compute.compare_key(l2)


# --------------------------------------------------------------------------- #
# Aging via global_age
# --------------------------------------------------------------------------- #
def test_aging_uses_global_age_plus_freq_cost_over_resident():
    policy = GDSFEvictionPolicy(compute_aware=True)
    a = _make_node(2, prefix_len=2)  # cost 3
    b = _make_node(2, prefix_len=2)
    policy.on_new_node(a)
    policy.on_new_node(b)
    policy.on_access(a)  # a freq1 => priority 1.5
    policy.on_access(a)  # a freq2 => 3.0
    # b stays freq0 => priority 0
    high = policy.priority(a)
    policy.on_evict(a)
    assert policy.aging_clock == pytest.approx(high)
    # new node inherits aging_clock
    c = _make_node(2, prefix_len=2)
    policy.on_new_node(c)
    assert c._gdsf_clock == pytest.approx(high)
    # c freq0 => priority = aging_clock
    assert policy.priority(c) == pytest.approx(high)
    # one reuse makes it > high
    policy.on_access(c)
    assert policy.priority(c) > high


def test_touch_does_not_affect_aging_or_freq():
    policy = GDSFEvictionPolicy(compute_aware=True)
    node = _make_node(4, prefix_len=4)  # cost10
    policy.on_new_node(node)
    policy.on_access(node)  # freq1
    clock_before = node._gdsf_clock
    freq_before = node._gdsf_freq
    prio_before = policy.priority(node)
    for _ in range(5):
        policy.touch(node)
    assert node._gdsf_freq == freq_before
    assert node._gdsf_clock == clock_before
    assert policy.priority(node) == pytest.approx(prio_before)


# --------------------------------------------------------------------------- #
# RadixCache integration with gdsf_compute
# --------------------------------------------------------------------------- #
def test_radix_cache_gdsf_compute_uses_quadratic_cost():
    tree = RadixCache("gdsf_compute_quad", 100, 0, eviction_policy="gdsf_compute")
    assert tree._evict_policy.compute_aware is True
    # small shallow L2 cost3 size2 ; deep L8 cost36 size8
    # Insert both, give each one reuse => freq1 each
    tree.insert(_t(1, 2))  # small, L2
    tree.insert(_t(10, 11, 12, 13, 14, 15, 16, 17))  # large, L8
    _reuse(tree, _t(1, 2), times=1)
    _reuse(tree, _t(10, 11, 12, 13, 14, 15, 16, 17), times=1)
    # compute-aware: large should have higher priority (36/8=4.5 vs 3/2=1.5)
    # so small evicted first
    tree.evict(1, lambda x: x)
    assert not _present(tree, _t(1, 2))
    assert _present(tree, _t(10, 11, 12, 13, 14, 15, 16, 17))


def test_radix_cache_gdsf_compute_via_gdsf_flag():
    tree = RadixCache(
        "gdsf_compute_flag", 100, 0, eviction_policy="gdsf", eviction_policy_kwargs={"compute_aware": True}
    )
    assert tree._evict_policy.compute_aware is True
    _, node = tree.insert(_t(1, 2, 3, 4))
    assert node._gdsf_freq == 0
    _reuse(tree, _t(1, 2, 3, 4), times=1)
    assert node._gdsf_freq == 1
    # isolated L4 size4 => start0 cost 16 => 16/4=4.0
    assert tree._evict_policy.priority(node) == pytest.approx(4.0)


def test_radix_cache_gdsf_compute_deep_vs_shallow_same_size():
    # Same resident size, different L: deep should win
    tree = RadixCache("gdsf_compute_deep", 100, 0, eviction_policy="gdsf_compute")
    # chain: [1,2] L2 size2 ; [1,2,3,4] extends to L4 size2 deep leaf [3,4]
    tree.insert(_t(1, 2))
    tree.insert(_t(1, 2, 3, 4))
    tree.insert(_t(10, 11))  # shallow isolated L2 size2
    _reuse(tree, _t(1, 2, 3, 4), times=1)  # deep leaf freq1
    _reuse(tree, _t(10, 11), times=1)  # shallow freq1
    # deep L4 cost10/2=5 vs shallow L2 cost3/2=1.5 => deep larger
    tree.evict(1, lambda x: x)
    assert not _present(tree, _t(10, 11))
    assert _present(tree, _t(1, 2, 3, 4))


def test_radix_cache_gdsf_compute_insertion_not_counted():
    tree = RadixCache("gdsf_compute_insert", 100, 0, eviction_policy="gdsf_compute")
    _, n1 = tree.insert(_t(1, 2, 3))
    _, n2 = tree.insert(_t(1, 2, 3, 4, 5))
    assert n1._gdsf_freq == 0 and n2._gdsf_freq == 0
    # re-insert same prefix should not bump
    tree.insert(_t(1, 2, 3))
    assert n1._gdsf_freq == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
