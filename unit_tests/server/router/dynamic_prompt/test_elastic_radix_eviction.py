"""
Elastic Radix Eviction 的单元测试。

关注点:
1. 默认行为不变（未开启开关时，回收仍然是整叶节点粒度）。
2. 开启后，冷叶节点只被裁掉尾部，保留的前缀依然是一段从 root 开始连续可复用的 kv。
3. 裁剪后的树不变量（node_value_len / node_prefix_total_len / key 与 value 等长 / 回收队列位置）。
4. 保留前缀在裁剪后仍可 match，重新插入后可以在其之上继续增长。
5. 不安全的场景（被引用节点、保留前缀过短）会回退到整节点回收。
6. 与可插拔回收策略（lru / gdsf_compute）的组合行为，见文件末尾的 GDSF 小节。
"""

import pytest
import torch

from lightllm.server.router.dynamic_prompt.radix_cache import RadixCache
from lightllm.server.router.dynamic_prompt.linear_att_radix_cache import LinearAttPagedTreeNode
from lightllm.server.router.dynamic_prompt.radix_eviction_policy import (
    ComputeAwareGDSFEvictionPolicy,
    LRUEvictionPolicy,
)


def _tensor(values):
    return torch.tensor(values, dtype=torch.int64, device="cpu")


def _make_tree(
    rank_in_node,
    enable_elastic_eviction=True,
    elastic_min_retain_tokens=1,
    eviction_policy="lru",
):
    return RadixCache(
        "elastic_radix_test",
        100,
        rank_in_node,
        eviction_policy=eviction_policy,
        enable_elastic_eviction=enable_elastic_eviction,
        elastic_min_retain_tokens=elastic_min_retain_tokens,
    )


def _collect(tree, need_remove_tokens):
    """执行一次 evict，返回被释放的 mem index（拼接成一个一维 tensor）。"""
    freed = []
    tree.evict(need_remove_tokens, lambda mem_index: freed.append(mem_index))
    if len(freed) == 0:
        return _tensor([])
    return torch.concat(freed)


def test_disabled_by_default_evicts_whole_leaf():
    """不开启开关时，回收 3 个 token 依然会把整个 10 token 的叶节点删掉。"""
    tree = RadixCache("elastic_radix_test", 100, 0)
    assert tree.enable_elastic_eviction is False

    key = _tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tree.insert(key, _tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]))
    assert tree.get_tree_total_tokens_num() == 10

    freed = _collect(tree, 3)

    assert len(freed) == 10
    assert tree.get_tree_total_tokens_num() == 0
    assert len(tree.root_node.children) == 0


def test_partial_tail_reclaim_frees_exactly_what_is_needed():
    """开启后，只裁掉尾部 3 个 token，保留前 7 个。"""
    tree = _make_tree(rank_in_node=0)

    key = _tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    value = _tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
    _, node = tree.insert(key, value)

    freed = _collect(tree, 3)

    # 只释放尾部 3 个槽位，且释放的正是尾部而不是任意 3 个。
    assert torch.equal(freed, _tensor([107, 108, 109]))
    assert tree.get_tree_total_tokens_num() == 7

    # 节点还在树上，保留的是原始的前缀。
    assert tree.root_node.children[0] is node
    assert torch.equal(node.token_id_key, _tensor([0, 1, 2, 3, 4, 5, 6]))
    assert torch.equal(node.token_mem_index_value, _tensor([100, 101, 102, 103, 104, 105, 106]))


def test_truncation_keeps_tree_invariants():
    tree = _make_tree(rank_in_node=0)

    key = _tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    _, node = tree.insert(key, _tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]))
    time_id_before = node.time_id

    _collect(tree, 4)

    assert len(node.token_id_key) == len(node.token_mem_index_value) == 6
    assert node.node_value_len == 6
    assert node.node_prefix_total_len == node.parent.node_prefix_total_len + 6
    assert node.parent is tree.root_node
    assert node.is_leaf()
    assert node.ref_counter == 0
    # 仍然留在回收队列里，且没有因为被裁剪而“变新”——它依旧是下一个回收目标。
    assert node in tree.evict_tree_set
    assert node.time_id == time_id_before

    # 树的账本与实际持有的 token 数一致。
    assert tree.get_tree_total_tokens_num() == 6
    assert tree.get_refed_tokens_num() == 0


def test_retained_prefix_is_still_matchable():
    tree = _make_tree(rank_in_node=0)

    key = _tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tree.insert(key, _tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]))
    _collect(tree, 3)

    # 保留下来的前缀可以整段命中。
    matched_node, size, values = tree.match_prefix(_tensor([0, 1, 2, 3, 4, 5, 6]), update_refs=False)
    assert size == 7
    assert torch.equal(values, _tensor([100, 101, 102, 103, 104, 105, 106]))
    assert matched_node.node_prefix_total_len == 7

    # 用完整的原始 key 去 match，只能命中保留下来的部分，绝不会返回已经被释放的槽位。
    matched_node, size, values = tree.match_prefix(key, update_refs=False)
    assert size == 7
    assert torch.equal(values, _tensor([100, 101, 102, 103, 104, 105, 106]))


def test_reinsert_after_partial_evict_reuses_retained_prefix():
    """裁剪 -> 重新插入完整序列: 保留的前缀被复用，只有尾部是新分配的。"""
    tree = _make_tree(rank_in_node=0)

    key = _tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tree.insert(key, _tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]))
    _collect(tree, 3)

    # 重新插入完整序列，尾部使用新的 mem index。
    prefix_len, _ = tree.insert(key, _tensor([100, 101, 102, 103, 104, 105, 106, 207, 208, 209]))
    assert prefix_len == 7  # 前 7 个 token 命中了保留下来的前缀

    assert tree.get_tree_total_tokens_num() == 10

    matched_node, size, values = tree.match_prefix(key, update_refs=False)
    assert size == 10
    # 前缀部分复用了原来的槽位，尾部是新插入的槽位。
    assert torch.equal(values, _tensor([100, 101, 102, 103, 104, 105, 106, 207, 208, 209]))
    assert matched_node.node_prefix_total_len == 10


def test_min_retain_tokens_falls_back_to_whole_node_evict():
    """保留的前缀短于 elastic_min_retain_tokens 时，不值得留，整节点回收。"""
    tree = _make_tree(rank_in_node=0, elastic_min_retain_tokens=8)

    key = _tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tree.insert(key, _tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]))

    # 需要 5 个 -> 只能留下 5 个 < 8，回退到整节点回收。
    freed = _collect(tree, 5)
    assert len(freed) == 10
    assert tree.get_tree_total_tokens_num() == 0
    assert len(tree.root_node.children) == 0


def test_min_retain_tokens_allows_truncation_when_prefix_long_enough():
    tree = _make_tree(rank_in_node=0, elastic_min_retain_tokens=8)

    key = _tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tree.insert(key, _tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]))

    # 需要 2 个 -> 留下 8 个 == 阈值，可以裁剪。
    freed = _collect(tree, 2)
    assert torch.equal(freed, _tensor([108, 109]))
    assert tree.get_tree_total_tokens_num() == 8


def test_exact_fit_removes_whole_node_without_leaving_empty_node():
    tree = _make_tree(rank_in_node=0)

    key = _tensor([0, 1, 2, 3, 4])
    tree.insert(key, _tensor([100, 101, 102, 103, 104]))

    freed = _collect(tree, 5)
    assert len(freed) == 5
    assert tree.get_tree_total_tokens_num() == 0
    assert len(tree.root_node.children) == 0


def test_only_the_last_victim_is_truncated():
    """跨多个叶节点回收时，前面的叶节点整体回收，只有最后一个被裁剪。"""
    tree = _make_tree(rank_in_node=0)

    # 两条独立分支，先插入的那条 time_id 更小，会先被回收。
    _, cold_node = tree.insert(_tensor([0, 1, 2, 3]), _tensor([100, 101, 102, 103]))
    _, warm_node = tree.insert(_tensor([9, 8, 7, 6, 5, 4]), _tensor([200, 201, 202, 203, 204, 205]))
    assert tree.get_tree_total_tokens_num() == 10

    # 需要 6 个: 冷分支 4 个全回收 + 热分支裁掉尾部 2 个。
    freed = _collect(tree, 6)

    assert len(freed) == 6
    assert torch.equal(freed, _tensor([100, 101, 102, 103, 204, 205]))
    assert cold_node.parent is None
    assert tree.root_node.children[9] is warm_node
    assert torch.equal(warm_node.token_id_key, _tensor([9, 8, 7, 6]))
    assert tree.get_tree_total_tokens_num() == 4


def test_refed_nodes_are_never_truncated():
    tree = _make_tree(rank_in_node=0)

    tree.insert(_tensor([0, 1, 2, 3]), _tensor([100, 101, 102, 103]))
    tree.insert(_tensor([0, 1, 2, 3, 4, 5, 6, 7]), _tensor([100, 101, 102, 103, 104, 105, 106, 107]))

    refed_node, _, _ = tree.match_prefix(_tensor([0, 1, 2, 3]), update_refs=True)
    assert refed_node.ref_counter == 1
    assert tree.get_refed_tokens_num() == 4

    # 只能从未被引用的尾部叶节点里回收。
    freed = _collect(tree, 2)
    assert torch.equal(freed, _tensor([106, 107]))

    # 被引用的前缀一个 token 都没有丢。
    assert torch.equal(refed_node.token_id_key, _tensor([0, 1, 2, 3]))
    assert torch.equal(refed_node.token_mem_index_value, _tensor([100, 101, 102, 103]))
    assert tree.get_refed_tokens_num() == 4
    assert tree.get_tree_total_tokens_num() == 6


def test_cannot_evict_more_than_unrefed_tokens():
    """弹性回收不会放宽“可回收量”的上界。"""
    tree = _make_tree(rank_in_node=0)

    tree.insert(_tensor([0, 1, 2, 3]), _tensor([100, 101, 102, 103]))
    tree.match_prefix(_tensor([0, 1, 2, 3]), update_refs=True)

    with pytest.raises(AssertionError):
        _collect(tree, 1)


def test_truncated_node_can_still_be_split():
    """裁剪后的节点依然是一个正常节点，可以继续分裂出兄弟分支。"""
    tree = _make_tree(rank_in_node=0)

    tree.insert(_tensor([0, 1, 2, 3, 4, 5, 6, 7]), _tensor([100, 101, 102, 103, 104, 105, 106, 107]))
    _collect(tree, 3)  # 保留 [0, 1, 2, 3, 4]

    tree.insert(_tensor([0, 1, 2, 9]), _tensor([100, 101, 102, 109]))

    split_node = tree.root_node.children[0]
    assert torch.equal(split_node.token_id_key, _tensor([0, 1, 2]))
    assert set(split_node.children.keys()) == {3, 9}
    assert tree.get_tree_total_tokens_num() == 6

    _, size, values = tree.match_prefix(_tensor([0, 1, 2, 3, 4]), update_refs=False)
    assert size == 5
    assert torch.equal(values, _tensor([100, 101, 102, 103, 104]))


def test_truncated_node_can_still_be_merged():
    tree = _make_tree(rank_in_node=0)

    tree.insert(_tensor([0, 1, 2]), _tensor([100, 101, 102]))
    tree.insert(_tensor([0, 1, 2, 3, 4, 5]), _tensor([100, 101, 102, 103, 104, 105]))
    _collect(tree, 2)  # 裁掉子节点尾部 2 个 -> 子节点只剩 [3]

    tree.merge_unreferenced_nodes()

    merged = tree.root_node.children[0]
    assert torch.equal(merged.token_id_key, _tensor([0, 1, 2, 3]))
    assert torch.equal(merged.token_mem_index_value, _tensor([100, 101, 102, 103]))
    assert merged.node_value_len == 4
    assert merged.node_prefix_total_len == 4
    assert tree.get_tree_total_tokens_num() == 4


def test_full_eviction_still_clears_the_tree():
    """需要回收的量大于等于全树时，弹性回收不会挡住整树清空。"""
    tree = _make_tree(rank_in_node=0)

    tree.insert(_tensor([0, 1, 2, 3]), _tensor([100, 101, 102, 103]))
    tree.insert(_tensor([9, 8, 7]), _tensor([200, 201, 202]))

    freed = _collect(tree, 7)
    assert len(freed) == 7
    assert tree.get_tree_total_tokens_num() == 0
    assert len(tree.root_node.children) == 0


def test_repeated_truncation_shrinks_the_same_cold_leaf():
    """连续多次小额回收会持续从同一个冷叶节点尾部取，直到它被整体回收。"""
    tree = _make_tree(rank_in_node=0)

    tree.insert(_tensor([0, 1, 2, 3, 4, 5]), _tensor([100, 101, 102, 103, 104, 105]))

    assert torch.equal(_collect(tree, 2), _tensor([104, 105]))
    assert tree.get_tree_total_tokens_num() == 4
    assert torch.equal(_collect(tree, 2), _tensor([102, 103]))
    assert tree.get_tree_total_tokens_num() == 2
    assert torch.equal(_collect(tree, 2), _tensor([100, 101]))
    assert tree.get_tree_total_tokens_num() == 0
    assert len(tree.root_node.children) == 0


def test_min_retain_tokens_must_be_positive():
    with pytest.raises(AssertionError):
        _make_tree(rank_in_node=0, elastic_min_retain_tokens=0)


class _FakeAllocator:
    def __init__(self, can_use_mem_size):
        self.can_use_mem_size = can_use_mem_size

    def free(self, free_index):
        self.can_use_mem_size += len(free_index)


class _FakeMemManager:
    def __init__(self, can_use_mem_size):
        self.allocator = _FakeAllocator(can_use_mem_size)
        self.freed = []

    def free(self, free_index):
        self.freed.append(free_index)
        self.allocator.free(free_index)


def test_free_radix_cache_to_get_enough_token_only_frees_the_tail():
    """端到端走一遍回收入口: 只有尾部的槽位被还给 mem manager。"""
    mem_manager = _FakeMemManager(can_use_mem_size=0)
    tree = RadixCache(
        "elastic_radix_test",
        100,
        0,
        mem_manager=mem_manager,
        enable_elastic_eviction=True,
        elastic_min_retain_tokens=1,
    )

    key = _tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    tree.insert(key, _tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109]))

    tree.free_radix_cache_to_get_enough_token(need_token_num=3)

    assert len(mem_manager.freed) == 1
    assert torch.equal(mem_manager.freed[0], _tensor([107, 108, 109]))
    assert mem_manager.allocator.can_use_mem_size == 3
    assert tree.get_tree_total_tokens_num() == 7

    # 保留下来的前缀依然可以命中复用。
    _, size, values = tree.match_prefix(key, update_refs=False)
    assert size == 7
    assert torch.equal(values, _tensor([100, 101, 102, 103, 104, 105, 106]))


def test_paged_linear_att_cache_does_not_support_elastic_eviction():
    """
    LinearAttPagedRadixCache 的节点按 page 对齐，并且每个 page 绑定 linear attention
    的 state buffer，裁掉 page 尾部会破坏 page 对齐、并让 state buffer 失效，
    因此这一类 cache 明确不支持弹性回收。
    """
    assert not hasattr(LinearAttPagedTreeNode, "truncate_tail")


# --------------------------------------------------------------------------------------
# 与回收策略的组合: elastic 只决定"从选中的受害者身上拿多少", 受害者的选择仍然完全由策略决定。
#
# GDSF 下 priority = age + freq * recompute_cost / resident_tokens, 其中
# recompute_cost / resident_tokens == (e**2 - s**2) / (e - s) == e + s
# (e = node_prefix_total_len, s = e - node_value_len)。裁尾只会降低 e, 所以裁剪后的
# 节点优先级只降不升, 依然是下一个受害者。
# --------------------------------------------------------------------------------------
class _FakeNode:
    """与 test_radix_eviction_policy.py 中一致的最小节点替身，用于纯策略测试。"""

    def __init__(self, node_prefix_total_len, node_value_len, time_id=0):
        self.node_prefix_total_len = node_prefix_total_len
        self.node_value_len = node_value_len
        self.ref_counter = 0
        self.time_id = time_id
        self.children = {}
        self.freq = 0
        self.age = 0.0
        self.priority = 0.0

    def update_time(self):
        self.time_id += 1


def _insert(tree, ids):
    return tree.insert(_tensor(ids))


def _reuse(tree, ids):
    """一次真正的 prompt cache 命中 (match_prefix(update_refs=True))。"""
    node, _, _ = tree.match_prefix(_tensor(ids), update_refs=True)
    if node is not None:
        tree.dec_node_ref_counter(node)
    return node


def test_lru_on_truncate_is_a_noop():
    """LRU 的 key 与几何无关；裁剪更不能刷新 time_id，否则冷节点会假装变热。"""
    policy = LRUEvictionPolicy()
    node = _FakeNode(node_prefix_total_len=5, node_value_len=5, time_id=7)

    policy.on_truncate(node)

    assert node.time_id == 7
    assert node.freq == 0


def test_gdsf_on_truncate_rescores_geometry_against_frozen_age():
    policy = ComputeAwareGDSFEvictionPolicy()
    policy.global_age = 100.0  # 当前时钟不允许泄漏进裁剪后的分数

    node = _FakeNode(node_prefix_total_len=8, node_value_len=8, time_id=7)
    node.freq = 2
    node.age = 10.0
    node.priority = policy.compute_priority(node)
    assert node.priority == pytest.approx(10.0 + 2 * (8 + 0))  # 26

    # 裁掉尾部 3 个 token: e 从 8 降到 5, s 仍然是 0。
    node.node_prefix_total_len = 5
    node.node_value_len = 5
    policy.on_truncate(node)

    assert node.priority == pytest.approx(10.0 + 2 * (5 + 0))  # 20, 只降不升
    # 裁剪不是一次复用: 频率、冻结的 age、recency 都不能变。
    assert node.freq == 2
    assert node.age == pytest.approx(10.0)
    assert node.time_id == 7


def test_gdsf_on_truncate_does_not_advance_the_aging_clock():
    """aging clock 只跟踪真正离开缓存的对象；被裁剪的节点还在树上。"""
    policy = ComputeAwareGDSFEvictionPolicy()
    node = _FakeNode(node_prefix_total_len=4, node_value_len=4)
    node.freq = 1
    policy.init_node(node)

    policy.on_truncate(node)
    assert policy.global_age == pytest.approx(0.0)

    # 整节点回收才推进时钟。
    policy.on_evict(node)
    assert policy.global_age == pytest.approx(node.priority)


def test_gdsf_elastic_trims_the_cheap_branch_and_keeps_the_expensive_one():
    tree = _make_tree(rank_in_node=0, eviction_policy="gdsf_compute", elastic_min_retain_tokens=1)

    _insert(tree, [0, 1, 2])  # 便宜分支 A: e=3 -> 每次命中价值 3
    _insert(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # 昂贵分支 B: e=8 -> 每次命中价值 8
    node_a = tree.root_node.children[0]
    node_b = tree.root_node.children[10]

    _reuse(tree, [0, 1, 2])  # freq_A = 1 -> priority 3
    _reuse(tree, [10, 11, 12, 13, 14, 15, 16, 17])  # freq_B = 1 -> priority 8
    assert node_a.priority == pytest.approx(3.0)
    assert node_b.priority == pytest.approx(8.0)

    freed = _collect(tree, 2)

    # 受害者仍由 GDSF 选出 (最低优先级的 A)，elastic 只把它裁短。
    assert torch.equal(freed, _tensor([1, 2]))
    assert torch.equal(node_a.token_id_key, _tensor([0]))
    # 昂贵分支一个 token 都没动。
    assert torch.equal(node_b.token_id_key, _tensor([10, 11, 12, 13, 14, 15, 16, 17]))
    assert tree.get_tree_total_tokens_num() == 9


def test_gdsf_elastic_truncation_lowers_priority_but_keeps_history():
    tree = _make_tree(rank_in_node=0, eviction_policy="gdsf_compute", elastic_min_retain_tokens=1)

    _insert(tree, [0, 1, 2, 3, 4, 5, 6, 7])
    node = tree.root_node.children[0]
    _reuse(tree, [0, 1, 2, 3, 4, 5, 6, 7])
    assert node.priority == pytest.approx(8.0)  # age 0 + freq 1 * (e 8 + s 0)

    _collect(tree, 3)  # 裁到 5 个 token

    assert node.node_value_len == 5
    assert node.priority == pytest.approx(5.0)  # age 0 + freq 1 * (e 5 + s 0)
    assert node.freq == 1  # 丢 token 不是复用
    assert node.age == pytest.approx(0.0)  # 冻结的 aging 基线保持不变


def test_gdsf_elastic_clock_advances_only_on_whole_node_eviction():
    tree = _make_tree(rank_in_node=0, eviction_policy="gdsf_compute", elastic_min_retain_tokens=1)
    policy = tree.eviction_policy

    _insert(tree, [0, 1, 2, 3])
    node = tree.root_node.children[0]
    _reuse(tree, [0, 1, 2, 3])
    assert node.priority == pytest.approx(4.0)

    _collect(tree, 2)  # 裁剪 -> 时钟不动
    assert policy.global_age == pytest.approx(0.0)
    assert node.priority == pytest.approx(2.0)

    _collect(tree, 2)  # 剩下的正好被整节点回收 -> 时钟推进到受害者的优先级
    assert policy.global_age == pytest.approx(2.0)
    assert len(tree.root_node.children) == 0


def test_gdsf_elastic_keeps_carving_the_same_cold_leaf():
    """被裁剪后优先级只降不升，所以 GDSF 会继续从同一个最便宜的前缀上取，而不是转去动贵的。"""
    tree = _make_tree(rank_in_node=0, eviction_policy="gdsf_compute", elastic_min_retain_tokens=1)

    _insert(tree, [0, 1, 2, 3])  # A: priority 4
    _insert(tree, [10, 11, 12, 13, 14, 15])  # B: priority 6
    node_a = tree.root_node.children[0]
    node_b = tree.root_node.children[10]
    _reuse(tree, [0, 1, 2, 3])
    _reuse(tree, [10, 11, 12, 13, 14, 15])

    assert torch.equal(_collect(tree, 1), _tensor([3]))
    assert node_a.priority == pytest.approx(3.0)
    assert torch.equal(_collect(tree, 1), _tensor([2]))
    assert node_a.priority == pytest.approx(2.0)

    assert torch.equal(node_a.token_id_key, _tensor([0, 1]))
    assert torch.equal(node_b.token_id_key, _tensor([10, 11, 12, 13, 14, 15]))


def test_gdsf_elastic_retained_prefix_is_reusable_and_rescored_on_reuse():
    tree = _make_tree(rank_in_node=0, eviction_policy="gdsf_compute", elastic_min_retain_tokens=1)

    _insert(tree, [0, 1, 2, 3, 4, 5])
    node = tree.root_node.children[0]
    _reuse(tree, [0, 1, 2, 3, 4, 5])
    _collect(tree, 2)  # 裁到 [0, 1, 2, 3]

    # 保留的前缀依然能命中，并且这次命中是一次正常的复用: freq 增长、age 重新锚定。
    matched = _reuse(tree, [0, 1, 2, 3])
    assert matched is node
    assert node.freq == 2
    assert node.priority == pytest.approx(2 * (4 + 0))  # age 0 + freq 2 * (e 4 + s 0)


def test_gdsf_without_elastic_still_evicts_whole_leaf():
    tree = _make_tree(rank_in_node=0, eviction_policy="gdsf_compute", enable_elastic_eviction=False)

    _insert(tree, [0, 1, 2, 3, 4, 5])
    _reuse(tree, [0, 1, 2, 3, 4, 5])

    freed = _collect(tree, 2)
    assert len(freed) == 6
    assert len(tree.root_node.children) == 0


if __name__ == "__main__":
    pytest.main()
