# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/router/radix_cache.py
import torch
import numpy as np
import collections
from typing import Tuple, Dict, Set, List, Optional, Union
from sortedcontainers import SortedSet
from .shared_arr import SharedArray
from .radix_eviction_policy import get_eviction_policy


class UniqueTimeIdGenerator:
    def __init__(self):
        self.counter = 0

    def generate_time_id(self):
        self.counter += 1
        return self.counter


time_gen = UniqueTimeIdGenerator()


class TreeNode:
    def __init__(self):
        self.children: Dict[int, TreeNode] = {}  # 这里的键 为 token_id_key 的第一个元素
        self.parent: TreeNode = None
        self.token_id_key: torch.Tensor = None
        self.token_mem_index_value: torch.Tensor = None  # 用于记录存储的 token_index 为每个元素在 token mem 中的index位置
        self.ref_counter = 0
        self.time_id = time_gen.generate_time_id()  # 用于标识时间周期

        self.node_value_len = 0
        self.node_prefix_total_len = 0

        # Eviction-policy bookkeeping. Unused by the default LRU policy; the
        # compute-aware GDSF policy uses these to score eviction candidates.
        # ``age`` is the frozen snapshot of the aging clock at the node's last
        # admission/reuse, so geometry-only rescoring (splits) preserves history.
        self.freq = 0
        self.age = 0.0
        self.priority = 0.0

        # Elastic eviction bookkeeping: per-step cap is enforced via
        # elastically_used_this_call dict; no persistent lifetime counter needed.
        # Keeping attribute for backward compatibility with merged nodes.
        self.elastic_evicted = 0

    def get_compare_key(self):
        return (0 if self.ref_counter == 0 else 1, len(self.children), self.time_id)

    def split_node(self, prefix_len):
        split_parent_node = TreeNode()
        split_parent_node.parent = self.parent
        split_parent_node.parent.children[self.token_id_key[0].item()] = split_parent_node
        split_parent_node.token_id_key = self.token_id_key[0:prefix_len]
        split_parent_node.token_mem_index_value = self.token_mem_index_value[0:prefix_len]
        split_parent_node.children = {}
        split_parent_node.children[self.token_id_key[prefix_len].item()] = self
        split_parent_node.ref_counter = self.ref_counter
        # The split prefix inherits the durable access history (frequency + frozen
        # age) of the node it came from. Its priority is (re)scored by the policy's
        # on_split hook, since its geometry differs from the pre-split node.
        split_parent_node.freq = self.freq
        split_parent_node.age = self.age
        # Elastic history preservation not needed for per-step cap, but keep
        # attribute for compatibility.
        split_parent_node.elastic_evicted = getattr(self, "elastic_evicted", 0)

        new_len = len(split_parent_node.token_mem_index_value)
        split_parent_node.node_value_len = new_len
        split_parent_node.node_prefix_total_len = split_parent_node.parent.node_prefix_total_len + new_len

        self.token_id_key = self.token_id_key[prefix_len:]
        self.token_mem_index_value = self.token_mem_index_value[prefix_len:]
        self.parent = split_parent_node
        new_len = len(self.token_mem_index_value)
        self.node_value_len = new_len
        self.node_prefix_total_len = self.parent.node_prefix_total_len + new_len
        return split_parent_node

    def add_and_return_new_child(self, token_id_key, token_mem_index_value):
        child = TreeNode()
        child.token_id_key = token_id_key
        child.token_mem_index_value = token_mem_index_value
        first_token_key = child.token_id_key[0].item()
        assert first_token_key not in self.children.keys()
        self.children[first_token_key] = child
        child.parent = self

        new_len = len(child.token_mem_index_value)
        child.node_value_len = new_len
        child.node_prefix_total_len = child.parent.node_prefix_total_len + new_len
        return child

    def remove_child(self, child_node: "TreeNode"):
        del self.children[child_node.token_id_key[0].item()]
        child_node.parent = None
        return

    def update_time(self):
        self.time_id = time_gen.generate_time_id()

    def is_leaf(self):
        return len(self.children) == 0


def match(t1: torch.Tensor, t2: torch.Tensor) -> int:
    # Ensure same shape for comparison: flatten and get min length
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    min_len = min(t1_flat.size(0), t2_flat.size(0))

    # Compare elements and find first mismatch
    diff = t1_flat[:min_len] != t2_flat[:min_len]
    mismatch_indices = torch.nonzero(diff)

    if mismatch_indices.numel() == 0:
        return min_len  # All matched up to min_len
    else:
        return mismatch_indices[0].item()


class RadixCache:
    """
    unique_name 主要用于解决单机，多实列部署时的shm冲突
    """

    def __init__(
        self,
        unique_name,
        total_token_num,
        rank_in_node,
        mem_manager=None,
        eviction_policy="lru",
        enable_elastic_eviction: bool = False,
        elastic_min_retained_tokens: int = 1,
        elastic_max_evict_tokens_per_leaf: Optional[int] = None,
    ):
        from lightllm.common.kv_cache_mem_manager import MemoryManager

        self.total_token_num = total_token_num
        self.mem_manager: MemoryManager = mem_manager
        self._key_dtype = torch.int64
        self._value_dtype = torch.int64

        # Pluggable eviction policy. Defaults to "lru" (the historical behavior);
        # "gdsf_compute" enables the optional compute-aware GDSF policy.
        self.eviction_policy = get_eviction_policy(eviction_policy)

        # Elastic radix eviction: reclaim tail of cold leaves while preserving prefix.
        # Disabled by default to preserve historical whole-leaf behavior.
        self.enable_elastic_eviction = bool(enable_elastic_eviction)
        self.elastic_min_retained_tokens = max(1, int(elastic_min_retained_tokens))
        if elastic_max_evict_tokens_per_leaf is not None:
            self.elastic_max_evict_tokens_per_leaf = max(1, int(elastic_max_evict_tokens_per_leaf))
        else:
            self.elastic_max_evict_tokens_per_leaf = None

        self.root_node = TreeNode()
        self.root_node.token_id_key = torch.zeros((0,), device="cpu", dtype=self._key_dtype)
        self.root_node.token_mem_index_value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
        self.root_node.ref_counter = 1  # 初始化为 1 保证永远不会被 evict 掉

        # 自定义比较器: eviction order is delegated to the policy.
        self.evict_tree_set: Set[TreeNode] = SortedSet(key=lambda x: self.eviction_policy.get_compare_key(x))
        self.evict_tree_set.add(self.root_node)

        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.refed_tokens_num.arr[0] = 0
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )
        self.tree_total_tokens_num.arr[0] = 0

    def insert(self, key, value=None) -> Tuple[int, Optional[TreeNode]]:
        if value is None:
            value = key

        assert len(key) == len(value)  # and len(key) >= 1
        if len(key) == 0:
            return 0, None
        return self._insert_helper(self.root_node, key, value)

    def _insert_helper(self, node: TreeNode, key, value) -> Tuple[int, Optional[TreeNode]]:
        handle_stack = collections.deque()
        update_list = collections.deque()
        handle_stack.append((node, key, value))

        ans_prefix_len = 0
        ans_node = None

        while len(handle_stack) != 0:
            node, key, value = handle_stack.popleft()
            ans_tuple = self._insert_helper_no_recursion(node=node, key=key, value=value)
            if len(ans_tuple) == 4:
                (_prefix_len, new_node, new_key, new_value) = ans_tuple
                ans_prefix_len += _prefix_len
                handle_stack.append((new_node, new_key, new_value))
            else:
                _prefix_len, ans_node = ans_tuple
                ans_prefix_len += _prefix_len

            update_list.append(node)

        while len(update_list) != 0:
            cur_node: TreeNode = update_list.pop()
            # Insertion (KV write-back) is not a reuse: refresh bookkeeping only.
            self.eviction_policy.touch_node(cur_node)
            if cur_node.is_leaf():
                self.evict_tree_set.add(cur_node)

        assert ans_node is not None

        return ans_prefix_len, ans_node

    def _insert_helper_no_recursion(
        self, node: TreeNode, key: torch.Tensor, value: torch.Tensor
    ) -> Union[Tuple[int, Optional[TreeNode]], Tuple[int, TreeNode, torch.Tensor, torch.Tensor]]:
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        first_key_id = key[0].item()
        if first_key_id in node.children.keys():
            child: TreeNode = node.children[first_key_id]
            prefix_len = match(key, child.token_id_key)
            if prefix_len == len(key):
                if prefix_len == len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)
                    # Re-inserting an already-cached prefix is not a reuse.
                    self.eviction_policy.touch_node(child)
                    if child.is_leaf():
                        self.evict_tree_set.add(child)
                    return prefix_len, child
                elif prefix_len < len(child.token_id_key):
                    if child.is_leaf():
                        self.evict_tree_set.discard(child)

                    split_parent_node = child.split_node(prefix_len)
                    # Both halves changed geometry: rescore each, preserving history.
                    self.eviction_policy.on_split(split_parent_node, child)

                    if split_parent_node.is_leaf():
                        self.evict_tree_set.add(split_parent_node)
                    if child.is_leaf():
                        self.evict_tree_set.add(child)

                    return prefix_len, split_parent_node
                else:
                    assert False, "can not run to here"

            elif prefix_len < len(key) and prefix_len < len(child.token_id_key):
                if child.is_leaf():
                    self.evict_tree_set.discard(child)

                key = key[prefix_len:]
                value = value[prefix_len:]
                split_parent_node = child.split_node(prefix_len)
                # Both halves changed geometry: rescore each, preserving history.
                self.eviction_policy.on_split(split_parent_node, child)
                new_node = split_parent_node.add_and_return_new_child(key, value)
                # Admitting a brand-new node anchors its aging baseline, not reuse.
                self.eviction_policy.init_node(new_node)
                # update total token num
                self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)

                if split_parent_node.is_leaf():
                    self.evict_tree_set.add(split_parent_node)
                if new_node.is_leaf():
                    self.evict_tree_set.add(new_node)

                if child.is_leaf():
                    self.evict_tree_set.add(child)
                return prefix_len, new_node
            elif prefix_len < len(key) and prefix_len == len(child.token_id_key):
                return (prefix_len, child, key[prefix_len:], value[prefix_len:])
            else:
                assert False, "can not run to here"

        else:
            new_node = node.add_and_return_new_child(key, value)
            # Admitting a brand-new node anchors its aging baseline, not reuse.
            self.eviction_policy.init_node(new_node)
            # update total token num
            self.tree_total_tokens_num.arr[0] += len(new_node.token_mem_index_value)
            if new_node.is_leaf():
                self.evict_tree_set.add(new_node)
            return 0, new_node

    def match_prefix(self, key, update_refs=False):
        assert len(key) != 0
        ans_value_list = []
        tree_node = self._match_prefix_helper(self.root_node, key, ans_value_list, update_refs=update_refs)
        if tree_node != self.root_node:
            if len(ans_value_list) != 0:
                value = torch.concat(ans_value_list)
            else:
                value = torch.zeros((0,), device="cpu", dtype=self._value_dtype)
            return tree_node, len(value), value
        else:
            if update_refs:
                self.dec_node_ref_counter(self.root_node)
            return None, 0, None

    def _match_prefix_helper(
        self, node: TreeNode, key: torch.Tensor, ans_value_list: list, update_refs=False
    ) -> TreeNode:
        handle_stack = collections.deque()
        update_list = collections.deque()
        handle_stack.append((node, key))

        ans_node = None

        while len(handle_stack) != 0:
            node, key = handle_stack.popleft()
            ans_tuple = self._match_prefix_helper_no_recursion(
                node=node, key=key, ans_value_list=ans_value_list, update_refs=update_refs
            )
            if isinstance(ans_tuple, tuple):
                new_node, new_key = ans_tuple
                handle_stack.append((new_node, new_key))
            else:
                ans_node = ans_tuple

            update_list.append(node)

        while len(update_list) != 0:
            cur_node: TreeNode = update_list.pop()
            # Only a real prompt-cache hit (update_refs=True) counts as reuse;
            # a read-only probe (update_refs=False) is bookkeeping only.
            if update_refs:
                self.eviction_policy.record_reuse(cur_node)
            else:
                self.eviction_policy.touch_node(cur_node)
            if cur_node.is_leaf():
                self.evict_tree_set.add(cur_node)

        return ans_node

    def _match_prefix_helper_no_recursion(
        self, node: TreeNode, key: torch.Tensor, ans_value_list: list, update_refs=False
    ) -> TreeNode:
        if node.is_leaf():
            self.evict_tree_set.discard(node)

        if update_refs:
            node.ref_counter += 1
            # from 0 to 1 need update refs token num
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)

        if len(key) == 0:
            return node

        first_key_id = key[0].item()
        if first_key_id not in node.children.keys():
            return node
        else:
            child = node.children[first_key_id]
            prefix_len = match(key, child.token_id_key)
            if prefix_len == len(child.token_id_key):
                ans_value_list.append(child.token_mem_index_value)
                return (child, key[prefix_len:])
            elif prefix_len < len(child.token_id_key):
                if child.is_leaf():
                    self.evict_tree_set.discard(child)

                split_parent_node = child.split_node(prefix_len)
                # Both halves changed geometry: rescore each, preserving history.
                self.eviction_policy.on_split(split_parent_node, child)
                ans_value_list.append(split_parent_node.token_mem_index_value)

                if update_refs:
                    split_parent_node.ref_counter += 1
                    # from 0 to 1 need update refs token num
                    if split_parent_node.ref_counter == 1:
                        self.refed_tokens_num.arr[0] += len(split_parent_node.token_mem_index_value)
                    # A partial hit genuinely reuses the matched shared prefix. This
                    # terminal node is not on the update_list, so count it here.
                    self.eviction_policy.record_reuse(split_parent_node)

                if child.is_leaf():
                    self.evict_tree_set.add(child)
                if split_parent_node.is_leaf():
                    self.evict_tree_set.add(split_parent_node)

                return split_parent_node
            else:
                assert False, "error state"

    def evict(self, need_remove_tokens, evict_callback):
        if self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0] < need_remove_tokens:
            assert False, f"""can not free tree tokens {need_remove_tokens},
                              tree_total_tokens_num {self.tree_total_tokens_num.arr[0]},
                              refed_tokens_num {self.refed_tokens_num.arr[0]}"""
        num_evicted = 0
        # Track per-leaf elastic usage within this evict() call.
        # Prevents GDSF (which keeps priority 0 for freq-0 leaves) from
        # repeatedly selecting the same truncated leaf and reclaiming beyond
        # per-leaf cap within a single eviction pass, and avoids repeated
        # cloning of the same leaf. Persistent per-node elastic_evicted caps
        # total elastic reclaim over the leaf's lifetime.
        # For distribution, elastically truncated leaves are deferred (not
        # immediately reinserted) so the same cold leaf is not selected again
        # within the same evict() call; the next victim is the next coldest
        # leaf. This ensures cap 2 with need 4 across two 8-token leaves
        # yields 2+2, not 2+6 from one leaf.
        elastically_used_this_call: Dict[int, int] = {}
        deferred_reinsert: List[TreeNode] = []
        while num_evicted < need_remove_tokens:
            # If all remaining leaves in the set are referenced, the deferred
            # unreferenced truncated leaves must be reinserted first. Otherwise
            # deferring would expose a referenced leaf as the next victim,
            # violating the invariant that only ref_counter==0 leaves are evicted.
            if deferred_reinsert and self.evict_tree_set:
                # Peek at the smallest element; ordering is (ref==0?0:1, ...)
                # so if the first is referenced, no unreferenced remain.
                first = next(iter(self.evict_tree_set))
                if first.ref_counter != 0:
                    for n in deferred_reinsert:
                        self.evict_tree_set.add(n)
                    deferred_reinsert.clear()
            if not self.evict_tree_set:
                # No more distinct leaves for elastic; reinsert deferred truncated
                # leaves so they can be considered for whole-leaf eviction if still needed.
                # Their LRU recency was preserved (not promoted), so they remain coldest.
                for n in deferred_reinsert:
                    self.evict_tree_set.add(n)
                deferred_reinsert.clear()
                if not self.evict_tree_set:
                    break
            node: TreeNode = self.evict_tree_set.pop(0)
            assert (
                node.ref_counter == 0 and len(node.children) == 0 and node != self.root_node
            ), "error evict tree node state"
            remaining = need_remove_tokens - num_evicted
            node_len = len(node.token_mem_index_value)
            # Elastic: if we only need a tail, shrink leaf and keep reusable prefix.
            # Conditions: enabled, need < node_len, can retain at least min_retained,
            # and per-leaf cap (persistent + per-call) not exhausted.
            node_id = id(node)
            already_this_call = elastically_used_this_call.get(node_id, 0)
            can_elastic = (
                self.enable_elastic_eviction
                and remaining < node_len
                and node_len > self.elastic_min_retained_tokens
            )
                # Enforce per-leaf per-step cap: at most cap tokens elastically
            # reclaimed from this leaf within this evict() call. Advertised as
            # per-step, not lifetime.
            if can_elastic and self.elastic_max_evict_tokens_per_leaf is not None:
                remaining_cap = self.elastic_max_evict_tokens_per_leaf - already_this_call
                if remaining_cap <= 0:
                    can_elastic = False

            if can_elastic:
                # Compute evict_len limited by remaining need, min_retained, and per-step cap.
                max_by_retained = node_len - self.elastic_min_retained_tokens
                evict_len = min(remaining, max_by_retained)
                if self.elastic_max_evict_tokens_per_leaf is not None:
                    remaining_cap = self.elastic_max_evict_tokens_per_leaf - already_this_call
                    evict_len = min(evict_len, remaining_cap)
                retain_len = node_len - evict_len
                if evict_len <= 0 or retain_len < self.elastic_min_retained_tokens:
                    can_elastic = False
                # Avoid pointless trim: if elastic cannot fully satisfy remaining
                # and other leaves cannot cover the remainder, the same leaf will
                # be whole-evicted next, making the elastic clone wasteful.
                # This covers both min_retained and per-step cap limits.
                if can_elastic and evict_len < remaining:
                    other_available = 0
                    # Sum tokens in remaining evict set + deferred truncated leaves
                    # (current node already popped, not counted)
                    for n in self.evict_tree_set:
                        other_available += len(n.token_mem_index_value)
                    for n in deferred_reinsert:
                        other_available += len(n.token_mem_index_value)
                    if other_available < remaining - evict_len:
                        # Not enough in others (considering both elastic and whole
                        # from others); this leaf will be whole-evicted immediately
                        # after trim, so skip elastic and do whole directly.
                        # Note: we check if elastic was limited (by min_retained or cap)
                        # — if evict_len == remaining, it fully satisfies need, not pointless.
                        can_elastic = False

                if can_elastic:
                    # Split suffix to free, keep prefix in same node object.
                    # Invariant: prefix remains contiguous from node.parent, children stays empty,
                    # ref_counter stays 0, so match_prefix and future insert still work.
                    # Avoid cloning suffix_key (unused) and clone prefix only once.
                    suffix_value = node.token_mem_index_value[retain_len:]
                    # Truncate leaf in-place to reusable prefix — single clone per tensor.
                    # Clone creates new smaller allocation so old tail storage can be freed.
                    node.token_id_key = node.token_id_key[:retain_len].clone()
                    node.token_mem_index_value = node.token_mem_index_value[:retain_len].clone()
                    node.node_value_len = retain_len
                    # node_prefix_total_len stays parent_total + retain_len
                    node.node_prefix_total_len = node.parent.node_prefix_total_len + retain_len
                    # Track per-step usage; do not enforce lifetime cap.
                    elastically_used_this_call[node_id] = already_this_call + evict_len
                    # Rescore for new geometry without counting as reuse and without
                    # refreshing LRU recency. The cold victim's prefix must stay cold;
                    # promoting to MRU would make the next eviction incorrectly target
                    # newer leaves. Defer reinsertion so same leaf is not immediately
                    # reselected within this call, distributing across leaves.
                    self.eviction_policy.on_elastic_truncate(node)
                    deferred_reinsert.append(node)
                    # Account and callback only for evicted tail
                    num_evicted += evict_len
                    self.tree_total_tokens_num.arr[0] -= evict_len
                    evict_callback(suffix_value)
                    continue

            # Default: whole-leaf eviction (historical behavior)
            self.eviction_policy.on_evict(node)
            num_evicted += len(node.token_mem_index_value)
            evict_callback(node.token_mem_index_value)
            # update total token num
            self.tree_total_tokens_num.arr[0] -= len(node.token_mem_index_value)
            parent_node: TreeNode = node.parent
            parent_node.remove_child(node)
            if parent_node.is_leaf():
                self.evict_tree_set.add(parent_node)

        # Reinsert any elastically truncated leaves deferred during this call.
        for n in deferred_reinsert:
            self.evict_tree_set.add(n)
        return

    def _try_merge(self, child_node: TreeNode) -> Optional[TreeNode]:
        """
        合并条件:
        1. 父节点不是根节点。
        2. 父节点的引用计数为 0。
        3. 子节点的引用计数为 0。
        4. 父节点只有一个子节点 (即 child_node)。
        """
        parent_node = child_node.parent
        # 条件检查
        if (
            parent_node is None
            or parent_node == self.root_node
            or parent_node.ref_counter != 0
            or len(parent_node.children) != 1
            or child_node.ref_counter != 0
        ):
            return None

        if child_node.is_leaf():
            self.evict_tree_set.discard(child_node)

        child_node.token_id_key = torch.cat([parent_node.token_id_key, child_node.token_id_key])
        child_node.token_mem_index_value = torch.cat(
            [parent_node.token_mem_index_value, child_node.token_mem_index_value]
        )
        child_node.node_value_len = len(child_node.token_mem_index_value)
        # Keep attribute for compatibility; per-step cap does not need merge logic.
        child_node.elastic_evicted = max(
            getattr(parent_node, "elastic_evicted", 0), getattr(child_node, "elastic_evicted", 0)
        )
        self.eviction_policy.on_merge(parent_node, child_node)

        grandparent_node = parent_node.parent
        key_in_grandparent = parent_node.token_id_key[0].item()
        grandparent_node.children[key_in_grandparent] = child_node
        child_node.parent = grandparent_node

        parent_node.parent = None

        if child_node.is_leaf():
            self.evict_tree_set.add(child_node)

        return child_node

    def merge_unreferenced_nodes(self):
        worklist = collections.deque(
            [
                node
                for node in self.evict_tree_set
                if node.ref_counter == 0 and node.parent is not None and node.parent != self.root_node
            ]
        )

        while worklist:
            node = worklist.popleft()
            if node.parent is None:
                continue
            merged_node = self._try_merge(node)
            if merged_node:
                worklist.append(merged_node)

    def clear_tree_nodes(self):
        """
        该函数只在测试时调用
        """
        while True:
            node: TreeNode = self.evict_tree_set.pop(0)
            if node != self.root_node:
                parent_node: TreeNode = node.parent
                parent_node.remove_child(node)
                if parent_node.is_leaf():
                    self.evict_tree_set.add(parent_node)
            else:
                break

        self.tree_total_tokens_num.arr[0] = 0
        self.refed_tokens_num.arr[0] = 0
        return

    def flush_cache(self):
        self.free_radix_cache_to_get_enough_token(need_token_num=self.total_token_num)
        return

    def dec_node_ref_counter(self, node: TreeNode):
        if node is None:
            return
        # 如果减引用的是叶节点，需要先从 evict_tree_set 中移除
        old_node = node
        if old_node.is_leaf():
            self.evict_tree_set.discard(old_node)

        while node is not None:
            if node.ref_counter == 1:
                self.refed_tokens_num.arr[0] -= len(node.token_mem_index_value)
            node.ref_counter -= 1
            node = node.parent

        # 加回。
        if old_node.is_leaf():
            self.evict_tree_set.add(old_node)
        return

    def add_node_ref_counter(self, node: TreeNode):
        if node is None:
            return
        # 如果减引用的是叶节点，需要先从 evict_tree_set 中移除
        old_node = node
        if old_node.is_leaf():
            self.evict_tree_set.discard(old_node)

        while node is not None:
            if node.ref_counter == 0:
                self.refed_tokens_num.arr[0] += len(node.token_mem_index_value)
            node.ref_counter += 1
            node = node.parent

        # 加回。
        if old_node.is_leaf():
            self.evict_tree_set.add(old_node)
        return

    def get_mem_index_value_by_node(self, node: TreeNode) -> Optional[torch.Tensor]:
        if node is None:
            return None

        ans_list = []
        while node is not None:
            ans_list.append(node.token_mem_index_value)
            node = node.parent

        ans_list.reverse()
        return torch.concat(ans_list, dim=0)

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def print_self(self, indent=0):
        self._print_helper(self.root_node, indent)

    def _print_helper(self, node: TreeNode, indent):
        print(
            " " * indent,
            f"k: {node.token_id_key[0:10]} v: {node.token_mem_index_value[0:10]} refs: {node.ref_counter} \
            time_id: {node.time_id} prefix_total_len: {node.node_prefix_total_len} \
            node_value_len: {node.node_value_len}",
        )
        for _, child in node.children.items():
            self._print_helper(child, indent=indent + 2)
        return

    def free_radix_cache_to_get_enough_token(self, need_token_num):
        assert self.mem_manager is not None
        if need_token_num > self.mem_manager.allocator.can_use_mem_size:
            need_evict_token_num = need_token_num - self.mem_manager.allocator.can_use_mem_size
            release_mems = []

            def release_mem(mem_index):
                release_mems.append(mem_index)
                return

            self.evict(need_evict_token_num, release_mem)
            mem_index = torch.concat(release_mems)
            self.mem_manager.free(mem_index)
        return


class _RadixCacheReadOnlyClient:
    """
    router 端只读用的客户端，用于从共享内存中读取树结构中的信息，用于进行prompt cache 的调度估计。
    """

    def __init__(self, unique_name, total_token_num, rank_in_node):
        self.refed_tokens_num = SharedArray(f"{unique_name}_refed_tokens_num_{rank_in_node}", (1,), dtype=np.int64)
        self.tree_total_tokens_num = SharedArray(
            f"{unique_name}_tree_total_tokens_num_{rank_in_node}", (1,), dtype=np.int64
        )

    def get_refed_tokens_num(self):
        return self.refed_tokens_num.arr[0]

    def get_tree_total_tokens_num(self):
        return self.tree_total_tokens_num.arr[0]

    def get_unrefed_tokens_num(self):
        return self.tree_total_tokens_num.arr[0] - self.refed_tokens_num.arr[0]


class RadixCacheReadOnlyClient:
    def __init__(self, unique_name, total_token_num, node_world_size, dp_world_size):
        self.dp_rank_clients: List[_RadixCacheReadOnlyClient] = [
            _RadixCacheReadOnlyClient(unique_name, total_token_num, rank_in_node)
            for rank_in_node in range(0, node_world_size, dp_world_size)
        ]

    def get_refed_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_refed_tokens_num()

    def get_tree_total_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_tree_total_tokens_num()

    def get_unrefed_tokens_num(self, dp_rank_in_node):
        return self.dp_rank_clients[dp_rank_in_node].get_unrefed_tokens_num()
