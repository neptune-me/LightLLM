"""Pluggable eviction policies for the GPU radix prompt cache.

The radix cache keeps its evictable leaves in a ``SortedSet`` ordered by a
per-node *compare key*. Popping index 0 yields the next victim. A policy owns:

* ``compare_key(node)`` - the sort key (smaller == evicted sooner). The first
  two tuple components MUST stay ``(ref_counter > 0, len(children))`` so that
  referenced nodes and non-leaves can never be chosen before a free leaf; this
  is a structural invariant of ``RadixCache.evict``.
* ``touch(node)`` - recency/structural signal, fired on every insert,
  re-insert and (read-only or not) prefix match.
* ``on_access(node)`` - genuine-reuse signal, fired only for
  ``match_prefix(update_refs=True)``. Size-aware LFU counts frequency here so
  that inserts and read-only scheduler probes do not pollute the score.
* ``on_new_node(node, split_from=...)`` - called right after a node is created.
* ``on_merge(parent, child)`` - called when two nodes are merged.
* ``on_evict(node)`` - called when a node is evicted (lets a policy age).

Two policies are provided:

* :class:`LRUEvictionPolicy` - the historical behavior (recency via ``time_id``).
* :class:`GDSFEvictionPolicy` - an experimental, configurable Greedy-Dual-Size
  Frequency policy: ``priority = aging_clock + freq * cost / size ** size_exp``.

Only the compare-key components change; all token bookkeeping in ``RadixCache``
is policy-independent.
"""
from typing import Optional, Tuple

from .radix_cache import TreeNode


class EvictionPolicy:
    """Base class. Default hooks reproduce plain LRU-by-recency behavior."""

    def compare_key(self, node: "TreeNode") -> Tuple:
        raise NotImplementedError

    def touch(self, node: "TreeNode") -> None:
        # Recency / structural signal: fired on every insert, re-insert and probe.
        node.update_time()

    def on_access(self, node: "TreeNode") -> None:
        # Genuine reuse signal: fired only for match_prefix(update_refs=True),
        # i.e. when a request actually pins and reuses the prefix.
        pass

    def on_new_node(self, node: "TreeNode", split_from: Optional["TreeNode"] = None) -> None:
        pass

    def on_merge(self, parent: "TreeNode", child: "TreeNode") -> None:
        child.time_id = max(parent.time_id, child.time_id)

    def on_evict(self, node: "TreeNode") -> None:
        pass


class LRUEvictionPolicy(EvictionPolicy):
    """Least-Recently-Used, identical to the legacy ``TreeNode.get_compare_key``.

    Evicts the unreferenced leaf with the smallest ``time_id`` (oldest touch).
    """

    def compare_key(self, node: "TreeNode") -> Tuple:
        return (0 if node.ref_counter == 0 else 1, len(node.children), node.time_id)


class GDSFEvictionPolicy(EvictionPolicy):
    """Experimental GDSF-style size-aware LFU policy.

    Each node stores a *frozen* access clock (``_gdsf_clock``) captured at its
    last access and an access frequency (``_gdsf_freq``). Its priority is::

        priority = _gdsf_clock + _gdsf_freq / size ** size_exponent

    where ``size`` is the number of tokens the node holds. The global
    ``aging_clock`` is bumped to the priority of every evicted node, so newly
    created / reaccessed nodes inherit a rising floor (classic GDSF inflation)
    that prevents stale-but-once-hot prefixes from polluting the cache forever.

    Tunable:

    * ``size_exponent`` - ``0`` recovers pure aged-LFU (no size awareness),
      ``1`` is classic GDSF-Size (default 1.0).

    Note: classic GDSF also carries a per-object fetch cost ``C(i)``. A *global*
    constant cost is deliberately omitted because it rescales every node's
    priority uniformly and therefore cannot change eviction ordering; the only
    size-relevant per-node signal here is ``size`` itself, controlled by
    ``size_exponent``.
    """

    def __init__(self, size_exponent: float = 1.0):
        self.size_exponent = float(size_exponent)
        self.aging_clock = 0.0

    @staticmethod
    def _size(node: "TreeNode") -> int:
        # number of tokens stored at this node; never zero for a real leaf
        if node.token_mem_index_value is not None:
            return max(len(node.token_mem_index_value), 1)
        return 1

    def priority(self, node: "TreeNode") -> float:
        size = self._size(node)
        return node._gdsf_clock + node._gdsf_freq / (size ** self.size_exponent)

    def compare_key(self, node: "TreeNode") -> Tuple:
        # keep the structural (ref, children) prefix; final time_id breaks ties
        return (
            0 if node.ref_counter == 0 else 1,
            len(node.children),
            self.priority(node),
            node.time_id,
        )

    def touch(self, node: "TreeNode") -> None:
        # Recency only. Inserts, re-inserts and read-only probes must not affect
        # the size-aware LFU score, so frequency and the aging clock are left
        # untouched here (see on_access).
        node.update_time()

    def on_access(self, node: "TreeNode") -> None:
        node._gdsf_freq += 1
        node._gdsf_clock = self.aging_clock

    def on_new_node(self, node: "TreeNode", split_from: Optional["TreeNode"] = None) -> None:
        if split_from is not None:
            # a split node covers part of an existing prefix: inherit its history
            node._gdsf_freq = split_from._gdsf_freq
            node._gdsf_clock = split_from._gdsf_clock
        else:
            node._gdsf_freq = 1
            node._gdsf_clock = self.aging_clock

    def on_merge(self, parent: "TreeNode", child: "TreeNode") -> None:
        child.time_id = max(parent.time_id, child.time_id)
        child._gdsf_freq = max(parent._gdsf_freq, child._gdsf_freq)
        child._gdsf_clock = max(parent._gdsf_clock, child._gdsf_clock)

    def on_evict(self, node: "TreeNode") -> None:
        # GDSF inflation: raise the floor to the evicted node's priority
        self.aging_clock = max(self.aging_clock, self.priority(node))


_POLICIES = {
    "lru": LRUEvictionPolicy,
    "gdsf": GDSFEvictionPolicy,
}


def create_eviction_policy(name: str = "lru", **kwargs) -> EvictionPolicy:
    """Build an eviction policy by name.

    ``lru`` ignores kwargs; ``gdsf`` accepts ``size_exponent``.
    """
    name = (name or "lru").lower()
    if name not in _POLICIES:
        raise ValueError(f"unknown radix cache eviction policy {name!r}; choices: {sorted(_POLICIES)}")
    if name == "lru":
        return LRUEvictionPolicy()
    return _POLICIES[name](**kwargs)
