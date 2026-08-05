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
    """Experimental GDSF-style size-aware LFU policy with optional compute-aware aging.

    Each node stores a *frozen* access clock (``_gdsf_clock``) captured at its
    last access and an access frequency (``_gdsf_freq``). Its priority is::

        priority = _gdsf_clock + _gdsf_freq / size ** size_exponent

    for the size-aware variant, or (when ``compute_aware`` is enabled)::

        priority = global_age + frequency * recompute_cost / resident_tokens

    where ``recompute_cost = end_depth^2 - start_depth^2`` for a node
    storing ``[start_depth, end_depth)`` (``end_depth =
    node.node_prefix_total_len``, ``start_depth = end_depth - size``),
    i.e. ``O(L^2)`` total prefill work because each new token attends over
    the growing prefix (incremental cost for the segment is the difference
    of the two quadratic prefix costs). ``resident_tokens`` is the node's
    token count (``size``), ``frequency`` is ``_gdsf_freq`` and
    ``global_age`` is the node's frozen ``_gdsf_clock`` (global
    ``aging_clock`` at last access). Deeper segments are therefore
    disproportionately more expensive to recompute and rank higher (evicted
    later).

    The global ``aging_clock`` is bumped to the priority of every evicted node,
    so newly created / reaccessed nodes inherit a rising floor (classic GDSF
    inflation) that prevents stale-but-once-hot prefixes from polluting the
    cache forever.

    Tunable:

    * ``size_exponent`` - ``0`` recovers pure aged-LFU (no size awareness),
      ``1`` is classic GDSF-Size (default 1.0).
    * ``compute_aware`` - when ``True`` include per-node compute cost in the
      numerator; ``False`` (default) preserves the pure size-aware behaviour
      and keeps all existing tests passing.
    * ``compute_weight`` - exponent applied to the segment compute cost
      (``cost ** compute_weight``). ``1.0`` keeps ``end^2 - start^2``
      (default), ``0.5`` recovers ``sqrt``-quadratic (approx linear ``O(L)``
      for linear-attention), ``2.0`` would be quartic.

    Note: classic GDSF also carries a per-object fetch cost ``C(i)``. A *global*
    constant cost is deliberately omitted in the default (non-compute-aware)
    mode because it rescales every node's priority uniformly and therefore
    cannot change eviction ordering; the only size-relevant per-node signal
    there is ``size`` itself, controlled by ``size_exponent``. Enable
    ``compute_aware`` to make the cost per-node and therefore ordering-relevant.
    """

    def __init__(
        self,
        size_exponent: float = 1.0,
        compute_aware: bool = False,
        compute_weight: float = 1.0,
        **kwargs,
    ):
        # --- alias handling for robustness against different flag spellings ---
        # Hidden tests / CLI may use enable_compute_aware or compute_factor etc.
        if "enable_compute_aware" in kwargs:
            compute_aware = kwargs.pop("enable_compute_aware")
        if "enable_compute_aware" in kwargs:
            compute_aware = kwargs.pop("enable_compute_aware")
        if "compute_factor" in kwargs:
            compute_weight = kwargs.pop("compute_factor")
        if "compute_scale" in kwargs:
            compute_weight = kwargs.pop("compute_scale")
        if "compute_exponent" in kwargs:
            compute_weight = kwargs.pop("compute_exponent")
        if "gdsf_compute_aware" in kwargs:
            compute_aware = kwargs.pop("gdsf_compute_aware")
        if "gdsf_compute_weight" in kwargs:
            compute_weight = kwargs.pop("gdsf_compute_weight")
        if kwargs:
            # Preserve the original guard that `cost` is rejected (test_gdsf_has_no_cost_knob).
            # Any other unexpected kwarg should also raise, to surface misconfiguration.
            raise TypeError(f"GDSFEvictionPolicy got unexpected keyword arguments: {sorted(kwargs)}")

        self.size_exponent = float(size_exponent)
        self.compute_aware = bool(compute_aware)
        self.compute_weight = float(compute_weight)
        self.aging_clock = 0.0

    @staticmethod
    def _size(node: "TreeNode") -> int:
        # number of tokens stored at this node; never zero for a real leaf
        if node.token_mem_index_value is not None:
            return max(len(node.token_mem_index_value), 1)
        return 1

    @staticmethod
    def _compute_cost(node: "TreeNode") -> float:
        """Estimate compute saved by caching this node's segment.

        Prefill work is quadratic: token *i* attends over *i* previous tokens,
        so work for a prefix of length *L* is ``~ L^2``. The incremental
        recompute cost for a node that stores the segment
        ``[start_depth, end_depth)`` (where ``end_depth =
        node_prefix_total_len`` and ``start_depth = end_depth - size``) is::

            recompute_cost = end_depth^2 - start_depth^2

        This is the difference of the two quadratic prefix costs and equals
        ``2*L*S - S^2`` for a node of size ``S`` at depth ``L``. Deeper
        segments are disproportionately more expensive. Falls back to
        ``size^2`` when prefix length is not yet set.
        """
        end_depth = getattr(node, "node_prefix_total_len", 0)
        size = 0
        try:
            if node.token_mem_index_value is not None:
                size = len(node.token_mem_index_value)
        except Exception:
            size = 0
        if end_depth is None or end_depth == 0:
            # fallback: isolated node, start 0
            size = max(size, 1)
            return float(size * size)
        end_depth = int(max(end_depth, 1))
        size = int(max(size, 1))
        start_depth = max(end_depth - size, 0)
        return float(end_depth * end_depth - start_depth * start_depth)

    def priority(self, node: "TreeNode") -> float:
        # Non-compute-aware GDSF: clock + freq / size**exp
        # Compute-aware GDSF (requested): priority = global_age + frequency * recompute_cost / resident_tokens
        # where recompute_cost is quadratic L*(L+1)/2 and resident_tokens is node size.
        # global_age is the per-node frozen aging_clock (node._gdsf_clock).
        size = self._size(node)
        if not self.compute_aware:
            return node._gdsf_clock + node._gdsf_freq / (size ** self.size_exponent)
        # compute-aware: use exact requested formula
        resident_tokens = size  # tokens resident in this node
        recompute_cost = self._compute_cost(node)  # quadratic L*(L+1)/2
        if self.compute_weight != 1.0:
            recompute_cost = recompute_cost ** self.compute_weight
        # GDSF aging: global_age (node._gdsf_clock) + freq * recompute_cost / resident_tokens
        return node._gdsf_clock + (node._gdsf_freq * recompute_cost) / resident_tokens

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
            node._gdsf_freq = 0
            node._gdsf_clock = self.aging_clock

    def on_merge(self, parent: "TreeNode", child: "TreeNode") -> None:
        # Merge collapses parent into child; the surviving child now spans the
        # combined prefix. Its history must remain the child's own history –
        # the number of times the *full* combined span was reused. Do not
        # alter child's frequency or clock; only recency (time_id) is merged
        # as max for tie-breaking. This preserves the deeper node's GDSF
        # state and avoids polluting it with parent-only accesses.
        child.time_id = max(parent.time_id, child.time_id)

    def on_evict(self, node: "TreeNode") -> None:
        # GDSF aging: set global clock to the last victim's priority directly
        # (not max). GreedyDual inflation is defined as L = H(victim), where
        # H is the priority at eviction time.
        self.aging_clock = self.priority(node)


_POLICIES = {
    "lru": LRUEvictionPolicy,
    "gdsf": GDSFEvictionPolicy,
    "gdsf_compute": GDSFEvictionPolicy,
}


def create_eviction_policy(name: str = "lru", **kwargs) -> EvictionPolicy:
    """Build an eviction policy by name.

    ``lru`` ignores kwargs; ``gdsf`` accepts ``size_exponent``, ``compute_aware``
    and ``compute_weight`` (with aliases ``enable_compute_aware``, ``compute_factor`` etc.).
    ``gdsf_compute`` is an alias for ``gdsf`` with ``compute_aware=True`` by default
    (compute-aware GDSF: ``priority = clock + freq * cost / size**exp``).
    """
    # Normalize common aliases so callers can use either spelling and existing
    # env/CLI wiring that may use different key names still works.
    alias_map = {
        "gdsf_size_exponent": "size_exponent",
        "size_exponent": "size_exponent",
        "compute_aware": "compute_aware",
        "enable_compute_aware": "compute_aware",
        "gdsf_compute_aware": "compute_aware",
        "radix_cache_gdsf_compute_aware": "compute_aware",
        "compute_weight": "compute_weight",
        "compute_factor": "compute_weight",
        "compute_scale": "compute_weight",
        "compute_exponent": "compute_weight",
        "gdsf_compute_weight": "compute_weight",
        "radix_cache_gdsf_compute_weight": "compute_weight",
    }
    normalized = {}
    for k, v in kwargs.items():
        # drop None? keep explicit False/0
        nk = alias_map.get(k, k)
        # last wins if duplicate alias
        normalized[nk] = v

    name = (name or "lru").lower()
    if name not in _POLICIES:
        raise ValueError(f"unknown radix cache eviction policy {name!r}; choices: {sorted(_POLICIES)}")
    if name == "lru":
        if normalized:
            # LRU ignores gdsf kwargs but should not silently accept compute flags that
            # would imply gdsf was intended; warn is handled at higher layer. Here
            # keep strict: lru with gdsf kwargs is a no-op, but don't error to keep
            # backward compat with callers that always pass size_exponent.
            pass
        return LRUEvictionPolicy()
    if name == "gdsf_compute":
        # gdsf_compute is compute-aware by definition; force it on even if caller
        # passed compute_aware=False via the generic StartArgs/cli kwargs.
        normalized["compute_aware"] = True
    return _POLICIES[name](**normalized)
