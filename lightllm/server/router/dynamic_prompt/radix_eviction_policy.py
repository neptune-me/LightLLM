"""
Pluggable eviction policies for the radix prompt KV cache.

The radix cache keeps its evictable leaf nodes in a ``SortedSet`` ordered by a
per-node *compare key*. Popping index ``0`` yields the next victim. A policy is
responsible for three things:

* ``get_compare_key(node)`` -- the sort key used to order eviction candidates.
* ``init_node(node)``       -- admission of a brand-new node. Anchors its aging
                               baseline; not a reuse.
* ``touch_node(node)``      -- non-reuse bookkeeping on an existing node (KV
                               write-back, read-only probe). Never counts as reuse:
                               no frequency growth and no age re-anchor. GDSF also
                               leaves recency (``time_id``) untouched -- only a
                               genuine reuse may improve a node's eviction standing.
* ``record_reuse(node)``    -- a genuine cache reuse, i.e. ``match_prefix`` with
                               ``update_refs=True`` (full OR partial prefix hit).
                               The ONLY hook that counts toward frequency.
* ``on_split(prefix, suffix)`` -- called when a node is split into a shared prefix
                               and the original suffix. Both halves have new
                               geometry and must be rescored independently while
                               preserving the frozen age / frequency they inherit.
* ``on_truncate(node)``     -- called when elastic eviction reclaims a node's tail
                               and leaves its reusable prefix resident. Like
                               ``on_split`` this is a geometry-only change: rescore
                               from the new span, but never re-anchor age, grow
                               frequency, or refresh recency.
* ``on_evict(node)``        -- called when a node is evicted (lets aging policies
                               advance their global clock).
* ``on_merge(parent, child)`` -- called when ``child`` absorbs ``parent`` during a
                               node merge, so metadata can be carried over.

Frequency policy: a node's ``freq`` reflects genuine reuse only. Inserting the
node's KV back into the tree, splitting/merging it, read-only probes, and
reference-counter bookkeeping do NOT count as reuse -- only a real prompt-cache
hit through ``match_prefix(update_refs=True)`` does. A partial hit (the request
matches a proper prefix of a resident node, splitting it) DOES count: the matched
shared-prefix half is a genuine reuse.

Two policies are provided:

* ``LRUEvictionPolicy`` (``"lru"``) -- the historical default. Evicts the oldest
  unreferenced leaf first. This preserves the original cache behavior exactly.
* ``ComputeAwareGDSFEvictionPolicy`` (``"gdsf_compute"``) -- a compute-aware
  variant of Greedy-Dual-Size-Frequency. It keeps prefixes whose KV cache is
  expensive to recompute and frequently reused, and drops cheap / cold ones.

GDSF priority of a node, scored at each genuine event (admission / reuse)::

    recompute_cost = end_depth ** 2 - start_depth ** 2
    priority       = global_age + frequency * recompute_cost / resident_tokens

where

* ``resident_tokens`` is the number of tokens the node occupies in KV memory
  (its "size", ``node_value_len``),
* ``recompute_cost`` approximates the compute needed to regenerate this node's KV
  cache. Prefill attention is quadratic in sequence length, so a node spanning
  prefix depths ``[start_depth, end_depth)`` costs ``end_depth**2 -
  start_depth**2`` -- the marginal quadratic attention work of its own tokens
  given all preceding context (``end_depth = node_prefix_total_len``,
  ``start_depth = end_depth - node_value_len``),
* ``frequency`` is how often the node has genuinely been reused,
* ``global_age`` is the GDSF aging clock, set to the priority of the most recently
  evicted node (``L = H(victim)``); it prevents a once-popular-but-now-cold node
  from lingering forever.

Each node also stores ``age``: the *frozen* snapshot of ``global_age`` taken when
the priority above was last scored (admission / reuse). Storing it per node lets
geometry-only events -- a split rescoring both halves, a merge -- re-derive
priority from the new span using the SAME baseline, so they preserve the node's
history instead of re-anchoring it to the current clock.
"""

from typing import Tuple


class EvictionPolicy:
    """Base class defining the eviction-policy interface used by ``RadixCache``."""

    #: stable name usable on the command line / in configs
    name: str = "base"

    def get_compare_key(self, node) -> Tuple:
        raise NotImplementedError

    def init_node(self, node) -> None:
        """Admit a brand-new node (anchor its aging baseline). Not a reuse."""
        raise NotImplementedError

    def touch_node(self, node) -> None:
        """Recency / priority bookkeeping without counting a reuse.

        Called on KV write-back and read-only probes. Must never increment a
        node's frequency or re-anchor its frozen age.
        """
        raise NotImplementedError

    def record_reuse(self, node) -> None:
        """Count a genuine cache reuse (``match_prefix`` with ``update_refs=True``)."""
        raise NotImplementedError

    def on_split(self, prefix_node, suffix_node) -> None:
        """Hook invoked when a node is split into ``prefix_node`` and ``suffix_node``.

        Both halves have new geometry and should be rescored independently while
        preserving the frozen age / frequency inherited from the pre-split node.
        No-op for policies whose key does not depend on geometry.
        """
        return

    def on_truncate(self, node) -> None:
        """Hook invoked after elastic eviction trims ``node``'s tail.

        The node stays resident with a shorter span, so this is a geometry-only
        event: rescore if the key depends on geometry, but never re-anchor the
        frozen age, grow frequency, or refresh recency. No-op for policies whose
        key does not depend on geometry.
        """
        return

    def on_evict(self, node) -> None:
        """Hook invoked when ``node`` is evicted. No-op for stateless policies.

        Only fired for whole-node eviction. A node whose tail was trimmed by
        elastic eviction is still in the cache, so it reports ``on_truncate``
        instead and must not advance an aging clock.
        """
        return

    def on_merge(self, parent_node, child_node) -> None:
        """Hook invoked when ``child_node`` absorbs ``parent_node`` on merge."""
        return


class LRUEvictionPolicy(EvictionPolicy):
    """
    Least-recently-used eviction (the original cache behavior).

    Unreferenced leaves (``ref_counter == 0``) are evicted before referenced ones,
    oldest ``time_id`` first.
    """

    name = "lru"

    def get_compare_key(self, node) -> Tuple:
        return (0 if node.ref_counter == 0 else 1, len(node.children), node.time_id)

    def init_node(self, node) -> None:
        # A freshly constructed node already carries the newest time_id; the split
        # prefix likewise. LRU's key does not depend on geometry, so nothing to do.
        return

    def touch_node(self, node) -> None:
        node.update_time()

    def record_reuse(self, node) -> None:
        # LRU is recency-only; reuse and any other touch both just refresh recency.
        node.update_time()

    # on_split: default no-op -- LRU ordering (children count, time_id) is unaffected
    # by the geometry change, matching the original cache behavior exactly.

    # on_truncate: default no-op, and deliberately so. LRU's key does not depend on
    # geometry, and refreshing time_id here would move the coldest leaf to the back
    # of the queue just because we trimmed it -- the next eviction would then have to
    # attack warmer data. A trimmed leaf must keep its place at the cold end.

    def on_merge(self, parent_node, child_node) -> None:
        child_node.time_id = max(parent_node.time_id, child_node.time_id)


class ComputeAwareGDSFEvictionPolicy(EvictionPolicy):
    """
    Compute-aware Greedy-Dual-Size-Frequency eviction.

    Nodes with the *lowest* priority are evicted first. Priority rewards prefixes
    that are expensive to recompute (quadratic prefill compute) and frequently
    reused, normalized by how much KV memory they occupy.
    """

    name = "gdsf_compute"

    def __init__(self):
        # GDSF aging clock (a.k.a. the "L" / inflation value): the priority of the
        # most recently evicted node.
        self.global_age: float = 0.0

    @staticmethod
    def _recompute_cost(node) -> float:
        """
        Marginal quadratic prefill/attention compute of this node's token span::

            recompute_cost = end_depth ** 2 - start_depth ** 2

        The node covers prefix depths ``[start_depth, end_depth)`` where
        ``end_depth = node_prefix_total_len`` and
        ``start_depth = end_depth - node_value_len``. Attention prefill compute
        grows quadratically with position, so the work attributable to this span is
        ``end_depth**2 - start_depth**2``.
        """
        end_depth = node.node_prefix_total_len
        start_depth = end_depth - node.node_value_len
        return float(end_depth * end_depth - start_depth * start_depth)

    def _score(self, node, aging_baseline) -> float:
        """priority = aging_baseline + frequency * recompute_cost / resident_tokens"""
        resident_tokens = max(node.node_value_len, 1)
        return aging_baseline + node.freq * self._recompute_cost(node) / resident_tokens

    def compute_priority(self, node) -> float:
        # Rescore from CURRENT geometry against the node's FROZEN aging baseline, so
        # a geometry-only change (split / merge) never re-anchors the age.
        return self._score(node, node.age)

    def get_compare_key(self, node) -> Tuple:
        # Evict unreferenced nodes first, then lowest priority, oldest as tiebreak.
        return (0 if node.ref_counter == 0 else 1, node.priority, node.time_id)

    def init_node(self, node) -> None:
        # Admission scores at the current aging floor:
        #   priority = global_age + frequency * recompute_cost / resident_tokens
        # Not a reuse, so freq stays 0 -> priority == global_age (a prime eviction
        # candidate until genuinely reused). Freeze global_age into node.age so later
        # geometry-only rescoring (splits) preserves this baseline.
        node.age = self.global_age
        node.priority = self._score(node, self.global_age)

    def touch_node(self, node) -> None:
        # Non-reuse bookkeeping (write-back / probe). Rescore for any geometry
        # change, but do NOT touch the aging component: frequency, the frozen age
        # baseline, AND recency (time_id) are all preserved. Refreshing recency here
        # would let a write-back or a read-only probe improve a node's eviction
        # standing -- only a genuine reuse (record_reuse) or admission (init_node)
        # may do that.
        node.priority = self.compute_priority(node)

    def record_reuse(self, node) -> None:
        # Genuine prompt-cache hit: the only path that grows frequency. Re-anchor the
        # aging baseline to the current clock and score with the canonical formula:
        #   priority = global_age + frequency * recompute_cost / resident_tokens
        node.update_time()
        node.freq += 1
        node.age = self.global_age
        node.priority = self._score(node, self.global_age)

    def on_split(self, prefix_node, suffix_node) -> None:
        # Split changes the geometry of both halves. Rescore each from its new span
        # while keeping the frozen age / frequency inherited from the pre-split node.
        prefix_node.priority = self.compute_priority(prefix_node)
        suffix_node.priority = self.compute_priority(suffix_node)

    def on_truncate(self, node) -> None:
        # Elastic eviction reclaimed the tail; the head is still resident. Geometry
        # changed, so rescore against the FROZEN age -- exactly like on_split, and
        # for the same reason: losing tokens is not a reuse and must not re-anchor
        # the node's aging baseline, grow its frequency, or refresh its recency.
        #
        # Note the priority can only move down here. With
        # cost/size == (e**2 - s**2) / (e - s) == e + s, trimming lowers the node's
        # end depth ``e`` while its start depth ``s`` is fixed, so a trimmed node
        # stays at (freq == 0) or below (freq > 0) the eviction floor and remains
        # the next victim -- which is what we want, we keep carving the coldest
        # prefix down rather than moving on to warmer data.
        node.priority = self.compute_priority(node)

    def on_evict(self, node) -> None:
        # Classic GDSF inflation: set the aging clock to the evicted victim's
        # priority directly (L = H(victim)), NOT max(L, H) -- global_age must track
        # the last victim, which is monotonic non-decreasing whenever the true
        # minimum is evicted.
        #
        # Deliberately NOT fired for elastic truncation. The clock tracks objects
        # that actually left the cache; a trimmed node is still resident. Firing it
        # there would also break the monotonicity argument above, because a trimmed
        # node is rescored strictly downwards and would drag L back down with it.
        self.global_age = node.priority

    def on_merge(self, parent_node, child_node) -> None:
        # The merged node keeps the richer history of the two, then is rescored for
        # its combined span.
        child_node.time_id = max(parent_node.time_id, child_node.time_id)
        child_node.freq = max(parent_node.freq, child_node.freq)
        child_node.age = max(parent_node.age, child_node.age)
        child_node.priority = self.compute_priority(child_node)


_POLICY_REGISTRY = {
    LRUEvictionPolicy.name: LRUEvictionPolicy,
    ComputeAwareGDSFEvictionPolicy.name: ComputeAwareGDSFEvictionPolicy,
}

#: names accepted by ``--radix_eviction_policy`` / ``get_eviction_policy``
EVICTION_POLICY_CHOICES = list(_POLICY_REGISTRY.keys())


def get_eviction_policy(policy) -> EvictionPolicy:
    """
    Resolve ``policy`` to an ``EvictionPolicy`` instance.

    ``policy`` may be an existing ``EvictionPolicy`` (returned as-is), ``None``
    (defaults to LRU), or one of the names in ``EVICTION_POLICY_CHOICES``.
    """
    if policy is None:
        return LRUEvictionPolicy()
    if isinstance(policy, EvictionPolicy):
        return policy
    if policy in _POLICY_REGISTRY:
        return _POLICY_REGISTRY[policy]()
    raise ValueError(f"unknown radix eviction policy {policy!r}, expected one of {EVICTION_POLICY_CHOICES}")
