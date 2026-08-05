"""
Pluggable eviction policies for the radix prompt KV cache.

The radix cache keeps its evictable leaf nodes in a ``SortedSet`` ordered by a
per-node *compare key*. Popping index ``0`` yields the next victim. A policy is
responsible for three things:

* ``get_compare_key(node)`` -- the sort key used to order eviction candidates.
* ``init_node(node)``       -- admission of a brand-new node. Anchors its aging
                               baseline; not a reuse.
* ``touch_node(node)``      -- recency / priority bookkeeping that is NOT a reuse.
                               Called on KV write-back and read-only probes. Never
                               increments frequency and never re-anchors the age.
* ``record_reuse(node)``    -- a genuine cache reuse, i.e. ``match_prefix`` with
                               ``update_refs=True`` (full OR partial prefix hit).
                               The ONLY hook that counts toward frequency.
* ``on_split(prefix, suffix)`` -- called when a node is split into a shared prefix
                               and the original suffix. Both halves have new
                               geometry and must be rescored independently while
                               preserving the frozen age / frequency they inherit.
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

GDSF priority of a node::

    priority = age + freq * recompute_cost / resident_tokens

where

* ``resident_tokens`` is the number of tokens the node occupies in KV memory
  (its "size"),
* ``recompute_cost`` approximates the compute needed to regenerate this node's KV
  cache. Prefill attention is quadratic in sequence length, so a node covering the
  prefix span ``[start, end)`` costs ``end**2 - start**2`` -- the marginal
  quadratic attention work of its own tokens given all preceding context,
* ``freq`` is how often the node has genuinely been reused,
* ``age`` is the node's *frozen* snapshot of the global aging clock, taken at its
  last admission/reuse. Because it is stored per node, geometry-only events (a
  split rescoring both halves) can re-derive priority from the current span
  WITHOUT re-anchoring the age -- preserving the node's history. ``global_age`` is
  the clock itself, advanced to the priority of the most recently evicted node; it
  prevents a once-popular-but-now-cold node from lingering forever.
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

    def on_evict(self, node) -> None:
        """Hook invoked when ``node`` is evicted. No-op for stateless policies."""
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
        # GDSF aging clock (a.k.a. the "L" / inflation value). Monotonic.
        self.global_age: float = 0.0

    @staticmethod
    def _recompute_cost(node) -> float:
        """
        Marginal quadratic prefill/attention compute of this node's token span.

        The node covers prefix positions ``[start, end)`` where
        ``end = node_prefix_total_len`` and ``start = end - node_value_len``.
        Attention prefill compute grows quadratically with position, so the work
        attributable to this span is ``end**2 - start**2``.
        """
        end = node.node_prefix_total_len
        start = end - node.node_value_len
        return float(end * end - start * start)

    def compute_priority(self, node) -> float:
        # Derived from the node's CURRENT geometry and its FROZEN age baseline, so
        # a geometry-only change (split) can be rescored without re-anchoring age.
        resident_tokens = max(node.node_value_len, 1)
        return node.age + node.freq * self._recompute_cost(node) / resident_tokens

    def get_compare_key(self, node) -> Tuple:
        # Evict unreferenced nodes first, then lowest priority, oldest as tiebreak.
        return (0 if node.ref_counter == 0 else 1, node.priority, node.time_id)

    def init_node(self, node) -> None:
        # Admission: anchor the aging baseline at the current clock. Not a reuse,
        # so freq stays 0 -> priority == global_age (a prime eviction candidate
        # until the node is genuinely reused).
        node.age = self.global_age
        node.priority = self.compute_priority(node)

    def touch_node(self, node) -> None:
        # Non-reuse bookkeeping (write-back / probe): refresh recency and rescore
        # for any geometry change, but PRESERVE the frozen age and frequency.
        node.update_time()
        node.priority = self.compute_priority(node)

    def record_reuse(self, node) -> None:
        # Genuine prompt-cache hit: the only path that grows frequency. Re-anchor
        # the aging baseline to the current clock (classic GDSF: H = L + f*c/s).
        node.update_time()
        node.freq += 1
        node.age = self.global_age
        node.priority = self.compute_priority(node)

    def on_split(self, prefix_node, suffix_node) -> None:
        # Split changes the geometry of both halves. Rescore each from its new span
        # while keeping the frozen age / frequency inherited from the pre-split node.
        prefix_node.priority = self.compute_priority(prefix_node)
        suffix_node.priority = self.compute_priority(suffix_node)

    def on_evict(self, node) -> None:
        # Advance the aging clock to the victim's priority (classic GDSF inflation).
        self.global_age = max(self.global_age, node.priority)

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
