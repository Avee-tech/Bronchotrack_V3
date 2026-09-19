"""Voting-based branch-level localization -- STRICT paper reference version
(paper section 4, "Localization", Eq. 8).

Fork of ``bronchotrack.localization`` with the temporal-smoothing/"sticky"
behavior removed entirely: this reports the raw per-frame vote every time,
exactly as Eq. 8 describes, with no history-based hysteresis. See
``bronchotrack.paper_exact`` (this package's ``__init__.py``) for the full
list of deltas against the main package.

Paper description (condensed): once tracklets carry anatomical labels,

    loc = g^{k-1}(l_i)   if n == 1   (single primary-level lumen visible)
    loc = g^k(l_i)       if n  > 1   (multiple primary-level lumens visible)

where g^k(l) is the k-th ancestor of branch l (g^0 = l itself), n is the
number of simultaneously-visible "primary-level" lumens, and all labeled
tracklets vote for their associated location; the most-voted branch is
reported as the bronchoscope's location.

Fidelity note (unchanged from the main package, kept because it is an
*interpretation* of an underspecified term, not an addition beyond the
paper): the paper does not spell out precisely how "primary-level" and the
base k are determined in general. Our reading, made concrete here: for each
currently-visible labeled tracklet, its "cohort" is the set of
currently-visible labeled tracklets that share its parent branch (i.e. its
visible siblings + itself). n is the size of that cohort.

  * n == 1  -> nothing ambiguous is visible alongside it; vote for the
               branch itself (k=0 relative to the paper's k-1 for k=1).
  * n  > 1  -> multiple sibling openings are visible at once, so we can't
               yet tell which one the scope will advance into; vote for
               their shared parent (k=1) instead of committing to one child.

This reduces exactly to the paper's g^{k-1} / g^k pattern with k=1.

Live vs. carried-forward votes (`Localizer.live_history` / `last_vote_was_live`)
-------------------------------------------------------------------------------
NOT a paper mechanism -- purely bookkeeping, added because the carry-forward
no-op above has a real, easy-to-miss consequence: `history` (and therefore
`FrameResult.location` / the JSON log's `"location"` field) is populated
almost every frame from shortly after initialization onward, whether or not
anything was actually re-confirmed that frame. A consumer reading only
`location` cannot tell "the scope is still here, freshly reconfirmed" from
"we have no current evidence at all and are just repeating our last guess" --
which matters, because the scope may well have advanced past a bifurcation,
unobserved, while every anchor was lost; a stale carried-forward label is a
real, silent accuracy risk, not just a cosmetic one.

`live_history` records, per frame (same length and indexing as `history`),
whether that frame's entry came from an actual vote (`True`) or was carried
forward with no visible labeled tracklet to vote with (`False`).
`last_vote_was_live()` exposes the latest entry for callers (`pipeline.py`'s
`FrameResult.location_is_live`, surfaced in the JSON log and, as a "(stale)"
suffix, in the overlay video's header -- see `viz.py`).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional

from ..graph import AirwayGraph
from ..types import Tracklet


class Localizer:
    def __init__(self, graph: AirwayGraph, ambiguous_k: int = 1):
        self.graph = graph
        self.ambiguous_k = ambiguous_k
        self.history: List[Optional[str]] = []  # per-frame raw vote, for logging/inspection
        self.live_history: List[bool] = []  # per-frame: True = fresh vote, False = carried forward

    def localize(self, tracklets: List[Tracklet]) -> Optional[str]:
        """Return this frame's raw Eq. 8 vote winner, or the last known
        location if nothing labeled is currently visible (there being
        nothing in the paper to vote with is not the same as the scope
        having left the airway -- carrying forward is the natural no-op).
        Either way, records whether THIS frame was a live vote or a
        carried-forward repeat in `live_history` -- see module docstring's
        "Live vs. carried-forward votes" section."""
        visible_labeled = [
            t
            for t in tracklets
            if t.time_since_update == 0 and t.label is not None and t.label in self.graph
        ]
        if not visible_labeled:
            last = self.history[-1] if self.history else None
            self.history.append(last)
            self.live_history.append(False)
            return last

        cohorts: Dict[Optional[str], List[Tracklet]] = defaultdict(list)
        for t in visible_labeled:
            parent = self.graph.parent(t.label)
            cohorts[parent].append(t)

        votes = Counter()
        for parent, members in cohorts.items():
            distinct_labels = {m.label for m in members}
            n = len(distinct_labels)
            k = 0 if n == 1 else self.ambiguous_k
            for label in distinct_labels:
                candidate = self.graph.ancestor(label, k)
                votes[candidate] += 1

        winner, _ = votes.most_common(1)[0]
        self.history.append(winner)
        self.live_history.append(True)
        return winner

    def last_vote_was_live(self) -> Optional[bool]:
        """Whether the most recent `localize()` call produced a fresh vote
        (True) or carried the last known location forward with no current
        evidence (False) -- None before the first call. See module
        docstring's "Live vs. carried-forward votes" section."""
        return self.live_history[-1] if self.live_history else None

    def current_generation(self) -> Optional[int]:
        if not self.history or self.history[-1] is None:
            return None
        return self.graph.generation(self.history[-1])
