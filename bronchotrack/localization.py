"""Voting-based branch-level localization (paper section 4, "Localization").

Paper description (condensed): once tracklets carry anatomical labels,

    loc = g^{k-1}(l_i)   if n == 1   (single primary-level lumen visible)
    loc = g^k(l_i)       if n  > 1   (multiple primary-level lumens visible)

where g^k(l) is the k-th ancestor of branch l (g^0 = l itself), n is the
number of simultaneously-visible "primary-level" lumens, and all labeled
tracklets vote for their associated location; the most-voted branch is
reported as the bronchoscope's location (this majority vote is what makes
the system robust to occasional wrong label assignments upstream).

Fidelity note: the paper does not spell out precisely how "primary-level"
and the base k are determined in general (only that this is the mechanism).
Our reading, made concrete here: for each currently-visible labeled
tracklet, its "cohort" is the set of currently-visible labeled tracklets
that share its parent branch (i.e. its visible siblings + itself). n is the
size of that cohort.

  * n == 1  -> nothing ambiguous is visible alongside it; vote for the
               branch itself (k=0 relative to the paper's k-1 for k=1).
  * n  > 1  -> multiple sibling openings are visible at once, so we can't
               yet tell which one the scope will advance into; vote for
               their shared parent (k=1) instead of committing to one child.

This reduces exactly to the paper's g^{k-1} / g^k pattern with k=1, and is
documented as an interpretation rather than a verbatim transcription --
see README "Fidelity notes" if your data suggests a different k should be
used (exposed as `ambiguous_k` below).

Temporal smoothing
-------------------
The raw per-frame vote above can flip between anatomically unrelated
branches frame to frame -- a single noisy low-confidence detection, a
momentarily mislabeled sibling, or a brief tracking glitch is enough to
swing the winner, even though the physical scope obviously didn't teleport
between branches. For a continuous-insertion recording that's not just
imprecise, it's actively misleading (e.g. flickering between two branches
in different subtrees frame to frame). So `localize()` reports a
**smoothed** location: it stays "sticky" on the current location unless a
challenger branch wins the raw per-frame vote at least `min_frames_to_switch`
times within the last `smoothing_window` frames -- i.e. a location change
has to earn some sustained support before it's actually reported, rather
than one noisy frame being enough to flip the display. The unsmoothed
per-frame vote is still recorded (`raw_history` / `last_raw`) if you want
to inspect or disable this.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Deque, Dict, List, Optional

from .graph import AirwayGraph
from .types import Tracklet


class Localizer:
    def __init__(
        self,
        graph: AirwayGraph,
        ambiguous_k: int = 1,
        smoothing_window: int = 5,
        min_frames_to_switch: int = 3,
    ):
        """
        smoothing_window : how many recent raw per-frame votes to consider
            when deciding whether to switch the reported location.
        min_frames_to_switch : a challenger branch must win the raw vote
            this many times within `smoothing_window` frames before the
            reported (smoothed) location switches to it. Set to 1 (or
            smoothing_window=1) to disable smoothing and report the raw
            per-frame vote directly, matching the original paper-literal
            behavior.
        """
        self.graph = graph
        self.ambiguous_k = ambiguous_k
        self.smoothing_window = smoothing_window
        self.min_frames_to_switch = min_frames_to_switch

        self.history: List[Optional[str]] = []  # smoothed, reported location per call
        self.raw_history: Deque[Optional[str]] = deque(maxlen=smoothing_window)

    @property
    def last_raw(self) -> Optional[str]:
        """The most recent *unsmoothed* per-frame vote (None if localize()
        hasn't been called yet, or the last call had nothing visible)."""
        return self.raw_history[-1] if self.raw_history else None

    def localize(self, tracklets: List[Tracklet]) -> Optional[str]:
        """Return the smoothed, reported branch location for the current
        frame (see module docstring), or None if nothing has ever been
        localized yet."""
        visible_labeled = [
            t
            for t in tracklets
            if t.time_since_update == 0 and t.label is not None and t.label in self.graph
        ]
        if not visible_labeled:
            smoothed = self.history[-1] if self.history else None
            self.history.append(smoothed)
            return smoothed

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

        raw_winner, _ = votes.most_common(1)[0]
        self.raw_history.append(raw_winner)

        smoothed = self._apply_smoothing(raw_winner)
        self.history.append(smoothed)
        return smoothed

    def _apply_smoothing(self, raw_winner: Optional[str]) -> Optional[str]:
        current = self.history[-1] if self.history else None
        if current is None:
            return raw_winner  # nothing to be sticky about yet
        if raw_winner == current:
            return current

        challenger_support = sum(1 for r in self.raw_history if r == raw_winner)
        if challenger_support >= self.min_frames_to_switch:
            return raw_winner
        return current

    def current_generation(self) -> Optional[int]:
        if not self.history or self.history[-1] is None:
            return None
        return self.graph.generation(self.history[-1])
