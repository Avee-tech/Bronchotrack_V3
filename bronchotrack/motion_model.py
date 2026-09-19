"""Graph-constrained motion model for branch-level localization.

This adds a proper predict/update (Bayes) filter over the airway graph's
discrete branch labels, seeded at the trachea and constrained to only ever
move along real graph edges (parent <-> self <-> children) -- i.e. exactly
the "start at the trachea, predict which lumen the camera is currently in,
doesn't need to be the exact position inside it" idea, kept at the same
branch-level granularity the rest of the pipeline already works at rather
than trying to recover a continuous 3D pose (see README "Not implemented"
-> "Position-aware matching vs. real depth/pose estimation" for why that
bigger version is a separate, much larger undertaking).

It runs *alongside* the existing `association.py` hard-labeling +
`localization.py` majority-vote smoothing, not in place of them -- both are
reported on every `FrameResult` so you can compare a simple heuristic
vote against a proper probabilistic filter (a natural ablation for the
thesis), rather than this silently replacing tested, working behavior.

How it works, per frame
------------------------
1. **Predict**: belief (a probability distribution over branch labels) is
   propagated one step along the tree. Each label's probability mass
   splits into "stayed", "advanced to a child" (split across children), and
   "retreated to the parent", using transition probabilities that default
   to favoring forward motion (bronchoscopy is normally an insertion
   procedure) and are nudged by `association.py`'s `approach_trend`
   diagnostic: a lumen that's visibly growing shifts probability toward
   "advance", a shrinking one toward "retreat". This is an *honest* use of
   that diagnostic -- it only ever biases a *direction* of motion, never a
   claimed distance or exact position (see association.py's module
   docstring for why the latter isn't reliable from monocular image size
   alone).
2. **Update**: the set of branch labels actually hard-matched to a visible
   tracklet this frame (`AirwayAssociation.process_frame`'s return value)
   is treated as a noisy observation -- seen labels get their probability
   boosted, everything else decays mildly (not zeroed: occlusion/missed
   detections are common and shouldn't be read as strong evidence the
   scope has left that branch).

The reported "current location" is just the argmax of the resulting
belief, with its probability as a confidence value -- deliberately not a
continuous in-branch position, per the request this was built for.
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import numpy as np

from .graph import AirwayGraph

DEFAULT_P_ADVANCE = 0.06
DEFAULT_P_STAY = 0.92
DEFAULT_P_RETREAT = 0.02
DEFAULT_TREND_GAIN = 1.5  # how strongly approach_trend nudges the transition probs
DEFAULT_OBSERVATION_CONFIDENCE = 0.85  # P(seen | truly there); also used as P(not-seen | not there)
DEFAULT_DISTANCE_DISCOUNT_PER_HOP = 0.0  # off by default -- see "A second real failure mode" below
DEFAULT_MIN_DISCOUNTED_CONFIDENCE = 0.5
_MIN_BELIEF = 1e-4  # prune entries below this after each normalize
_MAX_BELIEF_NODES = 64  # hard cap so belief can't spread across an entire large tree


class TreeMotionFilter:
    """Discrete Bayes filter over `graph`'s branch labels, seeded at the
    root (trachea). Call `predict()` then `update()` once per frame, in
    that order, then read `current_location()`.
    """

    def __init__(
        self,
        graph: AirwayGraph,
        p_advance: float = DEFAULT_P_ADVANCE,
        p_stay: float = DEFAULT_P_STAY,
        p_retreat: float = DEFAULT_P_RETREAT,
        trend_gain: float = DEFAULT_TREND_GAIN,
        observation_confidence: float = DEFAULT_OBSERVATION_CONFIDENCE,
        distance_discount_per_hop: float = DEFAULT_DISTANCE_DISCOUNT_PER_HOP,
        min_discounted_confidence: float = DEFAULT_MIN_DISCOUNTED_CONFIDENCE,
    ):
        total = p_advance + p_stay + p_retreat
        self.p_advance = p_advance / total
        self.p_stay = p_stay / total
        self.p_retreat = p_retreat / total
        self.trend_gain = trend_gain
        self.observation_confidence = observation_confidence
        self.distance_discount_per_hop = distance_discount_per_hop
        self.min_discounted_confidence = min_discounted_confidence

        self.graph = graph
        self.belief: Dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        """Re-seed belief as fully certain at the trachea (root)."""
        self.belief = {self.graph.root(): 1.0}

    # ------------------------------------------------------------------
    def _transition_probs(self, approach_trend: Optional[float]) -> Tuple[float, float, float]:
        if approach_trend is None:
            return self.p_advance, self.p_stay, self.p_retreat
        shift = float(np.clip(approach_trend * self.trend_gain, -0.3, 0.3))
        adv = float(np.clip(self.p_advance + shift, 0.02, 0.95))
        retreat = float(np.clip(self.p_retreat - shift, 0.02, 0.6))
        stay = max(1e-6, 1.0 - adv - retreat)
        return adv, stay, retreat

    def predict(self, approach_trend: Optional[float] = None) -> None:
        """Propagate belief one frame forward along graph edges only --
        mass can move to the current node's children, its parent, or stay,
        never to an unrelated branch."""
        p_adv, p_stay, p_retreat = self._transition_probs(approach_trend)
        new_belief: Dict[str, float] = {}

        def add(label: str, amount: float) -> None:
            if amount <= 0:
                return
            new_belief[label] = new_belief.get(label, 0.0) + amount

        for label, p in self.belief.items():
            if label not in self.graph:
                continue  # defensive: shouldn't happen, belief only ever holds real graph labels
            children = self.graph.children(label)
            parent = self.graph.parent(label)

            add(label, p * p_stay)

            if children:
                share = p * p_adv / len(children)
                for c in children:
                    add(c, share)
            else:
                add(label, p * p_adv)  # leaf branch: nowhere to advance to, stays

            if parent is not None:
                add(parent, p * p_retreat)
            else:
                add(label, p * p_retreat)  # root: nowhere to retreat to, stays

        self.belief = self._normalize(new_belief)

    def update(self, observed_labels: Set[str]) -> None:
        """Bayesian update from this frame's hard-matched branch labels
        (see module docstring). A no-op if nothing was observed this frame
        -- the predicted belief stands as-is rather than being penalized
        for a frame with no confirmed detections.

        `association.py` can legitimately hard-label several tracklets in
        the same frame at once (an anchor plus its children/siblings), so
        `observed_labels` isn't just "the one true current branch" -- and
        real detections are noisy enough (mean confidence ~0.45 in this
        project's own CPU baseline) that an occasional mislabel at a branch
        several generations away from where the filter otherwise has its
        belief concentrated is expected, not exceptional. An earlier
        version of this method discounted each observed label's evidence
        strength by its graph (tree-hop) distance from the filter's
        current highest-belief branch, floored at `min_discounted_confidence`
        (`distance_discount_per_hop`, still available below, just OFF by
        default -- see "A second real failure mode" in the README's
        "Motion-model filter" section for the full story). That fixed the
        original drift problem on a real 916-frame run, but a second real
        run then exposed a worse issue it introduced: discounting new
        evidence by its distance from the filter's *own* current belief is
        self-referential -- once that belief is confidently (and,
        mid-transition, increasingly wrongly) sitting on one branch, it
        discounts exactly the evidence that would correct it, harder the
        more confident it gets, so it can permanently lock onto a stale
        branch even as the scope genuinely advances. Simulated tests
        confirmed low `p_advance` alone (see module docstring / class
        constants) already resists the original one-off-mislabel drift
        just as well, without that self-reinforcing risk -- so the
        distance discount now defaults off, and `p_advance` is the primary
        defense against per-frame noise. `distance_discount_per_hop` is
        still exposed for experimentation, but re-enabling it should be
        paired with re-running `scripts/analyze_motion_model.py` against a
        real recording to check for this lock-in pattern (motion-model
        switches/agreement dropping while it sits unmoving on one branch
        for long stretches the vote-based localizer has already left).
        """
        if not observed_labels:
            return

        for label in observed_labels:
            if label in self.graph and label not in self.belief:
                self.belief[label] = _MIN_BELIEF

        map_label, _ = self.current_location()
        c = self.observation_confidence
        posterior = {}
        for label, p in self.belief.items():
            if label in observed_labels:
                dist = self._graph_distance(map_label, label) if map_label else 0
                eff_c = max(self.min_discounted_confidence, c - self.distance_discount_per_hop * dist)
                posterior[label] = p * eff_c
            else:
                posterior[label] = p * (1.0 - c)
        self.belief = self._normalize(posterior)

    def _graph_distance(self, a: Optional[str], b: str) -> int:
        """Tree-hop distance between two branch labels (0 if equal)."""
        if a is None or a == b:
            return 0
        path_a = self._ancestor_path(a)
        path_b = self._ancestor_path(b)
        idx_a = {label: i for i, label in enumerate(path_a)}
        for j, node in enumerate(path_b):
            if node in idx_a:
                return idx_a[node] + j
        return len(path_a) + len(path_b)  # shouldn't happen in a connected tree

    def _ancestor_path(self, label: str) -> list:
        """`label`, its parent, grandparent, ... up to the root, inclusive."""
        path = [label]
        while True:
            parent = self.graph.parent(path[-1])
            if parent is None:
                break
            path.append(parent)
        return path

    @staticmethod
    def _normalize(belief: Dict[str, float]) -> Dict[str, float]:
        if not belief:
            return belief
        # prune near-zero entries first so the cap below keeps the labels
        # that actually matter
        pruned = {l: p for l, p in belief.items() if p > _MIN_BELIEF}
        if not pruned:
            pruned = belief  # everything was tiny (e.g. very early frames) -- keep as-is
        if len(pruned) > _MAX_BELIEF_NODES:
            top = sorted(pruned.items(), key=lambda kv: -kv[1])[:_MAX_BELIEF_NODES]
            pruned = dict(top)
        total = sum(pruned.values())
        if total < 1e-12:
            return pruned
        return {l: p / total for l, p in pruned.items()}

    # ------------------------------------------------------------------
    def current_location(self) -> Tuple[Optional[str], float]:
        """(label, confidence) of the highest-probability branch in the
        current belief, or (None, 0.0) if belief is somehow empty."""
        if not self.belief:
            return None, 0.0
        label = max(self.belief, key=self.belief.get)
        return label, self.belief[label]
