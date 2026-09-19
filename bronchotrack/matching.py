"""Cost functions + assignment for multi-lumen tracking (paper section 3).

Implements, verbatim from the paper:

    C_m(i,j) = 1 - IoU(x_i, z_j)                     motion cost
    C_a(i,j) = 1 - (f_j^t)^T . e_i^{t-1}              appearance cost (cosine)
    C(i,j)   = lambda * C_a(i,j) + (1-lambda) * C_m(i,j)   combined, lambda=0.5

and a generic Hungarian-assignment-with-thresholding helper used by
tracker.py for the two-stage (high-conf combined-cost / low-conf
motion-only) association described in the paper.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .types import BBox, Detection, Tracklet
from .utils import cosine_similarity, hungarian, iou

DEFAULT_LAMBDA = 0.5


def motion_cost_matrix(
    predicted_boxes: List[BBox], detections: List[Detection]
) -> np.ndarray:
    """C_m(i, j) = 1 - IoU(predicted_i, detection_j)."""
    n, m = len(predicted_boxes), len(detections)
    cost = np.ones((n, m), dtype=np.float64)
    for i, pb in enumerate(predicted_boxes):
        for j, det in enumerate(detections):
            cost[i, j] = 1.0 - iou(pb, det.bbox)
    return cost


def appearance_cost_matrix(
    track_embeddings: List[Optional[np.ndarray]], detections: List[Detection]
) -> np.ndarray:
    """C_a(i, j) = 1 - cos(e_i^{t-1}, f_j^t).

    Tracklets or detections without an embedding get a neutral cost of 1.0
    for that entry (i.e. appearance provides no information), so this
    degrades gracefully if Re-ID is disabled or a crop failed to embed.
    """
    n, m = len(track_embeddings), len(detections)
    cost = np.ones((n, m), dtype=np.float64)
    for i, e in enumerate(track_embeddings):
        if e is None:
            continue
        for j, det in enumerate(detections):
            if det.embedding is None:
                continue
            cost[i, j] = 1.0 - cosine_similarity(e, det.embedding)
    return cost


def combined_cost_matrix(
    motion_cost: np.ndarray,
    appearance_cost: np.ndarray,
    lam: float = DEFAULT_LAMBDA,
) -> np.ndarray:
    """C(i,j) = lambda * C_a(i,j) + (1 - lambda) * C_m(i,j)."""
    return lam * appearance_cost + (1.0 - lam) * motion_cost


def assign(
    cost_matrix: np.ndarray, max_cost: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian assignment gated by a max-cost threshold.

    Returns (matches, unmatched_row_idx, unmatched_col_idx), where matches
    is a list of (row, col) pairs with cost <= max_cost.
    """
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    row_idx, col_idx = hungarian(cost_matrix)

    matches: List[Tuple[int, int]] = []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))

    unmatched_rows = [r for r in range(n) if r not in matched_rows]
    unmatched_cols = [c for c in range(m) if c not in matched_cols]
    return matches, unmatched_rows, unmatched_cols
