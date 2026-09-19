"""Tests for LumenDetector's mask acceptance threshold (detection.py's
"Mask acceptance threshold" docstring section): a detection below
`mask_conf_threshold` must still be returned (box, confidence, everything
else intact), just with `Detection.mask` dropped to `None` -- the same
shape every downstream consumer already falls back to for a plain, non-
segmentation checkpoint.

Runs without real weights or a GPU: `LumenDetector._ensure_model` is
monkeypatched to install a tiny fake ultralytics-shaped result object
instead of actually loading a checkpoint, so this only needs numpy.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from bronchotrack.detection import LumenDetector


class _FakeArray:
    """Minimal stand-in for a torch tensor: just enough of the
    detach().cpu().numpy() chain LumenDetector.infer() calls."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeBoxes:
    """Minimal stand-in for ultralytics' `result.boxes` -- adds `__len__`
    on top of _FakeArray-valued .xyxy/.conf/.cls, since detection.py checks
    `len(boxes)` before touching anything else."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.xyxy._arr)


def _make_detector(confs, xyxy, mask_conf_threshold=0.5, conf_threshold=0.1):
    """A LumenDetector whose (fake) model always returns one segmentation
    result: `len(confs)` detections, each with a valid 4-point square mask,
    at the given per-detection confidences."""
    det = LumenDetector("unused.pt", conf_threshold=conf_threshold, mask_conf_threshold=mask_conf_threshold)

    n = len(confs)
    fake_boxes = _FakeBoxes(
        xyxy=_FakeArray(xyxy),
        conf=_FakeArray(confs),
        cls=_FakeArray([0] * n),
    )
    square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    fake_masks = SimpleNamespace(xy=[square.copy() for _ in range(n)])
    fake_result = SimpleNamespace(boxes=fake_boxes, masks=fake_masks)

    class _FakeModel:
        def predict(self, **kwargs):
            return [fake_result]

    det._model = _FakeModel()
    det._resolved_device = "cpu"
    det._resolved_img_size = None
    return det


def test_mask_kept_when_confidence_at_or_above_threshold():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = _make_detector(confs=[0.9], xyxy=[[10, 10, 30, 30]])
    detections = det.infer(frame, frame_idx=0)
    assert len(detections) == 1
    assert detections[0].mask is not None
    assert len(detections[0].mask) == 4


def test_mask_dropped_below_threshold_but_detection_kept():
    """The whole point of this being a SEPARATE gate from conf_threshold:
    a detection at 0.3 confidence (well above the 0.1 conf_threshold that
    lets it exist at all) still gets its mask stripped at the default 0.5
    mask_conf_threshold -- the box/tracking path is unaffected."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = _make_detector(confs=[0.3], xyxy=[[10, 10, 30, 30]])
    detections = det.infer(frame, frame_idx=0)
    assert len(detections) == 1
    assert detections[0].mask is None
    assert detections[0].confidence == 0.3
    assert detections[0].bbox is not None  # box-based fallback path stays intact


def test_mask_exactly_at_threshold_is_kept():
    """>=, not >: a detection AT mask_conf_threshold should keep its mask,
    matching detection.py's `confs[i] >= self.mask_conf_threshold`."""
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = _make_detector(confs=[0.5], xyxy=[[10, 10, 30, 30]], mask_conf_threshold=0.5)
    detections = det.infer(frame, frame_idx=0)
    assert detections[0].mask is not None


def test_mixed_batch_gates_each_detection_independently():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = _make_detector(
        confs=[0.9, 0.3, 0.6],
        xyxy=[[10, 10, 30, 30], [40, 40, 60, 60], [70, 70, 90, 90]],
    )
    detections = det.infer(frame, frame_idx=0)
    assert len(detections) == 3
    assert detections[0].mask is not None  # 0.9 >= 0.5
    assert detections[1].mask is None      # 0.3 <  0.5
    assert detections[2].mask is not None  # 0.6 >= 0.5


def test_custom_mask_conf_threshold_is_respected():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    det = _make_detector(confs=[0.6], xyxy=[[10, 10, 30, 30]], mask_conf_threshold=0.75)
    detections = det.infer(frame, frame_idx=0)
    assert detections[0].mask is None  # 0.6 < custom 0.75 threshold


if __name__ == "__main__":
    test_mask_kept_when_confidence_at_or_above_threshold()
    test_mask_dropped_below_threshold_but_detection_kept()
    test_mask_exactly_at_threshold_is_kept()
    test_mixed_batch_gates_each_detection_independently()
    test_custom_mask_conf_threshold_is_respected()
    print("All detection mask-threshold tests passed.")
