"""Lumen detection module (paper section 1).

Wraps your trained YOLOv11 ``.pt`` weights (via the ``ultralytics`` package
-- YOLOv11 uses the same Python API as YOLOv8/v7 in that library) and turns
raw model output into the unified :class:`~bronchotrack.types.Detection`
objects the rest of the pipeline consumes.

The paper uses YOLOv7 at 256x256 input resolution with a 0.1 confidence
threshold at inference time; the confidence default follows that, but the
inference resolution does NOT default to 256 -- it's auto-detected from
your checkpoint's own training config (``train_args.imgsz`` in the
ultralytics ``.pt`` file) if possible, falling back to ultralytics' own
default of 640 otherwise. Running inference at a resolution far from what
the model was trained at (e.g. forcing 256 on a model trained at 640) can
noticeably hurt detection accuracy, so don't override `img_size` unless you
have a specific reason to.

``ultralytics`` + ``torch`` are imported lazily inside ``LumenDetector`` so
that the rest of this package (graph parsing, tracking math, etc.) can be
imported/tested without those heavy dependencies installed.

Segmentation checkpoints (e.g. a YOLO26-seg ``.pt``) work here unchanged --
Ultralytics still populates `result.boxes` for a `-seg` model exactly as
for a plain detector, so box-based tracking/matching downstream needs no
changes at all. The only difference: `result.masks` is also populated, and
each detection's polygon is attached to its `Detection.mask` (None for a
plain detection checkpoint). `bronchotrack.paper_exact.association` uses it,
when present, for a materially better lumen-diameter estimate than a
box's (w+h)/2 -- see that module's docstring.

Mask acceptance threshold (`mask_conf_threshold`)
---------------------------------------------------
Ultralytics gives one confidence score per detection, not a separate one
for its box vs. its mask -- there is no independent "how trustworthy is
this specific polygon" signal available from the model itself. That single
score is already used, via `conf_threshold` above, to decide whether a
detection exists at all (paper: kept deliberately low, 0.1, so the
BYTE-style tracker's second stage can still use low-confidence detections
for motion-only matching). `mask_conf_threshold` (default 0.55) is a
SEPARATE, stricter gate on top of that: a detection below it still exists
(box, tracking, everything else proceeds as normal), but its mask is
dropped -- `Detection.mask` stays `None` -- so every mask-dependent
consumer downstream (the diameter:distance cue's `image_diameter`/
`image_offset`, `_is_nested`'s point-in-polygon containment check, the
overlay's translucent fill) transparently falls back to its existing
box-based path instead, exactly as it already does for a plain
(non-segmentation) checkpoint. The reasoning: a box is a coarse, forgiving
shape (four numbers), while a segmentation polygon can be a confident BOX
around something that isn't actually shaped like the reported MASK -- a
foreshortened or partly-occluded lumen is exactly the case that both
produces a noisier mask and still scores a reasonable box confidence, so a
single shared threshold that's low enough to keep the box wouldn't also be
a meaningful bar for the mask. `mask_conf_threshold` must be >=
`conf_threshold` to have any effect (a detection can't fall below the
inference-time cutoff and still be returned to threshold against here).
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Set, Union

import numpy as np

from .types import BBox, Detection
from .utils import resolve_device

ClassIdSpec = Union[int, Iterable[int], None]


def _normalize_class_ids(class_id: ClassIdSpec) -> Optional[Set[int]]:
    if class_id is None:
        return None
    if isinstance(class_id, int):
        return {class_id}
    return set(class_id)




class LumenDetector:
    """Thin inference wrapper around an Ultralytics YOLO model.

    Example
    -------
    >>> detector = LumenDetector("weights/lumen_yolov11.pt", conf_threshold=0.1)
    >>> detections = detector.infer(frame_bgr, frame_idx=42)
    """

    def __init__(
        self,
        weights_path: str,
        conf_threshold: float = 0.1,
        mask_conf_threshold: float = 0.55,
        img_size: Optional[int] = None,
        device: Optional[str] = None,
        class_id: ClassIdSpec = None,
    ):
        """
        Parameters
        ----------
        weights_path : path to your trained YOLOv11 ``.pt`` file.
        conf_threshold : detection confidence threshold at inference
            (paper uses 0.1 -- deliberately low, because the downstream
            BYTE-style tracker in ``tracker.py`` is designed to make use of
            low-confidence detections rather than discard them outright).
        mask_conf_threshold : minimum confidence for a segmentation
            checkpoint's mask to actually be attached to a `Detection`
            (default 0.55); below it, the detection still exists (box,
            tracking, etc. all proceed normally) but `Detection.mask` is
            `None`, same as a plain (non-seg) checkpoint would give -- see
            module docstring's "Mask acceptance threshold" section. No
            effect on a checkpoint with no segmentation output at all.
        img_size : inference resolution. Leave as None (default) to
            auto-detect from the checkpoint's own training config; only set
            this explicitly if you specifically want a different resolution
            than the model was trained at.
        device : "cuda", "cuda:0", "cpu", or None to auto-select.
        class_id : which class id(s) count as a detectable "lumen" opening.
            Pass a single int, an iterable of ints (e.g. {0, 1} if your
            model has multiple airway-opening classes you want to treat
            uniformly for tracking), or None (default) to keep every class
            the model outputs. Check `LumenDetector(...).class_names` if
            you're not sure what classes your model has.
        """
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.mask_conf_threshold = mask_conf_threshold
        self._img_size_override = img_size
        self._resolved_img_size: Optional[int] = None
        self._device_override = device
        self._resolved_device: Optional[str] = None
        self.class_ids = _normalize_class_ids(class_id)
        self._model = None  # lazy-loaded

    @property
    def img_size(self) -> Optional[int]:
        """The inference resolution actually in use (after auto-detection,
        if applicable). None until the model has been loaded at least once
        (i.e. before the first `infer()` call)."""
        return self._img_size_override or self._resolved_img_size

    @property
    def device(self) -> Optional[str]:
        """The concrete device inference actually runs on (e.g. "cuda:0" or
        "cpu"), after auto-detection. None until the model has been loaded
        at least once (i.e. before the first `infer()`/`class_names` call);
        call `warm_up()` to resolve it eagerly."""
        return self._resolved_device

    @property
    def class_names(self) -> dict:
        """{class_id: class_name} as reported by the loaded model. Loads
        the model if it hasn't been already."""
        self._ensure_model()
        return dict(self._model.names)

    def warm_up(self) -> None:
        """Load the model and resolve img_size/device now, instead of
        lazily on the first `infer()` call -- useful so callers (e.g. the
        CLI) can report which device inference will actually run on before
        processing starts."""
        self._ensure_model()

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for LumenDetector. Install with "
                "`pip install ultralytics`."
            ) from e

        self._resolved_device = resolve_device(self._device_override)
        self._model = YOLO(self.weights_path)
        # move the model to the resolved device once at load time, rather
        # than relying on ultralytics to figure it out fresh on every
        # predict() call -- makes "is this actually on the GPU" a simple,
        # one-time fact instead of a per-frame implicit decision.
        self._model.to(self._resolved_device)

        if self._img_size_override is None:
            self._resolved_img_size = self._detect_training_imgsz()

    def _detect_training_imgsz(self) -> Optional[int]:
        """Best-effort read of the checkpoint's own training `imgsz`, so we
        don't silently run inference at a mismatched resolution. Several
        fallback lookup paths since checkpoint structure varies a bit
        across ultralytics versions; returns None (-> ultralytics' own
        640 default) if none of them work."""
        try:
            args = getattr(self._model.model, "args", None)
            if isinstance(args, dict) and args.get("imgsz"):
                return int(args["imgsz"])
        except Exception:
            pass
        try:
            import torch

            ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            imgsz = ckpt.get("train_args", {}).get("imgsz")
            if imgsz:
                return int(imgsz)
        except Exception:
            pass
        return None

    def infer(self, frame_bgr: np.ndarray, frame_idx: int) -> List[Detection]:
        """Run detection on a single BGR frame (as returned by cv2.VideoCapture).

        Returns a list of Detection, each carrying an image crop (so
        reid.py can compute an appearance embedding downstream) but no
        embedding yet.
        """
        self._ensure_model()

        predict_kwargs = dict(
            source=frame_bgr,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        if self.img_size is not None:
            predict_kwargs["imgsz"] = self.img_size

        results = self._model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        cls_ids = boxes.cls.detach().cpu().numpy().astype(int)

        # A YOLO-seg checkpoint's `result.masks` gives one polygon per
        # detection, index-aligned with `result.boxes` (same detections,
        # same order) -- None for a plain detection checkpoint, in which
        # case every Detection below just keeps mask=None as before.
        masks_xy = None
        if getattr(result, "masks", None) is not None:
            masks_xy = result.masks.xy

        h_img, w_img = frame_bgr.shape[:2]

        for i in range(len(xyxy)):
            if self.class_ids is not None and cls_ids[i] not in self.class_ids:
                continue
            x1, y1, x2, y2 = xyxy[i]
            x1 = float(np.clip(x1, 0, w_img - 1))
            y1 = float(np.clip(y1, 0, h_img - 1))
            x2 = float(np.clip(x2, 0, w_img - 1))
            y2 = float(np.clip(y2, 0, h_img - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            bbox = BBox.from_xyxy(x1, y1, x2, y2)
            crop = frame_bgr[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1].copy()
            mask = None
            if (
                masks_xy is not None
                and i < len(masks_xy)
                and len(masks_xy[i]) >= 3
                and confs[i] >= self.mask_conf_threshold
            ):
                mask = np.asarray(masks_xy[i], dtype=np.float64)

            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=float(confs[i]),
                    frame_idx=frame_idx,
                    class_id=int(cls_ids[i]),
                    crop=crop,
                    mask=mask,
                )
            )

        return detections


class PrecomputedDetectionSource:
    """Alternative to LumenDetector: replay detections you've already saved
    to disk (e.g. if you ran YOLOv11 separately and exported per-frame
    boxes to JSON), instead of running inference live.

    Expects a JSON/dict of the form::

        {
          "0": [{"xyxy": [x1,y1,x2,y2], "confidence": 0.93}, ...],
          "1": [...],
          ...
        }

    keyed by frame index (as string or int).
    """

    def __init__(self, per_frame_detections: dict):
        self._data = {int(k): v for k, v in per_frame_detections.items()}

    @classmethod
    def from_json(cls, path: str) -> "PrecomputedDetectionSource":
        import json

        with open(path, "r") as f:
            data = json.load(f)
        return cls(data)

    def infer(self, frame_bgr: Optional[np.ndarray], frame_idx: int) -> List[Detection]:
        raw = self._data.get(frame_idx, [])
        detections = []
        for r in raw:
            x1, y1, x2, y2 = r["xyxy"]
            bbox = BBox.from_xyxy(x1, y1, x2, y2)
            crop = None
            if frame_bgr is not None:
                crop = frame_bgr[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1].copy()
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=float(r.get("confidence", 1.0)),
                    frame_idx=frame_idx,
                    class_id=int(r.get("class_id", 0)),
                    crop=crop,
                )
            )
        return detections
