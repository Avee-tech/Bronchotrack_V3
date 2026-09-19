"""Appearance Re-ID module (paper section 3, "Appearance Matching").

The paper trains a ResNet50 with a softmax classification head on 3,630
cropped 128x128 lumen images (10 patients) and uses the penultimate-layer
feature as a 2048-d embedding, updated per-tracklet with an exponential
moving average:

    e_i^t = alpha * e_i^{t-1} + (1 - alpha) * f_i^t         (alpha = 0.9)

``ReIDEmbedder`` below gives you that same ResNet50-backbone embedding
extractor. Two ways to use it:

  1. Pass ``weights_path`` pointing at a checkpoint you've fine-tuned on
     your own lumen crops (recommended, and required to actually match the
     paper's reported robustness -- a generic ImageNet backbone will produce
     *some* signal but was not trained to distinguish airway lumens).
  2. Leave ``weights_path=None`` to fall back to an ImageNet-pretrained
     ResNet50 with its classification head removed, used purely as a
     generic feature extractor. This still gives the tracker useful
     appearance signal (texture/color of the lumen opening and surrounding
     mucosa) but is a placeholder for a properly fine-tuned model.

``torch``/``torchvision`` are imported lazily so the rest of the package
works without them installed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .types import Tracklet
from .utils import resolve_device

DEFAULT_ALPHA = 0.9
CROP_SIZE = 128


class ReIDEmbedder:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        crop_size: int = CROP_SIZE,
    ):
        self.weights_path = weights_path
        self.device = device
        self.crop_size = crop_size
        self._model = None
        self._torch = None
        self._transform = None

    def warm_up(self) -> None:
        """Load the model and resolve `self.device` now, instead of lazily
        on the first `embed()` call -- useful so callers (e.g. the CLI) can
        report which device Re-ID actually runs on before processing
        starts."""
        self._ensure_model()

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
            import torch.nn as nn
            import torchvision
            from torchvision import transforms
        except ImportError as e:
            raise ImportError(
                "torch and torchvision are required for ReIDEmbedder. "
                "Install with `pip install torch torchvision`."
            ) from e

        self._torch = torch
        device = resolve_device(self.device)
        self.device = device

        backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
            if self.weights_path is None
            else None
        )
        backbone.fc = nn.Identity()  # use 2048-d pooled feature as embedding

        if self.weights_path is not None:
            state_dict = torch.load(self.weights_path, map_location=device)
            # allow both a raw state_dict and a {"model": state_dict} checkpoint
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            backbone.load_state_dict(state_dict, strict=False)

        backbone.eval()
        backbone.to(device)
        self._model = backbone

        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Embed a single BGR image crop into a 2048-d L2-normalized vector."""
        self._ensure_model()
        torch = self._torch

        crop_rgb = crop_bgr[:, :, ::-1]  # BGR -> RGB
        tensor = self._transform(np.ascontiguousarray(crop_rgb)).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            feat = self._model(tensor)
        feat = feat.squeeze(0).cpu().numpy().astype(np.float64)
        norm = np.linalg.norm(feat)
        if norm > 1e-9:
            feat = feat / norm
        return feat

    def embed_batch(self, crops) -> np.ndarray:
        return np.stack([self.embed(c) for c in crops], axis=0) if crops else np.empty((0, 0))


def ema_update_embedding(
    prev_embedding: Optional[np.ndarray],
    new_embedding: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """e_i^t = alpha * e_i^{t-1} + (1 - alpha) * f_i^t, renormalized to unit
    length so cosine similarity in matching.py stays well-behaved."""
    if prev_embedding is None:
        updated = new_embedding
    else:
        updated = alpha * prev_embedding + (1.0 - alpha) * new_embedding
    norm = np.linalg.norm(updated)
    if norm > 1e-9:
        updated = updated / norm
    return updated


def update_tracklet_embedding(
    tracklet: Tracklet, new_embedding: np.ndarray, alpha: float = DEFAULT_ALPHA
) -> None:
    tracklet.embedding = ema_update_embedding(tracklet.embedding, new_embedding, alpha)
