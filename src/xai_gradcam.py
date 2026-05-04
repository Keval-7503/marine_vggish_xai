"""Grad-CAM for the VGGish convolutional stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.visualization import normalize_map, overlay_heatmap, save_heatmap


def generate_gradcam(model, x: torch.Tensor, mask: torch.Tensor, target: int, visual_logmel: np.ndarray, out_dir: Path, meta: dict) -> np.ndarray:
    x = x.detach().clone().requires_grad_(True)
    activations, gradients = [], []

    def fwd_hook(_, __, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    h1 = model.backbone.last_conv.register_forward_hook(fwd_hook)
    model.zero_grad(set_to_none=True)
    logits = model(x, mask)
    score = logits[0, target]
    score.backward()
    h1.remove()
    act = activations[-1]
    grad = gradients[-1]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(64, 96), mode="bilinear", align_corners=False).squeeze(1)
    patches = cam.detach().cpu().numpy()
    heatmap = normalize_map(np.concatenate([p for p in patches], axis=1))
    _save_outputs("Grad-CAM", heatmap, visual_logmel, out_dir, meta)
    return heatmap


def _save_outputs(name: str, heatmap: np.ndarray, visual_logmel: np.ndarray, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "raw_heatmap.npy", heatmap)
    title = f"{name}: true={meta['true_label']} pred={meta['predicted_label']} conf={meta['confidence']:.3f}"
    save_heatmap(heatmap, out_dir / "heatmap.png", title)
    overlay_heatmap(visual_logmel, heatmap, out_dir / "overlay.png", title)
    (out_dir / "prediction_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({**meta, "method": name}, indent=2), encoding="utf-8")
