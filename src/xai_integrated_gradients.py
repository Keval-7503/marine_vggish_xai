"""Integrated Gradients attribution over VGGish log-Mel input."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.visualization import normalize_map, overlay_heatmap, save_heatmap


def generate_integrated_gradients(model, x: torch.Tensor, mask: torch.Tensor, target: int, visual_logmel: np.ndarray, out_dir: Path, meta: dict) -> np.ndarray:
    try:
        from captum.attr import IntegratedGradients

        def forward_func(inp):
            return model(inp, mask)

        ig = IntegratedGradients(forward_func)
        attr = ig.attribute(x, baselines=torch.zeros_like(x), target=target, n_steps=32)
        arr = attr.detach().abs().sum(dim=2).squeeze(0).cpu().numpy()
    except Exception:
        arr = _manual_integrated_gradients(model, x, mask, target)
    heatmap = normalize_map(np.concatenate([p for p in arr], axis=1))
    _save_outputs("Integrated Gradients", heatmap, visual_logmel, out_dir, meta)
    return heatmap


def _manual_integrated_gradients(model, x, mask, target, steps: int = 24) -> np.ndarray:
    baseline = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for alpha in torch.linspace(0, 1, steps, device=x.device):
        inp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        score = model(inp, mask)[0, target]
        grad = torch.autograd.grad(score, inp)[0]
        total += grad
    attr = (x - baseline) * total / steps
    return attr.detach().abs().sum(dim=2).squeeze(0).cpu().numpy()


def _save_outputs(name: str, heatmap: np.ndarray, visual_logmel: np.ndarray, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "raw_attribution.npy", heatmap)
    title = f"{name}: true={meta['true_label']} pred={meta['predicted_label']} conf={meta['confidence']:.3f}"
    save_heatmap(heatmap, out_dir / "heatmap.png", title)
    overlay_heatmap(visual_logmel, heatmap, out_dir / "overlay.png", title)
    (out_dir / "prediction_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({**meta, "method": name}, indent=2), encoding="utf-8")
