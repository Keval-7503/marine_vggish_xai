"""Additional gradient-based XAI methods for VGGish log-Mel inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.visualization import normalize_map, overlay_heatmap, save_heatmap


def generate_saliency(model, x: torch.Tensor, mask: torch.Tensor, target: int, visual_logmel: np.ndarray, out_dir: Path, meta: dict) -> np.ndarray:
    """Absolute input gradient for the predicted class."""
    attr = _input_gradient(model, x, mask, target).abs()
    heatmap = _to_heatmap(attr)
    _save_outputs("Saliency", "raw_saliency.npy", heatmap, visual_logmel, out_dir, meta)
    return heatmap


def generate_input_x_gradient(model, x: torch.Tensor, mask: torch.Tensor, target: int, visual_logmel: np.ndarray, out_dir: Path, meta: dict) -> np.ndarray:
    """Elementwise input times gradient attribution."""
    grad = _input_gradient(model, x, mask, target)
    attr = (x.detach() * grad).abs()
    heatmap = _to_heatmap(attr)
    _save_outputs("Input x Gradient", "raw_input_x_gradient.npy", heatmap, visual_logmel, out_dir, meta)
    return heatmap


def generate_smoothgrad(model, x: torch.Tensor, mask: torch.Tensor, target: int, visual_logmel: np.ndarray, out_dir: Path, meta: dict, samples: int = 24, noise_std: float = 0.08) -> np.ndarray:
    """Average saliency over noisy copies of the input."""
    generator = torch.Generator(device=x.device)
    generator.manual_seed(0)
    total = torch.zeros_like(x)
    span = (x.max() - x.min()).detach().clamp_min(1e-6)
    for _ in range(samples):
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * noise_std * span
        noisy = (x.detach() + noise).clamp(0.0, 1.0)
        total += _input_gradient(model, noisy, mask, target).abs()
    attr = total / float(samples)
    heatmap = _to_heatmap(attr)
    _save_outputs("SmoothGrad", "raw_smoothgrad.npy", heatmap, visual_logmel, out_dir, meta)
    return heatmap


def generate_guided_backprop(model, x: torch.Tensor, mask: torch.Tensor, target: int, visual_logmel: np.ndarray, out_dir: Path, meta: dict) -> np.ndarray:
    """Guided backpropagation using ReLU backward hooks."""
    hooks = []
    relu_states = []

    def relu_hook(_module, grad_input, _grad_output):
        return tuple(None if g is None else torch.clamp(g, min=0.0) for g in grad_input)

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            relu_states.append((module, module.inplace))
            module.inplace = False
            hooks.append(module.register_full_backward_hook(relu_hook))
    try:
        attr = _input_gradient(model, x, mask, target).abs()
    finally:
        for hook in hooks:
            hook.remove()
        for module, inplace in relu_states:
            module.inplace = inplace
    heatmap = _to_heatmap(attr)
    _save_outputs("Guided Backpropagation", "raw_guided_backprop.npy", heatmap, visual_logmel, out_dir, meta)
    return heatmap


def _input_gradient(model, x: torch.Tensor, mask: torch.Tensor, target: int) -> torch.Tensor:
    inp = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    score = model(inp, mask)[0, target]
    grad = torch.autograd.grad(score, inp, retain_graph=False, create_graph=False)[0]
    return grad.detach()


def _to_heatmap(attr: torch.Tensor) -> np.ndarray:
    arr = attr.detach().sum(dim=2).squeeze(0).cpu().numpy()
    return normalize_map(np.concatenate([patch for patch in arr], axis=1))


def _save_outputs(name: str, filename: str, heatmap: np.ndarray, visual_logmel: np.ndarray, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / filename, heatmap)
    title = f"{name}: true={meta['true_label']} pred={meta['predicted_label']} conf={meta['confidence']:.3f}"
    save_heatmap(heatmap, out_dir / "heatmap.png", title)
    overlay_heatmap(visual_logmel, heatmap, out_dir / "overlay.png", title)
    (out_dir / "prediction_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({**meta, "method": name}, indent=2), encoding="utf-8")
