"""LIME-style local surrogate explanations over rectangular spectrogram segments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances

from src.visualization import normalize_map, overlay_heatmap, save_heatmap


def generate_lime(model, x: torch.Tensor, mask: torch.Tensor, target: int, visual_logmel: np.ndarray, out_dir: Path, meta: dict, freq_bins: int = 8, time_bins: int = 8, samples: int = 256) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    segments = _segments(x.shape[1], x.shape[-2], x.shape[-1], freq_bins, time_bins)
    n_seg = len(segments)
    z = rng.integers(0, 2, size=(samples, n_seg), endpoint=False)
    z[0, :] = 1
    y = []
    baseline_value = float(x.min().detach().cpu())
    for row in z:
        xp = x.clone()
        for keep, (p, f0, f1, t0, t1) in zip(row, segments):
            if not keep:
                xp[:, p, :, f0:f1, t0:t1] = baseline_value
        with torch.no_grad():
            y.append(torch.softmax(model(xp, mask), dim=1)[0, target].item())
    distances = pairwise_distances(z, np.ones((1, n_seg)), metric="cosine").ravel()
    weights = np.sqrt(np.exp(-(distances**2) / 0.25))
    reg = Ridge(alpha=1.0)
    reg.fit(z, np.array(y), sample_weight=weights)
    importance = np.zeros((x.shape[1], x.shape[-2], x.shape[-1]), dtype=np.float32)
    rows = []
    for coef, seg in zip(reg.coef_, segments):
        p, f0, f1, t0, t1 = seg
        importance[p, f0:f1, t0:t1] = coef
        rows.append({"patch": p, "freq_start": f0, "freq_end": f1, "time_start": t0, "time_end": t1, "importance": float(coef)})
    heatmap = normalize_map(np.concatenate([p for p in importance], axis=1))
    np.save(out_dir / "lime_importance_map.npy", heatmap)
    pd.DataFrame(rows).sort_values("importance", ascending=False).head(50).to_csv(out_dir / "top_contributing_segments.csv", index=False)
    title = f"LIME: true={meta['true_label']} pred={meta['predicted_label']} conf={meta['confidence']:.3f}"
    save_heatmap(heatmap, out_dir / "heatmap.png", title)
    overlay_heatmap(visual_logmel, heatmap, out_dir / "overlay.png", title)
    (out_dir / "prediction_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({**meta, "method": "LIME"}, indent=2), encoding="utf-8")
    return heatmap


def _segments(patches: int, freq: int, time: int, freq_bins: int, time_bins: int) -> list[tuple[int, int, int, int, int]]:
    segs = []
    f_edges = np.linspace(0, freq, freq_bins + 1, dtype=int)
    t_edges = np.linspace(0, time, time_bins + 1, dtype=int)
    for p in range(patches):
        for i in range(freq_bins):
            for j in range(time_bins):
                segs.append((p, int(f_edges[i]), int(f_edges[i + 1]), int(t_edges[j]), int(t_edges[j + 1])))
    return segs
