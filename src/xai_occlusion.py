"""Time-frequency occlusion sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.visualization import normalize_map, overlay_heatmap, save_heatmap


def generate_occlusion(model, x: torch.Tensor, mask: torch.Tensor, target: int, visual_logmel: np.ndarray, out_dir: Path, meta: dict, freq_step: int = 8, time_step: int = 12) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        base = torch.softmax(model(x, mask), dim=1)[0, target].item()
    scores = np.zeros((x.shape[1], x.shape[-2], x.shape[-1]), dtype=np.float32)
    rows = []
    baseline_value = float(x.min().detach().cpu())
    for p in range(x.shape[1]):
        for f0 in range(0, x.shape[-2], freq_step):
            for t0 in range(0, x.shape[-1], time_step):
                x_occ = x.clone()
                f1, t1 = min(f0 + freq_step, x.shape[-2]), min(t0 + time_step, x.shape[-1])
                x_occ[:, p, :, f0:f1, t0:t1] = baseline_value
                with torch.no_grad():
                    conf = torch.softmax(model(x_occ, mask), dim=1)[0, target].item()
                drop = base - conf
                scores[p, f0:f1, t0:t1] = drop
                rows.append({"patch": p, "freq_start": f0, "freq_end": f1, "time_start": t0, "time_end": t1, "base_confidence": base, "masked_confidence": conf, "confidence_drop": drop})
    heatmap = normalize_map(np.concatenate([p for p in scores], axis=1))
    np.save(out_dir / "raw_occlusion_map.npy", heatmap)
    pd.DataFrame(rows).to_csv(out_dir / "confidence_drop.csv", index=False)
    title = f"Occlusion: true={meta['true_label']} pred={meta['predicted_label']} conf={meta['confidence']:.3f}"
    save_heatmap(heatmap, out_dir / "heatmap.png", title)
    overlay_heatmap(visual_logmel, heatmap, out_dir / "overlay.png", title)
    (out_dir / "prediction_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({**meta, "method": "Occlusion"}, indent=2), encoding="utf-8")
    return heatmap
