"""Quantitative evaluation of generated explanation maps."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from src.config import project_root
from src.evaluate import load_trained_model
from src.utils import get_device, setup_logging, should_skip
from src.visualization import normalize_map, resize_like, save_matrix

LOGGER = setup_logging(__name__)


METHOD_FILES = {
    "gradcam": "raw_heatmap.npy",
    "integrated_gradients": "raw_attribution.npy",
    "occlusion": "raw_occlusion_map.npy",
    "lime": "lime_importance_map.npy",
    "saliency": "raw_saliency.npy",
    "smoothgrad": "raw_smoothgrad.npy",
    "input_x_gradient": "raw_input_x_gradient.npy",
    "guided_backprop": "raw_guided_backprop.npy",
}


def evaluate_xai_outputs(cfg: dict, overwrite: bool = False) -> None:
    root = project_root(cfg)
    out = root / "outputs/xai_evaluation"
    faith_path = out / "faithfulness_topk_deletion.csv"
    if should_skip(faith_path, overwrite):
        LOGGER.info("XAI evaluation already exists: %s", out)
        return
    pred_path = root / "outputs/test_predictions/xai_selected_examples.csv"
    if not pred_path.exists():
        raise FileNotFoundError("Run 07_run_xai.py before evaluating XAI.")
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(pred_path)
    model, _ = load_trained_model(cfg)
    device = get_device(cfg["project"]["device"])
    model.to(device).eval()
    maps = _load_maps(root, preds)
    _faithfulness(cfg, model, device, preds, maps, out)
    _agreement(maps, out)
    _stability(maps, out)
    _energy_alignment(preds, maps, out)


def _load_maps(root: Path, preds: pd.DataFrame) -> dict:
    result = {}
    for _, row in preds.iterrows():
        sid = row["sample_id"]
        result[sid] = {"row": row, "maps": {}}
        for method, filename in METHOD_FILES.items():
            p = root / "outputs/xai" / method / str(row["true_species"]) / sid / filename
            if p.exists():
                result[sid]["maps"][method] = normalize_map(np.load(p))
    return result


def _faithfulness(cfg, model, device, preds, maps, out: Path) -> None:
    rows = []
    rng = np.random.default_rng(int(cfg["project"]["seed"]))
    topks = [int(k) for k in cfg["xai"]["topk_percentages"]]
    for sid, item in tqdm(maps.items(), desc="Deletion faithfulness"):
        row = item["row"]
        x = torch.from_numpy(np.load(str(row["logmel_path"]).replace("_visual.npy", "_vggish.npy"))).float().unsqueeze(0).unsqueeze(2).to(device)
        mask = torch.ones((1, x.shape[1]), device=device)
        target = int(row["pred_label"])
        with torch.no_grad():
            base = torch.softmax(model(x, mask), dim=1)[0, target].item()
        for method, heat in item["maps"].items():
            heat_in = resize_like(heat, (x.shape[2] * x.shape[-2], x.shape[1] * x.shape[-1])).reshape(x.shape[2], x.shape[-2], x.shape[1], x.shape[-1]).transpose(2, 0, 1, 3)
            heat_in = heat_in.reshape(x.shape[1], x.shape[-2], x.shape[-1])
            for k in topks:
                conf = _masked_conf(model, x, mask, target, heat_in, k, rng=None)
                rows.append({"sample_id": sid, "method": method, "topk_percent": k, "baseline": "xai", "base_confidence": base, "masked_confidence": conf, "confidence_drop": base - conf})
                vals = []
                for _ in range(int(cfg["xai"]["random_baseline_trials"])):
                    vals.append(_masked_conf(model, x, mask, target, heat_in, k, rng=rng))
                rows.append({"sample_id": sid, "method": method, "topk_percent": k, "baseline": "random", "base_confidence": base, "masked_confidence": float(np.mean(vals)), "masked_confidence_std": float(np.std(vals)), "confidence_drop": base - float(np.mean(vals))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "faithfulness_topk_deletion.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=170)
    for (method, baseline), g in df.groupby(["method", "baseline"]):
        agg = g.groupby("topk_percent")["masked_confidence"].mean()
        ax.plot(agg.index, agg.values, marker="o", label=f"{method} {baseline}")
    ax.set_title("Faithfulness by top-k deletion")
    ax.set_xlabel("Masked top-k percentage")
    ax.set_ylabel("Predicted-class confidence")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "faithfulness_topk_deletion_plot.png")
    plt.close(fig)


def _masked_conf(model, x, mask, target, heat, k: int, rng=None) -> float:
    xp = x.clone()
    flat = heat.reshape(-1)
    n = max(1, int(flat.size * k / 100))
    if rng is None:
        idx = np.argpartition(flat, -n)[-n:]
    else:
        idx = rng.choice(flat.size, size=n, replace=False)
    mask_np = np.zeros(flat.size, dtype=bool)
    mask_np[idx] = True
    mask_np = mask_np.reshape(heat.shape)
    baseline = float(x.min().detach().cpu())
    for p in range(x.shape[1]):
        m = torch.from_numpy(mask_np[p]).to(x.device)
        xp[0, p, 0][m] = baseline
    with torch.no_grad():
        return torch.softmax(model(xp, mask), dim=1)[0, target].item()


def _agreement(maps: dict, out: Path) -> None:
    rows = []
    methods = list(METHOD_FILES)
    matrix = np.full((len(methods), len(methods)), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            vals = []
            for sid, item in maps.items():
                if a in item["maps"] and b in item["maps"]:
                    ma = _top_mask(item["maps"][a], 20)
                    mb = _top_mask(resize_like(item["maps"][b], ma.shape), 20)
                    inter = np.logical_and(ma, mb).sum()
                    union = np.logical_or(ma, mb).sum()
                    vals.append(inter / max(1, union))
            if vals:
                matrix[i, j] = float(np.mean(vals))
                if i < j:
                    rows.append({"method_a": a, "method_b": b, "mean_iou_top20": matrix[i, j]})
    pd.DataFrame(rows).to_csv(out / "explanation_agreement_iou.csv", index=False)
    save_matrix(np.nan_to_num(matrix), methods, out / "explanation_agreement_heatmap.png", "Explanation agreement IoU")


def _stability(maps: dict, out: Path) -> None:
    rows = []
    for sid, item in maps.items():
        for method, heat in item["maps"].items():
            perturbations = {
                "gaussian_noise": normalize_map(heat + np.random.default_rng(0).normal(0, 0.03, heat.shape)),
                "time_shift": np.roll(heat, shift=3, axis=1),
                "amplitude_scaling": normalize_map(heat * 0.9),
                "frequency_masking": _freq_mask(heat),
            }
            base_mask = _top_mask(heat, 20)
            for name, pert in perturbations.items():
                rho = spearmanr(heat.reshape(-1), pert.reshape(-1)).correlation
                pmask = _top_mask(pert, 20)
                iou = np.logical_and(base_mask, pmask).sum() / max(1, np.logical_or(base_mask, pmask).sum())
                rows.append({"sample_id": sid, "method": method, "perturbation": name, "spearman": float(np.nan_to_num(rho)), "top20_iou": float(iou), "confidence_change": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(out / "stability_results.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=170)
    df.groupby(["method", "perturbation"])["top20_iou"].mean().unstack(0).plot(kind="bar", ax=ax)
    ax.set_title("Explanation stability under small perturbations")
    ax.set_ylabel("Top-20% IoU")
    fig.tight_layout()
    fig.savefig(out / "stability_summary_plot.png")
    plt.close(fig)


def _energy_alignment(preds: pd.DataFrame, maps: dict, out: Path) -> None:
    rows = []
    for sid, item in maps.items():
        row = item["row"]
        visual = np.load(row["logmel_path"])
        energy = normalize_map(visual)
        high_energy = energy >= np.quantile(energy, 0.75)
        border = np.zeros_like(high_energy, dtype=bool)
        border[:, : max(1, high_energy.shape[1] // 20)] = True
        border[:, -max(1, high_energy.shape[1] // 20) :] = True
        low_energy = energy <= np.quantile(energy, 0.25)
        for method, heat in item["maps"].items():
            h = resize_like(heat, visual.shape)
            top = _top_mask(h, 20)
            rows.append({
                "sample_id": sid,
                "method": method,
                "true_species": row["true_species"],
                "correct": bool(row["correct"]),
                "high_energy_overlap": float(np.logical_and(top, high_energy).sum() / max(1, top.sum())),
                "low_energy_overlap": float(np.logical_and(top, low_energy).sum() / max(1, top.sum())),
                "clip_border_overlap": float(np.logical_and(top, border).sum() / max(1, top.sum())),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "energy_alignment.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=170)
    df.groupby("method")[["high_energy_overlap", "low_energy_overlap", "clip_border_overlap"]].mean().plot(kind="bar", ax=ax)
    ax.set_title("Shortcut/artifact audit by attribution overlap")
    ax.set_ylabel("Fraction of top attribution")
    fig.tight_layout()
    fig.savefig(out / "energy_alignment_plot.png")
    plt.close(fig)


def _top_mask(x: np.ndarray, percent: int) -> np.ndarray:
    flat = x.reshape(-1)
    n = max(1, int(flat.size * percent / 100))
    threshold = np.partition(flat, -n)[-n]
    return x >= threshold


def _freq_mask(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    h = max(1, y.shape[0] // 10)
    start = y.shape[0] // 3
    y[start : start + h, :] = 0
    return normalize_map(y)
