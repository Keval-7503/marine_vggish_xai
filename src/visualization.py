"""Plotting helpers and report asset assembly."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import project_root


def save_bar(counts: pd.Series, path: Path, title: str, xlabel: str = "Species", ylabel: str = "Clips") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    counts.plot(kind="bar", ax=ax, color="#3b6ea8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_curves(metrics: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("loss", "train_loss", "val_loss", "Loss", "training_loss_curve.png"),
        ("accuracy", "train_accuracy", "val_accuracy", "Accuracy", "training_accuracy_curve.png"),
        ("macro_f1", "train_macro_f1", "val_macro_f1", "Macro-F1", "training_macro_f1_curve.png"),
        ("weighted_f1", "train_weighted_f1", "val_weighted_f1", "Weighted-F1", "training_weighted_f1_curve.png"),
    ]
    for _, train_col, val_col, ylabel, name in specs:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
        ax.plot(metrics["epoch"], metrics[train_col], label="train")
        ax.plot(metrics["epoch"], metrics[val_col], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Training {ylabel}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / name)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.plot(metrics["epoch"], metrics["learning_rate"], color="#7a4fa3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning-rate schedule")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_rate_curve.png")
    plt.close(fig)


def save_matrix(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=170)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            txt = f"{value:.2f}" if matrix.dtype.kind == "f" else str(int(value))
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_spectrogram(logmel: np.ndarray, path: Path, title: str = "Log-Mel spectrogram") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=170)
    im = ax.imshow(logmel, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel frequency")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def overlay_heatmap(logmel: np.ndarray, heatmap: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heatmap = resize_like(heatmap, logmel.shape)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=190)
    ax.imshow(logmel, origin="lower", aspect="auto", cmap="magma")
    ax.imshow(heatmap, origin="lower", aspect="auto", cmap="jet", alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel frequency")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_heatmap(heatmap: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4), dpi=170)
    im = ax.imshow(heatmap, origin="lower", aspect="auto", cmap="jet")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel frequency")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def resize_like(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    try:
        from PIL import Image

        arr = normalize_map(x)
        im = Image.fromarray((arr * 255).astype(np.uint8))
        im = im.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
        return np.asarray(im).astype(np.float32) / 255.0
    except Exception:
        return np.resize(x, shape)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def generate_report_assets(cfg: dict, overwrite: bool = False) -> None:
    root = project_root(cfg)
    out = root / "outputs" / "report_assets"
    out.mkdir(parents=True, exist_ok=True)
    copy_if_exists(root / "outputs/plots/species_distribution_after.png", out / "dataset_species_distribution.png")
    copy_if_exists(root / "outputs/plots/training_loss_curve.png", out / "training_loss_curve.png")
    copy_if_exists(root / "outputs/confusion_matrices/confusion_matrix_normalized.png", out / "confusion_matrix_normalized.png")
    copy_if_exists(root / "outputs/xai_evaluation/faithfulness_topk_deletion_plot.png", out / "faithfulness_deletion_curve.png")
    copy_if_exists(root / "outputs/xai_evaluation/explanation_agreement_heatmap.png", out / "explanation_agreement_heatmap.png")
    copy_if_exists(root / "outputs/xai_evaluation/stability_summary_plot.png", out / "stability_summary_plot.png")
    copy_if_exists(root / "outputs/xai_evaluation/energy_alignment_plot.png", out / "energy_alignment_plot.png")
    _simple_diagram(out / "preprocessing_pipeline_diagram.png", ["Raw audio", "16 kHz mono", "5 s crop/pad", "64-bin VGGish log-Mel", "Classifier/XAI"])
    _simple_diagram(out / "model_architecture_diagram.png", ["Log-Mel input", "Pretrained VGGish conv stack", "128-d embedding", "MLP classifier", "Species logits"])
    _combine_training_curves(root, out / "training_curves_combined.png")
    _write_summary(root, out / "final_results_summary.md")


def _simple_diagram(path: Path, labels: Iterable[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 2.3), dpi=170)
    ax.axis("off")
    labels = list(labels)
    for i, label in enumerate(labels):
        x = i / max(1, len(labels) - 1)
        ax.text(x, 0.5, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#eef3f8", ec="#2f5f8f"))
        if i < len(labels) - 1:
            ax.annotate("", xy=((i + 0.72) / (len(labels) - 1), 0.5), xytext=((i + 0.28) / (len(labels) - 1), 0.5), arrowprops=dict(arrowstyle="->"))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _combine_training_curves(root: Path, path: Path) -> None:
    log = root / "outputs/logs/training_log.csv"
    if not log.exists():
        return
    df = pd.read_csv(log)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=170)
    for ax, col, title in zip(axes.ravel(), ["loss", "accuracy", "macro_f1", "weighted_f1"], ["Loss", "Accuracy", "Macro-F1", "Weighted-F1"]):
        ax.plot(df["epoch"], df[f"train_{col}"], label="train")
        ax.plot(df["epoch"], df[f"val_{col}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_summary(root: Path, path: Path) -> None:
    selected = root / "data/metadata/selected_balanced_metadata.csv"
    metrics = root / "outputs/metrics/test_metrics.json"
    train_log = root / "outputs/logs/training_log.csv"
    faith = root / "outputs/xai_evaluation/faithfulness_topk_deletion.csv"
    energy = root / "outputs/xai_evaluation/energy_alignment.csv"
    agreement = root / "outputs/xai_evaluation/explanation_agreement_iou.csv"
    lines = ["# Final Results Summary", ""]
    if selected.exists():
        df = pd.read_csv(selected)
        lines += ["## Dataset", ""]
        for species, count in df["species"].value_counts().items():
            lines.append(f"- {species}: {int(count)} source clips")
        lines.append("")
    if train_log.exists():
        tl = pd.read_csv(train_log)
        best = tl.sort_values("val_macro_f1", ascending=False).iloc[0].to_dict()
        lines += ["## Best Validation Metrics", "", f"- Epoch: {int(best['epoch'])}", f"- Validation macro-F1: {best['val_macro_f1']:.4f}", f"- Validation accuracy: {best['val_accuracy']:.4f}", ""]
    if metrics.exists():
        import json

        m = json.loads(metrics.read_text(encoding="utf-8"))
        lines += ["## Test Metrics", "", f"- Accuracy: {m.get('accuracy', float('nan')):.4f}", f"- Macro-F1: {m.get('macro_f1', float('nan')):.4f}", f"- Weighted-F1: {m.get('weighted_f1', float('nan')):.4f}", ""]
    lines += ["## XAI Findings", ""]
    if faith.exists():
        fdf = pd.read_csv(faith)
        xai = fdf[fdf["baseline"] == "xai"].copy()
        xai["confidence_drop"] = pd.to_numeric(xai["confidence_drop"], errors="coerce")
        best_rows = []
        for method, group in xai.groupby("method"):
            by_topk = group.groupby("topk_percent")["confidence_drop"].mean()
            topk = by_topk.idxmax()
            best_rows.append((method, int(topk), float(by_topk.loc[topk])))
        best_rows.sort(key=lambda row: row[2], reverse=True)
        lines.append("- Best deletion faithfulness by method:")
        for method, topk, drop in best_rows:
            lines.append(f"  - {method}: top-{topk}% deletion mean confidence drop = {drop:.4f}")
        lines.append("")
    if energy.exists():
        edf = pd.read_csv(energy)
        energy_summary = edf.groupby("method")[["high_energy_overlap", "low_energy_overlap", "clip_border_overlap"]].mean().sort_values("high_energy_overlap", ascending=False)
        best_method = energy_summary.index[0]
        best_high = float(energy_summary.iloc[0]["high_energy_overlap"])
        lines.append(f"- Highest high-energy alignment: {best_method} ({best_high:.4f}).")
        lines.append("- Mean energy/artifact overlaps:")
        for method, row in energy_summary.iterrows():
            lines.append(f"  - {method}: high={row['high_energy_overlap']:.4f}, low={row['low_energy_overlap']:.4f}, border={row['clip_border_overlap']:.4f}")
        lines.append("")
    if agreement.exists():
        adf = pd.read_csv(agreement)
        if not adf.empty:
            best = adf.sort_values("mean_iou_top20", ascending=False).iloc[0]
            lines.append(f"- Strongest explanation agreement: {best['method_a']} vs. {best['method_b']} (top-20% IoU = {float(best['mean_iou_top20']):.4f}).")
            lines.append("")
    if not (faith.exists() or energy.exists() or agreement.exists()):
        lines.append("Populate this section after running XAI evaluation. Use cautious language: the model appears to rely on visible high-energy acoustic regions when deletion and energy-alignment metrics support that interpretation.")
        lines.append("")
    lines += ["## Limitations", "", "The pipeline evaluates alignment with visible acoustic energy, not biological intent or causal vocalization semantics. Manual expert review remains necessary."]
    path.write_text("\n".join(lines), encoding="utf-8")
