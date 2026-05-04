"""Test evaluation and XAI orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, top_k_accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import project_root
from src.dataset import MarineVGGishDataset, collate_patches
from src.model import build_model
from src.utils import get_device, load_json, save_json, setup_logging, should_skip
from src.visualization import save_matrix
from src.xai_gradcam import generate_gradcam
from src.xai_gradient_methods import generate_guided_backprop, generate_input_x_gradient, generate_saliency, generate_smoothgrad
from src.xai_integrated_gradients import generate_integrated_gradients
from src.xai_lime import generate_lime
from src.xai_occlusion import generate_occlusion

LOGGER = setup_logging(__name__)


def load_trained_model(cfg: dict, checkpoint: Path | None = None):
    root = project_root(cfg)
    ckpt_path = checkpoint or root / "models/checkpoints/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_map = ckpt.get("label_map", load_json(root / "data/metadata/label_map.json"))
    model = build_model(cfg, len(label_map))
    model.load_state_dict(ckpt["model_state"])
    return model, label_map


def evaluate_test(cfg: dict, overwrite: bool = False) -> Path:
    root = project_root(cfg)
    metrics_path = root / "outputs/metrics/test_metrics.json"
    if should_skip(metrics_path, overwrite):
        LOGGER.info("Test metrics exist: %s", metrics_path)
        return metrics_path
    device = get_device(cfg["project"]["device"])
    model, label_map = load_trained_model(cfg)
    inv = {v: k for k, v in label_map.items()}
    labels = [inv[i] for i in range(len(inv))]
    model.to(device).eval()
    ds = MarineVGGishDataset(root / "data/processed/splits/test.csv", debug=bool(cfg["project"].get("debug")))
    loader = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=False, num_workers=int(cfg["training"]["num_workers"]), collate_fn=collate_patches)
    criterion = nn.CrossEntropyLoss()
    losses, rows, y_true, y_pred, probs_all = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test"):
            x, mask, y = batch["input"].to(device), batch["patch_mask"].to(device), batch["label"].to(device)
            logits = model(x, mask)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            losses.append(float(loss.cpu()))
            pred = probs.argmax(1)
            for i in range(len(pred)):
                rows.append({
                    "sample_id": batch["sample_id"][i],
                    "species": batch["species"][i],
                    "true_label": int(y[i].cpu()),
                    "pred_label": int(pred[i].cpu()),
                    "true_species": inv[int(y[i].cpu())],
                    "pred_species": inv[int(pred[i].cpu())],
                    "confidence": float(probs[i, pred[i]].cpu()),
                    "correct": bool(pred[i].cpu() == y[i].cpu()),
                    "processed_path": batch["processed_path"][i],
                    "logmel_path": batch["logmel_path"][i],
                    **{f"prob_{inv[j]}": float(probs[i, j].cpu()) for j in range(len(inv))},
                })
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            probs_all.extend(probs.cpu().numpy().tolist())
    pred_dir = root / "outputs/test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_dir / "test_predictions.csv", index=False)
    _save_confidence_slices(pred_df, pred_dir)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    save_matrix(cm, labels, root / "outputs/confusion_matrices/confusion_matrix_raw.png", "Raw confusion matrix")
    save_matrix(cm_norm, labels, root / "outputs/confusion_matrices/confusion_matrix_normalized.png", "Normalized confusion matrix")
    pr, rc, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(labels))), zero_division=0)
    metrics = {
        "test_loss": float(sum(losses) / max(1, len(losses))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top2_accuracy": float(top_k_accuracy_score(y_true, np.array(probs_all), k=min(2, len(labels)), labels=list(range(len(labels))))),
        "per_class": {labels[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(support[i])} for i in range(len(labels))},
    }
    save_json(metrics, metrics_path)
    (root / "outputs/metrics/classification_report.txt").write_text(classification_report(y_true, y_pred, target_names=labels, zero_division=0), encoding="utf-8")
    return metrics_path


def _save_confidence_slices(df: pd.DataFrame, out: Path) -> None:
    correct = df[df["correct"]].sort_values("confidence", ascending=False)
    incorrect = df[~df["correct"]].sort_values("confidence", ascending=False)
    correct.head(30).to_csv(out / "correct_high_confidence.csv", index=False)
    correct.tail(30).to_csv(out / "correct_low_confidence.csv", index=False)
    incorrect.head(30).to_csv(out / "incorrect_high_confidence.csv", index=False)
    incorrect.tail(30).to_csv(out / "incorrect_low_confidence.csv", index=False)


def select_xai_examples(cfg: dict) -> pd.DataFrame:
    root = project_root(cfg)
    df = pd.read_csv(root / "outputs/test_predictions/test_predictions.csv")
    selected = []
    for species, group in df.groupby("true_species"):
        selected.append(group[group["correct"]].sort_values("confidence", ascending=False).head(int(cfg["xai"]["examples_per_class_correct_high_conf"])))
        selected.append(group.sort_values("confidence", ascending=True).head(int(cfg["xai"]["examples_per_class_low_conf"])))
        selected.append(group[~group["correct"]].sort_values("confidence", ascending=False).head(int(cfg["xai"]["examples_per_class_incorrect"])))
    out = pd.concat(selected).drop_duplicates("sample_id").head(int(cfg["xai"]["max_examples_total"]))
    out.to_csv(root / "outputs/test_predictions/xai_selected_examples.csv", index=False)
    return out


def run_xai(cfg: dict, overwrite: bool = False) -> None:
    root = project_root(cfg)
    if not (root / "outputs/test_predictions/test_predictions.csv").exists():
        raise FileNotFoundError("Run 06_evaluate_test.py before XAI.")
    device = get_device(cfg["project"]["device"])
    model, label_map = load_trained_model(cfg)
    inv = {v: k for k, v in label_map.items()}
    model.to(device).eval()
    examples = select_xai_examples(cfg)
    methods = set(cfg["xai"]["methods"])
    for _, row in tqdm(examples.iterrows(), total=len(examples), desc="XAI examples"):
        x = torch.from_numpy(np.load(row["logmel_path"].replace("_visual.npy", "_vggish.npy"))).float().unsqueeze(0).unsqueeze(2).to(device)
        # The saved file is patches, mel, time; after unsqueeze -> batch, patches, channel, mel, time.
        mask = torch.ones((1, x.shape[1]), device=device)
        target = int(row["pred_label"])
        visual = np.load(row["logmel_path"])
        meta = {"sample_id": row["sample_id"], "true_label": row["true_species"], "predicted_label": row["pred_species"], "confidence": float(row["confidence"])}
        for method, fn in [
            ("gradcam", generate_gradcam),
            ("integrated_gradients", generate_integrated_gradients),
            ("occlusion", generate_occlusion),
            ("lime", generate_lime),
            ("saliency", generate_saliency),
            ("smoothgrad", generate_smoothgrad),
            ("input_x_gradient", generate_input_x_gradient),
            ("guided_backprop", generate_guided_backprop),
        ]:
            if method not in methods:
                continue
            out_dir = root / "outputs/xai" / method / str(row["true_species"]) / str(row["sample_id"])
            if out_dir.exists() and not overwrite:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["processed_path"], out_dir / "original_waveform.wav")
            fn(model, x, mask, target, visual, out_dir, meta)
