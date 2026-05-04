"""Training loop and checkpointing."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import project_root
from src.dataset import MarineVGGishDataset, collate_patches
from src.model import build_model
from src.utils import get_device, load_json, save_json, set_seed, setup_logging, should_skip
from src.visualization import plot_curves

LOGGER = setup_logging(__name__)


def train_model(cfg: dict, overwrite: bool = False) -> Path:
    root = project_root(cfg)
    best_path = root / "models/checkpoints/best_model.pt"
    if should_skip(best_path, overwrite):
        LOGGER.info("Best checkpoint exists: %s", best_path)
        return best_path
    set_seed(int(cfg["project"]["seed"]))
    label_map = load_json(root / "data/metadata/label_map.json")
    num_classes = len(label_map)
    device = get_device(cfg["project"]["device"])
    train_ds = MarineVGGishDataset(root / "data/processed/splits/train.csv", debug=bool(cfg["project"].get("debug")))
    val_ds = MarineVGGishDataset(root / "data/processed/splits/val.csv", debug=bool(cfg["project"].get("debug")))
    loader_args = dict(batch_size=int(cfg["training"]["batch_size"]), num_workers=int(cfg["training"]["num_workers"]), collate_fn=collate_patches)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    model = build_model(cfg, num_classes).to(device)
    optimizer = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["training"]["epochs"])))
    criterion = nn.CrossEntropyLoss(weight=_class_weights(train_ds.df, num_classes).to(device))
    ckpt_dir = root / "models/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    stale = 0
    history = []
    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_metrics = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = _run_epoch(model, val_loader, criterion, device, None)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}, "learning_rate": lr}
        history.append(row)
        epoch_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
        if cfg["training"].get("save_every_epoch", True):
            _save_checkpoint(epoch_path, model, cfg, label_map, row)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            stale = 0
            _save_checkpoint(best_path, model, cfg, label_map, row)
        else:
            stale += 1
        LOGGER.info("epoch=%d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f", epoch, train_metrics["loss"], val_metrics["loss"], val_metrics["macro_f1"])
        if stale >= int(cfg["training"]["early_stopping_patience"]):
            LOGGER.info("Early stopping after %d stale epochs.", stale)
            break
    final_path = ckpt_dir / "final_model.pt"
    _save_checkpoint(final_path, model, cfg, label_map, history[-1])
    logs = root / "outputs/logs"
    metrics_dir = root / "outputs/metrics"
    logs.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(logs / "training_log.csv", index=False)
    save_json({"epochs": history, "best_val_macro_f1": best_f1}, metrics_dir / "epoch_metrics.json")
    plot_curves(hist_df, root / "outputs/plots")
    return best_path


def _run_epoch(model, loader, criterion, device, optimizer=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    losses, y_true, y_pred = [], [], []
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, desc="train" if training else "eval", leave=False):
            x = batch["input"].to(device)
            mask = batch["patch_mask"].to(device)
            y = batch["label"].to(device)
            logits = model(x, mask)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            y_true.extend(y.detach().cpu().tolist())
            y_pred.extend(logits.argmax(1).detach().cpu().tolist())
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _class_weights(df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = df["label_id"].value_counts().reindex(range(num_classes), fill_value=1).sort_index().astype(float)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def _build_optimizer(model, cfg: dict):
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(cfg["training"].get("backbone_learning_rate", cfg["training"]["learning_rate"]))})
    if head_params:
        groups.append({"params": head_params, "lr": float(cfg["training"]["learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=float(cfg["training"]["weight_decay"]))


def _save_checkpoint(path: Path, model, cfg: dict, label_map: dict, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": cfg, "label_map": label_map, "metrics": metrics}, path)
