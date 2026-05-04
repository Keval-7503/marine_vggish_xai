"""Dataset classes and split creation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset

from src.config import project_root
from src.utils import setup_logging, should_skip
from src.visualization import save_bar

LOGGER = setup_logging(__name__)


class MarineVGGishDataset(Dataset):
    """Loads precomputed VGGish log-Mel examples and averages patch logits in the model."""

    def __init__(self, csv_path: Path, debug: bool = False):
        self.df = pd.read_csv(csv_path)
        if debug:
            self.df = self.df.groupby("label_id", group_keys=False).head(4).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.df.iloc[idx]
        x = np.load(row["vggish_path"]).astype(np.float32)
        return {
            "input": torch.from_numpy(x).unsqueeze(1),  # patches, channel, mel, time
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
            "sample_id": row["sample_id"],
            "species": row["species"],
            "logmel_path": row["logmel_path"],
            "processed_path": row["processed_path"],
        }


def collate_patches(batch: list[dict]) -> dict:
    max_patches = max(item["input"].shape[0] for item in batch)
    xs, masks = [], []
    for item in batch:
        x = item["input"]
        pad = max_patches - x.shape[0]
        if pad:
            x = torch.cat([x, torch.zeros((pad, *x.shape[1:]), dtype=x.dtype)], dim=0)
        xs.append(x)
        masks.append(torch.tensor([1] * item["input"].shape[0] + [0] * pad, dtype=torch.float32))
    return {
        "input": torch.stack(xs),
        "patch_mask": torch.stack(masks),
        "label": torch.stack([item["label"] for item in batch]),
        "sample_id": [item["sample_id"] for item in batch],
        "species": [item["species"] for item in batch],
        "logmel_path": [item["logmel_path"] for item in batch],
        "processed_path": [item["processed_path"] for item in batch],
    }


def create_splits(cfg: dict, overwrite: bool = False) -> None:
    root = project_root(cfg)
    in_csv = root / "data/metadata/preprocessed_metadata.csv"
    split_dir = root / "data/processed/splits"
    train_csv = split_dir / "train.csv"
    if should_skip(train_csv, overwrite):
        LOGGER.info("Split files already exist in %s", split_dir)
        return
    if not in_csv.exists():
        raise FileNotFoundError("Run 03_preprocess_audio.py first.")
    df = pd.read_csv(in_csv)
    if df["processed_path"].duplicated().any():
        raise RuntimeError("Duplicate processed paths detected before splitting.")
    seed = int(cfg["project"]["seed"])
    train_ratio = float(cfg["split"]["train_ratio"])
    val_ratio = float(cfg["split"]["val_ratio"])
    test_ratio = float(cfg["split"]["test_ratio"])
    temp_ratio = val_ratio + test_ratio
    if "source_file" in df.columns:
        train, temp = _grouped_stratified_split(df, test_size=temp_ratio, seed=seed)
        val_size = test_ratio / temp_ratio
        val, test = _grouped_stratified_split(temp, test_size=val_size, seed=seed)
    else:
        train, temp = train_test_split(df, test_size=temp_ratio, stratify=df["label_id"], random_state=seed)
        val_size = test_ratio / temp_ratio
        val, test = train_test_split(temp, test_size=val_size, stratify=temp["label_id"], random_state=seed)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.copy()
        part["split"] = name
        part.to_csv(split_dir / f"{name}.csv", index=False)
        LOGGER.info("%s split: %d rows", name, len(part))
    distributions = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        vc = part["species"].value_counts().rename_axis("species").reset_index(name="count")
        vc["split"] = name
        distributions.append(vc)
    dist = pd.concat(distributions)
    dist.to_csv(root / "outputs/metrics/split_class_distributions.csv", index=False)
    pivot = dist.pivot(index="species", columns="split", values="count").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(10, 5), title="Split class distribution")
    ax.set_ylabel("Clips")
    ax.figure.tight_layout()
    ax.figure.savefig(root / "outputs/plots/split_distribution.png", dpi=160)


def _grouped_stratified_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by source_file groups while approximating class stratification."""
    group_df = df.groupby("source_file", as_index=False).agg(label_id=("label_id", "first"))
    train_groups, test_groups = train_test_split(
        group_df,
        test_size=test_size,
        stratify=group_df["label_id"],
        random_state=seed,
    )
    train = df[df["source_file"].isin(train_groups["source_file"])].copy()
    test = df[df["source_file"].isin(test_groups["source_file"])].copy()
    return train, test
