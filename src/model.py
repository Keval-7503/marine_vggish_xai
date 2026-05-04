"""PyTorch VGGish-style classifier with optional pretrained weight loading."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.config import project_root
from src.utils import setup_logging

LOGGER = setup_logging(__name__)


class VGGishBackbone(nn.Module):
    """VGGish-compatible convolutional backbone for 1x64x96 log-Mel examples.

    The layer layout follows the public VGGish architecture. Users should place a
    PyTorch state dict at models/pretrained/vggish_pytorch.pt for true pretrained
    operation. The pipeline refuses to train from scratch unless debug mode is on
    or model.pretrained is set false intentionally.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.embeddings = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.embeddings(x)

    @property
    def last_conv(self) -> nn.Module:
        return self.features[13]


class VGGishClassifier(nn.Module):
    """Classifies one or more VGGish patches per clip by mean pooling embeddings."""

    def __init__(self, num_classes: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.backbone = VGGishBackbone()
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: batch, patches, channel, mel, time
        b, p, c, m, t = x.shape
        emb = self.backbone(x.reshape(b * p, c, m, t)).reshape(b, p, -1)
        if patch_mask is None:
            pooled = emb.mean(dim=1)
        else:
            mask = patch_mask.unsqueeze(-1).clamp_min(0.0)
            pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def build_model(cfg: dict, num_classes: int) -> VGGishClassifier:
    model = VGGishClassifier(
        num_classes=num_classes,
        hidden_dim=int(cfg["model"]["classifier_hidden_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    if cfg["model"].get("pretrained", True):
        load_pretrained_vggish(model, cfg)
    if cfg["model"].get("freeze_backbone", True) or cfg["model"].get("training_mode") == "linear_probe":
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif cfg["model"].get("training_mode") == "fine_tune_last":
        for p in model.backbone.parameters():
            p.requires_grad = False
        for p in model.backbone.features[-4:].parameters():
            p.requires_grad = True
    return model


def load_pretrained_vggish(model: VGGishClassifier, cfg: dict) -> None:
    root = project_root(cfg)
    candidates = [
        root / "models/pretrained/vggish_pytorch.pt",
        root / "models/pretrained/vggish.pth",
        root / "models/pretrained/vggish_state_dict.pt",
    ]
    ckpt = next((p for p in candidates if p.exists()), None)
    if ckpt is None:
        if cfg["project"].get("debug") or not cfg["model"].get("pretrained", True):
            LOGGER.warning("No pretrained VGGish weights found; continuing only because debug/non-pretrained mode is active.")
            return
        raise FileNotFoundError(
            "Pretrained VGGish weights are required. Place a PyTorch VGGish state dict at "
            "models/pretrained/vggish_pytorch.pt, or set project.debug=true for smoke tests."
        )
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for k, v in state.items():
        key = k.replace("module.", "")
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        cleaned[key] = v
    missing, unexpected = model.backbone.load_state_dict(cleaned, strict=False)
    LOGGER.info("Loaded VGGish weights from %s (missing=%d unexpected=%d)", ckpt, len(missing), len(unexpected))
