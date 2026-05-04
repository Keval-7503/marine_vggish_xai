"""Pretrained VGGish weight acquisition utilities."""

from __future__ import annotations

import shutil
from pathlib import Path

import torch

from src.config import project_root, resolve_path
from src.model import VGGishClassifier
from src.utils import setup_logging, should_skip

LOGGER = setup_logging(__name__)


def download_vggish_weights(cfg: dict, overwrite: bool = False) -> Path:
    """Save VGGish backbone weights to models/pretrained/vggish_pytorch.pt.

    Preferred path uses torchaudio's pretrained VGGish bundle when installed.
    A torch.hub fallback is provided for environments that can access GitHub.
    """
    root = project_root(cfg)
    out_path = resolve_path(cfg, cfg["model"].get("pretrained_weights_path", "models/pretrained/vggish_pytorch.pt"))
    assert out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if should_skip(out_path, overwrite):
        LOGGER.info("VGGish weights already exist: %s", out_path)
        return out_path

    errors: list[str] = []
    if _try_torchaudio(out_path, errors):
        return out_path
    if _try_torchhub(out_path, errors):
        return out_path

    message = "\n".join(errors)
    raise RuntimeError(
        "Could not automatically download VGGish weights. Install torchaudio with prototype VGGISH support "
        "or place a compatible PyTorch VGGish state dict at models/pretrained/vggish_pytorch.pt.\n"
        f"Attempts:\n{message}"
    )


def _try_torchaudio(out_path: Path, errors: list[str]) -> bool:
    try:
        from torchaudio.prototype import pipelines

        bundle = pipelines.VGGISH
        model = bundle.get_model()
        state = model.state_dict()
        # Torchaudio's module names may differ from this repo's VGGishBackbone.
        # Store the original state for provenance and a normalized copy when possible.
        torch.save(state, out_path.with_name("vggish_torchaudio_raw.pt"))
        normalized = _normalize_torchaudio_state(state)
        if normalized:
            torch.save(normalized, out_path)
        else:
            shutil.copy2(out_path.with_name("vggish_torchaudio_raw.pt"), out_path)
        LOGGER.info("Saved VGGish weights from torchaudio to %s", out_path)
        return True
    except Exception as exc:
        errors.append(f"torchaudio: {exc!r}")
        return False


def _try_torchhub(out_path: Path, errors: list[str]) -> bool:
    try:
        model = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
        state = model.state_dict()
        torch.save(state, out_path.with_name("vggish_torchhub_raw.pt"))
        normalized = _normalize_torchhub_state(state)
        if normalized:
            torch.save(normalized, out_path)
        else:
            shutil.copy2(out_path.with_name("vggish_torchhub_raw.pt"), out_path)
        LOGGER.info("Saved VGGish weights from torch.hub to %s", out_path)
        return True
    except Exception as exc:
        errors.append(f"torch.hub harritaylor/torchvggish: {exc!r}")
        return False


def _normalize_torchhub_state(state: dict) -> dict:
    return _normalize_by_order(state)


def _normalize_torchaudio_state(state: dict) -> dict:
    return _normalize_by_order(state)


def _normalize_by_order(state: dict) -> dict:
    """Map common VGGish state dicts to this repo's backbone by tensor order.

    This is intentionally conservative: only same-shaped tensors are copied into
    the local architecture. The model loader still reports missing/unexpected
    keys if a downloaded variant is incompatible.
    """
    reference = VGGishClassifier(num_classes=1).backbone.state_dict()
    source_items = [(k, v.detach().cpu()) for k, v in state.items() if hasattr(v, "shape")]
    used: set[int] = set()
    out = {}
    for ref_key, ref_tensor in reference.items():
        for idx, (_, src_tensor) in enumerate(source_items):
            if idx in used:
                continue
            if tuple(src_tensor.shape) == tuple(ref_tensor.shape):
                out[ref_key] = src_tensor
                used.add(idx)
                break
    return out
