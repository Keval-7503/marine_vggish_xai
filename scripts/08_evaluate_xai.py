"""Quantitatively evaluate XAI outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ensure_project_dirs, load_config  # noqa: E402
from src.xai_metrics import evaluate_xai_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    evaluate_xai_outputs(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
