"""Download external dataset and pretrained VGGish assets.

This script intentionally stores large external files under data/raw/ and
models/pretrained/, not in source-controlled code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ensure_project_dirs, load_config  # noqa: E402
from src.data_download import download_watkins_archive  # noqa: E402
from src.weights import download_vggish_weights  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--asset", choices=["all", "dataset", "weights"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap for dataset audio files, useful for quick tests.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    if args.asset in {"all", "dataset"}:
        download_watkins_archive(cfg, overwrite=args.overwrite, max_files=args.max_files)
    if args.asset in {"all", "weights"}:
        download_vggish_weights(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
