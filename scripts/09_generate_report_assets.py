"""Generate report-ready figures and a cautious summary markdown."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import ensure_project_dirs, load_config  # noqa: E402
from src.visualization import generate_report_assets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    generate_report_assets(cfg, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
