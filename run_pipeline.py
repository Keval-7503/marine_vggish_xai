"""Run the full marine VGGish XAI pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import ensure_project_dirs, load_config


STEPS = [
    ("assets", "scripts/00_download_assets.py"),
    ("collect", "scripts/01_collect_watkins_data.py"),
    ("balance", "scripts/02_prepare_balanced_subset.py"),
    ("preprocess", "scripts/03_preprocess_audio.py"),
    ("split", "scripts/04_split_dataset.py"),
    ("train", "scripts/05_train_vggish_classifier.py"),
    ("evaluate", "scripts/06_evaluate_test.py"),
    ("xai", "scripts/07_run_xai.py"),
    ("xai_eval", "scripts/08_evaluate_xai.py"),
    ("report", "scripts/09_generate_report_assets.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start_step", choices=[s[0] for s in STEPS], default="collect")
    parser.add_argument("--end_step", choices=[s[0] for s in STEPS], default="report")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    start = [s[0] for s in STEPS].index(args.start_step)
    end = [s[0] for s in STEPS].index(args.end_step)
    if start > end:
        raise SystemExit("--start_step must come before --end_step")
    root = Path(__file__).resolve().parent
    for name, script in STEPS[start : end + 1]:
        cmd = [sys.executable, str(root / script), "--config", str(Path(args.config).resolve())]
        if args.debug and name == "train":
            cmd.append("--debug")
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"\n=== Running step: {name} ===")
        subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
