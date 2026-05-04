"""Audio preprocessing and VGGish-compatible log-Mel extraction."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly, stft
from math import gcd
from tqdm import tqdm

from src.config import project_root
from src.utils import safe_stem, setup_logging, should_skip
from src.visualization import plot_spectrogram

LOGGER = setup_logging(__name__)


def load_audio_fixed(path: Path, sample_rate: int, clip_duration: float, normalize: bool = True, crop: str = "center") -> np.ndarray:
    y = read_audio_mono(path, sample_rate)
    target = int(sample_rate * clip_duration)
    if len(y) >= target:
        if crop == "energy":
            start = _energy_crop_start(y, target)
        else:
            start = 0 if crop == "start" else max(0, (len(y) - target) // 2)
        y = y[start : start + target]
    else:
        y = np.pad(y, (0, target - len(y)))
    if normalize:
        peak = np.max(np.abs(y))
        if peak > 1e-8:
            y = y / peak
    return y.astype(np.float32)


def load_audio_segment(path: Path, sample_rate: int, clip_duration: float, start: int, normalize: bool = True) -> np.ndarray:
    y = read_audio_mono(path, sample_rate)
    target = int(sample_rate * clip_duration)
    if len(y) <= target:
        y = np.pad(y, (0, target - len(y)))
    else:
        start = int(np.clip(start, 0, len(y) - target))
        y = y[start : start + target]
    if normalize:
        peak = np.max(np.abs(y))
        if peak > 1e-8:
            y = y / peak
    return y.astype(np.float32)


def read_audio_mono(path: Path, sample_rate: int) -> np.ndarray:
    """Read audio via soundfile and resample without librosa.load."""
    y, sr = sf.read(path, always_2d=True)
    y = y.astype(np.float32).mean(axis=1)
    if sr != sample_rate:
        factor = gcd(int(sr), int(sample_rate))
        up = int(sample_rate) // factor
        down = int(sr) // factor
        y = resample_poly(y, up, down).astype(np.float32)
    return y


def _energy_crop_start(y: np.ndarray, target: int) -> int:
    """Choose the fixed-length window with highest short-time energy."""
    if len(y) <= target:
        return 0
    hop = max(1, target // 20)
    starts = np.arange(0, len(y) - target + 1, hop)
    if starts[-1] != len(y) - target:
        starts = np.append(starts, len(y) - target)
    energies = [float(np.mean(y[s : s + target] ** 2)) for s in starts]
    return int(starts[int(np.argmax(energies))])


def _segment_starts(path: Path, cfg: dict, n_segments: int) -> list[int]:
    sample_rate = int(cfg["audio"]["sample_rate"])
    target = int(sample_rate * float(cfg["audio"]["clip_duration_sec"]))
    y = read_audio_mono(path, sample_rate)
    if len(y) <= target:
        return [0] * n_segments
    hop = max(1, target // 4)
    starts = np.arange(0, len(y) - target + 1, hop)
    if starts[-1] != len(y) - target:
        starts = np.append(starts, len(y) - target)
    energies = np.array([float(np.mean(y[s : s + target] ** 2)) for s in starts])
    order = np.argsort(-energies)
    chosen: list[int] = []
    min_gap = max(1, target // 3)
    for idx in order:
        s = int(starts[idx])
        if all(abs(s - c) >= min_gap for c in chosen):
            chosen.append(s)
        if len(chosen) == n_segments:
            break
    while len(chosen) < n_segments:
        chosen.append(int(starts[order[len(chosen) % len(order)]]))
    return sorted(chosen)


def _segment_plan(df: pd.DataFrame, cfg: dict) -> dict[int, int]:
    target_total = int(cfg["audio"].get("target_total_segments", 0) or 0)
    max_segments = int(cfg["audio"].get("max_segments_per_source", 1))
    if target_total <= len(df):
        return {int(i): 1 for i in df.index}
    classes = sorted(df["species"].unique())
    per_class_target = max(1, target_total // len(classes))
    plan: dict[int, int] = {}
    for species, group in df.groupby("species", sort=True):
        indices = list(group.index)
        desired = per_class_target
        base = max(1, desired // len(indices))
        remainder = max(0, desired - base * len(indices))
        for pos, idx in enumerate(indices):
            count = base + (1 if pos < remainder else 0)
            plan[int(idx)] = min(max_segments, count)
    return plan


def logmel(y: np.ndarray, cfg: dict, n_mels: int) -> np.ndarray:
    audio = cfg["audio"]
    sr = int(audio["sample_rate"])
    n_fft = int(audio["n_fft"])
    hop = int(audio["hop_length"])
    win = int(audio["win_length"])
    _, _, zxx = stft(
        y,
        fs=sr,
        window="hann",
        nperseg=win,
        noverlap=win - hop,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    mel_fb = mel_filterbank(sr, n_fft, int(n_mels), 0.0, sr / 2.0)
    mel = np.maximum(mel_fb @ power, 1e-10)
    ref = np.maximum(float(mel.max()), 1e-10)
    return (10.0 * np.log10(mel / ref)).astype(np.float32)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray | float) -> np.ndarray | float:
        return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for k in range(left, min(center, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(center, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(1, right - center)
    return fb


def vggish_examples_from_logmel(mel64: np.ndarray, frames: int = 96) -> np.ndarray:
    """Split 64-bin log-Mel into VGGish examples of 96 frames, padding as needed."""
    if mel64.shape[1] < frames:
        mel64 = np.pad(mel64, ((0, 0), (0, frames - mel64.shape[1])), mode="constant", constant_values=mel64.min())
    chunks = []
    for start in range(0, mel64.shape[1] - frames + 1, frames):
        chunks.append(mel64[:, start : start + frames])
    if not chunks:
        chunks = [mel64[:, :frames]]
    arr = np.stack(chunks, axis=0)  # patches, mel, time
    # Normalize to a stable range for PyTorch VGGish-style input.
    arr = (arr + 80.0) / 80.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def preprocess_dataset(cfg: dict, overwrite: bool = False) -> Path:
    root = project_root(cfg)
    in_csv = root / "data/metadata/selected_balanced_metadata.csv"
    out_csv = root / "data/metadata/preprocessed_metadata.csv"
    if should_skip(out_csv, overwrite):
        LOGGER.info("Preprocessed metadata exists: %s", out_csv)
        return out_csv
    if not in_csv.exists():
        raise FileNotFoundError("Run 02_prepare_balanced_subset.py first.")
    df = pd.read_csv(in_csv)
    clip_dir = root / "data/processed/audio_clips"
    mel_dir = root / "data/processed/logmel"
    png_dir = mel_dir / "visual_png"
    for d in [clip_dir, mel_dir, png_dir]:
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    segment_plan = _segment_plan(df, cfg)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        src = Path(row["file_path"])
        starts = _segment_starts(src, cfg, segment_plan.get(int(idx), 1))
        for segment_idx, start in enumerate(starts):
            sample_id = f"{idx:06d}_seg{segment_idx:02d}_{safe_stem(row['species'])}_{safe_stem(src.stem)}"
            wav_path = clip_dir / f"{sample_id}.wav"
            vgg_path = mel_dir / f"{sample_id}_vggish.npy"
            visual_path = mel_dir / f"{sample_id}_visual.npy"
            png_path = png_dir / f"{sample_id}.png"
            if overwrite or not (wav_path.exists() and vgg_path.exists() and visual_path.exists()):
                y = load_audio_segment(
                    src,
                    int(cfg["audio"]["sample_rate"]),
                    float(cfg["audio"]["clip_duration_sec"]),
                    start=start,
                    normalize=bool(cfg["audio"]["normalize"]),
                )
                sf.write(wav_path, y, int(cfg["audio"]["sample_rate"]))
                mel64 = logmel(y, cfg, int(cfg["audio"]["n_mels_vggish"]))
                visual = logmel(y, cfg, int(cfg["audio"]["n_mels_visual"]))
                np.save(vgg_path, vggish_examples_from_logmel(mel64))
                np.save(visual_path, visual)
                if bool(cfg["audio"].get("save_preprocessing_png", True)):
                    plot_spectrogram(visual, png_path, f"{row['species']} log-Mel spectrogram")
            rows.append(
                {
                    "sample_id": sample_id,
                    "processed_path": str(wav_path),
                    "vggish_path": str(vgg_path),
                    "logmel_path": str(visual_path),
                    "spectrogram_png": str(png_path),
                    "species": row["species"],
                    "label_id": int(row["label_id"]),
                    "duration_sec": float(cfg["audio"]["clip_duration_sec"]),
                    "segment_index": int(segment_idx),
                    "segment_start_sample": int(start),
                    "split": "",
                    "source_file": str(src),
                }
            )
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    LOGGER.info("Saved preprocessing metadata: %s", out_csv)
    return out_csv
