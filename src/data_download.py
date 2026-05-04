"""Watkins metadata collection and balanced subset preparation."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import librosa
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import path_from_config, project_root, resolve_path
from src.utils import safe_stem, set_seed, setup_logging, should_skip
from src.visualization import save_bar

LOGGER = setup_logging(__name__)


def normalize_species_name(name: str | None) -> str:
    if not name:
        return "unknown"
    name = Path(str(name)).stem
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    bad = {"unknown", "misc", "unidentified", "noise", "test", ""}
    return "unknown" if name in bad else name.title()


def collect_watkins_data(cfg: dict, overwrite: bool = False) -> Path:
    """Create metadata from a manual folder first, then try a lightweight web scrape."""
    meta_dir = path_from_config(cfg, "paths", "metadata_dir")
    out_csv = meta_dir / "watkins_metadata.csv"
    if should_skip(out_csv, overwrite):
        LOGGER.info("Metadata exists: %s", out_csv)
        return out_csv
    manual = resolve_path(cfg, cfg["data"].get("manual_data_dir"))
    if manual and manual.exists():
        LOGGER.info("Parsing manual Watkins folder: %s", manual)
        df = parse_audio_folder(cfg, manual)
    else:
        LOGGER.info("No manual_data_dir found; attempting conservative Watkins scrape.")
        df = try_collect_from_watkins(cfg)
    if df.empty:
        raise RuntimeError(
            "No audio files were found. Set data.manual_data_dir in config.yaml to a manually downloaded Watkins folder."
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    LOGGER.info("Saved metadata with %d rows to %s", len(df), out_csv)
    return out_csv


def download_watkins_archive(cfg: dict, overwrite: bool = False, max_files: int | None = None) -> Path:
    """Download Watkins audio files from archive.org item metadata.

    The Internet Archive item is used because it exposes stable file metadata.
    Files are saved under data/raw/watkins_202104/ and then parsed into the
    standard Watkins metadata CSV.
    """
    source = cfg["data"].get("watkins_download_source", "huggingface_parquet")
    if source == "huggingface_parquet":
        return download_watkins_huggingface_parquet(cfg, overwrite=overwrite, max_files=max_files)
    identifier = cfg["data"].get("watkins_archive_identifier", "watkins_202104")
    raw_root = path_from_config(cfg, "paths", "raw_data_dir") / identifier
    raw_root.mkdir(parents=True, exist_ok=True)
    meta_url = f"https://archive.org/metadata/{identifier}"
    LOGGER.info("Fetching archive.org metadata: %s", meta_url)
    metadata = requests.get(meta_url, timeout=30)
    metadata.raise_for_status()
    files = metadata.json().get("files", [])
    exts = {e.lower() for e in cfg["data"]["allowed_audio_extensions"]}
    candidates = [f for f in files if Path(f.get("name", "")).suffix.lower() in exts or f.get("name", "").lower().endswith(".zip")]
    if max_files is not None:
        candidates = candidates[:max_files]
    if not candidates:
        raise RuntimeError(f"No audio or zip files found in archive.org item {identifier}.")
    base = f"https://archive.org/download/{identifier}/"
    for f in tqdm(candidates, desc="Downloading Watkins archive files"):
        name = f["name"]
        target = raw_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            continue
        url = urljoin(base, name)
        _download_file(url, target)
        if target.suffix.lower() == ".zip":
            extract_dir = target.with_suffix("")
            if overwrite or not extract_dir.exists():
                try:
                    with zipfile.ZipFile(target) as zf:
                        zf.extractall(extract_dir)
                except zipfile.BadZipFile:
                    LOGGER.warning("Downloaded file is not a readable zip: %s", target)
    # Parse extracted and direct audio into standard metadata.
    cfg["data"]["manual_data_dir"] = str(raw_root)
    return collect_watkins_data(cfg, overwrite=True)


def download_watkins_huggingface_parquet(cfg: dict, overwrite: bool = False, max_files: int | None = None) -> Path:
    """Download Hugging Face parquet mirror and extract embedded audio files.

    This is a fallback for periods when WHOI pages are unavailable from the
    current network. The dataset card credits the Watkins Marine Mammal Sound
    Database and points back to the original archive.
    """
    repo = cfg["data"].get("huggingface_dataset_repo", "confit/wmms-parquet")
    raw_root = path_from_config(cfg, "paths", "raw_data_dir") / "wmms_huggingface"
    parquet_dir = raw_root / "parquet"
    audio_dir = raw_root / "audio"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://huggingface.co/api/datasets/{repo}/parquet"
    LOGGER.info("Fetching Hugging Face parquet metadata: %s", api_url)
    r = requests.get(api_url, timeout=30)
    r.raise_for_status()
    data = r.json()
    urls: list[str] = []
    for split_urls in data.get("default", {}).values():
        urls.extend(split_urls)
    if not urls:
        raise RuntimeError(f"No parquet URLs returned by Hugging Face for {repo}.")
    if max_files is not None:
        urls = urls[:max_files]
    rows = []
    file_counter = 0
    for url in urls:
        parquet_path = parquet_dir / f"{Path(url).parent.name}_{Path(url).name}"
        if overwrite or not parquet_path.exists():
            _download_file(url, parquet_path)
        df = pd.read_parquet(parquet_path)
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {parquet_path.name}"):
            species = normalize_species_name(row.get("species"))
            audio = row.get("audio")
            try:
                audio_bytes, ext = _audio_bytes_from_hf_cell(audio)
                out_dir = audio_dir / safe_stem(species)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{parquet_path.stem}_{idx:06d}{ext}"
                if overwrite or not out_path.exists():
                    out_path.write_bytes(audio_bytes)
                rows.append(
                    {
                        "file_path": str(out_path.resolve()),
                        "species": species,
                        "original_filename": out_path.name,
                        "duration_sec": None,
                        "sample_rate": None,
                        "channels": None,
                        "source": f"huggingface:{repo}",
                    }
                )
                file_counter += 1
            except Exception as exc:
                LOGGER.warning("Skipping parquet row %s:%s: %s", parquet_path.name, idx, exc)
    if not rows:
        raise RuntimeError("Downloaded parquet files, but no audio bytes could be extracted.")
    # Fill audio metadata using the same robust parser used for manual folders.
    cfg["data"]["manual_data_dir"] = str(audio_dir)
    parsed = parse_audio_folder(cfg, audio_dir)
    out_csv = path_from_config(cfg, "paths", "metadata_dir") / "watkins_metadata.csv"
    parsed.to_csv(out_csv, index=False)
    LOGGER.info("Extracted %d Hugging Face Watkins audio files to %s", len(parsed), audio_dir)
    return out_csv


def _audio_bytes_from_hf_cell(audio) -> tuple[bytes, str]:
    """Return bytes and extension from a Hugging Face Audio parquet cell."""
    if isinstance(audio, dict):
        if audio.get("bytes") is not None:
            path = str(audio.get("path") or "audio.wav")
            return audio["bytes"], Path(path).suffix or ".wav"
        if audio.get("array") is not None and audio.get("sampling_rate") is not None:
            import soundfile as sf

            buf = BytesIO()
            sf.write(buf, audio["array"], int(audio["sampling_rate"]), format="WAV")
            return buf.getvalue(), ".wav"
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio), ".wav"
    raise ValueError(f"Unsupported audio cell type: {type(audio)!r}")


def _download_file(url: str, target: Path) -> None:
    LOGGER.info("Downloading %s", url)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with target.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=target.name, leave=False) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))


def parse_audio_folder(cfg: dict, folder: Path) -> pd.DataFrame:
    exts = {e.lower() for e in cfg["data"]["allowed_audio_extensions"]}
    rows = []
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]
    for p in tqdm(files, desc="Inspecting audio"):
        try:
            duration = librosa.get_duration(path=str(p))
            info = librosa.get_samplerate(str(p))
            # Species is commonly encoded by parent folder in manual exports.
            species = normalize_species_name(p.parent.name)
            if species == "Unknown":
                species = normalize_species_name(p.stem.split("_")[0])
            rows.append(
                {
                    "file_path": str(p.resolve()),
                    "species": species,
                    "original_filename": p.name,
                    "duration_sec": float(duration),
                    "sample_rate": int(info),
                    "channels": None,
                    "source": "manual_watkins_folder",
                }
            )
        except Exception as exc:
            LOGGER.warning("Skipping unreadable file %s: %s", p, exc)
    return pd.DataFrame(rows)


def try_collect_from_watkins(cfg: dict) -> pd.DataFrame:
    """Best-effort scraper; site layout may change, so manual folder is the robust path."""
    raw_dir = path_from_config(cfg, "paths", "raw_data_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    base = cfg["data"].get("watkins_index_url")
    exts = tuple(cfg["data"]["allowed_audio_extensions"])
    try:
        html = requests.get(base, timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(exts):
                links.append(urljoin(base, href))
        rows = []
        for url in tqdm(links, desc="Downloading Watkins audio"):
            filename = Path(url).name
            local = raw_dir / filename
            if not local.exists():
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                local.write_bytes(r.content)
            species = normalize_species_name(Path(filename).stem.split("_")[0])
            rows.append({"file_path": str(local.resolve()), "species": species, "original_filename": filename, "duration_sec": librosa.get_duration(path=str(local)), "sample_rate": librosa.get_samplerate(str(local)), "channels": None, "source": url})
        return pd.DataFrame(rows)
    except Exception as exc:
        LOGGER.warning("Automated Watkins collection failed: %s", exc)
        return pd.DataFrame()


def prepare_balanced_subset(cfg: dict, overwrite: bool = False) -> Path:
    set_seed(int(cfg["project"]["seed"]))
    root = project_root(cfg)
    meta_csv = root / "data/metadata/watkins_metadata.csv"
    out_csv = root / "data/metadata/selected_balanced_metadata.csv"
    if should_skip(out_csv, overwrite):
        LOGGER.info("Balanced metadata exists: %s", out_csv)
        return out_csv
    if not meta_csv.exists():
        raise FileNotFoundError("Run 01_collect_watkins_data.py before preparing a balanced subset.")
    df = pd.read_csv(meta_csv)
    df["species"] = df["species"].map(normalize_species_name)
    df = df[~df["species"].str.lower().isin(["unknown", "ambiguous", "unidentified"])]
    counts = df["species"].value_counts()
    metrics_dir = root / "outputs/metrics"
    plots_dir = root / "outputs/plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    counts.rename_axis("species").reset_index(name="count").to_csv(metrics_dir / "species_counts.csv", index=False)
    save_bar(counts, plots_dir / "species_distribution_before.png", "Species distribution before balancing")
    selected_species = cfg["data"].get("selected_species") or []
    strategy = cfg["data"].get("subset_strategy", "balanced_top_counts")
    if selected_species:
        normalized_selected = [normalize_species_name(s) for s in selected_species]
        missing = [s for s in normalized_selected if counts.get(s, 0) < int(cfg["data"]["min_samples_per_species"])]
        if missing:
            raise RuntimeError(f"Configured selected_species below min_samples_per_species: {missing}. See outputs/metrics/species_counts.csv.")
        eligible = counts.loc[normalized_selected]
    else:
        eligible = counts[counts >= int(cfg["data"]["min_samples_per_species"])].head(int(cfg["data"]["num_species"]))
        if eligible.empty or len(eligible) < int(cfg["data"]["num_species"]):
            raise RuntimeError(f"Not enough species with at least {cfg['data']['min_samples_per_species']} samples. See outputs/metrics/species_counts.csv.")
    max_per_species = int(cfg["data"]["max_samples_per_species"])
    if strategy == "selected_species_max_available":
        per_species_by_label = {species: min(max_per_species, int(count)) for species, count in eligible.items()}
    else:
        per_species = min(max_per_species, int(eligible.min()))
        per_species_by_label = {species: per_species for species in eligible.index}
    selected_rows = []
    for species in eligible.index:
        selected_rows.append(df[df["species"] == species].sample(per_species_by_label[species], random_state=int(cfg["project"]["seed"])))
    balanced = pd.concat(selected_rows).sample(frac=1.0, random_state=int(cfg["project"]["seed"])).reset_index(drop=True)
    labels = {species: idx for idx, species in enumerate(sorted(balanced["species"].unique()))}
    balanced["label_id"] = balanced["species"].map(labels)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out_csv, index=False)
    (root / "data/metadata/label_map.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    save_bar(balanced["species"].value_counts(), plots_dir / "species_distribution_after.png", "Balanced selected species distribution")
    LOGGER.info("Saved balanced subset: %s (%d rows)", out_csv, len(balanced))
    return out_csv
