# Explaining Marine Mammal Sound Classification Using Pretrained VGGish and Time-Frequency XAI

This repository contains a reproducible PyTorch research pipeline for classifying marine mammal sounds from a balanced subset of Watkins Marine Mammal Sound Database recordings and explaining predictions in time-frequency space.

The pipeline uses VGGish-compatible 16 kHz mono log-Mel patches as the model input. The default training mode is `linear_probe`: the VGGish backbone is frozen and only a small classifier head is trained. The controlled experiment uses three balanced classes: Humpback Whale, Sperm Whale, and Spinner Dolphin.

## Research Question

Do predictions from a pretrained VGGish-based marine mammal classifier align with visible, high-energy acoustic events such as calls, clicks, whistles, and harmonics, or do explanations suggest shortcut behavior such as reliance on silence, background noise, clip borders, or recording artifacts?

The XAI evaluation is intentionally cautious. It reports alignment with visible/high-energy acoustic regions and faithfulness under deletion, not biological certainty.

## Installation

```bash
pip install -r requirements.txt
```

Place PyTorch VGGish weights at:

```text
models/pretrained/vggish_pytorch.pt
```

Accepted alternatives are `models/pretrained/vggish.pth` or `models/pretrained/vggish_state_dict.pt`. The checkpoint should be a VGGish backbone state dict compatible with `src/model.py`.

## Dataset Instructions

The primary dataset is the Watkins Marine Mammal Sound Database. Automated scraping can be brittle because public data portals change over time, so the robust workflow is:

1. Download Watkins audio files manually.
2. Organize them so species names are folder names when possible.
3. Set `data.manual_data_dir` in `config.yaml`.
4. Run the collection script, which recursively parses `.wav`, `.mp3`, `.flac`, `.aiff`, and `.aif` files.

The metadata CSV includes file path, normalized species label, original filename, duration, sample rate, channels, and source.

## Full Pipeline

Download external assets:

```bash
python scripts/00_download_assets.py --config config.yaml
```

Run the full workflow:

```bash
python run_pipeline.py --config config.yaml
```

Debug smoke test:

```bash
python run_pipeline.py --config config.yaml --debug
```

Run a range of steps:

```bash
python run_pipeline.py --config config.yaml --start_step train --end_step xai
```

## Individual Scripts

```bash
python scripts/01_collect_watkins_data.py --config config.yaml
python scripts/02_prepare_balanced_subset.py --config config.yaml
python scripts/03_preprocess_audio.py --config config.yaml
python scripts/04_split_dataset.py --config config.yaml
python scripts/05_train_vggish_classifier.py --config config.yaml
python scripts/06_evaluate_test.py --config config.yaml
python scripts/07_run_xai.py --config config.yaml
python scripts/08_evaluate_xai.py --config config.yaml
python scripts/09_generate_report_assets.py --config config.yaml
```

Add `--overwrite` to recompute existing outputs.

## Outputs

Key outputs are saved under:

- `data/metadata/selected_balanced_metadata.csv`
- `data/metadata/preprocessed_metadata.csv`
- `data/processed/audio_clips/`
- `data/processed/logmel/`
- `data/processed/splits/train.csv`, `val.csv`, `test.csv`
- `models/checkpoints/best_model.pt`
- `outputs/logs/training_log.csv`
- `outputs/metrics/test_metrics.json`
- `outputs/confusion_matrices/`
- `outputs/xai/{gradcam,integrated_gradients,occlusion,lime,saliency,smoothgrad,input_x_gradient,guided_backprop}/`
- `outputs/xai_evaluation/`
- `outputs/report_assets/`

Large raw data, generated outputs, and checkpoints are intentionally not included in version control.

## Model

The model is a PyTorch VGGish-style architecture:

1. 16 kHz mono waveform preprocessing
2. 64-bin log-Mel extraction
3. 0.96 s VGGish patches with 96 frames
4. Frozen pretrained VGGish convolutional/embedding backbone
5. Trainable MLP classifier head

The default `linear_probe` mode freezes all VGGish parameters. `fine_tune_last` can unfreeze final convolutional layers if the checkpoint and experiment design support it.

## XAI Methods

The pipeline applies eight explanation methods to the same VGGish log-Mel input representation:

- Grad-CAM: activation-gradient attribution from the final convolutional map.
- Integrated Gradients: Captum-based attribution from a zero baseline.
- Time-Frequency Occlusion: rectangular patch masking and predicted-class confidence drop.
- Spectrogram LIME: local surrogate over rectangular time-frequency segments.
- Saliency: absolute input gradients of the predicted-class score.
- SmoothGrad: noise-averaged saliency maps.
- Input x Gradient: elementwise input-weighted gradients.
- Guided Backpropagation: positive-gradient ReLU backward visualization.

Each method saves raw arrays, heatmaps, overlays, and metadata per selected example.

## XAI Evaluation

The evaluation scripts compute:

- Top-k deletion faithfulness at 5%, 10%, 20%, 30%, and 40%.
- Random deletion baseline with mean and standard deviation.
- Pairwise top-20% explanation agreement IoU.
- Stability proxies under Gaussian noise, time shift, amplitude scaling, and frequency masking.
- Energy-alignment and artifact audit metrics for high-energy, low-energy, and clip-border overlap.

These metrics support statements such as "the model appears to rely on high-energy vocalization regions" only when the data supports that pattern.

## Debug Mode

Debug mode is for CPU-safe pipeline smoke tests. It limits training to one epoch and tiny class subsets. It does not produce final experimental results and may run without pretrained weights only to test code paths.

```bash
python run_pipeline.py --config config.yaml --debug
```
