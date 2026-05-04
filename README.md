# Marine VGGish XAI

Explainable marine mammal sound classification with a pretrained VGGish-style audio model and time-frequency attribution methods.

This project studies whether a deep audio classifier for marine mammal vocalizations uses meaningful acoustic evidence in spectrogram space, such as calls, clicks, whistles, harmonic structure, and high-energy vocal regions, or whether predictions may depend on less reliable shortcut cues such as silence, background noise, clip boundaries, or recording artifacts.

## Project Summary

The repository implements a reproducible PyTorch pipeline for a controlled three-class marine mammal acoustic classification problem:

- Humpback Whale
- Sperm Whale
- Spinner Dolphin

Audio is converted into VGGish-compatible log-Mel patches and classified with a VGGish-style convolutional backbone plus a trainable classifier head. Predictions are then explained with multiple XAI methods and evaluated using faithfulness, agreement, stability, and energy/artifact alignment metrics.

The project is designed around model interpretability, not only predictive accuracy. Classification metrics show whether the model can separate the classes, while XAI analysis investigates what spectrogram evidence the model appears to use.

## Main Results

The final grouped held-out test evaluation produced:

| Metric | Value |
|---|---:|
| Test accuracy | 0.8273 |
| Macro-F1 | 0.8205 |
| Weighted-F1 | 0.8228 |
| Top-2 accuracy | 0.9712 |

Per-class behavior is uneven:

- Humpback Whale has the strongest F1 score.
- Spinner Dolphin has high recall but lower precision.
- Sperm Whale has high precision but weaker recall, making it the main class-specific weakness.

The XAI evaluation found:

- Input x Gradient produced the strongest high-energy spectrogram overlap.
- Integrated Gradients and Input x Gradient had the strongest pairwise explanation agreement.
- LIME produced the largest deletion-based confidence drop.
- SmoothGrad and Guided Backpropagation added useful gradient-family comparisons.
- Low-energy overlap and competitive random deletion baselines show that explanations should be interpreted cautiously.

The results support a careful conclusion: the classifier often appears to use visible acoustic evidence, but the explanations are not proof of biological reasoning without expert call-level annotation.

## Repository Structure

```text
marine_vggish_xai/
├── config.yaml
├── requirements.txt
├── run_pipeline.py
├── README.md
├── notebooks/
│   └── quick_visual_check.ipynb
├── scripts/
│   ├── 00_download_assets.py
│   ├── 01_collect_watkins_data.py
│   ├── 02_prepare_balanced_subset.py
│   ├── 03_preprocess_audio.py
│   ├── 04_split_dataset.py
│   ├── 05_train_vggish_classifier.py
│   ├── 06_evaluate_test.py
│   ├── 07_run_xai.py
│   ├── 08_evaluate_xai.py
│   ├── 09_generate_report_assets.py
│   └── generate_docx_report.py
├── src/
│   ├── config.py
│   ├── data_download.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── visualization.py
│   ├── weights.py
│   ├── xai_gradcam.py
│   ├── xai_gradient_methods.py
│   ├── xai_integrated_gradients.py
│   ├── xai_lime.py
│   ├── xai_metrics.py
│   └── xai_occlusion.py
└── outputs/
    ├── logs/
    ├── metrics/
    ├── plots/
    ├── confusion_matrices/
    ├── test_predictions/
    ├── xai/
    ├── xai_evaluation/
    └── report_assets/
```

Raw audio data and model checkpoints are not committed. They should be downloaded or generated locally.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The project expects a VGGish-compatible PyTorch checkpoint at:

```text
models/pretrained/vggish_pytorch.pt
```

Accepted alternatives are:

```text
models/pretrained/vggish.pth
models/pretrained/vggish_state_dict.pt
```

The model loader also supports the configured torchaudio-style pretrained source when available.

## Dataset

The intended dataset source is the Watkins Marine Mammal Sound Database or a local mirror of those recordings. The pipeline supports two practical workflows:

1. Use the configured dataset source in `config.yaml`.
2. Place manually downloaded recordings in a local directory and set `data.manual_data_dir`.

Supported audio extensions:

```text
.wav, .mp3, .flac, .aiff, .aif
```

The preprocessing code records metadata including file path, species label, source recording identifier, duration, sample rate, channels, and source.

## Preprocessing

The audio preprocessing stage performs:

- mono conversion
- resampling to 16 kHz
- amplitude normalization
- deterministic five-second clip construction
- energy-based crop/pad behavior
- 64-bin VGGish log-Mel feature extraction
- visual spectrogram generation for explanation overlays

The split stage uses grouped train/validation/test splitting by source recording. This reduces leakage caused by placing highly similar segments from the same original recording into different splits.

## Model

The classifier uses a VGGish-style audio CNN:

1. Input waveform is converted to log-Mel features.
2. Log-Mel features are grouped into VGGish-compatible patches.
3. A pretrained convolutional VGGish-style backbone extracts audio embeddings.
4. A classifier head maps embeddings to the three marine mammal classes.
5. Softmax probabilities are used for class prediction and XAI target selection.

The current configuration uses `fine_tune_last`, which keeps most pretrained representation structure while allowing final layers to adapt to the marine mammal task.

## XAI Methods

The pipeline applies eight explanation methods to the same trained model and input representation:

| Method | Explanation Type | Purpose |
|---|---|---|
| Grad-CAM | convolutional activation map | localizes class-discriminative CNN regions |
| Integrated Gradients | path attribution | attributes prediction from a baseline to the input |
| Occlusion Sensitivity | perturbation attribution | measures confidence drop after masking regions |
| LIME | local surrogate | estimates region importance with interpretable masks |
| Saliency | input gradient | shows local sensitivity of class score |
| SmoothGrad | averaged saliency | reduces noisy gradient artifacts |
| Input x Gradient | gradient-weighted input | combines sensitivity with input magnitude |
| Guided Backpropagation | modified backpropagation | gives sharper positive-gradient visualizations |

Using multiple methods is important because visually convincing explanation maps can still be unreliable. The project therefore compares explanation methods rather than relying on a single heatmap.

## XAI Evaluation

The evaluation scripts compute:

- top-k deletion faithfulness at 5%, 10%, 20%, 30%, and 40%
- random deletion baselines
- pairwise top-20% explanation agreement IoU
- stability under Gaussian noise, time shift, amplitude scaling, and frequency masking
- overlap with high-energy regions, low-energy regions, and clip-border regions

These metrics help distinguish useful attribution patterns from explanations that are only visually appealing.

## Running the Pipeline

Run the full pipeline:

```bash
python run_pipeline.py --config config.yaml
```

Run a debug smoke test:

```bash
python run_pipeline.py --config config.yaml --debug
```

Run a specific stage range:

```bash
python run_pipeline.py --config config.yaml --start_step train --end_step xai
```

Recompute existing artifacts:

```bash
python run_pipeline.py --config config.yaml --overwrite
```

## Individual Stages

```bash
python scripts/00_download_assets.py --config config.yaml
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

## Output Artifacts

The committed `outputs/` folder contains generated experiment artifacts such as:

- training logs
- test metrics
- classification reports
- confusion matrices
- per-class metric plots
- selected XAI examples
- attribution overlays
- deletion faithfulness results
- explanation agreement results
- perturbation stability results
- energy/artifact audit results

The most important result files are:

```text
outputs/metrics/test_metrics.json
outputs/logs/training_log.csv
outputs/metrics/classification_report.txt
outputs/xai_evaluation/faithfulness_topk_deletion.csv
outputs/xai_evaluation/pairwise_agreement.csv
outputs/xai_evaluation/stability_summary.csv
outputs/xai_evaluation/energy_alignment.csv
```

## Interpretation Notes

This project uses careful language when interpreting explanations:

- High-energy overlap suggests possible acoustic-event alignment, not biological certainty.
- Deletion faithfulness is useful but depends on the masking strategy.
- Perturbation-based methods are behaviorally meaningful but can create out-of-distribution inputs.
- Gradient-based methods are detailed and efficient but can be sensitive to small input changes.
- Expert-labeled call regions would be needed for stronger biological validation.

## Limitations

- The experiment is a controlled three-class task, not a general marine mammal recognizer.
- Raw recordings and pretrained weights are not included in the repository.
- XAI metrics are proxy evaluations and should not be treated as causal proof.
- The dataset may contain recording-source or background-condition biases.
- External validation across recording devices and environments remains future work.

## Future Work

Useful extensions include:

- larger and more diverse species coverage
- external test sets from different recording sources
- expert-labeled call, click, whistle, and harmonic regions
- saliency sanity checks with randomized model weights and labels
- counterfactual audio perturbations
- underwater-acoustics-specific model architectures
- stronger faithfulness metrics such as insertion curves

## License and Data Notice

This repository contains code and generated experiment artifacts. Raw Watkins audio recordings should be obtained from their original source or an authorized mirror and used according to the dataset's terms.
