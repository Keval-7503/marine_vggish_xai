# Final Results Summary

## Dataset

- Spinner Dolphin: 114 source clips
- Sperm Whale: 75 source clips
- Humpback Whale: 64 source clips

## Best Validation Metrics

- Epoch: 24
- Validation macro-F1: 0.8124
- Validation accuracy: 0.8120

## Test Metrics

- Accuracy: 0.8273
- Macro-F1: 0.8205
- Weighted-F1: 0.8228

## XAI Findings

- Best deletion faithfulness by method:
  - lime: top-40% deletion mean confidence drop = 0.3002
  - guided_backprop: top-30% deletion mean confidence drop = 0.2662
  - smoothgrad: top-10% deletion mean confidence drop = 0.2654
  - saliency: top-40% deletion mean confidence drop = 0.2476
  - integrated_gradients: top-10% deletion mean confidence drop = 0.2242
  - input_x_gradient: top-10% deletion mean confidence drop = 0.2140
  - occlusion: top-10% deletion mean confidence drop = 0.1925
  - gradcam: top-20% deletion mean confidence drop = 0.0801

- Highest high-energy alignment: input_x_gradient (0.7035).
- Mean energy/artifact overlaps:
  - input_x_gradient: high=0.7035, low=0.4216, border=0.1056
  - integrated_gradients: high=0.6874, low=0.4256, border=0.1050
  - guided_backprop: high=0.6581, low=0.3241, border=0.1309
  - saliency: high=0.6573, low=0.2730, border=0.1507
  - smoothgrad: high=0.6484, low=0.2766, border=0.1427
  - gradcam: high=0.6221, low=0.3381, border=0.1160
  - occlusion: high=0.6098, low=0.5717, border=0.1012
  - lime: high=0.5955, low=0.5720, border=0.0720

- Strongest explanation agreement: integrated_gradients vs. input_x_gradient (top-20% IoU = 0.6649).

## Limitations

The pipeline evaluates alignment with visible acoustic energy, not biological intent or causal vocalization semantics. Manual expert review remains necessary.