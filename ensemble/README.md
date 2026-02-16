# Ensemble Models

This directory contains scripts and tools for working with ensemble models in the fusion training project. The ensemble combines specialized models trained on individual modalities (vehicle, sign, human) into a unified prediction system.

## Overview

The ensemble approach addresses class imbalance in semantic segmentation by training dedicated models for different object categories. These specialized models are then merged into a single checkpoint that maintains separate branches for each modality while allowing unified inference.

## Key Components

### Core Files
- `zod_ensemble_predictor.py`: Main predictor class for loading and running the merged ensemble model
- `ensemble_merge.py`: Script to merge individual specialized checkpoints into a single ensemble checkpoint

### Testing and Evaluation
- `test_ensemble.py`: Validation testing script that evaluates ensemble performance on ZOD dataset
- `benchmark_ensemble.py`: Comprehensive benchmarking tool for measuring computational efficiency

### Visualization
- `visualize_ensemble.py`: Script for creating visual overlays of ensemble predictions on test images

## Usage

### 1. Merging Checkpoints

To create an ensemble from specialized checkpoints:

```bash
cd /path/to/fusion-training
python3 ensemble/ensemble_merge.py \
    --vehicle /path/to/vehicle_only/checkpoints/epoch_X.pth \
    --sign /path/to/sign_only/checkpoints/epoch_Y.pth \
    --human /path/to/human_only/checkpoints/epoch_Z.pth \
    --output logs/zod/clft/specialization/ensemble/merged_ensemble_checkpoint.pth
```

### 2. Testing Ensemble Performance

Run validation testing on the merged ensemble:

```bash
cd /path/to/fusion-training
PYTHONPATH=/path/to/fusion-training python3 ensemble/test_ensemble.py
```

This will evaluate the ensemble on ZOD validation data and output per-class IoU metrics.

### 3. Benchmarking

Measure computational performance:

```bash
cd /path/to/fusion-training
PYTHONPATH=/path/to/fusion-training python3 ensemble/benchmark_ensemble.py --modality cross_fusion
```

### 4. Visualization

Generate prediction visualizations:

```bash
cd /path/to/fusion-training
PYTHONPATH=/path/to/fusion-training python3 ensemble/visualize_ensemble.py \
    --input /path/to/input_images \
    --output /path/to/output_visualizations
```

## Ensemble Architecture

The ensemble uses a pseudo-merged approach where individual specialized models (trained for vehicle, sign, and human detection) are stored in a single checkpoint file. During inference:

1. The ensemble loads all three specialized models
2. Runs forward passes through each model branch
3. Combines predictions by taking maximum probabilities across classes
4. Outputs unified segmentation masks

## Performance

Current ensemble achieves:
- **mIoU**: 51.3% on ZOD validation
- **Vehicle IoU**: 70.8%
- **Sign IoU**: 44.0%
- **Human IoU**: 39.2%
- **FPS**: 10.3 (on NVIDIA RTX 5070 Ti)
- **Parameters**: 382.8M

## Requirements

- PyTorch
- CUDA-compatible GPU (recommended)
- ZOD dataset access
- Pre-trained specialized checkpoints

## Notes

- The ensemble requires significant computational resources due to running three models in parallel
- While it improves accuracy over single models, the speed trade-off may not be suitable for real-time applications
- All scripts assume the project structure and data paths are correctly set up