#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation testing of ZOD Merged Ensemble Model.
Tests the single merged model on validation data.
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from zod_ensemble_predictor import ZODMergedEnsemblePredictor
from tools.dataset_png import DatasetPNG
from core.metrics_calculator import MetricsCalculator
from utils.metrics import find_overlap_exclude_bg_ignore


def compute_ap_for_class(pred_probs, targets):
    """Compute Average Precision for a single class using VOC 2010 method."""
    if len(pred_probs) == 0 or targets.sum() == 0:
        return 0.0

    # Sort by prediction confidence (descending)
    sorted_indices = torch.argsort(pred_probs, descending=True)
    pred_probs = pred_probs[sorted_indices]
    targets = targets[sorted_indices]

    # Calculate cumulative true positives and false positives
    tp = torch.cumsum(targets, dim=0).float()
    fp = torch.cumsum(1 - targets, dim=0).float()

    # Calculate precision and recall
    precision = tp / (tp + fp + 1e-6)
    recall = tp / targets.sum()

    # Use VOC 2010 AP calculation method
    return voc_ap(recall, precision)


def voc_ap(recall, precision):
    """Calculate AP using VOC 2010 method."""
    if len(recall) == 0:
        return 0.0

    # Convert to numpy
    recall = recall.cpu().numpy()
    precision = precision.cpu().numpy()

    # Add sentinel values
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def test_merged_on_validation(device='cuda:0'):
    """Test merged ensemble on validation data."""

    # Load merged predictor
    predictor = ZODMergedEnsemblePredictor(device=device)

    # Load config for data loading (use baseline config with all classes)
    config_path = "/media/tom/ml/projects/fusion-training/config/zod/clft/specialization/baseline.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Setup validation data
    val_split_file = "/media/tom/ml/projects/fusion-training/zod_dataset/validation.txt"
    if not os.path.exists(val_split_file):
        print(f"Validation split file {val_split_file} not found")
        return None

    Dataset = DatasetPNG
    val_data = Dataset(config, 'val', val_split_file)

    # Create data loader
    val_dataloader = DataLoader(
        val_data,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=2
    )

    # Setup metrics calculator for multi-class evaluation
    multi_class_config = config
    num_eval_classes = sum(1 for cls in multi_class_config['Dataset']['train_classes'] if cls['index'] > 0)
    metrics_calc = MetricsCalculator(multi_class_config, num_eval_classes, find_overlap_exclude_bg_ignore)

    # Storage for AP calculation
    all_predictions = {'vehicle': [], 'sign': [], 'human': []}
    all_targets = {'vehicle': [], 'sign': [], 'human': []}
    all_pred_classes = []
    all_remapped_targets = []

    sample_count = 0
    progress_bar = tqdm(total=len(val_data), desc='Validation', unit='samples')

    for batch_idx, batch in enumerate(val_dataloader):
        rgb_batch = batch['rgb'].to(device)
        lidar_batch = batch['lidar'].to(device)
        target_batch = batch['anno'].to(device)

        with torch.no_grad():
            pred_batch = predictor.predict(rgb_batch, lidar_batch)

        # Get predictions
        pred_classes = torch.argmax(pred_batch, dim=1)

        # Remap target to match prediction format (background=0, vehicle=1, sign=2, human=3)
        remapped_target = torch.zeros_like(target_batch)
        for cls_config in config['Dataset']['train_classes']:
            if cls_config['index'] > 0:  # Skip background
                train_idx = cls_config['index']
                for dataset_idx in cls_config['dataset_mapping']:
                    remapped_target[target_batch == dataset_idx] = train_idx

        # Store for confusion matrix computation
        all_pred_classes.append(pred_classes)
        all_remapped_targets.append(remapped_target)

        # Update metrics calculator
        metrics_calc.update(pred_batch, remapped_target)

        # Store predictions and targets for AP calculation
        probs = torch.softmax(pred_batch, dim=1)

        for cls_idx, cls_name in enumerate(['vehicle', 'sign', 'human']):
            train_idx = cls_idx + 1

            cls_probs = probs[:, train_idx, :, :]
            cls_preds = (pred_classes == train_idx).float()
            cls_targets = (remapped_target == train_idx).float()

            cls_probs_flat = cls_probs.flatten()
            cls_targets_flat = cls_targets.flatten()

            relevant_mask = (cls_preds.flatten() > 0) | (cls_targets_flat > 0)

            if relevant_mask.sum() > 0:
                all_predictions[cls_name].append(cls_probs_flat[relevant_mask])
                all_targets[cls_name].append(cls_targets_flat[relevant_mask])

        batch_size = rgb_batch.shape[0]
        sample_count += batch_size
        progress_bar.update(batch_size)

    progress_bar.close()

    # Compute confusion matrix from all collected predictions
    print("Computing confusion matrix...")
    all_pred_flat = torch.cat([p.flatten() for p in all_pred_classes])
    all_target_flat = torch.cat([t.flatten() for t in all_remapped_targets])
    confusion_matrix = torch.bincount(
        4 * all_pred_flat + all_target_flat,
        minlength=16
    ).reshape(4, 4)

    # Compute final metrics
    metrics = metrics_calc.compute()

    # Compute AP for each class
    class_results = {}
    for cls_name in ['vehicle', 'sign', 'human']:
        if all_predictions[cls_name]:
            pred_probs = torch.cat(all_predictions[cls_name])
            pred_targets = torch.cat(all_targets[cls_name])
            ap = compute_ap_for_class(pred_probs, pred_targets)
        else:
            ap = 0.0

        cls_idx = ['vehicle', 'sign', 'human'].index(cls_name)
        class_results[cls_name] = {
            'iou': metrics['iou'][cls_idx].item(),
            'precision': metrics['precision'][cls_idx].item(),
            'recall': metrics['recall'][cls_idx].item(),
            'f1_score': metrics['f1'][cls_idx].item(),
            'ap': ap
        }

    # Overall metrics
    confusion_matrix_cpu = confusion_matrix.cpu()
    pixel_accuracy = confusion_matrix_cpu.diag().sum() / confusion_matrix_cpu.sum()
    mean_accuracy = sum(class_results[cls]['recall'] for cls in ['vehicle', 'sign', 'human']) / 3
    class_pixels = confusion_matrix_cpu.sum(dim=0)
    weights = class_pixels / class_pixels.sum()
    eval_iou = torch.tensor([class_results[cls]['iou'] for cls in ['vehicle', 'sign', 'human']])
    fw_iou = (weights[1:4] * eval_iou).sum().item()

    overall_results = {
        'mIoU_foreground': metrics['mean_iou'],
        'mean_accuracy': mean_accuracy,
        'fw_iou': fw_iou,
        'pixel_accuracy': pixel_accuracy.item(),
        'confusion_matrix': confusion_matrix.tolist(),
        'confusion_matrix_labels': ['background', 'vehicle', 'sign', 'human']
    }

    results = {
        **class_results,
        'overall': overall_results
    }

    # Print results
    print(f"\n{'='*40} MERGED MODEL VALIDATION RESULTS {'='*40}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Mean F1: {metrics['mean_f1']:.4f}")

    class_names = ['vehicle', 'sign', 'human']
    print("\nPer-class IoU:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {metrics['iou'][i].item():.4f}")

    print("\nPer-class Precision:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {metrics['precision'][i].item():.4f}")

    print("\nPer-class Recall:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {metrics['recall'][i].item():.4f}")

    print("\nPer-class F1:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}: {metrics['f1'][cls_idx].item():.4f}")

    print("\nPer-class AP:")
    for class_name in class_names:
        print(f"  {class_name}: {class_results[class_name]['ap']:.4f}")
    print(f"\nTested {sample_count} samples")
    return results


if __name__ == "__main__":
    # Run validation testing
    results = test_merged_on_validation(device='cuda:0')

    if results:
        print("\n" + "="*50)
        print("MERGED MODEL VALIDATION IoU RESULTS:")
        print("="*50)
        for cls in ['vehicle', 'sign', 'human']:
            print(f"{cls.capitalize()}: {results[cls]['iou']:.4f}")

        # Calculate mIoU for comparison with table
        miou = (results['vehicle']['iou'] + results['sign']['iou'] + results['human']['iou']) / 3
        print(f"\nmIoU: {miou:.4f}")

        print("\nCompare with ensemble results:")
        print("Original Ensemble: ~47.9% mIoU")
        print(f"Merged Model: {miou:.1f}%")

        # Save results to JSON file
        import json
        import uuid
        from datetime import datetime

        output_data = {
            "timestamp": datetime.now().isoformat(),
            "test_uuid": str(uuid.uuid4()),
            "validation_results": results,
            "model_type": "merged_ensemble"
        }

        output_file = "/media/tom/ml/projects/fusion-training/logs/zod/clft/specialization/ensemble/merged_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nResults saved to: {output_file}")