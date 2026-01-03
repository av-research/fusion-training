#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test aggregation utilities for collecting and analyzing results across multiple checkpoints.
"""
import os
import json
import datetime
import uuid
import numpy as np
from integrations.vision_service import send_test_results_from_file


def collect_checkpoint_results(checkpoint_paths, test_function, *test_args):
    """
    Test all checkpoints and collect results.

    Args:
        checkpoint_paths: List of checkpoint file paths
        test_function: Function that takes (checkpoint_path, *test_args) and returns results dict
        *test_args: Additional arguments to pass to test_function

    Returns:
        List of dicts with checkpoint data and results
    """
    all_checkpoint_results = []

    for i, checkpoint_path in enumerate(checkpoint_paths):
        print(f"\n{'='*50}")
        print(f"Testing checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
        print(f"{'='*50}")

        # Extract epoch info
        epoch_num, epoch_uuid = extract_epoch_info(checkpoint_path)

        # Run test function for this checkpoint
        checkpoint_results = test_function(checkpoint_path, *test_args)

        # Store results for this checkpoint
        checkpoint_data = {
            'epoch': epoch_num,
            'epoch_uuid': epoch_uuid,
            'checkpoint_path': checkpoint_path,
            'results': checkpoint_results
        }
        all_checkpoint_results.append(checkpoint_data)

        print(f'Completed testing checkpoint {i+1}/{len(checkpoint_paths)}')

    return all_checkpoint_results


def calculate_aggregated_statistics(all_checkpoint_results, weather_conditions, eval_classes):
    """
    Calculate aggregated statistics across all checkpoints.

    Args:
        all_checkpoint_results: List of checkpoint result dicts
        weather_conditions: List of weather condition names
        eval_classes: List of evaluation class names

    Returns:
        Dict with aggregated statistics
    """
    print(f"\n{'='*50}")
    print("Calculating aggregated statistics across all checkpoints...")
    print(f"{'='*50}")

    aggregated_results = {}

    for weather in weather_conditions:
        aggregated_results[weather] = {}

        # For all models: per-class metrics with iou, precision, recall, f1, ap
        # For each class
        for cls_name in eval_classes:
            # Collect metrics across all checkpoints
            iou_values = []
            precision_values = []
            recall_values = []
            f1_values = []
            ap_values = []

            for checkpoint_data in all_checkpoint_results:
                if weather in checkpoint_data['results'] and cls_name in checkpoint_data['results'][weather]:
                    metrics = checkpoint_data['results'][weather][cls_name]
                    iou_values.append(metrics['iou'])
                    precision_values.append(metrics['precision'])
                    recall_values.append(metrics['recall'])
                    f1_values.append(metrics['f1_score'] if 'f1_score' in metrics else metrics['f1'])
                    ap_values.append(metrics['ap'])

            # Calculate statistics
            aggregated_results[weather][cls_name] = {
                'iou': {
                    'mean': float(np.mean(iou_values)) if iou_values else 0.0,
                    'std': float(np.std(iou_values)) if iou_values else 0.0,
                    'min': float(np.min(iou_values)) if iou_values else 0.0,
                    'max': float(np.max(iou_values)) if iou_values else 0.0
                },
                'precision': {
                    'mean': float(np.mean(precision_values)) if precision_values else 0.0,
                    'std': float(np.std(precision_values)) if precision_values else 0.0,
                    'min': float(np.min(precision_values)) if precision_values else 0.0,
                    'max': float(np.max(precision_values)) if precision_values else 0.0
                },
                'recall': {
                    'mean': float(np.mean(recall_values)) if recall_values else 0.0,
                    'std': float(np.std(recall_values)) if recall_values else 0.0,
                    'min': float(np.min(recall_values)) if recall_values else 0.0,
                    'max': float(np.max(recall_values)) if recall_values else 0.0
                },
                'f1_score': {
                    'mean': float(np.mean(f1_values)) if f1_values else 0.0,
                    'std': float(np.std(f1_values)) if f1_values else 0.0,
                    'min': float(np.min(f1_values)) if f1_values else 0.0,
                    'max': float(np.max(f1_values)) if f1_values else 0.0
                },
                'ap': {
                    'mean': float(np.mean(ap_values)) if ap_values else 0.0,
                    'std': float(np.std(ap_values)) if ap_values else 0.0,
                    'min': float(np.min(ap_values)) if ap_values else 0.0,
                    'max': float(np.max(ap_values)) if ap_values else 0.0
                }
            }

        # Calculate mean IoU across classes for this weather condition
        mean_iou_per_checkpoint = []
        for checkpoint_data in all_checkpoint_results:
            if weather in checkpoint_data['results']:
                class_ious = [checkpoint_data['results'][weather][cls]['iou'] for cls in eval_classes if cls in checkpoint_data['results'][weather]]
                if class_ious:
                    mean_iou_per_checkpoint.append(np.mean(class_ious))

        if mean_iou_per_checkpoint:
            aggregated_results[weather]['mean_iou'] = {
                'mean': float(np.mean(mean_iou_per_checkpoint)),
                'std': float(np.std(mean_iou_per_checkpoint)),
                'min': float(np.min(mean_iou_per_checkpoint)),
                'max': float(np.max(mean_iou_per_checkpoint))
            }

    return aggregated_results


def save_and_upload_aggregated_results(config, checkpoint_paths, aggregated_results, all_checkpoint_results):
    """
    Save aggregated results to file and upload to vision service.

    Args:
        config: Configuration dict
        checkpoint_paths: List of checkpoint paths
        aggregated_results: Aggregated statistics dict
        all_checkpoint_results: Individual checkpoint results

    Returns:
        Path to saved results file
    """
    # Save aggregated results
    logdir = config['Log']['logdir']
    test_results_dir = os.path.join(logdir, 'test_results')
    os.makedirs(test_results_dir, exist_ok=True)

    # Generate filename for aggregated results
    test_uuid = str(uuid.uuid4())
    filename = f'aggregated_test_results_{test_uuid}.json'
    output_path = os.path.join(test_results_dir, filename)

    output_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "num_checkpoints_tested": len(checkpoint_paths),
        "checkpoint_paths": checkpoint_paths,
        "test_uuid": test_uuid,
        "aggregated_results": aggregated_results,
        "individual_checkpoint_results": all_checkpoint_results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nAggregated results saved to {output_path}")

    # Print summary
    print_aggregated_summary(aggregated_results)

    # Send aggregated results to vision service
    print("\nUploading aggregated test results to vision service...")
    upload_success = send_test_results_from_file(output_path)
    if upload_success:
        print("✅ Aggregated test results successfully uploaded to vision service")
    else:
        print("❌ Failed to upload aggregated test results to vision service")

    return output_path


def print_aggregated_summary(aggregated_results):
    """
    Print a summary of aggregated results.

    Args:
        aggregated_results: Aggregated statistics dict
    """
    print("\nAggregated Summary (Mean ± Std):")

    for weather, results in aggregated_results.items():
        if 'mean_iou' in results:
            mean_iou = results['mean_iou']
            print(f"{weather}: mIoU={mean_iou['mean']:.4f}±{mean_iou['std']:.4f} "
                  f"(min={mean_iou['min']:.4f}, max={mean_iou['max']:.4f})")


def extract_epoch_info(checkpoint_path):
    """
    Extract epoch number and UUID from checkpoint path.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Tuple of (epoch_num, epoch_uuid)
    """
    import re
    filename = os.path.basename(checkpoint_path)

    # Try new format first: epoch_{num}_{uuid}.pth
    epoch_match = re.search(r'epoch_(\d+)_([a-f0-9\-]+)\.pth', filename)
    if epoch_match:
        epoch_num = int(epoch_match.group(1))
        epoch_uuid = epoch_match.group(2)
        return epoch_num, epoch_uuid

    # Try old format: checkpoint_{num}.pth
    epoch_match = re.search(r'checkpoint_(\d+)\.pth', filename)
    if epoch_match:
        epoch_num = int(epoch_match.group(1))
        return epoch_num, None

    # Default
    return 0, None