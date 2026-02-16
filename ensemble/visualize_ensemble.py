#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for ZOD Merged Ensemble predictions.
Creates overlays of merged model predictions on visualization images.
"""
import os
import json
import argparse
import numpy as np
import torch
import cv2

from ensemble.zod_ensemble_predictor import ZODMergedEnsemblePredictor
from core.data_loader import DataLoader as InferenceDataLoader
from core.visualizer import Visualizer
from utils.helpers import get_annotation_path


def get_lidar_path(cam_path, dataset_name):
    """Get LiDAR path based on dataset."""
    if dataset_name == 'zod':
        return cam_path.replace('/camera', '/lidar_png')
    else:  # waymo
        return cam_path.replace('/camera/', '/lidar_png/')


def load_image_paths(path_arg, dataroot):
    """Load image paths from file or single path."""
    if path_arg.endswith(('.png', '.jpg', '.jpeg')):
        # Single image
        return [path_arg]
    else:
        # Text file with multiple paths
        with open(path_arg, 'r') as f:
            paths = f.read().splitlines()
        return paths


def create_multi_class_config():
    """Create multi-class config for visualization."""
    return {
        'Dataset': {
            'train_classes': [
                {"name": "background", "index": 0, "weight": 0.1, "dataset_mapping": [0, 1], "color": [0, 0, 0]},
                {"name": "vehicle", "index": 1, "weight": 10.0, "dataset_mapping": [2], "color": [128, 0, 128]},
                {"name": "sign", "index": 2, "weight": 15.0, "dataset_mapping": [3], "color": [255, 0, 0]},
                {"name": "human", "index": 3, "weight": 20.0, "dataset_mapping": [4, 5], "color": [0, 255, 255]}
            ],
            'classes': [
                {"name": "background", "training_index": 0, "color": [0, 0, 0]},
                {"name": "vehicle", "training_index": 1, "color": [128, 0, 128]},
                {"name": "sign", "training_index": 2, "color": [255, 0, 0]},
                {"name": "human", "training_index": 3, "color": [0, 255, 255]}
            ]
        }
    }


def remap_annotation(anno_tensor):
    """Remap annotation from dataset indices to train_class indices."""
    remapped = torch.zeros_like(anno_tensor)
    remapped[anno_tensor == 2] = 1  # vehicle
    remapped[anno_tensor == 3] = 2  # sign
    remapped[(anno_tensor == 4) | (anno_tensor == 5)] = 3  # human
    return remapped


def process_images(predictor, data_loader, visualizer, image_paths, dataroot,
                   modality, dataset_name, device, config):
    """Process and visualize all images with merged model predictions."""

    for idx, path in enumerate(image_paths, 1):
        # Construct full paths
        if os.path.isabs(path):
            cam_path = path
        else:
            cam_path = os.path.join(dataroot, path)

        anno_path = get_annotation_path(cam_path, dataset_name, config)
        lidar_path = get_lidar_path(cam_path, dataset_name)

        # Verify paths match
        rgb_name = os.path.basename(cam_path).split('.')[0]
        anno_name = os.path.basename(anno_path).split('.')[0]
        lidar_name = os.path.basename(lidar_path).split('.')[0]

        assert rgb_name == anno_name, f"RGB and annotation names don't match: {rgb_name} vs {anno_name}"
        assert rgb_name == lidar_name, f"RGB and LiDAR names don't match: {rgb_name} vs {lidar_name}"

        # Load data
        print(f'Processing image {idx}/{len(image_paths)}: {rgb_name}')
        rgb = data_loader.load_rgb(cam_path).to(device, non_blocking=True).unsqueeze(0)

        # Load LiDAR data
        lidar = data_loader.load_lidar(lidar_path).to(device, non_blocking=True).unsqueeze(0)

        # Run merged model prediction
        with torch.no_grad():
            pred_logits = predictor.predict(rgb, lidar)

        # Convert logits to class predictions (argmax) for later use if needed
        pred_classes = torch.argmax(pred_logits, dim=1).squeeze(0).cpu().numpy()

        # Load and remap ground truth
        anno_cv2 = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
        if anno_cv2 is None:
            print(f'Warning: Could not load annotation {anno_path}')
            continue

        anno_tensor = torch.from_numpy(anno_cv2).unsqueeze(0)
        anno_remapped = remap_annotation(anno_tensor).squeeze(0).numpy()

        # Create visualizations - pass raw logits, not argmax
        visualizer.visualize_prediction(pred_logits.squeeze(0), cam_path, anno_path, idx)

        print(f'  Saved visualizations for {rgb_name}')


def main():
    parser = argparse.ArgumentParser(description='ZOD Merged Ensemble Visualization')
    parser.add_argument('--config', type=str, default='/media/tom/ml/projects/fusion-training/config/zod/clft/specialization/human_only.json',
                       help='Path to config file (for data loading)')
    parser.add_argument('--images', type=str, default='/media/tom/ml/projects/fusion-training/zod_dataset/visualizations.txt',
                       help='Path to image list file or single image path')
    parser.add_argument('--dataroot', type=str, default='/media/tom/ml/projects/fusion-training/zod_dataset',
                       help='Root directory of dataset')
    parser.add_argument('--output', type=str, default='/media/tom/ml/projects/fusion-training/logs/zod/clft/specialization/ensemble/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--modal', type=str, default='fusion',
                       choices=['rgb', 'lidar', 'fusion'],
                       help='Fusion modality')
    parser.add_argument('--dataset', type=str, default='zod',
                       choices=['zod', 'waymo'],
                       help='Dataset name')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on')

    args = parser.parse_args()

    # Load config for data loading
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Create multi-class config for visualization
    vis_config = create_multi_class_config()

    # Setup components
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_loader = InferenceDataLoader(config)
    visualizer = Visualizer(vis_config, args.output)
    predictor = ZODMergedEnsemblePredictor(device=args.device)

    # Load image paths
    image_paths = load_image_paths(args.images, args.dataroot)

    print(f"Starting merged ensemble visualization on {len(image_paths)} images")
    print(f"Output directory: {args.output}")
    print(f"Device: {device}")

    # Process all images
    process_images(predictor, data_loader, visualizer, image_paths, args.dataroot,
                   args.modal, args.dataset, device, config)

    print("Visualization complete!")


if __name__ == '__main__':
    main()