#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization utilities for model predictions.
"""
import os
import cv2
import torch
import numpy as np
from utils.helpers import draw_test_segmentation_map, relabel_annotation


class Visualizer:
    """Handles visualization of model predictions."""
    
    def __init__(self, config, output_base):
        self.config = config
        self.output_base = output_base
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """Create output directories for visualizations."""
        self.dirs = {
            'segment': os.path.join(self.output_base, 'segment'),
            'overlay': os.path.join(self.output_base, 'overlay'),
            'compare': os.path.join(self.output_base, 'compare'),
            'correct_only': os.path.join(self.output_base, 'correct_only')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def visualize_prediction(self, output_seg, rgb_path, anno_path, idx):
        """
        Create all visualization outputs for a prediction.
        
        Args:
            output_seg: Model output segmentation
            rgb_path: Path to original RGB image
            anno_path: Path to ground truth annotation
            idx: Image index for logging
        """
        # Generate segmentation map (returns BGR format from config colors)
        segmented_image = draw_test_segmentation_map(output_seg, self.config)
        
        # Load original image
        rgb_cv2 = cv2.imread(rgb_path)
        if rgb_cv2 is None:
            print(f'Warning: Could not load RGB image {rgb_path}')
            if not os.path.exists(rgb_path):
                print(f'File does not exist: {rgb_path}')
            else:
                print(f'File exists but could not be read (possibly corrupted)')
            print('Skipping overlay for this image')
            return
        
        h, w = rgb_cv2.shape[:2]
        
        # Resize segmentation to original dimensions
        seg_resize_bgr = cv2.resize(segmented_image, (w, h), 
                               interpolation=cv2.INTER_NEAREST)
        
        # Save segmentation (already in BGR)
        seg_path = self._get_output_path('segment', rgb_path)
        print(f'Saving segment result {idx}...')
        cv2.imwrite(seg_path, seg_resize_bgr)
        
        # Save overlay (rgb_cv2 is BGR, seg_resize_bgr is BGR)
        overlay = self._create_overlay(rgb_cv2, seg_resize_bgr, alpha=0.6)
        overlay_path = self._get_output_path('overlay', rgb_path)
        print(f'Saving overlay result {idx}...')
        cv2.imwrite(overlay_path, overlay)
        
        # Save comparison and correct_only if annotation exists
        if os.path.exists(anno_path):
            self._save_comparison(output_seg, seg_resize_bgr, rgb_cv2, rgb_path, anno_path, 
                                 h, w, idx)
    
    def _save_comparison(self, output_seg, seg_resize, rgb_cv2, rgb_path, anno_path, 
                        h, w, idx):
        """Save comparison and correct_only visualizations."""
        # Load and process ground truth
        gt_anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
        if gt_anno is None:
            print(f'Warning: Could not load annotation for {anno_path}')
            return
        
        # Resize and relabel ground truth
        gt_anno_resized = cv2.resize(gt_anno, (w, h), 
                                     interpolation=cv2.INTER_NEAREST)
        gt_anno_tensor = torch.from_numpy(gt_anno_resized).unsqueeze(0).long()
        gt_relabeled = relabel_annotation(gt_anno_tensor, self.config)
        gt_relabeled = gt_relabeled.squeeze(0).squeeze(0).numpy()
        
        # Get predictions at the model's native resolution (before upsampling to original image size)
        # output_seg is already the network output (typically 384x384).  We perform the comparison
        # at this smaller scale so that the number of "correct" pixels matches what the metric
        # calculation used during testing.  We will then upsample the mask for visualization.
        pred_small = torch.argmax(output_seg.squeeze(), dim=0).detach().cpu().numpy().astype(np.uint8)

        # Build a corresponding ground‑truth map at the same resolution
        model_size = self.config['Dataset']['transforms']['resize']
        if model_size is None:
            # fallback to height/width if configuration is missing
            model_size = pred_small.shape[0]
        gt_small = cv2.resize(gt_anno, (model_size, model_size), interpolation=cv2.INTER_NEAREST)
        gt_small_tensor = torch.from_numpy(gt_small).unsqueeze(0).long()
        gt_small_relabeled = relabel_annotation(gt_small_tensor, self.config)
        gt_small_relabeled = gt_small_relabeled.squeeze(0).squeeze(0).numpy().astype(np.uint8)

        # Compute masks at model resolution (exclude background class 0)
        non_bg_small = (gt_small_relabeled != 0)
        correct_small = (pred_small == gt_small_relabeled) & non_bg_small
        incorrect_small = (pred_small != gt_small_relabeled) & non_bg_small

        # Upsample masks to original image size for saving
        correct_mask = cv2.resize(correct_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        incorrect_mask = cv2.resize(incorrect_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

        # Create comparison (green=correct, red=incorrect)
        # Note: OpenCV uses BGR format
        comparison = np.zeros((h, w, 3), dtype=np.uint8)
        # Green for correct, Red for incorrect, Black for background
        comparison[correct_mask] = [0, 255, 0]    # Green (BGR)
        comparison[incorrect_mask] = [0, 0, 255]  # Red (BGR)
        # Background remains black [0, 0, 0]
        
        compare_path = self._get_output_path('compare', rgb_path)
        print(f'Saving comparison result {idx}...')
        cv2.imwrite(compare_path, comparison)
        
        # Create correct_only - show only correctly predicted pixels on original image
        # we already computed incorrect_mask upstream (upsampled to original size)
        correct_only_segmented = seg_resize.copy()
        correct_only_segmented[incorrect_mask] = [0, 0, 0]

        # Overlay correct predictions on original image with transparency (like overlay)
        correct_only_overlay = self._create_overlay(rgb_cv2, correct_only_segmented, alpha=0.6)

        correct_only_path = self._get_output_path('correct_only', rgb_path)
        print(f'Saving correct_only result {idx}...')
        cv2.imwrite(correct_only_path, correct_only_overlay)
    
    def _create_overlay(self, image, segmented_image, alpha=0.6):
        """
        Create overlay with transparent masks on original image.
        Both image and segmented_image should be in BGR format.
        
        Args:
            image: Original image
            segmented_image: Segmented image to overlay
            alpha: Transparency level (1.0 = no dimming, 0.0 = fully transparent)
        """
        # Create a copy of the original image
        overlay = image.copy().astype(np.float32)

        # Find non-black pixels in segmented image (predicted classes)
        # Background is black [0, 0, 0] in BGR
        mask = np.any(segmented_image != [0, 0, 0], axis=2)

        # Apply alpha blending only to predicted regions
        overlay[mask] = alpha * segmented_image[mask].astype(np.float32) + (1 - alpha) * overlay[mask]

        return overlay.astype(np.uint8)

    def _get_output_path(self, output_type, input_path):
        """Get output path for a given visualization type."""
        filename = os.path.basename(input_path)
        return os.path.join(self.dirs[output_type], filename)
