#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data input handling for visualization.
"""
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.helpers import relabel_annotation


class DataLoader:
    """Handles loading and preprocessing of data for inference/visualization."""
    
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['Dataset']['name']
        self.cam_mean = config['Dataset']['transforms']['image_mean']
        self.cam_std = config['Dataset']['transforms']['image_std']
        self._setup_lidar_normalization()
        # create normalization transform if means/stds were set
        if hasattr(self, 'lidar_mean') and hasattr(self, 'lidar_std') and self.lidar_mean is not None and self.lidar_std is not None:
            self.lidar_normalize = transforms.Normalize(mean=self.lidar_mean, std=self.lidar_std)
        else:
            self.lidar_normalize = None
        self.resize = config['Dataset']['transforms']['resize']
    
    def _setup_lidar_normalization(self):
        """Setup dataset-specific LiDAR normalization."""
        t = self.config['Dataset']['transforms']
        # all datasets now use generic lidar_mean / lidar_std keys
        # fall back to None if missing (handled later)
        self.lidar_mean = t.get('lidar_mean')
        self.lidar_std  = t.get('lidar_std')
    
    def load_rgb(self, image_path):
        """Load and preprocess RGB image."""
        rgb_normalize = transforms.Compose([
            transforms.Resize((self.resize, self.resize), 
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.cam_mean, std=self.cam_std)
        ])
        
        rgb = Image.open(image_path).convert('RGB')
        return rgb_normalize(rgb)
    
    def load_annotation(self, anno_path):
        """Load and preprocess annotation."""
        anno = Image.open(anno_path)
        anno = np.array(anno)
        
        # Apply relabeling
        anno = relabel_annotation(anno, self.config)
        
        # Convert to tensor and resize
        anno_tensor = anno.float()
        anno_tensor = transforms.Resize(
            (self.resize, self.resize), 
            interpolation=transforms.InterpolationMode.NEAREST
        )(anno_tensor)
        
        return anno_tensor.squeeze(0)
    
    def load_lidar(self, lidar_path):
        """Load and preprocess LiDAR data for inference/visualization.

        All supported datasets use the same PNG format.  We mimic the
        preprocessing performed by :class:`tools.dataset_png.DatasetPNG`:
        resize the image with PIL, convert to tensor, and normalize if
        parameters are available.
        """
        lidar_pil = Image.open(lidar_path)
        lidar_pil = lidar_pil.resize((self.resize, self.resize), resample=Image.BILINEAR)
        lidar_tensor = TF.to_tensor(lidar_pil)
        if self.lidar_normalize is not None:
            lidar_tensor = self.lidar_normalize(lidar_tensor)
        return lidar_tensor
    

