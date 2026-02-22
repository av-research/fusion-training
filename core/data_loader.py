#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data input handling for visualization.
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.helpers import relabel_annotation
from utils.lidar_process import open_lidar, get_unresized_lid_img_val


class DataLoader:
    """Handles loading and preprocessing of data for inference/visualization."""
    
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['Dataset']['name']
        self.cam_mean = config['Dataset']['transforms']['image_mean']
        self.cam_std = config['Dataset']['transforms']['image_std']
        self._setup_lidar_normalization()
        self.resize = config['Dataset']['transforms']['resize']
    
    def _setup_lidar_normalization(self):
        """Setup dataset-specific LiDAR normalization."""
        t = self.config['Dataset']['transforms']
        if self.dataset_name == 'iseauto':
            self.lidar_mean = t['lidar_mean']
            self.lidar_std  = t['lidar_std']
        else:
            # Use .get() for both keys so neither is accessed when absent
            self.lidar_mean = t.get('lidar_mean') or t.get('lidar_mean_waymo')
            self.lidar_std  = t.get('lidar_std')  or t.get('lidar_std_waymo')
    
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
        """Load and preprocess LiDAR data (dataset-specific)."""
        if self.dataset_name == 'waymo':
            return self._load_waymo_lidar(lidar_path)
        elif self.dataset_name == 'iseauto':
            return self._load_zod_lidar(lidar_path)  # Iseauto uses same format as ZOD
        else:  # ZOD
            return self._load_zod_lidar(lidar_path)
    
    def _load_zod_lidar(self, lidar_path):
        """Load ZOD LiDAR from PNG projection."""
        lidar_pil = Image.open(lidar_path)
        lidar_tensor = TF.to_tensor(lidar_pil)
        
        # For ZOD PNG data, do NOT apply normalization (matches training)
        # The PNG values are already in [0, 1] range
        
        lidar_tensor = transforms.Resize((self.resize, self.resize))(lidar_tensor)
        
        return lidar_tensor
    
    def _load_waymo_lidar(self, lidar_path):
        """Load Waymo LiDAR from PNG projection."""
        lidar_pil = Image.open(lidar_path)
        lidar_tensor = TF.to_tensor(lidar_pil)
        
        # For Waymo PNG data, do NOT apply normalization (matches training)
        # The PNG values are already in the expected range
        
        lidar_tensor = transforms.Resize((self.resize, self.resize))(lidar_tensor)
        
        return lidar_tensor
