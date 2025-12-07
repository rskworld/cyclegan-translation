"""
Advanced Data Augmentation for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides advanced data augmentation techniques.
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random
import numpy as np
from PIL import Image


class AdvancedAugmentation:
    """
    Advanced data augmentation for CycleGAN training.
    """
    
    def __init__(self, 
                 flip_prob=0.5,
                 rotation_range=15,
                 color_jitter_strength=0.3,
                 gaussian_noise_std=0.02,
                 cutout_prob=0.3,
                 cutout_size=0.2):
        """
        Initialize augmentation parameters.
        
        Args:
            flip_prob: Probability of horizontal flip
            rotation_range: Rotation range in degrees
            color_jitter_strength: Strength of color jittering
            gaussian_noise_std: Standard deviation for Gaussian noise
            cutout_prob: Probability of cutout augmentation
            cutout_size: Size of cutout relative to image
        """
        self.flip_prob = flip_prob
        self.rotation_range = rotation_range
        self.color_jitter_strength = color_jitter_strength
        self.gaussian_noise_std = gaussian_noise_std
        self.cutout_prob = cutout_prob
        self.cutout_size = cutout_size
    
    def get_train_transform(self, size=256):
        """
        Get training transform with augmentations.
        
        Args:
            size: Target image size
            
        Returns:
            Transform pipeline
        """
        return transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=self.flip_prob),
            transforms.RandomRotation(self.rotation_range),
            transforms.ColorJitter(
                brightness=self.color_jitter_strength,
                contrast=self.color_jitter_strength,
                saturation=self.color_jitter_strength,
                hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            self._add_gaussian_noise,
            self._apply_cutout
        ])
    
    def _add_gaussian_noise(self, tensor):
        """Add Gaussian noise to tensor."""
        if random.random() < 0.3:
            noise = torch.randn_like(tensor) * self.gaussian_noise_std
            tensor = tensor + noise
        return tensor
    
    def _apply_cutout(self, tensor):
        """Apply cutout augmentation."""
        if random.random() < self.cutout_prob:
            _, h, w = tensor.shape
            cutout_h = int(h * self.cutout_size)
            cutout_w = int(w * self.cutout_size)
            
            y = random.randint(0, h - cutout_h)
            x = random.randint(0, w - cutout_w)
            
            tensor[:, y:y+cutout_h, x:x+cutout_w] = 0
        
        return tensor


class MixUpAugmentation:
    """
    MixUp augmentation for CycleGAN.
    """
    
    def __init__(self, alpha=0.2):
        """
        Initialize MixUp augmentation.
        
        Args:
            alpha: MixUp parameter
        """
        self.alpha = alpha
    
    def __call__(self, batch_A, batch_B):
        """
        Apply MixUp to batches.
        
        Args:
            batch_A: Batch A
            batch_B: Batch B
            
        Returns:
            Mixed batches
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        mixed_A = lam * batch_A + (1 - lam) * batch_B
        mixed_B = lam * batch_B + (1 - lam) * batch_A
        
        return mixed_A, mixed_B, lam

