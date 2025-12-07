"""
Comprehensive Metrics Calculator for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides comprehensive evaluation metrics for CycleGAN models.
"""

import torch
import numpy as np
from .fid_score import calculate_fid
from .inception_score import calculate_inception_score
from .lpips_score import calculate_lpips


class MetricsCalculator:
    """
    Comprehensive metrics calculator for CycleGAN evaluation.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the metrics calculator.
        
        Args:
            device: Device to run calculations on
        """
        self.device = device
    
    def calculate_all_metrics(self, real_images, fake_images, real_paths=None, fake_paths=None):
        """
        Calculate all available metrics.
        
        Args:
            real_images: Real images tensor or list of paths
            fake_images: Generated images tensor or list of paths
            real_paths: Paths to real images (if images are paths)
            fake_paths: Paths to fake images (if images are paths)
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # FID Score
        try:
            fid_score = self.calculate_fid_score(real_images, fake_images, real_paths, fake_paths)
            metrics['FID'] = fid_score
        except Exception as e:
            print(f"Error calculating FID: {e}")
            metrics['FID'] = None
        
        # Inception Score
        try:
            is_mean, is_std = self.calculate_inception_score(fake_images, fake_paths)
            metrics['IS_mean'] = is_mean
            metrics['IS_std'] = is_std
        except Exception as e:
            print(f"Error calculating IS: {e}")
            metrics['IS_mean'] = None
            metrics['IS_std'] = None
        
        # LPIPS Score
        try:
            lpips_score = self.calculate_lpips_score(real_images, fake_images, real_paths, fake_paths)
            metrics['LPIPS'] = lpips_score
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            metrics['LPIPS'] = None
        
        return metrics
    
    def calculate_fid_score(self, real_images, fake_images, real_paths=None, fake_paths=None):
        """Calculate FID score."""
        return calculate_fid(real_images, fake_images, real_paths, fake_paths, self.device)
    
    def calculate_inception_score(self, fake_images, fake_paths=None):
        """Calculate Inception Score."""
        return calculate_inception_score(fake_images, fake_paths, self.device)
    
    def calculate_lpips_score(self, real_images, fake_images, real_paths=None, fake_paths=None):
        """Calculate LPIPS score."""
        return calculate_lpips(real_images, fake_images, real_paths, fake_paths, self.device)

