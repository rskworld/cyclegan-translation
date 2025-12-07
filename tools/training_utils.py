"""
Training Utilities for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides utility functions for training.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np


class TrainingUtils:
    """
    Utility class for training operations.
    """
    
    @staticmethod
    def get_learning_rate(optimizer):
        """
        Get current learning rate from optimizer.
        
        Args:
            optimizer: Optimizer
            
        Returns:
            Learning rate
        """
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    @staticmethod
    def set_learning_rate(optimizer, lr):
        """
        Set learning rate for optimizer.
        
        Args:
            optimizer: Optimizer
            lr: Learning rate
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    @staticmethod
    def clip_gradients(model, max_norm=1.0):
        """
        Clip gradients to prevent exploding gradients.
        
        Args:
            model: Model
            max_norm: Maximum gradient norm
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    @staticmethod
    def exponential_moving_average(model, ema_model, decay=0.999):
        """
        Update exponential moving average model.
        
        Args:
            model: Current model
            ema_model: EMA model
            decay: Decay factor
        """
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)
    
    @staticmethod
    def create_mixed_precision_scaler():
        """Create gradient scaler for mixed precision training."""
        return GradScaler()
    
    @staticmethod
    def get_device():
        """Get available device."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def set_seed(seed=42):
        """
        Set random seed for reproducibility.
        
        Args:
            seed: Random seed
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

