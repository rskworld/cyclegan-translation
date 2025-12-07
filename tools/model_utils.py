"""
Model Utilities for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides utility functions for model management.
"""

import torch
import torch.nn as nn
import os
from collections import OrderedDict


class ModelUtils:
    """
    Utility class for model operations.
    """
    
    @staticmethod
    def count_parameters(model):
        """
        Count the number of trainable parameters in a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def save_checkpoint(model, optimizer, epoch, loss, filepath):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            loss: Current loss
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    @staticmethod
    def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to checkpoint
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            device: Device to load on
            
        Returns:
            Epoch and loss from checkpoint
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', float('inf'))
        
        print(f"Checkpoint loaded from {filepath}")
        return epoch, loss
    
    @staticmethod
    def freeze_model(model):
        """Freeze all model parameters."""
        for param in model.parameters():
            param.requires_grad = False
    
    @staticmethod
    def unfreeze_model(model):
        """Unfreeze all model parameters."""
        for param in model.parameters():
            param.requires_grad = True
    
    @staticmethod
    def get_model_size_mb(model):
        """
        Get model size in megabytes.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    @staticmethod
    def remove_module_prefix(state_dict):
        """
        Remove 'module.' prefix from state dict keys (for DataParallel models).
        
        Args:
            state_dict: State dictionary
            
        Returns:
            Cleaned state dictionary
        """
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        return new_state_dict

