"""
TensorBoard Logger for CycleGAN Training
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides TensorBoard logging functionality for training monitoring.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import torchvision.utils as vutils


class TensorBoardLogger:
    """
    TensorBoard logger for CycleGAN training.
    """
    
    def __init__(self, log_dir='./logs', experiment_name='cyclegan'):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")
        self.global_step = 0
    
    def log_scalar(self, tag, value, step=None):
        """
        Log a scalar value.
        
        Args:
            tag: Tag name
            value: Scalar value
            step: Step number (uses global_step if None)
        """
        step = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_image(self, tag, image, step=None, normalize=True):
        """
        Log an image.
        
        Args:
            tag: Tag name
            image: Image tensor or PIL Image
            step: Step number
            normalize: Whether to normalize image
        """
        step = step if step is not None else self.global_step
        
        if isinstance(image, Image.Image):
            # Convert PIL to tensor
            from torchvision import transforms
            transform = transforms.ToTensor()
            image = transform(image).unsqueeze(0)
        
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            self.writer.add_image(tag, image, step, dataformats='NCHW')
    
    def log_images(self, tag, images, step=None, nrow=8):
        """
        Log multiple images in a grid.
        
        Args:
            tag: Tag name
            images: List of image tensors
            step: Step number
            nrow: Number of images per row
        """
        step = step if step is not None else self.global_step
        
        if isinstance(images, list):
            images = torch.stack(images)
        
        grid = vutils.make_grid(images, nrow=nrow, normalize=True, scale_each=True)
        self.writer.add_image(tag, grid, step)
    
    def log_histogram(self, tag, values, step=None):
        """
        Log a histogram.
        
        Args:
            tag: Tag name
            values: Values to plot
            step: Step number
        """
        step = step if step is not None else self.global_step
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model, input_shape):
        """
        Log model graph.
        
        Args:
            model: Model to log
            input_shape: Input shape (C, H, W)
        """
        dummy_input = torch.randn(1, *input_shape)
        self.writer.add_graph(model, dummy_input)
    
    def log_losses(self, losses_dict, step=None):
        """
        Log multiple losses.
        
        Args:
            losses_dict: Dictionary of loss values
            step: Step number
        """
        step = step if step is not None else self.global_step
        for tag, value in losses_dict.items():
            self.writer.add_scalar(f'Losses/{tag}', value, step)
    
    def log_learning_rate(self, optimizer, step=None):
        """
        Log learning rates.
        
        Args:
            optimizer: Optimizer
            step: Step number
        """
        step = step if step is not None else self.global_step
        for i, param_group in enumerate(optimizer.param_groups):
            self.writer.add_scalar(f'LearningRate/group_{i}', param_group['lr'], step)
    
    def increment_step(self):
        """Increment global step counter."""
        self.global_step += 1
    
    def close(self):
        """Close the writer."""
        self.writer.close()

