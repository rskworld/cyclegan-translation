"""
File Logger for CycleGAN Training
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides file-based logging functionality.
"""

import os
import logging
from datetime import datetime


class FileLogger:
    """
    File logger for CycleGAN training.
    """
    
    def __init__(self, log_dir='./logs', experiment_name='cyclegan', level=logging.INFO):
        """
        Initialize file logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            level: Logging level
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(level)
        
        # Create file handler
        log_file = os.path.join(log_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message."""
        self.logger.debug(message)
    
    def log_epoch(self, epoch, losses_dict, time_taken):
        """
        Log epoch information.
        
        Args:
            epoch: Epoch number
            losses_dict: Dictionary of losses
            time_taken: Time taken for epoch
        """
        message = f"Epoch {epoch} - Time: {time_taken:.2f}s"
        for key, value in losses_dict.items():
            message += f" | {key}: {value:.4f}"
        self.logger.info(message)

