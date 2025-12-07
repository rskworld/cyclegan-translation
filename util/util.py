"""
General Utility Functions
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides general utility functions used throughout the project.
"""

import os
import torch


def mkdirs(paths):
    """
    Create directories if they don't exist.
    
    Args:
        paths: Single path or list of paths to create
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


def set_requires_grad(nets, requires_grad=False):
    """
    Set requires_grad=False for all the networks to avoid unnecessary computations.
    
    Args:
        nets: A list of networks or a single network
        requires_grad: Whether gradients are required
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

