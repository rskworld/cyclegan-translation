"""
Visualization Utilities for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides utilities for visualizing and saving generated images.
"""

import os
import numpy as np
from PIL import Image
import torch


def tensor2im(input_image, imtype=np.uint8):
    """
    Convert a Tensor array into a numpy image array.
    
    Args:
        input_image: Tensor image
        imtype: Type of the output image
        
    Returns:
        Numpy image array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """
    Save a numpy image to the disk.
    
    Args:
        image_numpy: Numpy image array
        image_path: Path to save the image
        aspect_ratio: Aspect ratio of the image
    """
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape
    
    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """
    Save images to the disk and create HTML visualization.
    
    Args:
        webpage: HTML webpage object
        visuals: Dictionary of images to save
        image_path: Path to save images
        aspect_ratio: Aspect ratio of images
        width: Width of images
    """
    image_dir = webpage.get_image_dir()
    short_path = os.path.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    
    webpage.add_header(name)
    ims, txts, links = [], [], []
    
    for label, im_data in visuals.items():
        im = tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)

