"""
Inception Score Calculation
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module calculates Inception Score for evaluating generated images.
"""

import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def calculate_inception_score(fake_images, fake_paths=None, device='cuda', splits=10):
    """
    Calculate Inception Score for generated images.
    
    Args:
        fake_images: Generated images tensor or list of image paths
        fake_paths: Paths to fake images (if images are paths)
        device: Device to run calculations on
        splits: Number of splits for calculation
        
    Returns:
        Mean and standard deviation of IS
    """
    # Load Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model = inception_model.to(device)
    
    # Get predictions
    preds = get_inception_predictions(fake_images, fake_paths, inception_model, device)
    
    # Calculate IS
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits)]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([np.sum(p * np.log(p / py)) for p in part])))
    
    return np.mean(scores), np.std(scores)


def get_inception_predictions(images, paths, model, device):
    """
    Get predictions from Inception model.
    
    Args:
        images: Images tensor or list of paths
        paths: Paths to images (if images are paths)
        model: Inception model
        device: Device to run on
        
    Returns:
        Predictions
    """
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    preds = []
    model.eval()
    
    with torch.no_grad():
        if isinstance(images, list) or paths is not None:
            # Load from paths
            image_paths = paths if paths is not None else images
            for img_path in image_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                pred = F.softmax(model(img_tensor), dim=1)
                preds.append(pred.cpu().numpy())
        else:
            # Process tensor
            batch_size = 32
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                if batch.shape[1] != 3:
                    batch = batch.repeat(1, 3, 1, 1)
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                pred = F.softmax(model(batch), dim=1)
                preds.append(pred.cpu().numpy())
    
    return np.concatenate(preds, axis=0)

