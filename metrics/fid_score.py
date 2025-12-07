"""
FID (Fréchet Inception Distance) Score Calculation
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module calculates FID score for evaluating generated images.
"""

import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from PIL import Image
import torchvision.transforms as transforms


def calculate_fid(real_images, fake_images, real_paths=None, fake_paths=None, device='cuda'):
    """
    Calculate FID score between real and fake images.
    
    Args:
        real_images: Real images tensor or list of image paths
        fake_images: Generated images tensor or list of image paths
        real_paths: Paths to real images (if images are paths)
        fake_paths: Paths to fake images (if images are paths)
        device: Device to run calculations on
        
    Returns:
        FID score
    """
    # Load Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = torch.nn.Identity()
    inception_model.eval()
    inception_model = inception_model.to(device)
    
    # Get features
    real_features = get_inception_features(real_images, real_paths, inception_model, device)
    fake_features = get_inception_features(fake_images, fake_paths, inception_model, device)
    
    # Calculate FID
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def get_inception_features(images, paths, model, device):
    """
    Extract features from images using Inception model.
    
    Args:
        images: Images tensor or list of paths
        paths: Paths to images (if images are paths)
        model: Inception model
        device: Device to run on
        
    Returns:
        Feature vectors
    """
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    features = []
    model.eval()
    
    with torch.no_grad():
        if isinstance(images, list) or paths is not None:
            # Load from paths
            image_paths = paths if paths is not None else images
            for img_path in image_paths:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                feat = model(img_tensor)
                features.append(feat.cpu().numpy())
        else:
            # Process tensor
            batch_size = 32
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                if batch.shape[1] != 3:
                    batch = batch.repeat(1, 3, 1, 1)
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                feat = model(batch)
                features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)

