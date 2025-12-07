"""
LPIPS (Learned Perceptual Image Patch Similarity) Score Calculation
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module calculates LPIPS score for evaluating image similarity.
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("LPIPS not available. Install with: pip install lpips")


def calculate_lpips(real_images, fake_images, real_paths=None, fake_paths=None, device='cuda'):
    """
    Calculate LPIPS score between real and fake images.
    
    Args:
        real_images: Real images tensor or list of image paths
        fake_images: Generated images tensor or list of image paths
        real_paths: Paths to real images (if images are paths)
        fake_paths: Paths to fake images (if images are paths)
        device: Device to run calculations on
        
    Returns:
        Average LPIPS score
    """
    if not LPIPS_AVAILABLE:
        return None
    
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    # Load and process images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    scores = []
    
    with torch.no_grad():
        if isinstance(real_images, list) or real_paths is not None:
            # Load from paths
            real_paths_list = real_paths if real_paths is not None else real_images
            fake_paths_list = fake_paths if fake_paths is not None else fake_images
            
            for real_path, fake_path in zip(real_paths_list, fake_paths_list):
                real_img = Image.open(real_path).convert('RGB')
                fake_img = Image.open(fake_path).convert('RGB')
                
                real_tensor = transform(real_img).unsqueeze(0).to(device)
                fake_tensor = transform(fake_img).unsqueeze(0).to(device)
                
                score = lpips_model(real_tensor, fake_tensor)
                scores.append(score.item())
        else:
            # Process tensors
            batch_size = 8
            for i in range(0, len(real_images), batch_size):
                real_batch = real_images[i:i+batch_size].to(device)
                fake_batch = fake_images[i:i+batch_size].to(device)
                
                # Normalize to [-1, 1] if needed
                if real_batch.max() > 1:
                    real_batch = (real_batch / 255.0 - 0.5) / 0.5
                if fake_batch.max() > 1:
                    fake_batch = (fake_batch / 255.0 - 0.5) / 0.5
                
                score = lpips_model(real_batch, fake_batch)
                scores.extend(score.cpu().numpy().flatten().tolist())
    
    return np.mean(scores) if scores else None

