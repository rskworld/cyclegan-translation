"""
Example Usage Script for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script demonstrates how to use the CycleGAN model for image translation.
"""

import torch
from PIL import Image
from torchvision import transforms
from models.networks import define_G
import os


def preprocess_image(image_path, size=256):
    """
    Preprocess an image for CycleGAN.
    
    Args:
        image_path: Path to the image
        size: Target size for the image
        
    Returns:
        Preprocessed tensor
    """
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor


def tensor_to_image(tensor):
    """
    Convert a tensor to a PIL Image.
    
    Args:
        tensor: Input tensor
        
    Returns:
        PIL Image
    """
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    to_pil = transforms.ToPILImage()
    return to_pil(tensor)


def translate_image(image_path, model_path, direction='AtoB', output_path=None):
    """
    Translate an image using a trained CycleGAN model.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained generator model
        direction: Translation direction ('AtoB' or 'BtoA')
        output_path: Path to save the translated image
        
    Returns:
        Translated image as PIL Image
    """
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load generator
    if direction == 'AtoB':
        netG = define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, 'normal', 0.02, [0] if torch.cuda.is_available() else [])
    else:
        netG = define_G(3, 3, 64, 'resnet_9blocks', 'instance', False, 'normal', 0.02, [0] if torch.cuda.is_available() else [])
    
    # Load weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(netG, torch.nn.DataParallel):
            netG = netG.module
        netG.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    
    netG.to(device)
    netG.eval()
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Translate
    with torch.no_grad():
        translated = netG(image_tensor)
    
    # Convert back to image
    result_image = tensor_to_image(translated)
    
    # Save if output path is provided
    if output_path:
        result_image.save(output_path)
        print(f"Translated image saved to {output_path}")
    
    return result_image


if __name__ == '__main__':
    print("CycleGAN Image Translation Example")
    print("=" * 50)
    print("\nAuthor: RSK World")
    print("Website: https://rskworld.in")
    print("Email: help@rskworld.in")
    print("Phone: +91 93305 39277\n")
    
    # Example usage
    print("Example usage:")
    print("1. Train your model first:")
    print("   python train.py --dataroot ./datasets/your_dataset --name experiment_name")
    print("\n2. Test the model:")
    print("   python test.py --dataroot ./datasets/your_dataset --name experiment_name")
    print("\n3. Use this script to translate a single image:")
    print("   python example_usage.py")
    print("\nNote: Make sure to update the paths in this script for your specific use case.")

