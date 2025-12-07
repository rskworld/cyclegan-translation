"""
Download Pre-trained Models Script
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script downloads pre-trained CycleGAN models.
"""

import os
import urllib.request
import zipfile
import shutil


PRETRAINED_MODELS = {
    'horse2zebra': {
        'url': 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/raw/master/pretrained/horse2zebra_pretrained.zip',
        'description': 'Horse to Zebra translation model'
    },
    'apple2orange': {
        'url': 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/raw/master/pretrained/apple2orange_pretrained.zip',
        'description': 'Apple to Orange translation model'
    },
    'summer2winter': {
        'url': 'https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/raw/master/pretrained/summer2winter_pretrained.zip',
        'description': 'Summer to Winter translation model'
    }
}


def download_file(url, destination):
    """
    Download a file from URL.
    
    Args:
        url: URL to download from
        destination: Destination path
    """
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, destination)
    print(f"Downloaded to {destination}")


def extract_zip(zip_path, extract_to):
    """
    Extract zip file.
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")


def download_pretrained_model(model_name, save_dir='./pretrained'):
    """
    Download a pre-trained model.
    
    Args:
        model_name: Name of the model to download
        save_dir: Directory to save the model
    """
    if model_name not in PRETRAINED_MODELS:
        print(f"Model {model_name} not available.")
        print(f"Available models: {list(PRETRAINED_MODELS.keys())}")
        return
    
    model_info = PRETRAINED_MODELS[model_name]
    print(f"\nDownloading {model_name}: {model_info['description']}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Download zip file
    zip_path = os.path.join(save_dir, f"{model_name}.zip")
    download_file(model_info['url'], zip_path)
    
    # Extract
    extract_to = os.path.join(save_dir, model_name)
    extract_zip(zip_path, extract_to)
    
    # Remove zip file
    os.remove(zip_path)
    
    print(f"\nModel {model_name} downloaded successfully!")
    print(f"Location: {extract_to}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        download_pretrained_model(model_name)
    else:
        print("Usage: python download_pretrained.py <model_name>")
        print(f"\nAvailable models:")
        for name, info in PRETRAINED_MODELS.items():
            print(f"  - {name}: {info['description']}")

