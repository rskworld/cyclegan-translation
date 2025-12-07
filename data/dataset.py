"""
Dataset and Data Loading Utilities for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides data loading functionality for unpaired image datasets.
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def get_transform(opt, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    """
    Create a transform pipeline for image preprocessing.
    
    Args:
        opt: Configuration object
        grayscale: Whether to convert to grayscale
        method: Interpolation method
        convert: Whether to convert to tensor
        
    Returns:
        Transform pipeline
    """
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    
    if 'crop' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    elif 'scale_width' in opt.preprocess or 'resize' in opt.preprocess:
        transform_list.append(transforms.RandomCrop(opt.crop_size))
    
    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width, method=transforms.InterpolationMode.BICUBIC):
    """
    Scale image width while maintaining aspect ratio.
    
    Args:
        img: PIL Image
        target_width: Target width
        method: Interpolation method
        
    Returns:
        Scaled image
    """
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


class UnalignedDataset(Dataset):
    """
    Dataset class for unpaired image-to-image translation.
    Loads images from two separate directories (domain A and domain B).
    """
    
    def __init__(self, opt):
        """
        Initialize the dataset.
        
        Args:
            opt: Configuration object with dataset parameters
        """
        self.opt = opt
        phase = getattr(opt, 'phase', 'train')
        self.dir_A = os.path.join(opt.dataroot, phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, phase + 'B')
        
        self.A_paths = sorted(self.__make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(self.__make_dataset(self.dir_B, opt.max_dataset_size))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        direction = getattr(self.opt, 'direction', 'AtoB')
        self.btoA = direction == 'BtoA'
        input_nc = self.opt.output_nc if self.btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if self.btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
    
    def __make_dataset(self, dir, max_dataset_size=float("inf")):
        """
        Create a list of image paths from a directory.
        
        Args:
            dir: Directory path
            max_dataset_size: Maximum number of images to load
            
        Returns:
            List of image paths
        """
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if self.__is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images[:min(max_dataset_size, len(images))]
    
    def __is_image_file(self, filename):
        """
        Check if a file is an image.
        
        Args:
            filename: File name
            
        Returns:
            True if file is an image, False otherwise
        """
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
            '.tif', '.TIF', '.tiff', '.TIFF',
        ]
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    
    def __getitem__(self, index):
        """
        Get a data sample.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing 'A', 'B', 'A_paths', 'B_paths'
        """
        A_path = self.A_paths[index % self.A_size]
        serial_batches = getattr(self.opt, 'serial_batches', False)
        if serial_batches:
            index_B = index % self.B_size
        else:
            import random
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)

