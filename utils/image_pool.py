"""
Image Pool for CycleGAN Training
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module implements an image pool to store generated images for discriminator training.
"""

import random
import torch


class ImagePool():
    """
    Image pool to store generated images.
    This buffer stores previously generated images to update the discriminator.
    """
    
    def __init__(self, pool_size):
        """
        Initialize the image pool.
        
        Args:
            pool_size: Size of the image buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
    
    def query(self, images):
        """
        Return an image from the pool.
        If the pool is not full, add the image to the pool.
        If the pool is full, randomly replace an image with the new one.
        
        Args:
            images: Batch of images to query/add
            
        Returns:
            Images from the pool
        """
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

