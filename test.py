"""
CycleGAN Testing Script
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script tests the CycleGAN model on test images.
"""

import os
import time
from config import get_config
from data import UnalignedDataset
from models import CycleGANModel
from torch.utils.data import DataLoader
from utils.visualization import save_image, tensor2im


if __name__ == '__main__':
    # Get test options
    opt = get_config()
    opt.isTrain = False
    opt.checkpoints_dir = './checkpoints'
    opt.pool_size = 50
    opt.serial_batches = False
    opt.load_iter = 0
    
    # Create dataset
    dataset = UnalignedDataset(opt)
    print('The number of test images = %d' % len(dataset))
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=opt.num_threads)
    
    # Create model
    model = CycleGANModel(opt)
    model.setup(opt)
    model.eval()
    
    # Create results directory
    results_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    os.makedirs(results_dir, exist_ok=True)
    
    # Test loop
    print('Testing...')
    for i, data in enumerate(dataloader):
        if i >= opt.num_test:
            break
        
        model.set_input(data)
        model.test()
        
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        
        print('Processing image %s' % img_path)
        
        for label, im_data in visuals.items():
            image_name = '%s_%s.png' % (os.path.basename(img_path[0]).split('.')[0], label)
            save_path = os.path.join(results_dir, image_name)
            im = tensor2im(im_data)
            save_image(im, save_path)
    
    print('Results saved to %s' % results_dir)

