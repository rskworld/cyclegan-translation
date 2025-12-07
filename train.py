"""
CycleGAN Training Script
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script trains the CycleGAN model for image-to-image translation.
"""

import time
import os
import sys
import argparse
from config import get_config
from data import UnalignedDataset
from models import CycleGANModel
from torch.utils.data import DataLoader
import torch


if __name__ == '__main__':
    # Get training options
    opt = get_config()
    opt.isTrain = True
    opt.checkpoints_dir = './checkpoints'
    opt.pool_size = 50
    opt.serial_batches = False
    opt.load_iter = 0
    
    # Create dataset
    dataset = UnalignedDataset(opt)
    print('The number of training images = %d' % len(dataset))
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches,
                            num_workers=opt.num_threads)
    
    # Create model
    model = CycleGANModel(opt)
    model.setup(opt)
    
    # Training loop
    total_iters = 0
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            
            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                # Display images (if using visdom)
                # model.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                print('epoch: %d, iters: %d, time: %.3f, data: %.3f' % 
                      (epoch, total_iters, t_comp, t_data))
                for k, v in losses.items():
                    print('%s: %.3f' % (k, v))
            
            if total_iters % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
            
            iter_data_time = time.time()
        
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % 
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

