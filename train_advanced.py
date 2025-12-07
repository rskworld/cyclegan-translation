"""
Advanced Training Script for CycleGAN with Mixed Precision and Distributed Training
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This script provides advanced training features including:
- Mixed precision training
- Distributed training support
- TensorBoard logging
- Advanced metrics tracking
"""

import time
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from config import get_config
from data import UnalignedDataset
from models import CycleGANModel
from torch.utils.data import DataLoader, DistributedSampler
from logger import TensorBoardLogger, FileLogger
from metrics import MetricsCalculator
from tools.training_utils import TrainingUtils
import argparse


def setup_distributed(rank, world_size):
    """
    Setup distributed training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def train_advanced(opt, use_mixed_precision=False, use_distributed=False, rank=0, world_size=1):
    """
    Advanced training function with mixed precision and distributed support.
    
    Args:
        opt: Configuration object
        use_mixed_precision: Whether to use mixed precision training
        use_distributed: Whether to use distributed training
        rank: Process rank (for distributed training)
        world_size: Total number of processes
    """
    # Setup device
    if use_distributed:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize loggers
    if rank == 0:
        tb_logger = TensorBoardLogger(log_dir='./logs', experiment_name=opt.name)
        file_logger = FileLogger(log_dir='./logs', experiment_name=opt.name)
        file_logger.info(f"Starting training: {opt.name}")
        file_logger.info(f"Mixed precision: {use_mixed_precision}")
        file_logger.info(f"Distributed: {use_distributed}")
    
    # Create dataset
    dataset = UnalignedDataset(opt)
    
    # Create sampler for distributed training
    if use_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        shuffle = False
    else:
        sampler = None
        shuffle = not opt.serial_batches
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=opt.num_threads,
        pin_memory=True
    )
    
    # Create model
    opt.isTrain = True
    model = CycleGANModel(opt)
    model.setup(opt)
    
    # Move model to device
    if use_distributed:
        model = DDP(model, device_ids=[rank])
    
    # Mixed precision scaler
    scaler = GradScaler() if use_mixed_precision else None
    
    # Training loop
    total_iters = 0
    best_loss = float('inf')
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        if use_distributed:
            sampler.set_epoch(epoch)
        
        epoch_start_time = time.time()
        epoch_losses = {}
        
        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            
            # Move data to device
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)
            
            model.set_input(data)
            
            # Forward and backward with mixed precision
            if use_mixed_precision:
                with autocast():
                    model.optimize_parameters()
                scaler.step(model.optimizer_G)
                scaler.step(model.optimizer_D)
                scaler.update()
            else:
                model.optimize_parameters()
            
            total_iters += opt.batch_size
            
            # Logging
            if rank == 0 and total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                
                # Update epoch losses
                for k, v in losses.items():
                    if k not in epoch_losses:
                        epoch_losses[k] = []
                    epoch_losses[k].append(v)
                
                # Log to TensorBoard
                tb_logger.log_losses(losses, total_iters)
                tb_logger.increment_step()
                
                # Print to console
                file_logger.info(f'Epoch {epoch}, Iter {total_iters}, Time: {t_comp:.3f}s')
                for k, v in losses.items():
                    file_logger.info(f'  {k}: {v:.4f}')
            
            # Save checkpoints
            if rank == 0 and total_iters % opt.save_latest_freq == 0:
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
        
        # End of epoch
        if rank == 0:
            # Calculate average losses
            avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
            epoch_time = time.time() - epoch_start_time
            
            file_logger.log_epoch(epoch, avg_losses, epoch_time)
            
            # Save epoch checkpoint
            if epoch % opt.save_epoch_freq == 0:
                model.save_networks('latest')
                model.save_networks(epoch)
            
            # Update learning rate
            model.update_learning_rate()
            
            # Log learning rate
            if hasattr(model, 'optimizer_G'):
                lr = TrainingUtils.get_learning_rate(model.optimizer_G)
                tb_logger.log_scalar('LearningRate/G', lr, epoch)
    
    # Cleanup
    if rank == 0:
        tb_logger.close()
        file_logger.info("Training completed!")
    
    if use_distributed:
        cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--rank', type=int, default=0, help='Process rank for distributed training')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for distributed training')
    args, remaining = parser.parse_known_args()
    
    # Get config
    sys.argv = [sys.argv[0]] + remaining
    opt = get_config()
    opt.isTrain = True
    opt.checkpoints_dir = './checkpoints'
    opt.pool_size = 50
    opt.serial_batches = False
    opt.load_iter = 0
    
    # Train
    train_advanced(
        opt,
        use_mixed_precision=args.mixed_precision,
        use_distributed=args.distributed,
        rank=args.rank,
        world_size=args.world_size
    )

