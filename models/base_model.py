"""
Base Model Class for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Base class for all models in CycleGAN.
    This class provides common functionality for model initialization,
    device management, and checkpoint saving/loading.
    """
    
    def __init__(self, opt):
        """
        Initialize the BaseModel.
        
        Args:
            opt: Configuration object with model parameters
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain if hasattr(opt, 'isTrain') else True
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = opt.checkpoints_dir if hasattr(opt, 'checkpoints_dir') else './checkpoints'
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'
    
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """
        Add new model-specific options, and rewrite default values for existing options.
        
        Args:
            parser: Original option parser
            is_train: Whether training phase or test phase
            
        Returns:
            Modified parser
        """
        return parser
    
    @abstractmethod
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        
        Args:
            input: Includes the data itself and its metadata information.
        """
        pass
    
    @abstractmethod
    def forward(self):
        """
        Run forward pass. This will be called by both functions <optimize_parameters> and <test>.
        """
        pass
    
    @abstractmethod
    def optimize_parameters(self):
        """
        Calculate losses, gradients, and update network weights; called in every training iteration
        """
        pass
    
    def setup(self, opt):
        """
        Load and print networks; create schedulers.
        
        Args:
            opt: Configuration object
        """
        if self.isTrain:
            self.schedulers = [self.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)
    
    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
    
    def test(self):
        """Forward function used in test time."""
        with torch.no_grad():
            self.forward()
            self.compute_visuals()
    
    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass
    
    def get_image_paths(self):
        """Return image paths that are used to load current data"""
        return self.image_paths
    
    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
    
    def get_scheduler(self, optimizer, opt):
        """
        Return a learning rate scheduler.
        
        Args:
            optimizer: The optimizer
            opt: Configuration object
            
        Returns:
            Learning rate scheduler
        """
        if opt.lr_policy == 'linear':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
                return lr_l
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif opt.lr_policy == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
        elif opt.lr_policy == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        elif opt.lr_policy == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler
    
    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = {}
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret
    
    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = {}
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret
    
    def save_networks(self, epoch):
        """
        Save all the networks to the disk.
        
        Args:
            epoch: Current epoch number
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)
    
    def load_networks(self, epoch):
        """
        Load all the networks from the disk.
        
        Args:
            epoch: Epoch number to load
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                net.load_state_dict(state_dict)
    
    def print_networks(self, verbose):
        """
        Print the total number of parameters in the network and (if verbose) network architecture.
        
        Args:
            verbose: If True, print network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
    
    def set_requires_grad(self, nets, requires_grad=False):
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations.
        
        Args:
            nets: A list of networks
            requires_grad: Whether gradients are required
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

