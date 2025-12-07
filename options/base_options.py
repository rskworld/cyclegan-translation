"""
Base Options for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides base options that are used in both training and testing.
"""

import argparse
import os
import torch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util import util
import models


class BaseOptions():
    """
    Base class for defining options used in both training and testing.
    """
    
    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.initialized = False
    
    def initialize(self, parser):
        """
        Define common options that are used in both training and test.
        
        Args:
            parser: Argument parser
            
        Returns:
            Parser with added options
        """
        # Basic parameters
        parser.add_argument('--dataroot', type=str, default='./datasets',
                            help='path to dataset')
        parser.add_argument('--name', type=str, default='cyclegan_experiment',
                            help='name of the experiment')
        parser.add_argument('--gpu_ids', type=str, default='0',
                            help='gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='models are saved here')
        
        # Model parameters
        parser.add_argument('--model', type=str, default='cycle_gan',
                            help='chooses which model to use')
        parser.add_argument('--input_nc', type=int, default=3,
                            help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3,
                            help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64,
                            help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64,
                            help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            help='specify discriminator architecture [basic | n_layers | pixel]')
        parser.add_argument('--netG', type=str, default='resnet_9blocks',
                            help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3,
                            help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true',
                            help='no dropout for the generator')
        
        # Dataset parameters
        parser.add_argument('--direction', type=str, default='AtoB',
                            help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', type=int, default=4,
                            help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1,
                            help='input batch size')
        parser.add_argument('--load_size', type=int, default=286,
                            help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256,
                            help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        
        # Additional parameters
        parser.add_argument('--verbose', action='store_true',
                            help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        
        self.initialized = True
        return parser
    
    def gather_options(self):
        """
        Gather all options.
        
        Returns:
            Parser with all options
        """
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        # Get the basic options
        opt, _ = parser.parse_known_args()
        
        # Modify model-related parser options
        model_name = opt.model
        model_option_setter = getattr(models, 'get_option_setter')
        parser = model_option_setter(parser, self.isTrain)
        
        self.parser = parser
        return parser.parse_args()
    
    def print_options(self, opt):
        """
        Print and save options.
        
        Args:
            opt: Options object
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        
        # Save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
    
    def parse(self):
        """
        Parse options.
        
        Returns:
            Options object
        """
        opt = self.gather_options()
        opt.isTrain = self.isTrain
        
        # Process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
        
        self.print_options(opt)
        
        # Set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])
        
        self.opt = opt
        return self.opt

