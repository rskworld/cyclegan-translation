"""
Testing Options for CycleGAN
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module defines options specific to testing.
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """
    Testing options class.
    """
    
    def initialize(self, parser):
        """
        Define test-specific options.
        
        Args:
            parser: Argument parser
            
        Returns:
            Parser with test options added
        """
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"),
                            help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results',
                            help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0,
                            help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test',
                            help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=50,
                            help='how many test images to run')
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--eval', action='store_true',
                            help='use eval mode during test time.')
        
        self.isTrain = False
        return parser

