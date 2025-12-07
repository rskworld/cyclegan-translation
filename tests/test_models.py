"""
Unit Tests for CycleGAN Models
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import torch
import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.networks import define_G, define_D
from models.cyclegan_model import CycleGANModel


class TestModels(unittest.TestCase):
    """Test cases for CycleGAN models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 2
        self.input_size = 256
        self.input_nc = 3
        self.output_nc = 3
    
    def test_generator_forward(self):
        """Test generator forward pass."""
        netG = define_G(self.input_nc, self.output_nc, 64, 'resnet_9blocks', 'instance', 
                       False, 'normal', 0.02, [0] if torch.cuda.is_available() else [])
        netG = netG.to(self.device)
        netG.eval()
        
        x = torch.randn(self.batch_size, self.input_nc, self.input_size, self.input_size).to(self.device)
        
        with torch.no_grad():
            y = netG(x)
        
        self.assertEqual(y.shape, (self.batch_size, self.output_nc, self.input_size, self.input_size))
        self.assertTrue(torch.all(y >= -1) and torch.all(y <= 1))  # Tanh output
    
    def test_discriminator_forward(self):
        """Test discriminator forward pass."""
        netD = define_D(self.input_nc, 64, 'basic', 3, 'instance', 'normal', 0.02,
                      [0] if torch.cuda.is_available() else [])
        netD = netD.to(self.device)
        netD.eval()
        
        x = torch.randn(self.batch_size, self.input_nc, self.input_size, self.input_size).to(self.device)
        
        with torch.no_grad():
            y = netD(x)
        
        self.assertEqual(y.shape[0], self.batch_size)
        self.assertEqual(y.shape[1], 1)
    
    def test_cyclegan_model_initialization(self):
        """Test CycleGAN model initialization."""
        class MockOpt:
            def __init__(self):
                self.input_nc = 3
                self.output_nc = 3
                self.ngf = 64
                self.ndf = 64
                self.netG = 'resnet_9blocks'
                self.netD = 'basic'
                self.n_layers_D = 3
                self.norm = 'instance'
                self.init_type = 'normal'
                self.init_gain = 0.02
                self.no_dropout = True
                self.gpu_ids = [0] if torch.cuda.is_available() else []
                self.isTrain = True
                self.lambda_A = 10.0
                self.lambda_B = 10.0
                self.lambda_identity = 0.5
                self.pool_size = 50
                self.checkpoints_dir = './checkpoints'
                self.name = 'test'
                self.epoch = 'latest'
        
        opt = MockOpt()
        model = CycleGANModel(opt)
        
        self.assertIsNotNone(model.netG_A2B)
        self.assertIsNotNone(model.netG_B2A)
        self.assertIsNotNone(model.netD_A)
        self.assertIsNotNone(model.netD_B)


if __name__ == '__main__':
    unittest.main()

