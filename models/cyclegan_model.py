"""
CycleGAN Model Implementation
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277

This module implements the CycleGAN model with cycle consistency loss.
"""

import torch
import itertools
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .base_model import BaseModel
from .networks import define_G, define_D, init_weights
try:
    from utils.image_pool import ImagePool
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.image_pool import ImagePool


class CycleGANModel(BaseModel):
    """
    CycleGAN model for unpaired image-to-image translation.
    Uses two generators (G_A2B and G_B2A) and two discriminators (D_A and D_B).
    """
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options, and rewrite default values for existing options."""
        parser.set_defaults(no_dropout=True)
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        
        return parser
    
    def __init__(self, opt):
        """
        Initialize the CycleGAN model.
        
        Args:
            opt: Configuration object
        """
        BaseModel.__init__(self, opt)
        
        # specify the training losses you want to print out
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display
        self.visual_names = ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']
        # specify the models you want to save to the disk
        self.model_names = ['G_A2B', 'G_B2A'] if self.isTrain else ['G_A2B']
        if self.isTrain:
            self.model_names.extend(['D_A', 'D_B'])
        
        # define networks
        self.netG_A2B = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                  not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B2A = define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                  not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            self.netD_A = define_D(opt.input_nc, opt.ndf, opt.netD,
                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = define_D(opt.output_nc, opt.ndf, opt.netD,
                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        
        if self.isTrain:
            if not hasattr(opt, 'pool_size'):
                opt.pool_size = 50
            if opt.lambda_identity > 0.0:
                # Only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = torch.nn.MSELoss()
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        
        Args:
            input: Dictionary containing 'A' and 'B' images and their paths
        """
        self.real_A = input['A'].to(self.device)
        self.real_B = input['B'].to(self.device)
        self.image_paths = input['A_paths']
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A2B(self.real_A)  # G_A2B(A)
        self.rec_A = self.netG_B2A(self.fake_B)   # G_B2A(G_A2B(A))
        self.fake_A = self.netG_B2A(self.real_B)  # G_B2A(B)
        self.rec_B = self.netG_A2B(self.fake_A)   # G_A2B(G_B2A(B))
    
    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator.
        
        Args:
            netD: Discriminator network
            real: Real images
            fake: Fake images from the pool
            
        Returns:
            Loss value
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
    
    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_B)
    
    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_A)
    
    def backward_G(self):
        """
        Calculate the loss for generators G_A2B and G_B2A.
        Includes adversarial loss, cycle consistency loss, and identity loss.
        """
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        # Identity loss
        if lambda_idt > 0:
            # G_A2B should be identity if real_B is fed: ||G_A2B(B) - B||
            self.idt_A = self.netG_A2B(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B2A should be identity if real_A is fed: ||G_B2A(A) - A||
            self.idt_B = self.netG_B2A(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
        
        # GAN loss D_A(G_A2B(A))
        self.loss_G_A2B = self.criterionGAN(self.netD_B(self.fake_B), torch.ones_like(self.netD_B(self.fake_B)))
        # GAN loss D_B(G_B2A(B))
        self.loss_G_B2A = self.criterionGAN(self.netD_A(self.fake_A), torch.ones_like(self.netD_A(self.fake_A)))
        # Forward cycle loss || G_B2A(G_A2B(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A2B(G_B2A(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # Combined loss and calculate gradients
        self.loss_G = self.loss_G_A2B + self.loss_G_B2A + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
    
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # G_A2B and G_B2A
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

