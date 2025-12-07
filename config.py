"""
CycleGAN Configuration File
Author: RSK World
Website: https://rskworld.in
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import argparse

def get_config():
    """
    Get configuration arguments for CycleGAN training and testing.
    
    Returns:
        argparse.Namespace: Configuration arguments
    """
    parser = argparse.ArgumentParser(description='CycleGAN Image-to-Image Translation')
    
    # Dataset parameters
    parser.add_argument('--dataroot', type=str, default='./datasets', 
                        help='path to dataset')
    parser.add_argument('--name', type=str, default='cyclegan_experiment',
                        help='name of the experiment')
    parser.add_argument('--model', type=str, default='cycle_gan',
                        help='chooses which model to use')
    parser.add_argument('--direction', type=str, default='AtoB', 
                        help='AtoB or BtoA')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of threads for data loading')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='input batch size')
    parser.add_argument('--load_size', type=int, default=286,
                        help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='then crop to this size')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='Maximum number of samples allowed per dataset')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                        help='scaling and cropping of images at load time')
    parser.add_argument('--no_flip', action='store_true',
                        help='if specified, do not flip the images for data augmentation')
    parser.add_argument('--display_id', type=int, default=-1,
                        help='window id of the web display')
    
    # Model parameters
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
    
    # Training parameters
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='gpu ids: e.g. 0 0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='initial learning rate for adam')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='momentum term of adam')
    parser.add_argument('--lr_policy', type=str, default='linear',
                        help='learning rate policy. [linear | step | plateau | cosine]')
    parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count')
    parser.add_argument('--niter', type=int, default=100,
                        help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training: load the latest model')
    parser.add_argument('--epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--verbose', action='store_true',
                        help='if specified, print more debugging information')
    parser.add_argument('--suffix', default='', type=str,
                        help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
    
    # Loss parameters
    parser.add_argument('--lambda_A', type=float, default=10.0,
                        help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0,
                        help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss.')
    
    # Testing parameters
    parser.add_argument('--num_test', type=int, default=50,
                        help='how many test images to run')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='saves results here.')
    
    # Visualization parameters
    parser.add_argument('--display_freq', type=int, default=400,
                        help='frequency of showing training results on screen')
    parser.add_argument('--display_ncols', type=int, default=4,
                        help='if positive, display all images in a single visdom web panel')
    parser.add_argument('--display_winsize', type=int, default=256,
                        help='display window size for both visdom and HTML')
    parser.add_argument('--display_server', type=str, default="http://localhost",
                        help='visdom server of the web display')
    parser.add_argument('--display_env', type=str, default='main',
                        help='visdom display environment name (default is "main")')
    parser.add_argument('--display_port', type=int, default=8097,
                        help='visdom port of the web display')
    parser.add_argument('--update_html_freq', type=int, default=1000,
                        help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000,
                        help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--save_by_iter', action='store_true',
                        help='whether saves model by iteration')
    parser.add_argument('--phase', type=str, default='train',
                        help='train, val, test, etc')
    
    return parser.parse_args()

