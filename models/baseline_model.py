import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool


class BaselineModel(BaseModel):
    """ This class implements the pix2pix models, for learning a mapping from input images to output images given paired data.

    The models training requires '--dataset_mode aligned' data.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new data-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- real option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the fake parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)

        if is_train:
            parser.add_argument('--lambda_rec', type=float, default=1, help='weight for reconstruction loss')
            parser.add_argument('--lambda_GAN', type=float, default=1, help='weight of GAN loss in the fake face or reconstruction face')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_rec', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake', 'rec']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc if opt.baseline != 'edge' else 1, opt.output_nc, opt.ngf, opt.netG, opt.normG,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.criterionRec = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                              lr=opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            self.rec_pool = ImagePool(self.opt.pool_size)

            self.loss_D_real = 0
            self.loss_D_fake = 0
            self.loss_G_GAN = 0
            self.loss_G_rec = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        for key, value in input.items():
            if key == 'real_path':
                self.image_paths = value
                continue
            if 'path' in key:
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            setattr(self, key, value.to(self.device))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.rec = self.netG(self.fake)  # G(A)

    def backward_D(self):
        rec = self.rec_pool.query(self.rec)
        logit_rec = self.netD(rec.detach())
        logit_real = self.netD(self.real)
        self.loss_D_fake = self.criterionGAN(logit_rec, False) * self.opt.lambda_GAN
        self.loss_D_real = self.criterionGAN(logit_real, True) * self.opt.lambda_GAN
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, _ = networks.cal_gradient_penalty(self.netD, self.real, rec.detach(), self.device)
            gradient_penalty.backward(retain_graph=True)
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        logit_rec = self.netD(self.rec)
        self.loss_G_GAN = self.criterionGAN(logit_rec, True) * self.opt.lambda_GAN
        self.loss_G_rec = self.criterionRec(self.real, self.rec) * self.opt.lambda_rec

        loss_G = self.loss_G_GAN + self.loss_G_rec
        loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
