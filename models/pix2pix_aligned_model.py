import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool


class Pix2PixAlignedModel(BaseModel):
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
            parser.add_argument('--lambda_id', type=float, default=1, help='weight for identity loss')
            parser.add_argument('--lambda_rec', type=float, default=1, help='weight for reconstruction loss')
            parser.add_argument('--lambda_fr', type=float, default=1, help='weight for face recognition loss')
            parser.add_argument('--lambda_GAN', type=float, default=1, help='weight of GAN loss in the fake face or reconstruction face')
            parser.add_argument('--fr_level', type=str, help='which classifier level in the sphere face model you use in fr loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        if self.opt.fr_level:
            self.opt.fr_level = self.opt.fr_level.split(',')
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_fr_fake_a', 'G_fr_rec_a', 'G_id', 'G_GAN_fake_a', 'G_GAN_rec_a',
                           'C_fr_real_a', 'C_fr_fake_a', 'C_fr_rec_a',
                           'D_real_a', 'D_fake_a', 'D_rec_a',
                           'G_rec']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_a', 'fake_a']
        if self.opt.lambda_rec > 0:
            self.visual_names.append('rec_a')
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G']
            if self.opt.lambda_fr > 0:
                self.model_names.append('C')
            if self.opt.lambda_GAN > 0:
                self.model_names.append('D')
            if self.opt.lambda_rec > 0:
                self.model_names.append('R')
        else:  # during test time, only load G
            self.model_names = ['G']
            if self.opt.lambda_rec > 0:
                self.model_names.append('R')
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            if self.opt.lambda_fr > 0.:
                self.netC = networks.define_C(opt.sphere_model_path, self.gpu_ids)
                from models.net_sphere import AngleLoss
                self.criterionCLS = AngleLoss().to(self.device)
                self.optimizer_C = torch.optim.Adam(self.netC.parameters(), lr=opt.lr * 0.1, betas=(self.opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_C)

            if self.opt.lambda_rec > 0.:
                self.netR = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                     not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                self.criterionRec = torch.nn.L1Loss().to(self.device)
                self.optimizer_G = torch.optim.Adam(list(self.netG.parameters()) + list(self.netR.parameters()),
                                                  lr=opt.lr, betas=(self.opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(self.opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)

            if self.opt.lambda_id > 0.:
                self.criterionL1 = torch.nn.L1Loss().to(self.device)

            if self.opt.lambda_GAN:
                self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, None, None,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(self.opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

                self.fake_pool = ImagePool(self.opt.pool_size)
                self.rec_pool = ImagePool(self.opt.pool_size)

            self.loss_D_real_a = 0
            self.loss_D_fake_a = 0
            self.loss_D_rec_a = 0
            self.loss_G_fr_fake_a = 0
            self.loss_G_fr_rec_a = 0
            self.loss_G_id = 0
            self.loss_G_GAN_fake_a = 0
            self.loss_G_GAN_rec_a = 0
            self.loss_G_rec = 0
            self.loss_C_fr_real_a = 0
            self.loss_C_fr_fake_a = 0
            self.loss_C_fr_rec_a = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        for key, value in input.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            setattr(self, key, value.to(self.device))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_a = self.netG(self.real_a)  # G(A)
        self.theta_a = self.theta_a.view(-1, 2, 3)
        self.theta_n = self.theta_n.view(-1, 2, 3)
        grid_a = F.affine_grid(self.theta_a, torch.Size((self.theta_a.shape[0], 3, 112, 96)))
        grid_n = F.affine_grid(self.theta_n, torch.Size((self.theta_n.shape[0], 3, 112, 96)))
        self.real_a_aligned = F.grid_sample(self.real_a, grid_a)
        self.fake_a_aligned = F.grid_sample(self.fake_a, grid_a)
        self.real_n_aligned = F.grid_sample(self.real_n, grid_n)
        if self.opt.lambda_rec > 0:
            self.rec_a = self.netR(self.fake_a)
            self.rec_a_aligned = F.grid_sample(self.rec_a, grid_a)


    def backward_C(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        logit_real_a_aligned = self.netC(self.real_a_aligned, self.opt.fr_level)
        logit_fake_a_aligned = self.netC(self.fake_a_aligned.detach(), self.opt.fr_level)
        if self.opt.lambda_rec > 0:
            logit_rec_a_aligned = self.netC(self.rec_a_aligned.detach(), self.opt.fr_level)
        self.loss_C_fr_real_a = self.loss_C_fr_fake_a = self.loss_C_fr_rec_a = 0
        for key in self.opt.fr_level:
            self.loss_C_fr_real_a += self.loss_C_fr_real_a + self.criterionCLS(logit_real_a_aligned[key], self.label_a)
            self.loss_C_fr_fake_a += self.loss_C_fr_fake_a + self.criterionCLS(logit_fake_a_aligned[key], self.label_a)
            if self.opt.lambda_rec > 0:
                self.loss_C_fr_rec_a += self.loss_C_fr_rec_a - self.criterionCLS(logit_rec_a_aligned[key], self.label_a)
        loss_C = (self.loss_C_fr_real_a + self.loss_C_fr_fake_a + self.loss_C_fr_rec_a) * self.opt.lambda_fr
        if self.opt.lambda_rec > 0:
            loss_C /= 3
        else:
            loss_C /= 2
        loss_C.backward()

    def backward_D(self):
        fake_a = self.fake_pool.query(self.fake_a)
        logit_fake_a = self.netD(fake_a.detach())
        logit_real_a = self.netD(self.real_a)
        if self.opt.lambda_rec > 0:
            rec_a = self.fake_pool.query(self.rec_a)
            logit_rec_a = self.netD(rec_a.detach())
            self.loss_D_rec_a = self.criterionGAN(logit_rec_a, False) * self.opt.lambda_GAN
        self.loss_D_fake_a = self.criterionGAN(logit_fake_a, False) * self.opt.lambda_GAN
        self.loss_D_real_a = self.criterionGAN(logit_real_a, True) * self.opt.lambda_GAN

        loss_D = self.loss_D_real_a + \
                 self.loss_D_fake_a + \
                 self.loss_D_rec_a
        if self.opt.lambda_rec > 0:
            loss_D /= 3
        else:
            loss_D /= 2
        loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        if self.opt.lambda_fr > 0:
            logit_fake_a_aligned = self.netC(self.fake_a_aligned, self.opt.fr_level)
            if self.opt.lambda_rec > 0:
                logit_rec_a_aligned = self.netC(self.rec_a_aligned, self.opt.fr_level)
            self.loss_G_fr_fake_a = 0
            self.loss_G_fr_rec_a = 0
            for key in self.opt.fr_level:
                self.loss_G_fr_fake_a = self.loss_G_fr_fake_a - self.criterionCLS(logit_fake_a_aligned[key], self.label_a) * self.opt.lambda_fr
                if self.opt.lambda_rec > 0:
                    self.loss_G_fr_rec_a = self.loss_G_fr_rec_a + self.criterionCLS(logit_rec_a_aligned[key], self.label_a) * self.opt.lambda_fr
        if self.opt.lambda_id > 0:
            self.loss_G_id = self.opt.lambda_id * self.criterionL1(self.fake_a, self.real_a)
        if self.opt.lambda_GAN > 0:
            logit_fake_a = self.netD(self.fake_a)
            self.loss_G_GAN_fake_a = self.criterionGAN(logit_fake_a, True) * self.opt.lambda_GAN
            if self.opt.lambda_rec > 0:
                logit_rec_a = self.netD(self.rec_a)
                self.loss_G_GAN_rec_a = self.criterionGAN(logit_rec_a, True) * self.opt.lambda_GAN
        if self.opt.lambda_rec > 0:
            self.loss_G_rec = self.opt.lambda_rec * self.criterionRec(self.real_a, self.rec_a)

        loss_G = self.loss_G_fr_fake_a + \
                 self.loss_G_fr_rec_a + \
                 self.loss_G_id + \
                 self.loss_G_GAN_fake_a + \
                 self.loss_G_GAN_rec_a + \
                 self.loss_G_rec

        loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        if self.opt.lambda_fr:
            # update C
            self.set_requires_grad(self.netC, True)
            self.optimizer_C.zero_grad()
            self.backward_C()
            self.optimizer_C.step()

        if self.opt.lambda_GAN:
            # update D on M
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update G and R
        if self.opt.lambda_fr:
            self.set_requires_grad(self.netC, False)
        if self.opt.lambda_GAN:
            self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
