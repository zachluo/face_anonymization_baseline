import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks


class Pix2PixModel(BaseModel):
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

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_fr_fake_a', 'G_id', 'G_GAN_real_a_fake_a', 'G_GAN_real_a_rec_a',
                           'C_fr_real_a', 'C_fr_fake_a', 'C_fr_real_n',
                           'D_real_a_fake_a', 'D_real_a_real_n', 'D_real_a_rec_a',
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
                self.optimizer_C = torch.optim.SGD(self.netC.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
                self.optimizers.append(self.optimizer_C)

            if self.opt.lambda_rec > 0.:
                self.netR = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
                self.criterionRec = torch.nn.L1Loss().to(self.device)
                self.optimizer_G = torch.optim.SGD(list(self.netG.parameters()) + list(self.netR.parameters()),
                                                   lr=opt.lr, momentum=0.9, weight_decay=0.0005)
            else:
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
            self.optimizers.append(self.optimizer_G)

            if self.opt.lambda_id > 0.:
                self.criterionL1 = torch.nn.L1Loss().to(self.device)

            if self.opt.lambda_GAN:
                self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
                self.optimizer_D = torch.optim.SGD(self.netD.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
                self.optimizers.append(self.optimizer_D)

            self.loss_D_real_a_fake_a = 0
            self.loss_D_real_a_real_n = 0
            self.loss_D_real_a_rec_a = 0
            self.loss_D_gradient_penalty = 0
            self.loss_G_fr_fake_a = 0
            self.loss_G_id = 0
            self.loss_G_GAN_real_a_fake_a = 0
            self.loss_G_GAN_real_a_rec_a = 0
            self.loss_G_rec = 0
            self.loss_C_fr_real_a = 0
            self.loss_C_fr_fake_a = 0
            self.loss_C_fr_real_n = 0


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


    def backward_C(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        logit_real_a_aligned = self.netC(self.real_a_aligned)
        logit_fake_a_aligned = self.netC(self.fake_a_aligned.detach())
        logit_real_n_aligned = self.netC(self.real_n_aligned)

        self.loss_C_fr_real_a = self.criterionCLS(logit_real_a_aligned, self.label_a) * self.opt.lambda_fr
        self.loss_C_fr_fake_a = self.criterionCLS(logit_fake_a_aligned, self.label_a) * self.opt.lambda_fr
        self.loss_C_fr_real_n = self.criterionCLS(logit_real_n_aligned, self.label_n) * self.opt.lambda_fr
        self.loss_C = self.loss_C_fr_real_a + self.loss_C_fr_fake_a + self.loss_C_fr_real_n

        self.loss_C.backward()

    def backward_D(self):
        real_a_fake_a = torch.cat((self.real_a, self.fake_a.detach()), 1)
        logit_real_a_fake_a = self.netD(real_a_fake_a)
        real_a_real_n = torch.cat((self.real_a, self.real_n), 1)
        logit_real_a_real_n = self.netD(real_a_real_n)
        if self.opt.lambda_rec > 0:
            real_a_rec_a = torch.cat((self.real_a, self.rec_a.detach()), 1)
            logit_real_a_rec_a = self.netD(real_a_rec_a)
            self.loss_D_real_a_rec_a = self.criterionGAN(logit_real_a_rec_a, False) * self.opt.lambda_GAN
        self.loss_D_real_a_fake_a = self.criterionGAN(logit_real_a_fake_a, False) * self.opt.lambda_GAN
        self.loss_D_real_a_real_n = self.criterionGAN(logit_real_a_real_n, True) * self.opt.lambda_GAN
        if self.opt.gan_mode == 'wgangp':
            self.loss_D_gradient_penalty, _ = networks.cal_gradient_penalty(self.netD, real_a_real_n.detach(), real_a_fake_a.detach(),
                                                                        self.device, lambda_gp=self.opt.lambda_GAN)

        self.loss_D = self.loss_D_real_a_fake_a + self.loss_D_real_a_real_n + self.loss_D_real_a_rec_a + self.loss_D_gradient_penalty
        self.loss_D.backward(retain_graph=True)

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        if self.opt.lambda_fr > 0:
            logit_fake_a_aligned = self.netC(self.fake_a_aligned)
            self.loss_G_fr_fake_a = (15 - self.criterionCLS(logit_fake_a_aligned, self.label_a)) * self.opt.lambda_fr
        if self.opt.lambda_id > 0:
            self.loss_G_id = self.opt.lambda_id * self.criterionL1(self.fake_a, self.real_a)
        if self.opt.lambda_GAN > 0:
            real_a_fake_a = torch.cat((self.real_a, self.fake_a), 1)
            logit_real_a_fake_a = self.netD(real_a_fake_a)
            self.loss_G_GAN_real_a_fake_a = self.criterionGAN(logit_real_a_fake_a, True) * self.opt.lambda_GAN
            if self.opt.lambda_rec > 0:
                real_a_rec_a = torch.cat((self.real_a, self.rec_a), 1)
                logit_real_a_rec_a = self.netD(real_a_rec_a)
                self.loss_G_GAN_real_a_rec_a = self.criterionGAN(logit_real_a_rec_a, True) * self.opt.lambda_GAN
        if self.opt.lambda_rec > 0:
            self.loss_G_rec = self.opt.lambda_rec * self.criterionRec(self.real_a, self.rec_a)

        self.loss_G = self.loss_G_fr_fake_a + self.loss_G_id + self.loss_G_GAN_real_a_fake_a + self.loss_G_GAN_real_a_rec_a + self.loss_G_rec

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        if self.opt.lambda_fr:
            # update C
            self.set_requires_grad(self.netC, True)  # enable backprop for D
            self.optimizer_C.zero_grad()     # set C's gradients to zero
            self.backward_C()                # calculate gradients for D
            self.optimizer_C.step()          # update D's weights

        if self.opt.lambda_GAN:
            # update D on M
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()
            self.optimizer_D.step()

        # update G and R
        if self.opt.lambda_fr:
            self.set_requires_grad(self.netC, False)  # D requires no gradients when optimizing G
        if self.opt.lambda_GAN:
            self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
