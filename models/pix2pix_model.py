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
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)

        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=10.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_fake_a', 'G_l1', 'C_real_a', 'C_fake_a', 'C_real_n']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_a', 'fake_a', 'real_n', 'real_a_aligned', 'fake_a_aligned', 'real_n_aligned']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'C']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netC = networks.define_C(opt.sphere_model_path, self.gpu_ids)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            from models.net_sphere import AngleLoss
            self.criterionCLS = AngleLoss().to(self.device)
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
            self.optimizer_C = torch.optim.SGD(self.netC.parameters(), lr=opt.lr, momentum=0.9, weight_decay=0.0005)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_C)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        for key, value in input.items():
            if not isinstance(value, torch.Tensor):
                value = torch.Tensor(value)
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


    def backward_C(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        out_real_a = self.netC(self.real_a_aligned)
        out_fake_a = self.netC(self.fake_a_aligned.detach())
        out_real_n = self.netC(self.real_n_aligned)

        self.loss_C_real_a = self.criterionCLS(out_real_a, self.label_a)
        self.loss_C_fake_a = self.criterionCLS(out_fake_a, self.label_a)
        self.loss_C_real_n = self.criterionCLS(out_real_n, self.label_n)
        self.loss_C = self.loss_C_real_a + self.loss_C_fake_a + self.loss_C_real_n

        self.loss_C.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator

        out_fake_a = self.netC(self.fake_a_aligned)
        self.loss_G_fake_a = 15 - self.criterionCLS(out_fake_a, self.label_a)

        self.loss_G_l1 = self.opt.lambda_L1 * self.criterionL1(self.fake_a, self.real_a)
        self.loss_G = self.loss_G_fake_a + self.loss_G_l1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netC, True)  # enable backprop for D
        self.optimizer_C.zero_grad()     # set D's gradients to zero
        self.backward_C()                # calculate gradients for D
        self.optimizer_C.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netC, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
