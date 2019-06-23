import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from util.image_pool import ImagePool
import numpy as np


class ConditionPix2PixNolandmarkModel(BaseModel):
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
            parser.add_argument('--margin', type=float, default=3, help='margin in hinge loss')
            parser.add_argument('--epoch_fix_global', type=int, default=30, help='number of epochs that we only train the outmost local enhancer')

        parser.add_argument('--n_downsampling_global', type=int, default=2, help='number of downsampling in global network')
        parser.add_argument('--n_blocks_global', type=int, default=6, help='number of residual blocks in global network')
        parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancer modules')
        parser.add_argument('--n_blocks_local', type=int, default=3, help='number of residual blocks in local enhancer modules')
        parser.add_argument('--celeba_id_list', type=str, help='path of the id txt of celeba dataset')
        parser.add_argument('--celeba_hq_list', type=str, help='path of image txt of celeba hq dataset')
        parser.add_argument('--fill_percent', type=float, default=1, help='path of image txt of celeba hq dataset')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_fr_fake', 'G_fr_rec', 'G_id', 'G_GAN_fake', 'G_GAN_rec', 'G_rec',
                          # 'C_fr_real', 'C_fr_fake', 'C_fr_rec',
                           'D_real', 'D_fake', 'D_rec']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real', 'fake', 'rec']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'C', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                      opt.n_downsampling_global, opt.n_blocks_global, opt.n_local_enhancers, opt.n_blocks_local)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            # define network C
            self.netC = networks.define_C(opt.sphere_model_path, opt.n_classes_C, opt.init_type, opt.init_gain, self.gpu_ids, True)
            from models.net_sphere import AngleLoss
            #self.criterionCLS = AngleLoss().to(self.device)
            #self.criterionCLS = torch.nn.L1Loss(reduction='none').to(self.device)
            self.criterionCLS = F.cosine_similarity
            # self.optimizer_C = torch.optim.Adam([p for _, p in self.netC.named_parameters('fc6')], lr=opt.lr * 0.1, betas=(self.opt.beta1, 0.999))
            # self.optimizers.append(self.optimizer_C)

            self.criterionL1 = torch.nn.L1Loss().to(self.device)

            # define network D
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type,
                                          opt.init_gain, self.gpu_ids)
            # self.netFE = networks.define_FE(opt.input_nc, opt.ndf, opt.n_layers_FE, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netD = networks.define_DAfterFE(opt.ndf, opt.n_layers_FE, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            # self.netQ = networks.define_InfoAfterFE(opt.ndf, opt.n_layers_FE, opt.n_layers_info, 2**opt.n_condition, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            # anything else
            self.fake_pool = ImagePool(self.opt.pool_size)
            self.rec_pool = ImagePool(self.opt.pool_size)

            #self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(self.opt.beta1, 0.999))
            # optimizer G
            if opt.epoch_fix_global > 0:
                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():
                    if key.startswith('module.model' + str(opt.n_local_enhancers)):
                        params += [value]
                print( '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.epoch_fix_global)
            else:
                params = self.netG.parameters()
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            # #generate condition
            # self.condition_map_fake = torch.ones([opt.batch_size, opt.n_condition, opt.load_size, opt.load_size], device=self.device)
            # self.condition_map_rec = torch.zeros([opt.batch_size, opt.n_condition, opt.load_size, opt.load_size], device=self.device)
            # self.criterionCondition = torch.nn.CrossEntropyLoss().to(self.device)
            # self.n_cases = (torch.pow(torch.tensor(2, dtype=torch.float32), self.opt.n_condition) - 1).to(self.device)

            # initialize losses
            self.loss_D_real = 0
            self.loss_D_fake = 0
            self.loss_D_rec = 0
            self.loss_G_fr_fake = 0
            self.loss_G_fr_rec = 0
            self.loss_G_id = 0
            self.loss_G_GAN_fake = 0
            self.loss_G_GAN_rec = 0
            self.loss_G_rec = 0
            # self.loss_C_fr_real = 0
            # self.loss_C_fr_fake = 0
            # self.loss_C_fr_rec = 0

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
            if not isinstance(value, torch.Tensor):
                value_tensor = torch.tensor(value)
            else:
                value_tensor = value

            setattr(self, key, value_tensor.to(self.device))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #import pdb;pdb.set_trace()
        # self.condition_map_rec_weight = self.condition_bin.view([self.opt.batch_size, self.opt.n_condition, 1, 1]).repeat([1, 1, self.opt.load_size, self.opt.load_size])
        self.fake = self.netG(self.real)
        if self.opt.fill_percent < 1:
            self.real_copy = self.real.clone()
            shape = self.real_copy.shape
            start_h = int((1 - self.opt.fill_percent) / 2 * shape[2])
            start_w = int((1 - self.opt.fill_percent) / 2 * shape[3])
            end_h = start_h + int(self.opt.fill_percent * shape[2])
            end_w = start_w + int(self.opt.fill_percent * shape[3])
            self.real_copy[:, :, start_h:end_h, start_w:end_w] = self.fake[:, :, start_h:end_h, start_w:end_w]
            self.fake = self.real_copy
        self.rec = self.netG(self.fake)
        self.real_aligned = F.interpolate(self.real, size=(112, 96), mode='bilinear')[:, [2, 1, 0]]
        self.fake_aligned = F.interpolate(self.fake, size=(112, 96), mode='bilinear')[:, [2, 1, 0]]
        self.rec_aligned = F.interpolate(self.rec, size=(112, 96), mode='bilinear')[:, [2, 1, 0]]

        # self.weight = self.condition_dec / self.n_cases

    # def backward_C(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     logit_real_aligned = self.netC(self.real_aligned)
    #     logit_fake_aligned = self.netC(self.fake_aligned.detach())
    #     logit_rec_aligned = self.netC(self.rec_aligned.detach())
    #     self.loss_C_fr_real = self.criterionCLS(logit_real_aligned, self.label)
    #     self.loss_C_fr_fake = self.criterionCLS(logit_fake_aligned, self.label)
    #     self.loss_C_fr_rec = -self.criterionCLS(logit_rec_aligned, self.label)
    #     loss_C = (self.loss_C_fr_real + self.loss_C_fr_fake + self.loss_C_fr_rec) / 3 * self.opt.lambda_fr
    #     loss_C.backward()

    def backward_D(self):
        fake = self.fake_pool.query(self.fake)
        rec = self.fake_pool.query(self.rec)
        logit_fake = self.netD(fake.detach())
        logit_real = self.netD(self.real)
        logit_rec = self.netD(rec.detach())
        self.loss_D_rec = self.criterionGAN(logit_rec, False)
        self.loss_D_fake = self.criterionGAN(logit_fake, False)
        self.loss_D_real = self.criterionGAN(logit_real, True)
        gradient_penalty = 0
        if self.opt.gan_mode == 'wgangp':
            gradient_penalty, _ = networks.cal_gradient_penalty(self.netD, self.real, fake.detach(), self.device)
            gradient_penalty.backward(retain_graph=True)
        loss_D = (self.loss_D_real + (self.loss_D_fake + self.loss_D_rec) / 2) / 2 * self.opt.lambda_GAN

        loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #import pdb; pdb.set_trace()
        # face recognition loss
        feature_real_aligned = self.netC(self.real_aligned)
        feature_fake_aligned = self.netC(self.fake_aligned)
        feature_rec_aligned = self.netC(self.rec_aligned)
        diff = self.criterionCLS(feature_fake_aligned, feature_real_aligned) - self.opt.margin
        self.loss_G_fr_fake = torch.max(diff, torch.zeros(diff.shape, device=self.device)).mean() * self.opt.lambda_fr
        self.loss_G_fr_rec = (1 - self.criterionCLS(feature_rec_aligned, feature_real_aligned).mean()) * self.opt.lambda_fr
        # identity loss
        self.loss_G_id = self.criterionL1(self.fake, self.real) * self.opt.lambda_id
        # GAN loss and condition loss
        logit_fake = self.netD(self.fake)
        logit_rec = self.netD(self.rec)
        self.loss_G_GAN_fake = self.criterionGAN(logit_fake, True) * self.opt.lambda_GAN
        self.loss_G_GAN_rec = self.criterionGAN(logit_rec, True) * self.opt.lambda_GAN
        # self.loss_G_condition = self.criterionCondition(condition_rec, self.condition_dec.type(torch.int64)) * self.opt.lambda_condition
        # reconstruction loss
        self.loss_G_rec = self.criterionL1(self.real, self.rec) * self.opt.lambda_rec
        # # reconstruction with weight loss
        # self.loss_G_rec_weight = self.criterionL1(self.real, self.rec_weight).mean([1,2,3]) * (1 - self.weight) + \
        #                         self.criterionL1(self.fake.detach(), self.rec_weight).mean([1,2,3]) * self.weight
        # self.loss_G_rec_weight = self.loss_G_rec_weight.mean() * self.opt.lambda_rec_weight

        loss_G = self.loss_G_fr_fake + \
                 self.loss_G_fr_rec + \
                 self.loss_G_id + \
                 self.loss_G_GAN_fake + \
                 self.loss_G_GAN_rec + \
                 self.loss_G_rec

        loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update C
        # self.set_requires_grad(self.netC, True)
        # self.optimizer_C.zero_grad()
        # self.backward_C()
        # self.optimizer_C.step()

        # update D
        self.set_requires_grad(self.netD, True)
        #self.set_requires_grad(self.netFE, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G and Q
        #self.set_requires_grad(self.netC, False)
        self.set_requires_grad(self.netD, False)
        #self.set_requires_grad(self.netFE, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def update_learning_param(self):
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
