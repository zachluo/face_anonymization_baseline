import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

import torch
import pickle
import numpy as np
from util.tools import *
import tqdm
import torchvision
from PIL import Image

seed = 2019
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# FID
from scipy import linalg
from scipy.misc import imread
from torch.nn.functional import adaptive_avg_pool2d
from inception import InceptionV3
from fid_score import calculate_frechet_distance

# LPIPS
from PerceptualSimilarity import dist_model as dm

# SSIM
from skimage.measure import compare_ssim

# VGGFACE 2
from models.resnet import resnet50
from verification import eval_acc

# Sphere face
from models.net_sphere import sphere20a

FID_DIMS = 2048
FID_DIMS_VGGFACE = 2048

def tensor2im(image_tensor, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + cent) * factor
    return image_numpy

def _compute_statistics_from_pred_arr(pred_arr):
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma

def vggface_transform_GPU(imgs, bboxs):
    imgs = imgs.cpu().numpy()
    all_img = []
    for idx, img in enumerate(imgs):
        img = (img + 1.) * 0.5 * 255
        img = img.transpose([1, 2, 0])
        img = img[bboxs[idx, 2]:bboxs[idx, 3], bboxs[idx, 0]:bboxs[idx, 1]]
        # if img.shape[2] == 1:
        #     img = np.repeat(img, 3, axis=2)
        img = Image.fromarray(img.astype(np.uint8), 'RGB')
        img = torchvision.transforms.Resize(256)(img)
        img = torchvision.transforms.CenterCrop(224)(img)
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= np.array([91.4953, 103.8827, 131.0912])
        img = img.transpose(2, 0, 1)  # C x H x W
        all_img.append(img)
    all_img = np.asarray(all_img)
    img = torch.from_numpy(all_img).float()

    return img

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    #opt.num_threads = 1   # test code only supports num_threads = 1
    #opt.batch_size = 1    # test code only supports batch_size = 1
    #opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    #opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    #opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    print('length of dataset:', len(dataset))
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    #web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    #webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()

    if 'casia' in opt.dataset_mode:
        best_thresh = 0.395
        best_thresh_aligned = 0.28
    elif 'lfw' in opt.dataset_mode:
        best_thresh = 0.4500
        best_thresh_aligned = 0.305

    vggface2_model = resnet50(num_classes=8631, include_top=False)
    with open('pretrain_model/resnet50_ft_weight.pkl', 'rb') as f:
        obj = f.read()
    weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    vggface2_model.load_state_dict(weights)
    vggface2_model.to(opt.gpu_ids[0])
    vggface2_model = torch.nn.DataParallel(vggface2_model, opt.gpu_ids)
    vggface2_model.eval()

    netFR = sphere20a(feature=True)
    netFR.to(opt.gpu_ids[0])
    netFR = torch.nn.DataParallel(netFR, opt.gpu_ids)
    netFR.module.load_state_dict(torch.load('./pretrain_model/sphere20a_20171020.pth', map_location='cpu'))
    netFR.eval()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
    fid_model = InceptionV3([block_idx], normalize_input=False)
    fid_model.to(opt.gpu_ids[0])
    fid_mode = torch.nn.DataParallel(fid_model, opt.gpu_ids)
    fid_model.eval()

    lpips_model = dm.DistModel()
    lpips_model.initialize(model='net-lin', net='alex', use_gpu=True)

    num_imgs = len(dataset) // opt.batch_size * opt.batch_size

    real_pred_arr = np.empty((num_imgs, FID_DIMS))
    fake_pred_arr = np.empty((num_imgs, FID_DIMS))
    recon_pred_arr = np.empty((num_imgs, FID_DIMS))

    real_pred_arr_aligned = np.empty((num_imgs, FID_DIMS))
    fake_pred_arr_aligned = np.empty((num_imgs, FID_DIMS))
    recon_pred_arr_aligned = np.empty((num_imgs, FID_DIMS))

    real_fake_arr = np.empty((num_imgs))
    real_recon_arr = np.empty((num_imgs))
    real_fake_aligned_arr = np.empty((num_imgs))
    real_recon_aligned_arr = np.empty((num_imgs))

    criterion_L1 = torch.nn.L1Loss().cuda()
    criterion_L2 = torch.nn.MSELoss().cuda()

    LPIPS = AverageMeter()
    L1 = AverageMeter()
    L2 = AverageMeter()
    DSSIM = AverageMeter()

    with torch.no_grad():
        for i, data in tqdm.tqdm(enumerate(dataset)):
            start = i * opt.batch_size
            end = start + opt.batch_size
            model.set_input(data)
            model.test()
            img, theta, bbox = model.real, model.theta, model.bbox
            fake, recon = model.fake, model.rec
            if fake.shape[1] == 1:
                fake = fake.repeat(1, 3, 1, 1)
            grid = torch.nn.functional.affine_grid(theta, torch.Size((opt.batch_size, 3, 112, 96)))

            real_aligned = torch.nn.functional.grid_sample(img, grid)
            fake_aligned = torch.nn.functional.grid_sample(fake, grid)
            recon_aligned = torch.nn.functional.grid_sample(recon, grid)

            # face verification
            real_feature = vggface2_model(vggface_transform_GPU(img, bbox).cuda()).squeeze()
            fake_feature = vggface2_model(vggface_transform_GPU(fake, bbox).cuda()).squeeze()
            recon_feature = vggface2_model(vggface_transform_GPU(recon, bbox).cuda()).squeeze()

            real_aligned_feature = netFR(real_aligned)
            fake_aligned_feature = netFR(fake_aligned)
            recon_aligned_feature = netFR(recon_aligned)

            cosdistance_real_fake = (real_feature * fake_feature).sum(dim=1) / (
            real_feature.norm(dim=1) * fake_feature.norm(dim=1) + 1e-5)
            cosdistance_real_recon = (real_feature * recon_feature).sum(dim=1) / (
            real_feature.norm(dim=1) * recon_feature.norm(dim=1) + 1e-5)

            cosdistance_real_fake_aligned = (real_aligned_feature * fake_aligned_feature).sum(dim=1) / (
                real_aligned_feature.norm(dim=1) * fake_aligned_feature.norm(dim=1) + 1e-5)
            cosdistance_real_recon_aligned = (real_aligned_feature * recon_aligned_feature).sum(dim=1) / (
                real_aligned_feature.norm(dim=1) * recon_aligned_feature.norm(dim=1) + 1e-5)

            real_fake_arr[start:end] = cosdistance_real_fake.cpu().numpy()
            real_recon_arr[start:end] = cosdistance_real_recon.cpu().numpy()

            real_fake_aligned_arr[start:end] = cosdistance_real_fake_aligned.cpu().numpy()
            real_recon_aligned_arr[start:end] = cosdistance_real_recon_aligned.cpu().numpy()

            # FID

            real_pred = fid_model(img)[0]
            fake_pred = fid_model(fake)[0]
            recon_pred = fid_model(recon)[0]

            real_pred_aligned = fid_model(real_aligned)[0]
            fake_pred_aligned = fid_model(fake_aligned)[0]
            recon_pred_aligned = fid_model(recon_aligned)[0]

            real_pred_arr[start:end] = real_pred.cpu().numpy().reshape(opt.batch_size, -1)
            fake_pred_arr[start:end] = fake_pred.cpu().numpy().reshape(opt.batch_size, -1)
            recon_pred_arr[start:end] = recon_pred.cpu().numpy().reshape(opt.batch_size, -1)

            real_pred_arr_aligned[start:end] = real_pred_aligned.cpu().numpy().reshape(opt.batch_size, -1)
            fake_pred_arr_aligned[start:end] = fake_pred_aligned.cpu().numpy().reshape(opt.batch_size, -1)
            recon_pred_arr_aligned[start:end] = recon_pred_aligned.cpu().numpy().reshape(opt.batch_size, -1)

            # Recon metrics: LPIPS, SSIM, L2, L1
            dist_lpips = lpips_model.forward(img, recon)
            LPIPS.update(np.mean(dist_lpips), n=img.size(0))

            L1.update(criterion_L1(img, recon).item(), n=opt.batch_size)
            L2.update(criterion_L2(img, recon).item(), n=opt.batch_size)

            # compare_ssim(p0, p1, data_range=range, multichannel=True)
            img_np = tensor2im(img)
            recon_np = tensor2im(recon)
            for img_idx in range(opt.batch_size):
                DSSIM.update((1. - compare_ssim(img_np[img_idx, ...], recon_np[img_idx, ...], data_range=255., multichannel=True)) / 2.)

        mu_real, sigma_real = _compute_statistics_from_pred_arr(real_pred_arr)
        mu_fake, sigma_fake = _compute_statistics_from_pred_arr(fake_pred_arr)
        mu_recon, sigma_recon = _compute_statistics_from_pred_arr(recon_pred_arr)

        mu_real_aligned, sigma_real_aligned = _compute_statistics_from_pred_arr(real_pred_arr_aligned)
        mu_fake_aligned, sigma_fake_aligned = _compute_statistics_from_pred_arr(fake_pred_arr_aligned)
        mu_recon_aligned, sigma_recon_aligned = _compute_statistics_from_pred_arr(recon_pred_arr_aligned)


        with open('training_stat_FID.pickle', 'rb') as f:
            training_stat = pickle.load(f)
        mu_real, mu_real_aligned, mu_real_vggface, \
        sigma_real, sigma_real_aligned, sigma_real_vggface = \
            training_stat['mu_real'], training_stat['mu_real_aligned'], training_stat['mu_real_vggface'], \
            training_stat['sigma_real'], training_stat['sigma_real_aligned'], training_stat['sigma_real_vggface']

        fid_real_fake = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        fid_real_recon = calculate_frechet_distance(mu_real, sigma_real, mu_recon, sigma_recon)

        fid_real_fake_aligned = calculate_frechet_distance(mu_real_aligned, sigma_real_aligned, mu_fake_aligned,
                                                           sigma_fake_aligned)
        fid_real_recon_aligned = calculate_frechet_distance(mu_real_aligned, sigma_real_aligned, mu_recon_aligned,
                                                            sigma_recon_aligned)


        predicts_fake = []
        predicts_recon = []
        predicts_fake_aligned = []
        predicts_recon_aligned = []
        for i in range(num_imgs):
            predicts_fake.append([real_fake_arr[i], 1])
            predicts_recon.append([real_recon_arr[i], 1])
            predicts_fake_aligned.append([real_fake_aligned_arr[i], 1])
            predicts_recon_aligned.append([real_recon_aligned_arr[i], 1])

        predicts_fake = np.asarray(predicts_fake)
        predicts_recon = np.asarray(predicts_recon)
        predicts_fake_aligned = np.asarray(predicts_fake_aligned)
        predicts_recon_aligned = np.asarray(predicts_recon_aligned)

        accuracy_fake = eval_acc(best_thresh, predicts_fake)
        accuracy_recon = eval_acc(best_thresh, predicts_recon)

        accuracy_fake_aligned = eval_acc(best_thresh_aligned, predicts_fake_aligned)
        accuracy_recon_aligned = eval_acc(best_thresh_aligned, predicts_recon_aligned)

        print('{} : {}'.format(opt.load_pretrain, opt.epoch))
        print('FID Inception: real vs fake {}; real vs recon {}'.format(fid_real_fake, fid_real_recon))
        print('FID Inception aligned: real vs fake {}; real vs recon {}'.format(fid_real_fake_aligned, fid_real_recon_aligned))
        print('Recon metrics: LPIPS {}; SSIM {}; L2 {}; L1 {}'.format(LPIPS.avg, DSSIM.avg, L2.avg, L1.avg))
        print('VGG Face verification accuracy: fake {}; recon {}'.format(accuracy_fake, accuracy_recon))
        print('Sphere Face verification accuracy: fake {}; recon {}'.format(accuracy_fake_aligned, accuracy_recon_aligned))
