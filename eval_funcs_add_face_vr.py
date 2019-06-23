from torchvision.utils import make_grid, save_image
seed = 2019
# import
# system libraries
import os, sys, glob
import os.path as osp
import time
import shutil
import numpy as np
np.random.seed(seed)
from PIL import Image
import gc
from collections import OrderedDict
# import GPUtil
import pickle
import tqdm

import torch
import torchvision.transforms as transforms
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# libraries within this package
from cmd_args import parse_args
from utils.tools import *
from utils.visualizer import Visualizer
from utils.util import print_param_info, save_model
import datasets
import models

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
from verification import KFold, eval_acc, find_best_threshold
from sklearn import metrics

FID_DIMS = 2048
FID_DIMS_VGGFACE = 2048

# FUNCTIONS, ignore

def alignment(src_pts):
    # TODO: different processing in Jason's code, here, and the paper (paper concatenate multi-scale cropping)

    ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
               [48.0252, 71.7366], [33.5493, 92.3655], [62.7299, 92.2041]]
    # crop_size = (96, 112)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    s = s / 125. - 1.
    r[:, 0] = r[:, 0] / 48. - 1
    r[:, 1] = r[:, 1] / 56. - 1

    all_tfms = np.empty((s.shape[0], 2, 3), dtype=np.float32)
    for idx in range(s.shape[0]):
        all_tfms[idx, :, :] = models.get_similarity_transform_for_cv2(r, s[idx, ...])
    all_tfms = torch.from_numpy(all_tfms).to(torch.device('cuda:0'))
    return all_tfms

def unsigned_long_to_binary_repr(unsigned_long, passwd_length):
    batch_size = unsigned_long.shape[0]
    target_size = passwd_length // 4

    binary = np.empty((batch_size, passwd_length), dtype=np.float32)
    for idx in range(batch_size):
        binary[idx, :] = np.array([int(item) for item in bin(unsigned_long[idx])[2:].zfill(passwd_length)])

    dis_target = np.empty((batch_size, target_size), dtype=np.long)
    for idx in range(batch_size):
        tmp = unsigned_long[idx]
        for byte_idx in range(target_size):
            dis_target[idx, target_size - 1 - byte_idx] = tmp % 16
            tmp //= 16
    return binary, dis_target

def generate_code(passwd_length, batch_size, device, inv):
    unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
    binary, dis_target = unsigned_long_to_binary_repr(unsigned_long, args.passwd_length)
    z = torch.from_numpy(binary).to(device)
    dis_target = torch.from_numpy(dis_target).to(device)

    repeated = True
    while repeated:
        rand_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
        repeated = np.any(unsigned_long - rand_unsigned_long == 0)
    rand_binary, rand_dis_target = unsigned_long_to_binary_repr(rand_unsigned_long, args.passwd_length)
    rand_z = torch.from_numpy(rand_binary).to(device)
    rand_dis_target = torch.from_numpy(rand_dis_target).to(device)

    if not inv:
        if args.use_minus_one:
            z = (z - 0.5) * 2
            rand_z = (rand_z - 0.5) * 2

        return z, dis_target, rand_z, rand_dis_target
    else:
        inv_unsigned_long = 2 ** args.passwd_length - 1 - unsigned_long
        inv_binary, inv_dis_target = unsigned_long_to_binary_repr(inv_unsigned_long, args.passwd_length)

        inv_z = torch.from_numpy(inv_binary).to(device)
        inv_dis_target = torch.from_numpy(inv_dis_target).to(device)

        repeated = True
        while repeated:
            another_rand_unsigned_long = np.random.randint(0, 2 ** passwd_length, size=(batch_size,), dtype=np.uint64)
            repeated = np.any(inv_unsigned_long - another_rand_unsigned_long == 0)
        another_rand_binary, another_rand_dis_target = unsigned_long_to_binary_repr(another_rand_unsigned_long,
                                                                                    args.passwd_length)
        another_rand_z = torch.from_numpy(another_rand_binary).to(device)
        another_rand_dis_target = torch.from_numpy(another_rand_dis_target).to(device)

        if args.use_minus_one:
            z = (z - 0.5) * 2
            rand_z = (rand_z - 0.5) * 2
            inv_z = z * -1.
            another_rand_z = (another_rand_z - 0.5) * 2

        return z, dis_target, rand_z, rand_dis_target, \
               inv_z, inv_dis_target, another_rand_z, another_rand_dis_target




TRAIN_MODE = True

# main
args = parse_args(sys.argv[1])
args.old_ckpt_dir = osp.dirname(sys.argv[1])
print('args.old_ckpt_dir', args.old_ckpt_dir)
args.evaluate = True

args.resume = osp.join(args.old_ckpt_dir, args.ckpt_name)

if not '_' in args.ckpt_name:
    list_of_files = glob.glob(osp.join(args.old_ckpt_dir, 'checkpoint_*'))
    if len(list_of_files) == 0:
        args.ckpt_epoch = 'last'
        args.ckpt_iter = 'iter_last'
    else:
        args.ckpt_name = osp.basename(max(list_of_files, key=os.path.getctime))
        print('found args.ckpt_name', args.ckpt_name)
        args.ckpt_epoch, args.ckpt_iter = args.ckpt_name.split('.')[0].split('_')[1:]
else:
    args.ckpt_epoch, args.ckpt_iter = args.ckpt_name.split('.')[0].split('_')[1:]

args.name = 'test_epoch_' + args.ckpt_epoch + '_' + args.ckpt_iter + '_' + args.name
# CHANGE ckpt_dir to new one!
args.ckpt_dir = osp.join(args.old_ckpt_dir, args.name)
args.display_id = -1

args.gpu_ids = list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
args.device = torch.device('cuda:0') if args.gpu_ids else torch.device('cpu')
args.batch_size = args.batch_size // 4 * len(args.gpu_ids)

os.makedirs(args.ckpt_dir, mode=0o777, exist_ok=True)
visualizer = Visualizer(args)
# visualizer.logger.log('sys.argv:\n' + ' '.join(sys.argv))
for arg in sorted(vars(args)):
    visualizer.logger.log('{:20s} {}'.format(arg, getattr(args, arg)))
visualizer.logger.log('')

# -------------------- code copy --------------------
# copy config yaml
shutil.copyfile(sys.argv[1], osp.join(args.ckpt_dir, osp.basename(sys.argv[1])))

# repo_basename = osp.basename(osp.dirname(osp.abspath('.')))
# repo_path = osp.join(args.ckpt_dir, repo_basename)
# os.makedirs(repo_path, mode=0o777, exist_ok=True)
#
# walk_res = os.walk('.')
# useful_paths = [path for path in walk_res if
#                 '.git' not in path[0] and
#                 'checkpoints' not in path[0] and
#                 'configs' not in path[0] and
#                 '__pycache__' not in path[0] and
#                 'tee_dir' not in path[0] and
#                 'tmp' not in path[0]]
# # print('useful_paths', useful_paths)
# for p in useful_paths:
#     for item in p[-1]:
#         if not (item.endswith('.py') or item.endswith('.c') or item.endswith('.h') or item.endswith('.md')):
#             continue
#         old_path = osp.join(p[0], item)
#         new_path = osp.join(repo_path, p[0][2:], item)
#         basedir = osp.dirname(new_path)
#         os.makedirs(basedir, mode=0o777, exist_ok=True)
#         shutil.copyfile(old_path, new_path)
shutil.copyfile(args.resume, osp.join(args.ckpt_dir, 'model_used.pth.tar'))

# -------------------- dataset & loader --------------------
test_dataset = datasets.__dict__[args.dataset](
    train=False,
    transform=transforms.Compose([
        transforms.Resize(args.imageSize, Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ]),
    args=args
)

visualizer.logger.log('test_dataset: ' + str(test_dataset))

# TODO: modify here for the visualization
if len(test_dataset) < 100:
    visualizer.logger.log('test img paths:')
    for anno in test_dataset.raw_annotations:
        visualizer.logger.log('%s %d %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f' % (
        anno[0], anno[1], anno[2], anno[3], anno[4], anno[5], anno[6], anno[7], anno[8], anno[9], anno[10], anno[11]))
    visualizer.logger.log('')

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
    worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
)

if not args.evaluate:
    args.evaluate = True
    _test_dataset = datasets.__dict__[args.dataset](
        train=False,
        transform=transforms.Compose([
            transforms.Resize(args.imageSize, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ]),
        args=args
    )
    args.evaluate = False
    _test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )
else:
    _test_dataset = test_dataset
    _test_loader = test_loader


model_dict = {}

if 'with_noise' in args.which_model_netG or args.lambda_dis == 0.:
    G_input_nc = args.input_nc
else:
    G_input_nc = args.input_nc + args.passwd_length
model_dict['G'] = models.define_G(G_input_nc, args.output_nc,
                                  args.ngf, args.which_model_netG, args.n_downsample_G,
                                  args.norm, not args.no_dropout,
                                  args.init_type, args.init_gain,
                                  args.gpu_ids,
                                  args.passwd_length,
                                  use_leaky=args.use_leakyG,
                                  use_resize_conv=args.use_resize_conv)
model_dict['G_nets'] = [model_dict['G']]

if args.resume:
    if osp.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        args.start_epoch = checkpoint['epoch'] + 1

        name = 'G'
        net = model_dict[name]
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        net.load_state_dict(checkpoint['state_dict_' + name])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
    gc.collect()
    torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = True

if TRAIN_MODE:
    model_dict['G'].train()
else:
    model_dict['G'].eval()

vggface2_model = resnet50(num_classes=8631, include_top=False)
with open('pretrained_models/resnet50_ft_weight.pkl', 'rb') as f:
    obj = f.read()
weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
vggface2_model.load_state_dict(weights)
vggface2_model.to(args.gpu_ids[0])
vggface2_model = torch.nn.DataParallel(vggface2_model, args.gpu_ids)
vggface2_model.eval()


block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[FID_DIMS]
fid_model = InceptionV3([block_idx], normalize_input=False)
fid_model.to(args.gpu_ids[0])
fid_mode = torch.nn.DataParallel(fid_model, args.gpu_ids)
fid_model.eval()


lpips_model = dm.DistModel()
lpips_model.initialize(model='net-lin',net='alex',use_gpu=True)


# # one hot code, not used yet
# a = np.array([1, 2, 4, 8, 16, 32, 64, 128]).astype(np.uint64)
# z_list = []
# for i in range(8):
#     rolled = np.roll(a, i)
#
#     one_hot_passwords = np.tile(rolled, 2)
#     binary_passwords, dis_target = unsigned_long_to_binary_repr(one_hot_passwords, args.passwd_length)
#     z = torch.from_numpy(binary_passwords).to(args.device)
#     z_list.append(z)


num_test_imgs = len(test_dataset) // args.batch_size * args.batch_size
_num_test_imgs = len(_test_dataset) // args.batch_size * args.batch_size
visualizer.logger.log('# test imgs ' + str(num_test_imgs))

real_pred_arr = np.empty((num_test_imgs, FID_DIMS))
fake_pred_arr = np.empty((num_test_imgs, FID_DIMS))
recon_pred_arr = np.empty((num_test_imgs, FID_DIMS))
wrong_pred_arr = np.empty((num_test_imgs, FID_DIMS))

real_pred_arr_aligned = np.empty((num_test_imgs, FID_DIMS))
fake_pred_arr_aligned = np.empty((num_test_imgs, FID_DIMS))
recon_pred_arr_aligned = np.empty((num_test_imgs, FID_DIMS))
wrong_pred_arr_aligned = np.empty((num_test_imgs, FID_DIMS))

real_pred_arr_vggface = np.empty((num_test_imgs, FID_DIMS_VGGFACE))
fake_pred_arr_vggface = np.empty((num_test_imgs, FID_DIMS_VGGFACE))
recon_pred_arr_vggface = np.empty((num_test_imgs, FID_DIMS_VGGFACE))
wrong_pred_arr_vggface = np.empty((num_test_imgs, FID_DIMS_VGGFACE))

real_positive_arr = np.empty((_num_test_imgs))
real_negative_arr = np.empty((_num_test_imgs))
real_fake_arr = np.empty((num_test_imgs))
real_recon_arr = np.empty((num_test_imgs))
real_wrong_recon_arr = np.empty((num_test_imgs))

criterion_L1 = nn.L1Loss().cuda()
criterion_L2 = nn.MSELoss().cuda()

LPIPS = AverageMeter()
L1 = AverageMeter()
L2 = AverageMeter()
DSSIM = AverageMeter()


def tensor2im(image_tensor, cent=1., factor=255./2.):
# def tensor2im(image_tensor, imtype=np.uint8, cent=1., factor=1.):
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1)) + cent) * factor
    return image_numpy

with torch.no_grad():
    for i, (img, label, landmarks, img_path, a_img, p_img, n_img) in tqdm.tqdm(enumerate(_test_loader)):
        if img.size(0) != args.batch_size:
            continue

        img_cuda = img.cuda()
        a_img_cuda = a_img.cuda()
        p_img_cuda = p_img.cuda()
        n_img_cuda = n_img.cuda()

        start = i * args.batch_size
        end = start + args.batch_size

        # face verification
        real_feature = vggface2_model(a_img_cuda).squeeze()
        positive_feature = vggface2_model(p_img_cuda).squeeze()
        negative_feature = vggface2_model(n_img_cuda).squeeze()

        cosdistance_real_positive = (real_feature * positive_feature).sum(dim=1) / (real_feature.norm(dim=1) * positive_feature.norm(dim=1) + 1e-5)
        cosdistance_real_negative = (real_feature * negative_feature).sum(dim=1) / (real_feature.norm(dim=1) * negative_feature.norm(dim=1) + 1e-5)

        real_positive_arr[start:end] = cosdistance_real_positive.cpu().numpy()
        real_negative_arr[start:end] = cosdistance_real_negative.cpu().numpy()

with torch.no_grad():
    for i, (img, label, landmarks, img_path, a_img, p_img, n_img) in tqdm.tqdm(enumerate(test_loader)):
        if img.size(0) != args.batch_size:
            continue

        theta = alignment(landmarks)
        grid = torch.nn.functional.affine_grid(theta, torch.Size((args.batch_size, 3, 112, 96)))

        img_cuda = img.cuda()
        a_img_cuda = a_img.cuda()
        p_img_cuda = p_img.cuda()
        n_img_cuda = n_img.cuda()

        z, dis_target, rand_z, rand_dis_target, \
        inv_z, inv_dis_target, another_rand_z, another_rand_dis_target = generate_code(args.passwd_length,
                                                                                       args.batch_size,
                                                                                       torch.device('cuda:0'),
                                                                                       inv=True)

        fake = model_dict['G'](img, z.cpu())
        recon = model_dict['G'](fake, inv_z)
        wrong_recon = model_dict['G'](fake, rand_z)

        real_aligned = torch.nn.functional.grid_sample(img_cuda, grid)
        fake_aligned = torch.nn.functional.grid_sample(fake, grid)
        recon_aligned = torch.nn.functional.grid_sample(recon, grid)
        wrong_recon_aligned = torch.nn.functional.grid_sample(wrong_recon, grid)

        # face verification
        real_feature = vggface2_model(a_img_cuda).squeeze()
        fake_feature = vggface2_model(test_dataset.vggface_transform_GPU(fake).cuda()).squeeze()
        recon_feature = vggface2_model(test_dataset.vggface_transform_GPU(recon).cuda()).squeeze()
        wrong_recon_feature = vggface2_model(test_dataset.vggface_transform_GPU(wrong_recon).cuda()).squeeze()

        cosdistance_real_fake = (real_feature * fake_feature).sum(dim=1) / (
        real_feature.norm(dim=1) * fake_feature.norm(dim=1) + 1e-5)
        cosdistance_real_recon = (real_feature * recon_feature).sum(dim=1) / (
        real_feature.norm(dim=1) * recon_feature.norm(dim=1) + 1e-5)
        cosdistance_real_wrong_recon = (real_feature * wrong_recon_feature).sum(dim=1) / (
        real_feature.norm(dim=1) * wrong_recon_feature.norm(dim=1) + 1e-5)

        real_fake_arr[start:end] = cosdistance_real_fake.cpu().numpy()
        real_recon_arr[start:end] = cosdistance_real_recon.cpu().numpy()
        real_wrong_recon_arr[start:end] = cosdistance_real_wrong_recon.cpu().numpy()

        # FID
        start = i * args.batch_size
        end = start + args.batch_size

        real_pred = fid_model(img_cuda)[0]
        fake_pred = fid_model(fake)[0]
        recon_pred = fid_model(recon)[0]
        wrong_recon_pred = fid_model(wrong_recon)[0]

        real_pred_aligned = fid_model(real_aligned)[0]
        fake_pred_aligned = fid_model(fake_aligned)[0]
        recon_pred_aligned = fid_model(recon_aligned)[0]
        wrong_recon_pred_aligned = fid_model(wrong_recon_aligned)[0]

        real_pred_arr[start:end] = real_pred.cpu().numpy().reshape(args.batch_size, -1)
        fake_pred_arr[start:end] = fake_pred.cpu().numpy().reshape(args.batch_size, -1)
        recon_pred_arr[start:end] = recon_pred.cpu().numpy().reshape(args.batch_size, -1)
        wrong_pred_arr[start:end] = wrong_recon_pred.cpu().numpy().reshape(args.batch_size, -1)

        real_pred_arr_aligned[start:end] = real_pred_aligned.cpu().numpy().reshape(args.batch_size, -1)
        fake_pred_arr_aligned[start:end] = fake_pred_aligned.cpu().numpy().reshape(args.batch_size, -1)
        recon_pred_arr_aligned[start:end] = recon_pred_aligned.cpu().numpy().reshape(args.batch_size, -1)
        wrong_pred_arr_aligned[start:end] = wrong_recon_pred_aligned.cpu().numpy().reshape(args.batch_size, -1)

        real_pred_arr_vggface[start:end] = real_feature.cpu().numpy().reshape(args.batch_size, -1)
        fake_pred_arr_vggface[start:end] = fake_feature.cpu().numpy().reshape(args.batch_size, -1)
        recon_pred_arr_vggface[start:end] = recon_feature.cpu().numpy().reshape(args.batch_size, -1)
        wrong_pred_arr_vggface[start:end] = wrong_recon_feature.cpu().numpy().reshape(args.batch_size, -1)

        # Recon metrics: LPIPS, SSIM, L2, L1
        dist_lpips = lpips_model.forward(img_cuda, recon)
        LPIPS.update(np.mean(dist_lpips), n=img.size(0))

        L1.update(criterion_L1(img_cuda, recon).item(), n=args.batch_size)
        L2.update(criterion_L2(img_cuda, recon).item(), n=args.batch_size)

        # compare_ssim(p0, p1, data_range=range, multichannel=True)
        img_np = tensor2im(img)
        recon_np = tensor2im(recon)
        for img_idx in range(args.batch_size):
            DSSIM.update((1. - compare_ssim(img_np[img_idx, ...], recon_np[img_idx, ...], data_range=255., multichannel=True)) / 2.)


        # print('Distance: ', dist01)
        # sys.exit(1)


def _compute_statistics_from_pred_arr(pred_arr):
    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)
    return mu, sigma


mu_real, sigma_real = _compute_statistics_from_pred_arr(real_pred_arr)
mu_fake, sigma_fake = _compute_statistics_from_pred_arr(fake_pred_arr)
mu_recon, sigma_recon = _compute_statistics_from_pred_arr(recon_pred_arr)
mu_wrong, sigma_wrong = _compute_statistics_from_pred_arr(wrong_pred_arr)

mu_real_aligned, sigma_real_aligned = _compute_statistics_from_pred_arr(real_pred_arr_aligned)
mu_fake_aligned, sigma_fake_aligned = _compute_statistics_from_pred_arr(fake_pred_arr_aligned)
mu_recon_aligned, sigma_recon_aligned = _compute_statistics_from_pred_arr(recon_pred_arr_aligned)
mu_wrong_aligned, sigma_wrong_aligned = _compute_statistics_from_pred_arr(wrong_pred_arr_aligned)

mu_real_vggface, sigma_real_vggface = _compute_statistics_from_pred_arr(real_pred_arr_vggface)
mu_fake_vggface, sigma_fake_vggface = _compute_statistics_from_pred_arr(fake_pred_arr_vggface)
mu_recon_vggface, sigma_recon_vggface = _compute_statistics_from_pred_arr(recon_pred_arr_vggface)
mu_wrong_vggface, sigma_wrong_vggface = _compute_statistics_from_pred_arr(wrong_pred_arr_vggface)

fid_real_fake = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
fid_real_recon = calculate_frechet_distance(mu_real, sigma_real, mu_recon, sigma_recon)
fid_real_wrong = calculate_frechet_distance(mu_real, sigma_real, mu_wrong, sigma_wrong)

fid_real_fake_aligned = calculate_frechet_distance(mu_real_aligned, sigma_real_aligned, mu_fake_aligned, sigma_fake_aligned)
fid_real_recon_aligned = calculate_frechet_distance(mu_real_aligned, sigma_real_aligned, mu_recon_aligned, sigma_recon_aligned)
fid_real_wrong_aligned = calculate_frechet_distance(mu_real_aligned, sigma_real_aligned, mu_wrong_aligned, sigma_wrong_aligned)

fid_real_fake_vggface = calculate_frechet_distance(mu_real_vggface, sigma_real_vggface, mu_fake_vggface, sigma_fake_vggface)
fid_real_recon_vggface = calculate_frechet_distance(mu_real_vggface, sigma_real_vggface, mu_recon_vggface, sigma_recon_vggface)
fid_real_wrong_vggface = calculate_frechet_distance(mu_real_vggface, sigma_real_vggface, mu_wrong_vggface, sigma_wrong_vggface)

folds = KFold(n=num_test_imgs, n_folds=10)
thresholds = np.arange(-1.0, 1.0, 0.005)
predicts_fake = []
predicts_recon = []
predicts_wrong_recon = []
predicts_positive = []
predicts_negative = []
for i in range(num_test_imgs):
    predicts_fake.append([real_fake_arr[i], 0])
    predicts_recon.append([real_recon_arr[i], 1])
    predicts_wrong_recon.append([real_wrong_recon_arr[i], 0])
    predicts_positive.append([real_positive_arr[i], 1])
    predicts_negative.append([real_negative_arr[i], 0])

predicts_fake = np.asarray(predicts_fake)
predicts_recon = np.asarray(predicts_recon)
predicts_wrong_recon = np.asarray(predicts_wrong_recon)
predicts_positive = np.asarray(predicts_positive)
predicts_negative = np.asarray(predicts_negative)
predicts_positive_negative = np.concatenate([predicts_positive, predicts_negative], axis=0)

best_thresh = find_best_threshold(thresholds, predicts_positive_negative)
accuracy_fake = eval_acc(best_thresh, predicts_fake)
accuracy_recon = eval_acc(best_thresh, predicts_recon)
accuracy_wrong_recon = eval_acc(best_thresh, predicts_wrong_recon)

fpr_fake, tpr_fake, _ = metrics.roc_curve(np.concatenate([predicts_fake[:, 1], predicts_positive[:, 1]], axis=0),
                                          np.concatenate([predicts_fake[:, 0], predicts_positive[:, 0]], axis=0), pos_label=1)
fpr_recon, tpr_recon, _ = metrics.roc_curve(np.concatenate([predicts_recon[:, 1], predicts_negative[:, 1]], axis=0),
                                          np.concatenate([predicts_recon[:, 0], predicts_negative[:, 0]], axis=0), pos_label=1)
fpr_wrong_recon, tpr_wrong_recon, _ = metrics.roc_curve(np.concatenate([predicts_wrong_recon[:, 1], predicts_positive[:, 1]], axis=0),
                                          np.concatenate([predicts_wrong_recon[:, 0], predicts_positive[:, 0]], axis=0), pos_label=1)
auc_fake = metrics.auc(fpr_fake, tpr_fake)
auc_recon = metrics.auc(fpr_recon, tpr_recon)
auc_wrong_recon = metrics.auc(fpr_wrong_recon, tpr_wrong_recon)

visualizer.logger.log('FID Inception: real vs fake {}; real vs recon {}; real vs wrong {}'.format(fid_real_fake, fid_real_recon, fid_real_wrong))
visualizer.logger.log('FID Inception aligned: real vs fake {}; real vs recon {}; real vs wrong {}'.format(fid_real_fake_aligned, fid_real_recon_aligned, fid_real_wrong_aligned))
visualizer.logger.log('FID VGGFACE: real vs fake {}; real vs recon {}; real vs wrong {}'.format(fid_real_fake_vggface, fid_real_recon_vggface, fid_real_wrong_vggface))
visualizer.logger.log('Recon metrics: LPIPS {}; SSIM {}; L2 {}; L1 {}'.format(LPIPS.avg, DSSIM.avg, L2.avg, L1.avg))
visualizer.logger.log('Face verification accuracy: fake {}; recon {}; wrong {}'.format(accuracy_fake, accuracy_recon, accuracy_wrong_recon))
visualizer.logger.log('Face verification auc: fake {}; recon {}; wrong {}'.format(auc_fake, auc_recon, auc_wrong_recon))



# dpi = 80.0
# xpixels, ypixels = 128 * BATCH_SIZE, 128 * 8
# def show(img):
#     npimg = img.numpy()
#     plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
#     plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')




# show(make_grid((torch.cat(new_fake_list, dim=0) + 1.) / 2., nrow=BATCH_SIZE))
# save_image((torch.cat(new_fake_list, dim=0) + 1.) / 2., filename='qualitative/' + SAVE_NAME + '_fake.png',
#            nrow=BATCH_SIZE)



