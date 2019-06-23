import sys
sys.path.append('.')
from data.base_dataset import BaseDataset, get_transform_baseline, get_transform
import os
import random
seed = 2019
random.seed(seed)
import numpy as np
from PIL import Image
from util.matlab_cp2tform import get_similarity_transform_for_cv2
import torch
import torchvision

class CASIABaselineDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        bbox_path = os.path.join(self.root, 'casia_detection.txt')
        self.bbox_dict = {}
        with open(bbox_path) as fd:
            bbox_lines = [bbox_line.strip().split() for bbox_line in fd.readlines()]
        for bbox_line in bbox_lines:
            oleft = float(bbox_line[1]) / 250. * 128.
            oup = float(bbox_line[2]) / 250. * 128.
            oright = float(bbox_line[3]) / 250. * 128.
            odown = float(bbox_line[4]) / 250. * 128.

            ocenter_x = (oleft + oright) / 2.
            ocenter_y = (oup + odown) / 2.

            ohalf_width = (oright - oleft) / 2.
            ohalf_height = (odown - oup) / 2.

            ehalf_width = 1.3 * ohalf_width
            ehalf_height = 1.3 * ohalf_height

            self.bbox_dict[bbox_line[0]] = [max(int(ocenter_x - ehalf_width), 0),
                                            min(int(ocenter_x + ehalf_width), 127),
                                            max(int(ocenter_y - ehalf_height), 0),
                                            min(int(ocenter_y + ehalf_height), 127)]


        with open(os.path.join(self.root, opt.landmark_path)) as f:
            self.indexlist = [line.rstrip('\n').split() for line in f]
            for index in range(len(self.indexlist)):
                self.indexlist[index] = [self.indexlist[index][0]] + \
                                        [int(self.indexlist[index][1])] + \
                                        [float(i) for i in self.indexlist[index][2:]]
            ids = set([i[1] for i in self.indexlist])
            self.split_train = max(ids) // 5 * 4
            self.split_val = (len(ids) - self.split_train) // 2 + self.split_train
            if opt.isTrain:
                self.indexlist = [i for i in self.indexlist if i[1] < self.split_train]
                random.shuffle(self.indexlist)
            else:
                if opt.val_test == 'val':
                    self.indexlist = [i for i in self.indexlist if i[1] >= self.split_train and i[1] < self.split_val]
                elif opt.val_test == 'test':
                    self.indexlist = [i for i in self.indexlist if i[1] >= self.split_val]
                else:
                    raise NotImplementedError

            self.id_dict = {}
            for i in self.indexlist:
                if i[1] not in self.id_dict.keys():
                    self.id_dict[i[1]] = [i[0]]
                else:
                    self.id_dict[i[1]].append(i[0])

        self.transform = get_transform(opt)
        self.transform_baseline = get_transform_baseline(opt)
        self.img_dir = 'images'

    def alignment(self, src_pts):
        ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
                    [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        src_pts = np.array(src_pts).reshape(5,2)
        s = np.array(src_pts).astype(np.float32)
        r = np.array(ref_pts).astype(np.float32)
        # center normalize for spatial transformer
        r[:, 0] = r[:, 0] / 48. - 1
        r[:, 1] = r[:, 1] / 56. - 1
        tfm = get_similarity_transform_for_cv2(r, s)
        return tfm

    def sample_negative(self, a_cls):
        while True:
            rand = random.randint(0, len(self.indexlist) - 1)
            if self.indexlist[rand][1] != a_cls:
                break
        return self.indexlist[rand][0]

    def sample_positive(self, a_cls, img_name):
        while True:
            rand = random.randint(0, len(self.id_dict[a_cls]) - 1)
            if self.id_dict[a_cls][rand] != img_name or len(self.id_dict[a_cls]) == 1:
                break
        return self.id_dict[a_cls][rand]

    def vggface_transform(self, img):
        img = torchvision.transforms.Resize(256)(img)
        img = torchvision.transforms.CenterCrop(224)(img)
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= np.array([91.4953, 103.8827, 131.0912])
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()

        return img

    def load_img(self, index):
        info = self.indexlist[index]
        cls = info[1]
        landmark = [(float(k)/125. - 1) for k in info[2:]]
        tfm = self.alignment(landmark).astype(np.float32)
        img_path = os.path.join(self.root, self.img_dir, info[0])
        p_path = os.path.join(self.root, self.img_dir, self.sample_positive(cls, info[0]))
        n_path = os.path.join(self.root, self.img_dir, self.sample_negative(cls))
        img = Image.open(img_path).convert('RGB')
        p_img = Image.open(p_path).convert("RGB")
        n_img = Image.open(n_path).convert("RGB")

        a_img = self.vggface_transform(img)
        p_img = self.vggface_transform(p_img)
        n_img = self.vggface_transform(n_img)

        bbox = np.asarray(self.bbox_dict[info[0]])

        return img, cls, tfm, img_path, a_img, p_img, n_img, bbox

    def load_path(self, index):
        info = self.indexlist[index]
        cls = info[1]
        p_path = self.sample_positive(cls, info[0])
        n_path = self.sample_negative(cls)
        img_path = os.path.join(self.root, self.img_dir, info[0])
        return img_path, p_path, n_path

    def __getitem__(self, index):
        _a, a_cls, a_tfm, img_path, a_img, p_img, n_img, bbox = self.load_img(index)

        # transform images
        real_a = self.transform(_a)
        fake_a = self.transform_baseline(_a)

        return {'real': real_a,
                'real_path': img_path,
                'fake': fake_a,
                'a_img': a_img,
                'p_img': p_img,
                'n_img': n_img,
                'theta': a_tfm,
                'bbox': bbox}

    # def __getitem__(self, index):
    #     a_path, p_path, n_path = self.load_path(index)
    #
    #     return {'real_path': a_path,
    #             'p_path': p_path,
    #             'n_path': n_path}

    def __len__(self):
        return len(self.indexlist)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm
    class option(object):
        def __init__(self):
            self.landmark_path = 'casia_landmark.txt'
            self.dataroot = '/p300/dataset/casia'
            self.load_size = 128
            self.n_condition = 4
            self.baseline = 'edge'
            self.isTrain = False

    def merge_path(path, label):
        s = path.split('/')[-2:]
        basename, extend = s[1].split('.')
        s = '_'.join([s[0], basename, label + '.png'])
        return s

    opt = option()
    casia = CASIABaselineDataset(opt)
    print('{} images in the testing set'.format(len(casia)))
    with open('fake.txt', 'w') as f:
        for i in tqdm.tqdm(iter(casia)):
            s = ' '.join([merge_path(i['real_path'], 'real'), merge_path(i['p_path'], 'real'), '1']) + '\n'
            s += ' '.join([merge_path(i['real_path'], 'real'), merge_path(i['n_path'], 'real'), '0']) + '\n'
            s += ' '.join([merge_path(i['real_path'], 'real'), merge_path(i['real_path'], 'fake'), '0']) + '\n'
            f.write(s)

    with open('rec.txt', 'w') as f:
        for i in tqdm.tqdm(iter(casia)):
            s = ' '.join([merge_path(i['real_path'], 'real'), merge_path(i['n_path'], 'real'), '0']) + '\n'
            s += ' '.join([merge_path(i['real_path'], 'real'), merge_path(i['p_path'], 'real'), '1']) + '\n'
            s += ' '.join([merge_path(i['real_path'], 'real'), merge_path(i['real_path'], 'rec'), '1']) + '\n'
            f.write(s)

