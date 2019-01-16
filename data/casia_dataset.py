import sys
sys.path.append('.')
from data.base_dataset import BaseDataset, get_transform
import os
import random
import numpy as np
from PIL import Image
from util.matlab_cp2tform import get_similarity_transform_for_cv2

class CASIADataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        with open(os.path.join(self.root, opt.landmark_path)) as f:
            self.indexlist = [line.rstrip('\n') for line in f]
        random.shuffle(self.indexlist)
        self.transform = get_transform(opt)
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
            rand = random.randint(0, len(self.indexlist)-1)
            name, cls = self.indexlist[rand].split()[0:2]
            if cls != a_cls:
                break
        return rand

    def load_img(self, index):
        info = self.indexlist[index].split()
        cls = int(info[1])
        landmark = [(float(k)/125. - 1) for k in info[2:]]

        img_path = os.path.join(self.root, self.img_dir, info[0])
        img = Image.open(img_path).convert('RGB')

        return img, landmark, cls

    def __getitem__(self, index):
        # Get the index of each image in the triplet
        a_name, a_cls = self.indexlist[index].split()[0:2]
        n_index = self.sample_negative(a_cls)

        _a, a_keys, a_cls = self.load_img(index)
        _n, n_keys, n_cls = self.load_img(n_index)

        # transform images
        img_a = self.transform(_a)
        img_n = self.transform(_n)

        tfm_a = self.alignment(a_keys)
        tfm_n = self.alignment(n_keys)

        return {'real_a': img_a,
                 'real_n': img_n,
                 'theta_a': tfm_a.astype(np.float32),
                 'theta_n': tfm_n.astype(np.float32),
                 'label_a': np.int(a_cls),
                 'label_n': np.int(n_cls)}

    def __len__(self):
        return len(self.indexlist)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    class option(object):
        def __init__(self):
            self.landmark_path = 'casia_landmark.txt'
            self.dataroot = '/p300/dataset/casia'
            self.load_size = 256

    opt = option()
    casia = iter(CASIADataset(opt))
    for _, d in next(casia).items():
        print(d.shape)
