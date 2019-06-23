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

class LFWBaselineDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        bbox_path = os.path.join('/p300/dataset', 'lfw_detection.txt')
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


        with open(os.path.join('/p300/dataset', opt.landmark_path)) as f:
            self.indexlist = [line.rstrip('\n').split() for line in f]
            for index in range(len(self.indexlist)):
                self.indexlist[index] = [self.indexlist[index][0]] + \
                                        [float(i) for i in self.indexlist[index][1:]]


        self.transform = get_transform(opt)
        self.transform_baseline = get_transform_baseline(opt)

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

    def load_img(self, index):
        info = self.indexlist[index]
        landmark = [(float(k)/125. - 1) for k in info[1:]]
        tfm = self.alignment(landmark).astype(np.float32)
        img_path = os.path.join(self.root, info[0])
        img = Image.open(img_path).convert('RGB')

        bbox = np.asarray(self.bbox_dict[info[0]])

        return img, tfm, img_path, bbox


    def __getitem__(self, index):
        _a, a_tfm, img_path, bbox = self.load_img(index)

        # transform images
        real_a = self.transform(_a)
        fake_a = self.transform_baseline(_a)

        return {'real': real_a,
                'real_path': img_path,
                'fake': fake_a,
                'theta': a_tfm,
                'bbox': bbox}

    def __len__(self):
        return len(self.indexlist)

if __name__ == '__main__':
    pass

