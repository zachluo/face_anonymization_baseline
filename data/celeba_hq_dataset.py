import sys
sys.path.append('.')
from data.base_dataset import BaseDataset, get_transform
import os
import random
import numpy as np
from PIL import Image

class CelebaHQDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.transform = get_transform(opt)
        self.celeba_id = {}
        with open(opt.celeba_id_list) as f:
            for line in f:
                info = line.rstrip('\n').split()
                self.celeba_id[info[0]] = int(info[1])

        self.celeba_hq_id = {}
        with open(opt.celeba_hq_list) as f:
            f.readline() # drop the first line
            for line in f:
                line = line.rstrip('\n').split()
                self.celeba_hq_id[int(line[0])] = self.celeba_id[line[2]]

        celeba_hq_id_unique = np.sort(np.unique(list(self.celeba_hq_id.values())))
        print('There is totally {} people in celeba_hq dataset'.format(len(celeba_hq_id_unique)))
        mapping_old2new = {}
        for index, old_id in enumerate(celeba_hq_id_unique):
            mapping_old2new[old_id] = index

        self.indexlist = []
        for img, id in self.celeba_hq_id.items():
            self.indexlist.append((img, mapping_old2new[id]))
        if opt.isTrain:
            self.indexlist = self.indexlist[:25000]
            random.shuffle(self.indexlist)
        else:
            self.indexlist = self.indexlist[25000:]



    def load_img(self, index):
        npy_path = os.path.join(self.root, 'imgHQ{:05d}.npy'.format(self.indexlist[index][0]))
        cls = self.indexlist[index][1]

        img = np.load(npy_path)
        # convert RGB to BGR
        #img = img[::-1]

        return img, cls, npy_path

    def __getitem__(self, index):
        # Get the index of each image in the triplet
        _a, a_cls,a_path = self.load_img(index)
        # transform images
        pil_img = Image.fromarray(np.transpose(_a, [1, 2, 0]))
        img_a = self.transform(pil_img)

        # get random condition
        c_bin = np.array([np.random.randint(0, 2) for i in range(self.opt.n_condition)], dtype=np.float32)
        #c_bin = np.array([1,1,1,1], dtype=np.float32)
        c_dec = np.sum([value * np.power(2, self.opt.n_condition - index - 1) for index, value in enumerate(c_bin)], dtype=np.float32)
        # import pdb; pdb.set_trace()
        return {'real': img_a,
                 'real_path': a_path,
                 'label': np.int(a_cls),
                 'condition_bin': c_bin,
                 'condition_dec': c_dec}

    def __len__(self):
        return len(self.indexlist)

if __name__ == '__main__':
    class option(object):
        def __init__(self):
            self.celeba_id_list = '/p300/project/download-celebA-HQ/celebA/Anno/identity_CelebA.txt'
            self.celeba_hq_list = '/p300/project/download-celebA-HQ/image_list.txt'
            self.dataroot = '/p300/project/download-celebA-HQ/celebA-HQ-256'
            self.load_size = 256
            self.n_condition = 4
            # self.isTrain = False

    opt = option()
    celeba_hq = iter(CelebaHQDataset(opt))
    for key, d in next(celeba_hq).items():
        if type(d) == int:
            print(key, d)
        else:
            print(key, d.shape)
