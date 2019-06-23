import glob
import os
import numpy as np
import cv2

def checkdir(d):
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def process(src, des, new_size):
    files = glob.glob(os.path.join(src, '*npy'))
    files.sort()
    assert len(files) == 30000, 'the number of npy is wrong, man'
    for file in files:
        data = np.load(file)[0]
        data = np.transpose(data, [1, 2, 0])
        data_new = cv2.resize(data, (new_size, new_size))
        data_new = np.transpose(data_new, [2, 0, 1])
        path_new = os.path.join(des, file.split('/')[-1])
        np.save(path_new, data_new)
        print(file, path_new)

if __name__ == '__main__':
    new_size = 256
    src = '/p300/project/download-celebA-HQ/celebA-HQ'
    des = checkdir(src + '-{}'.format(new_size))

    process(src, des, new_size)