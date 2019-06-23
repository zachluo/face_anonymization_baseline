import sys
import datetime

import torch
import torch.nn as nn
from torch.nn import init

# class Logger(object):
#     def __init__(self, out_fname):
#         self.out_fd = open(out_fname, 'w')
#
#     def log(self, out_str, end='\n'):
#         """
#         out_str: single object now
#         """
#         # for classes, if __str__() is not defined, then it's the same as __repr__()
#         self.out_fd.write(str(out_str) + end)
#         self.out_fd.flush()
#         print(out_str, end=end, flush=True)
#
#     def close(self):
#         self.out_fd.close()

# TODO: add logging time option
class Logger(object):
    def __init__(self, out_fname):
        self.log_name = out_fname

    def log(self, out_str, end='\n', log_time=False):
        """
        out_str: single object now
        """
        # for classes, if __str__() is not defined, then it's the same as __repr__()
        if log_time:
            out_str = '{0:%Y-%m-%d %H:%M:%S} '.format(datetime.datetime.now()) + out_str
        with open(self.log_name, "a") as log_file:
            log_file.write(out_str + end)
        print(out_str, end=end, flush=True)


def init_weights_multi(m, init_type, gain=1.):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
            init.orthogonal_(m.weight.data, gain=gain)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, gain)
        init.constant_(m.bias.data, 0.0)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count