"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import torch
import cv2
import numpy as np
from skimage.segmentation import slic

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of data.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add data-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new data-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the data."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def get_transform(opt):
    """Create a torchvision transformation function
    """
    transform_list = [transforms.Scale((opt.load_size, opt.load_size)),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_baseline(opt):
    transforms_list = [transforms.Resize((opt.load_size, opt.load_size))]
    if opt.baseline != 'edge' and opt.baseline != 'blur':
        transforms_list.append(transforms.ToTensor())

    def masked(image):
        ratio = 0.2
        image[:, int(ratio * image.shape[1]):int((1 - ratio) * image.shape[1]), int(ratio * image.shape[2]):int((1 - ratio) * image.shape[2])].fill_(0.5)
        return image
    def noise(image):
        image += (torch.randn(image.shape) * 0.5)
        return image
    def super_pixel(image):
        numpy_image = image.detach().numpy()
        numpy_image_new = image.detach().numpy()
        seg = slic(np.transpose(numpy_image, [1, 2, 0]))
        for j in np.unique(seg):
            numpy_image_new[:, seg == j] = np.mean(numpy_image[:, seg == j], axis=1, keepdims=True)
        return image.new(numpy_image_new)
    def edge(image):
        numpy_image = np.array(image)
        result = cv2.Canny(numpy_image, 100, 200)
        return Image.fromarray(result)

    if opt.baseline == 'blur':
        transforms_list.extend([transforms.Resize((8, 8)),
                                transforms.Resize((opt.load_size, opt.load_size)),
                                transforms.ToTensor()])
    elif opt.baseline == 'masked':
        transforms_list.append(transforms.Lambda(masked))
    elif opt.baseline == 'noise':
        transforms_list.append(transforms.Lambda(noise))
    elif opt.baseline == 'super_pixel':
        transforms_list.append(transforms.Lambda(super_pixel))
    elif opt.baseline == 'edge':
        transforms_list.extend([transforms.Grayscale(),
                                transforms.Lambda(edge),
                                transforms.ToTensor()])

    transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    print(transforms_list)
    return transforms.Compose(transforms_list)


def __adjust(img):
    """Modify the width and height to be multiple of 4.

    Parameters:
        img (PIL image) -- input image

    Returns a modified image whose width and height are mulitple of 4.

    the size needs to be a multiple of 4,
    because going through generator network may change img size
    and eventually cause size mismatch error
    """
    ow, oh = img.size
    mult = 4
    if ow % mult == 0 and oh % mult == 0:
        return img
    w = (ow - 1) // mult
    w = (w + 1) * mult
    h = (oh - 1) // mult
    h = (h + 1) * mult

    if ow != w or oh != h:
        __print_size_warning(ow, oh, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __scale_width(img, target_width):
    """Resize images so that the width of the output image is the same as a target width

    Parameters:
        img (PIL image)    -- input image
        target_width (int) -- target image width

    Returns a modified image whose width matches the target image width;

    the size needs to be a multiple of 4,
    because going through generator network may change img size
    and eventually cause size mismatch error
    """
    ow, oh = img.size

    mult = 4
    assert target_width % mult == 0, "the target width needs to be multiple of %d." % mult
    if (ow == target_width and oh % mult == 0):
        return img
    w = target_width
    target_height = int(target_width * oh / ow)
    m = (target_height - 1) // mult
    h = (m + 1) * mult

    if target_height != h:
        __print_size_warning(target_width, target_height, w, h)

    return img.resize((w, h), Image.BICUBIC)


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
