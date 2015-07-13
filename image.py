#!/usr/bin/env python

"""
This module contains functions useful for basic image analysis. For every function, the expected input image is a numpy array.

"""

import numpy as np

from scipy import ndimage
from skimage import exposure, filter, img_as_float, morphology

__author__ = "Charlie Wright"
__email__ = "charles.s.wright@gmail.com"

def removebackground(img, bgd=np.ones(1, dtype=np.float)):
    """
    Remove defined background from image. If no background image is given, the original image is returned as a float.

    args:
        img (ndarray): input image

    kwargs:
        bgd (ndarray): background image (dtype = float, shape = same as input)

    returns:
        ndarray: output image (dtype = float)

    """
    img = img_as_float(img)
    return img / bgd

def fixflatfield(img, siz=0):
    """
    Perform pseudo-flatfield correction by dividing the image by a large uniform (averaging) filter.

    args:
        img (ndarray): input image

    kwargs:
        siz (int): size of averaging filter (default = 25% of the smallest image dimension)

    returns:
        ndarray: output image (dtype = float)

    """
    img = img_as_float(img)
    if siz == 0:
        siz = np.ceil(min(img.shape) / 4)
    flt = ndimage.uniform_filter(img, siz)
    return img / flt

def complement(img, level=0.):
    """
    Invert the image around the specified level.

    args:
        img (ndarray): input image

    kwargs:
        level (float): background image (default = maximum value in image)

    returns:
        ndarray: output image

    """
    if level == 0.:
        level = img.max()
    return level - img

def rescale(img, p_range=(0, 100)):
    """
    Rescale the image between 0 and 1, optionally stretching the contrast by setting the new lower and upper bounds of grayscale pixel values.

    args:
        img (ndarray): input image

    kwargs:
        p_range (tuple): new (lower, upper) bounds (default = full range)

    returns:
        ndarray: output image

    """
    in_range = tuple([np.percentile(img, v) for v in p_range])
    return exposure.rescale_intensity(img, in_range=in_range)

def energy(img):
    """
    Calculate the image energy, defined here as the absolute value of the gradients in both row and column directions of the image.

    args:
        img (ndarray): input image

    returns:
        ndarray: output image

    """
    [d_x, d_y] = np.gradient(img)
    return np.sqrt(d_x ** 2 + d_y ** 2)

def fillholes(img, selem=None):
    """
    Fill holes in the grayscale input image using specified structuring element.

    args:
        img (ndarray): input image

    kwargs:
        selem (ndarray): structuring element (default = disk of radius 50)

    returns:
        ndarray: output image

    """
    if selem is None:
        selem = morphology.disk(50)
    seed = filter.rank.maximum(img, selem)
    return morphology.reconstruction(seed, img, method='erosion')

def img_as_uint12(img):
    """
    Convert image to 12-byte format (uint16, but with all values <4096), required for some functions in skimage. Input must be positive-valued.

    args:
        img (ndarray): input image

    returns:
        ndarray: output image (dtype = uint12)

    """
    return (img / img.max() * 4095).astype('uint16')

def threshold(img, use_otsu=True, level=1.):
    """
    Threshold the image at the defined level. If use_otsu = False, calculate threshold based on absolute levels. Otherwise, calculate threshold using the Otsu method and let the level be a fraction of this value.

    args:
        img (ndarray): input image

    kwargs:
        use_otsu (bool): specifies whether to calculate the threshold based on absolute levels (False) or using the Otsu method, with the input level representing a specified fraction of this value (True)
        level (float): threshold level, in range [0, 1] (default = 1)

    returns:
        ndarray: output image (dtype = bool)

    """
    if use_otsu:
        level = level * filter.threshold_otsu(img)
    return img > level
