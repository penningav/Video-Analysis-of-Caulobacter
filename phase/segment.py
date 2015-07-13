#!/usr/bin/env python

"""
This module is used to segment bacterial cells from phase-contrast images.

"""

import os
import sys
import traceback

from skimage.measure import label
from skimage.morphology import disk, remove_small_objects
from skimage.filter.rank import bottomhat
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import median_filter
from scipy.signal import wiener

sys.path.append(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
from image import *

__author__ = "Charlie Wright"
__email__ = "charles.s.wright@gmail.com"


def lowpassfilter(img, bgd_img):
    """
    Remove background from input image by dividing out pre-defined background 
    image, then perform psuedo-flatfield correction.

    args:
        img (ndarray): input image
        bgd_img (ndarray): background image

    returns:
        ndarray: output image (dtype = float)

    """
    img2 = removebackground(img, bgd_img)
    img3 = fixflatfield(img2)
    return img3


def segmentglobal(img, filter_size=5, level=0.7):
    """
    Segment image using gradients calculated globally. Apply a Wiener filter to
    remove salt-and-pepper noise and a median filter to smooth edges, then 
    calculate gradients across the entire image (between adjacent pixels in 
    both directions). Threshold the image constructed my multiplying the 
    original image with its derivative (this helps define the edges).

    args:
        img (ndarray): input image

    kwargs:
        filter_size (int): neighborhood size of Wiener and median filters
        level (float): threshold level, specified as a fraction of the 
            calculated Otsu threshold; must by in the range [0, 1]

    returns:
        ndarray: thresholded binary image (dtype = bool)

    """
    img2 = complement(img, 1.)
    img3 = wiener(img2, (filter_size, filter_size))
    img4 = median_filter(img3, filter_size)
    img5 = energy(img4)
    img6 = (1 + img5) * img4
    return threshold(img6, level=level).astype(bool)


def segmentlocal(img, filter_size=5, level=0.2, selem_size=3,
                 rescale_low=(0, 50), rescale_high=(50, 100)):
    """
    Segment image using gradients calculated locally. Apply a Wiener filter to 
    remove salt-and-pepper noise, then calculate gradients over local 
    neighborhoods using the specified structuring element. Threshold the image 
    constructed my multiplying the rescaled original image with its rescaled 
    derivative (this helps define the edges).

    args:
        img (ndarray): input image

    kwargs:
        filter_size (int): neighborhood size of Wiener filter
        selem_size (int): size of the disk structuring element
        level (float): threshold level, specified as a fraction of the 
            calculated Otsu threshold; must by in the range [0, 1]
        rescale_low (tuple): (low, high) values used to rescale original image
        rescale_high (tuple): (low, high) values used to rescale edges image

    returns:
        ndarray: thresholded binary image (dtype = bool)

    """
    img2 = wiener(img, (filter_size, filter_size))
    img3 = median_filter(img2, filter_size)

    img4 = complement(rescale(img3, rescale_low))

    img5 = bottomhat(img_as_uint12(img3), disk(selem_size))
    img6 = fillholes(img5)
    img7 = rescale(img6, rescale_high)

    img8 = img4 * img7
    return threshold(img8, level=level)


def cleanbinary(bw, min_size=400):
    """
    Clean up the segmented binary image.

    args:
        bw (ndarray): segmented binary image

    kwargs:
        min_size (int): minimum size of objects to keep

    return:
        ndarray: output image (dtype = bool)

    """
    bw2 = binary_fill_holes(bw)
    bw3 = remove_small_objects(bw2, min_size=min_size)
    return bw3
    

def main(img, d):
    """
    Segment the specified grayscale images, and save the binary image to file. 
    First, clean the image by removing the background and filtering it, then 
    find the edges and threshold it to convert it to a binary image.

    args:
        img (ndarray): input image
        d (dict): input parameters

    returns:
        labeled (ndarray): labeled image

    """

    # First perform lowpass filtering
    img2 = lowpassfilter(img, d['bgd_img'])

    # Segment using global or local method
    if d['method'] == 'global':
        bw = segmentglobal(img2,
                           filter_size=d['filter_size'],
                           level=d['level'])
    else:
        bw = segmentlocal(img2,
                          filter_size=d['filter_size'],
                          level=d['level'],
                          selem_size=d['selem_size'],
                          rescale_low=d['rescale_low'],
                          rescale_high=d['rescale_high'])

    # Clean the binary image and return a labeled integer array
    bw2 = cleanbinary(bw, min_size=d['min_size'])
    return label(bw2)
