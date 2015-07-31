#!/usr/bin/env python

"""
This module defines the general analysis workflow for phase-contrast images.

"""

import os
import sys
import time

import cPickle as pickle
import numpy as np

from pandas import DataFrame, MultiIndex
from skimage.io import imread, imsave
from scipy.interpolate import splev

sys.path.append(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
import read

from phase import *


def fillparams(params):
    """
    Read in dictionary of input parameters; set those not explicitly given to 
    default values.

    args:
        params (dict): input parameters, with keys giving the function to 
            perform and each value a dictionary of parameter-value pairs for 
            that function

    """

    # Set possible defaults used by every function
    for f in ('segment', 'extract', 'track', 'verify', 'collate'):
        d = params[f] if params.has_key(f) else {}
        if f == 'segment':
            # Set defaults used by every segmenting function
            p1 = {'pattern' : '^(phase)([0-9]+)(.tif)$',
                  'frame_range' : [0, +float("inf")],
                  'bgd_file' : (os.path.join(os.path.dirname(
                                os.path.abspath(__file__)),'background.pkl')),
                  'min_size' : 400}

            m = d['method'] if d.has_key('method') else 'default'
            if m == 'global':
                # Global segmentation method
                p2 = {'method' : 'global',
                      'filter_size' : 5,
                      'level' : 0.7}
            else:
                # Local segmentation method (default)
                p2 = {'method' : 'local',
                      'filter_size' : 5,
                      'level' : 0.2,
                      'selem_size' : 3,
                      'rescale_low' : (0, 50),
                      'rescale_high' : (50, 100)}
            p = dict(p1.items() + p2.items())
        elif f == 'extract':
            # Extract data
            p = {'smoothing' : 0.1,
                 'degree' : 5,
                 'alpha' : 0.,
                 'coverage' : 2,
                 'num_coefs' : 20,
                 'span' : 3,
                 'cutoff' : 0.5}
        elif f == 'track':
            # Track cells
            p = {'frame_steps' : [1, 2, 4, 8],
                 'pixel_steps' : [1, 3, 5, 10, 15, 25],
                 'num_points' : 5,
                 'len_frac' : 0.3}
        elif f == 'verify':
            # Verify data
            p = {'shape' : (1024, 1024),
                 'edge' : (1, 1),
                 'min_dist' : 250,
                 'num_points' : 100}
        elif f == 'collate':
            # Collate data
            p = {'min_dist' : 30,
                 'num_coefs' : 5,
                 'spans' : range(10, 50, 10),
                 'order' : 30,
                 'min_prob' : 0.7}
        else:
            # Other functions with unspecified defaults
            p = {}

        # Set each value in params to the default if it is not already set
        for k, v in p.iteritems():
            if not d.has_key(k):
                d[k] = v

        # When segmenting, set the background image only if necessary
        if f == 'segment' and not d.has_key('bgd_img'):
            d['bgd_img'] = pickle.load(open(d['bgd_file'], 'r'))

        # Set the original parameters dictionary
        params[f] = d
    return params


def preeditimage(input_file, output_dir, params):
    """
    Segment the specified grayscale images, and save the binary image to file.
    First, clean the image by removing the background and filtering it, then 
    find the edges and threshold it to convert it to a binary image. Extract 
    and verify the data from this image.

    args:
        input_file (file): input directory of raw data
        output_dir (path): output directory to save file
        params (dict): input parameters

    """

    # Do not overwrite existing output
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    if os.path.isfile(output_file):
        img = imread(output_file)
    else:
        # Segment the grayscale image and save to file
        img = segment.main(imread(input_file), params['segment'])
        imsave(output_file, img)

    print ' - segment: ' + time.asctime()

    # Do not overwrite existing output
    output_file2 = os.path.splitext(output_file)[0] + '.pickle'
    if os.path.isfile(output_file2):
        return

    # Extract properties from the labeled image and save as a DataFrame
    data = extract.preedit(img, params['extract'])
    columns = ('Area', 'BoundingBox', 'Centroid', 'EdgeSpline', 'FourierFit',
               'Length', 'MidSpline', 'Perimeter', 'StalkedPole', 'SwarmerPole')

    f = read.getframenum(input_file, params['segment']['pattern'])
    if data:
        # Make MultiIndex with frame and label info
        j = [f] * len(data)
        k = [v['Label'] for v in data]
    else:
        # Create empty DataFrame
        data = [dict.fromkeys(columns, np.nan)]
        j = [f]
        k = [-1]
    index = MultiIndex.from_arrays((j, k), names=('Frame', 'Label'))
    df = DataFrame(data, columns=columns, index=index)
    verify.preedit(df, params['verify'])
    df.to_pickle(output_file2)

    print ' - extract: ' + time.asctime()


def trackblock(input_dir, output_file, params):
    s = track.main(input_dir, params)
    print 'phase/workflow trackblock is writing tracking info to', output_file
    s.to_pickle(output_file)


def stitchblocks(input_dirs, params):
    track.stitch(input_dirs, params)


def collateblocks(input_dirs, output_file, params):
    df = collate.main(input_dirs, params)
    # debug -BK
    print 'Writing dataframe to ', output_file
    df.to_pickle(output_file)


def editmovie(expt_raw_data_dir, expt_analyses_dir, positions):
    """
    Interactive, manual editing.

    """
    edit.main(expt_raw_data_dir, expt_analyses_dir, positions)


def posteditblock(input_dir, params):
    """
    Automated, post-editing analysis.

    """
    postedit.main(input_dir, params)

