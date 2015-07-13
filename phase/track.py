#!/usr/bin/env python

"""
This module is used to track cells between phase-contrast images.

"""

import os
import sys
import traceback

import numpy as np

from scipy.interpolate import splev

sys.path.append(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
import read

__author__ = "Charlie Wright"
__email__ = "charles.s.wright@gmail.com"


def matchpoints(series1, series2, cutoff=10):
    """
    Track the points given by series1 into series2, whose Euclidean distance is 
    less than the distance given by the parameter max_dist. If more than two 
    objects lie within the same cutoff distance, neither is identified. The 
    dresults give the labels of matching objects in each series.

    args:
        series 1 (Series): 1st series of trajectories
        series 2 (Series): 1st series of trajectories

    kwargs:
        cutoff (int): search radius

    """
    matches = []
    for i1 in series1.index:
        try:
            x1, y1 = series1[i1]
        except TypeError:
            continue
        for i2 in series2.index:
            try:
                x2, y2 = series2[i2]
            except TypeError:
                continue
            dx = (x1 - x2)
            dy = (y1 - y2)
            dist = np.mean(np.sqrt(dx * dx + dy * dy))
            if dist <= cutoff:
                matches.append((i1, i2))
    if not matches:
        return []

    # Remove non-unique connections
    matches1, matches2 = zip(*matches)
    results = []
    for i1, i2 in matches:
        if matches1.count(i1) == 1 and matches2.count(i2) == 1:
            results.append((i1, i2))
    return results


def matchsearch(series1, series2, cutoffs=[1, 2, 3]):
    """
    Find matches at a series of cutoff distances, which should be a list of 
    monotonically increasing positive numbers.

    args:
        series 1 (Series): 1st series of trajectories
        series 2 (Series): 1st series of trajectories

    kwargs:
        cutoffs (list): monotonically-increasing search radii

    """

    # Labels of original points
    j1 = set(series1.index)
    j2 = set(series2.index)

    m1, m2 = [], []
    results = []
    for c in cutoffs:
        # Find unique matches between unmatched points within cutoff radius
        ms = matchpoints(series1[j1], series2[j2], cutoff=c)
        if ms:
            # Update labels of still-unmatched points
            m1, m2 = zip(*ms)
            j1 = j1 - set(m1)
            j2 = j2 - set(m2)

            # Save labels of points that have been matched
            results.extend(ms)
    return results


def matchtraces(series1, series2, matches):
    """
    Relabel unique matches between input series.

    args:
        series 1 (Series): 1st series of trajectories
        series 2 (Series): 1st series of trajectories
        matches (list): candidate matches

    """
    series2_copy = series2.copy()

    # Relabel points in second series to match the first
    for m1, m2 in matches:
        series2[m2] = series1[m1]

    # Reset any non-unique matches to the original value
    counts2 = series2.value_counts()
    for i in np.flatnonzero(counts2.values > 1):
        j = (series2 == counts2.index[i])
        series2[j] = series2_copy[j]


def getpoints(input_dir, num_points=5, num_pixels=10.0):
    """
    Evaluate spline at discrete points that will be compared to establish 
    continuous time trajectories.

    args:
        df (DataFrame): contains requested coordinates and trajectories

    kwargs:
        num_points (int): number of points to compare
        num_pixels (float): number of pixels along midline to compare

    """

    # Read the midline and any previous trajectories in from file
    df = read.makedataframe(input_dir, ('MidSpline', 'Length', 'Trace'))
    df = df.reindex(sorted(df.index))
    if 'Trace' not in df:
        df['Trace'] = [v for v in df.index]

    # Evaluate the midline at discrete points starting at the old pole
    u0 = np.linspace(0., 1., num_points)
    vs = []
    for i in df.index:
        u = u0 * min(1, num_pixels / df['Length'][i])
        try:
            v = splev(u, df['MidSpline'][i])
        except:
            # traceback.print_exc()
            v = [np.ones(num_points) * np.nan] * 2
        vs.append(v)
    df['Points'] = vs
    return df.ix[:, ('Points', 'Trace')]


def track(df, frame_steps=[1, 2], pixel_steps=[1, 2, 3]):
    """
    Track objects across the entire movie. The first input must be a multi-
    indexed DataFrame object, with first level corresponding to the frame 
    number, the second to the automatically assigned per-frame 'Label' and 
    columns 'Points' and 'Trace'. The 'Trace' value is updated after each 
    iteration of the tracking procedure. For each pair of frames that are 
    compared, the Euclidean distance beween points in 'Points' is calculated, 
    and a new trace identified only if a unique match exists between points at 
    a distance less than the current cutoff. This procedure repeats at various 
    intervals between frames.

    args:
        df (DataFrame): contains coordinates to match and names of traces

    kwargs:
        frame_steps (list): steps between different frames to compare
        pixel_steps (list): distances in pixels from each cell to compare

    """
    frames = df.index.levels[0]
    for j in frame_steps:
        for i in frames[j::]:
            try:
                df1 = df.ix[i-j]
                df2 = df.ix[i]
            except:
                # traceback.print_exc()
                continue
            matches = matchsearch(df1['Points'], df2['Points'], pixel_steps)
            matchtraces(df1['Trace'], df2['Trace'], matches)


def stitch(input_dirs, d):
    """
    Stitch tracked DataFrame traces together, saving the results in place.

    args:
        input_dirs: list of directories with independent tracking results
        d (dict): input parameters

    """

    # Stitch blocks together (overwriting previous trace identifications)
    for i, v in enumerate(input_dirs):
        df2 = getpoints(v, d['num_points'], d['num_pixels'])
        if i > 0:
            df = read.stitchdataframes(df1, df2, 1+max(d['frame_steps']))
            track(df, d['frame_steps'], d['pixel_steps'])
            frames = set(df.index.levels[0]) & set(df2.index.levels[0])
            for f in frames:
                for j, y in df2['Trace'][f].iteritems():
                    x = df['Trace'][f][j]
                    if x != y:
                        for k, z in df2['Trace'].iteritems():
                            if y == z:
                                df2['Trace'][k] = x
            df2['Trace'].to_pickle(os.path.join(v, 'Trace.pickle'))
        df1 = df2.copy()


def main(input_dir, d):
    """
    Track cells from the specified binary files.

    args:
        d (dict): input parameters

    returns:
        Series: each cell is labeled with a unique trace

    """
    df = getpoints(input_dir, d['num_points'], d['num_pixels'])
    track(df, d['frame_steps'], d['pixel_steps'])
    return df['Trace']

