#!/usr/bin/env python

import os
import sys
import traceback

import numpy as np
import cPickle as pickle

from pandas import concat, read_pickle, DataFrame

from scipy.signal import argrelmax
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

sys.path.append(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
import read

__author__ = "Charlie Wright"
__email__ = "charles.s.wright@gmail.com"


def splitdata(data, num_clusters=2):
    """
    Split data using K-means clustering.

    args:
        data (ndarray): array of values (M observations of N values)

    kwargs:
        num_clusters (int): number of clusters to identify

    returns:
        generator: indexes of edges

    """

    # Perform cluster analysis
    whitened = whiten(data)
    centroids, _ = kmeans(whitened, num_clusters)
    index, _ = vq(whitened, centroids)

    # Return indexes of edges
    for i in range(len(index)-1):
        if index[i] != index[i+1]:
            yield i


# TODO: proper docstrings for the following 3 functions --BK

def identify_linear_outliers(pts, win_size=7):
    # this runs a sliding window across the trace, performing a RANSAC regression
    # for each window. A point is considered an outlier if the moving-RANSAC
    # never considers it an inlier.
    regressor = RANSACRegressor()
    x = np.arange(win_size, dtype=np.float64)
    x = np.expand_dims(x, axis=1)
    inlier_count = np.zeros_like(pts)
    npts = len(pts)
    for i in range(npts-win_size+1):
        y = pts[i:i+win_size]

        # RANSAC of this section of the trace
        try:
            regressor.fit(x, y)
            inlier_inds = regressor.inlier_mask_
        except ValueError:  # no consensus -- (almost) all the points were bad
            inlier_inds = []

        # accumulate the number of times each point was an inlier
        for j, inlier in enumerate(inlier_inds):
            if inlier:
                inlier_count[i+j] += 1

    # Note: the following line will always consider the first and last points outliers!
    #       However, I don't think this will matter for downstream analysis.  -BK
    outlier_mask = np.logical_or(inlier_count < 2, pts == 0)
#    outlier_inds = np.where(outlier_mask)[0]
#
#    # points that are exactly zero are always considered outliers
#    outlier_inds = np.append(outlier_inds, np.where(pts==0)[0])
    return outlier_mask


def nan_moving_regression(pts, win_size=4):
    """
    Does a moving regression. Nan's are skipped over, but we alway ensure that
    there are exactly win_size points used for the regression.
    TODO: this is ugly
    """
    ista = win_size
    x = np.arange(win_size)
    x = np.expand_dims(x, 1)
    prev = pts[:win_size]
    while np.any(np.isnan(prev)):
        naninds = np.where(np.isnan(prev))[0]
        x = x[~np.isnan(prev)]
        prev = prev[~np.isnan(prev)]
        x = np.append(x, range(ista, ista+len(naninds)))
        prev = np.append(prev, pts[ista: ista+len(naninds)])
        ista += len(naninds)

    x = np.expand_dims(x, 1)
    regressor = LinearRegression()
    predicted = np.zeros_like(pts)
    predicted[:ista] = np.nan
    scores = np.zeros_like(pts)
    scores[:ista] = np.nan
    i = ista
    while i < len(pts):
        pt = pts[i]
        regressor.fit(x, prev)
        score = regressor.score(x, prev)
        y_pred = regressor.predict(i)
        predicted[i] = y_pred
        scores[i] = score
#        print i, prev, y_pred
        while np.isnan(pt):
            i += 1
            if i > len(pts)-1:
                return predicted, scores
            print i, len(pts)
            predicted[i] = y_pred
            scores[i] = score
            pt = pts[i]
        x = np.roll(x, -1)
        x[-1] = i
        prev = np.roll(prev, -1)
        prev[-1] = pt
        i += 1
    return predicted, scores



def division_inds(pts0, outlier_mask, win_size=4, thresh=-4):
    # Performs a sliding window regression with outliers masked. We threshold
    # based upon the difference between the linearly extrapolated point and the
    # actual point, weighted by the quality of the linear fit (if the fit is
    # poor, then the discrepancy is less significant)
    pts = np.copy(pts0)
    pts[outlier_mask] = np.nan
    pred, scores = nan_moving_regression(pts, win_size)
    inds = np.where(scores * (pts - pred) < thresh)[0]
    return inds.tolist()


def getdivntimes(coefs):
    pts = coefs[:, 0]
    outlier_mask = identify_linear_outliers(pts)
    outlier_inds = np.where(outlier_mask)[0]
    return division_inds(pts, outlier_inds)

#
# def getdivntimes(coefs, spans=range(10, 50, 10), order=30, min_prob=0.5):
#     """
#     Calculate division times from coefficients of fit.
#
#     args:
#         coefs (ndarray): coefficients of Fourier fit to splined cells
#
#     kwargs:
#         spans (list): list of timescales over which to cluster
#         order (int): separation of cluster edges
#         min_prob (float): value in [0, 1] giving minimum probability, above
#             which a point is marked as an edge between clusters
#
#     returns:
#         divns (list): list of division times
#
#     """
#     # Find division times from fit coefficients
#     num_frames, num_coefs = coefs.shape
#     probs = np.zeros(num_frames)
#     for span in spans:
#         v = 1. / span
#         w = np.arange(span)
#         for f in range(num_frames - span + 1):
#             fs = f + w
#             c = coefs[fs, :]
#             if np.sum(c == 0) > 0.5 * span * num_coefs:
#                 continue
#             for e in splitdata(c):
#                 probs[fs[e]] += v
#     probs /= len(spans)
#
#     # debug --BK
#     f = open('/home/brian/tmp/coefs.pickle', 'wb')
#     pickle.dump(coefs, f)
#     f = open('/home/brian/tmp/spans.pickle', 'wb')
#     pickle.dump(spans, f)
#     f = open('/home/brian/tmp/order.pickle', 'wb')
#     pickle.dump(order, f)
#     f = open('/home/brian/tmp/min_prob.pickle', 'wb')
#     pickle.dump(min_prob, f)
#     f = open('/home/brian/tmp/probs.pickle', 'wb')
#     pickle.dump(probs, f)
#
#     # Return peaks in division time probabilities
#     divns = []
#     k = np.hstack(argrelmax(probs, order=order))
#     if np.any(k):
#         divns = sorted(k[probs[k] > min_prob])
#
#     # debug --BK
#     f = open('/home/brian/tmp/divns.pickle', 'wb')
#     pickle.dump(divns, f)
#
#     return divns


def main(input_dirs, d):
    """
    Gather relevant quantities to a single .pickle file.

    args:
        input_dirs (path): input data directories
        d (dict): input parameters

    """

    # Find all analyzed frame blocks in the given directory
    read_names = 'Area', 'Trace', 'IsOnEdge', 'Nearest', 'FourierFit'
    r = dict.fromkeys(read_names)
    for input_dir in input_dirs:
        for read_name in read_names:
            df = read_pickle(os.path.join(input_dir, read_name + '.pickle'))
            if read_name not in r:
                r[read_name] = df
            else:
                r[read_name] = concat((r[read_name], df))

    # Initialize values organized by cell trajectory
    traces = np.unique(r['Trace'].values)
    num_traces = len(traces)
    frames = r['Trace'].index.levels[0]
    num_frames = len(frames) #max(frames) - min(frames) + 1; f0 = min(frames)
    lifetimes = r['Trace'].value_counts()
    min_lifetime = min(300, 0.8 * num_frames)

    # Data for the DataFrame to save to file
    data = [[] for _ in range(num_traces)]
    columns = 'Trace', 'Saved', 'Mother', 'Divns', 'Area', 'Label', 'Keep'
    index = range(num_traces)

    # Convert all values to arrays organized by cell trajectory
    for i, trace in enumerate(traces):
        # Create data dictionary for each trace
        trace_dict = dict.fromkeys(columns)
        trace_dict['Trace'] = trace
        trace_dict['Saved'] = lifetimes[trace] > min_lifetime
        trace_dict['Mother'] = None
        trace_dict['Label'] = np.zeros(num_frames, dtype=int)
        trace_dict['Keep'] = np.ones(num_frames, dtype=bool)
        trace_dict['Area'] = np.zeros(num_frames) * np.nan
        trace_dict['Divns'] = []
        coefs = np.zeros((num_frames, d['num_coefs']))
        for j, frame in enumerate(frames):
            k = trace == r['Trace'][frame]

            if k.any():
                try:
                #Add values for cell label and area

                    trace_dict['Label'][j] = r['Trace'][frame][k].index[0]
                    trace_dict['Area'][j] = r['Area'][frame][k]

                    # Save coefficients from Fourier fit
                    c = r['FourierFit'][frame][k].values[0]
                    coefs[j, :] = np.abs(c)[:d['num_coefs']]

                    # Do not keep point if checks failed
                    n = r['Nearest'][frame][k].values[0].values()

                    is_on_edge = r['IsOnEdge'][frame][k].values[0]
                    area_not_ok = np.isnan(r['Area'][frame][k]).values[0]
                    dist_too_small = (n and min(n) < d['min_dist'])

                    if (is_on_edge or area_not_ok or dist_too_small):
                        trace_dict['Keep'][j] = False
                except:
                    trace_dict['Keep'][j] = False

            else:
                trace_dict['Keep'][j] = False

        # Find division times
        if trace_dict['Saved']:
            #trace_dict['Divns'] = getdivntimes(coefs, spans=d['spans'], order=d['order'], min_prob=d['min_prob'])
            trace_dict['Divns'] = getdivntimes(coefs)
        data[i] = trace_dict

    # Return the collated Data Frame
    df = DataFrame(data, columns=columns, index=index)
    print '---- returning dataframe: '
    print df.head()
    return df
