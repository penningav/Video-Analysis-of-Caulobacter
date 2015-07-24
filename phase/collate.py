#!/usr/bin/env python

import os
import sys
import traceback

import numpy as np
import cPickle as pickle

from pandas import concat, read_pickle, DataFrame

from scipy.signal import argrelmax
from scipy.cluster.vq import kmeans, vq, whiten

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


def getdivntimes(coefs, spans=range(10, 50, 10), order=30, min_prob=0.5):
    """
    Calculate division times from coefficients of fit.

    args:
        coefs (ndarray): coefficients of Fourier fit to splined cells

    kwargs:
        spans (list): list of timescales over which to cluster
        order (int): separation of cluster edges
        min_prob (float): value in [0, 1] giving minimum probability, above 
            which a point is marked as an edge between clusters

    returns:
        divns (list): list of division times

    """

    # Find division times from fit coefficients
    num_frames, num_coefs = coefs.shape
    probs = np.zeros(num_frames)
    for s in spans:
        v = 1./ s
        w = np.arange(s)
        for f in range(num_frames - s + 1):
            fs = f + w
            c = coefs[fs, :]
            if np.sum(c == 0) > 0.5 * s * num_coefs:
                continue
            for e in splitdata(c):
                probs[fs[e]] += v
    probs /= len(spans)

    # Return peaks in division time probabilities
    divns = []
    k = np.hstack(argrelmax(probs, order=order))
    if np.any(k):
        divns = sorted(k[probs[k] > min_prob])
    return divns


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

    # debug --BK
    print '**** debug: r dict'
    for key in r:
        print
        print r[key].head()


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
                # try:
                # Add values for cell label and area

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
                # except:
                #     trace_dict['Keep'][j] = False

            else:
                trace_dict['Keep'][j] = False

        # Find division times
        if trace_dict['Saved']:
            trace_dict['Divns'] = getdivntimes(coefs,
                spans=d['spans'], order=d['order'], min_prob=d['min_prob'])
        data[i] = trace_dict

    # Return the collated Data Frame
    df = DataFrame(data, columns=columns, index=index)
    return df
