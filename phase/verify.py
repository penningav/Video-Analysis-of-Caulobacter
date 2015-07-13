#!/usr/bin/env python

"""
This module is used to verify the splined cell outline results.

"""

import traceback

import numpy as np

from numpy.linalg import norm
from pandas import Series
from scipy.interpolate import splev

__author__ = "Charlie Wright"
__email__ = "charles.s.wright@gmail.com"


def checkedges(df, shape=(1024, 1024), edge=(1, 1)):
    """
    Determine whether a cell touches the edge of the image. The edge is set by 
    default to 1 (just the image border); setting it <1 will not detect any 
    edges while setting it >1 will create larger edge regions. The required 
    input is a DataFrame with column 'BoundingBox'. The DataFrame is modified 
    in place and returned, with a new column 'IsOnEdge' added to give 'True' if 
    cell on edge and 'False' otherwise.

    args:
        df (DataFrame): must have column 'BoundingBox'

    kwargs:
        shape (tuple): shape of input image given as (row, col)
        edge (tuple): size of image border given as (row, col)

    """

    # Define boundaries of acceptable image
    lim = (edge[0], edge[1], shape[0] - edge[0], shape[1] - edge[1])
    is_on_edge = lambda x: (x[0] < lim[0] or # min_row
                            x[1] < lim[1] or # min_col
                            x[2] > lim[2] or # max_row
                            x[3] > lim[3])   # max_col

    # Search for cells that fall outside the edges
    s = Series(False, index=df.index)
    for i, v in df['BoundingBox'].iteritems():
        try:
            s[i] = is_on_edge(v)
        except:
            # traceback.print_exc()
            pass

    # Save in place to the original DataFrame
    df['IsOnEdge'] = s
    

def findnearest(df, min_dist=250., num_points=100):
    """
    For each cell in each frame, finds all cells within the specified search 
    radius. First identifies all cells whose centroids are within the minimum 
    specified distance of each other, then calculates the distance of nearest 
    approach from this subset of cells by comparing every pixel on the cell 
    borders. For each cell, the label of the closest cell (from the same frame) 
    is saved along with the distance between them into a dictionary with keys 
    giving cell labels.

    args:
        df (DataFrame): must have columns 'Centroid' and 'EdgeSpline'

    kwargs:
        min_dist (float): the centroids must be this minimum distance apart 
            before points on the borders of the respective cells are compared
        num_points (int): number of points along the cell border

    """

    # Make aperiodic points for comparison
    u = np.arange(0., 1., 1. / num_points)

    # Loop over each frame independently
    s = Series(index=df.index, dtype=object)
    for i in df.index.levels[0]:
        cs = df.ix[i]['Centroid']
        ks1 = df.ix[i].index
        xys = Series(index=ks1, dtype=object)
        for k1, tck in df.ix[i]['EdgeSpline'].iteritems():
            try:
                xys[k1] = np.asarray(zip(*splev(u, tck)))
            except:
                # traceback.print_exc()
                pass

        for j1, k1 in enumerate(ks1):
            xys1 = xys[k1]

            # Do not compare to the same cell
            ks2 = list(ks1)
            del ks2[j1]

            # First find centroids that are the minimum distance apart
            c1 = np.asarray(cs[k1])
            cs2 = np.asarray(cs[ks2])
            js2 = (j2 for j2, c2 in enumerate(cs2) if norm(c1 - c2) < min_dist)

            # Then calculate the closest distance by comparing boundaries
            v = {}
            try:
                for j2 in js2:
                    k2 = ks2[j2]
                    xys2 = xys[ks2[j2]]
                    m = min([min(np.sqrt(np.sum((v1 - xys2) ** 2, axis=1)))
                             for v1 in xys1])
                    v[k2] = m
            except:
                # traceback.print_exc()
                pass
            s[i, k1] = v
    # Save in place to the original DataFrame
    df['Nearest'] = s


def preedit(df, d):
    """
    Verify the quality of each observation.

    args:
        df (DataFrame): must have columns 'BoundingBox', 'Centroid' and 
            'EdgeSpline'
        d (dict): input parameters

    """
    checkedges(df, shape=d['shape'], edge=d['edge'])
    findnearest(df, min_dist=d['min_dist'], num_points=d['num_points'])

