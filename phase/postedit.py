#!/usr/bin/env python

"""
This module is used to analyze cells after the initials steps of automated 
analysis followed by manual editing.

"""

import os
import sys
import traceback

import cPickle as pickle
import numpy as np

from pandas import DataFrame

from numpy.linalg import norm as norm
from scipy.optimize import leastsq
from scipy.spatial import Voronoi
from scipy.signal import argrelmin, argrelmax
from scipy.interpolate import splev
from scipy.ndimage.filters import gaussian_filter1d

from extract import getpoles_inertia, makebspline, splitborder, splineprops

sys.path.append(os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]))
import read

__author__ = "Charlie Wright"
__email__ = "charles.s.wright@gmail.com"

def calc_radius(c, ps):
    """
    Calculate the distance of each data points from the center (xc, yc)

    """
    return np.sqrt(np.sum((ps - c) ** 2, axis=1))


def calc_distance(c, ps):
    """
    Calculate the algebraic distance between the 2D points and the mean circle 
    centered at c = (xc, yc).

    """
    Ri = calc_radius(c, ps)
    return Ri - Ri.mean()


def calc_jacobian(c, ps):
    """
    Jacobian of func
    The axis corresponding to derivatives must be coherent with the col_deriv 
    option of leastsq

    """
    Ri = calc_radius(c, ps)
    df_dc = np.transpose(c - ps) / Ri
    return df_dc - df_dc.mean(axis=1)[:, np.newaxis]


def fit_circle(ps):
    """
    Fit data to circle using a least-squares fit with the Jacobian.

    """
    c0 = np.mean(ps, axis=0)
    c, _, infodict, _, _ = leastsq(calc_distance, c0, ps, Dfun=calc_jacobian,
                                   col_deriv=True, full_output=True)
    r = calc_radius(c, ps).mean()
    ss_err = np.sum(infodict['fvec'] ** 2)
    ss_tot = np.sum((ps - np.mean(ps, axis=0)) ** 2)
    Rsq = 1 - ss_err / ss_tot
    return c, r, Rsq


def point_in_poly(p, es):
    """
    Determine whether point is within closed region.

    """
    xp, yp = p
    n = len(es)
    inside = False
    xe1, ye1 = es[0]
    for i in range(n+1):
        xe2, ye2 = es[i % n]
        if yp > min(ye1, ye2):
            if yp <= max(ye1, ye2):
                if xp <= max(xe1, xe2):
                    if ye1 != ye2:
                        xints = (yp - ye1) * (xe2 - xe1) / (ye2 - ye1) + xe1
                    if xe1 == xe2 or xp <= xints:
                        inside = not inside
        xe1, ye1 = xe2, ye2
    return inside


def find_midline(es):
    """
    Find Voronoi diagram of input data points, keeping only internal points.

    """
    # FIXME: scikit image has a medial axis transform, which is designed to do
    # FIXME: exactly this (I believe). skeletonize give similar results, different algo.
    vor = Voronoi(es)
    vs = vor.vertices
    return np.asarray([v for v in vs if point_in_poly(v, es)])


def prune_midline(es, des, vs, max_angle_diff=0.05, min_chord_diff=1.9,
                  max_chord_diff=2., max_dist_mult=2.):
    """
    Remove branch points on Voronoi diagram, and reorder coordinates.

    """

    if np.all(es[0] - es[-1] < 1e-12):
        es = es[:-1]

    # Reject points whose corresponding inscribing circles have:
    #   1. Chord connecting tangent points not equal to diameter
    #   2. Angle between radius and tangent not pi/2
    vs2 = []
    for i, v in enumerate(vs):
        rs = np.sqrt(np.sum((es - v) ** 2, axis=1))
        j = (np.diff(np.sign(np.diff(rs))) > 0).nonzero()[0] + 1
        if np.any(j):
            j = np.hstack((j[0], j[1:][np.diff(j) > 1]))
        if len(j) > 1:
            j = j[np.argsort(rs[j])[:2]]
            e = es[j]
            r = rs.min()
            d = np.sqrt(np.sum(np.diff(e, axis=0) ** 2))
            test1 = min_chord_diff < d / r < max_chord_diff

            m1 = (e - v)[:, 1] / (e - v)[:, 0]
            m2 = des[j][:, 1] / des[j][:, 0]
            theta = np.abs(np.arctan((m1 - m2) / (1 + m1 * m2)))
            test2 = np.all(np.abs(theta - np.pi / 2) < max_angle_diff)

            if test1 and test2:
                vs2.append(v)

    # Reorder the coordinates according to both x- and y-axes
    xv2, yv2 = map(np.asarray, zip(*vs2))
    xv3 = xv2[np.argsort(xv2)]
    yv3 = yv2[np.argsort(xv2)]

    xv4 = xv3[np.argsort(yv3)]
    yv4 = yv3[np.argsort(yv3)]

    cs = np.asarray(zip(xv4, yv4))

    # Remove any points that are far away from any other points
    dist = np.sum(np.diff(cs, axis=0) ** 2, axis=1)
    max_dist = np.mean(dist[np.isfinite(dist)])
    return cs[dist < max_dist_mult * max_dist]


def orient_midline(es, cs, ps):
    """
    Orient the midline to run from stalked to swarmer poles.

    """

    # Get the current ends of the centerline
    e1s = cs[0], cs[-1]

    # Find the stalked and swarmer poles and reorient accordingly
    if norm(ps[0] - e1s[0]) > norm(ps[1] - e1s[0]):
        cs = cs[::-1]
    return cs


def extend_midline(es, cs, Cs, ps):
    """
    Extend the midline all the way to the edges of the cell (at either pole).

    """
    # Average step size between points along centerline
    step_size = np.sqrt(np.mean(np.sum(np.diff(cs, axis=0) ** 2, axis=1)))

    # Get the current ends of the centerline
    e1s = cs[0], cs[-1]

    # Extend each end from centerline to pole along the fitted circle
    ds = [[], []]
    for i, (((xC, yC), r), (xp, yp), (xe1, ye1)) in enumerate(zip(Cs, ps, e1s)):

        # Find where the circle intersects the edge, closest to the pole
        d = r ** 2 - np.sum((es - (xC, yC)) ** 2, axis=1)
        j = np.flatnonzero(np.diff(np.sign(d)))
        if not np.any(j): continue
        e2 = np.mean((es[j], es[j+1]), axis=0)
        xe2, ye2 = e2[np.argmin(np.sum((e2 - (xp, yp)) ** 2, axis=1))]

        # Find the limits over which to evaluate the circle
        xMin = min(xe1, xe2) - 1e-2
        xMax = max(xe1, xe2) + 1e-2
        yMin = min(ye1, ye2) - 1e-2
        yMax = max(ye1, ye2) + 1e-2

        # Evaluate the circle within those limits
        xa = np.arange(xMin, xMax, step_size)
        ya1 = yC + np.sqrt(r ** 2 - (xa - xC) ** 2)
        ya2 = yC - np.sqrt(r ** 2 - (xa - xC) ** 2)

        yb = np.arange(yMin, yMax, step_size)
        xb1 = xC + np.sqrt(r ** 2 - (yb - yC) ** 2)
        xb2 = xC - np.sqrt(r ** 2 - (yb - yC) ** 2)

        x = np.hstack((xa, xa, xb1, xb2))
        y = np.hstack((ya1, ya2, yb, yb))

        jMin = (x >= xMin) & (y >= yMin)
        jMax = (x <= xMax) & (y <= yMax)
        x = x[jMin & jMax]
        y = y[jMin & jMax]

        #if np.any(x):
        ds[i] = np.vstack(((xe1, ye1), zip(x, y), (xe2, ye2)))
        #else:
        #    ds[i] = np.vstack(((xe1, ye1), (xe2, ye2)))

    # Sort all the coordinates
    a = np.vstack((ds[0], cs, ds[1]))
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * 2)))
    _, idx = np.unique(b, return_index=True)

    # Reorder the coordinates according to both x- and y-axes
    xv2, yv2 = map(np.asarray, zip(*a[idx]))
    xv3 = xv2[np.argsort(xv2)]
    yv3 = yv2[np.argsort(xv2)]

    xv4 = xv3[np.argsort(yv3)]
    yv4 = yv3[np.argsort(yv3)]

    cs2 = np.asarray([v for v in zip(xv4, yv4) if point_in_poly(v, es)])

    # Reorient so that always runs from stalked pole to swarmer pole
    if norm(ps[0] - cs2[0]) > norm(ps[1] - cs2[0]):
        cs2 = cs2[::-1]
    return cs2


def get_width(es, cs, dcs):
    """
    Calculate the distance between opposite sides of the cell that are 
    connected by a normal line through the medial axis.

    """
    # Points along the perimeter
    xes, yes = map(np.asarray, zip(*es))
    ne = len(es)

    # Points along the centerline
    xcs, ycs = map(np.asarray, zip(*cs))
    dxcs, dycs = map(np.asarray, zip(*dcs))

    # Calculate normal vector
    ms = -dxcs / dycs
    bs = ycs - ms * xcs
    ws = np.ones_like(ms) * np.nan
    for i, (b, m, xc, yc) in enumerate(zip(bs, ms, xcs, ycs)):
        # Find intersection with edge
        d = yes - (m * xes + b)
        j = np.flatnonzero(np.diff(np.sign(d)))
        if not np.any(j): continue
        xn = np.mean((xes[j], xes[j+1]), axis=0)
        yn = np.mean((yes[j], yes[j+1]), axis=0)

        # Keep intersection closest to midline point
        k = np.argsort(np.sqrt((xc - xn) ** 2 + (yc - yn) ** 2))
        xn1, yn1 = xn[k[0]], yn[k[0]]

        # Keep next point that falls on opposite side of contour
        xn2, yn2 = xn[k[1]], yn[k[1]]

        # Calculate the width
        ws[i] = np.sqrt((xn1 - xn2) ** 2 + (yn1 - yn2) ** 2)
    ws[0], ws[-1] = 0., 0.
    return ws


def get_divn_plane(es, c, dc):
    """
    Get points on contour opposite to given point on midline.

    """
    # Points along the perimeter
    xes, yes = map(np.asarray, zip(*es))
    ne = len(es)

    # Points along the centerline
    xc, yc = c
    dxc, dyc = dc

    # Calculate normal vector
    m = -dxc / dyc
    b = yc - m * xc

    # Find intersection with edge
    d = yes - (m * xes + b)
    j = np.flatnonzero(np.diff(np.sign(d)))
    if not np.any(j):
        return np.asarray(((np.nan, np.nan), (np.nan, np.nan)))

    xn = np.mean((xes[j], xes[j+1]), axis=0)
    yn = np.mean((yes[j], yes[j+1]), axis=0)

    # Keep intersection closest to midline point
    k = np.argsort(np.sqrt((xc - xn) ** 2 + (yc - yn) ** 2))
    xn1, yn1 = xn[k[0]], yn[k[0]]

    # Keep next point that falls on opposite side of contour
    xn2, yn2 = xn[k[1]], yn[k[1]]

    return np.asarray(((xn1, yn1), (xn2, yn2)))


def find_min_width(ws, edges=0.25):
    """
    Calculate the minimum cell width.

    """
    n = len(ws)

    # Find relative  minima not near the cell edges
    j_mins = argrelmin(ws)[0]
    j_mins = j_mins[(j_mins > n * edges) & (j_mins < n * (1 - edges))]
    if np.any(j_mins):
        w_mins = ws[j_mins]
        k_min = np.argmin(w_mins)
        return j_mins[k_min], w_mins[k_min]
    else:
        return j_mins, np.nan


def find_max_width(ws):
    """
    Calculate the maximum cell width(s).

    """

    # Find max value of the width
    k = np.flatnonzero(np.isfinite(ws))
    j_max = np.argmax(ws[k])
    w_max = ws[k[j_max]]
    return j_max, w_max


def split_lengths(u, tck, j_min, xs, ps, l):
    """
    Find the lengths of the parent and nascent daughter cells.

    args:
        u (ndarray): 1D array giving paramaterization of the spline
        j_min (ndarray): index of division plane on midline
        tck (list): spline representation of midline
        xs (ndarray): coordinates along the midline
        ps (ndarray): coordinates of cell poles
        l (float): length of the entire cell

    returns:
        lo (float): length of the parent cell
        ln (float): length of the daughter cell

    """

    # Split the midline around the division plane
    uo, un = u[:j_min+1], u[j_min:]

    # Verify the identifications
    po, pn = ps
    xo = splev(uo[0], tck)
    do = np.min(norm(po - xo))
    dn = np.min(norm(pn - xo))
    if dn < do:
        uo, un = un, uo

    # Return the lengths of each cell
    lo = splineprops(uo, tck, ('Length', ))['Length']
    ln = splineprops(un, tck, ('Length', ))['Length']

    # Scale the results to be commensurate with the full length
    f = l / (lo + ln)
    return f * lo, f * ln


def get_length(u, tck):
    return splineprops(u, tck, ('Length', ))['Length']


def split_areas(u, tck, z, xs, ps, a):
    """
    Find the areas of the parent and nascent daughter cells.

    args:
        u (ndarray): 1D array giving paramaterization of the spline
        z (ndarray): coordinates of division plane on midline
        tck (list): spline representation of border
        xs (ndarray): coordinates along the border
        ps (ndarray): coordinates of cell poles
        a (float): area of the entire cell

    returns:
        ao (float): area of the parent cell
        an (float): area of the daughter cell

    """

    # Split the border into two contours, one for each new cell
    uo, un = splitborder(u, xs, z)

    # Identify the contours with the old and new cells
    xso = np.asarray(zip(*splev(uo, tck)))
    xsn = np.asarray(zip(*splev(un, tck)))

    # Verify the identifications
    po, pn = ps
    do = np.min([norm(po - v) for v in xso])
    dn = np.min([norm(pn - v) for v in xso])
    if dn < do:
        xso, xsn = xsn, xso

    # Make new periodic splines for each cell
    tcko = makebspline(xso, 0., True)[0]
    tckn = makebspline(xsn, 0., True)[0]

    # Return the area of each cell
    ao = splineprops(u, tcko, ('Area', ))['Area']
    an = splineprops(u, tckn, ('Area', ))['Area']

    # Scale the results to be commensurate with the full area
    f = a / (ao + an)
    return f * ao, f * an


def get_width_profile_old(s, num_points=500):
    """
    Get the width profile and associated variables with the original midline.

    args:
        s (Series): data read in from file for a single frame

    returns:
        d (dict): calculated values

    """

    # Evaluate the midline at discrete points starting at the old pole
    u = np.linspace(0., 1., num_points)

    d = {'Widths' : np.empty(0),
         'WidthMin' : np.nan,
         'WidthStalkedMax' : np.nan,
         'WidthSwarmerMax' : np.nan,
         'WidthStalkedMin' : np.nan,
         'WidthSwarmerMin' : np.nan,
         'AreaStalked' : np.nan,
         'AreaSwarmer' : np.nan,
         'LengthStalked' : np.nan,
         'LengthSwarmer' : np.nan,
         'LengthStalkedMin' : np.nan,
         'LengthSwarmerMin' : np.nan,
         'Radius' : np.nan,
         'RadiusStalked' : np.nan,
         'RadiusSwarmer' : np.nan}
    try:
        # Rename values from file
        a = s['Area']
        l = s['Length']
        po = s['StalkedPole']
        pn = s['SwarmerPole']

        # Get coordinates and first derivative of contour
        tck = s['EdgeSpline']
        es = np.asarray(zip(*splev(u, tck)))

        # Spline the midline
        tck4 = s['MidSpline']
        cs4 = np.asarray(zip(*splev(u, tck4)))
        dcs4 = np.asarray(zip(*splev(u, tck4, der=1)))
#
#         # Get the index of the stalked pole
#         jo = np.argmin(np.sum((es - po) ** 2, axis=1))
#
#         # Fit best-fit circle to entire cell
#         cre0 = fit_circle(cs4)
#         d['Radius'] = cre0[1]

        # Get the cell width and minimum and maximum cell widths
        ws4 = get_width(es, cs4, dcs4)
        d['Widths'] = ws4
        j4_min, w_min = find_min_width(ws4)
        d['WidthMin'] = w_min
        if np.any(j4_min):
            # Split contours according to minimum cell width
            wso4, wsn4 = ws4[:j4_min], ws4[j4_min:]

            # Find the maximum widths of either side
            jo4_max, wo_max = find_max_width(wso4)
            jn4_max, wn_max = find_max_width(wsn4)

            # Find the minimum widths of either side
            jo4_min, wo_min = find_min_width(wso4)
            jn4_min, wn_min = find_min_width(wsn4)

            if not np.any(jo4_min): jo4_min = np.nan
            if not np.any(jn4_min): jn4_min = np.nan

            # Get points on contour at division plane
            es_min = get_divn_plane(es, cs4[j4_min], dcs4[j4_min])

            # Split the cell areas and lengths
            ao, an = split_areas(u, tck, es_min, es, (po, pn), a)
            lo, ln = split_lengths(u, tck4, j4_min, es, (po, pn), l)

            # Find the lengths to either secondary minimum
            ro_min = float(jo4_min) / float(len(wso4))
            rn_min = 1. - float(jn4_min) / float(len(wsn4))

            lo_min = lo * ro_min
            ln_min = ln * rn_min
#
#             # Find best-fit circle to each cell half
#             cre1 = fit_circle(cs4[:j4_min])
#             cre2 = fit_circle(cs4[j4_min:])
        else:
            wo_max, wn_max = np.nan, np.nan
            wo_min, wn_min = np.nan, np.nan
            ao, an = np.nan, np.nan
            lo, ln = np.nan, np.nan
#             cre1 = (np.nan, np.nan)
#             cre2 = (np.nan, np.nan)
            lo_min, ln_min = np.nan, np.nan

        d['WidthStalkedMax'] = wo_max
        d['WidthSwarmerMax'] = wn_max
        d['WidthStalkedMin'] = wo_min
        d['WidthSwarmerMin'] = wn_min
        d['AreaStalked'] = ao
        d['AreaSwarmer'] = an
        d['LengthStalked'] = lo
        d['LengthSwarmer'] = ln
        d['LengthStalkedMin'] = lo_min
        d['LengthSwarmerMin'] = ln_min
#         d['RadiusStalked'] = cre1[1]
#         d['RadiusSwarmer'] = cre2[1]
    except Exception:
        pass
    return d


def get_width_profile_new(s, num_points=500):
    """
    Get the width profile and associated variables with a recalculated midline.

    args:
        s (Series): data read in from file for a single frame

    returns:
        d (dict): calculated values

    """

    # Evaluate the midline at discrete points starting at the old pole
    u = np.linspace(0., 1., num_points)

    d = {'Radius' : np.nan,
         'RadiusStalked' : np.nan,
         'RadiusSwarmer' : np.nan,
         'Widths' : np.empty(0),
         'WidthMin' : np.nan,
         'WidthStalkedMax' : np.nan,
         'WidthSwarmerMax' : np.nan,
         'WidthStalkedMin' : np.nan,
         'WidthSwarmerMin' : np.nan,
         'Area' : np.nan,
         'AreaStalked' : np.nan,
         'AreaSwarmer' : np.nan,
         'Length' : np.nan,
         'LengthStalked' : np.nan,
         'LengthSwarmer' : np.nan,
         'LengthStalkedMin' : np.nan,
         'LengthSwarmerMin' : np.nan,
         'MidSpline' : []}
    try:
        # Get coordinates and first derivative of contour
        tck = s['EdgeSpline']
        es = np.asarray(zip(*splev(u, tck)))
        des = np.asarray(zip(*splev(u, tck, der=1)))

        # Find a crude approximation of the cell poles
        ps = (s['StalkedPole'], s['SwarmerPole'])

        # Approximate the midline by taking the Voronoi diagram
        vs = find_midline(es)

        # Prune the Voronoi diagram to remove branch points
        cs0 = prune_midline(es, des, vs)

        # Reorient the midline from stalked to swarmer pole
        cs1 = orient_midline(es, cs0, ps)

        # Find approximate width along the medial axis
        ws1 = get_width(es, cs1[:-1], np.diff(cs1, axis=0))

        # Find the radius of curvature of the entire cell
        cre0 = fit_circle(cs1)

        # Get the minimum cell width
        j1_min, w1_min = find_min_width(ws1)
        if np.any(j1_min):
            # Split contours according to minimum cell width
            wso1, wsn1 = ws1[:j1_min], ws1[j1_min:]

            # Find the maximum widths of either side
            jo1_max, wo1_max = find_max_width(wso1)
            jn1_max, wn1_max = find_max_width(wsn1)

            # Only keep exterior points within 98% of maximum width
            ko1_max = np.flatnonzero(wso1[jo1_max:] > wo1_max * 0.75)
            ko1 = np.hstack((range(jo1_max), jo1_max + ko1_max))
            kn1_max = np.flatnonzero(wsn1[:jn1_max] > wn1_max * 0.75)
            kn1 = np.hstack((kn1_max, range(jn1_max, len(wsn1))))

            # Fit circles to either cell
            cre1 = fit_circle(cs1[ko1])
            cre2 = fit_circle(cs1[j1_min + kn1])
        else:
            cre1, cre2, = cre0, cre0

        # Extend the midline in a way that maintains curvature
        cs2 = extend_midline(es, cs1, (cre1[:2], cre2[:2]), ps)

        # Spline the new midline
        tck4 = makebspline(cs2, smoothing=0.1)[0]
        d['MidSpline'] = tck4
        cs4 = np.asarray(zip(*splev(u, tck4)))
        dcs4 = np.asarray(zip(*splev(u, tck4, der=1)))

        # Get the correct cell poles
        po, pn = cs4[0], cs4[-1]

        # Get the index of the stalked pole
        jo = np.argmin(np.sum((es - po) ** 2, axis=1))

        # Overall cell area and length
        a = splineprops(u, tck, 'Area')['Area']
        d['Area'] = a
        l = splineprops(u, tck4, 'Length')['Length']
        d['Length'] = l

        # Fit best-fit circle to entire cell
        cre0 = fit_circle(cs4)
        d['Radius'] = cre0[1]

        # Get the cell width and minimum and maximum cell widths
        ws4 = get_width(es, cs4, dcs4)
        d['Widths'] = ws4
        j4_min, w_min = find_min_width(ws4)
        d['WidthMin'] = w_min
        if np.any(j4_min):
            # Split contours according to minimum cell width
            wso4, wsn4 = ws4[:j4_min], ws4[j4_min:]

            # Find the maximum widths of either side
            jo4_max, wo_max = find_max_width(wso4)
            jn4_max, wn_max = find_max_width(wsn4)

            # Find the minimum widths of either side
            jo4_min, wo_min = find_min_width(wso4)
            jn4_min, wn_min = find_min_width(wsn4)

            if not np.any(jo4_min): jo4_min = np.nan
            if not np.any(jn4_min): jn4_min = np.nan

            # Get points on contour at division plane
            es_min = get_divn_plane(es, cs4[j4_min], dcs4[j4_min])

            # Split the cell areas and lengths
            ao, an = split_areas(u, tck, es_min, es, (po, pn), a)
            lo, ln = split_lengths(u, tck4, j4_min, es, (po, pn), l)

            # Find the lengths to either secondary minimum
            ro_min = float(jo4_min) / float(len(wso4))
            rn_min = 1. - float(jn4_min) / float(len(wsn4))

            lo_min = lo * ro_min
            ln_min = ln * rn_min

            # Find best-fit circle to each cell half
            cre1 = fit_circle(cs4[:j4_min])
            cre2 = fit_circle(cs4[j4_min:])
        else:
            wo_max, wn_max = np.nan, np.nan
            wo_min, wn_min = np.nan, np.nan
            ao, an = np.nan, np.nan
            lo, ln = np.nan, np.nan
            cre1 = (np.nan, np.nan)
            cre2 = (np.nan, np.nan)
            lo_min, ln_min = np.nan, np.nan

        d['RadiusStalked'] = cre1[1]
        d['RadiusSwarmer'] = cre2[1]
        d['WidthStalkedMax'] = wo_max
        d['WidthSwarmerMax'] = wn_max
        d['WidthStalkedMin'] = wo_min
        d['WidthSwarmerMin'] = wn_min
        d['AreaStalked'] = ao
        d['AreaSwarmer'] = an
        d['LengthStalked'] = lo
        d['LengthSwarmer'] = ln
        d['LengthStalkedMin'] = lo_min
        d['LengthSwarmerMin'] = ln_min
    except Exception:
        pass
    return d


def get_width_profiles(input_dir, num_points=500):
    """
    Calculate the width profiles from the saved cell midlines.

    args:
        df (DataFrame): contains requested coordinates and trajectories

    """

    # Read the necessary variables in from file
    old_keys = ('Area', 'EdgeSpline', 'Length', 'MidSpline', 'Perimeter',
                'StalkedPole', 'SwarmerPole', 'Trace')
    df = read.makedataframe(input_dir, old_keys)

    # Evaluate each frame independently
    for i, v in enumerate(df.index):
        d = get_width_profile_new(df.ix[v], num_points)
        if i == 0:
            new_keys = d.keys()
            for k in new_keys:
                df[k] = None
        for k in new_keys:
            df[k].ix[v] = d[k]

    # Save each Series to file
    for k in new_keys:
        df[k].to_pickle(os.path.join(input_dir, k + '.pickle'))


def fftcoefs(xs, n=20):
    """
    Finds the Fourier series coefficients for (x, y) coordinates representing a 
    closed curve. Converts to polar coordinates then computes the 1D discrete 
    Fourier-transform of r(theta). Returns the truncated coefficients of the 
    decomposition.

    For example, to calculate the fit from the coefficients and compare:
        >>> n = 20
        >>> d = fftcoefs(xs, n=n)
        >>> m = len(xs)
        >>> a = np.zeros(m, dtype=complex)
        >>> for k, v in d.items():
        >>>     a[k] = v
        >>> z = np.fft.ifft(a * m)
        >>>
        >>> x, y = zip(*xs)
        >>> p, q = z.real, z.imag
        >>> p += np.mean(x)
        >>> q += np.mean(y)
        >>>
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111, aspect=1)
        >>> ax.plot(x, y, 'b', label='data')
        >>> ax.plot(p, q, 'r', label='fit')
        >>> ax.legend()
        >>> ax.set_title(r'$n = %d, R^2 = %0.4f$' % (n, np.sum((xs - zip(p, q)) ** 2)))
        >>> fig.show()

    args:
        xs (ndarray): array of coordinates

    kwargs:
        n (int): number of coefficients to keep

    returns:
        d (dict): keys = coefficient indexes, values = coefficient values

    """
    ys = map(np.asarray, zip(*xs))
    z = ys[0] + ys[1] * 1j
    f = np.fft.fft(z)
    f /= float(len(z))
    d = {}
    for i in xrange(-n, 0):
        d[i] = f[i]
    for i in xrange(1, n+1):
        d[i] = f[i]
    return d


def ifftcoefs(d, c=(0, 0), m=500):
    """
    Converts Fourier coefficients back into real coordinates.

    args:
        d (dict): results of fftcoefs(xs)

    kwargs:
        c (list): centroid of input data,  np.mean(xs, axis=0)
        m (int): length of input data, len(xs)

    returns:
        ys (array): real coordinates corresponding to Fourier fit

    """
    a = np.zeros(m, dtype=complex)
    for k, v in d.items():
        a[k] = v
    z = np.fft.ifft(a * m)
    x, y = z.real + c[0], z.imag + c[1]
    return np.asarray(zip(x, y))


def get_fft_coefs(input_dir, num_points=500, num_coefs=20):
    """
    Calculate the Fourier shape descriptors from the saved cell contours.

    args:
        df (DataFrame): contains requested coordinates and trajectories

    """

    # Read the necessary variables in from file
    df = read.makedataframe(input_dir, ('EdgeSpline', ))
    df['FourierCoef'] = None

    # Evaluate each frame independently
    u = np.arange(0., 1., 1. / num_points)
    for i, v in enumerate(df.index):
        tck = df['EdgeSpline'].ix[v]
        try:
            xs = splineprops(u, tck, 'Coordinates')['Coordinates']
            d = fftcoefs(xs, n=num_coefs)
        except Exception:
            d = {}
        df['FourierCoef'].ix[v] = d

    # Save new Series to file
    df['FourierCoef'].to_pickle(os.path.join(input_dir, 'FourierCoef.pickle'))


def smooth_widths(input_dir, sigma=2, cutoff=0.02):
    """
    Smooth the widths and recalculate the shape parameters.

    args:
        df (DataFrame): contains requested coordinates and trajectories

    """

    # Read the necessary variables in from file
    df = read.makedataframe(input_dir, ('Area', 'EdgeSpline', 'Length',
                                        'MidSpline', 'StalkedPole',
                                        'SwarmerPole', 'Widths'))

    # Save new and overwrite existing shape variables
    new_keys =('AreaStalked', 'LengthStalkedMaxPole', 'LengthStalkedMin',
               'LengthStalkedMaxCenter', 'LengthMin', 'LengthSwarmerMaxCenter',
               'LengthSwarmerMin', 'LengthSwarmerMaxPole', 'WidthStalkedMaxPole',
               'WidthStalkedMin', 'WidthStalkedMaxCenter', 'WidthMin',
               'WidthSwarmerMaxCenter', 'WidthSwarmerMin', 'WidthSwarmerMaxPole',
               'WidthStalkedMax', 'WidthSwarmerMax', 'WidthMax', 'LengthStalkedMax',
               'LengthSwarmerMax', 'LengthMax', 'WidthsSmoothed')
    for k in new_keys:
        df[k] = None

    # Evaluate each frame independently
    for v in df.index:
        # Assign default values
        y = np.ndarray(0)
        ast = np.nan
        lstmaxp = np.nan
        lstmin = np.nan
        lstmaxc = np.nan
        lst = np.nan
        lswmaxc = np.nan
        lswmin = np.nan
        lswmaxp = np.nan
        wstmaxp = np.nan
        wstmin = np.nan
        wstmaxc = np.nan
        wmin = np.nan
        wswmaxc = np.nan
        wswmin = np.nan
        wswmaxp = np.nan
        wstmax = np.nan
        wswmax = np.nan
        wmax = np.nan
        lstmax = np.nan
        lswmax = np.nan
        lmax = np.nan

        try:
            s = df.ix[v]

            # Load variables from file
            a = s['Area']
            l = s['Length']
            ps = (s['StalkedPole'], s['SwarmerPole'])
            w = s['Widths']
            tcke = s['EdgeSpline']
            tckm = s['MidSpline']

            # Reconstruct cell contour and midline
            n = len(w)
            u = np.linspace(0., 1., n)
            es = np.asarray(zip(*splev(u, tcke)))
            ms = np.asarray(zip(*splev(u, tckm)))
            dms = np.asarray(zip(*splev(u, tckm, der=1)))

            # Smooth the width profile
            y = gaussian_filter1d(w, sigma=sigma)

            # Truncate the width profile near the poles
            #y = y[m:-m]

            # Find all relative minima and maxima
            m = int(n * cutoff)
            imin = np.hstack(argrelmin(y, order=m))
            imax = np.hstack(argrelmax(y, order=m))

            if np.any(imin):
                # Find the smallest (primary) minimum
                jmin = imin[np.argmin(y[imin])]

                # Assign the minimum width
                wmin = y[jmin]

                # Find the normal line that split the cell at the minimum width
                esmin = get_divn_plane(es, ms[jmin], dms[jmin])

                # Split into stalked and swarmer parts
                ast, _ = split_areas(u, tcke, esmin, es, ps, a)
                lst = get_length(u[:jmin+1], tckm)

                # Sort the other minima to the stalked or swarmer part
                istmin = imin[imin < jmin]
                iswmin = imin[imin > jmin]

                # Sort the maxima to the stalked or swarmer part
                istmax = imax[imax < jmin]
                iswmax = imax[imax > jmin]

                if np.any(istmin):
                    # Find the smallest of the stalked minima
                    jstmin = istmin[np.argmin(y[istmin])]

                    # Assign the minimum stalked width
                    wstmin = y[jstmin]

                    # Find the length at the minimum stalked width
                    lstmin = get_length(u[:jstmin+1], tckm)

                    # Sort the stalked maxima to the pole or center
                    istmaxp = istmax[istmax < jstmin]
                    istmaxc = istmax[istmax > jstmin]

                    # Find the largest of the stalked maxima
                    jstmaxp = istmaxp[np.argmax(y[istmaxp])]
                    jstmaxc = istmaxc[np.argmax(y[istmaxc])]

                    # Assign the stalked pole width and length maxima
                    wstmaxp = y[jstmaxp]
                    lstmaxp = get_length(u[:jstmaxp+1], tckm)

                    # Assign the stalked center width and length maxima
                    wstmaxc = y[jstmaxc]
                    lstmaxc = get_length(u[:jstmaxc+1], tckm)
                else:
                    # Find the largest of the stalked maxima
                    jstmax = istmax[np.argmax(y[istmax])]

                    # Assign the stalked pole width and length maxima
                    wstmax = y[jstmax]
                    lstmax = get_length(u[:jstmax+1], tckm)
                if np.any(iswmin):
                    # Find the smallest of the swarmer minima
                    jswmin = iswmin[np.argmin(y[iswmin])]

                    # Assign the minimum swarmer width
                    wswmin = y[jswmin]

                    # Find the length at the minimum swarmer width
                    lswmin = get_length(u[:jswmin+1], tckm)

                    # Sort the swarmer maxima to the pole or center
                    iswmaxp = iswmax[iswmax > jswmin]
                    iswmaxc = iswmax[iswmax < jswmin]

                    # Find the largest of the swarmer maxima
                    jswmaxp = iswmaxp[np.argmax(y[iswmaxp])]
                    jswmaxc = iswmaxc[np.argmax(y[iswmaxc])]

                    # Assign the swarmer pole width and length maxima
                    wswmaxp = y[jswmaxp]
                    lswmaxp = get_length(u[:jswmaxp+1], tckm)

                    # Assign the swarmer center width and length maxima
                    wswmaxc = y[jswmaxc]
                    lswmaxc = get_length(u[:jswmaxc+1], tckm)
                else:
                    # Find the largest of the swarmer maxima
                    jswmax = iswmax[np.argmax(y[iswmax])]

                    # Assign the swarmer pole width and length maxima
                    wswmax = y[jswmax]
                    lswmax = get_length(u[:jswmax+1], tckm)
            else:
                # Find the largest of the maxima
                jmax = imax[np.argmax(y[imax])]

                # Assign the width and length maxima
                wmax = y[jmax]
                lmax = get_length(u[:jmax+1], tckm)
        except Exception as e:
            pass

        s['WidthsSmoothed'] = y
        s['AreaStalked'] = ast
        s['LengthStalkedMaxPole'] = lstmaxp
        s['LengthStalkedMin'] = lstmin
        s['LengthStalkedMaxCenter'] = lstmaxc
        s['LengthMin'] = lst
        s['LengthSwarmerMaxCenter'] = lswmaxc
        s['LengthSwarmerMin'] = lswmin
        s['LengthSwarmerMaxPole'] = lswmaxp
        s['WidthStalkedMaxPole'] = wstmaxp
        s['WidthStalkedMin'] = wstmin
        s['WidthStalkedMaxCenter'] = wstmaxc
        s['WidthMin'] = wmin
        s['WidthSwarmerMaxCenter'] = wswmaxc
        s['WidthSwarmerMin'] = wswmin
        s['WidthSwarmerMaxPole'] = wswmaxp
        s['WidthStalkedMax'] = wstmax
        s['WidthSwarmerMax'] = wswmax
        s['WidthMax'] = wmax
        s['LengthStalkedMax'] = lstmax
        s['LengthSwarmerMax'] = lswmax
        s['LengthMax'] = lmax

        df.ix[v] = s

    # Save new Series to file
    for k in new_keys:
        df[k].to_pickle(os.path.join(input_dir, k + '.pickle'))


def main(input_dir, d):
    """
    Analyze cells from the specified pickle files.

    args:
        d (dict): input parameters

    """
    get_fft_coefs(input_dir)
    get_width_profiles(input_dir)
    smooth_widths(input_dir)
