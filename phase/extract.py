#!/usr/bin/env python

"""
This module is used to extract data about bacterial cells from binary (segmented) 
images from phase-contrast data by fitting each cell to a spline.

"""

import traceback

import numpy as np

from numpy.linalg import norm
from scipy.integrate import trapz
from scipy.interpolate import *
from scipy.signal import argrelmin, argrelmax
from skimage.measure import find_contours

__author__ = "Charlie Wright"
__email__ = "charles.s.wright@gmail.com"

def movingaverage(data, span):
    """
    Calculate a moving average over a periodic data set.

    args:
        data (ndarray): input data vector
        span (int): size of window

    returns:
        ndarray: smoothed vector of same shape as input

    """
    window = np.ones(int(span)) / float(span)
    periodic = np.hstack((data[-span/2+1:], data, data[:span/2]))
    return np.convolve(periodic, window, 'valid')

def makebspline(xs, smoothing=0.1, per=False, degree=5, alpha=1., error=1.):
    """
    Fit coordinates to smoothing B-spline curve.

    args:
        xs (ndarray): array of coordinates to fit

    kwargs:
        smoothing (int): value of spline smoothing
        per (bool): if True, use periodic conditions to create closed spline
        degree (int): degree of the interpolating B-spline (1 <= degree <= 5)
        alpha (float): determines parameterization method (0 = uniformly spaced method, 1 = chord length method, otherwise = centripetal method)
        error (float): error on the precision of the input measurements

    returns:
        tcku (tuple): spline given as [knots, coefs, degree], parameterization

    """

    # First and last points must match
    if per and not (xs[0] == xs[-1]).all():
        xs = np.concatenate((xs, [xs[0]]))
    m = len(xs)

    if alpha > 0:   # Use chord length or centripetal method
        # Find distance along contour from 0 to 1
        d = np.sqrt(np.sum(np.diff(xs, axis=0) ** 2, axis=1))
        d = np.cumsum(np.hstack((0., d)))
        d /= d[-1]
        u = d ** alpha
    else:           # Use uniformly spaced method
        u = np.linspace(0., 1., m)

    # Smoothing factor depends on number of data points
    s = smoothing * (m - np.sqrt(2 * m))

    # Set the weights determined by the errors
    w = np.ones_like(u) / error

    # Return spline representation ([knots, coefs, degree], parameterization)
    return splprep(zip(*xs), u=u, w=w, s=s, per=per, k=degree)

def splineprops(u, tck, props):
    """
    Calculate properties of parametrically-defined spline curve:
    1. Coordinates: coordinates evaluated at discrete values determined by u
    2. Derivative:  coordinates differentiated with respect to u
    3. Length: length along the curve (line integral)
    4. Perimeter: same as length
    5. Area: area inside closed curve (surface integral)
    6. Centroid: geometric centroid (surface integral)
    7. Curvature: curvature

    args:
        u (ndarray): 1D array giving parameterization of the spline
        tck (list): spline given as knots, coefs, degree

    kwargs:
        props (tuple): names of properties to calculate

    returns:
        dict: dictionary of results, with keys corresponding to 'props'

    """

    # Properties input must be list-like
    if isinstance(props, basestring):
        props = (props, )

    # Evaluate the curve and its derivatives
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)

    results = dict.fromkeys(props)
    for k in props:
        if k == 'Coordinates': # (x, y) coordinates of the spline
            v = np.asarray(zip(x, y))
        elif k == 'Derivative': # (dx/du, dy/du) coordinates
            v = np.asarray(zip(dx, dy))
        elif k == 'Length' or k == 'Perimeter': # arc length
            v = trapz(np.sqrt(dx ** 2 + dy ** 2), u)
        elif k == 'Area' or k == 'Centroid':
            A = trapz(x * dy, u)
            if k == 'Area': # area inside closed curve
                v = np.abs(A)
            elif k == 'Centroid': # geometric centroid
                M_y = trapz(x * x * dy, u)
                M_x = trapz(y * y * dx, u)
                v = 0.5 * np.abs(np.asarray((M_y / A,  M_x / A)))
        elif k == 'Curvature': # curvature of the spline
            ddx, ddy = splev(u, tck, der=2)
            n = dx * ddy - dy * ddx
            d = np.sqrt(dx ** 2 + dy ** 2) ** 3
            v = n / d
        else:
            v = None
        results[k] = v
    return results

def fftcoefs(xs, c, n=20):
    """
    Finds the Fourier series coefficients for (x, y) coordinates representing a closed curve. Converts to polar coordinates then computes the 1D discrete Fourier-transform of r(theta). Returns the truncated coefficients of the decomposition.

    For example, to calculate the fit from the coefficients and compare:
        >>> n = 20
        >>> a = fftcoefs(xs, c, n=n)
        >>> r = np.sqrt(np.sum((xs - c) ** 2, axis=1))
        >>> x, y = map(np.asarray, zip(*d['Coordinates'] - d['Centroid']))
        >>> theta = np.arctan2(y, x)
        >>> f = np.fft.irfft(c, n=len(r))
        >>> plt.polar(theta, r, '.')
        >>> plt.polar(theta, f, '-')
        >>> plt.title('n = %d, R = %0.2f' % (n, np.sqrt(np.sum((r - f) ** 2))))
        >>> plt.show()

    args:
        xs (ndarray): array of coordinates
        c (ndarray): center of mass along each dimension in xs

    kwargs:
        n (int): number of coefficients to keep

    """
    r = np.sqrt(np.sum((xs - c) ** 2, axis=1))
    return np.fft.rfft(r)[:n] / float(len(r))

def getpoles_inertia(xs, c):
    """
    Find poles of distal ends of elongated cells. The mother pole is the upstream coordinate pair, assuming flow from the positive y-direction. The poles are located by finding the points of maximal moment of inertia on opposite sides of the centroid.

    args:
        xs (ndarray): array of coordinates along border
        c (ndarray): center of mass along each dimension in xs

    returns:
        po (tuple): old pole
        pn (tuple): new pole

    """

    # Remove repeated values in the (periodic) input array
    if np.all(np.abs(xs[0] - xs[-1]) < 1e-10):
        xs = xs[:-1]

    # Calculate moments of inertia for each coordinate
    js = [np.sum(np.sqrt(np.sum((x - xs) ** 2, axis=1))) for x in xs]
    js = np.asarray(js) / len(js)

    # Order pixels from highest to lowest moment of inertia
    xs2 = xs[np.argsort(js)][::-1]

    # The 1st pole is found at the pixel of largest moment of inertia
    p = xs2[0]

    # The 2nd pole is found on the opposite side of the midpoint with the
    # additional constraint that the two poles not be vertically aligned
    # (which would result in an ambiguous case of mother vs. daughter)
    if len(xs2) > 1:
        for x in xs2[1:]:
            if norm(x - p) > norm(c - p) and x[0] != p[0]:
                break
    else:
        # Replicate if only one point
        x = p
    ps = (p, x)

    # The new pole is assumed to be downstream of the old pole
    if ps[1][1] > ps[0][1]:
        pn, po = ps
    else:
        po, pn = ps
    return po, pn

def getpoles_curvature(tck_k, tck_x, c):
    """
    Find poles of distal ends of elongated cells. The mother pole is the upstream coordinate pair, assuming flow from the positive y-direction. The poles are first located by finding the points of maximal positive curvature on opposite sides of the centroid.

    args:
        tck_k (list): spline of curvature given as knots, coefs, degree
        tck_x (list): spline of coordinates given as knots, coefs, degree
        c (ndarray): center of mass along each dimension in xs

    returns:
        po (tuple): old pole
        pn (tuple): new pole

    """

    # Find all inflection points of the curvature
    u_ip = sproot(splder(tck_k), mest=100)

    # Keep only points that are convex down
    u_cd = u_ip[splev(u_ip, tck_k, der=2) < 0]

    # Order points from highest to lowest moment of curvature
    js = np.argsort(splev(u_cd, tck_k))[::-1]
    xs = np.asarray(zip(*splev(u_cd[js], tck_x)))

    # The 1st pole is found at the pixel of largest moment of inertia
    p = xs[0]

    # The 2nd pole is found on the opposite side of the midpoint with the
    # additional constraint that the two poles not be vertically aligned
    # (which would result in an ambiguous case of mother vs. daughter)
    if len(xs) > 1:
        for x in xs[1:]:
            if norm(x - p) > norm(c - p) and x[0] != p[0]:
                break
    else:
        # Replicate if only one point
        x = p
    ps = (p, x)

    # The new pole is assumed to be downstream of the old pole
    if ps[1][1] > ps[0][1]:
        pn, po = ps
    else:
        po, pn = ps
    return po, pn

def splitborder(u, xs, ps, num_points=None):
    """
    Split the closed B-spline curve at the specified points (must supply two). Return two new parameterization vectors that start at the one point and run in opposite directions around the spline to end at the second point.

    args:
        u (ndarray): 1D array giving parameterization of the spline
        xs (ndarray): array of coordinates along close spline curve
        ps (tuple): points around which to split the curve

    kwargs:
        num_points (int): number of points to force each output to

    returns:
        u1 (ndarray): 1st split of the parameterization vector u
        u2 (ndarray): 2nd split of the parameterization vector u

    """

    # Remove repeated values in the (periodic) input array
    if np.all(np.abs(xs[0] - xs[-1]) < 1e-10):
        xs = xs[:-1]
    n = len(xs)

    # Find the indexes of the poles in the input border
    k1, k2 = np.sort([np.argmin([norm(v) for v in p - xs]) for p in ps])

    # Create two new monotonically increasing vectors that encircle the cell
    if num_points is None:
        u1 = u[k1:k2]
        u2 = u[np.mod(np.arange(k2, k1+n), n)]
    else:
        # Different points, but in the same domain (0, 1) as the original
        u1 = np.linspace(u[k1], u[k2], num_points)
        u2 = np.linspace(u[k2], u[k1] + 1., num_points)
        u2[u2 > 1.] -= 1.

    # Return oriented vectors that each run from first to second point
    return u1, u2[::-1]

def getmidline(u, tck, xs, ps, num_points=100):
    """
    Calculate the midline of the closed curve.

    args:
        u (ndarray): 1D array giving paramaterization of the spline
        tck (list): spline representation given as (knots, coefs, degree)
        xs (ndarray): coordinates of points along the border
        ps (ndarray): positions of opposite poles

    kwargs:
        num_points(int): number of points used when resplining contours

    returns:
        tck2 (tuple): spline representation given as (knots, coefs, degree)

    """

    # Split the border into two contours
    u1, u2 = splitborder(u, xs, ps, num_points=num_points)

    # Evaluate each pole-to-pole B-spline
    xs1 = np.asarray(zip(*splev(u1, tck)))
    xs2 = np.asarray(zip(*splev(u2, tck)))

    # Calculate a spline representation of the midline
    return makebspline((xs1 + xs2) / 2, smoothing=0., per=False)[0]

def getnormallength(xs, dxs, span=3):
    """
    Calculate the length of the normal vector starting from each point on the edge of a closed curve and terminating at the first intersection with another point on the curve.

    args:
        xs (ndarray): coordinates along the curve
        dxs (ndarray): derivatives (of parametric spatial coordinates)

    kwargs:
        span (int): size of moving average filter to apply to final result

    returns:
        ds (ndarray): distances of each line connecting two points on the curve

    """

    # Remove repeated values in the (periodic) input array
    if np.all(np.abs(xs[0] - xs[-1]) < 1e-10):
        xs = xs[:-1]
        dxs = dxs[:-1]

    # Index of all coordinates to loop over
    n = len(xs)
    j = np.arange(n)

    # Remove immediately-adjacent points
    a = 25
    h = -a + j[2 * a + 1:]

    # Average over small window of adjacent points to calculate the distance
    r = np.arange(-4, 5)

    ds = np.ones(n) * np.nan
    for i, (x, dx) in enumerate(zip(xs, dxs)):
        # Calculate the slope of the normal vector
        m = -dx[0] / dx[1]

        # Remove neighboring points from the calculation
        k = np.take(j, i + h, mode='wrap')

        # Find the point that comes closest to intersecting the normal vector
        z = np.sum([-m, 1] * (xs[k] - x), axis=1)
        y = np.abs(z - 0.)
        p = argrelmin(y, order=10, mode='wrap')
        if np.any(p):
            q = k[p][y[p] < 5]
            if np.any(q):
                if len(q) == 1:
                    q = q[0]
                else:
                    q = q[np.argmin(np.sqrt(np.sum((xs[q] - x) ** 2, axis=1)))]

                # Calculate the distance between this point and its neighbors
                v = (xs[np.mod(q + r, n)] - x) ** 2
                ds[i] = np.mean(np.sqrt(np.sum(v, axis=1)))
    return movingaverage(ds, span)

def getdivnplane(xs, c, ps, ds, w, xsm, cutoff=0.5):
    """
    Find the point of invagination, the minimum distance between opposite points in the middle half of the cell. Must be the specified distance greater than the maximum distance between opposite points.

    args:
        xs (ndarray): coordinates along the border
        c (ndarray): centroid of border
        ps (ndarray): coordinates of cell poles
        ds (ndarray): distances of normal vectors contained within cell
        w (ndarray): average cell width
        xsm (ndarray): coordinates along midline

    kwargs:
        cutoff (float): maximum deviation from mean cell width

    returns:
        z (ndarray): coordinates of division plane on each border and midline

    """

    # Remove repeated values in the (periodic) input array
    if np.all(np.abs(xs[0] - xs[-1]) < 1e-10):
        xs = xs[:-1]
    n = len(ds)
    js = np.arange(n)

    # Find the minima (set threshold to half of mean width)
    k = ds < w * cutoff
    p = argrelmin(ds[k], order=10, mode='wrap')
    if np.any(p[0]):
        ms = js[k][p]
    else:
        ms = np.ones(0, dtype=bool)

    # Must be closer to the centroid than to either pole
    po, pn = ps
    dpo = np.sqrt(np.sum((xs[ms] - po) ** 2, axis=1))
    dpn = np.sqrt(np.sum((xs[ms] - pn) ** 2, axis=1))
    dc = np.sqrt(np.sum((xs[ms] - c) ** 2, axis=1))
    ms = ms[(dc < dpo) & (dc < dpn)]

    # There must be a corresponding point on the opposite side
    k1, k2 = np.sort([np.argmin([norm(v) for v in p - xs]) for p in ps])
    js1 = js[k1:k2]
    js2 = js[np.mod(np.arange(k2, k1+n), n)]
    ms1 = [v for v in ms if v in js1]
    ms2 = [v for v in ms if v in js2]

    # The distance between points on opposite sides must be small enough
    z1, z2 = None, None
    d = 10 * w
    for m1 in ms1:
        for m2 in ms2:
            e = np.sqrt(np.sum((xs[m1] - xs[m2]) ** 2))
            if e < w * cutoff and e < d:
                d = e
                z1 = xs[m1]
                z2 = xs[m2]

    if z1 is not None and z2 is not None:
        # Find distance on midline closest to each invagination on border
        h1 = np.argmin([norm(v) for v in z1 - xsm])
        h2 = np.argmin([norm(v) for v in z2 - xsm])

        # Average these indexes
        h = (h1 + h2) / 2

        z = np.concatenate(((z1, ), (xsm[h], ), (z2, )))
    else:
        z = np.ones((3, 2)) * np.nan
    return z

def splitlengths(u, z, tck, xs, ps, l):
    """
    Find the lengths of the parent and nascent daughter cells.

    args:
        u (ndarray): 1D array giving paramaterization of the spline
        z (ndarray): coordinates of division plane on each border and midline
        tck (list): spline representation of midline
        xs (ndarray): coordinates along the midline
        ps (ndarray): coordinates of cell poles
        l (float): length of the entire cell

    returns:
        lo (float): length of the parent cell
        ln (float): length of the daughter cell

    """

    # Split the midline around the division plane
    j = np.argmin([norm(z[1] - v) for v in xs])
    uo, un = u[:j+1], u[j:]

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

def splitareas(u, z, tck, xs, ps, a):
    """
    Find the areas of the parent and nascent daughter cells.

    args:
        u (ndarray): 1D array giving paramaterization of the spline
        z (ndarray): coordinates of division plane on each border and midline
        tck (list): spline representation of border
        xs (ndarray): coordinates along the border
        ps (ndarray): coordinates of cell poles
        a (float): area of the entire cell

    returns:
        ao (float): area of the parent cell
        an (float): area of the daughter cell

    """

    # Split the border into two contours, one for each new cell
    uo, un = splitborder(u, xs, (z[0], z[2]))

    # Identify the contours with the old and new cells
    xso = splineprops(uo, tck, ('Coordinates', ))['Coordinates']
    xsn = splineprops(un, tck, ('Coordinates', ))['Coordinates']

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

def getboundingbox(bw):
    """
    Return the bounding box of the objects in the binary input image.

    args:
        bw (ndarray): binary image

    returns:
        b (tuple): bounding box, given as (min(x), min(y), max(x), max(y))

    """
    b = [0] * 4
    for i, v in enumerate(np.nonzero(bw)):
        b[i] = min(v)
        b[i+2] = max(v) + 1
    return tuple(b)

def getcontours(bw, single=True):
    """
    Get the contour(s) of the objects in the binary image.

    args:
        bw (ndarray): binary image

    kwargs:
        single (bool): if True, just return the largest contour

    returns:
        contours, as list of arrays if single is False or as single array if True

    """
    cs = find_contours(~bw, level=0.5)
    if single:
        results = cs[np.argmax([len(c) for c in cs])]
    else:
        results = cs
    return results

def preedit(labeled, d):
    """
    Find iso-contours along cell boundaries, and return them in a closed
    B-spline form. Get properties of 2D surface described by closed B-curve that are time-independent (i.e., that can be calculated from one frame).

    args:
        labeled (ndarray): labeled binary image
        d (dict): input parameters

    returns:
        results (list): list of dictionaries

    """

    # The distinct cells should already be labeled (0 is the background)
    labels = list(np.unique(labeled[labeled > 0]))

    # Initialize the data dictionaries
    results = []
    for i, v in enumerate(labels):
        r = {'Area' : np.nan,
             'BoundingBox' : (0, 0, 0, 0),
             'Centroid' : np.ones((1, 2)) * np.nan,
             'Curvature' : [],
             'EdgeSpline' : [],
             'FourierFit' : np.zeros(d['num_coefs']),
             'Label' : v,
             'Length' : np.nan,
             'MidSpline' : [],
             'Perimeter' : np.nan,
             'StalkedPole' : np.ones((1, 2)) * np.nan,
             'SwarmerPole' : np.ones((1, 2)) * np.nan}
        results.append(r)

    # Loop over each cell to ensure that the data match up with the labeled images
    for i, (r, v) in enumerate(zip(results, labels)):
        try:
            # Create a binary image with the selected cell and find its contour
            bw = labeled == v
            c = getcontours(bw)

            # Find the bounding box of the cell
            r['BoundingBox'] = getboundingbox(bw)

            # Convert to spline and calculate properties of cell from this curve
            tck = makebspline(c, smoothing=d['smoothing'], degree=d['degree'],
                              per=True, alpha=d['alpha'])[0]

            # Recast in terms of chord length parameterization
            n = int(d['coverage'] * len(c))
            x = np.asarray(zip(*splev(np.arange(0., 1., 1 / float(n)), tck)))
            r['EdgeSpline'] = makebspline(x, smoothing=0., degree=d['degree'],
                                          per=True, alpha=1.)[0]

            # Get requested data values
            u = np.linspace(0., 1., n)
            e = splineprops(u, r['EdgeSpline'], ('Coordinates', 'Area',
                            'Perimeter', 'Centroid', 'Curvature'))
            r['Area'], r['Perimeter'] = e['Area'], e['Perimeter']

            # Interpolate curvature using a B-spline with minimal smoothing
            # (this will reduce the knot number with affecting the values)
            # and degree = 4 (to allow root finding of the derivative)
            r['Curvature'] = splrep(x=u[:-1], y=e['Curvature'][:-1],
                                    s=0.01 / float(n), k=4, per=True)

            # Calculate the positions of anterior and posterior poles
            try:
                r['StalkedPole'], r['SwarmerPole'] = getpoles_curvature(
                    r['Curvature'], r['EdgeSpline'], e['Centroid'])
            except:
                r['StalkedPole'], r['SwarmerPole'] = getpoles_inertia(
                    e['Coordinates'], e['Centroid'])

            # Calculate a spline representation of the midline
            r['MidSpline'] = getmidline(u, r['EdgeSpline'], e['Coordinates'],
                                        (r['StalkedPole'], r['SwarmerPole']))

            # Evaluate the B-spline midline and calculate its length
            m = splineprops(u, r['MidSpline'], ('Coordinates', 'Length'))
            r['Length'] = m['Length']

            # Find the closest point on the midline to the centroid
            j = np.argmin([norm(e['Centroid'] - y) for y in m['Coordinates']])
            r['Centroid'] = m['Coordinates'][j]

            # Calculate the coefficients of the Fourier decomposition
            r['FourierFit'] = fftcoefs(e['Coordinates'], r['Centroid'],
                                       n=d['num_coefs'])
        except:
            # traceback.print_exc()
            pass
        results[i] = r
    return results
