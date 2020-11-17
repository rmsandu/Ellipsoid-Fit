# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
from mpl_toolkits.mplot3d import axes3d
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
# import sklearn.datasets
from scipy.spatial import ConvexHull


def random_point_ellipsoid(a, b, c, x0, y0, z0):
    """Generate a random point on an ellipsoid defined by a,b,c"""
    u = np.random.rand()
    v = np.random.rand()
    theta = u * 2.0 * np.pi
    phi = np.arccos(2.0 * v - 1.0)
    sinTheta = np.sin(theta);
    cosTheta = np.cos(theta);
    sinPhi = np.sin(phi);
    cosPhi = np.cos(phi);
    rx = a * sinPhi * cosTheta;
    ry = b * sinPhi * sinTheta;
    rz = c * cosPhi;
    return rx, ry, rz


def random_point_ellipse(W, d):
    # random angle
    alpha = 2 * np.pi * np.random.random()
    # vector on that angle
    pt = np.array([np.cos(alpha), np.sin(alpha)])
    # Ellipsoidize it
    return W @ pt + d


def GetRandom(dims, Npts):
    if dims == 2:
        W = sklearn.datasets.make_spd_matrix(2)
        d = np.array([2, 3])
        points = np.array([random_point_ellipse(W, d) for i in range(Npts)])
    elif dims == 3:
        points = np.array([random_point_ellipsoid(3, 5, 7, 2, 3, 3) for i in range(Npts)])
    else:
        raise Exception("dims must be 2 or 3!")
    noise = np.random.multivariate_normal(mean=[0] * dims, cov=0.2 * np.eye(dims), size=Npts)
    return points + noise
