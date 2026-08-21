"""Generate mockup point clouds scattered on/near an ellipse or ellipsoid.

Used to produce reproducible example data for the inner/outer ellipsoid fits
and for the README figures. Pass ``seed`` for reproducible output.
"""

import numpy as np


def sample_ellipse_points(semi_axes=(3.0, 1.5), center=(0.0, 0.0), rotation=0.3,
                           n=60, noise_std=0.15, seed=None):
    """Sample ``n`` points scattered around a 2D ellipse.

    Parameters
    ----------
    semi_axes : (a, b)
        Ellipse semi-axis lengths before rotation.
    center : (x0, y0)
    rotation : float
        Rotation angle of the ellipse, in radians.
    noise_std : float
        Std-dev of isotropic Gaussian noise added to each sampled point.
    seed : int or None
        Seed for reproducible sampling.

    Returns
    -------
    (n, 2) array
    """
    rng = np.random.default_rng(seed)
    a, b = semi_axes
    theta = rng.uniform(0, 2 * np.pi, n)
    unit = np.column_stack((np.cos(theta), np.sin(theta)))

    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    W = R @ np.diag([a, b])

    points = unit @ W.T + np.asarray(center)
    points += rng.normal(scale=noise_std, size=points.shape)
    return points


def sample_ellipsoid_points(semi_axes=(3.0, 5.0, 7.0), center=(0.0, 0.0, 0.0),
                             n=200, noise_std=0.2, seed=None):
    """Sample ``n`` points scattered around a 3D ellipsoid surface.

    Parameters
    ----------
    semi_axes : (a, b, c)
        Ellipsoid semi-axis lengths.
    center : (x0, y0, z0)
    noise_std : float
        Std-dev of isotropic Gaussian noise added to each sampled point.
    seed : int or None
        Seed for reproducible sampling.

    Returns
    -------
    (n, 3) array
    """
    rng = np.random.default_rng(seed)
    a, b, c = semi_axes
    u = rng.uniform(0, 2 * np.pi, n)
    v = np.arccos(2 * rng.uniform(0, 1, n) - 1.0)

    x = a * np.sin(v) * np.cos(u)
    y = b * np.sin(v) * np.sin(u)
    z = c * np.cos(v)
    points = np.column_stack((x, y, z)) + np.asarray(center)
    points += rng.normal(scale=noise_std, size=points.shape)
    return points
