"""Minimum-volume ellipsoid enclosing a point cloud (the Löwner-John outer ellipsoid).

Solved with the Khachiyan algorithm: a fixed-point iteration on a probability
weighting of the points that converges to the minimum-volume enclosing
ellipsoid. See https://en.wikipedia.org/wiki/John_ellipsoid.
"""

import numpy as np


def outer_ellipsoid_fit(points, tol=1e-4, max_iterations=10_000):
    """Find the minimum-volume ellipsoid enclosing ``points``.

    Parameters
    ----------
    points : (N, dim) array
        2D or 3D point cloud.
    tol : float
        Convergence tolerance on the Khachiyan weight update.
    max_iterations : int
        Safety cap on the number of iterations.

    Returns
    -------
    A : (dim, dim) array
        Shape matrix such that the ellipsoid boundary is
        ``{x : (x - c).T @ A @ (x - c) = 1}``.
    c : (dim,) array
        Ellipsoid center.
    """
    points = np.asarray(points, dtype=float)
    n, dim = points.shape
    q = np.column_stack((points, np.ones(n))).T  # (dim+1, n)
    u = np.full(n, 1.0 / n)

    err = tol + 1
    n_iter = 0
    while err > tol and n_iter < max_iterations:
        X = q @ np.diag(u) @ q.T
        m = np.diag(q.T @ np.linalg.inv(X) @ q)
        j = np.argmax(m)
        step_size = (m[j] - dim - 1.0) / ((dim + 1) * (m[j] - 1.0))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u
        n_iter += 1

    c = u @ points
    A = np.linalg.inv(points.T @ np.diag(u) @ points - np.outer(c, c)) / dim
    return A, c
