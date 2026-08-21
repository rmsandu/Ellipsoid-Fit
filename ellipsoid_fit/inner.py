"""Maximum-volume ellipsoid inscribed in the convex hull of a point cloud.

Formulated as the semidefinite program described in
https://stackoverflow.com/questions/61859098/maximum-volume-inscribed-ellipsoid-in-a-polytope-set-of-points/61905793#61905793

The convex hull of the points is written as the polytope ``A @ x <= b``. The
inscribed ellipsoid is parameterized in "affine image of the unit ball" form,
``x = B @ u + d`` for ``||u|| <= 1``, and its volume is proportional to
``det(B)``. Maximizing ``log_det(B)`` subject to each hull facet containing
the ellipsoid gives a convex problem solvable directly with CVXPY.
"""

import cvxpy as cp
import numpy as np

from .hull import get_hull


def inner_ellipsoid_fit(points):
    """Find the maximum-volume ellipsoid inscribed in the convex hull of ``points``.

    Parameters
    ----------
    points : (N, dim) array
        2D or 3D point cloud.

    Returns
    -------
    B : (dim, dim) array
        Shape matrix; the ellipsoid boundary is ``{B @ u + d : ||u|| = 1}``.
    d : (dim,) array
        Ellipsoid center.
    """
    dim = points.shape[1]
    A, b, _hull = get_hull(points)

    B = cp.Variable((dim, dim), PSD=True)
    d = cp.Variable(dim)

    constraints = [cp.norm(B @ A[i], 2) + A[i] @ d <= b[i] for i in range(len(A))]
    problem = cp.Problem(cp.Minimize(-cp.log_det(B)), constraints)
    optval = problem.solve()
    if optval == np.inf:
        raise ValueError("No feasible inscribed ellipsoid was found for these points.")

    return B.value, d.value
