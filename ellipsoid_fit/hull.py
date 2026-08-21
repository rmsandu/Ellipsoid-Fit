"""Convex hull helper shared by the inner- and outer-ellipsoid fits."""

from scipy.spatial import ConvexHull


def get_hull(points):
    """Return the facet inequalities ``A @ x <= b`` of the convex hull of ``points``.

    Parameters
    ----------
    points : (N, dim) array
        2D or 3D point cloud.

    Returns
    -------
    A : (F, dim) array
    b : (F,) array
    hull : scipy.spatial.ConvexHull
    """
    dim = points.shape[1]
    hull = ConvexHull(points)
    A = hull.equations[:, 0:dim]
    b = -hull.equations[:, dim]
    return A, b, hull
