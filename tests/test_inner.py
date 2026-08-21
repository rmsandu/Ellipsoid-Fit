import numpy as np
import pytest

from ellipsoid_fit import inner_ellipsoid_fit, sample_ellipse_points, sample_ellipsoid_points
from ellipsoid_fit.hull import get_hull


def _assert_inside_hull(points, B, d, tol=1e-3):
    A, b, _hull = get_hull(points)
    # Support function of the ellipsoid {B@u + d : ||u||<=1} along each facet
    # normal must not exceed the facet offset, i.e. ||B.T @ A[i]|| + A[i]@d <= b[i].
    support = np.linalg.norm(B.T @ A.T, axis=0) + A @ d
    assert np.all(support <= b + tol)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_inner_ellipse_2d_within_hull(seed):
    points = sample_ellipse_points(n=30, noise_std=0.1, seed=seed)
    B, d = inner_ellipsoid_fit(points)
    assert B.shape == (2, 2)
    assert d.shape == (2,)
    _assert_inside_hull(points, B, d)


def test_inner_ellipsoid_3d_within_hull():
    points = sample_ellipsoid_points(n=60, noise_std=0.1, seed=0)
    B, d = inner_ellipsoid_fit(points)
    assert B.shape == (3, 3)
    assert d.shape == (3,)
    _assert_inside_hull(points, B, d)


def test_inner_ellipsoid_positive_volume():
    points = sample_ellipsoid_points(n=60, noise_std=0.1, seed=0)
    B, _d = inner_ellipsoid_fit(points)
    assert np.linalg.det(B) > 0
