import numpy as np
import pytest

from ellipsoid_fit import outer_ellipsoid_fit, sample_ellipse_points, sample_ellipsoid_points


def _assert_contains_points(points, A, c, tol=1e-3):
    diff = points - c
    mahalanobis = np.einsum("ij,jk,ik->i", diff, A, diff)
    assert np.all(mahalanobis <= 1 + tol)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_outer_ellipse_2d_contains_points(seed):
    points = sample_ellipse_points(n=30, noise_std=0.1, seed=seed)
    A, c = outer_ellipsoid_fit(points)
    assert A.shape == (2, 2)
    assert c.shape == (2,)
    _assert_contains_points(points, A, c)


def test_outer_ellipsoid_3d_contains_points():
    points = sample_ellipsoid_points(n=60, noise_std=0.1, seed=0)
    A, c = outer_ellipsoid_fit(points)
    assert A.shape == (3, 3)
    assert c.shape == (3,)
    _assert_contains_points(points, A, c)
