# -*- coding: utf-8 -*-
"""
@author: Raluca Sandu
"""
# from mpl_toolkits.mplot3d import axes3d
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


from generate_test_ellipse import GetRandom, random_point_ellipse, random_point_ellipsoid


def inner_ellipsoid_fit(points):
    """Find the inscribed ellipsoid into a set of points of maximum volume. Return its matrix-offset form."""
    dim = points.shape[1]
    A,b,hull = GetHull(points)

    B = cp.Variable((dim,dim), PSD=True)  # Ellipsoid
    d = cp.Variable(dim)                  # Center

    constraints = [cp.norm(B@A[i],2)+A[i]@d<=b[i] for i in range(len(A))]
    prob = cp.Problem(cp.Minimize(-cp.log_det(B)), constraints)
    optval = prob.solve()
    if optval==np.inf:
        raise Exception("No solution possible!")
    print(f"Optimal value: {optval}")

    return B.value, d.value


def Plot(points, hull, B, d):
    fig = plt.figure()
    if points.shape[1] == 2:
        ax = fig.add_subplot(111)
        ax.scatter(points[:, 0], points[:, 1])
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        display_points = np.array([random_point_ellipse([[1, 0], [0, 1]], [0, 0]) for i in range(100)])
        display_points = display_points @ B + d
        ax.scatter(display_points[:, 0], display_points[:, 1])
    elif points.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        display_points = np.array([random_point_ellipsoid(1, 1, 1, 0, 0, 0) for i in range(len(points))])
        display_points = display_points @ B + d
        ax.scatter(display_points[:, 0], display_points[:, 1], display_points[:, 2])
        return ax
    plt.show()


def GetHull(points):
    dim = points.shape[1]
    hull = ConvexHull(points)
    A = hull.equations[:,0:dim]
    b = hull.equations[:,dim]
    return A, -b, hull  #Negative moves b to the RHS of the inequality


if __name__ == '__main__':
    # points = GetRandom(dims=3, Npts=200)
    points = np.array([[ 0.53135758, -0.25818091, -0.32382715],
                       [ 0.58368177, -0.3286576,  -0.23854156,],
                       [ 0.18741533,  0.03066228, -0.94294771],
                       [ 0.65685862, -0.09220681, -0.60347573],
                       [ 0.63137604, -0.22978685, -0.27479238],
                       [ 0.59683195, -0.15111101, -0.40536606],
                       [ 0.68646128,  0.0046802,  -0.68407367],
                       [ 0.62311759,  0.0101013,  -0.75863324]])

    A, b, hull = GetHull(points)
    B, d = inner_ellipsoid_fit(points)
    Plot(points, hull, B, d)


