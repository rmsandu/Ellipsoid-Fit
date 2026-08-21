"""Shared 2D/3D plotting for point clouds plus their inner/outer ellipsoid fits."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.linalg import sqrtm

from .hull import get_hull

INNER_COLOR = "tab:orange"
OUTER_COLOR = "tab:green"
POINTS_COLOR = "tab:blue"


def _outer_to_affine(A, c):
    """Convert center-form (A, c) into the affine "B @ u + d" form used for drawing."""
    B = np.real(sqrtm(np.linalg.inv(A)))
    return B, c


def _boundary_2d(B, d, n=200):
    theta = np.linspace(0, 2 * np.pi, n)
    unit_circle = np.column_stack((np.cos(theta), np.sin(theta)))
    return unit_circle @ B.T + d


def _boundary_3d(B, d, n=40):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere = np.stack((x, y, z), axis=-1)  # (n, n, 3)
    ellipsoid = sphere @ B.T + d
    return ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2]


def plot_fit(points, inner=None, outer=None, show_hull=True, title=None,
             ax=None, save_path=None, show=False):
    """Plot a point cloud together with its fitted inner and/or outer ellipsoid.

    Parameters
    ----------
    points : (N, 2) or (N, 3) array
    inner : (B, d) tuple or None
        Result of :func:`ellipsoid_fit.inner_ellipsoid_fit`.
    outer : (A, c) tuple or None
        Result of :func:`ellipsoid_fit.outer_ellipsoid_fit`.
    show_hull : bool
        Draw the convex hull edges (2D) or wireframe (3D) of the points.
    title : str or None
    ax : matplotlib Axes or None
        Reuse an existing axes; otherwise a new figure/axes is created
        (3D projection is picked automatically from ``points`` dimensionality).
    save_path : str or None
        If given, save the figure to this path.
    show : bool
        If True, call ``plt.show()``.

    Returns
    -------
    fig, ax
    """
    points = np.asarray(points, dtype=float)
    dim = points.shape[1]
    if dim not in (2, 3):
        raise ValueError(f"Only 2D and 3D point clouds are supported, got dim={dim}")

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d" if dim == 3 else None)
    else:
        fig = ax.figure

    if dim == 2:
        ax.scatter(points[:, 0], points[:, 1], s=15, color=POINTS_COLOR, label="points", zorder=3)
        if show_hull:
            _, _, hull = get_hull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex, 0], points[simplex, 1], "k-", lw=1, alpha=0.6)
        if outer is not None:
            A, c = outer
            B, d = _outer_to_affine(A, c)
            boundary = _boundary_2d(B, d)
            ax.plot(boundary[:, 0], boundary[:, 1], color=OUTER_COLOR, lw=2, label="outer ellipse")
        if inner is not None:
            B, d = inner
            boundary = _boundary_2d(B, d)
            ax.plot(boundary[:, 0], boundary[:, 1], color=INNER_COLOR, lw=2, label="inner ellipse")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color=POINTS_COLOR)
        legend_handles = [Line2D([0], [0], marker="o", linestyle="", color=POINTS_COLOR, label="points")]
        if outer is not None:
            A, c = outer
            B, d = _outer_to_affine(A, c)
            X, Y, Z = _boundary_3d(B, d)
            ax.plot_wireframe(X, Y, Z, color=OUTER_COLOR, alpha=0.25, rstride=2, cstride=2)
            legend_handles.append(Line2D([0], [0], color=OUTER_COLOR, label="outer ellipsoid"))
        if inner is not None:
            B, d = inner
            X, Y, Z = _boundary_3d(B, d)
            ax.plot_surface(X, Y, Z, color=INNER_COLOR, alpha=0.35, linewidth=0)
            legend_handles.append(Patch(color=INNER_COLOR, alpha=0.35, label="inner ellipsoid"))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        try:
            ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass  # older matplotlib
        # 3D collections (plot_surface/plot_wireframe) don't support ax.legend()
        # directly in all matplotlib versions, so build the legend from proxies.
        ax.legend(handles=legend_handles, loc="best")

    if title:
        ax.set_title(title)
    if dim == 2:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
