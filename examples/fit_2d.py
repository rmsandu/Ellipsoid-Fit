"""Fit and plot both the inner and outer ellipse for a mockup 2D point cloud."""

import os

from ellipsoid_fit import inner_ellipsoid_fit, outer_ellipsoid_fit, plot_fit, sample_ellipse_points

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def main():
    points = sample_ellipse_points(semi_axes=(3.0, 1.5), rotation=0.4, n=40, noise_std=0.2, seed=0)

    B, d = inner_ellipsoid_fit(points)
    A, c = outer_ellipsoid_fit(points)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_fit(
        points,
        inner=(B, d),
        outer=(A, c),
        title="2D: inner (orange) and outer (green) ellipse",
        save_path=os.path.join(FIGURES_DIR, "fit_2d.png"),
        show=__name__ == "__main__" and os.environ.get("EF_NO_SHOW") is None,
    )


if __name__ == "__main__":
    main()
