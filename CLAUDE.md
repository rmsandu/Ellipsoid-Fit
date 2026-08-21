# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A small Python library that fits ellipsoids to 2D/3D point clouds:

- **Inner (max-volume inscribed) ellipsoid** — `ellipsoid_fit/inner.py`: convex-hull-constrained SDP solved with CVXPY (maximizes `log_det` of the shape matrix `B`; ellipsoid given as `x = B@u + d` for `||u|| <= 1`). This is the formulation from the author's Stack Overflow answer: https://stackoverflow.com/questions/61859098/maximum-volume-inscribed-ellipsoid-in-a-polytope-set-of-points/61905793#61905793
- **Outer (min-volume enclosing) ellipsoid** — `ellipsoid_fit/outer.py`: Khachiyan's algorithm (iterative fixed-point method), no CVXPY needed. Ellipsoid given in center form `(x-c).T @ A @ (x-c) = 1`.
- **Shared convex hull helper** — `ellipsoid_fit/hull.py`: wraps `scipy.spatial.ConvexHull`, expressing the hull as `A @ x <= b`. Used only by the inner fit (the outer fit doesn't need the hull).
- **Mockup data generators** — `ellipsoid_fit/sampling.py`: `sample_ellipse_points` (2D) / `sample_ellipsoid_points` (3D), both accept a `seed` for reproducible output.
- **Unified visualization** — `ellipsoid_fit/visualize.py`: `plot_fit(points, inner=(B,d), outer=(A,c), ...)` auto-detects 2D vs 3D from `points.shape[1]` and draws points + hull + both ellipsoid fits on one axes. Converts the outer fit's center-form `(A, c)` into the same affine `B@u + d` boundary representation used by the inner fit via `sqrtm(inv(A))`, so both fits share one boundary-drawing code path.
- **`examples/`** — runnable scripts (`fit_2d.py`, `fit_3d.py`) that generate mockup data, run both fits, and save the figures used in the README to `figures/`.
- **`tests/`** — pytest suite checking geometric correctness: the inner ellipsoid's support function must stay within each hull facet, and the outer ellipsoid's Mahalanobis distance must contain every input point. Numerical tolerance is `1e-3`, not tighter — SDP solvers (CVXPY's CLARABEL/SCS) converge to roughly `1e-5`-`1e-4` precision by default, so a `1e-6` tolerance produces flaky-looking failures that are not real bugs (learned this the hard way when debugging this repo).
- **`archive/ellipsoid_inner_outer.py`** — an older, project-specific version of this code (imports `VolumeMetrics`, `SimpleITK`, DICOM-related helpers from an unrelated project). Not runnable standalone; kept for historical reference only, do not try to fix its imports.

The previous top-level scripts (`inner_ellipsoid.py`, `outer_ellipsoid.py`, `max_inner_ellipsoid_v2.py`, `generate_test_ellipse.py`) were consolidated into the `ellipsoid_fit/` package. The iterative S-lemma-based inner-ellipsoid method that used to live in `max_inner_ellipsoid_v2.py` (ported from hongkai-dai/large_inscribed_ellipsoid) was dropped — the direct SDP formulation in `inner.py` solves the same problem in one shot without iteration/sampling heuristics, so keeping both was redundant.

## Commands

```bash
pip install -e ".[dev]"        # install package + pytest
pytest                          # run tests
python -m examples.fit_2d       # regenerate figures/fit_2d.png (set EF_NO_SHOW=1 to skip the plt.show() window)
python -m examples.fit_3d       # regenerate figures/fit_3d.png
```

CI (`.github/workflows/ci.yml`) runs `pytest` on Python 3.10 and 3.12 with `MPLBACKEND=Agg`.

## Key implementation details worth knowing before editing

- Ellipsoid representations differ between the two fits: `inner_ellipsoid_fit` returns `(B, d)` for `x = B@u + d`; `outer_ellipsoid_fit` returns center-form `(A, c)` for `(x-c).T @ A @ (x-c) = 1`. Don't assume these are interchangeable without converting (see `visualize._outer_to_affine`).
- `plot_fit`'s 3D branch builds its legend from manual `Line2D`/`Patch` proxies rather than relying on `ax.legend()` picking up labels from `plot_surface`/`plot_wireframe` — passing `label=` directly to those 3D collections is unreliable across matplotlib versions (some raise on `ax.legend()` with no explicit handles).
