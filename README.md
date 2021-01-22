# Elipsoid-Fit
Approximate a cloud of points (3D/2D) with the maximum volume inner and minimum volume outer ellipsoid. Also known as the inner and outer [Löwner-John ellipses](https://en.wikipedia.org/wiki/John_ellipsoid). 

![image](https://user-images.githubusercontent.com/20581812/99432875-0cbda180-290d-11eb-8ccd-4697f851cdc8.png)

## Installation
Using [CVXPY](https://www.cvxpy.org/) package. Install the required packages into a separate virtual environment. I advise using [Anaconda](https://anaconda.org/conda-forge/cvxpy) envinromment instead of pip, since CVXPY has all the necessary solvers compiled already there.
Scikit image is needed to visualize your points.

## Usage
I defined a set of random 3D points as a `numpy` array as input data. Alternatively, a 2D numpy array can be given.
- `inner_ellipsoid.py` generates the maximum volume inscribed ellipsoid approximating a set of points
- `outer_ellipsoid.py` generates the minimum volume enclosing ellipsoid around a set of points

## Results on some random data:
Original points in blue, Outer Ellipsoid in Green Wireframe and Inner Ellipsoid as Orange Points


![image](https://user-images.githubusercontent.com/20581812/99433589-fcf28d00-290d-11eb-8b0c-d2c6b975cc53.png)


Solution in `max_inner_ellipsoid_v2.py` was inspired by [Hongkai-Dai](https://github.com/hongkai-dai/large_inscribed_ellipsoid)








