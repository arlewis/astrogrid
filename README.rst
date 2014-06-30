astrogrid
=========

Tools for creating 2d grids from flattened data.

`astrogrid` provides the `Grid` class, which takes an unstructured set of
data, performs a user-defined calculation on it, and arranges the results
as a grid of the desired shape. The resulting grid is a 2d `numpy` array,
so it can easily be processed further, plotted, or written to an image
file. The main focus of the package is on calculating broadband fluxes from
star formation history (SFH) data, but the input data and calculated grid
values could be anything.

`astrogrid` requires the following packages:

- `astropy <http://www.astropy.org>`_
- `FSPS <http://people.ucsc.edu/~conroy/FSPS.html>`_ and
  `python-fsps <https://github.com/dfm/python-fsps>`_
  (or `this <https://github.com/bd-j/python-fsps>`_ fork)
- `numpy <http://www.numpy.org>`_
- `scipy <http://www.scipy.org>`_
- `scombine <https://github.com/bd-j/scombine>`_
- `sedpy <https://github.com/bd-j/sedpy>`_
