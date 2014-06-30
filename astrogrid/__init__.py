"""

===========
`astrogrid`
===========

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


Example
-------
To illustrate how this package might be used, suppose a rectangular patch
of sky has been divided into a grid with 3 rows and 4 columns. The SFH of
each cell is measured separately, resulting in a set of 12 SFH files for
the whole grid. The ultimate goal is to use these SFHs to construct an
image of modeled flux.

The first step is to write a function that calculates flux from an input
SFH. The `astrogrid.flux` module provides some tools to help do this, but
suppose the function looks like,

>>> import astrogrid
>>> def calc_flux(sfhfile, band, distance, extinction=None):
...     # Code to calculate flux in the band from the SFH data in stored in
...     # sfhfile, assuming a certain distance and an optional amount of
...     # extinction.
...     age, sfr = process_sfh(sfhfile)
...     wave, spec = astrogrid.flux.calc_sed(sfr, age, av=extinction)
...     mag = astrogrid.flux.calc_mag(wave, spec, band, dmod=distance):
...     flux = astrogrid.flux.mag2flux(mag, band)
...     return flux

Now create a list of arguments and a list of keyword arguments to pass to
`calc_flux` for each cell in the grid. The order of the cells should be
that of a flattened 2d `numpy` array (e.g., `np.ravel`). Assuming the cells
are numbered from 1 to 12 and cell 1 goes in the first row, first column of
the grid,

>>> file_list = ['sfhfile01', 'sfhfile02', ... 'sfhfile12']
>>> band_list = ['galex_fuv'] * len(file_list)
>>> distance_list = [distance] * len(file_list)
>>> args = zip(file_list, band_list, distance_list)
>>> extinction_vals = [av01, av02, ... av12]
>>> kwargs = [{'extinction': av} for av in av_list]

Next create a `Grid` instance and calculate the grid values:

>>> shape = (3, 4)  # 3 rows by 4 columns
>>> grid = astrogrid.Grid(shape, calc_flux, args, kwargs)
>>> grid.update()
>>> grid.data_grid  # The 2d array of grid values are accessed here
array([[...

Note that the grid values are not actually calculated until the `update`
method is called. The grid attributes can be modified in needed. For
example, suppose a different extinction value should be used for cell 4:

>>> grid.kwargs[3]['extinction'] = av04_new

Also suppose that `calc_flux` is very expensive and recomputing the entire
grid would take a long time. The `update` method has a `where` option to
compute only specific cells for exactly this purpose. Cell 4 can be indexed
using either ``where=3`` (list index) or ``where=(0, 3)`` (grid indices).

>>> grid.update(where=3)  # or grid.update(where=(0, 3))

The array in the `data_grid` attribute contains the desired image data, but
it would be nice to have an accompanying header with WCS information so
that, for example, the image could later be combined with other similar
images to produce a mosaic. The `astrogrid.wcs` module can fit a WCS to the
grid given the coordinates of a set of points. If the RA and dec of the
cell corners have been measured, then obtaining a header is easy:

>>> x, y = grid.edges  # pixel coordinates of the cell corners
>>> hdr = astrogrid.wcs.make_header(x, y, RA, dec)

where ``RA`` and ``dec`` are the same shape as ``x`` and ``y``, and ``hdr``
is an `astropy.io.fits.Header` instance. Finally, the grid can be saved as
an image in FITS format:

>>> import astropy.io.fits
>>> hdu = astropy.io.fits.PrimaryHDU(data=grid.data_grid, header=hdr)
>>> hdu.writeto(filename)


Modules
-------

====== ==================================================================
`flux` Utilities for calculating integrated SEDs and magnitudes from SFHs
       using FSPS.
`util` General utilities.
`wcs`  Utilities for working with world coordinate systems.
====== ==================================================================


Classes
-------

====== =================================
|Grid| Build a grid from flattened data.
====== =================================


============
Module Index
============

- `astrogrid.flux`
- `astrogrid.grid`
- `astrogrid.util`
- `astrogrid.wcs`


.. references

.. |Grid| replace:: `~astrogrid.grid.Grid`

"""
from . import flux
from .grid import Grid
from . import util
from . import wcs
