"""

===========
`astrogrid`
===========

Tools for creating 2d grids from flattened data.

`astrogrid` provides the `Grid` class, which takes an unstructured set of
data, performs a user-defined calculation on it, and arranges the results
as a grid of the desired shape. The resulting grid is a 2d ndarray, so it
can easily be processed further, plotted, or written to an image file. The
main focus of the package is on calculating broadband fluxes from star
formation history (SFH) data, but the input data and calculated grid values
could be anything.

`astrogrid` requires the following core packages:

- `numpy <http://www.numpy.org>`_

Supporting modules are imported if the required packages are installed:

- `astrogrid.flux`:

  - `FSPS <http://people.ucsc.edu/~conroy/FSPS.html>`_ and
    `python-fsps <https://github.com/dfm/python-fsps>`_
  - `scombine <https://github.com/bd-j/scombine>`_
  - `sedpy <https://github.com/bd-j/sedpy>`_

- `astrogrid.mwe`: all `astrogrid.wcs` dependencies, plus,

  - `astropy <http://www.astropy.org>`_
  - `Montage <http://montage.ipac.caltech.edu>`_ and
    `montage_wrapper <http://montage-wrapper.readthedocs.org>`_

- `astrogrid.wcs`:

  - `astropy <http://www.astropy.org>`_
  - `scipy <http://www.scipy.org>`_


Example
-------
To illustrate how this package might be used, suppose a rectangular patch
of sky has been divided into a grid with 3 rows and 4 columns. The SFH of
each cell is measured separately, resulting in a set of 12 SFH files for
the whole grid. The ultimate goal is to use these SFHs to construct an
image of modeled flux.

The first step is to write a function that calculates flux from an input
SFH. The `astrogrid.flux` module provides some tools to help do this (note
that the `flux` module is really just a convenience frontend for
`python-fsps`, `scombine`, and `sedpy`):

>>> import astrogrid
>>> def calc_flux(sfhfile, band, distance, extinction=None):
...     # Code to calculate flux in the band from the SFH data in stored in
...     # sfhfile, assuming a certain distance and an optional amount of
...     # extinction.
...     age, sfr = some_function_to_process_the_sfh(sfhfile)
...     wave, spec = astrogrid.flux.calc_sed(sfr, age, av=extinction)
...     mag = astrogrid.flux.calc_mag(wave, spec, band, dmod=distance):
...     flux = astrogrid.flux.mag2flux(mag, band)
...     return flux

Now create a list of arguments and a list of keyword arguments to pass to
`calc_flux` for each cell in the grid. The grid is represented by a 2d
ndarray, and it is always filled starting from row 0, column 0, then going
through the columns before moving to the next row, and so on. The order of
the cell arguments should therefore correspond to a flattened array (e.g.,
`numpy.ravel`). For this example, assume that the cells are labeld "a"
through "l" arranged in the following way::

     +---+---+---+---+
    2| i | j | k | l |
  ^  +---+---+---+---+
  r 1| e | f | g | h |
  o  +---+---+---+---+
  w 0| a | b | c | d |
     +---+---+---+---+
       0   1   2   3
       column >

The chosen arrangement is arbitrary and is essentially a matter of
convenience. For example, the rows and columns could have been reversed so
that cells "a" and "l" are at 2,3 and 0,0, respectively. With the
arrangement above, the arguments are listed such that the cells are in
alphabetical order:

>>> file_list = ['sfhfile_a', 'sfhfile_b', ... 'sfhfile_l']
>>> band_list = ['galex_fuv'] * len(file_list)
>>> distance_list = [distance] * len(file_list)
>>> args = zip(file_list, band_list, distance_list)
>>> extinction_vals = [av_a, av_b, ... av_l]
>>> kwargs = [{'extinction': av} for av in av_list]

Next create a `Grid` instance and calculate the grid values:

>>> shape = (3, 4)  # 3 rows by 4 columns
>>> grid = astrogrid.Grid(shape, calc_flux, args, kwargs)
>>> grid.update()
>>> grid.data_grid  # The 2d array of grid values are accessed here
array([[...

Note that the grid values are not actually calculated until the `update`
method is called. The grid attributes can be modified if needed. For
example, suppose a different extinction value should be used for cell "d":

>>> grid.kwargs[3]['extinction'] = av_d_new

Also suppose that running `calc_flux` is very expensive and recomputing the
entire grid would take a long time. The `update` method has a `where`
option to compute only specific cells for exactly this purpose. Cell "d"
can be indexed using either ``where=3`` (list index) or ``where=(0, 3)``
(grid indices):

>>> grid.update(where=3)  # or grid.update(where=(0, 3))

The array in the `data_grid` attribute now contains the desired image data.
It would be nice, however, to have an accompanying header with WCS
information so that the image could later be combined with other similar
images to produce a mosaic. The `astrogrid.wcs` module can fit a WCS to the
grid given the coordinates of a set of points. If the RA and dec
coordinates of the cell corners have been measured, then obtaining a header
is easy:

>>> x, y = grid.edges  # pixel coordinates of the cell corners
>>> hdr = astrogrid.wcs.make_header(x, y, RA, dec)

where ``RA`` and ``dec`` are the same shape as ``x`` and ``y``, and ``hdr``
is an `astropy.io.fits.Header` instance. Finally, the grid can be saved as
an image in FITS format:

>>> import astropy.io.fits
>>> hdu = astropy.io.fits.PrimaryHDU(data=grid.data_grid, header=hdr)
>>> hdu.writeto(filename)

A set of such modeled flux images can be stitched together as a mosaic
using the `astrogrid.mwe` ("montage-wrapper extension") module, which
provides a wrapper for `montage_wrapper.mosaic`. Of course, the
`montage_wrapper` package could always be used directly. Either way,
mosaicking requires a working installation of Montage and the
`montage_wrapper` package. Assuming ``input_files`` is a list of paths to
the individual flux images, the following command creates a mosaic at
``mosaic_file`` and a directory of intermediate files in ``work_dir``:

>>> astrogrid.mwe.mosaic(input_files, mosaic_file, work_dir)


Modules
-------
These support modules are only imported if their required dependencies are
available.

====== ==================================================================
`flux` Utilities for calculating integrated SEDs and magnitudes from SFHs
       using FSPS.
`mwe`  Mosaicking utilities.
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
- `astrogrid.mwe`
- `astrogrid.wcs`


.. references

.. |Grid| replace:: `~astrogrid.grid.Grid`

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .grid import Grid

try:
    from . import flux
except ImportError:
    pass

try:
    from . import mwe
except ImportError:
    pass

try:
    from . import wcs
except ImportError:
    pass
