"""

=========
`sfhmaps`
=========

Create maps from star formation history (SFH) data.

`sfhmaps` requires the following packages:

- `astropy <http://www.astropy.org>`_
- `FSPS <http://people.ucsc.edu/~conroy/FSPS.html>`_ and
  `python-fsps <https://github.com/dfm/python-fsps>`_
  (or `this <https://github.com/bd-j/python-fsps>`_ fork)
- `numpy <http://www.numpy.org>`_
- `scombine <https://github.com/bd-j/scombine>`_
- `sedpy <https://github.com/bd-j/sedpy>`_


Functions
---------

=============== ========================================================
|calc_sed|      Calculate the SED for a binned SFH.
|get_zmet|      Return the closest FSPS `zmet` integer for the given log
                metal abundance.
|calc_pixscale| Calculate the pixel scale from the WCS information in a
                FITS header.
|gcdist|        Calculate the great circle distance between two points.
=============== ========================================================


============
Module Index
============

- `sfhmaps.flux`
- `sfhmaps.util`
- `sfhmaps.wcs`


.. references

.. |calc_sed| replace:: `~sfhmaps.flux.calc_sed`
.. |get_zmet| replace:: `~sfhmaps.flux.get_zmet`

.. |calc_pixscale| replace:: `~sfhmaps.wcs.calc_pixscale`
.. |gcdist| replace:: `~sfhmaps.wcs.gcdist`

"""
from .flux import calc_sed, get_zmet
from .wcs import calc_pixscale, gcdist
