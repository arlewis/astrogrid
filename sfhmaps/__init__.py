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

=============== ==========================================================
|calc_mag|      Calculate the magnitude of an SED in a given filter.
|calc_sed|      Calculate the SED for a binned SFH.
|get_zmet|      Return the closest FSPS `zmet` integer for the given log
                metal abundance.
|make_header|   Create a FITS header from a set of points with known pixel
                and world coordinates given a celestial coordinate system
                and projection.
|mag2flux|      Convert AB magnitude in a filter to flux (erg s-1 cm-2
                A-1).
=============== ==========================================================


============
Module Index
============

- `sfhmaps.flux`
- `sfhmaps.util`
- `sfhmaps.wcs`


.. references

.. |calc_mag| replace:: `~sfhmaps.flux.calc_mag`
.. |calc_sed| replace:: `~sfhmaps.flux.calc_sed`
.. |get_zmet| replace:: `~sfhmaps.flux.get_zmet`

.. |make_header| replace:: `~sfhmaps.wcs.make_header`

"""
from .flux import calc_mag, calc_sed, get_zmet, mag2flux
from .wcs import make_header
