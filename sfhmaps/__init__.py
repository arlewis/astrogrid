"""

=========
`sfhmaps`
=========

Create maps from star formation history (SFH) data.

Dependencies:

- `FSPS <http://people.ucsc.edu/~conroy/FSPS.html>`_ and
  `python-fsps <https://github.com/dfm/python-fsps>`_
  (or `this <https://github.com/bd-j/python-fsps>`_ fork)
- `numpy <http://www.numpy.org>`_
- `scombine <https://github.com/bd-j/scombine>`_
- `sedpy <https://github.com/bd-j/sedpy>`_


Functions
---------

========== ==============================================================
|calc_sed| Calculate the SED for a binned SFH.
|get_zmet| Return the closest FSPS `zmet` integer for the given log metal
           abundance.
========== ==============================================================


============
Module Index
============

- `sfhmaps.config`
- `sfhmaps.flux`


.. references

.. |calc_sed| replace:: `~sfhmaps.flux.calc_sed`
.. |calc_sed| replace:: `~sfhmaps.flux.get_zmet`

"""
#from . import config  # Deprecate?
from .flux import calc_sed, get_zmet
