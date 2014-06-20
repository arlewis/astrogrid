"""

==============
`sfhmaps.util`
==============

General use utilities.


Functions
---------

============ =========================================================
`islistlike` True if the object is iterable like a list and is *not* a
             string.
`isstring`   True if the object is a string.
============ =========================================================

"""
import os
import errno


def isstring(obj):
    """True if the object is a string."""
    return isinstance(obj, basestring)


def islistlike(obj):
    """True if the object is iterable like a list and is *not* a string."""
    return ((hasattr(obj, '__iter__') or hasattr(obj, '__getitem__')) and
            not isstring(obj))


def leastsquares(x, y, z):
    """Use 2d least squares minimization to find the best-fit a and b for
    zi = a*xi + b*yi.

    Does not work where x == y! If all x == y, then NaN is returned.

    Parameters
    ----------
    x, y

    Notes
    -----
    Let, ::

      f(x,y) = a*x + b*y

    Given a set of measurements ``zi`` for each pair of ``xi`` and ``yi``,
    find ``a`` and ``b`` such that ``S = sum(ri**2)`` is minimized (the
    gradient of ``S`` is zero), where ``ri = zi - f(xi,yi)``.

    So, ::

      dS/da = dS/db = 0

    The derivatives are, ::

      dS/da = 2 * sum(ri * dri/da) = 0
      dS/db = 2 * sum(ri * dri/db) = 0

      dri/da = -xi
      dri/db = -yi

      ri * dri/da = a*xi**2 + b*xi*yi - xi*zi
      ri * dri/db = a*xi*yi + b*yi**2 - yi*zi

    The sums can be written as, ::

      sum(a*xi**2 + b*xi*yi - xi*zi) = 0
      sum(a*xi*yi + b*yi**2 - yi*zi) = 0

      a*sum(xi**2) + b*sum(xi*yi) - sum(xi*zi) = 0
      a*sum(xi*yi) + b*sum(yi**2) - sum(yi*zi) = 0

      a*sx + b*sxy = sxz
      a*sxy + b*sy = syz

    Therefore, ::

      a = (sxz - sxy*syz/sy) / (sx - sxy**2/sy)
      b = (sxz - a*sx) / sxy

    """
    i = x != y
    if np.sum(i) == 0:
        return np.nan

    sx, sxy, sxz = np.sum(x[i]**2), np.sum(x[i]*y[i]), np.sum(x[i]*z[i])
    sy,      syz = np.sum(y[i]**2),                    np.sum(y[i]*z[i])

    a = (sxz - sxy*syz/sy) / (sx - sxy**2/sy)
    b = (sxz - a*sx) / sxy

    return a, b
