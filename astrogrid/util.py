"""

================
`astrogrid.util`
================

General utilities.


Functions
---------

================ ==========================================================
`islistlike`     True if the object is iterable like a list and is *not* a
                 string.
`isstring`       True if the object is a string.
================ ==========================================================

"""
import os
import errno


def isstring(obj):
    """True if the object is a string."""
    return isinstance(obj, basestring)


def islistlike(obj):
    """True if the object is iterable like a list and is *not* a string."""
    cond = hasattr(obj, '__iter__') or hasattr(obj, '__getitem__')
    if cond and not isstring(obj):
        if hasattr(obj, 'shape') and not obj.shape:
            test = False  # Probably a numpy integer or float array element
        else:
            test = True
    else:
        test = False
    return test
