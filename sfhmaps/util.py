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
