"""
"""
import os
import errno


def safe_mkdir(path):
    """Create a directory only if it does not already exist."""
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def safe_symlink(src, dst):
    """Create a symlink only if it does not already exist."""
    try:
        os.symlink(src, dst)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def isstring(obj):
    """True if the object is a string."""
    return isinstance(obj, basestring)


def islistlike(obj):
    """True if the object is iterable like a list and is *not* a string."""
    return ((hasattr(obj, '__iter__') or hasattr(obj, '__getitem__')) and
            not isstring(obj))
