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
