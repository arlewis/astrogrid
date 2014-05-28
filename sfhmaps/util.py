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


