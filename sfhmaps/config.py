"""

================
`sfhmaps.config`
================

Configuration system for `sfhmaps`.

This module provides path names and parameters for the `sfhmaps` package so
that hard-coded values may be minimized. Settings are defined in a
configuration file whose path is specified by the `configfile` variable
below. The configuration file is a python module -- it does not need to
live in a package -- and therefore may define classes and functions in
addition to simple parameters. All of the names it defines (except for
modules and special members) are loaded into the `sfhmaps.config`
namespace.

.. caution:: The configuration file is loaded using ``__import__``, which
   executes all of its code, so be careful!


Parameters
----------
configfile : str
    Absolute path to the local configuration file. The file name is
    arbitrary and the ".py" extension is optional. There are a number of
    different ways to set this variable::

      # Manually specify the absolute path
      configfile = '/absolute/path/to/configfile.py'

      # The `os` module is handy, e.g., to expand "~" or any environmental
      # variables
      configfile = os.path.expanduser('~/path/to/configfile.py')
      configfile = os.path.expandvars('$PATHTOCONFIGFILE')

"""

import os
import sys


# Edit `configfile` here
# ----------------------


configfile = os.path.expandvars('$SFHMAPSCONFIG')


# Do not edit anything below this line
# ====================================


def _get_namespace(
        module_path, skipspecials=False, skipmodules=False, skipnames=None):
    """Retrieve the namespace dictionary from an individual python module.

    `module_path` can point to any python file (module), regardless of
    whether it lives in a package.

    .. caution:: The namespace is retrieved using ``__import__``, which
       executes all code in the module, so be careful about what
       `module_path` points to!

    Parameters
    ----------
    module_path : str
        Absolute path to a python module. The ".py" extension is optional.
    skipspecials : bool, optional
        If True, special members (those with leading and trailing double
        underscores) are excluded from the returned namespace dictionary.
        Default is False.
    skipmodules : bool, optional
        If True, imported modules are excluded from the namespace
        dictionary. Default is False.
    skipnames : list, optional
        List of names to exclude from the returned namespace dictionary.
        Default is None.

    Returns
    -------
    dict
        Dictionary of names in the specified python module.

    """
    from types import ModuleType

    if skipnames is None:
        skipnames = []

    dirname, basename = os.path.split(module_path)
    module_name = os.path.splitext(basename)[0]

    original_path = sys.path[:]
    sys.path.insert(0, dirname)
    try:
        module = __import__(module_name)
    finally:
        sys.path[:] = original_path

    namespace = {}
    for key, val in vars(module).items():
        specialtest = (skipspecials and
                       key.startswith('__') and key.endswith('__'))
        moduletest = skipmodules and isinstance(val, ModuleType)
        skiptest = key in skipnames
        if any((specialtest, moduletest, skiptest)):
            continue
        namespace[key] = val

    return namespace


globals().update(_get_namespace(configfile, skipspecials=True, skipmodules=True))
