"""

================
`astrogrid.grid`
================

Create 2d grids from flattened data.

This module defines the `Grid` class, which takes an unstructured set of
data, performs a user-defined calculation on it, and arranges the results
as a grid of the desired shape. The resulting grid is a 2d `numpy` array,
so it can easily be processed further, plotted, or written to an image
file.


Classes
-------

====== =================================
`Grid` Build a grid from flattened data.
====== =================================

"""
import numpy as np


class Grid(object):

    """Build a grid from flattened data.

    A grid is defined by a shape, a function, and list of arguments for the
    grid cells. The grid itself is represented by a 2d array of the given
    shape (the `data` attribute), where the value of each cell is computed
    using the function and the cell's arguments. Grid attributes can be
    modified freely. The grid's data array is only updated when the
    `update` method is called.

    Parameters
    ----------
    shape : tuple
        Initialize the `shape` attribute.
    func : function
        Initialize the `func` attribute.
    args : list
        Initialize the `args` attribute.
    kwargs : list, optional
        Initialize the `kwargs` attribute. Default is a list of empty
        dictionaries equal in length to `args`.
    fill : int or float, optional
        Initialize the `fill` attribute. Default is `np.nan`.
    update : bool, optional
        If True, the grid values are calculated on instantiation. If False
        (default), then the entire grid is set to `fill` until the `update`
        method is called.

    Attributes
    ----------
    shape
        Numbers of rows and columns in the grid.
    nrow
    ncol
    ij
    xy
    edges
    func
        Function that calculates the value of each cell in the grid.
    args
        List of tuples, one per cell, containing the arguments for `func`.
        The tuples are ordered as a list of cells from a flattened grid
        (e.g., `numpy.ravel`). The list is automatically exteneded with
        Nones if it is too short for the given grid shape, or truncated if
        it is too long.
    kwargs
        Similar to `args`, but a list of dictionaries containing any
        keyword arguments for `func`.
    fill : int or float
        Fallback value to assign a cell if its args or kwargs is None.
    data_list
    data_grid

    Methods
    -------
    update

    """

    def __init__(self, shape, func, args, kwargs=None, fill=np.nan,
                 update=False):
        self.shape = shape
        self.func = func
        self.args = args
        self.kwargs = [{} for a in args] if kwargs is None else kwargs
        self.fill = fill

        self._data_list = np.zeros(shape).ravel() * fill
        self._data_grid = self._data_list.reshape(shape)

        if update:
            self.update()

        return

    @property
    def nrow(self):
        """Number of rows in the grid."""
        return self.shape[0]

    @property
    def ncol(self):
        """Number of columns in the grid."""
        return self.shape[1]

    @property
    def ij(self):
        """Row and column array coordinates (row and column indices) of the
        cells in the grid.

        """
        return np.indices(self.shape)

    @property
    def xy(self):
        """x and y pixel coordinates of the centers of the cells in the grid.

        The pixel coordinate system places the center of the cell in the
        first row and first column (i,j = 0,0) at x,y = 1,1; the cell's
        outer corner is at x,y = 0.5,0.5.

        """
        return np.indices(self.shape)[::-1] + 1

    @property
    def edges(self):
        """x and y pixel coordinates of the edges of the cells in the grid.

        See `xy` for the definition of the pixel coordinate system.

        """
        return np.indices((self.nrow+1, self.ncol+1))[::-1] + 0.5

    @property
    def data_list(self):
        """Grid cell values as a flattened array.

        The order of the cells is the same as `args` and `kwargs`. If a
        cell's argument tuple or keyword dictionary are set to None, then
        the cell's value is set to `fill`.

        This is a read-only attribue because the value in a given cell is
        the result of evaluating a function with a set of arguments.
        Setting a cell's value directly would break this link.

        """
        return self._data_list

    @property
    def data_grid(self):
        """Grid cell values as a 2d array.

        This is a reshaped view of `data_list`.

        """
        return self._data_grid

    def _check_list(self, list_):
        """Fill or trim the list to the proper length."""
        n = self.nrow * self.ncol
        len_list = len(list_)
        if len_list < n:
            list_ = list_ + [None]*(n - len_list)
        elif n < len_list:
            list_ = list_[:n]
        return list_

    def _check_data(self, arr):
        """Fill or trim the array to the proper length."""
        n = self.nrow * self.ncol
        len_arr = arr.size
        if len_arr < n:
            arr = np.append(arr, [self.fill]*(n - len_arr))
        elif n < len_arr:
            arr = arr.copy()[:n]
        else:
            arr = arr.copy()
        return arr

    def _wrap_func(self):
        """Wrap self.func so that it returns self.fill when provided with
        None.

        """
        def wrapper(args, kwargs):
            if args is None or kwargs is None:
                val = self.fill
            else:
                val = self.func(*args, **kwargs)
            return val
        return wrapper

    def _apply_func(self, where=None):
        """Update the values in the data array at the given indices.

        See `update` for the `where` keyword.

        """
        func = self._wrap_func()

        if hasattr(where, 'dtype') and where.dtype == bool:
            where = np.where(where)  # get indices from boolean array
        if where is None:
            i_list = xrange(self.nrow*self.ncol)
        elif len(where) == 2 and hasattr(where[0], '__len__'):
            # get flat indices from row,col indices
            i_list = np.ravel_multi_index(where, self.shape)
        else:
            i_list = where

        for i in i_list:
            self._data_list[i] = func(self.args[i], self.kwargs[i])

        return

    def update(self, where=None):
        """Update the grid to the values of the current attributes.

        The grid data array is always copied before updating, thus breaking
        any references to `data_list` and `data_grid`.

        Parameters
        ----------
        where : list, optional
            A list of indices specifying which cells to update, as either a
            list of indices for the flattened grid or a list containing a
            list of row indices followed by a list of column indices. All
            cells are updated if None (default).

        Returns
        -------
        None

        """
        self.args = self._check_list(self.args)
        self.kwargs = self._check_list(self.kwargs)
        self._data_list = self._check_data(self._data_list)
        self._apply_func(where=where)
        self._data_grid = self._data_list.reshape(self.shape)
        return
