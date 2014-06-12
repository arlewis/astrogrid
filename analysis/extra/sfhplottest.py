"""
Here I experiment with some alternative ways to show SFHs.


TODO
----

- nbins = 20 seems line a good number for determining the PDF in an age bin,
  but it makes the figure look blocky. Could smooth the mesh by doing a 2d
  interpolation of x_grid, y_grid, and density_grid to a finer resolution
  in plot_ydensity, and then pass the interpolated grids to pcolormesh.
  Could also smooth the age bins individually by doing a 1d interpolation
  of hist and y_edges.

- Indicate narrowest 68% CI in each age bin? As for best-fit SFH, these
  limits could be displayed as steps, levels, or some kind of
  interpolation (plot in a different color).

- How does this compare to a "spaghetti plot", i.e., plotting SFR vs. age
  for every single HMC test?

"""
import numpy as np
from matplotlib import pyplot as plt

from uvregions import config, utils



# Get SFH data
# ------------
# Note that it takes a few minutes to create the hmc_sfr array, so it's
# wise to cache it in a file for quick loading in the future.
name = '4308'
sfhfile_list = config.regsfh_file(name, kind='hmczcbind')

# Get time bin data
sfhfile = sfhfile_list[0]
table = utils.parse_zcbtable(sfhfile)
agei, agef = 10**table['log(age_i)'], 10**table['log(age_f)']
normfactor = (agef[0] - agei[0]) / agef[0]  # for rescaling sfr in first age bin
agei[0] = 0  # stretch first age bin to 0

# Initialize SFR array (one row per HMC iteration)
shape = (len(sfhfile_list), len(agei))
hmc_sfr = np.zeros(shape)

for i, sfhfile in enumerate(sfhfile_list):
    table = utils.parse_zcbtable(sfhfile)
    hmc_sfr[i] = table['SFR']
    hmc_sfr[i,0] *= normfactor  # rescale first age bin

filename = 'hmc_sfr_cache_{:s}.npz'.format(name)
np.savez(filename, agei, agef, hmc_sfr)






# Plotting
# --------

def plot_ydensity(ax, x_edges, y_series, range=None, histkwargs=None,
                  pckwargs=None):
    """Plot the distributions of y values in bins of x.

    For example: several measurements of y for the same set of x values.
    Plotting the *distribution* of the y values each x is an alternative to
    plotting just the mean y value plus/minus an error bar. The
    distributions are plotted using `matplotlib.pyplot.pcolormesh`.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`-like
        The axes to plot in.
    x_edges : length M+1 array
        M histograms in y are computed, where the mth histogram is for y
        values corresponding to x between ``x_edges[m]`` and
        ``x_edges[m+1]``.
    y_series : KxM array
        A series of K sets of y values. Each set is a length-M array, where
        the mth y value corresponds to x between ``x_edges[m]`` and
        ``x_edges[m+1]``::

          y_series = np.array([[y11, y12, ... y1M],
                               [y21, y22, ... y2M],
                               ...
                               [yK1, yK2, ... yKM]])

        The mth histogram is computed for all ykm with k={1,2,..K}, i.e.,
        ``y_series[:,m]``.

    range : tuple, optional
        (ymin, ymax); each histogram is calculated for y values between
        ymin and ymax. Sets the `range` keyword for `np.histogram`.
    histkwargs : dict, optional
        Keyword arguments for `np.histogram`. Note that the `bins` keyword
        can be used to set the number of y bins (N).
    pckwargs : dict, optional
        Keyword arguments for `matplotlib.pyplot.pcolormesh`.

    Returns
    -------
    tuple
        x edges ((M+1)x(N+1) array), y edges ((M+1)x(N+1) array), density
        grid (MxN array), and `matplotlib.collections.QuadMesh`.

    """
    if histkwargs is None:
        histkwargs = {}
    histkwargs['range'] = range

    density_list = []
    for y_list in y_series.T:  # len K, m={1...M}
        hist, y_edges = np.histogram(y_list, **histkwargs)  # len N, len N+1
        density_list.append(hist)

    y_grid, x_grid = np.meshgrid(y_edges, x_edges)  # (M+1)x(N+1)
    density_grid = np.zeros(x_grid.shape)[:-1,:-1]  # MxN
    for m, density in enumerate(density_list):  # len N, m={1...M}
        density_grid[m] = density

    if pckwargs is None:
        pckwargs = {}

    mesh = ax.pcolormesh(x_grid, y_grid, density_grid, **pckwargs)

    return x_grid, y_grid, density_grid, mesh


# Get data
filename = 'hmc_sfr_cache_4308.npz'
agei = np.load(filename)['arr_0']
agef = np.load(filename)['arr_1']
hmc_sfr = np.load(filename)['arr_2']


# Settings
sfr_lim = None  # default to max
age_lim1 = 0  # yr
age_lim2 = 200e6  # yr
age_units = 1e6  # Desired age units for plotting
nbins = 20  # Number of bins for PDF calculation in each age bin
cmap = plt.cm.gray_r


# Slice to age range of interest; convert to age_units
i1 = np.where((agei <= age_lim1) & (age_lim1 < agef))[0][0]
i2 = np.where((agei <= age_lim2) & (age_lim2 < agef))[0][0]
agei, agef = agei[i1:i2+1], agef[i1:i2+1]
hmc_sfr = hmc_sfr[:,i1:i2+1]

agei /= age_units
agef /= age_units
age_lim1 /= age_units
age_lim2 /= age_units



# Layout
fig_dx, fig_dy = 8.0, 6.0
ax_x0, ax_y0 = 0.15, 0.15
ax_dx, ax_dy = 0.84, 0.84


# Draw figure
if sfr_lim is None:
    sfr_lim = hmc_sfr.max()
fig = plt.figure(figsize=(fig_dx, fig_dy))
ax = fig.add_axes([ax_x0, ax_y0, ax_dx, ax_dy])
ax.axis([age_lim2, age_lim1, 0, sfr_lim])


# Plot SFR densities
age_edges = np.append(agei, agef[-1])
age_grid, sfr_grid, density_grid, mesh = plot_ydensity(
        ax, age_edges, hmc_sfr, range=(0, sfr_lim),
        histkwargs={'bins': nbins}, pckwargs={'cmap': cmap}
        )

# Plot median SFR (could also use best-fit SFH)
sfr0 = np.median(hmc_sfr, axis=0)
x = np.vstack((agei, agef)).T.ravel()
y = np.repeat(sfr0, 2)
line, = ax.plot(x, y, 'r-')



# Experiment with different styles
def test(sfr_lim, nbins, mode='step', kind='linear'):
    plt.clf()
    ax = fig.add_axes([ax_x0, ax_y0, ax_dx, ax_dy])
    ax.axis([age_lim2, age_lim1, 0, sfr_lim])


    # Plot SFR densities
    age_edges = np.append(agei, agef[-1])
    age_grid, sfr_grid, density_grid, mesh = plot_ydensity(
            ax, age_edges, hmc_sfr, range=(0, sfr_lim),
            histkwargs={'bins': nbins}, pckwargs={'cmap': cmap}
            )

    # Plot median SFR (could also use best-fit SFH)
    sfr0 = np.median(hmc_sfr, axis=0)
    if mode == 'step':
        x = np.vstack((agei, agef)).T.ravel()
        y = np.repeat(sfr0, 2)
        line, = ax.plot(x, y, 'r-')
    elif mode == 'level':
        lines = ax.hlines(sfr0, agei, agef, colors='r', linestyles='-')
    elif mode == 'interp':
        x = (agef + agei) / 2.0
        interp = interpolate.interp1d(x, sfr0, kind=kind)
        x = np.linspace(x[0], x[-1], 1000)
        y = interp(x)
        line, = ax.plot(x, y, 'r-')

    return None

test(0.4, 20, mode='interp', kind=3)

