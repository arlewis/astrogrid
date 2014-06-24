"""

==============
`sfhmaps.flux`
==============

Utilities for calculating integrated SEDs and magnitudes from SFHs using
FSPS.


Functions
---------

========== ==============================================================
`calc_mag` Calculate the magnitude of an SED in a given filter.
`calc_sed` Calculate the SED for a binned SFH.
`get_zmet` Return the closest FSPS `zmet` integer for the given log metal
           abundance.
`mag2flux` Convert AB magnitude in a filter to flux (erg s-1 cm-2 A-1).
========== ==============================================================

"""
import bursty_sfh as bsp  # scombine package
import fsps
import numpy as np
from sedpy import observate, attenuation

from . import util


CURRENT_SP = []  # Container for fsps.StellarPopulation


def get_zmet(logZ):
    """Return the closest FSPS `zmet` integer for the given log metal
    abundance.

    Parameters
    ----------
    logZ : float
        Log metal abundance.

    Returns
    -------
    int

    Notes
    -----
    `zmet` parameter values vs. metal abundance for Padova isochrones and
    BaSeL stellar library from the FSPS manual (Zsun = 0.0190):

    ==== ====== ===========
    zmet Z      log(Z/Zsun)
    ==== ====== ===========
       1 0.0002       -1.98
       2 0.0003       -1.80
       3 0.0004       -1.68
       4 0.0005       -1.58
       5 0.0006       -1.50
       6 0.0008       -1.38
       7 0.0010       -1.28
       8 0.0012       -1.20
       9 0.0016       -1.07
      10 0.0020       -0.98
      11 0.0025       -0.89
      12 0.0031       -0.79
      13 0.0039       -0.69
      14 0.0049       -0.59
      15 0.0061       -0.49
      16 0.0077       -0.39
      17 0.0096       -0.30
      18 0.0120       -0.20
      19 0.0150       -0.10
      20 0.0190       +0.00
      21 0.0240       +0.10
      22 0.0300       +0.20
    ==== ====== ===========

    """
    zmet_lookup = np.array([
        [1, -1.98],
        [2, -1.80],
        [3, -1.68],
        [4, -1.58],
        [5, -1.50],
        [6, -1.38],
        [7, -1.28],
        [8, -1.20],
        [9, -1.07],
        [10, -0.98],
        [11, -0.89],
        [12, -0.79],
        [13, -0.69],
        [14, -0.59],
        [15, -0.49],
        [16, -0.39],
        [17, -0.30],
        [18, -0.20],
        [19, -0.10],
        [20, 0.00],
        [21, 0.10],
        [22, 0.20]
        ])

    i = np.abs(logZ - zmet_lookup[:,1]).argmin()
    zmet = int(zmet_lookup[i,0])

    return zmet


def calc_sed(sfr, age, **kwargs):
    """Calculate the SED for a binned SFH.

    The form of the input SFH is a set of age bins with a SFR value
    assigned to each bin. At each point in time in the SFH, an SED is
    generated using an `fsps.StellarPopulation` instance. It can take
    several seconds to create the instance initially, but it is saved in
    `CURRENT_SP` and is reused to save time for subsequent calls to
    `calc_sed`. If needed, the current `fsps.StellarPopulation` instance
    can be cleared with ``CURRENT_SP.pop(0)``.

    All stellar population synthesis is performed under the hood using FSPS
    with the Padova isochrones and BaSeL stellar library. `python-fsps`
    provides the python interface, and integrated SEDs are computed from
    input SFHs using the `scombine` and `sedpy` packages. This function
    merely puts everything together under a single interface for
    convenience.

    Parameters
    ----------
    sfr : array-like
        SFR values (Msun yr-1) for the bins in the SFH.
    age : tuple or array-like
        Ages (i.e., lookback times, yr) of the bin edges in the SFH. In the
        tuple form, the first element is an array of ages for the young
        edges of the bins, and the second element gives the ages for the
        old edges. Both arrays have the same length as `sfr`.
        Alternatively, a single array of length ``len(sfr)+1`` may be
        specified for the ages of all of the bin edges. SFH bins are
        assumed to be in order of increasing age, so the youngest bins
        (closest to the present) are listed first and the oldest bins
        (closest to the big bang) are listed last.
    age_observe : float or list, optional
        Age (i.e., lookback times, yr) at which the SED is calculated. May
        also be a list of ages. Default is 1. Note that 0 will throw a
        RuntimeWarning about dividing by zero. It is safer to use a small
        positive number instead (hence 1 yr as the default).
    bin_res : float, optional
        Time resolution factor used for resampling the input SFH. The time
        step in the resampled SFH is equal to the narrowest age bin divided
        by this number. Default is 20.
    av, dav : float, optional
        Foreground and differential V-band extinction parameters. Default
        is None (0). [1]_
    nsplit : int, optional
        Number of pieces in which to split the spectrum. Default is 30. [1]_
    dust_curve : function, optional
        Function that returns ``A_lambda/A_V`` (extinction normalized to
        the total V-band extinction) for a given array of input wavelengths
        in Angstroms (see the `attenuation` module in `sedpy`). Default is
        `sedpy.attentuation.cardelli`. [1]_
    fsps_kwargs : dict, optional
        Dictionary of valid keyword arguments for `fsps.StellarPopulation`.
        Default is an empty dictionary. 'sfh'=0 will always be used.

    Returns
    -------
    tuple
        A tuple containing an array of wavelengths (in Angstroms) and an
        array of spectral flux values (Lsun A-1) at each wavelength. If
        `age_observe` is a list, then the spectrum array has shape
        (len(age_observe), len(wave)), otherwise it has shape (len(wave),).
        
    Notes
    -----

    .. [1] The two-component extinction model assumes that V-band
       extinctions in a stellar population follow a uniform distribution
       from `av` to `av` + `dav`. `av` sets the total foreground V-band
       extinction common to all stars and `dav` is the maximum amount of
       differential V-band extinction. A reddened SED is obtained by
       splitting the intrinsic SED into `nsplit` equal pieces, reddening
       each piece according to the assumed extinction curve (`dust_curve`)
       and a random V-band extinction value drawn from the model, and then
       summing the pieces back together.

    """
    if len(age)==2 and util.islistlike(age[0]):
        age = np.append(age[0], age[1][-1])  # One array of bin edges

    age_list = kwargs.get('age_observe', 1)
    if util.islistlike(age_list):
        len_age_list = len(age_list)
    else:
        age_list = [age_list]
        len_age_list = 0

    bin_res = kwargs.get('bin_res', 20.0)
    av, dav = kwargs.get('av', None), kwargs.get('dav', None)
    nsplit = kwargs.get('nsplit', 30)
    dust_curve = kwargs.get('dust_curve', attenuation.cardelli)
    fsps_kwargs = kwargs.get('fsps_kwargs', {})

    # To save time, create StellarPopulation only when necessary
    try:
        sp = CURRENT_SP[0]
    except IndexError:
        sp = fsps.StellarPopulation()
        CURRENT_SP.append(sp)
    fsps_kwargs['sfh'] = 0
    for key, val in fsps_kwargs.items():
        sp.params[key] = val

    # Resample the SFH to a high time resolution
    #
    # Notes on bsp.burst_sfh:
    #
    # - `sfh` is a record array with columns 't1', 't2', and 'sfr'.
    # - Only interested in upsampling, not any burst stuff, so `f_burst`
    #   must be 0. Other parameters don't matter.
    #
    dtypes = [('t1', 'float'), ('t2', 'float'), ('sfr', 'float')]
    sfh = np.array(zip(age[:-1], age[1:], sfr), dtypes)
    age, sfr = bsp.burst_sfh(f_burst=0, sfh=sfh, bin_res=bin_res)[:2]

    # Spectrum at each observation age
    output = bsp.bursty_sps(age_list, age, sfr, sp, av=av, dav=dav,
                            nsplit=nsplit, dust_curve=dust_curve)
    if av is None or dav is None:
        wave, spec, weights = output
        lum_ir = None
    else:
        wave, spec, weights, lum_ir = output

    if not len_age_list:
        spec = spec[0]

    return wave, spec


def calc_mag(wave, spec, band, dmod=None):
    """Calculate the magnitude of an SED in a given filter.

    Parameters
    ----------
    wave : array
        1d array of wavelengths (in Angstroms) where the SED is defined.
    spec : array
        Spectral flux values (Lsun A-1) at each wavelength in `wave`. The
        array may be a single spectrum of shape (len(wave),), or contain N
        individual spectra such that the shape is (N, len(wave)).
    band : str or list
        Filter, or a list of filters, within which to calculate absolute
        magnitude. See `fsps.find_filter` or `fsps.fsps.FILTERS` for valid
        filter names.
    dmod : float, optional
        Distance modulus used to calculate apparent magnitudes. If given,
        the returned magnitudes are apparent instead of absolute. Default
        is None.

    Returns
    -------
    float or array
        AB magnitudes in the given filters. [1]_ If `spec` is a single
        spectrum and `band` is one name, then a float is returned. A 1d
        array is returned if either `spec` or `band` contain multiple
        values. If both `spec` and `band` contain multiple values, then a
        2d array of shape (len(spec), len(band)) is returned.

    Notes
    -----

    .. [1] The SED data is processed using `observate.getSED`, which always
       returns AB magnitudes.

    """
    spec_list = spec
    if spec_list.ndim == 1:
        spec_list = np.expand_dims(spec_list, 0)
        len_spec_list = 0
    else:
        len_spec_list = len(spec_list)

    band_list = band
    if util.islistlike(band_list):
        len_band_list = len(band_list)
    else:
        band_list = [band_list]
        len_band_list = 0

    band_list = observate.load_filters(band_list)
    spec_list = spec_list * bsp.to_cgs  # erg s-1 cm-2 A-1
    mags = observate.getSED(wave, spec_list, filterlist=band_list)  # Absolute

    if dmod is not None:
        mags += dmod  # Apparent

    if not len_spec_list and not len_band_list:
        mags = float(mags)
    else:
        if len_band_list == 1:
            mags = np.expand_dims(mags, 1)
        if len_age_list == 1:
            mags = np.expand_dims(mags, 0)

    return mags


def _galex_cps2flux(cps, band):
    """Counts per second to flux (erg s-1 cm-2 A-1)."""
    scale = {'galex_fuv': 1.40e-15, 'galex_nuv': 2.06e-16}
    return scale[band] * cps


def _galex_flux2cps(flux, band):
    """Flux (erg s-1 cm-2 A-1) to counts per second."""
    scale = {'galex_fuv': 1.40e-15, 'galex_nuv': 2.06e-16}
    return  flux / scale[band]


def _galex_cps2mag(cps, band):
    """Counts per second to AB magnitude."""
    zeropoint = {'galex_fuv': 18.82, 'galex_nuv': 20.08}
    return -2.5 * np.log10(cps) + zeropoint[band]


def _galex_mag2cps(mag, band):
    """AB magnitude to counts per second."""
    zeropoint = {'galex_fuv': 18.82, 'galex_nuv': 20.08}
    return 10**(0.4 * (zeropoint[band] - mag))


def _galex_flux2mag(flux, band):
    """Flux (erg s-1 cm-2 A-1) to AB magnitude."""
    return _galex_cps2mag(_galex_flux2cps(flux, band), band)


def _galex_mag2flux(mag, band):
    """AB magnitude to flux (erg s-1 cm-2 A-1)."""
    return _galex_cps2flux(_galex_mag2cps(mag, band), band)


def mag2flux(mag, band):
    """Convert AB magnitude in a filter to flux (erg s-1 cm-2 A-1).

    Parameters
    ----------
    mag : float or array
        Input AB magnitude(s) in the given filter.
    band : str
        Name of the filter corresponding to `mag`. See `fsps.find_filter`
        or `fsps.fsps.FILTERS` for valid filter names.       

    Returns
    -------
    float or array
        Flux in erg s-1 cm-2 A-1.

    """
    if band in ['galex_fuv', 'galex_nuv']:
        flux = _galex_mag2flux(mag, band)
    else:
        # Raise error?
        flux = None

    return flux
