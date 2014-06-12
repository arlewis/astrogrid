from astropy import table as tbl, wcs
import numpy as np
import os
import subprocess

from . import config



# Shell and misc.
# ---------------

def safe_symlink(src, dst):
    """Create a symlink only if it does not already exist."""
    try:
        os.symlink(src, dst)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def is_string(obj):
    """Check if an object is a string rather than a list of strings.

    Note that a slightly more straightforward test would be, ::

      return hasattr(obj, '__iter__')

    because lists and tuples have the __iter__ method while strings do not.
    However, this will only work in Python 2.x as strings in Python 3 have
    the __iter__ method.

    """
    return ''.join(obj) == obj



# Parsers
# -------
def parse_zcbtable(zcbfile):
    """Load a zcombine output file into an astropy table.

    =========  ======= ====================================================
    columns    units   description
    =========  ======= ====================================================
    log(age_i) yr      Log age of the young (most recent) side of each bin
    log(age_f) yr      Log age of the old side of each bin
    dmod               Distance modulus
    SFR        Msun/yr Star formation rate
    SFR_eu     Msun/yr Upper error for SFR
    SFR_el     Msun/yr Lower error for SFR
    [M/H]              Metallicity, where the solar value is [M/H] = 0 [1]_
    [M/H]_eu           Upper error for [M/H]
    [M/H]_el           Lower error for [M/H]
    d[M/H]             Metallicity spread
    d[M/H]_eu          Upper error for [M/H]
    d[M/H]_el          Lower error for [M/H]
    CSF                Cumulative mass formed as a fraction of total mass
    CSF_eu             Upper error for CSF
    CSF_el             Lower error for CSF
    =========  ======= ====================================================

    .. [1] The MATCH readme uses "logZ" for metallicity, but Z is typically
       reserved for metal abundance, for which the solar value is 0.02.

    """
    names = ['log(age_i)', 'log(age_f)', 'dmod',
             'SFR', 'SFR_eu', 'SFR_el', '[M/H]', '[M/H]_eu', '[M/H]_el',
             'd[M/H]', 'd[M/H]_eu', 'd[M/H]_el', 'CSF', 'CSF_eu', 'CSF_el']
    dtypes = ['float'] * 15

    data = []
    with open(zcbfile, 'r') as f:
        for row in f:
            row = row.split()
            if row:
                try:
                    float(row[0])
                except ValueError:
                    continue
                data.append(row)
    table = tbl.Table(zip(*data), names=names, dtype=dtypes)
    return table


def parse_coordfile(coordfile):
    """Load pixel vertices into a an array.

    The coordfile contains RA,dec coordinates of pixel vertices. The format
    is one pixel per line with the vertices listed in clockwise order::

      pix1_v1_RA pix1_v1_dec pix1_v2_RA pix1_v2_dec ... pix_1_v4_dec
      pix2_v1_RA pix2_v1_dec pix2_v2_RA pix2_v2_dec ... pix_1_v4_dec
      ...
      pixn_v1_RA pixn_v1_dec pixn_v2_RA pixn_v2_dec ... pix_1_v4_dec

    In a plot with RA and dec increasing to the left and up, respectively,
    the vertices are listed clockwise starting from the upper-leftish
    corner (vertex 1) and ending in the lower-leftish corner (vertex 4).
    The pixels are listed from left to right, top to bottom in a
    config.NROW by config.NCOL grid.

    """
    data = np.genfromtxt(coordfile, dtype='float')
    shape = (2, config.NROW, config.NCOL, 4)  # axes: coord, row, col, vertex
    vertices = np.zeros(shape)
    vertices[0] = data[:,0::2].reshape(shape[1:])  # RA
    vertices[1] = data[:,1::2].reshape(shape[1:])  # dec
    return vertices



# MATCH utilities
# ---------------
def zcombine(sfhfile, outfile, param=None, best=None, bestonly=False,
             unweighted=False, medbest=False, jeffreys=False, norun=False):
    """Wrapper for the MATCH zcombine utility.

    Parameters
    ----------
    sfhfile : str or list
        Path(s) to the calcsfh SFH output file(s) to be processed.
    outfile : str
        Path for the zcombine output file.
    param : str, optional
        Path to a zcombine parameter file specifying the time bins for
        ``outfile``. Default is None.
    best : str, optional
        Default is None.
    bestonly : bool, optional
        Default is False
    unweighted : bool, optional
        Default is False
    medbest : bool, optional
        Default is False
    jeffreys : bool, optional
        Default is False
    norun : bool, optional
        If True, then the command string is not passed to subprocess.call
        (useful for testing a command before running it). Default is False.

    Returns
    -------
    string
        zcombine command string.

    """
    cmd = 'zcombine'

    if param is not None:
        cmd += ' -param={:s}'.format(param)
    if unweighted:
        cmd += ' -unweighted'
    if medbest:
        cmd += ' -medbest'
    if jeffreys:
        cmd += ' -jeffreys'
    if best is not None:
        cmd += ' -best={:s}'.format(best)

    if is_string(sfhfile):
        sfhfile = [sfhfile]
    cmd += ' {:s}'.format(' '.join(sfhfile))

    if bestonly:
        cmd += ' -bestonly'

    cmd += ' > {:s}'.format(outfile)

    if not norun:
        subprocess.call(cmd, shell=True)
    return cmd


def zcmerge(zcbfile, outfile, absolute=False, norun=False):
    """Wrapper for the MATCH zcmerge utility.

    Parameters
    ----------
    zcbfile : str or list
        Path(s) to the zcombine SFH output file(s) to be processed.
    outfile : str
        Path for the zcombine output file.
    absolute : bool, optional
        Default is False
    norun : bool, optional
        If True, then the command string is not passed to subprocess.call
        (useful for testing a command before running it). Default is False.

    Returns
    -------
    string
        zcmerge command string.

    """
    cmd = 'zcmerge'

    if is_string(zcbfile):
        zcbfile = [zcbfile]
    cmd += ' {:s}'.format(' '.join(zcbfile))

    if absolute:
        cmd += ' -absolute'

    cmd += ' > {:s}'.format(outfile)

    if not norun:
        subprocess.call(cmd, shell=True)
    return cmd



# WCS stuff
# ---------
"""
astropy.wcs already handles these transformations, but I spent some time
figuring out how they work, so here they are! They are mostly limited to
gnomonic (TAN) projections with the celestial coordinates in deg.

"""
def pix2proj(x, y, hdr):
    """Convert pixel coordinates into projection plane coordinates (deg)
    according to Greisen & Calabretta (2002).

    Required header keywords:
    CRPIX1, CRPIX2
    CD1_1, CD1_2, CD2_1, CD2_2

    CDi_j are assumed to have units deg/pix.

    """
    # (G&C02 eq.3/pg.1063/pdf.3)
    u = hdr['cd1_1']*(x-hdr['crpix1']) + hdr['cd1_2']*(y-hdr['crpix2'])
    v = hdr['cd2_1']*(x-hdr['crpix1']) + hdr['cd2_2']*(y-hdr['crpix2'])

    return u, v


def proj2pix(u, v, hdr):
    """Convert projection plane coordinates (deg) into pixel coordinates
    according to Greisen & Calabretta (2002).

    Required header keywords:
    CRPIX1, CRPIX2
    CD1_1, CD1_2, CD2_1, CD2_2

    CDi_j are assumed to have units deg/pix.

    """
    # (G&C02 eq.3/pg.1063/pdf.3)
    c = hdr['cd1_1']*hdr['cd2_2'] - hdr['cd1_2']*hdr['cd2_1']
    x = (hdr['cd2_2']*u - hdr['cd1_2']*v) / c + hdr['crpix1']
    y = (-hdr['cd2_1']*u + hdr['cd1_1']*v) / c + hdr['crpix2']

    return x, y


def proj2natsph(u, v, hdr):
    """Convert projection plane coordinates (deg) into native spherical
    coordinates (deg) assuming a gnomonic (TAN) projection according to
    Calabretta & Greisen  (2002).

    CTYPE1 and CTYPE2 are assumed to end with 'TAN'.

    """
    u = u * np.pi/180
    v = v * np.pi/180
    p = np.arctan2(u, -v)  # (C&G02 eq.14/pg.1085/pdf.9)
    r = np.sqrt(u**2 + v**2)  # (C&G02 eq.15/pg.1085/pdf.9)
    t = np.arctan(1/r)  # (C&G02 eq.55/pg.1088/pdf.12)
    p *= 180/np.pi
    t *= 180/np.pi

    return p, t


def natsph2proj(p, t, hdr):
    """Convert native spherical coordinates (deg) into projection plane
    coordinates (deg) assuming a gnomonic (TAN) projection according to
    Calabretta & Greisen  (2002).

    CTYPE1 and CTYPE2 are assumed to end with 'TAN'.

    """
    p = p * np.pi/180
    t = t * np.pi/180
    r = 1/np.tan(t)  # (C&G02 eq.54/pg.1088/pdf.12)
    u = r*np.sin(p)  # (C&G02 eq.12/pg.1085/pdf.9)
    v = -r*np.cos(p)  # (C&G02 eq.13/pg.1085/pdf.9)
    u *= 180/np.pi
    v *= 180/np.pi

    return u, v


def natsph2celsph(p, t, hdr):
    """Convert native spherical coordinates (deg) into celestial spherical
    coordinates (deg) assuming a gnomonic (TAN) projection according to
    Calabretta & Greisen (2002). The celestial coordinate system is
    determined by CTYPE1 and CTYPE2. For example, if CTYPE1 and CTYPE2 are
    'RA---TAN' and 'DEC--TAN', then a and d correspond to RA and dec (as do
    CRVAL1 and CRVAL2). The units of CRVAL1 and CRVAL2 are assumed to be
    deg.

    Required header keywords:
    CRVAL1, CRVAL2

    CTYPE1 and CTYPE2 are assumed to end with 'TAN', and CRVAL1 and CRVAL2
    are assumed to have units deg.

    """
    # TAN-specific settings
    ap, dp = hdr['crval1'], hdr['crval2']
    pp = 180.0

    # (C&G02 eq.2/pg.1079/pdf.3)
    p = p * np.pi/180
    t = t * np.pi/180
    ap *= np.pi/180
    dp *= np.pi/180
    pp *= np.pi/180
    a = ap + np.arctan2(-np.cos(t)*np.sin(p-pp),
                        np.sin(t)*np.cos(dp)-np.cos(t)*np.sin(dp)*np.cos(p-pp))
    d = np.arcsin(np.sin(t)*np.sin(dp) + np.cos(t)*np.cos(dp)*np.cos(p-pp))
    a *= 180/np.pi
    d *= 180/np.pi

    return (a, d)


def celsph2natsph(a, d, hdr):
    """Convert celestial spherical coordinates (deg) into native spherical
    coordinates (deg) assuming a gnomonic (TAN) projection according to
    Calabretta & Greisen (2002). The celestial coordinate system is
    determined by CTYPE1 and CTYPE2. For example, if CTYPE1 and CTYPE2 are
    'RA---TAN' and 'DEC--TAN', then a and d correspond to RA and dec (as do
    CRVAL1 and CRVAL2). The units of CRVAL1 and CRVAL2 are assumed to be
    deg.

    Required header keywords:
    CRVAL1, CRVAL2

    CTYPE1 and CTYPE2 are assumed to end with 'TAN', and CRVAL1 and CRVAL2
    are assumed to have units deg.

    """
    # TAN-specific settings
    ap, dp = hdr['crval1'], hdr['crval2']
    pp = 180.0

    # (C&G02 eq.5/pg.1080/pdf.4)
    a = a * np.pi/180
    d = d * np.pi/180
    ap *= np.pi/180
    dp *= np.pi/180
    pp *= np.pi/180
    p = pp + np.arctan2(-np.cos(d)*np.sin(a-ap),
                        np.sin(d)*np.cos(dp)-np.cos(d)*np.sin(dp)*np.cos(a-ap))
    t = np.arcsin(np.sin(d)*np.sin(dp) + np.cos(d)*np.cos(dp)*np.cos(a-ap))
    p *= 180/np.pi
    t *= 180/np.pi

    return p, t


def pix2world(x, y, hdr):
    """Convert pixel coordinates into RA and dec assuming a gnomonic (TAN)
    projection accoring to Calabretta & Greisen (2002).

    Required keywords in ``hdr``:
    CRPIX1, CRPIX2
    CRVAL1, CRVAL2
    CD1_1, CD1_2, CD2_1, CD2_2

    CTYPE1 and CTYPE2 are assumed to be 'RA---TAN' and 'DEC--TAN'. CDi_j
    are assumed to have units deg/pix. CRVAL1 and CRVAL2 are assumed to
    have units deg.

    """
    u, v = pix2proj(x, y, hdr)
    p, t = proj2natsph(u, v, hdr)
    a, d = natsph2celsph(p, t, hdr)
    return a, d


def world2pix(a, d, hdr):
    """Convert RA and dec into pixel coordinates assuming a gnomonic (TAN)
    projection accoring to Calabretta & Greisen (2002).

    Required keywords in ``hdr``:
    CRPIX1, CRPIX2
    CRVAL1, CRVAL2
    CD1_1, CD1_2, CD2_1, CD2_2

    CTYPE1 and CTYPE2 are assumed to be 'RA---TAN' and 'DEC--TAN'. CDi_j
    are assumed to have units deg/pix. CRVAL1 and CRVAL2 are assumed to
    have units deg.

    """
    p, t = celsph2natsph(a, d, hdr)
    u, v = natsph2proj(p, t, hdr)
    x, y = proj2pix(u, v, hdr)
    return x, y


def gcdist(lon1, lat1, lon2, lat2, deg=True):
    """Calculate the great circle distance between two points."""
    if deg:
        lon1 *= np.pi / 180
        lat1 *= np.pi / 180
        lon2 *= np.pi / 180
        lat2 *= np.pi / 180

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    l = 2 * np.arcsin(np.sqrt(
            np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2))
    if deg:
        l *= 180 / np.pi

    return l


def sparea(lon, lat, R=1, units='deg'):
    """Calculate the area of a spherical polygon.

    It is assumed that the polygon contains no poles! The method is adapted
    from,

    http://trs-new.jpl.nasa.gov/dspace/bitstream/2014/40409/1/07-03.pdf

    Parameters
    ----------
    lon, lat : 1D array
        Lists or arrays of longitudes and latitudes of the polygon vertices.
    R : float, optional
        Radius of the sphere (default is 1).
    units : {'deg', 'rad'}, optional
        Sets the unit system for ``lon`` and ``lat``.

    Returns
    -------
    float
        Total area of the spherical polygon.

    """
    lon, lat = np.array(lon), np.array(lat)
    if units == 'deg':
        lon, lat = np.pi / 180 * lon, np.pi / 180 * lat

    # Great circle segments between vertices
    l = gcdist(lon[:-1], lat[:-1], lon[1:], lat[1:], deg=False)

    # Semiperimeter of each spherical triangle
    s = 0.5 * (l + np.pi + lat[:-1] + lat[1:])

    # Spherical excess of each spherical triangle from L'Huilier's theorem.
    # Note that none of the terms should be negative (not 100% sure about
    # that); assume that any negative values are within machine precision
    # of 0.
    term1 = (s - (np.pi / 2 + lat[:-1])) /2
    term1b = np.where(term1 < 0, 0, term1)
    term2 = (s - (np.pi / 2 + lat[1:])) /2
    term2b = np.where(term2 < 0, 0, term2)
    E = 4 * np.arctan(np.sqrt(
            np.tan(s/2) * np.tan((s-l)/2) * np.tan(term1b) * np.tan(term2b)))

    # Let A<0 for lon_i+1>lon_i, A>0 for lon_i+1<lon_i assuming ccw
    # traversal
    sign = (lon[1:] < lon[:-1]) * 2 - 1

    # Total area
    A = np.sum(E * sign) * R**2
    if units == 'deg':
        A = A * (180 / np.pi)**2
    if A < 0:  # Fix the sign in case the vertices are not listed ccw
        A = -A
    return A


def calc_pixscale(hdr):
    """Calculate the pixel scale from the WCS information in a header.

    World coordinates are assumed to be in deg (e.g., CRVAL1 and CRVAL2).
    The pixel scales are returned in arcsec/pixel.

    """
    hwcs = wcs.WCS(hdr)
    xy = np.array([[hdr['crpix1']+1, hdr['crpix2']],
                   [hdr['crpix1'], hdr['crpix2']+1]])

    ad0 = hdr['crval1'], hdr['crval2']
    ad = hwcs.wcs_pix2world(xy, 1)

    scale1 = gcdist(ad0[0], ad0[1], ad[0,0], ad[0,1], deg=True) * 3600
    scale2 = gcdist(ad0[0], ad0[1], ad[1,0], ad[1,1], deg=True) * 3600
    return scale1, scale2



# Analysis stuff
# --------------

