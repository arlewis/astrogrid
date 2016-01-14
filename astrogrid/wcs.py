"""

===============
`astrogrid.wcs`
===============

Utilities for working with world coordinate systems.


Functions
---------

======================== ==================================================
`calc_pixscale`          Calculate the x and y pixel scales from the WCS
                         information in a FITS header.
`fit_cdmatrix`           Fit a CD matrix for a set of points with known
                         pixel and world coordinates.
`make_header`            Create a FITS header from a set of points with
                         known pixel and world coordinates given a
                         celestial coordinate system and projection.
`gcdist`                 Calculate the great circle distance between the
                         given endpoints.
`sparea`                 Calculate the area of a spherical polygon.
`separation_deprojected` Calculate the deprojected linear separation
                         between coordinates in a spiral galaxy assuming a
                         distance, a position angle, and an inclination.
======================== ==================================================

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import astropy.coordinates
import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np
import scipy.optimize

from pdb import set_trace

def calc_pixscale(hdr, ref='crpix', units=None):
    """Calculate the x and y pixel scales from the WCS information in a
    FITS header.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        Header object with WCS keywords. Must have the keys 'naxis1',
        'naxis2', 'crpix1', 'crpix2', 'crval1', and 'crval1'.
    ref : {'crpix', 'center', tuple}, optional
        If 'crpix' (default), the reference pixel is set to CRPIX1,CRPIX2.
        'center' indicates that the central pixel in the image should be
        used. An x,y tuple of floats may be given to use a specific
        reference pixel instead.
    units : tuple, optional
        The units of the longitude and latitude coordinates as a tuple of
        `astropy.units.core.Unit` instances. If None (default), then the
        units will be determined from `hdr`.

    Returns
    -------
    astropy.coordinates.Angle
        An `Angle` instance containing the x and y pixel scales.

    Updates
    -------
    added naxis keyword to astropy.wcs.WCS call to ensure only 2 axes are
    used. - A. R. Lewis 03/30/2015

    """
    wcs = astropy.wcs.WCS(hdr, naxis=2)
    if ref == 'crpix':
        x, y = hdr['crpix1'], hdr['crpix2']
    elif ref == 'center':
        x, y = hdr['naxis1']//2, hdr['naxis2']//2
    else:
        x, y = ref

    if units is None:
        units = wcs.wcs.cunit

    lon, lat = wcs.wcs_pix2world([x, x+1, x], [y, y, y+1], 1)
    if (astropy.version.major < 1) & (astropy.version.minor < 4):
        # Makes no difference whether ICRS, Galactic, AltAz, etc.
        points = astropy.coordinates.ICRS(lon, lat, unit=units)
    else:
        # Makes no difference whether ICRS, Galactic, AltAz, etc.
        points = astropy.coordinates.SkyCoord(lon, lat, frame='icrs', unit=units)
    dxy = astropy.coordinates.Angle([points[0].separation(points[1]),
                                     points[0].separation(points[2])])
    return dxy


def fit_cdmatrix(x, y, lon, lat, hdr):
    """Fit a CD matrix for a set of points with known pixel and world
    coordinates.

    The world coordinates are transformed to projection plane coordinates
    for a given projection, and then a 2d least squares fitting method is
    used to determine the best-fit CD matrix that transforms the pixel
    coordinates into projection plane coordinates.

    Parameters
    ----------
    x, y : ndarray
        x and y pixel coordinates.
    lon, lat : ndarray
        Celestial longitude and latitude (world) coordinates.
    hdr : astropy.io.fits.Header
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2
        - CRPIX1, CRPIX2
        - CRVAL1, CRVAL2

    Returns
    -------
    ndarray
        The best-fit CD matrix, ``[[CD1_1, CD1_2], [CD2_1, CD2_2]]``.

    """
    def residuals(p, dx, dy, ip):
        # i represents x or y;
        # xp - (cd11*dx + cd12*dy) and yp - (cd21*dx + cd22*dy)
        cdi1, cdi2 = p
        err = ip - (cdi1*dx + cdi2*dy)
        return err

    # Calculate projection plane coords Converting celestial coordinates
    # into pixel coordinates involves 1) celestial spherical to native
    # spherical, 2) native spherical to projection plane, and 3) projection
    # plane to pixel. Steps 1 and 2 depend on the CTYPEi and CRVALi. Step 3
    # depends on CDi_j and CRPIXi. All are known except CDi_j. By setting
    # the CD matrix to a unity matrix, the `wcs_world2pix` method converts
    # celestial coordinates through step 2 to projection plane coordinates.
    # The CD matrix can then be fit because the coordinates before and
    # after step 3 are known.
    wcs = astropy.wcs.WCS(hdr)
    wcs.wcs.cd = np.array([[1, 0], [0, 1]])
    xp, yp = wcs.wcs_world2pix(lon, lat, 1)
    xp, yp = xp - hdr['CRPIX1'], yp - hdr['CRPIX2']

    # Solve for CD elements
    dx, dy = x - hdr['CRPIX1'], y - hdr['CRPIX2']
    p_init1 = (1, 0)
    p_init2 = (0, 1)
    args1 = (dx.ravel(), dy.ravel(), xp.ravel())
    args2 = (dx.ravel(), dy.ravel(), yp.ravel())
    cd11, cd12 = scipy.optimize.leastsq(residuals, p_init1, args=args1)[0]
    cd21, cd22 = scipy.optimize.leastsq(residuals, p_init2, args=args2)[0]

    return np.array([[cd11, cd12], [cd21, cd22]])


def make_header(x, y, lon, lat, ctype1='RA---TAN', ctype2='DEC--TAN',
                ref=None):
    """Create a FITS header from a set of points with known pixel and world
    coordinates given a celestial coordinate system and projection.

    `fit_cdmatrix` is used to find the best-fit CD matrix for the given
    data.

    Parameters
    ----------
    x, y : ndarray
        x and y pixel coordinates.
    lon, lat : ndarray
        Celestial longitude and latitude (world) coordinates.
    ctype1, ctype2 : str, optional
        Values for the CTYPE1 and CTYPE2 header keywords; describes the
        coordinate system of `lon` and `lat` and sets the projection.
    ref : tuple, optional
        A tuple of containing the (x, y, lon, lat) coordinates of a chosen
        reference pixel. If None (default), then the reference pixel is
        automatically chosen as the most central point in the x and y data.

    Returns
    -------
    astropy.io.fits.Header
        A minimal header containing the WCS information that describes the
        data, including the best-fit CD matrix.

    """
    # Make a new header
    hdr = astropy.io.fits.Header()
    hdr['WCSAXES'] = 2
    hdr['CTYPE1'], hdr['CTYPE2'] = ctype1, ctype2

    # Reference point
    if ref:
        hdr['CRPIX1'], hdr['CRPIX2'] = ref[0], ref[1]
        hdr['CRVAL1'], hdr['CRVAL2'] = ref[2], ref[3]
    else:
        # Use the most central data point
        xmid = (x.max() - x.min())/2 + x.min()
        ymid = (y.max() - y.min())/2 + y.min()
        r = np.sqrt((x-xmid)**2 + (y-ymid)**2)
        idx = tuple(i[0] for i in np.where(r == r.min()))
        hdr['CRPIX1'], hdr['CRPIX2'] = x[idx], y[idx]
        hdr['CRVAL1'], hdr['CRVAL2'] = lon[idx], lat[idx]

    # CD matrix
    cd = fit_cdmatrix(x, y, lon, lat, hdr)
    hdr['CD1_1'], hdr['CD1_2'] = cd[0, 0], cd[0, 1]
    hdr['CD2_1'], hdr['CD2_2'] = cd[1, 0], cd[1, 1]

    return hdr


def gcdist(lonlat1, lonlat2, unit='deg'):
    """Calculate the great circle distance between the given endpoints.

    Basically just a convenience function for making `SkyCoord` instances
    and calling the `separation` method.

    Parameters
    ----------
    lonlat1, lonlat2 : SkyCoord or (2,) tuple of array_like
        Longitude and latitude of the great circle line endpoints.
        (`SkyCoord` is `astropy.coordinates.sky_coordinate.SkyCoord`.)
    unit : str or astropy.units.core.Unit, optional
        Units of the input coordinates. Default is 'deg' (degrees).

    Returns
    -------
    astropy.coordinates.angles.Angle

    """
    ### Here is the haversine formula for reference:
    #l = 2 * np.arcsin(np.sqrt(np.sin((lat[1:] - lat[:-1]) / 2)**2 +
    #                          np.cos(lat[:-1]) * np.cos(lat[1:]) *
    #                          np.sin((lon[1:] - lon[:-1]) / 2)**2))

    if not isinstance(lonlat1, astropy.coordinates.SkyCoord):
        lonlat1 = astropy.coordinates.SkyCoord(*lonlat1, unit=unit)
        lonlat2 = astropy.coordinates.SkyCoord(*lonlat2, unit=unit)
    l = lonlat1.separation(lonlat2)
    return l


def sparea(lon, lat, R=1, unit='deg', axis=0):
    """Calculate the area of a spherical polygon.

    It is assumed that the polygon contains no poles! The method is adapted
    from "Some algorithms for polygons on a sphere" by Chamberlain and
    Duquette (http://trs-new.jpl.nasa.gov/dspace/handle/2014/40409)

    Parameters
    ----------
    lon, lat : array_like
        Longitude and latitude of each vertex in the polygon. The first
        vertex may be listed either only at the beginning of the sequence,
        or both at the beginning and at the end of the sequence.
        Multidimensional arrays can be used to compuate areas for more than
        one polygon at once (e.g., a 2d array where each row is a polygon
        and vertices are in the columns; see the `axis` keyword). All
        polygons must have the same number of vertices.
    R : float, optional
        Radius of the sphere. Default is 1.
    unit : str or astropy.units.core.Unit, optional
        Units for `lon` and `lat`. Default is 'deg' (degrees).
    axis : int, optional
        The axis along which the polygon vertices are listed. For example,
        if `lon` and `lat` are 2d arrays where each row is a polygon and
        vertices are in the columns, then `axis` should be set to 1.
        Default is 0.

    Returns
    -------
    astropy.units.quantity.Quantity
        Total area of the spherical polygon in units of ``lonunits *
        latunits * R**2``. Note: the `Quantity` instance only knows about
        the solid angle units; the units of `R` are not attached!

    """
    # Ensure ndarray and one polygon per row
    lon, lat = np.asarray(lon), np.asarray(lat)
    if lon.ndim == 1:
        lon, lat = lon[None,:], lat[None,:]
    else:
        lon = np.rollaxis(lon, axis).reshape(lon.shape[axis], -1).T
        lat = np.rollaxis(lat, axis).reshape(lat.shape[axis], -1).T

    # Ensure the polygon is closed (check first row for efficiency)
    if not lon[0,-1] == lon[0,0]:
        lon = np.append(lon, lon[:,0:1], axis=1)
        lat = np.append(lat, lat[:,0:1], axis=1)

    try:
        # Assuming unit is a string
        unit = getattr(astropy.units, unit)
    except TypeError:
        # unit is probably a member of astropy.units
        pass
    try:
        # Assuming unit is list-like
        lonunit, latunit = unit
    except TypeError:
        # unit is single unit instance
        lonunit, latunit = unit, unit
    lonlat = astropy.coordinates.SkyCoord(lon, lat, unit=(lonunit, latunit))
    lon, lat = lonlat.ra.rad, lonlat.dec.rad

    # Great circle segments between vertices
    l = lonlat[:,:-1].separation(lonlat[:,1:]).rad

    # Semiperimeter of each spherical triangle
    s = 0.5 * (l + np.pi + lat[:,:-1] + lat[:,1:])

    # Spherical excess of each spherical triangle from L'Huilier's theorem.
    # Note that none of the terms should be negative (not 100% sure about
    # that); assume that any negative values are within machine precision
    # of 0.
    term1 = (s - (np.pi / 2 + lat[:,:-1])) /2
    term1[term1<0] = 0
    term2 = (s - (np.pi / 2 + lat[:,1:])) /2
    term2[term2<0] = 0
    result = np.tan(s/2) * np.tan((s-l)/2) * np.tan(term1) * np.tan(term2)
    E = 4 * np.arctan(np.sqrt(result))

    # Let A<0 for lon[i]<lon[i+1], A>0 for lon[i+1]<lon[i] assuming ccw
    # traversal (looking at the sky, where lon increases to the east)
    sign = 2*(lon[:,1:] < lon[:,:-1]) - 1

    # Total area
    A = np.sum(sign * E, axis=1) * R**2
    A = np.absolute(A)  # Fix the sign in case the vertices are not listed ccw
    A = (A * astropy.units.rad**2).to(lonunit * latunit)  # Attach units
    return A


def separation_deprojected(coord1, coord2, dist, pa, inc):
    """Calculate the deprojected linear separation between coordinates in a
    spiral galaxy assuming a distance, a position angle, and an
    inclination.

    Parameters
    ----------
    coord1, coord2 : astropy.coordinates.sky_coordinates.SkyCoord
        Endpoints for the separation calculation.
    dist : float or astropy.units.Quantity
        Distance to the galaxy.
    pa : float
        Position angle of the galaxy in degrees.
    inc : float
        Inclination angle of the galaxy's disk in degrees.

    Returns
    -------
    float or astropy.units.Quantiy
        Type is determined by `dist`.

    Notes
    -----
    Probably should not use this to calculate areas (e.g., multiplying
    deprojected side lengths for a rectangular region). Rather, just
    calculate the spherical polygon area and deproject by dividing by the
    cosine of the disk inclination.

    """
    angular_sep = coord1.separation(coord2)
    pa_coords = coord1.position_angle(coord2)

    projected_length = dist * angular_sep.rad
    theta = np.radians(pa) - pa_coords.rad
    deprojected_length = projected_length * np.sqrt(
        np.cos(theta)**2 + np.sin(theta)**2 / np.cos(np.radians(inc))**2)
    return deprojected_length



# Tests
# -----
def _make_header_imwcs(filename, x, y, lon, lat):
    """Create a FITS header using imwcs; a sanity check for `make_header`.

    The program requires an existing .fits file (the WCS in the header is
    ignored). A new file is written with the suffix 'w.fits'.

    This seems to give results similar to `make_header`, providing some
    validation for the CD matrix fitting method. This was tested on most of
    the PHAT bricks, and the maximum difference between input and output
    x,y was ~2% (0.02 pixels) for both methods. For most bricks, imwcs was
    marginally better at reproducing the input x,y from the known RA,dec,
    however it produced a fairly inaccurate solution for brick 12
    (discrepancies up to 1-2 pixels between input and output x,y)
    suggesting that `make_header` is more robust.

    """
    import os
    import subprocess

    lon = lon * 24./360  # hours
    coords = []
    for xp, yp, RA, DEC in zip(x.ravel(), y.ravel(), lon.ravel(), lat.ravel()):
        h = int(RA)
        m = int((RA - h) * 60)
        s = (((RA - h) * 60) - m) * 60
        hms = '{0:02d} {1:d} {2:8.5f}'.format(h, m, s)

        d = int(DEC)
        m = int((DEC - d) * 60)
        s = (((DEC - d) * 60) - m) * 60
        dms = '{0:02d} {1:d} {2:8.5f}'.format(d, m, s)

        cstr = ('{0:15.8f} {1:15.8f}    {2:s}    {3:s}\n'
                .format(xp, yp, hms, dms))
        coords.append(cstr)

    dirname = os.path.dirname(filename)
    coordlistfile = os.path.join(dirname, 'coords.dat')
    with open(coordlistfile, 'w') as f:
        f.writelines(coords)

    filename = os.path.basename(filename)
    coordlistfile = os.path.basename(coordlistfile)
    cmd = ('cd {0:s}; imwcs -ew -n 8 -u {1:s} {2:s}'
           .format(dirname, coordlistfile, filename))
    subprocess.call(cmd, shell=True)

    return None
