"""

=============
`sfhmaps.wcs`
=============

Utilities for working with world coordinate systems.


Functions
---------

=============== =======================================================
`calc_pixscale` Calculate the pixel scale from the WCS information in a
                FITS header.
`gcdist`        Calculate the great circle distance between two points.
=============== =======================================================

"""
from astropy import wcs
import numpy as np


def gcdist(lon1, lat1, lon2, lat2, deg=True):
    """Calculate the great circle distance between two points.

    Uses the law of haversines.

    Parameters
    ----------
    lon1, lat1 : float or array-like
        Longitude and latitude coordinates of the first point.
    lon2, lat2 : float or array-like
        Longitude and latitude coordinates of the second point.
    deg : bool, optional
        If True (default), the coordinate values are assumed to be in
        degrees and the returned distance is in degrees as well. If False,
        then the coordinates and distance are in radians.

    Returns
    -------
    float or array

    """
    if deg:
        lon1 = lon1 * np.pi / 180
        lat1 = lat1 * np.pi / 180
        lon2 = lon2 * np.pi / 180
        lat2 = lat2 * np.pi / 180

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    haversine_lat = np.sin(dlat/2)**2
    haversine_lon = np.sin(dlon/2)**2
    haversine_l = haversine_lat + np.cos(lat1)*np.cos(lat2)*haversine_lon
    l = 2 * np.arcsin(np.sqrt(haversine_l))
    if deg:
        l *= 180 / np.pi

    return l


def calc_pixscale(hdr, ref='crpix'):
    """Calculate the pixel scale from the WCS information in a FITS header.

    World coordinates are assumed to be in deg (e.g., CRVAL1 and CRVAL2).
    The x and y pixel scales are returned in arcsec/pixel.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        Header object with WCS keywords. Must have the keys 'naxis1',
        'naxis2', 'crpix1', 'crpix2', 'crval1', and 'crval1'.
    ref : {'crpix', 'central', tuple}, optional
        If 'crpix' (default), the reference pixel is set to CRPIX1,CRPIX2.
        'central' indicates that the central pixel in the image should be
        used. An x,y tuple of floats may be given to use a specific
        reference pixel instead.

    Returns
    -------
    tuple
        The pixel scales along the x and y directions.

    """
    hwcs = wcs.WCS(hdr)

    if ref == 'crpix':
        x0, y0 = hdr['crpix1'], hdr['crpix2']
        ad0 = hdr['crval1'], hdr['crval2']
    elif ref == 'central':
        x0, y0 = hdr['naxis1']/2, hdr['naxis2']/2
        a0, d0 = hwcs.wcs_pix2world(x0, y0, 1)
        ad0 = [a0, d0]
    else:
        x0, y0 = ref
        a0, d0 = hwcs.wcs_pix2world(x0, y0, 1)
        ad0 = [a0, d0]

    xy = np.array([[x0+1, y0], [x0, y0+1]])

    ad = hwcs.wcs_pix2world(xy, 1)

    scale1 = gcdist(ad0[0], ad0[1], ad[0,0], ad[0,1], deg=True) * 3600
    scale2 = gcdist(ad0[0], ad0[1], ad[1,0], ad[1,1], deg=True) * 3600
    return scale1, scale2
