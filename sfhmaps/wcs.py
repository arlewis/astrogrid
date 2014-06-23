"""

=============
`sfhmaps.wcs`
=============

Utilities for working with world coordinate systems.


Functions
---------

=============== =======================================================
`parse_ctype`   Get the coordinate system and projection for a given value
                of the CTYPEi keyword.
`pix2proj`      Convert pixel coordinates into projection plane coordinates
                according to Eq. 3 in Greisen & Calabretta (2002).
`proj2pix`      Convert projection plane coordinates into pixel coordinates
                according to Eq. 3 in Greisen & Calabretta (2002).
`proj2natsph`   Convert projection plane coordinates into native spherical
                coordinates for a given projection according to Calabretta
                & Greisen (2002).
`natsph2proj`   Convert native spherical coordinates into projection plane
                coordinates for a given projection according to Calabretta
                & Greisen (2002).
`natsph2celsph` Convert native spherical coordinates for a given projection
                into the given celestial spherical coordinates according to
                Calabretta & Greisen (2002).
`celsph2natsph` Convert the given celestial spherical coordinates into
                native spherical coordinates for a given projection
                according to Calabretta & Greisen (2002).
`pix2world`     Convert pixel coordinates into celestial coordinates
                accoring to Calabretta & Greisen (2002).
`world2pix`     Convert celestial coordinates into pixel coordinates
                accoring to Calabretta & Greisen (2002).
`gcdist`        Calculate the great circle distance between two points.
`calc_pixscale` Calculate the pixel scale from the WCS information in a
                FITS header.
`fit_cdmatrix`  Attempt to fit a CD matrix for a set of points with known
                pixel and world coordinates.
=============== =======================================================

"""
from astropy import wcs
import numpy as np

from . import util


def parse_ctype(ctype_str):
    """Get the coordinate system and projection for a given value of the
    CTYPEi keyword.

    """
    elements = ctype_str.split('-')
    coordsys, projection = elements[0], elements[-1]
    return coordsys, projection


def pix2proj(x, y, hdr):
    """Convert pixel coordinates into projection plane coordinates
    according to Eq. 3 in Greisen & Calabretta (2002).

    Parameters
    ----------
    x, y : float or array
        x and y pixel coordinates.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CRPIX1, CRPIX2
        - CD1_1, CD1_2, CD2_1, CD2_2

        Optional keywords:

        - CUNIT1, CUNIT2: Used for converting the CD matrix into deg/pix.
          If omitted, the CD matrix is assumed to already have units
          deg/pix so that no conversion is necessary.

    Returns
    -------
    float or array
        x and y projection plane coordinates in degrees.

    """
    cunit1, cunit2 = hdr.get('CUNIT1'), hdr.get('CUNIT2')
    if cunit1 is not None:
        pass  # Always assume degrees for now
    if cunit2 is not None:
        pass  # Always assume degrees for now

    # (G&C02 eq.3/pg.1063/pdf.3)
    xp = hdr['CD1_1']*(x-hdr['CRPIX1']) + hdr['CD1_2']*(y-hdr['CRPIX2'])
    yp = hdr['CD2_1']*(x-hdr['CRPIX1']) + hdr['CD2_2']*(y-hdr['CRPIX2'])

    return xp, yp


def proj2pix(xp, yp, hdr):
    """Convert projection plane coordinates into pixel coordinates
    according to Eq. 3 in Greisen & Calabretta (2002).

    Parameters
    ----------
    xp, yp : float or array
        x and y projection plane coordinates in degrees.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CRPIX1, CRPIX2
        - CD1_1, CD1_2, CD2_1, CD2_2

        Optional keywords:

        - CUNIT1, CUNIT2: Used for converting the CD matrix into deg/pix.
          If omitted, the CD matrix is assumed to already have units
          deg/pix so that no conversion is necessary.

    Returns
    -------
    float or array
        x and y pixel coordinates.

    """
    cunit1, cunit2 = hdr.get('CUNIT1'), hdr.get('CUNIT2')
    if cunit1 is not None:
        pass  # Always assume degrees for now
    if cunit2 is not None:
        pass  # Always assume degrees for now

    # (G&C02 eq.3/pg.1063/pdf.3)
    c = hdr['CD1_1']*hdr['CD2_2'] - hdr['CD1_2']*hdr['CD2_1']
    x = (hdr['CD2_2']*xp - hdr['CD1_2']*yp) / c + hdr['CRPIX1']
    y = (-hdr['CD2_1']*xp + hdr['CD1_1']*yp) / c + hdr['CRPIX2']

    return x, y


def proj2natsph(xp, yp, hdr):
    """Convert projection plane coordinates into native spherical
    coordinates for a given projection according to Calabretta & Greisen
    (2002).

    Parameters
    ----------
    xp, yp : float or array
        x and y projection plane coordinates in degrees.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2

    Returns
    -------
    float or array
        Native spherical longitude (phi) and latitude (theta) in degrees
        for the projection specified by CTYPE1 and CTYPE2.

    """
    # Degrees to radians
    xp = xp * np.pi/180
    yp = yp * np.pi/180

    projection1 = parse_ctype(hdr['CTYPE1'])[1]
    projection2 = parse_ctype(hdr['CTYPE2'])[1]
    projection = projection1  # Always assume projection1 and projection2 are the same?

    if projection == 'TAN':
        phi = np.arctan2(xp, -yp)  # (C&G02 eq.14/pg.1085/pdf.9)
        r = np.sqrt(xp**2 + yp**2)  # (C&G02 eq.15/pg.1085/pdf.9)
        theta = np.arctan(1/r)  # (C&G02 eq.55/pg.1088/pdf.12)
    else:
        # Raise an error?
        phi, theta = None, None

    # Radians to degrees
    phi *= 180/np.pi
    theta *= 180/np.pi

    return phi, theta


def natsph2proj(phi, theta, hdr):
    """Convert native spherical coordinates into projection plane
    coordinates for a given projection according to Calabretta & Greisen
    (2002).

    Parameters
    ----------
    phi, theta : float or array
        Native spherical longitude (phi) and latitude (theta) in degrees
        for the projection specified by CTYPE1 and CTYPE2.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2

    Returns
    -------
    float or array
        x and y projection plane coordinates in degrees.

    """
    # Degrees to radians
    phi = phi * np.pi/180
    theta = theta * np.pi/180

    projection1 = parse_ctype(hdr['CTYPE1'])[1]
    projection2 = parse_ctype(hdr['CTYPE2'])[1]
    projection = projection1  # Always assume projection1 and projection2 are the same?

    if projection == 'TAN':
        r = 1/np.tan(theta)  # (C&G02 eq.54/pg.1088/pdf.12)
        xp = r*np.sin(phi)  # (C&G02 eq.12/pg.1085/pdf.9)
        yp = -r*np.cos(phi)  # (C&G02 eq.13/pg.1085/pdf.9)
    else:
        # Raise and error?
        xp, yp = None, None

    # Radians to degrees
    xp *= 180/np.pi
    yp *= 180/np.pi

    return xp, yp


def natsph2celsph(phi, theta, hdr):
    """Convert native spherical coordinates for a given projection into the
    given celestial spherical coordinates according to Calabretta & Greisen
    (2002).

    Parameters
    ----------
    phi, theta : float or array
        Native spherical longitude (phi) and latitude (theta) in degrees
        for the projection specified by CTYPE1 and CTYPE2.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2
        - CRVAL1, CRVAL2

        Optional keywords:

        - CUNIT1, CUNIT2: Used for converting CRVAL1 and CRVAL2 into
          degrees and converting celestial longitude and latitude into the
          proper units. If omitted, the everything is assumed to be in
          degrees.

    Returns
    -------
    float or array
        Celestial spherical longitude (lon) and latitude (lat) for the
        celestial coordinate system specified by CTYPE1 and CTYPE2.

    """
    cunit1, cunit2 = hdr.get('CUNIT1'), hdr.get('CUNIT2')
    if cunit1 is not None:
        pass  # Always assume CRVAL1 and longitude are in degrees for now
    if cunit2 is not None:
        pass  # Always assume CRVAL2 and latitude are in degrees for now

    # Degrees to radians
    phi = phi * np.pi/180
    theta = theta * np.pi/180

    coordsys1, projection1 = parse_ctype(hdr['CTYPE1'])
    coordsys2, projection2 = parse_ctype(hdr['CTYPE2'])
    projection = projection1  # Always assume projection1 and projection2 are the same?

    if projection == 'TAN':
        lon_p = hdr['CRVAL1'] * np.pi/180
        lat_p = hdr['CRVAL2'] * np.pi/180
        phi_p = 180.0 * np.pi/180
    else:
        # Raise an error?
        lon_p, lat_p, phi_p = None, None, None

    # (C&G02 eq.2/pg.1079/pdf.3)
    dphi = phi - phi_p
    lon = lon_p + np.arctan2(-np.cos(theta) * np.sin(dphi),
                             np.sin(theta) * np.cos(lat_p) -
                             np.cos(theta) * np.sin(lat_p) * np.cos(dphi))
    lat = np.arcsin(np.sin(theta) * np.sin(lat_p) +
                    np.cos(theta) * np.cos(lat_p) * np.cos(dphi))

    # Radians to degrees
    lon *= 180/np.pi
    lat *= 180/np.pi

    return (lon, lat)


def celsph2natsph(lon, lat, hdr):
    """Convert the given celestial spherical coordinates into native
    spherical coordinates for a given projection according to Calabretta &
    Greisen (2002).

    Parameters
    ----------
    lon, lat : float or array
        Celestial spherical longitude (lon) and latitude (lat) for the
        celestial coordinate system specified by CTYPE1 and CTYPE2.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2
        - CRVAL1, CRVAL2

        Optional keywords:

        - CUNIT1, CUNIT2: Used for converting CRVAL1, CRVAL2, and celestial
          longitude and latitude into degrees. If omitted, the everything
          is assumed to be in degrees.

    Returns
    -------
    float or array
        Native spherical longitude (phi) and latitude (theta) in degrees
        for the projection specified by CTYPE1 and CTYPE2.

    """
    cunit1, cunit2 = hdr.get('CUNIT1'), hdr.get('CUNIT2')
    if cunit1 is not None:
        pass  # Always assume CRVAL1 and longitude are in degrees for now
    if cunit2 is not None:
        pass  # Always assume CRVAL2 and latitude are in degrees for now

    # Degrees to radians
    lon = lon * np.pi/180
    lat = lat * np.pi/180

    coordsys1, projection1 = parse_ctype(hdr['CTYPE1'])
    coordsys2, projection2 = parse_ctype(hdr['CTYPE2'])
    projection = projection1  # Always assume projection1 and projection2 are the same?

    if projection == 'TAN':
        lon_p = hdr['CRVAL1'] * np.pi/180
        lat_p = hdr['CRVAL2'] * np.pi/180
        phi_p = 180.0 * np.pi/180
    else:
        # Raise an error?
        lon_p, lat_p, phi_p = None, None, None

    # (C&G02 eq.5/pg.1080/pdf.4)
    dlon = lon - lon_p
    phi = phi_p + np.arctan2(-np.cos(lat) * np.sin(dlon),
                             np.sin(lat) * np.cos(lat_p) -
                             np.cos(lat) * np.sin(lat_p) * np.cos(dlon))
    theta = np.arcsin(np.sin(lat) * np.sin(lat_p) +
                      np.cos(lat) * np.cos(lat_p) * np.cos(dlon))

    # Radians to degrees
    phi *= 180/np.pi
    theta *= 180/np.pi

    return phi, theta


def pix2world(x, y, hdr):
    """Convert pixel coordinates into celestial coordinates accoring to
    Calabretta & Greisen (2002).

    .. note:: This transformation is already supported by the `astropy.wcs`
       module.

    Parameters
    ----------
    x, y : float or array
        x and y pixel coordinates.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2
        - CRPIX1, CRPIX2
        - CRVAL1, CRVAL2
        - CD1_1, CD1_2, CD2_1, CD2_2

        Optional keywords:

        - CUNIT1, CUNIT2: Used for converting the CD matrix into deg/pix,
          CRVAL1 and CRVAL2 into degrees, and celestial longitude and
          latitude into the proper units. If omitted, everything is assumed
          to be in degrees (deg/pix for the CD matrix).

    Returns
    -------
    float or array
        Celestial spherical longitude (lon) and latitude (lat) for the
        celestial coordinate system specified by CTYPE1 and CTYPE2.

    """
    xp, yp = pix2proj(x, y, hdr)
    phi, theta = proj2natsph(xp, yp, hdr)
    lon, lat = natsph2celsph(phi, theta, hdr)
    return lon, lat


def world2pix(lon, lat, hdr):
    """Convert celestial coordinates into pixel coordinates accoring to
    Calabretta & Greisen (2002).

    .. note:: This transformation is already supported by the `astropy.wcs`
       module.

    Parameters
    ----------
    lon, lat : float or array
        Celestial spherical longitude (lon) and latitude (lat) for the
        celestial coordinate system specified by CTYPE1 and CTYPE2.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2
        - CRPIX1, CRPIX2
        - CRVAL1, CRVAL2
        - CD1_1, CD1_2, CD2_1, CD2_2

        Optional keywords:

        - CUNIT1, CUNIT2: Used for converting the CD matrix into deg/pix
          and CRVAL1, CRVAL2, and celestial longitude and latitude into
          degrees. If omitted, everything is assumed to be in degrees
          (deg/pix for the CD matrix).

    Returns
    -------
    float or array
        x and y pixel coordinates.

    """
    phi, theta = celsph2natsph(lon, lat, hdr)
    xp, yp = natsph2proj(phi, theta, hdr)
    x, y = proj2pix(xp, yp, hdr)
    return x, y


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
    ref : {'crpix', 'center', tuple}, optional
        If 'crpix' (default), the reference pixel is set to CRPIX1,CRPIX2.
        'center' indicates that the central pixel in the image should be
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


def fit_cdmatrix(x, y, lon, lat, hdr):
    """Attempt to fit a CD matrix for a set of points with known pixel and
    world coordinates.

    The world coordinates are transformed to projection plane coordinates
    for a given projection, and then a 2d least squares fitting method is
    used to determine the best-fit CD matrix that transforms the pixel
    coordinates into projection plane coordinates.

    Parameters
    ----------
    x, y : array
        x and y pixel coordinates.
    lon, lat : array
        Celestial longitude and latitude (world) coordinates.
    hdr : astropy.io.fits.Header or dictionary
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2
        - CRVAL1, CRVAL2

        Optional keywords:

        - CUNIT1, CUNIT2: Used for converting CRVAL1, CRVAL2, and celestial
          longitude and latitude into degrees. If omitted, the everything
          is assumed to be in degrees.

    Returns
    -------
    array
        The best-fit CD matrix, ``[[CD1_1, CD1_2], [CD2_1, CD2_2]]``.

    """
    # Calculate projection plane coords
    phi, theta = celsph2natsph(lon, lat, hdr)
    xp, yp = natsph2proj(phi, theta, hdr)

    # Solve for CD elements
    dx, dy = x - hdr['CRPIX1'], y - hdr['CRPIX2']
    cd11, cd12 = util.leastsquares2d(dx, dy, xp)
    cd21, cd22 = util.leastsquares2d(dx, dy, yp)

    return np.array([[cd11, cd12], [cd21, cd22]])


def make_header(xy, ad, ref='center'):
    """
    ref : {'center', tuple}, optional
        If 'center' (default), the reference pixel is set to the central
        pixel in the image. An x,y tuple of floats may be given to use a
        specific reference pixel instead.


    """
    # Make a new header; assume a TAN projection with RA,dec coords in deg
    hdr = fits.Header()
    hdr['wcsaxes'] = 2
    hdr['ctype1'] = 'RA---TAN'
    hdr['ctype2'] = 'DEC--TAN'

    # Reference point
    if ref == 'center':
        # j=(x.max()-x.min())/2+x.min(), etc.
        i, j = (config.NROW+1)/2, (config.NCOL+1)/2  # middle point in the grid
        hdr['CRPIX1'], hdr['CRPIX2'] = xy[:,i,j]
        hdr['CRVAL1'], hdr['CRVAL2'] = ad[:,i,j]
    else:
        pass  # 

    # CD matrix
    xy, ad = xy.T.reshape((-1, 2)), ad.T.reshape((-1, 2))
    cd = fit_cdmatrix(xy, ad, hdr)
    hdr['CD1_1'], hdr['CD1_2'] = cd[0, 0], cd[0, 1]
    hdr['CD2_1'], hdr['CD2_2'] = cd[1, 0], cd[1, 1]

    return hdr
