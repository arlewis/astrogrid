from astropy import wcs
import numpy as np
import os

from . import config



# Parsers
# -------
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


def sparea(lon, lat, R=1, units='deg'):
    """Calculate the area of a spherical polygon.

    It is assumed that the polygon contains no poles! The method is adapted
    from "Some algorithms for polygons on a sphere" by Chamberlain and
    Duquette (http://trs-new.jpl.nasa.gov/dspace/handle/2014/40409)

    Parameters
    ----------
    lon, lat : array-like
        Longitude and latitude of each vertex in the polygon. The first
        vertex may be listed either only at the beginning of the sequence,
        or both at the beginning and at the end of the sequence.
    R : float, optional
        Radius of the sphere (default is 1).
    units : {'deg', 'rad'}, optional
        Sets the unit system for `lon` and `lat`.

    Returns
    -------
    float
        Total area of the spherical polygon.

    """
    lon, lat = np.array(lon), np.array(lat)

    # Ensure the polygon is closed
    if not lon[-1]==lon[0] or not lat[-1]==lat[0]:
        lon, lat = np.append(lon, lon[0]), np.append(lat, lat[0])

    if units == 'deg':
        lon, lat = lon * np.pi / 180, lat * np.pi / 180

    # Great circle segments between vertices
    l = gcdist(lon[:-1], lat[:-1], lon[1:], lat[1:], deg=False)

    # Semiperimeter of each spherical triangle
    s = 0.5 * (l + np.pi + lat[:-1] + lat[1:])

    # Spherical excess of each spherical triangle from L'Huilier's theorem.
    # Note that none of the terms should be negative (not 100% sure about
    # that); assume that any negative values are within machine precision
    # of 0.
    term1 = (s - (np.pi / 2 + lat[:-1])) /2
    term1[term1<0] = 0
    term2 = (s - (np.pi / 2 + lat[1:])) /2
    term2[term2<0] = 0
    result = np.tan(s/2) * np.tan((s-l)/2) * np.tan(term1) * np.tan(term2)
    E = 4 * np.arctan(np.sqrt(result))

    # Let A<0 for lon[i]<lon[i+1], A>0 for lon[i+1]<lon[i] assuming ccw
    # traversal
    sign = 2*(lon[1:] < lon[:-1]) - 1

    # Total area
    A = np.sum(sign * E) * R**2
    if units == 'deg':
        A = A * (180 / np.pi)**2
    if A < 0:  # Fix the sign in case the vertices are not listed ccw
        A = -A
    return A


def calc_pixscale(hdr, ref='crpix'):
    """Calculate the pixel scale from the WCS information in a header.

    World coordinates are assumed to be in deg (e.g., CRVAL1 and CRVAL2).
    The pixel scales are returned in arcsec/pixel.

    ref : {'crpix', 'central', tuple}, optional
        The reference pixel is set to CRPIX1,CRPIX2 if 'crpix' (default).
        'central' indicates that the central pixel in the image should be
        used. An x,y tuple of floats may be given to instead use a specific
        reference pixel.

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



# Analysis stuff
# --------------

