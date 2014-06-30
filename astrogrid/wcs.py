"""

===============
`astrogrid.wcs`
===============

Utilities for working with world coordinate systems.


Functions
---------

=============== ===========================================================
`calc_pixscale` Calculate the x and y pixel scales from the WCS information
                in a FITS header.
`fit_cdmatrix`  Fit a CD matrix for a set of points with known pixel and
                world coordinates.
`make_header`   Create a FITS header from a set of points with known pixel
                and world coordinates given a celestial coordinate system
                and projection.
=============== ===========================================================

"""
import astropy.coordinates, astropy.io.fits, astropy.units, astropy.wcs
import numpy as np
import scipy.optimize


def calc_pixscale(hdr, ref='crpix', units=(astropy.units.deg, astropy.units.deg)):
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
        `astropy.units.core.Unit` instances. Both units are set to
        `astropy.units.deg` by default.

    Returns
    -------
    astropy.coordinates.Angle
        An `Angle` instance containing the x and y pixel scales.

    """
    wcs = astropy.wcs.WCS(hdr)

    if ref == 'crpix':
        x, y = hdr['crpix1'], hdr['crpix2']
    elif ref == 'center':
        x, y = hdr['naxis1']/2, hdr['naxis2']/2
    else:
        x, y = ref

    lon, lat = wcs.wcs_pix2world([x, x+1, x], [y, y, y+1], 1)
    # Makes no difference whether ICRS, Galactic, AltAz, etc.
    points = astropy.coordinates.ICRS(lon, lat, unit=units)
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
    x, y : array
        x and y pixel coordinates.
    lon, lat : array
        Celestial longitude and latitude (world) coordinates.
    hdr : astropy.io.fits.Header
        A FITS header. Required keywords:

        - CTYPE1, CTYPE2
        - CRPIX1, CRPIX2
        - CRVAL1, CRVAL2

    Returns
    -------
    array
        The best-fit CD matrix, ``[[CD1_1, CD1_2], [CD2_1, CD2_2]]``.

    """
    def residuals(p, dx, dy, ip):
        # i represents x or y; xp - (cd11*dx + cd12*dy) and yp - (cd21*dx + cd22*dy)
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


def make_header(x, y, lon, lat, ctype1='RA---TAN', ctype2='DEC--TAN', ref=None):
    """Create a FITS header from a set of points with known pixel and world
    coordinates given a celestial coordinate system and projection.

    `fit_cdmatrix` is used to find the best-fit CD matrix for the given
    data.

    Parameters
    ----------
    x, y : array
        x and y pixel coordinates.
    lon, lat : array
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



# Tests
# -----
def _make_header_imwcs(filename, x, y, lon, lat,
                       ctype1='RA---TAN', ctype2='DEC--TAN', ref=None):
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
    for xp, yp, RA, DEC in zip(x.ravel(), x.ravel(), lon.ravel(), lat.ravel()):
        h = int(RA)
        m = int((RA - h) * 60)
        s = (((RA - h) * 60) - m) * 60
        hms = '{0:02d} {1:d} {2:8.5f}'.format(h, m, s)

        d = int(DEC)
        m = int((DEC - d) * 60)
        s = (((DEC - d) * 60) - m) * 60
        dms = '{0:02d} {1:d} {2:8.5f}'.format(d, m, s)

        cstr = '{0:15.8f} {1:15.8f}    {2:s}    {3:s}\n'.format(xp, yp, hms, dms)
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
