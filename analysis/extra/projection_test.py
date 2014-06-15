from astropy.io import fits
from astropy import wcs
import montage_wrapper as montage
import numpy as np
import os


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


def calc_area1(arr, hdrwcs):
    """Calculate pixel areas from vertices assuming spherical rectangles."""
    ny, nx = arr.shape
    x, y = np.meshgrid(np.linspace(0.5, nx+0.5, nx+1),
                       np.linspace(0.5, ny+0.5, ny+1))

    # List of vertices for each pixel
    n = nx*ny
    corner0 = x[:-1,:-1].reshape(n, -1)
    corner1 = x[:-1,1:].reshape(n, -1)
    corner2 = x[1:,1:].reshape(n, -1)
    corner3 = x[1:,:-1].reshape(n, -1)
    x = np.hstack([corner0, corner1, corner2, corner3])

    corner0 = y[:-1,:-1].reshape(n, -1)
    corner1 = y[:-1,1:].reshape(n, -1)
    corner2 = y[1:,1:].reshape(n, -1)
    corner3 = y[1:,:-1].reshape(n, -1)
    y = np.hstack([corner0, corner1, corner2, corner3])

    a, d = hdrwcs.wcs_pix2world(x, y, 1)

    area = np.array([sparea(a, d) for a, d in zip(a, d)])
    area = area.reshape(arr.shape) * 3600**2  # arcsec2
    return area


def calc_area2(arr, hdrwcs):
    """Calculate pixel areas from pixel scales assuming planar rectangles."""
    ny, nx = arr.shape
    x, y = np.meshgrid(np.linspace(0.5, nx+0.5, nx+1),
                       np.linspace(0.5, ny+0.5, ny+1))

    a0, d0 = hdrwcs.wcs_pix2world(x[:-1,:-1], y[:-1,:-1], 1)
    a1, d1 = hdrwcs.wcs_pix2world(x[:-1,1:], y[:-1,1:], 1)
    a2, d2 = hdrwcs.wcs_pix2world(x[1:,:-1], y[1:,:-1], 1)
    dx = gcdist(a0, d0, a1, d1) * 3600
    dy = gcdist(a0, d0, a2, d2) * 3600
    area = dx*dy  # arcsec2
    return area


def calc_area3(hdr, ref='crpix'):
    """Calculate pixel area from pixel scale assuming a planar rectangle."""
    dx, dy = calc_pixscale(hdr, ref=ref)
    area = dx * dy  # arcsec2
    return area


def print_summary(f):
    output = ('      median = {0:.2e}\n'
              'mean +/- std = {1:.2e} +/- {2:.2e}\n'
              '    min, max = {3:.2e}, {4:.2e}\n'
              .format(np.median(f), np.mean(f), np.std(f), np.min(f), np.max(f))
              )
    print output
    return None


def sum_test():
    work_dir = '/Users/Jake/Research/PHAT/sfhmaps/__old__analysis/map'
    brick_list = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                  14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

    data_nat, data_rep = [], []  # brick native, brick reprojected
    for brick in brick_list:
        # Native, unprojected image
        filename = 'b{:02d}_mod_fuv_attenuated.fits'.format(brick)
        filename = os.path.join(work_dir, 'b{:02d}'.format(brick), filename)
        data = fits.getdata(filename)

        hdr = fits.getheader(filename)
        hwcs = wcs.WCS(hdr)
        #area = calc_area1(data, hwcs)
        area = calc_area3(hdr, ref='central')
        data_nat.append(data * area)

        # Reprojected image
        filename = 'hdu0_b{:02d}_mod_fuv_attenuated.fits'.format(brick)
        filename = os.path.join(work_dir, '_mod_fuv_attenuated', 'reproject', filename)
        data = fits.getdata(filename)

        filename = 'hdu0_b{:02d}_mod_fuv_attenuated_area.fits'.format(brick)
        filename = os.path.join(work_dir, '_mod_fuv_attenuated', 'reproject', filename)
        area_rep = fits.getdata(filename) * (180/np.pi*3600)**2  # arcsec2
        data_rep.append(data * area_rep)

    sum_nat = np.array([np.nansum(arr) for arr in data_nat])
    sum_rep = np.array([np.nansum(arr) for arr in data_rep])

    fractionaldiff = (sum_rep - sum_nat) / sum_nat

    print_summary(fractionaldiff)


if __name__ == '__main__':
    sum_test()


"""
Results
=======

Fractional differences for 21 bricks; each difference is between the pixel
sums of the native and the reprojected images of a brick. Reprojection
preserves flux to within ~0.01% at worst, ~0.001% on average:

        median = 2.47e-05
  mean +/- std = 3.27e-05 +/- 9.69e-05
      min, max = -3.47e-04, 1.71e-04


Notes
=====
- mProjExec and mProject produce nearly identical results (within ~0.01%).

- Settings that affect reprojection: mProject has the 'z' flag, which
  controls the drizzle factor. 1 seems to be the default, and appears to
  give the best results anyway so there is no reason to change it.
  mProjExec doesn't really have any settings that change the reprojection
  results.

- Pixel area varies slightly accross brick 15 in both the native and the
  reprojected images. The effect causes only ~0.001% difference between the
  minimum and maximum areas, however, so the pixels can be safely treated
  as having constant area.

- All three area calculation methods (`calc_area1`, `calc_area2`,
  `calc_area3`) agree to within ~0.01% to ~0.001%. Might as well just use
  the simplest method, which is the single area value calculated from the
  pixel scale in the FITS header.

"""
