from astropy.io import fits
from astropy import wcs
import montage_wrapper as montage
import os


def corners(x):
    n = (x.shape[0]-1) * (x.shape[1]-1)
    corner0 = x[:-1,:-1].reshape(n, -1)
    corner1 = x[:-1,1:].reshape(n, -1)
    corner2 = x[1:,1:].reshape(n, -1)
    corner3 = x[1:,:-1].reshape(n, -1)
    return np.hstack([corner0, corner1, corner2, corner3])


def grid(x, shape):
    """Inverse of `corners`."""
    y = np.zeros(shape, dtype=x.dtype)
    ny, nx = shape
    y[:-1,:-1] = x[:,0].reshape(ny-1, nx-1)
    y[:-1,1:] = x[:,1].reshape(ny-1, nx-1)
    y[1:,1:] = x[:,2].reshape(ny-1, nx-1)
    y[1:,:-1] = x[:,3].reshape(ny-1, nx-1)
    return y


def spherical_polygon_area(lon, lat, R=1, units='deg'):
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
    l = 2 * np.arcsin(np.sqrt(np.sin((lat[1:] - lat[:-1]) / 2)**2 +
                              np.cos(lat[:-1]) * np.cos(lat[1:]) *
                              np.sin((lon[1:] - lon[:-1]) / 2)**2))

    # Semiperimeter of each spherical triangle
    s = 0.5 * (l + np.pi + lat[:-1] + lat[1:])

    # Spherical excess of each spherical triangle from L'Huilier's theorem.
    # Note that none of the terms should be negative (not 100% sure about
    # that); assume that any negative values are within machine precision
    # of 0.
    term1 = (s - (np.pi / 2 + lat[:-1])) /2
    term2 = (s - (np.pi / 2 + lat[1:])) /2
    E = 4 * np.arctan(np.sqrt(np.tan(s / 2) * np.tan((s - l) / 2) *
                              np.tan(np.where(term1 < 0, 0, term1)) *
                              np.tan(np.where(term2 < 0, 0, term2))))

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


work_dir = '/Users/Jake/Research/PHAT/sfhmaps/__old__analysis/map/b15/test'
input_dir = os.path.join(work_dir, 'input')
input_img = os.path.join(input_dir, 'b15_mod_fuv_attenuated.fits')
proj_dir = os.path.join(work_dir, 'reproject')
hdrfile = os.path.join(work_dir, 'template.hdr')
metafile = os.path.join(work_dir, 'input.tbl')
statfile = os.path.join(work_dir, 'stat.tbl')


# Reproject
# ---------

# mProjExec
proj_img1 = os.path.join(proj_dir, 'test1.fits')
montage.mImgtbl(input_dir, metafile, corners=True)
montage.mProjExec(metafile, hdrfile, proj_dir, statfile, raw_dir=input_dir)
os.rename(os.path.join(proj_dir, 'hdu0_b15_mod_fuv_attenuated.fits'), proj_img1)

# mProject
proj_img2 = os.path.join(proj_dir, 'test2.fits')
montage.mProject(input_img, proj_img2, hdrfile, status_file=statfile)


# Analyze
# -------
data0, hdr0 = fits.getdata(input_img), fits.getheader(input_img)
data1, hdr1 = fits.getdata(proj_img1), fits.getheader(proj_img1)
data2 = fits.getdata(proj_img2)

s0, s1, s2 = np.nansum(data0), np.nansum(data1), np.nansum(data2)

ny, nx = data0.shape
X0, Y0 = np.meshgrid(np.linspace(0.5, nx+0.5, nx+1), np.linspace(0.5, ny+0.5, ny+1))
ny, nx = data1.shape
X1, Y1 = np.meshgrid(np.linspace(0.5, nx+0.5, nx+1), np.linspace(0.5, ny+0.5, ny+1))
ny, nx = data2.shape
X2, Y2 = np.meshgrid(np.linspace(0.5, nx+0.5, nx+1), np.linspace(0.5, ny+0.5, ny+1))

X0, Y0 = corners(X0), corners(Y0)
X1, Y1 = corners(X1), corners(Y1)
X2, Y2 = corners(X2), corners(Y2)

wcs0, wcs1 = wcs.WCS(hdr0), wcs.WCS(hdr1)
A0, D0 = wcs0.wcs_pix2world(X0, Y0, 1)
A1, D1 = wcs1.wcs_pix2world(X1, Y1, 1)
A2, D2 = wcs1.wcs_pix2world(X2, Y2, 1)

### There's a bug in the area calculation! ###
area0 = np.array([spherical_polygon_area(lon, lat) for lon, lat in zip(A0, D0)]).reshape(data0.shape)
area1 = np.array([spherical_polygon_area(lon, lat) for lon, lat in zip(A1, D1)]).reshape(data1.shape)
area2 = np.array([spherical_polygon_area(lon, lat) for lon, lat in zip(A2, D2)]).reshape(data2.shape)
a0, a1, a2 = np.mean(area0), np.mean(area1), np.mean(area2)

# Quick ds9 pixel measurements
dx0, dy0 = 23.7515, 26.6784
dx1 = dy1 = 23.7467

a0 = dx0*dy0
a1 = dx1*dy1


(s1 - s0) / s0 * 100  # 17.5%
(s1 * a1/a0 - s0) / s0 * 100  # 4.6%








MAPDIR = '/Users/Jake/Research/PHAT/sfhmaps/data/map'

brick_list = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
              20, 21, 22, 23]

brknat, brkrep = [], []  # brick native, brick reprojected
for brick in brick_list:
    filename = 'b{:02d}_mod_fuv_attenuated.fits'.format(brick)
    filename = os.path.join(MAPDIR, 'b{:02d}'.format(brick), filename)
    data = fits.getdata(filename)
    brknat.append(data)

    filename = 'hdu0_b{:02d}_mod_fuv_attenuated.fits'.format(brick)
    filename = os.path.join(MAPDIR, '_mod_fuv_attenuated', 'reproject', filename)
    data = fits.getdata(filename)
    brkrep.append(data)

bnsum = np.array([np.nansum(arr) for arr in brknat])
brsum = np.array([np.nansum(arr) for arr in brkrep])

fdiff = (brsum - bnsum) / bnsum
np.all(fdiff > 0)  # True; weird!!!
np.median(fdiff)  # 0.2560
np.mean(fdiff)  # 0.2688
np.std(fdiff)  # 0.0683
np.min(fdiff), np.max(fdiff)  # 0.1751, 0.4013

# Brick 15 (index 12)
fdiff[12]  # 0.1751

