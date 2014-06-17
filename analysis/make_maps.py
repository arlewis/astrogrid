from astropy.io import fits
import montage_wrapper as montage
import numpy as np
import os
import subprocess

from sfhmaps import config, util



# Basic functions
# ---------------
def make_coord_grid(vertices):
    """
    Parameters
    ----------
    vertices : array
        Output from parse_coordfile.

    Returns
    -------
    xy : array
        Vertices in x,y pixel coordinates; shape (NROW+1, NCOL+1).
    ad : array
        Vertices in RA,dec coordinates; shape (NROW+1, NCOL+1).

    """
    # RA,dec grid
    shape = (2, config.NROW+1, config.NCOL+1)
    ad = np.zeros(shape)
    ad[:,:-1,:-1] = vertices[...,0]  # all 1st vertices
    ad[:,:-1,-1] = vertices[...,-1,1]  # last column (unique 2nd vertices)
    ad[:,-1,:-1] = vertices[:,-1,:,3]  # last row (unique 4th vertices)
    ad[:,-1,-1] = vertices[:,-1,-1,2]  # last corner (unique 3rd vertex)
    ad = ad[:,::-1]  # reverse row order

    # x,y grid (pixel coords, i.e., origin is 0.5,0.5)
    xy = np.mgrid[0:config.NROW+1,0:config.NCOL+1][::-1] + 0.5

    return xy, ad


def leastsquares(x, y, z):
    """Get the best-fit a,b for zi = a*xi + b*yi).

    Does not work where x == y! If all x == y, then NaN is returned.

    Derivation
    ----------
    Let, ::

      f(x,y) = a*x + b*y

    Given a set of measurements, ``zi`` for each pair of ``xi`` and ``yi``,
    find ``a`` and ``b`` such that ``S = sum(ri**2)`` is minimized (the
    gradient of ``S`` is zero), where ``ri = zi - f(xi,yi)``.

    So, ::

      dS/da = dS/db = 0

    The derivatives are, ::

      dS/da = 2 * sum(ri * dri/da) = 0
      dS/db = 2 * sum(ri * dri/db) = 0

      dri/da = -xi
      dri/db = -yi

      ri * dri/da = a*xi**2 + b*xi*yi - xi*zi
      ri * dri/db = a*xi*yi + b*yi**2 - yi*zi

    The sums can be written as, ::

      sum(a*xi**2 + b*xi*yi - xi*zi) = 0
      sum(a*xi*yi + b*yi**2 - yi*zi) = 0

      a*sum(xi**2) + b*sum(xi*yi) - sum(xi*zi) = 0
      a*sum(xi*yi) + b*sum(yi**2) - sum(yi*zi) = 0

      a*sx + b*sxy = sxz
      a*sxy + b*sy = syz

    Therefore, ::

      a = (sxz - sxy*syz/sy) / (sx - sxy**2/sy)
      b = (sxz - a*sx) / sxy

    """
    i = x != y
    if np.sum(i) == 0:
        return np.nan

    sx, sxy, sxz = np.sum(x[i]**2), np.sum(x[i]*y[i]), np.sum(x[i]*z[i])
    sy,      syz = np.sum(y[i]**2),                    np.sum(y[i]*z[i])

    a = (sxz - sxy*syz/sy) / (sx - sxy**2/sy)
    b = (sxz - a*sx) / sxy

    return a, b


def fit_cdmatrix(xy, ad, hdr):
    # Calculate projection plane coords
    p, t = util.celsph2natsph(ad[:,0], ad[:,1], hdr)  # CRVALi
    u, v = util.natsph2proj(p, t, hdr)  # TAN projection
    uv = np.vstack((u, v)).T

    # Delta x, Delta y
    deltaxy = xy - np.array([hdr['CRPIX1'], hdr['CRPIX2']])

    # Solve for CD elements
    cd11, cd12 = leastsquares(deltaxy[:,0], deltaxy[:,1], uv[:,0])
    cd21, cd22 = leastsquares(deltaxy[:,0], deltaxy[:,1], uv[:,1])
    return np.array([[cd11, cd12], [cd21, cd22]])


def make_header(brick):
    # Make a new header; assume a TAN projection with RA,dec coords in deg
    hdr = fits.Header()
    hdr['wcsaxes'] = 2
    hdr['ctype1'] = 'RA---TAN'
    hdr['ctype2'] = 'DEC--TAN'

    # Load coordinate data for CD matrix fitting
    coordfile = config.get_file(brick=brick, kind='vert')
    vertices = util.parse_coordfile(coordfile)
    xy, ad = make_coord_grid(vertices)

    # Choose a reference point
    i, j = (config.NROW+1)/2, (config.NCOL+1)/2  # middle point in the grid
    hdr['CRPIX1'], hdr['CRPIX2'] = xy[:,i,j]
    hdr['CRVAL1'], hdr['CRVAL2'] = ad[:,i,j]

    # CD matrix
    xy, ad = xy.T.reshape((-1, 2)), ad.T.reshape((-1, 2))
    cd = fit_cdmatrix(xy, ad, hdr)
    hdr['CD1_1'], hdr['CD1_2'] = cd[0, 0], cd[0, 1]
    hdr['CD2_1'], hdr['CD2_2'] = cd[1, 0], cd[1, 1]

    return hdr


def make_image(brick, func):
    """Apply function to all pixels in given brick and assemble image."""
    data = []
    for pixel in config.PIXEL_LIST:
        if (brick, pixel) in config.MISSING:
            val = np.nan
        else:
            val = func(brick, pixel)
        data.append(val)
    data = np.array(data).reshape(config.NROW, config.NCOL)

    # Reverse rows
    data = data[::-1]

    # Make fits hdu
    hdr = make_header(brick)
    hdu = fits.PrimaryHDU(data, header=hdr)

    return hdu



# Custom functions
# ----------------
def calc_mean_sfr(brick, pixel):
    # Load data
    zcbfile = config.get_file(brick=brick, pixel=pixel, kind='bestzcb')
    table = util.parse_zcbtable(zcbfile)

    # Get SFH over last 100 Myr
    i = table['log(age_f)'] <= 8.00
    sfr = table['SFR'][i]
    agei, agef = 10**table['log(age_i)'][i], 10**table['log(age_f)'][i]

    # Calculate mean SFR
    dt = agef - agei
    mean = np.average(sfr, weights=dt)  # Equivalent to np.sum(mass)/1e8

    return mean


def make_spec(M_H):
    """Generate spectral evolution (spectrum observed at t_lookback for
    each sfh time bin), write to .fits file. File is only written if it
    does not exist in the current working directory.

    Assumes Kroupa02 IMF and present-day lookback time.

    Parameters
    ----------
    M_H : float
        Metallicity [M/H].

    """
    zmet = 10**M_H  # linear solar units, i.e., units of (M/H)_sun
    imf_type = 2  # Kroupa
    t_lookback = [0]  # yr
    outroot = 'spec'
    clobber = False

    specfile = config.get_file(kind='spec', M_H=M_H)

    if not os.path.exists(specfile):
        # Arbitrary SFH file (only the time bins are important)
        zcbfile = config.get_file(
                brick=config.BRICK_LIST[0], pixel=config.PIXEL_LIST[0],
                kind='bestzcb')

        # Make the spectrum evolution file
        filename = scombine.generate_basis(
                zcbfile, zmet=zmet, imf_type=imf_type,
                t_lookback=t_lookback, outroot=outroot, clobber=clobber)[0]

        # Relocate
        cmd = 'mv {0:s} {1:s}'.format(filename, specfile)
        subprocess.call(cmd, shell=True)

    return None


def __old__calc_flux(brick, pixel, magfilter, attenuated=False):
    """Calculate flux in magfilter for the given pixel."""
    if attenuated:
        bidx, pidx = config.BIDX[brick],config.PIDX[pixel]
        av, dav = config.AV[bidx, pidx], config.DAV[bidx, pidx]
    else:
        av, dav = 0, 0

    zcbfile = config.get_file(brick=brick, pixel=pixel, kind='bestzcb')
    specfile = config.get_file(kind='spec', M_H=0)

    # Initialize the spectrum combiner
    cb = scombine.Combiner(specfile, dust_law=attenuation.cardelli)
    filterlist = observate.load_filters([magfilter])

    # Get mag
    results = cb.combine(zcbfile, av=av, dav=dav, filterlist=filterlist)
    mag = float(results[2])
    mag += config.DMOD

    if magfilter == 'galex_FUV':
        flux = 1.40e-15 * 10**(-0.4*(mag - 18.82))
    else:
        flux = None

    return flux



sp_params = {
    'compute_vega_mags': False,
    'imf_type': 2,
    'zmet': 20,
    'sfh': 0}





def calc_flux(brick, pixel, sps, magfilter, attenuated=False):
    """Calculate flux in magfilter for the given pixel."""
    if attenuated:
        bidx, pidx = config.BRICK_LIST==brick, config.PIXEL_LIST==pixel
        av, dav = config.AV[bidx, pidx], config.DAV[bidx, pidx]
    else:
        av, dav = 0, 0

    zcbfile = config.get_file(brick=brick, pixel=pixel, kind='bestzcb')
    specfile = config.get_file(kind='spec', M_H=0)

    # Initialize the spectrum combiner
    cb = scombine.Combiner(specfile, dust_law=attenuation.cardelli)
    filterlist = observate.load_filters([magfilter])

    # Get mag
    results = cb.combine(zcbfile, av=av, dav=dav, filterlist=filterlist)
    mag = float(results[2])
    mag += config.DMOD

    if magfilter == 'galex_FUV':
        flux = 1.40e-15 * 10**(-0.4*(mag - 18.82))
    else:
        flux = None

    return flux



# Scripting
# ---------
def make_brickmaps_base(kind, func):
    for brick in config.BRICK_LIST:
        hdu = make_image(brick, func)
        mapfile = config.path(kind, brick=brick)
        dirname = os.path.dirname(mapfile)
        util.safe_mkdir(dirname)
        hdu.writeto(mapfile)
    return None


def make_mosaic_base(kind, work_dir, cdelt=None):
    """
    cdelt : float, optional
        Set the pixel scale (deg/pix) of the mosaic. If None, the mosaic of
        Alexia's pixel regions will have a pixel scale of 23.75 arcsec (in
        both x and y).

    """
    # Set up a working directory and subdirectories
    input_dir = os.path.join(work_dir, 'input')
    proj_dir = os.path.join(work_dir, 'reproject')
    for path in (work_dir, input_dir, proj_dir):
        util.safe_mkdir(path)

    # Symlink the input images
    for brick in config.BRICK_LIST:
        mapfile = config.path(kind, brick=brick)
        filename = os.path.basename(mapfile)
        linkpath = os.path.join(input_dir, filename)
        util.safe_symlink(mapfile, linkpath)

    # Metadata for input images
    metafile1 = os.path.join(work_dir, 'input.tbl')
    montage.mImgtbl(input_dir, metafile1, corners=True)

    # Make a template header
    hdrfile = os.path.join(work_dir, 'template.hdr')
    hdr = montage.mMakeHdr(metafile1, hdrfile, cdelt=cdelt)

    # Reproject to the template
    statfile = os.path.join(work_dir, 'stats.tbl')
    montage.mProjExec(metafile1, hdrfile, proj_dir, statfile, raw_dir=input_dir)

    # Metadata for reprojected images
    metafile2 = os.path.join(work_dir, 'reproject.tbl')
    montage.mImgtbl(proj_dir, metafile2, corners=True)

    # Build final mosaic
    mosaicfile = config.path(kind, brick='all')
    montage.mAdd(metafile2, hdrfile, mosaicfile, img_dir=proj_dir, exact=True)

    return None


def make_mean_sfr_100myr_brickmaps():
    make_brickmaps_base('mean_sfr_100myr', calc_mean_sfr)
    return None


def make_mean_sfr_100myr_mosaic(cdelt=None):
    work_dir = os.path.join(config.ANALYSIS_DIR, '_mean_sfr_100myr')
    make_mosaic_base('mean_sfr_100myr', work_dir, cdelt=cdelt)
    return None


def __old__make_mod_fuv_brickmaps(reddened=False):
    # wrap function based on `reddened`
    def func(brick, pixel):
        return calc_flux(brick, pixel, 'galex_FUV', reddened=reddened)

    if reddened:
        kind = 'mod_fuv_red'
    else:
        kind = 'mod_fuv'

    make_brickmaps_base(kind, func)
    return None


def __old__make_mod_fuv_mosaic(attenuated=False, cdelt=None):
    """
    reddened : bool
        If True, create a mosaic for reddened FUV flux. Else make a
        mosaic for intrinsic FUV flux (default).

    """
    if attenuated:
        kind = 'mod_fuv_red'
    else:
        kind = 'mod_fuv'

    work_dir = os.path.join(config.ANALYSIS_DIR, '_{:s}'.format(kind))
    work_dir = config.path('{0:s}_montage_dir'.append(kind))
    make_mosaic_base(kind, work_dir, cdelt=cdelt)
    return None


def make_mod_fuv_brickmaps(attenuated=False):
    sps = fsps.StellarPopulation()
    sps.params['sfh'] = 0
    sps.params['sfh'] = 0
    sps.params['zmet'] = 4

    # wrap function based on `sps` and `attenuated`
    def func(brick, pixel):
        return calc_flux(brick, pixel, sps, 'galex_FUV', attenuated=attenuated)

    if attenuated:
        kind = 'mod_fuv_attenuated'
    else:
        kind = 'mod_fuv_intrinsic'

    make_brickmaps_base(kind, func)
    return None


def make_galex_uv_mosaic(band):
    """`band` is either 'fuv' or 'nuv'."""
    if band == 'fuv':
        a = 1.40e-15
    elif band == 'nuv':
        a = 2.06e-16

    # Set up a working directory and subdirectories
    work_dir = os.path.join(config.MAP_DIR, '_galex_{:s}'.format(band))
    input_dir = os.path.join(work_dir, 'input')
    proj_dir = os.path.join(work_dir, 'reproject')
    for path in (work_dir, input_dir, proj_dir):
        util.safe_mkdir(path)

    # Prepare input images
    for field in config.GALEX_FIELD_LIST:
        filename = 'PS_M31_MOS{0:s}-{1:s}d-int.fits'.format(field, band[0])
        mapfile = os.path.join(config.GALEX_DIR, filename)
        hdr = fits.getheader(mapfile)
        data = fits.getdata(mapfile)

        # Set border pixels to NaN
        x0, y0 = 1920, 1920
        r = 1400
        y, x = np.indices(data.shape)
        incircle = (x+1)**2 + (y+1)**2 <= r**2
        iszero = data == 0
        data[-incircle & iszero] = np.nan

        # Convert input image units from cps to flux
        data = data * a

        # Write image
        mapfile = os.path.join(input_dir, filename)
        hdu = fits.PrimaryHDU(data, header=hdr)
        hdu.writeto(mapfile)

    # Metadata for input images
    metafile1 = os.path.join(work_dir, 'input.tbl')
    montage.mImgtbl(input_dir, metafile1, corners=True)

    # Copy template header
    hdrfile = '/Users/Jake/Research/PHAT/sfhmaps/data/map/_mod_fuv_attenuated/template.hdr'

    # Reproject to the template
    statfile = os.path.join(work_dir, 'stats.tbl')
    montage.mProjExec(metafile1, hdrfile, proj_dir, statfile, raw_dir=input_dir)

    # Metadata for reprojected images
    metafile2 = os.path.join(work_dir, 'reproject.tbl')
    montage.mImgtbl(proj_dir, metafile2, corners=True)

    # Build final mosaic
    mosaicfile = '/Users/Jake/Research/PHAT/sfhmaps/data/map/galex_{:s}.fits'.format(band)
    montage.mAdd(metafile2, hdrfile, mosaicfile, img_dir=proj_dir, exact=True)

    return None


def make_galex_uv_brickmaps(band, clean=False):
    """`band` is either 'fuv' or 'nuv'."""
    for brick in config.BRICK_LIST:

        # Write template header
        brickmap = config.get_file(brick=brick, kind='mod_fuv_attenuated')
        work_dir1 = os.path.dirname(brickmap)
        hdrfile = os.path.join(work_dir1, 'template.hdr')
        if clean:
            subprocess.call('rm {:s}'.format(hdrfile), shell=True)
        else:
            montage.mGetHdr(brickmap, hdrfile)

        # Metadata for input images
        work_dir2 = os.path.join(config.MAP_DIR, '_galex_{:s}'.format(band))
        input_dir = os.path.join(work_dir2, 'input')
        metafile1 = os.path.join(work_dir1, 'input.tbl')
        if clean:
            subprocess.call('rm {:s}'.format(metafile1), shell=True)
        else:
            montage.mImgtbl(input_dir, metafile1, corners=True)

        # Reproject to the template
        proj_dir = os.path.join(work_dir1, 'reproject')
        statfile = os.path.join(work_dir1, 'stats.tbl')
        if clean:
            subprocess.call('rm -rf {:s}'.format(proj_dir), shell=True)
            subprocess.call('rm {:s}'.format(statfile), shell=True)
        else:
            util.safe_mkdir(proj_dir)
            montage.mProjExec(metafile1, hdrfile, proj_dir, statfile, raw_dir=input_dir)

        # Metadata for reprojected images
        metafile2 = os.path.join(work_dir1, 'reproject.tbl')
        if clean:
            subprocess.call('rm {:s}'.format(metafile2), shell=True)
        else:
            montage.mImgtbl(proj_dir, metafile2, corners=True)

        # Build final mosaic
        mosaicfile = os.path.join(work_dir1, 'b{0:02d}_galex_{1:s}.fits'.format(brick, band))
        if clean:
            mosaicfile = os.path.splitext(mosaicfile)[0]
            subprocess.call('rm {:s}_area.fits'.format(mosaicfile), shell=True)
        else:
            montage.mAdd(metafile2, hdrfile, mosaicfile, img_dir=proj_dir, exact=True)

    return None


def main():
    #make_mean_sfr_100myr_brickmaps()
    #make_mean_sfr_100myr_mosaic()

    #make_mod_fuv_brickmaps(attenuated=True)
    #make_mod_fuv_mosaic(attenuated=True)

    #make_galex_uv_mosaic('nuv')
    #make_galex_uv_brickmaps('fuv', clean=True)

    # Tests
    # -----
    #for brick in config.BRICK_LIST:
    #    run_imwcs(brick)
    #compare_wcs()

    return None



# Tests
# -----
def run_imwcs(brick):
    """Fit a WCS to an image using points with known x,y and RA,dec.

    The program requires an existing .fits file (the WCS in the header is
    ignored). A new file is written with the suffix 'w.fits'.

    This seems to give results similar to ``make_header``, providing some
    validation for my CD matrix fitting method. This was tested on Brick
    15, where the imwcs solution was only marginally better at reproducing
    the input x,y from the known RA,dec (the maximum difference between
    input and output x,y was ~2% for both methods).

    """
    import subprocess

    coordfile = config.get_file(brick=brick, kind='vert')
    vertices = util.parse_coordfile(coordfile)
    xy, ad = make_coord_grid(vertices)

    hd = np.copy(ad)
    hd[0] = hd[0] * 24./360
    coords = []
    for xp, yp, RA, DEC in zip(xy[0].ravel(), xy[1].ravel(), hd[0].ravel(), hd[1].ravel()):
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

    mapfile = config.get_file(brick=brick, kind='mean_sfr_100myr')
    dirname = os.path.dirname(mapfile)
    coordlistfile = os.path.join(dirname, 'coords.dat')
    with open(coordlistfile, 'w') as f:
        f.writelines(coords)

    mapfile = os.path.basename(mapfile)
    coordlistfile = os.path.basename(coordlistfile)
    cmd = ('cd {0:s}; imwcs -ew -n 8 -u {1:s} {2:s}'
           .format(dirname, coordlistfile, mapfile))
    subprocess.call(cmd, shell=True)

    return None


def compare_wcs():
    """Compare WCS from ``make_header`` with that of ``run_imwcs`` (imwcs
    from wcstools).

    Results
    -------
    Overall, the differences between x,y calculated from the reference
    RA,dec are ~1e-2 pix or less for both methods.

    imwcs seems to semi-fail on Brick 12, with discrepancies up to 1-2 pix
    from the reference x,y (this is very noticeable when switching between
    the images in ds9). ``make_header`` therefore seems somewhat more
    robust.

    """
    from astropy import wcs

    for brick in config.BRICK_LIST:
        mapfile1 = config.get_file(brick=brick, kind='mean_sfr_100myr')
        mapfile2 = '{:s}w.fits'.format(mapfile1.split('.fits')[0])
        hdr1 = fits.getheader(mapfile1)
        hdr2 = fits.getheader(mapfile2)
        wcs1 = wcs.WCS(hdr1)
        wcs2 = wcs.WCS(hdr2)

        coordfile = config.get_file(brick=brick, kind='vert')
        vertices = util.parse_coordfile(coordfile)
        xy, ad = make_coord_grid(vertices)

        x, y = xy[0].ravel(), xy[1].ravel()
        x1, y1 = wcs1.wcs_world2pix(ad[0].ravel(), ad[1].ravel(), 1)
        x2, y2 = wcs2.wcs_world2pix(ad[0].ravel(), ad[1].ravel(), 1)

        dx1 = x1 - x
        meandx1, stddx1, maxdx1 = dx1.mean(), dx1.std(), np.abs(dx1).max()
        dy1 = y1 - y
        meandy1, stddy1, maxdy1 = dy1.mean(), dy1.std(), np.abs(dy1).max()

        dx2 = x2 - x
        meandx2, stddx2, maxdx2 = dx2.mean(), dx2.std(), np.abs(dx2).max()
        dy2 = y2 - y
        meandy2, stddy2, maxdy2 = dy2.mean(), dy2.std(), np.abs(dy2).max()

        print 'brick {:d}'.format(brick)
        print 'differences: mean       std        max'
        print 'make_header: {0: 10.5e} {1:10.5e} {2:10.5e}'.format(meandx1, stddx1, maxdx1)
        print '             {0: 10.5e} {1:10.5e} {2:10.5e}'.format(meandy1, stddy1, maxdy1)
        print '      imwcs: {0: 10.5e} {1:10.5e} {2:10.5e}'.format(meandx2, stddx2, maxdx2)
        print '             {0: 10.5e} {1:10.5e} {2:10.5e}'.format(meandy2, stddy2, maxdy2)
        print

    return None


def flux_conservation_test():
    """The following test shows that SFR is *not* conserved in reprojection."""
    filename0 = '/Users/Jake/Research/PHAT/sfh_maps/data/map/b15/b15_mean_sfr_100myr.fits'
    filename = '/Users/Jake/Research/PHAT/sfh_maps/data/map/_mean_sfr_100myr/reproject/hdu0_b15_mean_sfr_100myr.fits'

    data0 = fits.getdata(filename0)  # input image
    data = fits.getdata(filename)  # reprojected image

    np.nansum(data0)  # 0.03619
    np.nansum(data)  # 0.043404; 20% larger!!!


    """According to this blog post
    (http://montageblog.wordpress.com/2011/06/24/does-montage-conserve-flux-when-changing-the-image-resolution/),
    it may be necessary to distinguish between total/absolute quantities
    and densities (the post refers to "flux" vs. "flux density", though it
    is not clear what these terms actually mean... perhaps luminosity vs.
    flux, where flux is luminosity per unit area?). If I understand this
    correctly, then I should be able to normalize the input SFR maps by
    pixel area, and then recover SFR in the reprojected maps by multiplying
    by the area of a reprojected pixel. I test this below."""

    # set up directories for reprojection
    work_dir = '/Users/Jake/Research/PHAT/sfh_maps/data/map/test'
    input_dir = os.path.join(work_dir, 'input')
    proj_dir = os.path.join(work_dir, 'reproject')
    for path in (work_dir, input_dir, proj_dir):
        util.safe_mkdir(path)

    # make SFR density map
    testname0 = os.path.join(work_dir, 'input/test.fits')
    hdr0 = fits.getheader(filename0)
    from astropy import wcs
    w0 = wcs.WCS(hdr0)
    xy0 = np.mgrid[0:hdr0['naxis2']+1,0:hdr0['naxis1']+1][::-1] + 0.5
    ad0 = w0.wcs_pix2world(xy0.T.reshape(-1, 2), 1).reshape(xy0.T.shape).T
    area0 = []
    for i in range(data0.shape[0]):
        for j in range(data0.shape[1]):
            lon = np.array([ad0[0,i,j], ad0[0,i+1,j], ad0[0,i+1,j+1], ad0[0,i,j+1]])
            lat = np.array([ad0[1,i,j], ad0[1,i+1,j], ad0[1,i+1,j+1], ad0[1,i,j+1]])
            area0.append(util.sparea(lon, lat))
    area0 = np.array(area0).reshape(data0.shape)
    datat0 = data0 / area0  # SFR deg-2
    hdu = fits.PrimaryHDU(datat0, header=hdr0)
    hdu.writeto(testname0)

    # input image metadata
    metafile1 = os.path.join(work_dir, 'input.tbl')
    montage.mImgtbl(input_dir, metafile1, corners=True)

    # use the same header
    hdrfile = os.path.join(work_dir, 'template.hdr')
    import subprocess
    subprocess.call('cp /Users/Jake/Research/PHAT/sfh_maps/data/map/_mean_sfr_100myr/template.hdr {:s}'.format(hdrfile), shell=True)

    # reproject
    statfile = os.path.join(work_dir, 'stats.tbl')
    montage.mProjExec(metafile1, hdrfile, proj_dir, statfile, raw_dir=input_dir)

    # convert SFR density back to SFR
    testname = os.path.join(work_dir, 'reproject/hdu0_test.fits')
    datat = fits.getdata(testname)
    hdr = fits.getheader(testname)
    w = wcs.WCS(hdr)
    xy = np.mgrid[0:hdr['naxis2']+1,0:hdr['naxis1']+1][::-1] + 0.5
    ad = w.wcs_pix2world(xy.T.reshape(-1, 2), 1).reshape(xy.T.shape).T
    area= []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            lon = np.array([ad[0,i,j], ad[0,i+1,j], ad[0,i+1,j+1], ad[0,i,j+1]])
            lat = np.array([ad[1,i,j], ad[1,i+1,j], ad[1,i+1,j+1], ad[1,i,j+1]])
            area.append(util.sparea(lon, lat))
    area = np.array(area).reshape(data.shape)
    datat *= area

    # conservation test
    np.nansum(data0)  # 0.03619
    np.nansum(datat)  # 0.03840; 6% larger

    """
    6% is much better, but it's still large enough that I'm concerned that
    I'm doing something wrong.

    """
    return None


def plot_img(mode):
    from astropy.io import fits
    from matplotlib import pyplot as plt

    if mode == 'mean_sfr_100myr':
        imgfile = 'mean_sfr_100myr.fits'
        outfile = 'mean_sfr_100myr.pdf'
        fmin, fmax = 0, 1
        a = 1e3
    elif mode == 'mod_fuv_attenuated':
        imgfile = 'mod_fuv_attenuated.fits'
        outfile = 'mod_fuv_attenuated.pdf'
        fmin, fmax = 0, 1
        a = 1e15
    elif mode == 'galex_fuv':
        imgfile = 'galex_fuv.fits'
        outfile = 'galex_fuv.pdf'
        fmin, fmax = 0, 1
        a = 5e17
    elif mode == 'galex_nuv':
        imgfile = 'galex_nuv.fits'
        outfile = 'galex_nuv.pdf'
        fmin, fmax = 0, 1
        a = 5e17
    data = fits.getdata(imgfile)

    data = data[::-1].T  # landscape instead of portrait
    data = np.log10(a*data + 1)

    fig_dx = 10.0
    aspect = float(data.shape[0])/data.shape[1]
    fig_dy = fig_dx * aspect
    fig = plt.figure(figsize=(fig_dx, fig_dy))
    ax = fig.add_axes([0, 0, 1, 1])

    cmap = plt.cm.gist_heat_r
    vmin = (np.nanmax(data)-np.nanmin(data))*fmin + np.nanmin(data)
    vmax = (np.nanmax(data)-np.nanmin(data))*fmax + np.nanmin(data)
    img = ax.imshow(data, interpolation='nearest', origin='lower',
                    cmap=cmap, vmin=vmin, vmax=vmax)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(outfile)

    return None
#plot_img('mean_sfr_100myr')
#plot_img('galex_fuv')
#plot_img('galex_nuv')
#plot_img('mod_fuv_attenuated')


def plot_img2():
    from astropy.io import fits
    from matplotlib import pyplot as plt

    imgfile1 = 'mod_fuv_attenuated.fits'
    imgfile2 = 'galex_nuv.fits'
    outfile = 'nuv_ratio'
    data1 = fits.getdata(imgfile1)
    data2 = fits.getdata(imgfile2)

    data = data1 / data2
    hdr = fits.getheader(imgfile1)
    hdu = fits.PrimaryHDU(data, header=hdr)
    hdu.writeto('{:s}.fits'.format(outfile))

    data = data[::-1].T  # landscape instead of portrait

    fig_dx = 10.0
    aspect = float(data.shape[0])/data.shape[1]
    fig_dy = fig_dx * aspect
    fig = plt.figure(figsize=(fig_dx, fig_dy))
    ax = fig.add_axes([0, 0, 1, 1])

    fmin, fmax = 0, 1
    cmap = plt.cm.gist_heat_r
    vmin = (np.nanmax(data)-np.nanmin(data))*fmin + np.nanmin(data)
    vmax = (np.nanmax(data)-np.nanmin(data))*fmax + np.nanmin(data)
    img = ax.imshow(data, interpolation='nearest', origin='lower',
                    cmap=cmap, vmin=vmin, vmax=vmax)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig('{:s}.pdf'.format(outfile))

    return None
#plot_img2()



if __name__ == '__main__':
    main()



def test(**kwargs):
    plt.clf()
    gs = gridspec.GridSpec(3, 1, **kwargs)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])



"""
TODO
----
Once the mosaic creating ability is accomplished, figure out how to use Ben's code
to create a model FUV map.

Use the same input WCS header to create a corresponding GALEX FUV map.
- should reprojection be done using a header that fully covers the images?
  If so, then how do I later trim the full mosaic to just the area of
  interest?

"""



import match_wrapper as match
from sfhmaps import flux

zcbfile = '/Users/Jake/Research/PHAT/sfhmaps/analysis/b15/bestzcb/b15-001_best.zcb'
sfh = match.io.open_zcbfile(zcbfile)
sfh['age_i'] = 10**sfh['log(age_i)']  # Linearize ages
sfh['age_f'] = 10**sfh['log(age_f)']
sfh['SFR'][0] *= 1 - sfh['age_i'][0]/sfh['age_f'][0]  # Rescale 1st age bin
sfh['age_i'][0] = 0
age, sfr = (sfh['age_i'], sfh['age_f']), sfh['SFR']

fsps_kwargs = {
    'compute_vega_mags': False,
    'imf_type': 2,
    'zmet': flux.get_zmet(sfh['[M/H]'][sfh['SFR']>0][0]),
    'sfh': 0
    }

wave, spec = flux.calc_sed(sfr, age, fsps_kwargs=fsps_kwargs)

