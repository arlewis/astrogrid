"""

===============
`astrogrid.mwe`
===============

Mosaicking utilities.

"mwe" stands for `montage_wrapper` extension. Montage and `montage_wrapper`
are powerful and useful tools. The goal of this package is only to make
certain things in the Montage workflow a little easier.


Functions
---------

======== ==============
`mosaic` Make a mosaic.
======== ==============

"""
import astropy.io.fits
import montage_wrapper as montage
import numpy
import os
import shutil

from . import wcs


def mosaic(input_files, mosaic_file, work_dir, cleanup=False,
           density=False, full=False, header=None, preprocess=None,
           postprocess=None, **kwargs):
    """Make a mosiac.

    This is essentially a wrapper around `montage_wrapper.mosaic`. Files
    are handled slightly differently, however, and there is added support
    for processing the data before and after mosaicking. There are also
    options for using images in total flux units (Montage always assumes
    flux density units), and for creating an intermediate full mosaic
    before making the final mosaic for the given header.

    All `montage_wrapper.mosaic` keyword arguments are valid and have their
    usual defaults, except for,

    - `work_dir`: Reserved as an argument name. Internally, the
      `montage_wrapper.mosaic` keyword argument of the same name is set to
      ``work_dir/montage``.
    - `output_dir`: Set to ``work_dir/output``.
    - `imglist`: Set to None. Individual files are specified with the
      `input_files` argument.
    - `cleanup`: This is actually for the entire `work_dir`, not just the
      part used by `montage_wrapper.mosaic`, and the default is False.

    Parameters
    ----------
    input_files : list or string
        List of paths to the input images. This may also be the path to a
        directory containing all input images, mimicking the
        `montage_wrapper.mosaic` interface. In this case, `input_files`
        will automatically be set to a list of all files in the directory
        ending with ".fits".
    mosaic_file : str
        Path to the output mosaic file. It is either a symlink to
        ``work_dir/output/mosaic.fits`` or a new file depending on the
        `postprocess`, `density`, and `cleanup` keywords. The final mosaic
        always has the same units as the `input_files` images.
    work_dir : str
        Path to the working directory for all intermediate files produced
        by Montage, and has the following structure::

          work_dir/
            input/
              (Contains either symlinks to `input_files` or new files
              depending on the `preprocess` and `density` keywords.
              Assuming the `density` keyword has been set correctly, these
              images will always be in flux density units.)
            montage/
              (The working directory for `montage_wrapper.mosaic`. Contains
              all intermediate files produced by Montage, including the
              reprojected images and area files.)
            output/
              mosaic.fits
              mosaic_area.fits
              (The final mosaic and its accompanying area file. May also
              contain a full mosaic and area file depending on the `full`
              and `header` keywords. Mosaics are always in flux density
              units, and the area files are always in steradians.)

    cleanup : bool, optional
        If True, `work_dir` is be deleted after the mosaic is created.
        Default is False.
    density : bool, optional
        If True, the input images are in flux density units (i.e., signal
        per unit pixel area). If False (default), the input images are
        assumed to be in units of total flux, and are automatically scaled
        to flux density before reprojection.
    full : bool, optional
        If True, an intermediate mosaic that fully covers the input images
        is created first, and then the full mosaic is reprojected to
        `header`. The full mosaic is created using an automatically
        generated header (as if setting `header` to None). This option is
        useful in cases where `background_match` is True and `header`
        covers only a portion of the input images, as background matching
        works best when operating on the full extent of the input images.
        Default is False. This keyword is ignored if `header` is
        None.
    header : str, optional
        Path to the header file describing the output mosaic. Default is
        None, in which case an automatically generated header that
        optimally covers the input images is used.
    preprocess, postprocess : function, optional
        Functions for processing the raw input images before the input
        density images are created and after the final mosaic is created.
        The function arguments should be the image data array and the image
        header (`astropy.io.fits.Header`), and the return values should be
        the same. Default is None.

    """
    # Get list of files if `input_files` is a directory name
    if isinstance(input_files, basestring):
        dirname = os.path.dirname(input_files)
        input_files = [os.path.join(dirname, basename)
                       for basename in os.listdir(dirname)
                       if os.path.splitext(basename)[1] == '.fits']

    # Create working directory
    try:
        os.makedirs(work_dir)
    except OSError:
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)


    # Create input directory and populate it
    input_dir = os.path.join(work_dir, 'input')
    os.mkdir(input_dir)
    if preprocess or not density:
        # Create new input files
        for input_file in input_files:
            data, hdr = astropy.io.fits.getdata(input_file, header=True)
            if preprocess:
                data, hdr = preprocess(data, hdr)
            if not density:
                # Use pixel area to convert total flux into flux density
                dx, dy = wcs.calc_pixscale(hdr, ref='crpix').arcsec
                area = dx * dy  # arcsec2
                data /= area
            # Write
            basename, ext = os.path.splitext(os.path.basename(input_file))
            basename = '{0:s}_density{1:s}'.format(basename, ext)
            new_input_file = os.path.join(input_dir, basename)
            hdu = astropy.io.fits.PrimaryHDU(data, header=hdr)
            hdu.writeto(new_input_file)
    else:
        # Symlink existing files
        for input_file in input_files:
            basename = os.path.basename(input_file)
            new_input_file = os.path.join(input_dir, basename)
            os.symlink(input_file, new_input_file)

    # Mosaic
    output_dir = os.path.join(work_dir, 'output')
    kwargs['header'] = None if header and full else header
    kwargs['work_dir'] = os.path.join(work_dir, 'montage')
    kwargs['imglist'] = None
    kwargs['cleanup'] = False
    montage.mosaic(input_dir, output_dir, **kwargs)
    mtgmosaic_file = os.path.join(output_dir, 'mosaic.fits')
    mtgarea_file = os.path.join(output_dir, 'mosaic_area.fits')

    # Reproject full mosaic to given header
    if full:
        basename, ext = os.path.splitext(os.path.basename(mtgmosaic_file))
        fullmosaic_file = os.path.join(
            output_dir, '{0:s}_full{1:s}'.format(basename, ext))
        os.rename(mtgmosaic_file, fullmosaic_file)
        fullarea_file = os.path.join(
            output_dir, '{0:s}_full_area{1:s}'.format(basename, ext))
        os.rename(mtgarea_file, fullarea_file)
        montage.mProject(fullmosaic_file, mtgmosaic_file, header)

    # Write final mosaic, converting back into total flux if needed
    dirname = os.path.dirname(mosaic_file)
    try:
        os.makedirs(dirname)
    except OSError:
        pass
    if postprocess or not density:
        # Create new file
        data, hdr = astropy.io.fits.getdata(mtgmosaic_file, header=True)
        if not density:
            # Convert flux density into total flux
            area = astropy.io.fits.getdata(mtgarea_file)  # steradians
            area *= (180/np.pi*3600)**2  # arcsec2
            data *= area
        if postprocess:
            data, hdr = postprocess(data, hdr)
        # Write
        hdu = astropy.io.fits.PrimaryHDU(data, header=hdr)
        try:
            hdu.writeto(mosaic_file)
        except IOError:
            os.remove(mosaic_file)
            hdu.writeto(mosaic_file)
    elif cleanup:
        # Move existing file
        os.rename(mtgmosaic_file, mosaic_file)
    else:
        # Symlink existing file
        try:
            os.symlink(mtgmosaic_file, mosaic_file)
        except OSError:
            os.remove(mosaic_file)
            os.symlink(mtgmosaic_file, mosaic_file)

    # Cleanup
    if cleanup:
        shutil.rmtree(work_dir)

    return




"""

    Notes
    -----
    - Input and reprojected maps were compared for 21 PHAT bricks. For a
      given brick, the percent difference between the pixel sum of the
      native map and the pixel sum of the reprojected map was less than
      ~0.01%. The average percent difference was ~0.001%.

    - mProjExec and mProject produce nearly identical results (within
      ~0.01%).

    - Settings that affect reprojection: mProject has the 'z' flag, which
      controls the drizzle factor. 1 seems to be the default, and appears
      to give the best results anyway so there is no reason to change it.
      mProjExec doesn't really have any settings that change the
      reprojection results.

    - Pixel area varies slightly accross brick 15 in both the native and
      the reprojected images. The effect causes only ~0.001% difference
      between the minimum and maximum areas, however, so the pixels can be
      safely treated as having constant area.

    - Three pixel area calculation methods were tested: 1) calculate the
      areas of all pixels from the coordinates of their corners assuming
      spherical rectangles, 2) calculate the areas of all pixels from their
      x and y scales assuming planar rectangles, and 3) same as 2, but only
      calculate the area for the reference pixel and assume that value for
      all of the pixels. All three area calculation methods agree to within
      ~0.01% to ~0.001%. Might as well just use the simplest method (3).

"""

def _make_mosaic(kind, make_header=False, cdelt=None, background_match=False,
                level_only=True, preprocess=None, postprocess=None):
    """Make a mosiac.

    Montage is used to create the actual mosaic. The rest of this function
    is for processing the data before and after mosaicing, and moving the
    Montage output files to their desired locations.

    There are eight types of files involved in the construction of a
    mosaic, and their kinds are implied by the value of `kind` (the kinds,
    given in parentheses, are used with `m31flux.config.path` to define
    their paths):

    - Input images ('kind')
    - Density images ('kind.density')
    - Template header ('kind.hdr')
    - Reprojected images and their area files ('kind.reproject',
      'kind.area')
    - Density mosaic and its area file ('kind.reproject.add',
      'kind.area.add')
    - Final mosaic ('kind.add')

    The input images are used to create a set of "density" images (signal
    per unit pixel area). The density images are optionally preprocessed,
    and are then reprojected to the template header and added via Montage.
    The final mosaic is the product of the density mosaic and the
    corresponding area file ('kind.area.add'), which is optionally
    postprocessed before being written to file.

    Parameters
    ----------
    kind : str
        File kind for the input images (see `m31flux.config.path`).
    make_header : bool, optional
        If True, a template header for the mosaic will be created based on
        the input images. If False (default), then the file pointed to by
        `config.path` is used instead (kind.hdr).
    cdelt : float, optional
        Set the pixel scale (deg/pix) for the template header. If None
        (default), the scale is determined automatically. This keyword is
        ignored if `make_header` is False.
    background_match, level_only : bool, optional
        Background matching parameters. See `montage_wrapper.mosaic` for
        details.
    preprocess, postprocess : function, optional
        Functions for processing the raw input images before the input
        density images are created and after the final mosaic is created.
        The function arguments should be the image data array and the image
        header (`astropy.io.fits.Header`), and the return values should be
        the same. Default is None.

    Notes
    -----
    - Input and reprojected maps were compared for 21 PHAT bricks. For a
      given brick, the percent difference between the pixel sum of the
      native map and the pixel sum of the reprojected map was less than
      ~0.01%. The average percent difference was ~0.001%.

    - mProjExec and mProject produce nearly identical results (within
      ~0.01%).

    - Settings that affect reprojection: mProject has the 'z' flag, which
      controls the drizzle factor. 1 seems to be the default, and appears
      to give the best results anyway so there is no reason to change it.
      mProjExec doesn't really have any settings that change the
      reprojection results.

    - Pixel area varies slightly accross brick 15 in both the native and
      the reprojected images. The effect causes only ~0.001% difference
      between the minimum and maximum areas, however, so the pixels can be
      safely treated as having constant area.

    - Three pixel area calculation methods were tested: 1) calculate the
      areas of all pixels from the coordinates of their corners assuming
      spherical rectangles, 2) calculate the areas of all pixels from their
      x and y scales assuming planar rectangles, and 3) same as 2, but only
      calculate the area for the reference pixel and assume that value for
      all of the pixels. All three area calculation methods agree to within
      ~0.01% to ~0.001%. Might as well just use the simplest method (3).

    """
    input_files = config.path(kind)
    density_files = config.path('{:s}.density'.format(kind))
    header_file = config.path('{:s}.hdr'.format(kind))
    proj_files = config.path('{:s}.reproject'.format(kind))
    area_files = config.path('{:s}.area'.format(kind))
    projadd_file = config.path('{:s}.reproject.add'.format(kind))
    areaadd_file = config.path('{:s}.area.add'.format(kind))
    add_file = config.path('{:s}.add'.format(kind))

    # Make density images
    for input_file, density_file in zip(input_files, density_files):
        # Load input image, perform initial processing
        data, hdr = astropy.io.fits.getdata(input_file, header=True)
        if preprocess is not None:
            data, hdr = preprocess(data, hdr)

        # Divide by pixel area to create a density image
        dx, dy = astrogrid.wcs.calc_pixscale(hdr, ref='crpix').arcsec
        area = dx * dy  # arcsec2
        data = data / area

        # Write
        hdu = astropy.io.fits.PrimaryHDU(data, header=hdr)
        dirname = os.path.dirname(density_file)
        safe_mkdir(dirname)
        if os.path.exists(density_file):
            os.remove(density_file)
        hdu.writeto(density_file)

    # Temporary directory to hold all Montage inputs and outputs
    dirname, basename = os.path.split(add_file)
    basename = '_temp_{:s}'.format(os.path.splitext(basename)[0])
    temp_dir = os.path.join(dirname, basename)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # Create input dir and symlink density files
    input_dir = os.path.join(temp_dir, 'input')
    os.mkdir(input_dir)
    for density_file in density_files:
        filename = os.path.basename(density_file)
        os.symlink(density_file, os.path.join(input_dir, filename))

    # Specify output directory and Montage working directory
    output_dir = os.path.join(temp_dir, 'output')
    work_dir = os.path.join(temp_dir, 'work')

    # Template header file
    if make_header:
        header_file = None
    else:
        header_file = config.path('{:s}.hdr'.format(kind))

    # Make the mosaic; paths to the Montage output files
    montage.mosaic(input_dir, output_dir, header=header_file,
                   background_match=background_match, level_only=level_only,
                   exact_size=True, cleanup=False, work_dir=work_dir)
    mheader_file = os.path.join(work_dir, 'header.hdr')
    mproj_dir = os.path.join(work_dir, 'projected')
    mproj_files, marea_files = [], []
    for density_file in density_files:
        basename, ext = os.path.splitext(os.path.basename(density_file))
        ### Does Montage always prefix reprojected images with 'hdu0_'?
        mproj_file = 'hdu0_{0:s}{1:s}'.format(basename, ext)
        mproj_files.append(os.path.join(mproj_dir, mproj_file))
        marea_file = 'hdu0_{0:s}_area{1:s}'.format(basename, ext)
        marea_files.append(os.path.join(mproj_dir, marea_file))
    mprojadd_file = os.path.join(output_dir, 'mosaic.fits')
    mareaadd_file = os.path.join(output_dir, 'mosaic_area.fits')

    # Move and rename files:
    # Template header
    if make_header:
        header_file = config.path('{:s}.hdr'.format(kind))
        os.rename(mheader_file, header_file)
    # Reprojected images
    for mproj_file, proj_file in zip(mproj_files, proj_files):
        dirname = os.path.dirname(proj_file)
        safe_mkdir(dirname)
        if os.path.exists(proj_file):
            os.remove(proj_file)
        os.rename(mproj_file, proj_file)
    # Area files for reprojected images
    for marea_file, area_file in zip(marea_files, area_files):
        dirname = os.path.dirname(area_file)
        safe_mkdir(dirname)
        if os.path.exists(area_file):
            os.remove(area_file)
        os.rename(marea_file, area_file)
    # Density mosaic
    dirname = os.path.dirname(projadd_file)
    safe_mkdir(dirname)
    if os.path.exists(projadd_file):
        os.remove(projadd_file)
    os.rename(mprojadd_file, projadd_file)
    # Area file for density mosaic
    dirname = os.path.dirname(areaadd_file)
    safe_mkdir(dirname)
    if os.path.exists(areaadd_file):
        os.remove(areaadd_file)
    os.rename(mareaadd_file, areaadd_file)

    # Clean up
    #shutil.rmtree(temp_dir)

    # Final mosaic
    data, hdr = astropy.io.fits.getdata(projadd_file, header=True)
    area = astropy.io.fits.getdata(areaadd_file) * (180/np.pi*3600)**2 # arcsec2
    data = data * area
    if postprocess is not None:
        data, hdr = postprocess(data, hdr)
    add_file = config.path('{:s}.add'.format(kind))
    dirname = os.path.dirname(add_file)
    safe_mkdir(dirname)
    if os.path.exists(add_file):
        os.remove(add_file)
    hdu = astropy.io.fits.PrimaryHDU(data, header=hdr)
    hdu.writeto(add_file)

    return

def _montage_test():
    # create density images

    input_dir = os.path.dirname(density_files[0])

    # image metadata
    meta1_file = os.path.join(input_dir, 'meta1.tbl')
    montage.mImgtbl(input_dir, meta1_file, corners=True)

    # make header
    #lon, lat = [], []
    #for density_file in density_files:
    #    data, hdr = astropy.io.fits.getdata(density_file, header=True)
    #    wcs = astropy.wcs.WCS(hdr)
    #    x1, y1 = 0.5, 0.5
    #    y2, x2 = data.shape
    #    x2, y2 = x2 + 0.5, y2 + 0.5
    #    x, y = [x1, x2, x2, x1], [y1, y1, y2, y2]
    #    ln, lt = wcs.wcs_pix2world(x, y, 1)
    #    lon += list(ln)
    #    lat += list(lt)
    #lon1, lon2 = np.min(lon), np.max(lon)
    #lat1, lat2 = np.min(lat), np.max(lat)
    hdr_file = os.path.join(os.path.dirname(input_dir), 'test.hdr')
    montage.mMakeHdr(meta1_file, hdr_file)

    # reproject
    proj_dir = os.path.dirname(proj_files[0])
    safe_mkdir(proj_dir)
    stats_file = os.path.join(proj_dir, 'stats.tbl')
    montage.mProjExec(meta1_file, hdr_file, proj_dir, stats_file,
                      raw_dir=input_dir, exact=True)

    # image metadata
    meta2_file = os.path.join(proj_dir, 'meta2.tbl')
    montage.mImgtbl(proj_dir, meta2_file, corners=True)

    # Background modeling
    diff_dir = os.path.join(os.path.dirname(proj_dir), 'difference')
    safe_mkdir(diff_dir)
    diff_file = os.path.join(diff_dir, 'diffs.tbl')
    montage.mOverlaps(meta2_file, diff_file)
    montage.mDiffExec(diff_file, hdr_file, diff_dir, proj_dir)
    fits_file = os.path.join(diff_dir, 'fits.tbl')
    montage.mFitExec(diff_file, fits_file, diff_dir)

    # Background matching
    corr_dir = os.path.join(os.path.dirname(proj_dir), 'correct')
    safe_mkdir(corr_dir)
    corr_file = os.path.join(corr_dir, 'corrections.tbl')
    montage.mBgModel(meta2_file, fits_file, corr_file, level_only=False)
    montage.mBgExec(meta2_file, corr_file, corr_dir, proj_dir=proj_dir)

    # Native mosaic
    projadd_file = config.path('{:s}.reproject.add'.format(kind))
    projadd_dir, filename = os.path.split(projadd_file)
    filename, ext = os.path.splitext(filename)
    filename = '{0:s}_native{1:s}'.format(filename, ext)
    projaddnative_file = os.path.join(projadd_dir, filename)
    safe_mkdir(projadd_dir)
    montage.mAdd(meta2_file, hdr_file, projaddnative_file, img_dir=corr_dir, exact=True)

    # Reproject to final header
    header_file = config.path('{:s}.hdr'.format(kind))
    montage.mProject(projaddnative_file, projadd_file, header_file)

    # Postprocess
    data, hdr = astropy.io.fits.getdata(projaddnative_file, header=True)
    x1, x2 = 900, 1900
    y1, y2 = 3000, 4500
    val = np.mean(data[y1:y2,x1:x2])

    data, hdr = astropy.io.fits.getdata(projadd_file, header=True)
    data = data - val
    areaadd_file = config.path('{:s}.area.add'.format(kind))
    area = astropy.io.fits.getdata(areaadd_file) * (180/np.pi*3600)**2 # arcsec2
    data = data * area

    add_file = config.path('{:s}.add'.format(kind))
    dirname = os.path.dirname(add_file)
    safe_mkdir(dirname)
    if os.path.exists(add_file):
        os.remove(add_file)
    hdu = astropy.io.fits.PrimaryHDU(data, header=hdr)
    hdu.writeto(add_file)
