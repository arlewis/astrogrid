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
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import astropy.io.fits
import montage_wrapper as montage
import numpy as np
import os
import shutil

from . import wcs


def mosaic(input_files, mosaic_file, work_dir, background_match=False,
           cdelt=None, density=False, equinox=None, header=None,
           level_only=False, north_aligned=False, postprocess=None,
           preprocess=None, system=None, weights_file=None):
    """Make a mosiac.

    High-level wrapper around several Montage operations similar to
    `montage_wrapper.mosaic`. The main differences are 1) added support for
    preprocessing the input images before reprojection and postprocessing
    the final image after mosaicking, 2) options for using images in total
    flux units instead of flux density (as assumed by Montage), 3) more of
    the `montage_wrapper.mMakeHdr` keywords available for header creation,
    and 4) the `whole` keyword for `montage_wrapper.mProjExec` is
    automatically set to True when `background_match` is True. The latter
    is important since backround matching behaves unreliably otherwise.

    Parameters
    ----------
    input_files : list or string
        List of paths to the input images. This may also be the path to a
        directory containing all input images, in which case `input_files`
        will automatically be set to a list of all files in the directory
        ending with ".fits".
    mosaic_file : str
        Path to the output mosaic file. The final mosaic always has the
        same units as the `input_files` images.
    work_dir : str
        Path to the working directory for all intermediate files produced
        by Montage. The directory has the following structure::

          work_dir/
            input/
              Contains either symlinks to `input_files` or new files
              depending on the `preprocess` and `density` keywords.
              Assuming the `density` keyword has been set correctly, these
              images will always be in flux density units.
            reprojected/
              The reprojected images.
            differences/
              Difference calculations for background matching (only if
              `background_match` is True).
            corrected/
              Background-matched images (only if `background_match` is
              True).
            output/
              The intermediate mosiac used to produce the final mosaic
              file, depending on the `density` and `postprocess` keywords.

    background_match : bool, optional
        If True, match the background levels of the reprojected images
        before mosaicking. Automatically sets ``whole = True`` in
        `montage_wrapper.mProjExec`. Default is False.
    cdelt : float, optional
        See `header` and `montage_wrapper.mMakeHdr`. Default is None.
    density : bool, optional
        If True, the input images are in flux density units (i.e., signal
        per unit pixel area). If False (default), the input images are
        assumed to be in units of total flux, and are automatically scaled
        to flux density before reprojection.
    equinox : str, optional
        See `header` and `montage_wrapper.mMakeHdr`. Default is None.
    header : str, optional
        Path to the template header file describing the output mosaic.
        Default is None, in which case a template header is created
        automatically using `montage_wrapper.mMakeHdr` and the `cdelt`,
        `equinox`, `north_aligned`, and `system` keyword arguments.
    level_only : bool, optional
        See `montage_wrapper.mBgModel`. Ignored if `background_match` is
        False. Default is False.
    north_aligned : bool, optional
        See `header` and `montage_wrapper.mMakeHdr`. Default is None.
    postprocess, preprocess : function, optional
        Functions for processing the raw input images before the input
        density images are created (`preprocess`) and after the final
        mosaic is created (`postprocess`). The function arguments should be
        the image data array and the image header
        (`astropy.io.fits.Header`), and the return values should be the
        same. Default is None.
    system : str, optional
        See `header` and `montage_wrapper.mMakeHdr`. Default is None.
    weights_file : str, optional
        Path to output pixel weights file. Pixel weights are derived from
        the final mosaic area file. Weights are normalized to 1, and
        represent coverage of the mosaic area by the input images. Unlike
        Montage area files, regions where the input images overlap are not
        considered. Default is None.

    Returns
    -------
    None

    """
    # Get list of files if input_files is a directory name
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


    # Create input directory, populate it, and get image metadata
    input_dir = os.path.join(work_dir, 'input')
    os.mkdir(input_dir)

    if preprocess or not density:
        # Create new input files
        for input_file in input_files:
            data, hdr = astropy.io.fits.getdata(input_file, header=True)

            if preprocess:
                data, hdr = preprocess(data, hdr)

            if not density:
                # Convert total flux into flux density
                dx, dy = wcs.calc_pixscale(hdr, ref='crpix').arcsec
                pixarea = dx * dy  # arcsec2
                data /= pixarea

            # Write
            basename = os.path.basename(input_file)
            basename = '_density'.join(os.path.splitext(basename))
            new_input_file = os.path.join(input_dir, basename)
            hdu = astropy.io.fits.PrimaryHDU(data, header=hdr)
            hdu.writeto(new_input_file)

    else:
        # Symlink existing files
        for input_file in input_files:
            basename = os.path.basename(input_file)
            new_input_file = os.path.join(input_dir, basename)
            os.symlink(input_file, new_input_file)

    input_table = os.path.join(input_dir, 'input.tbl')
    montage.mImgtbl(input_dir, input_table, corners=True)


    # Template header
    if header is None:
        template_header = os.path.join(work_dir, 'template.hdr')
        montage.mMakeHdr(input_table, template_header, cdelt=cdelt,
                         equinox=equinox, north_aligned=north_aligned,
                         system=system)
    else:
        template_header = header


    # Create reprojection directory, reproject, and get image metadata
    proj_dir = os.path.join(work_dir, 'reprojected')
    os.makedirs(proj_dir)

    whole = True if background_match else False
    stats_table = os.path.join(proj_dir, 'mProjExec_stats.log')
    montage.mProjExec(input_table, template_header, proj_dir, stats_table,
                      raw_dir=input_dir, whole=whole)

    reprojected_table = os.path.join(proj_dir, 'reprojected.tbl')
    montage.mImgtbl(proj_dir, reprojected_table, corners=True)


    # Background matching
    if background_match:
        diff_dir = os.path.join(work_dir, 'differences')
        os.makedirs(diff_dir)

        # Find overlaps
        diffs_table = os.path.join(diff_dir, 'differences.tbl')
        montage.mOverlaps(reprojected_table, diffs_table)

        # Calculate differences between overlapping images
        montage.mDiffExec(diffs_table, template_header, diff_dir,
                          proj_dir=proj_dir)

        # Find best-fit plane coefficients
        fits_table = os.path.join(diff_dir, 'fits.tbl')
        montage.mFitExec(diffs_table, fits_table, diff_dir)

        # Calculate corrections
        corr_dir = os.path.join(work_dir, 'corrected')
        os.makedirs(corr_dir)
        corrections_table = os.path.join(corr_dir, 'corrections.tbl')
        montage.mBgModel(reprojected_table, fits_table, corrections_table,
                         level_only=level_only)

        # Apply corrections
        montage.mBgExec(reprojected_table, corrections_table, corr_dir,
                        proj_dir=proj_dir)

        img_dir = corr_dir

    else:
        img_dir = proj_dir


    # Make mosaic
    output_dir = os.path.join(work_dir, 'output')
    os.makedirs(output_dir)

    out_image = os.path.join(output_dir, 'mosaic.fits')
    montage.mAdd(reprojected_table, template_header, out_image,
                 img_dir=img_dir, exact=True)


    # Pixel areas and weights
    if weights_file or not density:
        area_file = '_area'.join(os.path.splitext(out_image))
        area, hdr = astropy.io.fits.getdata(area_file, header=True)  # steradians
        area *= (180/np.pi*3600)**2  # arcsec2
        dx, dy = wcs.calc_pixscale(hdr, ref='crpix').arcsec
        pixarea = dx * dy  # arcsec2
        area = np.clip(area, 0, pixarea)  # Don't care about overlaps
        if weights_file:
            weights = area / pixarea  # Normalize to 1
            hdu = astropy.io.fits.PrimaryHDU(weights, header=hdr)
            try:
                hdu.writeto(weights_file)
            except IOError:
                os.remove(weights_file)
                hdu.writeto(weights_file)


    # Write final mosaic
    dirname = os.path.dirname(mosaic_file)
    try:
        os.makedirs(dirname)
    except OSError:
        pass

    if postprocess or not density:
        # Create new file
        data, hdr = astropy.io.fits.getdata(out_image, header=True)

        if not density:
            # Convert flux density into total flux
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

    else:
        # Move existing file
        os.rename(out_image, mosaic_file)

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
