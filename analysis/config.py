import numpy as np
import os


def is_string(obj):
    """Check if an object is a string rather than a list of strings.

    Note that a slightly more straightforward test would be, ::

      return hasattr(obj, '__iter__')

    because lists and tuples have the __iter__ method while strings do not.
    However, this will only work in Python 2.x as strings in Python 3 have
    the __iter__ method.

    """
    return ''.join(obj) == obj



# Brick pixel parameters
# ----------------------
NROW, NCOL = 15, 30  # Bricks are divided into grids NROW rows by NCOL columns
BRICK_LIST = np.r_[2, np.arange(4,24)]  # Bricks are IDd by integers
PIXEL_LIST = np.arange(NROW*NCOL) + 1  # Each brick has NROW*NCOL pixels, IDd by integer



# Paths
# -----
DATA_DIR = '/Users/Jake/Research/PHAT/sfhmaps/data'
SFH_DIR = os.path.join(DATA_DIR, 'sfh')
SFHPROC_DIR = os.path.join(DATA_DIR, 'proc')


def _get_file_extpar(**kwargs):
    dirname = os.path.join(SFH_DIR, 'B{0:02d}')
    filename = 'b{0:02d}_region_AvdAv.dat'
    pth = os.path.join(dirname, filename)
    path_list = [pth.format(brick) for brick in kwargs['brick']]
    return path_list


def _get_file_vert(**kwargs):
    dirname = os.path.join(SFH_DIR, 'B{0:02d}')
    filename = 'M31-B{0:02d}_15x30_subregion-exact-vertices.dat'
    pth = os.path.join(dirname, filename)
    path_list = [pth.format(brick) for brick in kwargs['brick']]
    return path_list


def _get_file_phot(**kwargs):
    dirname = os.path.join(SFH_DIR, 'B{0:02d}', 'phot')
    filename = 'M31-B{0:02d}_15x30-{1:03d}.gst.match'
    pth = os.path.join(dirname, filename)
    path_list = [pth.format(brick, pixel) for brick, pixel in kwargs['brickpixel']]
    return path_list


def _get_file_sfh(**kwargs):
    dirname = os.path.join(SFH_DIR, 'B{0:02d}', 'sfh')
    filename = 'M31-B{0:02d}_15x30-{1:03d}_{2:.1f}-{3:.1f}_best.sfh'
    pth = os.path.join(dirname, filename)
    path_list = []
    for brick, pixel in kwargs['brickpixel']:
        i = (BRICK_LIST==brick, PIXEL_LIST==pixel)
        path_list.append(pth.format(brick, pixel, AV[i][0], DAV[i][0]))
    return path_list


def _get_file_cmd(**kwargs):
    dirname = os.path.join(SFH_DIR, 'B{0:02d}', 'cmd')
    filename = 'M31-B{0:02d}_15x30-{1:03d}_{2:.1f}-{3:.1f}_best.sfh.cmd'
    pth = os.path.join(dirname, filename)
    path_list = []
    for brick, pixel in kwargs['brickpixel']:
        i = (BRICK_LIST==brick, PIXEL_LIST==pixel)
        path_list.append(pth.format(brick, pixel, AV[i][0], DAV[i][0]))
    return path_list


def _get_file_bestzcb(**kwargs):
    dirname = os.path.join(SFHPROC_DIR, 'b{0:02d}', 'bestzcb')
    filename = 'b{0:02d}-{1:03d}_best.zcb'
    pth = os.path.join(dirname, filename)
    path_list = [pth.format(brick, pixel) for brick, pixel in kwargs['brickpixel']]
    return path_list


def _get_mosaic(kind, **kwargs):
    brick_list = kwargs.get('brick')
    path_list = []
    if brick_list == 'full':
        pass  # path_list.append(mosaic_file)
    else:
        for brick in brick_list:
            pass  # path_list.append(brick_file)
    return path_list


def _parse_extparfile():
    """Load pixel Av and dAv extinction parameters for all bricks into an
    array (rows are bricks, columns are pixels). This is required for some
    filenames.

    """
    av_arr = np.zeros((BRICK_LIST.size, PIXEL_LIST.size))
    dav_arr = np.zeros((BRICK_LIST.size, PIXEL_LIST.size))
    for i, brick in enumerate(BRICK_LIST):
        extparfile = _get_file_extpar(brick=[brick])[0]

        with open(extparfile, 'r') as f:
            for j, line in enumerate(f):
                av, dav = line.split()[1:3]
                av_arr[i,j], dav_arr[i,j] = float(av), float(dav)

    return av_arr, dav_arr
AV, DAV = _parse_extparfile()


def _find_missing():
    """Find all brick,pixel pairs for which no data are available."""
    missing = []

    # Missing brick numbers, missing pixel numbers
    for brick in range(1, 24):
        if brick not in BRICK_LIST:
            for pixel in PIXEL_LIST:
                missing.append((brick, pixel))

    # Find brick,pixel with entry in AV and DAV arrays, but with no actual
    # data (flagged by `badval`)
    badval = 99
    i, j = np.where((AV==badval) & (DAV==badval))
    for brick, pixel in zip(BRICK_LIST[i], PIXEL_LIST[j]):
        missing.append((brick, pixel))

    return missing
MISSING = _find_missing()


def path(kind, **kwargs):
    """
    Parameters
    ----------
    kind : str
        The general type of file. The set of files returned is refined
        using keyword arguments. See the Notes section below.
    brick : {int, list of ints, 'all'}, optional
        Return the file(s) of type `kind` for a brick or list of bricks.
        Whether a single file or a list of files is returned for a given
        brick depends on `pixel`. None (default) is equivalent to a list of
        all bricks. 'full' returns a file for the combination of all
        bricks, e.g., a mosaic (pixels are irrelevant in this case). Valid
        uses of this keyword depend on `kind`.
    pixel : int, list of ints, optional
        Return the file(s) for type `kind` for a pixel or list of pixels.
        Whether a single file or list of files is returned for a given
        pixel depends on `brick`. None (default) is equivalent to a list of
        all pixels. Valid uses of this keyword depend on `kind`.

    Returns
    -------
    string or list of strings
        List of file paths.

    Notes
    -----
    Values of the `kind` parameter:

    vert
       File of RA,dec coordinates of the corners of all pixels in a brick.
    extpar
        File with best-fit Av and dAv extinction parameters for all pixels
        in a brick.
    phot, sfh, cmd, bestzcb
        SFH-related files for a given pixel. They are produced by the
        following procedure (kind names are in brackets)::

          calcsfh par [phot] fake [sfh] -zinc -mcdata -dAvy=0 -dAv=dAv  # other output: [cmd], hmcdat
          zcombine [sfh] -bestonly > [bestzcb]

        .. note:: In case additional kinds are supported in the future,
           should probably stick to a naming scheme similar to this::

             hybridMC hmcdat [hmcsfh] -tint=2.0 -nmc=10000 -dt=0.015
             zcombine -unweighted -medbest -jeffreys -best=[bestzcb] [hmcsfh] > [hmczcb]
             zcmerge [bestzcb] [hmczcb] -absolute > [besthmczcb]

             zcombine [extsyssfh]* > [extsyszcb]  # extinction systematics
             zcmerge [bestzcb] [extsyszcb] -absolute > [bestextzcb]

             zcombine [isosyssfh]* > [isosyszcb]  # isochrone systematics
             zcmerge [bestzcb] [isosyszcb] -absolute > [bestisozcb]

    """
    brick = kwargs.get('brick')
    pixel = kwargs.get('pixel')

    if brick == 'full' and kind not in []:
        kwargs['brick'] = []
    else:
        if brick is None:
            brick = BRICK_LIST
        elif not hasattr(brick, '__iter__'):
            brick = [brick]  # convert to list
        brick = [brk for brk in brick if brk in BRICK_LIST]
        kwargs['brick'] = brick

        if kind in ['phot', 'sfh', 'cmd', 'bestzcb']:
            if pixel is None:
                pixel = PIXEL_LIST
            elif not hasattr(pixel, '__iter__'):
                pixel = [pixel]  # convert to list
            pixel = [pix for pix in pixel if pix in PIXEL_LIST]
            kwargs['pixel'] = pixel

            # list of brick,pixel pairs
            brickpixel_list = []
            for brick in kwargs['brick']:
                for pixel in kwargs['pixel']:
                    if (brick, pixel) in MISSING:
                        continue
                    brickpixel_list.append((brick, pixel))
            kwargs['brickpixel'] = brickpixel_list

    kind_dict = {
        'vert': _get_file_vert,
        'extpar': _get_file_extpar,
        'phot': _get_file_phot,
        'sfh': _get_file_sfh,
        'cmd': _get_file_cmd,
        'bestzcb': _get_file_bestzcb,
        }

    kwargs['kind'] = kind
    path_list = kind_dict[kind](**kwargs)

    if not path_list:
        path_list = None
    elif len(path_list) == 1:
        # contains just one string, so return only that string
        path_list = path_list[0]

    return path_list



def old_stuff():
    # Brick pixel parameters
    # ----------------------
    BIDX = {brick: idx for idx, brick in enumerate(BRICK_LIST)}
    PIDX = {pixel: idx for idx, pixel in enumerate(PIXEL_LIST)}



    # Paths
    # -----
    PROC_DIR = os.path.join(DATA_DIR, 'proc')
    MAP_DIR = os.path.join(DATA_DIR, 'map')
    SPEC_DIR = os.path.join(DATA_DIR, 'spec')
    ANALYSIS_DIR = os.path.join(PROJECT_DIR, 'analysis')

    def get_file(brick=None, pixel=None, kind=None, M_H=None):
        """Return various types of files in the project.

        Parameters
        ----------
        brick : int, optional
            The number of the brick containing the pixel(s) of interest.
            Default is None. [1]_
        pixel : int, optional
            The number of the pixel of interest. Default is None. [1]_
        kind : {'vert', 'sfh', 'cmd', 'phot', 'bestzcb', '100myr_mean_sfr',
                'spec', 'mod_fuv_intrinsic', 'mod_fuv_attenuated'}, optional
            Selects the specific kind of data file. Default is None. [1]_

        .. [1] See Kinds below for more information.

        Returns
        -------
        string or list of strings

        Kinds
        -----
        'vert'
            File containing the RA,dec coordinates of the corners (vertices) of
            all pixels in a brick. If ``brick`` is None, then a list of
            files for all bricks is returned.
        'mean_sfr_100myr'
            fits file of <SFR> over 100 Myr for a brick. If `brick` is None,
            then a list of files for all bricks is returned. Setting `brick` to
            'mosaic' will return the mosaic of all bricks.
        'phot', 'sfh', 'cmd', 'bestzcb'
            SFH-related files for a given pixel. These are best described by
            the procedure that created them (files in parentheses are not
            available)::

              # Calculate best fit SFH and run zcombine on the output
              calcsfh (par) phot (fake) sfh ... -mcdata  # other output: cmd, (hmcdat)
              zcombine sfh -bestonly > bestzcb

        'spec'
            Spectral evolution file from `scombine.generate_basis`. Assumes a
            present-day "lookback time" (`t_lookback=[0]`), and a Kroupa02 IMF.
            Metallicity is selected with `zmet` keyword.
        'mod_fuv_intrinsic', 'mod_fuv_attenuated'
            fits file of modeled FUV flux for a brick. If `brick` is None, then
            a list of files for all bricks is returned. Setting `brick` to
            'mosaic' will return the mosaic of all bricks.

        """
        if brick == 'mosaic':
            brick_list = [brick]
        else:
            if brick is None:
                brick_list = BRICK_LIST
            else:
                brick_list = [brick]
            brick_list = ['{:02d}'.format(brick) for brick in brick_list]

        if pixel is None:
            pixel_list = PIXEL_LIST
        else:
            pixel_list = [pixel]
        pixel_list = ['{:03d}'.format(pixel) for pixel in pixel_list]

        file_list = []

        if kind == 'vert':
            for brick in brick_list:
                if int(brick) not in BRICK_LIST:
                    continue
                dirname = os.path.join(SFH_DIR, 'B{:s}'.format(brick))
                filename = 'M31-B{:s}_15x30_subregion-exact-vertices.dat'.format(brick)
                file_list.append(os.path.join(dirname, filename))

        elif kind in ('mean_sfr_100myr', 'mod_fuv_intrinsic', 'mod_fuv_attenuated'):
            if brick_list[0] == 'mosaic':
                dirname = MAP_DIR
                filename = '{:s}.fits'.format(kind)
                file_list.append(os.path.join(dirname, filename))
            else:
                for brick in brick_list:
                    if int(brick) not in BRICK_LIST:
                        continue
                    dirname = os.path.join(MAP_DIR, 'b{:s}'.format(brick))
                    filename = 'b{0:s}_{1:s}.fits'.format(brick, kind)
                    file_list.append(os.path.join(dirname, filename))

        elif kind == 'sfh':
            for brick in brick_list:
                dirname = os.path.join(SFH_DIR, 'B{:s}'.format(brick), 'sfh')
                for pixel in pixel_list:
                    if (int(brick), int(pixel)) in MISSING:
                        continue
                    i, j = BIDX[int(brick)], PIDX[int(pixel)]
                    av, dav = '{:.1f}'.format(AV[i,j]), '{:.1f}'.format(DAV[i,j])
                    filename = ('M31-B{0:s}_15x30-{1:s}_{2:s}-{3:s}_best.sfh'
                                .format(brick, pixel, av, dav))
                    file_list.append(os.path.join(dirname, filename))

        elif kind == 'cmd':
            for brick in brick_list:
                dirname = os.path.join(SFH_DIR, 'B{:s}'.format(brick), 'cmd')
                for pixel in pixel_list:
                    if (int(brick), int(pixel)) in MISSING:
                        continue
                    i, j = BIDX[int(brick)], PIDX[int(pixel)]
                    av, dav = '{:.1f}'.format(AV[i,j]), '{:.1f}'.format(DAV[i,j])
                    filename = ('M31-B{0:s}_15x30-{1:s}_{2:s}-{3:s}_best.sfh.cmd'
                                .format(brick, pixel, av, dav))
                    file_list.append(os.path.join(dirname, filename))

        elif kind == 'phot':
            for brick in brick_list:
                dirname = os.path.join(SFH_DIR, 'B{:s}'.format(brick), 'phot')
                for pixel in pixel_list:
                    if (int(brick), int(pixel)) in MISSING:
                        continue
                    filename = 'M31-B{0:s}_15x30-{1:s}.gst.match'.format(brick, pixel)
                    file_list.append(os.path.join(dirname, filename))

        elif kind == 'bestzcb':
            for brick in brick_list:
                dirname = os.path.join(PROC_DIR, 'b{:s}'.format(brick), 'bestzcb')
                for pixel in pixel_list:
                    if (int(brick), int(pixel)) in MISSING:
                        continue
                    filename = 'b{0:s}-{1:s}_best.zcb'.format(brick, pixel)
                    file_list.append(os.path.join(dirname, filename))

        elif kind == 'spec':
            filename = 'spec_kroupa_z{:.1f}_tl0.0000Gyr.fits'.format(M_H)
            dirname = SPEC_DIR
            file_list.append(os.path.join(dirname, filename))


        if not file_list:
            # file_list is empty
            file_list = None
        elif len(file_list) == 1:
            # file_list contains just one string, so return only that string
            file_list = file_list[0]

        return file_list



    # M31 parameters
    # --------------
    DMOD = 24.47  # Distance modulus; McConnachie, A. W., Irwin, M. J., Ferguson, A. M. N., et al. 2005, MNRAS, 356, 979
    DIST_PC = 10**(DMOD/5. + 1)  # Distance in pc
    DIST_CM = DIST_PC * 3.08567758e18  # Distance in cm

    return None
