from astropy.io import fits
import montage_wrapper as montage
import os


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


B15DIR = '/Users/Jake/Research/PHAT/sfhmaps/data/map/b15'
hdrfile = os.path.join(MAPDIR, '_mod_fuv_attenuated/template.hdr')

def reproject(tblargs, projargs):
    work_dir = os.path.join(B15DIR, 'test')
    input_dir = os.path.join(work_dir, 'input')
    proj_dir = os.path.join(work_dir, 'reproject')


    # Symlink the input image
    inputfile = os.path.join(B15DIR, 'b15_mod_fuv_attenuated.fits')
    filename = os.path.basename(inputfile)
    linkpath = os.path.join(input_dir, filename)
    try:
        os.symlink(inputfile, linkpath)
    except OSError:
        pass

    # Metadata for input image
    metafile = os.path.join(work_dir, 'input.tbl')
    montage.mImgtbl(input_dir, metafile, **tblargs)

    # Reproject to the template
    statfile = os.path.join(work_dir, 'stats.tbl')
    montage.mProjExec(metafile, hdrfile, proj_dir, statfile, raw_dir=input_dir, **projargs)

    # Compare
    filename = os.path.join(MAPDIR, 'b15',  'b15_mod_fuv_attenuated.fits')
    data1 = fits.getdata(filename)

    filename = os.path.join(proj_dir,  'hdu0_b15_mod_fuv_attenuated.fits')
    data2 = fits.getdata(filename)

    sum1, sum2 = np.nansum(data1), np.nansum(data2)
    fdiff = (sum2 - sum1) / sum1

    return fdiff


"""
flux vs. flux density? Try multiplying by the ratio of pixel areas.

"""
