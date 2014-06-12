import sys, os, glob
import numpy as np
import fsps
import sfhutils as utils
import bursty_sfh as bsp
import attenuation, observate


sps = fsps.StellarPopulation()
sps.params['sfh'] = 0
sps.params['zmet'] = 4

filters = ['galex_FUV']
filterlist = observate.load_filters(filters)
maggies_to_cgs = 10**(-0.4*(2.406 + 5*np.log10([f.wave_effective for f in filterlist])))
dm =24.47

files = glob.glob('sfh/*zcb')
out = open('data/b15_predicted.dat' ,'w')
out.write('F_lambda,fuv  F_lambda,fuv,int  LIR(L_sun)  av  dav\n')

for ipix, filen in enumerate(files):
    sfh = utils.load_angst_sfh(filen)
    #pull out av, dav from filenames
    f = open(filen, 'r')
    av, dav = f.readline().split('/')[-1].split('_')[-2].split('-')
    av, dav = float(av), float(dav)
    f.close()
    
    sfh['t1'] = 10.**sfh['t1']
    sfh['t2'] = 10.**sfh['t2']
    sfh['sfr'][0] *=  1 - (sfh['t1'][0]/sfh['t2'][0])
    sfh[0]['t1'] = 0.
    mtot = ((sfh['t2'] - sfh['t1']) * sfh['sfr']).sum()

    lt, sfr, fb = bsp.burst_sfh(fwhm_burst = 0.05, f_burst = 0., contrast = 1.,
                                sfh = sfh, bin_res = 20.)

    t_lookback = 1e5
    #dav = 0
    wave, spec, aw, lirp = bsp.bursty_sps(t_lookback, lt, sfr, sps,
                                          av = av, dav = dav, nsplit = 30)
    mags = np.atleast_1d(observate.getSED(wave, spec * bsp.to_cgs, filterlist = filterlist))
    wave, spec, aw = bsp.bursty_sps(t_lookback, lt, sfr, sps, av = None, dav = None)
    mags_int = np.atleast_1d(observate.getSED(wave, spec * bsp.to_cgs, filterlist = filterlist))

    fl, flint = maggies_to_cgs * 10**(-0.4 *(mags + dm)), maggies_to_cgs * 10**(-0.4 *(mags_int + dm))
    
    out.write('{0:6.3f} {1:6.3f} {2:6.3e} {3:6.3f} {4:6.3f}\n'.format(fl, flint, lirp[0], av, dav))

out.close()
