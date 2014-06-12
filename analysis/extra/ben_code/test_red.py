import numpy as np
import matplotlib.pyplot as pl

import attenuation
from bursty_sfh import redden_analytic, redden_pieces
import fsps

av, dav, tage = 0.5, 1.0, 0.20

sps = fsps.StellarPopulation()
sps.params['sfh']= 0
wave, spec = sps.get_spectrum(peraa =True, tage = tage)



ra, _ = redden_analytic(wave, spec, av = av, dav = dav,
                        dust_curve = attenuation.cardelli,
                        wlo = 1216., whi = 2e4)

rp, _ = redden_pieces(wave, spec, av = av, dav = dav, nsplit = 50,
                      dust_curve = attenuation.cardelli,
                      wlo = 1216., whi = 2e4)

pl.figure()
pl.plot(wave, ra, label = 'analytic')
pl.plot(wave, rp, label = 'numerical')
pl.plot(wave, spec, label = 'intrinsic')
pl.xlim(1e3,1e4)
pl.ylabel(r'$F_\lambda$')
pl.title('age = {0:4.0f}Myr, Av = {1:3.1}, dAv = {2:3.1f}'.format(tage*1e3, av, dav))
pl.legend()
pl.show()
