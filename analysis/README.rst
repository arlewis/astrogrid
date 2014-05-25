Project is documented here.

SFH data
========

Starting with data from Alexia:

- For each brick:

  - list of RA,dec coordinates of the corners of all pixels in the brick
    (`vert`)
  - list of best-fit Av,dAv extinction parameters of all pixels in the brick
    (`extpar`)

  - for each pixel in the brick:

    - F475W,F814W .gst photometry (`phot`), and the SFH and CMD output files
      from calcsfh (`sfh` and `cmd`)


Process raw calcsfh data
========================

Used zcombine on the `sfh` files to create tables of SFR vs age (`zcb`)

(Process HMC results?)


