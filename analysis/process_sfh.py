"""
Process calcsfh output files using zcombine.

"""
import os

import match-wrapper as match
from sfhmaps import config, util


def make_zcb(norun=False):
    """Create zcombine files for the best SFH solutions only, no errors."""
    sfhfile_list = config.path('sfh', brick=brick)
    zcbfile_list = config.path('zcb', brick=brick)

    for sfhfile, zcbfile in zip(sfhfile_list, zcbfile_list):
        dirname = os.path.dirname(zcbfile)
        util.safe_mkdir(dirname)
        match.zcombine(sfhfile, zcbfile, bestonly=True, norun=norun)


def main():
    make_bestzcb()


if __name__ == "__main__":
    main()
