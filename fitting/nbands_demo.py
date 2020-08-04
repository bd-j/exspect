#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time, sys
import numpy as np
import matplotlib.pyplot as pl

from sedpy.observate import load_filters
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from exspect.examples.nband import parser
from exspect.examples.nband import build_obs, build_model, build_sps, build_noise


# Here we are going to put together some filter names
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
spitzer = ['spitzer_irac_ch'+n for n in "1234"]
twomass = ['twomass_{}'.format(b) for b in ['J', 'H', 'Ks']]
acs = ['acs_wfc_{}'.format(b) for b in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']]
wfc3ir = ['wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']


def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__ == "__main__":

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["param_file"] = __file__
    run_params["sps_libraries"] = sps.ssp.libraries

    print(model)

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = "{0}_{1}_result.h5".format(args.outfile, ts)

    output = fit_model(obs, model, sps, noise, **run_params)

    print("writing to {}".format(hfile))
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      sps=sps,
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
