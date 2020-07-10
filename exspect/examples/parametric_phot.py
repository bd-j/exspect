#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" ---- Parametric Photometry fit ----
This is a parameter file for fitting photometry only (no spectra) with a
single composite stellar population (tau-model).
 --------------------------------------
"""

import time, sys
from copy import deepcopy
import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from .utils import build_mock

# --------------
# MODEL SETUP
# --------------


def build_model(uniform_priors=False, add_neb=True, add_duste=True, **kwargs):
    """Instantiate and return a ProspectorParams model subclass.

    This model is a parametric SFH with KC13 dust attenuation

    :param add_neb: (optional, default: False)
        If True, turn on nebular emission and add relevant parameters to the
        model.
    """
    from prospect.models.templates import TemplateLibrary, describe
    from prospect.models import priors, sedmodel

    # Basic parameteric SFH with nebular & dust emission
    model_params = TemplateLibrary["parametric_sfh"]
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        #model_params["gas_logu"]["isfree"] = True
    if add_duste:
        model_params.update(TemplateLibrary["dust_emission"])

    # --- Complexify dust attenuation ---
    # Switch to Kriek and Conroy 2013
    model_params["dust_type"]["init"] = 4
    # Slope of the attenuation curve, expressed as the index of the power-law
    # that modifies the base Kriek & Conroy/Calzetti shape.
    # I.e. a value of zero is basically calzetti with a 2175AA bump
    model_params["dust_index"]  = {"N": 1, "isfree": False, "init": 0.0}
    # young star dust
    model_params["dust1"]       = {"N": 1, "isfree": False, "init": 0.0}
    model_params["dust1_index"] = {"N": 1, "isfree": False, "init": -1.0}
    model_params["dust_tesc"]   = {"N": 1, "isfree": False, "init": 7.0}

    # Alter parameter values based on keyword arguments
    for p in list(model_params.keys()):
        if (p in kwargs):
            model_params[p]["init"] = kwargs[p]

    # Alter some priors?
    minit = model_params["mass"]["init"]
    if uniform_priors:
        model_params["tau"]["prior"] = priors.TopHat(mini=0.1, maxi=10)
        model_params["mass"]["prior"] = priors.TopHat(mini=minit/3., maxi=minit*3)
    else:
        model_params["mass"]["prior"].params["maxi"] = minit * 10
        model_params["mass"]["prior"].params["mini"] = minit / 10

    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1, maxi=0.2)
    model_params["tage"]["prior"] = priors.TopHat(mini=0.1, maxi=13.8)

    return sedmodel.SedModel(model_params)

# --------------
# SPS Object
# --------------


def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps


# --------------
# Observational data
# --------------

# Here we are going to put together some filter names
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
galex = ['galex_FUV', 'galex_NUV']
sdss = ['sdss_{0}0'.format(b) for b in 'ugriz']
twomass = ['twomass_{}'.format(b) for b in ['J', 'H', 'Ks']]
wise = ['wise_w{}'.format(b) for b in '1234']


def build_obs(filterset=galex + sdss + twomass,
              snr_phot=10.0, add_noise=False, seed=101, **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.

    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.

    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.

    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock photometry.

    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.
    """
    # We need the models to make a mock
    sps = build_sps(**kwargs)
    model = build_model(**kwargs)
    mock = build_mock(sps, model, filterset=filterset, snr_phot=snr_phot,
                      add_noise=add_noise, seed=seed, *kwargs)

    return mock


# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------


def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__ == "__main__":

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--zred', type=float, default=0.1,
                        help="Redshift for the model (and mock).")

    # Fitted Model specification
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, dust emission in the model (and mock).")
    parser.add_argument('--uniform_priors', action="store_true",
                        help="If set, use uniform priors for tau and mass.")

    # Mock data construction
    parser.add_argument('--filterset', type=str, nargs="*",
                        default=galex + sdss + twomass,
                        help="Names of filters through which to produce photometry.")
    parser.add_argument('--snr_phot', type=float, default=20,
                        help="S/N ratio for the mock photometry.")
    parser.add_argument('--add_noise', action="store_true",
                        help="If set, noise up the mock.")
    parser.add_argument('--seed', type=int, default=101,
                        help=("RNG seed for the noise. Negative values result"
                              "in random noise."))

    # Mock physical parameters
    parser.add_argument('--tage', type=float, default=12.,
                        help="Age of the mock, Gyr.")
    parser.add_argument('--tau', type=float, default=3.,
                        help="SFH timescale parameter of the mock, Gyr.")
    parser.add_argument('--dust2', type=float, default=0.3,
                        help="Dust attenuation V band optical depth of the mock.")
    parser.add_argument('--logzsol', type=float, default=-0.5,
                        help="Metallicity of the mock; log(Z/Z_sun)")
    parser.add_argument('--mass', type=float, default=1e10,
                        help="Stellar mass of the mock; solar masses formed")

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["param_file"] = __file__
    run_params["sps_libraries"] = sps.ssp.libraries

    print(model)

    if args.debug:
        sys.exit()

    #hfile = setup_h5(model=model, obs=obs, **run_params)
    hfile = "{0}_{1}_mcmc.h5".format(args.outfile, int(time.time()))
    output = fit_model(obs, model, sps, noise, **run_params)

    print("writing to {}".format(hfile))
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
