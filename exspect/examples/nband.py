#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""nband.py - Non-Parametric Photometry fit
This is a parameter file for fitting photometry with a full model including
'non-parameteric' (continuity) SFH and AGN. The filter set can be easily
changed.
"""

import os, sys, time
from copy import deepcopy
import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from .utils import build_mock

# -------------
# FILTERS
# ------------

# Filter names
galex = ['galex_FUV', 'galex_NUV']
sdss = ['sdss_{0}0'.format(b) for b in "ugriz"]
twomass = ['twomass_{}'.format(b) for b in ['J', 'H', 'Ks']]
wise = ['wise_w{}'.format(b) for b in "1234"]
pacs = ["herschel_pacs_{}".format(b) for b in ["70", "100", "160"]]
spire = ["herschel_spire_{}".format(b) for b in ["250", "250", "500"]]

# Filter Sets
filtersets = {"oneband": sdss[2:3],  # r
              "twoband": sdss[1:3],  # g, r
              "optical": sdss,       # ugriz
              "opt_nir": sdss + twomass,
              "uv_opt": galex + sdss,
              "uv_to_nir": galex + sdss + twomass,
              "uv_to_mir": galex + sdss + twomass + wise,
              "full": galex + sdss + twomass + wise + pacs
              }


# - Parser with default arguments -
parser = prospect_args.get_parser(["optimize", "dynesty"])
# Fitted Model specification
parser.add_argument('--parametric_sfh', action="store_true",
                    help="If set, fit a delay-tau model")
parser.add_argument('--nbins_sfh', type=int, default=6,
                    help="Number of bins in the SFH")
parser.add_argument('--add_neb', action="store_true",
                    help="If set, add nebular emission in the model (and mock).")
parser.add_argument('--add_duste', action="store_true",
                    help="If set, dust emission in the model (and mock).")
parser.add_argument('--free_neb_met', action="store_true",
                    help="If set, use a nebular metallicity untied to the stellar Z")
parser.add_argument("--free_duste", action="store_true",
                    help="If set, let dust DL07 dust emission parameters and the AGN parameters vary")
parser.add_argument('--complex_dust', action="store_true",
                    help="If set, let attenuation curve slope and young star dust vary")
# Mock data construction
parser.add_argument('--snr_phot', type=float, default=20,
                    help="S/N ratio for the mock photometry.")
parser.add_argument('--filterset', type=str, default="optical",
                    help="Names of the filterset through which to produce photometry.")
parser.add_argument('--add_noise', action="store_true",
                    help="If set, noise up the mock.")
parser.add_argument('--seed', type=int, default=101,
                    help=("RNG seed for the noise. Negative values result"
                          "in random noise."))
# Mock physical parameters
parser.add_argument('--zred', type=float, default=0.1,
                    help="Redshift for the model (and mock).")
parser.add_argument('--logmass', type=float, default=10,
                    help="Stellar mass of the mock; solar masses formed")
parser.add_argument('--dust2', type=float, default=0.3,
                    help="Dust attenuation V band optical depth")
parser.add_argument('--logzsol', type=float, default=-0.3,
                    help="Metallicity of the mock; log(Z/Z_sun)")
parser.add_argument('--tage', type=float, default=12.,
                    help="Age of the mock, Gyr.")
parser.add_argument('--tau', type=float, default=3.,
                    help="SFH timescale parameter of the mock, Gyr.")
parser.add_argument('--fagn', type=float, default=0.05,
                    help="Dust attenuation V band optical depth")
parser.add_argument('--agn_tau', type=float, default=20,
                    help="AGN torus optical depth")
parser.add_argument('--duste_umin', type=float, default=2,
                    help="Dust heating intensity")
parser.add_argument('--duste_qpah', type=float, default=1.,
                    help="Dust heating intensity")


# --------------
# MODEL
# --------------


def build_model(add_neb=True, add_duste=True, complex_dust=True,
                free_neb_met=False, free_duste=True,
                parametric_sfh=False, nbins_sfh=10, tuniv=13.7, **kwargs):
    """Load the model object.
    """

    from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins
    from prospect.models import priors, sedmodel
    from prospect.models.transforms import dustratio_to_dust1

    # --- Basic + SFH ----
    if parametric_sfh:
        model_params = TemplateLibrary["parametric_sfh"]
    else:
        model_params = TemplateLibrary["continuity_sfh"]
        model_params = adjust_continuity_agebins(model_params, nbins=nbins_sfh, tuniv=tuniv)

    # --- Nebular & dust emission ---
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params["gas_logu"]["isfree"] = True
        if free_neb_met:
            model_params["gas_logz"]["isfree"] = True
            _ = model_params["gas_logz"].pop("depends_on")
    if add_duste:
        model_params.update(TemplateLibrary["dust_emission"])
        model_params.update(TemplateLibrary["agn"])
        if free_duste:
            model_params["duste_qpah"]["isfree"] = True
            model_params["duste_umin"]["isfree"] = True
            model_params["duste_gamma"]["isfree"] = True
            model_params["fagn"]["isfree"] = True
            model_params["agn_tau"]["isfree"] = True

    # --- Complexify Dust attenuation ---
    # Switch to Kriek and Conroy 2013
    model_params["dust_type"]["init"] = 4
    # Center screen Av prior broadly on 0.3 +/- 1.0
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
    # Slope of the attenuation curve, as delta from Calzetti
    model_params["dust_index"]  = dict(N=1, isfree=False, init=0.0,
                                       prior=priors.ClippedNormal(mini=-1, maxi=0.4, mean=0, sigma=0.5))
    # Young star dust, as a ratio to old star dust
    model_params["dust_ratio"]  = dict(N=1, isfree=False, init=1,
                                       prior=priors.ClippedNormal(mini=0, maxi=1.5, mean=1.0, sigma=0.3))
    model_params["dust1"]       = dict(N=1, isfree=False, init=0.0, depends_on=dustratio_to_dust1)
    model_params["dust1_index"] = dict(N=1, isfree=False, init=-1.0)
    model_params["dust_tesc"]   = dict(N=1, isfree=False, init=7.0)
    if complex_dust:
        model_params["dust_index"]["isfree"] = True
        model_params["dust_ratio"]["isfree"] = True

    # Alter parameter values based on keyword arguments
    for p in list(model_params.keys()):
        if (p in kwargs):
            model_params[p]["init"] = kwargs[p]

    # Alter some priors?
    minit = model_params["logmass"]["init"]
    model_params["logmass"]["prior"].params["maxi"] = minit + 1.5
    model_params["logmass"]["prior"].params["mini"] = minit - 1.5
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2, maxi=0.2)

    return sedmodel.SpecModel(model_params)

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, parametric_sfh=False, **extras):
    from prospect.sources import FastStepBasis, CSPSpecBasis
    if parametric_sfh:
        sps = CSPSpecBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    else:
        sps = FastStepBasis(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_mags)
    return sps


# --------------
# OBS
# --------------

def build_obs(filterset="optical", snr_phot=10., add_noise=False, seed=101,
              **kwargs):
    """Build a mock observation

    :param filterset:
        Name of a filterset that is a key of the `filtersets` dictionary.
    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.
    :param add_noise: (optional, boolean, default: True)
        Whether to add a noise reealization to the spectroscopy.
    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.
    :returns obs:
        Dictionary of observational data.
    """
    # choose the set of filters
    filters = filtersets[filterset]

    # We need the models to make a mock.
    sps = build_sps(**kwargs)
    model = build_model(**kwargs)

    # --- Make the Mock ----
    mock = build_mock(sps, model, filterset=filters, snr_phot=snr_phot,
                      add_noise=add_noise, seed=seed)

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
