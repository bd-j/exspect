#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""globular_cluster.py - SSP spectrum & photometry fitting

Fit GGC spectra (Schiavon 2005) and photometry with an SSP.

We remove the spectral continuum shape by optimizing out a polynomial
at each model call, if continuum_optimize is True
"""

import time, sys
from copy import deepcopy
import numpy as np

from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.sources.constants import cosmo

from .utils import set_ggc_lsf
from .utils import fit_continuum, eline_mask


# Here we are going to put together some filter names
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
galex = ['galex_FUV', 'galex_NUV']
sdss = ['sdss_{0}0'.format(b) for b in ['g','r','i','z']]
twomass = ['twomass_{}'.format(b) for b in ['J', 'H', 'Ks']]
bessell = ['bessell_{}'.format(b) for b in "UBVRI"]


# - Parser with default arguments -
parser = prospect_args.get_parser(["optimize", "dynesty"])
# Initial parameters
parser.add_argument('--zred', type=float, default=0.0,
                    help="Redshift for the model.")
parser.add_argument('--zred_disp', type=float, default=1e-3,
                    help="Redshift for the model (and mock).")
# Fitted Model specification
parser.add_argument('--add_neb', action="store_true",
                    help="If set, add nebular emission in the model (and mock).")
parser.add_argument('--add_realism', action="store_true",
                    help="If set, Add realistic instrumental dispersion.")
parser.add_argument('--continuum_order', type=int, default=0,
                    help="If > 0, optimize out the continuum shape.")
parser.add_argument('--outlier_model', action="store_true",
                    help="If set, mask windows around bright emission lines")
# Data construction
parser.add_argument('--ggc_data', type=str, default="data/ggc.h5",
                    help="Full path of GGC data HDF5 file")
parser.add_argument('--ggc_id', type=str, default="NGC104",
                    help="Name of the GGC object.")
parser.add_argument('--ggc_index', type=int, default=-1,
                    help="Index of the GGC object. Overrides ggc_id if >0")
parser.add_argument('--wave_lo', type=float, default=3800.,
                    help="Minimum wavelength to fit")
parser.add_argument('--wave_hi', type=float, default=6250.,
                    help="Maximum wavelength to fit")
parser.add_argument('--filterset', type=str, nargs="*",
                    default=[],
                    help="Names of filters through which to produce photometry.")
parser.add_argument('--snr_phot', type=float, default=20,
                    help="S/N ratio for the photometry.")
parser.add_argument('--mask_elines', action="store_true",
                    help="If set, mask windows around bright emission lines")

# --------------
# MODEL SETUP
# --------------


def build_model(continuum_order=0, add_neb=False, zred=0., zred_disp=1e-3, **kwargs):
    """Instantiate and return a ProspectorParams model subclass.
    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Basic SSP at fixed distance (10pc) ---
    model_params = TemplateLibrary["ssp"]
    model_params["lumdist"] = {"N": 1, "isfree": False, "init": 0.01}
    model_params["tage"]["init"] = 10.0

    # --- Complexify dust attenuation ---
    # Cardelli
    model_params["dust_type"]["init"] = 1
    # Slope of the attenuation curve, expressed as R_v
    model_params["mwr"] = dict(N=1, isfree=False, init=3.1)

    # --- Add smoothing parameters ---
    model_params.update(TemplateLibrary["spectral_smoothing"])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=0, maxi=50)
    model_params["sigma_smooth"]["init"] = 10.0
    # --- Add spectral fitting parameters ---
    if continuum_order > 0:
        model_params.update(TemplateLibrary["optimize_speccal"])
        model_params["polyorder"]["init"] = continuum_order

    # redshift
    model_params['zred']["isfree"] = True
    model_params['zred']["init"] = zred
    model_params['zred']['prior'] = priors.Normal(mean=zred, sigma=zred_disp)

    # Alter parameter values based on keyword arguments
    for p in list(model_params.keys()):
        if (p in kwargs):
            model_params[p]["init"] = kwargs[p]

    # Alter some priors?
    model_params["dust2"]["prior"].params["maxi"] = 2.5
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2, maxi=0.2)
    model_params["tage"]["prior"] = priors.TopHat(mini=1, maxi=13.8)

    if continuum_order > 0:
        return sedmodel.PolySpecModel(model_params)
    else:
        return sedmodel.SpecModel(model_params)

# --------------
# SPS Object
# --------------


def build_sps(zcontinuous=1, compute_vega_mags=False, add_realism=False, **extras):
    """Load the SPS object.  If add_realism is True, set up to convolve the
    library spectra to an sdss resolution
    """
    from prospect.sources import FastSSPBasis
    sps = FastSSPBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    if add_realism:
        set_ggc_lsf(sps.ssp, **extras)

    return sps

# -----------------
# Noise Model
# ------------------


def build_noise(**extras):
    return None, None


# -----------------
# Helper Functions
# ------------------


def normalize_ggc_spec(obs, norm_band="bessell_B"):
    """Normalize the spectrum to a photometric band
    """
    from sedpy.observate import getSED
    from prospect.sources.constants import lightspeed, jansky_cgs

    bands = list([f.name for f in obs['filters']])
    norm_index = bands.index(norm_band)

    synphot = getSED(obs['wavelength'], obs['spectrum'], obs['filters'])
    synphot = np.atleast_1d(synphot)
    # Factor by which the observed spectra should be *divided* to give you the
    #  photometry (or the cgs apparent spectrum), using the given filter as
    #  truth.  Alternatively, the factor by which the model spectrum (in cgs
    #  apparent) should be multiplied to give you the observed spectrum.
    norm = 10**(-0.4 * synphot[norm_index]) / obs['maggies'][norm_index]
    wave = obs["wavelength"]
    flambda_to_maggies = wave * (wave/lightspeed) / jansky_cgs / 3631
    flambda_cgs = obs["spectrum"] / norm
    maggies = flambda_cgs * flambda_to_maggies
    obs["spectrum"] = maggies
    obs["unc"] = obs["unc"] / norm * flambda_to_maggies
    if "sky" in obs:
        obs["sky"] = obs["sky"] / norm * flambda_to_maggies

    obs["norm_band"] = norm_band
    return obs


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
