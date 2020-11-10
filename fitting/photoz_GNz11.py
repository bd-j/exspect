#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" photoz.py - Photomteric redshift fitting
Fit for redshift using only photometry of GNz-11
"""

import time, sys
from copy import deepcopy
import numpy as np

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from prospect.sources.constants import cosmo

from sedpy.observate import load_filters


# - Parser with default arguments -
parser = prospect_args.get_parser(["optimize", "dynesty"])
# - Add custom arguments -
# Fitted Model specification
parser.add_argument('--nbins_sfh', type=int, default=5,
                    help="Number of bins in the SFH")
parser.add_argument('--add_neb', action="store_true",
                    help="If set, add nebular emission in the model (and mock).")
parser.add_argument('--add_duste', action="store_true",
                    help="If set, dust emission in the model (and mock).")
parser.add_argument('--free_neb_met', action="store_true",
                    help="If set, use a nebular metallicity untied to the stellar Z")
parser.add_argument('--free_igm', action="store_true",
                    help="If set, allow for the IGM attenuation to vary")
parser.add_argument('--complex_dust', action="store_true",
                    help="If set, let attenuation curve slope and young star dust vary")
parser.add_argument('--zmax', type=float, default=40.,
                    help="Maximum redshift for SF to occur")
parser.add_argument('--zmean', type=float, default=-1,
                    help="mean of redshift prior; use uniform prior if negative")
parser.add_argument('--zdisp', type=float, default=1.,
                    help="dispersion of redshift prior")

# --------------
# SPS Object
# --------------


def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import FastStepBasis, CSPSpecBasis
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)
    return sps

# --------------
# MODEL SETUP
# --------------


def build_model(zmean=-1, zdisp=None, zmax=20, nbins_sfh=6,
                add_neb=True, free_neb_met=True, add_duste=True, free_igm=True,
                complex_dust=False, **kwargs):

    from prospect.models.templates import TemplateLibrary, adjust_continuity_agebins
    from prospect.models.transforms import dustratio_to_dust1
    from prospect.models import priors, sedmodel

    # --- Basic non-parametric SFH parameter set ---
    model_params = TemplateLibrary["continuity_sfh"]

    # adjust number of bins for SFH
    model_params = adjust_continuity_agebins(model_params, nbins=nbins_sfh)
    model_params["zmax"] = dict(N=1, isfree=False, init=zmax)
    model_params["nbins_sfh"] = dict(N=1, isfree=False, init=nbins_sfh)

    # add redshift scaling to agebins, such that there is one 0-10 Myr bin and the
    # rest are evenly spaced in log(age) up to the age of the universe at that redshift
    model_params["agebins"]["depends_on"] = zred_to_agebins
    model_params["mass"]["depends_on"] = zlogsfr_ratios_to_masses

    # --- We *are* fitting for redshift ---
    model_params["zred"]["isfree"] = True
    if zmean > 0 :
        assert zdisp is not None
        model_params["zred"]["prior"] = priors.Normal(mean=zmean, sigma=zdisp)
    else:
        model_params["zred"]["prior"] = priors.TopHat(mini=1, maxi=13)

    # --- Complexify Dust attenuation ---
    # Switch to Kriek and Conroy 2013
    model_params["dust_type"]["init"] = 4
    # Center screen Av prior broadly on 0.3 +/- 1.0
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
    # Slope of the attenuation curve, as delta from Calzetti
    model_params["dust_index"]  = dict(N=1, isfree=False, init=0.0,
                                       prior=priors.ClippedNormal(mini=-1, maxi=0.4, mean=0, sigma=0.5))
    # Young star dust, as a ratio to old star dust
    model_params["dust_ratio"]  = dict(N=1, isfree=False, init=0,
                                       prior=priors.ClippedNormal(mini=0, maxi=1.5, mean=1.0, sigma=0.3))
    model_params["dust1"]       = dict(N=1, isfree=False, init=0.0, depends_on=dustratio_to_dust1)
    model_params["dust1_index"] = dict(N=1, isfree=False, init=-1.0)
    model_params["dust_tesc"]   = dict(N=1, isfree=False, init=7.0)
    if complex_dust:
        model_params["dust_index"]["isfree"] = True
        model_params["dust_ratio"]["isfree"] = True

    # --- IGM, nebular and dust emission ---
    model_params.update(TemplateLibrary["igm"])
    if free_igm:
        # Allow IGM transmission scaling to vary
        model_params["igm_factor"]['isfree'] = True
        model_params["igm_factor"]["prior"] = priors.ClippedNormal(mean=1.0, sigma=0.3, mini=0.0, maxi=2.0)
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params["gas_logu"]["isfree"] = True
        if free_neb_met:
            # Fit for independent gas metallicity
            model_params["gas_logz"]["isfree"] = True
            _ = model_params["gas_logz"].pop("depends_on")
    if add_duste:
        # Add dust emission (with fixed dust SED parameters)
        model_params.update(TemplateLibrary["dust_emission"])

    return sedmodel.SpecModel(model_params)


def zred_to_agebins(zred=None, nbins_sfh=5, zmax=20.0, **extras):
    """Construct `nbins_sfh` bins in lookback time from 0 to age(zmax).  The
    first bin goes from 0-10 Myr, the rest are evenly spaced in log time
    """
    tuniv = cosmo.age(zred).value*1e9
    tbinmax = tuniv-cosmo.age(zmax).value*1e9
    agelims = np.append(np.array([0.0, 7.0]), np.linspace(7.0, np.log10(tbinmax), int(nbins_sfh))[1:])
    agebins = np.array([agelims[:-1], agelims[1:]])
    return agebins.T


def zlogsfr_ratios_to_masses(logmass=None, logsfr_ratios=None, zred=None,
                             **extras):
    """This converts from an array of log_10(SFR_j / SFR_{j+1}) and a value of
    log10(\Sum_i M_i) to values of M_i.  j=0 is the most recent bin in lookback
    time; it incorporates changes in the agebins due to changing redshift.
    """
    agebins = zred_to_agebins(zred, **extras)
    nbins = agebins.shape[0]
    sratios = 10**np.clip(logsfr_ratios, -100, 100)  # numerical issues...
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    coeffs = np.array([ (1. / np.prod(sratios[:i])) * (np.prod(dt[1: i+1]) / np.prod(dt[: i]))
                        for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()

    return m1 * coeffs

# -----------------
# Noise Model
# ------------------


def build_noise(**extras):
    return None, None

# --------------
# OBS
# --------------


# Here we are going to put together some filter names
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
spitzer = ['spitzer_irac_ch'+n for n in "1234"]
twomass = ['twomass_{}'.format(b) for b in ['J', 'H', 'Ks']]
acs = ['acs_wfc_{}'.format(b) for b in ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']]
wfc3ir = ['wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']


def build_obs(**kwargs):
    """Load GNz-11 photometry from Oesch 16
    """
    filterset = acs + wfc3ir + twomass[-1:] + spitzer[:2]

    obs = {}
    obs['wavelength'] = None  # No spectrum
    obs['filters'] = load_filters(filterset)

    # From Oesch 16
    obs['maggies'] = 1e-9/3631 * np.array([7., 2., 5., 3., 17., -7, 11., 64., 152., 137., 139., 144.])
    obs['maggies_unc'] = 1e-9/3631 * np.array([9., 7., 10., 7., 11., 9., 8., 13., 10., 67., 21., 27])
    obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])

    return obs


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
