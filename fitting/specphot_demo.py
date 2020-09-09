#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" parametric_mock_specphot.py -  Parametric Spectrum and Photometry fit
This is a parameter file with build_* methods for fitting spectra and photometry
with a single composite stellar population (tau-model.) We optionally include
spectrum or photometry, and can remove the spectral continuum shape by
optimizing out a polynomial at each model call, if use_continuum is False
"""

import time, sys
from copy import deepcopy
import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

from prospect.sources.constants import cosmo

try:
    from exspect.utils import build_mock
    from exspect.utils import set_sdss_lsf, load_sdss
    from exspect.utils import fit_continuum, eline_mask
except(ImportError):
    pass


# Here we are going to put together some filter names
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
galex = ['galex_FUV', 'galex_NUV']
sdss = ['sdss_{0}0'.format(b) for b in 'ugriz']
twomass = ['twomass_{}'.format(b) for b in ['J', 'H', 'Ks']]
wise = ['wise_w{}'.format(b) for b in '1234']


# - Parser with default arguments -
parser = prospect_args.get_parser(["optimize", "dynesty"])
# - Add custom arguments -
parser.add_argument('--zred', type=float, default=0.1,
                    help="Redshift for the model (and mock).")
parser.add_argument('--zred_disp', type=float, default=1e-3,
                    help="Redshift prior dispersion. Use 0 for fixed redshift.")
# Fitted Model specification
parser.add_argument('--add_neb', action="store_true",
                    help="If set, add nebular emission in the model (and mock).")
parser.add_argument('--add_duste', action="store_true",
                    help="If set, dust emission in the model (and mock).")
parser.add_argument('--uniform_priors', action="store_true",
                    help="If set, use uniform priors for tau and mass.")
parser.add_argument("--free_neb_met", action="store_true",
                    help="If set, allow nebular metallicity != stellar metallicity")
parser.add_argument("--free_duste", action="store_true",
                    help="If set, let dust DL07 dust emission parameters vary")
# Mock data construction
parser.add_argument('--snr_spec', type=float, default=0,
                    help="S/N ratio for the mock spectroscopy.")
parser.add_argument('--snr_phot', type=float, default=20,
                    help="S/N ratio for the mock photometry.")
parser.add_argument('--filterset', type=str, nargs="*",
                    default=galex + sdss + twomass,
                    help="Names of filters through which to produce photometry.")
parser.add_argument('--add_noise', action="store_true",
                    help="If set, noise up the mock.")
parser.add_argument('--seed', type=int, default=101,
                    help=("RNG seed for the noise. Negative values result"
                          "in random noise."))
# Mock spectrum parameters
parser.add_argument('--wave_lo', type=float, default=3800.,
                    help="Minimum (restframe) wavelength for the mock spectrum")
parser.add_argument('--wave_hi', type=float, default=7200.,
                    help="Minimum (restframe) wavelength for the mock spectrum")
parser.add_argument('--dlambda_spec', type=float, default=2.0,
                    help="Minimum (restframe) wavelength for the mock spectrum")
parser.add_argument('--add_realism', action="store_true",
                    help="If set, Add realistic noise and instrumental dispersion.")
parser.add_argument('--sdss_filename', type=str, default="",
                    help="Full path to the SDSS spectral data file for adding realism.")
parser.add_argument('--mask_elines', action="store_true",
                    help="If set, mask windows around bright emission lines")
parser.add_argument('--continuum_optimize', action="store_true",
                    help="If set, optimize out the continuum shape.")
# Mock physical parameters
parser.add_argument('--tage', type=float, default=12.,
                    help="Age of the mock, Gyr.")
parser.add_argument('--tau', type=float, default=3.,
                    help="SFH timescale parameter of the mock, Gyr.")
parser.add_argument('--dust2', type=float, default=0.3,
                    help="Dust attenuation V band optical depth")
parser.add_argument('--logzsol', type=float, default=-0.5,
                    help="Metallicity of the mock; log(Z/Z_sun)")
parser.add_argument('--mass', type=float, default=1e10,
                    help="Stellar mass of the mock; solar masses formed")
parser.add_argument('--sigma_smooth', type=float, default=200.,
                    help="Velocity dispersion, km/s")


# --------------
# MODEL SETUP
# --------------


def build_model(uniform_priors=False, add_neb=True, add_duste=True,
                free_neb_met=False, free_duste=False, zred_disp=0,
                snr_spec=0, continuum_optimize=False,
                **kwargs):
    """Instantiate and return a ProspectorParams model subclass.

    :param add_neb: (optional, default: False)
        If True, turn on nebular emission and add relevant parameters to the
        model.
    """
    from prospect.models.templates import TemplateLibrary, describe
    from prospect.models import priors, sedmodel
    has_spectrum = np.any(snr_spec > 0)

    # --- Basic parameteric SFH ---
    model_params = TemplateLibrary["parametric_sfh"]

    # --- Nebular & dust emission ---
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params["gas_logu"]["isfree"] = True
        if free_neb_met:
            # Fit for independent gas metallicity
            model_params["gas_logz"]["isfree"] = True
            _ = model_params["gas_logz"].pop("depends_on")
    if add_duste | free_duste:
        model_params.update(TemplateLibrary["dust_emission"])
        if free_duste:
            # could also adjust priors here
            model_params["duste_qpah"]["isfree"] = True
            model_params["duste_umin"]["isfree"] = True
            model_params["duste_alpha"]["isfree"] = True

    # --- Complexify dust attenuation ---
    # Switch to Kriek and Conroy 2013
    model_params["dust_type"]["init"] = 4
    # Slope of the attenuation curve, expressed as the index of the power-law
    # that modifies the base Kriek & Conroy/Calzetti shape.
    # I.e. a value of zero is basically calzetti with a 2175AA bump
    model_params["dust_index"]  = dict(N=1, isfree=False, init=0.0)
    # young star dust
    model_params["dust1"]       = dict(N=1, isfree=False, init=0.0)
    model_params["dust1_index"] = dict(N=1, isfree=False, init=-1.0)
    model_params["dust_tesc"]   = dict(N=1, isfree=False, init=7.0)

    # --- Add smoothing parameters ---
    if has_spectrum:
        model_params.update(TemplateLibrary["spectral_smoothing"])
        model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=150, maxi=250)
        # --- Add spectroscopic calibration ---
        if continuum_optimize:
            model_params.update(TemplateLibrary["optimize_speccal"])
            # Could change the polynomial order here
            model_params["polyorder"]["init"] = 12

    # Alter parameter values based on keyword arguments
    for p in list(model_params.keys()):
        if (p in kwargs):
            model_params[p]["init"] = kwargs[p]

    # Now set redshift free and adjust prior
    z = np.copy(model_params['zred']["init"])
    if zred_disp > 0:
        model_params['zred']["isfree"] = True
        model_params['zred']['prior'] = priors.Normal(mean=z, sigma=zred_disp)

    # Alter some priors?
    if uniform_priors:
        minit = model_params["mass"]["init"]
        model_params["tau"]["prior"] = priors.TopHat(mini=0.1, maxi=10)
        model_params["mass"]["prior"] = priors.TopHat(mini=minit/10., maxi=minit*10)

    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.5, maxi=0.2)
    tuniv = cosmo.age(z).to("Gyr").value
    model_params["tage"]["prior"] = priors.TopHat(mini=0.1, maxi=tuniv)

    if has_spectrum & continuum_optimize:
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
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    if add_realism:
        set_sdss_lsf(sps.ssp, **extras)

    return sps


# --------------
# OBS
# --------------


def build_obs(dlambda_spec=2.0, wave_lo=3800, wave_hi=7000.,
              filterset=galex+sdss+twomass,
              snr_spec=10., snr_phot=20., add_noise=False, seed=101,
              add_realism=False, mask_elines=False,
              continuum_optimize=True, **kwargs):
    """Load a mock

    :param wave_lo:
        The (restframe) minimum wavelength of the spectrum.

    :param wave_hi:
        The (restframe) maximum wavelength of the spectrum.

    :param dlambda_spec:
        The (restframe) wavelength sampling or spacing of the spectrum.

    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.

    :param snr_spec:
        S/N ratio for the spectroscopy per pixel.  scalar.

    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.

    :param add_noise: (optional, boolean, default: True)
        Whether to add a noise reealization to the spectroscopy.

    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.

    :param add_realism:
        If set, add a realistic S/N and instrumental dispersion based on a
        given SDSS spectrum.

    :returns obs:
        Dictionary of observational data.
    """
    # --- Make the Mock ----
    # In this demo we'll make a mock.  But we need to know which wavelengths
    # and filters to mock up.
    has_spectrum = np.any(snr_spec > 0)
    if has_spectrum:
        a = 1 + kwargs.get("zred", 0.0)
        wavelength = np.arange(wave_lo, wave_hi, dlambda_spec) * a
    else:
        wavelength = None

    if np.all(snr_phot <= 0):
        filterset = None

    # We need the models to make a mock.
    sps = build_sps(add_realism=add_realism, **kwargs)
    model = build_model(conintuum_optimize=continuum_optimize,
                        mask_elines=mask_elines, **kwargs)

    # Make spec uncertainties realistic ?
    if has_spectrum & add_realism:
        # This uses an actual SDSS spectrum to get a realistic S/N curve,
        # renormalized to have median S/N given by the snr_spec parameter
        sdss_spec, _, _ = load_sdss(**kwargs)
        snr_profile = sdss_spec['flux'] * np.sqrt(sdss_spec['ivar'])
        good = np.isfinite(snr_profile)
        snr_vec = np.interp(wavelength, 10**sdss_spec['loglam'][good], snr_profile[good])
        snr_spec = snr_spec * snr_vec / np.median(snr_vec)

    mock = build_mock(sps, model, filterset=filterset, snr_phot=snr_phot,
                      wavelength=wavelength, snr_spec=snr_spec,
                      add_noise=add_noise, seed=seed)

    # continuum normalize ?
    if has_spectrum & continuum_optimize:
        # This fits a low order polynomial to the spectrum and then divides by
        # that to get a continuum normalized spectrum.
        cont, _ = fit_continuum(mock["wavelength"], mock["spectrum"], normorder=6, nreject=3)
        cont = cont / cont.mean()
        mock["spectrum"] /= cont
        mock["unc"] /= cont
        mock["continuum"] = cont

    # Spectroscopic Masking
    if has_spectrum & mask_elines:
        mock['mask'] = np.ones(len(mock['wavelength']), dtype=bool)
        a = (1 + model.params['zred'])  # redshift the mask
        # mask everything > L(Ha)/100
        lines = np.array([3727, 3730, 3799.0, 3836.5, 3870., 3890.2, 3970,  # OII + H + NeIII
                          4103., 4341.7, 4862.7, 4960.3, 5008.2,            # H[b,g,d]  + OIII
                          4472.7, 5877.2, 5890.0,           # HeI + NaD
                          6302.1, 6549.9, 6564.6, 6585.3,   # OI + NII + Halpha
                          6680.0, 6718.3, 6732.7, 7137.8])  # HeI + SII + ArIII
        mock['mask'] = mock['mask'] & eline_mask(mock['wavelength'], lines * a, 9.0 * a)

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
