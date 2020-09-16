# ---- Non-Parametric Spectrum fit --------
# This is a parameter file for fitting spectra only (no photometry) with a
# nonparameteric (continuity) SFH.
# We also optionally remove the continuum shape by optimizing out a polynomial
# at each model call, if continuum_order > 0
# -------------------------------------

import os, sys, time
from copy import deepcopy
import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer

try:
    from exspect.utils import build_mock
    from exspect.utils import set_sdss_lsf, load_sdss
    from exspect.utils import fit_continuum, eline_mask
except(ImportError):
    pass

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
parser.add_argument('--continuum_order', type=int, default=0,
                    help="If set, optimize out the continuum shape.")
# Mock data construction
parser.add_argument('--illustris_sfh_file', type=str, default="",
                    help="File with the Illustris SFH")
parser.add_argument('--snr_phot', type=float, default=0,
                    help="S/N ratio for the mock photometry.")
parser.add_argument('--snr_spec', type=float, default=100,
                    help="S/N ratio for the mock photometry.")
parser.add_argument('--filterset', type=str, nargs="*",
                    default=[],
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
# Mock physical parameters
parser.add_argument('--zred', type=float, default=0.01,
                    help="Redshift for the model (and mock).")
parser.add_argument('--logmass', type=float, default=10,
                    help="Stellar mass of the mock; solar masses formed")
parser.add_argument('--logmass', type=float, default=1e10,
                    help="Stellar mass of the mock; solar masses formed")
parser.add_argument('--dust2', type=float, default=0.3,
                    help="Dust attenuation V band optical depth")
parser.add_argument('--logzsol', type=float, default=-0.3,
                    help="Metallicity of the mock; log(Z/Z_sun)")
parser.add_argument('--duste_umin', type=float, default=2,
                    help="Dust heating intensity")
parser.add_argument('--duste_qpah', type=float, default=1.,
                    help="Dust heating intensity")


# --------------
# MODEL
# --------------

def build_model(parametric_sfh=False, nbins_sfh=14, tuniv=13.7,
                snr_spec=0, continuum_order=0, zred_disp=1e-4,
                add_neb=False, free_neb_met=False,
                add_duste=False, free_duste=False, complex_dust=False,
                **kwargs):
    """Load the model object.
    """
    has_spectrum = np.any(snr_spec > 0)

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
        if free_duste:
            model_params["duste_qpah"]["isfree"] = True
            model_params["duste_umin"]["isfree"] = True
            model_params["duste_gamma"]["isfree"] = True

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

    # --- Add smoothing parameters ---
    if has_spectrum:
        model_params.update(TemplateLibrary["spectral_smoothing"])
        model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=150, maxi=250)
        model_params["sigma_smooth"]["init"] = 150.
        # --- Add spectroscopic calibration ---
        if continuum_order > 0:
            model_params.update(TemplateLibrary["optimize_speccal"])
            # Could change the polynomial order here
            model_params["polyorder"]["init"] = continuum_order

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
    if parametric_sfh:
        minit = model_params["mass"]["init"]
        model_params["mass"]["prior"].params["maxi"] = minit * 10**1.0
        model_params["mass"]["prior"].params["mini"] = minit / 10**1.0
    else:
        minit = model_params["logmass"]["init"]
        model_params["logmass"]["prior"].params["maxi"] = minit + 1.0
        model_params["logmass"]["prior"].params["mini"] = minit - 1.0
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-2, maxi=0.2)

    if continuum_order > 0:
        return sedmodel.PolySpecModel(model_params)
    else:
         return sedmodel.SpecModel(model_params)


# --------------
# SPS Object
# --------------

from prospect.sources import SSPBasis

class TabularBasis(SSPBasis):
    """Subclass of :py:class:`SSPBasis` that implements a fixed tabular SFH.
    The user must add the `tabular_time`, `tabular_sfr`, and `mtot` attributes
    """

    def get_galaxy_spectrum(self, **params):
        """Construct the tabular SFH and feed it to the ``ssp``.
        """
        self.update(**params)
        self.ssp.params["sfh"] = 3  # Hack to avoid rewriting the superclass
        self.ssp.set_tabular_sfh(self.tabular_time, self.tabular_sfr)
        wave, spec = self.ssp.get_spectrum(tage=-99, peraa=False)
        return wave, spec / self.mtot, self.ssp.stellar_mass / self.mtot


def build_sps(zcontinuous=1, compute_vega_mags=False, add_realism=False,
              illustris_sfh_file="", use_table=False, parametric_sfh=False,
              tuniv=13.7, **extras):
    """Load the SPS object.  If add_realism is True, set up to convolve the
    library spectra to an sdss instrumental resolution
    """
    if use_table & illustris_sfh_file:
        sps = TabularBasis(zcontinuous=zcontinuous,
                           compute_vega_mags=compute_vega_mags)
        time, sfr = np.genfromtxt(illustris_sfh_file).T
        if tuniv is not None:
            inds = slice(0, np.argmin(np.abs(time - tuniv)))
        else:
            inds = slice(None)
        sps.tabular_time = time[inds]
        sps.tabular_sfr = sfr[inds]
        sps.mtot = np.trapz(sfr[inds], time[inds]) * 1e9
    elif parametric_sfh:
        from prospect.sources import CSPSpecBasis
        sps = CSPSpecBasis(zcontinuous=zcontinuous,
                           compute_vega_mags=compute_vega_mags)
    else:
        from prospect.sources import FastStepBasis
        sps = FastStepBasis(zcontinuous=zcontinuous,
                            compute_vega_mags=compute_vega_mags)

    if add_realism:
        set_sdss_lsf(sps.ssp, **extras)

    return sps


# --------------
# OBS
# --------------

def build_obs(illustris_sfh_file="",
              dlambda_spec=2.0, wave_lo=3800, wave_hi=7000.,
              filterset=None,
              snr_spec=100., snr_phot=0., add_noise=False, seed=101,
              add_realism=False, mask_elines=False,
              continuum_order=0, **kwargs):
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
    continuum_optimize = continuum_order > 0
    if has_spectrum:
        a = 1 + kwargs.get("zred", 0.0)
        wavelength = np.arange(wave_lo, wave_hi, dlambda_spec) * a
    else:
        wavelength = None

    if np.all(snr_phot <= 0):
        filterset = None

    # We need the models to make a mock.
    # For the SPS we use the Tabular SFH from illustris
    sps = build_sps(add_realism=add_realism, use_table=True, **kwargs)
    assert len(sps.tabular_time) > 0
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
    mock['tabular_time'] = sps.tabular_time.copy()
    mock['tabular_sfr'] = sps.tabular_sfr.copy()
    mock['tabular_mtot'] = sps.mtot

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
