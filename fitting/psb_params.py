# import modules
import sys, os, time
import numpy as np

from prospect import prospect_args
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer

from astropy.io import fits
from sedpy import observate
from prospect.models.sedmodel import PolySpecModel

from prospect.sources import FastStepBasis
from prospect.sources.constants import cosmo


parser = prospect_args.get_parser()
# --- Add custom arguments ---
# Data arguments
parser.add_argument('--objname', type=str, default='92942',
                    help="Name of the object to fit; has no effect.")
parser.add_argument('--zred', type=float, default=0.073,
                    help="Redshift.  Do not change.")
parser.add_argument('--err_floor', type=float, default=0.05,
                    help="Force fractional photometric errors to be larger than this")
# Model options
parser.add_argument('--nbins_sfh', type=int, default=8,
                    help="Number of bins in the non-parameteric SFH")
parser.add_argument('--continuum_order', type=int, default=12,
                    help="If set, fit continuum.")
parser.add_argument('--smooth_instrument', action="store_true",
                    help="If set, smooth the SSPs to the instrumetnal resolution")
# Noise model
parser.add_argument('--jitter_model', action="store_true",
                    help="If set, add a jitter term.")
parser.add_argument('--mixture_model', action="store_true",
                    help="If set, add a mixture model term.")
# Nebular emission model
parser.add_argument('--add_neb', action="store_true",
                    help="If set, add nebular emission in the model (and mock).")
parser.add_argument('--marginalize_neb', action="store_true",
                    help="If set, add nebular emission in the model (and mock).")
parser.add_argument('--free_neb_met', action="store_true",
                    help="If set, add nebular emission in the model (and mock).")


# ------------------
# Observational data
# ------------------
def build_obs(err_floor=0.05, **kwargs):
    """Load photometry and spectra.  This is specific to a particular object,
    but more general functionality could be added
    """

    tloc = '../data/spec-2101-53858-0220.fits'  # FIXME: HARDCODED!

    # --- u,g,r,i,z
    mags = [18.81, 17.10, 16.43, 16.07, 15.85]  # FIXME: HARDCODED!
    magunc = [0.02, 0.00, 0.00, 0.00, 0.01]
    filters = ['sdss_{}0'.format(b) for b in "ugriz"]

    # --- convert to flux
    # use taylor expansion for uncertainties
    flux = 10**(-0.4*np.array(mags))
    unc = flux*np.array(magunc)/1.086
    unc = np.clip(unc, flux*err_floor, np.inf)

    # --- define photometric mask
    phot_mask = (flux != unc) & (flux != -99.0) & (unc > 0)

    # --- load up obs dictionary for photometry
    obs = {}
    obs['redshift'] = 0.073
    obs['filters'] = observate.load_filters(filters)
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    obs['maggies'] = flux
    obs['maggies_unc'] = unc

    # --- now spectra
    # load target list first
    with fits.open(tloc) as hdus:
        dat = hdus[1].data

    # generate observables
    spec = dat['flux']
    spec_err = 1 / np.sqrt(dat['ivar'])
    wave_obs = 10**dat['loglam']
    instrumental_sigma_v = np.log(10) * 2.998e5 * 1e-4 * dat["wdisp"]

    # convert to maggies
    # spectra are 10**-17 erg s-1 cm-2 Angstrom-1
    c_angstrom = 2.998e18
    factor = ((wave_obs)**2 / c_angstrom) * 1e-17 * 1e23 / 3631.
    spec, spec_err = spec * factor, spec_err * factor

    # create spectral mask
    # approximate cut-off for MILES library at 7500 A rest-frame, using SDSS redshift,
    # also mask Sodium D absorption
    # also mask Ca H & K absorption
    wave_rest = wave_obs / (1+obs['redshift'])
    mask = ((spec_err != 0) &
            (spec != 0) &
            (wave_rest < 7500) &
            (np.abs(wave_rest-5892.9) > 25) &
            (np.abs(wave_rest-3935.0) > 10) &
            (np.abs(wave_rest-3969.0) > 20)
            )

    obs['wavelength'] = wave_obs[mask]
    obs['spectrum'] = spec[mask]
    obs['unc'] = spec_err[mask]
    obs['mask'] = np.ones(mask.sum(), dtype=bool)
    obs["sigma_v"] = instrumental_sigma_v[mask]

    # plot SED to ensure everything is on the same scale
    if False:
        import matplotlib.pyplot as plt
        smask = obs['mask']
        plt.plot(obs['wavelength'][smask], obs['spectrum'][smask], '-', lw=2, color='red')
        plt.plot(obs['wave_effective'], obs['maggies'], 'o', color='black', ms=8)
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(obs['spectrum'].min()*0.5,obs['spectrum'].max()*2)
        #plt.xlim(6000,10000)
        plt.show()

    return obs


# --------------
# Model Definition
# --------------
def build_model(zred=0.073, nbins_sfh=8,
                mixture_model=True, jitter_model=True,
                add_neb=True, marginalize_neb=True, free_neb_met=True,
                continuum_order=12, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.
    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    """

    from prospect.models.templates import TemplateLibrary, describe
    from prospect.models import priors, sedmodel

    # --- input basic continuity SFH ---
    model_params = TemplateLibrary["continuity_sfh"]
    model_params["logmass"]["prior"] = priors.TopHat(mini=9, maxi=12)

    #  --- fit for redshift ---
    # use catalog value as center of the prior
    model_params["zred"]['isfree'] = True
    model_params["zred"]["init"] = zred
    model_params["zred"]["prior"] = priors.TopHat(mini=zred-0.01, maxi=zred+0.01)

    # --- modify SFH bins ---
    model_params["nbins_sfh"] = dict(N=1, isfree=False, init=nbins_sfh)
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh-1
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1, 0.0)  # constant SFH
    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1, 0.0),
                                                             scale=np.full(nbins_sfh-1, 0.3),
                                                             df=np.full(nbins_sfh-1, 2))

    # --- Use C3K everywhere ---
    model_params["use_wr_spectra"] = dict(N=1, isfree=False, init=0)
    model_params["logt_wmb_hot"] = dict(N=1, isfree=False, init=np.log10(5e4))


    # add redshift scaling to agebins, such that t_max = t_univ
    def zred_to_agebins(zred=None, nbins_sfh=None, **extras):
        tuniv = np.squeeze(cosmo.age(zred).to("yr").value)
        ncomp = np.squeeze(nbins_sfh)
        tbinmax = (tuniv*0.9)
        agelims = [0.0, 7.4772] + np.linspace(8.0, np.log10(tbinmax), ncomp-2).tolist() + [np.log10(tuniv)]
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

    def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
        agebins = zred_to_agebins(zred=zred, **extras)
        logsfr_ratios = np.clip(logsfr_ratios, -10, 10)  # numerical issues...
        nbins = agebins.shape[0]
        sratios = 10**logsfr_ratios
        dt = (10**agebins[:, 1] - 10**agebins[:, 0])
        coeffs = np.array([(1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
        m1 = (10**logmass) / coeffs.sum()
        return m1 * coeffs

    model_params['agebins']['depends_on'] = zred_to_agebins
    model_params['mass']['depends_on'] = logmass_to_masses

    # --- metallicity (flat prior) ---
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.19)

    # --- complexify the dust ---
    model_params['dust_type']['init'] = 4
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=2.0, mean=0.3, sigma=1)
    model_params["dust_index"] = dict(N=1, isfree=True, init=0,
                                      prior=priors.TopHat(mini=-1.0, maxi=0.2))

    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return dust1_fraction*dust2

    model_params['dust1'] = dict(N=1, isfree=False, init=0,
                                 prior=None, depends_on=to_dust1)
    model_params['dust1_fraction'] = dict(N=1, isfree=True, init=1.0,
                                          prior=priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3))

    # --- spectral smoothing ---
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=50.0, maxi=300.0)

    # --- Nebular emission ---
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params['nebemlineinspec'] = dict(N=1, isfree=False, init=False)
        model_params['gas_logu']['isfree'] = True
        if free_neb_met:
            model_params['gas_logz']['isfree'] = True
            _ = model_params["gas_logz"].pop("depends_on")

        if marginalize_neb:
            model_params.update(TemplateLibrary['nebular_marginalization'])
            #model_params.update(TemplateLibrary['fit_eline_redshift'])
            model_params['eline_prior_width']['init'] = 1.0
            model_params['use_eline_prior']['init'] = True

            # only marginalize over a few (strong) emission lines
            if False:
                to_fit = ['H delta 4102', 'H gamma 4340', '[OIII]4364', 'HeI 4472',
                          'H beta 4861', '[OIII]4960', '[OIII]5007', '[ArIII]5193',
                          '[NII]6549', 'H alpha 6563', '[NII]6585', '[SII]6717', '[SII]6732']
                model_params['lines_to_fit']['init'] = to_fit

            # model_params['use_eline_prior']['init'] = False
        else:
            model_params['nebemlineinspec']['init'] = True

    # This removes the continuum from the spectroscopy. Highly recommend
    # using when modeling both photometry & spectroscopy
    if continuum_order > 0:
        model_params.update(TemplateLibrary['optimize_speccal'])
        model_params['spec_norm']['isfree'] = False
        model_params["polyorder"]["init"] = continuum_order

    # This is a pixel outlier model. It helps to marginalize over
    # poorly modeled noise, such as residual sky lines or
    # even missing absorption lines
    if mixture_model:
        model_params['nsigma_outlier_spec'] = dict(N=1, isfree=False, init=50.)
        model_params['f_outlier_spec'] = dict(N=1, isfree=True, init=0.01,
                                              prior=priors.TopHat(mini=1e-5, maxi=0.1))
        model_params['nsigma_outlier_phot'] = dict(N=1, isfree=False, init=50.)
        model_params['f_outlier_phot'] = dict(N=1, isfree=False, init=0.0,
                                              prior=priors.TopHat(mini=0, maxi=0.5))

    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a statistically acceptable fit.
    if jitter_model:
        model_params['spec_jitter'] = dict(N=1, isfree=True, init=1.0,
                                           prior=priors.TopHat(mini=1.0, maxi=3.0))

    # Now instantiate the model using this new dictionary of parameter specifications
    model = PolySpecModel(model_params)

    return model


# --------------
# SPS Object
# --------------
def build_sps(zcontinuous=1, compute_vega_mags=False,
              zred=0.073, smooth_instrument=False, obs=None, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)
    if (obs is not None) and (smooth_instrument):
        #from exspect.utils import get_lsf
        wave_obs = obs["wavelength"]
        sigma_v = obs["sigma_v"]
        speclib = sps.ssp.libraries[1].decode("utf-8")
        wave, delta_v = get_lsf(wave_obs, sigma_v, speclib=speclib, zred=zred, **extras)
        sps.ssp.params['smooth_lsf'] = True
        sps.ssp.set_lsf(wave, delta_v)

    return sps


def get_lsf(wave_obs, sigma_v, speclib="miles", zred=0.0, **extras):
    """This method takes an instrimental resolution curve and returns the
    quadrature difference between the instrumental dispersion and the library
    dispersion, in km/s, as a function of restframe wavelength

    :param wave_obs: ndarray
        Observed frame wavelength (AA)

    :param sigma_v: ndarray
        Instrumental spectral resolution in terms of velocity dispersion (km/s)

    :param speclib: string
        The spectral library.  One of 'miles' or 'c3k_a', returned by
        `sps.ssp.libraries[1]`
    """
    lightspeed = 2.998e5  # km/s
    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave_obs, sigma_v = wave_obs[good], sigma_v[good]
    wave_rest = wave_obs / (1 + zred)

    # Get the library velocity resolution function at the corresponding
    # *rest-frame* wavelength
    if speclib == "miles":
        miles_fwhm_aa = 2.54
        sigma_v_lib = lightspeed * miles_fwhm_aa / 2.355 / wave_rest
        # Restrict to regions where MILES is used
        good = (wave_rest > 3525.0) & (wave_rest < 7500)
    elif speclib == "c3k_a":
        R_c3k = 3000
        sigma_v_lib = lightspeed / (R_c3k * 2.355)
        # Restrict to regions where C3K is used
        good = (wave_rest > 2750.0) & (wave_rest < 9100.0)
    else:
        sigma_v_lib = sigma_v
        good = slice(None)
        raise ValueError("speclib of type {} not supported".format(speclib))

    # Get the quadrature difference
    # (Zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_lib**2, 0, np.inf))

    # return the broadening of the rest-frame library spectra required to match
    # the observed frame instrumental lsf
    return wave_rest[good], dsv[good]


# -----------------
# Noise Model
# ------------------
def build_noise(jitter_model=False, **extras):
    if jitter_model:
        from prospect.likelihood import NoiseModel
        from prospect.likelihood.kernels import Uncorrelated
        jitter = Uncorrelated(parnames=['spec_jitter'])
        spec_noise = NoiseModel(kernels=[jitter], metric_name='unc', weight_by=['unc'])
    else:
        spec_noise = None

    return spec_noise, None


# -----------
# Everything
# ------------
def build_all(**kwargs):
    obs = build_obs(**kwargs)
    sps = build_sps(obs=obs, **kwargs)
    return (obs, build_model(**kwargs),
            sps, build_noise(**kwargs))


if __name__ == '__main__':

    # - Parser with default arguments -

    args = parser.parse_args()
    run_params = vars(args)

    # override some dynesty defaults
    run_params["nested_rwalks"] = 48
    run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
    run_params['nested_maxcall'] = 7500000
    run_params['nested_maxcall_init'] = 7500000
    run_params['nested_maxbatch'] = None
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}
    run_params['objname'] = str(run_params['objname'])

    obs, model, sps, noise = build_all(**run_params)
    run_params["param_file"] = __file__

    if args.debug:
        sys.exit()

    ts = time.strftime("%y%b%d-%H.%M", time.localtime())
    hfile = "{0}_{1}_result.h5".format(args.outfile, ts)
    output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      sps=sps,
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
