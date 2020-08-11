# import modules
import sys, os
import numpy as np
from sedpy.observate import load_filters
from prospect import prospect_args
from prospect.fitting import fit_model, lnprobfn
from prospect.io import write_results as writer
from astropy.io import fits
from sedpy import observate
from prospect.models.sedmodel import PolySpecModel, gauss
from scipy import optimize
from prospect.sources import FastStepBasis
from prospect.models.templates import TemplateLibrary
from prospect.models import priors
from astropy.cosmology import WMAP9 as cosmo
from prospect.likelihood import NoiseModel
from prospect.likelihood.kernels import Uncorrelated

# define user-specific paths and filter names
apps = os.getenv('APPS')

def build_obs(objname=92942, extra_phot=False,
              err_floor=0.05, **kwargs):
    """Load photometry and spectra
    """

    ### u,g,r,i,z
    mags = [18.81,17.10,16.43,16.07,15.85]
    magunc = [0.02,0.00,0.00,0.00,0.01]
    filters = ['sdss_'+filt+'0' for filt in ['u','g','r','i','z']]

    ### convert to flux
    # use taylor expansion for uncertainties
    flux = 10**((-2./5)*np.array(mags))
    unc = flux*np.array(magunc)/1.086
    unc = np.clip(unc, flux*err_floor, np.inf)

    ### define photometric mask
    phot_mask = (flux != unc) & (flux != -99.0) & (unc > 0)

    ### load up obs dictionary for photometry
    obs = {}
    obs['redshift'] = 0.073
    obs['filters'] = observate.load_filters(filters)
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    obs['maggies'] = flux
    obs['maggies_unc'] = unc

    ### now spectra
    # load target list first
    tloc = apps+'/prospector_alpha/data/spec-2101-53858-0220.fits'
    dat = fits.open(tloc)[1].data

    # generate observables
    spec = dat['flux']
    spec_err = 1/np.sqrt(dat['ivar'])
    wave_obs = 10**dat['loglam']

    # convert to maggies
    # spectra are 10**-17 erg s-1 cm-2 Angstrom-1
    c_angstrom = 2.998e18
    factor = ((wave_obs)**2/c_angstrom) * 1e-17 * 1e23 / 3631.
    spec, spec_err = spec*factor, spec_err*factor

    # create spectral mask
    # approximate cut-off for MILES library at 7500 A rest-frame, using SDSS redshift,
    # also mask Sodium D absorption
    wave_rest = wave_obs / (1+obs['redshift'])
    mask = (spec_err != 0) & \
           (spec != 0) & \
           (wave_rest < 7500) & \
           (np.abs(wave_rest-5892.9) > 25)      

    obs['wavelength'] = wave_obs[mask]
    obs['spectrum'] = spec[mask]
    obs['unc'] = spec_err[mask]
    obs['mask'] = np.ones(mask.sum(),dtype=bool)

    # plot SED to ensure everything is on the same scale
    if False:
        import matplotlib.pyplot as plt
        smask = obs['mask']
        plt.plot(obs['wavelength'][smask],obs['spectrum'][smask],'-',lw=2,color='red')
        plt.plot(obs['wave_effective'],obs['maggies'],'o',color='black',ms=8)
        plt.xscale('log')
        plt.yscale('log')
        #plt.ylim(obs['spectrum'].min()*0.5,obs['spectrum'].max()*2)
        #plt.xlim(6000,10000)
        plt.show()

    return obs

# --------------
# Model Definition
# --------------
def build_model(objname=92942, add_neb=True, mixture_model=True,
                remove_spec_continuum=True, switch_off_spec=False, 
                marginalize_neb=True, **extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.
    :param add_neb: (optional, default: False)
        Switch to add (fixed) parameters relevant for nebular emission, and
        turn nebular emission on.
    """

    # input basic continuity SFH
    model_params = TemplateLibrary["continuity_sfh"]

    # fit for redshift
    # use catalog value as center of the prior
    zred = 0.073   # best-guess from the SDSS pipeline
    model_params["zred"]['isfree'] = True
    model_params["zred"]["init"] =  zred
    model_params["zred"]["prior"] = priors.TopHat(mini=zred-0.01, maxi=zred+0.01)

    # modify to increase nbins
    nbins_sfh = 8
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh-1
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                      scale=np.full(nbins_sfh-1,0.3),
                                                                      df=np.full(nbins_sfh-1,2))
    # add redshift scaling to agebins, such that
    # t_max = t_univ
    def zred_to_agebins(zred=None,agebins=None,**extras):
        tuniv = cosmo.age(zred).value[0]*1e9
        tbinmax = (tuniv*0.9)
        agelims = [0.0,7.4772] + np.linspace(8.0,np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
        agebins = np.array([agelims[:-1], agelims[1:]])
        return agebins.T

    def logmass_to_masses(logmass=None, logsfr_ratios=None, zred=None, **extras):
        agebins = zred_to_agebins(zred=zred)
        logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
        nbins = agebins.shape[0]
        sratios = 10**logsfr_ratios
        dt = (10**agebins[:,1]-10**agebins[:,0])
        coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
        m1 = (10**logmass) / coeffs.sum()
        return m1 * coeffs

    model_params['agebins']['depends_on'] = zred_to_agebins
    model_params['mass']['depends_on'] = logmass_to_masses

    # metallicity (flat prior)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-1.0, maxi=0.19)

    # complexify the dust
    model_params['dust_type']['init'] = 4
    model_params["dust2"]["prior"] = priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)
    model_params["dust_index"] = {"N": 1, 
                                  "isfree": True,
                                  "init": 0.0, "units": "power-law multiplication of Calzetti",
                                  "prior": priors.TopHat(mini=-1.0, maxi=0.4)}

    def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
        return dust1_fraction*dust2

    model_params['dust1'] = {"N": 1, 
                             "isfree": False, 
                             'depends_on': to_dust1,
                             "init": 0.0, "units": "optical depth towards young stars",
                             "prior": None}
    model_params['dust1_fraction'] = {'N': 1,
                                      'isfree': True,
                                      'init': 1.0,
                                      'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    # velocity dispersion
    model_params.update(TemplateLibrary['spectral_smoothing'])
    model_params["sigma_smooth"]["prior"] = priors.TopHat(mini=40.0, maxi=400.0)

    if add_neb:
        # Add nebular emission
        model_params.update(TemplateLibrary["nebular"])
        model_params['gas_logu']['isfree'] = True
        model_params['gas_logz']['isfree'] = True
        model_params['nebemlineinspec'] = {'N': 1,
                                           'isfree': False,
                                           'init': False}
        _ = model_params["gas_logz"].pop("depends_on")

        if marginalize_neb:
            model_params.update(TemplateLibrary['nebular_marginalization'])
            #model_params.update(TemplateLibrary['fit_eline_redshift'])
            model_params['eline_prior_width']['init'] = 1.0
            model_params['use_eline_prior']['init'] = True

            # only marginalize over a few (strong) emission lines
            if False:
                to_fit = ['H delta 4102', 'H gamma 4340', '[OIII]4364', 'HeI 4472','H beta 4861','[OIII]4960','[OIII]5007','[ArIII]5193',
                          '[NII]6549','H alpha 6563','[NII]6585','[SII]6717','[SII]6732']
                model_params['lines_to_fit']['init'] = to_fit

            # model_params['use_eline_prior']['init'] = False
        else:
            model_params['nebemlineinspec']['init'] = True

    # This removes the continuum from the spectroscopy. Highly recommend
    # using when modeling both photometry & spectroscopy
    if remove_spec_continuum:
        model_params.update(TemplateLibrary['optimize_speccal'])
        model_params['spec_norm']['isfree'] = False
        model_params['spec_norm']['prior'] = priors.Normal(mean=1.0, sigma=0.3)

    # This is a pixel outlier model. It helps to marginalize over
    # poorly modeled noise, such as residual sky lines or
    # even missing absorption lines
    if mixture_model:
        model_params['f_outlier_spec'] = {"N": 1, 
                                          "isfree": True, 
                                          "init": 0.01,
                                          "prior": priors.TopHat(mini=1e-5, maxi=0.5)}
        model_params['nsigma_outlier_spec'] = {"N": 1, 
                                              "isfree": False, 
                                              "init": 50.0}
        model_params['f_outlier_phot'] = {"N": 1, 
                                          "isfree": False, 
                                          "init": 0.00,
                                          "prior": priors.TopHat(mini=0, maxi=0.5)}
        model_params['nsigma_outlier_phot'] = {"N": 1, 
                                              "isfree": False, 
                                              "init": 50.0}

    # This is a multiplicative noise inflation term. It inflates the noise in
    # all spectroscopic pixels as necessary to get a statistically acceptable fit.
    model_params['spec_jitter'] = {"N": 1, 
                                   "isfree": True, 
                                   "init": 1.0,
                                   "prior": priors.TopHat(mini=1.0, maxi=3.0)}


    # Now instantiate the model using this new dictionary of parameter specifications
    model = PolySpecModel(model_params)

    return model


# --------------
# SPS Object
# --------------
def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)  # special to remove redshifting issue
    return sps

# -----------------
# Noise Model
# ------------------
def build_noise(**extras):
    jitter = Uncorrelated(parnames = ['spec_jitter'])
    spec_noise = NoiseModel(kernels=[jitter],metric_name='unc',weight_by=['unc'])
    return spec_noise, None

# -----------
# Everything
# ------------
def build_all(**kwargs):
    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()    

    # - Add custom arguments -
    parser.add_argument('--add_neb', action="store_true",default=True,
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--remove_spec_continuum', action="store_true",default=True,
                        help="If set, fit continuum.")
    parser.add_argument('--objname', default='92942',
                        help="Name of the object to fit.")
    parser.add_argument('--switch_off_spec', action="store_true", default=False,
                        help="If set, remove spectrum from obs.")
    parser.add_argument('--extra_phot', action="store_true", default=False,
                        help="If set, remove photometry from obs.")

    args = parser.parse_args()
    run_params = vars(args)

    # add in dynesty settings
    run_params['dynesty'] = True
    run_params['nested_weight_kwargs'] = {'pfrac': 1.0}
    run_params['nested_nlive_batch'] = 200
    run_params['nested_walks'] = 48  # sampling gets very inefficient w/ high S/N spectra
    run_params['nested_nlive_init'] = 500 
    run_params['nested_dlogz_init'] = 0.01
    run_params['nested_maxcall'] = 7500000
    run_params['nested_maxcall_init'] = 7500000
    run_params['nested_method'] = 'rwalk'
    run_params['nested_maxbatch'] = None
    run_params['nested_posterior_thresh'] = 0.03
    run_params['nested_first_update'] = {'min_ncall': 20000, 'min_eff': 7.5}
    run_params['objname'] = str(run_params['objname'])

    obs, model, sps, noise = build_all(**run_params)
    run_params["param_file"] = __file__

    if args.debug:
        sys.exit()

    hfile = apps+'/prospector_alpha/results/psb/psb_sdss_mcmc.h5'
    output = fit_model(obs, model, sps, noise, lnprobfn=lnprobfn, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])

    try:
        hfile.close()
    except(AttributeError):
        pass
