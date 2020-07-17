from copy import deepcopy
import numpy as np
from prospect.models import priors, sedmodel
from prospect.sources import CSPSpecBasis

from sedpy.observate import load_filters
from astropy.cosmology import WMAP9 as cosmo


# --------------
# OBS
# --------------

# Here we are going to put together some filter names
# All these filters are available in sedpy.  If you want to use other filters,
# add their transmission profiles to sedpy/sedpy/data/filters/ with appropriate
# names (and format)
spitzer = ['spitzer_irac_ch'+n for n in "1234"]
twomass = ['twomass_{}'.format(b) for b in ['J', 'H', 'Ks']]
acs = ['acs_wfc_{}'.format(b) for b in
       ['f435w', 'f606w', 'f775w', 'f814w', 'f850lp']]
wfc3ir = ['wfc3_ir_f105w', 'wfc3_ir_f125w', 'wfc3_ir_f140w', 'wfc3_ir_f160w']


def build_obs(**kwargs):
    """Load GNz-11 photometry
    """
    filterset = acs + wfc3ir + twomass[-1:] + spitzer[:2]

    obs = {}
    obs['wavelength'] = None  # No spectrum
    obs['filters'] = load_filters(filterset)

    obs['maggies'] = 1e-9/3631 * np.array([7., 2., 5., 3., 17., -7, 11., 64., 152., 137., 139., 144.])
    obs['maggies_unc'] = 1e-9/3631 * np.array([9., 7., 10., 7., 11., 9., 8., 13., 10., 67., 21., 27])
    obs['phot_wave'] = np.array([f.wave_effective for f in obs['filters']])

    return obs


# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# --------------
# MODEL SETUP
# --------------


def build_model(add_neb=True, add_duste=True,
                free_nebZ=True, free_igm=True,
                **kwargs):

    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Basic non-parameteric SFH with nebular & dust emission ---
    model_params = TemplateLibrary["parametric_sfh"]

    # --- we are fitting for redshift ---
    model_params["zred"]["isfree"] = True
    model_params["zred"]["prior"] = priors.TopHat(mini=2, maxi=12)

    # --- Dust attenuation ---
    # Switch to Kriek and Conroy 2013
    model_params["dust_type"]["init"] = 4
    # Slope of the attenuation curve, as delta from Calzetti
    model_params["dust_index"]  = {"N": 1, "isfree": False, "init": 0.0}
    # young star dust
    model_params["dust1"]       = {"N": 1, "isfree": False, "init": 0.0}
    model_params["dust1_index"] = {"N": 1, "isfree": False, "init": -1.0}
    model_params["dust_tesc"]   = {"N": 1, "isfree": False, "init": 7.0}

    # --- IGM, nebular and dust emission ---
    model_params.update(TemplateLibrary["igm"])
    if add_neb:
        model_params.update(TemplateLibrary["nebular"])
        model_params["gas_logu"]["isfree"] = True
        if free_nebZ:
            # Fit for independent gas metallicity
            model_params["gas_logz"]["isfree"] = True
            _ = model_params["gas_logz"].pop("depends_on")
    if add_duste:
        model_params.update(TemplateLibrary["dust_emission"])

    return sedmodel.SpecModel(model_params)

# -----------------
# Noise Model
# ------------------


def build_noise(**extras):
    return None, None


if __name__ == "__main__":

    # - Parser with default arguments -
    parser = prospect_args.get_parser(["optimize", "dynesty"])
    # - Add custom arguments -

    # Fitted Model specification
    parser.add_argument('--add_neb', action="store_true",
                        help="If set, add nebular emission in the model (and mock).")
    parser.add_argument('--add_duste', action="store_true",
                        help="If set, dust emission in the model (and mock).")
    parser.add_argument('--free_nebZ', action="store_true",
                        help="If set, use a nebular metallicity untied to the stellar Z")
    parser.add_argument('--free_igm', action="store_true",
                        help="If set, allow for the IGM attenuation to vary")


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
