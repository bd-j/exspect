#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as pl

from sedpy.observate import load_filters

__all__ = ["build_mock", "get_lsf",
           "load_sdss", "set_sdss_lsf",
           "eline_mask", "fit_continuum"]

lightspeed = 2.998e5  # km/s


def build_mock(sps, model,
               filterset=None,
               wavelength=None,
               snr_spec=10.0, snr_phot=20., add_noise=False,
               seed=101, **kwargs):
    """Make a mock dataset.  Feel free to add more complicated kwargs, and put
    other things in the run_params dictionary to control how the mock is
    generated.

    :param filterset:
        A list of `sedpy` filter names.  Mock photometry will be generated
        for these filters.

    :param wavelength:
        A vector

    :param snr_phot:
        The S/N of the phock photometry.  This can also be a vector of same
        lngth as the number of filters, for heteroscedastic noise.

    :param snr_spec:
        The S/N of the phock spectroscopy.  This can also be a vector of same
        lngth as `wavelength`, for heteroscedastic noise.

    :param add_noise: (optional, boolean, default: True)
        If True, add a realization of the noise to the mock photometry.

    :param seed: (optional, int, default: 101)
        If greater than 0, use this seed in the RNG to get a deterministic
        noise for adding to the mock data.
    """
    # We'll put the mock data in this dictionary, just as we would for real
    # data.  But we need to know which filters (and wavelengths if doing
    # spectroscopy) with which to generate mock data.
    mock = {"filters": None, "maggies": None, "wavelength": None, "spectrum": None}
    mock['wavelength'] = wavelength
    if filterset is not None:
        mock['filters'] = load_filters(filterset)

    # Now we get any mock params from the kwargs dict
    params = {}
    for p in model.params.keys():
        if p in kwargs:
            params[p] = np.atleast_1d(kwargs[p])

    # And build the mock
    model.params.update(params)
    spec, phot, mfrac = model.predict(model.theta, mock, sps=sps)

    # Now store some output
    mock['true_spectrum'] = spec.copy()
    mock['true_maggies'] = np.copy(phot)
    mock['mock_params'] = deepcopy(model.params)

    # store the mock photometry
    if filterset is not None:
        pnoise_sigma = phot / snr_phot
        mock['maggies'] = phot.copy()
        mock['maggies_unc'] = pnoise_sigma
        mock['mock_snr_phot'] = snr_phot
        # And add noise
        if add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            pnoise = np.random.normal(0, 1, size=len(phot)) * pnoise_sigma
            mock['maggies'] += pnoise

        mock['phot_wave'] = np.array([f.wave_effective for f in mock['filters']])

    # store the mock spectrum
    if wavelength is not None:
        snoise_sigma = spec / snr_spec
        mock['spectrum'] = spec.copy()
        mock['unc'] = snoise_sigma
        mock['mock_snr_spec'] = snr_spec
        # And add noise
        if add_noise:
            if int(seed) > 0:
                np.random.seed(int(seed))
            snoise = np.random.normal(0, 1, size=len(spec)) * snoise_sigma
            mock['spectrum'] += snoise

    return mock


# -----------------
# Helper Functions
# ------------------

def get_lsf(wave_obs, sigma_v, speclib="miles", zred=0.0, **extras):
    """This method takes a spec file and returns the quadrature difference
    between the instrumental dispersion and the MILES dispersion, in km/s, as a
    function of wavelength

    :param wave_obs: ndarray
        observed frame wavelength

    :param sigma_v: ndarray
        spectral resolution in terms of velocity dispersion
    """
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
    # the obserrved frame instrumental lsf
    return wave_rest[good], dsv[good]


def load_sdss(sdss_filename="", **extras):
    """This method loads an SDSS spectral file (BOSS style spec-PLATE-MJD-FIBER.fits,
    by DR16 and possibly earlier)
    """
    import astropy.io.fits as fits
    with fits.open(sdss_filename) as hdus:
        spec = np.array(hdus[1].data)
        info = np.array(hdus[2].data)
        line = np.array(hdus[3].data)
    return spec, info, line


def set_sdss_lsf(ssp, zred=0.0, sdss_filename='', **extras):
    """Method to make the SSPs have the same (rest-frame) resolution as the
    SDSS spectrographs.  This is only correct if the redshift is fixed, but is
    a decent approximation as long as redshift does not change much.
    """
    sdss_spec, _, _ = load_sdss(sdss_filename)
    wave_obs = 10**sdss_spec['loglam']  # observed frame wavelength
    # instrumental resolution as velocity dispersion
    sigma_v = np.log(10) * lightspeed * 1e-4 * sdss_spec['wdisp']
    speclib = ssp.libraries[1].decode("utf-8")
    wave, delta_v = get_lsf(wave_obs, sigma_v, speclib=speclib, zred=zred, **extras)
    ssp.params['smooth_lsf'] = True
    ssp.set_lsf(wave, delta_v)


def set_ggc_lsf(ssp, zred=0.0, wave_lo=3500, wave_hi=7500, **extras):
    """Method to make the SSPs have the same (rest-frame) resolution as the
    GGC data.  This is only correct if the redshift is fixed, but is
    a decent approximation as long as redshift does not change much.
    """
    wave_obs = np.arange(wave_lo, wave_hi, 1.0)
    fwhm = ggc_lsf(wave_obs)  # angstroms
    sigma_v = lightspeed * fwhm / 2.355 / wave_obs  # km/s
    speclib = ssp.libraries[1].decode("utf-8")
    wave, delta_v = get_lsf(wave_obs, sigma_v, speclib=speclib, zred=zred, **extras)
    ssp.params['smooth_lsf'] = True
    ssp.set_lsf(wave, delta_v)


def ggc_lsf(wave):
    """Line spread function of the GGC spectroscopy from Schiavon.

    :param wave:
        Observed frame wavelengths in angstroms

    :returns disp:
        The FWHM at each wavelength given by wave.
    """
    coeffs = np.array([15.290, -6.079e-3, 9.472e-7, -4.395e-11])
    powers = np.arange(len(coeffs))
    fwhm = np.dot(coeffs, wave[None, :] ** powers[:, None])
    return fwhm


def eline_mask(wave, lines, pad=None):
    """A little method to apply emission line masks based on wavelength
    intervals like used in Starlight.
    """
    isline = np.zeros(len(wave), dtype=bool)
    for w in lines:
        try:
            lo, hi = w
        except(TypeError):
            lo, hi = w-pad, w+pad
        isline = isline | ((wave > lo) & (wave < hi))

    return ~isline


def fit_continuum(wave, spec, normorder=6, nreject=1):
    good = np.ones(len(spec), dtype=bool)
    for i in range(nreject+1):
        p = np.polyfit(wave[good], spec[good], normorder)
        poly = np.poly1d(p)
        cal = poly(wave)
        normed = spec / cal
        lastgood = good.copy()
        good = good & ((1 - normed) < normed[good].std())

    return cal, lastgood


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