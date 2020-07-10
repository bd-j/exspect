#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as pl

from sedpy.observate import load_filters

__all__ = ["build_mock", "get_lsf",
           "load_sdss", "set_sdss_lsf",
           "eline_mask", "fit_continuum"]


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
    mock['true_maggies'] = phot.copy()
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

def get_lsf(spec, miles_fwhm_aa=2.54, zred=0.0, **extras):
    """This method takes a spec file and returns the quadrature difference
    between the instrumental dispersion and the MILES dispersion, in km/s, as a
    function of wavelength
    """
    lightspeed = 2.998e5  # km/s
    # Get the SDSS instrumental resolution for this plate/mjd/fiber
    wave_obs = 10**spec['loglam']  # observed frame wavelength
    # This is the instrumental velocity resolution in the observed frame
    sigma_v = np.log(10) * lightspeed * 1e-4 * spec['wdisp']
    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave_obs, sigma_v = wave_obs[good], sigma_v[good]
    # Get the miles velocity resolution function at the corresponding
    # *rest-frame* wavelength
    wave_rest = wave_obs / (1 + zred)
    sigma_v_miles = lightspeed * miles_fwhm_aa / 2.355 / wave_rest

    # Get the quadrature difference
    # (Zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_miles**2, 0, np.inf))
    # Restrict to regions where MILES is used
    good = (wave_rest > 3525.0) & (wave_rest < 7500)

    # return the broadening of the rest-frame library spectra required to match
    # the obserrved frame instrumental lsf
    return wave_rest[good], dsv[good]


def load_sdss(sdss_filename="", **extras):
    """This method loads an SDSS spectral file
    """
    import astropy.io.fits as pyfits
    with pyfits.open(sdss_filename) as hdus:
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
    wave, delta_v = get_lsf(sdss_spec, zred=zred, **extras)
    ssp.libraries[1] == 'miles', "Please change FSPS to the MILES libraries."
    ssp.params['smooth_lsf'] = True
    ssp.set_lsf(wave, delta_v)


def eline_mask(wave, lines, pad):
    """A little method to apply emission line masks based on wavelength
    intervals like used in Starlight.
    """
    isline = np.zeros(len(wave), dtype=bool)
    for w in lines:
        lo, hi = w-pad, w+pad
        #print(lo, hi)
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
