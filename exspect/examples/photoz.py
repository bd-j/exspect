#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" photoz.py - Photomteric redshift fitting
Fit for redshift using only photometry of GNz-11
"""


import numpy as np
from prospect.sources.constants import cosmo

# --------------
# SPS Object
# --------------

def build_sps(zcontinuous=1, compute_vega_mags=False, **extras):
    from prospect.sources import FastStepBasis, CSPSpecBasis
    sps = FastStepBasis(zcontinuous=zcontinuous,
                        compute_vega_mags=compute_vega_mags)
    return sps


def zred_to_agebins(zred=None, nbins_sfh=5, zmax=30.0, **extras):
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

