#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import gamma, gammainc

from prospect.sources.constants import cosmo
from .utils import step

__all__ = ["show_sfh", "params_to_sfh",
           "delay_tau", "delay_tau_cmf", "delay_tau_mwa", "delay_tau_ssfr",
           "ratios_to_sfrs", "sfh_to_cmf", "nonpar_mwa", "nonpar_recent_sfr"]


def show_sfh(ages, sfrs=None, ax=None,
             ndraw=0, draw_kwargs={"color": "g", "linewidth": 2, "alpha": 0.5},
             quantiles=[0.16, 0.5, 0.84], post_kwargs={"alpha": 0.5, "color": "slateblue"}):

    sfrs = np.atleast_2d(sfrs)
    binned = ages.ndim > 1
    if binned:
        cages = np.array(ages[:, 0].tolist() + [ages[-1, 1]])
    else:
        cages = ages

    if quantiles is not None:
        qq = np.percentile(sfrs, np.array(quantiles) * 100, axis=0)
        if binned:
            step(ages[:, 0], ages[:, 1], ylo=qq[0,:], yhi=qq[2,:], ax=ax, **post_kwargs)
            step(ages[:, 0], ages[:, 1], qq[1,:], ax=ax, **post_kwargs)
        else:
            _ = fill_between(ages, qq[0, :], qq[2, :], ax=ax, **post_kwargs)
            ax.plot(ages, qq[1, :], **post_kwargs)

    if ndraw > 0:
        if binned:
            [step(ages[:, 0], ages[:, 1], s, ax=ax, **draw_kwargs)
             for s in sfrs[:ndraw]]
        else:
            [ax.plot(ages, s, **draw_kwargs)
             for s in sfrs[:ndraw]]

    return ax


def params_to_sfh(params, time=None, agebins=None):

    parametric = (time is not None)

    if parametric:
        taus, tages, masses = params["tau"], params["tage"], params["mass"]
        sfhs = []
        cmfs = []
        for tau, tage, mass in zip(taus, tages, masses):
            x = np.array([1.0, tau, tage])
            sfr_model = delay_tau(x, time)
            A = mass / (np.trapz(sfr_model, time) * 1e9)
            x = np.array([A, tau, tage])
            sfhs.append(delay_tau(x, time))
            cmfs.append(delay_tau_cmf(x[-2:], time))
        lookback = time.max() - time
        sfhs = np.array(sfhs)
        cmfs = np.array(cmfs)

    else:
        logmass = params["logmass"]
        logsfr_ratios = params["logsfr_ratios"]
        sfhs = np.array([ratios_to_sfrs(logm, sr, agebins)
                        for logm, sr in zip(logmass, logsfr_ratios)])
        cmfs = sfh_to_cmf(sfhs, agebins)
        lookback = 10**(agebins-9)

    return lookback, sfhs, cmfs


def stoch_params_to_sfh(params, sig=0.01):
    """Take a stochastic parameter set and return an sfh

    :returns lookback:
        lookback time, Gyr

    :returns sfr:
        SFR, in Msun/yr
    """
    tuniv = cosmo.age(params["redshift"]).to("Gyr").value
    nt = max(tuniv*2*10/sig, 1000)
    lookback = np.linspace(0, tuniv, nt)
    sfr = np.zeros_like(lookback)
    const = params["mass"][0] / params["tage"][0]
    sfr[lookback < params["tage"][0]] = const
    # Bursts
    stop = params["nburst"] + 1
    tb = params["tage"][1:stop]
    mb = params["mass"][1:stop]
    sb = sig #* tb
    exparg = -0.5 * ((lookback[:, None] - tb[None, :]) / sb)**2
    bs = mb / (sb*np.sqrt(2*np.pi)) * np.exp(exparg)
    sfr += bs.sum(axis=-1)

    return lookback, sfr * 1e-9


def delay_tau(theta, time=None):
    A, tau, age = theta
    tstart = time.max() - age
    tt = (time - tstart) / tau
    sfr = A * tt * np.exp(-tt)
    sfr[tt < 0] = 0
    if (age > time.max()) or (age < 0):
        sfr *= 0.0
    return sfr


def delay_tau_cmf(theta, time=None):
    tau, age = theta
    if (age > time.max()) or (age < 0):
        return np.zeros_like(time)
    tstart = time.max() - age
    tt = (time - tstart) / tau
    tt[tt < 0] = 0.0
    cmf = gammainc(2, tt)
    cmf /= cmf[-1]
    return cmf


def delay_tau_mwa_numerical(theta):
    ''' this is done numerically
    '''
    tau, tage = theta
    t = np.linspace(0, tage, 1000)
    tavg = np.trapz((t**2)*np.exp(-t/tau), t) / np.trapz(t*np.exp(-t/tau), t)
    return tage - tavg


def delay_tau_mwa(theta):
    ''' this is done analytic
    '''
    power = 1
    tau, tage = theta
    tt = tage / tau
    mwt = gammainc(power+2, tt) * gamma(power+2) / gammainc(power+1, tt) * tau

    return tage - mwt


def delay_tau_ssfr(theta, power=1):
    ''' just for the last age
    '''
    tau, tage = theta
    tt = tage / tau
    sfr = tt**power * np.exp(-tt) / tau
    mtot = gammainc(power+1, tt)
    return sfr/mtot * 1e-9


def ratios_to_sfrs(logmass, logsfr_ratios, agebins):
    """scalar
    """
    from prospect.models.transforms import logsfr_ratios_to_masses
    masses = logsfr_ratios_to_masses(np.squeeze(logmass),
                                     np.squeeze(logsfr_ratios),
                                     agebins)
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    sfrs = masses / dt
    return sfrs


def nonpar_recent_sfr(logmass, logsfr_ratios, agebins, sfr_period=0.1):
    """vectorized
    """
    from prospect.models.transforms import logsfr_ratios_to_masses
    masses = [logsfr_ratios_to_masses(np.squeeze(logm), np.squeeze(sr), agebins)
              for logm, sr in zip(logmass, logsfr_ratios)]
    masses = np.array(masses)
    ages = 10**(agebins - 9)
    # fractional coverage of the bin by the sfr period
    ft = np.clip((sfr_period - ages[:, 0]) / (ages[:, 1] - ages[:, 0]), 0., 1)
    mformed = (ft * masses).sum(axis=-1)
    return mformed / (sfr_period * 1e9)


def nonpar_mwa(logmass, logsfr_ratios, agebins):
    """mass-weighted age, vectorized
    """
    sfrs = np.array([ratios_to_sfrs(logm, sr, agebins)
                     for logm, sr in zip(logmass, logsfr_ratios)])
    ages = 10**(agebins)
    dtsq = (ages[:, 1]**2 - ages[:, 0]**2) / 2
    mwa = [(dtsq * sfr).sum() / 10**logm
           for sfr, logm in zip(sfrs, logmass)]
    return np.array(mwa) / 1e9


def sfh_to_cmf(sfrs, agebins):
    sfrs = np.atleast_2d(sfrs)
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    masses = (sfrs * dt)[..., ::-1]
    cmfs = masses.cumsum(axis=-1)
    cmfs /= cmfs[..., -1][..., None]
    zshape = list(cmfs.shape[:-1]) + [1]
    zeros = np.zeros(zshape)
    cmfs = np.append(zeros, cmfs, axis=-1)
    ages = 10**(np.array(agebins) - 9)
    ages = np.array(ages[:, 0].tolist() + [ages[-1, 1]])
    return ages, np.squeeze(cmfs[...,::-1])
