#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""show_illustris.py - Par/Nonpar fits to illustris: SFHs

This script is intended to show the SFHs inferred from high-S/N optical
spectroscopy using parameteric and non-parameteric models.  The true SFH is
given by several illustris galaxies.
"""

import sys, os
import numpy as np

import matplotlib.ticker as ticker
import matplotlib.pyplot as pl
from matplotlib import rcParams
from matplotlib.lines import Line2D

from prospect.io import read_results as reader
from prospect.models.sedmodel import PolySedModel

from prospect.plotting import FigureMaker, pretty, chain_to_struct
from prospect.plotting.sfh import ratios_to_sfrs
from prospect.plotting.corner import quantile

from defaults import plot_defaults, colorcycle

from scipy.special import gamma, gammainc
def parametric_sfr(tau=4, tage=13.7, power=1, mass=None, logmass=None,
                   times=None, **extras):
    """Return the SFR (Msun/yr) for the given parameters of an exponential or
    delayed exponential SFH. Does not account for burst, constant components,
    or truncations.

    :param power: (optional, default: 1)
        Use 0 for exponential decline, and 1 for te^{-t} (delayed exponential decline)

    :param times: (optional, ndarray)
        If given, a set of loockback times where you want to calculate the sfr
    """
    if (mass is None) and (logmass is not None):
        mass = 10**logmass
    if times is None:
        tt = tage
    else:
        assert len(np.atleast_1d(tage)) == 1
        assert len(np.atleast_1d(tau)) == 1
        tt = tage - times
    p = power + 1
    psi = mass * (tt/tau)**power * np.exp(-tt/tau) / (tau * gamma(p) * gammainc(p, tage/tau))
    psi[tt < 0] = 0
    return psi * 1e-9


def binup_illustris(time, sfr, agebins=None):
    dt = (10**agebins[:, 1] - 10**agebins[:, 0])
    ages = 10**(agebins - 9)
    # Convert to lookback time
    lookback = time.max() - time
    ll, sl = lookback[::-1], sfr[::-1]
    mass = np.zeros(len(ages))
    for i, age in enumerate(ages):
        ends = np.interp(age, ll, sl)
        g = (ll >= age[0]) & (ll < age[1])
        tt = np.insert(age, 1, ll[g])
        ss = np.insert(ends, 1, sl[g])
        mass[i] = np.trapz(ss, tt) * 1e9
    sfrs = mass / dt
    return sfrs


def normalize_illustris(time, sfr, mtrue=1e10):
    nt = len(time)
    cmf = [np.trapz(sfr[:i], time[:i]) for i in range(nt)]
    norm = mtrue / (cmf[-1] * 1e9)
    cmf /= cmf[-1]
    normed_sf = sfr * norm
    return normed_sf


def bins_to_vec(tvec, sfrs, bins, eps=0.01):
    sfrs = np.atleast_2d(sfrs)
    ns = sfrs.shape[0]
    tlook = 10**bins / 1e9 * np.array([1.+eps, 1.-eps])
    tlook = np.tile(tlook, (ns, 1, 1))
    tt = tlook.reshape(tlook.shape[0], -1)
    ss = np.array([sfrs, sfrs]).transpose(1, 2, 0).reshape(tlook.shape[0], -1)
    sfhs = np.array([np.interp(tvec, t, s, left=0, right=0) for t, s in zip(tt, ss)])
    return sfhs


class Plotter(FigureMaker):

    def reorder_params(self):
        order = self.result["theta_labels"]
        free = np.array(self.model.free_params)
        ind = np.zeros(len(free), dtype=int)
        for i, p in enumerate(self.model.free_params):
            if p in order:
                ind[i] = order.index(p)
            else:
                ind[i] = order.index(p+"_1")
        par_order = free[np.argsort(ind)].tolist()
        param_order = par_order + [k for k in self.model.config_dict.keys() if k not in par_order]
        return param_order

    def read_in(self, results_file):
        """Read a prospector results file, cache important components,
        and do any parameter transformations.

        :param results_file: string
            full path of the file with the prospector results.
        """
        self.result, self.obs, self.model = reader.results_from(results_file)
        from prospect.models.sedmodel import PolySedModel
        param_order = self.reorder_params()
        self.model = PolySedModel(self.model.config_dict, param_order=param_order)

        self.sps = None
        self.chain = chain_to_struct(self.result["chain"], self.model)
        self.weights = self.result.get("weights", None)
        self.ind_best = np.argmax(self.result["lnprobability"])
        self.parchain = self.convert(self.chain)

    @property
    def agebins(self):
        try:
            return self.model.params["agebins"]
        except(KeyError):
            return None

    @property
    def is_parametric(self):
        return "tau" in self.model.params

    @property
    def illustris_sfh(self):
        ttime = self.obs["tabular_time"]
        tsfr = self.obs["tabular_sfr"]
        return ttime, normalize_illustris(ttime, tsfr)

    def sfh_samples(self, tvec):

        if self.is_parametric:
            sfhs = [parametric_sfr(s["tau"], s["tage"], mass=s["mass"], times=tvec)
                    for s in self.chain]
        else:
            agebins = self.agebins
            binsfr = [ratios_to_sfrs(s["logmass"], s["logsfr_ratios"], agebins=agebins)
                      for s in self.chain]
            sfhs = bins_to_vec(tvec, binsfr, agebins)

        return np.array(sfhs)

    def plot_sfh(self, sfhax, tvec, q=[0.16, 0.50, 0.84]):
        sfhs = self.sfh_samples(tvec)
        sq = quantile(sfhs.T, q=q, weights=self.weights)

        # --- plot SFH ---
        sfhax.plot(tvec, sq[:, 1], '-', lw=1.5, **self.pkwargs)
        sfhax.fill_between(tvec, sq[:, 0], sq[:, 2], **self.pkwargs)
        #sfhax.plot(tvec, sq[:, 0], '-', color='k', alpha=0.3, lw=1.5)
        #sfhax.plot(tvec, sq[:, 2], '-', color='k', alpha=0.3, lw=1.5)


if __name__ == "__main__":

    odir = "../fitting/output/illustris"
    files = ["illustris_sfh2_snr100_nebFalse_mcmc.h5",
             "illustris_nonpar_sfh2_nbins14_slice_snr100_noiseTrue_mcmc.h5",
             "illustris_sfh2orig_snr100_nebFalse_mcmc.h5",
             "illustris_nonpar_sfh2orig_nbins14_slice_snr100_noiseTrue_mcmc.h5"]
    files = [os.path.join(odir, f) for f in files]

    # instantiate the plotters
    plotters = [Plotter(results_file=f) for f in files]
    # make styles with different color for each plotter
    [p.styles() for p in plotters]
    cind = [0, 1, 0, 1]
    for i, p in enumerate(plotters):
        p.pkwargs["color"] = colorcycle[cind[i]]
        p.pkwargs["alpha"] = 0.8
        p.dkwargs = dict(color="gray", linestyle="-", linewidth=0.75, marker="")
        p.make_art()

    truth_kwargs = dict(color="k", linewidth=2)

    # --- Plot SFHs ---
    rcParams = plot_defaults(rcParams)
    fig, axes = pl.subplots(2, 1, squeeze=False, sharex=True,
                            figsize=(7.15, 9.4))

    for i, ax in enumerate(axes.flat):
        it, isf = plotters[i*2 + 1].illustris_sfh
        agebins = plotters[i*2 + 1].agebins

        tlook = np.linspace(1e-3, it.max(), 500)
        plotters[i*2].plot_sfh(ax, tlook)
        plotters[i*2 + 1].plot_sfh(ax, tlook)

        bill = binup_illustris(it, isf, agebins=agebins)
        ax.plot(it.max() - it, isf, **truth_kwargs)
        bv = bins_to_vec(tlook, bill, agebins)[0]
        ax.plot(tlook, bv, alpha=0.5, linestyle=":", **truth_kwargs)

        ax.set_ylabel(r"SFR (M$_\odot$ / yr)", fontsize=16)
        ax.set_xlim(0, it.max())

    # --- prettify ---
    axes.flat[-1].set_xlabel(r"Lookback Time (Gyr)", fontsize=16)
    art = dict(truth=Line2D([], [], **truth_kwargs),
               binned=Line2D([], [], alpha=0.5, linestyle=":", **truth_kwargs))


    # --- Inset recent SFH ---
    if True:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset_locs = [1, 2]
        for i, ax in enumerate(axes.flat):
            inax = inset_axes(ax, width="25%", height="30%", borderpad=3, loc=inset_locs[i])

            it, isf = plotters[i*2 + 1].illustris_sfh
            agebins = plotters[i*2 + 1].agebins

            tlook = np.linspace(1e-3, it.max(), 500)
            plotters[i*2].plot_sfh(inax, tlook)
            plotters[i*2 + 1].plot_sfh(inax, tlook)

            bill = binup_illustris(it, isf, agebins=agebins)
            inax.plot(it.max() - it, isf, **truth_kwargs)
            bv = bins_to_vec(tlook, bill, agebins)[0]
            inax.plot(tlook, bv, alpha=0.5, linestyle=":", **truth_kwargs)

            inax.set_xlim(0, 1)
            g = (it.max() - it) < 1.0
            inax.set_ylim(0, isf[g].max() * 1.5)
            inax.tick_params(axis="both", labelsize=10)
            inax.set_xticks([0.2, 0.4, 0.6, 0.8])
            inax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.1f}"))
            #inax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.2f}"))

    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    legends = ["Parametric", "Non-parametric", "Illustris", "Illustris (binned)"]
    artists = [plotters[0].art["posterior"], plotters[1].art["posterior"], art["truth"], art["binned"]]
    fig.legend(artists, legends, loc='upper right', bbox_to_anchor=(0.96, 0.94),
               frameon=True, fontsize=12, ncol=4)

    fig.savefig("paperfigures/illustris1.png", dpi=400)

    pl.show()
