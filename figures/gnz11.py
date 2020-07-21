#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct
from prospect.sources.constants import cosmo

from exspect.plotting import plot_defaults
from exspect.plotting.utils import sample_posterior#, pretty
from exspect.plotting.corner import marginal, allcorner, _quantile
from exspect.plotting.sfhplot import ratios_to_sfrs


from exspect.examples.photoz import zred_to_agebins

pretty = {"logzsol": r"$\log (Z_{\ast}/Z_{\odot})$",
          "logmass": r"$\log M_{formed}$",
          "gas_logu": r"$U_{neb}$",
          "dust2": r"$\tau_V$",
          "dust_index": r"$\Gamma_{dust}$"}


def step(xlo, xhi, y=None, ylo=None, yhi=None, ax=None,
         label=None, linewidth=2, **kwargs):
    """A custom method for plotting step functions as a set of horizontal lines
    """
    clabel = label
    for i, (l, h) in enumerate(zip(xlo, xhi)):
        if y is not None:
            ax.plot([l,h],[y[i],y[i]], label=clabel, linewidth=linewidth, **kwargs)
        if ylo is not None:
            ax.fill_between([l,h], [ylo[i], ylo[i]], [yhi[i], yhi[i]], linewidth=0, **kwargs)
        clabel = None


def sfh_quantiles(time, bins, sfrs, q=[16, 50, 84]):
    tt = bins.reshape(bins.shape[0], -1)
    ss = np.array([sfrs, sfrs]).T.reshape(bins.shape[0], -1)
    sf = np.array([np.interp(tarr, t, s, left=0, right=0) for t, s in zip(tt, ss)])
    qq = np.percentile(sf, axis=0, q=q)
    return qq


if __name__ == "__main__":
    pl.ion()

    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, default="")
    parser.add_argument("--fignum", type=int, default=1)
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--n_sample", type=int, default=1000)
    args = parser.parse_args()

    result, obs, model = reader.results_from(args.results_file)
    chain = chain_to_struct(result["chain"], model=model)
    weights = result["weights"]

    samples = sample_posterior(result["chain"], result["weights"], nsample=args.n_sample)
    samples = chain_to_struct(samples, model)
    agebins = np.array([zred_to_agebins(s["zred"], 5, 20) for s in samples])
    sfh_samples = np.array([ratios_to_sfrs(s["logmass"], s["logsfr_ratios"], agebins=a)
                            for s, a in zip(samples, agebins)])

    show = ["logzsol", "dust2", "logmass", "gas_logu", "dust_index"]
    # set up axes
    rcParams = plot_defaults(rcParams)
    fig = pl.figure(figsize=(8.5, 10.5))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, len(show), width_ratios=len(show) * [10],
                  left=0.1, right=0.98, wspace=0.15, hspace=0.38, top=0.95, bottom=0.1)

    # plot redshift posterior
    zkwargs = dict(color="tomato", alpha=0.8)
    zax = fig.add_subplot(gs[1, :])
    z = np.squeeze(chain["zred"])
    marginal(z, ax=zax, weights=weights, **zkwargs)
    zp = model.config_dict["zred"]["prior"].params
    zax.set_xlim(zp["mini"], zp["maxi"])
    zax.set_xlabel("Redshift")
    zax.set_ylabel("Probability")
    q = _quantile(z, weights=weights, q=[0.16, 0.50, 0.84])
    zax.text(0.1, 0.8, r"$z_{{phot}}={{{:3.2f}}}^{{+{:3.2f}}}_{{-{:3.2f}}}$".format(q[1], q[2]-q[1], q[1]-q[0]), transform=zax.transAxes)

    zax.axvline(11.09, color="royalblue", linestyle="--", linewidth=0.75)

    # plot SED and SED posteriors
    sax = fig.add_subplot(gs[0, :])
    sax.errorbar(obs["phot_wave"]/1e4, obs["maggies"], obs["maggies_unc"], linestyle="", color="black")
    sax.plot(obs["phot_wave"]/1e4, obs["maggies"], marker="o", linestyle="", color="black")
    sax.set_xscale("log")
    sax.set_yscale("log")
    sax.set_ylim(1e-13, 1e-10)
    sax.set_xlim(0.3, 5)

    # plot SFH posteriors
    hkwargs = dict(color="tomato", alpha=0.5, linewidth=0.5)
    hax = fig.add_subplot(gs[3, :])
    tlook = (10**agebins.T/1e9).T
    #for i in range(20):
    #    t, r, s = tlook[i], sfh_samples[i], samples[i]
    #    step(*t.T, r, ax=hax, **hkwargs)

    tarr = np.linspace(tlook.min(), tlook.max(), 500)
    sq = sfh_quantiles(tarr, tlook, sfh_samples)
    median_sfh = hax.plot(tarr, sq[1, :], color=hkwargs["color"])
    hax.fill_between(tarr, sq[0, :], sq[2,:], **hkwargs)

    hax.set_xlabel(r"$t_{lookback}$ (Gyr)")
    hax.set_ylabel("SFR")


    # plot parameter posteriors (logzsol, Gamma, Av, Mformed, gas_logu)
    pkwargs = dict(color="tomato", alpha=0.5)
    paxes = []
    for i, p in enumerate(show):
        pax = fig.add_subplot(gs[2, i])
        marginal(np.squeeze(chain[p]), weights=weights, ax=pax, **pkwargs)
        pax.set_xlabel(pretty.get(p, p))
        paxes.append(pax)
    [ax.set_yticklabels("") for ax in paxes]
