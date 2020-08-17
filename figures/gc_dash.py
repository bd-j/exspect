#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" gc_dash.py - dashboard for individual fits to GGC spectra + photometry
"""

import os, glob
from copy import deepcopy
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct, dict_to_struct

from exspect.plotting.corner import allcorner, marginal, scatter, get_spans
from exspect.plotting.utils import violinplot
from exspect.plotting.utils import get_simple_prior, sample_prior, sample_posterior
from exspect.plotting.sed import to_nufnu, convolve_spec

from exspect.examples.globular_cluster import build_sps
from defaults import pretty, plot_defaults, colorcycle


def convert(chain, agebins=None):
    """compute quantities from a structured chain (or dictionary)
    """
    niter = len(chain)
    cols = ["av", "logzsol", "tage"]
    dt = np.dtype([(c, np.float) for c in cols])
    params = np.zeros(niter, dtype=dt)

    for c in cols:
        if c in chain.dtype.names:
            params[c] = np.squeeze(chain[c])

    # --- dust attenuation
    params["av"] = np.squeeze(1.086 * chain["dust2"])

    return params


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fignum", type=str, default="")
    parser.add_argument("--figext", type=str, default="png")
    parser.add_argument("--result_file", type=str, default="")
    parser.add_argument("--prior_samples", type=int, default=int(1e4))
    parser.add_argument("--n_seds", type=int, default=0)
    args = parser.parse_args()

    show = ["tage", "logzsol", "av"]

    # --- Read in ---
    # ---------------
    result, obs, model = reader.results_from(args.result_file)
    chain = chain_to_struct(result["chain"], model)
    weights = result["weights"]
    dummy = deepcopy(obs)
    dummy["wavelength"] = None
    dummy["spectrum"] = None

    # --- Axes ---
    # ------------
    from matplotlib.pyplot import rcParams
    rcParams = plot_defaults(rcParams)
    from matplotlib.gridspec import GridSpec
    fig = pl.figure(figsize=(14., 7))
    gsc = GridSpec(3, 3, left=0.58, right=0.95, hspace=0.05, wspace=0.05)
    gss = GridSpec(4, 1, left=0.08, right=0.5)
    caxes = [fig.add_subplot(gsc[i, j]) for i in range(3) for j in range(3)]
    caxes = np.array(caxes).reshape(3, 3)
    sax = fig.add_subplot(gss[:2, 0])
    rax = fig.add_subplot(gss[2, 0])
    lax = fig.add_subplot(gss[3, 0])

    # -- Legend & Styles --
    # ---------------------
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    label_kwargs = {"fontsize": 14}
    tick_kwargs = {"labelsize": 12}

    alpha = 0.65
    pkwargs = dict(color=colorcycle[0], alpha=alpha, linewidth=0)
    skwargs = dict(color=colorcycle[1])
    hkwargs = dict(histtype="stepfilled", alpha=pkwargs["alpha"])
    dkwargs = dict(color="gray", linestyle="", linewidth=0.75, marker="o", mec="k", mfc="w")
    rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)

    tkwargs = dict(color="k", linestyle="--", linewidth=1.5, mfc="k", mec="k")
    lkwargs = dict(color="k", linestyle="-", linewidth=0.75, marker="")

    # --- Corner plots ----
    # ---------------------
    params = convert(chain)
    xx = np.array([params[p] for p in show])
    spans = get_spans(None, xx, weights=weights)
    labels = [pretty.get(p, p) for p in show]

    caxes = allcorner(xx, labels, caxes, weights=weights, span=spans,
                      color=colorcycle[3], hist_kwargs=hkwargs,
                      samples_kwargs={"color": tkwargs["mfc"], "edgecolor": "k"},
                      label_kwargs=label_kwargs, tick_kwargs=tick_kwargs, max_n_ticks=4)

    # --- SED inset plot ---
    # ----------------------
    nufnu = True
    q = [16, 50, 84]
    wc = 10**(4 * nufnu)

    ind_best = np.argmax(result["lnprobability"])
    xbest = result["chain"][ind_best, :]

    owave, ophot, ounc = obs["phot_wave"], obs["maggies"], obs["maggies_unc"]
    phot_width = np.array([f.effective_width for f in obs["filters"]])
    maxw, minw = np.max(owave + phot_width) * 1.02, np.min(owave - phot_width) / 1.02
    phot_width /= wc
    if nufnu:
        _, ophot = to_nufnu(owave, ophot)
        owave, ounc = to_nufnu(owave, ounc)

    # Phot data
    sax.plot(owave, ophot, zorder=20, **dkwargs)
    sax.errorbar(owave, ophot, ounc, zorder=20, color="k", linestyle="")
    pleg = Line2D([], [], **dkwargs)

    if args.n_seds > 0:
        # --- get samples ---
        raw_samples = sample_posterior(result["chain"], result["weights"], nsample=args.n_seds)
        sps = build_sps(**result["run_params"])
        spec, phot, cal, sed = [], [], [], []
        for x in raw_samples[:args.n_seds]:
            s, p, m = model.predict(x, obs=obs, sps=sps)
            spec.append(s)
            phot.append(p)
            cal.append(model._speccal)
            s, p, m = model.predict(x, obs=dummy, sps=sps)
            sed.append(model._sed)
        spec, phot = np.array(spec), np.array(phot)
        cal, sed = np.array(cal), np.array(sed)

        # --- get best & convert---
        spec_best, phot_best, mfrac_best = model.predict(xbest, obs=obs, sps=sps)
        sed_best = model._sed.copy()
        dwave = obs["wavelength"]
        zwave = sps.wavelengths * (1 + chain[ind_best]["zred"])
        m = obs["mask"]

        swave, ssed = convolve_spec(zwave, sed, R=500*2.35, nufnu=nufnu,
                                    maxw=maxw, minw=minw)
        minw, maxw = dwave[m].min(), dwave[m].max()
        bwave, sed_best = convolve_spec(dwave, [sed_best], R=500*2.35, nufnu=nufnu,
                                        maxw=maxw, minw=minw)
        if nufnu:
            awave, spec = to_nufnu(dwave, spec)
            _, phot = to_nufnu(obs["phot_wave"], phot)
            _, cphot_best = to_nufnu(obs["phot_wave"], phot_best)
        else:
            cphot_best = phot_best
            awave = dwave

        sed_best = np.interp(awave, bwave, np.squeeze(sed_best))

        # SED
        qq = np.percentile(ssed, axis=0, q=q)
        violinplot([p for p in phot.T], owave, phot_width, ax=sax, **pkwargs)
        #sax.plot(owave, cphot_best, marker="o", mec="k", **pkwargs)
        sax.fill_between(swave, qq[0, :], qq[-1, :], alpha=alpha, **skwargs)
        sax.plot(awave[m], sed_best[m], linewidth=0.5, alpha=1, color="k")

        spost = Patch(alpha=alpha, **skwargs)
        ppost = Patch(**pkwargs)
        sleg = Line2D([], [], linewidth=0.5, alpha=1, color="k")

        # Residual
        chi = (spec - obs["spectrum"]) / obs["unc"]
        schi_best = (spec_best - obs["spectrum"]) / obs["unc"]
        pchi_best = (phot_best - obs["maggies"]) / obs["maggies_unc"]
        rax.plot(awave[m], schi_best[m], linewidth=0.5, alpha=1, **skwargs)
        rax.plot(owave, pchi_best, **dkwargs)

        # Calibration
        qq = np.percentile(cal, axis=0, q=q)
        lax.fill_between(awave[m], qq[0, m], qq[-1, m], alpha=alpha, **skwargs)

    # Prettify
    sax.set_ylim(7e-9, 6e-8)
    sax.set_ylabel(r"$\nu f_\nu$")
    #sax.set_yscale('log')
    artists = [pleg, sleg, spost, ppost]
    legends = [r"Observed photometry", r"Bestfit Spectrum", r"Posterior (Spectrum)", r"Posterior (Photometry)"]
    sax.legend(artists, legends, loc="lower right", fontsize=12)

    rax.axhline(0.0, linestyle=':', color='k')
    rax.set_ylim(-7, 7)
    rax.set_ylabel(r'$\chi \, (\frac{model-data}{\sigma})$')
    rax.text(0.7, 0.8, r'Residuals', transform=rax.transAxes)

    lax.axhline(1.0, linestyle=':', color='k')
    lax.set_xlabel(r'$\lambda (\mu{\rm m})$')
    lax.set_ylabel(r'$\mathcal{C}\, (\lambda)$', fontsize=12)
    lax.ticklabel_format(axis='x', style='sci', scilimits=(-10, 10))
    lax.text(0.58, 0.60, r"Calibration vector:"+"\n"+r"$F_\nu({\rm model}) \, = \, \mathcal{C}(\lambda) \cdot F_\nu({\rm intrinsic})$",
             transform=lax.transAxes)

    [ax.set_xlim(3300 / wc, 8700 / wc) for ax in [sax, rax, lax]]
    [ax.set_xticklabels('') for ax in [sax, rax]]

    # --- Saving ----
    # ---------------
    if args.fignum:
        fig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()
