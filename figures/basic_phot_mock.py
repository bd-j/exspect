#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic parameteric fit

This script is intended to show a corner plot of the posterior PDF and the
quality of the fit of a simple parameteric model to mock broadband photometry.
"""

import os, glob
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct, dict_to_struct
from prospect.utils.plotting import get_truths

from exspect.plotting.corner import allcorner, marginal, scatter, get_spans
from exspect.plotting.utils import violinplot
from exspect.plotting.utils import get_simple_prior, sample_prior, sample_posterior
from exspect.plotting.sed import to_nufnu, convolve_spec

from exspect.examples.parametric_mock_specphot import build_sps
from defaults import pretty, plot_defaults, colorcycle


def convert(chain, agebins=None):
    """compute quantities from a structured chain (or dictionary)
    """
    cols = ["av", "logzsol"]
    sfh, sfh_label = construct_sfh_measure(chain, agebins)
    niter = len(sfh[0])
    cols += sfh_label
    dt = np.dtype([(c, np.float) for c in cols])
    params = np.zeros(niter, dtype=dt)

    for c in cols:
        if c in chain.dtype.names:
            params[c] = np.squeeze(chain[c])

    # --- dust attenuation
    params["av"] = np.squeeze(1.086 * chain["dust2"])

    # --- stellar ---
    for i, c in enumerate(sfh_label):
        params[c] = np.squeeze(sfh[i])
    return params


def construct_sfh_measure(chain, agebins=None):
    """Compute SFH measures from a structured chain (or dictionary)
    """
    tage = np.atleast_1d(np.squeeze(chain["tage"]))
    tau = np.atleast_1d(np.squeeze(np.log10(chain["tau"])))
    mass = np.atleast_1d(np.squeeze(chain["mass"]))
    mass = np.log10(mass)
    return [mass, tage, tau], ["logmass", "tage", "logtau"]


def show_priors(model, diagonals, spans, show=[], nsample=int(1e4),
                smooth=0.1, color="g", **linekwargs):
    """
    """
    samples, _ = sample_prior(model, nsample=nsample)
    priors = chain_to_struct(samples, model)
    params = convert(priors)
    smooth = np.zeros(len(diagonals)) + np.array(smooth)
    for i, p in enumerate(show):
        ax = diagonals[i]
        if p in priors.dtype.names:
            x, y = get_simple_prior(model.config_dict[p]["prior"], spans[i])
            ax.plot(x, y * ax.get_ylim()[1] * 0.96, color=color, **linekwargs)
        else:
            marginal(params[p], ax, span=spans[i], smooth=smooth[i],
                     color=color, histtype="step", peak=ax.get_ylim()[1], **linekwargs)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fignum", type=str, default="")
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--phot_file", type=str, default="")
    parser.add_argument("--prior_samples", type=int, default=int(1e4))
    parser.add_argument("--n_seds", type=int, default=0)

    args = parser.parse_args()

    # --- Axes ---
    # ------------
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib import rcParams
    from matplotlib.gridspec import GridSpec
    rcParams = plot_defaults(rcParams)

    cfig, caxes = pl.subplots(5, 5, figsize=(14.5, 12))
    cfig.subplots_adjust(hspace=0.1, wspace=0.1)
    sax = cfig.add_subplot(6, 2, 2)
    #rax = cfig.add_subplot(6, 2, 4, sharex=sax)

    # --- Legend stuff ---
    # --------------------
    label_kwargs = {"fontsize": 14}
    tick_kwargs = {"labelsize": 12}
    hkwargs = dict(alpha=0.5)
    pkwargs = dict(color=colorcycle[0], alpha=0.8)
    skwargs = dict(color=colorcycle[1], alpha=0.8)
    dkwargs = dict(color=colorcycle[3], linestyle="", marker="o", mec="k", linewidth=0.75)
    rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)
    tkwargs = dict(color="k", linestyle="dashed", marker="o", linewidth=2)
    lkwargs = dict(color="k", marker="", linestyle="dashed", linewidth=2)

    data = Line2D([], [], **dkwargs)
    truth = Line2D([], [], **tkwargs)
    post = Patch(**pkwargs)
    prior = Line2D([], [], **rkwargs)

    # -- Read in ---
    # --------------
    result, obs, model = reader.results_from(args.phot_file)
    chain = chain_to_struct(result["chain"], model)
    weights = result["weights"]
    show = ["logmass", "logtau", "tage", "logzsol", "av"]

    # --- Corner plots ---
    # --------------------
    params = convert(chain)
    xx = np.array([params[p] for p in show])
    labels = [pretty.get(p, p) for p in show]
    spans = get_spans(None, xx, weights=weights)

    truths = convert(dict_to_struct(obs["mock_params"]))
    tvec = np.array([truths[p] for p in show])
    caxes = allcorner(xx, labels, caxes, weights=weights, span=spans,
                      color=colorcycle[0], hist_kwargs=hkwargs,
                      psamples=tvec, samples_kwargs={"color": tkwargs["color"], "edgecolor": "k"},
                      label_kwargs=label_kwargs,
                      tick_kwargs=tick_kwargs, max_n_ticks=4)

    if args.prior_samples > 0:
        show_priors(model, np.diag(caxes), spans, nsample=args.prior_samples,
                    show=show, smooth=0.1, **rkwargs)

    # --- SED inset plot ---
    # ----------------------
    nufnu = True
    wc = 10**(4 * nufnu)

    owave, ophot, ounc = obs["phot_wave"], obs["maggies"], obs["maggies_unc"]
    phot_width = np.array([f.effective_width for f in obs["filters"]]) / wc
    maxw, minw = np.max(owave + phot_width) * 1.02, np.min(owave - phot_width) / 1.02
    if nufnu:
        _, ophot = to_nufnu(owave, ophot)
        owave, ounc = to_nufnu(owave, ounc)

    if args.n_seds > 0:
        # --- get samples ---
        raw_samples = sample_posterior(result["chain"], result["weights"], nsample=args.n_seds)
        sps = build_sps(**result["run_params"])
        sed_samples = [model.predict(p, obs=obs, sps=sps) for p in raw_samples[:args.n_seds]]
        phot = np.array([sed[1] for sed in sed_samples])
        spec = np.array([sed[0] for sed in sed_samples])

        ind_best = np.argmax(result["lnprobability"])
        pbest = result["chain"][ind_best, :]
        spec_best, phot_best, mfrac_best = model.predict(pbest, obs=obs, sps=sps)
        swave = sps.wavelengths * (1 + chain[ind_best]["zred"])
        if nufnu:
            swave, spec_best = convolve_spec(swave, [spec_best], R=500 * 2.35, maxw=maxw, minw=minw, nufnu=True)
            spec_best = np.squeeze(spec_best)
            _, phot = to_nufnu(obs["phot_wave"], phot)

        violinplot([p for p in phot.T], owave, phot_width, ax=sax, **pkwargs)
        sax.plot(swave, spec_best, color=skwargs["color"], linewidth=0.5,
                 label=r"Highest probability spectrum ($z=${:3.2f})".format(chain[ind_best]["zred"][0]))

    sax.errorbar(owave, ophot, ounc, color="k", linestyle="")
    sax.plot(owave, ophot, **dkwargs)
    #sax.set_ylim(3e-9, 1e-7)
    sax.set_xlim(3e3 / wc, 10e3 / wc)
    sax.set_xscale("log")
    sax.set_yscale("log")
    sax.set_xlabel(r"$\lambda_{\rm obs}$ ($\mu$m)")
    sax.set_ylabel(r"$\nu f_\nu$")
    #sax.set_xticklabels([])

    artists = [data, post, spec]
    legends = ["Observed photometry", "Posterior SED", "MAP spectrum"]
    #sax.legend(loc="upper left")

    artists = [post, truth, prior]
    legends = ["Posterior", "Truth", "Prior"]
    cfig.legend(artists, legends, (0.8, 0.3), frameon=True)

    # --- Saving ----
    # ---------------
    if args.fignum:
        cfig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()