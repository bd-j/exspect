#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""fig1.py - Basic parameteric fit

This script is intended to show a corner plot of the posterior PDF and the
quality of the fit of a simple parameteric model to mock broadband photometry.
"""


import os, glob
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl


#from matplotlib.ticker import MaxNLocator, NullLocator
#from matplotlib.ticker import ScalarFormatter

from prospect.io import read_results as reader

from exspect.utils import get_truths, sample_prior
from exspect.cornerplot import allcorner, scatter, marginal, get_spans
from exspect.sedplot import show_sed, posterior_sed, truespec


def convert(chain, labels):
    dust = 1.086 * chain[..., labels.index("dust2")]
    z = chain[..., labels.index("logzsol")]
    sfh, sfh_label = construct_sfh_measure(chain, labels)
    return np.array(sfh + [z, dust]), sfh_label + ["log Z/Z$_\odot$", "A$_V$"]


def construct_sfh_measure(chain, labels):
    tage = chain[..., labels.index("tage")]
    tau = np.log10(chain[..., labels.index("tau")])
    mass = chain[..., labels.index("mass")]
    mass = np.log10(mass)
    return [mass, tage, tau], ["$log M_*$", "$Age$", "$log(\\tau)$"]


def rectify_res(res, start=0):
    chain = res["chain"][start:]
    wghts = res["weights"][start:]
    labels = res["theta_labels"]
    values, newlabels = convert(chain, labels)

    return values, newlabels, wghts


def show_priors(model, diagonals, spans, smooth=0.1, nsample=int(1e4),
                color="g", **linekwargs):
    """
    """
    ps, _ = convert(*sample_prior(model, nsample=nsample))
    smooth = np.zeros(len(diagonals)) + np.array(smooth)
    for i, (x, ax) in enumerate(zip(ps, diagonals)):
        marginal(x, ax, span=spans[i], smooth=smooth[i],
                 color=color, histtype="step", peak=ax.get_ylim()[1], **linekwargs)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fignum", type=int, default=1)
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--result_file", type=str, default="")
    args = parser.parse_args()

    # -- Read in ---
    pfile = args.result_file
    result, obs, model = reader.results_from(pfile)

    # --- Formatting ---
    from matplotlib import rcParams
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['axes.grid'] = False
    rcParams["errorbar.capsize"] = 5

    colorcycle = ["slateblue", "maroon", "orange"]

    label_kwargs = {"fontsize": 14}
    tick_kwargs =  {"labelsize": 10}

    prior_kwargs = {"color": "g", "linestyle": ":", "linewidth": 2}
    hist_kwargs =  {"alpha": 0.5, "histtype": "stepfilled"}
    post_kwargs =  {"color": colorcycle[0], "alpha": 0.5, "linewidth": 2}
    draw_kwargs =  {"color": "r", "linewidth": 1.0, "alpha": 0.5}
    truth_kwargs = {"color": "k", "marker": "o"}
    sed_kwargs =   {"color": 'k', "linewidth": 1, "alpha":0.5}
    data_kwargs =  {"color": "royalblue", "markerfacecolor": "none", "marker": "o",
                    "markersize": 6, "linestyle": "", "markeredgewidth": 2}

    # -- Legend stuff --
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    patches = [Patch(color=c, alpha=hist_kwargs["alpha"]) for c in colorcycle[:1]]
    dot = Line2D([], [], linestyle="", **truth_kwargs)
    prior = Line2D([], [], **prior_kwargs)
    artists = patches + [dot, prior]
    legends = ["Posterior", "Truth", "Prior"]

    # --- Corner plots ---
    if True:

        xx, labels, wghts = rectify_res(result, start=1000)
        spans = get_spans(None, xx, weights=wghts)

        t = (get_truths(result)[0], result["theta_labels"])
        t = convert(*t)
        tvec = np.atleast_2d(t[0]).T

        cfig, caxes = pl.subplots(len(xx), len(xx), figsize=(14.5, 12))
        cfig.subplots_adjust(hspace=0.1, wspace=0.1)
        caxes = allcorner(xx, labels, caxes, weights=wghts, span=spans,
                          color=colorcycle[0], hist_kwargs=hist_kwargs,
                          psamples=tvec, samples_kwargs=truth_kwargs,
                          label_kwargs=label_kwargs,
                          tick_kwargs=tick_kwargs, max_n_ticks=4)

        show_priors(model, np.diag(caxes), spans,
                    smooth=0.1, **prior_kwargs)



    # ---- SED inset plot ------
    if True:
        # --- get samples ---
        sps = reader.get_sps(result)
        sample_kwargs = {"nsample": int(1e4), "start": 0.2,
                         "weights": result["weights"]}
        thetas, fluxes = posterior_sed(result["chain"], model, obs, sps, sample_kwargs=sample_kwargs)
        twave, tspec = truespec(obs, model, sps, R=500 * 2.35, nufnu=True)

        # --- plot ----
        sax = cfig.add_subplot(6, 2, 2)
        rax = cfig.add_subplot(6, 2, 4, sharex=sax)

        truth = {"maggies": obs["true_maggies"].copy(),
                 "filters": obs["filters"],
                 "maggies_unc": np.zeros_like(obs["maggies_unc"]),
                 "phot_mask": slice(None)}

        sax.plot(twave, tspec, **sed_kwargs)
        #sax.scatter(obs["phot_wave"], obs["true_maggies"], **truth_kwargs)
        sax = show_sed(truth, phot=fluxes[1], ax=sax, masked=True, nufnu=True,
                       showtruth=True, truth_kwargs=truth_kwargs,
                       ndraw=0, quantiles=None)
        sax = show_sed(obs, phot=fluxes[1], ax=sax, masked=True, nufnu=True,
                       showtruth=True, truth_kwargs=data_kwargs,
                       ndraw=0, quantiles=None)
        rax = show_sed(obs, phot=fluxes[1], ax=rax, masked=True, nufnu=True,
                       residual=True,
                       ndraw=6, draw_kwargs=draw_kwargs,
                       post_kwargs=post_kwargs)

        # --- prettify ---
        rax.axhline(0.0, linestyle=":", color="k")
        xmin, xmax = np.min(twave), np.max(twave)
        ymin, ymax = tspec.min()*0.9, tspec.max()/0.9
        sax.set_xscale("log")
        sax.set_yscale("log")
        sax.set_xlim(xmin, xmax)
        sax.set_ylim(ymin, ymax)
        rax.set_ylim(-2.5, 2.5)
        rax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))
        sax.set_xticklabels([])
        sax.set_ylabel(r"$\\nu \, f_\\nu \, (erg/s/cm^2)$")
        rax.set_ylabel(r"$\\chi$")
        rax.set_xlabel(r"$\\lambda \, (\AA)$")

        data = Line2D([], [], **data_kwargs)
        draws = Line2D([], [], **draw_kwargs)

        artists += [data, draws]
        legends += ["Mock Data", "Posterior Draws"]

    cfig.legend(artists, legends, (0.77, 0.43), frameon=True)
    cfig.savefig("paperfigures/fig{}.{}".format(args.fignum, args.ext), dpi=400)
