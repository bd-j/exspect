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

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct
from prospect.utils.plotting import get_truths

from exspect.plotting.corner import allcorner, marginal, scatter, get_spans
from exspect.plotting.utils import sample_prior, sample_posterior, violinplot #, get_truths
from exspect.plotting import plot_defaults, colorcycle
#from exspect.plotting.sedplot import show_sed, posterior_sed, truespec

from exspect.examples.parametric_mock_specphot import build_sps


def convert(chain):
    """compute quantities from a structured chain (or dictionary)
    """
    dust = np.squeeze(1.086 * chain["dust2"])
    z = np.squeeze(chain["logzsol"])
    sfh, sfh_label = construct_sfh_measure(chain)
    return np.array(sfh + [z, dust]), sfh_label + [r"$\log ({\rm Z}/{\rm Z}_\odot)$", r"A$_V$"]


def construct_sfh_measure(chain):
    """Compute SFH measures from a structured chain (or dictionary)
    """
    tage = np.squeeze(chain["tage"])
    tau = np.squeeze(np.log10(chain["tau"]))
    mass = np.squeeze(chain["mass"])
    mass = np.log10(mass)
    return [mass, tage, tau], [r"$\log({\rm M}_\star)$", r"${\rm Age}$", r"$\log(\tau)$"]


def show_priors(model, diagonals, spans, smooth=0.1, nsample=int(1e4),
                color="g", **linekwargs):
    """
    """
    samples, _ = sample_prior(model, nsample=nsample)
    priors = chain_to_struct(samples, model)
    ps, _ = convert(priors)
    smooth = np.zeros(len(diagonals)) + np.array(smooth)
    for i, (x, ax) in enumerate(zip(ps, diagonals)):
        marginal(x, ax, span=spans[i], smooth=smooth[i],
                 color=color, histtype="step", peak=ax.get_ylim()[1], **linekwargs)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fignum", type=int, default=1)
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--phot_file", type=str, default="")
    parser.add_argument("--prior_samples", type=int, default=int(1e4))
    parser.add_argument("--n_seds", type=int, default=0)

    args = parser.parse_args()

    # -- Read in ---
    result, obs, model = reader.results_from(args.phot_file)
    chain = chain_to_struct(result["chain"], model)
    weights = result["weights"]
    if args.n_seds > 0:
        raw_samples = sample_posterior(result["chain"], result["weights"], nsample=args.n_seds)
        sps = build_sps(**result["run_params"])
        sed_samples = [model.predict(p, obs=obs, sps=sps) for p in raw_samples[:args.n_seds]]
        phot = np.array([sed[1] for sed in sed_samples])
        spec = np.array([sed[0] for sed in sed_samples])
        wave = sps.wavelengths

        ind_best = np.argmax(result["lnprobability"])
        pbest = result["chain"][ind_best, :]
        spec_best, phot_best, mfrac_best = model.predict(pbest, obs=obs, sps=sps)
        phot_width = np.array([f.effective_width for f in obs["filters"]])

    # --- Axes & Styles & legends ---
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib import rcParams
    from matplotlib.gridspec import GridSpec
    rcParams = plot_defaults(rcParams)

    cfig, caxes = pl.subplots(5, 5, figsize=(14.5, 12))
    cfig.subplots_adjust(hspace=0.1, wspace=0.1)
    sax = cfig.add_subplot(6, 2, 2)
    rax = cfig.add_subplot(6, 2, 4, sharex=sax)

    #fig = pl.figure()
    #gs = GridSpec(4, 4)

    label_kwargs = {"fontsize": 14}
    tick_kwargs = {"labelsize": 12}

    hkwargs = dict(alpha=0.5, histtype="stepfilled")
    pkwargs = dict(color=colorcycle[0], alpha=0.8)
    skwargs = dict(color=colorcycle[1], alpha=0.8)
    tkwargs = dict(color=colorcycle[3], linestyle="", marker="o", mec="k", linewidth=0.75)
    rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)

#    prior_kwargs = {"color": "g", "linestyle": ":", "linewidth": 2}
#    hist_kwargs =  {"alpha": 0.5, "histtype": "stepfilled"}
#    post_kwargs =  {"color": colorcycle[0], "alpha": 0.5, "linewidth": 2}
#    draw_kwargs =  {"color": "r", "linewidth": 1.0, "alpha": 0.5}
#    truth_kwargs = {"color": "k", "marker": "o"}
#    sed_kwargs =   {"color": 'k', "linewidth": 1, "alpha":0.5}
#    data_kwargs =  {"color": "royalblue", "markerfacecolor": "none", "marker": "o",
#                    "markersize": 6, "linestyle": "", "markeredgewidth": 2}

    # -- Legend stuff --
    patches = [Patch(color=c, alpha=pkwargs["alpha"]) for c in colorcycle[:1]]
    dot = Line2D([], [], **tkwargs)
    prior = Line2D([], [], **rkwargs)
    artists = patches + [dot, prior]
    legends = ["Posterior", "Truth", "Prior"]

    # --- Corner plots ---

    xx, labels = convert(chain)
    spans = get_spans(None, xx, weights=weights)

    t = get_truths(result)
    t, _ = convert(t)
    tvec = np.atleast_2d(np.squeeze(t)).T
    caxes = allcorner(xx, labels, caxes, weights=weights, span=spans,
                      color=colorcycle[0], hist_kwargs=hkwargs,
                      psamples=tvec, samples_kwargs=tkwargs,
                      label_kwargs=label_kwargs,
                      tick_kwargs=tick_kwargs, max_n_ticks=4)

    show_priors(model, np.diag(caxes), spans, nsample=args.prior_samples,
                smooth=0.1, **rkwargs)


    # ---- SED inset plot ------
    pwave = obs["phot_wave"] / 1e4
    if args.n_seds > 0:
        violinplot([p for p in phot.T], pwave, phot_width / 1e4, ax=sax, **pkwargs)
        sax.plot(wave * (1 + chain[ind_best]["zred"]) / 1e4, spec_best, color=skwargs["color"], linewidth=0.5,
                 label=r"Highest probability spectrum ($z=${:3.2f})".format(chain[ind_best]["zred"][0]))

    sax.errorbar(pwave, obs["maggies"], obs["maggies_unc"], color="k", linestyle="")
    sax.plot(pwave, obs["maggies"], **tkwargs)
    sax.set_ylim(3e-9, 1e-7)
    sax.set_xlim(0.3, 5)
    sax.set_xscale("log")
    sax.set_yscale("log")
    sax.set_xticklabels([])
    #sax.legend(loc="upper left")

    if False:
        sax = show_sed(truth, phot=phot, ax=sax, masked=True, nufnu=True,
                       showtruth=True, truth_kwargs=truth_kwargs,
                       ndraw=0, quantiles=None)
        sax = show_sed(obs, phot=phot, ax=sax, masked=True, nufnu=True,
                       showtruth=True, truth_kwargs=data_kwargs,
                       ndraw=0, quantiles=None)
        rax = show_sed(obs, phot=phot, ax=rax, masked=True, nufnu=True,
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

    #cfig.legend(artists, legends, (0.77, 0.43), frameon=True)
    #cfig.savefig("paperfigures/fig{}.{}".format(args.fignum, args.ext), dpi=400)
    pl.ion()
    pl.show()