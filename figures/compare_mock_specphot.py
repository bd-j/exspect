#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""fig3.py - Phot/Spec cornerplots

This script is intended to show a corner plot of posterior PDF constraints for
a parameteric SFH inferred from
    * photometry alone,
    * spectroscopy alone, and
    * photometry + spectroscopy,
"""


import os, glob
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct
from prospect.utils.plotting import get_truths

from exspect.plotting.corner import allcorner, marginal, scatter, get_spans, corner, prettify_axes
#from sedplot import truespec
from exspect.plotting.utils import sample_prior
from exspect.plotting import plot_defaults, colorcycle


def convert(chain, labels):
    dust = 1.086 * chain[..., labels.index("dust2")]
    z = chain[..., labels.index("logzsol")]
    sfh, sfh_label = construct_sfh_measure(chain, labels)
    return np.array([sfh, z, dust]), [sfh_label, "log Z/Z$_\odot$", "A$_V$"]


def construct_sfh_measure(chain, labels):
    sfh = chain[..., labels.index("tage")] / chain[..., labels.index("tau")]
    return sfh, "$Age/\\tau$"
    #sfh = chain[..., labels.index("tage")]
    #return sfh, "$Age$" # "Age" | "$Age/\\tau$"


def rectify(res, start=0):
    chain = res["chain"][start:]
    wghts = res["weights"][start:]
    labels = res["theta_labels"]
    values, newlabels = convert(chain, labels)

    return values, newlabels, wghts


def multispan(results):
    spans = []
    for r in results:
        xx, labels, wghts = rectify(r, start=500)
        spans.append(get_spans(None, xx, weights=wghts))

    spans = np.array(spans)
    span = spans[:,:, 0].min(axis=0), spans[:,:, 1].max(axis=0)

    return tuple(np.array(span).T)


def multicorner(results, colors, smooth=0.02,
                tick_kwargs={}, max_n_ticks=3,
                label_kwargs={}, truth_kwargs={},
                hist_kwargs={}, hist2d_kwargs={}):
    span = multispan(results)
    ndim = len(span)
    pfig, paxes = pl.subplots(ndim, ndim, figsize=(15, 12.5))
    for i, r in enumerate(results):
        xx, labels, wghts = rectify(r, start=500)
        paxes = corner(xx, paxes, weights=wghts, span=span,
                       smooth=smooth, color=colors[i],
                       hist_kwargs=hist_kwargs, hist2d_kwargs=hist2d_kwargs)

    prettify_axes(paxes, labels, max_n_ticks=max_n_ticks,
                  label_kwargs=label_kwargs, tick_kwargs=tick_kwargs)

    t = convert(get_truths(r)[0], r["theta_labels"])
    tvec = np.atleast_2d(t[0]).T
    scatter(tvec, paxes, zorder=10, **truth_kwargs)

    return pfig, paxes


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
    parser.add_argument("--fignum", type=int, default=2)
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--phot_file", type=str, default="")
    parser.add_argument("--spec_file", type=str, default="")
    parser.add_argument("--specphot_file", type=str, default="")
    parser.add_argument("--n_seds", type=int, default=0)
    args = parser.parse_args()

    import matplotlib
    fmtr = matplotlib.ticker.ScalarFormatter()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib import rcParams
    from matplotlib.gridspec import GridSpec

    # --- Read-in ---
    tag = "tau4_tage12_noiseTrue_nebFalse_maskFalse_normTrue_snr20"
    v = -1
    files = [args.phot_file, args.spec_file, args.specphot_file]

    results, obs, models, sps = setup(files, sps=-1)
    data = [o[f] for (o, f) in zip(obs, ftype)]

    # --- Axes & Styles ---
    rcParams = plot_defaults(rcParams)
    #fig = pl.figure()
    #gs = GridSpec(4, 4)

    label_kwargs = {"fontsize": 14}
    tick_kwargs = {"labelsize": 12}

    prior_kwargs = {"color": colorcycle[4], "linestyle": ":", "linewidth": 2}
    hist_kwargs = {"alpha": 0.5, "histtype": "stepfilled"}
    truth_kwargs = {"color": "k", "marker": "o"}
    sed_kwargs = {"color": 'k', "linewidth": 1, "alpha":1.0}
    data_kwargs =  {"color": colorcycle[3], "linestyle": "-", "markeredgewidth": 2}

    # --- Legend stuff ---
    rtypes = ["Only Photometry", "Only Spectroscopy", "Photometry & Spectroscopy"]

    patches = [Patch(color=c, alpha=hist_kwargs["alpha"]) for c in colorcycle]
    dot = Line2D([], [], linestyle="", **truth_kwargs)
    prior = Line2D([], [], **prior_kwargs)
    artists = [dot, prior] + patches
    legends = ["True Parameters", "Prior"] + rtypes

    # --- Corner plots ---
    if True:
        t = [convert(get_truths(r)[0], r["theta_labels"])
             for r in results]
        truths = dict(zip(t[0][1], t[0][0]))
        pfig, paxes = multicorner(results, colorcycle,
                                  label_kwargs=label_kwargs, tick_kwargs=tick_kwargs,
                                  truth_kwargs=truth_kwargs, hist_kwargs=hist_kwargs)
        show_priors(models[0], np.diag(paxes), multispan(results),
                    smooth=[0.1, 0.3, 0.3], **prior_kwargs)


    # ---- Spectrum inset plot ------
    if True:
        # --- get samples ---
        #sps = reader.get_sps(result)
        #sample_kwargs = {"nsample": int(1e4), "start": 0.2,
        #                "weights": result["weights"]}
        #thetas, fluxes = posterior_sed(result["chain"], model, obs, sps, sample_kwargs=sample_kwargs)

        # --- True spectrum (normalized) ---
        #twave, tspec = truespec(obs, model, sps, R=500 * 2.35, nufnu=False)
        ii = 2
        o = obs[ii]
        tspec = o["true_spectrum"] / o["continuum"]
        twave = o["wavelength"]
        tcont = o["continuum"].copy()
        tmask = o["mask"].copy()

        #order = models[1].params["polyorder"]
        #tcal = polyopt(tspec, obs[1], order=12)[0]
        #tspec *= tcal
        #truth = {"spectrum": tspec,
        #         "wavelength": twave,
        #         "unc": np.zeros_like(tspec),
        #         "mask": slice(None)}

        norm = np.median(tspec)

        # --- plot ----
        sax = pfig.add_subplot(4, 2, 2)
        sax.plot(obs[ii]["wavelength"], obs[ii]["spectrum"] / norm, **data_kwargs)
        sax.plot(twave, tspec / norm, **sed_kwargs)

        data = Line2D([], [], **data_kwargs)
        sed = Line2D([], [], **sed_kwargs)

        artists = [sed, data] + artists
        legends = ["True Spectrum\n(Continuum Normalized)", "Mock Data"] + legends

        sax.set_ylabel("$Flux$")
        sax.set_xlabel("$\\lambda \, (\AA)$")



    pfig.subplots_adjust(hspace=0.1, wspace=0.1)
    pfig.suptitle(tag)
    pfig.legend(artists, legends, loc='upper right', bbox_to_anchor=(0.85, 0.6), frameon=True)

    pl.show()
    pfig.savefig("paperfigures/fig{}.{}".format(args.fignum, args.figext), dpi=400)
