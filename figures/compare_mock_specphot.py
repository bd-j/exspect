#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""fig3.py - Phot/Spec cornerplots

This script is intended to show a corner plot of posterior PDF constraints for
a parameteric SFH inferred from
    * photometry alone,
    * spectroscopy alone, and
    * photometry + spectroscopy,
"""

from copy import deepcopy
import os, glob
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl
import matplotlib.ticker as ticker

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct, dict_to_struct

from exspect.plotting.corner import marginal, scatter, get_spans, corner, prettify_axes
from exspect.plotting.utils import sample_prior
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
    sfh = chain["tage"] / chain["tau"]
    sfh = np.atleast_1d(np.squeeze(sfh))
    return [sfh], ["ageprime"]
    #sfh = chain["tage"]
    #return sfh, "$Age$" # "Age" | "$Age/\\tau$"


def multispan(params, weights, show):
    spans, xx = [], []
    for par, w in zip(params, weights):
        x = np.array([par[p] for p in show])
        spans.append(get_spans(None, x, weights=w))
        xx.append(x)
    spans = np.array(spans)
    span = spans[:,:, 0].min(axis=0), spans[:,:, 1].max(axis=0)
    span = tuple(np.array(span).T)
    return span, xx


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
    parser.add_argument("--figext", type=str, default="png")
    parser.add_argument("--phot_file", type=str, default="")
    parser.add_argument("--spec_file", type=str, default="")
    parser.add_argument("--specphot_file", type=str, default="")
    parser.add_argument("--prior_samples", type=int, default=int(1e4))
    parser.add_argument("--n_seds", type=int, default=0)
    args = parser.parse_args()

    show = ["ageprime", "logzsol", "av"]

    # --- Axes ---
    # ------------
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib import rcParams
    from matplotlib.gridspec import GridSpec
    rcParams = plot_defaults(rcParams)

    pfig, paxes = pl.subplots(len(show), len(show), figsize=(15, 12.5))
    sax = pfig.add_subplot(4, 2, 2)
    #rax = cfig.add_subplot(6, 2, 4, sharex=sax)

    # --- Legend stuff ---
    # --------------------
    label_kwargs = {"fontsize": 18}
    tick_kwargs = {"labelsize": 16}
    pkwargs = dict(color=colorcycle[0], alpha=0.65)
    skwargs = dict(color=colorcycle[1], alpha=0.65)
    akwargs = dict(color=colorcycle[3], alpha=0.65)
    dkwargs = dict(color="gray", linestyle="-", linewidth=0.75, marker="")
    rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)
    tkwargs = dict(color="gray", linestyle="", linewidth=2.0, marker="o", mfc="k", mec="k")
    lkwargs = dict(color="k", linestyle="-", linewidth=1.25, marker="")
    mkwargs = dict(alpha=0.5, histtype="stepfilled")
    hkwargs = [pkwargs, skwargs, akwargs]

    data = Line2D([], [], **dkwargs)
    truth = Line2D([], [], **tkwargs)
    posts = [Patch(**kwargs) for kwargs in hkwargs]
    prior = Line2D([], [], **rkwargs)

    # --- Read-in ---
    # ---------------
    rtypes = ["Only Photometry", "Only Spectroscopy", "Photometry & Spectroscopy"]
    files = [args.phot_file, args.spec_file, args.specphot_file]

    out = [reader.results_from(os.path.expandvars(f)) for f in files]
    chains = [chain_to_struct(result["chain"], model) for result, obs, model in out]
    weights = [result["weights"] for result, obs, model in out]
    phot, spec, specphot = [o[1] for o in out]

    result = out[-1][0]
    model = out[-1][-1]

    # --- Corner plots ---
    # --------------------
    params = [convert(chain) for chain in chains]
    spans, xx = multispan(params, weights, show)
    truths = convert(dict_to_struct(specphot["mock_params"]))
    tvec = np.array([truths[p] for p in show])
    labels = [pretty.get(p, p) for p in show]

    for i, (x, w) in enumerate(zip(xx, weights)):
        kwargs, color = deepcopy(hkwargs[i]), hkwargs[i]["color"]
        _ = kwargs.pop("color")
        kwargs["histtype"] = "stepfilled"
        paxes = corner(x, paxes, weights=w, span=spans,
                       smooth=0.02, color=color,
                       hist_kwargs=kwargs, hist2d_kwargs=mkwargs)

    scatter(tvec, paxes, zorder=10, color=tkwargs["mfc"], edgecolor="k")
    prettify_axes(paxes, labels, label_kwargs=label_kwargs, tick_kwargs=tick_kwargs)
    if args.prior_samples > 0:
        show_priors(model, np.diag(paxes), spans, nsample=args.prior_samples,
                    show=show, smooth=0.1, **rkwargs)

    # ---- Spectrum inset plot ------

    # --- True spectrum (normalized) ---
    truespec = np.atleast_2d(specphot["true_spectrum"])
    tspec = np.squeeze(truespec / specphot["continuum"])
    ind_best = np.argmax(result["lnprobability"])
    pbest = result["chain"][ind_best, :]

    renorm = 1/np.median(tspec)

    if args.n_seds > 0:
        # --- get samples ---
        raw_samples = sample_posterior(result["chain"], result["weights"], nsample=args.n_seds)
        sps = build_sps(**result["run_params"])
        sed_samples = [model.predict(p, obs=specphot, sps=sps) for p in raw_samples[:args.n_seds]]
        pphot = np.array([sed[1]*renorm for sed in sed_samples])
        pspec = np.array([sed[0]*renorm for sed in sed_samples])
        qq = np.percentile(pspec, [16, 50, 84], axis=0)
        spec_best, phot_best, mfrac_best = model.predict(pbest, obs=specphot, sps=sps)
        sax.plot(specphot["wavelength"], spec_best * renorm, **skwargs)

    m = specphot["mask"]
    sax.plot(specphot["wavelength"][m], specphot["spectrum"][m] * renorm, **dkwargs)
    sax.plot(specphot["wavelength"], tspec * renorm, **lkwargs)

    sax.set_ylabel(r"Flux (Arbitrary", fontsize=18)
    sax.set_xlabel(r"$\lambda_{\rm obs} \, (\AA)$", fontsize=18)
    miny = specphot["spectrum"].min() * 0.9
    maxy = np.median(specphot["spectrum"] * renorm) * 5
    sax.set_ylim(miny, maxy)

    sp = Line2D([], [], **lkwargs)

    artists = [sp, data, truth, prior] + posts
    legends = ["True Spectrum\n(Continuum Normalized)", "Mock Data", "True Parameters", "Prior"] + rtypes
    pfig.subplots_adjust(hspace=0.1, wspace=0.1)
    pfig.legend(artists, legends, loc='upper right', bbox_to_anchor=(0.9, 0.6), frameon=True, fontsize=14)

    # --- Saving ----
    # ---------------
    if args.fignum:
        pfig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()