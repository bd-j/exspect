#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""show_nbands.py

This script is intended to show some posteriors and the SED for a fit to a
variable number of photometric bands
"""


import os, glob
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import gridspec, rcParams

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct, dict_to_struct

from exspect.examples.nband import build_sps
from exspect.plotting.utils import sample_prior, sample_posterior, get_simple_prior
from exspect.plotting.sed import convolve_spec, to_nufnu
from exspect.plotting.corner import marginal
from exspect.plotting.sfh import nonpar_recent_sfr, nonpar_mwa

from defaults import pretty, plot_defaults, colorcycle


none = r""
sdss = r"SDSS: $ugriz$"
tmass = r"2MASS: $JHK_s$"
galex = r"GALEX: FUV, NUV"
wise = r"WISE: W1,W2,W3,W4"
herschel = r"Herschel/PACS: 70,100,160"
filters = {"oneband": "\nSDSS: $r$",
           "twoband": "\nSDSS: $gr$",
           "optical": "\n".join([none, sdss]),
           "opt_nir": "\n".join([none, sdss, tmass]),
           "uv_to_nir": "\n".join([galex, sdss, tmass]),
           "uv_to_mir": "\n".join([galex, sdss, tmass, wise]),
           "full": "\n".join([galex, sdss, tmass, wise, herschel])}


def convert(chain, agebins):
    """Convert a chain (as structured ndarray) to structured array of derived
    parameters.
    """
    cols = ["logmass", "logzsol", "gas_logu", "gas_logz",
            "av", "av_bc", "dust_index",
            "duste_umin", "duste_qpah", "duste_gamma",
            "log_fagn", "agn_tau"]

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
    params["av_bc"] = params["av"] * np.squeeze(1 + chain["dust_ratio"])

    # --- agn ---
    params["log_fagn"] = np.squeeze(np.log10(chain["fagn"]))

    # --- stellar ---
    for i, c in enumerate(sfh_label):
        params[c] = np.squeeze(sfh[i])
    return params


def construct_sfh_measure(chain, agebins):
    logmass = np.squeeze(chain["logmass"])
    lm, sr = np.atleast_2d(chain["logmass"]), np.atleast_2d(chain["logsfr_ratios"])
    age = nonpar_mwa(lm, sr, agebins=agebins)
    sfr = nonpar_recent_sfr(lm, sr, agebins, sfr_period=0.1)
    ssfr = np.log10(sfr) - logmass
    return [age, ssfr], ["mwa", "ssfr"]


def show_priors(model, diagonals, spans, show=[], smooth=0.1, nsample=int(1e4),
                color="g", **linekwargs):
    """
    """
    samples, _ = sample_prior(model, nsample=nsample)
    priors = chain_to_struct(samples, model)
    params = convert(priors, model.params["agebins"])
    smooth = np.zeros(len(diagonals)) + np.array(smooth)
    for i, p in enumerate(show):
        ax = diagonals[i]
        if p in priors.dtype.names:
            x, y = get_simple_prior(model.config_dict[p]["prior"], spans[i])
            ax.plot(x, y * ax.get_ylim()[1] * 0.96, color=color, **linekwargs)
        else:
            marginal(params[p], ax, span=spans[i], smooth=smooth[i],
                     color=color, histtype="step", peak=ax.get_ylim()[1], **linekwargs)


def set_lims(caxes):
    caxes[0].set_xlim(8.9, 11.4)
    caxes[1].set_xlim(-14.9, -8.1)
    caxes[2].set_xlim(-2, 0.18)
    caxes[3].set_xlim(0.1, 13.7)

    caxes[4].set_xlim(0, 4)
    caxes[5].set_xlim(0, 5)
    caxes[6].set_xlim(-1, 0.4)

    caxes[7].set_xlim(0.5, 25)
    caxes[8].set_xlim(0.5,7)
    caxes[9].set_xlim(0.001, 0.10)

    caxes[10].set_xlim(-5, 0.0)
    caxes[11].set_xlim(5, 120)


if __name__ == "__main__":

    pl.ion()

    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, default="")
    parser.add_argument("--fignum", type=str, default="")
    parser.add_argument("--figext", type=str, default="png")
    parser.add_argument("--prior_samples", type=int, default=int(1e5))
    parser.add_argument("--n_seds", type=int, default=0)
    args = parser.parse_args()

    # --- Axes ---
    # ------------
    rcParams = plot_defaults(rcParams)
    rcParams.update({'font.size': 15})
    rcParams.update({'xtick.labelsize': 12})
    rcParams.update({'ytick.labelsize': 12})
    fig = pl.figure(figsize=(15, 8.3))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 8,
                  left=0.1, right=0.98, wspace=0.2, hspace=0.65, top=0.95, bottom=0.1)

    caxes = [fig.add_subplot(gs[0, 4+i]) for i in range(4)]
    caxes += [fig.add_subplot(gs[1, 4+i]) for i in range(3)]
    caxes += [fig.add_subplot(gs[2, 4+i]) for i in range(3)]
    caxes += [fig.add_subplot(gs[3, 4+i]) for i in range(2)]
    caxes = np.array(caxes)
    sax = fig.add_subplot(gs[:4, :4])

    # --- Styles & Legend ---
    # -----------------------
    label_kwargs = {"fontsize": 14}
    tick_kwargs = {"labelsize": 10}
    pkwargs = dict(color=colorcycle[0], alpha=0.65)
    dkwargs = dict(mfc=colorcycle[3], marker="o", linestyle="", mec="black", markersize=10, mew=2)
    rkwargs = dict(color=colorcycle[4], linestyle="--", linewidth=2)
    lkwargs = dict(color="black", marker="", linestyle="-", linewidth=2)
    tkwargs = dict(color="black", linestyle="--", linewidth=2, marker="")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    post = Patch(**pkwargs)
    data = Line2D([], [], **dkwargs)
    prior = Line2D([], [], **rkwargs)
    truth_sed = Line2D([], [], **lkwargs)
    truth_par = Line2D([], [], **tkwargs)

    show = ["logmass", "ssfr", "logzsol", "mwa",
            "av", "av_bc", "dust_index",
            "duste_umin", "duste_qpah", "duste_gamma",
            "log_fagn", "agn_tau"]

    # -- Read in ---
    # --------------
    result, obs, model = reader.results_from(args.results_file)
    chain = chain_to_struct(result["chain"], model=model)
    weights = result["weights"]
    agebins = model.params["agebins"]
    filterset = result["run_params"]["filterset"]

    # --- Marginal plots ---
    # ----------------------
    params = convert(chain, agebins)
    truths = convert(dict_to_struct(obs['mock_params']), agebins)

    for i, p in enumerate(show):
        ax = caxes.flat[i]
        ax.set_xlabel(pretty.get(p, p))
        marginal(params[p], ax, weights=weights, peak=1.0,
                 histtype="stepfilled", **pkwargs)
        # Plot truth
        ax.axvline(truths[p], **tkwargs)

    set_lims(caxes)
    if args.prior_samples > 0:
        spans = [ax.get_xlim() for ax in caxes.flat]
        show_priors(model, caxes.flat, spans, nsample=args.prior_samples,
                    smooth=0.02, show=show, **rkwargs)

    [ax.set_yticklabels([]) for ax in caxes.flat]
    artists = [truth_par, prior, post]
    legends = ["True Parameters", "Prior", "Posterior"]
    fig.legend(artists, legends, (0.78, 0.1), frameon=True)

    # --- SED plot ---
    # -----------------------
    nufnu = True
    wc = 10**(4 * nufnu)

    owave, ophot, ounc = obs["phot_wave"], obs["maggies"], obs["maggies_unc"]
    maxw = np.max(owave > 10e4) * 520e4 + np.max(owave < 10e4) * 30e4
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
        truespec = np.atleast_2d(obs["true_spectrum"])

        wave = sps.ssp.wavelengths * (1 + model.params["zred"])
        swave, sspec = convolve_spec(wave, spec, R=500*2.35, nufnu=nufnu, maxw=maxw)
        twave, tspec = convolve_spec(wave, truespec, R=500*2.35, nufnu=nufnu, maxw=maxw)

        qq = np.percentile(sspec, [16, 50, 84], axis=0)
        sax.fill_between(swave, qq[0, :], qq[-1, :], **pkwargs)
        sax.plot(twave, tspec[0], **lkwargs)

    sax.plot(owave, ophot, **dkwargs)
    sax.errorbar(owave, ophot, ounc, color="k", linestyle="")

    sax.set_yscale("log")
    sax.set_xscale("log")
    sax.set_xlim(0.13, maxw/1e4)
    sax.set_xlabel(r"$\lambda_{\rm obs} (\mu{\rm m})$")
    sax.set_ylabel(r"$\nu f_\nu$")
    if nufnu:
        sax.set_ylim(1e-15, 1e-11)

    artists = [truth_sed, data, post]
    legends = ["True SED", "Observed Photometry", "Posterior SED"]
    sax.legend(artists, legends, loc="lower left")
    sax.text(0.58, 0.3, filters[filterset], transform=sax.transAxes,
             verticalalignment="top", fontsize=20)
    [item.set_fontsize(22) for item in [sax.xaxis.label, sax.yaxis.label]]

    # --- Saving ---
    # --------------
    if args.fignum:
        fig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()