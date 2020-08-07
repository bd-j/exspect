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
from exspect.plotting.utils import sample_prior, sample_posterior, sample_prior
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
filters = {"oneband": r"\nSDSS: $r$",
           "twoband": r"\nSDSS: $gr$",
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


def get_simple_prior(prior, xlim, num=1000):
    xx = np.linspace(*xlim, num=num)
    px = np.array([prior(x) for x in xx])
    px = np.exp(px)
    return xx, px / px.max()


if __name__ == "__main__":

    pl.ion()

    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, default="")
    parser.add_argument("--fignum", type=str, default="")
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--prior_samples", type=int, default=int(1e5))
    parser.add_argument("--n_seds", type=int, default=0)
    args = parser.parse_args()

    # --- Axes ---
    # ------------
    rcParams = plot_defaults(rcParams)
    fig = pl.figure(figsize=(19, 10.5))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(4, 8,
                  left=0.1, right=0.98, wspace=0.15, hspace=0.3, top=0.95, bottom=0.1)

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
    hkwargs = dict(alpha=0.5)
    pkwargs = dict(color=colorcycle[0], alpha=0.8)
    skwargs = dict(color=colorcycle[1], alpha=0.8)
    tkwargs = dict(color=colorcycle[3], linestyle="", marker="o", mec="k", linewidth=0.75)
    rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)
    lkwargs = dict(color="k", marker="", linestyle="dashed", linewidth=2)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    post = Patch(**pkwargs)
    data = Line2D([], [], **tkwargs)
    prior = Line2D([], [], **rkwargs)
    truth = Line2D([], [], **lkwargs)

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
    if True:

        params = convert(chain, agebins)
        truths = convert(dict_to_struct(obs['mock_params']), agebins)

        for i, p in enumerate(show):
            ax = caxes.flat[i]
            ax.set_xlabel(pretty.get(p, p))
            marginal(params[p], ax, weights=weights, peak=1.0,
                     histtype="stepfilled", **pkwargs)
            # Plot truth
            ax.axvline(truths[p], **lkwargs)

        if args.prior_samples > 0:
            spans = [ax.get_xlim() for ax in caxes.flat]
            show_priors(model, caxes.flat, spans, nsample=args.prior_samples,
                        smooth=0.02, show=show, **rkwargs)

        [ax.set_yticklabels([]) for ax in caxes.flat]
        artists = [post, truth, prior]
        legends = ["Posterior", "Truth", "Prior"]
        fig.legend(artists, legends, (0.78, 0.1), frameon=True)

    # --- SED plot ---
    # -----------------------
    nufnu = True
    wc = 10**(4 * nufnu)

    owave, ophot, ounc = obs["phot_wave"], obs["maggies"], obs["maggies_unc"]
    maxw = np.max(owave > 30e4) * 520e4 + np.max(owave < 30e4) * 30e4
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

    sax.plot(owave, ophot, **tkwargs)
    sax.errorbar(owave, ophot, ounc, color="k", linestyle="")

    sax.set_yscale("log")
    sax.set_xscale("log")
    sax.set_xlim(0.1, maxw/1e4)
    sax.set_xlabel(r"$\lambda_{\rm obs} (\mu{\rm m})$")
    sax.set_ylabel(r"$\nu f_\nu$")

    artists = [data, post, truth]
    legends = ["Observed Photometry", "Posterior SED", "True SED"]
    sax.legend(artists, legends, loc="lower left")
    sax.text(0.7, 0.3, filters[filterset], transform=sax.transAxes)

    # --- Saving ---
    # --------------
    if args.fignum:
        fig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()