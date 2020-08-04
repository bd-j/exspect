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
from prospect.io.write_results import chain_to_struct

from exspect.examples.nband import build_sps
from exspect.plotting import plot_defaults, colorcycle
from exspect.plotting.utils import sample_prior, sample_posterior, convolve_spec
from exspect.plotting.corner import marginal, get_spans
from exspect.plotting.sfhplot import show_sfh, nonpar_recent_sfr, nonpar_mwa


pretty = {"logzsol": r"$\log (Z_{\star}/Z_{\odot})$",
          "logmass": r"$\log {\rm M}_{\star, {\rm formed}}$",
          "gas_logu": r"${\rm U}_{\rm neb}$",
          "gas_logz": r"$\log (Z_{\neb}/Z_{\odot})$",
          "dust2": r"$\tau_V$",
          "av": r"$A_{V, diffuse}$",
          "av_bc": r"$A_{V, young}$",
          "dust_index": r"$\Gamma_{\rm dust}$",
          "igm_factor": r"${\rm f}_{\rm IGM}$",
          "duste_umin": r"$U_{min, dust}$",
          "duste_qpah": r"$Q_{PAH}$",
          "duste_gamma": r"$\gamma_{dust}$",
          "log_fagn": r"log(AGN fraction)",
          "agn_tau": r"$\\tau_{AGN}$",
          "mwa": r"$\langle t \\rangle_M$",
          "ssfr": r"$\log (sSFR)"}


def convert(chain, agebins):
    """Convert a chain (as structured ndarray) to structured array of derivative parameters.
    """

    cols = ["logzsol", "gas_logu", "gas_logz",
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
            params[c] = chain[c]

    # --- dust attenuation
    params["av"] = 1.086 * chain["dust2"]
    params["av_bc"] = params["av"] * (1 + chain["dustratio"])

    # --- agn ---
    params["log_fagn"] = np.log10(chain["fagn"])

    # --- stellar ---
    for i, c in enumerate(sfh_label):
        params[c] = sfh[i]
    return params


def construct_sfh_measure(chain, agebins):
    logmass = np.squeeze(chain["logmass"])
    age = nonpar_mwa(chain["logmass"], chain["logsfr_ratios"], agebins=agebins)
    sfr = nonpar_recent_sfr(chain["logmass"], chain["logsfr_ratios"], agebins, sfr_period=0.1)
    ssfr = np.log10(sfr) - logmass
    return [age, ssfr], ["mwa", "ssfr"]


def show_priors(model, diagonals, spans, smooth=0.1, nsample=int(1e4),
                color="g", **linekwargs):
    """
    """
    ps, _ = convert(sample_prior(model, nsample=nsample)[0], model)
    smooth = np.zeros(len(diagonals)) + np.array(smooth)
    for i, (x, ax) in enumerate(zip(ps, diagonals)):
        marginal(x, ax, span=spans[i], smooth=smooth[i],
                 color=color, histtype="step", peak=ax.get_ylim()[1], **linekwargs)


if __name__ == "__main__":

    pl.ion()

    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, default="")
    parser.add_argument("--fignum", type=int, default=8)
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--n_sample", type=int, default=1000)
    parser.add_argument("--n_seds", type=int, default=0)
    args = parser.parse_args()

    # -- Read in ---
    result, obs, model = reader.results_from(args.results_file)
    chain = chain_to_struct(result["chain"], model=model)
    weights = result["weights"]
    agebins = model.params["agebins"]

    show = ["logmass", "ssfr", "logzsol", "mwa",
            "av", "av_bc", "dust_index",
            "duste_umin", "duste_qpah", "duste_gamma",
            "log_fagn", "agn_tau"]
    # --- Set up axes ---
    rcParams = plot_defaults(rcParams)
    fig = pl.figure(figsize=(8.5, 10.5))
    from matplotlib.gridspec import GridSpec

    rcParams["errorbar.capsize"] = 5

    # --- set up styles ---
    label_kwargs = {"fontsize": 14}
    tick_kwargs =  {"labelsize": 10}
    hkwargs = dict(alpha=0.5)
    pkwargs = dict(color=colorcycle[0], alpha=0.8)
    skwargs = dict(color=colorcycle[1], alpha=0.8)
    tkwargs = dict(color=colorcycle[3], linestyle="", marker="o", mec="k", linewidth=0.75)
    rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)

    #truth_kwargs = {"color": "k", "marker": "", "linestyle": "dashed", "linewidth": 2}
    #draw_kwargs =  dict(color="r", linewidth=1.0, alpha=0.5)
    #sed_kwargs =   {"color": 'k', "linewidth": 1, "alpha": 0.5}
    #data_kwargs =  {"color": "royalblue", "markerfacecolor": "none", "marker": "o",
    #                "markersize": 6, "linestyle": "", "markeredgewidth": 2}

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    post = Patch(color=colorcycle[0], alpha=hkwargs["alpha"])
    phot_data = Line2D([], [], **data_kwargs)
    prior = Line2D([], [], **prior_kwargs)

    # --- Marginal plots ---
    if True:

        xx = convert(chain, agebins)
        spans = get_spans(None, xx, weights=weights)

        truths = convert(obs['mock_params'], agebins)
        truth = Line2D([], [], **tkwargs)

        cfig, caxes = pl.subplots(4, 4, figsize=(16, 13))

        for i, p in enumerate(show):
            ax = caxes.flat[i]
            ax.set_xlabel(pretty.get(p, p))
            marginal(xx[p], ax, weights=weights, peak=1.0,
                     histtype="stepfilled", **pkwargs)
            # Plot truth
            ax.axvline(truths[p], **tkwargs)
        # Plot prior
        show_priors(model, caxes.flat, spans, nsample=int(1e5),
                    smooth=0.02, **rkwargs)

        [ax.set_yticklabels([]) for ax in caxes.flat]
        cfig.subplots_adjust(hspace=0.3)

        artists = [post, truth, prior]
        legends = ["Posterior", "Truth", "Prior"]

        cfig.legend(artists, legends, (0.77, 0.43), frameon=True)
        cfig.savefig("figures/nband_{}_onecorner.pdf".format(len(obs["maggies"])))
        cfig.savefig("paperfigures/fig8a.pdf")

    # ---- SED inset plot ------
    if args.n_seds > 0:
        # --- get samples ---
        raw_samples = sample_posterior(result["chain"], result["weights"], nsample=args.n_seds)
        sps = build_sps(**result["run_params"])
        sed_samples = [model.predict(p, obs=obs, sps=sps) for p in raw_samples[:args.n_seds]]
        phot = np.array([sed[1] for sed in sed_samples])
        spec = np.array([sed[0] for sed in sed_samples])

        wave = sps.ssp.wavelengths * (1 + model.params["zred"])
        swave, smoothed_spec = convolve_spec(wave, spec, R=500*2.35)
        twave, tspec = truespec(obs, model, sps, R=500 * 2.35, nufnu=False)

        # --- plot ----
        true_obs = {"maggies": obs["true_maggies"].copy(),
                    "wavelength": twave.copy(),
                    "spectrum": tspec.copy(),
                    "unc": np.zeros_like(twave),
                    "mask": slice(None),
                    "filters": obs["filters"],
                    "maggies_unc": np.zeros_like(obs["maggies_unc"]),
                    "phot_mask": slice(None)}
        truth_kwargs = {"color": "k", "marker": "", "linewidth": 1}
        truth = Line2D([],[], **tkwargs)

        #sax.plot(twave, tspec, **sed_kwargs)
        #sax.scatter(obs["phot_wave"], obs["true_maggies"], **truth_kwargs)
        sax = show_sed(true_obs, spec=smoothed_spec, ax=sax, masked=True, nufnu=True,
                       showtruth=True, truth_kwargs=truth_kwargs,
                       ndraw=0, post_kwargs=post_kwargs)
        sax = show_sed(obs, phot=phot, ax=sax, masked=True, nufnu=True,
                       showtruth=True, truth_kwargs=data_kwargs,
                       ndraw=0, quantiles=None)

        xmin, xmax = np.min(twave), np.max(twave)
        ymin, ymax = 1e-14, 1e-11 #tspec.min()*0.9, tspec.max()/0.9
        sax.set_xscale("log")
        sax.set_yscale("log")
        sax.set_xlim(xmin, xmax)
        sax.set_ylim(ymin, ymax)
        sax.set_xticklabels([])
        sax.set_ylabel("$\\nu \, f_\\nu \, (erg/s/cm^2)$")

        data = Line2D([], [], **data_kwargs)
        draws = Line2D([], [], **draw_kwargs)

        artists = [post, truth, phot_data]
        legends = ["Posterior", "Truth", "Mock Data"]

        sfig.legend(artists, legends, (0.7, 0.13), frameon=True)
        sfig.suptitle(tag)
        sfig.savefig("figures/nband_{}_onesed.pdf".format(len(obs["maggies"])))
        sfig.savefig("paperfigures/fig8b.pdf")
        pl.show()
