#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""show_gnz11.py - show a photometric redshift fit to GNz-11
"""

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct
from prospect.sources.constants import cosmo

from exspect.plotting.utils import get_simple_prior, sample_prior, sample_posterior
from exspect.plotting.utils import violinplot, step
from exspect.plotting.sed import to_nufnu, convolve_spec
from exspect.plotting.corner import marginal, allcorner, _quantile
from exspect.plotting.sfh import ratios_to_sfrs

from exspect.examples.photoz import zred_to_agebins, build_sps
from defaults import pretty, plot_defaults, colorcycle


def sfh_quantiles(time, bins, sfrs, q=[16, 50, 84]):
    tt = bins.reshape(bins.shape[0], -1)
    ss = np.array([sfrs, sfrs]).T.reshape(bins.shape[0], -1)
    sf = np.array([np.interp(tarr, t, s, left=0, right=0) for t, s in zip(tt, ss)])
    qq = np.percentile(sf, axis=0, q=q)
    return qq


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, default="")
    parser.add_argument("--fignum", type=str, default="")
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--n_sample", type=int, default=1000)
    parser.add_argument("--n_seds", type=int, default=0)
    args = parser.parse_args()

    show = ["logzsol", "dust2", "logmass", "gas_logu", "dust_index", "igm_factor"]

    # --- Set up axes & styles ---
    # ----------------------------
    rcParams = plot_defaults(rcParams)
    fig = pl.figure(figsize=(8.5, 10.5))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(8, len(show), width_ratios=len(show) * [10],
                  height_ratios=[3, 1, 1.25, 3, 1.25, 2, 1.25, 3],
                  left=0.1, right=0.98, wspace=0.15, hspace=0.03, top=0.95, bottom=0.1)
    sax = fig.add_subplot(gs[0, :])
    rax = fig.add_subplot(gs[1, :], sharex=sax)
    zax = fig.add_subplot(gs[3, :])
    paxes = [fig.add_subplot(gs[5, i]) for i in range(len(show))]
    hax = fig.add_subplot(gs[7, :])

    pkwargs = dict(color=colorcycle[0], alpha=0.8)
    zkwargs = pkwargs
    skwargs = dict(color=colorcycle[1], linewidth=0.75)
    tkwargs = dict(color=colorcycle[3], linestyle="--", linewidth=0.75)
    rkwargs = dict(color=colorcycle[4], linestyle=":", linewidth=2)
    dkwargs = dict(marker="o", linestyle="", mec="black", markerfacecolor=colorcycle[3])
    hkwargs = pkwargs

    data = Line2D([], [], **dkwargs)
    post = Patch(**pkwargs)
    prior = Line2D([], [], **rkwargs)
    sp = Line2D([], [], )

    # -- Read in ---
    # --------------
    result, obs, model = reader.results_from(args.results_file)
    chain = chain_to_struct(result["chain"], model=model)
    weights = result["weights"]
    raw_samples = sample_posterior(result["chain"], result["weights"], nsample=args.n_sample)
    samples = chain_to_struct(raw_samples, model)
    agebins = np.array([zred_to_agebins(s["zred"], 5, 20) for s in samples])
    ind_best = np.argmax(result["lnprobability"])
    zbest = chain[ind_best]["zred"][0]

    # --- plot SED and SED posteriors ---
    # -----------------------------------
    nufnu = False
    wc = 10**(4 * nufnu)

    owave, ophot, ounc = obs["phot_wave"], obs["maggies"], obs["maggies_unc"]
    phot_width = np.array([f.effective_width for f in obs["filters"]]) / wc
    maxw, minw = np.max(owave + phot_width) * 1.02, 900.0
    if nufnu:
        _, ophot = to_nufnu(owave, ophot)
        owave, ounc = to_nufnu(owave, ounc)

    if args.n_seds > 0:
        sps = build_sps(**result["run_params"])
        sed_samples = [model.predict(p, obs=obs, sps=sps) for p in raw_samples[:args.n_seds]]
        phot = np.array([sed[1] for sed in sed_samples])
        spec = np.array([sed[0] for sed in sed_samples])

        pbest = result["chain"][ind_best, :]
        spec_best, phot_best, mfrac_best = model.predict(pbest, obs=obs, sps=sps)
        swave = sps.wavelengths * (1 + chain[ind_best]["zred"])
        if nufnu:
            swave, spec_best = convolve_spec(swave, [spec_best], R=500 * 2.35, maxw=maxw, minw=minw, nufnu=True)
            spec_best = np.squeeze(spec_best)
            pwave = obs["phot_wave"]
            _, phot = to_nufnu(pwave, phot)
            _, phot_best = to_nufnu(pwave, phot_best)

        violinplot([p for p in phot.T], owave, phot_width, ax=sax, **pkwargs)
        sax.plot(swave, spec_best, **skwargs)

    sax.errorbar(owave, ophot, ounc, linestyle="", color="black",)
    sax.plot(owave, ophot, **dkwargs)
    sax.set_xscale("log")
    sax.set_yscale("log")
    if not nufnu:
        sax.set_ylim(1e-13, 5e-10)
    sax.set_xlim(3e3 / wc, 5e4 / wc)
    artists = [data, post, sp]
    legends = [r"Data (??)", r"Posterior SED", r"MAP spectrum ($z=${:3.2f})".format(zbest)]
    sax.legend(artists, legends, loc="upper left")

    # --- SED residuals ---
    # ---------------------
    if args.n_seds > 0:
        chi = (ophot - phot_best) / ounc
        rax.plot(owave, chi, **dkwargs)#marker="o", linestyle="", color="black")
    rax.axhline(0, linestyle=":", color="black")
    rax.set_ylim(-2.8, 2.8)
    rax.set_ylabel(r"$\chi$")
    rax.set_xlabel(r"Observed wavelength ($\mu$m)")

    # --- Redshift posterior ---
    # --------------------------------
    z = np.squeeze(chain["zred"])
    marginal(z, ax=zax, weights=weights, **zkwargs)
    zp = model.config_dict["zred"]["prior"].params
    zax.set_xlim(zp["mini"], zp["maxi"])
    zax.set_xlabel("Redshift")
    zax.set_ylabel("Probability")
    q = _quantile(z, weights=weights, q=[0.16, 0.50, 0.84])
    zstr = r"$z_{{phot}}={{{:3.2f}}}^{{+{:3.2f}}}_{{-{:3.2f}}}$"
    zax.text(0.1, 0.8, zstr.format(q[1], q[2]-q[1], q[1]-q[0]), transform=zax.transAxes)

    zax.axvline(11.09, **tkwargs)

    # --- parameter posteriors ---
    # ----------------------------
    for i, p in enumerate(show):
        pax = paxes[i]
        marginal(np.squeeze(chain[p]), weights=weights, ax=pax, **pkwargs)
        xx, px = get_simple_prior(model.config_dict[p]["prior"], pax.get_xlim())
        pax.plot(xx, px * pax.get_ylim()[1] * 0.96, **rkwargs)
        pax.set_xlabel(pretty.get(p, p))
        paxes.append(pax)
    [ax.set_yticklabels("") for ax in paxes]

    # --- SFH posteriors ---
    # ----------------------------
    sfh_samples = np.array([ratios_to_sfrs(s["logmass"], s["logsfr_ratios"], agebins=a)
                            for s, a in zip(samples, agebins)])

    tlook = (10**agebins.T/1e9).T
    # show a few samples?
    #for i in range(20):
    #    t, r, s = tlook[i], sfh_samples[i], samples[i]
    #    step(*t.T, r, ax=hax, **hkwargs)

    tarr = np.linspace(tlook.min(), tlook.max(), 500)
    sq = sfh_quantiles(tarr, tlook, sfh_samples)
    median_sfh = hax.plot(tarr, sq[1, :], color=hkwargs["color"], linewidth=2)
    hax.fill_between(tarr, sq[0, :], sq[2, :], **hkwargs)

    hax.set_xlabel(r"Lookback Time (Gyr)")
    hax.set_ylabel(r"SFR (M$_\odot$/year)")

    # --- Saving ---
    # --------------
    if args.fignum:
        fig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()