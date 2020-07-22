#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams

from prospect.io import read_results as reader
from prospect.io.write_results import chain_to_struct
from prospect.sources.constants import cosmo

from exspect.plotting import plot_defaults
from exspect.plotting.utils import sample_posterior, violinplot, step #, pretty
from exspect.plotting.corner import marginal, allcorner, _quantile
from exspect.plotting.sfhplot import ratios_to_sfrs


from exspect.examples.photoz import zred_to_agebins, build_sps

pretty = {"logzsol": r"$\log (Z_{\star}/Z_{\odot})$",
          "logmass": r"$\log M_{\star, {\rm formed}}$",
          "gas_logu": r"$U_{\rm neb}$",
          "dust2": r"$\tau_V$",
          "dust_index": r"$\Gamma_{\rm dust}$",
          "igm_factor": r"${\rm f}_{\rm IGM}$"}


def sfh_quantiles(time, bins, sfrs, q=[16, 50, 84]):
    tt = bins.reshape(bins.shape[0], -1)
    ss = np.array([sfrs, sfrs]).T.reshape(bins.shape[0], -1)
    sf = np.array([np.interp(tarr, t, s, left=0, right=0) for t, s in zip(tt, ss)])
    qq = np.percentile(sf, axis=0, q=q)
    return qq


if __name__ == "__main__":
    pl.ion()

    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, default="")
    parser.add_argument("--fignum", type=int, default=1)
    parser.add_argument("--figext", type=str, default="pdf")
    parser.add_argument("--n_sample", type=int, default=500)
    args = parser.parse_args()

    result, obs, model = reader.results_from(args.results_file)
    chain = chain_to_struct(result["chain"], model=model)
    weights = result["weights"]

    raw_samples = sample_posterior(result["chain"], result["weights"], nsample=args.n_sample)
    samples = chain_to_struct(raw_samples, model)
    agebins = np.array([zred_to_agebins(s["zred"], 5, 20) for s in samples])
    sfh_samples = np.array([ratios_to_sfrs(s["logmass"], s["logsfr_ratios"], agebins=a)
                            for s, a in zip(samples, agebins)])

    sps = build_sps(**result["run_params"])
    sed_samples = [model.predict(p, obs=obs, sps=sps) for p in raw_samples]
    phot = np.array([sed[1] for sed in sed_samples])
    spec = np.array([sed[0] for sed in sed_samples])
    wave = sps.wavelengths

    ind_best = np.argmax(result["lnprobability"])
    pbest = result["chain"][ind_best, :]
    spec_best, phot_best, mfrac_best = model.predict(pbest, obs=obs, sps=sps)
    phot_width = np.array([f.effective_width for f in obs["filters"]])

    show = ["logzsol", "dust2", "logmass", "gas_logu", "dust_index", "igm_factor"]

    # --- set up axes ---
    rcParams = plot_defaults(rcParams)
    fig = pl.figure(figsize=(8.5, 10.5))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(8, len(show), width_ratios=len(show) * [10],
                  height_ratios=[3, 1, 1.25, 3, 1.25, 2, 1.25, 3],
                  left=0.1, right=0.98, wspace=0.15, hspace=0.03, top=0.95, bottom=0.1)

    # --- plot SED and SED posteriors ---
    post_kwargs = dict(color="tomato", alpha=0.8)
    sax = fig.add_subplot(gs[0, :])
    violinplot([p for p in phot.T], obs["phot_wave"]/1e4, phot_width/1e4, ax=sax, **post_kwargs)
    sax.plot(wave * (1 + chain[ind_best]["zred"]) / 1e4, spec_best, color="tomato", linewidth=0.5,
             label=r"Highest probability spectrum ($z=${:3.2f})".format(chain[ind_best]["zred"][0]))
    sax.errorbar(obs["phot_wave"]/1e4, obs["maggies"], obs["maggies_unc"], linestyle="", color="black",)
    sax.plot(obs["phot_wave"]/1e4, obs["maggies"], marker="o", linestyle="", mec="black",
             markerfacecolor="cornflowerblue", label="Data (??)")
    sax.set_xscale("log")
    sax.set_yscale("log")
    sax.set_ylim(1e-13, 5e-10)
    sax.set_xlim(0.3, 5)
    sax.legend(loc="upper left")

    # --- plot residuals ---
    rax = fig.add_subplot(gs[1, :], sharex=sax)
    chi = (obs["maggies"] - phot_best) / obs["maggies_unc"]
    rax.plot(obs["phot_wave"]/1e4, chi, marker="o", linestyle="", color="black")
    rax.axhline(0, linestyle=":", color="black")
    rax.set_ylim(-2.8, 2.8)
    rax.set_ylabel(r"$\chi$")
    rax.set_xlabel(r"Observed wavelength ($\mu$m)")

    # --- plot redshift posterior ---
    zkwargs = dict(color="tomato", alpha=0.8)
    zax = fig.add_subplot(gs[3, :])
    z = np.squeeze(chain["zred"])
    marginal(z, ax=zax, weights=weights, **zkwargs)
    zp = model.config_dict["zred"]["prior"].params
    zax.set_xlim(zp["mini"], zp["maxi"])
    zax.set_xlabel("Redshift")
    zax.set_ylabel("Probability")
    q = _quantile(z, weights=weights, q=[0.16, 0.50, 0.84])
    zstr = r"$z_{{phot}}={{{:3.2f}}}^{{+{:3.2f}}}_{{-{:3.2f}}}$"
    zax.text(0.1, 0.8, zstr.format(q[1], q[2]-q[1], q[1]-q[0]), transform=zax.transAxes)

    zax.axvline(11.09, color="royalblue", linestyle="--", linewidth=0.75)

    # --- plot parameter posteriors (logzsol, Gamma, Av, Mformed, gas_logu) ---
    pkwargs = dict(color="tomato", alpha=0.8, linewidth=2)
    paxes = []
    for i, p in enumerate(show):
        pax = fig.add_subplot(gs[5, i])
        marginal(np.squeeze(chain[p]), weights=weights, ax=pax, **pkwargs)
        pax.set_xlabel(pretty.get(p, p))
        paxes.append(pax)
    [ax.set_yticklabels("") for ax in paxes]

    # --- plot SFH posteriors ---
    hkwargs = dict(color="tomato", alpha=0.5, linewidth=0.5)
    hax = fig.add_subplot(gs[7, :])
    tlook = (10**agebins.T/1e9).T
    #for i in range(20):
    #    t, r, s = tlook[i], sfh_samples[i], samples[i]
    #    step(*t.T, r, ax=hax, **hkwargs)

    tarr = np.linspace(tlook.min(), tlook.max(), 500)
    sq = sfh_quantiles(tarr, tlook, sfh_samples)
    median_sfh = hax.plot(tarr, sq[1, :], color=hkwargs["color"])
    hax.fill_between(tarr, sq[0, :], sq[2, :], **hkwargs)

    hax.set_xlabel(r"Lookback Time (Gyr)")
    hax.set_ylabel("SFR")

