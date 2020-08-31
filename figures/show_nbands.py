#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""show_nbands.py

This script is intended to show some posteriors and the SED for a fit to a
variable number of photometric bands
"""

from argparse import ArgumentParser
import numpy as np

import matplotlib.pyplot as pl
from matplotlib import gridspec, rcParams
from matplotlib import ticker

from prospect.plotting import FigureMaker, chain_to_struct, dict_to_struct
from prospect.plotting import boxplot
from prospect.plotting.corner import marginal
from prospect.plotting.sed import to_nufnu, convolve_spec
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


def construct_sfh_measure(chain, agebins):
    logmass = np.squeeze(chain["logmass"])
    lm, sr = np.atleast_2d(chain["logmass"]), np.atleast_2d(chain["logsfr_ratios"])
    age = nonpar_mwa(lm, sr, agebins=agebins)
    sfr = nonpar_recent_sfr(lm, sr, agebins, sfr_period=0.1)
    ssfr = np.log10(sfr) - logmass
    return [age, ssfr], ["mwa", "ssfr"]


class Plotter(FigureMaker):

    show = ["logmass", "ssfr", "logzsol", "mwa",
            "av", "av_bc", "dust_index",
            "duste_umin", "duste_qpah", "duste_gamma",
            "log_fagn", "agn_tau"]

    @property
    def agebins(self):
        return self.model.params["agebins"]

    def convert(self, chain):
        """Convert a chain (as structured ndarray) to structured array of derived
        parameters.
        """
        cols = ["logmass", "logzsol", "gas_logu", "gas_logz",
                "av", "av_bc", "dust_index",
                "duste_umin", "duste_qpah", "duste_gamma",
                "log_fagn", "agn_tau"]

        sfh, sfh_label = construct_sfh_measure(chain, self.agebins)
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

    def plot_all(self):
        self.make_axes()
        self.styles()
        self.lkwargs["linewidth"] = 2
        self.tkwargs["marker"] = ""
        self.make_art()

        self.plot_post(self.caxes)
        self.plot_sed(self.sax)
        self.make_legend(self.sax)

    def make_axes(self):

        fig = pl.figure(figsize=(15, 8.3))
        from matplotlib.gridspec import GridSpec
        gs = gridspec.GridSpec(4, 8, wspace=0.2, hspace=0.65,
                               left=0.1, right=0.98, top=0.95, bottom=0.1)

        caxes = [fig.add_subplot(gs[0, 4+i]) for i in range(4)]
        caxes += [fig.add_subplot(gs[1, 4+i]) for i in range(3)]
        caxes += [fig.add_subplot(gs[2, 4+i]) for i in range(3)]
        caxes += [fig.add_subplot(gs[3, 4+i]) for i in range(2)]
        self.caxes = np.array(caxes)
        self.fig = fig
        self.sax = self.fig.add_subplot(gs[:4, :4])

    def set_lims(self, caxes):
        caxes[0].set_xlim(8.9, 11.4)
        caxes[1].set_xlim(-14.9, -8.1)
        caxes[2].set_xlim(-2, 0.18)
        caxes[3].set_xlim(0.1, 13.7)

        caxes[4].set_xlim(0, 4)
        caxes[5].set_xlim(0, 5)
        caxes[6].set_xlim(-1, 0.4)

        caxes[7].set_xlim(0.5, 25)
        caxes[8].set_xlim(0.5, 7)
        caxes[9].set_xlim(0.001, 0.10)

        caxes[10].set_xlim(-5, 0.0)
        caxes[11].set_xlim(5, 120)

    def plot_post(self, caxes, lfactor=1.75):
        truths = self.convert(dict_to_struct(self.obs['mock_params']))

        for i, p in enumerate(self.show):
            ax = caxes.flat[i]
            ax.set_xlabel(pretty.get(p, p))
            marginal(self.parchain[p], ax, weights=self.weights,
                     peak=1.0, histtype="stepfilled", **self.pkwargs)
            # Plot truth
            ax.axvline(truths[p], **self.tkwargs)

        peak = np.ones(len(self.show)) * 0.96
        peak[self.show.index("duste_gamma")] = lfactor
        peak[self.show.index("agn_tau")] = lfactor
        self.set_lims(caxes)
        if self.prior_samples > 0:
            spans = [ax.get_xlim() for ax in caxes.flat]
            self.show_priors(caxes.flat, spans, peak=peak, smooth=0.02, **self.rkwargs)

        # --- prettify ---
        [ax.set_yticklabels([]) for ax in caxes.flat]

    def plot_sed(self, sax):
        """ Plot the SED: data and posterior predictions
        """
        wc = 10**(4 * self.nufnu)

        # --- Photometric data ---
        owave, ophot, ounc = self.obs["phot_wave"], self.obs["maggies"], self.obs["maggies_unc"]
        maxw = np.max(owave > 10e4) * 520e4 + np.max(owave < 10e4) * 30e4
        minw = 900
        if self.nufnu:
            _, ophot = to_nufnu(owave, ophot)
            owave, ounc = to_nufnu(owave, ounc)

        truespec = np.atleast_2d(self.obs["true_spectrum"])

        # --- posterior samples ---
        if args.n_seds > 0:
            self.make_seds()
            self.spec_wave = self.sps.wavelengths * (1 + self.model.params["zred"])
            ckw = dict(minw=minw, maxw=maxw, R=500*2.35, nufnu=self.nufnu)
            swave, sspec = convolve_spec(self.spec_wave, self.spec_samples, **ckw)
            twave, tspec = convolve_spec(self.spec_wave, truespec, **ckw)

            qq = np.percentile(sspec, [16, 50, 84], axis=0)
            sax.fill_between(swave, qq[0, :], qq[-1, :], **self.pkwargs)
            sax.plot(twave, tspec[0], **self.lkwargs)

        # --- plot data ---
        sax.plot(owave, ophot, **self.dkwargs)
        sax.errorbar(owave, ophot, ounc, color="k", linestyle="")

        # --- prettify ---
        sax.set_yscale("log")
        sax.set_xscale("log")
        sax.set_xlim(1300./wc, maxw/wc)
        sax.set_xlabel(r"$\lambda_{\rm obs} (\mu{\rm m})$")
        sax.set_ylabel(r"$\nu f_\nu$")
        if self.nufnu:
            sax.set_ylim(1e-15, 1e-11)

    def make_legend(self, sax):
        # posterior
        artists = self.art["truth"], self.art["prior"], self.art["posterior"]
        legends = ["True Parameters", "Prior", "Posterior"]
        self.fig.legend(artists, legends, (0.78, 0.1), frameon=True)

        # sed
        filterset = self.result["run_params"]["filterset"]
        artists = [self.art["spec_data"], self.art["phot_data"], self.art["posterior"]]
        legends = ["True SED", "Observed Photometry", "Posterior SED"]
        sax.legend(artists, legends, loc="lower left")
        sax.text(0.58, 0.3, filters[filterset], transform=sax.transAxes,
                 verticalalignment="top", fontsize=20)
        [item.set_fontsize(22) for item in [sax.xaxis.label, sax.yaxis.label]]


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

    plotter = Plotter(nufnu=True, **vars(args))
    plotter.plot_all()

    # --- Saving ---
    # --------------
    if args.fignum:
        plotter.fig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()