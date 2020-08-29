#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Basic parameteric fit

This script is intended to show a corner plot of the posterior PDF and the
quality of the fit of a simple parameteric model to mock broadband photometry.
"""

import os, glob
from argparse import ArgumentParser
import numpy as np

import matplotlib.pyplot as pl
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib import rcParams, gridspec

from prospect.plotting import FigureMaker, chain_to_struct, dict_to_struct
from prospect.plotting import boxplot, get_simple_prior, sample_prior, sample_posterior
from prospect.plotting.corner import allcorner, marginal, get_spans, _quantile
from prospect.plotting.sed import to_nufnu, convolve_spec

from exspect.examples.parametric_mock_specphot import build_sps
from defaults import pretty, plot_defaults, colorcycle

rcParams = plot_defaults(rcParams)


def construct_sfh_measure(chain, agebins=None):
    """Compute SFH measures from a structured chain (or dictionary)
    """
    tage = np.atleast_1d(np.squeeze(chain["tage"]))
    tau = np.atleast_1d(np.squeeze(np.log10(chain["tau"])))
    mass = np.atleast_1d(np.squeeze(chain["mass"]))
    mass = np.log10(mass)
    return [mass, tage, tau], ["logmass", "tage", "logtau"]


class Plotter(FigureMaker):

    show = ["logmass", "logtau", "tage", "logzsol", "av"]

    def convert(self, chain, agebins=None):
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

    def plot_all(self):
        self.make_axes()
        self.styles()
        self.plot_sed(self.sax, self.rax)
        self.plot_posteriors(self.caxes)
        self.make_legend()

    def make_axes(self):
        self.fig, self.caxes = pl.subplots(len(self.show), len(self.show), figsize=(14.5, 12),
                                           gridspec_kw={"top": 0.96, "right": 0.96})
        self.fig.subplots_adjust(hspace=0.1, wspace=0.1)
        self.sax = self.fig.add_axes([0.55, 0.75, 0.96-0.55, 0.96-0.75])
        self.rax = self.fig.add_axes([0.55, 0.65, 0.96-0.55, 0.75-0.65], sharex=self.sax)

    def plot_posteriors(self, caxes):
        """Make a corner plot of selected parameters
        """
        xx = np.array([self.parchain[p] for p in self.show])
        labels = [pretty.get(p, p) for p in self.show]
        spans = get_spans(None, xx, weights=self.weights)

        truths = self.convert(dict_to_struct(self.obs["mock_params"]))
        tvec = np.array([truths[p] for p in self.show])
        caxes = allcorner(xx, labels, caxes, weights=self.weights, span=spans,
                          color=self.pkwargs["color"], hist_kwargs=self.hkwargs,
                          psamples=tvec, samples_kwargs={"color": self.tkwargs["mfc"], "edgecolor": "k"},
                          label_kwargs=self.label_kwargs,
                          tick_kwargs=self.tick_kwargs, max_n_ticks=4)
        # Plot truth
        for ax, p in zip(np.diagonal(caxes), self.show):
            ax.axvline(truths[p], marker="", **self.tkwargs)

        # plot priors
        if self.prior_samples > 0:
            self.show_priors(np.diag(caxes), spans, smooth=0.05, **self.rkwargs)

    def plot_sed(self, sax, rax):
        nufnu = self.nufnu
        wc = 10**(4 * self.nufnu)

        # --- Data ---
        owave, ophot, ounc = self.obs["phot_wave"], self.obs["maggies"], self.obs["maggies_unc"]
        phot_width = np.array([f.effective_width for f in self.obs["filters"]])
        maxw, minw = np.max(owave + phot_width) * 1.02, np.min(owave - phot_width) / 1.02
        phot_width /= wc
        if self.nufnu:
            _, ophot = to_nufnu(owave, ophot)
            owave, ounc = to_nufnu(owave, ounc)
        renorm = 1 / np.mean(ophot)

        # --- posterior samples ---
        if self.n_seds > 0:
            self.make_seds()
            ckw = dict(minw=minw, maxw=maxw, R=500*2.35, nufnu=self.nufnu)
            if self.nufnu:
                swave, spec_best = convolve_spec(self.spec_wave, [self.spec_best], **ckw)
                twave, spec_true = convolve_spec(self.spec_wave, [self.obs["true_spectrum"]], **ckw)
                spec_true = np.squeeze(spec_true)
                spec_best = np.squeeze(spec_best)
                pwave, phot_best = to_nufnu(self.obs["phot_wave"], self.phot_best)
                pwave, phot = to_nufnu(self.obs["phot_wave"], self.phot_samples)
            else:
                twave, spec_true = self.spec_wave, self.obs["true_spectrum"]
                phot, phot_best = self.phot_samples, self.phot_best

            # --- plot spec & phot samples ---
            self.bkwargs = dict(alpha=0.8,
                                facecolor=self.pkwargs["color"], edgecolor="k")
            self.art["sed_post"] = Patch(**self.bkwargs)
            widths = 0.05 * owave  # phot_width
            boxplot((phot * renorm).T, owave, widths, ax=sax, **self.bkwargs)
            sax.plot(twave, spec_true * renorm, **self.lkwargs, label=r"True spectrum")

            # --- plot residuals ---
            if rax is not None:
                chi = (ophot - phot_best) / ounc
                rax.plot(owave, chi, **self.dkwargs)
                rax.axhline(0, linestyle=":", color="black")

        # --- plot data ---
        sax.errorbar(owave, ophot * renorm, ounc * renorm, color="k", linestyle="", linewidth=2)
        sax.plot(owave, ophot * renorm, **self.dkwargs)

        # --- prettify ---
        if nufnu:
            sax.set_ylim(1.3e-13 * renorm, 0.7e-12 * renorm)
        else:
            sax.set_ylim(3e-9 * renorm, 1e-7 * renorm)
        sax.set_xlim(minw / wc, maxw / wc)
        sax.set_xscale("log")
        sax.set_yscale("log")
        sax.set_ylabel(r"$\nu f_\nu$ (Arbitrary)", fontsize=18)
        sax.set_xticklabels([])
        sax.set_yticks([0.4, 0.6, 1.0, 1.5])
        sax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.1f}"))

        rax.set_ylim(-2.8, 2.8)
        rax.set_ylabel(r"$\chi_{\rm best}$", fontsize=18)
        rax.set_xlabel(r"$\lambda_{\rm obs}$ ($\mu$m)", fontsize=18)
        wpos = [0.2, 0.5, 1.0, 1.5]
        rax.set_xticks(wpos)
        rax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:0.1f}"))

    def make_legend(self):
        artists = [self.art["spec_data"], self.art["phot_data"]]
        legends = ["True Spectrum", "Mock photometry"]

        if "sed_post" in self.art:
            artists += [self.art["sed_post"]]
            legends += ["Posterior SED"]

        artists += [self.art["truth"], self.art["prior"], self.art["phot_post"]]
        legends += ["True Parameters", "Prior", "Posterior"]
        self.fig.legend(artists, legends, loc=(0.81, 0.45), frameon=True, fontsize=14)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--fignum", type=str, default="")
    parser.add_argument("--figext", type=str, default="png")
    parser.add_argument("--results_file", type=str, default="")
    parser.add_argument("--prior_samples", type=int, default=int(1e4))
    parser.add_argument("--n_seds", type=int, default=0)
    args = parser.parse_args()

    plotter = Plotter(nufnu=True, **vars(args))
    plotter.plot_all()

    # --- Saving ----
    # ---------------
    if args.fignum:
        plotter.fig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()