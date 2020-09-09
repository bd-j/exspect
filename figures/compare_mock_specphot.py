#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Phot/Spec cornerplots

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
from matplotlib import rcParams
import matplotlib.ticker as ticker

from prospect.plotting import FigureMaker, dict_to_struct
from prospect.plotting.corner import marginal, scatter, get_spans, corner, prettify_axes
from prospect.plotting.sed import to_nufnu, convolve_spec

from defaults import pretty, plot_defaults, colorcycle


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
    span = spans[:, :, 0].min(axis=0), spans[:, :, 1].max(axis=0)
    span = tuple(np.array(span).T)
    return span, xx


rcParams = plot_defaults(rcParams)


class Plotter(FigureMaker):

    show = ["ageprime", "logzsol", "av"]

    def make_axes(self):
        self.fig, self.paxes = pl.subplots(len(show), len(show), figsize=(15, 12.5))
        self.sax = self.fig.add_subplot(4, 2, 2)

    def plot_corner(self, paxes, spans):
        color = self.pkwargs["color"]
        mkwargs = dict(alpha=0.5, histtype="stepfilled")
        kwargs = dict(alpha=self.pkwargs["alpha"], histtype="stepfilled")

        truths = self.convert(dict_to_struct(self.obs["mock_params"]))
        tvec = np.array([truths[p] for p in self.show])
        labels = [pretty.get(p, p) for p in self.show]

        xx = np.array([self.parchain[p] for p in self.show])
        paxes = corner(xx, paxes, weights=self.weights, span=spans,
                       smooth=0.02, color=color,
                       hist_kwargs=kwargs, hist2d_kwargs=mkwargs)
        scatter(tvec, paxes, zorder=10, color=self.tkwargs["mfc"], edgecolor="k")
        prettify_axes(paxes, labels, label_kwargs=self.label_kwargs, tick_kwargs=self.tick_kwargs)
        if self.prior_samples > 0:
            self.show_priors(np.diag(paxes), spans, smooth=0.05, **self.rkwargs)

    def convert(self, chain):
        """compute quantities from a structured chain (or dictionary)
        """
        cols = ["av", "logzsol"]
        sfh, sfh_label = construct_sfh_measure(chain, None)
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

    def plot_spectrum(self, sax):
        # --- True spectrum (normalized) ---
        truespec = np.atleast_2d(self.obs["true_spectrum"])
        tspec = np.squeeze(truespec / self.obs["continuum"])
        wave = self.obs["wavelength"].copy()

        renorm = 1 / np.median(tspec)

        if args.n_seds > 0:
            self.make_seds()
            # --- get samples ---

            pspec = self.spec_samples * renorm
            qq = np.percentile(pspec, [16, 50, 84], axis=0)
            sax.plot(wave, self.spec_best * renorm, **self.skwargs)

        m = self.obs["mask"]
        # data
        data = self.obs["spectrum"].copy()
        sax.plot(wave[m], data[m] * renorm, **self.dkwargs)
        # truth
        sax.plot(wave, tspec * renorm, **self.lkwargs)

        # --- prettify ---
        sax.set_ylabel(r"Flux (Arbitrary)", fontsize=18)
        sax.set_xlabel(r"$\lambda_{\rm obs} \, (\AA)$", fontsize=18)
        miny, maxy = (data * renorm).min() * 0.9, np.median(data * renorm) * 5
        sax.set_ylim(miny, maxy)

    def make_legend(self, fig, eleg=[], eart=[], loc=(0.91, 0.62), fontsize=14):
        legends = ["True Spectrum\n(Continuum Normalized)", "Mock Data", "True Parameters", "Prior"]
        artists = [self.art["spec_data"], self.art["phot_data"], self.art["truth"], self.art["prior"]]
        legends += eleg
        artists += eart

        fig.legend(artists, legends, loc='upper right', bbox_to_anchor=loc, frameon=True, fontsize=fontsize)


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

    show = Plotter.show

    # instantiate the plotters
    plotters = [Plotter(results_file=f, **vars(args)) for f in
                [args.phot_file, args.spec_file, args.specphot_file]]

    # make styles with different color for each plotter
    [p.styles() for p in plotters]
    cind = [0, 1, 3]
    for i, p in enumerate(plotters):
        p.pkwargs["color"] = colorcycle[cind[i]]
        p.dkwargs = dict(color="gray", linestyle="-", linewidth=0.75, marker="")
        p.make_art()
        if i < 2:
            p.prior_samples = 0

    # get limits for the plots
    spans, xx = multispan([p.parchain for p in plotters], [p.weights for p in plotters], show)

    # make and fill figures
    fig, paxes = pl.subplots(len(show), len(show), figsize=(9.6, 8.25))
    sax = fig.add_subplot(4, 2, 2)
    [p.plot_corner(paxes, spans) for p in plotters]
    plotters[-1].plot_spectrum(sax)

    leg = ["Only Photometry", "Only Spectroscopy", "Phot. & Spec."]
    art = [p.art["posterior"] for p in plotters]
    plotters[-1].make_legend(fig, eleg=leg, eart=art, fontsize=12)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)


    # --- Saving ----
    # ---------------
    if args.fignum:
        fig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()