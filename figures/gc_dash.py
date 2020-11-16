#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" gc_dash.py - dashboard for individual fits to GGC spectra + photometry
"""

from copy import deepcopy
from argparse import ArgumentParser
import numpy as np

import matplotlib.pyplot as pl
from matplotlib import rcParams, gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import ticker

from prospect.plotting import FigureMaker, chain_to_struct, boxplot, sample_posterior
from prospect.plotting.corner import allcorner, marginal, scatter, get_spans
from prospect.plotting.sed import to_nufnu, convolve_spec
from prospect.plotting.sfh import ratios_to_sfrs, sfh_quantiles

from defaults import pretty, plot_defaults, colorcycle

rcParams = plot_defaults(rcParams)


class Plotter(FigureMaker):

    show = ["tage", "logzsol", "av"]

    def convert(self, chain):
        """compute quantities from a structured chain (or dictionary)
        """
        niter = len(chain)
        cols = ["av", "logzsol", "tage"]
        dt = np.dtype([(c, np.float) for c in cols])
        params = np.zeros(niter, dtype=dt)

        for c in cols:
            if c in chain.dtype.names:
                params[c] = np.squeeze(chain[c])

        # --- dust attenuation
        params["av"] = np.squeeze(1.086 * chain["dust2"])

        return params

    def plot_all(self):
        self.make_axes()
        self.styles()
        self.pkwargs["color"] = colorcycle[3]
        self.make_art()
        self.skwargs = dict(color=colorcycle[1], alpha=0.65)
        self.art["spec_post"] = Patch(**self.skwargs)

        self.plot_corner(self.caxes)
        if self.n_seds >= 0:
            self.make_seds(full=True)
        self.plot_sed(self.sax, self.rax, lax=self.lax)
        self.make_legends(self.sax)

    def make_axes(self, figsize=(14., 7)):

        N = len(self.show)

        fig = pl.figure(figsize=figsize)
        gsc = gridspec.GridSpec(N, N, left=0.58, right=0.95, hspace=0.05, wspace=0.05)
        gss = gridspec.GridSpec(4, 1, left=0.08, right=0.5)
        caxes = [fig.add_subplot(gsc[i, j]) for i in range(N) for j in range(N)]
        self.caxes = np.array(caxes).reshape(N, N)
        self.sax = fig.add_subplot(gss[:2, 0])
        self.rax = fig.add_subplot(gss[2, 0])
        self.lax = fig.add_subplot(gss[3, 0])
        self.fig = fig

    def plot_corner(self, caxes):
        """Make a corner plot of selected parameters
        """
        xx = np.array([self.parchain[p] for p in self.show])
        labels = [pretty.get(p, p) for p in self.show]
        spans = get_spans(None, xx, weights=self.weights)

        caxes = allcorner(xx, labels, caxes, weights=self.weights, span=spans,
                          color=self.pkwargs["color"], hist_kwargs=self.hkwargs,
                          label_kwargs=self.label_kwargs,
                          tick_kwargs=self.tick_kwargs, max_n_ticks=4)

    def plot_sed(self, sax, rax, lax=None,
                 nufnu=True, microns=True, normalize=True,
                 fullsed=True, q=[16, 50, 84]):
        """Inset plot of SED
        """
        wc = 10**(4 * microns)

        # --- Data ---
        pmask = self.obs["phot_mask"]
        ophot, ounc = self.obs["maggies"][pmask], self.obs["maggies_unc"][pmask]
        owave = np.array([f.wave_effective for f in self.obs["filters"]])[pmask]
        phot_width = np.array([f.effective_width for f in self.obs["filters"]])[pmask]
        maxw, minw = np.max(owave + phot_width) * 1.02, np.min(owave - phot_width) / 1.02
        phot_width /= wc
        if nufnu:
            _, ophot = to_nufnu(owave, ophot, microns=microns)
            owave, ounc = to_nufnu(owave, ounc, microns=microns)
        if normalize:
            renorm = 1. / np.mean(ophot)
        else:
            renorm = 1.

        # --- plot phot data ---
        sax.plot(owave, ophot * renorm, zorder=20, **self.dkwargs)
        sax.errorbar(owave, ophot * renorm, ounc * renorm, zorder=20, color="k", linestyle="")

        # --- posterior samples ---
        self.spec_wave = self.sps.wavelengths * (1 + self.chain["zred"][self.ind_best])
        ckw = dict(minw=minw, maxw=maxw, R=500*2.35, nufnu=nufnu, microns=microns)
        swave, ssed = convolve_spec(self.spec_wave, self.sed_samples, **ckw)
        phot, phot_best = self.phot_samples, self.phot_best
        if nufnu:
            _, phot_best = to_nufnu(self.obs["phot_wave"], self.phot_best, microns=microns)
            _, phot = to_nufnu(self.obs["phot_wave"], self.phot_samples, microns=microns)

        # Photometry posterior as boxes
        self.bkwargs = dict(alpha=0.8, facecolor=colorcycle[0], edgecolor="k")
        self.art["sed_post"] = Patch(**self.bkwargs)
        widths = 0.05 * owave  # phot_width
        boxplot((phot * renorm).T, owave, widths, ax=sax, **self.bkwargs)

        # SED posterior, full wavelength range
        qq = np.percentile(ssed * renorm, axis=0, q=q)
        sax.fill_between(swave, qq[0, :], qq[-1, :], **self.skwargs)

        # best fit SED, but only where data exist
        sed_best = np.atleast_2d(self.spec_best / self.cal_best)
        dwave = self.obs["wavelength"]
        m = self.obs["mask"]
        ckw["minw"], ckw["maxw"] = dwave[m].min(), dwave[m].max()
        bwave, sed_best = convolve_spec(dwave, sed_best, **ckw)
        sed_best = np.interp(dwave / wc, bwave, np.squeeze(sed_best))
        sax.plot(dwave[m] / wc, sed_best[m] * renorm, **self.lkwargs)

        # --- Residual ---
        schi_best = (self.spec_best - self.obs["spectrum"]) / self.obs["unc"]
        pchi_best = (phot_best - ophot) / ounc
        self.xkwargs = dict(linestyle="-", linewidth=0.5, alpha=1, color=self.skwargs["color"])
        self.art["spec_residual"] = Line2D([], [], **self.xkwargs)
        rax.plot(dwave[m] / wc, schi_best[m], **self.xkwargs)
        rax.plot(owave, pchi_best, **self.dkwargs)

        # --- Calibration ---
        qq = np.percentile(self.cal_samples, axis=0, q=q)
        lax.fill_between(dwave[m] / wc, qq[0, m], qq[-1, m], **self.skwargs)

        # --- Prettify ---
        sax.set_ylim(0.1, 1.5)
        sax.set_ylabel(r"$\nu f_\nu \times$ Constant")
        #sax.set_yscale('log')

        rax.axhline(0.0, linestyle=':', color='k')
        rax.set_ylim(-7, 7)
        rax.set_ylabel(r'$\chi_{\rm best} \, (\frac{{\rm model}-{\rm data}}{\sigma})$')

        lax.axhline(1.0, linestyle=':', color='k')
        lax.set_xlabel(r'$\lambda (\mu{\rm m})$')
        lax.set_ylabel(r'$\mathcal{C}\, (\lambda)$', fontsize=12)
        #lax.ticklabel_format(axis='x', style='sci', scilimits=(-10, 10))

        [ax.set_xlim(3300 / wc, 8700 / wc) for ax in [sax, rax, lax]]
        [ax.set_xticklabels('') for ax in [sax, rax]]

        # --- annotate ---
        sax.text(0.1, 0.8, self.obs["cluster"], transform=sax.transAxes, fontsize=14)
        rax.text(0.7, 0.8, r'Residuals', transform=rax.transAxes)
        lax.text(0.58, 0.60, r"Calibration vector:"+"\n"+r"$F_\nu({\rm model}) \, = \, \mathcal{C}(\lambda) \cdot F_\nu({\rm intrinsic})$",
                 transform=lax.transAxes)

    def make_legends(self, sax):

        artists = [self.art["phot_data"], self.art["spec_data"], self.art["spec_post"], self.art["sed_post"]]
        legends = [r"Observed photometry", r"Bestfit Spectrum", r"Posterior (Spectrum)", r"Posterior (Photometry)"]
        sax.legend(artists, legends, loc="lower right", fontsize=12)


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
