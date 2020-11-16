#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" fig10.py - comparison of inferred GGC parameters from prospector and the literature.
"""
import sys, glob, os
import argparse

import numpy as np
import matplotlib.pyplot as pl
from astropy.io import fits
from numpy.lib.recfunctions import append_fields

import prospect.io.read_results as reader
from prospect.io.write_results import chain_to_struct

from prospect.plotting.corner import _quantile
from defaults import plot_defaults, pretty

from matplotlib.pyplot import FixedLocator

lmap = {"tage": "age",
        "logzsol": "feh",
        "av": "av"}


class Plotter:

    show = "logzsol", "tage", "av"

    def __init__(self, search="*h5", koleva_file="", **extras):
        self.files = glob.glob(search)
        self.koleva_file = koleva_file

    def read_in(self):
        self.names = []
        self.chains, self.weights, self.models, self.ebv = {}, {}, {}, []
        for i, f in enumerate(self.files):
            result, obs, model = reader.results_from(f)
            name = obs["cluster"]
            self.names.append(name)
            self.models[name] = model
            self.chains[name] = self.convert(chain_to_struct(result["chain"], model))
            self.weights[name] = result["weights"]
            self.ebv.extend(obs["ebv"])

        # match to koleva
        gcnames = np.array(self.names).copy()
        self.literature, linds = self.get_literature(gcnames, self.koleva_file)
        self.literature = append_fields(self.literature, "av", 3.1 * np.array(self.ebv))
        sel = self.literature['age'] > 0
        self.literature = self.literature[sel]
        self.matched_names = gcnames[sel]

    def get_literature(self, gcnames, kfile):

        lit = fits.getdata(kfile)
        lnames = list(lit['name'])
        lit = np.array(lit)
        out = np.zeros(len(gcnames), dtype=lit.dtype)
        oinds = []
        for k, n in enumerate(gcnames):
            if n == 'NGC104':
                n = 'NGC0104'
            inds = [i for i, g in enumerate(lnames) if n in str(g)]
            #print(n, inds)
            if len(inds) == 1:
                out[k] = lit[inds[0]]
                oinds.append(n)
            else:
                print(n)
        return out, oinds

    def convert(self, chain):
        """compute quantities from a structured chain (or dictionary)
        """
        niter = len(chain)
        cols = ["av", "logzsol", "tage", "sigma_smooth", "spec_jitter", "zred"]
        dt = np.dtype([(c, np.float) for c in cols])
        params = np.zeros(niter, dtype=dt)

        for c in cols:
            if c in chain.dtype.names:
                params[c] = np.squeeze(chain[c])

        # --- dust attenuation
        params["av"] = np.squeeze(1.086 * chain["dust2"])

        return params

    def make_axes(self):
        self.fig, self.axes = pl.subplots(3, 2, figsize=(4.5, 8), sharey="row", sharex="row")
        self.fig.subplots_adjust(wspace=0.2, hspace=0.5)

    def styles(self):
        self.ekwargs = {"marker": '', "linestyle": '', "ecolor": 'grey',
                        "elinewidth": 1.5, "alpha": 0.5, "zorder": 50}
        self.skwargs = dict(cmap="magma", vmin=-1.5, vmax=1.5)

    def plot_comparison(self, axes, q=[0.16, 0.50, 0.84]):

        # --- Compare ---
        # ---------------
        ltypes = ["{}", "{}_koleva"]
        for i, p in enumerate(self.show):
            for j, fmt in enumerate(ltypes):
                pp, ax = fmt.format(lmap[p]), axes[i, j]
                if pp not in self.literature.dtype.names:
                    continue
                qq = np.array([_quantile(self.chains[n][p], q, weights=self.weights[n])
                               for n in self.matched_names])
                y = qq[:, 1]
                yerr = y - qq[:, 0], qq[:, 2] - y
                ax.errorbar(self.literature[pp], y, yerr, **self.ekwargs)
                cm = ax.scatter(self.literature[pp], y, c=self.literature["hbr"], **self.skwargs)
                ke = "{}_unc".format(pp)
                if ke in self.literature.dtype.names:
                    xerr = self.literature[ke]
                    ax.errorbar(self.literature[pp], y, xerr=xerr, **self.ekwargs)
                # priors (assuming topHat)
                m = list(plotter.models.values())[0]
                try:
                    prior = m.config_dict[p]["prior"].params.values()
                    [ax.axhline(l, linestyle=":", color="k", alpha=0.5) for l in prior]
                except(KeyError):
                    pass

        # --- Prettify ---
        for i, p in enumerate(self.show):
            for j, l in enumerate(["Literature", "K08"]):
                ax = axes[i, j]
                pp = pretty[p]
                if j == 0:
                    ax.set_ylabel("{} [Prospector]".format(pp))
                ax.set_xlabel("{} [{}]".format(pp, l))
                lo, hi = ax.get_xlim()
                xx = np.linspace(lo, hi, 10)
                ax.plot(xx, xx, linestyle="--", color="k")

        # limits and ticks
        [ax.set_ylim(-2.5, 0.3) for ax in axes[0, :]]
        zloc = FixedLocator([-2, -1, 0])
        [ax.yaxis.set_major_locator(zloc) for ax in axes[0,:]]
        [ax.xaxis.set_major_locator(zloc) for ax in axes[0,:]]
        [ax.set_ylim(3, 17) for ax in axes[1, :]]

        dloc = FixedLocator([0, 1, 2])
        axes[-1, 0].xaxis.set_major_locator(dloc)
        axes[-1, 0].yaxis.set_major_locator(dloc)

        axes.flat[-1].set_visible(False)

        cax = self.fig.add_subplot(12, 2, 22)
        self.fig.colorbar(cm, cax=cax, label="HBR", orientation="horizontal")

    def plot_all(self):
        self.read_in()
        self.make_axes()
        self.styles()
        self.plot_comparison(self.axes)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fignum", type=str, default="")
    parser.add_argument("--figext", type=str, default="png")
    parser.add_argument("--search", type=str, default="../fitting/output/ggc/ggc*h5")
    parser.add_argument("--koleva_file", type=str, default="../data/koleva08_table1.fits")
    args = parser.parse_args()

    from matplotlib.pyplot import rcParams
    rcParams = plot_defaults(rcParams)

    plotter = Plotter(**vars(args))
    plotter.plot_all()

    if args.fignum:
        plotter.fig.savefig("paperfigures/{}.{}".format(args.fignum, args.figext), dpi=400)
    else:
        pl.ion()
        pl.show()
