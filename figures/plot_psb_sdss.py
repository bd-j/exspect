#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, pickle
from argparse import ArgumentParser
import numpy as np

from matplotlib import rcParams, gridspec
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import prospect.io.read_results as reader
from prospect.plotting import FigureMaker, chain_to_struct, boxplot, sample_prior
from prospect.plotting.corner import marginal, quantile, _quantile
from prospect.plotting.sed import convolve_spec, to_nufnu
from prospect.plotting.sfh import nonpar_recent_sfr, nonpar_mwa, ratios_to_sfrs, sfh_quantiles

from defaults import pretty, plot_defaults, colorcycle

# plot options
pl.ioff()  # don't pop up window
rcParams = plot_defaults(rcParams)
ms = 5
alpha = 0.8
fs = 16
ticksize = 12
lw = 0.5


def construct_sfh_measure(chain, agebins):
    logmass = np.squeeze(chain["logmass"])
    lm, sr = np.atleast_2d(chain["logmass"]), np.atleast_2d(chain["logsfr_ratios"])
    age = nonpar_mwa(lm, sr, agebins=agebins)
    sfr = nonpar_recent_sfr(lm, sr, agebins, sfr_period=0.1)
    ssfr = np.log10(sfr) - logmass
    return [age, np.log10(sfr)], ["mwa", "logsfr"]


def params_to_sfh(model, samples, ntime=50):
    agebins = model.params["agebins"]
    sfh_samples = np.array([ratios_to_sfrs(s["logmass"], s["logsfr_ratios"], agebins=agebins)
                            for s in samples])
    tlook = 10**agebins / 1e9
    tvec = np.exp(np.linspace(np.log(max(tlook.min(), 0.01)), np.log(tlook.max()), ntime))

    return tlook, tvec, sfh_samples


class Plotter(FigureMaker):
    """ To generate plots: instatiate the Plotter object, then try:
        `` plotter.plot_all()``
    """

    show = ['logmass', 'logsfr', 'mwa', 'av', 'logzsol', 'sigma_smooth']

    def convert(self, chain):
        cols = ["logmass", "logzsol", "gas_logu", "gas_logz",
                "av", "av_bc", "dust_index",
                "zred", "sigma_smooth", "spec_jitter"]

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
        #params["av_bc"] = params["av"] * np.squeeze(1 + chain["dust_ratio"])

        # --- stellar ---
        for i, c in enumerate(sfh_label):
            params[c] = np.squeeze(sfh[i])

        return params

    def read_in(self, results_file):
        self.result, self.obs, self.model = reader.results_from(results_file)
        if self.model is None:
            self.model = psb_params.build_model(**self.result["run_params"])
            self.sps = psb_params.build_sps(**self.result["run_params"])
            #self.obs = psb_params.build_obs(**self.result["run_params"])

        self.sps = None
        self.chain = chain_to_struct(self.result["chain"], self.model)
        self.weights = self.result.get("weights", None)
        self.ind_best = np.argmax(self.result["lnprobability"])

        # get agebins for the best redshift
        xbest = self.result["chain"][self.ind_best]
        self.model.set_parameters(xbest)
        self.agebins = np.array(self.model.params["agebins"])

        self.parchain = self.convert(self.chain)

    def plot_all(self):

        self.qu = np.array([16, 50, 84])
        self.setup_geometry(len(self.show))
        self.styles()
        self.extra_art()

        self.plot_posteriors(self.paxes)
        self.plot_sfh(self.sfhax)

        self.make_seds()
        self.plot_sed(self.sedax, self.resax,
                      nufnu=self.nufnu, microns=self.microns)
        self.plot_spec(self.specax, self.sresax, calax=self.calax,
                       nufnu=self.nufnu, microns=False)
        self.restframe_axis(self.specax, microns=False, fontsize=fs, ticksize=ticksize)
        self.make_inset(self.specax, microns=False)

    def setup_geometry(self, npar):

        self.fig = pl.figure(figsize=(9.5, 14.))

        gs = gridspec.GridSpec(10, npar, width_ratios=npar * [10], wspace=0.15, hspace=0.03,
                               height_ratios=[3, 1, 1.25, 3, 1, 1, 1.25, 2.0, 0.5, 2.0],
                               left=0.1, right=0.98, top=0.99, bottom=0.05)
        self.sedax = self.fig.add_subplot(gs[0, :])
        self.resax = self.fig.add_subplot(gs[1, :], sharex=self.sedax)
        self.specax = self.fig.add_subplot(gs[3, :])
        self.sresax = self.fig.add_subplot(gs[4, :], sharex=self.specax)
        self.calax = self.fig.add_subplot(gs[5, :], sharex=self.specax)

        #self.paxes = [self.fig.add_subplot(gs[7, i]) for i in range(npar)]
        #self.sfhax = self.fig.add_subplot(gs[9, 2:-2])
        self.sfhax = self.fig.add_subplot(gs[7:, 0:3])
        #self.paxes = np.array([self.fig.add_subplot(gs[int(i/3)*2 + 7, 3 + (i % 3)]) for i in range(npar)])
        self.paxes = np.array([self.fig.add_subplot(gs[int(i/3)*2 + 7, 3 + (i % 3)]) for i in range(npar)])

    def plot_sfh(self, sfhax):
        """ Plot the SFH
        """
        agebins = self.model.params["agebins"]
        sfh_samples = np.array([ratios_to_sfrs(s["logmass"], s["logsfr_ratios"], agebins=agebins)
                                for s in self.chain])
        tlook = 10**agebins / 1e9
        tvec = np.exp(np.linspace(np.log(max(tlook.min(), 0.01)), np.log(tlook.max()), 50))

        # -- shrink the bins to get a prettier SFH ---
        tlook *= np.array([1.05, 0.95])
        tlook = np.tile(tlook, (len(self.chain), 1, 1))
        sq = sfh_quantiles(tvec, tlook, sfh_samples, weights=self.weights, q=self.qu)

        # --- plot SFH ---
        sfhax.plot(tvec, sq[:, 1], '-', color='k', lw=1.5)
        sfhax.fill_between(tvec, sq[:, 0], sq[:, 2], **self.akwargs)
        sfhax.plot(tvec, sq[:, 0], '-', color='k', alpha=0.3, lw=1.5)
        sfhax.plot(tvec, sq[:, 2], '-', color='k', alpha=0.3, lw=1.5)

        # --- prettify ---
        sfhax.set_ylabel(r'SFR (M$_{\odot}$/yr)', fontsize=fs, labelpad=1.5)
        sfhax.set_xlabel(r'Lookback Time (Gyr)', fontsize=fs, labelpad=1.5)

        sfhax.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
        sfhax.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
        sfhax.set_xscale('log', subsx=([2, 5]))
        sfhax.set_yscale('log', subsy=([2, 5]))
        sfhax.tick_params('both', length=lw*3, width=lw, which='both', labelsize=12)

        #sfhax.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
        #sfhax.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
        sfhax.yaxis.set_major_formatter(FormatStrFormatter('%2.5g'))

        # --- annotate ---
        sfhax.annotate('post-starburst\nevent', (0.25, 5), (0.03, 10),
                       ha="center", va="center", weight='bold',
                       size=fs*0.8, color="red",
                       arrowprops=dict(shrinkA=1.5, shrinkB=1.5, fc="red", ec="red"),
                       zorder=2)

    def plot_posteriors(self, paxes, show_extra=False,
                        title_kwargs=dict(fontsize=fs*0.75),
                        label_kwargs=dict(fontsize=fs*0.6)):

        for i, p in enumerate(self.show):
            x = np.squeeze(self.parchain[p]).flatten()
            ax = paxes.flat[i]
            ax.set_xlabel(pretty.get(p, p), **label_kwargs)

            marginal(x, weights=self.weights, ax=ax, histtype="stepfilled", **self.akwargs)

            if show_extra:
                # --- quantiles ---
                qs = _quantile(x, self.qu/100., weights=self.weights)
                for j, q in enumerate(qs):
                    lw = 1 + int(j == 1)
                    paxes[i].axvline(q, ls="dashed", color='k', alpha=0.75, lw=lw)
                qm, qp = np.diff(qs)
                title = r"${{{0:.2f}}}_{{-{1:.2f}}}^{{+{2:.2f}}}$"
                title = title.format(qs[1], qm, qp)
                ax.set_title(title, va='bottom', pad=2.0, **title_kwargs)

        # priors
        if self.prior_samples > 0:
            spans = [ax.get_xlim() for ax in paxes.flat]
            self.show_priors(paxes.flat, spans, smooth=0.10, **self.rkwargs)

        # --- Prettify ---
        [ax.set_yticklabels("") for ax in paxes.flat]

    def plot_sed(self, sedax, resax=None, nufnu=True, microns=True,
                 normalize=True, logify=False, **kwargs):

        # --- Photometric data ---
        pmask = self.obs["phot_mask"]
        pwave = self.obs["wave_effective"][pmask]
        ophot, ounc = self.obs["maggies"][pmask], self.obs["maggies_unc"][pmask]
        maxw, minw = np.max(pwave*1.05) * 1.02, np.min(pwave * 0.95) * 0.98
        # units
        if nufnu:
            _, ophot = to_nufnu(pwave, ophot, microns=microns)
            owave, ounc = to_nufnu(pwave, ounc, microns=microns)
        else:
            owave = pwave / 10**(4 * microns)
        if normalize:
            renorm = 1. / np.mean(ophot)
        else:
            renorm = 1.

        # --- model SEDs ---
        if self.n_seds > 0:
            self.spec_wave = self.obs["wavelength"].copy()
            swave = self.spec_wave.copy() / 10**(4 * microns)
            # photometry
            phot = self.phot_samples
            phot_best = self.phot_best
            # phot units
            if nufnu:
                _, phot = to_nufnu(pwave, self.phot_samples, microns=microns)
                _, phot_best = to_nufnu(pwave, self.phot_best, microns=microns)
            # spec convolve & units
            ckw = dict(minw=minw, maxw=maxw, R=500*2.35, nufnu=nufnu, microns=microns)
            _, spec = convolve_spec(self.spec_wave, self.sed_samples, **ckw)
            cswave, spec_best = convolve_spec(self.spec_wave, self.sed_best, **ckw)
            # interpolate back onto obs wavelengths, get quantiles
            spec = np.array([np.interp(swave, cswave, s) for s in spec])
            spec_best = np.interp(swave, cswave, spec_best)
            spec_pdf = np.percentile(spec, axis=0, q=self.qu).T
            mask = slice(10, -10)  # remove edges that get convolved wrong

            # --- plot spectrum posteriors ---
            sedax.plot(swave[mask], spec_pdf[mask, 1] * renorm, **self.skwargs)
            sedax.fill_between(swave[mask], spec_pdf[mask, 0] * renorm, spec_pdf[mask, 2] * renorm,
                               **self.skwargs)

            # --- plot phot posterior ---
            self.bkwargs = dict(alpha=0.8,
                                facecolor=self.pkwargs["color"], edgecolor="k")
            self.art["phot_post"] = Patch(**self.bkwargs)
            widths = 0.05 * owave  # phot_width
            boxplot((phot * renorm).T, owave, widths, ax=sedax, **self.bkwargs)

        # --- plot phot data ---
        sedax.errorbar(owave, ophot * renorm, ounc * renorm, color="k", linestyle="", linewidth=2)
        sedax.plot(owave, ophot * renorm, **self.dkwargs)

        # --- phot residuals ---
        phot_chi = (phot_best - ophot) / ounc
        resax.plot(owave, phot_chi, **self.dkwargs)

        # --- prettify ---
        # limits & lines
        resax.axhline(0, linestyle=':', color='grey')
        resax.yaxis.set_major_locator(MaxNLocator(5))
        ymin, ymax = 0.8 * (ophot * renorm).min(), 1.2 * (ophot * renorm).max()
        sedax.set_ylim(ymin, ymax)
        resax.set_ylim(-2.8, 2.8)

        # set labels
        wave_unit = microns*r"$\mu$m" + (not microns)*r"$\AA$"
        fl = (int(logify)*r"$\log$" + int(nufnu)*r'$\nu$' +
              r'$f_{\nu}$' + int(normalize)*r' $\times$ Constant')
        sedax.set_ylabel(fl, fontsize=fs)
        sedax.set_xticklabels([])
        resax.set_ylabel(r'$\chi_{\rm Best}$', fontsize=fs)
        resax.set_xlabel(r'$\lambda_{{\rm obs}}$ ({})'.format(wave_unit), fontsize=fs, labelpad=-1)

        if logify:
            sedax.set_yscale('log', nonposy='clip')
            sedax.set_xscale('log', nonposx='clip')
            resax.set_xscale('log', nonposx='clip', subsx=(2, 5))
        resax.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
        resax.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
        resax.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both', labelsize=ticksize)
        sedax.tick_params('y', which='major', labelsize=ticksize)

        # --- annotate ---
        chisq = np.sum(phot_chi**2)
        ndof = self.obs["phot_mask"].sum()
        reduced_chisq = chisq / (ndof)
        sedax.text(0.01, 0.9, r'Photometric fit', fontsize=18, transform=sedax.transAxes, color='k')
        sedax.text(0.01, 0.81, r'best-fit $\chi^2$/N$_{\mathrm{phot}}$='+"{:.2f}".format(reduced_chisq),
                   fontsize=10, ha='left', transform=sedax.transAxes, color='k')
        if "zred" not in self.model.free_params:
            zred = self.model.params["zred"]
            sedax.text(0.02, 0.9, 'z='+"{:.2f}".format(zred),
                       fontsize=10, ha='left', transform=sedax.transAxes)

        # --- Legend ---
        artists = [self.art["phot_data"], self.art["phot_post"], self.art["sed_post"]]
        labels = ["Observed", "Model Photometry", "Model SED"]
        sedax.legend(artists, labels, loc='lower right',
                     fontsize=10, scatterpoints=1, fancybox=True)

    def plot_spec(self, specax, sresax, calax=None,
                  nufnu=True, microns=False, normalize=True, **kwargs):
        """Plot the spectroscopy for the model and data (with error bars), and
        plot residuals
            -- pass in a list of [res], can iterate over them to plot multiple results
        good complimentary color for the default one is '#FF420E', a light red
        """
        # --- Spec Data ---
        mask = self.obs['mask']
        wave = self.obs['wavelength'][mask]
        ospec, ounc = self.obs['spectrum'][mask], self.obs['unc'][mask]
        # units
        if nufnu:
            _, ospec = to_nufnu(wave, ospec, microns=microns)
            owave, ounc = to_nufnu(wave, ounc, microns=microns)
        else:
            owave = wave / 10**(4 * microns)
        if normalize:
            renorm = 1. / np.median(ospec)
        else:
            renorm = 1.

        # --- Model spectra ---
        if self.n_seds > 0:
            spec = self.spec_samples[:, mask]
            spec_best = self.spec_best[mask]
            # units
            if nufnu:
                _, spec = to_nufnu(wave, spec, microns=microns)
                wave, spec_best = to_nufnu(wave, spec_best, microns=microns)
            spec_pdf = np.percentile(spec, axis=0, q=self.qu).T

            # --- plot posterior ---
            #specax.plot(wave, spec_pdf[:, 1] * renorm, **self.skwargs)
            specax.fill_between(owave, spec_pdf[:, 0] * renorm, spec_pdf[:, 2] * renorm, **self.skwargs)
            specax.plot(owave, spec_best * renorm, **self.spkwargs)

        # --- plot data ---
        specax.plot(owave, ospec*renorm, **self.lkwargs)

        # --- plot residuals ---
        spec_chi = (ospec - spec_best) / ounc
        sresax.plot(owave, spec_chi, linewidth=0.75, **self.spkwargs)

        # --- plot calibration ---
        calax.plot(owave, self.cal_best, linewidth=2.0, **self.spkwargs)

        # --- prettify ---
        # limits
        xlim = (owave.min()*0.95, owave.max()*1.05)
        specax.set_xlim(*xlim)
        ymin, ymax = (ospec*renorm).min()*0.9, (ospec*renorm).max()*1.1
        specax.set_ylim(ymin, ymax)
        sresax.set_ylim(-5, 5)

        # extra line
        for ax, factor in zip([sresax, calax], [0, 1]):
            ax.axhline(factor, linestyle=':', color='grey')
            ax.yaxis.set_major_locator(MaxNLocator(5))
        calax.set_ylim(0.79, 1.21)

        # set labels
        wave_unit = microns*r"$\mu$m" + (not microns)*r"$\AA$"
        fl = int(nufnu)*r'$\nu$' + r'$f_{\nu}$' + int(normalize)*r' $\times$ Constant'
        sresax.set_ylabel(r'$\chi_{\rm Best}$', fontsize=fs)
        specax.set_ylabel(fl, fontsize=fs)
        calax.set_ylabel('calibration\nvector', fontsize=fs)
        calax.set_xlabel(r'$\lambda_{{\rm obs}}$ ({})'.format(wave_unit), fontsize=fs)
        sresax.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both', labelsize=ticksize)
        calax.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both', labelsize=ticksize)
        specax.tick_params('y', which='major', labelsize=ticksize)

        # --- annotate ---
        chisq = np.sum(spec_chi**2)
        ndof = mask.sum()
        reduced_chisq = chisq/(ndof)

        specax.text(0.01, 0.9, 'Spectroscopic fit', fontsize=18, transform=specax.transAxes, color='k')
        specax.text(0.01, 0.81, r'best-fit $\chi^2$/N$_{\mathrm{spec}}$='+"{:.2f}".format(reduced_chisq),
                    fontsize=10, ha='left', transform=specax.transAxes, color='black')

        # --- Legend ---
        artists = [self.art["spec_data"], self.art["spec_best"]]
        labels = ["Observed", "Best posterior sample"]
        specax.legend(artists, labels, loc='upper right',
                      fontsize=10, scatterpoints=1, fancybox=True)

    def spectral_components(self, x):
        # generate all three spectra
        spec_bfit, _, _ = self.model.predict(x, sps=self.sps, obs=self.obs)
        self.model.params['marginalize_elines'] = False
        self.model.params['nebemlineinspec'] = True
        spec_nomarg, _, _ = self.model.predict(x, sps=self.sps, obs=self.obs)
        self.model.params['marginalize_elines'] = False
        self.model.params['nebemlineinspec'] = True
        self.model.params['add_neb_emission'] = False
        spec_nolines, _, _ = self.model.predict(x, sps=self.sps, obs=self.obs)

        # return the model to its original state
        self.model.params["marginalize_elines"] = True
        self.model.params["nebemlineinspec"] = False
        self.model.params["add_neb_emission"] = True

        return spec_bfit, spec_nomarg, spec_nolines

    def make_inset(self, ax, minw=6500, maxw=6600,
                   label=r'H$\alpha$ + [NII]', microns=False):
        # H-alpha, NII inset
        # create inset axis
        axi = inset_axes(ax, width="25%", height="35%", loc="lower right",
                         borderpad=2)

        if "zred" in self.model.free_params:
            zred = self.model.params["zred"]
        else:
            zred = self.parchain["zred"][self.ind_best]
        wave, ospec = self.obs["wavelength"].copy(), self.obs["spectrum"].copy()
        xbest = self.result["chain"][self.ind_best]
        blob = self.spectral_components(xbest)
        bfit, bfit_nomarg, bfit_nolines = blob

        # find region around H-alpha
        idx = (wave / (1 + zred) > minw) & (wave / (1 + zred) < maxw)
        wave /= 10**(4 * microns)

        lw_inset = lw*2
        common = dict(alpha=alpha, lw=lw_inset, linestyle="-")
        obs_kwargs = dict(color="k", label='Observed')
        marg_kwargs = dict(zorder=20, color=self.skwargs["color"], label='Nebular marginalization')
        cloudy_kwargs = dict(zorder=20, color=colorcycle[0], label='CLOUDY grid')
        cont_kwargs = dict(zorder=10, color='grey', label='Continuum model')
        axi.plot(wave[idx], ospec[idx], **dict(**common, **obs_kwargs))
        axi.plot(wave[idx], bfit[idx], **dict(**common, **marg_kwargs))
        axi.plot(wave[idx], bfit_nomarg[idx], **dict(**common, **cloudy_kwargs))
        axi.plot(wave[idx], bfit_nolines[idx], **dict(**common, **cont_kwargs))

        # labels
        axi.set_title(label, fontsize=fs*0.7, weight='semibold', pad=-3)
        axi.set_yticklabels([])
        axi.tick_params('both', which='both', labelsize=fs*0.4)
        wave_unit = microns*r"$\mu$m" + (not microns)*r"$\AA$"
        axi.set_xlabel(r"$\lambda_{{\rm obs}}$ ({})".format(wave_unit), fontsize=fs*0.6, labelpad=-1.5)

        # legend
        axi.legend(prop={'size': 4.5}, loc='upper left')

    def show_sfh_prior(self, ax, nsample=10000, renorm=True):

        samples, _ = sample_prior(self.model, nsample=nsample)
        priors = chain_to_struct(samples, self.model)
        tlook, tvec, sfh_samples = params_to_sfh(self.model, priors)

        # renormalize to SFR in most recent bin, to get just the *shape*
        if renorm:
            pbest = self.chain[self.ind_best]
            sfh_best = ratios_to_sfrs(pbest["logmass"], pbest["logsfr_ratios"],
                                      agebins=self.model.params["agebins"])
            sfh_samples = sfh_samples / (sfh_samples[:, :1]) * sfh_best[0]

        # -- shrink the bins to get a prettier SFH ---
        tlook *= np.array([1.05, 0.95])
        tlook = np.tile(tlook, (len(priors), 1, 1))
        sq = sfh_quantiles(tvec, tlook, sfh_samples, weights=None, q=self.qu)

        ax.fill_between(tvec, sq[0, :], sq[2, :], zorder=0, alpha=0.4, **self.rkwargs)

    def extra_art(self):
        self.skwargs = dict(color=colorcycle[1], alpha=0.65)
        self.akwargs = dict(color=colorcycle[3], alpha=0.65)
        self.spkwargs = dict(color=colorcycle[1], alpha=1.0)

        self.art["sed_post"] = Patch(**self.skwargs)
        self.art["all_post"] = Patch(**self.akwargs)
        self.art["spec_best"] = Line2D([], [], **self.spkwargs)


        # add in arrows for negative fluxes
        #if pflux.sum() != len(obsmags):
        #    downarrow = [u'\u2193']
        #    y0 = 10**((np.log10(ymax) - np.log10(ymin))/20.)*ymin
        #    for x0 in phot_wave_eff[~pflux]:
        #        phot.plot(x0, y0, linestyle='none', marker=u'$\u2193$',
        #                  markersize=16,alpha=alpha,mew=0.5, mec='k',color=self.data_color)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--results_file", type=str, default="../fitting/output/psb_results/psb_sdss_mcmc.h5")
    parser.add_argument("--fignum", type=str, default="sdss_psb")
    parser.add_argument("--figext", type=str, default="png")
    parser.add_argument("--prior_samples", type=int, default=100000)
    parser.add_argument("--n_seds", type=int, default=500)
    args = parser.parse_args()

    plotter = Plotter(nufnu=True, microns=True, **vars(args))
    plotter.plot_all()
    #plotter.show_sfh_prior(plotter.sfhax)

    # --- Saving ---
    # --------------
    if args.fignum:
        n = "paperfigures/{}.{}".format(args.fignum, args.figext)
        plotter.fig.savefig(n, dpi=400)
        os.system('open {}'.format(n))
    else:
        pl.ion()
        pl.show()
