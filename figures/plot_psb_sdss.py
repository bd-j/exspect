import os,pickle
from prospect.io.read_results import results_from
from matplotlib import rcParams, gridspec
import matplotlib.pyplot as plt
import numpy as np
from prosp_dutils import smooth_spectrum
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter as norm_kde
from dynesty.plotting import _quantile as quantile
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.ioff() # don't pop up window

# plot options
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["STIXGeneral"]
rcParams["font.size"] = 12
rcParams["mathtext.fontset"] = "custom"
rcParams["mathtext.rm"] = "serif"
rcParams["mathtext.sf"] = "serif"
rcParams['mathtext.it'] = 'serif:italic'
ms = 5 
alpha = 0.8 
fs = 16 
ticksize = 12
lw = 0.5


class Plotter():
    """ To generate plots: instatiate the Plotter object, then try:
        `` plotter.build_all()``
    """

    def __init__(self, regenerate=True):

        # load data
        self.res, _, _ = results_from('psb_results/psb_sdss_mcmc.h5')
        with open("psb_results/psb_sdss_eout.pickle", 'rb') as f:
            self.eout = pickle.load(f,encoding='unicode_escape')

        # rebuild model to generate best-fit spectrum
        import psb_params
        self.model = psb_params.build_model(extra_phot=0)
        self.obs = psb_params.build_obs(extra_phot=0)
        self.sps = psb_params.build_sps(extra_phot=0)

        # generate best-fit spectrum
        if regenerate:
            # generate all three spectra
            bfit_theta = self.res['chain'][self.eout['sample_idx'][0]]
            self.spec_bfit, _, _ = self.model.predict(bfit_theta,sps=self.sps,obs=self.obs)
            self.model.params['marginalize_elines'] = False
            self.model.params['nebemlineinspec'] = True
            self.spec_nomarg, _, _ = self.model.predict(bfit_theta,sps=self.sps,obs=self.obs)
            self.model = psb_params.build_model(extra_phot=0)
            self.model.params['nebemlineinspec'] = False
            self.model.params['marginalize_elines'] = False
            self.spec_nolines, _, _ = self.model.predict(bfit_theta,sps=self.sps,obs=self.obs)
        else:
            self.spec_bfit = np.ones_like(self.obs['wavelength'])
            self.spec_nomarg = np.ones_like(self.obs['wavelength'])
            self.spec_nolines = np.ones_like(self.obs['wavelength'])

        # match plotting options
        self.colorcycle = ["royalblue", "firebrick", "indigo", "darkorange", "seagreen"]
        self.data_color = 'k'
        self.posterior_color = self.colorcycle[1]
        self.extra_color = self.colorcycle[0]


    def plot_all(self):

        pshow = ['stellar_mass', 'sfr_100', 'avg_age', 'dust2', 'logzsol', 'sigma_smooth']
        pnames = [r'log(M/M$_{\odot}$)', r'log(SFR) (M$_{\odot}$/yr)', r'<stellar age> (Gyr)', 
                  r'A$_{\mathrm{V}}$', r'log(Z/Z$_{\odot}$)', r'$\sigma_{\mathrm{smooth}}$ (km/s)']

        self.setup_geometry(len(pshow))
        self.plot_sed()
        self.plot_spec()
        self.plot_posteriors(pshow,pnames)
        self.plot_sfh()

        plt.savefig('sdss_psb.pdf',dpi=250)
        plt.close()
        os.system('open sdss_psb.pdf')

    def setup_geometry(self,npar):

        self.fig = plt.figure(figsize=(9.5,14.))

        gs = gridspec.GridSpec(10, npar, width_ratios=npar * [10],
                      #height_ratios=[3, 1, 1.25, 3, 1, 1, 1.25, 2, 1.25, 3],
                      height_ratios=[3, 1, 1.25, 3, 1, 1, 1.25, 1.75, 1., 1.75],
                      left=0.1, right=0.98, wspace=0.15, hspace=0.03, top=0.99, bottom=0.05)
        self.sedax = self.fig.add_subplot(gs[0, :])
        self.sedresidax = self.fig.add_subplot(gs[1, :], sharex=self.sedax)
        self.specax = self.fig.add_subplot(gs[3, :])
        self.specresidax = self.fig.add_subplot(gs[4,:], sharex=self.specax)
        self.speccalibax = self.fig.add_subplot(gs[5,:], sharex=self.specax)

        #self.paxes = [self.fig.add_subplot(gs[7, i]) for i in range(npar)]
        #self.sfhax = self.fig.add_subplot(gs[9, 2:-2])
        self.sfhax = self.fig.add_subplot(gs[7:10,0:3])
        self.paxes = [self.fig.add_subplot(gs[int(i/3)*2+7,3+(i%3)]) for i in range(npar)]

        plt.show()

    def plot_sfh(self):
        """ Plot the SFH
        """

        xmin, ymin = np.inf, np.inf
        xmax, ymax = -np.inf, -np.inf
            
        # create master time bin
        min_time = np.max([self.eout['sfh']['t'].min(),0.01])
        max_time = self.eout['sfh']['t'].max()
        tvec = 10**np.linspace(np.log10(min_time),np.log10(max_time),num=50)

        # create median SFH
        perc = np.zeros(shape=(len(tvec),3))
        for jj in range(len(tvec)): 
            # nearest-neighbor 'interpolation'
            # exact answer for binned SFHs
            if len(self.eout['sfh']['t'].shape) == 2:
                idx = np.abs(self.eout['sfh']['t'][0,:] - tvec[jj]).argmin(axis=-1)
            else:
                idx = np.abs(self.eout['sfh']['t'] - tvec[jj]).argmin(axis=-1)
            perc[jj,:] = quantile(self.eout['sfh']['sfh'][:,idx],[0.16,0.50,0.84],weights=self.eout['weights'])

        #### plot SFH
        self.sfhax.plot(tvec, perc[:,1],'-',color='k',lw=1.5)
        self.sfhax.fill_between(tvec, perc[:,0], perc[:,2], color='k', alpha=0.3)
        self.sfhax.plot(tvec, perc[:,0],'-',color='k',alpha=0.3,lw=1.5)
        self.sfhax.plot(tvec, perc[:,2],'-',color='k',alpha=0.3,lw=1.5)

        #### update plot ranges
        xmin = np.min([xmin,tvec.min()])
        xmax = np.max([xmax,tvec.max()])
        ymin = np.min([ymin,perc[perc>0].min()])
        ymax = np.max([ymax,perc.max()])

        xmin = np.min(tvec[tvec>0.01])
        ymin = np.clip(ymin,ymax*1e-5,np.inf)

        axlim_sfh = [xmax*1.01, xmin*1.0001, ymin*.7, ymax*1.4]
        self.sfhax.axis(axlim_sfh)

        #### labels, format, scales !
        self.sfhax.set_ylabel(r'SFR [M$_{\odot}$/yr]',fontsize=fs,labelpad=1.5)
        self.sfhax.set_xlabel(r't$_{\mathrm{lookback}}$ [Gyr]',fontsize=fs,labelpad=1.5)
        
        self.sfhax.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
        self.sfhax.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
        self.sfhax.set_xscale('log',subsx=([2,5]))
        self.sfhax.set_yscale('log',subsy=([2,5]))
        self.sfhax.tick_params('both', length=lw*3, width=lw, which='both',labelsize=12)

        self.sfhax.xaxis.set_minor_formatter(FormatStrFormatter('%2.5g'))
        self.sfhax.xaxis.set_major_formatter(FormatStrFormatter('%2.5g'))
        self.sfhax.yaxis.set_major_formatter(FormatStrFormatter('%2.5g'))

        # now annotate
        self.sfhax.annotate('post-starburst\nevent',
                            (0.25, 5),(0.03, 10),
                            ha="center", va="center",weight='bold',
                            size=fs*0.8, color="red",
                            arrowprops=dict(shrinkA=1.5,
                                            shrinkB=1.5,
                                            fc="red", ec="red",
                                             ),
                            zorder=2)

    def plot_posteriors(self,pshow,pnames):

        title_kwargs = {'fontsize':fs*.75}
        label_kwargs = {'fontsize':fs*.6}
        logify = ['stellar_mass','sfr_100','ssfr_100']
        for i, par in enumerate(pshow):

            ax = self.paxes[i]

            # pull out chain, quantiles, weights
            weights = self.eout['weights']
            key = 'thetas'
            if par in self.eout['extras'].keys(): 
                key = 'extras'
                pchain = self.eout[key][par]['chain']
            else:
                idx = self.res['theta_labels'].index(par) 
                pchain = self.res['chain'][self.eout['sample_idx'],idx]
            qvalues = np.array([self.eout[key][par]['q16'],
                                self.eout[key][par]['q50'],
                                self.eout[key][par]['q84']])

            # logify and transform 
            if par == 'dust2':
                pchain *= 1.086
                qvalues *= 1.086

            if par in logify:
                pchain = np.log10(pchain)
                qvalues = np.log10(qvalues)

            # complex smoothing routine to match dynesty
            bins = int(round(10. / 0.02))
            n, b = np.histogram(pchain, bins=bins, weights=weights,
                                range=[pchain.min(),pchain.max()])
            n = norm_kde(n, 10.)
            x0 = 0.5 * (b[1:] + b[:-1])
            y0 = n
            ax.fill_between(x0, y0, color=self.posterior_color, alpha = 0.65)

            # plot and show quantiles
            for j,q in enumerate(qvalues): 
                lw = 1
                if j == 1: lw = 2
                ax.axvline(q, ls="dashed", color='k',alpha=0.75, lw=lw)

            q_m = qvalues[1]-qvalues[0]
            q_p = qvalues[2]-qvalues[1]
            fmt = "{{0:{0}}}".format(".2f").format
            title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
            title = title.format(fmt(float(qvalues[1])), fmt(float(q_m)), fmt(float(q_p)))
            ax.set_title(title, va='bottom',pad=2.0,**title_kwargs)
            ax.set_xlabel(pnames[i],labelpad=-1,**title_kwargs)

            # look for truth
            min, max = np.percentile(pchain,0.5), np.percentile(pchain,99.5)

            # set range
            if par in logify:
                min = min - 0.08
                max = max + 0.08
            else:
                min, max = min*0.95, max*1.05
            ax.set_xlim(min,max)
            ax.set_ylim(0, 1.1 * np.max(n))

            # set labels
            ax.set_yticklabels([])
            ax.xaxis.set_major_locator(MaxNLocator(4))
            [l.set_rotation(45) for l in ax.get_xticklabels()]
            ax.xaxis.set_tick_params(labelsize=label_kwargs['fontsize'])

    def plot_sed(self, transcurves=False, ergs_s_cm=False,
                 normalize=True,logify=False,**kwargs):

        # define axes
        phot = self.sedax
        resid = self.sedresidax

        # pull out data
        mask = self.obs['phot_mask']
        phot_wave_eff_obs = self.obs['wave_effective'][mask].copy()
        obsmags = self.obs['maggies'][mask].copy()
        obsmags_unc = self.obs['maggies_unc'][mask].copy()

        # model information
        zred = self.eout['thetas']['zred']['q50']
        modmags_bfit = self.eout['obs']['mags'][0,mask].copy()
        modspec_lam_obs = self.eout['obs']['lam_obs'].copy()
        modspec_lam_obs /= (1+zred)
        nspec = modspec_lam_obs.shape[0]

        try:
            spec_pdf = np.zeros(shape=(nspec,3))
            if 'zred' in self.res['theta_labels']: # renormalize if we're fitting redshift
                zred_draw = self.res['chain'][self.eout['sample_idx'],self.res['theta_labels'].index('zred')]
                #self.eout['obs']['spec'] *= (1+zred_draw)[:,None]
            for jj in range(spec_pdf.shape[0]): 
                spec_pdf[jj,:] = np.percentile(self.eout['obs']['spec'][:,jj],[16.0,50.0,84.0])
        except:
            spec_pdf = np.stack((self.eout['obs']['spec']['q16'],self.eout['obs']['spec']['q50'],self.eout['obs']['spec']['q84']),axis=1)

        # units
        factor = 3e18
        if normalize:
            factor = 1./np.median(obsmags/phot_wave_eff_obs)

        # photometry
        modmags_bfit *= factor/phot_wave_eff_obs
        obsmags *= factor/phot_wave_eff_obs
        obsmags_unc *= factor/phot_wave_eff_obs
        photchi = (obsmags-modmags_bfit)/obsmags_unc
        phot_wave_eff = phot_wave_eff_obs

        # spectra
        spec_pdf *= (factor/modspec_lam_obs/(1+zred)).reshape(nspec,1)
        modspec_lam = modspec_lam_obs*(1+zred)
        if self.res['obs']['spectrum'] is not None:
            spec_pdf *= (1+zred)**2

        # plot MAP photometry and residuals
        phot.plot(phot_wave_eff, modmags_bfit, color=self.posterior_color, 
                  marker='o', ms=ms, linestyle=' ', label = 'model photometry', alpha=alpha, 
                  markeredgecolor='k')
        
        resid.plot(phot_wave_eff, photchi, color=self.posterior_color,
                 marker='o', linestyle=' ',
                 ms=ms,alpha=alpha,markeredgewidth=0.7,markeredgecolor='k')        

        # model spectra
        yplt = spec_pdf[:,1]
        pspec = smooth_spectrum(modspec_lam,yplt,200,minlam=1e3,maxlam=1e5)
        nz = pspec > 0
        phot.plot(modspec_lam[nz], pspec[nz], linestyle='-',
                  color=self.posterior_color, alpha=0.9,zorder=-1,label = 'model spectrum')  
        #phot.fill_between(modspec_lam[nz], spec_pdf[nz,0], spec_pdf[nz,2],
        #                  color=self.posterior_color, alpha=0.3,zorder=-1)

        # calculate and show reduced chi-squared for MAP
        chisq = np.sum(photchi**2)
        ndof = mask.sum()
        reduced_chisq = chisq/(ndof)

        phot.text(0.01, 0.9, 'photometric fit', fontsize=18,weight='bold',transform=phot.transAxes,color='k')
        phot.text(0.01, 0.81, r'best-fit $\chi^2$/N$_{\mathrm{phot}}$='+"{:.2f}".format(reduced_chisq),
                  fontsize=10, ha='left',transform = phot.transAxes,color='k')

        # plot observations
        pflux = obsmags > 0
        phot.errorbar(phot_wave_eff[pflux], obsmags[pflux], yerr=obsmags_unc[pflux],
                      color=self.data_color, marker='o', label='observed', alpha=alpha, linestyle=' ',ms=ms,
                      zorder=10,markeredgecolor='k')

        # limits
        xlim = (phot_wave_eff[pflux].min()*0.9,phot_wave_eff[pflux].max()*1.02)
        resid.set_xlim(xlim)
        ymin, ymax = obsmags[pflux].min()*0.8, obsmags[pflux].max()*1.2

        # add transmission curves
        if transcurves:
            dyn = 10**(np.log10(ymin)+(np.log10(ymax)-np.log10(ymin))*0.2)
            for f in self.res['obs']['filters']: phot.plot(f.wavelength, f.transmission/f.transmission.max()*dyn+ymin,lw=1.5,color='0.3',alpha=0.7)

        # add in arrows for negative fluxes
        if pflux.sum() != len(obsmags):
            downarrow = [u'\u2193']
            y0 = 10**((np.log10(ymax) - np.log10(ymin))/20.)*ymin
            for x0 in phot_wave_eff[~pflux]: phot.plot(x0, y0, linestyle='none',marker=u'$\u2193$',markersize=16,alpha=alpha,mew=0.5,
                                                       mec='k',color=self.data_color)
        phot.set_ylim(ymin, ymax)
        resid_ymax = np.abs(resid.get_ylim()).max()*2
        resid.set_ylim(-resid_ymax,resid_ymax)

        # redshift text
        if 'zred' not in self.res['theta_labels']:
            phot.text(0.02, 0.9, 'z='+"{:.2f}".format(zred),
                      fontsize=10, ha='left',transform = phot.transAxes)
        
        # extra line
        resid.axhline(0, linestyle=':', color='grey')
        resid.yaxis.set_major_locator(MaxNLocator(5))

        # legend
        leg = phot.legend(loc=kwargs.get('legend_loc','lower right'), prop={'size':fs*0.7},
                          scatterpoints=1,fancybox=True)

        # set labels
        resid.set_ylabel( r'$\chi$',fontsize=fs)
        for tl in phot.get_xticklabels():tl.set_visible(False)
        if ergs_s_cm:
            phot.set_ylabel(r'$\nu f_{\nu}$ [erg/s/cm$^2$]',fontsize=fs)
        else:
            phot.set_ylabel(r'$\nu f_{\nu}$',fontsize=fs)
        resid.set_xlabel(r'$\lambda_{\mathrm{obs}}$ [$\mu$m]',fontsize=fs,labelpad=-1)

        if logify:
            phot.set_yscale('log',nonposy='clip')
            phot.set_xscale('log',nonposx='clip')
            resid.set_xscale('log',nonposx='clip',subsx=(2,5))
        resid.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
        resid.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
        resid.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=ticksize)
        phot.tick_params('y', which='major', labelsize=ticksize)
        
        # set second x-axis (self.rest-frame wavelength)
        if 'zred' not in self.res['theta_labels']:
            y1, y2=phot.get_ylim()
            x1, x2=phot.get_xlim()
            ax2=phot.twiny()
            ax2.set_xticks(np.arange(0,10,0.2))
            ax2.set_xlim(x1/(1+zred), x2/(1+zred))
            ax2.set_xlabel(r'$\lambda_{\mathrm{self.rest}}$ [$\mu$m]',fontsize=fs)
            ax2.set_ylim(y1, y2)
            if logify:
                ax2.set_xscale('log',nonposx='clip',subsx=(2,5))
                ax2.xaxis.set_minor_formatter(FormatStrFormatter('%2.4g'))
                ax2.xaxis.set_major_formatter(FormatStrFormatter('%2.4g'))
            ax2.tick_params('both', pad=2.5, size=3.5, width=1.0, which='both',labelsize=ticksize)

    def plot_spec(self,plot_maxprob=False, nufnu=True,
                  fig=None, spec=None, resid=None,
                  micron=False,normalize=True,
                  **kwargs):
        """Plot the spectroscopy for the model and data (with error bars), and
        plot residuals
            -- pass in a list of [res], can iterate over them to plot multiple results
        good complimentary color for the default one is '#FF420E', a light red
        """

        # pull out data
        mask = self.obs['mask']
        wave = self.obs['wavelength'][mask]
        if micron: wave /= 1e4
        specobs = self.obs['spectrum'][mask]
        specobs_unc = self.obs['unc'][mask]

        # model information
        modspec_lam = self.eout['obs']['lam_obs']
        if micron: modspec_lam /= 1e4
        spec_pdf = np.zeros(shape=(modspec_lam.shape[0],3))
        for jj in range(spec_pdf.shape[0]): spec_pdf[jj,:] = np.percentile(self.eout['obs']['spec'][:,jj],[16.0,50.0,84.0])
        modspec_lam = modspec_lam[mask]
        spec_pdf = spec_pdf[mask,:]
        spec_bfit = self.eout['obs']['spec'][0,mask]

        # units
        if nufnu:
            units = 3e18 / wave
            if normalize: units = 1./ (wave * np.median(specobs/wave))
            spec_pdf *= units[:,None]
            specobs *= units
            specobs_unc *= units
            spec_bfit *= units

        # plot observations
        self.specax.errorbar(wave, specobs, #yerr=specobs_unc,
                     color=self.data_color, label='observed', alpha=alpha, linestyle='-',lw=lw)

        # posterior median spectra
        self.specax.plot(modspec_lam, spec_pdf[:,1], linestyle='-',
                  color=self.posterior_color, alpha=alpha,label = 'posterior median', lw=lw)  
        self.specax.fill_between(modspec_lam, spec_pdf[:,0], spec_pdf[:,2],
                          color=self.posterior_color, alpha=0.3)
        self.specresidax.fill_between(modspec_lam, (specobs - spec_pdf[:,0]) / specobs_unc, (specobs - spec_pdf[:,2]) / specobs_unc,
                          color=self.posterior_color, alpha=0.8)

        # plot maximum probability model
        specchi = (specobs - spec_bfit) / specobs_unc
        if plot_maxprob:
            self.specax.plot(modspec_lam, spec_bfit, color=self.extra_color, 
                      linestyle='-', label = 'best-fit', alpha=alpha, lw=lw)
            self.specresidax.plot(modspec_lam, specchi, color=self.extra_color,
                       linestyle='-', lw=lw, alpha=alpha)        

        # calculate and show reduced chi-squared
        chisq = np.sum(specchi**2)
        ndof = mask.sum()
        reduced_chisq = chisq/(ndof)

        self.specax.text(0.01, 0.9, 'spectroscopic fit', fontsize=18,weight='bold',transform=self.specax.transAxes,color='k')
        self.specax.text(0.01, 0.81, r'best-fit $\chi^2$/N$_{\mathrm{spec}}$='+"{:.2f}".format(reduced_chisq),
                  fontsize=10, ha='left',transform = self.specax.transAxes,color='black')

        # calibration
        try:
            self.speccalibax.plot(modspec_lam,self.model._speccal,lw=2,color='k')
        except:
            self.speccalibax.plot(modspec_lam,np.ones_like(modspec_lam),lw=lw,color='k')

        # limits
        xlim = (wave.min()*0.95,wave.max()*1.05)
        self.specax.set_xlim(xlim)

        ymin, ymax = specobs.min()*0.9, specobs.max()*1.1
        self.specax.set_ylim(ymin, ymax)
        resid_ymax = np.min([np.abs(self.specresidax.get_ylim()).max(),5])
        self.specresidax.set_ylim(-resid_ymax,resid_ymax)

        # extra line
        for ax,factor in zip([self.specresidax,self.speccalibax],[0,1]):
            ax.axhline(factor, linestyle=':', color='grey')
            ax.yaxis.set_major_locator(MaxNLocator(5))
        self.speccalibax.set_ylim(0.79,1.21)

        # legend
        self.specax.legend(loc='upper right', prop={'size':10},
                    scatterpoints=1,fancybox=True)

        # set labels
        self.specresidax.set_ylabel( r'$\chi$',fontsize=fs)
        self.specax.set_ylabel(r'$\nu f_{\nu}$',fontsize=fs)
        self.speccalibax.set_ylabel('calibration\nvector',fontsize=fs)
        self.speccalibax.set_xlabel(r'$\lambda_{\mathrm{obs}}$ [$\AA$]',fontsize=fs)
        if micron: self.specresidax.set_xlabel(r'$\lambda_{\mathrm{obs}}$ [$\mu$m]',fontsize=fs)
        self.specresidax.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=ticksize)
        self.speccalibax.tick_params('both', pad=3.5, size=3.5, width=1.0, which='both',labelsize=ticksize)
        self.specax.tick_params('y', which='major', labelsize=ticksize)
        
        # set second x-axis (rest-frame wavelength)
        zred = self.eout['thetas']['zred']['q50']
        y1, y2 = self.specax.get_ylim()
        x1, x2 = self.specax.get_xlim()
        ax2 = self.specax.twiny()
        ax2.set_xlim(x1/(1+zred), x2/(1+zred))
        ax2.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\AA$]',fontsize=fs)
        ax2.set_ylim(y1, y2)
        ax2.tick_params('both', pad=2.5, size=3.5, width=1.0, which='both',labelsize=ticksize)

        # H-alpha, NII inset
        # create inset axis
        axi = inset_axes(ax2, width="25%", height="35%", loc="lower right",borderpad=2)

        bfit = self.spec_bfit.copy()
        bfit_nomarg = self.spec_nomarg.copy()
        bfit_nolines = self.spec_nolines.copy()
        if nufnu:
            bfit *= units
            bfit_nomarg *= units
            bfit_nolines *= units

        # plot region around H-alpha
        idx = (modspec_lam/(1+zred) > 6500) & (modspec_lam/(1+zred) < 6600)

        lw_inset = lw*2
        axi.plot(modspec_lam[idx], bfit[idx], linestyle='-',
                  color=self.posterior_color, alpha=alpha,label = 'nebular marginalization', lw=lw_inset)
        axi.plot(modspec_lam[idx], bfit_nomarg[idx], linestyle='-',
                  color=self.extra_color, alpha=alpha,label = 'CLOUDY grid', lw=lw_inset)
        axi.plot(modspec_lam[idx], bfit_nolines[idx], linestyle='-',
                  color='grey', alpha=alpha,label = 'continuum model', lw=lw_inset)
        axi.plot(modspec_lam[idx], specobs[idx], linestyle='-',
                  color=self.data_color, alpha=alpha,label = 'observed', lw=lw_inset)

        # labels
        axi.set_title(r'H$\alpha$ + [NII]',fontsize=fs*0.7,weight='semibold',pad=-3)
        for tl in axi.get_yticklabels():tl.set_visible(False)  # turn off y-labels
        axi.tick_params('both', which='both',labelsize=fs*0.4)
        axi.set_xlabel('obs. wavelength [$\AA$]',fontsize=fs*0.6,labelpad=-1.5)

        # legend
        axi.legend(prop={'size':4.5},loc='upper left')


























