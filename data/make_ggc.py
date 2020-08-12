#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""make_ggc.py - Code for generating an HDF5 file of homogenous GGC data,
including spectra and photometry
"""

import os, glob
import numpy as np
import matplotlib.pyplot as pl

import h5py
from astropy.io import fits

from sedpy import observate

ggcdir = '.'


def make_dataset(objname, hfile):

    grp = hfile.create_group(objname)

    catdat = ggc_catalog_info(objname)
    info = grp.create_dataset("info", data=catdat)

    specdat, units = ggc_spec(objname)
    spec = grp.create_dataset("spec", data=specdat)
    spec.attrs.update(units)


def ggc_spec(objname, exp='a', ext='1', fluxtype=None,
             to_vacuum=True, **extras):
    """
    :param fluxtype: (default: None)
        Flag describing the the type of flux calibration:

        * None: Use the flux calibrated spectrum
        * 0: Use the variance-weighted, CR-cleaned, sky-subtracted,
          spectrum.
        * 1: Use the unweighted, uncleaned, sky-subtracted, spectrum.
          Don't ever use this.
    """
    name = objname.upper().strip().replace(' ', '')

    sfile = '{0}_{1}_{2}.fits'.format(name.upper(), exp, ext)
    sfile = os.path.join(ggcdir, 'spectra', sfile)
    auxfile = sfile.replace('.fits', '.aux.fits')
    if not os.path.exists(sfile):
        raise ValueError('{0} does not exist!'.format(sfile))

    shdr, sdata = fits.getheader(sfile), fits.getdata(sfile)
    ahdr, adata = fits.getheader(auxfile), fits.getdata(auxfile)
    crpix = (shdr['CRPIX1'] - 1)  # convert from FITS to numpy indexing
    assert sdata.shape[0] == adata.shape[1]
    assert sdata.ndim == 1

    try:
        cd = shdr['CDELT1']
    except(KeyError):
        cd = shdr['CD1_1']

    cols = ["wavelength", "spectrum", "unc", "sky", "calibration"]
    dt = np.dtype([(c, np.float) for c in cols])
    obs = np.zeros(len(sdata), dtype=dt)

    obs['wavelength'] = (np.arange(len(sdata)) - crpix) * cd + shdr['CRVAL1']
    if to_vacuum:
        obs['wavelength'] = observate.air2vac(obs['wavelength'])

    if fluxtype is None:
        obs['spectrum'] = sdata.copy()
        bunit = shdr['BUNIT']
        obs['calibration'] = adata[0, :] / obs['spectrum']
    else:
        obs['spectrum'] = adata[fluxtype, :].copy()
        bunit = 'Counts'
        obs['calibration'] = 1.0
    obs['unc'] = obs['spectrum'] / adata[3, :]
    obs['sky'] = adata[2, :] / obs["calibration"]

    units = {}
    units["flux_unit"] = bunit
    units["wave_unit"] = ("AA, Restframe," +
                          to_vacuum * " Vacuum" +
                          (~to_vacuum) * " Air")
    return obs, units


def ggc_catalog_info(objname):

    filters = ["U", "B", "V", "R", "I",
               "g", "r", "i", "z",
               "J", "H", "Ks",
               "FUV", "NUV"]

    harris_cols = {"ra": "RA2000", "dec": "DE2000", "l": "GLON", "b": "GLAT",
                   "dist": "Rsun", "vrad": "Vr", "feh_harris": "__Fe_H_", "hbr": "HBR",
                   "c": "c", "r_c": "Rc", "r_h": "Rh", "mu_V": "muV", "ebv": "E_B-V_"}

    cols = (harris_cols.keys() + filters +
            ["{}_err".format(f) for f in filters])

    dt = [(c, np.float) for c in cols]
    dt += [("griz_telescope", "S4")]
    row = np.zeros(1, dtype=np.dtype(dt))

    # Harris
    row = get_harris(objname, row, harris_cols)

    # Van der beke
    maggies, mags_unc, tel = vanderbeke_maggies(objname, redden=True)
    row["griz_telescope"] = tel
    for b, f, u in zip("griz", maggies, mags_unc):
        row[b] = -2.5 * np.log10(f)
        row["{}_err".format(b)] = 1.086 * u / f

    # 2MASS
    row = twomass_maggies(objname, row)

    # Convert to AB
    for b in "UBVRI":
        f = observate.Filter("bessell_{}".format(b))
        row[b] -= f.ab_to_vega
    for b in ["J", "H", "Ks"]:
        f = observate.Filter("twomass_{}".format(b))
        row[b] -= f.ab_to_vega

    return row


def get_harris(objname, row, harris_cols):

    harris = fits.getdata('photometry/harris97.fits')
    hnames = [n.upper().strip().replace(' ', '') for n in harris['ID']]
    hrow = harris[hnames.index(objname)]
    for k, v in list(harris_cols.iteritems()):
        row[k] = hrow[v]

    # look at all these gross correlations
    row["V"] = hrow["Vt"]
    row["B"] = hrow["__B-V_t"] + row["V"]
    row["U"] = hrow["__U-B_t"] + row["B"]
    row["R"] = row["V"] - hrow["__V-R_t"]
    row["I"] = row["V"] - hrow["__V-I_t"]

    return row


def vanderbeke_maggies(name, bands="griz", redden=True):
    """
    :returns maggies:
        g,r,i,z GC maggies within the half-light radius, on the AB
        system.
    :returns mags_unc:
        g,r,i,z *magnitude* uncertainties
    """
    datadir = os.path.join(ggcdir, "photometry")

    tel = "    "
    sep, tels = ['_', ''], ['ctio', 'sdss']
    for s, t in zip(sep, tels):
        try:
            cat = fits.getdata(os.path.join(datadir, '{}.fits'.format(t)))
            cnames = [n.upper().strip().replace(' ', '') for n in cat['Name']]
            opt = cat[cnames.index(name)]
            mags = np.array([opt['{}{}mag'.format(b, s)] for b in bands]).flatten()
            mags_unc = np.array([opt['e_{}{}mag'.format(b, s)] for b in bands]).flatten()
            tel = t
        except(ValueError):
            pass

    if redden & (tel != "    "):
        # Reapply the reddening that Vanderbeke corrected for.
        cat = fits.getdata(os.path.join(datadir, 'reddening.fits'))
        cnames = [n.upper().strip().replace(' ', '') for n in cat['name']]
        rc = cat[cnames.index(name)]
        red = np.array([rc['A_{}'.format(b)] for b in bands]).flatten()
        mags = mags + red

    if tel == "    ":
        mags = np.zeros(4) + np.nan
        mags_unc = np.zeros(4) + np.nan
        print('Warning: no optical photometry found for {0}'.format(name))

    return 10**(-0.4*mags), mags_unc, tel


def twomass_maggies(objname, row):
    """
    :returns maggies:
        J,H,K integrated GC maggies, on the Vega system.
    :returns mags_unc:
        J,H,K *magnitude* uncertainties
    """
    datadir = os.path.join(ggcdir, "photometry")

    r_c = 60.0 * row['r_c']
    r_t, r_h = r_c * 10**row['c'], 60.0 * row['r_h']

    tmass = fits.getdata(os.path.join(datadir, 'twomass.fits'))
    tnames = [n.upper().strip().replace(' ', '') for n in tmass['Name']]
    try:
        nir = tmass[tnames.index(objname)]
    except(ValueError):
        print('Warning: {0} not found in 2mass catalog, setting NIR mags to NaN'.format(objname))
        for b in ["J", "H", "Ks"]:
            row[b] = np.nan
            row["{}_err".format(b)] = np.nan
        return row
    integral = king(r_h, r_c=nir['Jrc'], r_t=r_t, f_0=1.0)[1]
    magzp = np.array([20.45, 20.90, 19.93])
    dn0 = np.array([nir[col] for col in ['Ja0', 'Ha0', 'Ka0']])
    dn_unc = np.array([nir['e_'+col] for col in ['Ja0', 'Ha0', 'Ka0']])
    maggies, unc = integral * dn0 * 10**(-0.4 * magzp), 1.086 * dn_unc/dn0

    for b, f, u in zip(["J", "H", "Ks"], maggies, unc):
        row[b] = -2.5 * np.log10(f)
        row["{}_err".format(b)] = 1.086 * u / f

    return row


def king(r, r_c=1., r_t=30., f_0=1.):
    x, xt = (r/r_c)**2, (r/r_t)**2
    xtc = (r_t/r_c)**2
    t1, t2 = 1 + x, 1 + xt
    k = f_0 * (1 - 1/(1 + xtc)**(0.5))**(-2)
    value = k * ((1 + x)**(-0.5) - (1 + xtc)**(-0.5))**(2)
    integral = (np.log(1 + x) + x/(1 + xt) -
                4 * ((1+x)**(0.5) - 1) / (1 + xt)**(0.5))
    integral *= np.pi * r_c**2 * k

    return value, integral


if __name__ == "__main__":

    names = glob.glob("spectra/*a_1.fits")
    objnames = [os.path.basename(n).split('_')[0] for n in names]

    import h5py
    with h5py.File("ggc.h5", "w") as hfile:
        for objname in objnames:
            make_dataset(objname, hfile)



