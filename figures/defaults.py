#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

# colorcycle = ["slateblue", "maroon", "orange"]
colorcycle = ["royalblue", "firebrick", "indigo", "darkorange", "seagreen"]


def plot_defaults(rcParams):
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["STIXGeneral"]
    rcParams["font.size"] = 12
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.rm"] = "serif"
    rcParams["mathtext.sf"] = "serif"
    rcParams['mathtext.it'] = 'serif:italic'
    return rcParams


pretty = {"logzsol": r"$\log (Z_{\star}/Z_{\odot})$",
          "logmass": r"$\log {\rm M}_{\star, {\rm formed}}$",
          "gas_logu": r"${\rm U}_{\rm neb}$",
          "gas_logz": r"$\log (Z_{\neb}/Z_{\odot})$",
          "dust2": r"$\tau_{\rm V}$",
          "av": r"${\rm A}_{\rm V, diffuse}$",
          "av_bc": r"${\rm A}_{\rm V, young}$",
          "dust_index": r"$\Gamma_{\rm dust}$",
          "igm_factor": r"${\rm f}_{\rm IGM}$",
          "duste_umin": r"$U_{\rm min, dust}$",
          "duste_qpah": r"$Q_{\rm PAH}$",
          "duste_gamma": r"$\gamma_{\rm dust}$",
          "log_fagn": r"$\log({\rm f}_{\rm AGN})$",
          "agn_tau": r"$\tau_{\rm AGN}$",
          "mwa": r"$\langle t \rangle_M$ (Gyr)",
          "ssfr": r"$\log ({\rm sSFR})$ $({\rm M}_\odot/{\rm yr})$",
          "tau": r"$\tau$ (Gyr)",
          "logtau": r"$\log(\tau)$ (Gyr)",
          "tage": r"Age (Gyr)",
          "ageprime": r"Age/$\tau$"}
