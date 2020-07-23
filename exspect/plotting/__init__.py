#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
