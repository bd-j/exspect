#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl
from .corner import _quantile
from prospect.models.priors import TopHat

__all__ = ["get_simple_prior", "sample_prior", "sample_posterior",
           "violinplot", "step"]


def get_simple_prior(prior, xlim, num=1000):
    xx = np.linspace(*xlim, num=num)
    px = np.array([prior(x) for x in xx])
    px = np.exp(px)
    return xx, px / px.max()


def sample_prior(model, nsample=1e6):
    """Generate samples from the prior.

    :param model:
        A ProspectorParams instance.

    :param nsample: (int, optional, default: 1000000)
        Number of samples to take
    """
    labels = model.free_params
    chain = np.zeros([nsample, model.ndim])
    #chain = []
    for l in labels:
        prior = model.config_dict[l]["prior"]
        if isinstance(prior, TopHat):
            val = np.linspace(prior.params["mini"], prior.params["maxi"], nsample)
            val = np.atleast_2d(val).T
        else:
            val = np.array([prior.sample() for i in range(int(nsample))])
        chain[:, model.theta_index[l]] = np.array(val)
        # chain.append()
    #chain = np.concatenate([c.T for c in chain]).T
    return chain, labels


def sample_posterior(chain, weights=None, nsample=int(1e4),
                     start=0, thin=1, extra=None):
    """
    :param chain:
        ndarray of shape (niter, ndim) or (niter, nwalker, ndim)

    :param weights:
        weights for each sample, of shape (niter,)

    :param nsample: (optional, default: 10000)
        Number of samples to take

    :param start: (optional, default: 0.)
        Fraction of the beginning of the chain to throw away, expressed as a float in the range [0,1]

    :param thin: (optional, default: 1.)
        Thinning to apply to the chain before sampling (why would you do that?)

    :param extra: (optional, default: None)
        Array of extra values to sample along with the parameters of the chain.
        ndarray of shape (niter, ...)
    """
    start_index = np.floor(start * (chain.shape[-2] - 1)).astype(int)
    if chain.ndim > 2:
        flatchain = chain[:, start_index::thin, :]
        nwalker, niter, ndim = flatchain.shape
        flatchain = flatchain.reshape(niter * nwalker, ndim)
    elif chain.ndim == 2:
        flatchain = chain[start_index::thin, :]
        niter, ndim = flatchain.shape

    if weights is not None:
        p = weights[start_index::thin]
        p /= p.sum()
    else:
        p = None

    inds = np.random.choice(niter, size=nsample, p=p)
    if extra is None:
        return flatchain[inds, :]
    else:
        return flatchain[inds, :], extra[inds, ...]


def violinplot(data, pos, widths, ax=None,
               violin_kwargs={"showextrema": False},
               color="slateblue", alpha=0.5, span=None, **extras):
    ndim = len(data)
    clipped_data = []

    if type(color) is str:
        color = ndim * [color]

    if span is None:
        span = [0.999999426697 for i in range(ndim)]

    for i, _ in enumerate(span):
        try:
            xmin, xmax = span[i]
        except(TypeError):
            q = [0.5 - 0.5 * span[i], 0.5 + 0.5 * span[i]]
            xmin, xmax = _quantile(data[i], q)
        good = (data[i] > xmin) & (data[i] < xmax)
        clipped_data.append(data[i][good])

    parts = ax.violinplot(data, positions=pos, widths=widths, **violin_kwargs)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(color[i])
        pc.set_alpha(alpha)


def step(xlo, xhi, y=None, ylo=None, yhi=None, ax=None,
         label=None, linewidth=2, **kwargs):
    """A custom method for plotting step functions as a set of horizontal lines
    """
    clabel = label
    for i, (l, h) in enumerate(zip(xlo, xhi)):
        if y is not None:
            ax.plot([l,h], [y[i], y[i]], label=clabel, linewidth=linewidth, **kwargs)
        if ylo is not None:
            ax.fill_between([l,h], [ylo[i], ylo[i]], [yhi[i], yhi[i]], linewidth=0, **kwargs)
        clabel = None

