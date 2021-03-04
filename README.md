# Exspect

Example code for fits and plots with [prospector](https://prospect.readthedocs.io/en/latest/), as presented in [2012.01426](https://arxiv.org/abs/2012.01426)

## Installation

If you have Anaconda installed, follow the steps in `conda_install.sh` to install the relevant codes into an environment named `prox`.  The `environment_paper.yml` files gives the code versions used to generate fits and plots in the paper; `environment.yml` is a more modern build.

## Running Fits

The `fitting` directory contains parameter files that can be used to conduct fits, as described in the README.  These fits will produce date stamped files including posterior samples/chains in the `fitting/output/` directory.

## Making plots

First, soft-link the date stamped filenames to more generic filenames.  Then, run `make_figures.sh` in the `figures/` directory, which will produce output in the `figures/paperfigures/` directory, using the scripts in the `figures/` directory.
