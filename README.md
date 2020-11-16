# Exspect

Example code for fits and plots, as presented in |paper|

## Installation

If you have Anaconda installed, follow the steps in `install_conda.sh` to install the relevant codes into an environment named `prox`

## Running Fits

The `fitting` directory contains parameter files that can be used to conduct fits, as described in the README.  These fits will produce date stamped files including posterior samples/chains in the `fitting/output/` directory.

## Making plots

First, soft-link the date stamped filenames to more generic filenames.  Then, run `make_figures.sh` in the `figures/` directory, which will produce output in the `figures/paperfigures/` directory, using the scripts in the `figures/` directory.
