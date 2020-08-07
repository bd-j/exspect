#!/bin/bash

# Make figures for prospector paper
# you need to copy (or soft-link) timestamped output files to generic filenames

rdir=../fitting/output
nprior=10000
nseds=1000

mkdir -p paperfigures

# Basic figure
python basic_phot_mock.py --prior_samples=$nprior --n_seds=$nseds \
                          --fignum=basic --phot_file=${rdir}/mock_parametric_phot.h5

# compare photometry, spectroscopy
python compare_mock_specphot --prior_samples=$nprior --n_seds=$nseds \
                             --fignum=mock_compare \
                             --phot_file=${rdir}/mock_parametric_phot.h5 \
                             --phot_file=${rdir}/mock_parametric_spec.h5 \
                             --specphot_file=${rdir}/mock_parametric_specphot.h5

# nbands figure
filtersets=(oneband twoband optical opt_nir uv_to_nir uv_to_mir full)
for f in $filtersets;
  do python show_nbands.py --prior_samples=$nprior --n_seds=$nseds \
                           --fignum=nband_${f} --results_file=$rdir/nband_fit_$f.h5
done

# Photo-z
python show_gnz11.py --n_sample=1000 --n_seds=$nseds --fignum=gnz11 \
                     --results_file=${rdir}/photoz_gnz11.h5