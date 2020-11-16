#!/bin/bash

# Make figures for prospector paper
# you need to copy (or soft-link) timestamped output files to generic filenames

rdir=../fitting/output
nprior=100000
nseds=1000

mkdir -p paperfigures

# General rules
# --------------
# Inferences (best fit, PDFs, etc) have *color*
#  - use different colors for photometry, spectrum, phot + spec
# True input values are *black*
# Data (mock or real) is black/gray


# Basic figure
echo "Basic"
python basic_phot_mock.py --prior_samples=$nprior --n_seds=$nseds \
                          --fignum=basic --results_file=${rdir}/mock_parametric_phot.h5

# compare photometry, spectroscopy
echo "Phot + Spec"
python compare_mock_specphot.py  --prior_samples=$nprior --n_seds=0 \
                                 --fignum=mock_specphot \
                                 --phot_file=${rdir}/mock_parametric_phot.h5 \
                                 --spec_file=${rdir}/mock_parametric_spec.h5 \
                                 --specphot_file=${rdir}/mock_parametric_specphot.h5

# Illustris
echo "Illustris"
python show_illustris.py --fignum=illustris1

# nbands figure
echo "Nbands"
filtersets=(oneband twoband optical opt_nir uv_to_nir uv_to_mir full)
#filtersets=(oneband full)

for ((i = 0; i < ${#filtersets[@]}; ++i)); do
    # bash arrays are 0-indexed
    idx=$(( $i + 1 ));
    f=${filtersets[$i]};
    echo "$f,$idx"
    python show_nbands.py --prior_samples=$nprior --n_seds=$nseds \
                          --fignum=nband_${f} --results_file=$rdir/nband_fit_$f.h5 \
                          #--adjust_maxw
    cp paperfigures/nband_${f}.png paperfigures/nband${idx}.png
done


# GCs
echo "GC example"
python gc_dash.py --n_seds=$nseds --fignum=ggc1 --results_file=$rdir/ggc1.h5

# GC comparison
echo "GC comparison"
python gc_compare.py --fignum=ggc_all

# Photo-z
echo "Photo-z"
python show_gnz11.py --prior_samples=$nprior --n_seds=$nseds --fignum=gnz11 \
                     --results_file=${rdir}/photoz_gnz11.h5

# SDSS PSB
echo "PSB"
python plot_psb_sdss.py --prior_samples=$nprior --n_seds=$nseds --fignum=sdss_psb \
                        --results_file=${rdir}/psb_sdss.h5
