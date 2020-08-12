#!/bin/bash

#SBATCH -J exspect_gcs
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 12:00:00 # Runtime
#SBATCH -p conroy,shared # Partition to submit to
#SBATCH --constraint=intel
#SBATCH --mem-per-cpu=4000 #in MB
#SBATCH -o /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_gcs_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_gcs_%A_%a.err # Standard err goes to this file

export GROUP=conroy_lab
export MYSCRATCH=$SCRATCH/$GROUP/$USER
export SPS_HOME=$SCRATCH/$GROUP/$USER/fsps

source activate prox
cd $MYSCRATCH/exspect/fitting

ggc_index=$SLURM_ARRAY_TASK_ID

opts="--jitter_model --add_realism --continuum_order=15"
fit="--dynesty --nested_method=rwalk"
data="--ggc_data=../data/ggc.h5 --ggc_index=${ggc_index} --mask_elines"

python ggc.py $fit $opts $data \
              --outfile=../output/ggc_$ggc_index
