#!/bin/bash

#SBATCH -J exspect_psb
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 144:00:00 # Runtime
#SBATCH -p conroy,shared # Partition to submit to
#SBATCH --constraint=intel
#SBATCH --mem-per-cpu=4000 #in MB
#SBATCH -o /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_psb_%A.out # Standard out goes to this file
#SBATCH -e /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_psb_%A.err # Standard err goes to this file

module purge
module load gcc/9.2.0-fasrc01
module load Anaconda3/5.0.1-fasrc01

export GROUP=conroy_lab
export MYSCRATCH=$SCRATCH/$GROUP/$USER
export SPS_HOME=$SCRATCH/$GROUP/$USER/fsps

source activate prox
cd $MYSCRATCH/exspect/fitting

# data flags
objnum=92942
data="--objname $objnum --zred=0.073"

# model flags
model="--continuum_order=12 --add_neb --free_neb_met --marginalize_neb"
model=$model" --nbins_sfh=8 --jitter_model --mixture_model"

# fitting flags
fit="--dynesty --nested_method=rwalk --nlive_batch=200 --nlive_init 500"
fit=$fit" --nested_dlogz_init=0.01 --nested_posterior_thresh=0.03"


mkdir -p output/psb_results
python psb_params.py $fit $model $data \
                     --outfile=output/psb_results/psb_$objnum