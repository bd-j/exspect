#!/bin/bash

#SBATCH -J exspect_nbands
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 36:00:00 # Runtime
#SBATCH -p conroy,shared # Partition to submit to
#SBATCH --constraint=intel
#SBATCH --mem-per-cpu=4000 #in MB
#SBATCH -o /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_nbands_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_nbands_%A_%a.err # Standard err goes to this file

module purge
module load gcc/9.2.0-fasrc01
module load Anaconda3/5.0.1-fasrc01

export GROUP=conroy_lab
export MYSCRATCH=$SCRATCH/$GROUP/$USER
export SPS_HOME=$SCRATCH/$GROUP/$USER/fsps

source activate prox
cd $MYSCRATCH/exspect/fitting

# list of possible filtersets
filtersets=(oneband twoband optical opt_nir uv_to_nir uv_to_mir full)
filterset=${filtersets[$SLURM_ARRAY_TASK_ID]}

# model and data flags
model="--add_neb --add_duste --complex_dust --free_duste"
data="--snr_phot=20 --add_noise"

# fitting flags
fit="--dynesty --nested_method=rwalk"

# mock parameters
zred=0.1
logzsol=-0.3
dust2=0.5
logmass=10
nbins_sfh=6
duste_umin=2
duste_qpah=1
fagn=0.05
agn_tau=20

python nbands_demo.py $fit $model $data --filterset=$filterset \
                      --zred=$zred \
                      --nbins_sfh=$nbins_sfh --logzsol=$logzsol --logmass=$logmass --dust2=$dust2 \
                      --duste_umin=$duste_umin --duste_qpah=$duste_qpah --fagn=$fagn --agn_tau=$agn_tau \
                      --outfile=output/nband_fit_$filterset