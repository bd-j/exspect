#!/bin/bash

#SBATCH -J exspect_basic
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 12:00:00 # Runtime
#SBATCH -p conroy,shared # Partition to submit to
#SBATCH --constraint=intel
#SBATCH --mem-per-cpu=4000 #in MB
#SBATCH -o /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_basic_%A_%a.out # Standard out goes to this file
#SBATCH -e /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_basic_%A_%a.err # Standard err goes to this file

module purge
module load git/2.17.0-fasrc01
module load gcc/9.2.0-fasrc01
module load Anaconda3/5.0.1-fasrc01

export GROUP=conroy_lab
export MYSCRATCH=$SCRATCH/$GROUP/$USER
export SPS_HOME=$SCRATCH/$GROUP/$USER/fsps

source activate prox
cd $MYSCRATCH/exspect/fitting

tags=(phot spec specphot)
snrp=(20 0 20)
snrs=(0 10 10)

tag=${tags[$SLURM_ARRAY_TASK_ID]}
snr_phot=${snrp[$SLURM_ARRAY_TASK_ID]}
snr_spec=${snrs[$SLURM_ARRAY_TASK_ID]}

# model flags
model="--add_neb --add_duste"

# fitting flags
fit="--dynesty --nested_method=rwalk"

# mock parameters
zred=0.1
logzsol=-0.3
dust2=0.3
mass=1e10
tau=4
tage=12

# photometry only
python specphot_demo.py $fit $model \
                        --snr_spec=$snr_spec --snr_phot=$snr_phot --add_noise --continuum_optimize \
                        --zred=$zred --zred_disp=1e-3 \
                        --tau=$tau --tage=$tage --logzsol=$logzsol --mass=$mass --dust2=$dust2 \
                        --outfile=output/mock_parametric_$tag
