#!/bin/bash

#SBATCH -J exspect_basic
#SBATCH -n 20 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 12:00:00 # Runtime
#SBATCH -p conroy,shared # Partition to submit to
#SBATCH --constraint=intel
#SBATCH --mem-per-cpu=8000 #in MB
#SBATCH -o /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_basic_%A.out # Standard out goes to this file
#SBATCH -e /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_basic_%A.err # Standard err goes to this file

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
                        --outfile=../output/mock_parametric_$tag
