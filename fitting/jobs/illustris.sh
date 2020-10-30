#!/bin/bash

#SBATCH -J exspect_illustris
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 144:00:00 # Runtime
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

# list of possible SFHs
galaxies=(00 01 02 03 04 05)
igal=${galaxies[$SLURM_ARRAY_TASK_ID]}
sfh="--illustris_sfh_file=../data/illustris/illustris_sfh_galaxy${igal}.dat"

# model flags
model="--continuum_order 0"
# choose parametric or nonparametric
mtypes=(parametric nonparametric)
model=$model" --parametric_sfh"
model=$model" --nbins_sfh 14"

# data flags
data="--snr_phot=0 --snr_spec=100 --add_noise"

# fitting flags
fit="--dynesty --nested_method=rwalk"

# mock parameters
zred=0.01
logzsol=-0.3
dust2=0.5
logmass=10
mass=1e10
mock="--logzsol=${logzsol} --logmass=${logmass} --mass=${mass} --dust2=${dust2}"

mkdir -p output/illustris
python illustris.py $fit $model $data \
                    $mock $sfh --zred=$zred \
                    --outfile=output/illustris/illustris${igal}_${mtype}