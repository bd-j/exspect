#SBATCH -J exspect_gnz11
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 6:00:00 # Runtime
#SBATCH -p conroy,shared # Partition to submit to
#SBATCH --constraint=intel
#SBATCH --mem-per-cpu=4000 #in MB
#SBATCH -o /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_gnz11_%A.out # Standard out goes to this file
#SBATCH -e /n/holyscratch01/conroy_lab/bdjohnson/exspect/fitting/logs/exspect_gnz11_%A.err # Standard err goes to this file

export GROUP=conroy_lab
export MYSCRATCH=$SCRATCH/$GROUP/$USER
export SPS_HOME=$SCRATCH/$GROUP/$USER/fsps

source activate prox
cd $MYSCRATCH/exspect/fitting


python photoz_GNz11.py --free_igm --add_neb --complex_dust --free_neb_met \
                       --nbins_sfh 5 --zmax=35 \
                       --dynesty --nested_method=rwalk --outfile output/photoz_gnz11
