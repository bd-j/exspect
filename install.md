# Installation on Cannon

## Setup

Add to your `.bashrc` or just do before you run any code:

```sh
export GROUP=conroy_lab
export MYSCRATCH=$SCRATCH/$GROUP/$USER

module purge
module load git/2.17.0-fasrc01
module load gcc/9.2.0-fasrc01
module load Anaconda3/5.0.1-fasrc01
export SPS_HOME=$SCRATCH/$GROUP/$USER/fsps
export F90FLAGS=-fPIC
```

## Build the environment:

```sh
cd $MYSCRATCH
git clone git@github.com:bd-j/exspect.git
cd exspect
conda env create -f environment.yml
conda activate prox
cd ..

# Install FSPS from source
git clone git@github.com:cconroy20/fsps
export SPS_HOME="$PWD/fsps"
cd $SPS_HOME/src
make clean
make all

# Install other repos from source
repos=( dfm/python-fsps bd-j/sedpy bd-j/prospector )
for r in "${repos[@]}"; do
    git clone git@github.com:$r
    cd ${r##*/}
    python setup.py install
    cd ..
done

echo "Add 'export SPS_HOME=${SPS_HOME}' to your .bashrc"
```

## Run code

e.g.:

```sh
export GROUP=conroy_lab
export SPS_HOME=$SCRATCH/$GROUP/$USER/fsps
source activate prox
cd fitting
python nbands_demo.py --dynesty --outfile=../output/hello_world
```