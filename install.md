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
git clone git@github.com:cconroy20/fsps.git
git clone git@github.com:dfm/python-fsps.git
git clone git@github.com:bd-j/sedpy.git
git clone git@github.com:bd-j/prospector.git

cd exspect
conda env create -f environment.yml
source activate prox

cd ../fsps/src
make clean
make all

cd ../../python-fsps
python setup.py install

cd ../sedpy
python setup.py install

cd ../prospector
python setup.py install

cd ../exspect
git pull
python setup.py install
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