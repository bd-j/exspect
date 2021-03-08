#!/usr/bin/bash

# Script to install code with same versions as used
# for fits and figures in prospector paper

# change this if you want to install elsewhere;
# or, copy and run this script in the desired location
CODEDIR=$HOME
cd $CODEDIR

# Install FSPS from source
git clone git@github.com:cconroy20/fsps
export SPS_HOME="$PWD/fsps"
cd $SPS_HOME/src
make clean
make all

# Create and activate environment (named 'prox')
git clone git@github.com:bd-j/exspect.git
cd exspect
conda env create -f environment_paper.yml
conda activate prox_paper
cd ..

# Install other repos from source
repos=( dfm/python-fsps bd-j/sedpy bd-j/prospector )
hashes=( a1edca9f8 1a20c81 fc0c36f )
for i in "${!repos[@]}"; do
    r="${repos[$i]}"
    git clone git@github.com:{$r}
    cd ${r##*/}
    git checkout "${hashes[$i]}"
    python -m pip install .
    git checkout main
    cd ..
done

cd exspect
python -m pip install .

echo "Add 'export SPS_HOME=${SPS_HOME}' to your .bashrc"