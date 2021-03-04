#!/usr/bin/bash

# change this if you want to install elsewhere;
# or, copy and run this script in the desired location
CODEDIR=$PWD
cd $CODEDIR

# Install FSPS from source
git clone git@github.com:cconroy20/fsps
export SPS_HOME="$PWD/fsps"
#cd $SPS_HOME/src
#make clean
#make all

# Create and activate environment (named 'prox')
git clone git@github.com:bd-j/exspect.git
cd exspect
conda env create -f environment.yml
conda activate prox
cd ..

# Install other repos from source
repos=( bd-j/prospector )
for r in "${repos[@]}"; do
    git clone git@github.com:$r
    cd ${r##*/}
    python setup.py install
    cd ..
done

echo "Add 'export SPS_HOME=${SPS_HOME}' to your .bashrc"