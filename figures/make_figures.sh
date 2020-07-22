#!/bin/bash

# Make figures for prospector paper

python phot_mock_basic.py --fignum 1
python specphot_mock_compare --fignum 2
python gnz11.py --fignum 4