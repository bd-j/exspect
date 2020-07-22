#!/bin/bash

# Make figures for prospector paper

python basic_phot_mock.py --fignum 1
python compare_mock_specphot --fignum 2
python show_gnz11.py --fignum 11 --make_seds