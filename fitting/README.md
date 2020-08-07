# Scripts for conducting fits

## S3.1: Mock parametric

```sh
# mock parameters
zred=0.1
logzsol=-0.3
dust2=0.3
mass=1e10
tau=4
tage=12

# photometry only
python specphot_demo.py --add_duste --zred=$zred --zred_disp=1e-3 \
                        --tau=$tau --tage=$tage --logzsol=$logzsol --mass=$mass dust2=$dust2 \
                        --snr_spec=0 --snr_phot=20 --add_noise \
                        --dynesty --nested_method=rwalk \
                        --outfile=../output/mock_parametric_phot

# spectroscopy only
python specphot_demo.py --add_duste --zred=$zred --zred_disp=1e-3 \
                        --tau=$tau --tage=$tage --logzsol=$logzsol --mass=$mass dust2=$dust2 \
                        --snr_spec=10 --snr_phot=0 --add_noise --continuum_optimize \
                        --dynesty --nested_method=rwalk \
                        --outfile=../output/mock_parametric_spec

# photometry + spectroscopy
python specphot_demo.py --add_duste --zred=$zred --zred_disp=1e-3 \
                        --tau=$tau --tage=$tage --logzsol=$logzsol --mass=$mass dust2=$dust2 \
                        --snr_spec=10 --snr_phot=20 --add_noise --continuum_optimize \
                        --dynesty --nested_method=rwalk \
                        --outfile=../output/mock_parametric_specphot
```

## S3.2: Illustris SFHs

## S3.3: Inference as a function of number of bands

```sh
# mock parameters
zred=0.1
logzsol=-0.3
dust2=0.5
logmass=10
nbins_sfh=6
duste_umin=2
duste_qpah=1
fagn=0.05
agn_tau=20
# filters to use
filterset=optical

python nbands_demo.py --add_neb --add_duste --complex_dust --zred=$zred --zred_disp=1e-3 \
                      --nbins_sfh=$nbins_sfh --logzsol=$logzsol --logmass=$logmass dust2=$dust2 \
                      --duste_umin=$duste_umin --duste_qpah=$duste_qpah --fagn=$fagn --agn_tau=$agn_tau \
                      --free_duste \
                      --snr_phot=20 --add_noise --filterset=$filterset
```

## Galactic Globular Clusters (S4.1)

## Photo-z (S4.2)

```sh
python photoz_GNz11.py --free_igm --add_neb --complex_dust --free_neb_met --nbins_sfh 5 \
                       --dynesty --nested_method=rwalk --outfile photoz_gnz11
```

## SDSS post-starburst (S4.3)