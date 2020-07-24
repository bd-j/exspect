# Scripts for conducting fits

## Mock parametric (S3.1)

```sh
# mock parameters
zred=0.1
tau=4
tage=12
logzsol=-0.3
dust2=0.3
mass=1e10

python specphot_demo.py --add_duste --zred=$zred --zred_disp=1e-3 \
                        --tau=$tau --tage=$tage --logzsol=$logzsol --mass=$mass dust2=$dust2 \
                        --snr_spec=0 --snr_phot=20 --add_noise \
                        --dynesty --nested_method=rwalk \
                        --outfile=../output/mock_parametric_phot --debug

python specphot_demo.py --add_duste --zred=$zred --zred_disp=1e-3 \
                        --tau=$tau --tage=$tage --logzsol=$logzsol --mass=$mass dust2=$dust2 \
                        --snr_spec=10 --snr_phot=0 --add_noise --continuum_optimize \
                        --outfile=../output/mock_parametric_spec

python specphot_demo.py --add_duste --zred=$zred --zred_disp=1e-3 \
                        --tau=$tau --tage=$tage --logzsol=$logzsol --mass=$mass dust2=$dust2 \
                        --snr_spec=10 --snr_phot=20 --add_noise --continuum_optimize \
                        --outfile=../output/mock_parametric_specphot
```

## Illustris SFHs (S3.2)

## Inference as a function of number of bands (S3.3)

## Galactic Globular Clusters (S4.1)

## Photo-z (S4.2)

```sh
python photoz_GNz11.py --free_igm --add_neb --complex_dust --free_neb_met --nbins_sfh 5 \
                       --dynesty --nested_method=rwalk --outfile photoz_gnz11
```

## SDSS post-starburst (S4.3)