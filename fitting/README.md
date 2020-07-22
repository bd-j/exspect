# Scripts for conducting fits on Cannon

## Mock parametric

``sh
python specphot_demo.py --add_neb --add_duste --zred=0.1 --zred_disp=1e-3 \
                        --snr_spec=0 --snr_phot=20 --add_noise \
                        --outfile=../output/mock_parametric_phot

## Photo-z

```sh
python photoz_GNz11.py --free_igm --add_neb --complex_dust --free_neb_met --nbins_sfh 5 \
                       --dynesty --nested_method=rwalk --outfile photoz_gnz11
```