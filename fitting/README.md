# Scripts for conducting fits

## Section 3: Mocks

### S3.1: Mock parametric

```sh
# mock parameters
zred=0.1
logzsol=-0.3
dust2=0.3
mass=1e10
tau=4
tage=12
mock="--zred=${zred} --tau=$tau --tage=$tage --logzsol=$logzsol --mass=$mass --dust2=$dust2"

opts="--add_duste --add_neb"
data="--add_noise --mask_elines --continuum_optimize"
fit="--dynesty --nested_method=rwalk"

# photometry only
python specphot_demo.py $fit $mock $opts --zred_disp=1e-3 \
                        $data --snr_spec=0 --snr_phot=20 \
                        --outfile=../output/mock_parametric_phot

# spectroscopy only
python specphot_demo.py $fit $mock $opts --zred_disp=1e-3 \
                        $data --snr_spec=10 --snr_phot=0 \
                        --outfile=../output/mock_parametric_spec

# photometry + spectroscopy
python specphot_demo.py $fit $mock $opts --zred_disp=1e-3 \
                        $data --snr_spec=10 --snr_phot=20 \
                        --outfile=../output/mock_parametric_specphot
```

### S3.2: Illustris SFHs

### S3.3: Inference as a function of number of bands

```sh

# mock parameters
zred=0.1
logzsol=-0.3
dust2=0.5
logmass=10
duste_umin=2
duste_qpah=1
fagn=0.05
agn_tau=20
nbins_sfh=6
mock="--logzsol=${logzsol} --logmass=${logmass} dust2=${dust2}"
mock=$mock" --duste_umin=${duste_umin} --duste_qpah=${duste_qpah}"
mock=$mock" --fagn=${fagn} --agn_tau=${agn_tau}"

opts="--add_neb --add_duste --complex_dust --free_duste"
data="--snr_phot=20 --add_noise"
fit="--dynesty --nested_method=rwalk"

# filters to use
filterset=optical

python nbands_demo.py $fit $opts $data --filterset=$filterset \
                      $mock --zred=$zred --nbins_sfh=$nbins_sfh  \
                      --outfile=../output/nbands_fit_$filterset
```

## Section 4: Real data

### Galactic Globular Clusters (S4.1)

```sh
ggc_index=1
opts="--jitter_model --outlier_model --continuum_order=15"
fit="--dynesty --nested_method=rwalk"
data="--ggc_data=../data/ggc.h5 --ggc_index=${ggc_index} --mask_elines"

python ggc.py $fit $opts $data \
              --outfile=../output/ggc_$ggc_index
```

### Photo-z (S4.2)

```sh
python photoz_GNz11.py --free_igm --add_neb --complex_dust --free_neb_met --nbins_sfh 5 \
                       --dynesty --nested_method=rwalk --outfile photoz_gnz11
```

### SDSS post-starburst (S4.3)