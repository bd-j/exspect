# Scripts for conducting fits

The parameter files here can be run, with suitable options, to perform
inferences with Prospector as presented in \paper\.

Detailed descriptions of the options provided are given in the SLURM submission
scripts in the `jobs` directory.

## Section 3: Mocks

### S3.1: Mock parametric

```sh
mock="--zred=0.1 --tau=4 --tage=12 --logzsol=-0.3 --mass=1e10 --dust2=0.3"
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

```sh
igal=01
sfh="--illustris_sfh_file=../data/illustris/illustris_sfh_galaxy${igal}.dat"
mock="--logzsol=-0.3 --logmass=10 --mass=1e10 --dust2=0.5"
data="--snr_phot=0 --snr_spec=100 --add_noise"
fit="--dynesty --nested_method=rwalk"

# Non-parametric
model="--continuum_order 0  --nbins_sfh 14"
python illustris.py $fit $model $data \
                    $mock $sfh --zred=$zred \
                    --outfile=output/illustris/illustris${igal}_nonparametric

# parametric
model="--continuum_order 0  --parametric_sfh"
python illustris.py $fit $model $data \
                    $mock $sfh --zred=0.01 \
                    --outfile=output/illustris/illustris${igal}_parametric
```

### S3.3: Inference as a function of number of bands

```sh
zred=0.1
nbins_sfh=6
mock="--logzsol=-0.3 --logmass=10 --dust2=0.5"
mock=$mock" --duste_umin=2 --duste_qpah=1"
mock=$mock" --fagn=0.05 --agn_tau=20"

data="--snr_phot=20 --add_noise"
opts="--add_neb --add_duste --complex_dust --free_duste"
fit="--dynesty --nested_method=rwalk"

# filters to use
filterset=optical

python nbands_demo.py $fit $opts $data --filterset=$filterset \
                      $mock --zred=$zred --nbins_sfh=$nbins_sfh  \
                      --outfile=output/nbands_fit_$filterset
```

## Section 4: Real data

### Galactic Globular Clusters (S4.1)

```sh
ggc_index=1
data="--ggc_data=../data/ggc.h5 --ggc_index=${ggc_index} --mask_elines"
opts="--jitter_model --add_realism --continuum_order=15"
fit="--dynesty --nested_method=rwalk"

python ggc.py $fit $opts $data \
              --outfile=output/ggc_id$ggc_index
```

### Photo-z (S4.2)

```sh
opts="--free_igm --add_neb --complex_dust --free_neb_met"
fit="--dynesty --nested_method=rwalk"

python photoz_GNz11.py $fit $opts --nbins_sfh=5 \
                       --outfile output/photoz_gnz11
```

### SDSS post-starburst (S4.3)

```sh
data="--objname 92942 --zred=0.073"
model="--continuum_order=12 --add_neb --free_neb_met --marginalize_neb"
model=$model" --nbins_sfh=8 --jitter_model --mixture_model"
fit="--dynesty --nested_method=rwalk --nlive_batch=200 --nlive_init 500"
fit=$fit" --nested_dlogz_init=0.01 --nested_posterior_thresh=0.03"

python psb_params.py $fit $model $data \
                     --outfile=output/psb_results/psb_92942
```