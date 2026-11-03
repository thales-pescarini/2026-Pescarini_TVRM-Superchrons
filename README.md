# Slow cooling and the geomagnetic polarity history link protracted thermoviscous remagnetization to superchrons

Code and derived data for Pescarini et al., submitted to *Geophysical Journal International*.

The scripts model thermoviscous resetting and continuous-cooling blocking of single-domain pyrrhotite and magnetite, and the vector cancellation of the acquired moment under a stochastic geomagnetic polarity history. The remagnetized carbonates of the Nama Group (Namibia) are the case study; the TROUT unmixing of the demagnetization data and the Raman thermometry that constrain the model are included. The scripts reproduce Figs 1 and 3–7 and Tables S1–S7.

Reference parameters (Table 1): T_peak = 266 °C, peak conditions from 507 Ma, cooling from 500 Ma, cooling rate 2 °C/Myr, f_Po = 0.85 below 320 °C, τ0 = 1e-9 s, laboratory time 40 min, Moyero Reversed Superchron (MRS) 485.0–466.1 Ma. The ensemble is the 31 specimens in `data/trout_run/ensemble_membership.txt`; the polarity ensemble is 400 histories with seeds 200–599.

## Requirements

Python ≥ 3.10, numpy ≥ 2.0, scipy, pandas, matplotlib. Fig. 1 also needs cartopy, shapely and pmagpy (Natural Earth coastlines are downloaded on the first run); the TROUT fits and Fig. 3 need pmagpy, jax, emcee, scikit-learn and adjustText.

```bash
git clone https://github.com/thales-pescarini/2026-Pescarini_TVRM-Superchrons.git
cd 2026-Pescarini_TVRM-Superchrons
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Contents

```
trout/          TROUT fits (trout_batch.py) and specimen selection (gates.py)
model/          blocking model (blocking.py) and the two ensemble inputs of Fig. 7
figures/        one script per figure
supplementary/  Tables S1-S7; reference results in supplementary/outputs/
data/           derived data
```

| Data | |
|---|---|
| `data/trout_run/C2_SPECTRA/` | C2 unblocking spectra of the 69 fitted specimens |
| `data/trout_run/PARAMS/` | TROUT parameters of the 69 fits |
| `data/trout_run/trout_status.csv` | fitting window and seed of every fit |
| `data/trout_run/ensemble_membership.txt` | the 31 ensemble specimens |
| `data/ensemble31_mean_C2_spectrum.csv` | ensemble-mean C2 spectrum (`model/build_mean_spectrum.py`) |
| `data/ensemble31_instantaneous_blocking_2C_Myr.npz` | blocking ages and moments of all elements at 2 °C/Myr (`model/build_blocking_npz.py`) |
| `data/raman_spectra/` | the 20 Raman spectra of carbonaceous material |
| `data/nama_rscm_per_measurement.csv`, `data/nama_rscm_per_sample.csv` | band fits and peak temperatures (`figures/fig4_rscm.py`) |
| `data/Fig5_fractions_per_specimen.csv`, `data/Fig6_per_specimen.csv` | per-specimen outputs of Figs 5 and 6 |

The demagnetization data are those of Pescarini et al. (2025), https://doi.org/10.5281/zenodo.13974746. The Raman spectra, with the native instrument files, are at https://doi.org/10.5281/zenodo.23896190.

## Running

From the repository root; outputs go to the working directory.

```
python figures/fig1_gondwana_apwp.py
python figures/fig4_rscm.py
python figures/fig5_pullaiah_reset.py
python figures/fig6_blocking_ages.py
python figures/fig7_coherence.py
python figures/fig3_trout_examples.py <MagIC data root>

python supplementary/tableS1_fpo.py
python supplementary/tableS2_plate_motion.py
python supplementary/tableS3_convergence.py
python supplementary/tableS4_S7_build.py
python supplementary/tableS5_reset_Tpeak.py
python supplementary/tableS6_Tpeak_rate.py
```

Fig. 2 (map) is not produced by code here. Fig. S1 shows the spectra in `data/trout_run/C2_SPECTRA/`. `fig7_coherence.py` also writes `Fig7_MRS_SHARE.json` and prints the time-step convergence test of Text S3. The supplementary scripts write to `supplementary/outputs/`; Table S3 takes about 10 min, the others a few minutes or less. Results may differ in the last digit with other numpy and scipy versions.

## Rebuilding the ensemble inputs

The reference inputs used by Fig. 7 are included in `data/`. To rebuild them without replacing those files:

```bash
python model/build_mean_spectrum.py
python model/build_blocking_npz.py
```

Both commands read `data/trout_run/` and write their products to the current working directory.

## TROUT

`trout/trout_batch.py <data root> [output directory]` fits two-component TROUT models (Cych et al. 2023). Its default output is `trout_run_new/`; the archived reference fits are in `data/trout_run/`. `trout/gates.py [run directory]` selects the ensemble specimens. To refit into a separate directory and apply the selection:

```bash
python trout/trout_batch.py "<MagIC data root>" trout_run_new
python trout/gates.py trout_run_new
```

The MagIC tables (`measurements.txt`, `specimens.txt`, `samples.txt`, `sites.txt`) of Pescarini et al. (2025) are read from

```
<data root>/_TROUT_MAGIC_STAGING/DT   <data root>/1B/1B_TT_set1_   <data root>/1B/1B_TT_set2_
<data root>/1G                        <data root>/1A/1A_T_1_       <data root>/1A/1A_T_2_
```

`trout.py` is not included. Download it from https://github.com/bcych/TROUT (Cych et al. 2023) and place it in `trout/`. The fits in `data/trout_run/` were made with the version of `trout.py` with MD5 checksum 560a52ca8babb695ca8e6ce932ace96d.

## References

Cych, B., Morzfeld, M., Heslop, D., Maher, S., Gee, J. & Tauxe, L., 2023. Thermal Resolution of Unblocking Temperatures (TROUT): a method for "unmixing" multi-component magnetizations, *Geochem. Geophys. Geosyst.*, 24, e2023GC010920, https://doi.org/10.1029/2023GC010920.

Pescarini, T., Trindade, R.I.F., Evans, D.A.D., Kirschvink, J.L., Pierce, J. & Fernandes, H.A., 2025. Magnetic mineralogy and paleomagnetic record of the Nama Group, Namibia: implications for the large-scale remagnetization of West Gondwanaland and its tectonic evolution, *J. geophys. Res.: Solid Earth*, 130, e2024JB030612, https://doi.org/10.1029/2024JB030612.

## License

MIT.
