#!/bin/bash
#SBATCH --job-name=cdx_scorer
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_scorer_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/_omip_merge
codex exec --sandbox read-only -C $WT \
"Adversarial review of commit HEAD (git show HEAD): --use-mean-fields on scripts/validate/compare_omip_nemo.py (_load_legoesm use_mean, MLD block) and scripts/validate/ocean_fidelity/compare_three_way_nemo.py (_lego_mld, three load sites, flag), plus tests in tests/ocean/unit/test_compare_omip_nemo_bands.py. Context: NEMO 5-day files are window means (tos/sos plain hourly means; to/so = @toce_e3t/@e3t thickness-weighted; mldr10_1 hourly mean of MLDs). The driver writes T_mean, S_mean, T_mean_hw, S_mean_hw, mld_mean, flux_mean_window_s per snapshot when --state-accumulate --mld-accumulate. Numbered, severity-tagged, file:line: (1) any path where use_mean silently falls back to the instantaneous state or mixes the two (e.g. a consumer reading T3d[...,0] as SST, the also-mask footprint, the wet mask built from T3d NaNs on dry columns where T_mean_hw is NaN)? (2) does the three-way scorer's --expect-day / oracle-record pinning still describe the right window when the snapshot is a window mean ending at that day? (3) are the two new tests non-vacuous, and is anything downstream (plots, band stats, deep-MLD tail) broken by NaN in T_mean_hw on dry cells where T had a fill value? (4) any default behaviour changed for callers not passing the flag?" 2>&1 | tail -120
echo "=== codex exit ${PIPESTATUS[0]} ==="



