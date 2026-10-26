#!/bin/bash
#SBATCH --job-name=cdx_reint
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_reint_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_reint_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "$(cat $WT/docs/ocean/fidelity/reinterp_2026-09-07.txt)

Repo pointers: the RGB shortwave branch is packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py around line 4020 and packages/ocean/legoesm/ocean/physics/shortwave_penetration.py; the TKE closure card is orca1_zdftke_config in scripts/run/run_omip_core2.py; the EVD composition is packages/ocean/legoesm/ocean/physics/vertical_mixing/k_profiles.py; NEMO source at /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE (traqsr.F90 for its RGB, zdftke.F90, zdfevd.F90) and the ORCA1 namelist at cfgs/ORCA1/EXP00/namelist_cfg. Verify claims against the code rather than accepting my summary." > "$LOG" 2>&1
rc=$?
echo "rc=$rc"; tail -50 "$LOG"
echo DONE
