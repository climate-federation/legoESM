#!/bin/bash
#SBATCH --job-name=cdx_refr
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_refr_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_refr_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "$(cat $WT/docs/ocean/fidelity/reframing_2026-09-08.txt)

Verify against the code, not my summary. Pointers: the TKE closure card is orca1_zdftke_config in scripts/run/run_omip_core2.py; the closure itself packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py and k_profiles.py; the delivered diffusivity comes from diagnose_vertical_K in packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py; the RGB shortwave branch is in ocean_pe_latlon_cgrid.py near line 4020 and packages/ocean/legoesm/ocean/physics/shortwave_penetration.py; NEMO source at /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE (zdftke.F90, traqsr.F90, sbcblk.F90) and the ORCA1 namelist at cfgs/ORCA1/EXP00/namelist_cfg." > "$LOG" 2>&1
rc=$?
echo "rc=$rc"; tail -45 "$LOG"
echo DONE
