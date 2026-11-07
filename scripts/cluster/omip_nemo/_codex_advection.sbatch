#!/bin/bash
#SBATCH --job-name=cdx_adv
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_adv_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_adv_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "$(cat $WT/docs/ocean/fidelity/advection_claim_2026-09-08.txt)

Verify against the code and the data rather than my summary. Pointers: the box budget instrument is scripts/validate/ocean_fidelity/equatorial_thermocline.py (--box-budget), the isopycnal one equatorial_thermocline_structure.py (--isopycnal); the vertical velocity and effective transports come from the model's own mass_flux_w in the snapshots; NEMO's are in cfgs/ORCA1/EXP00/RUN_GATEWAY/*grid_[TUVW].nc; the momentum path is packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py and ocean_pe_latlon_cgrid.py." > "$LOG" 2>&1
rc=$?
echo "rc=$rc"; tail -45 "$LOG"
echo DONE
