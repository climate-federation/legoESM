#!/bin/bash
#SBATCH --job-name=cdx_lens
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_lens_%j.out
# Claim review BEFORE code: the cold-tongue warm lens / missing EVD claim.
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_lens_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "$(cat $WT/docs/ocean/fidelity/lens_claim_2026-09-06.txt)

Repo pointers for verification: NEMO source at /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/src/OCE/ZDF/zdfevd.F90, zdftke.F90, zdfphy.F90; ORCA1 namelist cfgs/ORCA1/EXP00/namelist_cfg; our EVD module packages/ocean/legoesm/ocean/physics/convection/enhanced_diffusion.py + config.py; driver scripts/run/run_omip_core2.py (--convection at ~line 5951, EnhancedDiffusionConfig built at ~line 1268); RGB branch packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py ~line 4020." > "$LOG" 2>&1
rc=$?
echo "rc=$rc"; tail -80 "$LOG"
echo DONE
