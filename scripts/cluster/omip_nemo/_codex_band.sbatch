#!/bin/bash
#SBATCH --job-name=cdx_band
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_band_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/_omip_wt
unset CODEX_HOME
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_band_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "$(cat $WT/docs/ocean/fidelity/band_mixing_claim_2026-09-08.txt)

Verify against the code rather than my summary. Our saved diffusivities come from scripts/run/run_omip_core2.py --kprofile-snapshots and the TKE closure in packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py; the band comparison itself is scripts/cluster/omip_nemo/_top20_mixing_ratio.sbatch; NEMO's avt/avm are in cfgs/ORCA1/EXP00/RUN_TRD2/*trd1h_T.nc and its closure in src/OCE/ZDF/zdftke.F90." > "$LOG" 2>&1
rc=$?
echo "rc=$rc"; tail -45 "$LOG"
echo DONE
