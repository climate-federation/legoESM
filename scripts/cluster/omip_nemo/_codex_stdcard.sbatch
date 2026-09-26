#!/bin/bash
#SBATCH --job-name=cdx_std
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_std_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_std_body_${SLURM_JOB_ID}.txt
echo "=== repo $WT at $(git -C $WT rev-parse --short HEAD) ==="
codex exec --sandbox read-only -C $WT \
"Adversarially review commit 54d535c31 (git show 54d535c31): it declares a
standard OMIP 1-degree configuration and adds an MPAS cross-grid arm.
ATTACK:
1. run_standard_faithful_1deg.sbatch claims to be BYTE-IDENTICAL (in the
   model card) to scripts/cluster/omip_nemo/_longrun_180d_merged.sbatch
   except for parameterized days/name/snapshot cadence.  Diff the two
   run_omip_core2.py invocations flag by flag; any silent difference in a
   physics flag is a CRITICAL finding (the standard would not be the
   validated card).
2. The YEARS computation: python print(days/365.0) into --years — check
   run_omip_core2.py parses --years as float days*365 consistently with the
   long180 run's --years 0.4932 for 180 days (0.4932*365=180.0).  For
   DAYS=90 the standard gives 0.24657...; does the driver's step-count
   rounding produce day-90 snapshots (snapshot_day0090.npz) or day-89?
   Check how snapshot filenames are generated.
3. _mpas_modfrc_d30.sbatch: verify every flag passed is legal on --grid mpas
   per the guards in run_omip_core2.py (dm2dc/isf/bbl/sw-rgb-chl/iwm/
   nemo-monthly-init/tke-* are hard errors on MPAS — confirm none slipped
   in), and that omitting --min-levels and --mesh is correct for mpas.
4. The scoring loops call compare_omip_nemo.py with --nemo-time-idx 5/11/17
   for days 30/60/90.  Verify against the GATEWAY file's 5-day record
   convention (rec 5 = days 26-30) documented in _rescore_vs_gateway.sbatch.
Number findings, severity-tagged; GO/NO-GO for using the standard card."  > "$LOG" 2>&1
rc=$?
echo "=== codex rc=$rc ==="
grep -qiE "Failed to refresh token|refresh token was already used|not logged in" "$LOG" \
  && echo "FATAL: codex auth failed -- review DID NOT RUN"
cat "$LOG"
echo DONE
