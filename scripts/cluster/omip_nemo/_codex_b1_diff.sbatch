#!/bin/bash
#SBATCH --job-name=cdx_b1
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:50:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_b1_%j.out
set -u
WT=/burg-archive/glab/users/pg2328/legoESM
unset CODEX_HOME
LOG=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_b1_${SLURM_JOB_ID}.log
codex exec --sandbox read-only -C $WT \
  "Adversarially review commit e237821bd (git show e237821bd): --grid fesom
stage B1 in scripts/run/run_omip_core2.py (build_fesom_ocean +
validate_fesom_stage + run_fesom_b1_smoke + CLI tests). Attack ONLY:
(1) _FESOM_B1_UNSUPPORTED completeness — grep the driver for OMIP
selectors/flags that reach physics or forcing and are MISSING from the
list (would be silently dropped on the fesom lane); (2) the smoke loop —
n_steps/snapshot arithmetic, is_first_step handling across steps, anything
that makes 'rc=0' misleading; (3) build_fesom_ocean — z shim consumers
that need more than z_full_ref/n_levels, H_bathy sign, PHC IC wrapped via
from_fesom vs the mesh the model rescaled (constants='legoesm' rescales
coriolis — does the STATE see the same mesh object?); (4) the return 0
early-exit — any cleanup/manifest writing the other lanes do that fesom
now skips silently. Rank CRITICAL/MAJOR/MINOR, one line each + file:line.
No summary." > "$LOG" 2>&1
rc=$?
tail -60 "$LOG"
echo "=== codex rc=$rc ==="
echo DONE


