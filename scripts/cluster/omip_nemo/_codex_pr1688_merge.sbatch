#!/bin/bash
#SBATCH --job-name=cdx_1688
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/cdx_1688_%j.out
set -u
unset CODEX_HOME
WT=/burg-archive/glab/users/pg2328/legoESM
LOG=$WT/omip_nemo/cdx_1688_body_${SLURM_JOB_ID}.txt
codex exec --sandbox read-only -C $WT \
"Review merge-conflict resolution commit 30dc0fa08 (git show 30dc0fa08; it merges
cf/main into the fix/twin-fp64-snapshots PR branch, PR #1688). The ONLY content
conflict was in tests/ocean/unit/test_dino_1226_instruments.py: this PR asserted
run_twin source contains '_restart_elapsed_seconds(' (underscore) while main
renamed it to 'restart_elapsed_seconds(' (no underscore). I took main's naming and
kept the PR's ~360 lines of fp64-snapshot tests (union). VERIFY, numbered:
1. The merged scripts/validate/ocean_fidelity/dino_1226/kamm_twin_90d.py actually
   DEFINES restart_elapsed_seconds (no underscore) and has ZERO _restart_elapsed_
   seconds — so the kept assertion is TRUE, not vacuous. Confirm.
2. NONE of this PR's fp64 tests were dropped in the resolution (count test_ defs in
   the merged test file vs the PR head 811241885 pre-merge; they should all survive).
3. The fp64 API the tests exercise (snapshot_dtype, capture_snapshot,
   LEGACY_STORAGE_DTYPES, snapshot_storage_dtypes, fp64_3d, _parse_args --fp64-3d)
   exists in the merged kamm_twin_90d.py so the tests are non-vacuous.
4. The merge did not silently drop any main change to these 6 PR files or leave any
   conflict marker anywhere (grep the tree).
5. Any OTHER semantic issue from combining main + this PR (import errors, a test
   that now references a symbol that moved, a duplicated def).
GO/NO-GO on merging PR #1688 to main." > "$LOG" 2>&1
echo "rc=$?"; grep -qiE "Failed to refresh token|refresh token was already used" "$LOG" && echo AUTHFAIL
cat "$LOG"; echo DONE
