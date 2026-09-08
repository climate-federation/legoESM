#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=cdx_fp32f
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --output=/burg-archive/glab/users/pg2328/fv3_duo_gaps/codex_fp32f_%j.log
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_fv3jax
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID; mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version
cd $WT; echo "PINNED sha: $(git rev-parse HEAD)"; echo "dirty(tracked): $(git status --porcelain --untracked-files=no | wc -l)"
codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of an UNCOMMITTED working-tree diff -- the FOUNDATION
increment for fv3_duo coarse fp32/mixed precision. Run `git diff` (also
`git status`). This is increment 1 of N: it does NOT make fp32 runnable;
it makes the DEFAULT fp64 path byte-identical while relaxing the dtype
gates and adding the config/factory plumbing, and REFUSES fp32 loudly at
construction until the in-phase workspaces/metrics are threaded (a later
increment).

VERIFY, cite file:line:
1. fp64 BYTE-IDENTICAL: the 8 dtype gates (shared require_uniform_float_jax
   in fv3_phase3d_common.py + the 7 private _require_f64_jax copies in
   fv3_duo_stepper/fv3_mapz/fv3_nh_core/fv3_duo_sw_core/fv3_pgrad/
   fv3_tp_core/grids/fv3_duo_halos) changed from "== float64" to
   "uniform float in {f32,f64}". For an ALL-f64 run is behaviour
   identical (every operand f64 -> uniform -> passes exactly where strict
   f64 did)? Any gate whose new body changed a NON-dtype behaviour (e.g.
   dropped the None-skip, changed control flow, reordered)?
2. The back-compat alias `require_f64_jax = require_uniform_float_jax` in
   fv3_phase3d_common: do all ~40 call sites + importers still resolve?
   Is it in __all__? Does the numpy SPEC lane's strict gate
   (fv3_native_nh_core.py `_require_f64`) stay STRICT (it must -- it is the
   fp64 oracle reference, not the runtime)?
3. The gate now catches MIXED dtypes but no longer catches a UNIFORM f32
   run that was INTENDED as fp64 -- is that anti-silent-downcast guarantee
   truly preserved at the model boundary? Read FV3DuoDynamicsModel.__init__:
   does it actually assert/guard the storage dtype, or does the guard only
   refuse non-fp64 storage_dtype (so a stray f32 IC under storage_dtype=
   'float64' would now slip past the relaxed gates)? Is that a REGRESSION
   of the old protection?  [This is the key question -- the old strict
   gate caught an f32 IC; the new uniform gate + fp64-only construction
   must still catch it, or it is a real hole.]
4. FV3DuoConfig.storage_dtype default 'float64'; model raises
   NotImplementedError for anything else. Factory maps precision->dtype
   and passes it. Is the mapping total/correct? Does dropping the factory's
   own `precision != 'fp64'` ValueError leave any OTHER path that assumed
   fp64-only? Is 'precision' allow-listed in _FV3_DUO_ALLOWED_NONDEFAULT
   so precision=fp32 is not rejected by _refuse_fv3_duo_non_default?
5. The 2 test edits (test_fv3_phase3d_common uniformity tests;
   test_fv3_dsw_phase_3d docstring): non-vacuous? Does the dsw non-vacuity
   test still actually prove the 3-D gate fires (a MIXED dict still raises
   under the new gate)?
6. Any comment/docstring now false. Any NEW bug the diff introduces.
Output BLOCKER/MAJOR/MINOR + file:line, or CLEAN. Focus hardest on #1 and #3.
PROMPT
)"
echo "CODEX RC=$?"; echo CODEX_FP32F_DONE
