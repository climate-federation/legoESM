#!/bin/bash
#SBATCH --account=glab
#SBATCH -p short
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /burg-archive/glab/users/pg2328/legoESM/logs/joint_cdx_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  C="$_CH/codex_home_${SLURM_JOB_ID:-$$}"; mkdir -p "$C" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$C/" 2>/dev/null || continue
  [ -f "$C/auth.json" ] || continue; export CODEX_HOME="$C"; break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no CODEX_HOME"; exit 90; }
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
cd $WT; HEAD_SHA=$(git rev-parse HEAD)
[ "$HEAD_SHA" = "$PIN" ] || { echo PREFLIGHT_FAIL; exit 93; }
git diff --quiet || { echo DIRTY; exit 94; }
codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a multi-SST tuning change to a merged, tested numerics
file. Break it. The CURRENT checked-out HEAD adds joint tuning of convection
schemes across three RCEMIP SSTs (295/300/305 K), and a prior review round's
blockers have ALREADY been fixed in this HEAD -- verify the FIXES, do not
re-report issues that the current code no longer has.

READ THE CURRENT CODE (not any older commit): the multi-SST logic in
scripts/run/run_scm_rce_campaign.py (tune_category_winner, the _joint_eval
closure, _config_cache_key, tune_focused_params) and
scripts/run/run_scm_rce_convection_intercomparison.py (evaluate_scheme, main's
--ssts / joint_refs handling, _SIGNATURE_FIELDS). Quote the CURRENT line
content at each file:line you cite; if a line matches a fixed state, say so.

ATTACK, each with file:line:
1. CACHE KEY. _config_cache_key gained sst_K. Confirm EVERY run_cached call in
   the tuner passes the sst_K that matches the ref it runs against -- a run at
   ref_305 with sst_K=300 (or vice versa) would score the 305 profile against a
   300 K column. Check the _joint_eval closure in tune_category_winner: does it
   pass sst_K=_sst paired with the CORRECT _r? Does the prior run in
   evaluate_scheme pass sst_K=primary_sst_K with ref (the primary ref)?
2. THE JOINT SCORE. _joint_eval averages objective_value over SSTs and returns
   inf unless ALL equilibrate. Confirm: (a) best_score is initialised from the
   joint default_score (not a single-SST objective_value); (b) score_default /
   score_tuned in the records are the joint scores, not objective_value(run);
   (c) the refinement loop (step_fracs) also uses _joint_eval, so refinement
   optimises the joint objective, not 300 K alone.
3. THE PRIMARY RUN. _joint_eval returns `primary` = the FIXED_SST_K run for
   profile reporting. If none of the joint SSTs is FIXED_SST_K (300), it falls
   back to runs[0]. Is that correct, and is the reported profile then at a
   different SST than the albedo/base_cfg was built for?
4. IC + ALBEDO CONSISTENCY. wing_initial_profiles(ref, sst_K) picks q0 from
   WING_Q_SFC_BY_SST and raises on an undefined SST. But base_cfg's albedo is
   built ONCE at primary_sst_K and reused for all three joint runs (comment
   claims negligible). Is the albedo actually SST-dependent in
   make_physics_config, and is holding it fixed defensible, or does it bias the
   295/305 runs?
5. tune_focused_params got an sst_K param but is NOT made joint. Confirm it
   still works single-SST and that the stray sst_K reference introduced earlier
   now resolves.
6. BACK-COMPAT. joint_refs=None must reproduce the exact previous single-SST
   behaviour. Confirm the default path (no --ssts, or one SST) changes no
   numbers: same cache key additions default to FIXED_SST_K, same scoring.
7. Anything that SILENTLY produces a wrong number rather than an error:
   mismatched ref/SST pairing, a cache collision across SSTs, the mean taken
   over a list that dropped a failed SST, or an inf leaking into a reported
   score.

SPECIFICALLY CONFIRM OR REFUTE that these prior blockers are now fixed in the
CURRENT code, quoting the current line:
 - best_score is initialised from default_score (the joint mean), not
   objective_value(default_run).
 - _joint_eval selects the primary run by primary_sst_K and RAISES if absent.
 - main validates each --joint-reference-dirs trailing SST against its paired
   --ssts value.
 - primary_sst_K is threaded so a single non-300 SST tunes at that SST.
 - focused tuning takes sst_K=primary_sst_K on default AND trial runs and
   rejects a multi-SST request.
 - "ssts" is in _SIGNATURE_FIELDS.
Then report any REMAINING issue in the CURRENT code, BLOCKER / SHOULD-FIX /
NIT, file:line, one-line fix. If a prior blocker is genuinely fixed, say
"FIXED" for it rather than re-listing it.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"; echo ALLDONE
