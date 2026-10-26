#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtune_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_cdx_%j.out
set +e
# ---------------------------------------------------------------------------
# Adversarial review of the merged convection-tuning branch: the three-way
# merge resolution (RCEMIP protocol IC vs gSAM-sounding calibration), the new
# saturation-treatment flags, the JIT-cache fix, and the cross-lane IFS-cap
# probe.  Read-only sandbox; the reviewer may run git and read files.
#
# WT is exported by the launcher so the review always runs on a pinned tree
# that nothing is editing.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
# Codex needs a NODE-LOCAL CODEX_HOME: a shared-FS home has failed before, and
# a bad CODEX_HOME exits 0 having reviewed NOTHING.
# CODEX_HOME must be node-local AND must still exist when codex reads it.
# On g059 (job 9361561) the /tmp dir vanished between `cp` and codex start —
# "Error finding codex home: ... that path does not exist" — so the location is
# chosen by TESTING it, and codex's own exit status is checked rather than
# assumed (a bad CODEX_HOME has historically exited 0 having reviewed nothing).
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"
  break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no usable CODEX_HOME"; exit 90; }
echo "CODEX_HOME=$CODEX_HOME (auth staged: $(ls -1 $CODEX_HOME/auth.json 2>/dev/null | wc -l))"
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; echo "CODEX_HOME=$CODEX_HOME"; codex --version || exit 91

cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
FOCUSED ADVERSARIAL REVIEW. Answer the numbered questions and stop. Do not
read the whole repository. Report only defects that would make a PUBLISHED
RESULT WRONG or a run silently do something other than what its command line
says. For each: file:line, what breaks, and the smallest fix.

The change under review is the last four commits on this branch:
  git log --oneline -5
  git show --stat HEAD~3   # the three-way merge
  git diff HEAD~4..HEAD -- scripts/ packages/ tests/

CONTEXT. A campaign tunes every convection scheme's parameters inside a
single-column RCE model and ranks the schemes by profile RMSE against an
RCEMIP CRM reference, before and after tuning. This branch merges three lines
of work so the campaign can run with the IFS/SAM supersaturation treatment
active: an external RCEMIP SAM_CRM reference, a uniform in-scheme liquid
saturation guard, and the IFS homogeneous-freezing ICE supersaturation
allowance (thermo.homogeneous_freezing_rh_factor).

Q1. THE IC FORK. The merge had to choose between two calibrations of the same
three RCEMIP1 analytic-IC constants: the published protocol (Wing et al. 2018
Eq. 3, T_v0 = T_sfc*(1+0.608*q0), Gamma = 0.0067 K/m, q0 = 12/18.65/24 g/kg by
SST) and a fit to gSAM's own sounding (T_v0 = 299.274, Gamma = 0.0069901,
q0 = 14.2 g/kg). The protocol won as the module default; the gSAM triple is
kept as GSAM_SND_*. Find anything left INCONSISTENT after that choice:
a comment or docstring that still asserts the gSAM fit is the default, a test
that still pins a gSAM-calibrated number, or any consumer that would now mix
one constant from each set (which is a third, uncalibrated column). Check
packages/atmosphere/legoesm/atmosphere/idealized/rcemip_initial_conditions.py,
scripts/run/run_rcemip_plane.py, scripts/data/build_rcemip1_small_reference.py,
tests/unit/test_rcemip_gsam_oracle_ic.py, tests/unit/test_rcemip_wing_ic.py,
tests/validation/test_rcemip_plane_smoke.py.

Q2. THE UNION MERGE in scripts/run/run_rcemip_plane.py. One side added
--sounding (tabulated SAM sounding IC), the other added --sam-grd (oracle
vertical levels) and SST-dependent q0; I merged them into one if/elif/else.
Is any branch now missing a definition it needs (theta_ref_fn, hc, snd,
q_sfc_rce)? Is any combination of --sounding / --sam-grd / --stretched-vertical
silently ignored rather than refused? Did the merge drop any behaviour that
existed on either side?

Q3. THE FLAG COLLISION. Both sides added an argparse flag literally named
--hard-saturation-adjustment for two DIFFERENT interventions (in-scheme
iterated adjustment vs post-step drain). I kept the name for the in-scheme one
and renamed the drain to --hard-sat-drain/--hard-sat-drain-threshold/
--hard-sat-drain-max-heating-k. Find any remaining reference to the old drain
flag names or to args.hard_sat_threshold / args.hard_sat_max_heating anywhere
(code, tests, sbatch, docs, shell wrappers).

Q4. THE GUARD THREADING in run_scm_rce_campaign.make_physics_config. Does
micro_cfg._replace(**{microphysics: apply_microphysics_experiment_flags(...)})
do the right thing for every scheme name the campaign can select? Does the
"silently inert override" refusal fire in exactly the right cases? Is there any
path where the returned PhysicsConfig has scheme=X but the guard applied to
sub-config Y?

Q5. THE JIT-CACHE FIX. run_cached now calls jax.clear_caches() before each new
evaluation, to stop XLA:CPU aborting with "LLVM ERROR: Unable to allocate
section memory" after ~40 tuning evaluations. Is that diagnosis plausible given
the code, or is there a likelier cause? Can clear_caches() break anything in
this driver (a cached executable something else still relies on, an interaction
with the persistent compilation cache, a silent slowdown that is worse than the
problem)?

Q6. THE PROBE, scripts/validate/check_ifs_supersaturation_cap.py, and its test
tests/unit/test_check_ifs_supersaturation_cap.py. Can the probe PASS while the
IFS cap is in fact broken or unreachable in one of the three lanes? Is the
ON-vs-OFF sign expectation right (with the ice allowance ON the scheme should
retain MORE vapour, i.e. a LESS negative dq_v/dt)? Is the synthetic column
physically in the regime the ramp governs, and is section 2 genuinely
non-vacuous? Does the checker's own test prove it can fail?

Q7. THE EXPERIMENT DESIGN. The campaign holds microphysics, radiation,
timestep, substep counts and the transport kernel fixed across all 10
convection schemes, and tunes only each scheme's own tunable_tier<=extended
parameters against the CRM profiles, reporting RMSE before and after. Name the
CONFOUNDS that would still make the resulting ranking misleading, ordered by
how much they could move it. Be specific about which held-fixed choice hurts
which scheme.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"
echo ALLDONE
