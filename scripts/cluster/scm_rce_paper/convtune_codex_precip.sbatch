#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=convtune_cdxp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/convtune_cdxp_%j.out
set +e
# ---------------------------------------------------------------------------
# ONE question: why does the SCM-RCE column precipitate ~0 while the SAME
# physics package precipitates normally in the global model?  The user's prior
# is that this is a basic setup/plumbing mistake in the SCM harness, not a
# physics failure, and the global model running fine is the evidence for it.
# ---------------------------------------------------------------------------
WT=${WT:?export WT=<pinned worktree>}
PIN=${PIN:?export PIN=<sha> the worktree must be at}
export CODEX_HOME=/tmp/codex_home_$SLURM_JOB_ID
mkdir -p $CODEX_HOME
cp /burg-archive/home/pg2328/.codex/auth.json $CODEX_HOME/ 2>/dev/null || {
  echo "FATAL: could not stage codex auth"; exit 90; }
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version || exit 91

cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD)
echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
DIAGNOSIS, not a code review. ONE question, answered from the code:

  The SCM RCE campaign (scripts/run/run_scm_rce_campaign.py) reports surface
  precipitation of 1e-18 to 3e-5 mm/day for EVERY convection scheme, against a
  CRM reference of 2.395 mm/day, while the column is in steady state
  (drift_qv 1e-7..4e-5). The SAME physics package precipitates normally in the
  global model. So this is very likely a harness/plumbing defect in the SCM,
  not a physics failure. WHERE IS IT?

MEASURED FACTS (do not re-derive; use them):
  scheme      tuned score  precip [mm/day]   condensate RMSE  drift_qv
  sbm         1.698        2.98e-05          3.096            3.8e-06
  mass_flux   2.385        2.39e-18          4.664            3.8e-05
  kuo         4.936        6.27e-18          9.828            1.9e-05
  dca         9.459        4.16e-18         18.897            1.9e-07
  Every scheme scores precip_rmse = 0.798 = 2.395/3 exactly.
  sbm's tuned column: max condensate 1.454e-04 kg/kg, surface q_v 1.974e-02
  kg/kg (initial q0 was 1.865e-02), so the column is moist and not empty.
  Config: 100 days, dt 600 s, RRTMGP, turbulence louis, microphysics morrison
  (SAM flavour), SST prescribed at 300 K, surface wind 5 m/s, no large-scale
  forcing, convection sub-stepped 10x, microphysics sub-stepped 30x.

READ AT LEAST THESE, and say which line is responsible:
  * run_scm_rce_campaign.py: the stepping body, apply_split_microphysics_step,
    apply_convection_substeps, _make_microphysics_precip_diagnostic,
    _make_convective_precip_diagnostic,
    _convective_precip_rate_from_tendency, and the
    `convective_precip_for_score` expression that zeroes the convective
    contribution when convection is sub-stepped and microphysics != "none".
  * packages/atmosphere/legoesm/atmosphere/physics/convection/output.py, which
    states that convection owns NO surface-precipitation diagnostic and that
    microphysics owns it.
  * how the GLOBAL model accumulates surface precipitation for the same
    schemes (driver/physics_pipeline.py, driver/model_driver.py), since that
    path demonstrably works — and say precisely what the SCM does differently.
  * whether the SCM applies a surface MOISTURE flux at all: the forcing is
    SCMForcing(prescribe="T_s", ...), turbulence is louis with
    SurfaceConfig(bulk_scheme="constant", Ch_neutral=1.5e-3); trace whether an
    evaporation tendency reaches q_v, and whether the tracer set even contains
    the slots morrison needs to sediment (q_r/q_s/q_g), including
    _preseed_column_tracers and min_tracer_slots.

ANSWER IN THIS ORDER:
  1. Is the column CLOSED (no surface moisture flux) or LEAKY (evaporation
     live, water removed without being counted)? Give the file:line evidence
     that decides it, not a plausibility argument.
  2. Name the single smallest change that makes the SCM's reported
     precipitation the physically correct one, and say what it would read
     for sbm and for mass_flux if applied.
  3. Say whether `_convective_precip_rate_from_tendency` — which integrates
     max(dq_c/dt, 0), i.e. condensate PRODUCTION — is a correct surface
     precipitation proxy for ANY of the ten schemes, and for which it is not.
  4. State what is WRONG with my current reading, if anything. I believe the
     zeroing of convective precip plus a condensate-production proxy are the
     two defects; tell me if the real defect is elsewhere.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"
echo ALLDONE
