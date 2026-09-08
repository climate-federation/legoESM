#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=qvbias_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/qvbias_cdx_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"; break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no usable CODEX_HOME"; exit 90; }
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version || exit 91
cd $WT || exit 92
HEAD_SHA=$(git rev-parse HEAD); echo "PINNED sha: $HEAD_SHA"
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL: $HEAD_SHA != $PIN"; exit 93; }
git diff --quiet && git diff --cached --quiet || {
  echo "PREFLIGHT FAIL: DIRTY TREE at $HEAD_SHA — uncommitted edits would run"
  git status --porcelain | head -20
  exit 94
}

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
FIND THE ROOT CAUSE of a boundary-layer humidity bias in a single-column model.
Not a style review: locate the mechanism in the code, with file:line.

MEASURED, 10 convection schemes, identical column, 100-day RCE equilibria at
SST 300 K vs a SAM CRM reference (bias = SCM - CRM):
* surface q_v bias +5.1 g/kg MEAN over all ten schemes, IDENTICAL before and
  after parameter tuning -> the cause is in the SHARED path, not convection.
* bias decays upward: +5.1 sfc, +3.6 at 0.5 km, +2.8 at 1 km, +1.8 at 2 km.
* BIMODAL temperature: the two adjustment schemes (dca, sbm) are ~1 K COLD at
  the surface with SST-T_air ~ 4 K and near-correct evaporation (2.8-3.1 vs
  CRM 2.73 mm/day); the seven mass-flux/EDMF schemes are +2.0..+3.3 K WARM with
  SST-T_air collapsed to 0.3-1.5 K (CRM: 3.06) and E starved at 0.7-1.6.
* near-surface RH 0.81-0.997 everywhere vs CRM 0.752.
* prescribed surface wind 5 m/s; the SST anchor is OFF in these runs.
* ADJUSTMENT schemes relax the whole column INCLUDING the lowest levels; the
  mass-flux schemes act above cloud base and leave sub-cloud ventilation to the
  turbulence scheme + surface fluxes. That split is the strongest clue.

THE SHARED PATH TO REVIEW (the SCM loop is scripts/run/run_scm_rce_campaign.py,
turbulence=louis, bulk fluxes):
  packages/atmosphere/legoesm/atmosphere/physics/turbulence/louis.py
  packages/atmosphere/legoesm/atmosphere/physics/turbulence/surface_layer.py
  packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py
  packages/core/legoesm/core/bulk_flux.py  (and whatever compute_most_fluxes it calls)
  packages/atmosphere/legoesm/atmosphere/physics/turbulence/config.py (gustiness_w_zi, bulk scheme selection)
  scripts/run/run_scm_rce_campaign.py (how the surface wind, SST, and fluxes enter the column; DEFAULT_SCM_RCE_SURFACE_WIND_M_S; what bulk scheme the campaign resolves)

QUESTIONS, each with file:line:
A. Which bulk scheme does the SCM campaign actually resolve (coare3? louis-native?),
   and does the COARE convective-gustiness enhancement (gustiness_w_zi / w* term)
   actually ACTIVATE in this configuration, or is it inert because of a default,
   a None, or a scheme mismatch? A 5 m/s fixed wind with no gustiness is
   candidate cause (b).
B. In the Louis surface-layer/stability functions: as the surface layer becomes
   near-neutral or weakly stable (SST-T_air -> 0.3-1 K, warm-biased air), do the
   exchange coefficients collapse faster than COARE/observations say they
   should? Look for the stability-function form and any floor on C_H/C_E.
C. Trace the full path of the surface LATENT flux into the lowest model level:
   is there any factor (area, density, dt, implicitness, tri-diagonal coupling)
   that under-applies it, or any asymmetry between how heat and moisture are
   deposited? The adjustment schemes get E right in the SAME path, which argues
   the flux law responds correctly when the state is right - confirm that
   reading.
D. Does the turbulence scheme mix the lowest level with the sub-cloud layer
   strongly enough under a warm, moist, near-neutral surface layer, or does
   K collapse at the first model level (Ri-based cutoffs, length-scale floors)?
E. Is there anything that DRIES or VENTILATES the sub-cloud layer that ships
   disabled in this configuration (downdraft terms, cold-pool gustiness,
   entrainment-drying), i.e. candidate cause (a)?
F. Name the SINGLE cheapest one-variable experiment that discriminates your top
   two candidates, runnable as one 100-day SCM run.

Numbered findings with file:line and one-line fixes. Say explicitly where a
suspicion is NOT supported by the code. Do not pad.
PROMPT
)" 2>&1
CODEX_RC=$?
echo "CODEX_RC=$CODEX_RC"
[ $CODEX_RC -eq 0 ] || echo "REVIEW DID NOT RUN"
echo ALLDONE
