#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=blqv_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/blqv_cdx_%j.out
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
[ "$HEAD_SHA" = "$PIN" ] || { echo "PREFLIGHT FAIL"; exit 93; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
ADVERSARIAL REVIEW of a measurement chain and its conclusion. Your job is to
BREAK it: find any confound, code defect, or logical gap that would invalidate
the verdict. If it survives, say so plainly.

CLAIM UNDER REVIEW: In our single-column model (SCM) at RCEMIP1 RCE
(SST 300 K fixed, 100-day runs vs a SAM CRM reference), the +5-6 g/kg
near-surface humidity excess common to all ten convection schemes is NOT
caused by boundary conditions (SST, wind, flux law), NOT by the turbulence
closure choice, and NOT curable by any existing single knob; the missing
physics is cold-pool/downdraft delivery of cool dry air to the surface layer.

EVIDENCE CHAIN (all from scripts/validate/scm_rce_blqv_discriminator.py,
one-variable 100-day arms vs a control reproducing the campaign bias; results
in results/scm_rce_tqtune/blqv_discriminator_*.json):
1. control (louis, bulk constant, wind 5 m/s): dqv_sfc +6.20, SST-Ta 0.63 K
   (CRM: 3.06), RH 0.933 (CRM 0.752), E 1.25 mm/day (CRM 2.73).
2. cloudtop_entrainment 0->0.2: +6.19 - no effect.
3. bulk constant->coare3 (+w* gustiness): +6.03 - no effect.
4. turbulence louis->holtslag_boville (nonlocal K-profile): +6.27 - no effect.
5. emanuel +unsaturated downdraft: moistens 0.5-2 km, surface unchanged.
6. wind 2 m/s: dqv +4.47, SST-Ta 1.54, E 1.02. wind 8 m/s: dqv +6.73,
   SST-Ta 0.37, E 1.38. Monotonic: more wind = wetter surface.
7. Adjustment schemes (dca, sbm) through the SAME BC get E 2.8-3.1 correct.
INTERPRETATION: the wind/flux knob moves E and surface qv in the SAME
direction; the CRM needs E UP and qv DOWN simultaneously; only a process
removing moisture/heat from the surface layer independent of the surface flux
can decouple them; the only class of scheme that does it (whole-column
adjustment) is also the only class with the right surface state.

ATTACK THESE SPECIFIC POINTS, each with file:line evidence:
A. Read scripts/validate/scm_rce_blqv_discriminator.py and the functions it
   calls in scripts/run/run_scm_rce_campaign.py (run_cached, run_scm_rce,
   _config_cache_key, build_forcing / the SCMForcing construction near line
   1130, and the SingleColumnModel.create call near 1490). Is each arm truly
   ONE variable? In particular: surface_wind_m_s sets BOTH the bulk-flux wind
   AND the initial/forced wind profile (line ~1136 wind_profile = full(...)).
   Does changing it alter anything else (shear seen by Louis, momentum
   forcing, gustiness floors, drag)? Uniform profile => zero shear at any
   value - confirm nothing else leaks.
B. Cache correctness: could any arm have returned another arm's cached
   result? _config_cache_key must separate configs AND wind. Check the hb arm
   (different turbulence scheme) and downdraft arm (config _replace) hash
   differently.
C. The window: diagnostics average the last analysis_days=5 of 100. Is any
   arm plausibly NOT equilibrated, making its number a transient? What in
   RunDiagnostics would reveal it (drift fields), and are those recorded?
D. The E comparison: is evap_mm_day in the SCM computed on the same
   definition as the CRM's 2.73 (bulk latent flux vs water-budget residual)?
   Trace where evap_mm_day comes from (run_scm_rce_campaign.py ~1657-1692).
E. The 'adjustment schemes get E right' premise: dca/sbm reach E 2.8-3.1
   while cooling the surface air ~4 K below SST. Could their E be right for a
   WRONG reason (excess SST-Ta gap overcompensating the flux law), making
   'flux law proven' an overstatement?
F. Missing candidates: name any code path we have NOT tested that could
   decouple E and surface qv (radiative cooling of the lowest level,
   large-scale subsidence forcing choice, microphysics rain evaporation at
   the lowest level, saturation adjustment) - and say whether each is ON in
   this configuration and where.
G. The RCEMIP protocol point: our fixed 5 m/s flux wind - is SAM's RCE_small
   surface flux actually gust-driven with no mean wind, and is 5 m/s a
   defensible surrogate magnitude, or systematically high/low vs SAM's
   effective flux wind?

VERDICT format: numbered findings with file:line; each labelled
BREAKS-THE-CLAIM / WEAKENS / SURVIVES. End with one line: does the cold-pool
conclusion stand?
PROMPT
)" 2>&1
echo "CODEX_RC=$?"
echo ALLDONE
