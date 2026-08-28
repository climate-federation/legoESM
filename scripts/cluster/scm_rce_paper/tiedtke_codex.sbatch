#!/bin/bash
#SBATCH --account=glab
#SBATCH -p burst
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /burg-archive/glab/users/pg2328/legoESM/logs/tiedtke_cdx_%j.out
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
cd $WT; [ "$(git rev-parse HEAD)" = "$PIN" ] || { echo PREFLIGHT_FAIL; exit 93; }
git diff --quiet || { echo DIRTY; exit 94; }
codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
DIAGNOSE + PROPOSE A FIX. The Tiedtke convection scheme drives water vapour
slightly NEGATIVE (min -6.8e-5 kg/kg) at a near-dry upper level in tropical RCE,
which trips a positivity check and fails the run at all SSTs. Default params.
Temperature profile is otherwise sane. Emanuel/EDMF do not hit this.

READ:
- packages/atmosphere/legoesm/atmosphere/physics/convection/tiedtke.py
  (the dq_v_dt assembly: apply_mass_flux_kernel, the evap_rate moistening near
  line 544, and everything that adds to dq_v_dt before it is returned).
- packages/atmosphere/legoesm/atmosphere/physics/convection/_shared.py
  apply_mass_flux_kernel — the environmental drying (compensating subsidence +
  detrainment) that produces the vapour sink.
- How the SCM applies the convection tendency:
  scripts/run/run_scm_rce_campaign.py apply_convection_substeps (~line 1596)
  and whether ANY moisture-positivity fixer runs after it.

ANSWER, each with file:line:
1. EXACT SOURCE of the negative: which term in dq_v_dt can drive
   q_v + dq_v_dt*dt < 0 at a level where q_v ~ 1e-4? Is it the mass-flux kernel
   drying, the detrainment, or the evap? Show the line.
2. Does tiedtke (or apply_mass_flux_kernel) have ANY per-level limiter capping
   the vapour SINK at the available q_v? Compare with emanuel.py and edmf --
   what do THEY do that prevents this (a taper as q_v->0, a
   max(dq_v_dt, -q_v/dt) clip, or sub-stepping)? Quote it.
3. Is there a CONSERVATION coupling: if the vapour sink is clipped, must the
   paired dq_c / precip source and the latent heating dT_dt be reduced by the
   same amount to keep column water AND MSE closed? Point to where dT_dt and
   dq_c are computed relative to dq_v_dt.
4. Propose the MINIMAL, differentiable, conserving fix and say exactly where it
   goes (which function, which line). Prefer a smooth taper of the drying as
   q_v -> 0 over a hard post-hoc clip if the code structure allows it; if a
   clip, show how to keep dT_dt/dq_c consistent. It must be a static-shape,
   AD-safe, jnp op (no Python branching on traced values).
5. Confirm the fix would NOT change emanuel/edmf/other schemes (tiedtke-scoped),
   and name the smallest test that proves q_v stays >= 0 for default tiedtke.

Be concrete with line numbers. Do not propose disabling the realism check.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"; echo ALLDONE
