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
ADVERSARIAL REVIEW of a physics fix at the CURRENT HEAD. Verify the fix, break
it if you can. The Tiedtke convection scheme drove q_v negative (-6.8e-5 kg/kg)
at the near-dry cold point in SCM-RCE, failing all SSTs; the fix caps the
convective vapour sink.

READ the CURRENT code (quote current lines):
packages/atmosphere/legoesm/atmosphere/physics/convection/tiedtke.py -- the
"Vapour positivity" block (~line 555): a per-column scale
col_scale = min_over_levels(clip(q_v,0)/max(-dt*dq_v_dt,1e-20)) clipped to
[0,1], applied to dT_dt, dq_v_dt, dq_c_conv_dt before the cloud/rain split.

CONFIRM OR REFUTE, each with file:line:
1. POSITIVITY: does q_v + dt*dq_v_dt >= 0 hold at EVERY level after the scale,
   for any dt? Is the min-over-levels the correct reduction (the tightest
   level binds the whole column)?
2. CONSERVATION: scaling dT/dq_v/dq_c by ONE factor -- does it preserve the
   column water budget and the MSE/latent relation the scheme had before? Does
   the cloud/rain split downstream inherit the scaling correctly so dq_c_pos
   and dq_r_conv_dt are consistent?
3. OVER-DAMPING: col_scale can be ~0.6 when one level overshoots, cutting the
   WHOLE column's convection 40% that substep. Measured equilibrium scores are
   sane (295/300/305 K -> 2.40/4.71/5.96, was BLOWUP), but is there a regime
   where the column-min scaling spuriously kills convection in a healthy column
   because ONE upper level is transiently near-dry? Is a per-level scale safer,
   and if so does it break the flux telescoping as claimed?
4. AD/JIT: is the block static-shape, NaN-safe at dq_v_dt=0 (the 1e-20 floor),
   and does it retrace? The 3 committed tests
   (tests/atmosphere/unit/test_tiedtke_vapour_positivity.py) cover positivity,
   no-op-when-stable, and gradient finiteness -- do they actually FAIL against
   the pre-fix code, and is any assertion vacuous?
5. SCOPE: is this genuinely tiedtke-only? mass_flux uses the SAME
   apply_mass_flux_kernel subsidence term and ALSO blows up at 295 K -- confirm
   this fix does NOT touch mass_flux/emanuel/edmf, and note whether mass_flux
   needs the same fix (a one-line follow-up finding, not this task).

Numbered findings, CONFIRMED / BLOCKER / SHOULD-FIX / NIT, file:line. Say where
you checked and found nothing wrong.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"; echo ALLDONE
