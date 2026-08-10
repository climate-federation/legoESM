#!/bin/bash
#SBATCH --job-name=rce_oracle
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/gsam_rcemip/gate_%j.log
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# RCEMIP gSAM-oracle gate, from a PINNED worktree of the pushed branch.
#
#   MODE=extract  regenerate the tracked tiny baseline from the gSAM sounding
#                 (writes into the WORKTREE; copy back by hand after review)
#   MODE=test     run the oracle unit tests (default)
#   MODE=smoke    2-step run of run_rcemip_plane.py --sounding (wiring check)
#   MODE=all      test + smoke
#
# JAX_ENABLE_X64=1 throughout: this is a numerics/conservation-class check.
set -uo pipefail

REPO=/burg-archive/glab/users/pg2328/legoESM
BRANCH="${BRANCH:-feat/rcemip-gsam-oracle-sounding}"
MODE="${MODE:-test}"
WT=/burg-archive/glab/users/pg2328/legoESM_wt/rce_oracle_${SLURM_JOB_ID}
GSAM_ROOT=/burg/glab/users/pg2328/gsam_oracle/gSAM1.8.8
SND="$GSAM_ROOT/CASES/RCEMIP1/snd_rcemip_300s6.11.2"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

mkdir -p "$(dirname "$WT")" "$REPO/results/gsam_rcemip"
git -C "$REPO" worktree add --detach "$WT" "origin/$BRANCH" >/dev/null 2>&1
echo "PINNED sha $(git -C "$WT" rev-parse HEAD)  branch=$BRANCH  MODE=$MODE"
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true' EXIT

cd "$WT"
_pp="$WT"
for _d in "$WT"/packages/*/; do
  [ -d "${_d}legoesm" ] && _pp="${_pp}:${_d}"
done
export PYTHONPATH="${_pp}:${WT}/src:${PYTHONPATH:-}"
export JAX_ENABLE_X64=1
export JAX_PLATFORMS=cpu
export LEGOESM_GSAM_ROOT="$GSAM_ROOT"
rc_all=0

if [ "$MODE" = "extract" ] || [ "$MODE" = "bootstrap" ]; then
  echo "=== EXTRACT tiny baseline ==="
  "$PY" scripts/data/extract_gsam_rcemip_baseline.py \
      --snd "$SND" --out "$REPO/results/gsam_rcemip/gsam_rcemip300_snd.json"
  echo "EXTRACT rc=$?"
  [ "$MODE" = "extract" ] && exit 0
fi

if [ "$MODE" = "bootstrap" ]; then
  # Surface EVERY anchor the RCE300 recalibration moves, with the measured
  # value, so the anchors are updated from output rather than from arithmetic.
  echo "=== BLAST RADIUS: tests that pin the old RCE300 calibration ==="
  "$PY" -m pytest -q --no-header --tb=short \
      tests/unit/test_rcemip_wing_ic.py \
      tests/unit/test_compute_reference_state_p_sfc_iter95.py \
      tests/validation/test_rcemip_plane_smoke.py \
      tests/atmosphere/hydrostatic/test_build_rcemip1_small_reference.py \
      2>&1 | tail -80
  echo "BLAST rc=${PIPESTATUS[0]}"
  echo "=== MEASURED analytic anchors under the new constants ==="
  "$PY" - <<'EOF'
import jax.numpy as jnp, numpy as np
from legoesm.atmosphere.idealized import rcemip_initial_conditions as ic
z = jnp.array([0.0, 4000.0, 5000.0, 15000.0, 20000.0])
th = np.asarray(ic.wing2018_theta_profile(z))
T = np.asarray(ic.wing2018_temperature_profile(z))
print("WING_T_V0", ic.WING_T_V0, "WING_GAMMA", ic.WING_GAMMA,
      "WING_Q_SFC_DEFAULT", ic.WING_Q_SFC_DEFAULT)
print("theta at 0/4k/5k/15k/20k:", [round(float(v), 4) for v in th])
print("T     at 0/4k/5k/15k/20k:", [round(float(v), 4) for v in T])
print("cold point T_v(z_t) =", ic.WING_T_V0 - ic.WING_GAMMA * ic.WING_Z_T)
print("lapse 0-5km K/km =", (float(T[0]) - float(T[2])) / 5.0)
EOF
  exit 0
fi

if [ "$MODE" = "test" ] || [ "$MODE" = "all" ]; then
  echo "=== PYTEST oracle gate (LEGOESM_GSAM_ROOT set => full-column checks ON) ==="
  "$PY" -m pytest -q --tb=short \
      tests/unit/test_rcemip_gsam_oracle_ic.py \
      tests/unit/test_rcemip_wing_ic.py \
      tests/unit/test_compute_reference_state_p_sfc_iter95.py \
      tests/validation/test_rcemip_plane_smoke.py \
      tests/atmosphere/hydrostatic/test_build_rcemip1_small_reference.py \
      tests/atmosphere/nonhydrostatic/integration/test_plane_crm_end_to_end_smoke.py
  rc=$?; echo "PYTEST(with-oracle) rc=$rc"; [ $rc -ne 0 ] && rc_all=1

  # CONTROL: the plane-CRM end-to-end smoke writes into /local pytest tmpdirs
  # and failed on 9331622 with FileNotFoundError on progress.txt — a scratch
  # issue, not a physics one.  "A new failure is not yours until reproduced
  # with your change stashed": run the SAME file from origin/main, same node,
  # same env, and print both verdicts side by side.
  echo "=== CONTROL: same CRM e2e file on origin/main (unchanged tree) ==="
  MAINWT=/burg-archive/glab/users/pg2328/legoESM_wt/rce_main_${SLURM_JOB_ID}
  git -C "$REPO" worktree add --detach "$MAINWT" origin/main >/dev/null 2>&1
  echo "CONTROL sha $(git -C "$MAINWT" rev-parse HEAD)"
  ( cd "$MAINWT"
    _mp="$MAINWT"; for _d in "$MAINWT"/packages/*/; do
      [ -d "${_d}legoesm" ] && _mp="${_mp}:${_d}"; done
    PYTHONPATH="${_mp}:${MAINWT}/src" "$PY" -m pytest -q --no-header --tb=line \
        tests/atmosphere/nonhydrostatic/integration/test_plane_crm_end_to_end_smoke.py \
        2>&1 | tail -12 )
  echo "CONTROL(main) rc=$?"
  git -C "$REPO" worktree remove --force "$MAINWT" >/dev/null 2>&1 || true

  echo "=== PYTEST oracle gate, oracle path ABSENT (skip-cleanly check) ==="
  env -u LEGOESM_GSAM_ROOT "$PY" -m pytest -q \
      tests/unit/test_rcemip_gsam_oracle_ic.py
  rc=$?; echo "PYTEST(no-oracle) rc=$rc"; [ $rc -ne 0 ] && rc_all=1
fi

if [ "$MODE" = "table" ] || [ "$MODE" = "all" ]; then
  echo "=== ACCEPTANCE TABLE: --sounding IC and the analytic fallback vs gSAM ==="
  "$PY" - "$SND" <<'EOF'
import sys
import numpy as np
from legoesm import constants
from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_snd
from legoesm.atmosphere.idealized import rcemip_initial_conditions as ic
from legoesm.thermo import saturation_mixing_ratio
sys.path.insert(0, "tests/unit")
from test_rcemip_gsam_oracle_ic import _analytic_at, _build_ic_from, _rh_from

snd = read_sam_snd(sys.argv[1])
z_o = np.asarray(snd.z); q_o = np.asarray(snd.q_v)
T_o, rh_o = _rh_from(np.asarray(snd.theta), np.asarray(snd.p), q_o)
z_m, T_m, q_m, rh_m = _build_ic_from(snd)
k = int(np.argmin(z_m))
Ta, qa, _, rha = _analytic_at(z_o, T_v0=ic.WING_T_V0, Gamma=ic.WING_GAMMA,
                              q_sfc=ic.WING_Q_SFC_DEFAULT)
Tp, qp, _, rhp = _analytic_at(z_o, T_v0=295.0, Gamma=0.0067, q_sfc=0.01865)

print(f"model lowest level z = {z_m[k]:.1f} m; oracle lowest level z = "
      f"{z_o[0]:.1f} m (oracle interpolated to the model height)")
rows = [
 ("surface T [K]", np.interp(z_m[k], z_o, T_o), T_m[k], Ta[0], Tp[0]),
 ("surface q_v [g/kg]", np.interp(z_m[k], z_o, q_o)*1e3, q_m[k]*1e3,
  qa[0]*1e3, qp[0]*1e3),
 ("surface RH", np.interp(z_m[k], z_o, rh_o), rh_m[k], rha[0], rhp[0]),
 ("max RH over column", rh_o.max(), rh_m.max(), rha.max(), rhp.max()),
 ("cold point T [K]", T_o.min(), T_m.min(),
  ic.WING_T_V0 - ic.WING_GAMMA*ic.WING_Z_T, 295.0 - 0.0067*ic.WING_Z_T),
]
print(f"{'quantity':22s} {'gSAM oracle':>12s} {'--sounding':>12s} "
      f"{'analytic new':>13s} {'analytic OLD':>13s}")
for n, o, m, a, pre in rows:
    print(f"{n:22s} {o:12.4f} {m:12.4f} {a:13.4f} {pre:13.4f}")
print(f"\nconstants: T_v0={ic.WING_T_V0} Gamma={ic.WING_GAMMA} "
      f"q_sfc={ic.WING_Q_SFC_DEFAULT}")
EOF
  echo "TABLE rc=$?"
  [ "$MODE" = "table" ] && exit 0
fi

if [ "$MODE" = "smoke" ] || [ "$MODE" = "all" ]; then
  # nlev=64 keeps the stretch ratio under the dz_top/dz_sfc>30 warning;
  # --semi-implicit is required by the driver whenever
  # substep_horizontal_acoustic is on (its default) — omitting it made BOTH
  # arms exit 1 on 9331622, which is why the analytic arm is run as a control
  # rather than assumed unaffected.
  COMMON="--nx 8 --ny 8 --nlev 64 --dx 4000.0 --dt 2.0 --steps 2 --H 20000.0
          --stretched-vertical --semi-implicit --radiation gray
          --microphysics kessler"

  echo "=== SMOKE: run_rcemip_plane.py --sounding (2 steps) ==="
  "$PY" scripts/run/run_rcemip_plane.py $COMMON --sounding "$SND" \
      --output "$REPO/results/gsam_rcemip/smoke_sounding_${SLURM_JOB_ID}"
  rc=$?; echo "SMOKE(sounding) rc=$rc"; [ $rc -ne 0 ] && rc_all=1

  echo "=== SMOKE: analytic default unchanged (2 steps) — CONTROL ==="
  "$PY" scripts/run/run_rcemip_plane.py $COMMON \
      --output "$REPO/results/gsam_rcemip/smoke_analytic_${SLURM_JOB_ID}"
  rc=$?; echo "SMOKE(analytic) rc=$rc"; [ $rc -ne 0 ] && rc_all=1

  echo "=== NEGATIVE: --sounding without --stretched-vertical must FAIL ==="
  neg_out=$("$PY" scripts/run/run_rcemip_plane.py \
      --nx 8 --ny 8 --nlev 64 --dx 4000.0 --dt 2.0 --steps 1 --H 20000.0 \
      --semi-implicit --sounding "$SND" \
      --output "$REPO/results/gsam_rcemip/smoke_neg_${SLURM_JOB_ID}" 2>&1)
  rc=$?
  # rc!=0 alone is NOT evidence — any unrelated crash would satisfy it. Require
  # the guard's OWN message (the exit-code-is-not-evidence rule).
  if [ $rc -ne 0 ] && printf '%s' "$neg_out" | grep -q -- "--sounding requires --stretched-vertical"; then
    echo "NEGATIVE rc=$rc with the expected guard message OK"
  else
    echo "NEGATIVE rc=$rc but the --stretched-vertical guard message was ABSENT -> GATE FAILED"
    printf '%s\n' "$neg_out" | tail -5
    rc_all=1
  fi
fi

echo "=== GATE DONE rc_all=$rc_all ==="
exit $rc_all
