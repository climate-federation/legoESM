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
  "$PY" -m pytest -q --no-header -x --tb=short \
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
  "$PY" -m pytest -q \
      tests/unit/test_rcemip_gsam_oracle_ic.py \
      tests/unit/test_rcemip_wing_ic.py \
      tests/unit/test_compute_reference_state_p_sfc_iter95.py
  rc=$?; echo "PYTEST(with-oracle) rc=$rc"; [ $rc -ne 0 ] && rc_all=1

  echo "=== PYTEST oracle gate, oracle path ABSENT (skip-cleanly check) ==="
  env -u LEGOESM_GSAM_ROOT "$PY" -m pytest -q \
      tests/unit/test_rcemip_gsam_oracle_ic.py
  rc=$?; echo "PYTEST(no-oracle) rc=$rc"; [ $rc -ne 0 ] && rc_all=1
fi

if [ "$MODE" = "smoke" ] || [ "$MODE" = "all" ]; then
  echo "=== SMOKE: run_rcemip_plane.py --sounding (2 steps) ==="
  "$PY" scripts/run/run_rcemip_plane.py \
      --nx 8 --ny 8 --nlev 40 --dx 4000.0 --dt 2.0 --steps 2 \
      --H 20000.0 --stretched-vertical --sounding "$SND" \
      --radiation gray --microphysics kessler \
      --output "$REPO/results/gsam_rcemip/smoke_sounding_${SLURM_JOB_ID}"
  rc=$?; echo "SMOKE(sounding) rc=$rc"; [ $rc -ne 0 ] && rc_all=1

  echo "=== SMOKE: analytic default unchanged (2 steps) ==="
  "$PY" scripts/run/run_rcemip_plane.py \
      --nx 8 --ny 8 --nlev 40 --dx 4000.0 --dt 2.0 --steps 2 \
      --H 20000.0 --stretched-vertical \
      --radiation gray --microphysics kessler \
      --output "$REPO/results/gsam_rcemip/smoke_analytic_${SLURM_JOB_ID}"
  rc=$?; echo "SMOKE(analytic) rc=$rc"; [ $rc -ne 0 ] && rc_all=1

  echo "=== NEGATIVE: --sounding without --stretched-vertical must FAIL ==="
  "$PY" scripts/run/run_rcemip_plane.py \
      --nx 8 --ny 8 --nlev 40 --dx 4000.0 --dt 2.0 --steps 1 \
      --H 20000.0 --sounding "$SND" \
      --output "$REPO/results/gsam_rcemip/smoke_neg_${SLURM_JOB_ID}"
  rc=$?
  if [ $rc -eq 0 ]; then echo "NEGATIVE rc=0 -> GATE FAILED (should have exited)"; rc_all=1
  else echo "NEGATIVE rc=$rc (expected non-zero) OK"; fi
fi

echo "=== GATE DONE rc_all=$rc_all ==="
exit $rc_all
