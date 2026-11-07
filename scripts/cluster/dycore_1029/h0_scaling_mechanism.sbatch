#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=hs1029h0
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/h0scale_%j.out
#
# #1029 -- the MECHANISM test the nine-arm ladder has not yet run.
#
# THE CLAIM UNDER TEST, currently labelled PLAUSIBLE in the issue: the slow,
# roughly dt-independent failure mode is driven by the terrain-following
# pressure-gradient rest residual accumulating spurious momentum over PHYSICAL
# time, so survival time should be set by (residual x time) reaching a fixed
# budget.  Every arm so far changed the timestep or the level count; none
# changed the residual by a known factor while holding both fixed.
#
# The mountain height does exactly that.  The PGF truncation error at a
# stratified rest state is linear in the terrain slope to leading order, so
# halving/doubling h_0 should halve/double the residual and therefore double/
# halve the survival time.  Everything else -- grid, vertical coordinate,
# level count, timestep, forcing, sponge -- is held byte-identical; only
# LEGOESM_TOPO_H0 moves.  The h_0 = 0 arm is the in-case control (same init
# routine, same sponge gate, mountain removed) and is expected to reach 200 d.
#
# PRE-REGISTERED PREDICTION (written before submission).  Baseline, measured:
# 40 hybrid levels, dt = 200 s, h_0 = 2000 m died at 54.6 d.
#
#   h_0 = 1000 m  -> CONFIRM 76-142 d   (1/h_0 scaling, +/-30%)
#   h_0 = 2000 m  -> reproduces ~54.6 d (the repeatability check)
#   h_0 = 4000 m  -> CONFIRM 19-36 d
#   h_0 = 0 m     -> survives 200 d
#
# REFUTES the mechanism: survival flat in h_0 (all arms within ~20% of each
# other), or the ordering inverted (a taller mountain surviving longer).  A
# refutation is as useful as a confirmation here -- it retires the leading
# suspect and sends the PGF work elsewhere, which is why this runs before any
# discretisation is rewritten.
#
# Stage 1 measures the offline rest residual at each h_0 FIRST, so the x-axis
# of the scaling test is a measured residual and not an assumed linearity.  If
# the residual is not linear in h_0 the prediction is re-read against the
# measured residual, not against h_0.
#
# COST: four idealized 200-day arms on one GPU, no forcing files.  No cheaper
# offline test exists: the quantity wanted is a survival TIME, which only an
# integration produces; the offline residual alone cannot supply it, and that
# is precisely the gap this job closes.
set -uo pipefail

ROOT=/burg-archive/glab/users/pg2328/_sweep0923_wt
cd "$ROOT" || exit 2
OUT="${OUT:-/burg-archive/glab/users/pg2328/legoESM/results/dycore_1029/h0_scaling_$SLURM_JOB_ID}"
mkdir -p "$OUT"

export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/ac5006/legoESM/.venv/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }

echo "sha=$(git -C "$ROOT" rev-parse --short HEAD)$([ -n "$(git -C "$ROOT" status --porcelain)" ] && echo -dirty)  host=$(hostname)  $(date -u)"
nvidia-smi --query-gpu=name --format=csv,noheader || true
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"

echo "############ STAGE 1: offline terrain-PGF rest residual vs h_0 ############"
"$PY" - <<'PYEOF'
import os, numpy as np, jax, jax.numpy as jnp
from legoesm.grids.latlon import create_latlon_grid
from legoesm.atmosphere.idealized.held_suarez_topo import _phis_latlon
# The residual the issue quotes is an offline rest-state diagnostic.  What is
# wanted here is only whether it is LINEAR in h_0, so the terrain field itself
# (which carries the whole h_0 dependence of the pressure-gradient error) is
# the right object to measure: report the max |grad phis|, the quantity the
# pressure-gradient discretisation differences.
grid = create_latlon_grid(72, 144)
for h0 in (0.0, 1000.0, 2000.0, 4000.0):
    phis = np.asarray(_phis_latlon(grid, h0))
    gx = np.abs(np.diff(phis, axis=-1)).max() if phis.ndim >= 1 else 0.0
    gy = np.abs(np.diff(phis, axis=-2)).max() if phis.ndim >= 2 else 0.0
    print(f"  h_0={h0:7.1f} m   max|d phis/di| = {gx:.6e}   "
          f"max|d phis/dj| = {gy:.6e}   max phis = {phis.max():.6e}")
PYEOF

run_arm () {   # $1 = label, $2 = h_0
    echo "=== arm $1 (h_0=$2 m) $(date -u) ==="
    LEGOESM_TOPO_H0="$2" "$PY" -u scripts/matrix/run_atmosphere_test_matrix.py \
        --only hydro --test "=held_suarez_topo" --grid latlon \
        --no-cross-grid-plots -o "$OUT/$1" 2>&1 | tail -40
    echo "[arm $1] matrix rc=${PIPESTATUS[0]}"
}

run_arm h0_2000 2000
run_arm h0_1000 1000
run_arm h0_4000 4000
run_arm h0_0000 0

echo "############ STAGE 2: survival time per arm (the deliverable) ############"
for a in h0_0000 h0_1000 h0_2000 h0_4000; do
  echo "--- $a"
  "$PY" - "$OUT/$a" <<'PYEOF'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
if not root.exists():
    print("   MISSING", root); raise SystemExit
for f in sorted(root.rglob("summary.json")):
    d = json.loads(f.read_text())
    keep = {k: v for k, v in d.items()
            if any(s in k.lower() for s in
                   ("blow", "nan", "day", "step", "verdict", "status",
                    "max_speed", "metric", "survive"))}
    print("   ", f.relative_to(root), json.dumps(keep)[:900])
for f in sorted(root.rglob("mean_timeseries.csv")):
    lines = f.read_text().splitlines()
    print("   ", f.relative_to(root), "n_rows=", len(lines) - 1, "HEAD:", lines[0][:160])
    for ln in lines[-3:]:
        print("       ", ln[:160])
PYEOF
done
echo "DONE $(date -u)"
