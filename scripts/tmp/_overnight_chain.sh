#!/usr/bin/env bash
# Overnight chain:
#   Wait for the 10-yr implicit spinup to finish, then run:
#     1. 1-yr verification from the spinup endpoint (Crit 1.2/1.3 check)
#     2. 40-yr continuation to reach sim-yr 50 (the original-experiment redo)
#     3. Comparison plots vs the broken-solver 50yr run
#
# All output goes under results/ocean/, with timestamped messages in
# results/ocean/_overnight_chain.log.

set -e
set -o pipefail

# repo root (robust to this script's depth; it now lives in scripts/tmp/)
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PY=/Users/dhruvbalwada/code/miniforge3/bin/python
LOG=results/ocean/_overnight_chain.log
SPINUP_LOG=results/ocean/global_overturning_implicit_spinup/run.log
SPINUP_RESTART=results/ocean/global_overturning_implicit_spinup/restart_day003650.npz

ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] Overnight chain started" | tee -a "$LOG"

# --- Step 1: wait for spinup ---
echo "[$(ts)] Waiting for spinup completion..." | tee -a "$LOG"
until grep -q "Spinup complete" "$SPINUP_LOG" 2>/dev/null \
   || grep -qE "BLOWUP|Traceback" "$SPINUP_LOG" 2>/dev/null; do
    sleep 60
done

if grep -qE "BLOWUP|Traceback" "$SPINUP_LOG" 2>/dev/null; then
    echo "[$(ts)] !! Spinup FAILED — chain aborted" | tee -a "$LOG"
    tail -20 "$SPINUP_LOG" | tee -a "$LOG"
    exit 1
fi

if [ ! -f "$SPINUP_RESTART" ]; then
    echo "[$(ts)] !! Final spinup restart missing: $SPINUP_RESTART" | tee -a "$LOG"
    exit 1
fi
echo "[$(ts)] Spinup complete; final restart $SPINUP_RESTART exists" | tee -a "$LOG"

# --- Step 2: 1-yr verification ---
VERIFY_DIR=results/ocean/momentum_budget_online_implicit_postspinup
echo "[$(ts)] Step 2: 1-yr verification from clean spinup state" | tee -a "$LOG"
mkdir -p "$VERIFY_DIR"
JAX_ENABLE_X64=1 "$PY" -c "
import os, sys
sys.path.insert(0, 'scripts')
os.environ.setdefault('JAX_ENABLE_X64', '1')
from pathlib import Path
import scripts.run_drake_momentum_budget_implicit as r
r.RESTART_PATH = Path('$SPINUP_RESTART')
r.OUTPUT_DIR = Path('$VERIFY_DIR')
r.main()
" > "$VERIFY_DIR/run.log" 2>&1
echo "[$(ts)] Step 2 done — see $VERIFY_DIR/run.log" | tee -a "$LOG"
tail -8 "$VERIFY_DIR/run.log" | tee -a "$LOG"

# Extract Crit 2 number for quick visibility
"$PY" - <<'PYEOF' 2>&1 | tee -a "$LOG"
import os
os.environ.setdefault('JAX_ENABLE_X64','1')
import numpy as np, sys
sys.path.insert(0,'scripts')
from legoesm.ocean.vertical import create_ocean_z_star
from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.experiments.global_overturning import GlobalOverturningConfig

cfg = GlobalOverturningConfig(use_gm_redi=True)
z_coord = create_ocean_z_star(n_levels=cfg.n_levels, H_max=cfg.H_max,
                              dz_surface=cfg.dz_surface, dz_deep=cfg.dz_deep)
grid = create_latlon_grid(36, 72)
dz = np.asarray(z_coord.dz_ref, dtype=np.float64)
H_max = cfg.H_max

r = np.load('results/ocean/global_overturning_implicit_spinup/restart_day003650.npz')
v_mask = np.asarray(r['v_mask'], dtype=np.float64)
u_mask = np.asarray(r['u_mask'], dtype=np.float64)
H_bathy = np.asarray(r['H_bathy'], dtype=np.float64)
mask = np.asarray(r['land_mask'], dtype=np.float64)

cos_lat = np.cos(np.clip(np.asarray(grid.lat), -np.pi/2 + 1e-9, np.pi/2 - 1e-9))
dx_u = cos_lat * float(grid.radius) * float(grid.dlon)
dy = float(grid.radius) * float(grid.dlat)
J_DRAKE = np.arange(2, 7); RHO_0 = 1027.0; n_lon = 72

f_cell = np.asarray(grid.f)
f_u = 0.5*(np.roll(f_cell,1,axis=1)+f_cell)
f_u = np.concatenate([f_u, f_u[:,0:1]], axis=1)

n_wet = np.sum(u_mask[:,:n_lon], axis=1)
area_band = float(np.sum((dx_u * n_wet * dy)[J_DRAKE]))

d = np.load('results/ocean/momentum_budget_online_implicit_postspinup/tendency_3d_means.npz')
eta_mean = d['state_eta_mean']; v_mean = d['state_v_mean']
H_total_cell = (eta_mean + H_bathy) * mask
H_v_int = 0.5*(H_total_cell[:-1]+H_total_cell[1:])
H_v = np.concatenate([np.zeros((1,72)), H_v_int, np.zeros((1,72))], axis=0)
H_u_core = 0.5*(np.roll(np.where(mask>0.5,H_total_cell,0),1,axis=1) +
                np.where(mask>0.5,H_total_cell,0))
H_u = np.concatenate([H_u_core, H_u_core[:,0:1]], axis=1)
h_v = dz[None,None,:] * (H_v/H_max)[..., None]
V_bar = np.where(H_v>1e-3, np.sum(v_mean*h_v,axis=-1)/np.maximum(H_v,1e-3), 0)
V_west = np.roll(V_bar, 1, axis=1)
V_at_u_core = 0.25*(V_bar[:-1]+V_bar[1:]+V_west[:-1]+V_west[1:])
V_at_u = np.concatenate([V_at_u_core, V_at_u_core[:,0:1]], axis=1)
cor = RHO_0 * H_u * f_u * V_at_u * u_mask
crit2 = float(np.sum(cor[J_DRAKE,:n_lon].sum(axis=1) * dx_u[J_DRAKE] * dy)) / area_band
print(f"  POSTSPINUP Crit 2  rho*H*f*<V_baro>_Drake = {crit2*1000:+.3f} mPa")
print(f"    target |...| < 5 mPa")
print(f"    50yr-old-restart was -8.018 mPa (was failing)")

# Crit 1.2: max |<V_baro>| off polar
off_polar = np.zeros_like(v_mask, dtype=bool); off_polar[2:35] = True
off_polar &= (v_mask > 0.5)
print(f"  POSTSPINUP Crit 1.2 max|<V_baro>|_off_polar = {float(np.max(np.abs(V_bar*v_mask)[off_polar])):.4e} m/s")
print(f"    target < 5e-3 m/s")
PYEOF

echo "[$(ts)] === Crit 1.2 / Crit 2 numbers from clean-IC verification logged above ===" | tee -a "$LOG"

# --- Step 3: 40-yr continuation ---
echo "[$(ts)] Step 3: 40-yr continuation to sim-yr 50" | tee -a "$LOG"
mkdir -p results/ocean/global_overturning_50yr_implicit
JAX_ENABLE_X64=1 "$PY" scripts/global_overturning/run_global_overturning_50yr_implicit_continuation.py \
    > results/ocean/global_overturning_50yr_implicit/run.log 2>&1
echo "[$(ts)] Step 3 done" | tee -a "$LOG"
tail -10 results/ocean/global_overturning_50yr_implicit/run.log | tee -a "$LOG"

# --- Step 4: comparison plots ---
echo "[$(ts)] Step 4: comparison plots vs original 50yr run" | tee -a "$LOG"
JAX_ENABLE_X64=1 "$PY" scripts/plot_implicit_spinup_restart.py 18250 \
    > /dev/null 2>&1 || true

echo "[$(ts)] Overnight chain COMPLETE" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Key outputs to inspect when you wake up:" | tee -a "$LOG"
echo "  - $VERIFY_DIR/tendency_3d_means.npz (1-yr verification)" | tee -a "$LOG"
echo "  - results/ocean/global_overturning_50yr_implicit/restart_day018250.npz (yr-50 state)" | tee -a "$LOG"
echo "  - $LOG (this chain log)" | tee -a "$LOG"
