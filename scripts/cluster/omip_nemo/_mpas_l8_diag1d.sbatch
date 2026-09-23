#!/bin/bash
#SBATCH --job-name=l8diag
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --constraint="a40|rtx8000"
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpasfwnemo_%j.out
#
# THE HARMONIZED MPAS CARD, EXTENDED TO 180 DAYS (matched pair with the
# tripole trp_visc180 run).  The 30-day A/B (jobs 9561558/9561560) confirmed
# the full harmonized card (mle + monthly-IC + dm2dc + sw-rgb-chl) is
# regression-safe: SST rmse 0.491, bias +0.005, Arctic SSS bias +0.26 (was
# +0.48 pre-harmonization).  The residual nino3 +1.00 at day 30 is the same
# shared warm-tongue transient the tripole shows; the tripole's plateaus
# +1.42 -> +0.70 by day 180, so the real cross-grid + vs-NEMO comparison
# needs matched 180-day trajectories.
#
# NEMO-FAITHFUL FRESHWATER ARM (30 d).  Identical to _mpas_unified_180d
# except for ONE freshwater flag, so this is a controlled pair against
# that run's OWN day-15/30 snapshots (same code, same forcing, same IC).
#
# WHY: the unified 180 d run drives 16 Amazon-mouth cells to salinity
# EXACTLY 0.0000 by day 60 (0 at day 30, 6 at day 45).  Bit-exact zeros over
# 16 cells are the salinity clamp in ocean_model_mpas.py firing, i.e. the
# model computed NEGATIVE salinity and silently created salt; that plume
# carries 100.9% of the day-60 tropical SSS rmse growth (0.575 -> 0.803).
# The older arm reaches the same state by day 90, so the defect is
# PRE-EXISTING, not caused by the unified card.
# Codex (job 9610534) CONFIRMED by code read that (i) the virtual-salt term
# runs ON TOP of the live eta/z-star volume path -- a double count NEMO does
# not have -- (ii) `--freshwater-salinity local` does reach the RUNOFF
# channel, not only P-E/ice, and (iii) the clamp violates salt conservation
# with no compensating normalization.  GLM ranked the double count above the
# constant-S_ref term and required measurement before any default flip.
#
# PRE-REGISTERED, day 15 and day 30 vs the unified arm's matched snapshots:
# SCOPE (user-approved 2026-09-03, "full NEMO stack"): the three flags the
# driver requires TOGETHER -- real_freshwater closure, restoring routed as a
# WATER flux, live_s normalization.  Read off the oracle, not inferred:
# ORCA1 namelist_cfg nn_sssr=2 is documented in sbcssr.F90:127 as a "volume
# flux (emp) and associated heat flux", i.e. NEMO restores with water, not
# with a direct salinity tendency as our default does.
# NOT changed: global freshwater normalization stays ON.  ORCA1 runs
# nn_fwb=1, so a global budget correction is what the oracle does too;
# --no-normalize-freshwater would be a deviation, not a fix.
# NOT passed: --freshwater-salinity local is moot here -- real_freshwater
# drops the virtual-salt term entirely, so dilution is by local salinity
# through the volume channel by construction.
#
#   CONFIRM (local-salinity adopted for production): zero clamped cells at day
#     30, tropical SSS rmse no worse than the unified arm's 0.540 at day 30,
#     and global SST rmse within 0.05 C of its 0.496.
#   REFUTE (defect owner is elsewhere -- regridding or mesh concentration):
#     clamped-cell count still > 0 at day 30, or the Amazon minimum surface
#     salinity still falling.
#   ABSTAIN: zeros gone but global SST rmse degrades > 0.05 C -- then the
#     current default is masking a compensating error and the next step is
#     the Arctic salt-budget check GLM asked for, not adoption.
set -uo pipefail
PIN=${1:?usage: sbatch _mpas_fwnemo_l8_diag1d.sbatch <pinned-sha> [leg]}
LEG=${2:-1}
# LEVEL 8 = 655,362 cells (~29 km, 2x finer than the level-7 arm); dt halved
# for CFL (150 -> 75 s) => ~8x the cost, so the 180 days run as a 72 h
# restart-chained leg sequence (--restart-save every 15 d, --restart-from on
# legs >= 2, ABSOLUTE --years target). ONE VARIABLE vs mpas_fwnemo_180d:
# resolution. PRE-REGISTERED: nino3 bias at matched d30/60/90 vs the
# level-7 arm (+1.00/+0.58/+0.10 on the earlier card) — CONFIRM resolution
# sensitivity if |dnino3| > 0.2 C at d30; otherwise the early transient is
# not resolution-limited on this grid.
RESTART_FROM=""
# Level 8 OOMed a 21.8 GiB allocation under JAX's default 75% preallocation
# cap (job 9631427); let XLA use the whole card.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.98

ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
NEMOIN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS
OUT=$ROOT/results/omip_nemo/mpas_fwnemo_l8_diag1d
[ "$LEG" -ge 2 ] && RESTART_FROM="--restart-from $OUT/restart.npz"

echo "=== gate: corrected forcing is the default ==="
JAX_PLATFORMS=cpu $PY -m pytest -q tests/ocean/unit/test_core2_corrected_cache_default.py tests/unit/test_run_omip_core2_diag_mpas.py 2>&1 | tail -2
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: gate rc=$rc"; exit 1; }

JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid mpas \
  --mpas-level 8 --mpas-lloyd 20 --mpas-vmix tke \
  --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic \
  --tke-kappa-convention veros_sqrte \
  --dt 75 --nemo-vertical --partial-cell \
  --mle --mle-ce 0.06 \
  --dm2dc --sw-rgb-chl \
  --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 \
  --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice \
  --prognostic-ice-dynamics free_drift --ice-lead-freeze-source nemo_qlead \
  --isf --isf-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc \
  --iwm --iwm-forcing-file $NEMOIN/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc \
  --bbl-adv --years 0.00274 --snapshot-every-days 1 \
  --freshwater-closure real_freshwater --sss-restore-channel water_flux \
  --sss-restore-normalization live_s \
  --diag-every-days 0.0009 --tracer-advection superbee --runoff-depth-nemo-ini \
  --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic \
  --ice-ocean-heat-coeff 70 \
  --woa-init \
  --nemo-monthly-init $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc \
    $NEMOIN/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc \
  --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc \
  --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc \
  --sss-restore-file $NEMOIN/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc \
  --ice-init $NEMOIN/Ice_initialization.nc \
  --restart-save $OUT/restart.npz $RESTART_FROM --output $OUT 2>&1 | grep -vE "^\s*$" | tail -60
run_rc=$?
echo "=== run rc=$run_rc"
[ $run_rc -eq 0 ] || { echo "FATAL: run failed rc=$run_rc"; exit 1; }

export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc

echo "=== matched-day scores vs GATEWAY (rec 5/11/17 = d26-30/56-60/86-90) ==="
for pair in "0030 5" "0060 11" "0090 17"; do
  set -- $pair
  SNAP=$OUT/snapshot_day$1.npz
  [ -f "$SNAP" ] || { echo "no day $1 snapshot"; continue; }
  echo "### day $1 vs GATEWAY rec $2 ###"
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GW_T" --nemo-time-idx "$2" --freeze-clamp-C -1.8 \
    --sss-faithful --grid-label mpas_fwnemo30_d$1 \
    --output-dir $ROOT/results/omip_nemo/compare_mpasfwnemo30_d$1 2>&1 | tail -38
done

echo "=== OUR nino3 SST trajectory (plateau test, no oracle) ==="
$PY - "$OUT" <<'PY'
import sys, glob, numpy as np
out = sys.argv[1]
prev = None
for d in range(15, 181, 15):
    snap = f"{out}/snapshot_day{d:04d}.npz"
    if not glob.os.path.exists(snap):
        continue
    z = np.load(snap)
    lat = z["lat_T"]; lon = z["lon_T"] % 360
    T0 = z["T"][..., 0]; wet = z["land_mask"] > 0.5
    box = (np.abs(lat) <= 5) & (lon >= 210) & (lon <= 270) & wet
    # codex 9562869: MPAS SCVT cells are quasi-uniform area; uniform weights,
    # not cos(lat), when the snapshot carries no cell_area.
    area = z["cell_area"] if "cell_area" in z else np.ones_like(lat)
    w = area[box]; v = T0[box]
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    sst = float((v[ok] * w[ok]).sum() / w[ok].sum())
    dd = "" if prev is None else f"  d{sst - prev:+.3f}"
    print(f"day {d:3d}  nino3 SST {sst:.3f} C{dd}")
    prev = sst
print("\nPLATEAU (|d| shrinking, <0.3 K over d90->180) => bounded residual; "
      "LINEAR growth => implementation drift remains.")
PY
echo DONE
