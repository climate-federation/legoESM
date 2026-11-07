#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=faith_trp2tke
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=60:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/faith_trp2tke_%j.out

# NEMO zdftke closure on the MAXIMALLY-FAITHFUL tripole config (P4/P5
# campaign, LAST audited namelist gap): trp2_p46 levers + --tripole-vmix tke
# (ORCA1 &namzdf_tke namelist mapping; composes additively with zdfiwm).
#
# SMOKE-GATED: trp2 runs superbee tracer advection, and TKE x superbee blew
# up DINO in ~15 days (KE ~700x; open bug task).  So:
#   arm 0: unit tests for the new wiring (CPU, fast) — abort on red;
#   arm 1: 10-day smoke of the FULL trp2 config + --tripole-vmix tke,
#          realism gate on the day-10 snapshot (finite T/S/u/v/eta,
#          max|u|,|v| < 3 m/s, KE finite);
#   arm 2: only if arm 1 is clean — the 90-day A/B twin of trp2_p46
#          (LABEL trp2_tke), scored vs NEMO month-3 exactly like p46.
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/nemo-gaps-p4p5; cd $WT
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
INP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree ==="; git -C $WT log --oneline -1; git -C $WT status --short | head -20

GA="--grid tripole --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc --dt 150 \
  --min-levels 2 --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal \
  --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
P45="--bottom-drag-scheme nemo_quadratic \
  --iwm --iwm-forcing-file $INP/zdfiwm_forcing_TRA.nc \
  --dm2dc \
  --isf --isf-forcing-file $INP/runoff-icb_DaiTrenberth_Depoorter.nc \
  --bbl-adv"
INITD=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions
SSSCLIM=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc
IC="--woa-init --woa-t $WTIC --woa-s $WSIC \
  --nemo-monthly-init $INITD/woce_temp_monthly_init_4p2.nc $INITD/woce_salt_monthly_init_4p2.nc \
  --sss-restore-file $SSSCLIM"
TKE="--tripole-vmix tke"

# ---------------- arm 0: unit tests for the new wiring ----------------
echo "=== ARM 0: unit tests (test_tripole_vmix_cli) ==="
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 $PY -m pytest -q \
  tests/ocean/unit/test_tripole_vmix_cli.py 2>&1 | tail -15
UT_RC=${PIPESTATUS[0]}
echo "unit_rc=$UT_RC"
if [ "$UT_RC" -ne 0 ]; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!!! ARM 0 FAILED: --tripole-vmix unit tests red — NOT running !!"
  echo "!!! the smoke or the 90-day A/B.  Fix the wiring first.       !!"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 1
fi

# ---------------- arm 1: 10-day smoke + realism gate ----------------
SLABEL=trp2_tke_smoke
SOUT=$ROOT/results/omip_nemo/nemolev_${SLABEL}
echo "=== ARM 1: RUN $SLABEL (10-day smoke, full trp2 config + TKE) ==="; date
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $P45 $IC --sw-rgb-chl $TKE \
  --years 0.0274 --snapshot-every-days 5 \
  --output $SOUT 2>&1 | tail -120
SMOKE_RC=${PIPESTATUS[0]}
echo "smoke_run_rc=$SMOKE_RC"; date
if [ "$SMOKE_RC" -ne 0 ]; then
  echo "!!! ARM 1 SMOKE run exited rc=$SMOKE_RC (blowup/crash) — SKIPPING"
  echo "!!! the 90-day A/B.  Inspect $SOUT"
  exit 1
fi

# The runner writes periodic day snapshots only for step != n_steps; the
# day-10 state of a 10-day run is snapshot_final.npz.
SNAP10=$SOUT/snapshot_final.npz
echo "=== ARM 1 GATE: realism check on $SNAP10 (day-10 final state) ==="
$PY - "$SNAP10" <<'PYEOF'
import sys
import numpy as np
path = sys.argv[1]
try:
    d = np.load(path)
except Exception as e:
    print(f"[gate] FAIL: no day-10 final snapshot ({type(e).__name__}: {e})"
          " -> run died before day 10 (blowup or crash).")
    sys.exit(1)
ok = True
u, v = d["u"], d["v"]
for name in ("T", "S", "u", "v", "eta"):
    if name not in d:
        continue
    a = d[name]
    if not np.isfinite(a).all():
        n = int((~np.isfinite(a)).sum())
        print(f"[gate] FAIL: {name} has {n} non-finite values")
        ok = False
mu = float(np.max(np.abs(u))) if np.isfinite(u).all() else float("nan")
mv = float(np.max(np.abs(v))) if np.isfinite(v).all() else float("nan")
ke = 0.5 * (float(np.mean(u * u)) + float(np.mean(v * v)))
T = d["T"]
print(f"[gate] day-10: max|u|={mu:.3f} m/s  max|v|={mv:.3f} m/s  "
      f"KE~{ke:.3e} m2/s2  T[{np.min(T):.2f},{np.max(T):.2f}] C")
if not (mu < 3.0 and mv < 3.0):
    print("[gate] FAIL: |u| or |v| >= 3 m/s (realism gate)")
    ok = False
if not np.isfinite(ke):
    print("[gate] FAIL: KE non-finite")
    ok = False
print("[gate] " + ("PASS: smoke clean -> 90-day A/B may start"
                   if ok else "FAIL"))
sys.exit(0 if ok else 1)
PYEOF
GATE_RC=$?
echo "smoke_gate_rc=$GATE_RC"
if [ "$GATE_RC" -ne 0 ]; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!!! ARM 1 SMOKE FAILED the realism gate — SKIPPING the 90-day  !!!"
  echo "!!! trp2_tke A/B.  Suspect TKE x superbee (DINO blew up ~day   !!!"
  echo "!!! 15, KE ~700x).  Inspect $SOUT"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 1
fi

# ---------------- arm 2: 90-day A/B vs trp2_p46 ----------------
LABEL=trp2_tke
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
echo "=== ARM 2: RUN $LABEL (trp2_p46 + --tripole-vmix tke, 90d A/B) ==="; date
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $P45 $IC --sw-rgb-chl $TKE \
  --years 0.3 --snapshot-every-days 30 \
  --output $OUT 2>&1 | tail -250
AB_RC=${PIPESTATUS[0]}
echo "run_rc=$AB_RC"; date
if [ "$AB_RC" -ne 0 ]; then
  echo "!!! ARM 2 run exited rc=$AB_RC — scoring whatever day-90 snapshot"
  echo "!!! exists (p46 semantics), but treat the scorecard with care."
fi

SNAP=$OUT/snapshot_day0090.npz
if [ -f "$SNAP" ]; then
  echo "=== SCORE + BATTERY (month-3, vs NEMO) ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "nemolev_${LABEL}" \
    --output-dir $ROOT/results/omip_nemo/compare_nemolev_${LABEL}_m03 2>&1 | tail -18
  $PY scripts/validate/diag_omip_nemo_battery.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --domain-cfg $INP/domain_cfg.nc \
    --nemo-gridv $RUN/ORCA1_1y_20000101_20041231_grid_V.nc \
    --nemo-gridu $RUN/ORCA1_1y_20000101_20041231_grid_U.nc \
    --grid-label "nemolev_${LABEL}" \
    --output-dir $ROOT/results/omip_nemo/battery_nemolev_${LABEL}_m03 2>&1 | tail -18
else
  echo "[run] no day-90 snapshot -> check $OUT for blowup."
fi
echo "=== DONE ==="; date
