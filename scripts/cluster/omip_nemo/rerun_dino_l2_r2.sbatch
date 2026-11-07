#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=dino_l2_r2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/dino_l2_%j.out
#
# DINO "L2" two-level intercomparison
# ===================================
# Runs legoESM at the DINO 1deg R1 lat-lon config under each MODEL RECIPE (a pure
# config overlay selecting another ocean model's canonical DINO blocks — EOS,
# advection/limiter, vertical mixing, momentum/Coriolis/integrator), then maps
# where the recipes diverge from the NEMO reference and from each other.
#
#   L1 anchors : legoesm_default (Wright + KPP), nemo_paper (Kamm 2025: nemo_seos + TKE)
#   L2 cards   : veros (Vallis EOS + TKE + superbee + streamfunction/AB2)
#                mitgcm (JMD95-ish EOS + KPP + DST3 + flux-form/AB2/unsplit-FS)
#                oceananigans (TEOS-10 + CATKE + WENO + AB2)
#
# recipe->config: legoesm.ocean.experiments.dino.DINO_RECIPES
# dispatch test : tests/ocean/unit/test_dino_experiment.py::TestDINORecipes
#
# FIDELITY / STABILITY CAVEATS (see the scoping report):
#  * Veros + nemo_paper use TKE, which is only stable to ~day 226 on our 1deg
#    DINO (the SW-corner mode on DINOConfig.vmix_scheme). The common map day
#    (MAP_DAY, default 90) and run length (DAYS, default 180) stay under that.
#    mitgcm + legoesm_default use KPP (multi-year-stable) and may be extended.
#  * Oceananigans CATKE at DINO config is UNVERIFIED -> the SMOKE phase NaN-gates
#    every recipe (SMOKE days) before committing to the full run.
#  * MITgcm EOS uses unesco80 (~JMD95); MDJWF is not bit-exact in legoESM.
#  * L2 cards are lat-lon-only (they select lat-lon-C-grid dycore blocks); a
#    single MPAS reference run (legoesm_default/KPP) feeds the vs-paper panels.
#
# Launch (from the repo root):
#   sbatch scripts/cluster/omip_nemo/dino_l2_intercomparison.sbatch
# Optional overrides (env):
#   DAYS=180 SMOKE=30 SNAP=15 MAP_DAY=90 \
#   NEMO_RESTART=/path/to/restart_prod_1deg.nc \
#     sbatch scripts/cluster/omip_nemo/dino_l2_intercomparison.sbatch
set +e
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RD="$PY -u scripts/run/run_dino.py"
R="$ROOT/results/dino_l2_r2"; mkdir -p "$R"
DAYS=${DAYS:-180}
SMOKE=${SMOKE:-30}
SNAP=${SNAP:-15}
MAP_DAY=${MAP_DAY:-90}
NEMO_RESTART=${NEMO_RESTART:-$ROOT/data/dino_paper/Reference_experiments/EXP_R1/restart_prod_1deg.nc}
flt() { grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin|tensorflow|TpuPlatform|Updating files|jax\." ; }
echo "=== HOST ==="; hostname; date
echo "DAYS=$DAYS SMOKE=$SMOKE SNAP=$SNAP MAP_DAY=$MAP_DAY"
echo "NEMO_RESTART=$NEMO_RESTART  (exists: $([ -f "$NEMO_RESTART" ] && echo yes || echo NO))"

echo; echo "########## PHASE 0: recipe->config dispatch tests (compute node) ##########"
$PY -m pytest -q \
  tests/ocean/unit/test_dino_experiment.py::TestDINORecipes \
  "tests/ocean/unit/test_dino_experiment.py::TestDINOConfig::test_vmix_scheme_default_and_dispatch" \
  2>&1 | tail -8

# --- one recipe: smoke (NaN-gate) then full length; ACC spin-up figure ---
RECIPES="legoesm_default nemo_paper veros mitgcm oceananigans"
run_recipe() {
  local rec=$1
  echo; echo "########## RECIPE: $rec  (smoke ${SMOKE}d -> full ${DAYS}d) ##########"
  $RD --recipe "$rec" --grid latlon --days "$SMOKE" --snapshot-every-days "$SNAP" \
      --output-dir "$R/${rec}_smoke" > "$R/${rec}_smoke.log" 2>&1
  tail -4 "$R/${rec}_smoke.log" | flt
  if grep -qi 'nan' "$R/${rec}_smoke.log"; then
    echo "!! $rec SMOKE produced NaN -> SKIPPING full run (unstable at DINO config)"
    return 1
  fi
  # Realism gate, not just NaN (inspect-realism rule): the veros card passed the
  # NaN check while ~700x too energetic (|v| 4.5 m/s, KE 1.3e14) and only NaN'd
  # at day ~60-75. Bound the LAST smoke diag row: |u|,|v| < 3 m/s (1-deg basin
  # jets are O(1)), |eta| < 3 m. Diag columns: step day |u| |v| |eta| Tmax Tmin KE.
  read -r _u _v _eta <<< "$(awk '/^[[:space:]]*[0-9]+[[:space:]]/{u=$3; v=$4; e=$5} END{print u, v, e}' "$R/${rec}_smoke.log")"
  if awk -v u="$_u" -v v="$_v" -v e="$_eta" \
       'BEGIN{exit !(u>3.0 || v>3.0 || e>3.0)}' 2>/dev/null; then
    echo "!! $rec SMOKE UNPHYSICAL (|u|=$_u |v|=$_v |eta|=$_eta m; bounds 3 m/s, 3 m)"
    echo "   -> SKIPPING full run (would blow later; the NaN gate alone missed this)"
    return 1
  fi
  echo "-- smoke clean (NaN + realism: |u|=$_u |v|=$_v |eta|=$_eta); full ${DAYS}d --"
  $RD --recipe "$rec" --grid latlon --days "$DAYS" --snapshot-every-days "$SNAP" \
      --output-dir "$R/$rec" > "$R/${rec}.log" 2>&1
  tail -4 "$R/${rec}.log" | flt
  grep -qi 'nan' "$R/${rec}.log" && \
    echo "!! $rec FULL hit NaN before ${DAYS}d (TKE cards expected ~day 226); "\
"comparisons use the last finite snapshot (plot tools skip NaN tails)."
  $PY scripts/plot/plot_dino_acc.py "$R/$rec" --out "$R/dino_l2_${rec}_acc.png" 2>&1 | flt | tail -2
  return 0
}
for rec in $RECIPES; do run_recipe "$rec"; done

echo; echo "########## MPAS reference (legoesm_default/KPP, ${MAP_DAY}d) for vs-paper panels ##########"
$RD --recipe legoesm_default --grid mpas --days "$MAP_DAY" --snapshot-every-days "$SNAP" \
    --output-dir "$R/mpas_ref" > "$R/mpas_ref.log" 2>&1
tail -3 "$R/mpas_ref.log" | flt

echo; echo "########## PHASE 3: maps — each recipe vs NEMO R1 equilibrium (day ${MAP_DAY}) ##########"
for rec in $RECIPES; do
  [ -d "$R/$rec/snapshots" ] || { echo "skip $rec (no full run)"; continue; }
  if [ -f "$NEMO_RESTART" ] && [ -d "$R/mpas_ref/snapshots" ]; then
    echo "-- $rec vs NEMO equilibrium --"
    $PY scripts/plot/plot_dino_vs_paper.py "$NEMO_RESTART" "$R/$rec" "$R/mpas_ref" \
        --day "$MAP_DAY" --out "$R/dino_l2_${rec}_vs_paper.png" 2>&1 | flt | tail -3
  else
    echo "(NEMO restart or MPAS ref missing -> skipping vs-paper maps for $rec)"
  fi
done

echo; echo "########## L1 cross-grid sanity (nemo_paper lat-lon vs MPAS ref) ##########"
if [ -d "$R/nemo_paper/snapshots" ] && [ -d "$R/mpas_ref/snapshots" ]; then
  $PY scripts/plot/plot_dino_cross_grid.py "$R/nemo_paper" "$R/mpas_ref" \
      --day "$MAP_DAY" --out "$R/dino_l2_crossgrid_nemo_paper.png" 2>&1 | flt | tail -3
fi

echo; echo "########## INTER-RECIPE DIVERGENCE MATRIX (regridded SST/SSS RMS @ day ${MAP_DAY}) ##########"
# Reuses the shared regrid helpers from plot_dino_cross_grid (XG) — no re-derived
# interpolation. Prints the pairwise RMS(SST) + RMS(SSS) between every recipe on
# the common 1deg box: the quantified "map where the models diverge".
$PY - "$R" "$MAP_DAY" $RECIPES <<'PY' 2>&1 | flt
import sys, importlib.util, itertools
from pathlib import Path
import numpy as np
R = Path(sys.argv[1]); day = float(sys.argv[2]); recipes = sys.argv[3:]
here = Path("scripts/plot")
spec = importlib.util.spec_from_file_location("xg", here / "plot_dino_cross_grid.py")
XG = importlib.util.module_from_spec(spec); spec.loader.exec_module(XG)
fields = {}
for rec in recipes:
    d = R / rec
    if not (d / "snapshots").exists():
        print(f"  {rec}: no run — skipped"); continue
    try:
        # Walk BACKWARD from the target day to the last FINITE snapshot -- a
        # blown run (veros: NaN from ~day 60-75) otherwise poisons every one of
        # its matrix rows with NaN while claiming day-90 coverage.
        dd = day
        T = S = None
        while dd >= 0:
            snap, dday = XG._snap_for_day(d / "snapshots", dd)
            T, S = XG._regrid_TS(XG._latlon_coords(d), snap)
            if np.isfinite(T).all() and np.isfinite(S).all():
                break
            dd = dday - 1.0e-6 - 15.0   # step back one snapshot interval
            T = S = None
        if T is None:
            print(f"  {rec}: NO finite snapshot at any day <= {day} — skipped")
            continue
        if dday != day:
            print(f"  {rec}: blown after day {dday:.0f} — comparing its LAST "
                  f"FINITE state (day {dday:.0f}), not day {day:.0f}")
        fields[rec] = (T, S, dday)
    except Exception as e:  # noqa: BLE001 — best-effort diagnostic
        print(f"  {rec}: load failed ({e})")
labels = list(fields)
print("  recipes compared:", labels)
for a, b in itertools.combinations(labels, 2):
    Ta, Sa, da = fields[a]; Tb, Sb, db = fields[b]
    _, rT = XG._stats(Ta, Tb); _, rS = XG._stats(Sa, Sb)
    tag = "" if da == db else f"  [days {da:.0f} vs {db:.0f} — NOT synchronous]"
    print(f"  {a:16s} vs {b:16s}  RMS(SST)={rT:6.3f} C   RMS(SSS)={rS:6.3f} PSU{tag}")
PY

echo; echo "=== ALL DONE ==="; ls -lh "$R"/*.png 2>/dev/null
