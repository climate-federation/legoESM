#!/bin/bash
#SBATCH --job-name=score3e
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/score3e_%j.out
#
# Pass E: the control that actually tests the regrid-asymmetry objection.
#
# The round-2 review rejected the earlier 2-degree rerun as a control, and it
# was right: --res-deg moves the target sample points but leaves the IDW
# regridder taking k=4 source neighbours, whose PHYSICAL footprint still
# differs per mesh (~60 km on MPAS ico7 vs ~111 km on ORCA1).  Because NEMO is
# itself ORCA1, that asymmetry could flatter tripole-vs-NEMO and inflate the
# headline "the two grids agree with each other 3.5-3.9x better than either
# agrees with NEMO".
#
# --smooth-radius-deg imposes ONE physical top-hat on all three fields AFTER
# regridding.  Prediction if the artifact is REAL: tripole-vs-NEMO degrades
# relative to MPAS-vs-NEMO as the radius grows, and the cross-grid/NEMO ratio
# falls.  Prediction if ABSENT: the ratio is roughly radius-independent.
# Both arms are the same two snapshots, so nothing but the filter changes.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
export PYTHONPATH="$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
[ -r "$GT" ] || { echo "FATAL: NEMO grid_T unreadable"; exit 1; }

echo; echo "########## 0. SCORER UNIT TESTS (gate)"
$PY -m pytest tests/validate/test_compare_three_way_nemo.py -q
trc=$?; echo "=== scorer tests rc=$trc"
[ $trc -eq 0 ] || { echo "FATAL: scorer tests failed -- refusing to produce numbers"; exit 1; }

NFAIL=0
XT=$R/xgrid_trp_d30/snapshot_final.npz
XM=$R/xgrid_mpas_d30/snapshot_final.npz

for rad in 0 2 4; do
  if [ "$rad" = "0" ]; then
    OUT=$R/xgrid3_smooth_none; SM=""
  else
    OUT=$R/xgrid3_smooth_${rad}deg; SM="--smooth-radius-deg $rad"
  fi
  echo; echo "########## common-footprint radius = ${rad} deg"
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $XT --mpas $XM --nemo-gridt "$GT" --nemo-month 1 \
    --mask-mode nearest --res-deg 1.0 $SM \
    --label-tripole "tripole eORCA1" --label-mpas "MPAS ico7" --out-dir $OUT
  rc=$?
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
  echo "=== radius ${rad} rc=$rc"
done

echo; echo "########## ratio table (the quantity the control tests)"
$PY - <<'PYEOF'
import json
base = "/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo"
rows = [("none", "xgrid3_smooth_none"), ("2deg", "xgrid3_smooth_2deg"),
        ("4deg", "xgrid3_smooth_4deg")]
print(f"{'radius':8s} {'field':5s} {'trp-NEMO':>9s} {'mpas-NEMO':>10s} "
      f"{'trp-mpas':>9s} {'ratio(mean/xgrid)':>18s}")
for tag, d in rows:
    try:
        r = json.load(open(f"{base}/{d}/report.json"))
    except OSError as exc:
        print(f"{tag:8s} MISSING: {exc}")
        continue
    for f in ("SST", "SSS", "MLD"):
        F = r["fields"][f]
        t = F["tripole_vs_nemo"]["global"]["rmse"]
        m = F["mpas_vs_nemo"]["global"]["rmse"]
        x = F["tripole_vs_mpas"]["global"]["rmse"]
        print(f"{tag:8s} {f:5s} {t:9.4f} {m:10.4f} {x:9.4f} "
              f"{((t + m) / 2) / x:18.2f}")
PYEOF
echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
