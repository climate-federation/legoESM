#!/bin/bash
#SBATCH --job-name=fesomscore
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomscore_%j.out
#
# Score the FESOM2 (fesom-jax CORE2) day-30 arm against NEMO and against the
# two legoESM grids, using the codex-cleaned three-way comparator UNCHANGED:
# its arm labels are parameters, so two extra invocations cover every pair
# (JSON keys keep the tripole_/mpas_ names -- read the labels, not the keys).
#
# PROTOCOL CAVEAT (travels with every number): the FESOM2 arm runs the
# published fesom-jax hindcast protocol (JRA55-do 1958, PHC3.0 winter cold
# start, fesom-jax physics), NOT the xgrid matched-pair protocol (CORE-II
# normal-year, annual WOA IC, ORCA1-card TKE).  This is a THREE-MODEL
# comparison against the shared NEMO January reference, not a controlled
# pair; grid-vs-grid numbers here bound model+grid difference, not grid alone.
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
FS=$R/fesom_core2_d30/snapshot_day0030.npz
[ -r "$FS" ] || { echo "FATAL: FESOM day-30 snapshot missing"; exit 1; }
[ -r "$GT" ] || { echo "FATAL: NEMO grid_T unreadable"; exit 1; }

# REPAIR (idempotent): snapshots written before the writer sliced off the
# fesom-jax padded bottom tracer slot carry 48 levels vs 47 z_center_ref
# mids; slice tracers to the mid count so the (T, z) pairing is consistent.
$PY - "$FS" <<'EOF'
import sys
import numpy as np
p = sys.argv[1]
z = dict(np.load(p))
n = z["z_center_ref"].size
if z["T"].shape[-1] != n:
    print(f"[repair] {p}: tracers {z['T'].shape[-1]} levels -> {n}")
    z["T"] = z["T"][..., :n]
    z["S"] = z["S"][..., :n]
    np.savez_compressed(p, **z)
else:
    print(f"[repair] {p}: already consistent ({n} levels)")
EOF

NFAIL=0
run_stage () {
  local lab="$1"; shift
  echo; echo "########## $lab"
  "$@"
  local rc=$?
  echo "=== $lab rc=$rc"
  [ $rc -eq 0 ] || NFAIL=$((NFAIL + 1))
}

run_stage "tripole vs FESOM2 vs NEMO (month 1)" \
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $R/xgrid_trp_d30/snapshot_final.npz \
    --mpas "$FS" --label-mpas FESOM2 \
    --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --out-dir $R/xgrid_trp_fesom_d30

run_stage "MPAS vs FESOM2 vs NEMO (month 1)" \
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole $R/xgrid_mpas_d30/snapshot_final.npz --label-tripole MPAS \
    --mpas "$FS" --label-mpas FESOM2 \
    --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
    --out-dir $R/xgrid_mpas_fesom_d30

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
