#!/bin/bash
#SBATCH --job-name=fesom_b23_5d
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesomb23_%j.out
#
# FIRST FORCED FESOM ARM (stage B2+B3): CORE-II corrected forcing through the
# SHARED legoESM bulk, injected at the SurfaceFluxes seam. dm2dc + RGB chl ON
# (the harmonized physics); P-E ON via the freshwater channel. NO ice, NO SSS
# restore, NO runoff yet (stage B4) — polar bands are known-wrong by design.
# PRE-REGISTERED (plumbing gates, NOT a faithfulness score):
#  (a) 5 days complete, finite state; |u|max < 2 m/s;
#  (b) day-5 SST vs GATEWAY rec 1: rmse < 1.5 C in the TROPICS band (23S-23N,
#      away from the missing ice/restore) — REFUTES a sign/scale error in the
#      injected fluxes (a flipped stress or doubled SW blows past this in 5d);
#  (c) SSS global drift < 0.2 PSU (no restore, P-E only).
set -uo pipefail
PIN=${1:?usage: sbatch _fesom_b23_forced_5d.sbatch <pinned-sha>}
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT || exit 1
HEAD=$(git rev-parse HEAD)
[ "$HEAD" = "$PIN" ] || { echo "FATAL: HEAD $HEAD != pinned $PIN"; exit 1; }
echo "HEAD: $(git log --oneline -1)"
echo "fesom_jax: $(git -C /burg-archive/glab/users/pg2328/fesom_jax_repo log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:/burg-archive/glab/users/pg2328/fesom_jax_repo:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
MI=$ROOT/data/fesom2_core2/mesh_ic/core2_mesh_ic
JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py --grid fesom \
  --fesom-mesh-dir "$MI/mesh_core2" --fesom-ic-dir "$MI/ic_core2" \
  --dm2dc --sw-rgb-chl \
  --dt 1800 --years 0.0137 --snapshot-every-days 1 \
  --output $ROOT/results/omip_nemo/fesom_b23_forced5d 2>&1 | tail -12
echo "=== run rc=$? ==="
export JAX_PLATFORMS=cpu
GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
SNAP=$ROOT/results/omip_nemo/fesom_b23_forced5d/snapshot_day0005.npz
[ -f "$SNAP" ] || { echo "FATAL: day-5 snapshot missing"; exit 1; }
echo "=== day 5 vs GATEWAY rec 1 (plumbing gate) ==="
$PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
  --nemo-gridt "$GW_T" --nemo-time-idx 1 --freeze-clamp-C -1.8 \
  --grid-label fesom_b23_d5 \
  --output-dir $ROOT/results/omip_nemo/compare_fesom_b23_d5 2>&1 | tail -30
echo DONE
