#!/bin/bash
#SBATCH --job-name=fesom30
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=8:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesom30_%j.out
#
# FESOM2 (fesom-jax) CORE2 realistic arm, 30 days -- the third grid of the
# cross-grid comparison.  Unzips the forcing on first use (sha-verified),
# then runs scripts/run/run_fesom_core2.py.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
FJX=/burg-archive/glab/users/pg2328/fesom_jax_repo
D=$REPO/data/fesom2_core2
cd $REPO || exit 1
echo "=== HEAD $(git -C $REPO rev-parse HEAD)"
echo "=== fesom_jax $(git -C $FJX rev-parse --short HEAD 2>/dev/null || echo untracked)"

# --- forcing: verify + unzip once -----------------------------------------
if [ ! -d "$D/forcing" ]; then
  [ -s "$D/core2_forcing_1958.zip" ] || { echo "FATAL: forcing zip missing/incomplete"; exit 1; }
  echo "[sha256] core2_forcing_1958.zip"
  want=501c1be7673f7c2509235c2c34f02d431d3773289f712f3dd6c98763831126b0
  got=$(sha256sum "$D/core2_forcing_1958.zip" | cut -d' ' -f1)
  [ "$got" = "$want" ] || { echo "FATAL: sha mismatch $got"; exit 1; }
  mkdir -p "$D/forcing"
  unzip -q -o "$D/core2_forcing_1958.zip" -d "$D/forcing"
fi
echo "[forcing] tree:"
find "$D/forcing" -maxdepth 3 -type d | head; find "$D/forcing" -type f | head -8

# JRA dir = the directory holding the per-variable .1958.nc files.
JRA=$(dirname "$(find "$D/forcing" -name "tas*.nc" | head -1)")
[ -n "$JRA" ] || { echo "FATAL: no tas*.nc in forcing tree"; exit 1; }
MI=$D/mesh_ic/core2_mesh_ic
export FESOM_JRA_DIR="$JRA"
export FESOM_SSS_PATH="$MI/PHC2_salx.nc"
export FESOM_RUNOFF_PATH="$MI/CORE2_runoff.nc"
export FESOM_CHL_PATH="$MI/Sweeney_2005.nc"
export FESOM_PHC_PATH="$MI/phc3.0_winter.nc"
for f in "$FESOM_SSS_PATH" "$FESOM_RUNOFF_PATH" "$FESOM_CHL_PATH"; do
  [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

export PYTHONPATH="$FJX:$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_ENABLE_X64=0     # fesom-jax published hindcast runs f32; keep ITS convention
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader || true

$PY -u scripts/run/run_fesom_core2.py \
  --mesh-dir "$MI/mesh_core2" \
  --ic-dir "$MI/ic_core2" \
  --days 30 --dt 1800 --year 1958 \
  --snapshot-every-days 10 \
  --output $REPO/results/omip_nemo/fesom_core2_d30
rc=$?
echo "=== rc=$rc ==="
exit $rc
