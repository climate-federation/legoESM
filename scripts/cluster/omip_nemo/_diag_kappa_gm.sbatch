#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=kappa_gm_diag
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/kappa_gm_%j.out
set +e
ROOT=/burg/glab/users/pg2328/legoESM; cd $ROOT
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
SNAP=$ROOT/results/omip_nemo/nemolev_trpsbc_ndir_mxl3_prog/snapshot_day0030.npz
echo "=== kappa_GM: Treguier (NEMO ldf_eiv) vs Visbeck (OMIP default) ==="
$PY scripts/tmp/_diag_kappa_gm_treguier_vs_visbeck.py "$SNAP" 1800 2>&1 \
  | grep -vE "FutureWarning|warnings.warn|XLA|cuda|CUDA|Plugin|tensorflow"
echo "=== DONE ==="
