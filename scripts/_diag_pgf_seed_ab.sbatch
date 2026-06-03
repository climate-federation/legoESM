#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=pgf_seed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/pgfab_%j.out

# Is the spurious equatorial-bottom meridional PGF seed (KE_PGF_v ~8.6e-3 m/s^2
# @ lat4.4,lon123.5,k19 = Indonesian seas, deepest level) a PGF-discretisation
# artifact? A/B the adcroft PGF (tripole default) vs smc03 (density-Jacobian,
# the documented partial-cell-over-steep-topo mitigation) on the SAME tripole
# grid + WOA IC. If smc03's KE_PGF_v << adcroft's -> discretisation artifact
# (smc03 is the fidelity fix). If ~equal -> a real steep-topo gradient needing
# bathymetry smoothing.
set +e
cd /burg-archive/glab/users/pg2328/legoESM
export PYTHONPATH=/burg-archive/glab/users/pg2328/legoESM/src:$PYTHONPATH
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
echo "=== HOST ==="; hostname; date; nvidia-smi --query-gpu=name --format=csv,noheader

echo; echo "### ARM adcroft (tripole default)"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/_diag_omip_per_term_tendency.py \
  --grid tripole --steps 6 --dt 600 --top-n 5 --pgf-scheme adcroft 2>&1
echo "[adcroft_rc]=$?"

echo; echo "### ARM smc03 (density-Jacobian partial-cell PGF)"
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY scripts/_diag_omip_per_term_tendency.py \
  --grid tripole --steps 6 --dt 600 --top-n 5 --pgf-scheme smc03 2>&1
echo "[smc03_rc]=$?"

echo; echo "=== DONE ==="; date
