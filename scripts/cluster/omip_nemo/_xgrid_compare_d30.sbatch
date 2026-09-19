#!/bin/bash
#SBATCH --job-name=xgcmp
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/xgcmp_%j.out
#
# CROSS-GRID AGREEMENT, tripole vs MPAS, day 30, on the arms ALREADY on disk.
# Both were run as a MATCHED pair (_xgrid_trp.sbatch / _xgrid_mpas.sbatch), so
# this is the legitimate grid comparison. Both predate the signed-TEOS-10 N2
# card change, so this is the BASELINE agreement number, not a statement about
# the current card -- do not quote it as one.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
R=$WT/results/omip_nemo
echo "HEAD: $(git -C $WT log --oneline -1)"
echo "--- manifests (confirm the pair is matched on days/dt/IC):"
for a in xgrid_trp_d30 xgrid_mpas_d30; do
  /burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -c "
import json,sys
m=json.load(open('$R/$a/run_manifest.json'))
cl=m.get('command_line','')
print('$a:', ' '.join(w for w in cl.split() if w.startswith(('--years','--dt','--woa-init','--tracer'))) or cl[:120])
"
done
echo "--- comparison:"
/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python -u \
  scripts/validate/ocean_fidelity/compare_grids_tripole_mpas.py \
  --tripole $R/xgrid_trp_d30/snapshot_final.npz \
  --mpas    $R/xgrid_mpas_d30/snapshot_final.npz \
  --out $WT/omip_nemo/xgrid_compare_d30.png 2>&1
echo "=== rc=$?"
