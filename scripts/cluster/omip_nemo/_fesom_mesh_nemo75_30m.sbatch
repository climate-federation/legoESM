#!/bin/bash
#SBATCH --job-name=fesom_m30
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesom_m30_%j.out
# FESOM on NEMO's 75 levels with a 30 m minimum column (user decision 8,
# 2026-09-06: NEMO's floor, not FESOM's 5th interface = 4.5 m on this ladder).
# Level rule self-check on the shipped 47-level files must be exact first.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_wt; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
D=/burg-archive/glab/users/pg2328/legoESM/data/fesom2_core2
RAW=$D/mesh_ic/core2_mesh_ic/mesh_core2_raw
$PY scripts/validate/ocean_fidelity/fesom_level_rule.py --raw $RAW \
  --nemo-domain-cfg /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/domain_cfg.nc \
  --out $D/mesh_nemo75_30m_raw --min-depth-m 30
echo "rule rc=$?"
export PYTHONPATH=/burg-archive/glab/users/pg2328/fesom_jax_repo
$PY /burg-archive/glab/users/pg2328/fesom_jax_repo/scripts/prepare_mesh.py $D/mesh_nemo75_30m_raw $D/mesh_nemo75_30m 2>&1 | tail -8
echo "prepare rc=$?"
$PY - <<'PYEOF'
from fesom_jax.mesh import load_mesh
import numpy as np
m = load_mesh("/burg-archive/glab/users/pg2328/legoESM/data/fesom2_core2/mesh_nemo75_30m")
print("loaded:", m.nod2D, m.elem2D, m.nl, "zbar[:16]", np.round(np.asarray(m.zbar)[:16], 2), "min nlevels_nod2D", int(np.asarray(m.nlevels_nod2D).min()))
PYEOF
echo DONE
