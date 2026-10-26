#!/bin/bash
#SBATCH --job-name=fesom_prep75
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:40:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/fesom_prep75_%j.out
# fesom_jax mesh bundle for the CORE2 mesh on NEMO's 75 levels (level files
# from the ported FESOM2 find_levels; self-check 0 mismatches on the shipped ladder).
set -uo pipefail
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export PYTHONPATH=/burg-archive/glab/users/pg2328/fesom_jax_repo
D=/burg-archive/glab/users/pg2328/legoESM/data/fesom2_core2
$PY /burg-archive/glab/users/pg2328/fesom_jax_repo/scripts/prepare_mesh.py $D/mesh_nemo75_raw $D/mesh_nemo75 2>&1 | tail -20
echo "rc=$?"
$PY - <<'PYEOF'
from fesom_jax.mesh import load_mesh
m = load_mesh("/burg-archive/glab/users/pg2328/legoESM/data/fesom2_core2/mesh_nemo75")
import numpy as np
print("loaded:", m.nod2D, m.elem2D, m.nl, "zbar[:5]", np.asarray(m.zbar)[:5], "min nlevels_nod2D", int(np.asarray(m.nlevels_nod2D).min()))
PYEOF
echo DONE
