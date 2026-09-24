#!/bin/bash
#SBATCH --job-name=sweep0923_g2
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/sweep0923_g2.%j.out
# Round 2 of the issue-sweep gates.  Round 1 selected with `pytest -k` over the
# whole tree, which dies during COLLECTION on two unrelated DINO test modules
# (they exec a validation probe at import and it raises), so two gates returned
# "19 skipped, 2 errors" and measured nothing.  Named paths only here.
#   #1729 -- NEMO DINO initial-state transcription, bit-exact gate
#   #1712 -- subgrid-orography scale decomposition
#   #1321 -- MPAS interactive land under MPI, single rank and 2 ranks
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/_sweep0923_wt
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
[ -x "$PY" ] || { echo "FATAL: interpreter $PY missing"; exit 2; }
cd "$WT"
export PYTHONPATH="$WT/src:$WT/packages/atmosphere:$WT/packages/core:$WT/packages/coupler:$WT/packages/ice:$WT/packages/land:$WT/packages/ml:$WT/packages/ocean:$WT/packages/tools:$WT"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
echo "HEAD: $(git -C "$WT" log --oneline -1)"

echo "########## GATE #1729 -- NEMO DINO from-rest initial state ##########"
"$PY" -m pytest -q tests/ocean/fidelity/test_nemo_dino_fromrest_gate.py 2>&1 | tail -15

echo "########## GATE #1712 -- subgrid orography scale decomposition ##########"
"$PY" -m pytest -q tests/unit/test_prep_subgrid_orography.py \
    tests/unit/test_sso_residual_variance.py \
    tests/unit/test_amip_production_subgrid_orography.py 2>&1 | tail -14

echo "########## GATE #1321 -- can the distributed suite run here AT ALL? ##########"
# Round 1 reported this gate as "skipped", which reads like a pass and is not
# one.  Say out loud what is missing instead.
"$PY" -c "import mpi4jax" 2>&1 | tail -2
echo "mpi4py: $("$PY" -c 'import mpi4py; print(mpi4py.__version__)' 2>&1 | tail -1)"
command -v mpirun || echo "mpirun: ABSENT from PATH (needs an openmpi module)"
module load openmpi/gcc/64 2>/dev/null && command -v mpirun && echo "mpirun appeared after module load"

echo "########## GATE #1321 item 2 -- qv del2 filter ##########"
"$PY" -m pytest -q tests/unit/test_mpas_qv_smoothing.py 2>&1 | tail -8

echo "########## the collection defect that broke round 1 ##########"
"$PY" -m pytest -q --collect-only tests/ocean/unit/test_dino_1455_channel_rescore.py 2>&1 | tail -12
echo DONE
