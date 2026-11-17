#!/usr/bin/env bash
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=00:25:00
#SBATCH --job-name=gdr_probe
#SBATCH --output=/scratch/b/b384341/amip_gpu_profile/gdr/dbg/%x-%j.out
# Launch-variant probe for the GPU-direct halo benchmark (phase 0 debug):
# a 4-rank ring sendrecv of an 8 MB float64 GPU array through mpi4jax under
# each MPI stack / launcher combination; prints whether values arrive and the
# per-call time.  Diagnostic only.
set -u
REPO=/home/b/b384341/projects/legoESM-perf
P=/scratch/b/b384341/amip_gpu_profile/gdr/dbg/mpi_probe.py
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false MPI4JAX_NO_WARN_JAX_VERSION=1
unset SLURM_CPUS_PER_TASK
S=( srun --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=map_gpu:0,1,2,3 --cpus-per-task=8 )
try() { local name=$1; shift; echo "--- $name: $*"; PROBE_NAME=$name timeout 240 "$@" 2>&1 | grep -E "PROBE|rror|cannot|failed|abort" | grep -vE "Warning" | head -8; echo "    rc=${PIPESTATUS[0]}"; }
( export LD_LIBRARY_PATH=/sw/spack-levante/openmpi-4.1.2-mnmady/lib:/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64
  try A_default "${S[@]}" /usr/bin/env MPI4JAX_USE_CUDA_MPI=0 PROBE_NAME=A_default $REPO/.venv/bin/python $P )
source $REPO/scripts/cluster/levante/hpcx_gdr_env.sh
echo "=== ucx_info"; ucx_info -v | head -3; ucx_info -d | grep -E "^#\s+(Transport|Device)" | sort | uniq | head -30
for mpiflag in "" "--mpi=pmix" "--mpi=pmix_v3"; do
  for pml in "" "ucx"; do
    name="B${mpiflag:+_${mpiflag#--mpi=}}${pml:+_pml$pml}"
    try "$name" "${S[@]}" $mpiflag /usr/bin/env MPI4JAX_USE_CUDA_MPI=0 PROBE_NAME=$name ${pml:+OMPI_MCA_pml=$pml} UCX_LOG_LEVEL=warn $REPO/.venv-gdr/bin/python $P
  done
done
for mpiflag in "" "--mpi=pmix" "--mpi=pmix_v3"; do
  name="C${mpiflag:+_${mpiflag#--mpi=}}_pmlucx"
  try "$name" "${S[@]}" $mpiflag /usr/bin/env MPI4JAX_USE_CUDA_MPI=1 PROBE_NAME=$name OMPI_MCA_pml=ucx UCX_LOG_LEVEL=warn $REPO/.venv-gdr/bin/python $P
done
cat > /scratch/b/b384341/amip_gpu_profile/gdr/dbg/gpu_wrap.sh <<'W'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK}
exec "$@"
W
chmod +x /scratch/b/b384341/amip_gpu_profile/gdr/dbg/gpu_wrap.sh
try C_ucxtls_prev "${S[@]}" --mpi=pmix /usr/bin/env MPI4JAX_USE_CUDA_MPI=1 PROBE_NAME=C_ucxtls_prev OMPI_MCA_pml=ucx OMPI_MCA_osc=ucx UCX_TLS=rc,cuda_copy,cuda_ipc,gdr_copy,sm,self UCX_MEMTYPE_CACHE=n UCX_RNDV_SCHEME=put_zcopy UCX_LOG_LEVEL=warn $REPO/.venv-gdr/bin/python $P
try C_mpirun mpirun -np 4 --bind-to none -x MPI4JAX_USE_CUDA_MPI=1 -x PROBE_NAME=C_mpirun -x JAX_PLATFORMS -x XLA_PYTHON_CLIENT_PREALLOCATE -x MPI4JAX_NO_WARN_JAX_VERSION -x LD_LIBRARY_PATH -x PATH /scratch/b/b384341/amip_gpu_profile/gdr/dbg/gpu_wrap.sh $REPO/.venv-gdr/bin/python $P
