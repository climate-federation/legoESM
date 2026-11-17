# GPU-direct MPI stack for the legoESM-perf .venv-gdr: NVIDIA HPC-X 2.20
# (OpenMPI 4.1.7, UCX, CUDA-aware, CUDA 12.x) from nvhpc 25.11 comm_libs/12.9.
HPCX_HOME=/sw/spack-levante/nvhpc-25.11-3is7tk/Linux_x86_64/25.11/comm_libs/12.9/hpcx/hpcx-2.20
export HPCX_HOME
export PATH="$HPCX_HOME/ompi/bin:$HPCX_HOME/ucx/bin:$PATH"
export LD_LIBRARY_PATH="$HPCX_HOME/ompi/lib:$HPCX_HOME/ucx/lib:$HPCX_HOME/ucx/lib/ucx:$HPCX_HOME/hcoll/lib:$HPCX_HOME/ucc/lib:$HPCX_HOME/sharp/lib:/sw/spack-levante/gcc-13.4.0-c4mgfk/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OPAL_PREFIX="$HPCX_HOME/ompi"
export MPICC="$HPCX_HOME/ompi/bin/mpicc"
# The HPC-X wrappers default to nvc; build the Python extensions with gcc
# (the same compiler family as the rest of the venv).
export OMPI_CC=gcc OMPI_CXX=g++
