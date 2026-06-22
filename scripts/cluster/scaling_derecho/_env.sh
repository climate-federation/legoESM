# Shared environment for NCAR Derecho scaling jobs (sourced by PBS scripts).
# ---------------------------------------------------------------------------
# EDIT the three marked values for your account/paths before first submit.
# Everything here can also be overridden from the qsub environment, e.g.
#   qsub -v LEGOESM_CONDA_ENV=my-jax-env scaling_derecho/gpu_scaling.pbs
# ---------------------------------------------------------------------------

# --- (1) Project allocation (matches the PBS -A directive in the job scripts) -
export PBS_ACCOUNT="${PBS_ACCOUNT:-P08010000}"

# --- (2) Repo location on GLADE -- EDIT to where you cloned legoESM ----------
REPO="${LEGOESM_REPO:-/glade/work/$USER/legoESM}"
export REPO

# NCAR normally exports $SCRATCH; default it defensively so the job scripts'
# `set -u` can't abort on an unset $SCRATCH (used below + for OUTDIR).
export SCRATCH="${SCRATCH:-/glade/derecho/scratch/$USER}"

# --- (3) Conda env that has JAX with CUDA support -- EDIT to your env name ---
CONDA_ENV="${LEGOESM_CONDA_ENV:-legoesm-gpu}"

# --- Federation PYTHONPATH (belt-and-braces; an editable `pip install -e .`
#     in the conda env makes this redundant but harmless) -------------------
PP="$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"

# --- Modules + conda activation ---------------------------------------------
module load conda 2>/dev/null || true
module load cuda  2>/dev/null || true
conda activate "$CONDA_ENV"
PY="$(command -v python)"
export PY

# --- JAX / runtime knobs ----------------------------------------------------
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export MPI4JAX_NO_WARN_JAX_VERSION=1
export MPLBACKEND="${MPLBACKEND:-Agg}"          # headless plotting
# Persistent JIT cache reuses compiles across runs; set empty to force a cold
# compile (true compile_time_s). Lives on SCRATCH so it survives between jobs.
export LEGOESM_JIT_CACHE_DIR="${LEGOESM_JIT_CACHE_DIR:-$SCRATCH/legoesm_jit_cache}"

# Scratch tmp (mirrors the user's reference Casper job script).
export TMPDIR="${TMPDIR:-$SCRATCH/temp}"
mkdir -p "$TMPDIR"
