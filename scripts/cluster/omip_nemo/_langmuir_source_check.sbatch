#!/bin/bash
#SBATCH --job-name=lcsrc
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/lcsrc_%j.out
# How big is the Langmuir source our production function produces on NEMO's
# own hour-12 column, against the extra dissipation NEMO's LC-run TKE carries
# over its noLC twin? PRE-REGISTERED: src/d(eps) ~ 1 at 8-15 m -> the LC
# source alone explains NEMO's subsurface TKE and our weaker response is a
# feedback (length/transport) question; src/d(eps) << 1 -> NEMO's LC run gets
# TKE from a channel the formula I read does not contain (re-read zdftke).
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
git -C $ROOT rev-parse --short HEAD | grep -q 0d4c394b6 || { echo "worktree HEAD moved"; exit 2; }
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
B=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00
for H in 12 16; do
  echo "##### hour $H"
  $PY -u scripts/validate/ocean_fidelity/equatorial_langmuir_source_on_nemo_state.py --restart-glob-lc "$B/RUN_HOURLY/ORCA1_000000${H}_restart_oce_*.nc" --restart-glob-nolc "$B/RUN_HOURLY_NOLC/ORCA1_000000${H}_restart_oce_*.nc" 2>&1 | grep -vE "cuInit|jax_plugins|plugin_module|_check_cuda|cuda_versions|Traceback|^\s*\^|Jax plugin"
done
echo "=== card exit: $? ==="
