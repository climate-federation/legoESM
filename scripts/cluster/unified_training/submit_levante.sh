#!/usr/bin/env bash
# Ready-to-paste DKRZ Levante submitter for the unified WB+AIMIP training driver.
#
# Submits one self-chaining SLURM job per selected AIMIP variant (and/or a WB
# campaign job) via scripts/cluster/unified_training/train_levante.slurm. Each
# AIMIP job self-resumes across the 12 h walltime until params.eqx exists
# (CHAIN_MAX links); WB is single-link (the WB trainer has no resume yet).
#
# USAGE (from the repo root on a Levante login node):
#   # AIMIP, all three headline variants (ALLOC is REQUIRED on Levante):
#   ALLOC=bd1083 ./scripts/cluster/unified_training/submit_levante.sh
#
#   # pick variants:
#   ALLOC=bd1083 VARIANTS="classical sfno_full" ./scripts/cluster/unified_training/submit_levante.sh
#
#   # WeatherBench campaign instead (modes are '+'-separated, never commas):
#   ALLOC=bd1083 CAMPAIGN=wb WB_SUITE=config/wb/campaign/spectral_t106.yaml \
#     WB_MODES="physics+neural_gcm+sfno" ./scripts/cluster/unified_training/submit_levante.sh
#
# PREREQUISITES:
#   - Edit scripts/cluster/scaling_levante/_env.sh (or export LEGOESM_REPO /
#     CONDA_ENV) so REPO + the conda env resolve on Levante.
#   - ALLOC must be your DKRZ project account (Levante rejects jobs without -A).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SLURM="$REPO_ROOT/scripts/cluster/unified_training/train_levante.slurm"
CAMPAIGN="${CAMPAIGN:-aimip}"
ALLOC="${ALLOC:-}"
CHAIN_MAX="${CHAIN_MAX:-24}"

if [ ! -f "$SLURM" ]; then
  echo "error: launcher not found: $SLURM" >&2
  exit 1
fi
if [ -z "$ALLOC" ]; then
  echo "error: set ALLOC=<your DKRZ project account> (Levante requires -A)" >&2
  exit 1
fi

# SLURM carries per-job vars via --export; the launcher's resubmit_self exports
# them + re-invokes with --export=ALL so CHAIN advances (see train_levante.slurm).
case "$CAMPAIGN" in
  aimip)
    VARIANTS="${VARIANTS:-classical column_nn sfno_full}"
    for v in $VARIANTS; do
      suite="config/aimip/scale/suite_${v}.yaml"
      if [ ! -f "$REPO_ROOT/$suite" ]; then
        echo "SKIP $v: no suite $suite" >&2
        continue
      fi
      echo "[submit] aimip variant=$v suite=$suite"
      sbatch -A "$ALLOC" \
        --export="ALL,CAMPAIGN=aimip,SUITE=$suite,VARIANT=$v,CHAIN_MAX=$CHAIN_MAX" \
        "$SLURM"
    done
    ;;
  wb)
    WB_SUITE="${WB_SUITE:-config/wb/campaign/spectral_t106.yaml}"
    WB_MODES="${WB_MODES:-physics+neural_gcm+sfno}"   # '+'-separated (not ',')
    if [ ! -f "$REPO_ROOT/$WB_SUITE" ]; then
      echo "error: WB suite not found: $WB_SUITE" >&2
      exit 1
    fi
    echo "[submit] wb suite=$WB_SUITE modes=$WB_MODES (single link)"
    sbatch -A "$ALLOC" \
      --export="ALL,CAMPAIGN=wb,SUITE=$WB_SUITE,VARIANT=$WB_MODES,CHAIN_MAX=0" \
      "$SLURM"
    ;;
  *)
    echo "error: CAMPAIGN must be aimip|wb (got '$CAMPAIGN')" >&2
    exit 1
    ;;
esac

echo "[submit] done. Track with: squeue -u \$USER   |   logs under the suite output_dir."
