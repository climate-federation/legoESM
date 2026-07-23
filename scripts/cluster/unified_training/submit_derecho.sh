#!/usr/bin/env bash
# Ready-to-paste Derecho submitter for the unified WB+AIMIP training driver.
#
# Submits one self-chaining PBS job per selected AIMIP variant (and/or a WB
# campaign job) via scripts/cluster/unified_training/train_derecho.pbs. Each
# AIMIP job self-resumes across the 12 h walltime until params.eqx exists
# (CHAIN_MAX links); WB is single-link (the WB trainer has no resume yet).
#
# USAGE (from the repo root on a Derecho login node):
#   # AIMIP, all three headline variants:
#   ALLOC=P08010000 ./scripts/cluster/unified_training/submit_derecho.sh
#
#   # pick variants:
#   ALLOC=P08010000 VARIANTS="classical sfno_full" ./scripts/cluster/unified_training/submit_derecho.sh
#
#   # WeatherBench campaign instead (modes are '+'-separated):
#   ALLOC=P08010000 CAMPAIGN=wb WB_SUITE=config/wb/campaign/spectral_t106.yaml \
#     WB_MODES="physics+neural_gcm+sfno" ./scripts/cluster/unified_training/submit_derecho.sh
#
# PREREQUISITES:
#   - Edit scripts/cluster/scaling_derecho/_env.sh (or export LEGOESM_REPO /
#     CONDA_ENV) so REPO + the conda env resolve on Derecho.
#   - Set ALLOC to your project allocation (falls back to the .pbs default).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PBS="$REPO_ROOT/scripts/cluster/unified_training/train_derecho.pbs"
CAMPAIGN="${CAMPAIGN:-aimip}"
ALLOC="${ALLOC:-}"                 # empty -> use the #PBS -A default in the .pbs
CHAIN_MAX="${CHAIN_MAX:-24}"

_alloc_flag=()
[ -n "$ALLOC" ] && _alloc_flag=(-A "$ALLOC")

if [ ! -f "$PBS" ]; then
  echo "error: launcher not found: $PBS" >&2
  exit 1
fi

case "$CAMPAIGN" in
  aimip)
    # variant -> its scale suite (config/aimip/scale/suite_<variant>.yaml).
    VARIANTS="${VARIANTS:-classical column_nn sfno_full}"
    for v in $VARIANTS; do
      suite="config/aimip/scale/suite_${v}.yaml"
      if [ ! -f "$REPO_ROOT/$suite" ]; then
        echo "SKIP $v: no suite $suite" >&2
        continue
      fi
      echo "[submit] aimip variant=$v suite=$suite"
      qsub "${_alloc_flag[@]}" \
        -v LEGOESM_REPO="$REPO_ROOT",CAMPAIGN=aimip,SUITE="$suite",VARIANT="$v",CHAIN_MAX="$CHAIN_MAX" \
        "$PBS"
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
    qsub "${_alloc_flag[@]}" \
      -v LEGOESM_REPO="$REPO_ROOT",CAMPAIGN=wb,SUITE="$WB_SUITE",VARIANT="$WB_MODES",CHAIN_MAX=0 \
      "$PBS"
    ;;
  *)
    echo "error: CAMPAIGN must be aimip|wb (got '$CAMPAIGN')" >&2
    exit 1
    ;;
esac

echo "[submit] done. Track with: qstat -u \$USER   |   logs under the suite output_dir."
