#!/usr/bin/env bash
# Submit a global CRU-JRA P-model LMIP run to a Ginsburg GPU, in chained blocks.
#
# Splits [--year-start, --year-end] into blocks of --block-years.  Each block is
# its own experiment directory (init_experiment.py: resolved config + provenance
# tag) and its own Slurm job; block N+1 waits for block N (afterok) and warm
# starts from the restart block N writes.
#
#   # one-year test
#   scripts/run/submit_lmip_ginsburg.sh --arm least_cost --year-start 2000 --year-end 2000
#   # full period, 5-year blocks
#   scripts/run/submit_lmip_ginsburg.sh --arm least_cost --year-start 2000 --year-end 2014
#   # another partition / skip full nodes (repeatable, passed to sbatch)
#   ... --partition short --sbatch-arg=--exclude=g097,g098
#   # extend an existing run
#   scripts/run/submit_lmip_ginsburg.sh --arm least_cost --year-start 2001 --year-end 2014 \
#       --restart-from <root>/least_cost/2000-2000/restart_2001_d000h00.npz
#   # warm start from the published 1985 soil state (no P-model acclimation state
#   # in it: the first block resets that explicitly), start only after job 123 ends well
#   ... --restart-from data/lmip_soil_ic/restart_1985_d000h00.npz --pmodel-cold-restart --after 123
#   # any config override, repeatable (same syntax as init_experiment.py -o)
#   ... --override forcing.co2_file=
#
# Arms (tests/land/integration/test_run_lmip_biophys.py):
#   baseline    prescribed capacities + table g1
#   least_cost  P-model capacities + P-model g1, beta(theta) stress
#   phydro      P-model capacities + P-model g1, P-hydro stress
set -euo pipefail

ARM=least_cost
YEAR_START=2000
YEAR_END=2000
BLOCK_YEARS=5
HOURS_PER_YEAR=8          # walltime budget; ~5.4 h/yr measured on an RTX 8000
RESTART_FROM=""
TEMPLATE=biophysics/lmip_pmodel_crujra
ROOT="/burg/glab/users/$USER/lmip_runs/pmodel_crujra"
PYTHON="${LEGOESM_PYTHON:-/burg/glab/users/$USER/.conda/envs/legoesm-gpu/bin/python}"
ACCOUNT=glab
PARTITION=glab1
MAIL_USER="${LEGOESM_MAIL_USER:-}"   # set to get END/FAIL emails
SBATCH_EXTRA=()          # e.g. --sbatch-arg=--partition=short --sbatch-arg=--exclude=g098
MIN_FREE_GB=40           # host RAM one forcing year needs; nodes here are often oversubscribed
OVERRIDES=()             # extra -o key=value for init_experiment.py
PMODEL_COLD_RESTART=0    # first block only: --pmodel-cold-restart when restarting from a file
AFTER=""                 # first block waits for this job (afterok)
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arm)            ARM="$2"; shift ;;
        --year-start)     YEAR_START="$2"; shift ;;
        --year-end)       YEAR_END="$2"; shift ;;
        --block-years)    BLOCK_YEARS="$2"; shift ;;
        --hours-per-year) HOURS_PER_YEAR="$2"; shift ;;
        --restart-from)   RESTART_FROM="$2"; shift ;;
        --template)       TEMPLATE="$2"; shift ;;
        --root)           ROOT="$2"; shift ;;
        --account)        ACCOUNT="$2"; shift ;;
        --partition)      PARTITION="$2"; shift ;;
        --mail-user)      MAIL_USER="$2"; shift ;;
        --sbatch-arg=*)   SBATCH_EXTRA+=("${1#--sbatch-arg=}") ;;
        --min-free-gb)    MIN_FREE_GB="$2"; shift ;;
        --override)       OVERRIDES+=(-o "$2"); shift ;;
        --pmodel-cold-restart) PMODEL_COLD_RESTART=1 ;;
        --after)          AFTER="$2"; shift ;;
        --dry-run)        DRY_RUN=1 ;;
        -h|--help)        sed -n '2,28p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

case "$ARM" in
    baseline)   ARM_OVR=(canopy_capacity_scheme=prescribed canopy_g1_source=table transpiration_stress=beta_theta) ;;
    least_cost) ARM_OVR=(canopy_capacity_scheme=p_model canopy_g1_source=p_model transpiration_stress=beta_theta) ;;
    phydro)     ARM_OVR=(canopy_capacity_scheme=p_model canopy_g1_source=p_model transpiration_stress=phydro) ;;
    *) echo "unknown --arm '$ARM' (baseline | least_cost | phydro)" >&2; exit 2 ;;
esac
(( YEAR_END >= YEAR_START )) || { echo "--year-end < --year-start" >&2; exit 2; }
(( BLOCK_YEARS >= 1 )) || { echo "--block-years must be >= 1" >&2; exit 2; }
[[ -x "$PYTHON" ]] || { echo "python not found: $PYTHON (set LEGOESM_PYTHON)" >&2; exit 2; }
if [[ -n "$RESTART_FROM" ]]; then
    RESTART_FROM="$(realpath "$RESTART_FROM")"
    [[ -f "$RESTART_FROM" ]] || { echo "restart not found: $RESTART_FROM" >&2; exit 2; }
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

prev_job=""
restart="$RESTART_FROM"
y0=$YEAR_START
while (( y0 <= YEAR_END )); do
    y1=$(( y0 + BLOCK_YEARS - 1 )); (( y1 > YEAR_END )) && y1=$YEAR_END
    nyears=$(( y1 - y0 + 1 ))
    name="${ARM}_${y0}-${y1}"
    exp="$ROOT/$ARM/${y0}-${y1}"
    hours=$(( nyears * HOURS_PER_YEAR + 1 ))
    walltime="$(( hours / 24 ))-$(printf '%02d' $(( hours % 24 ))):00:00"

    ovr=(-o forcing.year_start="$y0" -o forcing.year_end="$y1"
         -o time.n_steps=$(( nyears * 8760 )))
    for o in "${ARM_OVR[@]}"; do ovr+=(-o "physics.$o"); done
    ovr+=("${OVERRIDES[@]}")
    # The explicit acclimation reset applies only to the block that reads the
    # external file; later blocks restart from this chain's own files.
    cold=""
    if (( PMODEL_COLD_RESTART )) && [[ -n "$restart" && "$restart" == "$RESTART_FROM" ]]; then
        cold=" --pmodel-cold-restart"
    fi
    dep="${prev_job:-$AFTER}"

    if (( DRY_RUN )); then
        echo "[dry-run] $name -> $exp | walltime $walltime | restart: ${restart:-cold start}${cold} | after: ${dep:-none}"
        JAX_PLATFORMS=cpu "$PYTHON" scripts/run/init_experiment.py "$TEMPLATE" --name "$name" \
            --output-dir "$exp" "${ovr[@]}" --dry-run > /dev/null
    else
        JAX_PLATFORMS=cpu "$PYTHON" scripts/run/init_experiment.py "$TEMPLATE" --name "$name" \
            --output-dir "$exp" "${ovr[@]}" > /dev/null
        cat > "$exp/job.sbatch" <<EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
${MAIL_USER:+#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=$MAIL_USER
}#SBATCH --job-name=lmip_$name
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=$walltime
#SBATCH --output=$exp/slurm-%j.out
set -euo pipefail
cd "$REPO"
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR="$REPO/.mplconfig"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Slurm does not enforce --mem here, so a node can be full despite the request.
free_gb=\$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo)
echo "node \$(hostname -s): \${free_gb} GB RAM available"
if (( free_gb < $MIN_FREE_GB )); then
    echo "ERROR: need $MIN_FREE_GB GB free RAM; resubmit with --sbatch-arg=--exclude=\$(hostname -s)" >&2
    exit 3
fi
"$PYTHON" scripts/run/run_lmip_biophys.py --config "$exp/config.yaml" \\
    --output-dir "$exp"${restart:+ --restart-from "$restart"}$cold
EOF
        job=$(sbatch --parsable -A "$ACCOUNT" "${SBATCH_EXTRA[@]}" ${dep:+--dependency=afterok:$dep} "$exp/job.sbatch")
        echo "$name -> job $job | $exp | walltime $walltime | restart: ${restart:-cold start}${cold} | after: ${dep:-none}"
        prev_job=$job
    fi
    restart="$exp/restart_$(printf '%04d' $(( y1 + 1 )))_d000h00.npz"
    y0=$(( y1 + 1 ))
done
