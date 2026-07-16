#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1,short,burst
#SBATCH --job-name=amip_convcmp
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=11:59:00
#SBATCH --requeue
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM_chunk/results/amip/convcmp/slurm_%j.out
# =============================================================================
# C48/L40 AMIP convection-scheme comparison — self-chaining, one job per scheme.
#
# Runs the AUTHORITATIVE production config (config/amip/amip_production.yaml)
# with ONLY the convection scheme + GWD swapped:
#   --convection $CONV                      (sbm control | tiedtke | bechtold | edmf)
#   --gravity-wave-drag hines+mcfarlane     (production non-oro Hines + McFarlane
#                                            orographic, composite)
#   --subgrid-orography-file $SSO_FILE      (per-column h_topo stddev from ETOPO;
#                                            without it McFarlane launches from a
#                                            uniform 500 m mountain — ocean too)
#
# Submit (per scheme; OUTBASE must exist before sbatch opens --output):
#   mkdir -p /burg-archive/glab/users/pg2328/legoESM_chunk/results/amip/convcmp
#   for c in sbm tiedtke bechtold edmf; do
#     sbatch -J amip_cv_$c --export=ALL,CONV=$c \
#       scripts/cluster/amip/amip_convection_campaign.sbatch
#   done
#
# Chain mechanics identical to amip_ginsburg_production_chain.sbatch: each link
# checkpoints + exits inside the wallclock budget, then re-submits itself until
# TARGET_DAYS; JAX compile cache in OUTDIR amortizes the rrtmg JIT across links.
# CODE is a PINNED worktree (branch amip-convection-gwd-campaign) so concurrent
# sessions on the main checkout cannot mutate a multi-day run mid-flight.
# =============================================================================
set -uo pipefail

CODE="${CONVCMP_CODE:-/burg-archive/glab/users/pg2328/legoESM_convcmp}"
# ginsburg.sh reads LEGOESM_REPO for the GPU .venv + staged AMIP inputs (ERA5 IC,
# ETOPO, input4MIPs SST/SIC, forcing deck) — ac5006's checkout hosts both.
export LEGOESM_REPO="${LEGOESM_REPO:-/burg-archive/glab/users/ac5006/legoESM}"

: "${CONV:?set CONV=sbm|tiedtke|bechtold|edmf}"
: "${GWD:=hines+mcfarlane}"
: "${SSO_FILE:=/burg-archive/glab/users/pg2328/legoESM_chunk/data_pg/amip/sso_stdh_2deg.nc}"
: "${OUTBASE:=/burg-archive/glab/users/pg2328/legoESM_chunk/results/amip/convcmp}"
: "${OUTDIR:=$OUTBASE/conv_${CONV}}"
# Publication resolution: OVERRIDE the amip_production.yaml default, which #821
# set to a coarse C12/L20 dt450 SCREEN (not publication grade). C48/L40 dt150 is
# the realistic AMIP config (matches Pierre's tuned production; CFL-safe at C48).
: "${RESOLUTION:=48}"
: "${NLEV:=40}"
: "${DT:=150.0}"
: "${TARGET_DAYS:=90}"
: "${CHECKPOINT_DAYS:=5}"
: "${MAX_WALL:=41400}"          # ~11.5 h; driver checkpoints+exits before 11:59
: "${EXTRA_FLAGS:=}"            # per-scheme tuning knobs on later iterations
mkdir -p "$OUTDIR"

# CLM surfdata for the multilayer land (ac5006's data tree does not carry it;
# ginsburg.sh consumes this env override into --clm-surfdata-path).
export CLM_SURFDATA="${CLM_SURFDATA:-/burg-archive/glab/users/pg2328/legoESM_chunk/data_pg/clm/surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc}"

source "$CODE/config/amip/amip_production.ginsburg.sh"
export PYTHONPATH="$(ls -d "$CODE"/packages/*/ | tr '\n' ':')${PYTHONPATH:-}"
export JAX_COMPILATION_CACHE_DIR="$OUTDIR/jax_cache"; mkdir -p "$JAX_COMPILATION_CACHE_DIR"

[ -f "$SSO_FILE" ] || { echo "[convcmp] missing SSO_FILE=$SSO_FILE — run scripts/data/prep_subgrid_orography.py first"; exit 2; }

# EXTRA_FLAGS is a space-separated flag string of SIMPLE tokens
# (e.g. "--bechtold-cape-threshold 50"). Shell quoting is NOT honored by the
# word-split below, and commas would corrupt the comma-separated --export list
# on chain resubmission — reject both loudly instead of mangling silently.
case "$EXTRA_FLAGS" in (*[\'\",]*)
  echo "[convcmp] EXTRA_FLAGS must be simple space-separated tokens (no quotes/commas), got: $EXTRA_FLAGS"; exit 2;;
esac
read -r -a EXTRA <<< "$EXTRA_FLAGS"
SCHEME_FLAGS=(--convection "$CONV" --gravity-wave-drag "$GWD"
              --subgrid-orography-file "$SSO_FILE"
              --resolution "$RESOLUTION" --nlev "$NLEV" --dt "$DT"
              # The clear-sky 2nd RRTMGP pass (#843) ~doubles the radiation
              # compile + per-step cost and the convection-scheme comparison
              # scores on all-sky precip/albedo/OLR; turn it off for speed.
              # (Override with CLEAR_SKY=--clear-sky-diag for a final run.)
              ${CLEAR_SKY:---no-clear-sky-diag})

_ckpt_day() { local f; f=$(ls -1 "$OUTDIR"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1); [ -n "$f" ] && basename "$f" | sed -E 's/.*checkpoint_day_0*([0-9]+)\.npz/\1/' || echo 0; }
cur_day=$(_ckpt_day); latest=$(ls -1 "$OUTDIR"/checkpoint_day_*.npz 2>/dev/null | sort -V | tail -1)
echo "=== HOST $(hostname) $(date) CODE=$CODE conv=$CONV gwd=$GWD ==="
echo "[convcmp] OUTDIR=$OUTDIR cur_day=$cur_day target=$TARGET_DAYS extra=${EXTRA_FLAGS:-none}"
[ "${cur_day:-0}" -ge "$TARGET_DAYS" ] && { echo "[convcmp] target reached"; exit 0; }
RESTART=(); [ -n "$latest" ] && RESTART=(--restart-from "$latest")

JAX_ENABLE_X64=1 "$PY" -u "$CODE/scripts/run/run_amip.py" \
  --config "$CODE/config/amip/amip_production.yaml" \
  "${AMIP_PATH_FLAGS[@]}" \
  "${SCHEME_FLAGS[@]}" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  --days "$TARGET_DAYS" --checkpoint-days "$CHECKPOINT_DAYS" \
  --max-wallclock-seconds "$MAX_WALL" \
  "${RESTART[@]}" --output "$OUTDIR"
rc=$?; new_day=$(_ckpt_day)
echo "[convcmp] rc=$rc $cur_day -> $new_day $(date)"
[ "$rc" -ne 0 ] && { echo "[convcmp] FAILED rc=$rc — NOT chaining"; exit "$rc"; }
[ "${new_day:-0}" -ge "$TARGET_DAYS" ] && { echo "[convcmp] COMPLETE $new_day/$TARGET_DAYS"; exit 0; }
[ "${new_day:-0}" -le "${cur_day:-0}" ] && { echo "[convcmp] NO PROGRESS — NOT chaining"; exit 1; }
echo "[convcmp] $new_day/$TARGET_DAYS; submitting next link."
sbatch -J "amip_cv_${CONV}" --export="ALL,CONV=$CONV,GWD=$GWD,SSO_FILE=$SSO_FILE,OUTDIR=$OUTDIR,RESOLUTION=$RESOLUTION,NLEV=$NLEV,DT=$DT,TARGET_DAYS=$TARGET_DAYS,CHECKPOINT_DAYS=$CHECKPOINT_DAYS,MAX_WALL=$MAX_WALL,EXTRA_FLAGS=$EXTRA_FLAGS,CONVCMP_CODE=$CODE,LEGOESM_REPO=$LEGOESM_REPO,CLM_SURFDATA=$CLM_SURFDATA" "$0"
