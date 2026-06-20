#!/bin/bash
#SBATCH --account=CUSTOMIZE_ACCOUNT
#SBATCH --partition=CUSTOMIZE_PARTITION
#SBATCH --job-name=ck_sensitivity_preflight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=results/compare_reanalysis/preflight_%j.out

# ---------------------------------------------------------------------------
# REALISTIC (rrtmgp) C_K-SENSITIVITY go/no-go pre-flight (docs/compare_reanalysis_runbook.md
# §1b). rrtmgp's radiation graph is COMPUTE-heavy (iter 443: a res-4/nlev-20/2-day realistic
# run did not finish in ~16 min on a login node), so the REALISTIC pre-flight runs as a batch
# job, NOT interactively. It exits 0 (GO -- the LES->C_K loop can lower the real-ERA5 bias) /
# 1 (NO-GO -- idealization-dominated), so the campaign launch can be gated on THIS job's exit
# code (or on it via a dependency), e.g.:
#   jid=$(sbatch --parsable <this>); sbatch --dependency=afterok:$jid run_correction_campaign.sbatch
#
# CUSTOMIZE the #SBATCH account/partition + the REPO/PY paths and the LOCAL_ERA5_* below.
# ---------------------------------------------------------------------------
set +e
REPO="${LEGOESM_REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 2; }
PY="${LEGOESM_PY:-.venv/bin/python}"
# A LOCAL NCAR-RDA ll025 ERA5 archive (e5.oper.an.{pl,sfc}.*.ll025*.nc) + the YYYYMMDD date.
LOCAL_ERA5_DIR="${LOCAL_ERA5_DIR:?set LOCAL_ERA5_DIR to a local NCAR-RDA ll025 ERA5 archive}"
LOCAL_ERA5_DATE="${LOCAL_ERA5_DATE:?set LOCAL_ERA5_DATE=YYYYMMDD}"
RESOLUTION="${RESOLUTION:-8}"
NLEV="${NLEV:-20}"        # rrtmgp wants ~20 levels for accurate radiation (runbook §2b)
DAYS="${DAYS:-10}"
mkdir -p results/compare_reanalysis

echo "=== HOST ==="; hostname; date
# Run the REALISTIC (rrtmgp) two-C_K sensitivity vs real ERA5; the tool's exit code IS the
# go/no-go gate (0=GO, 1=NO-GO). Swap --days for "--days-sweep 1 3 6" to ask if run length
# (equilibration) is the lever instead.
JAX_ENABLE_X64=1 "$PY" scripts/experiment/ck_sensitivity_vs_era5.py \
  --local-era5-dir "$LOCAL_ERA5_DIR" --local-era5-date "$LOCAL_ERA5_DATE" \
  --resolution "$RESOLUTION" --nlev "$NLEV" --radiation rrtmgp --days "$DAYS" --c-k 0.4 1.0
rc=$?
echo "preflight_rc=$rc (0=GO -> launch the campaign; 1=NO-GO -> add realism / run longer)"; date
exit "$rc"
