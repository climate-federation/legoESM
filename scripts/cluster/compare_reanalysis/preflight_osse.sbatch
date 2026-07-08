#!/bin/bash
#SBATCH --account=CUSTOMIZE_ACCOUNT
#SBATCH --partition=CUSTOMIZE_PARTITION
#SBATCH --job-name=osse_preflight
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=results/compare_reanalysis/osse_preflight_%j.out

# ---------------------------------------------------------------------------
# PERFECT-MODEL OSSE go/no-go pre-flight (docs/compare_reanalysis_runbook.md §1b/§6).
# The OSSE is the CONTROLLED-TWIN go/no-go: it runs the model with a KNOWN C_K as the
# pseudo-truth + a BIASED start, runs the LES->C_K correction loop, and checks it LOWERS
# the bias AND RECOVERS the known coefficient. It needs NO real ERA5 (the model's own run is
# the truth), but it runs the model multiple times (true + biased + per-iteration re-runs) so
# it is COMPUTE-heavy -> a batch job, NOT interactive. Exits 0 (GO -- the loop recovers the
# parameter in the twin, so the closure inversion works) / 1 (NO-GO -- it does not, fix that
# before spending HPC on real ERA5), so the campaign launch can be gated as a dependency:
#   jid=$(sbatch --parsable <this>); sbatch --dependency=afterok:$jid run_correction_campaign.sbatch
#
# CUSTOMIZE the #SBATCH account/partition + the REPO/PY paths and CONFIG below.
# ---------------------------------------------------------------------------
set +e
REPO="${LEGOESM_REPO:-$PWD}"
cd "$REPO" || { echo "cannot cd to REPO=$REPO"; exit 2; }
PY="${LEGOESM_PY:-.venv/bin/python}"
# The base ExperimentConfig the campaign uses (its grid/nlev/radiation define the twin); the
# OSSE needs NO real ERA5 -- the model's own run with the KNOWN coefficient is the pseudo-truth.
CONFIG="${CONFIG:?set CONFIG to the base ExperimentConfig JSON (the campaign --config)}"
MODE="${MODE:-amip}"
TRUE_CK="${TRUE_CK:-0.4}"
BIASED_CK="${BIASED_CK:-0.6}"
ITERATIONS="${ITERATIONS:-3}"
mkdir -p results/compare_reanalysis

ARGS=(--config "$CONFIG" --mode "$MODE" --iterations "$ITERATIONS")
# Single-coefficient (default) vs SIMULTANEOUS multi-coefficient: set COEFFICIENTS=C_K,Pr_t,C_eps
# to validate the multi-coefficient campaign (it then ignores TRUE_CK/BIASED_CK and biases each
# coefficient by MULTI_BIAS_FACTOR). Otherwise the single-C_K twin uses TRUE_CK (pseudo-truth)
# vs BIASED_CK (the wrong starting value the loop must recover from).
if [ -n "${COEFFICIENTS:-}" ]; then
  ARGS+=(--coefficients "$COEFFICIENTS")
  if [ -n "${MULTI_BIAS_FACTOR:-}" ]; then ARGS+=(--multi-bias-factor "$MULTI_BIAS_FACTOR"); fi
else
  ARGS+=(--true-ck "$TRUE_CK" --biased-ck "$BIASED_CK")
fi
# CROSS-RESOLUTION twin: FINE_RESOLUTION=N validates that the env-kernel TRANSFERS coarse->fine
# (the grid-agnostic deploy, iter 72). Single-coefficient only (mutually exclusive with
# COEFFICIENTS -- the CLI fails loud on the combination).
if [ -n "${FINE_RESOLUTION:-}" ]; then ARGS+=(--fine-resolution "$FINE_RESOLUTION"); fi
# Diagnosis-path PARITY with the campaign (iter 471): match these to the campaign's flags so the
# go/no-go predicts the REALISTIC diagnosis (--surface-flux changes HOW the closure is diagnosed,
# not just which columns; --orographic-forcing the LES large-scale forcing over terrain).
if [ -n "${SURFACE_FLUX:-}" ]; then ARGS+=(--surface-flux); fi
if [ -n "${OROGRAPHIC_FORCING:-}" ]; then ARGS+=(--orographic-forcing "$OROGRAPHIC_FORCING"); fi
if [ -n "${COUPLED_PRESET:-}" ]; then ARGS+=(--coupled-preset "$COUPLED_PRESET"); fi

echo "=== HOST ==="; hostname; date
# The tool's exit code IS the go/no-go gate (0=GO, 1=NO-GO); it prints the verdict + the
# per-coefficient parameter recovery.
JAX_ENABLE_X64=1 "$PY" scripts/validate/run_perfect_model_osse.py "${ARGS[@]}"
rc=$?
echo "osse_preflight_rc=$rc (0=GO -> the loop recovers the parameter in the twin; 1=NO-GO -> fix the closure inversion before real ERA5)"; date
exit "$rc"
