#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=bech_cdx
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/logs/bech_cdx_%j.out
set +e
WT=${WT:?}; PIN=${PIN:?}
for _CH in "${SLURM_TMPDIR:-}" "/tmp" "/burg-archive/glab/users/pg2328/_codexhome"; do
  [ -n "$_CH" ] || continue
  CANDIDATE="$_CH/codex_home_${SLURM_JOB_ID:-$$}"
  mkdir -p "$CANDIDATE" 2>/dev/null || continue
  cp /burg-archive/home/pg2328/.codex/auth.json "$CANDIDATE/" 2>/dev/null || continue
  [ -f "$CANDIDATE/auth.json" ] || continue
  export CODEX_HOME="$CANDIDATE"; break
done
[ -n "${CODEX_HOME:-}" ] || { echo "FATAL: no CODEX_HOME"; exit 90; }
export PATH=/burg-archive/home/pg2328/.nvm/versions/node/v24.16.0/bin:$PATH
echo "host: $(hostname)"; codex --version || exit 91
cd $WT || exit 92
[ "$(git rev-parse HEAD)" = "$PIN" ] || { echo "PREFLIGHT FAIL"; exit 93; }
git diff --quiet || { echo "DIRTY"; exit 94; }

codex exec --sandbox read-only -C $WT "$(cat <<'PROMPT'
HARNESS AUDIT. A ranking metric is producing an incoherent result and I need you
to read the SCORING CODE and find out why - this is a "validate the instrument"
review, not a physics review.

THE INCOHERENCE (tuned SCM-RCE profiles vs a SAM CRM, all schemes):
  scheme      thermo_score  heldout_T_rmse_K  trop_qv_rmse_g_kg  sfc_RH
  mass_flux   1.92 (BEST)   7.40              0.78               0.726
  emanuel     2.67          2.45              2.04               0.574
  bechtold    4.74          2.87              2.29               0.890
mass_flux is ranked FIRST with a 7.4 K temperature error; bechtold is 9th with
2.87 K. A scheme 2.6x worse in temperature cannot be the better thermodynamic
fit. The metric is dominated by something other than temperature, and I need to
know exactly what, with file:line.

READ THESE AND REPORT THE ACTUAL FORMULA:
- packages/ml/legoesm/training/scm_rce_metrics.py  (score_thermo_jax,
  thermo_terms_jax, weighted_rmse, log_humidity_profile,
  relative_humidity_profile, the DEFAULT_THERMO_* scale constants, and the
  tropospheric_mass_weights / tropospheric_min_pressure it uses)
- scripts/run/run_scm_rce_campaign.py  (how thermo_score is assembled from the
  T term and the humidity term for the RunDiagnostics record, which humidity
  representation is the DEFAULT - logq / rh / absolute qv - and how the T term
  and humidity term are WEIGHTED relative to each other)

ANSWER, each with file:line:
1. Write the thermo_score formula as actually implemented: term by term, with
   the numeric scale/normalisation on each. Is it T_rmse/scale_T +
   qv_rmse/scale_q, and what are the two scales? A scale that makes 1 g/kg of
   humidity error "cost" far more than 1 K of temperature error is the whole
   story.
2. Which humidity variable does the DEFAULT objective use - log q, RH, or
   absolute q_v - and over what vertical range/mass-weighting? bechtold's
   near-surface RH is 0.89 (moist) vs the CRM 0.752; if the metric is a
   surface-heavy RH or log-q term, bechtold is being penalised almost entirely
   on BOUNDARY-LAYER HUMIDITY.
3. Is the T term "heldout_T_rmse_K" (tropospheric) or a DIFFERENT, full-column
   or surface-weighted temperature that would explain why mass_flux's 7.4 K
   held-out error does not hurt its thermo_score? i.e. does thermo_score even
   use the same T RMSE I am reading in the table?
4. Given the campaign's OWN finding (documented in
   docs/atmosphere/scm_rce_tq_tuning_strategy.md and the memory of this work)
   that the near-surface humidity bias is COMMON to all schemes and set by the
   surface-flux / cold-pool physics OUTSIDE convection: does the thermo_score
   effectively rank the convection schemes on a NON-CONVECTIVE variable? Quote
   the weighting that decides this.
5. Is there a defensible alternative score already present in the code (a
   temperature-only, or troposphere-above-BL, or held-out metric) that would
   NOT be dominated by the shared surface-humidity bias, and what would the
   ranking be under it? Name the field.

Do not speculate about Bechtold's physics. Report only what the scoring code
does, with line numbers, and whether the ranking is measuring convection or
measuring the boundary-layer humidity we already know is exogenous.
PROMPT
)" 2>&1
echo "CODEX_RC=$?"
echo ALLDONE
