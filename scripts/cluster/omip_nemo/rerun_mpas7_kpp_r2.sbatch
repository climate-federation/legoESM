#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=kppdeep_r2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/kppdeep_%x_%j.out
# VMIX STRUCTURAL-RESIDUAL SENSITIVITY (verify-first, 2026-06-22).
# Diagnosis: the OMIP subtropical mixed layer is ~half of NEMO (MLD bias -35 m
# NH-subtropics, model-wide MPAS==tripole at 75-lev, NOT spin-up) -> surface
# heat+freshwater trapped in a too-thin top cell -> the subtropical surface
# warm(+1.4 trop)+fresh(-0.6) bias.  KPP sets h_bl where Ri_b crosses Ri_crit;
# RAISING Ri_crit (0.3->0.5) AND Cv (1.6->2.5) deepens the boundary layer.
# This run is a CLEAN 1-knob-group delta vs nemolev_ico7_superbee_nogate_1yr
# (IDENTICAL stack: superbee + nogate + spread-150, ONLY the two KPP knobs move).
# Does MLD deepen toward NEMO AND the surface warm+fresh shrink?  If yes the
# vmix lever is confirmed; attribute Ri_crit vs Cv in a follow-up single-knob run.
set +e
ROOT=/burg-archive/glab/users/pg2328/legoESM; cd $ROOT
# packages/* (federation namespace) + ROOT (scripts.* package imports) +
# scripts/validate (the diag scripts do a bare `from compare_omip_nemo import`).
export PYTHONPATH="$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
RUN=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF
INP=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null; date

# IDENTICAL faithful stack to the baseline nemolev_ico7_superbee_nogate_1yr.
COMMON="--nemo-vertical --woa-init --partial-cell --pgf-scheme smc03 \
  --adaptive-implicit-vertadv --momentum-rk3 --geothermal \
  --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 \
  --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift \
  --years 0.3 --snapshot-every-days 30 --diag-every-days 5"
GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
LEVERS="--tracer-advection superbee --runoff-depth-spread-m 150"   # nogate (no --river-mouth-restoring-gate)
KPP="--kpp-ri-crit 0.5 --kpp-cv 2.5"

LABEL=ico7_kppdeep_ri05cv25_r2
OUT=results/omip_nemo/nemolev_${LABEL}
echo "=== RUN $LABEL (KPP Ri_crit=0.5 Cv=2.5; deepen subtropical MLD) ==="
$PY -u scripts/run/run_omip_core2.py $GA $COMMON $LEVERS $KPP --output $OUT 2>&1 | tail -40
echo "run_rc=${PIPESTATUS[0]}"; date

echo "=== SCORE + BATTERY (month-3, 75-vs-75) ==="
SNAP=$OUT/snapshot_day0090.npz
if [ -f "$SNAP" ]; then
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "nemolev_${LABEL}" --output-dir results/omip_nemo/compare_nemolev_${LABEL}_m03 2>&1 | tail -18
  $PY scripts/validate/diag_omip_nemo_battery.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 3 --domain-cfg $INP/domain_cfg.nc \
    --nemo-gridv $RUN/ORCA1_1y_20000101_20041231_grid_V.nc \
    --nemo-gridu $RUN/ORCA1_1y_20000101_20041231_grid_U.nc \
    --grid-label "nemolev_${LABEL}" --output-dir results/omip_nemo/battery_nemolev_${LABEL}_m03 2>&1 | tail -18
  echo "=== MLD + T/S depth-profile bands (model-NEMO) ==="
  $PY scripts/tmp/_diag_mld_bands.py "$SNAP" "$GT" 3 2>/dev/null
  $PY scripts/tmp/_diag_ts_depth_profile.py "$SNAP" "$GT" 3 2>/dev/null
else
  echo "[score] no day-90 snapshot -> check $OUT for blowup / log above."
fi
echo "=== DONE ==="; date
