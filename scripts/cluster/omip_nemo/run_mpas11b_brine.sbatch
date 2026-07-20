#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=mpas11b_brine
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|a40"
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpas11b_brine_%j.out
# MPAS11b = mpas11a + NEMO SI3 nn_icesal=2 (--ice-sinew 0.75 --ice-drain-days 15).
# nn_eice under-ice attenuation of the KPP velocity scales), newly WIRED on the
# MPAS KPP bridge (_run_mpas_kpp now threads surface_forcing.ice_concentration
# to the shared _kpp_ice_attenuation).  Controlled A/B: EVERYTHING is identical
# to mpas9_parity except --kpp-eice 3, so the day-30 Arctic delta is the eice
# effect alone.  mpas9 Arctic MLD was +62.6 m / Siberian SSS +2.56 (KPP, no
# eice); this measures how far eice narrows the gap toward the tripole TKE-eice
# endgame (+6.7 m / +1.10) — expected PARTIAL: eice attenuates the wave/wind
# w-scale but not the brine B_f convection (ll8 latlon showed ~28% on Siberian).
# 30-day probe: the halocline signature is fully formed by day 30.  Score vs
# NEMO m1 AND diff the SAME-protocol mpas9 day-30 snapshot (the eice A/B).
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/vfs-tke-fix; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
RUN=$ROOT2/cfgs/ORCA1/EXP00/RUN_REF
INP=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
SSSCLIM=$INP/sss_climatology_for_restoring.nc
ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree (fix tree) ==="; git -C $WT log --oneline -1
GA="--grid mpas --mpas-level 7 --mpas-lloyd 20 --dt 150"
COMMON="--nemo-vertical --partial-cell --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.085 --snapshot-every-days 30 --diag-every-days 5"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
PARITY="--bottom-drag-scheme nemo_quadratic"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC --sss-restore-file $SSSCLIM --ice-init $ICEIC"
FIX="--kpp-eice 3 --freshwater-salinity local --no-normalize-freshwater --ice-sinew 0.75 --ice-drain-days 15"
LABEL=mpas11b_brine
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
MPAS9=$ROOT/results/omip_nemo/nemolev_mpas10_kppeice
echo "=== RUN $LABEL (mpas9 + --kpp-eice 3, the MPAS-KPP under-ice lever, day-30 A/B) ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $PARITY $IC $FIX --kpp-ri-crit 0.15 --output $OUT 2>&1 | tail -60
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0030.npz
if [ -f "$SNAP" ]; then
  echo "=== $LABEL day-30 vs NEMO m1 ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "$LABEL" \
    --output-dir $ROOT/results/omip_nemo/compare_${LABEL}_m01 2>&1 | tail -25
  echo "=== $LABEL Arctic-SSS regions day-30 (eice ON) ==="
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" "mpas11b_brine d30" 1
  echo "=== A/B: mpas9 day-30 (eice OFF, same protocol) ==="
  [ -f "$MPAS9/snapshot_day0030.npz" ] && \
    $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$MPAS9/snapshot_day0030.npz" "$GT" "mpas10_kppeice d30 (prev rung)" 1
else
  echo "[run] no day-30 snapshot -> check $OUT for blowup."
fi
echo "=== DONE ==="; date
