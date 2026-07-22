#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=mpas14a_eice1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|a40"
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/mpas14a_eice1_%j.out
# mpas14a_eice1 = mpas12 baseline + ONE tuning knob.  TUNE arm A: --tke-eice 1 (linear) vs mpas12's card default 3 — under-ice suppression strength
# (baseline = the
# NEMO ORCA1 &namzdf_tke card — nn_mxl=2, lc, etau nn_etau=1 w/ nn_htau=1
# latitude profile, eice=3 card default — through the SAME grid-agnostic
# kernel the tripole runs; molecular backgrounds + preset convection stripped
# to the tripole closure environment).  Controlled A/B: EVERYTHING identical
# to mpas9/mpas10 except the closure, so the day-30 Arctic delta is the
# TKE-vs-KPP scheme effect on MPAS.  The 2x2 (trp5/trp6/mpas10) put the
# scheme half at +0.78 Siberian / +36-50 m Arctic MLD — if the card closes
# it, mpas12 should land near Siberian ~+1.0 and MLD well under +42.9, and
# mpas12-vs-trp5 becomes the SAME-CARD cross-grid TKE pair (grid effect
# only).  Score d30 vs NEMO m1 + A/B vs mpas10 (KPP-eice rung).
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
FIX="--mpas-vmix tke --tke-eice 1"
LABEL=mpas14a_eice1
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
MPAS9=$ROOT/results/omip_nemo/nemolev_mpas12_tke
echo "=== RUN $LABEL (mpas9 + --kpp-eice 3, the MPAS-KPP under-ice lever, day-30 A/B) ==="
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $LEVERS $PARITY $IC $FIX --output $OUT 2>&1 | tail -60
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0030.npz
if [ -f "$SNAP" ]; then
  echo "=== $LABEL day-30 vs NEMO m1 ==="
  $PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
    --nemo-gridt "$GT" --nemo-month 1 --freeze-clamp-C -1.8 --sss-faithful \
    --grid-label "$LABEL" \
    --output-dir $ROOT/results/omip_nemo/compare_${LABEL}_m01 2>&1 | tail -25
  echo "=== $LABEL Arctic-SSS regions day-30 (eice ON) ==="
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" "mpas14a_eice1 d30" 1
  echo "=== A/B: mpas9 day-30 (eice OFF, same protocol) ==="
  [ -f "$MPAS9/snapshot_day0030.npz" ] && \
    $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$MPAS9/snapshot_day0030.npz" "$GT" "mpas12_tke d30 (baseline)" 1
else
  echo "[run] no day-30 snapshot -> check $OUT for blowup."
fi
echo "=== DONE ==="; date
