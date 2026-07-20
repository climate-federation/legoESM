#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --job-name=ll9_kpppair
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/ll9_kpppair_%j.out
# LL8 = the ll4/ll5 latlon KPP config + THE LEVER UNDER TEST: --kpp-eice 3
# (NEMO nn_eice under-ice attenuation of the KPP velocity scales).  ll5 proved
# the LEVERS-bug fix was a NULL on latlon Arctic (Siberian +2.96 = ll4); the
# residual is KPP under-ice over-mixing, which TKEConfig.eice (F2) fixed only
# on the TKE grid.  KPPConfig.eice is the KPP mirror — this run tests whether
# Siberian +2.96 drops like the tripole's +2.62 -> +0.24 (trp4a, eice-only).
# 30-day probe: the halocline signature is fully formed by day 30.  From the
# vfs-tke-fix tree (carries both fix pairs).  s_ref + normalize kept
# (protocol-clean vs ll4/ll5).
set +e
WT=/burg-archive/glab/users/pg2328/legoESM_worktrees/vfs-tke-fix; cd $WT
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT:$WT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
ROOT=/burg-archive/glab/users/pg2328/legoESM
ROOT2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1
RUN=$ROOT2/cfgs/ORCA1/EXP00/RUN_REF
INP=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields
INITD=$ROOT2/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions
GT=$RUN/ORCA1_1m_20000101_20041231_grid_T.nc
WTIC=$ROOT/data/woce_gouretski/woce_t_woa57_annual.nc
WSIC=$ROOT/data/woce_gouretski/woce_s_woa57_annual.nc
SSSCLIM=$INP/sss_climatology_for_restoring.nc
ICEIC=$ROOT2/cfgs/ORCA1/INPUTS/Ice_initialization.nc
echo "=== HOST ==="; hostname; nvidia-smi -L 2>/dev/null | head -1; date
echo "=== worktree (fix tree) ==="; git -C $WT log --oneline -1
GA="--grid latlon_bathy --latlon-res 180x360 --mesh $ROOT/data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 --mask-marginal-seas"
COMMON="--nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.085 --snapshot-every-days 30 --diag-every-days 5"
P45="--bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file $INP/zdfiwm_forcing_TRA.nc --dm2dc --isf --isf-forcing-file $INP/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv"
LEVERS="--tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate"
IC="--woa-init --woa-t $WTIC --woa-s $WSIC --nemo-monthly-init $INITD/woce_temp_monthly_init_4p2.nc $INITD/woce_salt_monthly_init_4p2.nc --sss-restore-file $SSSCLIM --ice-init $ICEIC"
FIXES="--kpp-eice 3 --freshwater-salinity local --no-normalize-freshwater"
LABEL=ll9_kpppair
OUT=$ROOT/results/omip_nemo/nemolev_${LABEL}
echo "=== RUN $LABEL (ll8 + F1 local-S: the FULL fix pair on the KPP grid (eice + local-S), mirroring the tripole endgame) ==="
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 $PY -u scripts/run/run_omip_core2.py \
  $GA $COMMON $P45 $LEVERS $IC $FIXES --kpp-ri-crit 0.15 --sw-rgb-chl --output $OUT 2>&1 | tail -60
echo "run_rc=${PIPESTATUS[0]}"; date
SNAP=$OUT/snapshot_day0030.npz
if [ -f "$SNAP" ]; then
  echo "=== Arctic SSS regions day-30 vs NEMO m1 (A/B vs ll5 d30: Siberian ~+2.9) ==="
  $PY $ROOT/scripts/tmp/_diag_arctic_sss_regions.py "$SNAP" "$GT" "ll9_kpppair d30 (eice3 + local-S)" 1
else
  echo "[run] no day-30 snapshot -> check $OUT for blowup."
fi
echo "=== DONE ==="; date
