#!/bin/bash
#SBATCH --job-name=insn2_30
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/insn2_30_%j.out
#
# THE MISSING CONTROL for the signed-TEOS-10 arm 9409057.
#
# Every other day-30 arm on disk differs from 9409057 in THREE card fields
# (alpha_tke, kappa_convention, n2_mode) and the closest 90-day arm also
# differs in dissipation_discretization. Without this run the signed arm
# cannot be attributed to the stratification change at all.
#
# Byte-identical to 9409057's command line except:
#   --tke-n2-mode insitu   (the one variable)
#   a distinct --output dir
# n2_eos_form stays teos10 and is INERT under insitu, so it is not a second
# variable -- verified by the roundtrip check in _n2_flag_roundtrip.sbatch.
#
# PRE-REGISTERED, Southern Ocean mean MLD bias vs NEMO at day 30. The sqrt(2)
# and alpha fixes drove it from +1.5 m to -19.7 m; the question is how much
# of that the stratification change gives back.
#   CONFIRMED : signed lands -8 to +8 m.
#   PARTIAL   : -15 to -8 m.
#   OVERSHOOT : beyond +8 m.
#   REFUTED   : still below -15 m.
# Arctic SSS is the counter-prediction already on record: calm-column K went
# 1.37x -> 1.80x NEMO, so Arctic SSS should get slightly WORSE. If it
# IMPROVES, calm-column K is not the controlling term for the halocline.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM
PIN=063f6bd003e7177e219ce95d3b184407affd59df
cd $WT || exit 1
HAVE=$(git -C $WT rev-parse HEAD)
[ "$HAVE" = "$PIN" ] || { echo "FATAL: pin drift, HEAD=$HAVE want=$PIN"; exit 1; }
echo "=== insitu-N2 control, pin $PIN ==="
git -C $WT log --oneline -1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -u scripts/run/run_omip_core2.py --grid tripole --mesh data/grids/eORCA1.2_mesh_mask.nc --dt 150 --min-levels 2 --C-smag-lap 3.0 --smag-cfl-safety 0.125 --ew-cyclic-overlap --nemo-vertical --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv --momentum-rk3 --geothermal --runoff --sss-restore --sss-restore-tau-days 45.5 --sss-restore-bound-mmday 4 --sss-ice-gate-nemo --prognostic-sea-ice --prognostic-ice-dynamics free_drift --years 0.0822 --snapshot-every-days 10 --diag-every-days 5 --tracer-advection superbee --runoff-depth-nemo-ini --river-mouth-restoring-gate --bottom-drag-scheme nemo_quadratic --iwm --iwm-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc --dm2dc --isf --isf-forcing-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc --bbl-adv --woa-init --woa-t /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_t_woa57_annual.nc --woa-s /burg/glab/users/pg2328/legoESM/data/woce_gouretski/woce_s_woa57_annual.nc --nemo-monthly-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions/woce_temp_monthly_init_4p2.nc /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/initial_conditions/woce_salt_monthly_init_4p2.nc --sss-restore-file /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/sss_climatology_for_restoring.nc --tripole-vmix tke --ice-init /burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/Ice_initialization.nc --tke-surface-bc nemo_dirichlet --tke-mxl-choice 3 --tke-prognostic --tke-kappa-convention veros_sqrte --sw-rgb-chl --ice-ocean-heat-coeff 70 --gateway-transports --output /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/nemolev_trp_insitun2_d30 --mle --mle-ce 0.06 --tke-n2-mode insitu
rc=$?; echo "=== insitu-N2 control rc=$rc ==="; exit $rc
