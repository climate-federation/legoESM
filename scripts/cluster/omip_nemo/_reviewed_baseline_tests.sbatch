#!/bin/bash
#SBATCH --job-name=revbase
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/revbase_%j.out
#
# The card after codex 9405307 refuted three of five: tke_mxl_choice=4 and
# dissipation_discretization="nemo_1p5_split" survive; veros_dz_slots,
# n2_mode=nemo_bn2 and tke_surface_bc_level=nemo_z0 are reverted and pinned OFF.
# Re-measures the SURVIVING pair offline and re-runs every suite that asserts
# the card, including the two codex named as expecting choice 3.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM
cd $WT || exit 1
export PYTHONPATH="$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
TRD2=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
echo "HEAD: $(git -C $WT log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

run "card audit after the revert" \
  $PY -u scripts/validate/ocean_fidelity/audit_orca1_tke_card.py --iwm

run "the REVIEWED card, measured offline (was 1.846 amplitude-only, 0.634 +alpha)" \
  $PY -u scripts/validate/ocean_fidelity/nemo_zero_step_closure.py \
    --restart-npz $TRD2/restart_8760_full.npz \
    --mesh-mask $WT/data/grids/eORCA1.2_mesh_mask.nc \
    --trd-tfile $TRD2/ORCA1_1h_20010101_20010101_trd1h_T.nc --trd-rec 0 \
    --out-json $WT/results/omip_nemo/zero_step_closure/reviewed_card.json

run "every suite that asserts this card" \
  $PY -m pytest -q tests/unit/test_run_omip_core2_kappa_convention_cli.py \
      tests/unit/test_run_omip_core2_kpp_override.py \
      tests/ocean/unit/test_tripole_vmix_cli.py \
      tests/unit/test_omip_config_parity.py \
      tests/unit/test_run_omip_core2_freshwater_closure_cli.py

run "MPAS must still BUILD on the shared card (codex: it fail-loud rejects n2_mode/dz_slots)" \
  $PY -m pytest -q tests/ocean/unit/test_mpas_tke.py

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
