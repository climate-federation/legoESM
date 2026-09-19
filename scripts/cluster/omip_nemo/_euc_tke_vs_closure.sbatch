#!/bin/bash
#SBATCH --job-name=euctke
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/euctke_%j.out
#
# IS THE EQUATORIAL MOMENTUM DEFICIT THE CLOSURE, OR THE TKE?
#
# WHAT SURVIVES from the last three probes. In the nino3/eqpac calm columns
# above 100 m -- the EUC core, the depth Z20 is measured at -- our TRACER
# diffusivity matches NEMO (K_H/avt 1.058) and our MOMENTUM viscosity is half
# it (K_M/avm 0.472). Both numbers are unchanged, to three digits, by the
# matched --iwm-backgrounds arm (job 9417344), so neither is a background-floor
# artifact.
#
# WHAT IS REFUTED, by that same matched arm: the whole-column 2.311 is NOT the
# rn_avm0/rn_avt0-vs-molecular floor mismatch. Swapping our floors to the
# molecular pair NEMO's zdfiwm_init actually installs moved arctic/calm from
# 124.694 to 124.699 and nino3/calm from 2.311 to 2.311. An rms ratio is
# carried by its largest values; the floors only touch the smallest. The
# background mismatch is a real card defect and it is NOT this ratio's cause.
#
# THE OPEN QUESTION, and the one thing that can answer it cheaply. Two probes
# disagree about our momentum viscosity at the equator:
#     Stage A     (our own Mode-B equilibrium TKE)   K_M/avm  0.472
#     zero step   (HANDED NEMO's own restart en)     K_M/avm_k 0.999
# but they differ in the TKE *and* in the state, so neither is decisive. Stage
# A2 Mode-A takes NEMO's OWN en on NEMO's OWN state and now scores the same
# split with the same reduction and the same depth cuts, leaving the TKE as the
# only difference from Stage A.
#
#   A2 K_M/avm ~ 1 above 100 m  -> the closure is right and our EQUILIBRIUM TKE
#        is ~4x too small at the equator (K ~ sqrt(e), so 0.47^2 ~ 0.22). The
#        lever is TKE production/dissipation, not the mixing length.
#   A2 K_M/avm ~ 0.47 as well   -> the TKE is not the difference; the deficit
#        is in the closure or the length, and the 0.999 zero-step number is
#        explained by its state or its 1.78%-by-volume coverage instead.
#   A2 between                  -> report the split, claim neither alone.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM; cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
echo "=== git SHA: $(git -C $REPO rev-parse HEAD) ==="
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
$PY -m pytest -q tests/validate/test_prandtl_split_report.py 2>&1 | tail -3
D=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile $D/ORCA1_1h_20010101_20010101_trd1h_T.nc \
  --ufile $D/ORCA1_1h_20010101_20010101_trd1h_U.nc \
  --vfile $D/ORCA1_1h_20010101_20010101_trd1h_V.nc \
  --restart-npz $D/restart_8760_full.npz \
  --rec 1 --skip-maps --mxl-choice 3 \
  --output-dir $REPO/results/omip_nemo/euc_tke_vs_closure
echo "=== rc=$? ==="
