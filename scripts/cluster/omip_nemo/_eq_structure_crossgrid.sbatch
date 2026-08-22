#!/bin/bash
#SBATCH --job-name=eqstruct
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/eqstruct_%j.out
#
# DOES FESOM GET THE COLD TONGUE RIGHT, OR RIGHT BY CANCELLATION?
#
# FESOM's cold-tongue bias is +0.17 C against the tripole's +1.42 on identical
# forcing, and that was about to be used to argue the tripole's remaining bias
# is tripole-specific.  The second reviewer refused the inference: FESOM is
# 0.65 C cold globally with mixed layers 44 to 48 m too deep, so its correct
# equatorial temperature may be over-mixing cancelling its own structural warm
# bias -- which is visible, since FESOM on RAW winds sits at +1.55.
#
# The reviewer's discriminator, from output that already exists: equatorial
# 20 C isotherm depth and thermocline SHARPNESS alongside SST.  A matching SST
# on a deeper, more diffuse thermocline is cancellation and kills the
# inference.  A matching SST with the oracle's structure supports it.
set -uo pipefail
R=/burg-archive/glab/users/pg2328/legoESM; cd $R || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$(ls -d $R/packages/*/ | sed 's:/$::' | tr '\n' ':')$R:$R/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu

echo "=== gate: the probe must be able to SEE a smeared thermocline ==="
$PY -m pytest -q tests/ocean/unit/test_equatorial_thermocline_structure.py 2>&1 | tail -4
rc=$?; [ $rc -eq 0 ] || { echo "FATAL: probe gate failed rc=$rc"; exit 1; }

GW_T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
$PY -u scripts/validate/ocean_fidelity/equatorial_thermocline_structure.py \
  --snapshot $R/results/omip_nemo/nemolev_trp_modfrc_d30/snapshot_day0030.npz \
  --label tripole_corrected \
  --snapshot $R/results/omip_nemo/fesom_modfrc_d30/snapshot_day0030.npz \
  --label fesom_corrected \
  --snapshot $R/results/omip_nemo/fesom_anchor_d30/snapshot_day0030.npz \
  --label fesom_raw \
  --snapshot $R/results/omip_nemo/nemolev_trp_signedn2_d30/snapshot_day0030.npz \
  --label tripole_raw \
  --nemo-gridt "$GW_T" --nemo-rec 5
echo "=== structure rc=$?"

echo
echo "=== is the cold-tongue bias still DRIFTING at day 30? ==="
echo "The reviewer's free prerequisite: a day-30 number describes climatology"
echo "only if it has stopped moving.  Days 10, 20, 30 against GATEWAY records"
echo "1, 3 and 5 -- matched windows on both sides."
for pair in "nemolev_trp_modfrc_d30:tripole" "fesom_modfrc_d30:fesom"; do
  D=${pair%%:*}; L=${pair##*:}
  for dr in 0010:1 0020:3 0030:5; do
    DAY=${dr%%:*}; REC=${dr##*:}
    SNAP=$R/results/omip_nemo/$D/snapshot_day$DAY.npz
    [ -f "$SNAP" ] || { echo "$L day $DAY: no snapshot"; continue; }
    N3=$($PY scripts/validate/compare_omip_nemo.py --legoesm-snapshot "$SNAP" \
      --nemo-gridt "$GW_T" --nemo-time-idx $REC --freeze-clamp-C -1.8 \
      --sss-faithful --grid-label ${L}_d$DAY \
      --output-dir $R/results/omip_nemo/compare_gw_drift_${L}_d$DAY 2>&1 \
      | grep -E "nino3_5S5N" | head -1)
    echo "$L day $DAY (rec $REC): $N3"
  done
done
echo DONE
