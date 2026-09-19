#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=tkeprem
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/tkeprem_%j.out
#
# RE-MEASURE THE PREMISE BEFORE BUILDING ON IT.
#
# The claim driving the next piece of work is: over one matched 3600 s hour,
# seeded with NEMO's OWN `en` on NEMO's OWN state, our K SHRINKS ~14% while
# NEMO's avm GROWS 5%. That number was measured on card 2f04628d6. The card has
# since gained tke_mxl_choice=4, dissipation_discretization=nemo_1p5_split,
# alpha_tke=1 and kappa_convention=veros_sqrte, and main has been merged in
# twice. Quoting the old number as if it still held would be exactly the
# failure this project keeps paying for, so it is re-measured first.
#
# Step 1 stitches NEMO's per-rank ocean restart (the ONLY source of `en` --
# no XIOS stream carries it) into global fields. Step 2 runs the EXISTING
# Stage-A2 Mode-A comparator on it: --restart-npz with --rec 1, because the
# restart stores no avt and record 0 is an initialisation transient.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
OUT=$ROOT/omip_nemo/tke_premise
RST=$OUT/nemo_restart_8760.npz
mkdir -p $OUT

RC=0

echo "=== 1. stitch NEMO's per-rank restart (carries en, avm_k, avt_k, dissl) ==="
if [ -f "$RST" ]; then
  echo "[skip] $RST already present"
else
  $PY -u scripts/validate/ocean_fidelity/rebuild_nemo_restart.py \
    --pattern "$TRD/ORCA1_00008760_restart_oce_*.nc" \
    --fields tn sn un vn en avm_k avt_k dissl \
    --out "$RST" 2>&1 | tail -20
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: stitch failed"; exit 1; }
fi
[ -f "$RST" ] || { echo "FATAL: no $RST"; exit 1; }

echo "=== 2. Stage-A2 Mode-A on the CURRENT card ==="
# --rec 1 is required by the comparator when --restart-npz is given.
$PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
  --tfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_T.nc" \
  --ufile "$TRD/ORCA1_1h_20010101_20010101_trd1h_U.nc" \
  --vfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_V.nc" \
  --rec 1 --restart-npz "$RST" --skip-maps \
  --output-dir "$OUT/stage_a2" 2>&1 | tail -60
[ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: comparator failed"; RC=1; }

echo "DONE rc=$RC"
exit $RC
