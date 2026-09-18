#!/bin/bash
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --job-name=iwmpr
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/_omip_merge/omip_nemo/iwmpr_%j.out
#
# IS THE PRANDTL GAP REAL, OR IS IT WHAT EACH SIDE INCLUDES?
#
# Stage-A2 reported our Prandtl pinned at its ceiling of 10 across 94.9% of
# Antarctic interfaces while NEMO sits at 1.681 there. That was retracted
# (7e2176b7b) as confounded: ORCA1 runs ln_zdfiwm=.true. (namelist_cfg:434),
# so NEMO's avt AND avm both carry a wave-driven diffusivity, while the probe
# calls tke_vertical_mixing alone. Adding the SAME K to both sides of a ratio
# drives it toward 1, which is exactly the direction of the reported gap.
#
# Three arms, ONE variable each, so the effect is attributable:
#   base      as before, TKE only                     (the retracted number)
#   iwm       + ORCA1's own de Lavergne wave maps     (zdfiwm_forcing_TRA.nc)
#   iwmbg     + NEMO's molecular backgrounds too      (avmb 1.4e-6, avtb 1e-10)
#
# The third arm exists because NEMO's zdfiwm_init OVERWRITES the namzdf
# backgrounds when the wave field is on -- running the wave mixing on top of
# our 1.2e-4/1.2e-5 backgrounds would double-count the interior floor. Whether
# that matters is measured here rather than assumed.
#
# PRE-REGISTERED: if the wave mixing is the explanation, Pr_ours collapses
# from ~10 toward NEMO's ~1.7 and the ceiling fraction drops far below 94.9%.
# If Pr_ours stays pinned, the gap is real and survives its main confound.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

TRD=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
OUT=$ROOT/omip_nemo/tke_premise
RST=$OUT/nemo_restart_8760.npz
# The file namzdf_iwm itself names (sn_mpb..sn_dsc all point at _TRA).
# NOT $TRD/zdfiwm_forcing_TRA.nc: every RUN_* directory carries that name as a
# relative symlink into RUN_REF, and all of them are BROKEN -- `find` listed
# them because it does not dereference, and `[ -e ]` then failed the card in
# one second. The actual data lives in the INPUTS tree.
IWM=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/zdfiwm_forcing_TRA.nc

for f in "$RST" "$IWM"; do
  [ -e "$f" ] || { echo "FATAL: missing $f"; exit 1; }
done

RC=0
arm() {   # arm <label> <extra flags...>
  local label="$1"; shift
  echo ""
  echo "=== ARM $label : $* ==="
  $PY -u scripts/validate/ocean_fidelity/compare_tendencies_nemo.py \
    --tfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_T.nc" \
    --ufile "$TRD/ORCA1_1h_20010101_20010101_trd1h_U.nc" \
    --vfile "$TRD/ORCA1_1h_20010101_20010101_trd1h_V.nc" \
    --rec 1 --restart-npz "$RST" --skip-maps \
    --output-dir "$OUT/ab_$label" "$@" 2>&1 | tail -25
  [ ${PIPESTATUS[0]} -eq 0 ] || { echo "FATAL: arm $label failed"; RC=1; }
}

arm base
arm iwm    --iwm-forcing "$IWM"
arm iwmbg  --iwm-forcing "$IWM" --iwm-backgrounds

echo ""
echo "DONE rc=$RC  (read ab_*/tendency_match_rec1.json — the logs are tail-truncated)"
exit $RC
