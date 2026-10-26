#!/bin/bash
#SBATCH --job-name=n2seed
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/n2seed_%j.out
#
# IS THE REMAINING LENGTH DEFICIT THE N-SQUARED INSTRUMENT?
#
# With NEMO's real wind anchor wired in (job 9406984) the zero-step ratio is
# 0.897 Antarctic / 0.914 tropics / 0.792 Arctic -- a real ~10-21% shortfall
# in our mixing length, and the half of the day-30 Southern Ocean regression
# that the sqrt(2) fix exposed rather than caused.
#
# The buoyancy length is sqrt(2e)/N, so a TOO-LARGE N gives a TOO-SHORT
# length. The card computes N2 from the in-situ density gradient, clipped at
# zero, and that form carries a compressibility bias measured at +4.3e-5 in a
# prior session -- biased too stable, i.e. exactly the direction of the
# deficit. This swaps ONLY the N2 that seeds the length.
#
# CAVEAT, stated because codex 9405307 caught me wiring this as if it were
# NEMO's: 'nemo_bn2' is legoESM's S-EOS adiabatic bn2, while ORCA1 runs
# TEOS-10. It is much closer to NEMO's rn2 than a clipped in-situ gradient,
# but it is NOT rn2. This BOUNDS the effect; it does not reproduce NEMO.
#
# PRE-REGISTERED, antarctic zero-step K_M/avm_k (insitu baseline 0.897):
#   N2 IS THE CAUSE     : >= 0.95
#   PARTIAL             : 0.92 - 0.95
#   NOT THE CAUSE       : <= 0.92 -- the length shortfall is in the sweeps or
#                         the anchor chain, not the buoyancy seed.
set -uo pipefail
WT=/burg-archive/glab/users/pg2328/legoESM; cd $WT || exit 1
export PYTHONPATH="$WT/src:$(ls -d $WT/packages/*/ | sed 's:/$::' | tr '\n' ':')$WT"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
T=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_TRD2
R=$WT/results/omip_nemo/zero_step_closure
echo "HEAD: $(git -C $WT log --oneline -1)"
NFAIL=0
arm () {
  echo; echo "########## N2 seed = $1"
  $PY -u scripts/validate/ocean_fidelity/nemo_zero_step_closure.py \
    --restart-npz $T/restart_8760_full.npz \
    --mesh-mask $WT/data/grids/eORCA1.2_mesh_mask.nc \
    --trd-tfile $T/ORCA1_1h_20010101_20010101_trd1h_T.nc --trd-rec 0 \
    --n2-for-length "$1" --out-json $R/n2seed_$1.json 2>&1 \
    | grep -aE "^\[anchor\]|^\[N2\]|^\[C3\]|^band|^antarctic|^tropics|^arctic|^ONE STEP|FATAL"
  local rc=${PIPESTATUS[0]}; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1))
}
arm insitu
arm nemo_bn2
echo; echo "=== ALL DONE (failed arms: $NFAIL) ==="
exit $((NFAIL > 0))
