#!/bin/bash
#SBATCH --job-name=scpair
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scpair_%j.out
#
# ============================================================================
# PRE-REGISTERED SCORING of the amplitude / diffusion-coefficient pair.
# Written BEFORE the snapshots exist, so the protocol is fixed in advance.
# ============================================================================
#
# THE PAIR, one variable apart:
#   9401841  nemolev_trp_kappanemo_d90   kappa fix only          (alpha_tke=30)
#   9405182  nemolev_trp_alphatke_d90    kappa fix + alpha_tke=1
# Between their two commits the ONLY behavioural change is alpha_tke 30 -> 1;
# the rest of the diff is auditors, tests and docstrings, inert in a forward
# run. Both carry --tke-mxl-choice 3 explicitly on the command line, so the
# card default moving to 4 afterwards does not disturb either.
#
# BASELINE for both: nemolev_trp_mlefix_d90, Southern Ocean MLD bias +19.2 m
# at day 90 vs NEMO month 3, corr 0.257.
#
# PRE-REGISTERED, band antarctic_S_of_45S, day-90 snapshot vs NEMO month 3,
# compare_omip_nemo.py --res-deg 1.0 (the protocol that produced +19.2):
#   CONFIRMED  : bias <= +8 m
#   REFUTED    : bias >= +16 m
#   INDETERMINATE: +8 to +16 m
#
# THE RISK, pre-registered and reported either way: both fixes REDUCE mixing,
# and the offline one-hour test showed alpha_tke=1 overshooting -- our
# diffusivity fell 22% over the hour where NEMO's rose 5%. A too-SHALLOW
# Southern Ocean, or a tropical warm bias returning (prognostic TKE had
# removed +0.55 -> +0.12 C), are both live outcomes. The matched day-5
# diagnostic already shows the alpha arm warmer by 0.24 in the UNWEIGHTED cell
# mean -- which is not a global mean on a tripolar grid and is not quoted as
# one; these area-weighted snapshot scores are the real test.
#
# Day 30 is scored too, because the offline work says the two fixes should be
# nearly indistinguishable early and separate as convection develops.
set -uo pipefail
REPO=/burg-archive/glab/users/pg2328/legoESM
cd $REPO || exit 1
export PYTHONPATH="$REPO/src:$(ls -d $REPO/packages/*/ | sed 's:/$::' | tr '\n' ':')$REPO"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=$REPO/results/omip_nemo
GT=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_REF/ORCA1_1m_20000101_20041231_grid_T.nc
echo "HEAD: $(git -C $REPO log --oneline -1)"
NFAIL=0
run () { echo; echo "########## $1"; shift; "$@"; local rc=$?; echo "=== rc=$rc"; [ $rc -eq 0 ] || NFAIL=$((NFAIL+1)); }

for pair in "kappanemo:kappa-only (alpha_tke=30)" "alphatke:kappa + alpha_tke=1"; do
  tag="${pair%%:*}"; lbl="${pair#*:}"
  for D in 0030:1 0090:3; do
    day="${D%%:*}"; mon="${D##*:}"
    snap=$R/nemolev_trp_${tag}_d90/snapshot_day${day}.npz
    if [ -r "$snap" ]; then
      run "$lbl  day ${day#00} vs NEMO month $mon" \
        $PY -u scripts/validate/compare_omip_nemo.py \
          --legoesm-snapshot "$snap" --nemo-gridt "$GT" --nemo-month $mon \
          --res-deg 1.0 --grid-label "tripole ($lbl, d${day#00})" \
          --output-dir $R/compare_${tag}_d${day#00}_m0$mon
    else
      echo; echo "########## SKIP $lbl day ${day#00}: no $snap yet"
    fi
  done
done

# Three-way: the two arms against each other AND NEMO, for the near-land bands
# compare_omip_nemo does not report.
A=$R/nemolev_trp_kappanemo_d90/snapshot_day0030.npz
B=$R/nemolev_trp_alphatke_d90/snapshot_day0030.npz
if [ -r "$A" ] && [ -r "$B" ]; then
  run "pair vs NEMO, near-land subdomains, day 30" \
    $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
      --tripole "$A" --mpas "$B" \
      --label-tripole "kappa only" --label-mpas "kappa + alpha_tke=1" \
      --nemo-gridt "$GT" --nemo-month 1 --res-deg 1.0 \
      --out-dir $R/pair_kappa_alpha_d30
fi

echo; echo "=== ALL DONE (failed stages: $NFAIL) ==="
exit $((NFAIL > 0))
