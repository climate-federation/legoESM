#!/bin/bash
#SBATCH --job-name=trpfwch
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/trpfwch_%j.out
#
# DOES THE SALINITY RESTORING CHANNEL EXPLAIN FESOM'S COASTAL SALINITY EXCESS?
#
# WHY THIS ARM EXISTS. FESOM's day-30 surface salinity scores rmse 0.465
# against the ORCA1 oracle where the tripole scores 0.258, the two arms' mean
# offsets agree to 7e-4 so it is scatter, and 86% of the gap sits in the single
# ring of target cells touching the edge of the scored ocean. Four measurements
# then removed the scoring explanation: smoothing every source to a common
# 330 km left the ratio at 1.80-1.85 while SST's fell 1.18 -> 1.07 (job
# 9934795); the oracle's own field carried through FESOM's mesh accounts for
# only 6% of FESOM's error (9935164); narrowing each arm to its single nearest
# source removed NONE of FESOM's coastal error while costing the tripole 9.6%
# (9935186); and inside that ring the excess survives at the CLOSEST
# nearest-node distances, 1.5383 against 0.9358 where the node sits within
# 9 km of the target-cell centre and a barrier crossing is unavailable
# (9935199). A coastal model difference now carries the weight.
#
# THE LEVER. Of everything the two lanes do differently, one acts ON SALINITY
# and diverges HARDEST AT COASTS: FESOM restores surface salinity through a
# FRESHWATER FLUX, the tripole through a VIRTUAL SALT edit. The two agree only
# where salinity is near the reference the virtual-salt form divides by; at
# river mouths and on fresh shelves they differ by the ratio of local to
# reference salinity. That is a coastal signature by construction.
#
# THE ARM IS TWO FLAGS, NOT ONE, AND THE DRIVER MAKES THAT CHOICE, NOT ME.
# `--sss-restore-channel water_flux` on the tripole REQUIRES
# `--freshwater-closure real_freshwater` (run_omip_core2.py:8006): the default
# virtual-salt closure builds its salt tendency from the NET freshwater, which
# already contains the restoring term, so routing restoring as water without
# switching the closure would apply it TWICE -- once as volume through z-star
# and again as virtual salt. FESOM is exempt from that guard precisely because
# its lane has no virtual-salt closure at all; its freshwater is real by
# construction. So the two flags are one physical choice on this lane, and
# "the tripole on FESOM's restoring channel" necessarily means "on FESOM's
# freshwater treatment". `--sss-restore-normalization live_s` is a third flag
# and is likewise forced: water_flux refuses the default s_target conversion,
# which would move water at the virtual-salt denominator's rate.
#
# WHAT THAT COSTS, STATED UP FRONT. If this arm MOVES, the channel and the
# closure cannot be separated by it, because the driver will not run them
# apart. Separating them needs a second arm with the closure changed alone,
# which is legal (the guard only fires for water_flux) and is NOT launched
# here -- it is only worth its GPU hours if this one moves. If this arm does
# NOT move, both are exonerated together and the lever closes on one run.
#
# PRE-REGISTERED, on day-30 surface-salinity rmse against the oracle scored by
# compare_three_way_nemo at the matched record, ARM against trp_pinned_d30:
#   arm ring-1 rmse <= 0.55   -> the channel/closure pair is a LEADING term in
#                                the coastal salinity error (the control sits
#                                at 0.6605 and FESOM at 1.3917, so a drop of
#                                this size on the tripole means the treatment
#                                moves coastal salinity by the right order)
#   arm ring-1 rmse >= 0.75   -> the pair is REFUTED as the explanation; it
#                                either does nothing or makes the tripole
#                                worse, and the FESOM excess needs another
#                                lane difference
#   0.55 to 0.75              -> NO VERDICT
# The global figure is reported alongside but is NOT the primary: the whole
# finding is that this error lives in one ring, and a global rmse averages it
# against 24315 interior cells that carry 0.9% of the gap.
#
# BUILT FROM THE CONTROL'S OWN COMMAND LINE, not written fresh -- a card
# written from scratch for this campaign once NaN'd and cost a 78-minute run
# plus eight arms. `_arm_trp_fwchan_d30.txt` is `_arm_trp_pinned_d30.txt` with
# the output directory redirected and six tokens appended, 119 -> 125, with
# the count, the single --output and the absence of any --restart asserted at
# build time.
#
# RUNS FROM _omip_merge, THE TREE THE CONTROL RAN FROM. Reproducing a run means
# reproducing its tree, not just its flags; main's driver is older and dies on
# thirteen unrecognised arguments. This card lives in _fluxacc_wt and only
# READS that tree, so it adds nothing to a checkout with a job already in it.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
# The control's manifest names the tree it ran in; if this is not that tree the
# comparison is two lineages again, which is the exact confound that forced the
# control to be re-run in the first place.
CTL=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_pinned_d30/run_manifest.json
grep -q "_omip_merge" "$CTL" || { echo "FATAL: control manifest does not name _omip_merge"; exit 1; }

export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_ENABLE_X64=1
CMD=$(cat /burg-archive/glab/users/pg2328/legoESM/omip_nemo/_arm_trp_fwchan_d30.txt)
echo "CMD: $PY scripts/run/run_omip_core2.py $CMD"
$PY -u scripts/run/run_omip_core2.py $CMD
rc=$?
echo "=== card exit: $rc ==="
ls /burg-archive/glab/users/pg2328/legoESM/results/omip_nemo/trp_fwchan_d30/*.npz 2>/dev/null
# A job's exit status is its LAST command's. This card ends on the run's status
# so a died-at-day-0 run cannot be recorded COMPLETED, which happened once.
exit $rc
