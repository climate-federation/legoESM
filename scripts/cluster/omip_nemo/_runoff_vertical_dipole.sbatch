#!/bin/bash
#SBATCH --job-name=rnfdipole
#SBATCH --account=glab
#SBATCH --partition=glab1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/rnfdipole_%j.out
#
# MISPLACED RIVER WATER, OR A DIFFERENT AMOUNT OF IT?
#
# Two candidates are left for FESOM's coastal surface-salinity excess and both
# act from t=0: the initial condition interpolated onto its mesh, and where
# the lane puts river runoff. Every horizontal statistic confounds them,
# because river cells carry the ocean's sharpest salinity gradients and are
# therefore the hardest cells for any interpolation. Two attempts to separate
# them geographically came out VACUOUS -- the runoff-free stratum held 1.5%
# then 1.9% of the ring's area, too small to refute anything. Both bands were
# withdrawn (90111a318, and this card's predecessor).
#
# WHY THAT KEPT FAILING, correcting my own first diagnosis: I blamed the
# 2.5-degree inverse-distance smear and switched to nearest-source runoff. It
# moved the stratum from 289 to 317 cells. The real reason is that NEMO's
# Dai-Trenberth file is nonzero on nearly every coastal cell to begin with --
# it carries ice-shelf and iceberg melt as well as rivers. There is almost no
# river-free coast to stratify on, so no horizontal split can ever isolate it.
#
# THE COLUMN SEPARATES THEM, AND IT NEEDS NO ORACLE. Both lanes receive the
# SAME runoff from the SAME file. The tripole spreads it down to NEMO's
# ln_rnf_depth_ini depth, up to 150 m; FESOM cannot and puts all of it in the
# surface cell. So if vertical placement is the whole story, FESOM is fresher
# at the surface and saltier just below, and the column integral of the
# difference CANCELS -- no water was added, only moved. An initial-condition
# difference means the columns hold different amounts of salt, and nothing
# cancels.
#
# Comparing our two lanes directly also sidesteps the salinity-convention
# difference between the models and the NEMO output, which would otherwise sit
# inside any absolute anomaly.
#
# PRE-REGISTERED, on the cancellation ratio at river-mouth columns -- the
# column-integrated difference divided by the integral of its absolute value,
# both over 0 to 150 m. It is 0 for pure redistribution and 1 for a net
# difference no rearrangement can explain:
#   * cancel <= 0.35 AND the surface lobe negative while the subsurface lobe
#     is positive  ->  RUNOFF PLACEMENT LEADS. The lanes hold the same water,
#     put in different places, and wiring the depth spread into the FESOM lane
#     becomes the fix to try.
#   * cancel >= 0.65  ->  RUNOFF PLACEMENT IS REFUTED as the leading term. The
#     columns hold different amounts of salt, which vertical placement cannot
#     produce, and the initial condition carries the gap.
#   * 0.35 to 0.65  ->  NO VERDICT, both contribute.
#
# THE CONTROL IS IN THE SAME TABLE AND IT IS WHY THIS IS NOT ANOTHER VACUOUS
# BAND: the identical two numbers are computed on LOW-RUNOFF coastal columns,
# which are half the wet coast by count rather than 2% of it. Neither the
# spreading lever nor the river water exists there, so under the runoff
# hypothesis that row must show NO dipole. A dipole in BOTH rows means the
# structure is not about rivers and the verdict above does not stand whatever
# the river row says. Read the control first.
#
# BOTH MARKS. Day 5 is the cleaner one -- closest to t=0, where an inherited
# difference has had least time to be mixed away. Day 30 says whether the
# structure persists.
set -uo pipefail
ROOT=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $ROOT || exit 1
echo "HEAD: $(git -C $ROOT log --oneline -1)"
echo "dirty: $(git -C $ROOT status --porcelain | wc -l)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python

R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
RUNOFF=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/INPUTS/orca1_inputs/data_repository/input_fields/runoff-icb_DaiTrenberth_Depoorter.nc

echo "=== probe unit tests ==="
JAX_ENABLE_X64=1 $PY -m pytest -q tests/validate/test_runoff_vertical_dipole.py 2>&1 | tail -15
rc=${PIPESTATUS[0]}
[ "$rc" -eq 0 ] || { echo "FATAL: probe tests are red (rc=$rc) -- refusing to measure"; exit 1; }

[ -r "$RUNOFF" ] || { echo "FATAL: no runoff file at $RUNOFF"; exit 1; }

rc=0
for DAY in 0005 0030; do
  FES=$R/fesom_orca1faithtint_d30/snapshot_day${DAY}.npz
  TRP=$R/trp_pinned_d30/snapshot_day${DAY}.npz
  [ -r "$TRP" ] || TRP=$R/trp_pinned_d30/snapshot_final.npz
  if [ ! -r "$FES" ] || [ ! -r "$TRP" ]; then
    echo "FATAL: missing a snapshot for day $DAY"; rc=1; continue
  fi
  # The tripole stamps its day; FESOM stamps nothing, so its day is the
  # filename and that is stated rather than assumed.
  $PY - "$TRP" "$DAY" <<'GATE' || { echo "FATAL: tripole file is not day $DAY"; rc=1; continue; }
import sys
import numpy as np
d = np.load(sys.argv[1], allow_pickle=True)
want = float(sys.argv[2])
if "time_days" not in d.files:
    print(f"  {sys.argv[1]}: NO TIME STAMP")
    sys.exit(1)
got = float(np.asarray(d["time_days"]).ravel()[0])
print(f"  tripole day stamp {got:g} (want {want:g})")
sys.exit(0 if abs(got - want) < 1e-6 else 1)
GATE
  echo
  echo "===== day $DAY ====="
  echo "  FESOM day taken from the filename, unverified (the lane stamps none)"
  JAX_ENABLE_X64=1 $PY -u scripts/validate/ocean_fidelity/runoff_vertical_dipole.py \
    --fesom "$FES" --tripole "$TRP" --runoff-file "$RUNOFF" || rc=1
done
echo "=== card exit: $rc ==="
exit $rc
