#!/bin/bash -l
#SBATCH --job-name=cube_census
#SBATCH --account=bb1596
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=cube_census.%j.log
# ===========================================================================
# WHY THIS EXISTS. The cubed-sphere lane gives every device the SAME local
# tile (96 x 96 columns x 26 levels) in three of its CPU rows, so those
# three rows are a controlled weak-scaling triple. Ideal weak scaling is a
# flat wall clock. Measured instead, and reproduced in BOTH precisions:
#
#   tiles/face   devices   ms/step f32   ms/step f64   cost vs kt=2
#            2        24         258.4         470.2           1.00
#            3        54        1282.2        2414.0      4.96/5.13
#            4        96         924.6        1718.7      3.58/3.66
#
# The compiled-step census on those exact executables shows the neighbour
# exchanges barely moving (396 -> 416 -> 440 collective-permutes) while
# twelve all-to-alls become twelve (then sixteen) all-gathers exactly at
# the tile count where the cost appears. An all-gather hands every member
# of its group the whole group's data; an all-to-all hands it one slice.
#
# A first census run at resolution 96 did NOT reproduce that mix (it gave
# six gathers and eight all-to-alls at three tiles per face), so the
# lowering is resolution-dependent and the attribution must be made at the
# resolutions the timed rows actually ran.
#
# (a) NUMBER: per collective family, the instruction count and the
#     operand/result SHAPES and replica groups, at the three timed
#     configurations -- 192/kt2, 288/kt3, 384/kt4, nlev 26 -- i.e. the
#     weak triple at a fixed 96 x 96 local tile.
# (b) THIS PROBE DECIDES NOTHING ABOUT CAUSE. Both reviewers rejected an
#     earlier version of this header that promised to confirm or refute
#     the all-gather attribution from a byte ratio: result bytes are not
#     network bytes, a 5x rise would not confirm the attribution and a
#     flat ratio would not refute a latency or placement cost. What this
#     probe is FOR is the descriptive question both reviewers asked for
#     first -- which tensors are gathered, over which replica groups, and
#     whether the sharded extent is divisible by the tile count. The
#     causal question is answered by a separate timed trace.
# (c) Offline: compile only, virtual CPU devices in one process. No
#     allocation, no GPU hour, and the timed rows already exist.
#
# KNOWN-ANSWER CHECK: the kt=2 census must reproduce the timed row's
# 12 all-to-all / 0 all-gather / 396 collective-permute. If it does not,
# this probe is compiling a different executable and NOTHING here may be
# quoted.
#
# SUBMIT: sbatch scripts/cluster/scaling_levante/cube_collective_census.sbatch
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR" || exit 2
REPO="$SUBMIT_DIR"
PP="$REPO/src"
for p in atmosphere core coupler ice land ml ocean tools; do
  PP="$PP:$REPO/packages/$p"
done
export PYTHONPATH="$PP:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
PY=/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python
OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/cube_census_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"

set -- "2 192" "3 288" "4 384"
for arm in "$@"; do
  set -- $arm; KT=$1; RES=$2
  ND=$((6 * KT * KT))
  echo "=== kt=$KT res=$RES devices=$ND"
  XLA_FLAGS=--xla_force_host_platform_device_count=$ND \
    "$PY" scripts/tmp/_cube_collective_shapes.py \
      --kt "$KT" --resolution "$RES" --nlev 26 2>&1 \
    | grep -v "UserWarning\|jnp.array(offsets"
done | tee "$OUTDIR/cube_census.log"

echo "--- known-answer gate"
"$PY" - "$OUTDIR" <<'EOF'
import json, sys, re
log = open(f"{sys.argv[1]}/cube_census.log").read()
rows = [json.loads(m) for m in re.findall(r"^SUMMARY (.*)$", log, re.M)]
for r in rows:
    print(r["kt"], r["counts"], {k: round(v / 1e6, 3)
                                 for k, v in r["result_bytes"].items()})
missing = [kt for kt in (2, 3, 4)
           if not any(r["kt"] == kt for r in rows)]
if missing:
    print(f"KNOWN_ANSWER_FAILED: arms missing for kt={missing} — a partial "
          "census is not the weak triple this job exists to describe")
    sys.exit(4)
k2 = next(r for r in rows if r["kt"] == 2)
want = {"all-to-all": 12, "all-gather": 0, "collective-permute": 396}
got = {k: k2["counts"].get(k) for k in want}
if got != want:
    print(f"KNOWN_ANSWER_FAILED: kt=2 census {got} != timed row {want} "
          "-- different executable, quote NOTHING")
    sys.exit(4)
print("KNOWN_ANSWER_OK")
for r in rows:
    tot = sum(r["result_bytes"].values())
    # Descriptive only: result bytes are NOT network bytes, so this
    # number ranks the arms by how much collective RESULT they
    # materialise and says nothing about which arm waits longer.
    print(f"kt={r['kt']} collective result bytes/step {tot/1e6:.2f} MB "
          "(descriptive, not a cost)")
EOF
GATE_RC=$?
echo "CENSUS_EXIT=$GATE_RC"
exit $GATE_RC
