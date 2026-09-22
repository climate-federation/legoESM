#!/bin/bash
#SBATCH --job-name=d90repro
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/d90repro_%j.out
#
# DOES THE 90-DAY RUN ACTUALLY REPRODUCE THE 30-DAY CONTROL?
#
# The 90-day run's card claims it is "the same configuration as trp_faithz0_d30
# run three times as long". Nothing in this session has tested that claim, and
# every conclusion the 90-day run is meant to support rests on it: if the two
# runs differ over days that they BOTH integrate, then the day 35-90 numbers
# are not an extension of the 30-day series, they are a different experiment.
#
# The test became available for free once the run's real provenance was read.
# The 90-day run resumes from trp_faithz0_d30/restart_leg1.npz, which is step
# 8640 = day 15 -- NOT the day-0 state the card asserts. So both runs integrate
# days 15 to 30 from the SAME initial state with the SAME settings, and both
# write a day-20 snapshot. Those two files should agree.
#
# WHAT WOULD CONFIRM AND WHAT WOULD REFUTE, pre-registered:
#   * bitwise identical, or agreement at the 1e-12 level on every field:
#     the configuration claim holds and the 90-day series extends the 30-day
#     one. Small non-zero differences are expected from GPU non-determinism
#     (atomics in scatter/reduce) even at identical settings, so bitwise is
#     the hoped-for outcome and NOT the pass condition.
#   * differences that are small but far above rounding, say 1e-6 relative on
#     a prognostic field: the runs share a configuration but not a code path
#     or a reduction order, which is worth knowing before day-90 numbers are
#     compared term-by-term against day-30 ones.
#   * a field differing at O(1) relative, or a field present in one file and
#     absent from the other: the claim is FALSE and the 90-day run is a
#     different experiment. Everything downstream of it is confounded.
#
# The comparison is over the union of keys, so a field appearing in only one
# file is reported rather than silently skipped -- a missing key is exactly
# the kind of difference that a per-field loop over the intersection hides.
set -uo pipefail
M=/burg-archive/glab/users/pg2328/_fluxacc_wt; cd $M || exit 1
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
A=$R/trp_faithz0_d30/snapshot_day0020.npz
B=$R/trp_faithz0_d90/snapshot_day0020.npz
for f in "$A" "$B"; do [ -r "$f" ] || { echo "FATAL: missing $f"; exit 1; }; done

$PY -u - "$A" "$B" <<'EOF'
import sys
import numpy as np

a_path, b_path = sys.argv[1], sys.argv[2]
A = np.load(a_path, allow_pickle=True)
B = np.load(b_path, allow_pickle=True)
ka, kb = set(A.files), set(B.files)
print(f"d30 file: {a_path}")
print(f"d90 file: {b_path}")
print(f"keys: d30 {len(ka)}, d90 {len(kb)}")
only_a, only_b = sorted(ka - kb), sorted(kb - ka)
if only_a or only_b:
    print(f"KEY MISMATCH  only-in-d30={only_a}  only-in-d90={only_b}")

worst = 0.0
worst_key = None
rows = []
for k in sorted(ka & kb):
    x, y = np.asarray(A[k]), np.asarray(B[k])
    if x.shape != y.shape:
        rows.append((k, "SHAPE", str(x.shape), str(y.shape), float("inf")))
        worst, worst_key = float("inf"), k
        continue
    if x.dtype.kind not in "fiu":
        rows.append((k, str(x.dtype), "", "", 0.0 if np.array_equal(x, y) else float("inf")))
        continue
    xf, yf = x.astype(np.float64), y.astype(np.float64)
    fin = np.isfinite(xf) & np.isfinite(yf)
    nan_mismatch = int(np.sum(np.isfinite(xf) != np.isfinite(yf)))
    d = np.abs(xf[fin] - yf[fin])
    amax = float(d.max()) if d.size else 0.0
    scale = float(np.abs(xf[fin]).max()) if d.size else 0.0
    rel = amax / scale if scale > 0 else amax
    nd = int(np.sum(d > 0))
    rows.append((k, f"{amax:.3e}", f"{rel:.3e}", f"{nd}/{d.size}", rel))
    if nan_mismatch:
        print(f"  !! {k}: {nan_mismatch} cells finite in one file and not the other")
    if rel > worst:
        worst, worst_key = rel, k

print(f"{'field':28s} {'max|diff|':>12s} {'rel':>12s} {'differing':>18s}")
for k, c1, c2, c3, _ in rows:
    print(f"{k:28s} {c1:>12s} {c2:>12s} {c3:>18s}")

print()
print(f"WORST RELATIVE DIFFERENCE: {worst:.3e} on {worst_key}")
if worst == 0.0 and not only_a and not only_b:
    print("VERDICT: BITWISE IDENTICAL -- the two runs share day 15-30 exactly.")
elif worst < 1e-12:
    print("VERDICT: identical to rounding; same configuration, same code path.")
elif worst < 1e-6:
    print("VERDICT: same configuration, but the agreement is looser than "
          "rounding -- reduction order or a non-deterministic kernel differs.")
else:
    print("VERDICT: THE RUNS DIFFER. The 90-day run is NOT the 30-day "
          "configuration run longer, and its numbers are not comparable "
          "term-by-term with the 30-day series.")
EOF
rc=$?
echo "=== card exit: $rc ==="
exit $rc
