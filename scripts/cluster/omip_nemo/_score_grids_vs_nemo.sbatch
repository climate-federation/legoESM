#!/bin/bash
#SBATCH --job-name=score_grids
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/burg-archive/glab/users/pg2328/legoESM/omip_nemo/scoregrids_%j.out
#
# ONE NEMO comparison for all three grids, on ONE cell set.
#
# The three-way scorer has two legoESM slots, so three grids take three pair
# runs.  Every run passes the third snapshot as --also-mask, which intersects
# its wet footprint (and MLD coverage) into the common mask without scoring
# it, so the three reports share exactly one cell set.  The check that they
# do is built in: every grid-vs-NEMO score appears in two of the pair runs
# and the two copies must agree bit for bit, and this card FAILS if not.
#
# Same protocol for every grid, no per-grid options: GATEWAY 5-day grid_T,
# 0-based record DAY/5-1 (the window ENDING at the snapshot day; measured
# from time_counter_bounds 2026-09-15), 1-degree target, nearest-wet mask,
# dsigma=0.01 MLD, no SST clamp (the older one-arm cards floored SST at
# -1.8 C with --freeze-clamp-C, which is not comparable).
#
# Usage: sbatch --export=ALL,DAY=30,TRP=trp_base2_d30,MPAS=mpas_base2_d30,FESOM=fesom_base2_d30 \
#            scripts/cluster/omip_nemo/_score_grids_vs_nemo.sbatch
#   TRP / MPAS / FESOM are run directories under results/omip_nemo; each must
#   hold snapshot_day<DAY>.npz (or snapshot_final.npz when DAY is that run's
#   target day -- pass FINAL_OK=1 to allow the fallback for ALL three, since a
#   run's snapshot_final is its target day, not necessarily DAY).
set -euo pipefail   # -e: a failed pair run or check FAILS the job
ROOT=/burg-archive/glab/users/pg2328/_omip_merge; cd $ROOT || exit 1
echo "HEAD: $(git log --oneline -1)"
export PYTHONPATH="$ROOT/src:$(ls -d $ROOT/packages/*/ | sed 's:/$::' | tr '\n' ':')$ROOT:$ROOT/scripts/validate"
PY=/burg-archive/glab/users/jn2808/.conda/envs/legoesm/bin/python
export JAX_PLATFORMS=cpu
R=/burg-archive/glab/users/pg2328/legoESM/results/omip_nemo
GW=/burg-archive/glab/users/pg2328/nemo_orca1/nemo_5.0.1/cfgs/ORCA1/EXP00/RUN_GATEWAY/ORCA1_5d_20000101_20000331_grid_T.nc
DAY=$(( 10#${DAY:-30} ))   # 10#: a zero-padded DAY would parse as octal
[ $(( DAY % 5 )) -eq 0 ] && [ "$DAY" -ge 5 ] || { echo "FATAL: DAY=$DAY is not a positive multiple of 5"; exit 1; }
REC=$(( DAY / 5 - 1 ))
TAG=$(printf 'day%04d' "$DAY")
TRP=${TRP:-trp_base2_d30}; MPAS=${MPAS:-mpas_base2_d30}; FESOM=${FESOM:-fesom_base2_d30}
snap() {  # run dir -> snapshot path for DAY
  local f=$R/$1/snapshot_$TAG.npz
  if [ ! -r "$f" ] && [ "${FINAL_OK:-0}" = 1 ]; then f=$R/$1/snapshot_final.npz; fi
  [ -r "$f" ] || { echo "FATAL: $1 has no snapshot for day $DAY ($f)" >&2; return 1; }
  echo "$f"
}
ST=$(snap "$TRP") || exit 1; SM=$(snap "$MPAS") || exit 1; SF=$(snap "$FESOM") || exit 1
OUT=$R/score_grids_d${DAY}_${TRP#trp_}
mkdir -p "$OUT"
echo "=== inputs; oracle record $REC (0-based) = days $((DAY-5))-$DAY ==="
ls -l --time-style=+%Y-%m-%d_%H:%M "$ST" "$SM" "$SF"

run_pair() {  # label_a snap_a label_b snap_b mask_snap subdir
  echo; echo "########## $1 vs $3, day $DAY, GATEWAY record $REC ##########"
  rm -rf "$OUT/$6"   # a failed run must not pass the check on a stale report
  $PY -u scripts/validate/ocean_fidelity/compare_three_way_nemo.py \
    --tripole "$2" --label-tripole "$1" \
    --mpas "$4" --label-mpas "$3" \
    --also-mask "$5" --expect-day "$DAY" \
    --nemo-gridt "$GW" --nemo-time-idx "$REC" --res-deg 1.0 \
    --out-dir "$OUT/$6" 2>&1 | grep -vE "Warning|warn" | tail -45
  [ -s "$OUT/$6/report.json" ] || { echo "FATAL: pair $6 wrote no report"; exit 1; }
}
run_pair tripole "$ST" MPAS   "$SM" "$SF" trp_mpas
run_pair tripole "$ST" FESOM2 "$SF" "$SM" trp_fesom
run_pair MPAS    "$SM" FESOM2 "$SF" "$ST" mpas_fesom

echo; echo "=== shared-cell-set check and per-grid headline (global, vs NEMO) ==="
if ! OUT="$OUT" $PY - <<'PY'
import json, os, pathlib, sys
out = pathlib.Path(os.environ["OUT"])
rep = {k: json.load(open(out / k / "report.json")) for k in ("trp_mpas", "trp_fesom", "mpas_fesom")}
n = {k: r["n_common_cells"] for k, r in rep.items()}
print("common cells per pair run:", n)
ok = len(set(n.values())) == 1
# Cell IDENTITY, not just count: the per-field digest of the scored mask
# must be one value across the three runs.
for fld in rep["trp_mpas"]["fields"]:
    digests = {k: r["scored_mask_sha1"][fld] for k, r in rep.items()}
    same = len(set(digests.values())) == 1
    ok &= same
    print(f"  {fld:4s} scored-cell-set digest identical across pair runs: {same}")
# Every grid-vs-NEMO score appears in two pair runs (once per slot); each
# pair of appearances must be bit-identical, or the runs did not share a cell set.
for fld in rep["trp_mpas"]["fields"]:
    for key, r1, r2 in (("tripole_vs_nemo", "trp_mpas", "trp_fesom"),
                        ("MPAS_vs_nemo", "trp_mpas", "mpas_fesom"),
                        ("FESOM2_vs_nemo", "trp_fesom", "mpas_fesom")):
        a = rep[r1]["fields"][fld][key]["global"]
        b = rep[r2]["fields"][fld][key]["global"]
        same = all(a[k] == b[k] for k in ("bias", "rmse", "corr"))
        ok &= same
        print(f"  {fld:4s} {key:15s} identical across pair runs: {same}")
for k, r in rep.items():
    for lab, chk in r.get("node_cloud_check", {}).items():
        print(f"  [node-cloud] {k}: {lab} carries no land; nearest node over scored cells "
              f"p50 {chk['nearest_node_p50_km']:.0f} km, p99 {chk['nearest_node_p99_km']:.0f} km, "
              f"max {chk['nearest_node_max_km']:.0f} km; {chk['n_far']}/{chk['n_scored']} cells "
              f"beyond {chk['far_factor']:g}x the median node spacing ({chk['median_node_spacing_km']:.0f} km)")
for fld in rep["trp_mpas"]["fields"]:
    print(f"  {fld:4s}", end="")
    for lab, key, r in (("tripole", "tripole_vs_nemo", "trp_mpas"),
                        ("MPAS", "MPAS_vs_nemo", "trp_mpas"),
                        ("FESOM2", "FESOM2_vs_nemo", "trp_fesom")):
        g = rep[r]["fields"][fld][key]["global"]
        print(f"  {lab}: bias {g['bias']:+.4f} rmse {g['rmse']:.4f}", end="")
    print()
if not ok:
    print("FATAL: the pair runs did not score the same cell set"); sys.exit(1)
print("MASK-CONSISTENT: all three pair runs scored one cell set")
PY
then echo "FAILED: shared-cell-set check"; exit 1; fi
echo "OUT=$OUT"
echo DONE
