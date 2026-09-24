#!/bin/bash -l
#SBATCH --job-name=latlon_band_vs_tile
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --output=latlon_band_vs_tile_%j.log
# ===========================================================================
# DOES A 2-D TILING REMOVE THE LAT-LON CPU SCALING FLOOR?
#
# The lane does not scale on CPU past ~128 ranks. Strong scaling at
# 4096x8192x26 single precision, steady-state median ms/step:
#     32:2120  64:1387  128:1110  256:1015  512:997
# 16x the hardware buys 2.1x. Fitting T = P/n + C*log2(n) gives C = 99 ms per
# doubling with essentially no residual, so ~86-90% of the 512-rank step is
# cost that does not shrink with rank count.
#
# The mechanism is read off the code and the path is proven to execute: with
# --p-lon 1 (the default, and what the production ladder passes) the device
# mesh is 1-D over latitude, so every rank owns a band spanning the WHOLE
# longitude circle and its halo exchange ships two complete rows. That payload
# is halo*n_lon*nlev at 8 ranks and the same at 512, while local work falls as
# 1/n.
#
# The alternative is already built and nothing selects it. --p-lon p makes the
# mesh 2-D ("lat","lon") of shape (n/p, p) and routes the halo through the
# tiled body, which ships tile boundaries. The flag's own help text records
# that at 128 ranks a 16x8 tiling moves 1,280 boundary cells per rank against
# 8,192 for a band.
#
# THE NOCOMM CONTROLS, AND WHY THERE ARE TWO. LEGOESM_LATLON_HALO_NOCOMM
# deliberately breaks the answer: each rank keeps its own edge rows instead of
# the neighbour's. '1' replaces the collective with an optimization barrier
# (no wire traffic); 'wire' RUNS the collective and discards the result, so it
# pays the wire time while following the SAME wrong trajectory as '1'. The
# module docstring is explicit that full-minus-'1' is only an UPPER BOUND,
# because the two arms carry different fields after the first step and their
# local work can therefore differ too, while 'wire'-minus-'1' differs by the
# collective ALONE and is the number to quote as wire time. Both arms are
# numerically meaningless as physics.
#
# ALL FOUR ARMS RUN AT THE SAME PER-RANK CORE SHARE (4 ranks/node, 64 threads),
# which is the production layout and already measured to be the best of 8/16/
# 32/64.
# ===========================================================================
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

N_LAT=4096
N_LON=8192
NLEV=26
STEPS=12
WARMUP=3
# Worst case is 12 arms; at this cap that is 4 h against a 6 h wall.
ARM_TIMEOUT=1200
# The band arms are the KNOWN ANSWER. Anything further off than this and no
# ratio may be read off this job.
WALL_TOL=0.15
BAND_REF_128=1110
BAND_REF_512=997

# Closest-to-square factorization with an EVEN lon split (the packed tiled
# body raises on an odd one) subject to n_lat % p_lat == 0, n_lon % p_lon == 0.
P_LON_128=8      # 16 x 8
P_LON_512=32     # 16 x 32

OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/latlon_bvt_j${SLURM_JOB_ID}}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"

# Both gates default ON in the library (read as != "0"), so they are set
# IDENTICALLY on every arm rather than varied -- a difference here would be a
# second variable inside an A/B that is supposed to isolate the decomposition.
export LEGOESM_LATLON_PACKED_EXCHANGE=1
export LEGOESM_LATLON_SPMD_FUSED_HALO=1

echo "== header =="
echo "sha       $(git -C "$REPO" rev-parse --short HEAD)"
echo "py        $PY"
echo "provenance $($PY -c 'import legoesm.parallel.latlon_spmd as m; print(m.__file__)')"
echo "outdir    $OUTDIR"
echo "problem   ${N_LAT}x${N_LON}x${NLEV} strong, steps=$STEPS warmup=$WARMUP"
echo "layout    4 ranks/node, 64 threads/rank (production)"
echo "env       LEGOESM_LATLON_PACKED_EXCHANGE=1 LEGOESM_LATLON_SPMD_FUSED_HALO=1 (both on for EVERY arm)"
echo "tiling    128 ranks -> $((128 / P_LON_128)) x ${P_LON_128};  512 ranks -> $((512 / P_LON_512)) x ${P_LON_512}"
echo "reference band ${BAND_REF_128} ms at 128, ${BAND_REF_512} ms at 512, tolerance ${WALL_TOL}"

want=0; ok=0

run_arm() {   # $1=tag $2=nd $3=p_lon $4=nocomm('0'|'1'|'wire') $5=rep
    local tag=$1 nd=$2 plon=$3 nocomm=$4 rep=$5
    local nn=$(( nd / 4 ))
    want=$(( want + 1 ))
    echo "=== ${tag} rep${rep} nd=${nd} p_lon=${plon} nocomm=${nocomm} $(date +%T)"
    if timeout --kill-after=30 "$ARM_TIMEOUT" srun --export=ALL \
            --nodes="$nn" --ntasks="$nd" --ntasks-per-node=4 \
            --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 \
            env LEGOESM_LATLON_HALO_NOCOMM="$nocomm" \
            "$PY" scripts/bench/bench_atm_latlon_spmd_scaling.py \
            --multicontroller --n-devices "$nd" --mode strong \
            --n-lat "$N_LAT" --n-lon "$N_LON" --nlev "$NLEV" \
            --p-lon "$plon" --steps "$STEPS" --warmup "$WARMUP" \
            --out "$OUTDIR/${tag}_r${rep}.jsonl"; then
        ok=$(( ok + 1 ))
    else
        echo "ARM FAILED ${tag} rep${rep} rc=$? (124 = timeout)"
    fi
}

# Largest first, and the arm ORDER alternates between repeats: running a fixed
# order lets a warm-up or a thermal drift land on the same arm every time and
# read as an effect.
for rep in 1 2; do
    if [ "$rep" = 1 ]; then
        run_arm band512   512 1              0     "$rep"
        run_arm tile512   512 "$P_LON_512"   0     "$rep"
        run_arm skip512   512 1              1     "$rep"
        run_arm wire512   512 1              wire  "$rep"
        run_arm band128   128 1              0     "$rep"
        run_arm tile128   128 "$P_LON_128"   0     "$rep"
    else
        run_arm wire512   512 1              wire  "$rep"
        run_arm skip512   512 1              1     "$rep"
        run_arm tile512   512 "$P_LON_512"   0     "$rep"
        run_arm band512   512 1              0     "$rep"
        run_arm tile128   128 "$P_LON_128"   0     "$rep"
        run_arm band128   128 1              0     "$rep"
    fi
done

echo "=== arms: $ok of $want completed ==="

$PY - "$OUTDIR" "$BAND_REF_128" "$BAND_REF_512" "$WALL_TOL" \
     "$P_LON_128" "$P_LON_512" <<'PYEOF'
import glob, json, os, sys

out_dir, ref128, ref512, tol, pl128, pl512 = (
    sys.argv[1], float(sys.argv[2]), float(sys.argv[3]),
    float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))


def arm(tag):
    """Mean over the repeats of each file's last steady median, or None."""
    vals = []
    for f in sorted(glob.glob(os.path.join(out_dir, f"{tag}_r*.jsonl"))):
        last = None
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line).get("steady_median_ms")
            except Exception:
                continue
            if v:
                last = float(v)
        if last is not None:
            vals.append(last)
    return (sum(vals) / len(vals), len(vals), max(vals) - min(vals)) if vals else None


tags = ["band128", "tile128", "band512", "tile512", "skip512", "wire512"]
got = {t: arm(t) for t in tags}
for t in tags:
    v = got[t]
    print(f"  {t:<9} " + ("NO DATA" if v is None
                          else f"{v[0]:8.1f} ms  (n={v[1]}, spread {v[2]:.1f})"))

# KNOWN ANSWER, and it must be able to FAIL on a wrong value, not only on a
# missing one. The band arms at both rank counts are the production setting
# and have published medians; if this job cannot reproduce them, its ratios
# describe some other experiment.
suspect = []
for tag, ref in (("band128", ref128), ("band512", ref512)):
    v = got[tag]
    if v is None:
        suspect.append(f"{tag} produced no data")
        continue
    off = abs(v[0] - ref) / ref
    verdict = "ok" if off <= tol else "OUT OF TOLERANCE"
    print(f"\nknown answer {tag}: {v[0]:.0f} ms against {ref:.0f} ms "
          f"recorded, {off * 100:.0f}% off -- {verdict}")
    if off > tol:
        suspect.append(f"{tag} is {off * 100:.0f}% off its recorded median")

if suspect:
    print("\nHARNESS SUSPECT: " + "; ".join(suspect) + ".")
    print("No ratio is read off this job. Resolve before interpreting.")
    print("LATLON_BVT_DONE")
    raise SystemExit(0)

print(f"\ntiling used: 128 ranks {128 // pl128} x {pl128}; "
      f"512 ranks {512 // pl512} x {pl512}")

for nd, b, t in ((128, got["band128"], got["tile128"]),
                 (512, got["band512"], got["tile512"])):
    if b and t:
        print(f"{nd:>4} ranks: band {b[0]:7.1f}  tile {t[0]:7.1f}  "
              f"band/tile {b[0] / t[0]:.2f}x")

skip, wire, band = got["skip512"], got["wire512"], got["band512"]
if skip and wire and band:
    wire_ms = wire[0] - skip[0]
    print(f"\nwire time at 512 ranks: {wire_ms:.1f} ms, which is "
          f"{100 * wire_ms / band[0]:.0f}% of the band step.")
    print(f"  (upper bound from full-minus-skip would say "
          f"{100 * (band[0] - skip[0]) / band[0]:.0f}%, which is NOT the wire "
          f"time -- the skip arm follows a different trajectory, so its local "
          f"work differs too.)")
    print("  Both nocomm arms are numerically WRONG BY DESIGN and are valid "
          "only as a cost measurement.")

if got["band512"] and got["tile512"]:
    s = got["band512"][0] / got["tile512"][0]
    if s >= 1.25:
        print(f"\nTILING WINS: the tile is {100 * (s - 1):.0f}% faster at 512 "
              f"ranks. The one-dimensional decomposition is the floor and the "
              f"fix is to select a two-dimensional mesh.")
    elif s >= 1.10:
        print(f"\nPARTIAL: the tile is {100 * (s - 1):.0f}% faster at 512 "
              f"ranks. Real, but smaller than the halo-payload argument "
              f"predicts, so something else binds as well.")
    elif s >= 0.90:
        print(f"\nTILING NEUTRAL: {s:.2f}x at 512 ranks, inside the "
              f"registered band. The decomposition is not the floor.")
    else:
        print(f"\nTILING LOSES: the tile is {100 * (1 - s):.0f}% SLOWER at "
              f"512 ranks.")
print("LATLON_BVT_DONE")
PYEOF
