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
# WHY DOES THE LAT-LON LANE STOP SCALING ON CPU, AND CAN A 2-D MESH FIX IT?
#
# Strong scaling, 4096x8192x26 single precision, steady-state median ms/step:
#     32:2120  64:1387  128:1110  256:1015  512:997
# 16x the hardware buys 2.1x. Fitting T = P/n + C*log2(n) gives C = 99 ms per
# doubling with essentially no residual constant, so ~86-90% of the 512-rank
# step is cost that does not shrink with rank count.
#
# TWO MECHANISMS EXPLAIN THAT SHAPE AND THEY IMPLY OPPOSITE FIXES.
#
#   PAYLOAD. With --p-lon 1 (the default, and what the production ladder
#   passes) the mesh is 1-D over latitude, so every rank's halo ships two
#   complete longitude circles -- a payload identical at 8 ranks and at 512
#   while local work falls as 1/n. Fix: a 2-D mesh, which ships tile
#   boundaries instead.
#
#   PER-OPERATION COST. Constant bytes produce a CONSTANT term, not a term
#   that grows with log2(n). Growth with rank count is the signature of
#   latency and synchronization depth. A 2-D mesh CUTS BYTES BUT ADDS
#   COLLECTIVES, so under this mechanism tiling is neutral at best and can
#   lose.
#
# The fitted curve prefers the second reading, and the repo's own ballast
# knob docstring already records a measurement pointing the same way (halo
# time doubling between 32 and 64 devices at constant bytes). That number is
# three orders of magnitude below this plateau and was taken on a different
# instrument, so it is treated here as a pointer, not as a fact.
#
# THE DISCRIMINATOR IS THE BALLAST ARM AND IT RUNS FIRST.
# LEGOESM_LATLON_HALO_BALLAST=N sends each halo message N times and drops the
# copies: bytes scale, while the collective count, the schedule, the
# arithmetic AND THE ANSWER do not. It is bit-identical, unlike the nocomm
# knobs. The step's response to N IS the bandwidth term.
#   response proportional to N  -> payload binds  -> tiling should win
#   response flat in N          -> per-operation  -> tiling cannot win by
#                                                    cutting bytes
#
# THE BAND/TILE PAIR IS MATCHED ON PACKING. LEGOESM_LATLON_PACKED_EXCHANGE=1
# is production for the band lane, but packed_exchange_mesh() returns None
# for a 2-D mesh, so the tile arm silently runs UNPACKED. Comparing the
# production band against the tile would therefore compare two things at
# once. A band_unpacked arm is included and IT, not the production band, is
# the tile's control. (The tiled packed body exists in the library and has no
# caller anywhere under packages/ -- it is dead code, which is a finding in
# its own right and is why the confound cannot simply be switched off.)
#
# The nocomm arms ('1' = collective replaced by a barrier; 'wire' = collective
# runs, result discarded) are NUMERICALLY WRONG BY DESIGN. wire-minus-skip
# differs by the collective alone; full-minus-skip is a confounded difference
# whose sign is not established, so it is not reported as a bound.
# ===========================================================================
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
# _env.sh defaults JAX_PLATFORMS to cuda. This is a CPU job on a partition
# with no GPUs, so it is pinned BEFORE the source, which honours an existing
# value.
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

N_LAT=4096
N_LON=8192
NLEV=26
# 60-step blocks, matching the protocol of the receipts this job is held to.
# The fused lane ignores --warmup: it times blocks after separate compile and
# probe calls. A 12-step block would be a different experiment and the
# reference medians would no longer certify anything.
STEPS=60
WARMUP=3
ARM_TIMEOUT=900
WALL_TOL=0.15
BAND_REF_128=1110
BAND_REF_512=997

# Closest-to-square tiles subject to an EVEN lon split, n_lat % p_lat == 0 and
# n_lon % p_lon == 0:
#   128 ranks, p_lon 16 -> 8 x 16 -> 512 x 512 cells per tile
#   512 ranks, p_lon 32 -> 16 x 32 -> 256 x 256 cells per tile
P_LON_128=16
P_LON_512=32

OUTDIR="${OUTDIR:-/scratch/b/b381103/legoesm_scaling/latlon_bvt_j${SLURM_JOB_ID}}"
[ -e "$OUTDIR" ] && { echo "REFUSING: $OUTDIR exists" >&2; exit 1; }
mkdir -p "$OUTDIR"

echo "== header =="
echo "sha        $(git -C "$REPO" rev-parse --short HEAD)"
echo "py         $PY"
echo "provenance $($PY -c 'import legoesm.parallel.latlon_spmd as m; print(m.__file__)')"
echo "platforms  JAX_PLATFORMS=$JAX_PLATFORMS"
echo "outdir     $OUTDIR"
echo "problem    ${N_LAT}x${N_LON}x${NLEV} strong, steps=$STEPS (matches the reference receipts)"
echo "layout     4 ranks/node, 64 threads/rank (production; the best of 8/16/32/64 as measured)"
echo "tiling     128 ranks -> $((128 / P_LON_128)) x ${P_LON_128};  512 ranks -> $((512 / P_LON_512)) x ${P_LON_512}"
echo "reference  band ${BAND_REF_128} ms at 128, ${BAND_REF_512} ms at 512, tolerance ${WALL_TOL}"

want=0; ok=0

# run_arm TAG ND P_LON PACKED NOCOMM BALLAST REP
run_arm() {
    local tag=$1 nd=$2 plon=$3 packed=$4 nocomm=$5 ballast=$6 rep=$7
    local nn=$(( nd / 4 ))
    want=$(( want + 1 ))
    echo "=== ${tag} rep${rep} nd=${nd} p_lon=${plon} packed=${packed}" \
         "nocomm=${nocomm} ballast=${ballast} $(date +%T)"
    if timeout --kill-after=30 "$ARM_TIMEOUT" srun --export=ALL \
            --nodes="$nn" --ntasks="$nd" --ntasks-per-node=4 \
            --cpus-per-task=64 --cpu-bind=cores --kill-on-bad-exit=1 \
            env LEGOESM_LATLON_PACKED_EXCHANGE="$packed" \
                LEGOESM_LATLON_SPMD_FUSED_HALO=1 \
                LEGOESM_LATLON_HALO_NOCOMM="$nocomm" \
                LEGOESM_LATLON_HALO_BALLAST="$ballast" \
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

# The BALLAST arms lead: they decide whether the payload mechanism is live at
# all, and if it is not, the band/tile pair is answering a question whose
# premise is already dead. Arm order alternates between repeats so a warm-up
# or a thermal drift cannot land on the same arm every time.
for rep in 1 2; do
    if [ "$rep" = 1 ]; then
        run_arm ball1_512  512 1            1 0    1 "$rep"
        run_arm ball4_512  512 1            1 0    4 "$rep"
        run_arm ball2_512  512 1            1 0    2 "$rep"
        run_arm band_512   512 1            1 0    1 "$rep"
        run_arm unpk_512   512 1            0 0    1 "$rep"
        run_arm tile_512   512 "$P_LON_512" 0 0    1 "$rep"
        run_arm band_128   128 1            1 0    1 "$rep"
        run_arm unpk_128   128 1            0 0    1 "$rep"
        run_arm tile_128   128 "$P_LON_128" 0 0    1 "$rep"
    else
        run_arm tile_512   512 "$P_LON_512" 0 0    1 "$rep"
        run_arm unpk_512   512 1            0 0    1 "$rep"
        run_arm band_512   512 1            1 0    1 "$rep"
        run_arm ball2_512  512 1            1 0    2 "$rep"
        run_arm ball4_512  512 1            1 0    4 "$rep"
        run_arm tile_128   128 "$P_LON_128" 0 0    1 "$rep"
        run_arm unpk_128   128 1            0 0    1 "$rep"
        run_arm band_128   128 1            1 0    1 "$rep"
    fi
done
# The wrong-answer controls run once each, last: they bound the halo's share
# but decide nothing on their own.
run_arm skip_512 512 1 1 1    1 1
run_arm wire_512 512 1 1 wire 1 1

echo "=== arms: $ok of $want completed ==="

$PY - "$OUTDIR" "$BAND_REF_128" "$BAND_REF_512" "$WALL_TOL" \
     "$P_LON_128" "$P_LON_512" <<'PYEOF'
import glob, json, math, os, sys

out_dir = sys.argv[1]
ref128, ref512, tol = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
pl128, pl512 = int(sys.argv[5]), int(sys.argv[6])

# Arms whose numbers are meaningless as physics: they deliberately give every
# rank its OWN edge rows instead of the neighbour's.
WRONG_BY_DESIGN = {"skip_512", "wire_512"}


def arm(tag, expect_reps):
    """Mean of each repeat's last ACCEPTED steady median, or a refusal reason.

    A receipt is accepted only if it is marked valid, its median is finite and
    strictly positive, and the expected number of repeats is present. A run
    that crashed after writing one line, a non-finite state reported with
    valid=true, and a half-finished arm all have to REFUSE rather than
    contribute a number -- the failure mode worth paying for here is a wrong
    number, not a missing one.
    """
    vals, rejected = [], []
    for f in sorted(glob.glob(os.path.join(out_dir, f"{tag}_r*.jsonl"))):
        last = None
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            v = d.get("steady_median_ms")
            if v is None:
                continue
            if d.get("valid") is False:
                rejected.append(f"{os.path.basename(f)}: valid=false")
                continue
            if d.get("finite_ok") is False:
                rejected.append(f"{os.path.basename(f)}: finite_ok=false")
                continue
            v = float(v)
            if not math.isfinite(v) or v <= 0.0:
                rejected.append(f"{os.path.basename(f)}: median {v!r}")
                continue
            last = v
        if last is not None:
            vals.append(last)
    if rejected:
        return None, "; ".join(rejected)
    if len(vals) != expect_reps:
        return None, f"{len(vals)} of {expect_reps} repeats produced a receipt"
    return (sum(vals) / len(vals), len(vals), max(vals) - min(vals)), None


SPEC = [("ball1_512", 1), ("ball2_512", 2), ("ball4_512", 2),
        ("band_512", 2), ("unpk_512", 2), ("tile_512", 2),
        ("band_128", 2), ("unpk_128", 2), ("tile_128", 2),
        ("skip_512", 1), ("wire_512", 1)]
got, why = {}, {}
for tag, reps in SPEC:
    got[tag], why[tag] = arm(tag, reps)

print("\n== arms ==")
for tag, _ in SPEC:
    v = got[tag]
    mark = "  [WRONG BY DESIGN]" if tag in WRONG_BY_DESIGN else ""
    print(f"  {tag:<10} " + (f"REFUSED ({why[tag]})" if v is None
                             else f"{v[0]:8.1f} ms  (n={v[1]}, "
                                  f"spread {v[2]:.1f})") + mark)

# ---- known answer, and it must fail on a WRONG value, not only a missing one
suspect = []
for tag, ref in (("band_128", ref128), ("band_512", ref512)):
    v = got[tag]
    if v is None:
        suspect.append(f"{tag} refused ({why[tag]})")
        continue
    off = abs(v[0] - ref) / ref
    print(f"\nknown answer {tag}: {v[0]:.0f} ms against {ref:.0f} ms recorded "
          f"at the same step protocol, {off * 100:.0f}% off -- "
          f"{'ok' if off <= tol else 'OUT OF TOLERANCE'}")
    if off > tol:
        suspect.append(f"{tag} is {off * 100:.0f}% off its recorded median")

if suspect:
    print("\nHARNESS SUSPECT: " + "; ".join(suspect) + ".")
    print("No ratio is read off this job. Resolve before interpreting.")
    print("LATLON_BVT_DONE")
    raise SystemExit(3)

# ---- the discriminator, reported before anything that depends on it
b1, b2, b4 = got["ball1_512"], got["ball2_512"], got["ball4_512"]
payload_live = None
if b1 and b2 and b4:
    print("\n== does payload bind? (bit-identical ballast, 512 ranks) ==")
    for n, v in ((1, b1), (2, b2), (4, b4)):
        print(f"  {n}x halo bytes: {v[0]:8.1f} ms  "
              f"({v[0] / b1[0]:.3f}x of the 1x arm)")
    slope = (b4[0] - b1[0]) / b1[0] / 3.0     # fractional rise per extra copy
    spread = max(b1[2], b2[2], b4[2]) / b1[0]
    print(f"  fractional rise per extra copy of the halo: {slope:.3f}; "
          f"worst repeat spread {spread:.3f} of the step")
    if slope <= max(0.02, spread):
        payload_live = False
        print("  PER-OPERATION: quadrupling the bytes did not move the step "
              "beyond its own repeat spread. The non-shrinking cost is "
              "latency and synchronization, NOT payload, so cutting halo "
              "bytes cannot recover it.")
    elif slope >= 0.10:
        payload_live = True
        print("  PAYLOAD: the step tracks the halo bytes. Cutting them is a "
              "real lever and a 2-D mesh should pay.")
    else:
        print("  PARTIAL: the step responds to bytes, but weakly. Payload is "
              "part of the cost and not the whole of it.")
else:
    print("\n== ballast arms incomplete; the payload question is unanswered ==")

# ---- band vs tile, controlled on packing
print(f"\n== band vs tile (tiles {4096 // (128 // pl128)} x {8192 // pl128} at "
      f"128 ranks, {4096 // (512 // pl512)} x {8192 // pl512} at 512) ==")
print("  The tile arm CANNOT use the packed exchange (packed_exchange_mesh "
      "returns None for a 2-D mesh, and the tiled packed body has no caller), "
      "so the unpacked band is the tile's control. The production band is "
      "shown for scale only.")
for nd, bt, ut, tt in ((128, "band_128", "unpk_128", "tile_128"),
                       (512, "band_512", "unpk_512", "tile_512")):
    b, u, t = got[bt], got[ut], got[tt]
    if b and u and t:
        print(f"  {nd:>4} ranks: band(packed) {b[0]:7.1f}   "
              f"band(unpacked) {u[0]:7.1f}   tile {t[0]:7.1f}   "
              f"unpacked/tile {u[0] / t[0]:.2f}x   "
              f"packing is worth {100 * (u[0] - b[0]) / u[0]:.0f}%")

# ---- the wire controls
skip, wire, band = got["skip_512"], got["wire_512"], got["band_512"]
if skip and wire and band:
    wire_ms = wire[0] - skip[0]
    print(f"\n== halo cost at 512 ranks (BOTH arms are numerically wrong by "
          f"design) ==")
    print(f"  wire minus skip = {wire_ms:.1f} ms, "
          f"{100 * wire_ms / band[0]:.0f}% of the band step. These two arms "
          f"follow the SAME wrong trajectory and differ by the collective "
          f"alone, which is why this subtraction is quotable.")
    print(f"  full minus skip would say "
          f"{100 * (band[0] - skip[0]) / band[0]:.0f}%, which is NOT a bound "
          f"in either direction: the skip arm carries different fields, so "
          f"its local work differs too and the sign of that difference is "
          f"not established.")

# ---- verdict
if got["unpk_512"] and got["tile_512"]:
    s = got["unpk_512"][0] / got["tile_512"][0]
    print("\n== verdict, pre-registered, judged at 512 ranks on the "
          "packing-matched pair ==")
    if s >= 1.25:
        print(f"  TILING WINS: the tile is {100 * (s - 1):.0f}% faster. The "
              f"one-dimensional decomposition is a real floor.")
    elif s >= 1.10:
        print(f"  PARTIAL: the tile is {100 * (s - 1):.0f}% faster -- real, "
              f"but far below the 6x that the boundary-cell ratio alone "
              f"would give, so most of the cost is not payload.")
    elif s >= 0.90:
        print(f"  TILING NEUTRAL: {s:.2f}x, inside the registered band. "
              f"Cutting halo bytes does not move this lane.")
    else:
        print(f"  TILING LOSES: the tile is {100 * (1.0 / s - 1):.0f}% "
              f"SLOWER. The extra collectives cost more than the saved bytes.")
    if payload_live is False and s >= 1.25:
        print("  DISAGREEMENT: the ballast arm said payload does not bind, "
              "yet tiling won. Those cannot both be right about the same "
              "cost; do not report either until it is resolved.")
    if payload_live is True and s < 1.10:
        print("  DISAGREEMENT: the ballast arm said payload binds, yet "
              "tiling did not pay. The tile's extra collectives are the "
              "first thing to count.")
print("LATLON_BVT_DONE")
PYEOF
