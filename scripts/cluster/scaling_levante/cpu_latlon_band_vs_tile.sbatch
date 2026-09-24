#!/bin/bash -l
#SBATCH --job-name=latlon_band_vs_tile
#SBATCH --account=bb1596
#SBATCH --partition=compute
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
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
# TWO MECHANISMS FIT THAT SHAPE AND THEY IMPLY OPPOSITE FIXES.
#
#   PAYLOAD. With --p-lon 1 (the default, and what the production ladder
#   passes) the mesh is 1-D over latitude, so every rank's halo ships two
#   complete longitude circles -- a payload identical at 8 ranks and at 512
#   while local work falls as 1/n. Fix: a 2-D mesh, which ships tile
#   boundaries instead.
#
#   PER-OPERATION COST. Constant bytes give a CONSTANT term, not one growing
#   with log2(n). Growth with rank count is the signature of latency and
#   synchronization depth. A 2-D mesh CUTS BYTES BUT ADDS COLLECTIVES, so
#   under this mechanism tiling is neutral at best and can lose.
#
# THE DISCRIMINATOR: wire(N) MINUS skip(N), AT TWO PAYLOAD MULTIPLES.
#
# Three measurement knobs, all documented as never-production:
#   LEGOESM_LATLON_HALO_BALLAST=N sends each halo message N times and drops
#     the copies. Bit-identical: bytes scale, collective count, schedule,
#     arithmetic and answer do not.
#   LEGOESM_LATLON_HALO_NOCOMM='1' replaces the collective with an
#     optimization barrier -- the extra concatenate and slice are still paid,
#     the wire is not.
#   ...='wire' runs the collective and discards the result, following the SAME
#     wrong trajectory as '1'.
#
# The knobs COMPOSE, and the library says so explicitly: the payload multiple
# is built BEFORE the no-comm branch precisely so that skip-at-N pays the
# multiplier's device-side packing with no wire at all. That matters because
# the raw ballast delta came out LARGER than the whole communication term in
# an earlier measurement, which is only possible if packing was being counted
# as payload. So the quantity to read is
#
#     c(N) = wire(N) - skip(N)          the collective alone, at N x bytes
#
# with the packing cost present on BOTH sides of each subtraction and
# therefore cancelled. A collective costs latency plus bytes over bandwidth,
# so model c(N) = L + N*B and solve the two measurements for both terms:
#
#     B = (c4 - c1) / 3        L = (4*c1 - c4) / 3
#     payload fraction at production bytes = B / c(1)
#
# The FRACTION is what is reported, not a ratio. An earlier version gated on
# r = c(4)/c(1) with a payload cut at 2.5, which is wrong: under this model a
# collective whose cost is HALF latency and HALF bytes gives r = 5B/2B = 2.5
# exactly, so every collective less payload-bound than one-half would have
# been reported as per-operation. The fraction has no such bias.
#     fraction >= 0.5 -> payload dominates      -> cutting bytes is a lever
#     fraction <= 0.2 -> per-operation dominates -> cutting bytes cannot help
# Anything between, or bounds that straddle a cut, is MIXED or INCONCLUSIVE
# and is reported as such rather than resolved in either direction.
#
# THE DISCRIMINATOR RUNS UNPACKED, matching the packing state of the tile arm
# it informs. LEGOESM_LATLON_PACKED_EXCHANGE=1 is production for the band,
# but packed_exchange_mesh() returns None for a 2-D mesh, so the tile arm
# runs unpacked whatever is asked of it. The unpacked band is therefore the
# tile's control; the packed band appears only as the known-answer arm.
# (The tiled packed body exists in the library with no caller anywhere under
# packages/. That is dead code and a finding in its own right; it is also why
# the confound cannot simply be switched off.)
#
# FINITENESS. The fused timing lane carries no in-graph finite check and
# stamps finite_ok=null with valid=true, so a positive elapsed time could in
# principle describe a diverged trajectory. The reference receipts were taken
# on that lane, so the timed arms stay on it to keep the comparison honest,
# and ONE segment-lane arm runs the production band as a finiteness
# certificate -- that lane carries the in-graph scalar and refuses to
# serialize a diverging run as valid.
# ===========================================================================
set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
# _env.sh defaults JAX_PLATFORMS to cuda; this is a CPU partition. Pinned
# before the source, which honours an existing value.
export JAX_PLATFORMS=cpu
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }

N_LAT=4096
N_LON=8192
NLEV=26
# 60-step blocks, matching the protocol of the receipts the known-answer gate
# holds this job to. The fused lane ignores --warmup: it times blocks after a
# separate compile and probe call.
STEPS=60
BLOCKS=2
WARMUP=3
# The discriminator arms are compared only against EACH OTHER, so they are not
# held to the reference protocol and split the same total work into 8 short
# blocks instead of 2 long ones. Two run-level numbers per arm is a terrible
# uncertainty estimate -- the range of two samples is itself mostly noise, and
# a step time that is a max over 512 ranks has a straggler tail that two
# samples either miss entirely (falsely tight) or catch once (falsely wide).
# Eight blocks x two repeats gives sixteen samples per arm to bootstrap from.
DISC_STEPS=15
DISC_BLOCKS=8
# Per arm: one precompile scan plus two timed scans of 60 steps, plus first-
# call compilation. 19 arms at this cap is 6.3 h inside an 8 h wall.
ARM_TIMEOUT=1200
WALL_TOL=0.15
BAND_REF_128=1110
BAND_REF_512=997

# Closest-to-square tiles with an EVEN lon split (the tiled packed body raises
# on an odd one), n_lat % p_lat == 0 and n_lon % p_lon == 0:
#   128 ranks, 8 x 16  -> 512 x 512 cells per tile
#   512 ranks, 16 x 32 -> 256 x 256 cells per tile
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

# run_arm TAG ND P_LON PACKED NOCOMM BALLAST SEGSTEPS REP [STEPS] [BLOCKS]
run_arm() {
    local tag=$1 nd=$2 plon=$3 packed=$4 nocomm=$5 ballast=$6 seg=$7 rep=$8
    local steps=${9:-$STEPS} blocks=${10:-$BLOCKS}
    local nn=$(( nd / 4 ))
    local -a extra=()
    [ "$seg" != 0 ] && extra=(--segment-steps "$seg")
    want=$(( want + 1 ))
    echo "=== ${tag} rep${rep} nd=${nd} p_lon=${plon} packed=${packed}" \
         "nocomm=${nocomm} ballast=${ballast} seg=${seg}" \
         "steps=${steps} blocks=${blocks} $(date +%T)"
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
            --p-lon "$plon" --steps "$steps" --blocks "$blocks" \
            --warmup "$WARMUP" "${extra[@]}" \
            --out "$OUTDIR/${tag}_r${rep}.jsonl"; then
        ok=$(( ok + 1 ))
    else
        echo "ARM FAILED ${tag} rep${rep} rc=$? (124 = timeout)"
    fi
}

# The four discriminator arms lead: they decide whether the payload premise is
# alive before any tile arm is interpreted. Arm order alternates between the
# repeats so a warm-up or a thermal drift cannot land on the same arm twice.
for rep in 1 2; do
    if [ "$rep" = 1 ]; then
        run_arm skip_b1  512 1            0 1    1 0 "$rep" "$DISC_STEPS" "$DISC_BLOCKS"
        run_arm wire_b1  512 1            0 wire 1 0 "$rep" "$DISC_STEPS" "$DISC_BLOCKS"
        run_arm skip_b4  512 1            0 1    4 0 "$rep" "$DISC_STEPS" "$DISC_BLOCKS"
        run_arm wire_b4  512 1            0 wire 4 0 "$rep" "$DISC_STEPS" "$DISC_BLOCKS"
        run_arm band_512 512 1            1 0    1 0 "$rep"
        run_arm unpk_512 512 1            0 0    1 0 "$rep"
        run_arm tile_512 512 "$P_LON_512" 0 0    1 0 "$rep"
        run_arm band_128 128 1            1 0    1 0 "$rep"
        run_arm tile_128 128 "$P_LON_128" 0 0    1 0 "$rep"
    else
        run_arm wire_b4  512 1            0 wire 4 0 "$rep" "$DISC_STEPS" "$DISC_BLOCKS"
        run_arm skip_b4  512 1            0 1    4 0 "$rep" "$DISC_STEPS" "$DISC_BLOCKS"
        run_arm wire_b1  512 1            0 wire 1 0 "$rep" "$DISC_STEPS" "$DISC_BLOCKS"
        run_arm skip_b1  512 1            0 1    1 0 "$rep" "$DISC_STEPS" "$DISC_BLOCKS"
        run_arm tile_512 512 "$P_LON_512" 0 0    1 0 "$rep"
        run_arm unpk_512 512 1            0 0    1 0 "$rep"
        run_arm band_512 512 1            1 0    1 0 "$rep"
        run_arm tile_128 128 "$P_LON_128" 0 0    1 0 "$rep"
        run_arm band_128 128 1            1 0    1 0 "$rep"
    fi
done
# Finiteness certificate, production band, the lane that carries the in-graph
# scalar and refuses to serialize a diverging run as valid. Not compared
# against anything -- it exists so the fused-lane timings above cannot be
# describing a trajectory that has blown up.
run_arm cert_512 512 1 1 0 1 10 1

echo "=== arms: $ok of $want completed ==="

$PY - "$OUTDIR" "$BAND_REF_128" "$BAND_REF_512" "$WALL_TOL" \
     "$P_LON_128" "$P_LON_512" <<'PYEOF'
import glob, json, math, os, random, statistics, sys

out_dir = sys.argv[1]
ref128, ref512, tol = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
pl128, pl512 = int(sys.argv[5]), int(sys.argv[6])

# Arms whose numbers are meaningless as physics: they deliberately give every
# rank its OWN edge rows instead of the neighbour's.
WRONG_BY_DESIGN = {"skip_b1", "skip_b4", "wire_b1", "wire_b4"}


def arm(tag, expect_reps):
    """Pooled per-BLOCK step times across the repeats, or a refusal reason.

    The fused lane times whole blocks, so a run yields one sample per block
    rather than one per step. Pooling every block of every repeat is what
    makes an uncertainty statement possible at all: the range of two run-level
    numbers is itself mostly noise, and a step time that is a max over 512
    ranks has a straggler tail that two samples either miss (falsely tight) or
    catch once (falsely wide).

    A receipt counts only if it is not marked invalid, not marked non-finite,
    and its block times are finite and strictly positive -- and only if every
    expected repeat produced one. A run that crashed after writing a line, a
    non-finite state, and a half-finished arm each have to REFUSE rather than
    contribute: the failure worth paying to avoid here is a wrong number.
    """
    samples, seen, rejected = [], 0, []
    for f in sorted(glob.glob(os.path.join(out_dir, f"{tag}_r*.jsonl"))):
        got_one = False
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("steady_median_ms") is None:
                continue
            base = os.path.basename(f)
            if d.get("valid") is False:
                rejected.append(f"{base}: valid=false")
                continue
            if d.get("finite_ok") is False:
                rejected.append(f"{base}: finite_ok=false")
                continue
            steps = d.get("steps") or 0
            blocks = d.get("block_ms") or []
            if steps and blocks:
                per_step = [b / steps for b in blocks]
            else:
                per_step = [float(d["steady_median_ms"])]
            per_step = [float(v) for v in per_step]
            if not all(math.isfinite(v) and v > 0.0 for v in per_step):
                rejected.append(f"{base}: a block time is {per_step!r}")
                continue
            samples.extend(per_step)
            got_one = True
        if got_one:
            seen += 1
    if rejected:
        return None, "; ".join(rejected)
    if seen != expect_reps:
        return None, f"{seen} of {expect_reps} repeats produced a receipt"
    if len(samples) < 2:
        return None, f"only {len(samples)} block sample(s)"
    return samples, None


def med(samples):
    return statistics.median(samples)


def boot(samples, n=2000, lo_q=0.05, hi_q=0.95):
    """Percentile bootstrap interval of the median."""
    rng = random.Random(20260924)
    k = len(samples)
    meds = sorted(statistics.median(rng.choices(samples, k=k))
                  for _ in range(n))
    return (meds[int(lo_q * n)], meds[min(n - 1, int(hi_q * n))])


def boot_diff(a, b, n=2000, lo_q=0.05, hi_q=0.95):
    """Percentile bootstrap interval of median(a) - median(b)."""
    rng = random.Random(20260925)
    ka, kb = len(a), len(b)
    ds = sorted(statistics.median(rng.choices(a, k=ka))
                - statistics.median(rng.choices(b, k=kb))
                for _ in range(n))
    return (ds[int(lo_q * n)], ds[min(n - 1, int(hi_q * n))])


SPEC = [("skip_b1", 2), ("wire_b1", 2), ("skip_b4", 2), ("wire_b4", 2),
        ("band_512", 2), ("unpk_512", 2), ("tile_512", 2),
        ("band_128", 2), ("tile_128", 2), ("cert_512", 1)]
got, why = {}, {}
for tag, reps in SPEC:
    got[tag], why[tag] = arm(tag, reps)

print("\n== arms ==")
for tag, _ in SPEC:
    v = got[tag]
    mark = "  [WRONG BY DESIGN -- not physics]" if tag in WRONG_BY_DESIGN else ""
    if v is None:
        print(f"  {tag:<9} REFUSED ({why[tag]})" + mark)
    else:
        lo, hi = boot(v)
        print(f"  {tag:<9} {med(v):8.1f} ms  "
              f"(90% CI {lo:.1f}-{hi:.1f} from {len(v)} block samples)" + mark)

fail = []

# ---- finiteness certificate
cert = got["cert_512"]
if cert is None:
    print(f"\nfiniteness certificate: REFUSED ({why['cert_512']}). The timed "
          f"lane carries no in-graph finite check, so without this arm a "
          f"positive elapsed time cannot be distinguished from a diverged "
          f"trajectory.")
    fail.append("no finiteness certificate")
else:
    print(f"\nfiniteness certificate: the production band ran {med(cert):.1f} "
          f"ms/step on the lane that refuses to serialize a diverging run as "
          f"valid, so the fused-lane timings are not describing a blown-up "
          f"state.")

# ---- known answer, and it must fail on a WRONG value, not only a missing one
for tag, ref in (("band_128", ref128), ("band_512", ref512)):
    v = got[tag]
    if v is None:
        fail.append(f"{tag} refused ({why[tag]})")
        continue
    off = abs(med(v) - ref) / ref
    print(f"known answer {tag}: {med(v):.0f} ms against {ref:.0f} ms recorded at "
          f"the same step protocol, {off * 100:.0f}% off -- "
          f"{'ok' if off <= tol else 'OUT OF TOLERANCE'}")
    if off > tol:
        fail.append(f"{tag} is {off * 100:.0f}% off its recorded median")

if fail:
    print("\nHARNESS SUSPECT: " + "; ".join(fail) + ".")
    print("No mechanism verdict and no ratio are read off this job.")
    print("LATLON_BVT_DONE")
    raise SystemExit(3)

# ---- the discriminator
s1, w1, s4, w4 = (got["skip_b1"], got["wire_b1"],
                  got["skip_b4"], got["wire_b4"])
payload_live = None
if None in (s1, w1, s4, w4):
    missing = [t for t in ("skip_b1", "wire_b1", "skip_b4", "wire_b4")
               if got[t] is None]
    print(f"\n== the payload question is UNANSWERED: {', '.join(missing)} "
          f"refused. Nothing below may be read as a mechanism. ==")
else:
    print("\n== does payload bind? collective cost at 1x and 4x halo bytes, "
          "unpacked band, 512 ranks ==")

    # LIVENESS CHECK, and it comes first because failing it would manufacture
    # the payload verdict. The skip arm is supposed to still pay for building
    # the N-times-larger buffer; that only holds if the barrier carries a DATA
    # dependence on it. If the compiler instead eliminated the buffer build in
    # the skip arm alone, then skip(4) - skip(1) is zero and that missing
    # build cost lands in wire(4) - wire(1) instead -- which is exactly the
    # signature of a payload-bound collective, whatever the wire is doing.
    d_skip = med(s4) - med(s1)
    dlo, dhi = boot_diff(s4, s1)
    print(f"  liveness: skip(4) - skip(1) = {d_skip:7.1f} ms "
          f"(90% CI {dlo:.1f} to {dhi:.1f}). The skip arm still builds the "
          f"4x buffer, so this should be clearly positive.")
    liveness_ok = dlo > 0.0
    if not liveness_ok:
        print("  LIVENESS FAILED: the no-wire arm did not get more expensive "
              "when its buffer quadrupled, so the compiler probably deleted "
              "the buffer build there. The build cost then appears only in "
              "the wire arm and imitates a payload-bound collective exactly. "
              "No mechanism verdict is read from this job.")

    c1, c4 = med(w1) - med(s1), med(w4) - med(s4)
    c1lo, c1hi = boot_diff(w1, s1)
    c4lo, c4hi = boot_diff(w4, s4)
    print(f"  wire - skip at 1x bytes: {c1:8.1f} ms  "
          f"(90% CI {c1lo:.1f} to {c1hi:.1f})")
    print(f"  wire - skip at 4x bytes: {c4:8.1f} ms  "
          f"(90% CI {c4lo:.1f} to {c4hi:.1f})")
    print("  The device-side packing the multiplier itself adds is present in "
          "BOTH terms of each subtraction and cancels; that is why the two "
          "knobs are composed rather than used alone.")

    if not liveness_ok:
        pass
    elif c1 <= 0 or c1lo <= 0:
        print("  INCONCLUSIVE: the collective's own cost at 1x is not "
              "resolved above zero, so a fraction computed against it means "
              "nothing.")
    else:
        # c(N) = L + N*B solved on the two measurements. The reported quantity
        # is the PAYLOAD FRACTION at production bytes, not the ratio
        # c(4)/c(1): the two are related by ratio = 1 + 3*fraction, so a ratio
        # cut at 2.5 is a fraction cut at exactly one half and would call
        # everything less payload-bound than that "per-operation".
        def split(a1, a4):
            return ((a4 - a1) / 3.0) / a1 if a1 > 0 else float("nan")

        frac = split(c1, c4)
        cands = [split(a, b) for a in (c1lo, c1, c1hi)
                 for b in (c4lo, c4, c4hi) if a > 0]
        lo, hi = (min(cands), max(cands)) if cands else (float("nan"),) * 2
        print(f"  fitted latency {(4 * c1 - c4) / 3.0:8.1f} ms, "
              f"per-copy bytes term {(c4 - c1) / 3.0:8.1f} ms")
        print(f"  payload fraction of the collective at production bytes: "
              f"{frac:.2f}, bounds [{lo:.2f}, {hi:.2f}]")
        if lo >= 0.5:
            payload_live = True
            print("  PAYLOAD DOMINATES: at least half the collective's cost "
                  "is its bytes. Cutting them is a real lever and a 2-D mesh "
                  "should pay.")
        elif hi <= 0.2:
            payload_live = False
            print("  PER-OPERATION DOMINATES: a fifth or less of the "
                  "collective's cost is its bytes. The non-shrinking cost is "
                  "latency and synchronization, so cutting halo bytes cannot "
                  "recover it.")
        elif lo >= 0.2 and hi <= 0.5:
            print("  MIXED: the bytes are a real but minority share of the "
                  "collective. Cutting them helps by less than the boundary "
                  "ratio suggests.")
        else:
            print("  INCONCLUSIVE: the bounds straddle a registered cut. A "
                  "larger payload multiple, or more blocks per arm, would be "
                  "needed to separate the two mechanisms.")
        if not (0.0 <= frac <= 1.5):
            print(f"  WARNING: a payload fraction of {frac:.2f} is outside "
                  f"what the model c = L + N*B can produce. gloo may switch "
                  f"algorithm between the two message sizes, in which case "
                  f"the two points do not lie on one line and neither term "
                  f"is trustworthy.")

        # The fraction is a share of the COLLECTIVE, not of the step, and it
        # is not by itself a prediction. Both candidate fixes trade bytes for
        # message count, so the decision needs the two slopes applied to each
        # fix's change in bytes and in message count.
        if got["band_512"]:
            print(f"  For scale: the whole production step at 512 ranks is "
                  f"{med(got['band_512']):.0f} ms, so this collective's own "
                  f"cost at production bytes is "
                  f"{100 * c1 / med(got['band_512']):.0f}% of it. A 2-D mesh "
                  f"cuts bytes AND multiplies message count, so neither "
                  f"number predicts the fix on its own.")

# ---- band vs tile, controlled on packing
print(f"\n== band vs tile (tiles {4096 // (128 // pl128)} x {8192 // pl128} at "
      f"128 ranks, {4096 // (512 // pl512)} x {8192 // pl512} at 512) ==")
print("  The tile arm cannot use the packed exchange, so the UNPACKED band is "
      "its control. Tiling also changes local array shapes, pole handling and "
      "the communication schedule, so a ratio here says which implementation "
      "is faster -- it is not by itself a payload measurement.")
for nd, bt, ut, tt in ((128, "band_128", None, "tile_128"),
                       (512, "band_512", "unpk_512", "tile_512")):
    b, t = got[bt], got[tt]
    u = got[ut] if ut else None
    if b and t:
        line = (f"  {nd:>4} ranks: band(packed) {med(b):7.1f}   "
                f"tile {med(t):7.1f}")
        if u:
            line += (f"   band(unpacked) {med(u):7.1f}   "
                     f"unpacked/tile {med(u) / med(t):.2f}x   "
                     f"packing is worth "
                     f"{100 * (med(u) - med(b)) / med(u):.0f}%")
        else:
            line += "   (no unpacked control at this rank count)"
        print(line)

# ---- verdict
if got["unpk_512"] and got["tile_512"]:
    s = med(got["unpk_512"]) / med(got["tile_512"])
    rlo, rhi = boot_diff(got["unpk_512"], got["tile_512"])
    print(f"\n  unpacked band minus tile: {med(got['unpk_512']) - med(got['tile_512']):.1f} ms "
          f"(90% CI {rlo:.1f} to {rhi:.1f})")
    print("\n== verdict, pre-registered, judged at 512 ranks on the "
          "packing-matched pair ==")
    if s >= 1.25:
        print(f"  TILING WINS: the tile is {100 * (s - 1):.0f}% faster.")
    elif s >= 1.10:
        print(f"  PARTIAL: the tile is {100 * (s - 1):.0f}% faster -- real, "
              f"but far below the ~16x boundary-cell ratio, so most of the "
              f"cost is not boundary volume.")
    elif s >= 0.90:
        print(f"  TILING NEUTRAL: {s:.2f}x, inside the registered band.")
    else:
        print(f"  TILING LOSES: the tile is {100 * (1.0 / s - 1):.0f}% "
              f"SLOWER.")
    if payload_live is False and s >= 1.25:
        print("  DISAGREEMENT: the collective did not track its bytes, yet "
              "tiling won. Those cannot both be about the same cost -- the "
              "tile's other changes (shapes, pole handling, schedule) are the "
              "first thing to count. Report neither as a mechanism until it "
              "is resolved.")
    if payload_live is True and s < 1.10:
        print("  DISAGREEMENT: the collective tracked its bytes, yet tiling "
              "did not pay. Count the tile's extra collectives before "
              "reporting either.")
print("LATLON_BVT_DONE")
PYEOF
