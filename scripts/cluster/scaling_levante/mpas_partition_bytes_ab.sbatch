#!/bin/bash -l
#SBATCH --job-name=mpas_part_ab
#SBATCH --account=bd1083_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=8
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclude=l50100
#SBATCH --output=mpas_part_ab.%j.log
# ===========================================================================
# Does an EXISTING partitioner's padded-byte cut convert to wall time?
#
# Offline receipt (probe bench_mpas_halo_degree_partition.py, PR #1584 +
# byte follow-up): padding is PER-ROUND MAX — every pair in a round ships
# the round's worst pair's payload (sharded_dynamics.py:2151-2179). At
# s9@32 metis holds the SAME 12 rounds as sfc while cutting padded bytes
# 169.3 -> 121.4 MB (-28.3%). A purpose-built byte-aware ownership pass
# was REFUTED at 0.0%, so the shipped partitioners are the lever.
#
# Lever arm from the one bytes-only A/B at fixed rounds (26857404 +
# audits 26855933/26856854): -26.2% bytes bought -20.4% step, i.e.
# ~0.78 step-% per byte-%. -28.3% bytes therefore projects ~-22% step.
#
# Falsifiability BEFORE submit:
#  (a) number produced: steady_median_ms at s9@32, sfc vs metis.
#  (b) CONFIRM if metis/sfc <= 0.85 (the projection is -22%; a >=15%
#      cut is the conservative read).
#      REFUTE if >= 0.98 — then padded BYTES are not what this lane
#      pays for at fixed round count, and the 0.78 lever arm (derived
#      from a colouring-era audit) does not transfer.
#      Arm spread > half the effect => INCONCLUSIVE regardless.
#  (c) no cheaper test: the byte-to-wall conversion is a GPU/NCCL
#      property; the offline probe prices bytes, not time.
#
# Protocol: byte-identical arms except --partition-method. Interleaved
# with a discarded warm-up arm and 40 timed steps, per the harness
# lesson from jobs 26880294/26880368 (12 steps put 7.7% spread on
# IDENTICAL control arms; 60 steps gave 0.3%).
#
# CAVEAT recorded up front: metis scores 70 connected components at
# s9@32 vs sfc's 40. The offline probe treats that as a ship-blocker by
# pre-registration, NOT by physical argument — these are shipped
# partitioners and the equal-block re-cut already leaves sfc itself
# disconnected (40 components for 32 devices). This job measures speed
# only; a parity gate runs first, and connectivity stays an open
# question for the default-flip decision, not for this measurement.
# ===========================================================================
set -uo pipefail
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
export JAX_PLATFORMS=cuda,cpu
export LEGOESM_XLA_OVERLAP=0
export LEGOESM_MESH_CACHE_DIR=/work/bd1083/b309178/diffESM/legoesm_mesh_cache
export LEGOESM_REPO="${LEGOESM_REPO:-$SUBMIT_DIR}"
export LEGOESM_PYTHON="${LEGOESM_PYTHON:-/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.venv/bin/python}"
source "${SUBMIT_DIR}/scripts/cluster/scaling_levante/_env.sh"
cd "$REPO" || { echo "cd $REPO FAILED"; exit 2; }
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/mpas_part_j${SLURM_JOB_ID}}"
mkdir -p "$OUTDIR"; echo "outdir=$OUTDIR"
unset XLA_FLAGS
rc=0

PA_NODES="${PA_NODES:-8}"
PA_GPN="${PA_GPN:-4}"
PA_SUBDIV="${PA_SUBDIV:-9}"
PA_STEPS="${PA_STEPS:-40}"
PA_NDEV=$((PA_NODES * PA_GPN))

# pymetis is what makes --partition-method metis resolvable; without it
# the arm would silently fall back and the A/B would compare sfc to sfc.
"$PY" -c "import pymetis" 2>/dev/null || { echo "pymetis MISSING — metis arm would silently fall back to RCB; refusing to run a fake A/B"; exit 4; }

run_arm () {  # method tag
  echo "=== s${PA_SUBDIV}@${PA_NDEV} partition=$1 [$2] ==="
  sleep 15
  JAX_ENABLE_X64=0 \
    srun --nodes="$PA_NODES" --ntasks="$PA_NDEV" \
      --ntasks-per-node="$PA_GPN" --gpus-per-node="$PA_GPN" \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices "$PA_NDEV" \
      --subdivision "$PA_SUBDIV" --nlev 26 \
      --steps "$PA_STEPS" --warmup 5 --lloyd 0 \
      --partition-method "$1" --reorder-for "$PA_NDEV" \
      --out "$OUTDIR/$2.jsonl" || { echo "$2 FAILED"; rc=1; }
}

# Parity first: a partition change must not change the answer.
echo "=== parity gate: s6@4 metis ==="
# The gate compares against the single-device reference with EXACT-exchange
# tolerances, so it must run with the narrow halo even when the timing arms
# run the deep one. The deep halo is physically equivalent, not bit-exact --
# it recomputes an outer ring, which puts u about 3e-3 off the serial
# reference and stays there at double precision. Submitting this job with
# the deep halo exported once made the gate fail and killed both arms
# before any timing ran.
LEGOESM_MPAS_WIDE_HALO=0 \
JAX_ENABLE_X64=0 srun --nodes=1 --ntasks=4 --ntasks-per-node=4 \
    --gpus-per-node=4 --gpu-bind=none --kill-on-bad-exit=1 \
  "$PY" scripts/bench/bench_mpas_spmd_scaling.py \
    --multicontroller --n-devices 4 --subdivision 6 --nlev 8 \
    --steps 4 --warmup 1 --lloyd 0 --partition-method metis \
    --parity-gate --out "$OUTDIR/parity_metis.jsonl" \
  || { echo "METIS PARITY GATE FAILED — timing arms not run"; exit 5; }

run_arm sfc   warmup_discard
run_arm sfc   sfc_a
run_arm metis metis_b
run_arm sfc   sfc_a2
run_arm metis metis_b2

echo "=== RECEIPTS ==="
set +e
PA_NDEV="$PA_NDEV" "$PY" - "$OUTDIR" <<'EOF'
import json, math, os, sys, pathlib
out = pathlib.Path(sys.argv[1])
nd = int(os.environ["PA_NDEV"])
t, meth = {}, {}
for tag, want in (("sfc_a", "sfc"), ("metis_b", "metis"),
                  ("sfc_a2", "sfc"), ("metis_b2", "metis")):
    try:
        rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    except Exception as e:
        print(f"{tag}: MISSING -> {e}"); sys.exit(1)
    if len(rows) != 1:
        print(f"{tag}: {len(rows)} rows"); sys.exit(1)
    r = rows[0]
    got = r.get("partition_method")
    ok = (r.get("n_devices") == nd and r.get("platform") in ("gpu", "cuda")
          and r.get("n_processes") == nd and got == want
          and math.isfinite(r["steady_median_ms"])
          and r["steady_median_ms"] > 0)
    if not ok:
        print(f"{tag}: binding fail method={got} (want {want}) "
              f"nd={r.get('n_devices')} platform={r.get('platform')}")
        sys.exit(1)
    t[tag] = r["steady_median_ms"]; meth[tag] = got
    print(f"{tag:12s} {t[tag]:9.3f} ms  ({got}, "
          f"CPs={r.get('hlo_collective_permutes')})")
sfc = sorted((t["sfc_a"], t["sfc_a2"]))
met = sorted((t["metis_b"], t["metis_b2"]))
base = sum(sfc) / 2.0
ratio = (sum(met) / 2.0) / base
sfc_spread = (sfc[1] - sfc[0]) / sfc[0]
met_spread = (met[1] - met[0]) / met[0]
print(f"sfc {sfc[0]:.3f}/{sfc[1]:.3f}  metis {met[0]:.3f}/{met[1]:.3f}")
print(f"within-sfc {100*sfc_spread:.1f}%  within-metis {100*met_spread:.1f}%"
      f"  |  metis/sfc ratio {ratio:.3f}")
if max(sfc_spread, met_spread) > 2.0 * abs(1.0 - ratio):
    print("VERDICT: INCONCLUSIVE (arm spread exceeds half the effect)")
    sys.exit(3)
if ratio <= 0.85:
    print("VERDICT: CONFIRMED — the padded-byte cut converts to wall time")
elif ratio >= 0.98:
    print("VERDICT: REFUTED — bytes are not what this lane pays at fixed "
          "round count; the 0.78 step-%/byte-% lever arm does not transfer")
    sys.exit(2)
else:
    print("VERDICT: intermediate — real but shallower; reprice the lever arm")
    sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
