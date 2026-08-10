#!/bin/bash -l
#SBATCH --job-name=wh_census_graphs
#SBATCH --account=bb1596_gpu
#SBATCH --partition=gpu
#SBATCH --constraint=a100_80
#SBATCH --nodes=16
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:30:00
#SBATCH --exclude=l50100
#SBATCH --output=wh_census_graphs.%j.log
# ===========================================================================
# WIDE-HALO post-refutation instruments (job 26846337: wide/off 0.970 —
# the 33->11 collective cut moved the step 3%, so per-collective
# latency is NOT the binding term at s9@64).
#
# Two questions, one job:
#
# (1) CENSUS: what does the compiled WIDE step actually contain at 64
#     GPUs? Rank-0 XLA dump -> collective count (MUST be 11 CP — if 33
#     the wide mode silently did not engage on the GPU lane and the
#     0.970 verdict is void) + entry-instruction count (the sequential
#     kernel-slot census that the slot-floor hypothesis needs:
#     default = 301 top-level ops; wall/slots ~ 34 us/slot).
#
# (2) GRAPHS x WIDE: wide mode creates ONE long collective-free RK
#     segment (~150+ fusions). The earlier CUDA-graph refutation
#     (26680791) ran on the DEFAULT step, whose 33 CPs slice the
#     program into ~4-op segments below capture size. A/B:
#       G1 wide=1 + command_buffer=FUSION, min_graph_size=2
#       C0 wide=1, stock flags        (baseline + dump)
#       G2 wide=0 + same graph flags  (control: graphs need the segment)
#     A priori gate: G1/C0 <= 0.80 CONFIRMS the graphs-need-wide lever;
#     >= 0.95 refutes it and the next instrument is an nsys gap
#     attribution on the wide arm. G2 expected ~null (segments too
#     short) — if G2 wins too, the segment story is wrong.
#
# Protocol: byte-identical to the 26846337 arms except flags/dump.
# SUBMIT: sbatch scripts/cluster/scaling_levante/mpas_wide_census_graphs.sbatch
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
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/wh_cg_j${SLURM_JOB_ID}}"
DUMP="$OUTDIR/dump_rank0"
mkdir -p "$DUMP"; echo "outdir=$OUTDIR"
# Persistent compile cache OFF for this job (LEGOESM_JIT_CACHE_DIR=""
# = disabled, backend.py contract): two consecutive runs died at the
# first init kernel with "Failed to load in-memory CUBIN" while 64
# ranks raced a COLD per-job cache (26847346 pinned on l50003, 26848090
# on a clean node set — not hardware). 26846337 survived the same cache
# because its 4-rank parity arm prewarmed it. Bench arms recompile
# (~7 s/arm) — noise against a 20-min arm.
export LEGOESM_JIT_CACHE_DIR=""
rc=0

cat > "$OUTDIR/rank_wrapper.sh" <<'WRAP'
#!/usr/bin/env bash
set -uo pipefail
if [ "${SLURM_PROCID:-1}" = "0" ] && [ -n "${DUMP_DIR:-}" ]; then
  export XLA_FLAGS="--xla_dump_to=${DUMP_DIR}${XLA_FLAGS:+ $XLA_FLAGS}"
fi
exec "$PY_BIN" "$@"
WRAP
chmod +x "$OUTDIR/rank_wrapper.sh"
export PY_BIN="$PY"

GRAPH_FLAGS="--xla_gpu_enable_command_buffer=FUSION --xla_gpu_graph_min_graph_size=2"

run_arm () {  # wide extra_xla_flags dump tag
  local WH=$1 XF=$2 DP=$3 TAG=$4
  echo "=== s9@64 wide=$WH xla='$XF' dump=$DP [$TAG] ==="
  DUMP_DIR=$([ "$DP" = "1" ] && echo "$DUMP" || echo "") \
  XLA_FLAGS="$XF" \
  LEGOESM_MPAS_WIDE_HALO=$WH LEGOESM_MPAS_RAGGED_HALO=0 JAX_ENABLE_X64=0 \
    srun --nodes=16 --ntasks=64 --ntasks-per-node=4 --gpus-per-node=4 \
      --gpu-bind=none --kill-on-bad-exit=1 \
    "$OUTDIR/rank_wrapper.sh" scripts/bench/bench_mpas_spmd_scaling.py \
      --multicontroller --n-devices 64 --subdivision 9 --nlev 26 \
      --steps 12 --warmup 3 --lloyd 0 --partition-method sfc \
      --reorder-for 64 \
      --out "$OUTDIR/${TAG}.jsonl" || { echo "${TAG} FAILED"; rc=1; }
}

run_arm 1 ""             1 c0_wide_dump
run_arm 1 "$GRAPH_FLAGS" 0 g1_wide_graphs
run_arm 0 "$GRAPH_FLAGS" 0 g2_default_graphs

echo "=== CENSUS (wide executable) ==="
set +e
"$PY" scripts/bench/census_xla_dump.py --dump-dir "$DUMP" \
  --out "$OUTDIR/census_wide.json"
"$PY" - "$OUTDIR" <<'EOF'
import json, math, re, sys, pathlib
out = pathlib.Path(sys.argv[1])

# --- census: CP count MUST be the offline 11; entry-op slot census ---
cen = json.load(open(out / "census_wide.json"))
cp = cen["counts"].get("collective-permute", 0) + cen["counts"].get(
    "collective-permute-start", 0)
mod = out / "dump_rank0" / cen["module"]
in_entry, n_entry = False, 0
for line in open(mod):
    if line.startswith("ENTRY "):
        in_entry = True; continue
    if in_entry:
        s = line.strip()
        if s == "}":
            break
        if re.match(r"^(ROOT )?%?[\w\.\-]+ = ", s):
            n_entry += 1
print(f"census: wide CP={cp} entry_ops={n_entry} (default ref: 33 CP / "
      f"301 ops, dump 26791341)")
if cp != 11:
    print("CENSUS VERDICT: wide step does NOT contain 11 CP — the "
          "26846337 0.970 verdict is VOID; debug the GPU wide lane")
    sys.exit(4)

# --- timing arms ---
t = {}
for tag, wide in (("c0_wide_dump", True), ("g1_wide_graphs", True),
                  ("g2_default_graphs", False)):
    rows = [json.loads(l) for l in open(out / f"{tag}.jsonl")]
    r = rows[0]
    ok = (r.get("wide_halo") == wide and r.get("n_devices") == 64
          and r.get("platform") == "gpu"
          and math.isfinite(r["steady_median_ms"]))
    if not ok:
        print(f"{tag}: binding fail"); sys.exit(1)
    t[tag] = r["steady_median_ms"]
    print(f"{tag}: {t[tag]:9.2f} ms")
g1 = t["g1_wide_graphs"] / t["c0_wide_dump"]
print(f"graphs-on-wide ratio G1/C0 = {g1:.3f} | "
      f"G2 control {t['g2_default_graphs']:.2f} ms "
      f"(26846337 off arms: 10.31/10.33)")
if g1 <= 0.80:
    print("VERDICT: CONFIRMED — command buffers collapse the wide "
          "step's launch floor")
elif g1 >= 0.95:
    print("VERDICT: REFUTED — slot floor not launch-bound; next "
          "instrument nsys gap attribution on the wide arm")
    sys.exit(2)
else:
    print("VERDICT: intermediate — reprice")
    sys.exit(3)
EOF
rcv=$?
[ "$rcv" -ne 0 ] && rc=$rcv
echo "DONE rc=$rc"; exit $rc
