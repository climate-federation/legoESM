#!/bin/bash -l
#SBATCH --account=bb1596_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=nature_fesom_%j.log

# FESOM2 lane of the Nature scaling figure: fesom_jax (the JAX port of FESOM2,
# separate package + AWI env) on the forca20 mesh (2.13M nodes, 5-22 km,
# 70 levels), FULL model (JRA55 1958 + PHC IC + prognostic ice).  fesom_jax is
# x64-only, so this lane is float64 only.  Strong scaling over the
# decompositions that exist for forca20 (dist_16/32/64/128).
#
# GPU ONLY.  fesom_jax/docs/PARALLELISM.md: "Do not attempt multi-node CPU on
# jaxlib 0.10.1" - all four CPU variants crash, hang, or run 6x slower.  There
# is deliberately NO CPU lane; BACKEND=cpu is refused below.
#
# Halo transport per point: ragged AND coloured at npes<=32 (so the choice is
# measured); coloured AND the HYBRID 'coloured+sshpadded' at npes>=64 —
# coloured for the 3-D fields, padded dense a2a for the free-surface CG's
# ~2x70 single-level exchanges (fesom_jax 0d1fe62; owner-adopted 2026-09-22
# on job 27593181: 253.1 -> 227.6/228.3 ms at 64 GPUs, -10%).  FORCE_HALO / HALO_LIST
# override and then run ONLY the requested transport.  Each (npes, transport)
# arm is run TWICE (STEPS=150).  One honest per-arm receipt per arm is written;
# the plotter keys by (component, grid, backend, precision, mode, resolution,
# n_devices) and keeps the FASTEST row, so no winner selection here.
#
#   BACKEND=gpu sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 32 nature_fesom.sbatch
set -uo pipefail
BACKEND="${BACKEND:-gpu}"
if [[ "$BACKEND" != gpu ]]; then
  echo "ERROR: this lane is GPU-only (BACKEND='$BACKEND' refused)." >&2
  echo "fesom_jax/docs/PARALLELISM.md: \"Do not attempt multi-node CPU on jaxlib 0.10.1\"" >&2
  echo "- all four CPU variants crash, hang, or run 6x slower; JAX-CPU is effectively single-node." >&2
  exit 1
fi
FJ=/work/bd1083/b309178/diffESM/fesom_jax
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
D=/work/ab0995/a270088/meshes/FORCA20
SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/nature_fesom_${BACKEND}_j${SLURM_JOB_ID}}"; mkdir -p "$OUTDIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"
STEPS="${STEPS:-150}"
ARM_TIMEOUT="${ARM_TIMEOUT:-2400}"
# Physics flag sets: BOTH include --full 1, otherwise the toggles are
# ineffective (unforced ocean-only physics).
PHYS_SET="${PHYS_SET:-ours}"
if [[ "$PHYS_SET" == ours ]]; then
  PHYS=(--full 1)
elif [[ "$PHYS_SET" == paper ]]; then
  PHYS=(--full 1 --ice 1 --mevp 1 --tke 1 --kpp 0 --gm 0 --zstar 1)  # authors' production flags
else
  echo "ERROR: PHYS_SET must be ours|paper, got '$PHYS_SET'"; exit 1
fi
PHYS_FLAGS="$(printf '%s ' "${PHYS[@]}")"
# Halo selection: ragged for npes<=32, coloured for npes>=64.  An explicit
# request (FORCE_HALO, or a per-point entry in HALO_LIST "16:ragged ...") is
# honoured verbatim and runs ONLY that transport for that point.
HALO_FOR() {
  local n=$1 pair
  if [[ -n "${FORCE_HALO:-}" ]]; then echo "$FORCE_HALO"; return; fi
  for pair in ${HALO_LIST:-}; do
    [[ "${pair%%:*}" == "$n" ]] && { echo "${pair#*:}"; return; }
  done
  if (( n <= 32 )); then echo ragged; else echo coloured; fi
}
HALO_EXPLICIT() {  # true if the point has an explicit request
  local n=$1 pair
  [[ -n "${FORCE_HALO:-}" ]] && return 0
  for pair in ${HALO_LIST:-}; do [[ "${pair%%:*}" == "$n" ]] && return 0; done
  return 1
}
cd "$FJ" || { echo "ERROR: cannot cd to fesom_jax checkout '$FJ'" >&2; exit 1; }
export PYTHONPATH="$FJ:${PYTHONPATH:-}"
unset FESOM_CG_CHEB   # the authors' production benchmark scripts unset this
# A receipt that cannot name the code it measured is worthless, so an
# unreadable checkout is fatal rather than an empty SHA field.
FJ_SHA=$(git rev-parse HEAD) || { echo "ERROR: '$FJ' is not a readable git checkout" >&2; exit 1; }
[[ -n "$FJ_SHA" ]] || { echo "ERROR: empty fesom_jax SHA" >&2; exit 1; }
echo "backend=$BACKEND nodes=$NODES outdir=$OUTDIR steps=$STEPS phys_set=$PHYS_SET fesom_jax=$(git rev-parse --short HEAD)"
rc=0
run_arm() {
  # $1=npes $2=halo $3=repeat ; uses $nn, $OUTDIR, $STEPS, $PHYS_FLAGS
  local npes=$1 halo=$2 rep=$3
  local log="$OUTDIR/forca20_d${npes}_${halo}_r${rep}.log"
  local arm="$OUTDIR/forca20_d${npes}_${halo}_r${rep}.jsonl"
  echo "=== forca20 npes=$npes halo=$halo rep=$rep nn=$nn $(date '+%F %T')"
  local HALO_FLAGS
  case "$halo" in
    coloured+sshpadded) HALO_FLAGS=(--halo coloured --ssh-halo padded) ;;
    ragged|coloured|padded|allgather) HALO_FLAGS=(--halo "$halo") ;;
    *) echo "!!! unknown halo tag '$halo'"; rc=1; return 1 ;;
  esac
  if ! timeout --kill-after=30 "$ARM_TIMEOUT" srun --export=ALL "${SR[@]}" --cpus-per-task=64 --kill-on-bad-exit=1 \
       "$PY" scripts/bench/bench_forward_scaling.py --name forca20 --mesh-dir "$M" --dist-dir "$D" \
         --npes "$npes" "${HALO_FLAGS[@]}" --steps "$STEPS" --dt 240 --ic-dir "$M" \
         --year 1958 "${PHYS[@]}" < /dev/null > "$log" 2>&1; then
    echo "!!! FAILED npes=$npes halo=$halo rep=$rep"; rc=1; return 1
  fi
  # Per-arm receipt, written atomically (tmp + os.replace).  steady_median_ms
  # carries the same value as per_step_ms (one number per invocation) because
  # the plotter reads that name; the plotter does the transport/repeat
  # selection, so this row is honest, not a winner.
  if ! python3 - "$log" "$arm" "$npes" "$halo" "$rep" "$BACKEND" "$SLURM_JOB_ID" "$STEPS" "$FJ_SHA" "$PHYS_SET" "$PHYS_FLAGS" <<'EOF'
import json, math, os, re, sys
# 10 argv values after the script name: argv[1:11]
log, out, npes, halo, rep, backend, job, steps, sha, phys_set, phys_flags = sys.argv[1:12]
try:
    txt = open(log).read()
except OSError as e:
    print("cannot read log:", e); sys.exit(1)
m = re.search(r"nod2D=(\d+) nl=(\d+) npes=(\d+).*?per_step=\s*([\d.]+) ms.*?throughput=\s*([\d.]+)", txt)
fin = re.search(r"\[bench-finite\].*?nonfinite_T=(\d+) nonfinite_uv=(\d+)", txt)
if not m or not fin or int(m.group(3)) != int(npes):
    print("no valid summary/finite line in", log); sys.exit(1)
if int(fin.group(1)) or int(fin.group(2)):
    print("NONFINITE state in", log, fin.group(0)); sys.exit(1)
per_step = float(m.group(4))
if not (math.isfinite(per_step) and math.isfinite(float(m.group(5))) and per_step > 0):
    print("NONFINITE/nonpositive timing in", log); sys.exit(1)
# The bench line reports the platform it actually ran on; trusting the
# launcher's intent instead would let a silent CPU fallback be published as a
# GPU point.
plat = re.search(r"plat=(\w+)", txt)
if not plat or plat.group(1) != backend:
    print("PLATFORM MISMATCH in", log, "expected", backend, "got",
          plat.group(1) if plat else "none"); sys.exit(1)
row = dict(component="fesom_ocean", grid_type="fesom", resolution="forca20", mode="strong",
           n_devices=int(npes), nod2d=int(m.group(1)), n_levels=int(m.group(2)),
           precision="float64", backend=backend, halo=halo, repeat=int(rep),
           steps=int(steps), per_step_ms=per_step,
           steady_median_ms=per_step,  # same value; the plotter reads this name
           mnodlev_per_s=float(m.group(5)), dt=240.0,
           vertices_per_device=int(m.group(1)) / int(npes),
           phys_set=phys_set, phys_flags=phys_flags,
           fesom_jax_sha=sha, slurm_job_id=job, log=log,
           valid=True, finite_ok=True)
tmp = out + ".tmp"
with open(tmp, "w") as f:
    f.write(json.dumps(row) + "\n")
os.replace(tmp, out)
print(row)
EOF
then
    echo "!!! PARSE/VALIDITY failure npes=$npes halo=$halo rep=$rep"; rc=1; return 1
  fi
}
for NPES in ${NPES_LIST:-16 32 64 128}; do
  nn=$(( NPES / 4 )); export JDIST=1 GPUS_PER_NODE=4; unset JDIST_CPU
  SR=(--nodes="$nn" --ntasks="$nn" --ntasks-per-node=1 --gpus-per-node=4 --cpu-bind=none)
  if (( nn > NODES )); then echo "[skip] npes=$NPES needs $nn nodes"; continue; fi
  BASE=$(HALO_FOR "$NPES")
  if HALO_EXPLICIT "$NPES"; then HALOS=("$BASE")
  elif (( NPES <= 32 )); then HALOS=(ragged coloured)
  else HALOS=(coloured coloured+sshpadded); fi
  for HALO in "${HALOS[@]}"; do
    for REP in 1 2; do
      arm="$OUTDIR/forca20_d${NPES}_${HALO}_r${REP}.jsonl"
      # Resumability: skip only if this arm already has a valid receipt that
      # measured THIS configuration. A receipt from a different physics set,
      # step count or code revision answers a different question, so it is
      # re-run rather than reused. Never delete existing arm files.
      if python3 -c 'import json,sys
try: r=json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
ok=(r.get("valid") is True and r.get("phys_set")==sys.argv[2]
    and str(r.get("steps"))==sys.argv[3] and r.get("fesom_jax_sha")==sys.argv[4]
    and r.get("halo")==sys.argv[5] and r.get("backend")==sys.argv[6])
sys.exit(0 if ok else 1)' "$arm" "$PHYS_SET" "$STEPS" "$FJ_SHA" "$HALO" "$BACKEND"; then
        echo "[skip-arm] npes=$NPES halo=$HALO rep=$REP has a valid per-arm receipt"; continue
      fi
      # The receipt that survives here measured a DIFFERENT configuration, so
      # retire it before the rerun: if the rerun then fails, the stale row must
      # not stay eligible for the figure.
      if [[ -e "$arm" ]] && ! mv -f "$arm" "$arm.superseded.$(date +%s%N)"; then
        echo "!!! cannot retire stale receipt $arm - refusing to rerun over it"; rc=1; continue
      fi
      run_arm "$NPES" "$HALO" "$REP" || true
    done
  done
  sleep 5
done
echo "=== DONE rc=$rc outdir=$OUTDIR ==="; exit "$rc"
