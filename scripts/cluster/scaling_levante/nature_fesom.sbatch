#!/bin/bash -l
#SBATCH --account=bd1179_gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=nature_fesom_%j.log

# FESOM2 lane of the Nature scaling figure: fesom_jax (the JAX port of FESOM2,
# separate package + AWI env) on the forca20 mesh (2.13M nodes, 5-22 km,
# 70 levels), FULL model (JRA55 1958 + PHC IC + prognostic ice), ragged halo.
# fesom_jax is x64-only, so this lane is float64 only.  Strong scaling over
# the decompositions that exist for forca20 (dist_16/32/64/128); the FESOM2
# Fortran partitioner would be needed for other counts, so no weak ladder.
#
#   BACKEND=gpu sbatch -p gpu --constraint=a100_80 --gpus-per-node=4 -N 32 nature_fesom.sbatch
#   BACKEND=cpu sbatch -p compute -A bd1179 -N 16 nature_fesom.sbatch   (JDIST_CPU, gloo)
set -uo pipefail
BACKEND="${BACKEND:-gpu}"
FJ=/work/bd1083/b309178/diffESM/fesom_jax
PY=/work/ab0995/a270088/mambaforge/envs/fesom-jax/bin/python
M=/work/ab0995/a270088/fesom_jax_meshes/mesh_forca20
D=/work/ab0995/a270088/meshes/FORCA20
SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/$USER}"
OUTDIR="${OUTDIR:-$SCRATCH/legoesm_scaling/nature_fesom_${BACKEND}_j${SLURM_JOB_ID}}"; mkdir -p "$OUTDIR"
NODES="${SLURM_JOB_NUM_NODES:-1}"
cd "$FJ"; export PYTHONPATH="$FJ:${PYTHONPATH:-}"
echo "backend=$BACKEND nodes=$NODES outdir=$OUTDIR fesom_jax=$(git rev-parse --short HEAD)"
rc=0
for NPES in ${NPES_LIST:-16 32 64 128}; do
  if [[ "$BACKEND" == gpu ]]; then
    nn=$(( NPES / 4 )); export JDIST=1 GPUS_PER_NODE=4; unset JDIST_CPU
    SR=(--nodes="$nn" --ntasks="$nn" --ntasks-per-node=1 --gpus-per-node=4 --cpu-bind=none)
    RAGGED=1
  else
    PPN="${PPN:-32}"; nn=$(( (NPES + PPN - 1) / PPN )); export JDIST_CPU=1 JAX_PLATFORMS=cpu; unset JDIST
    SR=(--nodes="$nn" --ntasks="$NPES" --ntasks-per-node="$PPN" --cpu-bind=cores --distribution=block:cyclic)
    RAGGED=0   # ragged_all_to_all is GPU-only
  fi
  if (( nn > NODES )); then echo "[skip] npes=$NPES needs $nn nodes"; continue; fi
  log="$OUTDIR/forca20_d${NPES}.log"
  echo "=== forca20 npes=$NPES nn=$nn $(date '+%F %T')"
  if ! timeout --kill-after=30 2400 srun --export=ALL "${SR[@]}" --kill-on-bad-exit=1 \
       "$PY" scripts/bench/bench_forward_scaling.py --name forca20 --mesh-dir "$M" --dist-dir "$D" \
         --npes "$NPES" --steps "${STEPS:-40}" --warmup 5 --dt 240 --full 1 --ic-dir "$M" \
         --year 1958 --ragged "$RAGGED" < /dev/null > "$log" 2>&1; then
    echo "!!! FAILED npes=$NPES"; rc=1; continue
  fi
  # receipt row: the bench prints one summary line; keep it machine-readable
  python3 - "$log" "$OUTDIR/forca20_d${NPES}.jsonl" "$NPES" "$BACKEND" "$SLURM_JOB_ID" <<'EOF'
import json, re, sys
log, out, npes, backend, job = sys.argv[1:]
txt = open(log).read()
m = re.search(r"nod2D=(\d+) nl=(\d+) npes=(\d+).*?per_step=\s*([\d.]+) ms.*?throughput=\s*([\d.]+)", txt)
fin = re.search(r"\[bench-finite\].*?nonfinite_T=(\d+) nonfinite_uv=(\d+)", txt)
if not m or not fin or int(m.group(3)) != int(npes):
    print("no valid summary/finite line in", log); sys.exit(1)
if int(fin.group(1)) or int(fin.group(2)):
    print("NONFINITE state in", log, fin.group(0)); sys.exit(1)
row = dict(component="fesom_ocean", grid_type="fesom", resolution="forca20", mode="strong",
           n_devices=int(npes), nod2d=int(m.group(1)), n_levels=int(m.group(2)),
           precision="float64", backend=backend, steady_median_ms=float(m.group(4)),
           mnodlev_per_s=float(m.group(5)), dt=240.0, slurm_job_id=job, log=log,
           valid=True, finite_ok=True)
open(out, "w").write(json.dumps(row) + "\n"); print(row)
EOF
  [[ $? -ne 0 ]] && { echo "!!! no receipt for npes=$NPES"; rc=1; }
  sleep 5
done
echo "=== DONE rc=$rc outdir=$OUTDIR ==="; exit "$rc"
