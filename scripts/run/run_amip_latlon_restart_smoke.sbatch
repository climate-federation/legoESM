#!/bin/bash
#SBATCH --job-name=amip_latlon_restart_smoke
#SBATCH --output=results/amip_latlon_restart_smoke/slurm_%j.out
#SBATCH --error=results/amip_latlon_restart_smoke/slurm_%j.err
#SBATCH --account=glab
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Integration smoke for the Stage-3-D lat-lon MPI checkpoint chain.
#
# The unit tests in ``tests/distributed/test_latlon_mpi_checkpoint.py``
# cover the gather/scatter primitives + the collective-safety
# assertions in isolation (SimpleNamespace fake drivers).  They do NOT
# exercise the production ``ModelDriver.save_checkpoint`` →
# ``load_checkpoint`` path: the rank-0 grid swap, the ``.meta.json``
# digest validation, the npz dtype round-trip, ``self._carry_aux``
# preservation, or the chained ``--restart-from`` invocation.
#
# Codex review round 2 of commit 0ade35b0 flagged this gap as MEDIUM
# coverage debt.  This sbatch closes it WITHOUT bloating the
# fast-path halo smoke: a small 2°/4-rank AMIP run that completes a
# first checkpoint at day 2 and then a restart leg from day 2 to
# day 4.  Day-4 state must equal day-4 state of the non-restart
# reference run (run separately under the same seed at 4 ranks).
#
# Usage::
#
#   sbatch scripts/run_amip_latlon_restart_smoke.sbatch
#
# This gates the production 100 y AMIP submission
# (``run_amip_1deg_latlon_mpi.sbatch``).

set -euo pipefail

REPO_ROOT="/burg-archive/glab/users/pg2328/legoESM"
cd "${REPO_ROOT}"
mkdir -p results/amip_latlon_restart_smoke

PY="${HOME}/.venvs/legoesm-mpi/bin/python"
if [[ ! -x "${PY}" ]]; then
    echo "FATAL: ${PY} not found.  Build the venv first:" >&2
    echo "    sbatch scripts/data/setup_legoesm_mpi_venv.sbatch" >&2
    exit 2
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /etc/profile.d/modules.sh 2>/dev/null || true
module load openmpi/gcc/64/4.1.7a1 2>/dev/null || true

if [[ -n "${SLURM_JOB_ID:-}" ]] && command -v srun >/dev/null; then
    MPI_LAUNCH=(srun --mpi=pmix --cpus-per-task=1)
else
    MPI_LAUNCH=(mpirun)
fi

RES=90  # 2° at the equator
# At n_lat=90, n_lon=180 → dx_equator = 2π R / n_lon ≈ 2.22e5 m.
# Equatorial advective CFL with c=300 m/s, CFL=0.8 → dt_max ≈ 593 s.
# DT=540 leaves ~9% margin and avoids the factory clamp.  The polar
# filter (--use-polar-filter, below) handles the pole-cell CFL by
# truncating modes that violate it at this dt — Stage 3-E commit
# 79837145.  Before Stage 3-E, dt would have been clamped to ~10 s.
DT=540.0
RAD=7    # ~1-h radiation cadence at dt=540 (3780 s)

REF_DIR="${REPO_ROOT}/results/amip_latlon_restart_smoke/reference"
CHAIN_DIR="${REPO_ROOT}/results/amip_latlon_restart_smoke/chain"
mkdir -p "${REF_DIR}" "${CHAIN_DIR}"

echo "========================================"
echo "  AMIP lat-lon MPI restart-smoke"
echo "  Job ID: ${SLURM_JOB_ID:-<no-slurm>}"
echo "  Resolution: ${RES} (n_lat=${RES}, n_lon=$((2*RES)))"
echo "  4 ranks, dt=${DT} s"
echo "  Reference run dir: ${REF_DIR}"
echo "  Chain run dir:     ${CHAIN_DIR}"
echo "========================================"

# ----------------------------------------------------------------------
# Leg 1: reference 4-day run (no restart).  Yields the canonical
# day-4 checkpoint we compare the chained run against.
# ----------------------------------------------------------------------
echo
echo "--- Reference: 4-day continuous run ---"
"${MPI_LAUNCH[@]}" -n 4 "${PY}" "${REPO_ROOT}/scripts/run_amip.py" \
    --resolution "${RES}" \
    --grid-type latlon \
    --discretization finite_volume \
    --days 4 \
    --dt "${DT}" \
    --rad-update-steps "${RAD}" \
    --diag-days 1 \
    --checkpoint-days 2 \
    --output "${REF_DIR}" \
    --dataset analytical \
    --use-polar-filter --time-integrator ssp_rk3_scan

REF_DAY4="${REF_DIR}/checkpoint_day_0004.npz"
if [[ ! -f "${REF_DAY4}" ]]; then
    echo "FATAL: reference day-4 checkpoint not produced at ${REF_DAY4}" >&2
    exit 3
fi

# ----------------------------------------------------------------------
# Leg 2: chained 2-day run + restart from day-2 → day-4.  Exercises
# save_checkpoint(day-2) AND load_checkpoint(day-2) on the production
# Field-wrapped HydrostaticState under MPI.
# ----------------------------------------------------------------------
echo
echo "--- Chain leg 1: 0 → 2 days, writes day-2 checkpoint ---"
"${MPI_LAUNCH[@]}" -n 4 "${PY}" "${REPO_ROOT}/scripts/run_amip.py" \
    --resolution "${RES}" \
    --grid-type latlon \
    --discretization finite_volume \
    --days 2 \
    --dt "${DT}" \
    --rad-update-steps "${RAD}" \
    --diag-days 1 \
    --checkpoint-days 2 \
    --output "${CHAIN_DIR}" \
    --dataset analytical \
    --use-polar-filter --time-integrator ssp_rk3_scan

CHAIN_DAY2="${CHAIN_DIR}/checkpoint_day_0002.npz"
if [[ ! -f "${CHAIN_DAY2}" ]]; then
    echo "FATAL: chain day-2 checkpoint missing at ${CHAIN_DAY2}" >&2
    exit 3
fi

echo
echo "--- Chain leg 2: restart day-2 → day-4 ---"
# ``--start-day 2`` makes ``cfg.start_day = 2`` so the elapsed-day
# math in ``ModelDriver.save_checkpoint`` (``elapsed_day = day -
# cfg.start_day``) produces ``day_0004.npz``.  Without this flag
# ``cfg.start_day`` defaults to 0 and the chain-leg's first save lands
# at ``day_0006.npz`` (the absolute ``START_DAY + N_DAYS`` from
# ``_run_compiled``), which would mismatch the reference filename
# even though the underlying STATE is bit-equivalent.  Codex review
# round 3 caught this filename-vs-state inconsistency.
"${MPI_LAUNCH[@]}" -n 4 "${PY}" "${REPO_ROOT}/scripts/run_amip.py" \
    --resolution "${RES}" \
    --grid-type latlon \
    --discretization finite_volume \
    --start-day 2 \
    --days 4 \
    --dt "${DT}" \
    --rad-update-steps "${RAD}" \
    --diag-days 1 \
    --checkpoint-days 2 \
    --output "${CHAIN_DIR}" \
    --dataset analytical \
    --use-polar-filter --time-integrator ssp_rk3_scan \
    --restart-from "${CHAIN_DAY2}"

CHAIN_DAY4="${CHAIN_DIR}/checkpoint_day_0004.npz"
if [[ ! -f "${CHAIN_DAY4}" ]]; then
    echo "FATAL: chain day-4 checkpoint missing at ${CHAIN_DAY4}" >&2
    exit 3
fi

# ----------------------------------------------------------------------
# Compare: REF day-4 vs CHAIN day-4.  The state-digest in
# ``.meta.json`` is the natural integrity invariant — save_restart
# computes it from the sorted-key + raw-bytes serialization of the
# state.  If the chain leg's reload + 2-day continuation diverges
# from the reference at all, the digests will differ.
# ----------------------------------------------------------------------
echo
echo "--- Verify: REF day-4 vs CHAIN day-4 ---"
"${PY}" - <<'PYEOF'
import json
import pathlib
import sys

ref_meta = pathlib.Path(
    "results/amip_latlon_restart_smoke/reference/checkpoint_day_0004.meta.json"
)
chain_meta = pathlib.Path(
    "results/amip_latlon_restart_smoke/chain/checkpoint_day_0004.meta.json"
)
if not ref_meta.exists() or not chain_meta.exists():
    print(f"FATAL: meta.json missing.  ref={ref_meta.exists()} "
          f"chain={chain_meta.exists()}",
          file=sys.stderr)
    sys.exit(4)

r = json.loads(ref_meta.read_text())
c = json.loads(chain_meta.read_text())

print(f"  ref   day-4 digest: {r['state_digest'][:32]}...")
print(f"  chain day-4 digest: {c['state_digest'][:32]}...")
print(f"  ref   day-4 step:   {r['step']}")
print(f"  chain day-4 step:   {c['step']}")

if r["state_digest"] != c["state_digest"]:
    print(
        "FATAL: state digest mismatch between reference and restart-"
        "chain run.  The lat-lon MPI save → load path is NOT a "
        "round-trip identity; AMIP 100 y restart chain would corrupt.",
        file=sys.stderr,
    )
    sys.exit(5)
if r["step"] != c["step"]:
    print(
        f"FATAL: step counter mismatch: ref={r['step']} "
        f"chain={c['step']}",
        file=sys.stderr,
    )
    sys.exit(5)
print("OK: chain day-4 matches reference day-4 (digest + step).")
PYEOF

echo
echo "========================================"
echo "  AMIP lat-lon MPI restart smoke PASSED"
echo "  Date: $(date -u)"
echo "========================================"
