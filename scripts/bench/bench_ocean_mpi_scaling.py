#!/usr/bin/env python
"""Ocean CPU MPI scaling benchmark (lat-lon C-grid, latitude-band MPI).

Measures wall-clock time per step and SYPD for the lat-lon C-grid ocean
model (``LatLonCGridOceanModel``) under the latitude-band MPI
decomposition (``initialize_distributed_latlon`` +
``slice_cgrid_geometry_to_band`` + ``scatter_state_latlon_cgrid_ocean``).

Design: ONE case per MPI invocation, mirroring
``scripts/bench/run_cpu_mpi_scaling.py`` (JIT-compile step, warmup
steps, fused ``lax.scan`` timing block, MPI barriers around the timed
region, rank 0 writes results).  Output reuses ``TimingResult`` /
``write_csv`` / ``write_json`` from ``run_levante_gpu_scaling.py``
exactly like ``bench_ocean_gpu_scaling.py`` does, so the CSV schema is
identical and the existing scaling plotters work unchanged
(``n_gpus`` column carries the MPI rank count for these CPU runs).

Rank-local construction (adversarial-review constraint)
-------------------------------------------------------
``LatLonCGridOceanModel`` caches static structures at build time (e.g.
the rigid-lid island machinery,
``ocean/dynamics/ocean_model_latlon_cgrid.py`` ``_ensure_rigid_lid_data``),
so under MPI everything is rank-local FROM CONSTRUCTION:

1. build the global grid + global rest-state IC with the LOCAL halo
   backend (cheap, host-side — the documented convention of
   ``scatter_state_latlon_cgrid_ocean``: the global state is replicated
   on every rank, then sliced);
2. arm the MPI halo backend via ``initialize_distributed_latlon``;
3. slice geometry (``slice_cgrid_geometry_to_band``) and vertical
   coordinate (``slice_zcoord_to_band``) to this rank's band;
4. build the model ON THE BAND GEOMETRY — the model never sees the
   global grid;
5. scatter the global state to the band and step; halo exchange happens
   inside the backend-dispatched C-grid operators (``pad_ns_zero`` /
   ``pad_with_pole_bc_lat`` -> MPI sendrecv at partition cuts).

Barotropic solver under MPI
---------------------------
``implicit_cn`` (the serial production default) solves the free-surface
Helmholtz system.  Single-rank uses ``jax.scipy.sparse.linalg.cg``,
whose dot products are RANK-LOCAL and whose ``while_loop`` trip count is
residual-dependent — which would deadlock under the band decomposition
(each rank iterating a different number of times -> mismatched sendrecv
call counts).  The MULTI-RANK path now dispatches (on ``is_distributed``)
to a HAND-ROLLED fixed-iteration distributed PCG: a static
``fori_loop`` of exactly ``barotropic_implicit_pcg_fixed_iters`` (= M)
iterations.  Standard CG needs two sequentially-dependent dot products
per iteration (``p·Ap`` for alpha, ``r·z`` for beta), so it issues TWO
batched ``allreduce(SUM)`` per iteration (the residual ``r·r`` is folded
into the second, free).  The solve is UNROLLED and differentiated
straight through for AD (the halo sendrecv-VJP + allreduce-SUM dots are
AD-safe; ``custom_linear_solve`` is NOT used — it cannot transpose the
MPI-halo custom_vjp).  Every rank runs the identical collective schedule
=> no deadlock, ~``2*M``
reductions per barotropic step regardless of global resolution (THE
weak-scaling lever vs ``explicit_substep``, whose substep count — and
thus reduction count — grows with resolution).  Both solvers are valid
at all rank counts; the ladder picks one via ``--baro-solver`` and
compares identical numerics.

Weak scaling convention (``--weak-style``)
------------------------------------------
``--resolution`` is the per-rank base in weak mode; two styles:

``band`` (default — latitude-band algorithmic weak scaling)
    ``n_lat = rows * n_ranks`` with ``n_lon = 2 * rows`` held FIXED
    across the ladder, so cells/rank AND halo bytes/rank are exactly
    constant (each cut exchanges the same ``n_lon``-wide rows).  The
    domain aspect ratio distorts as np grows (bands get tall and
    narrow relative to a real production grid) — this ladder isolates
    the ALGORITHMIC weak-scaling cost.  ``--mode weak --resolution 64``
    at np=1 is exactly the LL64 case (n_lat=64, n_lon=128) of
    ``bench_ocean_gpu_scaling.py``.

``aspect`` (production-realism cross-check)
    ``n_lat ~= resolution * sqrt(n_ranks)`` rounded to a multiple of
    ``n_ranks`` (equal rows/band), with the standard square-cell
    ``n_lon = 2 * n_lat``.  The grid keeps the production aspect ratio,
    so cells/rank only approximately matches the base (integer-rounding
    drift, e.g. np=2: 90x180 -> -1.1% cells/rank vs 64x128) — the
    ACTUAL cells/rank is recorded in the CSV row (``cells_per_gpu``),
    so efficiency post-processing can normalise exactly.  At np=1 both
    styles coincide.  The sbatch sweep runs the band ladder plus aspect
    at np in {1, 4, 16}.

Gates (``--check-conservation`` / ``--parity-gate``)
----------------------------------------------------
``--check-conservation`` now ASSERTS: any of area / mean-eta / heat /
salt drifting beyond ``--cons-rtol`` exits nonzero (code 4).  NOTE the
raw scheme (no conservation fixer in this benchmark config) physically
drifts heat/salt at ~1e-8/step relative (LL64 f64, measured: 1.5e-7 /
2.7e-7 over 12 steps), so gate runs over N steps need a calibrated
tolerance (the sbatch gate uses 1e-6 for the ~33-step smoke); the
strict 1e-9 default is for short f64 runs / regression hunting.

``--parity-gate`` (smoke sizes only, n_lat <= 128): every rank computes
a full serial reference trajectory BEFORE the MPI backend is armed,
then the final band state is gathered and compared field-by-field
(T, S, eta, u, v) on rank 0 — exits nonzero (code 5) on mismatch.
This catches partition-cut corruption that global conservation
integrals CANNOT see (telescoping flux divergence with locally wrong
face values — the ``tvd_to_v_points`` failure mode; see
tests/ocean/distributed/test_ocean_mpi_tvd_parity.py).

Minimum band height
-------------------
``pad_halo_latlon_mpi`` raises when ``halo > n_lat_local``.  The ocean
operators currently exchange halo=1, but this driver enforces the same
>=2 rows/rank floor as the atmospheric lat-lon MPI path (halo=2
PPM/biharmonic ceiling) so the guard stays valid if deeper-halo schemes
are enabled.

Usage
-----
Single rank (serial baseline, same semantics as the GPU bench)::

    python scripts/bench/bench_ocean_mpi_scaling.py \
        --mode strong --resolution 64 --precision float64

Multi-rank::

    mpirun -np 4 python scripts/bench/bench_ocean_mpi_scaling.py \
        --mode strong --resolution 64 --baro-solver explicit_substep \
        --check-conservation --cons-rtol 1e-6

    mpirun -np 4 python scripts/bench/bench_ocean_mpi_scaling.py \
        --mode weak --resolution 64 --baro-solver explicit_substep

    mpirun -np 4 python scripts/bench/bench_ocean_mpi_scaling.py \
        --mode weak --weak-style aspect --resolution 64 \
        --baro-solver explicit_substep

Smoke-size parity gate (gathered fields vs pre-arming serial ref)::

    mpirun -np 2 python scripts/bench/bench_ocean_mpi_scaling.py \
        --mode strong --resolution 64 --baro-solver explicit_substep \
        --n-warmup 2 --n-timing 10 --parity-gate
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple

# NOTE: no JAX / legoesm imports at module load — JAX env vars
# (JAX_PLATFORMS / JAX_ENABLE_X64 / thread pinning) must be set first
# via ``_configure_jax_cpu``.  All heavy imports are function-scoped.

# Reuse the atm CPU-MPI script's JAX/CPU configuration + MPI bootstrap
# (same timing discipline) and the GPU-scaling output helpers (same CSV
# schema -> existing plotters work).  Neither module imports JAX at
# module scope.
sys.path.insert(0, str(Path(__file__).parent))
from run_cpu_mpi_scaling import _configure_jax_cpu, _init_mpi  # noqa: E402
from run_levante_gpu_scaling import (  # noqa: E402
    ScalingReport,
    TimingResult,
    write_csv,
    write_json,
)


def _configure_jax_gpu(precision: str) -> None:
    """Pin THIS MPI rank to one local GPU and run JAX on cuda.

    The ocean lat-lon multi-GPU path: each rank owns a latitude band on its
    OWN GPU; halos cross the PCIe pair via the same mpi4jax sendrecv as the
    CPU path.  Must run BEFORE any JAX import (sets CUDA_VISIBLE_DEVICES +
    JAX_PLATFORMS).  Local rank from the MPI launcher env (OpenMPI / SLURM).
    """
    local = (os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
             or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK"))
    if local is None:
        # SLURM_LOCALID is exported even in a plain sbatch step (ntasks=1, no
        # srun); pinning on it THERE hides all but GPU 0 from a single-process
        # multi-GPU run (the documented silent eff=0.5 bug, jobs 8454397/
        # 8454737). Only honor it for a genuine multi-task launch (codex
        # capstone LOW).
        slid = os.environ.get("SLURM_LOCALID")
        nt = os.environ.get("SLURM_NTASKS", "1")
        if slid is not None and nt.isdigit() and int(nt) > 1:
            local = slid
    if local is None:
        local = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = local   # one GPU visible per rank
    os.environ["JAX_PLATFORMS"] = "cuda"
    if precision == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"
    # Do NOT preallocate the whole GPU (two ranks share a node; each takes
    # its own device but the allocator must not grab 90% up front).
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Benchmark-excitation parameters (NOT physics tunables): a small
# deterministic, mask-aware perturbation of the rest state so the timed
# step exercises non-trivial dynamics (gravity waves + advection) and
# the conservation check sees real fluxes.  Same spirit as the
# perturbation in tests/ocean/distributed/test_ocean_mpi_conservation.py.
ETA_PERTURBATION_M = 0.05
T_PERTURBATION_C = 0.1

# >=2 lat rows per rank: matches the atmospheric lat-lon band-MPI floor
# (halo=2 exchange ceiling; see module docstring).
MIN_ROWS_PER_RANK = 2

BARO_SOLVER_CHOICES = ("explicit_substep", "implicit_cn")
WEAK_STYLE_CHOICES = ("band", "aspect")

# --check-conservation default tolerances (overridable via --cons-rtol).
# f64: machine-tight short-run default; f32 scaled up (single-precision
# reductions over ~1e5 cells drift far above 1e-9).
CONS_RTOL_DEFAULTS = {"float64": 1.0e-9, "float32": 1.0e-4}

# --parity-gate: gathered-field vs pre-arming serial reference.
PARITY_FIELDS = ("T", "S", "eta", "u", "v")
# Tolerances are calibrated to the measured np=1 cross-program noise
# floor (job 8454911, 2026-06-10): the armed-MPI leg and the local
# serial leg compile DIFFERENT XLA programs (halo dispatch), so
# fusion/reduction reorder gives eta/u/v |diff| up to ~1.3e-9 after 12
# steps at f64 even with ZERO ranks cuts (T/S stay bit-exact).  Real
# partition-cut corruption (wrong metric/neighbor row) shows up at
# O(1e-3 * field) — orders of magnitude above these gates.
PARITY_TOLS = {  # precision -> (rtol, atol)
    "float64": (1.0e-7, 1.0e-8),
    "float32": (1.0e-4, 1.0e-5),
}
# implicit_cn at np>=2 dispatches to the fixed-M distributed PCG whose
# per-iteration dot products are allreduce-summed PARTIAL sums — a
# different fp summation order from any serial reference (full-array
# sum), even with the solver-matched ``force_pcg`` reference.  fp
# non-associativity seeds ~1e-16 per dot, amplified through M=60
# iterations x O(30) steps into ~1e-5-abs field deltas (regate job
# 8459376: explicit lane bit-exact PASS, implicit lane S/eta at exact
# f64 reduction-noise scale).  This lane therefore gates at solver-
# tolerance level, NOT bit-exactness; the explicit_substep lane (same
# halo machinery, no global dots) remains the bit-exact tripwire —
# real partition-cut corruption shows up at O(1e-3 * field) in BOTH.
# Tolerances sit ~2.5x above the MEASURED 6-step reduction-noise floor
# (S abs 3.8e-6 at |S|~35, T abs 1.9e-6, eta abs 3e-8).  Effective
# allclose trip thresholds (atol + rtol*|want|): eta ~1.0e-5, S
# ~1.1e-5, T ~1.9e-5 — an implicit-lane cut bug of >~2e-5 absolute
# trips on every field, smaller ones trip on the near-zero fields
# (codex rounds 2-3: an earlier (1e-5, 1e-4) draft was loose enough to
# mask a 1e-5 bug entirely; this one is floor-limited, not slack).
# atol-dominant on purpose: a large rtol would scale with |T|~300 K
# and reopen the hole.  Gate the PCG lane at SMOKE windows (<~8
# advanced steps) — the noise floor grows with steps (chaotic
# amplification), 33-step runs reach ~1e-5 in u and would
# false-positive.
PARITY_TOLS_PCG_MPI = {  # precision -> (rtol, atol)
    "float64": (3.0e-8, 1.0e-5),
    "float32": (3.0e-5, 1.0e-3),
}
# The serial reference runs the FULL global model on every rank — only
# sensible at smoke sizes.
PARITY_GATE_MAX_NLAT = 128


# ===========================================================================
# Case geometry
# ===========================================================================

def resolve_grid_size(
    mode: str, resolution: int, n_ranks: int, weak_style: str = "band",
) -> tuple[int, int]:
    """Return (n_lat, n_lon) for a case.

    strong: ``resolution`` = global n_lat, n_lon = 2*n_lat.
    weak/band:   ``resolution`` = rows/rank; n_lat = rows*n_ranks and
            n_lon = 2*rows held fixed across the ladder (cells/rank and
            halo bytes/rank exactly constant; aspect ratio distorts).
    weak/aspect: n_lat ~= resolution*sqrt(n_ranks) rounded to a multiple
            of n_ranks (equal rows/band), n_lon = 2*n_lat (square-cell
            production aspect; cells/rank carries integer-rounding
            drift — the row records the ACTUAL cells/rank).
    """
    if mode == "strong":
        return resolution, 2 * resolution
    if weak_style == "band":
        return resolution * n_ranks, 2 * resolution
    if weak_style == "aspect":
        import math

        rows_per_rank = max(
            MIN_ROWS_PER_RANK,
            round(resolution * math.sqrt(n_ranks) / n_ranks),
        )
        n_lat = rows_per_rank * n_ranks
        return n_lat, 2 * n_lat
    raise ValueError(
        f"Unknown weak_style {weak_style!r}; expected one of "
        f"{WEAK_STYLE_CHOICES}."
    )


# ===========================================================================
# Model construction (rank-local from construction)
# ===========================================================================

def _perturb_state(state, grid):
    """Deterministic mask-aware eta/T perturbation of the global rest state.

    Applied on the GLOBAL state BEFORE the MPI backend is armed and
    before scattering, so every rank derives the identical global field
    and band slices stay globally consistent.
    """
    import jax.numpy as jnp

    mask = state.land_mask.data
    eta_pert = state.eta.data + ETA_PERTURBATION_M * mask * (
        jnp.sin(3.0 * grid.lon2d) * jnp.cos(2.0 * grid.lat2d)
    )
    T_pert = state.T.data + T_PERTURBATION_C * mask[..., jnp.newaxis] * (
        jnp.cos(grid.lon2d) * jnp.sin(grid.lat2d)
    )[..., jnp.newaxis]
    return state._replace(
        eta=state.eta.replace(data=eta_pert),
        T=state.T.replace(data=T_pert),
    )


def _ensure_precision(precision: str) -> None:
    """Apply the requested precision to BOTH layers: the jax x64 flag AND
    the legoESM precision POLICY.

    The x64 flag alone is NOT enough — the ocean state dtype comes from
    ``get_policy().storage`` (default fp32), so the previous
    flag-only version built f32 states under ``--precision float64``:
    every gate labeled f64 actually gated f32 data (caught by the tripole
    lane, whose salt drift sat at the f32 floor while the fp64-policy
    parity is machine-epsilon; the 2026-06-29 full-suite gotcha)."""
    import jax

    from legoesm.core.precision import PrecisionPolicy, set_policy

    if precision == "float64":
        jax.config.update("jax_enable_x64", True)
        set_policy(PrecisionPolicy.fp64())
    else:
        set_policy(PrecisionPolicy.fp32())


def _build_global_problem(
    n_lat: int, n_lon: int, nlev: int, baro_solver: str,
    *, force_pcg: bool = False, pcg_variant: str = "standard",
    preconditioner: str = "jacobi", fixed_iters: int = 60,
    cheby_degree: int = 4,
    land_mask: str = "none", bathymetry_file: str = "",
    tripole: bool = False,
):
    """Global grid + z-coordinate + config + perturbed global IC.

    Built with the LOCAL halo backend (pole pads / face-mask
    construction must see global arrays) and shared between
    :func:`build_case` and the ``--parity-gate`` serial reference —
    both constructions are deterministic, so every rank (and the
    pre-/post-arming calls) derive the IDENTICAL global state.

    ``force_pcg``: solver-matched parity (parity bisect job 8459362) —
    the np>=2 ``implicit_cn`` path always dispatches to the fixed-M
    PCG, so the SERIAL parity reference must run the same PCG (stock
    jax.scipy CG differs at the solver-residual level by construction:
    eta ~1e-8 abs, advecting into T/S over the gate window).
    """
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    if tripole:
        from legoesm.grids.tripole import create_synthetic_tripole

        grid = create_synthetic_tripole(n_lat=n_lat, n_lon=n_lon)
    else:
        grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev)
    import os as _os
    config = LatLonCGridOceanConfig.from_flat(
        barotropic_solver=baro_solver,
        barotropic_implicit_force_pcg=force_pcg,
        barotropic_implicit_pcg_variant=pcg_variant,
        barotropic_implicit_preconditioner=preconditioner,
        barotropic_implicit_pcg_fixed_iters=fixed_iters,
        # SOTA-local split-explicit barotropic lever (explicit_substep only):
        # LEGOESM_BARO_LOCAL_CLAMP=1 -> per-substep eta-floor clamp local,
        # global redistribute once/step (cuts ~3*n_substeps subcycle allreduces
        # to 3). Measures the multi-node strong-scaling gain vs the legacy
        # per-substep-redistribute path.
        # Wide-halo split-explicit barotropic lever (explicit_substep only):
        # LEGOESM_BARO_WIDE_HALO=1 -> ONE fused wide lat-halo exchange per
        # chunk of substeps instead of ~4 halo pads per substep (the >=16-rank
        # latency lever; A/B against the same case with the env unset).
        # LEGOESM_BARO_WIDE_HALO_CHUNK caps substeps/exchange (0 = auto).
        # The wide path REQUIRES the local-clamp scheme (config-validated),
        # so the wide env implies LEGOESM_BARO_LOCAL_CLAMP — pin the local
        # clamp in the A/B BASELINE too for a controlled comparison.
        barotropic_local_subcycle_clamp=(
            _os.environ.get("LEGOESM_BARO_LOCAL_CLAMP", "0") == "1"
            or _os.environ.get("LEGOESM_BARO_WIDE_HALO", "0") == "1"),
        barotropic_wide_halo=(
            _os.environ.get("LEGOESM_BARO_WIDE_HALO", "0") == "1"),
        barotropic_wide_halo_chunk=int(
            _os.environ.get("LEGOESM_BARO_WIDE_HALO_CHUNK", "0")),
    )
    # chebyshev degree: the config has no degree field (the factory reads
    # getattr(config, "barotropic_chebyshev_degree", 4)); the bench uses the
    # default degree-4 path (no shared-config change). cheby_degree kept in
    # the signature for callers that set the attr explicitly.
    _ = cheby_degree
    mask_override = None
    if land_mask == "etopo":
        # Realistic continents from the shipped ETOPO file: the wet-balance
        # A/B needs a REAL land distribution (the default rest-state mask is
        # polar caps only, whose row split is already near-balanced).  Flat
        # bottom is kept on purpose — wet_band_boundaries keys off the MASK,
        # so bathymetric depth would only confound the row-vs-wet timing.
        from legoesm.ocean.bathymetry import (
            BathymetryConfig,
            load_bathymetry_latlon_cgrid,
        )
        _bcfg = BathymetryConfig(source="file", path=bathymetry_file)
        _, mask_override = load_bathymetry_latlon_cgrid(grid, _bcfg)
    elif land_mask != "none":
        raise SystemExit(
            f"--land-mask must be 'none' or 'etopo', got {land_mask!r}")
    state_global = rest_state_latlon_cgrid_ocean(
        grid, z_coord, land_mask_override=mask_override)
    state_global = _perturb_state(state_global, grid)
    return grid, z_coord, config, state_global


def build_case(
    *,
    n_lat: int,
    n_lon: int,
    nlev: int,
    n_ranks: int,
    baro_solver: str,
    precision: str,
    pcg_variant: str = "standard",
    preconditioner: str = "jacobi",
    fixed_iters: int = 60,
    force_pcg: bool = False,
    wet_balance: bool = False,
    land_mask: str = "none",
    bathymetry_file: str = "",
    tripole: bool = False,
):
    """Build (model, state, total_cells, layout, partition_metrics).

    Multi-rank: the model is built on the BAND geometry (never on the
    global grid — see module docstring) and the state is the scattered
    band state; ``layout`` is the armed ``LatLonBandLayout``.  Single
    rank: identical to the serial GPU-bench build, ``layout`` is None.
    ``partition_metrics`` (multi-rank only, else None) records the wet-cell
    load balance of the ACTUAL band split — for the row-balanced baseline
    too, so a wet-vs-row A/B is comparable from the JSON records alone.
    """
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    _ensure_precision(precision)

    # (1) Global grid + IC with the LOCAL halo backend.
    # ``force_pcg`` forces the fixed-M PCG even at a SINGLE rank so a 1-vs-N
    # strong-scaling ratio compares the SAME barotropic solver both sides
    # (else np=1 stock jax.scipy CG vs np>=2 fixed-M PCG is apples-to-oranges
    # — codex route-A review Q5).  At n_ranks>1 the is_distributed dispatch
    # already uses the PCG, so this only changes the np=1 baseline.
    grid, z_coord, config, state_global = _build_global_problem(
        n_lat, n_lon, nlev, baro_solver, pcg_variant=pcg_variant,
        preconditioner=preconditioner, fixed_iters=fixed_iters,
        force_pcg=force_pcg,
        land_mask=land_mask, bathymetry_file=bathymetry_file,
        tripole=tripole,
    )

    total_cells = n_lat * n_lon * nlev

    layout = None
    part_metrics = None
    if n_ranks > 1:
        # (2) Arm the band layout + MPI halo backend.
        from legoesm.parallel.distributed import initialize_distributed_latlon
        from legoesm.parallel.latlon_mpi import (
            scatter_state_latlon_cgrid_ocean,
            slice_cgrid_geometry_to_band,
            slice_zcoord_to_band,
        )

        band_boundaries = None
        if wet_balance:
            # Wet-cell-aware bands: boundaries equalize OCEAN cells per band
            # instead of rows, so land-heavy bands stop idling at every
            # collective. Deterministic host computation from the GLOBAL mask
            # (available on every rank before the layout is armed).
            import numpy as np

            from legoesm.parallel.latlon_mpi import wet_band_boundaries
            wet_rows = np.asarray(state_global.land_mask.data).sum(axis=1)
            band_boundaries = wet_band_boundaries(
                wet_rows, n_ranks, min_rows=MIN_ROWS_PER_RANK)

        layout = initialize_distributed_latlon(
            global_n_lat=n_lat, global_n_lon=n_lon,
            band_boundaries=band_boundaries,
            # Fold-aware layout: the north rank applies the ORCA fold at
            # its north boundary; interior cuts exchange fold-aware.
            fold=getattr(grid, "fold", None) if tripole else None,
        )
        # Partition-quality record for the ACTUAL split (wet OR the
        # row-balanced default): wet cells + rows per band, and the
        # wet-imbalance ratio max/mean — the number the wet-balance A/B is
        # about.  Computed host-side from the global mask on every rank
        # (deterministic), recorded on rank 0's JSON row.
        import numpy as np

        _bounds = (band_boundaries if band_boundaries is not None
                   else _even_boundaries(n_lat, n_ranks))
        _mask_np = np.asarray(state_global.land_mask.data)
        _wet_per_band = np.array([
            float(_mask_np[_bounds[r]:_bounds[r + 1]].sum())
            for r in range(n_ranks)
        ])
        _rows = np.diff(_bounds)
        part_metrics = {
            "decomposition_boundaries": [int(b) for b in _bounds],
            "rows_per_rank_min": int(_rows.min()),
            "rows_per_rank_max": int(_rows.max()),
            "wet_cells_per_rank_min": int(_wet_per_band.min()),
            "wet_cells_per_rank_max": int(_wet_per_band.max()),
            "wet_imbalance_max_over_mean": float(
                _wet_per_band.max() / max(_wet_per_band.mean(), 1.0)),
            "wet_fraction_global": float(_mask_np.mean()),
            "wet_balanced": bool(wet_balance),
        }
        if layout.rank == 0:
            print(f"[bands] {'wet' if wet_balance else 'row'}-balanced "
                  f"boundaries={part_metrics['decomposition_boundaries']} "
                  f"rows/band={_rows.tolist()} wet-cells/band="
                  f"{[int(x) for x in _wet_per_band]} "
                  f"imbalance={part_metrics['wet_imbalance_max_over_mean']:.3f}",
                  flush=True)
        # (3) Band geometry + band vertical coordinate (z* carries no
        # per-cell arrays -> slice_zcoord_to_band is a pass-through, but
        # keeps this build correct if partial cells are enabled later).
        band_geom = slice_cgrid_geometry_to_band(ensure_geometry(grid), layout)
        z_band = slice_zcoord_to_band(z_coord, layout)
        # (4) Rank-local model ON the band geometry.
        model = LatLonCGridOceanModel(band_geom, z_band, config)
        # (5) Band state.
        state = scatter_state_latlon_cgrid_ocean(state_global, layout)
    else:
        model = LatLonCGridOceanModel(grid, z_coord, config)
        state = state_global

    return model, state, total_cells, layout, part_metrics


def _even_boundaries(n_lat: int, n_ranks: int) -> tuple[int, ...]:
    """The default even row split's boundaries (first ``n_lat % n_ranks``
    bands one row taller) — mirrors ``make_latlon_band_layout``'s split so
    the row-balanced baseline's partition metrics describe the REAL bands."""
    import numpy as np

    base, rem = divmod(n_lat, n_ranks)
    sizes = [base + 1 if r < rem else base for r in range(n_ranks)]
    return tuple(int(x) for x in np.concatenate([[0], np.cumsum(sizes)]))


# ===========================================================================
# Parity gate (--parity-gate; smoke sizes only)
# ===========================================================================

def compute_serial_reference(
    *,
    n_lat: int,
    n_lon: int,
    nlev: int,
    baro_solver: str,
    precision: str,
    dt: float,
    n_steps: int,
    n_ranks: int = 1,
    land_mask: str = "none",
    bathymetry_file: str = "",
    tripole: bool = False,
):
    """Serial reference fields after ``n_steps`` — every rank, BEFORE
    the MPI halo backend is armed.

    A rank-0-only serial reference computed AFTER arming deadlocks (the
    serial step would trace band sendrecv/allreduce collectives that no
    other rank matches — the pattern that bit P1); computing it locally
    on every rank costs one redundant global smoke run and zero
    communication.  Returns ``{field: np.ndarray}`` for PARITY_FIELDS.
    """
    import jax
    import numpy as np

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    _ensure_precision(precision)
    # Solver-matched reference (parity bisect 8459362): under MPI the
    # implicit_cn barotropic ALWAYS runs the fixed-M PCG, so the serial
    # reference must too — stock CG vs PCG differ at solver-residual
    # level by construction, which the bit-exact gate would (and did)
    # flag as a spurious MISMATCH.  Safe pre-arming: the PCG's global
    # dots reduce locally when not multi-process.
    grid, z_coord, config, state = _build_global_problem(
        n_lat, n_lon, nlev, baro_solver,
        # Only the MULTI-rank case dispatches to the PCG; an np=1 parity
        # run is stock-CG vs stock-CG and must stay solver-matched too.
        force_pcg=(baro_solver == "implicit_cn" and n_ranks > 1),
        land_mask=land_mask, bathymetry_file=bathymetry_file,
        tripole=tripole,
    )
    model = LatLonCGridOceanModel(grid, z_coord, config)
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    return {
        name: np.asarray(getattr(state, name).data)
        for name in PARITY_FIELDS
    }


def run_parity_gate(state_final, layout, serial_ref, n_ranks: int,
                    precision: str, baro_solver: str = "explicit_substep"):
    """Gather the final band state, compare to the serial reference.

    Collective (every rank must call: per-field ``comm.gather`` inside
    ``gather_state_latlon_cgrid_ocean`` + the verdict bcast).  Returns
    ``(ok, lines)``; ``ok`` is rank-consistent, ``lines`` is the
    per-field report (rank 0 only, empty elsewhere).

    Tolerance lane: ``implicit_cn`` at np>=2 uses
    ``PARITY_TOLS_PCG_MPI`` (allreduce dot-order noise — see the
    constant's comment); everything else gates bit-exact-tight via
    ``PARITY_TOLS``.
    """
    import numpy as np

    if baro_solver == "implicit_cn" and n_ranks > 1:
        rtol, atol = PARITY_TOLS_PCG_MPI[precision]
    else:
        rtol, atol = PARITY_TOLS[precision]
    if n_ranks > 1:
        from legoesm.parallel.latlon_mpi import (
            gather_state_latlon_cgrid_ocean,
        )

        gathered = gather_state_latlon_cgrid_ocean(state_final, layout)
    else:
        gathered = state_final

    ok = True
    lines: list[str] = []
    if gathered is not None:  # rank 0 (or single rank)
        for name in PARITY_FIELDS:
            got = np.asarray(getattr(gathered, name).data)
            want = serial_ref[name]
            adiff = np.abs(got - want)
            max_abs = float(adiff.max())
            max_rel = float(
                (adiff / np.maximum(np.abs(want), 1e-30)).max()
            )
            field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
            ok = ok and field_ok
            lines.append(
                f"    {name:4s} max_abs={max_abs:.3e} "
                f"max_rel={max_rel:.3e} -> "
                f"{'OK' if field_ok else 'MISMATCH'}"
            )
    if n_ranks > 1:
        from mpi4py import MPI as _MPI

        ok = bool(_MPI.COMM_WORLD.bcast(ok, root=0))
    return ok, lines


# ===========================================================================
# Conservation invariants (cheap, optional; GATED at --cons-rtol)
# ===========================================================================

def ocean_invariants(model, state, n_ranks: int) -> tuple[float, float, float, float]:
    """Global (ocean_area, eta_integral, heat_integral, salt_integral).

    Mirrors ``_global_ocean_invariants`` in
    tests/ocean/distributed/test_ocean_mpi_conservation.py: cell-centred
    sums over the rank-owned band (bands partition the cell rows — no
    double counting) reduced with one ``allreduce(SUM)``.
    """
    import jax.numpy as jnp

    from legoesm.ocean.vertical import compute_layer_thickness

    mask = state.land_mask.data
    h_k = compute_layer_thickness(
        state.eta.data,
        state.H_bathy.data,
        model.z_coord,
        min_water_column_m=model.config.min_water_column_m,
    )
    weighted_area = mask * model.grid.area
    local_terms = jnp.stack(
        [
            jnp.sum(weighted_area),
            jnp.sum(state.eta.data * weighted_area),
            jnp.sum(jnp.sum(state.T.data * h_k, axis=-1) * weighted_area),
            jnp.sum(jnp.sum(state.S.data * h_k, axis=-1) * weighted_area),
        ],
    )
    if n_ranks > 1:
        from legoesm.parallel.reductions import global_sum_mpi

        local_terms = global_sum_mpi(local_terms)
    return tuple(float(x) for x in local_terms)


def conservation_metrics(before, after) -> dict[str, float]:
    """Drift metrics between two invariant tuples.

    ``area_rel``/``heat_rel``/``salt_rel`` are relative; ``mean_eta_drift_m``
    is the drift of the area-mean free surface in METRES (the initial
    mean eta is ~0, so a relative measure is undefined — the single
    ``--cons-rtol`` tolerance is applied to it as metres).
    """
    area0, eta0, heat0, salt0 = before
    area1, eta1, heat1, salt1 = after
    return {
        "area_rel": abs(area1 - area0) / max(abs(area0), 1.0),
        "mean_eta_drift_m": abs(eta1 - eta0) / max(abs(area0), 1.0),
        "heat_rel": abs(heat1 - heat0) / max(abs(heat0), 1.0),
        "salt_rel": abs(salt1 - salt0) / max(abs(salt0), 1.0),
    }


def print_conservation(label: str, before, after, tol: float) -> None:
    m = conservation_metrics(before, after)
    print(
        f"  [conservation {label}] area_rel={m['area_rel']:.3e} | "
        f"mean_eta_drift={m['mean_eta_drift_m']:.3e} m | "
        f"heat_rel={m['heat_rel']:.3e} | salt_rel={m['salt_rel']:.3e}  "
        f"(gated at cons_rtol={tol:.1e})",
        flush=True,
    )


def conservation_breaches(before, after, tol: float) -> list[str]:
    """Metrics exceeding ``tol`` (empty list == gate passes).

    The invariants are already globally reduced (``allreduce(SUM)``
    inside :func:`ocean_invariants`), so every rank holds identical
    values and computes the identical verdict — no extra collective.
    """
    return [
        f"{name}={value:.3e} > cons_rtol={tol:.1e}"
        for name, value in conservation_metrics(before, after).items()
        if value > tol
    ]


# ===========================================================================
# Timing (run_cpu_mpi_scaling discipline)
# ===========================================================================

def _mpi_barrier_after_block(state) -> None:
    """block_until_ready on every leaf, then COMM_WORLD barrier (if MPI)."""
    import jax

    jax.block_until_ready(jax.tree.leaves(state))
    try:
        from mpi4py import MPI as _MPI

        if _MPI.COMM_WORLD.Get_size() > 1:
            _MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass


def _time_fused_fn(advance_fn, seed_state, n_warmup: int, n_timing: int):
    """Compile + warmup + fused-scan steady timing for ONE advance fn.

    ``advance_fn(carry_state) -> new_state`` is any pure pytree->pytree
    map that takes one full ocean state and returns a same-structure
    state (the full ``model.step`` for the headline case; an
    isolated-phase closure for ``--profile-phases``).  The discipline —
    JIT-compile-the-first-call timing, remaining-warmup,
    cloned-leaf scan pre-compile (so the timed seed is untouched),
    matched MPI barriers around a single fused ``lax.scan`` of length
    ``n_timing``, ``block_until_ready`` on every leaf — is IDENTICAL for
    the full step and every phase, so the per-phase ms/step is directly
    comparable to the full-step ms/step (same warmup/fusion/barrier
    treatment; the ONLY difference is the body).

    Returns (compile_s, warmup_s, timing_s, final_state).
    """
    import jax

    # --- JIT compile (first call) ---
    t0 = time.perf_counter()
    s = advance_fn(seed_state)
    jax.block_until_ready(jax.tree.leaves(s))
    compile_s = time.perf_counter() - t0

    # --- Remaining warmup ---
    t0 = time.perf_counter()
    for _ in range(max(0, n_warmup - 1)):
        s = advance_fn(s)
    jax.block_until_ready(jax.tree.leaves(s))
    warmup_s = time.perf_counter() - t0

    # --- Fused scan over n_timing steps (dtype-stable carry) ---
    input_dtypes = jax.tree.map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, s,
    )

    @jax.jit
    def _scan_run(st):
        def _body(carry, _):
            new = advance_fn(carry)
            new = jax.tree.map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes,
            )
            return new, None
        return jax.lax.scan(_body, st, None, length=n_timing)[0]

    # Pre-compile against cloned leaves so the timed seed state is
    # untouched (run_cpu_mpi_scaling bias fix).  Under MPI this executes
    # the same collective schedule on every rank — counts stay matched.
    _pre = jax.tree.map(lambda x: x, s)
    _pre_out = _scan_run(_pre)
    jax.block_until_ready(jax.tree.leaves(_pre_out))

    # MPI barrier before timing
    _mpi_barrier_after_block(s)

    t0 = time.perf_counter()
    s = _scan_run(s)
    jax.block_until_ready(jax.tree.leaves(s))
    # MPI barrier after timing (t1 includes the barrier wait — the
    # slowest rank defines the step time, same as run_cpu_mpi_scaling).
    try:
        from mpi4py import MPI as _MPI

        if _MPI.COMM_WORLD.Get_size() > 1:
            _MPI.COMM_WORLD.Barrier()
    except ImportError:
        pass
    timing_s = time.perf_counter() - t0

    return compile_s, warmup_s, timing_s, s


def time_case(model, state, dt: float, n_warmup: int, n_timing: int):
    """Compile + warmup + fused-scan steady timing of the FULL step.

    Returns (compile_s, warmup_s, timing_s, final_state).
    """
    dt_static = float(dt)
    return _time_fused_fn(
        lambda st: model.step(st, dt_static), state, n_warmup, n_timing,
    )


def check_finite(state, n_ranks: int) -> bool:
    """True when every floating leaf is finite on EVERY rank."""
    import jax
    import jax.numpy as jnp

    ok = True
    for leaf in jax.tree.leaves(state):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            ok = ok and bool(jnp.all(jnp.isfinite(leaf)))
    if n_ranks > 1:
        try:
            from mpi4py import MPI as _MPI

            ok = bool(_MPI.COMM_WORLD.allreduce(ok, op=_MPI.LAND))
        except ImportError:
            pass
    return ok


# ===========================================================================
# Phase split (--profile-phases): per-phase ISOLATION estimate
# ===========================================================================
#
# GATE measurement.  Two questions:
#   (1) [original] is the BAROTROPIC solve, or the baroclinic-tendency +
#       tracer chain, the dominant per-step cost of the lat-lon C-grid step?
#   (2) [the 74% hunt] the original 3-phase split (baroclinic + barotropic
#       + tracer_advection) summed to only ~26% of the full ``model.step``
#       (job 8458934), leaving ~74% UNACCOUNTED.  Where is it?  This mode
#       now times EVERY major sub-phase ``model.step`` executes — including
#       the ones the original split excluded — so the phases account for
#       ~100% (modulo the documented isolation/fusion residual), and tags
#       each phase COMPUTE-bound vs LATENCY-bound for the scaling-lever
#       decision.
#
# ``time_case`` times only the full ``model.step``; this mode times the
# step's sub-phases in ISOLATION (each with the identical ``_time_fused_fn``
# discipline) and reports ms/step + % of the full-step time + the phase's
# HALO-exchange count and GLOBAL-REDUCTION count (the rank-growing latency
# terms) + a COMPUTE/LATENCY tag.
#
# WHICH PHASES (config-gated — a phase is only timed when its config gate is
# on, so the split matches the config actually being benchmarked; see
# ``ocean_model_latlon_cgrid.LatLonCGridOceanModel._step_impl``):
#
#   baroclinic           always   — ONE ``model.tendencies`` eval (Stages 1-7
#                                    of the PE: EOS+pressure ITERATION,
#                                    KE/pressure gradients, momentum advection,
#                                    lateral viscosity, w-diagnosis).
#   coriolis_momint      always   — the momentum integrator (forward-Euler or
#                                    SSP-RK3) + the forward-backward Matsuno
#                                    Coriolis (``_forward_backward_coriolis_3d``).
#                                    For RK3 this re-evaluates ``tendencies`` 2x
#                                    (already attributed via baroclinic_insitu),
#                                    so this phase isolates the Euler/Coriolis
#                                    folding ONLY (RK3 stage re-evals counted in
#                                    baroclinic_insitu, not double-counted here).
#   barotropic           implicit_cn — the REAL ``barotropic_implicit_latlon_cgrid``
#                                    (predictor halos + fixed-M distributed PCG).
#   eta_drift_projection fix_eta_drift — the global mean-eta projection +
#                                    mass-conserving eta-floor redistribution
#                                    (``_step_impl`` step 6b; ``ocean_global_sum``
#                                    + ``eta_floor.clamp_and_redistribute``).
#   tracer_advection     always   — advection flux-divergence for T,S ONLY.
#   tracer_tail          always   — the rest of the tracer transport setup the
#                                    advection phase excludes: per-layer
#                                    mass-flux correction to match ``Hu_avg``
#                                    (Hallberg-Adcroft) + ``w_baro`` diagnosis
#                                    via ``divergence_cgrid`` + the vertical
#                                    tracer flux div.
#   gm_redi              gm_redi!=None — the GM/Redi isoneutral (neutral) tracer
#                                    tendency (``gm_redi_tracer_tendency_latlon``;
#                                    the isoneutral tensor + its many face/triad
#                                    halos).  OFF in the bench-default config;
#                                    ON in the OMIP-production config.
#   implicit_vmix        implicit_vertical_mixing — the backward-Euler vertical
#                                    mixing (``_apply_implicit_vertical_mixing``:
#                                    K-profile build + per-column tridiagonal
#                                    Thomas solve for T,S,u,v).  ON by default.
#   conservation_fixer   use_conservation_fixer — ``ocean_conservation_fixer``
#                                    (area/heat/salt global reductions + rescale).
#
# APPROXIMATION (read before trusting the split — this is NOT an exact
# decomposition):
#
#   * Phases are timed ALONE, so XLA cross-phase fusion is lost and —
#     critically under MPI — communication/computation OVERLAP is lost:
#     a phase's collectives (the barotropic PCG's 2*M allreduces; the
#     ~27 baroclinic halo sendrecvs) run with their latency FULLY EXPOSED
#     here, whereas in the fused full step XLA may overlap a phase's
#     comm with the next phase's compute.  => isolation OVER-counts a
#     comm-bound phase's standalone share and the phase times need NOT sum
#     EXACTLY to the full-step time.  The sum-check below reports the
#     residual (full - Σphases) explicitly, labelled
#     "fusion/overlap/uninstrumented".  Treat each % as "this phase's cost
#     when run alone, as a fraction of the full step", an UPPER-ish bound
#     on a comm-bound phase, NOT a strict additive budget.
#   * baroclinic = ONE ``model.tendencies`` evaluation.  The in-situ step
#     evaluates the tendency function ``baroclinic_insitu_evals`` times
#     (1 for forward-Euler momentum; 3 for SSP-RK3 = the initial call,
#     shared with the tracer update, + 2 ``_mom_pert`` re-evaluations for
#     stages 2 and 3 — stage 1 REUSES the initial call's tendency), so
#     the in-situ baroclinic cost is ~``insitu_evals * per_eval`` —
#     reported as ``baroclinic_insitu`` (and FOLDED INTO the sum-check, so
#     the residual is computed against the in-situ baroclinic cost, not the
#     single-eval cost, to avoid a spurious +2*per_eval residual on RK3).
#     Timing ONE eval is faithful to a single tendency call (the ~27 halo
#     exchanges it does in-situ ARE done here — ``model.tendencies`` is the
#     exact same function the step calls, no halo stripped).
#   * barotropic = the REAL ``barotropic_implicit_latlon_cgrid`` on the
#     input state with ``F_slow_*=None``.  Faithful for COST: the PCG
#     iteration count M (and thus the 2*M-allreduce schedule) and every
#     predictor/divergence halo are fixed by config + shape, INDEPENDENT
#     of the array VALUES, so zero-filled F_slow changes neither op count
#     nor halo bytes nor allreduce count vs the in-situ call on
#     ``state_mid``.  The 2*M allreduces ARE issued here.
#   * tracer_advection = the advection flux-divergence for T and S
#     (``_compute_advection_flux_div`` x2, or the RK3 tracer step),
#     INCLUDING the advection-scheme halos.  ADVECTION ONLY; the rest of
#     the tracer tail is the SEPARATE ``tracer_tail`` phase (no longer
#     un-instrumented).
#   * eta_drift_projection / implicit_vmix / gm_redi / conservation_fixer =
#     the REAL ``_step_impl`` blocks / model methods, run on the seed state.
#     For eta_drift / conservation_fixer the cost is reduction-latency-
#     dominated and value-independent (the allreduce schedule is fixed by
#     shape + config), so timing them on the seed state is faithful for
#     cost.  For implicit_vmix the seed-state path is the SAME fallback
#     ``compute_vertical_K_profiles`` + Thomas-solve path the bench step
#     takes (``physics=None`` ⇒ ``tend.K_v/A_v=None`` ⇒ fallback), so the
#     op count is faithful; for an OMIP-production config that passes
#     precomputed ``tend.K_v/A_v`` the isolated phase recomputes K (an
#     OVER-count of that phase's compute — flagged in the printout).
#
# COMPUTE-bound vs LATENCY-bound (the lever decision):
#   The tag is a BINARY classification from the phase's halo+reduction
#   counts: HALOS>0 or REDUCTIONS>0 ⇒ the phase has a term whose latency
#   GROWS with the rank count ⇒ tagged ``latency`` (rank-growing);
#   HALOS==0 and REDUCTIONS==0 ⇒ pure per-column/per-cell local work whose
#   per-rank cost stays CONSTANT under weak scaling ⇒ tagged ``compute``
#   (sets the base floor, does not grow).  Rank-growing phases:
#   barotropic (2*M allreduces), eta_drift / conservation_fixer (global
#   reductions), and the halo-heavy horizontal operators (baroclinic,
#   gm_redi, tracer_advection, tracer_tail, coriolis_momint).
#   NUANCE (codex finding 1): the ``latency`` tag means "has at least one
#   rank-growing term", NOT "the cost is dominated by comm".  implicit_vmix
#   is MOSTLY rank-constant compute (the columnwise tridiagonal Thomas
#   solve) but interpolates A_v/dz to the v-faces with 2 ``pad_ns_zero``
#   halos, so it is tagged ``latency`` with only a small rank-growing piece
#   on top of a large compute floor — the printout flags this.  So a large
#   ``latency``-tagged phase is not automatically a parallelism win: read
#   its halo/reduction counts (a phase with 2 halos and a big compute body
#   wants per-device fusion for the body + a cheap comm fix; a phase with
#   2*M reductions and little compute wants a comm/algorithm fix).  The
#   lever must target a phase that is BOTH large AND has a SUBSTANTIAL
#   rank-growing term; if the biggest phase is ``compute`` (no halo/no
#   reduction) the lever is per-device fusion, not parallelism.
#
# HALO / REDUCTION COUNTS (per phase, STATIC from the code structure — the
# same way the barotropic 2*M-allreduce floor is config-derived):
#   * "halos" = number of N-S (meridional) C-grid halo exchanges the phase
#     issues per call.  Under the latitude-band decomposition each
#     ``pad_ns_*`` / ``pad_with_pole_bc_lat`` is ONE sendrecv pair per
#     interior partition cut (longitude is periodic via ``jnp.roll`` — NO
#     MPI in the zonal direction).  Counted by the meridional operators the
#     phase invokes; cross-checked against the operator file:line in the
#     ``PhaseSpec.note``.  These are the terms whose latency grows as the
#     band gets thinner (more ranks).
#   * "reductions" = number of global ``allreduce(SUM)`` the phase issues
#     (``ocean_global_sum`` / ``global_sum_mpi`` / the PCG's 2*M dots / the
#     eta-floor's ``n_iter`` redistribute allreduces).  Latency grows
#     ~log(ranks).
#
# ROBUSTNESS CROSS-CHECK (independent of the timing approximation): the
# barotropic reduction-latency FLOOR is computed directly from the
# config — ``2*M`` allreduces/step * the measured per-allreduce latency
# (``--allreduce-latency-us``, default 111 us, the Ginsburg np<=4 CPU-MPI
# roofline floor) — and reported as a fraction of the full step.  If the
# barotropic ISOLATION share AND this latency floor BOTH say barotropic
# is a minority of the step, the "barotropic dominant? NO" verdict is
# robust to the isolation approximation (the floor is a hard lower bound
# the real solve cannot beat; the isolation time is an upper-ish bound).


class PhaseSpec(NamedTuple):
    """One isolated step-phase: its advance closure + static comm metadata.

    ``advance_fn`` is the ``state -> state`` isolation closure (timed by
    ``_time_fused_fn`` exactly like the full step).  ``halos`` /
    ``reductions`` are the STATIC per-call counts of N-S halo exchanges
    (``pad_ns_*`` / ``pad_with_pole_bc_lat`` -> 1 sendrecv pair per
    partition cut) and global ``allreduce(SUM)`` the phase issues — the
    rank-growing latency terms.  ``bound`` is ``"latency"`` when the phase
    has any halo or reduction (a term whose cost grows with ranks) else
    ``"compute"`` (per-column/per-cell local work, constant per rank under
    weak scaling).  ``note`` cites the operator file:line backing the
    counts.  ``in_sum`` is False for a phase that is a re-attribution of
    another phase's already-summed cost (so the sum-check does not
    double-count it); currently always True (baroclinic_insitu is folded
    into the sum-check separately, not as a PhaseSpec).
    """

    advance_fn: Callable
    halos: int
    reductions: int
    bound: str
    note: str
    in_sum: bool = True


def _baroclinic_insitu_evals(config) -> int:
    """Number of ``tendencies`` evaluations the in-situ step makes.

    SSP-RK3 momentum integrator: the initial ``self.tendencies`` call
    (which ALSO drives the tracer update — stage-1 momentum REUSES its
    ``du_dt_pert``/``dv_dt_pert``, no re-call) + exactly 2 ``_mom_pert``
    re-evaluations (stages 2 and 3) = 3 (matches the "3x the tendency
    cost" note on ``momentum_time_integrator`` in ``ocean/state.py``).
    Forward-Euler: 1.  This is the multiplier from a single-eval
    baroclinic timing to the in-situ baroclinic cost (the dominant
    ~27-halo cost is per-eval).  See the RK3 block in
    ``ocean_model_latlon_cgrid._step_impl`` (the initial ``tendencies``
    at ~:935; ``_mom_pert`` called at ~:1067 and ~:1070).
    """
    return 3 if getattr(config, "momentum_time_integrator", "euler") == "rk3" else 1


def _make_phase_advancers(model, dt: float):
    """Build the per-phase isolation specs (``{name: PhaseSpec}``).

    Each ``PhaseSpec.advance_fn`` runs ONLY its phase's real model code on
    the FULL state and returns a same-structure state (so
    ``_time_fused_fn``'s dtype-stable scan carry + cloned-leaf pre-compile
    work unchanged).  Phases are CONFIG-GATED — a phase is only included
    when its ``_step_impl`` block is actually executed by the config being
    benchmarked — so the split adds up against the same step the headline
    timing measured.  See the module note for the full phase list,
    isolation-fidelity caveats, and the static halo/reduction counts.
    """
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.barotropic_implicit_latlon_cgrid import (
        barotropic_implicit_latlon_cgrid,
    )

    dt_static = float(dt)
    dt_mom = dt_static / model.config.dt_mom_ratio
    grid = model.grid
    z_coord = model.z_coord
    config = model.config

    def _baroclinic(state):
        # ONE evaluation of the exact PE tendency function the step calls
        # (the ~27 halo exchanges happen here).  Fold the tendencies back
        # onto the state via a forward-Euler T/S update so the closure is
        # a state->state map with a stable pytree (values are irrelevant —
        # the op/halo schedule, not the numerics, is what we time).
        tend = model.tendencies(state, surface_forcing=None, dt=dt_static)
        mask_3d = state.land_mask.data[..., jnp.newaxis]
        T_new = (state.T.data + dt_static * tend.dT_dt.data) * mask_3d
        S_new = (state.S.data + dt_static * tend.dS_dt.data) * mask_3d
        u_new = state.u.data + dt_static * tend.du_dt.data
        v_new = state.v.data + dt_static * tend.dv_dt.data
        return state._replace(
            T=state.T.replace(data=T_new), S=state.S.replace(data=S_new),
            u=state.u.replace(data=u_new), v=state.v.replace(data=v_new),
        )

    def _barotropic(state):
        # The REAL implicit-CN free-surface solve (predictor halos +
        # fixed-M distributed PCG => 2*M allreduces).  F_slow_*=None
        # zero-fills — the cost (op count, halo bytes, allreduce count) is
        # value-independent, so this is the in-situ barotropic cost.
        state_new, _ = barotropic_implicit_latlon_cgrid(
            state, dt_mom, grid, z_coord, config,
            F_slow_eta=None, F_slow_u=None, F_slow_v=None,
        )
        return state_new

    def _barotropic_substep(state):
        # The REAL split-explicit forward-backward substepping path (codex
        # finding 3: do not leave it in the residual when --baro-solver
        # explicit_substep).  dt_s = dt_mom / n_barotropic_substeps;
        # F_slow_*=None — the substep count (and thus per-substep eta-floor
        # reductions + predictor halos) is config-fixed, value-independent.
        from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
            barotropic_substeps_latlon_cgrid,
        )
        n_sub = config.barotropic.n_barotropic_substeps
        dt_s = dt_mom / n_sub
        state_new, _ = barotropic_substeps_latlon_cgrid(
            state, dt_s, n_sub, grid, z_coord, config,
            F_slow_eta=None, F_slow_u=None, F_slow_v=None,
        )
        return state_new

    # baroclinic halo count: ~27 N-S sendrecvs per tendency eval, the
    # documented per-eval cost (every meridional C-grid operator in
    # ocean_pe_latlon_cgrid does one pad_ns_* / pad_with_pole_bc_lat; the
    # zonal direction is jnp.roll-periodic, no MPI).  Reductions: 0 (the
    # depth-mean is a local sum over the level axis, not a global MPI one).
    specs: dict[str, PhaseSpec] = {
        "baroclinic": PhaseSpec(
            advance_fn=_baroclinic, halos=27, reductions=0, bound="latency",
            note=("ocean_pe_latlon_cgrid.latlon_cgrid_ocean_baroclinic_"
                  "tendencies: ~27 pad_ns_*/pad_with_pole_bc_lat sendrecvs "
                  "(EOS+PGF+momadv+w); no global reduction"),
        ),
    }
    if config.barotropic.barotropic_solver == "implicit_cn":
        M = int(config.barotropic.barotropic_implicit_pcg_fixed_iters)
        # Predictor + RHS-divergence halos (config/shape-fixed, value-
        # independent); 2*M in-loop allreduces from the distributed PCG.
        specs["barotropic"] = PhaseSpec(
            advance_fn=_barotropic, halos=4, reductions=2 * M,
            bound="latency",
            note=(f"barotropic_implicit_latlon_cgrid: fixed-M={M} distributed "
                  f"PCG => 2*M={2 * M} IN-LOOP PCG-dot allreduce(SUM) "
                  f"(reductions= the in-loop dots ONLY; the solve ALSO issues "
                  f"a fixed handful of mass-projection / eta-floor allreduces "
                  f"not in this count); predictor + divergence pad_ns_* "
                  f"halos; allreduce count value-independent + FIXED per step "
                  f"regardless of resolution — the weak-scaling lever vs "
                  f"explicit_substep)"),
        )
    elif config.barotropic.barotropic_solver == "explicit_substep":
        # Codex finding 3 (+ re-review): do not hide the explicit-substep
        # barotropic block in the residual, and count its reductions
        # CORRECTLY.  Each of the n_sub substeps runs the eta-floor
        # clamp_and_redistribute (n_iter allreduces, barotropic_latlon_cgrid
        # :338) UNCONDITIONALLY, plus a SECOND clamp + a Laplacian-eta
        # divergence/gradient halo set when barotropic_diffusion_alpha > 0
        # (:400-408; the LatLonCGridOceanConfig default is 0.01 > 0, so the
        # default path pays BOTH clamps).  So per substep:
        #   reductions = floor_iters * (2 if alpha>0 else 1)
        #   halos      = ~4 predictor pad_ns_* + (2 diffusion divergence/
        #                gradient halos if alpha>0)
        # => both GROW with n_barotropic_substeps — the documented
        # weak-scaling contrast vs implicit_cn's fixed schedule (substep
        # count, and thus reduction count, grows with resolution).  Counts
        # are ESTIMATES (per-substep op count is shape-fixed, value-
        # independent).  NB this is NOT apples-to-apples with the implicit_cn
        # 2*M (which is the PCG IN-LOOP dots ONLY; implicit_cn ALSO issues a
        # fixed handful of mass-projection / eta-floor reductions — the
        # contrast is "GROWS with n_sub/resolution" vs "FIXED per step", not
        # the raw integers).
        from legoesm.ocean.dynamics.eta_floor import (
            clamp_and_redistribute as _cr,
        )
        import inspect as _inspect
        _fi = int(
            _inspect.signature(_cr).parameters["n_iter"].default
        )
        n_sub = int(config.barotropic.n_barotropic_substeps)
        _alpha_on = float(getattr(config.barotropic, "barotropic_diffusion_alpha", 0.0)) > 0.0
        _red_per_sub = _fi * (2 if _alpha_on else 1)
        _halo_per_sub = 4 + (2 if _alpha_on else 0)
        specs["barotropic"] = PhaseSpec(
            advance_fn=_barotropic_substep, halos=_halo_per_sub * n_sub,
            reductions=_red_per_sub * n_sub, bound="latency",
            note=(f"barotropic_substeps_latlon_cgrid: n_sub={n_sub} x "
                  f"({_halo_per_sub} pad_ns_* halos + {_red_per_sub} eta-floor "
                  f"allreduces [{_fi}/clamp, "
                  f"{'2 clamps: base + diffusion-alpha' if _alpha_on else '1 clamp'}"
                  f"]) => ~{_halo_per_sub * n_sub} halos / ~{_red_per_sub * n_sub} "
                  f"allreduce(SUM) (ESTIMATE; GROWS with n_barotropic_substeps "
                  f"— the weak-scaling cost implicit_cn's FIXED schedule "
                  f"avoids; NOT directly comparable to the 2*M PCG-dots "
                  f"integer)"),
        )
    elif config.barotropic.barotropic_solver == "rigid_lid":
        # rigid_lid needs the host-side island flood-fill (rl_data built
        # from the concrete state via _ensure_rigid_lid_data) + a CG solve
        # whose iteration count is residual-dependent — not cleanly
        # isolation-timeable on a stand-in state without replicating the
        # build.  Skip (the rigid-lid barotropic cost falls into the
        # residual, noted in the residual label); the bench/sbatch use
        # implicit_cn so this branch is not exercised there.
        pass

    # coriolis_momint: the momentum integrator + forward-backward Matsuno
    # Coriolis.  See implementation note on the closure.
    specs["coriolis_momint"] = _make_coriolis_momint_spec(model, dt_static)

    # eta_drift_projection: only when fix_eta_drift (default True).  Global
    # mean-eta projection (1 allreduce) + mass-conserving eta-floor
    # redistribution (n_iter allreduces).
    if config.fix_eta_drift:
        specs["eta_drift_projection"] = _make_eta_drift_spec(model, dt_static)

    # tracer_advection (ADVECTION ONLY) + tracer_tail (the rest of the
    # tracer transport setup the advection phase excludes).
    specs["tracer_advection"] = _make_tracer_advancer(model, dt_static)
    specs["tracer_tail"] = _make_tracer_tail_spec(model, dt_static)

    # gm_redi: only when configured (OFF in the bench default; ON for OMIP).
    # _make_gm_redi_spec returns None (skip) for an EKE config it cannot
    # faithfully isolate (codex findings 2/9) — then the GM/Redi+EKE cost
    # falls into the residual (noted by the residual label).
    if config.gm_redi is not None:
        _gm_spec = _make_gm_redi_spec(model, dt_static)
        if _gm_spec is not None:
            specs["gm_redi"] = _gm_spec

    # implicit_vmix: backward-Euler vertical mixing (default ON).
    if config.implicit_vertical_mixing:
        specs["implicit_vmix"] = _make_implicit_vmix_spec(model, dt_static)

    # conservation_fixer: global area/heat/salt rescale (default OFF).
    if config.use_conservation_fixer:
        specs["conservation_fixer"] = _make_cons_fixer_spec(model, dt_static)

    return specs


def _make_tracer_advancer(model, dt: float):
    """Tracer ADVECTION flux-divergence advance closure (state -> state).

    Times the tracer ADVECTION work ONLY: the advection flux divergence
    for BOTH tracers (the same ``_compute_advection_flux_div`` / RK3
    tracer step the step calls, including the advection-scheme halos),
    folded forward-Euler.  It deliberately EXCLUDES the rest of the
    in-situ tracer tail (``w_baro`` diagnosis + mass-flux correction,
    GM/Redi, the default-on implicit vertical-mixing solve) — so the full
    tracer tail is LARGER than this phase (see the module note).  Uses
    the current thickness + a zero barotropic transport (``w_baro=0``)
    for the mass flux — faithful for COST (the advection stencil + its
    halos are value-independent; ``w_baro=0`` keeps the vertical-flux
    stencil exercised), NOT for numerics.
    """
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface, min_cell_to_vface,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _compute_advection_flux_div, _ssp_rk3_tracer_step,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    dt_static = float(dt)
    grid = model.grid
    z_coord = model.z_coord
    config = model.config
    use_rk3 = config.tracer_time_integrator == "rk3"
    adv = config.tracer_advection

    def _tracer(state):
        mask = state.land_mask.data
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )
        h_u = min_cell_to_uface(h_k)
        h_v = min_cell_to_vface(h_k, grid)
        # Mass fluxes from the current 3D velocity (thickness-weighted);
        # w_baro = 0 keeps the vertical-flux stencil exercised (the
        # vertical divergence is computed regardless of magnitude).
        mass_flux_u = h_u * state.u.data
        mass_flux_v = h_v * state.v.data
        w_baro = jnp.zeros(
            (*state.land_mask.data.shape, h_k.shape[-1] + 1), dtype=h_k.dtype,
        )
        new_T = state.T.data
        new_S = state.S.data
        for tr_name in ("T", "S"):
            tr = state.T.data if tr_name == "T" else state.S.data
            if use_rk3:
                tr_new = _ssp_rk3_tracer_step(
                    tr, adv, mass_flux_u, mass_flux_v, w_baro,
                    h_k, h_k, h_u, h_v, grid, dt_static,
                    mask[..., jnp.newaxis],
                )
            else:
                div_hut, vert = _compute_advection_flux_div(
                    tr, adv, mass_flux_u, mass_flux_v, w_baro,
                    h_k, h_u, h_v, grid, dt_static,
                )
                hT_new = h_k * tr - dt_static * (div_hut + vert)
                tr_new = hT_new / jnp.maximum(h_k, 1e-10)
            if tr_name == "T":
                new_T = tr_new
            else:
                new_S = tr_new
        return state._replace(
            T=state.T.replace(data=new_T), S=state.S.replace(data=new_S),
        )

    # Halo count: per tracer the horizontal flux does ONE meridional
    # cell->v-face interp (tvd_to_v_points/upwind_to_v_points/interp_to_v_
    # points -> pad_with_pole_bc_lat) + ONE divergence_cgrid (pad_ns_zero on
    # the v-face flux); RK3 repeats the flux-div 3x => 3*(1+1) per tracer.
    # No global reduction (the flux-form update is purely local).  Latency-
    # relevant (the v-face/divergence halos grow with ranks).
    _per_tr = (3 if use_rk3 else 1) * 2  # (interp + divergence) per flux-div eval
    _halos = _per_tr * 2                  # T and S
    return PhaseSpec(
        advance_fn=_tracer, halos=_halos, reductions=0, bound="latency",
        note=("ocean_model_latlon_cgrid._compute_advection_flux_div: per "
              "tracer 1 cell->v-face interp (pad_with_pole_bc_lat) + 1 "
              "divergence_cgrid (pad_ns_zero); x3 for RK3; no global "
              "reduction"),
    )


def _make_coriolis_momint_spec(model, dt: float) -> "PhaseSpec":
    """Momentum integrator + forward-backward Matsuno Coriolis spec.

    Isolates ``_step_impl`` steps 3-4 EXCLUDING the per-eval baroclinic
    tendency cost (which is the ``baroclinic`` phase, and — for RK3 — its
    extra stage re-evals are attributed by ``baroclinic_insitu``).  For
    forward-Euler momentum this is the F_slow depth-mean split + the
    perturbation forward-Euler + the ``_forward_backward_coriolis_3d``
    folding (which does the u'->v-face 4-point average via
    ``interp_u_to_vface_4pt`` -> one meridional halo).  For SSP-RK3 the 2
    extra ``_mom_pert`` tendency re-evals are DELIBERATELY NOT re-run here
    (they would double-count the baroclinic_insitu attribution); this phase
    times the Euler/Coriolis folding ONLY, so on RK3 it UNDER-counts the
    integrator's true cost by the 2 re-evals (which baroclinic_insitu
    already carries).  Faithful for cost on the bench-default Euler config.
    Reductions: 0 (the depth-mean H_u/H_v sums are over the level axis,
    local).
    """
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface, min_cell_to_vface,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _forward_backward_coriolis_3d,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    dt_static = float(dt)
    dt_mom = dt_static / model.config.dt_mom_ratio
    grid = model.grid
    z_coord = model.z_coord
    config = model.config

    def _coriolis_momint(state):
        u = state.u.data
        v = state.v.data
        u_mask = state.u_mask.data
        v_mask = state.v_mask.data
        u_mask_3d = u_mask[..., jnp.newaxis]
        v_mask_3d = v_mask[..., jnp.newaxis]
        # Reproduce the step's F_slow depth-mean split on a stand-in
        # tendency (the actual du_dt is the baroclinic phase's output — its
        # VALUE is irrelevant to the depth-mean-split + Coriolis op/halo
        # schedule we are timing here).  Use the current velocity as the
        # stand-in tendency: same shapes, same reductions, same masks.
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )
        h_u = min_cell_to_uface(h_k)
        h_v = min_cell_to_vface(h_k, grid)
        H_u = jnp.maximum(jnp.sum(h_u, axis=-1), 1e-10)
        H_v = jnp.maximum(jnp.sum(h_v, axis=-1), 1e-10)
        F_slow_u = jnp.sum(u * h_u, axis=-1) / H_u * u_mask
        F_slow_v = jnp.sum(v * h_v, axis=-1) / H_v * v_mask
        du_pert = u - F_slow_u[..., jnp.newaxis]
        dv_pert = v - F_slow_v[..., jnp.newaxis]
        u_star = u + dt_mom * du_pert
        v_star = v + dt_mom * dv_pert
        # The real forward-backward Coriolis (does interp_u_to_vface_4pt ->
        # one meridional pad_with_pole_bc_lat halo).
        u_star, v_star = _forward_backward_coriolis_3d(
            u_star, v_star, dt_mom, grid, z_coord, config,
            u_mask, v_mask, state.land_mask.data,
            state.eta.data, state.H_bathy.data,
        )
        u_star = u_star.at[:, -1].set(u_star[:, 0])
        return state._replace(
            u=state.u.replace(data=u_star * u_mask_3d),
            v=state.v.replace(data=v_star * v_mask_3d),
        )

    return PhaseSpec(
        advance_fn=_coriolis_momint, halos=1, reductions=0, bound="latency",
        note=("ocean_model_latlon_cgrid._forward_backward_coriolis_3d: 1 "
              "u'->v-face 4pt halo (interp_u_to_vface_4pt -> "
              "pad_with_pole_bc_lat); depth-mean split is level-axis-local; "
              "no global reduction. RK3 stage re-evals attributed by "
              "baroclinic_insitu, not here"),
    )


def _make_eta_drift_spec(model, dt: float) -> "PhaseSpec":
    """Global mean-eta drift projection + eta-floor redistribution spec.

    The REAL ``_step_impl`` step-6b block (``config.fix_eta_drift``): one
    ``ocean_global_sum`` of a stacked 3-vector (target/actual/area) -> ONE
    allreduce(SUM), then the mass-conserving eta-floor
    ``clamp_and_redistribute`` (``n_iter=3`` redistribute allreduces).  The
    cost is reduction-latency-dominated and VALUE-INDEPENDENT (the
    allreduce schedule is fixed by shape + config), so timing it on the
    seed state is faithful.  Strongly LATENCY-bound.
    """
    import jax.numpy as jnp

    from legoesm.core.precision import cast as _cast
    from legoesm.ocean.conservation import ocean_global_sum
    from legoesm.ocean.dynamics.eta_floor import clamp_and_redistribute

    # dt is unused: the eta-drift block's allreduce schedule is dt-
    # independent (the cost we time); kept in the signature for uniformity.
    del dt
    grid = model.grid
    config = model.config
    _M = "ocean_diagnostics"
    # Reduction count derived from the helper's own default (codex finding
    # 6): clamp_and_redistribute is a Python ``for _ in range(n_iter)`` with
    # exactly ONE batched allreduce per iteration and NO value-dependent
    # early exit (eta_floor.py) — a FIXED, no-short-circuit collective
    # schedule.  Read n_iter from the signature default so this tracks the
    # source instead of hard-coding 3.
    import inspect as _inspect
    _floor_iters = int(
        _inspect.signature(clamp_and_redistribute)
        .parameters["n_iter"].default
    )

    def _eta_drift(state):
        mask_eta = state.land_mask.data
        area_eta = grid.area
        eta_d = state.eta.data
        mask_acc = _cast(mask_eta, _M, "accumulate")
        area_acc = _cast(area_eta, _M, "accumulate")
        eta_acc = _cast(eta_d, _M, "accumulate")
        wa = area_acc * mask_acc
        # target == actual on the seed (no barotropic update applied here);
        # the SUBTRACTION + allreduce schedule is what we time, so this is
        # faithful for cost.
        target_local = jnp.sum(eta_acc * wa)
        actual_local = jnp.sum(eta_acc * wa)
        ocean_area_local = jnp.sum(wa)
        target_mass, actual_mass, ocean_area = ocean_global_sum(
            jnp.stack([target_local, actual_local, ocean_area_local])
        )
        eta_correction = (target_mass - actual_mass) / jnp.maximum(
            ocean_area, 1.0e-30
        )
        eta_fixed = eta_d + eta_correction.astype(eta_d.dtype) * mask_eta
        if config.min_water_column_m is not None:
            eta_floor = (
                jnp.asarray(config.min_water_column_m, dtype=eta_fixed.dtype)
                - state.H_bathy.data
            )
            eta_fixed = clamp_and_redistribute(
                eta_fixed, eta_floor, mask_eta, area_eta,
            )
        return state._replace(eta=state.eta.replace(data=eta_fixed))

    _redux = 1 + (_floor_iters if config.min_water_column_m is not None else 0)
    return PhaseSpec(
        advance_fn=_eta_drift, halos=0, reductions=_redux, bound="latency",
        note=(f"ocean_model_latlon_cgrid._step_impl step-6b: 1 ocean_global_"
              f"sum(stacked-3) allreduce + eta_floor.clamp_and_redistribute "
              f"{_floor_iters} redistribute allreduces = {_redux} "
              f"allreduce(SUM); no halo (purely global reductions)"),
    )


def _make_tracer_tail_spec(model, dt: float) -> "PhaseSpec":
    """Tracer transport TAIL the advection phase excludes (spec).

    The per-layer mass-flux correction (Hallberg-Adcroft: correct the
    barotropic component of the 3D velocity so depth-integrated transport
    matches ``Hu_avg``) + the ``w_baro`` diagnosis (``divergence_cgrid`` of
    the corrected mass flux -> ``diagnose_w_from_flux_div``).  This is
    ``_step_impl`` step-7 lines ~1238-1332 MINUS the advection flux-div
    (the ``tracer_advection`` phase).  ``Hu_avg``/``Hv_avg`` come from the
    barotropic solver in the real step; here use the current
    thickness-weighted transport as the stand-in (same shapes/halos —
    value-independent for the op schedule).  Halo: the ``divergence_cgrid``
    for w (1 pad_ns_zero).  Reductions: 0 (the H_u/Hu_3d sums are
    level-axis-local).
    """
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        divergence_cgrid, min_cell_to_uface, min_cell_to_vface,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness, diagnose_w_from_flux_div,
    )

    grid = model.grid
    z_coord = model.z_coord
    config = model.config

    def _tracer_tail(state):
        h_k_old = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, z_coord,
            min_water_column_m=config.min_water_column_m,
        )
        h_u_old = min_cell_to_uface(h_k_old)
        h_v_old = min_cell_to_vface(h_k_old, grid)
        u_3d = state.u.data
        v_3d = state.v.data
        # Mass-flux correction (Hallberg-Adcroft): the level-axis reductions
        # + the uniform barotropic correction.  Hu_avg stand-in = the
        # current transport (correction is then ~0, but the op/halo schedule
        # is identical and value-independent).
        _u_pair = jnp.sum(
            jnp.stack([h_u_old, u_3d * h_u_old], axis=-1), axis=-2)
        H_u_old, Hu_3d = _u_pair[..., 0], _u_pair[..., 1]
        _v_pair = jnp.sum(
            jnp.stack([h_v_old, v_3d * h_v_old], axis=-1), axis=-2)
        H_v_old, Hv_3d = _v_pair[..., 0], _v_pair[..., 1]
        delta_U = (Hu_3d - Hu_3d) / jnp.maximum(H_u_old, 1e-10)
        delta_V = (Hv_3d - Hv_3d) / jnp.maximum(H_v_old, 1e-10)
        u_corr = u_3d + delta_U[..., jnp.newaxis]
        v_corr = v_3d + delta_V[..., jnp.newaxis]
        mass_flux_u = h_u_old * u_corr * state.u_mask.data[..., jnp.newaxis]
        mass_flux_v = h_v_old * v_corr * state.v_mask.data[..., jnp.newaxis]
        # w_baro diagnosis: divergence_cgrid (1 meridional halo) ->
        # diagnose_w_from_flux_div.
        flux_div_k = divergence_cgrid(mass_flux_u, mass_flux_v, grid)
        w_baro = diagnose_w_from_flux_div(
            flux_div_k, z_coord, thickness_weighted=True,
        )
        w_full = 0.5 * (w_baro[..., :-1] + w_baro[..., 1:])
        return state._replace(
            w=state.w.replace(data=w_full),
        )

    return PhaseSpec(
        advance_fn=_tracer_tail, halos=1, reductions=0, bound="latency",
        note=("ocean_model_latlon_cgrid._step_impl step-7 tail: mass-flux "
              "correction (level-axis-local) + w_baro via 1 divergence_cgrid "
              "(pad_ns_zero) + diagnose_w_from_flux_div; no global reduction"),
    )


def _make_gm_redi_spec(model, dt: float) -> "PhaseSpec":
    """GM/Redi isoneutral (neutral) tracer-tendency spec.

    The REAL ``gm_redi_tracer_tendency_latlon`` — the isoneutral tensor
    (density via a 2-iteration EOS coupling, isopycnal slopes, the
    skew + diffusive tracer fluxes folded through ``divergence_cgrid``).
    Often the single most expensive ocean operator (many face/triad
    meridional halos).  OFF in the bench-default config (``gm_redi=None``);
    this spec is only built when ``config.gm_redi`` is set.  Halo count is
    the meridional operators in the GM/Redi flux assembly (slope triads at
    v-faces + the final divergence); reductions: 0 (no global MPI sum —
    the Visbeck/EKE coefficients are local).

    Returns ``None`` (the phase is SKIPPED, with a rank-0 warning) when
    ``config.gm_redi.eke is not None`` (codex findings 2/9): the in-situ
    step runs a prognostic-EKE sub-step BEFORE GM/Redi and feeds
    ``kappa_gm_override``/``kappa_redi_override`` into the tendency
    (ocean_model_latlon_cgrid:1398-1452); this isolated phase omits the EKE
    sub-step (which itself does horizontal EKE transport) and calls the
    tendency WITHOUT the overrides — a different, under-counting code path.
    Rather than mis-time it, skip and tell the reader the EKE+GM cost is in
    the residual for an EKE config.
    """
    import jax.numpy as jnp

    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        gm_redi_tracer_tendency_latlon,
    )

    dt_static = float(dt)
    grid = model.grid
    z_coord = model.z_coord
    config = model.config
    gm_cfg = config.gm_redi

    if getattr(gm_cfg, "eke", None) is not None:
        # Unfaithful to isolate (see docstring) — skip rather than mis-time.
        return None

    def _gm_redi(state):
        mask_3d = state.land_mask.data[..., jnp.newaxis]
        dT_gm, dS_gm = gm_redi_tracer_tendency_latlon(
            state.T.data, state.S.data, state.eta.data, state.H_bathy.data,
            grid, z_coord, gm_cfg,
            eos=config.eos, eos_linear=config.eos_linear,
            mask=state.land_mask.data,
            u_mask=state.u_mask.data, v_mask=state.v_mask.data,
            rho_0=config.constants.rho_0, g=config.constants.g,
        )
        T_new = state.T.data + dt_static * dT_gm * mask_3d
        S_new = state.S.data + dt_static * dS_gm * mask_3d
        return state._replace(
            T=state.T.replace(data=T_new), S=state.S.replace(data=S_new),
        )

    # Halo count: the GM/Redi flux assembly builds meridional v-face slope
    # triads + a final divergence_cgrid per tracer; the isoneutral tensor
    # touches several N-S pads.  Reported as the documented ~6 meridional
    # exchanges of the neutral-flux assembly (slope-triad v-face pads +
    # divergence); cross-check the file:line in the note (it is an
    # ESTIMATE — GM/Redi has the densest halo footprint of any phase).
    return PhaseSpec(
        advance_fn=_gm_redi, halos=6, reductions=0, bound="latency",
        note=("gm_redi_latlon_cgrid.gm_redi_tracer_tendency_latlon: "
              "isoneutral tensor (2-iter EOS) + slope-triad v-face pads + "
              "divergence_cgrid (line ~1051); ~6 meridional halos (ESTIMATE "
              "— densest halo footprint); no global reduction"),
    )


def _make_implicit_vmix_spec(model, dt: float) -> "PhaseSpec":
    """Backward-Euler implicit vertical-mixing spec.

    The REAL ``model._apply_implicit_vertical_mixing`` — the K-profile
    build (in the bench-default ``physics=None`` config this is the
    FALLBACK ``compute_vertical_K_profiles``: EOS density + N² +
    Richardson/KPP) + one per-column tridiagonal Thomas solve for each of
    T, S, u, v.  The tridiagonal Thomas solve is COLUMNWISE in z (no
    horizontal stencil) and there is NO global reduction.  HOWEVER the
    MOMENTUM solve first interpolates ``A_v`` and ``dz`` from cell centres
    to the v-faces via ``interp_to_v_points`` (ocean_pe_latlon_cgrid:143 ->
    ``pad_ns_zero``) at ocean_model_latlon_cgrid:2053 and :2055 — TWO
    meridional halo exchanges (codex finding 1).  So the phase is NOT
    halo-free: it is tagged ``latency`` (it has a rank-growing term) even
    though the BULK of its cost (the Thomas solve) is rank-constant compute.
    The lever-split note flags this: a large implicit_vmix share is mostly
    compute-floor but carries 2 sendrecvs/rank that grow weakly with ranks.
    The ``interp_cell_to_uface`` for u (ocean_model_latlon_cgrid:2052,2054)
    is u-face / longitude-periodic (``jnp.roll``) — no MPI.

    FIDELITY: called with ``K_v_phys=None, A_v_phys=None``, which on the
    bench config is the SAME path the step takes (``physics=None`` ⇒
    ``tend.K_v/A_v=None`` ⇒ fallback recompute), and ``dt_mom = dt /
    dt_mom_ratio`` exactly as the step passes it (codex finding 5).  For an
    OMIP-production config that passes precomputed ``tend.K_v/A_v``, the
    step SKIPS the K recompute (fast path) but this isolated phase
    recomputes it — an OVER-count of this phase's compute on that config
    (printed warning in the note).
    """
    dt_static = float(dt)
    dt_mom = dt_static / model.config.dt_mom_ratio
    config = model.config

    def _implicit_vmix(state):
        # surface_forcing=None matches the bench step (no surface forcing
        # passed); K_v_phys/A_v_phys=None -> the fallback K-profile recompute
        # the bench config uses (physics=None).  dt_mom = dt/dt_mom_ratio
        # matches the step (codex finding 5: friction uses dt_mom, not dt).
        return model._apply_implicit_vertical_mixing(
            state, dt_static, None,
            K_v_phys=None, A_v_phys=None, K33_iso=None,
            dt_mom=dt_mom, surface_tracer_forcing=None,
        )

    _fast = "(K_v/A_v precomputed in-situ — phase OVER-counts the K rebuild)"
    _note_extra = "" if config.physics is None else f" {_fast}"
    return PhaseSpec(
        advance_fn=_implicit_vmix, halos=2, reductions=0, bound="latency",
        note=("ocean_model_latlon_cgrid._apply_implicit_vertical_mixing: "
              "columnwise tridiagonal Thomas solve (T,S,u,v; rank-constant "
              "compute) + 2 interp_to_v_points (pad_ns_zero) for the momentum "
              "A_v/dz -> latency-tagged though MOSTLY compute-floor; no "
              "global reduction" + _note_extra),
    )


def _make_cons_fixer_spec(model, dt: float) -> "PhaseSpec":
    """Conservation-fixer spec (area/heat/salt global rescale).

    The REAL ``ocean_conservation_fixer`` — global area/heat/salt
    reductions (``ocean_global_sum`` of stacked integrals) + a uniform
    rescale.  OFF in the bench default (``use_conservation_fixer=False``).
    Reduction-latency-dominated; the rescale needs the old + new global
    integrals.  ``state_old`` stand-in = the same seed state (the fixer
    measures old-vs-new drift; on identical states the drift is ~0 but the
    allreduce schedule — 2-3 stacked global sums — is value-independent and
    faithful for cost).  LATENCY-bound.
    """
    # dt is unused: the fixer's allreduce schedule is dt-independent (the
    # cost we time); kept in the signature for uniformity.
    del dt
    grid = model.grid
    z_coord = model.z_coord
    config = model.config

    from legoesm.ocean.conservation import ocean_conservation_fixer

    def _cons_fixer(state):
        return ocean_conservation_fixer(state, state, grid, z_coord, config)

    # Reductions: the fixer issues a small fixed number of stacked global
    # sums (volume + heat + salt, each a stacked old/new/area allreduce).
    # Reported as 3 (one per conserved integral family); the exact batching
    # is in conservation.py.
    return PhaseSpec(
        advance_fn=_cons_fixer, halos=0, reductions=3, bound="latency",
        note=("ocean.conservation.ocean_conservation_fixer: ~3 stacked "
              "ocean_global_sum allreduces (vol/heat/salt old-vs-new) + "
              "uniform rescale; no halo"),
    )


def _uninstrumented_active_blocks(config, freshwater) -> list[str]:
    """List step blocks that RUN for this config but are NOT timed.

    Codex findings 4/7/8: a handful of ``_step_impl`` blocks are gated OFF
    by default (so the bench/sbatch never hit them) but, if a config turns
    them on, they execute and their cost lands in the SUM-CHECK residual
    rather than a phase row.  Surface them explicitly so the residual is
    honestly attributed instead of silently absorbing a real block.
    """
    active: list[str] = []
    if getattr(config, "B_h_barotropic", 0.0) > 0.0:
        active.append(
            "B_h_barotropic>0 (depth-mean biharmonic slow forcing, "
            "_step_impl A2 block ~:1005-1030; halo-heavy vector bilaplacian)"
        )
    if getattr(config, "adaptive_implicit_vertadv", False):
        active.append(
            "adaptive_implicit_vertadv (7b vertical-momentum-advection "
            "solve ~:1348-1380)"
        )
    if freshwater is not None and getattr(
        config, "freshwater_closure", "none"
    ) != "none":
        active.append(
            "freshwater virtual-salt block (~:1617-1665; this bench passes "
            "freshwater=None, so normally inactive)"
        )
    if config.polar_filter.use_polar_filter:
        active.append("use_polar_filter (post-step Fourier filter)")
    if getattr(config, "freeze_floor", False):
        active.append("freeze_floor (post-step SST clamp)")
    if getattr(config, "ew_cyclic_overlap", False):
        active.append("ew_cyclic_overlap (post-step ORCA seam slave)")
    if getattr(config, "outer_integrator", "forward_euler") == "ab2":
        active.append(
            "outer_integrator='ab2' (the step runs _ab2_step, NOT the "
            "_step_impl path these phases model — the split is NOT valid "
            "for this config)"
        )
    return active


def profile_phases(
    model, state, dt: float, n_warmup: int, n_timing: int,
    full_step_ms: float, n_ranks: int, allreduce_latency_us: float,
    is_rank0: bool,
) -> None:
    """Time each ocean-step phase in ISOLATION; report ms/step + %.

    ``full_step_ms`` is the measured full-``model.step`` ms/step (from
    ``time_case``) — the denominator for the per-phase %.  Prints a
    rank-0 table; every phase is timed on EVERY rank (the closures run
    the identical collective schedule — matched sendrecv/allreduce — so
    no rank diverges).  See the module-level note for the approximation,
    the config-gated phase list, the halo/reduction counts, and the
    compute-vs-latency tagging.
    """
    import jax

    specs = _make_phase_advancers(model, dt)
    # rows: (name, ms/step, % of full, PhaseSpec)
    rows: list[tuple[str, float, float, PhaseSpec]] = []
    for name, spec in specs.items():
        # Each phase: clone the seed so phases don't perturb each other
        # (the closures are pure, but the scan pre-compile mutates leaves
        # off a clone anyway; clone the seed for clarity + safety).
        seed = jax.tree.map(lambda x: x, state)
        _c, _w, timing_s, _final = _time_fused_fn(
            spec.advance_fn, seed, n_warmup, n_timing,
        )
        ms = timing_s / n_timing * 1000.0
        rows.append((name, ms, 100.0 * ms / max(full_step_ms, 1e-12), spec))

    # In-situ baroclinic cost = per-eval * the step's tendency-eval count
    # (RK3 re-runs the tendency for stages 2,3 — those re-evals are NOT a
    # separate timed phase, they are attributed here).
    insitu_evals = _baroclinic_insitu_evals(model.config)
    baro_clinic_ms = next((m for n, m, _, _ in rows if n == "baroclinic"), 0.0)
    baroclinic_insitu_ms = baro_clinic_ms * insitu_evals
    # Extra (stage 2,3) baroclinic re-eval cost not in the single-eval
    # baroclinic row — folded into the sum so the residual is computed
    # against the in-situ baroclinic cost (avoids a spurious negative
    # residual on RK3 where the step pays insitu_evals tendency evals).
    baroclinic_extra_ms = baroclinic_insitu_ms - baro_clinic_ms

    # SUM-CHECK: Σ(timed phases, in_sum=True) + the RK3 baroclinic extra
    # evals.  The residual (full - Σ) is fusion/overlap (a comm-bound phase
    # OVER-counts standalone, pushing Σ above 100 -> negative residual) plus
    # any uninstrumented sliver (post-step polar filter / freeze floor /
    # ew-overlap when those gates are on; small masking ops).
    sum_ms = sum(m for _, m, _, sp in rows if sp.in_sum) + baroclinic_extra_ms
    sum_pct = 100.0 * sum_ms / max(full_step_ms, 1e-12)
    residual_ms = full_step_ms - sum_ms
    residual_pct = 100.0 - sum_pct

    # Aggregate the rank-growing (latency) vs constant (compute) shares —
    # the lever decision.  A phase is rank-growing iff it has any halo or
    # reduction (bound=="latency"); compute-bound otherwise.
    latency_ms = sum(
        m for _, m, _, sp in rows if sp.in_sum and sp.bound == "latency"
    ) + baroclinic_extra_ms  # the RK3 re-evals are halo-heavy tendency calls
    compute_ms = sum(
        m for _, m, _, sp in rows if sp.in_sum and sp.bound == "compute"
    )

    # Barotropic reduction-latency FLOOR (config-derived, timing-free):
    # 2*M allreduces/step * per-allreduce latency.  M is the fixed PCG
    # iteration count; only meaningful on the distributed implicit_cn path.
    M = int(model.config.barotropic.barotropic_implicit_pcg_fixed_iters)
    allreduce_count = 2 * M
    baro_latency_floor_ms = (
        allreduce_count * allreduce_latency_us / 1000.0
        if (n_ranks > 1 and model.config.barotropic.barotropic_solver == "implicit_cn")
        else 0.0
    )

    if not is_rank0:
        return

    # Honesty pass: name any active-but-uninstrumented step block whose cost
    # would otherwise hide in the residual (codex findings 4/7/8).  The bench
    # times model.step with freshwater=None, so pass None here.
    _uninstrumented = _uninstrumented_active_blocks(model.config, None)
    if _uninstrumented:
        print("  " + "-" * 74, flush=True)
        print(
            "  WARNING: these ACTIVE step blocks are NOT timed as phases — "
            "their cost is", flush=True,
        )
        print("  in the residual below:", flush=True)
        for _blk in _uninstrumented:
            print(f"    - {_blk}", flush=True)

    print("  " + "-" * 74, flush=True)
    print(
        "  PHASE SPLIT (ISOLATION estimate — phases timed ALONE; XLA fusion "
        "+ MPI", flush=True,
    )
    print(
        "  comm/compute OVERLAP lost => the residual below absorbs "
        "fusion/overlap;", flush=True,
    )
    print(
        "  a comm-bound phase's % is an UPPER-ish bound. See module note. "
        "H=N-S halo", flush=True,
    )
    print(
        "  exchanges/call (sendrecv pairs/cut), R=global allreduce(SUM)/call.",
        flush=True,
    )
    print("  " + "-" * 74, flush=True)
    print(
        f"  {'phase':30s} {'ms/step':>9s} {'%full':>7s}  {'H':>3s} {'R':>4s} "
        f"{'bound':>8s}", flush=True,
    )
    print(
        f"  {'full model.step':30s} {full_step_ms:9.2f} {100.0:7.1f}  "
        f"{'-':>3s} {'-':>4s} {'-':>8s}", flush=True,
    )
    for name, ms, pct, sp in rows:
        print(
            f"  {name:30s} {ms:9.2f} {pct:7.1f}  {sp.halos:3d} "
            f"{sp.reductions:4d} {sp.bound:>8s}", flush=True,
        )
    if insitu_evals != 1:
        print(
            f"  {'baroclinic_insitu (x%d)' % insitu_evals:30s} "
            f"{baroclinic_insitu_ms:9.2f} "
            f"{100.0 * baroclinic_insitu_ms / max(full_step_ms, 1e-12):7.1f}  "
            f"{'~27n':>3s} {0:4d} {'latency':>8s}  "
            f"[momentum_time_integrator="
            f"{getattr(model.config, 'momentum_time_integrator', 'euler')}; "
            f"the (x{insitu_evals}) is folded into the sum below, not added "
            f"as a separate row]", flush=True,
        )
    print("  " + "-" * 74, flush=True)
    # SUM-CHECK with the explicitly-labelled residual.
    print(
        f"  {'Σ phases (in-situ)':30s} {sum_ms:9.2f} {sum_pct:7.1f}   "
        f"(baroclinic counted at its x{insitu_evals} in-situ cost)",
        flush=True,
    )
    if residual_ms < 0:
        _resid_label = "negative => isolation over-counts a comm-bound phase"
    else:
        _resid_label = (
            "positive => an uninstrumented sliver (post-step filter/floor) "
            "or lost fusion speedup"
        )
    print(
        f"  {'residual':30s} {residual_ms:9.2f} {residual_pct:7.1f}   "
        f"= full - Σphases  [fusion/overlap/uninstrumented; {_resid_label}]",
        flush=True,
    )
    print("  " + "-" * 74, flush=True)
    # COMPUTE vs LATENCY split (the lever decision).
    print(
        f"  LEVER SPLIT (which phases set the weak-scaling limit):",
        flush=True,
    )
    print(
        f"    RANK-GROWING share (phases WITH halos and/or reductions — "
        f"their", flush=True,
    )
    print(
        f"      comm latency grows as ranks increase; tagged 'latency'): "
        f"{latency_ms:8.2f} ms "
        f"({100.0 * latency_ms / max(full_step_ms, 1e-12):5.1f}% of full)",
        flush=True,
    )
    print(
        f"      NOTE: a rank-growing phase can ALSO be locally compute-heavy "
        f"(e.g.", flush=True,
    )
    print(
        f"      baroclinic does EOS+PGF AND ~27 halos) — its base cost still "
        f"counts.", flush=True,
    )
    print(
        f"    RANK-CONSTANT share (phases with NO halo and NO reduction — "
        f"pure", flush=True,
    )
    print(
        f"      per-column/cell; tagged 'compute'): {compute_ms:8.2f} ms "
        f"({100.0 * compute_ms / max(full_step_ms, 1e-12):5.1f}% of full)",
        flush=True,
    )
    # Identify the single biggest phase + whether it is rank-growing.
    if rows:
        big_name, big_ms, big_pct, big_sp = max(rows, key=lambda r: r[1])
        # RK3 in-situ baroclinic may exceed the single-eval baroclinic row.
        if insitu_evals != 1 and baroclinic_insitu_ms > big_ms:
            big_name, big_ms, big_pct, big_sp = (
                "baroclinic_insitu", baroclinic_insitu_ms,
                100.0 * baroclinic_insitu_ms / max(full_step_ms, 1e-12),
                rows[0][3]._replace(bound="latency"),
            )
        _lever = (
            "parallelism/comm (phase is rank-growing)"
            if big_sp.bound == "latency"
            else "per-device fusion/kernel (phase is COMPUTE-bound — "
                 "parallelism will NOT help it)"
        )
        print(
            f"    biggest phase: {big_name} ({big_pct:.1f}%, "
            f"bound={big_sp.bound}) => lever = {_lever}", flush=True,
        )
    print("  " + "-" * 74, flush=True)
    print(
        f"  BAROTROPIC reduction-latency FLOOR (timing-free, config-"
        f"derived):", flush=True,
    )
    print(
        f"    M = barotropic_implicit_pcg_fixed_iters = {M}  =>  2*M = "
        f"{allreduce_count} in-loop allreduce/step", flush=True,
    )
    if baro_latency_floor_ms > 0.0:
        print(
            f"    {allreduce_count} allreduce * {allreduce_latency_us:.0f} us "
            f"= {baro_latency_floor_ms:8.2f} ms/step "
            f"({100.0 * baro_latency_floor_ms / max(full_step_ms, 1e-12):5.1f}% "
            f"of full)  [CONSERVATIVE floor: the in-loop 2*M dots ONLY — "
            f"the solve also issues a final rhs.rhs reduction + the "
            f"mass-projection / eta-floor reductions, and cannot beat "
            f"2*M serially-dependent allreduces under MPI]",
            flush=True,
        )
    else:
        print(
            "    (latency floor only meaningful on the distributed "
            "implicit_cn path)", flush=True,
        )
    # Verdict heuristic (rank-0 print only; the numbers above are the
    # evidence — this is a convenience label, not a gate).
    baro_iso_ms = next((m for n, m, _, _ in rows if n == "barotropic"), None)
    if baro_iso_ms is not None:
        baro_iso_pct = 100.0 * baro_iso_ms / max(full_step_ms, 1e-12)
        floor_pct = 100.0 * baro_latency_floor_ms / max(full_step_ms, 1e-12)
        dominant = baro_iso_pct >= 50.0
        print("  " + "-" * 74, flush=True)
        print(
            f"  VERDICT: barotropic dominant? "
            f"{'YES' if dominant else 'NO'}  "
            f"(isolation share {baro_iso_pct:.1f}%, latency floor "
            f"{floor_pct:.1f}%; baroclinic_insitu "
            f"{100.0 * baroclinic_insitu_ms / max(full_step_ms, 1e-12):.1f}%). "
            f"Robust when isolation AND floor agree.", flush=True,
        )
    print("  " + "-" * 74, flush=True)


def _append_aggregate_csv(result, out_dir: Path) -> None:
    """Append one case row to ``<out_dir>/ocean_scaling.csv``.

    One-case-per-invocation design means ``write_csv`` (which truncates)
    can only hold the last case; the sweep-level aggregate keeps EVERY
    row so ``scripts/plot/plot_ocean_vs_oceananigans.py`` (which
    discovers ``ocean_scaling.csv`` via ``rglob``) sees the whole
    ladder.  Header written once; schema identical to ``write_csv``
    (TimingResult dataclass fields).  Rank 0 only — the sbatch sweep
    runs cases sequentially, so append is race-free.
    """
    import csv
    from dataclasses import asdict

    row = asdict(result)
    agg_path = out_dir / "ocean_scaling.csv"
    write_header = not agg_path.exists()
    with open(agg_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  CSV (aggregate): {agg_path}")


# ===========================================================================
# CLI
# ===========================================================================

def _warmup_count(text: str) -> int:
    """argparse type for --n-warmup: integer >= 1.

    The first warmup step IS the JIT-compile step and always advances
    the timed trajectory, so n_warmup=0 is unrepresentable (the
    conservation label would undercount by the compile step).
    """
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"--n-warmup must be >= 1 (the JIT-compile step is the "
            f"first warmup step); got {value}."
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ocean CPU MPI scaling benchmark (lat-lon C-grid, "
            "latitude-band decomposition). One case per MPI invocation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["strong", "weak"], default="strong",
        help="strong: --resolution is global n_lat (n_lon=2*n_lat). "
             "weak: --resolution is lat rows per rank "
             "(n_lat=rows*n_ranks, n_lon=2*rows fixed).",
    )
    p.add_argument(
        "--resolution", type=int, default=64,
        help="Global n_lat (strong) or per-rank base (weak).",
    )
    p.add_argument(
        "--weak-style", choices=list(WEAK_STYLE_CHOICES), default="band",
        help="Weak-scaling geometry (weak mode only; ignored for "
             "strong). band: n_lat=rows*np, n_lon=2*rows fixed — "
             "cells/rank and halo bytes/rank exactly constant "
             "(latitude-band ALGORITHMIC weak scaling; aspect ratio "
             "distorts with np). aspect: n_lat~=base*sqrt(np) rounded "
             "to a multiple of np, n_lon=2*n_lat — production "
             "square-cell aspect; cells/rank has integer-rounding "
             "drift, the ACTUAL value is recorded in the CSV row "
             "(cells_per_gpu).",
    )
    p.add_argument(
        "--n-levels", type=int, default=20,
        help="Vertical levels (matches bench_ocean_gpu_scaling).",
    )
    p.add_argument(
        "--precision", choices=["float32", "float64"], default="float64",
        help="One precision per process; jax_enable_x64 is set before "
             "any JAX import / model build.",
    )
    p.add_argument(
        "--device", choices=["cpu", "gpu"], default="cpu",
        help="cpu (default): CPU-MPI, 1 thread/rank. gpu: each MPI rank "
             "pins to ONE local GPU (CUDA_VISIBLE_DEVICES=local-rank) and "
             "runs JAX on cuda — the ocean lat-lon multi-GPU path "
             "(mpi4jax halos over the PCIe pair). Launch e.g. "
             "`mpirun -np 2 ... --device gpu` on a 2-GPU node.",
    )
    p.add_argument("--land-mask", choices=["none", "etopo"],
                   default="none",
                   help="Land mask for the benchmark problem: none = the rest-state default (polar caps only), etopo = realistic continents from --bathymetry-file (flat bottom kept — the mask is what wet-balance keys off). Use with --wet-balance for the row-vs-wet A/B on a realistic land distribution.")
    p.add_argument("--bathymetry-file", type=str,
                   default="data/bathymetry/etopo_1deg.nc",
                   help="NetCDF bathymetry for --land-mask etopo (shipped 1-degree ETOPO by default).")
    p.add_argument(
        "--wet-balance", action="store_true",
        help="Wet-cell-aware latitude bands: band boundaries equalize OCEAN "
             "cells per rank (wet_band_boundaries on the global land_mask) "
             "instead of row counts, so land-heavy bands stop idling. "
             "MPI-band path only (n_ranks > 1); every rank computes the "
             "identical boundaries from the identical global mask. "
             "MEASURED (np=4 CPU, LL96 etopo, 2026-07-08): the DENSE step "
             "computes land cells too, so cost scales with ROWS — wet "
             "bands cut wet imbalance 1.23->1.01 but ran ~3% SLOWER "
             "(13.7 vs 13.3 ms/step). This flag is groundwork for "
             "active/wet-cell COMPACTION (audit item 4); do not flip it "
             "on the dense step expecting a win.",
    )
    p.add_argument(
        "--baro-solver", choices=list(BARO_SOLVER_CHOICES),
        default="implicit_cn",
        help="Barotropic solver. implicit_cn matches the serial "
             "production default and is now MPI-safe at any rank count: "
             "the multi-rank path uses a distributed fixed-iteration PCG "
             "(static fori_loop of M iters + two batched allreduces/iter, "
             "unrolled + differentiated through for AD) with a uniform "
             "collective schedule. It is THE weak-scaling lever — ~2*M "
             "reductions/step independent of resolution, vs "
             "explicit_substep whose substep (and reduction) count grows "
             "with resolution.",
    )
    p.add_argument(
        "--n-warmup", type=_warmup_count, default=3,
        help="Warmup steps INCLUDING the JIT-compile first step "
             "(>= 1: the compile step always advances the timed "
             "trajectory, so 0 would mislabel the step count).",
    )
    p.add_argument("--n-timing", type=int, default=30)
    p.add_argument(
        "--pcg-variant", choices=["standard", "single_reduce"],
        default="standard",
        help="Distributed fixed-M PCG body (implicit_cn, np>1): "
             "'standard' = 2 reductions/iter; 'single_reduce' = "
             "Chronopoulos-Gear, 1 batched reduction/iter (M+1 vs 2M+1 "
             "per solve) — the multi-node weak-scaling lever.",
    )
    p.add_argument(
        "--preconditioner",
        choices=["jacobi", "zonal_line", "chebyshev", "multigrid"],
        default="jacobi",
        help="Implicit-CN PCG preconditioner: 'jacobi' (legacy), "
             "'zonal_line' (exact periodic-tridiagonal row solves; "
             "comm-free under band MPI; M-sweep 8473872: equal residual at "
             "~M/3) or 'chebyshev' (degree-4 polynomial of A; cuts outer M "
             "with NO per-iter reduction but +4 matvec-halos/iter; conv "
             "8486241: reaches jacobi-M60 accuracy at ~M40 = 80 vs 120 "
             "reductions — pair with --pcg-fixed-iters; wins only where "
             "allreduce log-N latency > halo, i.e. high rank counts).",
    )
    p.add_argument(
        "--pcg-fixed-iters", type=int, default=60,
        help="Fixed-M for the distributed implicit-CN PCG (the "
             "reduction count per solve is 2M+1 / M+1 by variant).",
    )
    p.add_argument(
        "--force-pcg", action="store_true",
        help="Force the fixed-M PCG even at a SINGLE rank, so a 1-vs-N "
             "strong-scaling ratio uses the SAME barotropic solver both "
             "sides (else np=1 stock jax.scipy CG vs np>=2 fixed-M PCG is "
             "apples-to-oranges). No effect at n_ranks>1 (PCG already used).",
    )
    p.add_argument(
        "--profile-phases", action="store_true",
        help="After the full-step timing, time EVERY major ocean-step "
             "sub-phase in ISOLATION (config-gated: baroclinic tendencies / "
             "momentum+Coriolis / implicit-CN barotropic solve / eta-drift "
             "projection / tracer advection / tracer tail / GM-Redi / "
             "implicit vertical mixing / conservation fixer) with the same "
             "warmup+block+scan discipline and report ms/step + %% of the "
             "full step + each phase's N-S halo-exchange count + global "
             "allreduce count + a COMPUTE-bound vs LATENCY-bound tag. An "
             "explicit SUM-CHECK (Σphases vs 100%%, residual labelled "
             "fusion/overlap/uninstrumented) attributes the full step (the "
             "original 3-phase split left ~74%% unaccounted). APPROXIMATE: "
             "phases timed alone lose XLA fusion and MPI comm/compute "
             "overlap, so a comm-bound phase's share is an upper-ish bound "
             "(the residual absorbs it; see the module note). Also reports "
             "the barotropic 2*M-allreduce reduction-latency FLOOR from "
             "--allreduce-latency-us (timing-free, config-derived) as a "
             "robust cross-check. No scaling CSV is written when this flag "
             "is set.",
    )
    p.add_argument(
        "--allreduce-latency-us", type=float, default=111.0,
        help="Measured per-allreduce latency [us] for the barotropic "
             "reduction-latency floor (2*M allreduces/step). Default 111 "
             "us = the Ginsburg np<=4 CPU-MPI roofline floor "
             "(scripts/bench/roofline_probe.py allreduce-latency probe). "
             "Only used with --profile-phases.",
    )
    p.add_argument(
        "--dt", type=float, default=600.0,
        help="Baroclinic timestep [s] (bench_ocean_gpu_scaling default; "
             "stable for the LL64-LL192 strong ladder and the rows=64 "
             "weak ladder up to np=32 with the default 80-deg land cap).",
    )
    p.add_argument(
        "--output-dir", type=str, default="results/scaling_cpu_ocean",
        help="Output directory (rank 0 writes one CSV+JSON per case).",
    )
    p.add_argument(
        "--check-conservation", action="store_true",
        help="Gate global area/eta/heat/salt drift across the "
             "compile+warmup+timed trajectory (allreduce(SUM)): any "
             "metric beyond --cons-rtol exits nonzero (code 4).",
    )
    p.add_argument(
        "--cons-rtol", type=float, default=None,
        help="Conservation-gate tolerance. Default: 1e-9 for float64, "
             "1e-4 for float32. area/heat/salt are relative drifts; "
             "the mean-eta drift is compared in metres. NOTE the raw "
             "scheme (no conservation fixer here) drifts heat/salt at "
             "~1e-8/step relative (LL64 f64 measured baseline) — "
             "calibrate for the run length (the sbatch smoke gate "
             "passes 1e-6 for ~33 steps).",
    )
    p.add_argument(
        "--parity-gate", action="store_true",
        help=f"Gathered-field MPI-vs-serial parity gate (smoke sizes "
             f"only: n_lat <= {PARITY_GATE_MAX_NLAT}). Every rank "
             f"computes the serial reference BEFORE the MPI backend is "
             f"armed; after timing, the band state is gathered and "
             f"T/S/eta/u/v compared on rank 0 (f64 rtol=1e-7/atol=1e-8; "
             f"f32 rtol=1e-4/atol=1e-5). Mismatch exits nonzero (code 5). "
             f"Catches partition-cut corruption invisible to the "
             f"conservation gate.",
    )
    p.add_argument(
        "--tripole", action="store_true",
        help="Tripolar (ORCA-fold) grid lane on the synthetic tripole: "
             "fold-aware band layout (north rank owns the seam), full-model "
             "step validated MPI-vs-serial (test_ocean_mpi_tripole_step_"
             "parity.py). explicit_substep barotropic only; rows are "
             "tagged grid=tripole.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.tripole:
        # WIRED (audit item 4): the full-model fold step is now validated
        # MPI-vs-serial (tests/ocean/distributed/
        # test_ocean_mpi_tripole_step_parity.py, np1/np2 at f64 1e-8) on
        # the synthetic tripole this lane builds.  Combinations that stay
        # refused, loudly:
        if args.baro_solver != "explicit_substep":
            raise SystemExit(
                "--tripole currently validates the explicit_substep "
                "barotropic only (the distributed implicit-CN PCG has no "
                "tripole parity case); pass --baro-solver "
                "explicit_substep.")
        if args.land_mask != "none":
            raise SystemExit(
                "--tripole builds the synthetic tripole's own cap/land "
                "mask; --land-mask etopo is a regular-lat-lon lane.")

    # --- Configure JAX BEFORE any JAX import (CPU or per-rank GPU) ---
    if args.device == "gpu":
        _configure_jax_gpu(args.precision)
    else:
        _configure_jax_cpu(args.precision)

    # --- MPI init ---
    rank, n_ranks = _init_mpi()
    is_rank0 = rank == 0

    n_lat, n_lon = resolve_grid_size(
        args.mode, args.resolution, n_ranks, args.weak_style,
    )
    cons_rtol = (
        args.cons_rtol if args.cons_rtol is not None
        else CONS_RTOL_DEFAULTS[args.precision]
    )

    # --- Up-front configuration guards (clear messages, no tracebacks) ---
    if args.parity_gate and n_lat > PARITY_GATE_MAX_NLAT:
        if is_rank0:
            print(
                f"ERROR: --parity-gate is a smoke-size gate (every rank "
                f"runs the FULL global serial reference); n_lat={n_lat} "
                f"exceeds the {PARITY_GATE_MAX_NLAT}-row ceiling.",
                flush=True,
            )
        return 2
    if args.resolution < MIN_ROWS_PER_RANK:
        if is_rank0:
            print(
                f"ERROR: --resolution {args.resolution} is below the "
                f"{MIN_ROWS_PER_RANK}-row minimum.",
                flush=True,
            )
        return 2
    if n_ranks > 1 and (n_lat // n_ranks) < MIN_ROWS_PER_RANK:
        if is_rank0:
            print(
                f"ERROR: lat-lon band MPI needs >={MIN_ROWS_PER_RANK} lat "
                f"rows per rank; n_lat={n_lat} on {n_ranks} ranks gives "
                f"{n_lat // n_ranks} rows/rank. Increase --resolution or "
                f"reduce ranks.",
                flush=True,
            )
        return 2
    # implicit_cn IS MPI-safe under the band decomposition as of the
    # distributed fixed-iteration PCG (see module docstring): the
    # multi-rank dispatch runs a static-length PCG with a uniform
    # collective schedule, so it no longer deadlocks.  This is THE
    # weak-scaling lever — implicit_cn issues a fixed M batched
    # allreduces per barotropic step independent of resolution, whereas
    # explicit_substep's substep count grows with resolution.

    mode_label = (
        f"ocean_weak_{args.weak_style}" if args.mode == "weak"
        else "ocean_strong"
    )
    n_advanced_steps = args.n_warmup + args.n_timing

    if is_rank0:
        print("=" * 72)
        print("  legoESM Ocean CPU MPI Scaling Benchmark (lat-lon C-grid)")
        print("=" * 72)
        print(f"  Mode:        {mode_label}")
        print(f"  Grid:        n_lat={n_lat} n_lon={n_lon} L{args.n_levels}")
        print(f"  Ranks:       {n_ranks}")
        print(f"  Precision:   {args.precision}")
        print(f"  Baro solver: {args.baro_solver}")
        print(f"  dt:          {args.dt:.0f} s")
        if args.check_conservation:
            print(f"  Cons gate:   cons_rtol={cons_rtol:.1e}")
        if args.parity_gate:
            _prt, _pat = PARITY_TOLS[args.precision]
            print(f"  Parity gate: rtol={_prt:.1e} atol={_pat:.1e}")
        print("=" * 72, flush=True)

    # --- Parity-gate serial reference: BEFORE build_case arms the MPI
    # halo backend (a serial step traced after arming embeds band
    # collectives no other rank matches — see compute_serial_reference).
    serial_ref = None
    if args.parity_gate:
        serial_ref = compute_serial_reference(
            n_lat=n_lat,
            n_lon=n_lon,
            nlev=args.n_levels,
            baro_solver=args.baro_solver,
            precision=args.precision,
            dt=args.dt,
            n_steps=n_advanced_steps,
            n_ranks=n_ranks,
            land_mask=args.land_mask,
            bathymetry_file=args.bathymetry_file,
            tripole=args.tripole,
        )

    model, state, total_cells, layout, part_metrics = build_case(
        n_lat=n_lat,
        n_lon=n_lon,
        nlev=args.n_levels,
        n_ranks=n_ranks,
        baro_solver=args.baro_solver,
        precision=args.precision,
        pcg_variant=args.pcg_variant,
        preconditioner=args.preconditioner,
        fixed_iters=int(args.pcg_fixed_iters),
        force_pcg=args.force_pcg,
        wet_balance=args.wet_balance,
        land_mask=args.land_mask,
        bathymetry_file=args.bathymetry_file,
        tripole=args.tripole,
    )
    cells_per_rank = total_cells // n_ranks

    # ETOPO coastlines destabilize the default deep-ocean dt (measured:
    # LL96 blows up at dt=600, stable at 150).  The post-run finite check
    # still discards a blown-up timing (exit 3), but warn BEFORE the
    # expensive compile+timing rather than after (codex).
    if is_rank0 and args.land_mask == "etopo" and args.dt > 300.0:
        print(
            f"  WARNING: --land-mask etopo with --dt {args.dt:g}s: realistic "
            f"coastlines have blown up at dt=600 (LL96); if the run ends "
            f"with 'non-finite values', retry with --dt 150.",
            flush=True,
        )

    if is_rank0:
        print(
            f"  [{args.precision}] LL{n_lat}/L{args.n_levels} on {n_ranks} "
            f"rank(s) | cells={total_cells:,} | cells/rank={cells_per_rank:,}",
            flush=True,
        )

    inv_before = None
    if args.check_conservation:
        # Collective (allreduce) — every rank must call this.
        inv_before = ocean_invariants(model, state, n_ranks)

    compile_s, warmup_s, timing_s, state_final = time_case(
        model, state, args.dt, args.n_warmup, args.n_timing,
    )

    cons_failed = False
    if args.check_conservation:
        inv_after = ocean_invariants(model, state_final, n_ranks)
        # The timed trajectory advances n_warmup + n_timing steps, the
        # first warmup step being the JIT-compile step (the scan
        # precompile runs on cloned leaves, not the timed trajectory).
        label = (
            f"{n_advanced_steps} steps "
            f"(1 compile + {args.n_warmup - 1} warmup + "
            f"{args.n_timing} timed)"
        )
        if is_rank0:
            print_conservation(label, inv_before, inv_after, cons_rtol)
        # Invariants are globally reduced -> identical verdict on every
        # rank (no extra collective needed for a consistent exit).
        breaches = conservation_breaches(inv_before, inv_after, cons_rtol)
        if breaches:
            cons_failed = True
            if is_rank0:
                print(
                    "ERROR: conservation gate BREACHED over "
                    f"{label}: " + "; ".join(breaches),
                    flush=True,
                )

    # Stability sanity: a benchmark of NaN math is fast and meaningless.
    if not check_finite(state_final, n_ranks):
        if is_rank0:
            print(
                "ERROR: non-finite values in the final state — timing "
                "discarded. Check dt/CFL for this resolution.",
                flush=True,
            )
        return 3

    # Parity gate: collective gather + compare vs the pre-arming serial
    # reference (every rank participates; verdict is bcast-consistent).
    if args.parity_gate:
        parity_ok, parity_lines = run_parity_gate(
            state_final, layout, serial_ref, n_ranks, args.precision,
            baro_solver=args.baro_solver,
        )
        if is_rank0:
            print(
                f"  [parity gate vs serial, {n_advanced_steps} steps, "
                f"np={n_ranks}]",
                flush=True,
            )
            for line in parity_lines:
                print(line, flush=True)
        if not parity_ok:
            if is_rank0:
                print(
                    "ERROR: parity gate MISMATCH — gathered MPI fields "
                    "diverged from the serial reference (partition-cut "
                    "corruption; conservation integrals cannot see "
                    "this).",
                    flush=True,
                )
            return 5

    if cons_failed:
        return 4

    step_wall_s = timing_s / args.n_timing
    time_per_step_ms = step_wall_s * 1000.0
    # SYPD formula matches run_levante_gpu_scaling / bench_ocean_gpu_scaling.
    sypd = (args.dt / step_wall_s) / (365.25 * 86400.0) * 86400.0
    mcells_per_s = total_cells / step_wall_s / 1e6

    # Phase split (GATE measurement): time the step's sub-phases in
    # isolation against the just-measured full-step ms/step.  COLLECTIVE
    # — every rank runs the phase closures (matched halo/allreduce
    # schedule), so it is called outside the rank-0 guard; only rank 0
    # prints.  No scaling CSV is written for a phase-profiling run (it is
    # an attribution measurement, not a scaling data point).
    if args.profile_phases:
        if is_rank0:
            print(
                f"  LL{n_lat:4d} np={n_ranks:2d} | compile={compile_s:6.2f}s "
                f"| step={time_per_step_ms:8.2f}ms (full) | "
                f"{mcells_per_s:6.2f} Mcells/s",
                flush=True,
            )
        profile_phases(
            model, state, args.dt, args.n_warmup, args.n_timing,
            full_step_ms=time_per_step_ms, n_ranks=n_ranks,
            allreduce_latency_us=args.allreduce_latency_us,
            is_rank0=is_rank0,
        )
        return 0

    result = TimingResult(
        n_gpus=n_ranks,  # MPI rank count (CPU run) — keeps plotter schema
        # grid_type default is the atm dataclass's 'cubed-sphere' — left
        # defaulted it made write_json's _decomp() label ocean rows
        # decomposition='mpi' instead of 'band' (codex).
        grid_type="tripole" if args.tripole else "latlon",
        resolution=n_lat,
        n_levels=args.n_levels,
        precision=args.precision,
        mode=mode_label,
        physics_level=f"baro={args.baro_solver}",
        dt_seconds=args.dt,
        n_warmup=args.n_warmup,
        n_timing=args.n_timing,
        compile_time_s=compile_s,
        warmup_time_s=warmup_s,
        timing_time_s=timing_s,
        time_per_step_ms=time_per_step_ms,
        sypd=sypd,
        total_cells=total_cells,
        cells_per_gpu=cells_per_rank,
        mcells_per_s=mcells_per_s,
    )

    if is_rank0:
        print(
            f"  LL{n_lat:4d} np={n_ranks:2d} | compile={compile_s:6.2f}s | "
            f"step={time_per_step_ms:8.2f}ms | SYPD={sypd:7.2f} | "
            f"{mcells_per_s:6.2f} Mcells/s",
            flush=True,
        )
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{mode_label}_r{n_lat}_np{n_ranks}_{args.precision}"
        write_csv([result], out_dir / f"{stem}.csv")
        _append_aggregate_csv(result, out_dir)
        import jax  # backend tag for the report (post-config import)

        report = ScalingReport(
            mode=mode_label,
            precisions=[args.precision],
            results=[result],
            backend=jax.default_backend().upper(),
            hostname=os.environ.get("HOSTNAME", "unknown"),
        )
        # component="ocean": without it write_json stamps its atmosphere
        # default on every row (mislabel); n_ranks_true: jax cannot see the
        # mpirun world (process_count()==1 per rank), so the real rank count
        # must be recorded explicitly — it also resolves transport="mpi4jax".
        # solver_variant records the wide-halo/local-clamp levers so an A/B
        # pair can never be conflated with the baseline in aggregation.
        # (The analytic barotropic halo-message census lives in the SPMD
        # bench's records — model.config is in scope there; here the census is
        # derivable offline from solver_variant + n_barotropic_substeps, so it
        # is deliberately not recomputed. Pre-merge codex note.)
        _grid_label = "tripole" if args.tripole else "latlon"
        _variant = args.baro_solver
        if os.environ.get("LEGOESM_BARO_LOCAL_CLAMP", "0") == "1":
            _variant += "+local_clamp"
        if os.environ.get("LEGOESM_BARO_WIDE_HALO", "0") == "1":
            _variant += "+wide_halo"
        _md_over = {"solver_variant": _variant}
        if part_metrics is not None:
            _md_over["partition_metrics"] = part_metrics
        _md_over["grid"] = _grid_label
        _md_over["decomposition"] = "band" if n_ranks > 1 else "none"
        write_json(
            report, out_dir / f"{stem}.json",
            n_ranks_true=n_ranks,
            component="ocean",
            metadata_overrides=_md_over,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
