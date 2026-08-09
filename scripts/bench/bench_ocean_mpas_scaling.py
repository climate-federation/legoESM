"""Weak/strong scaling bench for the MPAS/Voronoi OCEAN (CPU-MPI route-A).

The scaling-status audit tagged MPAS-ocean "(c) infrastructure-ready,
unmeasured — missing bench lane, not missing capability": ``MPASOceanModel``
steps rank-locally on a ``VoronoiPartitionLayout`` local mesh (mpi4jax halo
exchange; owned-cell-masked global reductions), but no harness drove it
multi-rank.  This is that lane — BENCHMARK ONLY, no numerics touched.

  strong: fixed subdivision level, vary np  -> speedup = t(1)/t(np).
  weak:   pick the level whose cells/rank best matches --cells-per-rank
          (icosahedral levels quantize by 4x per level — the ACHIEVED
          cells/rank is recorded on the row; never compare rows without it).

Correctness gates (fail-fast, BEFORE any timing is reported):
  --parity-gate         gathered owned cells vs the serial trajectory at the
                        re-association floor (smoke windows only).
  --check-conservation  global volume/heat/salt drift over the run.

Anti-fake-scaling guards: multi-rank REQUIRES an armed VoronoiPartitionLayout
(a replicated global-mesh run cannot masquerade as decomposed); rows carry
the shared self-describing metadata (transport resolves to mpi4jax via
n_ranks > process_count) + voronoi partition-quality metrics.

M1 measurement contract (scaling-M3d increment-1): the headline number is a
fused ``lax.scan`` block (``metadata.timed_scan_blocks``; per-step
dispatch latency probed SEPARATELY), cross-rank MAX-reduced, and it is what
the aggregator-facing ``steady_median_ms`` carries (the same deliberate
naming as ``bench_ocean_latlon_spmd_scaling``); the host-synced gate-loop
median is dispatch+sync LATENCY and is recorded only under
``step_latency_gate_loop_ms``, never as the headline.  Rows also carry
``wet_cell_metrics``, solver-iteration mode + a post-run zero-forcing
Helmholtz residual probe (implicit_cn only, outside the timed loop; under
``--halo-refresh none`` one packed exchange precedes the probe so it
measures a cleanly-assembled system, not rotten halos), and
``--halo-refresh`` (default auto => one PACKED full-state halo exchange per
step at n_ranks > 1 via ``exchange_state_mpas_ocean``; ``--halo-refresh
in_step`` — auto's multi-rank choice — additionally arms the IN-STEP
stage-frontier refreshes, making the row stage-correct).  STAGE CORRECTNESS:
the ocean step consumes more stencil hops per step than halo_depth between
refreshes, so multi-rank rows are labeled ``stage_halo_correct=false`` — see
docs/performance/scaling/mpas_ocean_distributed_stage_audit.md.  The parity
gate bounds halo-staleness error over a smoke window; it does not certify
stage correctness.

Run (CPU-MPI):
  mpirun -np 4 python scripts/bench/bench_ocean_mpas_scaling.py \
      --mode strong --subdivision 4 --nlev 10 --steps 12
GPU: this lane is route-A (one rank per GPU pinned via CUDA_VISIBLE_DEVICES,
same convention as bench_ocean_mpi_scaling --device gpu); a single-process
multi-device SPMD ocean-voronoi path does not exist (the sharded step builder
is the ATMOSPHERE TRiSK model) — this bench REFUSES to fake one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Bench dir for the shared metadata module (sibling-script import pattern).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from metadata import (  # noqa: E402
    annotate_incomplete,
    scaling_metadata,
    timed_scan_blocks,
    wet_cell_metrics,
)

#: Parity tolerances (gathered MPI vs serial, f64/f32) — the re-association
#: floor of the rank-local step + owned-masked reductions over a SMOKE
#: window (measured np=2 L2 x 2 steps: eta ~9e-9 f64); a real halo/partition
#: defect is orders above.  The floor grows with steps, hence the cap.
MPAS_OCEAN_PARITY_TOLS = {  # precision -> (rtol, atol)
    "float64": (1.0e-6, 1.0e-6),
    "float32": (1.0e-3, 1.0e-3),
}
PARITY_MAX_STEPS = 8
PARITY_FIELDS = ("eta", "T", "S")
#: Conservation drift tolerances over the timed window (volume: mean-eta
#: drift in METERS; heat/salt: relative).  Calibrated to the scheme's own
#: smoke-window floor (measured L2 x 4 steps f64: volume 2.4e-11 m,
#: heat 0, salt 1.2e-7 — identical with the serial model, i.e. SCHEME
#: drift, owned by the conservation suite, not this gate).  This gate
#: exists to catch DISTRIBUTED breakage — a halo double-count in the
#: reductions is O(halo/owned ~ 10%), five orders above the tolerance.
CONS_RTOL_DEFAULTS = {"float64": 1.0e-6, "float32": 1.0e-3}

#: Icosahedral subdivision levels this lane will consider for weak mode.
WEAK_LEVELS = (2, 3, 4, 5, 6, 7)


def _ncells(level: int) -> int:
    """Cells of the icosahedral bisection at ``level`` (10*4^L + 2)."""
    return 10 * 4 ** level + 2


def weak_level_for(cells_per_rank: int, n_ranks: int) -> int:
    """Subdivision level whose cells/rank best matches the target.

    Icosahedral meshes quantize by 4x per level, so exact weak scaling is
    impossible — pick the closest RATIO (mirrors run_cpu_mpi_scaling's
    ``_weak_resolution_ico``) and record the achieved value on the row.
    """
    best, best_ratio = WEAK_LEVELS[0], float("inf")
    for lv in WEAK_LEVELS:
        cpr = _ncells(lv) / n_ranks
        ratio = max(cpr / cells_per_rank, cells_per_rank / cpr)
        if ratio < best_ratio:
            best, best_ratio = lv, ratio
    return best


def reduce_block_times(all_block_ms, block_steps: int) -> dict:
    """Cross-rank per-block MAX reduction of fused-scan block times (M1).

    ``all_block_ms``: (n_ranks, n_blocks) — every rank's per-block wall
    times in ``comm.allgather`` order.  The parallel time of block ``b``
    is the SLOWEST rank in that block (a per-rank median gathered alone
    hides an alternating straggler); rank imbalance is the per-block
    max/median ratio.  Pure NumPy so the reduction is unit-testable
    without an MPI stack (codex finding 6).
    """
    arr = np.asarray(all_block_ms, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            "reduce_block_times expects (n_ranks, n_blocks), got shape "
            f"{arr.shape}")
    if block_steps < 1:
        raise ValueError(
            f"reduce_block_times needs block_steps >= 1, got {block_steps}")
    par = np.max(arr, axis=0)
    med = np.median(arr, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = np.where(med > 0.0, par / med, np.nan)
    return {
        "parallel_block_ms": [round(float(b), 2) for b in par],
        "rank_imbalance_per_block": [round(float(r), 4) for r in imb],
        "rank_imbalance": (round(float(np.nanmedian(imb)), 4)
                           if np.isfinite(imb).any() else float("nan")),
        "fused_step_ms": round(float(np.median(par)) / block_steps, 4),
    }


def stage_halo_note_for(n_ranks: int, halo_refresh: str):
    """Row-metadata companion to ``stage_halo_correct``.

    Must describe the ACTUAL refresh selection (codex finding 5): a
    ``--halo-refresh none`` row suffers ACROSS-STEP halo rot on top of
    the within-step staleness every multi-rank row has — labeling it
    "per-step packed refresh" would misdescribe the evidence.
    """
    if n_ranks == 1:
        return None
    doc = "docs/performance/scaling/mpas_ocean_distributed_stage_audit.md"
    if halo_refresh == "in_step":
        return ("per-step packed entry refresh + in-step stage-frontier "
                "refreshes (make_mpas_ocean_halo_refresh) — stage-correct; "
                "see " + doc)
    if halo_refresh == "per_step":
        return ("per-step packed refresh only; within-step staleness — "
                "see " + doc)
    return ("NO refresh: across-step halo rot (halo_refresh=none) on top "
            "of within-step staleness — see " + doc)


def build_global_problem(subdivision: int, nlev: int, seed: int = 0,
                         barotropic_solver: str = "explicit_substep",
                         pcg_variant: str = "standard",
                         n_barotropic_substeps: int = 10,
                         conservation_fixer: bool = True,
                         eta_floor_clamp_iters: int = 3):
    """Global mesh + z-coordinate + config + perturbed global IC.

    Deterministic and mesh-cache-backed, so every rank derives the
    IDENTICAL global problem before the partition is armed.
    """
    import jax.numpy as jnp

    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.mpas_config import MPASOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    mesh = create_voronoi_mesh(subdivision_level=subdivision)
    z_coord = create_ocean_z_star(
        n_levels=nlev, H_max=4000.0, dz_surface=20.0, dz_deep=400.0)
    config = MPASOceanConfig(
        A_h=1e3, K_h=1e2, A_v=1e-3, K_v=1e-4,
        n_barotropic_substeps=n_barotropic_substeps,
        eta_floor_clamp_iters=eta_floor_clamp_iters,
        # implicit_cn at n_ranks > 1 dispatches to the DISTRIBUTED fixed-M
        # PCG (halo-composed A_op + owned-masked dots) when the layout is
        # armed — the OMIP production barotropic path.  The model ctor
        # validates the literal (raises on unknown).
        barotropic_solver=barotropic_solver,
        # Reduction strategy inside the distributed fixed-M PCG.  At M=60
        # (the config default) "standard" costs 1 + 2*M batched allreduces
        # per implicit solve and "single_reduce" (Chronopoulos-Gear) costs
        # 1 + M -- 121 vs 61 latency-serialized global reductions, which is
        # what actually binds as the communicator reaches 256-512 ranks.
        # Not the default: it is a different (equivalent-in-exact-arithmetic)
        # recurrence, so it is opt-in and parity-gated, per the audit.
        barotropic_implicit_pcg_variant=pcg_variant,
        # Production-like conservation fixers: without them the explicit
        # subcycle's raw volume drift (~1e-4 over a smoke window) would
        # trip the gate — and their global reductions are exactly the
        # collective cost a scaling row should include.
        use_conservation_fixer=conservation_fixer,
        fix_volume=conservation_fixer, fix_heat=conservation_fixer,
        fix_salt=conservation_fixer,
    )
    state = rest_state_mpas_ocean(
        mesh, z_coord, T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=4000.0, land_lat_threshold=85.0)
    # Perturbation exercising advection/PGF/barotropic (the conservation
    # test's recipe): zonal-wavenumber eta + latitude-structured T.
    mask = state.land_mask.data
    T_pert = state.T.data + 0.5 * jnp.sin(
        4 * mesh.latCell)[:, None] * mask[:, None]
    eta_pert = state.eta.data + 0.01 * jnp.sin(3 * mesh.lonCell) * mask
    state = state._replace(
        T=state.T.replace(data=T_pert),
        eta=state.eta.replace(data=eta_pert))
    return mesh, z_coord, config, state


def slice_state_to_local(state, mesh, part):
    """Rank-local state: index every leaf by owned+halo cell/edge ids.

    Leaf classification by LEADING DIM against the global mesh (the same
    convention the PCG parity test uses): nCells -> ``local_cells``,
    nEdges -> ``local_edges``, anything else replicated (scalars, per-level
    reference arrays).
    """
    import jax
    import jax.numpy as jnp

    lc = np.asarray(part.local_cells)
    le = np.asarray(part.local_edges)
    nC, nE = int(mesh.nCells), int(mesh.nEdges)

    def _slice(x):
        a = np.asarray(x)
        if a.ndim >= 1 and a.shape[0] == nC:
            return jnp.asarray(a[lc])
        if a.ndim >= 1 and a.shape[0] == nE:
            return jnp.asarray(a[le])
        return x

    return jax.tree.map(_slice, state)


def gather_owned_cells(field_local, part, comm, n_global: int):
    """Gather a rank-local CELL field's OWNED entries to rank 0 (None on
    other ranks).  NaN-filled destination proves full coverage."""
    n_owned = int(part.n_owned_cells)
    ids = np.asarray(part.local_cells)[:n_owned]
    vals = np.asarray(field_local)[:n_owned]
    pieces = comm.gather((ids, vals), root=0)
    if comm.Get_rank() != 0:
        return None
    out = np.full((n_global,) + vals.shape[1:], np.nan, dtype=vals.dtype)
    for i, v in pieces:
        out[i] = v
    if np.isnan(out).any():
        raise RuntimeError("gather left unowned cells (partition coverage)")
    return out


def ocean_invariants(state, mesh, z_coord, config, layout=None, comm=None):
    """Global volume/heat/salt integrals, OWNED-cell-masked under MPI.

    Same integrals as the serial conservation suite (eta / T·h_k / S·h_k,
    area-weighted, via the shared ``compute_layer_thickness``), but summed
    over OWNED cells only and allreduced host-side (diagnostics-only, not
    differentiable, outside the timed loop) — a halo-including sum would
    double-count ghost cells.  Collective when ``comm`` is given: every
    rank must call it."""
    from legoesm.ocean.vertical import compute_layer_thickness

    eta = np.asarray(state.eta.data)
    H_bathy = np.asarray(state.H_bathy.data)
    mask = np.asarray(state.land_mask.data)
    area = np.asarray(mesh.areaCell)
    h_k = np.asarray(compute_layer_thickness(
        state.eta.data, state.H_bathy.data, z_coord,
        min_water_column_m=config.min_water_column_m))
    own = (np.asarray(layout.owned_mask_cells, dtype=np.float64)
           if layout is not None else np.ones_like(mask))
    w = own * mask * area
    vals = np.array([
        float((w * eta).sum()),
        float((w[:, None] * h_k * np.asarray(state.T.data)).sum()),
        float((w[:, None] * h_k * np.asarray(state.S.data)).sum()),
        float(w.sum()),
    ])
    _ = H_bathy  # consumed via compute_layer_thickness above
    if comm is not None:
        vals = comm.allreduce(vals)
    return {"volume": vals[0], "heat": vals[1], "salt": vals[2],
            "ocean_area": vals[3]}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subdivision", type=int, default=4,
                   help="Icosahedral level (strong mode; L4=2,562 cells, "
                        "L6=40,962, L8=655,362).")
    p.add_argument("--mode", choices=["strong", "weak"], default="strong")
    p.add_argument("--cells-per-rank", type=int, default=2562,
                   help="weak mode: target cells/rank (level quantized; "
                        "achieved value recorded).")
    p.add_argument("--nlev", type=int, default=10)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=2,
                   help="Steps excluded from the steady median (step 0 is "
                        "the JIT compile).")
    p.add_argument("--dt", type=float, default=60.0)
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="gpu: each MPI rank pins to ONE local GPU "
                        "(CUDA_VISIBLE_DEVICES=local-rank, "
                        "bench_ocean_mpi_scaling convention) BEFORE the "
                        "first JAX import; halos stay on mpi4jax "
                        "(route-A).")
    p.add_argument("--partition-method",
                   choices=["auto", "geometric", "metis", "sfc"],
                   default="auto")
    p.add_argument("--pcg-variant",
                   choices=["standard", "single_reduce"],
                   default="standard",
                   help="reduction strategy inside the distributed fixed-M "
                        "PCG (implicit_cn only). 'standard' costs 1+2M "
                        "batched allreduces per solve, 'single_reduce' "
                        "(Chronopoulos-Gear) costs 1+M -- at the M=60 "
                        "default that is 121 vs 61 latency-serialized "
                        "global reductions, the term that binds at "
                        "256-512 ranks. NOT a 2x step speedup: it also adds "
                        "one extra A_op (hence one extra halo exchange) in "
                        "the initialization, and only the reduction term is "
                        "removed. Requires n_ranks>1, implicit_cn and f64; "
                        "refused otherwise rather than silently ignored.")
    p.add_argument("--eta-floor-iters", type=int, default=3,
                   help="eta-floor clamp refinement iterations per call "
                        "(config eta_floor_clamp_iters). 2 calls x "
                        "n_substeps x iters batched allreduces per step: "
                        "the sync-count knob for the rank-count term. "
                        "1 keeps positivity (final maximum) but leaves "
                        "the redistribution un-refined (absorbed by the "
                        "step fixer). Scaling instrument only.")
    p.add_argument("--conservation-fixer", choices=["on", "off"],
                   default="on",
                   help="off removes ALL non-scan global reductions (3 "
                        "fixer reductions + the 2 fixer-gated unbatched "
                        "dHeat/dSalt sums, ocean_model_mpas.py:1026) — "
                        "the rank-count-term attribution knob for the "
                        "non-barotropic 60-70%% of the 32->128 delta. "
                        "Scaling instrument only; conservation gates "
                        "obviously void when off.")
    p.add_argument("--n-substeps", type=int, default=10,
                   help="Barotropic explicit_substep subcycle count "
                        "(config n_barotropic_substeps). The 10-substep "
                        "scan holds 30 of the step's 35 halo epochs and "
                        "60 of its 65 global reductions (2026-08-09 "
                        "census), so this is the rank-count-term "
                        "discriminator knob: halving it should halve the "
                        "32->128-rank cost delta IF the term is "
                        "barotropic-comm-bound. Physics validity of the "
                        "subcycle is NOT asserted off 10 — scaling "
                        "instrument only.")
    p.add_argument("--barotropic-solver",
                   choices=["explicit_substep", "implicit_cn"],
                   default="explicit_substep",
                   help="implicit_cn at n_ranks>1 runs the DISTRIBUTED "
                        "fixed-M PCG (halo-composed A_op, owned-masked "
                        "dots) and enables the post-run residual probe.")
    p.add_argument("--halo-refresh",
                   choices=["auto", "in_step", "per_step", "none"],
                   default="auto",
                   help="Halo refresh mode. 'in_step' (auto's choice at "
                        "n_ranks>1) = the per-step packed entry exchange "
                        "PLUS the in-step stage-frontier refreshes "
                        "(make_mpas_ocean_halo_refresh threaded through "
                        "model.step — the STAGE-CORRECT mode; rows earn "
                        "stage_halo_correct=true). 'per_step' = entry "
                        "exchange only (legacy: within-step staleness "
                        "remains). 'none' = no refresh at all (halos rot "
                        "across the window; timing omits exchange cost; "
                        "the zero-forcing residual probe still refreshes "
                        "ONCE before probing).  See the stage audit doc.")
    p.add_argument("--block-steps", type=int, default=8,
                   help="Steps per fused lax.scan timing block (M1 "
                        "contract; runs AFTER the gates). 0 disables the "
                        "fused measurement.")
    p.add_argument("--blocks", type=int, default=2,
                   help="Number of fused timing blocks.")
    p.add_argument("--probe-steps", type=int, default=3,
                   help="Individually-synced dispatch-latency probe steps "
                        "(reported separately, never mixed into the fused "
                        "number).")
    p.add_argument("--parity-gate", action="store_true",
                   help="Gathered-vs-serial gate (smoke windows only; the "
                        "re-association floor grows with steps).")
    p.add_argument("--check-conservation", action="store_true")
    p.add_argument("--cons-rtol", type=float, default=None)
    p.add_argument("--out", type=str,
                   default="results/a1/ocean_mpas_scaling.jsonl")
    args = p.parse_args()

    if args.steps < 1:
        raise SystemExit(f"--steps must be >= 1, got {args.steps}")
    if not (0 <= args.warmup < args.steps):
        raise SystemExit(
            f"--warmup must satisfy 0 <= warmup < steps "
            f"(got warmup={args.warmup}, steps={args.steps})")
    if args.block_steps < 0:
        raise SystemExit(
            f"--block-steps must be >= 0, got {args.block_steps}")
    if args.blocks < 1:
        raise SystemExit(f"--blocks must be >= 1, got {args.blocks}")
    if args.probe_steps < 0:
        raise SystemExit(
            f"--probe-steps must be >= 0, got {args.probe_steps}")

    if args.device == "gpu":
        # Pin BEFORE the first JAX import (sibling-bench convention: local
        # rank from the MPI launcher env; never SLURM_LOCALID on a
        # single-task step — the documented silent eff=0.5 bug).  An
        # EXISTING CUDA_VISIBLE_DEVICES (external wrapper pin, e.g. a PALS
        # shim exporting $PALS_LOCAL_RANKID) is respected — clobbering it
        # with "0" binds every rank to GPU 0, the exact contended-GPU fake
        # row this pin exists to prevent (codex).
        if os.environ.get("CUDA_VISIBLE_DEVICES"):
            pass  # external wrapper already pinned this rank — respect it
        else:
            local = (os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
                     or os.environ.get("MV2_COMM_WORLD_LOCAL_RANK")
                     or os.environ.get("PALS_LOCAL_RANKID"))
            if local is None:
                slid = os.environ.get("SLURM_LOCALID")
                nt = os.environ.get("SLURM_NTASKS", "1")
                if slid is not None and nt.isdigit() and int(nt) > 1:
                    local = slid
            os.environ["CUDA_VISIBLE_DEVICES"] = local or "0" 
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.precision == "float64":
        import jax

        jax.config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp  # noqa: F401  (post-x64 config)

    if args.device == "gpu" and jax.default_backend() not in (
            "gpu", "cuda", "rocm"):
        raise SystemExit(
            "--device gpu requested but the JAX backend is "
            f"{jax.default_backend()!r} — refusing to record a mislabeled "
            "GPU scaling row.")

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank, n_ranks = comm.Get_rank(), comm.Get_size()
    except Exception:
        comm, rank, n_ranks = None, 0, 1

    subdivision = args.subdivision
    if args.mode == "weak":
        subdivision = weak_level_for(args.cells_per_rank, n_ranks)

    if args.parity_gate and args.steps > PARITY_MAX_STEPS:
        raise SystemExit(
            f"--parity-gate is a smoke gate; --steps {args.steps} > "
            f"{PARITY_MAX_STEPS} cap.")

    if args.parity_gate and args.barotropic_solver == "implicit_cn" \
            and n_ranks > 1:
        # The parity SERIAL reference steps the GLOBAL mesh before the
        # partition layout is armed, and implicit_cn refuses a layout-less
        # multi-rank launch at the solver entry (the stock-CG mass
        # projection would silently run rank-local).  Refuse the
        # combination loudly instead of crashing mid-reference; PCG
        # parity is covered by
        # tests/ocean/distributed/test_barotropic_pcg_mpas_mpi.py and the
        # conservation gate remains available.
        raise SystemExit(
            "--parity-gate with --barotropic-solver implicit_cn at "
            "n_ranks > 1 is unsupported: the serial reference cannot be "
            "computed under the solver's layout-less multi-rank refusal. "
            "Drop --parity-gate (keep --check-conservation), or gate "
            "parity on the explicit_substep solver.")

    # Resolve --halo-refresh (dispatch-hardened: an explicit per_step
    # request that cannot be honored is a hard error, never a silent
    # no-op; auto degrades to none single-rank where there is no
    # partition to exchange over).
    if args.halo_refresh == "auto":
        halo_refresh = "in_step" if n_ranks > 1 else "none"
    elif args.halo_refresh in ("per_step", "in_step") and n_ranks == 1:
        raise SystemExit(
            f"--halo-refresh {args.halo_refresh} needs n_ranks > 1 (no "
            "partition layout exists single-rank); use 'auto' or 'none'.")
    else:
        halo_refresh = args.halo_refresh

    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel

    # --pcg-variant reaches the solver ONLY on the distributed implicit path
    # (ocean_model_mpas dispatches to barotropic_implicit_mpas for
    # implicit_cn, and that hands config.barotropic_implicit_pcg_variant to
    # solve_helmholtz_implicit only when a partition layout is armed;
    # single-rank takes the stock-CG branch instead).  Refuse the two
    # combinations where the flag would be silently inert rather than emit a
    # row whose solver label is not what ran (codex review).
    if args.pcg_variant != "standard":
        if args.barotropic_solver != "implicit_cn":
            raise SystemExit(
                f"--pcg-variant {args.pcg_variant} only affects the implicit "
                "barotropic solve, but --barotropic-solver is "
                f"{args.barotropic_solver!r}. Pass --barotropic-solver "
                "implicit_cn, or drop --pcg-variant (a silently ignored "
                "performance flag would make a scaling row unattributable).")
        # Precision before rank count, deliberately: precision is a static
        # property of the invocation, so this branch stays reachable (and
        # testable) on a login node at np=1, whereas an n_ranks-first order
        # would hide it behind an MPI-only path.
        if args.precision != "float64":
            raise SystemExit(
                f"--pcg-variant {args.pcg_variant} is refused at "
                f"--precision {args.precision}: Chronopoulos-Gear replaces "
                "the two independent dots with a reconstructed recurrence, "
                "which is less forgiving in floating point, and this repo "
                "has no f32 single_reduce validation gate. Run f64, or use "
                "--pcg-variant standard.")
        if n_ranks == 1:
            raise SystemExit(
                f"--pcg-variant {args.pcg_variant} needs n_ranks > 1: "
                "single-rank implicit_cn runs stock jax.scipy CG, not the "
                "distributed fixed-M PCG, so the variant would do nothing "
                "while the row claimed it. This flag is a >=256-rank lever; "
                "run it under mpirun/srun.")
    if args.check_conservation and args.conservation_fixer == "off":
        raise SystemExit("--check-conservation with --conservation-fixer off "
                         "cannot pass (the fixer IS the closure); refusing")
    mesh, z_coord, config, state_global = build_global_problem(
        subdivision, args.nlev, barotropic_solver=args.barotropic_solver,
        pcg_variant=args.pcg_variant,
        n_barotropic_substeps=args.n_substeps,
        conservation_fixer=(args.conservation_fixer == "on"),
        eta_floor_clamp_iters=args.eta_floor_iters)
    is_rank0 = rank == 0

    # Serial reference for the parity gate: EVERY rank, BEFORE arming MPI
    # (a rank-0-only reference traced after arming embeds collectives no
    # other rank matches — the band-MPI deadlock discipline).
    serial_ref = None
    if args.parity_gate:
        m_ser = MPASOceanModel(mesh, z_coord, config)
        s = state_global
        for _ in range(args.steps):
            s = m_ser.step(s, args.dt)
        jax.block_until_ready(jax.tree.leaves(s))
        serial_ref = {nm: np.asarray(getattr(s, nm).data)
                      for nm in PARITY_FIELDS}

    part_metrics = None
    if n_ranks > 1:
        from legoesm.parallel.voronoi_mpi import (
            initialize_voronoi_mpi,
            reduce_partition_metrics,
            voronoi_partition_metrics,
        )

        _r, _n, layout = initialize_voronoi_mpi(
            mesh, method=args.partition_method)
        # Anti-fake-scaling guard: multi-rank REQUIRES the armed partition
        # (initialize raises on failure; belt-and-braces assert here so a
        # future refactor can never fall back to a replicated global mesh
        # and report it as a scaling row).
        if layout is None or layout.partition is None:
            raise SystemExit(
                "voronoi partition layout not armed at n_ranks > 1 — "
                "refusing to time a replicated global-mesh run.")
        part = layout.partition
        model = MPASOceanModel(layout.local_mesh, z_coord, config)
        state = slice_state_to_local(state_global, mesh, part)
        part_metrics = reduce_partition_metrics(
            voronoi_partition_metrics(layout))
    else:
        model = MPASOceanModel(mesh, z_coord, config)
        state = state_global

    _layout = layout if n_ranks > 1 else None

    # ONE production step for both the gated per-step loop and the fused
    # scan blocks.  At n_ranks > 1 with halo_refresh, a PACKED full-state
    # exchange (one batched union-neighbor message per neighbor per dtype
    # group) follows each step so the NEXT step's input halos are fresh
    # and the timed number pays representative exchange cost.  On owned
    # cells the exchange is the identity, so the parity gate semantics
    # are unchanged.  Within-step staleness remains (stage audit doc).
    if halo_refresh in ("per_step", "in_step"):
        from legoesm.parallel.voronoi_mpi import (
            exchange_state_mpas_ocean,
            make_mpas_ocean_halo_refresh,
        )

        # 'in_step': arm the stage-frontier refreshes inside the step
        # (the stage-correctness lever) ON TOP of the per-step entry
        # exchange.  Build ONCE — the refresh callables are traced into
        # the jitted graph (and the fused scan) like any collective.
        _in_step_refresh = (make_mpas_ocean_halo_refresh(_layout)
                            if halo_refresh == "in_step" else None)

        # jit the COMPOSED step+exchange (one dispatch per step; the
        # sendrecv wrapper always runs traced, exactly as the atmosphere
        # step and the distributed PCG use it).  ``_step_impl`` avoids a
        # nested-JIT boundary inside this wrapper; timed_scan_blocks
        # re-traces the whole thing into the fused scan — same graph.
        @jax.jit
        def advance(st):
            return exchange_state_mpas_ocean(
                model._step_impl(st, args.dt,
                                 halo_refresh=_in_step_refresh),
                _layout)
    else:
        def advance(st):
            return model.step(st, args.dt)

    # Wet-cell weak metric (M1 audit item 9): MPAS land_mask is a
    # (nCells,) column mask; a wet column is wet at all nlev levels.
    # Multi-rank: per-rank OWNED wet columns (halo cells excluded —
    # owned cells partition the globe, so the allgathered list sums to
    # the global count exactly).
    _mask_np = np.asarray(state.land_mask.data)
    if n_ranks > 1:
        _own_np = np.asarray(layout.owned_mask_cells, dtype=np.float64)
        _wet_local = float((_mask_np * _own_np).sum())
        _wet_per_rank = comm.allgather(_wet_local)
    else:
        _wet_per_rank = [float(_mask_np.sum())]
    wet_rec = wet_cell_metrics(
        wet_columns=float(sum(_wet_per_rank)),
        nlev=args.nlev,
        total_cells=int(mesh.nCells) * args.nlev,
        n_devices=n_ranks,
        wet_columns_per_device=_wet_per_rank,
    )

    inv_before = None
    if args.check_conservation:
        inv_before = ocean_invariants(
            state, model.mesh, z_coord, config, layout=_layout, comm=comm)

    # Per-step timing with MPI barriers bracketing (route-A convention).
    # Feeds the gates (exact step count); the M1 fused-scan headline is
    # measured separately below.
    per_step_ms = []
    for _ in range(args.steps):
        if comm is not None:
            comm.Barrier()
        t0 = time.perf_counter()
        state = advance(state)
        jax.block_until_ready(jax.tree.leaves(state))
        if comm is not None:
            comm.Barrier()
        step_ms = (time.perf_counter() - t0) * 1e3
        if comm is not None:
            # Barrier-release skew means rank 0's local elapsed can
            # undercount the slowest rank — record the MAX (codex).
            from mpi4py import MPI as _MPI

            step_ms = comm.allreduce(step_ms, op=_MPI.MAX)
        per_step_ms.append(step_ms)

    # --- Correctness gates (before any timing is reported) -----------------
    if args.check_conservation:
        inv_after = ocean_invariants(
            state, model.mesh, z_coord, config, layout=_layout, comm=comm)
        tol = (args.cons_rtol if args.cons_rtol is not None
               else CONS_RTOL_DEFAULTS[args.precision])
        breach = False
        for k in ("volume", "heat", "salt"):
            if k == "volume":
                # Sigma(eta) of a wave IC is ~0, so a relative-to-itself
                # drift is degenerate — normalize by the ocean AREA
                # instead: mean-eta drift in METERS (the bench_ocean_mpi
                # convention).
                rel = (abs(inv_after[k] - inv_before[k])
                       / max(inv_before["ocean_area"], 1e-30))
                label = "mean-eta drift [m]"
            else:
                rel = (abs(inv_after[k] - inv_before[k])
                       / max(abs(inv_before[k]), 1e-30))
                label = "rel drift"
            if is_rank0:
                print(f"    conservation {k}: {label}={rel:.3e} "
                      f"(tol {tol:.1e})", flush=True)
            breach |= rel > tol
        if breach:
            if is_rank0:
                print("ERROR: conservation gate BREACHED.", flush=True)
            return 4

    if args.parity_gate:
        rtol, atol = MPAS_OCEAN_PARITY_TOLS[args.precision]
        ok = True
        for nm in PARITY_FIELDS:
            if n_ranks > 1:
                got = gather_owned_cells(
                    getattr(state, nm).data, part, comm, int(mesh.nCells))
            else:
                got = np.asarray(getattr(state, nm).data)
            if is_rank0:
                want = serial_ref[nm]
                field_ok = bool(np.allclose(got, want, rtol=rtol, atol=atol))
                ok &= field_ok
                mx = float(np.max(np.abs(got - want))) if want.size else 0.0
                print(f"    parity {nm:>4s}: max|diff|={mx:.3e} "
                      f"{'OK' if field_ok else 'MISMATCH'}", flush=True)
        if comm is not None:
            ok = comm.bcast(ok, root=0)
        if not ok:
            if is_rank0:
                print("ERROR: MPI parity gate MISMATCH vs the serial "
                      "reference.", flush=True)
            return 5

    # --- M1 fused-scan measurement (AFTER the gates; timing-only) ----------
    # The trustworthy production-like number: multi-step lax.scan blocks
    # with sync only AROUND the block (the per-step host-synced loop above
    # measures dispatch+sync latency — kept for the gates and legacy
    # comparability, never as the fused headline).  Route-A: mpi4jax
    # collectives inside the scan keep ranks in lockstep (the distributed
    # PCG already runs sendrecv inside fori_loop); block times are
    # allgathered so the parallel time of block b is the SLOWEST rank in
    # that block (the same slowest-rank convention as the per-step MAX).
    fused = None
    if args.block_steps > 0:
        state, _t = timed_scan_blocks(
            advance, state,
            block_steps=args.block_steps, n_blocks=args.blocks,
            probe_steps=args.probe_steps,
            sync_label="ocean_mpas_mpi_bench")
        if comm is not None and n_ranks > 1:
            _t.update(reduce_block_times(
                comm.allgather(np.asarray(_t["block_ms"],
                                          dtype=np.float64)),
                args.block_steps))
        fused = _t

    # --- Solver iterations + zero-forcing residual probe (M1) --------------
    # implicit_cn: distributed (layout armed) runs the fixed-M PCG whose
    # in-loop residual is not exposed; single-rank runs the stock adaptive
    # CG (count not exposed; tol/maxiter recorded).  The probe below is ONE
    # standalone free-surface solve at the FINAL state with zero slow
    # forcing, OUTSIDE any timed loop, via return_residual=True — the
    # owned-masked global relative Helmholtz residual of the returned eta.
    # Solver-HEALTH evidence, not the benchmarked solve's residual
    # (bench_ocean_latlon_spmd_scaling convention).  Collective at
    # n_ranks > 1: every rank calls it.
    zero_forcing_probe_residual = None
    zero_forcing_probe_measured = False
    if args.barotropic_solver == "implicit_cn":
        if n_ranks > 1:
            solver_iters = int(config.barotropic_implicit_pcg_fixed_iters)
            solver_iters_mode = (
                f"distributed_fixed_pcg[{config.barotropic_implicit_pcg_variant}]")
        else:
            solver_iters = None
            solver_iters_mode = (
                "adaptive_stock_cg(maxiter="
                f"{int(config.barotropic_implicit_pcg_maxiter)},"
                f"tol={config.barotropic_implicit_pcg_tol:g}) — iteration "
                "count not exposed by jax.scipy CG")
        residual_reason = (
            "in-loop residual not captured (fixed-iteration PCG exposes "
            "none in the hot path); zero_forcing_probe_residual is a "
            "standalone zero-slow-forcing solve at the final state — "
            "solver-health evidence, not the benchmarked solve's residual")
        from legoesm.ocean.dynamics.barotropic_implicit_mpas import (
            barotropic_implicit_mpas,
        )
        if n_ranks > 1 and halo_refresh != "per_step":
            # --halo-refresh none: the final state's edge/cell halos have
            # been stale for every preceding step, and the implicit
            # solver's predictor/RHS read u, eta and derived transports
            # BEFORE its in-loop cell exchange — a residual assembled
            # from rotten halos is not solver-health evidence.  One
            # packed refresh here (collective; OUTSIDE every timed loop)
            # so the probe measures a cleanly-assembled system (codex
            # finding 2).  Under per_step the last advance() already
            # ended with this exact exchange, so the probe input is
            # fresh on every path.
            from legoesm.parallel.voronoi_mpi import (
                exchange_state_mpas_ocean,
            )
            state = exchange_state_mpas_ocean(state, _layout)
        _probe_out = barotropic_implicit_mpas(
            state, model.mesh, z_coord, config, args.dt,
            return_residual=True)
        zero_forcing_probe_residual = float(
            jax.block_until_ready(_probe_out[3]))
        zero_forcing_probe_measured = True
    else:
        solver_iters = None
        solver_iters_mode = "explicit_substep (no iterative solve)"
        residual_reason = ("explicit_substep barotropic has no iterative "
                           "solve — no solver residual exists to measure")

    steady = per_step_ms[args.warmup:]
    gate_loop_med = float(np.median(steady))
    # M1 headline contract (mirrors bench_ocean_latlon_spmd_scaling): the
    # aggregator-facing ``steady_median_ms`` carries the FUSED per-step
    # number (cross-rank MAX-reduced per block); the host-synced gate-loop
    # median measures dispatch+sync latency and is demoted to the
    # explicitly-named ``step_latency_gate_loop_ms``.  Honest null when
    # the fused measurement is disabled (--block-steps 0) — a latency
    # number must never masquerade as fused throughput (codex finding 1).
    fused_step_ms = fused.get("fused_step_ms") if fused is not None else None
    rec = dict(
        component="ocean",
        grid="voronoi",
        mode=args.mode, subdivision=subdivision, n_ranks=n_ranks,
        n_cells=int(mesh.nCells), n_edges=int(mesh.nEdges), nlev=args.nlev,
        partition_method=args.partition_method,
        steps=args.steps, dt=args.dt,
        platform=jax.default_backend(),
        compile_ms=round(per_step_ms[0], 1),
        steady_median_ms=(round(fused_step_ms, 4)
                          if fused_step_ms is not None else None),
        step_latency_gate_loop_ms=round(gate_loop_med, 2),
        step_latency_gate_loop_min_ms=round(float(np.min(steady)), 2),
        per_step_ms=[round(x, 1) for x in per_step_ms],
        cells=int(mesh.nCells) * args.nlev,
        # HORIZONTAL cells/rank — the same unit as --cells-per-rank, so a
        # weak-mode row is comparable to its target (codex: the 3-D count
        # made rows look nlev-x larger).
        cells_per_rank_achieved=int(mesh.nCells) // n_ranks,
        # Fix 4 honesty flag: at np=1 the parity reference pre-runs the
        # SAME shape before the timed loop, so per_step_ms[0] may not
        # contain the real JIT compile.
        compile_prewarmed_by_parity_ref=bool(
            args.parity_gate and n_ranks == 1),
        # --- M1 lane fields (scaling-M3d increment-1) ---
        barotropic_solver=args.barotropic_solver,
        halo_refresh=halo_refresh,
        # 'in_step' rows run the stage-frontier refreshes inside the step
        # (audited insertion points R1-R3/T1-T2/B0-B2/I0-I1) on top of the
        # per-step entry exchange -> stage-correct.  'per_step'/'none'
        # multi-rank rows keep within-step staleness and stay FALSE.
        # Single-rank rows have no partition, hence trivially true.
        stage_halo_correct=bool(n_ranks == 1 or halo_refresh == "in_step"),
        stage_halo_note=stage_halo_note_for(n_ranks, halo_refresh),
        fused=fused,
        wet_cell=wet_rec,
        solver_iters=solver_iters,
        solver_iters_mode=solver_iters_mode,
        zero_forcing_probe_residual=zero_forcing_probe_residual,
        zero_forcing_probe_measured=zero_forcing_probe_measured,
        residual_reason=residual_reason,
    )
    rec["metadata"] = annotate_incomplete(scaling_metadata(
        grid="voronoi",
        component="ocean",
        resolution=f"L{subdivision}",
        n_levels=args.nlev,
        precision=args.precision,
        n_ranks=n_ranks,
        decomposition="cell_partition" if n_ranks > 1 else "none",
        # Route-A pin: a multi-node run that ALSO initialized
        # jax.distributed would auto-resolve to nccl/gloo and mislabel the
        # mpi4jax halo fabric (codex).
        transport=("mpi4jax" if n_ranks > 1 else None),
        n_gpus=(n_ranks if args.device == "gpu" else 0),
        # The PCG variant changes the per-solve allreduce COUNT, so two rows
        # that differ only in it are not comparable -- it belongs in the
        # solver identity, not buried in extra{} (the plot's own receipts
        # already lost the solver setting for the historic s7 row, which is
        # why its PCG attribution is only conditional).
        #
        # Suffix ONLY when the variant actually reached the solver: that is
        # the distributed implicit path.  Single-rank implicit_cn runs stock
        # CG, so labelling it with a PCG variant would be a false attribution
        # (codex review).  This also keeps the unsuffixed identity for every
        # pre-existing row, so historic receipts stay comparable.
        solver_variant=(
            f"mpas_ocean_{args.barotropic_solver}"
            + (f"_{args.pcg_variant}"
               if (args.barotropic_solver == "implicit_cn"
                   and n_ranks > 1
                   and args.pcg_variant != "standard")
               else "")),
        cells_per_rank=int(mesh.nCells) * args.nlev // n_ranks,
        scaling_kind=args.mode,
        partition_metrics=(dict(part_metrics) if part_metrics else None),
        extra={
            "partition_method": args.partition_method,
            "steps": args.steps,
            "warmup": args.warmup,
            "parity_gate": bool(args.parity_gate),
            "check_conservation": bool(args.check_conservation),
            "halo_refresh": halo_refresh,
            "barotropic_solver": args.barotropic_solver,
            "pcg_variant": args.pcg_variant,
            "n_barotropic_substeps": args.n_substeps,
            "conservation_fixer": args.conservation_fixer,
            "eta_floor_clamp_iters": args.eta_floor_iters,
            "block_steps": args.block_steps,
            "blocks": args.blocks,
            "probe_steps": args.probe_steps,
        },
    ))
    if is_rank0:
        _outdir = os.path.dirname(args.out)
        if _outdir:
            os.makedirs(_outdir, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec))
        # Foreground the FUSED headline (M1 contract); the gate-loop
        # median is explicitly labeled as latency, never the headline.
        _fused_txt = (
            f"{fused_step_ms}ms/step"
            f" (probe_latency={fused['step_latency_ms']}ms)"
            if fused_step_ms is not None
            else "n/a (--block-steps 0: fused measurement disabled)")
        print(f"[mpas-ocean np={n_ranks} L{subdivision} "
              f"nCells={mesh.nCells} nlev={args.nlev} "
              f"solver={args.barotropic_solver} halo={halo_refresh}] "
              f"compile={rec['compile_ms']}ms fused={_fused_txt} "
              f"gate_loop_latency={gate_loop_med:.2f}ms/step")
        if rec["metadata"]["virtual_cpu_devices"]:
            print("[virtual-cpu] forced host-platform CPU devices: this row "
                  "is a communication-overhead / correctness proxy, NOT "
                  "hardware scaling — do not report it as a speedup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
