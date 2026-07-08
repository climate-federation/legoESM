"""Ocean GPU scaling benchmark (single-device).

Sweeps latlon C-grid and MPAS Voronoi ocean models at increasing
resolution. Records per-step wall-clock and throughput. Reuses
TimingResult/write_csv/write_json from run_levante_gpu_scaling so
output is plottable by plot_scaling.py.

Run:
    PYTHONPATH=. .venv/bin/python scripts/bench_ocean_gpu_scaling.py \
        --output-dir results/scaling_gpu_ocean --no-timestamp
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_CUDA_GRAPH_FLAG = (
    "--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,CUBLAS,CUDNN"
)


def _has_xla_token(existing: str, token: str) -> bool:
    """Whitespace-tokenized flag check.

    Avoids substring false-positives like another flag value containing
    the token name.
    """
    for part in existing.split():
        # Each XLA flag looks like ``--name=value``
        name = part.split("=", 1)[0]
        if name == token:
            return True
    return False


def _enable_cuda_graphs():
    """Inject CUDA-graphs XLA flag if not already set or explicitly disabled.

    Must be called BEFORE any JAX import. Idempotent.
    """
    existing = os.environ.get("XLA_FLAGS", "")
    if _has_xla_token(existing, "--xla_gpu_enable_command_buffer"):
        return
    os.environ["XLA_FLAGS"] = f"{existing} {_CUDA_GRAPH_FLAG}".strip()


# Enable CUDA graphs unless caller pre-sets XLA_FLAGS or asks --no-cuda-graphs
# (parsed below). Done BEFORE the run_levante_gpu_scaling import to keep us
# ahead of any transitive JAX import. Fixes MPAS-ocean fp32 anomaly that
# bloated step time 2.7-2.9× without graphs (iter-6, 2026-05-26).
_disable_cg = "--no-cuda-graphs" in sys.argv
if not _disable_cg:
    _enable_cuda_graphs()

# Reuse atm-script helpers — must import before JAX so JAX_PLATFORMS sticks.
sys.path.insert(0, str(Path(__file__).parent))
from run_levante_gpu_scaling import (  # noqa: E402
    TimingResult, ScalingReport, write_csv, write_json,
    _configure_jax,
)

LATLON_RES = [32, 64, 96, 128, 192]      # n_lat; n_lon = 2*n_lat
MPAS_LEVELS = [4, 5, 6]                  # 10*4^L+2 cells

OCEAN_NLEV = 20
DT_BAROCLINIC = 600.0   # 10 min
N_WARMUP = 3
N_TIMING = 30


def _build_latlon(n_lat: int, dtype_x64: bool,
                  baro_solver: str = "explicit_substep"):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    n_lon = 2 * n_lat
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z = create_ocean_z_star(n_levels=OCEAN_NLEV)
    cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver=baro_solver)
    model = LatLonCGridOceanModel(grid, z, cfg)
    state = rest_state_latlon_cgrid_ocean(grid, z)
    n_cells = n_lat * n_lon
    return model, state, n_cells


def _build_mpas(level: int, dtype_x64: bool, baro_solver: str = "explicit_substep"):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.dynamics.ocean_model_mpas import (
        MPASOceanModel, MPASOceanConfig,
    )
    mesh = create_voronoi_mesh(subdivision_level=level, lloyd_iterations=5)
    z = create_ocean_z_star(n_levels=OCEAN_NLEV)
    cfg = MPASOceanConfig(barotropic_solver=baro_solver)
    model = MPASOceanModel(mesh, z, cfg)
    state = rest_state_mpas_ocean(mesh, z)
    n_cells = mesh.nCells
    return model, state, n_cells


def _block_state(state):
    """Force completion on every leaf array."""
    import jax
    leaves = jax.tree_util.tree_leaves(state)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _time_step(model, state, n_warmup: int, n_timing: int, dt: float):
    """Warmup + time model.step. Returns (compile_s, warmup_s, timing_s).

    Timed region uses ``jax.lax.scan`` to fuse N_TIMING steps into a single
    XLA program — eliminates Python host-dispatch overhead that dominates
    at small problem sizes (cf. atm scaling bench iter-5 fix).
    """
    import jax
    # First call = JIT compile + first warmup
    t0 = time.perf_counter()
    s = model.step(state, dt)
    _block_state(s)
    compile_s = time.perf_counter() - t0

    # Remaining warmup
    t0 = time.perf_counter()
    for _ in range(max(0, n_warmup - 1)):
        s = model.step(s, dt)
    _block_state(s)
    warmup_s = time.perf_counter() - t0

    # Fused scan over N_TIMING steps. Preserve dtypes (scan requires
    # carry types to match across iterations).
    input_dtypes = jax.tree.map(
        lambda x: x.dtype if hasattr(x, "dtype") else None, s,
    )
    dt_static = float(dt)

    @jax.jit
    def _scan_run(st):
        def _body(carry, _):
            new = model.step(carry, dt_static)
            new = jax.tree.map(
                lambda x, d: x.astype(d)
                if d is not None and hasattr(x, "astype") else x,
                new, input_dtypes,
            )
            return new, None
        return jax.lax.scan(_body, st, None, length=n_timing)[0]

    # Pre-compile against cloned leaves so the timed seed is untouched.
    _pre = jax.tree.map(lambda x: x, s)
    _pre_out = _scan_run(_pre)
    _block_state(_pre_out)

    t0 = time.perf_counter()
    s = _scan_run(s)
    _block_state(s)
    timing_s = time.perf_counter() - t0
    return compile_s, warmup_s, timing_s


def _bench_one(label: str, build_fn, key, prec: str,
               solver_tag: str = "default") -> TimingResult:
    import jax, jax.numpy as jnp
    model, state, n_cells = build_fn(key, prec == "float64")
    compile_s, warmup_s, timing_s = _time_step(
        model, state, N_WARMUP, N_TIMING, DT_BAROCLINIC,
    )
    # Stability sanity check: re-step from initial state once and verify
    # all prognostic leaves are finite. Catches solver/numerical blow-up
    # that would otherwise be reported as fast (NaN math is fast).
    sanity = model.step(state, DT_BAROCLINIC)
    leaves = jax.tree_util.tree_leaves(sanity)
    for leaf in leaves:
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            ok = bool(jnp.all(jnp.isfinite(leaf)))
            if not ok:
                raise RuntimeError(
                    f"{label}: non-finite values after 1 step "
                    f"(solver={solver_tag})"
                )
    step_wall_s = timing_s / N_TIMING
    ms = step_wall_s * 1000.0
    # SYPD = simulated-years per wall-clock-day. Atm bench formula
    # (run_levante_gpu_scaling.py:814,1281): (dt/step_wall_s) / (365.25*86400) * 86400.
    sypd = (DT_BAROCLINIC / step_wall_s) / (365.25 * 86400.0) * 86400.0
    mcells_per_s = (n_cells * OCEAN_NLEV) / step_wall_s / 1e6
    total = n_cells * OCEAN_NLEV
    return TimingResult(
        n_gpus=1, resolution=key, n_levels=OCEAN_NLEV,
        precision=prec, mode="ocean_strong",
        # Encode solver tag in physics_level (otherwise unused for ocean)
        # so CSVs from different solvers stay distinguishable when pooled.
        physics_level=f"baro={solver_tag}",
        dt_seconds=DT_BAROCLINIC, n_warmup=N_WARMUP, n_timing=N_TIMING,
        compile_time_s=compile_s, warmup_time_s=warmup_s,
        timing_time_s=timing_s, time_per_step_ms=ms, sypd=sypd,
        total_cells=total, cells_per_gpu=total,
        mcells_per_s=mcells_per_s, scaling_efficiency=1.0,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="results/scaling_gpu_ocean")
    p.add_argument("--no-timestamp", action="store_true")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--grids", default="latlon,mpas",
                   help="Comma list of {latlon,mpas}")
    p.add_argument("--latlon-res", default=None,
                   help="Override LATLON_RES (comma list)")
    p.add_argument("--mpas-levels", default=None,
                   help="Override MPAS_LEVELS (comma list)")
    p.add_argument("--mpas-baro-solver",
                   choices=["explicit_substep", "implicit_cn"],
                   default="implicit_cn",
                   help="MPAS barotropic solver. implicit_cn solves a "
                        "Crank-Nicolson free-surface system once per "
                        "baroclinic step; explicit_substep iterates "
                        "n_barotropic_substeps small steps (config-defined). "
                        "Default implicit_cn MATCHES production (run_omip.py) "
                        "and is ~2.5x faster on GPU than the 30-substep "
                        "explicit path, which serialises 30 barotropic updates "
                        "(+ per-substep allreduce) per baroclinic step.")
    p.add_argument("--ll-baro-solver",
                   choices=["explicit_substep", "implicit_cn"],
                   default="implicit_cn",
                   help="Lat-lon C-grid barotropic solver (same options "
                        "as --mpas-baro-solver).  Default implicit_cn matches "
                        "production (run_omip.py); ~1.3x faster than explicit.")
    p.add_argument("--no-cuda-graphs", action="store_true",
                   help="Disable XLA CUDA-graphs flag (default: enabled to "
                        "fix MPAS-ocean fp32 anomaly; recognised at import "
                        "time, this flag is purely for docs/help)")
    args = p.parse_args()

    _configure_jax(args.precision)
    import jax
    out_dir = Path(args.output_dir)
    if not args.no_timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend().upper()

    print(f"Backend: {backend}")
    print(f"Devices: {jax.devices()}")
    print(f"Precision: {args.precision}")
    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    ll_res = [int(x) for x in (args.latlon_res.split(",")
              if args.latlon_res else [str(r) for r in LATLON_RES])]
    mp_lev = [int(x) for x in (args.mpas_levels.split(",")
              if args.mpas_levels else [str(r) for r in MPAS_LEVELS])]

    results: list[TimingResult] = []

    failures: list[str] = []

    if "latlon" in grids:
        print(f"\nLatLon C-grid ocean — resolutions {ll_res}  "
              f"(baro={args.ll_baro_solver})")
        def _mk_latlon(n, x64, _solver=args.ll_baro_solver):
            return _build_latlon(n, x64, baro_solver=_solver)
        for n in ll_res:
            try:
                r = _bench_one(f"LL{n}", _mk_latlon, n, args.precision,
                               solver_tag=args.ll_baro_solver)
                print(f"  LL{n:4d} n_cells={r.total_cells:>10,}  "
                      f"compile={r.compile_time_s:6.2f}s  "
                      f"step={r.time_per_step_ms:7.2f}ms  "
                      f"throughput={r.mcells_per_s:6.1f}Mcells/s  "
                      f"SYPD={r.sypd:6.1f}")
                results.append(r)
            except Exception as exc:
                import traceback
                print(f"  LL{n}: FAILED {exc}", flush=True)
                traceback.print_exc()
                failures.append(f"latlon LL{n}: {exc}")

    if "mpas" in grids:
        print(f"\nMPAS Voronoi ocean — levels {mp_lev}  "
              f"(baro={args.mpas_baro_solver})")
        def _mk_mpas(level, x64, _solver=args.mpas_baro_solver):
            return _build_mpas(level, x64, baro_solver=_solver)
        for L in mp_lev:
            try:
                # NOTE: TimingResult.resolution is int (used as `:5d`). Tag
                # solver via physics_level field instead so the CSV is
                # self-describing without merging incompatible runs.
                r = _bench_one(f"I{L}", _mk_mpas, L, args.precision,
                               solver_tag=args.mpas_baro_solver)
                print(f"  I{L} n_cells={r.total_cells:>10,}  "
                      f"compile={r.compile_time_s:6.2f}s  "
                      f"step={r.time_per_step_ms:7.2f}ms  "
                      f"throughput={r.mcells_per_s:6.1f}Mcells/s  "
                      f"SYPD={r.sypd:6.1f}")
                results.append(r)
            except Exception as exc:
                import traceback
                print(f"  I{L}: FAILED {exc}", flush=True)
                traceback.print_exc()
                failures.append(f"mpas I{L}: {exc}")

    if not results:
        print("\nNo successful benchmark results — exiting nonzero.")
        return 1
    write_csv(results, out_dir / "ocean_scaling.csv")
    report = ScalingReport(
        mode="ocean_strong",
        precisions=[args.precision],
        results=results,
        backend=backend,
        hostname=os.environ.get("HOSTNAME", "unknown"),
    )
    # component="ocean" (write_json's default stamps "atmosphere");
    # scaling_kind="throughput": single-device size sweep — a saturation
    # curve, NOT device-count scaling (SCALING_STATUS_AUDIT.md tag (b)).
    write_json(
        report, out_dir / "ocean_scaling.json",
        component="ocean",
        metadata_overrides={"scaling_kind": "throughput"},
    )
    print(f"\nResults: {out_dir}/ocean_scaling.{{csv,json}}")
    if failures:
        print(f"\n{len(failures)} case(s) FAILED:")
        for fmsg in failures:
            print(f"  - {fmsg}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
