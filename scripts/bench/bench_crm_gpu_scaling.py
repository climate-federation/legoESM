"""Single-GPU scaling benchmark for the plane CRM (f-plane compressible Euler).

Mirrors `bench_plane_crm_dd_scaling.py` (MPI, R8) for single-GPU. Sweeps
horizontal grid sizes on the plane dycore + reports throughput, ms/step,
and Mcells/s. Reuses TimingResult / write_csv / write_json from the atm
GPU scaling script.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/bench_crm_gpu_scaling.py \\
        --nx 64 128 256 --nlev 30 --dx 2000.0 --dt 2.0 \\
        --output-dir results/scaling_crm_gpu --no-timestamp
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_levante_gpu_scaling import (  # noqa: E402
    TimingResult, ScalingReport, write_csv, write_json,
    _configure_jax,
)


N_WARMUP = 3
N_TIMING = 30


def _build_model(nx: int, ny: int, nlev: int, dx: float, dtype_x64: bool,
                 n_acoustic_substeps: int = 12,
                 semi_implicit_acoustic: bool = True,
                 substep_horizontal_acoustic: bool = True,
                 fix_mass: bool = False):
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        PlaneCompressibleEulerModel, make_flat_plane_terrain_metric,
        make_rest_state,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_stretched_height_coordinate

    dtype = jnp.float64 if dtype_x64 else jnp.float32
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx, dtype=dtype)
    hc = create_stretched_height_coordinate(nlev, H=20_000.0, dz_sfc=100.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        n_acoustic_substeps=n_acoustic_substeps,
        semi_implicit_acoustic=semi_implicit_acoustic,
        # Match the PRODUCTION CRM runners (run_rcemip/les/gate/lba_plane), which
        # split the HORIZONTAL acoustic terms onto the substeps. The vertical-only
        # default (False) is cheaper and under-represents production cost.
        substep_horizontal_acoustic=substep_horizontal_acoustic,
        sponge_coeff=0.05, sponge_width=5000.,
        hyperdiff_coeff=1e6, hyperdiff_rho_coeff=1e6, hyperdiff_w_coeff=1e6,
        smagorinsky_cs=0.0, use_coriolis=True,
        fix_mass=fix_mass, anchor_mass_to_initial=fix_mass,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = make_rest_state(grid, hc, dtype=dtype)
    # Tiny theta_prime kick so dycore has work.
    import jax
    rng = jax.random.PRNGKey(0)
    kick = 0.001 * jax.random.normal(rng, state.theta_prime.data.shape, dtype=dtype)
    state = state._replace(theta_prime=state.theta_prime.replace(data=kick))
    # Pre-cache target_mass so fix_mass=True is lax.scan-compatible
    # (iter-37 fix).
    if fix_mass:
        model.precompute_target_mass(state)
    return model, state, nx * ny


def _block_state(state):
    import jax
    for leaf in jax.tree_util.tree_leaves(state):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _time_step(model, state, dt: float, n_warmup: int, n_timing: int):
    import jax, jax.numpy as jnp
    t0 = time.perf_counter()
    s = model.step(state, dt)
    _block_state(s)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(max(0, n_warmup - 1)):
        s = model.step(s, dt)
    _block_state(s)
    warmup_s = time.perf_counter() - t0

    # Post-warmup finite assertion. Catches NaN/Inf that would otherwise
    # be silently reported as fast wall-clock numbers (codex iter-2 #1).
    for leaf in jax.tree.leaves(s):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            if not bool(jnp.all(jnp.isfinite(leaf))):
                raise RuntimeError(
                    f"NaN/Inf in state leaf after {n_warmup} warmup steps; "
                    f"check dt={dt}, n_acoustic_substeps, IC amplitude"
                )

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

    _pre = jax.tree.map(lambda x: x, s)
    _pre_out = _scan_run(_pre)
    _block_state(_pre_out)

    t0 = time.perf_counter()
    s = _scan_run(s)
    _block_state(s)
    timing_s = time.perf_counter() - t0

    # Post-timing finite assert — same as warmup. NaN math is fast.
    for leaf in jax.tree.leaves(s):
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
            if not bool(jnp.all(jnp.isfinite(leaf))):
                raise RuntimeError(
                    f"NaN/Inf after timing loop ({n_timing} steps)"
                )
    return compile_s, warmup_s, timing_s


def _acoustic_cfl(dt: float, n_acoustic_substeps: int, dx: float,
                  c_sound: float = 340.0) -> float:
    """Horizontal acoustic CFL = c_sound * (dt/nsub) / dx.

    Semi-implicit acoustic relaxes vertical CFL; horizontal still
    constrained by sound-speed Courant number. Stable when ≤ ~0.7.
    """
    return c_sound * (dt / n_acoustic_substeps) / dx


def _bench_one(nx: int, ny: int, nlev: int, dx: float, dt: float,
               prec: str, n_acoustic_substeps: int = 12,
               semi_implicit_acoustic: bool = True,
               substep_horizontal_acoustic: bool = True,
               allow_unsafe_cfl: bool = False,
               repeat: int = 1) -> TimingResult:
    cfl = _acoustic_cfl(dt, n_acoustic_substeps, dx)
    if cfl > 0.7:
        msg = (f"horizontal acoustic CFL = {cfl:.2f} (>0.7) "
               f"at dx={dx}m, dt={dt}s, nsub={n_acoustic_substeps}")
        if allow_unsafe_cfl:
            print(f"  WARN: {msg} — proceeding (--allow-unsafe-cfl)")
        else:
            raise SystemExit(f"REFUSE: {msg}. Pass --allow-unsafe-cfl to override.")
    elif cfl > 0.15:
        # Empirical iter-73 sweep: nsub=2 (CFL=0.17) NaN's, nsub=3
        # (CFL=0.113) stable. Warn between empirical bound and canonical 0.7.
        print(f"  WARN: horizontal acoustic CFL = {cfl:.3f} "
              f"is above the empirical stability bound (~0.15). "
              f"Consider increasing n_acoustic_substeps if NaN's appear.")
    if not semi_implicit_acoustic:
        # Compute actual dz_min from the height coord (codex iter-16 #8)
        import jax.numpy as jnp
        from legoesm.grids.vertical import create_stretched_height_coordinate
        _hc = create_stretched_height_coordinate(nlev, H=20_000.0, dz_sfc=100.0)
        dz_min = float(jnp.min(_hc.dz_half))
        cfl_v = 340.0 * (dt / n_acoustic_substeps) / dz_min
        if cfl_v > 0.5:
            msg = (f"vertical acoustic CFL = {cfl_v:.2f} (>0.5) "
                   f"at dz_sfc≈100m, dt_a={dt/n_acoustic_substeps:.3f}s")
            if allow_unsafe_cfl:
                print(f"  WARN: {msg} — proceeding (--allow-unsafe-cfl)")
            else:
                raise SystemExit(f"REFUSE: {msg}. Pass --allow-unsafe-cfl to override.")
    model, state, n_horiz = _build_model(
        nx, ny, nlev, dx, prec == "float64",
        n_acoustic_substeps=n_acoustic_substeps,
        semi_implicit_acoustic=semi_implicit_acoustic,
        substep_horizontal_acoustic=substep_horizontal_acoustic,
    )
    # Median of `repeat` timing runs reduces noise on sub-millisecond
    # cases where scan-amortization + cache warmth can dominate
    # (codex iter-9 #1).
    timings = []
    last_compile_s = last_warmup_s = 0.0
    for i in range(max(1, repeat)):
        c_s, w_s, t_s = _time_step(model, state, dt, N_WARMUP, N_TIMING)
        timings.append(t_s)
        if i == 0:
            last_compile_s, last_warmup_s = c_s, w_s
    timings.sort()
    timing_s = timings[len(timings) // 2]
    # Per-repeat spread (deferred MEDIUM #1/#2 codex review): (max-min)/median
    # quantifies measurement noise. Stable bench at spread<5%; noisy >15%.
    if len(timings) >= 2:
        spread_frac = (timings[-1] - timings[0]) / timing_s
    else:
        spread_frac = 0.0
    compile_s, warmup_s = last_compile_s, last_warmup_s
    step_wall_s = timing_s / N_TIMING
    ms = step_wall_s * 1000.0
    sypd = (dt / step_wall_s) / (365.25 * 86400.0) * 86400.0
    total = n_horiz * nlev
    mcells_per_s = total / step_wall_s / 1e6
    acoustic_tag = "si" if semi_implicit_acoustic else "exp"
    result = TimingResult(
        n_gpus=1, resolution=nx, n_levels=nlev,
        precision=prec, mode="crm_plane_strong",
        physics_level=(f"f-plane_dx{int(dx)}m_L{nlev}_"
                       f"nsub{n_acoustic_substeps}_{acoustic_tag}_dt{dt}"),
        dt_seconds=dt, n_warmup=N_WARMUP, n_timing=N_TIMING,
        compile_time_s=compile_s, warmup_time_s=warmup_s,
        timing_time_s=timing_s, time_per_step_ms=ms, sypd=sypd,
        total_cells=total, cells_per_gpu=total,
        mcells_per_s=mcells_per_s, scaling_efficiency=1.0,
    )
    # Attach per-repeat spread as a runtime attribute — TimingResult is
    # a shared NamedTuple (run_levante_gpu_scaling.py owner) so we don't
    # extend its schema for this bench-only diagnostic.
    object.__setattr__ if False else None  # noqa: E711 (NamedTuple frozen)
    # Use a side-channel dict keyed on id — simpler than schema change.
    _SPREAD_REGISTRY[id(result)] = spread_frac
    return result


_SPREAD_REGISTRY: dict = {}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, nargs="+", default=[48, 96, 192, 384],
                   help="Horizontal grid (assumed square, ny=nx)")
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=2000.0, help="meters")
    p.add_argument("--dt", type=float, default=2.0, help="seconds")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--n-acoustic-substeps", type=int, default=12,
                   help="Inner acoustic substep count per RK3 stage. "
                        "Default 12 (conservative). 4-6 typically stable "
                        "for short integrations; verify CFL for production.")
    p.add_argument("--explicit-acoustic", action="store_true",
                   help="Use explicit acoustic instead of semi-implicit. "
                        "Requires small dt (~0.5 s at dz_sfc=100m, nsub=4). "
                        "Empirically faster on consumer GPUs where fp64 ALU "
                        "is throttled — bypasses column-Thomas serial path.")
    p.add_argument("--vertical-only-acoustic", action="store_true",
                   help="Use the VERTICAL-ONLY semi-implicit acoustic substeps "
                        "(column-local). Default OFF ⇒ substep_horizontal_acoustic"
                        "=True, matching the production CRM runners "
                        "(run_rcemip/les/gate/lba_plane). The vertical-only mode is "
                        "cheaper and under-represents production cost.")
    p.add_argument("--allow-unsafe-cfl", action="store_true",
                   help="Continue even when horiz CFL >0.7 or vertical CFL "
                        ">0.5 (explicit). Default = refuse (raise SystemExit).")
    p.add_argument("--repeat", type=int, default=1,
                   help="Number of repeated timing runs per case (median "
                        "reported when >1). 3 recommended for fast cases <1 ms.")
    p.add_argument("--output-dir", default="results/scaling_crm_gpu")
    p.add_argument("--no-timestamp", action="store_true")
    p.add_argument("--check-tridiag-backends", action="store_true",
                   help="Re-run each --nx with LEGOESM_TRIDIAG forced to "
                        "pcr / cusparse / legacy and report a comparison "
                        "table. Useful for verifying the default choice on "
                        "a new GPU. Multiplies runtime by ~3x.")
    args = p.parse_args()

    _configure_jax(args.precision)
    import jax
    out_dir = Path(args.output_dir)
    if not args.no_timestamp:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = out_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {jax.default_backend().upper()}  Devices: {jax.devices()}")
    cfl = _acoustic_cfl(args.dt, args.n_acoustic_substeps, args.dx)
    acoustic = "explicit" if args.explicit_acoustic else "semi_implicit"
    horiz = ("vertical-only"
             if (args.vertical_only_acoustic or args.explicit_acoustic)
             else "si_horizontal(PROD)")
    print(f"Precision: {args.precision}  nlev: {args.nlev}  "
          f"dx: {args.dx} m  dt: {args.dt} s  "
          f"nsub: {args.n_acoustic_substeps}  acoustic: {acoustic}/{horiz}  "
          f"horiz CFL: {cfl:.3f}")

    results = []
    failures = []
    print(f"\nPlane CRM (f-plane) GPU sweep — nx={args.nx}")
    for n in args.nx:
        try:
            r = _bench_one(
                n, n, args.nlev, args.dx, args.dt, args.precision,
                n_acoustic_substeps=args.n_acoustic_substeps,
                semi_implicit_acoustic=not args.explicit_acoustic,
                # si_horizontal requires semi_implicit; explicit acoustic can only
                # run vertical-only. Forcing the flag True under --explicit-acoustic
                # would drop the horizontal PG/div from the slow tendency without a
                # substep to carry them (codex 2026-06-09).
                substep_horizontal_acoustic=(not args.vertical_only_acoustic
                                             and not args.explicit_acoustic),
                allow_unsafe_cfl=args.allow_unsafe_cfl,
                repeat=args.repeat,
            )
            # HBM utilization estimate — TWO bounds:
            #   - upper: 20 × dtype_bytes per cell-lev × passes (per-kernel
            #     full-state materialization, no L2 reuse)
            #   - lower: 6 × dtype_bytes per cell-lev × passes (only 3
            #     prognostic fields read+write per substep, full state per
            #     slow_tend), assuming aggressive XLA fusion + L2 reuse
            # True HBM traffic is between these. Report the upper bound
            # (matches the way roofline models typically count traffic) but
            # this is a relative-not-absolute indicator. Reference ceiling
            # 730 GB/s = consumer-mobile RTX 5090 sustained.
            dtype_bytes = 4 if args.precision == "float32" else 8
            passes_per_step = 3 * (1 + args.n_acoustic_substeps)
            bw_upper = (r.mcells_per_s * 1e6 * 20 * dtype_bytes
                        * passes_per_step / 1e9)
            bw_lower = (r.mcells_per_s * 1e6 * 6 * dtype_bytes
                        * passes_per_step / 1e9)
            # Display upper bound; if user wants lower-bound, see doc.
            bw_used = bw_upper
            hbm_pct = 100.0 * bw_used / 730.0
            spread = _SPREAD_REGISTRY.get(id(r), 0.0)
            spread_tag = f"spread={100*spread:4.1f}%" if args.repeat >= 2 else ""
            print(f"  N{n:>4d} cells={r.total_cells:>10,}  "
                  f"compile={r.compile_time_s:6.2f}s  "
                  f"step={r.time_per_step_ms:7.2f}ms  "
                  f"throughput={r.mcells_per_s:6.1f}Mcells/s  "
                  f"SYPD={r.sypd:8.2f}  "
                  f"HBM≈{bw_used:5.0f}GB/s({hbm_pct:4.0f}%)  "
                  f"{spread_tag}")
            results.append(r)
        except Exception as exc:
            import traceback
            print(f"  N{n}: FAILED {exc}", flush=True)
            traceback.print_exc()
            failures.append(f"N{n}: {exc}")

    if args.check_tridiag_backends and results:
        # Per cavecrew note: LEGOESM_TRIDIAG is read at trace time, so the
        # forced backend takes effect only for a freshly-launched process.
        # Spawn a clean subprocess for each backend to bypass JAX's JIT cache.
        import subprocess, sys, tempfile, shutil
        print("\n--check-tridiag-backends: rerunning each --nx with forced "
              "PCR / cuSPARSE / legacy backends...")
        header = f"  {'backend':10s}  {'N':>5s}  {'Mc/s':>7s}  {'step ms':>8s}  {'HBM%':>5s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        # Use a single auto-cleaned tempdir for all backend subprocess
        # outputs — they're not needed after the comparison print.
        tmpdir = tempfile.mkdtemp(prefix="bench_check_tridiag_")
        for backend in ("pcr", "cusparse", "legacy"):
            for nx in args.nx:
                env_extra = {"LEGOESM_TRIDIAG": backend}
                cmd = [
                    sys.executable, __file__,
                    "--nx", str(nx),
                    "--nlev", str(args.nlev),
                    "--dx", str(args.dx),
                    "--dt", str(args.dt),
                    "--precision", args.precision,
                    "--n-acoustic-substeps", str(args.n_acoustic_substeps),
                    "--repeat", str(args.repeat),
                    "--no-timestamp",
                    "--output-dir", f"{tmpdir}/{backend}_n{nx}",
                ]
                if args.explicit_acoustic:
                    cmd.append("--explicit-acoustic")
                if args.vertical_only_acoustic:
                    cmd.append("--vertical-only-acoustic")
                if args.allow_unsafe_cfl:
                    cmd.append("--allow-unsafe-cfl")
                env = {**os.environ, **env_extra}
                try:
                    out = subprocess.check_output(
                        cmd, env=env, stderr=subprocess.STDOUT,
                    ).decode()
                    for line in out.splitlines():
                        if f"N{nx:>4d} cells=" in line:
                            import re
                            m_mc = re.search(r"throughput=\s*([\d.]+)Mcells", line)
                            m_ms = re.search(r"step=\s*([\d.]+)ms", line)
                            m_hbm = re.search(r"\((\s*\d+)%\)", line)
                            if m_mc and m_ms:
                                hbm = (f"{m_hbm.group(1).strip()}%"
                                       if m_hbm else "  -")
                                print(f"  {backend:10s}  {nx:5d}  "
                                      f"{float(m_mc.group(1)):7.1f}  "
                                      f"{float(m_ms.group(1)):8.2f}  "
                                      f"{hbm:>5s}")
                            break
                except subprocess.CalledProcessError as exc:
                    # Surface the underlying failure reason (NaN/CFL/etc.)
                    # rather than just the exit code. Per cavecrew #3.
                    stdout = (exc.output or b"").decode(errors="replace")
                    last_err = "no output"
                    for line in stdout.splitlines():
                        if any(tag in line for tag in
                               ("NaN", "RuntimeError", "FAILED N", "Error:")):
                            last_err = line.strip()[:80]
                    print(f"  {backend:10s}  {nx:5d}  FAILED ({exc.returncode}): {last_err}")
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not results:
        print("\nNo successful runs — exiting nonzero.")
        return 1
    write_csv(results, out_dir / "crm_scaling.csv")
    write_json(ScalingReport(
        mode="crm_plane_strong", precisions=[args.precision], results=results,
        backend=jax.default_backend().upper(),
        hostname=os.environ.get("HOSTNAME", "unknown"),
    ), out_dir / "crm_scaling.json")
    print(f"\nResults: {out_dir}/crm_scaling.{{csv,json}}")
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
