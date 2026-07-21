"""Coupled atm + slab-ocean latitude-band MPI scaling bench (C10).

The first coupled-model scaling lane: the atm C-grid dycore + a co-located
slab SST on the SAME lat-lon grid, SAME latitude-band decomposition, coupled by
a band-local sensible-heat exchange (no cross-rank comm, no regrid — see
``legoesm.coupler.coupled_latlon_band``).  Because the coupling is explicit at
the interval boundary the atm step compiles ONCE, so the timed loop has no
per-interval recompile (unlike a naive SST-in-the-closure coupling).

Serial reference build (rank-local band, allreduced sphere area) mirrors the
component ocean/atm band benches; the standard weak/strong sweep is a thin
wrapper around ``run_coupled_bench`` (rows are steps/s per rank + the coupled
surface-energy drift of one exchange).  ``--parity-gate`` runs the correctness
check inline (needs >=2 ranks) and refuses to emit timings if it fails.

    mpirun -np 4 .venv/bin/python scripts/bench/bench_coupled_latlon_scaling.py \
        --n-lat 64 --nlev 10 --n-intervals 20 --n-atm-substeps 4
"""
from __future__ import annotations

import argparse
import time


def run_coupled_bench(*, n_lat, nlev, n_intervals, n_atm_substeps, dt,
                      n_warmup=2, rank=0, n_ranks=1):
    """Build + time the coupled band step; return a metrics dict.

    Single-process (n_ranks=1) runs the serial band-equivalent (the halo
    backend's single-member ring is the local wrap), so this is importable +
    smoke-testable without a launcher.  Returns steps/s + the one-exchange
    coupled-surface-energy drift + shapes.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState,
        CGridLatLonPrimitiveEquationConfig,
        CGridLatLonPrimitiveEquationModel,
    )
    from legoesm.coupler.coupled_latlon_band import (
        CoupledSlabConfig,
        apply_surface_coupling,
        coupling_energy,
        make_coupled_latlon_band_step,
    )
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout,
        scatter_state_latlon,
        slice_latlon_grid_to_band,
    )

    from legoesm import constants

    n_lon = 2 * n_lat
    grid = create_latlon_grid(
        n_lat=n_lat, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=nlev)
    config = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False,
        use_ppm_transport=False, time_integrator="ssp_rk3")
    cpl = CoupledSlabConfig()

    rng = np.random.default_rng(31337)
    eps = 1.0e-3
    st_g = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((n_lat, n_lon + 1, nlev))),
        v=jnp.asarray(eps * rng.standard_normal((n_lat + 1, n_lon, nlev))),
        T=jnp.asarray(290.0 + eps * rng.standard_normal((n_lat, n_lon, nlev))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon))),
        phis=jnp.zeros((n_lat, n_lon)), tracers={})
    sst_g = jnp.asarray(292.0 + 3.0 * rng.standard_normal((n_lat, n_lon)))

    layout = make_latlon_band_layout(rank, n_ranks, n_lat, n_lon)
    band_grid = slice_latlon_grid_to_band(grid, layout)
    band_model = CGridLatLonPrimitiveEquationModel(band_grid, sigma, config)
    st = scatter_state_latlon(st_g, layout)
    sst = sst_g[layout.lat_start:layout.lat_end]
    band_area = band_grid.area

    coupled_step = make_coupled_latlon_band_step(
        band_model, layout, cpl, n_atm_substeps=n_atm_substeps)

    # One-exchange coupled-surface-energy drift (the conservation diagnostic).
    from legoesm.parallel.reductions import global_sum_mpi

    def _e(_st, _sst):
        return float(global_sum_mpi(coupling_energy(_st, _sst, cpl, band_area)))
    e_pre = _e(st, sst)
    st_c, sst_c, _ = apply_surface_coupling(st, sst, cpl, dt * n_atm_substeps)
    energy_drift = abs(_e(st_c, sst_c) - e_pre) / abs(e_pre)

    def _block(pair):
        jax.block_until_ready((pair[0].T, pair[1]))

    for _ in range(n_warmup):
        st, sst = coupled_step(st, sst, dt)
    _block((st, sst))
    t0 = time.perf_counter()
    for _ in range(n_intervals):
        st, sst = coupled_step(st, sst, dt)
    _block((st, sst))
    elapsed = time.perf_counter() - t0

    return {
        "n_lat": n_lat, "n_lon": n_lon, "nlev": nlev, "n_ranks": n_ranks,
        "n_intervals": n_intervals, "n_atm_substeps": n_atm_substeps,
        "coupled_steps_per_s": n_intervals / elapsed if elapsed > 0 else 0.0,
        "energy_drift_one_exchange": energy_drift,
        "band_rows": int(layout.lat_end - layout.lat_start),
    }


def _parity_gate():
    """Inline correctness gate (needs >=2 MPI ranks): the band-MPI coupled loop
    must match the serial coupled loop.  Returns True/False; refuses timings on
    failure.  Delegates to the same logic as the pytest gate."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState,
        CGridLatLonPrimitiveEquationConfig,
        CGridLatLonPrimitiveEquationModel,
    )
    from legoesm.coupler.coupled_latlon_band import (
        CoupledSlabConfig,
        apply_surface_coupling,
        make_coupled_latlon_band_step,
    )
    from legoesm.grids.halo import set_halo_backend
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.parallel.latlon_mpi import (
        gather_field_latlon,
        gather_state_latlon,
        make_latlon_band_layout,
        scatter_state_latlon,
        slice_latlon_grid_to_band,
    )
    from mpi4py import MPI

    from legoesm import constants

    comm = MPI.COMM_WORLD
    rank, nproc = comm.Get_rank(), comm.Get_size()
    n_lat, nlev, dt, n_int, n_sub = 16, 6, 100.0, 3, 2
    n_lon = 2 * n_lat
    grid = create_latlon_grid(n_lat=n_lat, radius=constants.R_earth,
                              omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=nlev)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False, use_ppm_transport=False,
        time_integrator="ssp_rk3")
    cpl = CoupledSlabConfig()
    rng = np.random.default_rng(31337)
    st_g = CGridLatLonHydrostaticState(
        u=jnp.asarray(1e-3 * rng.standard_normal((n_lat, n_lon + 1, nlev))),
        v=jnp.asarray(1e-3 * rng.standard_normal((n_lat + 1, n_lon, nlev))),
        T=jnp.asarray(290.0 + 1e-3 * rng.standard_normal((n_lat, n_lon, nlev))),
        p_s=jnp.asarray(1e5 + 10.0 * rng.standard_normal((n_lat, n_lon))),
        phis=jnp.zeros((n_lat, n_lon)), tracers={})
    sst_g = jnp.asarray(292.0 + 3.0 * rng.standard_normal((n_lat, n_lon)))

    set_halo_backend("local")
    ser = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    st_s, sst_s = st_g, sst_g
    for _ in range(n_int):
        for _ in range(n_sub):
            st_s, _ = ser._step_cgrid(st_s, dt, physics_fn=None)
        st_s, sst_s, _ = apply_surface_coupling(st_s, sst_s, cpl, dt * n_sub)
    st_s = jax.block_until_ready(st_s)

    layout = make_latlon_band_layout(rank, nproc, n_lat, n_lon)
    bg = slice_latlon_grid_to_band(grid, layout)
    bm = CGridLatLonPrimitiveEquationModel(bg, sigma, cfg)
    st_l = scatter_state_latlon(st_g, layout)
    sst_l = sst_g[layout.lat_start:layout.lat_end]
    step = make_coupled_latlon_band_step(bm, layout, cpl, n_atm_substeps=n_sub)
    for _ in range(n_int):
        st_l, sst_l = step(st_l, sst_l, dt)
    gathered = gather_state_latlon(st_l, layout)
    sst_gathered = gather_field_latlon(sst_l, layout)
    ok = True
    if rank == 0:
        for f in ("u", "v", "T", "p_s"):
            ok = ok and bool(np.allclose(
                np.asarray(getattr(gathered, f)),
                np.asarray(getattr(st_s, f)), rtol=1e-10, atol=1e-10))
        # SST must match too — a defect corrupting the final SST while leaving
        # the atmosphere intact would otherwise pass the gate (audit P2).
        ok = ok and bool(np.allclose(
            np.asarray(sst_gathered), np.asarray(sst_s),
            rtol=1e-10, atol=1e-10))
    set_halo_backend("local")
    return bool(comm.bcast(ok, root=0))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-lat", type=int, default=32)
    p.add_argument("--nlev", type=int, default=6)
    p.add_argument("--n-intervals", type=int, default=20)
    p.add_argument("--n-atm-substeps", type=int, default=2)
    p.add_argument("--dt", type=float, default=100.0)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--parity-gate", action="store_true",
                   help="run the serial==band-MPI correctness gate first "
                        "(needs >=2 ranks); refuse timings on failure")
    args = p.parse_args()

    rank, n_ranks = 0, 1
    try:
        from mpi4py import MPI
        rank, n_ranks = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    except Exception:
        pass

    if args.parity_gate:
        if n_ranks < 2:
            if rank == 0:
                print("--parity-gate needs >=2 ranks; skipping gate")
        elif not _parity_gate():
            if rank == 0:
                print("PARITY GATE FAILED — refusing to emit timings")
            return 1
        elif rank == 0:
            print("parity gate PASS")

    m = run_coupled_bench(
        n_lat=args.n_lat, nlev=args.nlev, n_intervals=args.n_intervals,
        n_atm_substeps=args.n_atm_substeps, dt=args.dt,
        n_warmup=args.n_warmup, rank=rank, n_ranks=n_ranks)
    if rank == 0:
        print(f"[coupled] n_lat={m['n_lat']} nlev={m['nlev']} "
              f"n_ranks={m['n_ranks']} band_rows={m['band_rows']} "
              f"coupled_steps/s={m['coupled_steps_per_s']:.2f} "
              f"energy_drift={m['energy_drift_one_exchange']:.2e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
