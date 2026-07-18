"""Coupled atm + slab-ocean latitude-band MPI — the first coupled MPI step
(C10).  Gates that the band-decomposed coupled loop reproduces the serial
coupled loop AND conserves the coupled surface energy across ranks.

The atm C-grid dycore + a co-located slab SST are stepped on the SAME lat-lon
grid, SAME latitude-band decomposition; the sensible-heat exchange is pointwise
per cell and fully band-local (no cross-rank comm, no regrid).  Two truth-tier
checks, no external reference:

1. **serial == band-MPI parity** — the gathered coupled (atm state + SST) after
   N coupling intervals matches the serial single-process coupled loop on the
   same global state (a wrong band scatter/gather or a lat-dependent coupling
   bug would diverge).
2. **global energy conservation** — the area-weighted coupled surface energy
   (Σ area·(c_atm·T_sfc_air + c_ocean·SST)), allreduced across bands, drifts
   below round-off over the run (warm SST, no freezing clamp).

Run: mpirun -np {2,4} python -m pytest tests/distributed/test_coupled_latlon_mpi.py
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

mpi4jax = pytest.importorskip("mpi4jax")  # noqa: F841
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (  # noqa: E402
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.coupler.coupled_latlon_band import (  # noqa: E402
    CoupledSlabConfig,
    apply_surface_coupling,
    coupling_energy,
    make_coupled_latlon_band_step,
)
from legoesm.grids.halo import set_halo_backend  # noqa: E402
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.parallel.latlon_mpi import (  # noqa: E402
    gather_field_latlon,
    gather_state_latlon,
    make_latlon_band_layout,
    scatter_state_latlon,
    slice_latlon_grid_to_band,
)
from legoesm.parallel.reductions import global_sum_mpi  # noqa: E402

from legoesm import constants  # noqa: E402

_N_INTERVALS = 3
_N_ATM_SUB = 2
_DT = 100.0


def _global(grid, nlev, seed=31337):
    rng = np.random.default_rng(seed)
    n_lat, n_lon = grid.n_lat, grid.n_lon
    eps = 1.0e-3
    st = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((n_lat, n_lon + 1, nlev))),
        v=jnp.asarray(eps * rng.standard_normal((n_lat + 1, n_lon, nlev))),
        T=jnp.asarray(290.0 + eps * rng.standard_normal((n_lat, n_lon, nlev))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon))),
        phis=jnp.zeros((n_lat, n_lon)),
        tracers={})
    # Warm SST (no freezing clamp), spatially varying.
    sst = jnp.asarray(292.0 + 3.0 * rng.standard_normal((n_lat, n_lon)))
    return st, sst


def _cfg():
    return CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False,
        use_ppm_transport=False, time_integrator="ssp_rk3")


def test_coupled_band_matches_serial_and_conserves_energy():
    comm = MPI.COMM_WORLD
    rank, n = comm.Get_rank(), comm.Get_size()
    if n not in (2, 4):
        pytest.skip("run under mpirun -np {2,4}")

    grid = create_latlon_grid(
        n_lat=16, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=6)
    config = _cfg()
    cpl = CoupledSlabConfig()
    st_g, sst_g = _global(grid, sigma.n_levels)

    # --- Serial reference FIRST, on EVERY rank, LOCAL backend (deadlock-safe:
    #     a serial _step_cgrid traced after arming MPI embeds band collectives
    #     no other rank matches). ---
    set_halo_backend("local")
    ser_model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
    st_s, sst_s = st_g, sst_g
    for _ in range(_N_INTERVALS):
        for _ in range(_N_ATM_SUB):
            st_s, _ = ser_model._step_cgrid(st_s, _DT, physics_fn=None)
        st_s, sst_s, _ = apply_surface_coupling(
            st_s, sst_s, cpl, _DT * _N_ATM_SUB)
    st_s = jax.block_until_ready(st_s)
    sst_s = jax.block_until_ready(sst_s)

    # --- Band-MPI coupled loop ---
    layout = make_latlon_band_layout(rank, n, grid.n_lat, grid.n_lon)
    band_grid = slice_latlon_grid_to_band(grid, layout)
    band_model = CGridLatLonPrimitiveEquationModel(band_grid, sigma, config)
    st_l = scatter_state_latlon(st_g, layout)
    # SST is a cell field on the SAME grid — scatter by the same latitude band.
    s, e = layout.lat_start, layout.lat_end
    sst_l = sst_g[s:e]
    band_area = band_grid.area

    coupled_step = make_coupled_latlon_band_step(
        band_model, layout, cpl, n_atm_substeps=_N_ATM_SUB)

    # (2) Global coupled-surface-energy conservation of the EXCHANGE itself:
    #     bracket ONE apply_surface_coupling (dynamics EXCLUDED — the atm
    #     dycore freely changes T_sfc_air by advection, which is not a coupling
    #     non-conservation; only the exchange must conserve c_atm·T_air +
    #     c_ocean·SST).  Allreduce so the invariant is checked GLOBALLY across
    #     bands.  Warm SST -> no freezing clamp (the only physical sink).
    e_pre = global_sum_mpi(coupling_energy(st_l, sst_l, cpl, band_area))
    st_c, sst_c, _ = apply_surface_coupling(st_l, sst_l, cpl, _DT * _N_ATM_SUB)
    e_post = global_sum_mpi(coupling_energy(st_c, sst_c, cpl, band_area))
    drift = abs(float(e_post) - float(e_pre)) / abs(float(e_pre))
    assert drift < 1e-13, (
        f"coupled surface energy drift {drift:.3e} across ONE exchange — the "
        f"band-local sensible exchange is not conservative")

    # Run the full coupled loop (for the parity check below).
    for _ in range(_N_INTERVALS):
        st_l, sst_l = coupled_step(st_l, sst_l, _DT)

    # (1) serial == band-MPI parity, on rank 0.
    gathered = gather_state_latlon(st_l, layout)
    sst_gathered = gather_field_latlon(sst_l, layout)
    if rank == 0:
        for f in ("u", "v", "T", "p_s"):
            np.testing.assert_allclose(
                np.asarray(getattr(gathered, f)),
                np.asarray(getattr(st_s, f)),
                rtol=1e-11, atol=1e-11,
                err_msg=(f"coupled band-MPI '{f}' diverged from the serial "
                         f"coupled loop"))
        np.testing.assert_allclose(
            np.asarray(sst_gathered), np.asarray(sst_s),
            rtol=1e-11, atol=1e-11,
            err_msg="coupled band-MPI SST diverged from serial")
        print("COUPLED_LATLON_MPI_OK serial==band-MPI + energy-conserved",
              flush=True)
    set_halo_backend("local")
