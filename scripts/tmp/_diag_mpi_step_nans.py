#!/usr/bin/env python
"""Diagnostic: bisect where NaN first appears in the lat-lon MPI step.

Stages: input → pad → padded-tendency → padded-RK-step → strip → fixer.
Each stage's `(u, v, T, p_s)` is checked for finiteness; the first
non-finite stage is reported.  Run after the smoke test fails with
NaN so we know which phase to fix.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
    cgrid_latlon_hydrostatic_tendencies,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.parallel.latlon_mpi import (
    build_padded_grid,
    make_latlon_band_layout,
    pad_state_halos,
    strip_halos,
)


def _check(label, *fields):
    bad = []
    for name, arr in fields:
        a = np.asarray(arr)
        if not np.all(np.isfinite(a)):
            n_nan = int(np.sum(np.isnan(a)))
            n_inf = int(np.sum(np.isinf(a)))
            bad.append(f"{name}(shape={a.shape} nan={n_nan} inf={n_inf})")
    status = "OK" if not bad else f"BAD: {', '.join(bad)}"
    print(f"[{label}] {status}")
    return not bad


def main():
    n_lat, nlev = 16, 6
    halo = 2
    grid = create_latlon_grid(
        n_lat=n_lat, radius=constants.R_earth, omega=constants.Omega,
    )
    sigma = create_sigma_coordinate(n_levels=nlev)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False,
        use_ppm_transport=True, time_integrator="ssp_rk3",
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)

    rng = np.random.default_rng(31337)
    n_lon = grid.n_lon
    eps = 1.0e-3
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((n_lat, n_lon + 1, nlev))),
        v=jnp.asarray(eps * rng.standard_normal((n_lat + 1, n_lon, nlev))),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((n_lat, n_lon, nlev))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((n_lat, n_lon))),
        phis=jnp.zeros((n_lat, n_lon)),
    )

    _check("input",
           ("u", state.u), ("v", state.v),
           ("T", state.T), ("p_s", state.p_s))

    layout = make_latlon_band_layout(0, 1, n_lat, n_lon)
    padded_state = pad_state_halos(state, layout, halo=halo)
    _check("padded_state",
           ("u", padded_state.u), ("v", padded_state.v),
           ("T", padded_state.T), ("p_s", padded_state.p_s))

    padded_grid = build_padded_grid(grid, layout, halo=halo)
    print(f"[padded_grid] n_lat={padded_grid.n_lat} "
          f"dy.shape={padded_grid.dy.shape} "
          f"area.shape={padded_grid.area.shape} "
          f"cos_lat[:4]={np.asarray(padded_grid.cos_lat[:4])} "
          f"cos_lat[-4:]={np.asarray(padded_grid.cos_lat[-4:])}")

    # Tendencies on padded grid + state, dry PE.
    du, dv, dT, dps, _ = cgrid_latlon_hydrostatic_tendencies(
        padded_state, padded_grid, sigma, cfg,
    )
    _check("tendency_on_padded",
           ("du", du), ("dv", dv), ("dT", dT), ("dps", dps))

    # Step into the tendency to find WHICH sub-op first hits NaN/Inf.
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        _face_to_cell_u, _face_to_cell_v,
    )
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        absolute_vorticity_coriolis,
    )
    from legoesm.grids.vertical import (
        compute_geopotential, pressure_from_sigma,
        compute_pressure_velocity, vertical_advection,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        divergence_cgrid, gradient_x_cgrid, gradient_y_cgrid,
        interp_cell_to_uface, interp_cell_to_vface,
    )

    T2 = jnp.maximum(padded_state.T, cfg.T_min)
    p_s2 = jnp.clip(padded_state.p_s, cfg.p_floor, 2.0e6)
    p_full = pressure_from_sigma(sigma.sigma_full, p_s2)
    _check("p_full", ("p_full", p_full))

    Phi = compute_geopotential(T2, p_s2, sigma, padded_state.phis)
    _check("Phi", ("Phi", Phi))

    u_c = _face_to_cell_u(padded_state.u)
    v_c = _face_to_cell_v(padded_state.v)
    KE = 0.5 * (u_c**2 + v_c**2)
    B = Phi + KE
    _check("B", ("B", B))

    ln_ps = jnp.log(p_s2)
    _check("ln_ps", ("ln_ps", ln_ps))

    _Bln = jnp.concatenate([B, ln_ps[..., jnp.newaxis]], axis=-1)
    dBln_dx = gradient_x_cgrid(_Bln, padded_grid)
    _check("dBln_dx", ("dBln_dx", dBln_dx))

    dBln_dy = gradient_y_cgrid(_Bln, padded_grid)
    _check("dBln_dy", ("dBln_dy", dBln_dy))

    # Mass-flux continuity
    dsigma = sigma.dsigma
    dp_ = p_s2[..., jnp.newaxis] * dsigma
    dp_u = interp_cell_to_uface(dp_)
    dp_v = interp_cell_to_vface(dp_)
    div_dp = divergence_cgrid(dp_u * padded_state.u, dp_v * padded_state.v, padded_grid)
    _check("div_dp", ("div_dp", div_dp))

    cor_u, cor_v = absolute_vorticity_coriolis(
        padded_state.u, padded_state.v, padded_grid,
    )
    _check("coriolis", ("cor_u", cor_u), ("cor_v", cor_v))

    # Full padded step through the model (includes pole_v_bc + RK).
    padded_cfg = cfg._replace(
        fix_mass=False, zero_mean_ps_tendency=False,
        pole_v_bc=(True, True), pole_v_bc_offset=halo,
    )
    padded_model = CGridLatLonPrimitiveEquationModel(
        padded_grid, sigma, padded_cfg,
    )
    padded_new, _ = padded_model._step_cgrid(
        padded_state, 100.0, target_mass=None, physics_fn=None,
    )
    _check("padded_step",
           ("u", padded_new.u), ("v", padded_new.v),
           ("T", padded_new.T), ("p_s", padded_new.p_s))

    stripped = strip_halos(padded_new, layout, halo=halo)
    _check("stripped",
           ("u", stripped.u), ("v", stripped.v),
           ("T", stripped.T), ("p_s", stripped.p_s))


if __name__ == "__main__":
    main()
