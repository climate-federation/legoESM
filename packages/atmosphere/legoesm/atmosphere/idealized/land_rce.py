"""Idealized LAND-RCE slab-surface physics for the plane NH dycore.

Column-local surface physics shared by the RCEMIP-LAND drivers
(:mod:`scripts.run.run_rcemip_plane`, ``scripts/run/run_rce_mpi_long.py``).
Factored out of ``run_rcemip_plane.py`` per the CLAUDE.md "No duplicate
numerics" rule so a second (MPI) driver can reuse the exact same slab
surface-energy-balance + bulk-flux tendency numerics.

Every routine here is **column-local**: it operates independently per
``(ny, nx)`` horizontal cell (fluxes + slab update at the lowest model
level ``k = nlev - 1`` under top-down indexing) with no grid-global
reduction and no halo exchange. This is required so the same functions
run per-rank under an MPI horizontal domain decomposition — each rank
passes its LOCAL ``(ny_local, nx_local)`` fields and gets back
LOCAL tendencies / skin temperature.

Contents:

- :func:`land_surface_flux_tendencies` — SAM ``oceflx`` bulk surface-flux
  tendencies for a 2-D prognostic slab skin temperature with a fixed
  moisture-availability ``beta``.
- :func:`apply_plane_tendency_forward_euler` — host-side forward-Euler
  increment of a plane NH tendency onto the state.
- :func:`update_land_slab_temperature` — forward-Euler slab surface
  energy balance.
- :func:`sum_plane_tendencies` — field-wise sum of
  ``PlaneNonHydrostaticTendencies``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.state import PlaneNonHydrostaticTendencies


def land_surface_flux_tendencies(
    state, grid, height_coord, terrain_metric,
    T_sfc: jax.Array, p_sfc: float, beta: float, wd: float = 0.0,
):
    """SAM bulk surface-flux tendencies for prognostic slab land.

    This is the LAND-RCE counterpart of :func:`_make_surface_flux_physics`.
    It keeps the same SAM ``oceflx`` transfer coefficients, but takes a
    2-D prognostic skin temperature field and applies a fixed moisture
    availability ``beta`` to the potential latent-heat flux.
    """
    del grid, terrain_metric
    from legoesm.core.bulk_flux import (
        compute_sam_oceflx_fluxes, sam_ocean_surface_q,
    )

    ny, nx, nlev = state.theta_prime.data.shape
    n_tracers = state.tracers.data.shape[-1]
    T_sfc = jnp.asarray(T_sfc, dtype=state.theta_prime.data.dtype)
    if T_sfc.shape != (ny, nx):
        raise ValueError(
            f"land T_sfc shape {T_sfc.shape!r} must match horizontal grid "
            f"{(ny, nx)!r}"
        )
    beta = jnp.clip(jnp.asarray(beta, dtype=T_sfc.dtype), 0.0, 1.0)

    rho_0 = height_coord.rho_ref
    theta_0 = height_coord.theta_ref
    theta_total = theta_0 + state.theta_prime.data
    rho_total = rho_0 + state.rho_prime.data

    k_sfc = nlev - 1
    u_lo = state.u.data[..., k_sfc]
    v_lo = state.v.data[..., k_sfc]
    rho_lo = rho_total[..., k_sfc]
    theta_lo = theta_total[..., k_sfc]
    pi_sfc = height_coord.exner_ref[k_sfc]
    if n_tracers > 0:
        q_lo = state.tracers.data[..., k_sfc, 0]
    else:
        q_lo = jnp.zeros_like(theta_lo)
    z_bot = height_coord.z_full[k_sfc]

    q_sfc_sat = sam_ocean_surface_q(T_sfc, p_sfc, salt_factor=1.0)
    tau_x, tau_y, shflx, lhflx_potential, _ = compute_sam_oceflx_fluxes(
        u_atm=u_lo, v_atm=v_lo, theta_atm=theta_lo, q_atm=q_lo,
        T_sfc=T_sfc, q_sfc=q_sfc_sat,
        rho=rho_lo, z_bot=z_bot, exner_sfc=pi_sfc, wd=wd,
    )
    lhflx = beta * lhflx_potential

    dz_sfc = height_coord.dz[k_sfc]
    du_sfc = tau_x / (rho_lo * dz_sfc)
    dv_sfc = tau_y / (rho_lo * dz_sfc)
    du_dt_data = jnp.zeros_like(state.u.data).at[..., k_sfc].set(du_sfc)
    dv_dt_data = jnp.zeros_like(state.v.data).at[..., k_sfc].set(dv_sfc)

    dT_sfc_air = shflx / (rho_lo * constants.c_pd * dz_sfc)
    dtheta_sfc = dT_sfc_air / pi_sfc
    dtheta_p_data = jnp.zeros_like(
        state.theta_prime.data
    ).at[..., k_sfc].set(dtheta_sfc)

    dtracers_data = jnp.zeros_like(state.tracers.data)
    if n_tracers > 0:
        dq_sfc = lhflx / (rho_lo * constants.L_v * dz_sfc)
        dtracers_data = dtracers_data.at[..., k_sfc, 0].add(dq_sfc)

    tendencies = PlaneNonHydrostaticTendencies(
        du_dt=state.u.replace(data=du_dt_data),
        dv_dt=state.v.replace(data=dv_dt_data),
        dw_dt=state.w.replace(data=jnp.zeros_like(state.w.data)),
        dtheta_prime_dt=state.theta_prime.replace(data=dtheta_p_data),
        drho_prime_dt=state.rho_prime.replace(
            data=jnp.zeros_like(state.rho_prime.data)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        dtracers_dt=state.tracers.replace(data=dtracers_data),
    )
    diagnostics = {
        "tau_x": tau_x,
        "tau_y": tau_y,
        "shflx": shflx,
        "lhflx": lhflx,
        "lhflx_potential": lhflx_potential,
        "q_sfc_sat": q_sfc_sat,
        "beta": jnp.full_like(T_sfc, beta),
    }
    return tendencies, diagnostics


def apply_plane_tendency_forward_euler(state, tend, dt: float):
    """Apply a plane NH tendency as a host-side forward-Euler increment."""
    return state._replace(
        u=state.u.replace(data=state.u.data + dt * tend.du_dt.data),
        v=state.v.replace(data=state.v.data + dt * tend.dv_dt.data),
        w=state.w.replace(data=state.w.data + dt * tend.dw_dt.data),
        theta_prime=state.theta_prime.replace(
            data=state.theta_prime.data + dt * tend.dtheta_prime_dt.data),
        rho_prime=state.rho_prime.replace(
            data=state.rho_prime.data + dt * tend.drho_prime_dt.data),
        phis=state.phis.replace(data=state.phis.data + dt * tend.dphis_dt.data),
        tracers=state.tracers.replace(
            data=state.tracers.data + dt * tend.dtracers_dt.data),
    )


def update_land_slab_temperature(
    T_sfc: jax.Array,
    sw_down: jax.Array,
    lw_down: jax.Array,
    shflx: jax.Array,
    lhflx: jax.Array,
    dt: float,
    heat_capacity: float,
    albedo: float,
    emissivity: float = 1.0,
):
    """Forward-Euler slab surface energy balance for LAND-RCE.

    Positive turbulent fluxes are upward from the surface, so they cool the
    slab:

        C_slab dT_s/dt = SW_down (1 - alpha) + LW_down - eps sigma T_s^4
                         - SH - LH
    """
    from legoesm.core.surface_energy import surface_radiation_fluxes

    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        sw_down, lw_down, T_sfc, albedo, emissivity,
    )
    net = sw_net + lw_net - shflx - lhflx
    T_new = T_sfc + dt * net / heat_capacity
    diagnostics = {
        "sw_net": sw_net,
        "lw_net": lw_net,
        "lw_up": lw_up,
        "R_net": sw_net + lw_net,
        "Q_slab": net,
    }
    return T_new, diagnostics


def sum_plane_tendencies(*tendencies):
    """Sum a list of ``PlaneNonHydrostaticTendencies`` field-wise.

    Each input is a NamedTuple of ``Field`` leaves; the output is one
    NamedTuple whose ``.data`` arrays are the element-wise sum across
    all inputs. Field metadata (dims/units/name) comes from the FIRST
    tendency input. Raises ``ValueError`` on an empty input (caller
    bug: there's no canonical empty tendency).
    """
    if not tendencies:
        raise ValueError(
            "sum_plane_tendencies requires at least one tendency; "
            "got an empty argument list."
        )
    if len(tendencies) == 1:
        return tendencies[0]
    head = tendencies[0]
    # Codex review 2026-05-24: dims must match across all tendencies
    # so the sum is semantically well-defined; mismatched dims would
    # indicate a wiring bug (e.g., feeding NH cubed-sphere tendencies
    # into the plane composer).
    for t in tendencies[1:]:
        for fname in head._fields:
            if getattr(t, fname).dims != getattr(head, fname).dims:
                raise ValueError(
                    f"plane tendency dim mismatch on field {fname!r}: "
                    f"first={getattr(head, fname).dims!r} "
                    f"vs other={getattr(t, fname).dims!r}"
                )
    out = {}
    for fname in head._fields:
        head_f = getattr(head, fname)
        summed = sum(
            (getattr(t, fname).data for t in tendencies[1:]),
            start=head_f.data,
        )
        out[fname] = head_f.replace(data=summed)
    return PlaneNonHydrostaticTendencies(**out)
