"""Hydrostatic primitive equations written once over the grid operator contract.

One form per term, shared by the lat-lon C-grid and the MPAS / TRiSK mesh
(test-only pilot: the models still run their own tendencies)::

    dvel/dt = -grad(Phi + KE) - R_d T_face grad(L) + vorticity term
              + vertical momentum advection + dissipation
    dT/dt   = -div(T_face vel) + T div(vel)                 (horizontal)
              + theta-form vertical transport                (includes kappa T F / p)
              + kappa T omega_ps / (p + eps) + kappa T vel.grad(L)
    dp_s/dt = -sum_k div(dp_face vel) / (B or sigma range)  (raw continuity)

``L`` is the Simmons-Burridge (1981) full-level ``ln p_k`` on a hybrid
coordinate and ``ln p_s`` on sigma; the pressure gradient and ``vel.grad(L)``
use differences of the SAME discrete ``L``.  ``omega_ps`` is the surface-pressure
part of omega (``B_k dp_s/dt`` or ``sigma_k dp_s/dt``); the vertical-motion part
sits inside the theta-form vertical term, so it is not counted twice.  There is
no in-tendency temperature floor.

Grid-specific numerics are plug-ins (:class:`PEOperators`): face interpolation,
kinetic energy, the Coriolis / PV-flux term, ``vel.grad(L)`` (lat-lon averages
face products to cells; MPAS uses ``div(vel L_face) - L div(vel)``), the
optional lat-lon PPM temperature flux, damping, tracers, the surface-pressure
post-hook (lat-lon zero-mean, MPAS del-4) and the velocity boundary condition.
Lat-lon SPMD packed exchanges are not used: the plain operators are called.

Vertical form, an explicit required choice (``vertical_form``, no default):

* ``"sb81_theta"`` -- the form above (SB81 ``ln p_k`` PGF, theta-form vertical
  T, omega_ps split).  Not energy-conserving: the upwind theta transport
  creates energy (6-day JW drift +4.6e-4 lat-lon, +5.5e-4 MPAS).
* ``"as83"`` -- Arakawa-Suarez (1983), hybrid coordinate only: layer Exner
  ``pi_k``, PGF ``-grad(Phi + K) - c_p theta_f grad(pi)``, hydrostatic
  ``Phi_k - Phi_{k+1} = c_p theta_{k+1/2} (pi_{k+1} - pi_k)``, mass-weighted
  theta transport with the SAME face theta, centred vertical transports, and on
  lat-lon the TRiSK-form kinetic energy and an energy-neutral Coriolis term.
  The semi-discrete total energy is conserved to round-off.
  KNOWN DIFFERENCE (#1029): an isothermal atmosphere at rest over terrain is
  not exactly at rest -- the hybrid column quadrature is not isothermal-exact
  (measured on the DCMIP 2-0-0 mountain, hybrid L20: max |dv/dt| 9.1e-6 m/s^2
  MPAS, 1.3e-5 lat-lon; the ``"sb81_theta"`` form: 2e-16 / 2e-15).
* ``"as83_logmean"`` -- ``"as83"`` with the log-mean face theta (exact for
  isothermal pairs; same energy identity).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from legoesm.grids.operator_adapters import latlon_sw_grid_operators, mpas_neighbour_operators
from legoesm.grids.vertical import (
    VERTICAL_ADVECTION_SCHEMES,
    HybridSigmaPressureCoordinate,
    as83_geopotential,
    as83_interface_theta,
    as83_layer_exner,
    as83_logmean_face_weight,
    as83_vertical_theta_flux,
    compute_geopotential,
    compute_geopotential_hybrid,
    compute_mass_flux_from_cumsum,
    compute_sigma_dot_from_cumsum,
    dp_from_hybrid,
    pressure_from_hybrid,
    pressure_from_sigma,
    sb81_full_level_ln_p,
    vertical_advection,
    vertical_advection_hybrid,
    vertical_advection_hybrid_sb,
    vertical_advection_theta,
    vertical_advection_theta_hybrid,
)

from legoesm import constants

Edge = tuple[jax.Array, ...]
Ctx = dict[str, Any]

# Smooth 1/p protection of the explicit omega/p term (kept from the lat-lon core).
_P_EPS = 1e-10


class PEOperators(NamedTuple):
    """Grid operator contract plus the grid plug-ins around the shared core."""

    grid_ops: Any
    to_face: Callable[[jax.Array], Edge]          # cell (..., k) -> faces
    kinetic_energy: Callable[[Edge], jax.Array]
    vorticity_term: Callable[[Ctx, Edge], Edge]   # (ctx, dvel) -> dvel
    v_dot_grad_L: Callable[[Ctx], jax.Array]
    horizontal_T: Callable[[Ctx], jax.Array] | None   # None -> shared flux form
    dissipation: Callable[[Ctx, Edge, jax.Array], tuple[Edge, jax.Array]]
    tracers: Callable[[Ctx], Any]
    ps_post: Callable[[Ctx, jax.Array], jax.Array]
    boundary: Callable[[Edge], Edge]
    vert_scheme: str
    p_floor: float
    p_ceil: float
    vertical_form: str
    face_pairs: Callable[[jax.Array], tuple]      # cell -> ((f1, f2) per face family)


VERTICAL_FORMS = ("as83", "as83_logmean", "sb81_theta")


def _check_vertical_form(vertical_form: str) -> None:
    if vertical_form not in VERTICAL_FORMS:
        raise ValueError(f"unknown vertical_form {vertical_form!r}; expected one of "
                         f"{VERTICAL_FORMS}")


def _as_tuple(x: Any) -> tuple:
    return x if isinstance(x, tuple) else (x,)


def vertical_momentum_advection(vel: jax.Array, vv_face: jax.Array, ps_face: jax.Array,
                                sigma_coord: Any, hybrid: bool, scheme: str,
                                centred: bool = False) -> jax.Array:
    """Column vertical advection of one velocity family at its faces; ``centred``
    selects the Simmons-Burridge centred form (conserves kinetic energy)."""
    if hybrid:
        if centred:
            return vertical_advection_hybrid_sb(vel, vv_face, ps_face, sigma_coord)
        return vertical_advection_hybrid(vel, vv_face, ps_face, sigma_coord)
    return vertical_advection(vel, vv_face, sigma_coord, scheme=scheme)


def primitive_equation_tendencies(
    T: jax.Array, vel: Edge, p_s: jax.Array, phis: jax.Array,
    sigma_coord: Any, ops: PEOperators,
) -> tuple[Edge, jax.Array, jax.Array, Any, Ctx]:
    """``(dvel, dT, dp_s, tracer_tends, ctx)`` for any grid; ``vel`` is a tuple of
    face families with the level axis last."""
    R_d, kappa = constants.R_d, constants.kappa
    hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)
    _check_vertical_form(ops.vertical_form)
    as83 = ops.vertical_form != "sb81_theta"
    if as83 and not hybrid:
        raise ValueError("the Arakawa-Suarez vertical form needs a hybrid coordinate")
    gops = ops.grid_ops
    p_s = jnp.clip(p_s, ops.p_floor, ops.p_ceil)
    if hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)
        if as83:   # Arakawa-Suarez (1983): Exner pi replaces ln p, c_p theta replaces R T
            P_half, pi, dpi_dps = as83_layer_exner(sigma_coord, p_s)
            theta = T / pi
            theta_half = as83_interface_theta(theta, P_half, pi)
            Phi = as83_geopotential(theta, theta_half, P_half, pi, phis)
            L = pi
        else:
            Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
            L = sb81_full_level_ln_p(sigma_coord, p_s)            # (..., nlev)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        dp = p_s[..., None] * sigma_coord.dsigma.astype(p_s.dtype)
        Phi = compute_geopotential(T, p_s, sigma_coord, phis)
        L = jnp.log(p_s)[..., None]                           # (..., 1)
    nlev = T.shape[-1]

    # Pressure gradient: grad(Phi + KE) + R_d T_face grad(L), one batched gradient.
    B = Phi + ops.kinetic_energy(vel)
    grads = _as_tuple(gops.gradient(jnp.concatenate([B, L], axis=-1)))
    dB = tuple(g[..., :nlev] for g in grads)
    dL = tuple(g[..., nlev:] for g in grads)
    T_face = ops.to_face(T)
    if as83:
        if ops.vertical_form == "as83_logmean":
            theta_f = tuple(
                as83_logmean_face_weight(p1, p2) * (t1 - t2) + t2
                for (t1, t2), (p1, p2) in zip(ops.face_pairs(theta), ops.face_pairs(pi)))
        else:
            theta_f = ops.to_face(theta)
    # PGF coefficient on faces: R_d T_f (ln p form) or c_p theta_f (Exner form)
    pgf_coef = (tuple(constants.c_pd * f for f in theta_f) if as83
                else tuple(R_d * f for f in T_face))
    dvel = tuple(-(db + c * dl) for db, c, dl in zip(dB, pgf_coef, dL))

    dp_face = ops.to_face(dp)
    ctx: Ctx = dict(T=T, vel=vel, p_s=p_s, p_full=p_full, dp=dp, dp_face=dp_face,
                    L=L, dL=dL, T_face=T_face, pgf_coef=pgf_coef, hybrid=hybrid, sigma_coord=sigma_coord,
                    Phi=Phi)
    # Each additive term, for budget attribution (unused outputs are dead code under jit).
    terms: Ctx = {"pgf": dvel}
    dvel_in = dvel
    dvel = ops.vorticity_term(ctx, dvel)
    terms["coriolis"] = tuple(a - b for a, b in zip(dvel, dvel_in))

    # Raw continuity (no global correction here).
    div_dp = gops.divergence(*(df * v for df, v in zip(dp_face, vel)))
    cumsum = jnp.cumsum(div_dp, axis=-1)
    D = cumsum[..., -1:]
    if hybrid:
        dps = -D[..., 0] / sigma_coord.B_range
        vv = compute_mass_flux_from_cumsum(cumsum, D, sigma_coord)
        vert_T = None if as83 else vertical_advection_theta_hybrid(
            T, vv, p_s, sigma_coord, conservative=(ops.vert_scheme == "sb"))
        omega_ps = sigma_coord.B_full * dps[..., None]
    else:
        dps = -D[..., 0] / (1.0 - sigma_coord.sigma_half[0])
        vv = compute_sigma_dot_from_cumsum(cumsum, D, p_s, sigma_coord)
        vert_T = vertical_advection_theta(T, vv, p_s, sigma_coord, scheme=ops.vert_scheme)
        omega_ps = sigma_coord.sigma_full * dps[..., None]
    ctx.update(div_dp=div_dp, vert_vel=vv, dps_raw=dps)

    vv_face = ops.to_face(vv)
    ps_face = tuple(f[..., 0] for f in ops.to_face(p_s[..., None]))
    terms["vert_mom"] = tuple(
        vertical_momentum_advection(v, vf, pf, sigma_coord, hybrid, ops.vert_scheme, centred=as83)
        for v, vf, pf in zip(vel, vv_face, ps_face))
    dvel = tuple(d + a for d, a in zip(dvel, terms["vert_mom"]))

    div_u = gops.divergence(*vel)
    ctx["div_u"] = div_u
    if as83:   # mass-weighted theta transport, same face theta as the PGF
        flux = gops.divergence(*(v * d * t for v, d, t in zip(vel, dp_face, theta_f)))
        horiz_T = pi * (-flux + theta * div_dp) / dp
        vert_T = pi * as83_vertical_theta_flux(theta, theta_half, vv) / dp
        conv_ps = theta * dpi_dps * dps[..., None]
        conv_vgl = jnp.zeros_like(T)
    else:
        if ops.horizontal_T is None:
            horiz_T = -gops.divergence(*(tf * v for tf, v in zip(T_face, vel))) + T * div_u
        else:
            horiz_T = ops.horizontal_T(ctx)
        conv_ps = kappa * T * omega_ps / (p_full + _P_EPS)
        conv_vgl = kappa * T * ops.v_dot_grad_L(ctx)
    dT = horiz_T + vert_T + (conv_ps + conv_vgl)
    terms.update(horiz_T=horiz_T, vert_T=vert_T, conv_ps=conv_ps, conv_vgl=conv_vgl,
                 dps=dps)
    ctx["terms"] = terms

    dvel, dT = ops.dissipation(ctx, dvel, dT)
    tracer_tends = ops.tracers(ctx)
    dps = ops.ps_post(ctx, dps)
    return ops.boundary(dvel), dT, dps, tracer_tends, ctx


# ============================================================================
# Lat-lon C-grid plug-ins
# ============================================================================

def latlon_face_metrics(grid: Any, gops: Any) -> tuple[jax.Array, ...]:
    """``(l_u, d_u, l_v, d_v)``: face length and centre distance of the u faces
    ``(n_lat,)`` and v faces ``(n_lat+1,)``, read off the divergence and gradient
    operators themselves so ``A_f = l_f d_f`` is exactly their summation-by-parts
    weight.  Pole v faces have ``l_v = d_v = 0``."""
    nj, ni = grid.n_lat, grid.n_lon
    area = jnp.asarray(grid.area)[:, 0]
    col = jnp.broadcast_to(jnp.arange(ni + 1.0), (nj, ni + 1))
    l_u = area * gops.divergence(col, jnp.zeros((nj + 1, ni)))[:, 0]
    d_u = 1.0 / gops.gradient(jnp.broadcast_to(jnp.arange(ni * 1.0), (nj, ni)))[0][:, 1]
    div_v = gops.divergence(jnp.zeros((nj, ni + 1)), jnp.ones((nj + 1, ni)))[:, 0]
    l_v = jnp.concatenate([jnp.zeros(1), jnp.cumsum(area * div_v)])
    g = gops.gradient(jnp.broadcast_to(jnp.arange(nj * 1.0)[:, None], (nj, ni)))[1][:, 0]
    d_v = jnp.where(g != 0, 1.0 / jnp.where(g != 0, g, 1.0), 0.0)
    l_v = l_v.at[0].set(0.0).at[-1].set(0.0)
    return l_u, d_u, l_v, d_v


def latlon_pe_operators(grid: Any, config: Any, tracers: Any = None, dt: Any = None, *,
                        vertical_form: str) -> PEOperators:
    """Lat-lon C-grid plug-ins (fields ``(n_lat, n_lon(+1), nlev)``).  The
    Arakawa-Suarez forms use the TRiSK-form kinetic energy and the
    energy-neutral Coriolis term; ``"sb81_theta"`` keeps the current ones."""
    _check_vertical_form(vertical_form)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        latlon_dissipation,
        latlon_pole_bc,
        latlon_tracer_tendencies,
    )
    from legoesm.atmosphere.dynamics.gcm.shallow_water_generic import (
        cell_winds_cgrid,
        kinetic_energy_cgrid,
    )
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        absolute_vorticity_coriolis,
    )
    from legoesm.core.conservation import zero_mean_tendency
    from legoesm.core.operators_fv_latlon_3d import cgrid_fv_scalar_advection_latlon_3d
    from legoesm.grids.operators_latlon_cgrid import (
        interp_cell_to_uface,
        interp_cell_to_vface_halo,
    )

    gops_ll = latlon_sw_grid_operators(grid)
    ni = grid.n_lon
    west = (jnp.arange(ni + 1) - 1) % ni      # cell west of u face i (periodic)
    east = jnp.arange(ni + 1) % ni            # cell east of u face i

    def to_face(f):
        return (interp_cell_to_uface(f), interp_cell_to_vface_halo(f))

    def face_pairs(f):
        # pole v faces pair a row with itself; their v tendency is zeroed by the wall
        rows = jnp.arange(grid.n_lat + 1)
        south, north = jnp.clip(rows - 1, 0, grid.n_lat - 1), jnp.clip(rows, 0, grid.n_lat - 1)
        return ((f[:, west], f[:, east]), (f[south], f[north]))

    def vorticity_term_current(ctx, dvel):
        cor_u, cor_v = absolute_vorticity_coriolis(ctx["vel"][0], ctx["vel"][1], grid)
        return (dvel[0] + cor_u, dvel[1] + cor_v)

    l_u, d_u, l_v, d_v = latlon_face_metrics(grid, gops_ll)
    A_u, A_v = l_u * d_u, l_v * d_v
    area = jnp.asarray(grid.area)
    inv_dv = jnp.where(d_v > 0, 1.0 / jnp.where(d_v > 0, d_v, 1.0), 0.0)
    lat = jnp.asarray(grid.lat)
    two_omega = 2.0 * grid.omega
    f_vert = jnp.pad(two_omega * jnp.sin(0.5 * (lat[:-1] + lat[1:])), (1, 1),
                     constant_values=(-two_omega, two_omega))     # (n_lat+1,)

    def kinetic_energy_dual(vel):
        """K_c = sum_f A_f u_f^2 / (4 A_c) (the TRiSK form)."""
        u, v = vel
        ku = A_u[:, None, None] * (u[:, :-1] ** 2 + u[:, 1:] ** 2)
        kv = A_v[:-1, None, None] * v[:-1] ** 2 + A_v[1:, None, None] * v[1:] ** 2
        return (ku + kv) / (4.0 * area[..., None])

    def vorticity_term_energy(ctx, dvel):
        """Vertex-PV Coriolis on mass transports U = l dp u, V = l dp v:
        cor_u = (1/4 d_u) sum Q V, cor_v = -(1/4 d_v) sum Q U over the four
        (face, vertex) pairs -> zero kinetic-energy work for any state."""
        u, v = ctx["vel"]
        dp_u, dp_v = ctx["dp_face"]
        dp = ctx["dp"]
        U = l_u[:, None, None] * dp_u * u
        V = l_v[:, None, None] * dp_v * v
        zeta = gops_ll.vorticity(u, v)                            # (n_lat+1, n_lon+1, nlev)
        dpx = 0.5 * (dp[:, west] + dp[:, east])                   # (n_lat, n_lon+1, nlev)
        dpv = jnp.concatenate([dpx[:1], 0.5 * (dpx[:-1] + dpx[1:]), dpx[-1:]], axis=0)
        Q = (zeta + f_vert[:, None, None]) / dpv
        sV = V[:, west] + V[:, east]                              # (n_lat+1, n_lon+1, nlev)
        cor_u = (Q[:-1] * sV[:-1] + Q[1:] * sV[1:]) / (4.0 * d_u[:, None, None])
        Up = jnp.pad(U, ((1, 1), (0, 0), (0, 0)))
        sU = Up[:-1] + Up[1:]                                     # (n_lat+1, n_lon+1, nlev)
        cor_v = -(Q[:, :-1] * sU[:, :-1] + Q[:, 1:] * sU[:, 1:]) * (inv_dv[:, None, None] / 4.0)
        return (dvel[0] + cor_u, dvel[1] + cor_v)

    as83 = vertical_form != "sb81_theta"
    vorticity_term = vorticity_term_energy if as83 else vorticity_term_current
    kinetic_energy = (kinetic_energy_dual if as83
                      else (lambda vel: kinetic_energy_cgrid(vel[0], vel[1])))

    def v_dot_grad_L(ctx):
        u_c, v_c = cell_winds_cgrid(*ctx["vel"])
        gx_c = 0.5 * (ctx["dL"][0][:, :-1] + ctx["dL"][0][:, 1:])
        gy_c = 0.5 * (ctx["dL"][1][:-1] + ctx["dL"][1][1:])
        return u_c * gx_c + v_c * gy_c

    horizontal_T = None
    if config.use_ppm_transport:
        def horizontal_T(ctx):
            return cgrid_fv_scalar_advection_latlon_3d(ctx["T"], *ctx["vel"], grid)

    def dissipation(ctx, dvel, dT):
        du, dv, dT = latlon_dissipation(
            dvel[0], dvel[1], dT, *ctx["vel"], ctx["T"], ctx["p_s"], grid,
            ctx["sigma_coord"], config, dt, ctx["hybrid"])
        return (du, dv), dT

    def tracer_fn(ctx):
        u_c, v_c = cell_winds_cgrid(*ctx["vel"])
        return latlon_tracer_tendencies(
            tracers, *ctx["vel"], u_c, v_c, *ctx["dp_face"], ctx["div_dp"], ctx["dp"],
            ctx["vert_vel"], ctx["p_s"], grid, ctx["sigma_coord"], config, ctx["hybrid"])

    def ps_post(ctx, dps):
        if config.zero_mean_ps_tendency and not config.fix_mass:
            return zero_mean_tendency(dps, grid)
        return dps

    def boundary(dvel):
        return (dvel[0], latlon_pole_bc(dvel[1], config))

    return PEOperators(
        gops_ll, to_face, kinetic_energy,
        vorticity_term, v_dot_grad_L, horizontal_T, dissipation, tracer_fn,
        ps_post, boundary, "upwind", config.p_floor, config.p_ceil,
        vertical_form, face_pairs)


def cgrid_latlon_pe_tendencies(state: Any, grid: Any, sigma_coord: Any,
                               config: Any = None, *, vertical_form: str,
                               geom_pads: Any = None, dt: Any = None):
    """Drop-in twin of ``cgrid_latlon_hydrostatic_tendencies`` on the shared core:
    ``(du_dt, dv_dt, dT_dt, dp_s_dt, tracer_tends)``."""
    if geom_pads is not None:
        raise ValueError("the shared PE core does not take SPMD geometry pads")
    if config is None:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationConfig,
        )
        config = CGridLatLonPrimitiveEquationConfig()
    ops = latlon_pe_operators(grid, config, state.tracers, dt, vertical_form=vertical_form)
    (du, dv), dT, dps, tt, _ = primitive_equation_tendencies(
        state.T, (state.u, state.v), state.p_s, state.phis, sigma_coord, ops)
    return du, dv, dT, dps, tt


# ============================================================================
# MPAS / TRiSK plug-ins
# ============================================================================

def mpas_pe_operators(mesh: Any, config: Any, tracers: Any = None, dt: float = 0.0, *,
                      vertical_form: str) -> PEOperators:
    """MPAS / TRiSK plug-ins (fields ``(nCells|nEdges, nlev)``)."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        mpas_momentum_dissipation,
        mpas_tracer_tendencies,
        validate_vert4_flux_form_tracers,
        vertical_del4_T_tendency,
    )
    from legoesm.core.operators_voronoi import (
        apvm_correction_3d,
        cell_to_edge_avg_3d,
        kinetic_energy_cell_3d,
        potential_vorticity_vertex_3d,
        pv_flux_energy_conserving_3d,
        pv_flux_enstrophy_conserving_3d,
    )

    validate_vert4_flux_form_tracers(config.vert4_flux_form_tracers,
                                     config.vert4_log_form_tracers)
    scheme = config.vert_advection_scheme
    if scheme not in VERTICAL_ADVECTION_SCHEMES:
        raise ValueError(f"unknown vert_advection_scheme {scheme!r}; "
                         f"expected one of {VERTICAL_ADVECTION_SCHEMES}")
    if config.pv_scheme == "energy":
        pv_flux = pv_flux_energy_conserving_3d
    elif config.pv_scheme == "enstrophy":
        pv_flux = pv_flux_enstrophy_conserving_3d
    else:
        raise ValueError(f"Unknown pv_scheme {config.pv_scheme!r}; "
                         "expected one of: 'energy', 'enstrophy'.")
    gops = mpas_neighbour_operators(mesh)

    def check_scheme(hybrid):
        if hybrid and scheme not in ("upwind", "sb"):
            raise ValueError(f"vert_advection_scheme={scheme!r} is sigma-only; "
                             "the hybrid lane takes 'upwind' or 'sb'.")
        if not hybrid and scheme == "sb":
            raise ValueError("vert_advection_scheme='sb' is hybrid-only.")

    def vorticity_term(ctx, dvel):
        check_scheme(ctx["hybrid"])
        u = ctx["vel"][0]
        h = ctx["dp"] / config.g
        h_edge = ctx["dp_face"][0] / config.g if ctx["hybrid"] else None
        q = potential_vorticity_vertex_3d(u, h, mesh.fVertex, mesh)
        if config.apvm_scale > 0:
            q = apvm_correction_3d(q, u, mesh, config.apvm_scale * dt)
        return (dvel[0] + pv_flux(u, h, q, mesh, h_edge_3d=h_edge),)

    def v_dot_grad_L(ctx):
        L = ctx["L"]
        return gops.divergence(ctx["vel"][0] * cell_to_edge_avg_3d(L, mesh)) - L * ctx["div_u"]

    def dissipation(ctx, dvel, dT):
        du = mpas_momentum_dissipation(dvel[0], ctx["vel"][0], mesh, config)
        T = ctx["T"]
        if config.K_h > 0:
            dT = dT + config.K_h * gops.divergence(gops.gradient(T))
        if config.nu_vert4_T > 0.0 and T.shape[-1] > 2:
            w = ctx["dp"] if ctx["hybrid"] else ctx["sigma_coord"].dsigma
            dT = dT + vertical_del4_T_tendency(T, config.nu_vert4_T, w)
        return (du,), dT

    def ps_del4(ctx):
        if config.nu_del4_ps <= 0:
            return None
        if "dps_del4" not in ctx:
            del2 = gops.divergence(gops.gradient(ctx["p_s"]))
            ctx["dps_del4"] = -config.nu_del4_ps * gops.divergence(gops.gradient(del2))
        return ctx["dps_del4"]

    def tracer_fn(ctx):
        hybrid = ctx["hybrid"]
        w = ctx["dp"] if hybrid else ctx["sigma_coord"].dsigma
        return mpas_tracer_tendencies(
            tracers, ctx["vel"][0], mesh, ctx["sigma_coord"], config, hybrid,
            ctx["vert_vel"] if hybrid else None, None if hybrid else ctx["vert_vel"],
            ctx["p_s"], ps_del4(ctx), w)

    def ps_post(ctx, dps):
        d4 = ps_del4(ctx)
        return dps if d4 is None else dps + d4

    _check_vertical_form(vertical_form)
    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    return PEOperators(
        gops, lambda f: (cell_to_edge_avg_3d(f, mesh),),
        lambda vel: kinetic_energy_cell_3d(vel[0], mesh),
        vorticity_term, v_dot_grad_L, None, dissipation, tracer_fn, ps_post,
        lambda dvel: dvel, scheme, config.p_floor, config.p_ceil,
        vertical_form, lambda f: ((f[c1], f[c2]),))


def mpas_pe_tendencies(state: Any, mesh: Any, sigma_coord: Any, config: Any = None,
                       physics_tendency: Any = None, dt: float = 0.0, *, vertical_form: str):
    """Drop-in twin of ``mpas_hydrostatic_tendencies`` on the shared core."""
    if physics_tendency is not None:
        raise ValueError("the shared PE core does not add physics tendencies")
    from legoesm.core.field import Field
    from legoesm.core.state import MPASHydrostaticTendencies

    if config is None:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationConfig,
        )
        config = MPASPrimitiveEquationConfig()
    ops = mpas_pe_operators(mesh, config, state.tracers, dt, vertical_form=vertical_form)
    (du,), dT, dps, tt, _ = primitive_equation_tendencies(
        state.T.data, (state.u.data,), state.p_s.data, state.phis.data, sigma_coord, ops)
    return MPASHydrostaticTendencies(
        du_dt=Field(data=du, name="du_dt", dims=("nEdges", "nlev"), units="m/s²"),
        dT_dt=Field(data=dT, name="dT_dt", dims=("nCells", "nlev"), units="K/s"),
        dp_s_dt=Field(data=dps, name="dp_s_dt", dims=("nCells",), units="Pa/s"),
        dphis_dt=Field(data=jnp.zeros_like(state.phis.data), name="dphis_dt",
                       dims=("nCells",), units="m²/s³"),
        tracer_tendencies=tt,
    )
