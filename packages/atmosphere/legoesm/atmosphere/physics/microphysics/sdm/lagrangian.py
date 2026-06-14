"""Advected-Lagrangian Super-Droplet Method for the spectral plane LES.

This module is the persistent-particle SDM path. It carries a fixed-size
structure-of-arrays particle population across LES steps, advects it with the
resolved wind, adds terminal-velocity sedimentation, applies Shima
collision-coalescence independently in each Eulerian cell, and exchanges vapor
and latent heat with the Eulerian thermodynamic fields during condensation.

The implementation is deliberately opt-in and separate from ``column.py``. The
existing ``scheme="sdm"`` column adapter remains a reconstructed box closure
for the keyless Eulerian microphysics interface; this module is a stateful
driver that must be carried by the caller.

Differentiability contract:

* ``collision_mode="stochastic"`` (default) is forward-only: pairing consumes a
  PRNG key, active particles are sorted into cell segments, and integer
  accept/reject collision counts are discontinuous.
* ``collision_mode="deterministic"`` keeps the same fixed candidate-pair layout
  but replaces random integer collision counts with a smooth expected
  mean-field increment. For fixed cell membership, advection interpolation,
  terminal velocity, deterministic coalescence, condensation, vapor/latent-heat
  coupling, and q_c/q_r scatter-add binning all have JAX reverse-mode VJPs with
  respect to particle radii/multiplicities and Eulerian thermodynamic fields.
  Particle-to-cell assignment itself uses floor/segment sorting, so gradients
  with respect to positions across cell boundaries are not meaningful.

Simplifications relative to full production SDM implementations such as PySDM
or SuperDropGPU:

* particles are stored in one global fixed-size pool. Per-cell coalescence
  sorts the active pool by cell each step and pairs adjacent droplets inside
  each sorted cell block; this is fixed-size and JIT-friendly, but still not a
  persistent GPU-resident cell-list container;
* stochastic coalescence is forward-only. Deterministic coalescence is
  differentiable for fixed cell membership as described above;
* aerosol activation is represented by initialized super-droplets (wet radius
  plus solute), not by a separate source/recycling operator.

Despite those limits, the conserved exchanges are explicit: condensation moves
mass between vapor and represented liquid in the containing cell, latent heat is
added with ``L_v/c_p``, sedimented particles accumulate surface precipitation,
and diagnostic cloud/rain mixing ratios are binned from particles rather than
prognosed as Eulerian condensate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, random

from legoesm import constants
from legoesm.thermo import relative_humidity
from legoesm.atmosphere.physics.microphysics.sdm.coalescence import (
    coalescence_step_pairs,
    coalescence_step_pairs_deterministic,
)
from legoesm.atmosphere.physics.microphysics.sdm.condensation import integrate_radius
from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.kernels import terminal_velocity
from legoesm.atmosphere.physics.microphysics.sdm.particles import (
    SuperDropletState,
    represented_water_mass,
    water_mass_per_droplet,
)

_FOUR_THIRDS_PI = 4.0 / 3.0 * jnp.pi

__physics_contract__ = {
    "summary": (
        "Persistent advected Lagrangian SDM for the spectral plane LES: "
        "fixed-size super-droplet pool with resolved-wind advection, terminal "
        "sedimentation, per-cell Shima coalescence, condensation, and two-way "
        "vapor/heat coupling."
    ),
    "inputs": {
        "x,y,z": "m",
        "u,v,w": "m/s",
        "theta": "K",
        "q_v": "kg/kg",
        "rho,p,T": "kg/m^3, Pa, K",
        "dt": "s",
    },
    "outputs": {
        "x,y,z": "m",
        "surface_precip": "kg/m^2",
        "q_v,q_c,q_r": "kg/kg",
        "theta": "K",
    },
    "sign_convention": (
        "Terminal velocity is a positive fall speed subtracted from vertical "
        "motion. Condensation removes q_v and warms; evaporation moistens and "
        "cools. q_c/q_r are diagnostic bins from particles."
    ),
    "conserves": ["moisture", "energy"],
    # Contract bool = the DEFAULT (stochastic) path, which is forward-only.
    # SDMConfig(collision_mode='deterministic') + fixed cell membership makes the
    # whole step (advection, terminal velocity, mean-field coalescence,
    # condensation, vapor/latent-heat coupling, q_c/q_r scatter-add) reverse-mode
    # differentiable wrt radii/multiplicities/Eulerian fields — see the module
    # docstring; floor-based cell assignment stays discrete wrt position.
    "differentiable": False,
    "reference": (
        "Shima et al. (2009) QJRMS 135:1307; PySDM/SuperDropGPU algorithmic "
        "pattern; ERF SuperDropletsMoist coupling semantics"
    ),
    "idealized_test": (
        "Uniform resolved wind translates particles by u*dt with periodic x/y "
        "wrapping and no count change; still-column sedimentation removes "
        "surface-crossing droplets and adds exactly their represented mass to "
        "surface precipitation; per-cell coalescence conserves represented "
        "liquid; condensation conserves vapor+particle water and applies "
        "dT=L_v/c_p*dq_l."
    ),
}


class LagrangianSDMState(NamedTuple):
    """Fixed-size persistent super-droplet state carried beside the LES state."""

    droplets: SuperDropletState
    x: jax.Array                 # (n_sd,) [m], periodic in x
    y: jax.Array                 # (n_sd,) [m], periodic in y
    z: jax.Array                 # (n_sd,) [m], clamped to [0, Lz]
    key: jax.Array               # PRNG key for stochastic coalescence
    surface_precip: jax.Array    # (ny,nx) cumulative surface precip [kg/m^2]


def make_lagrangian_sdm_state(
    droplets: SuperDropletState,
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    key: jax.Array,
    grid,
    surface_precip: jax.Array | None = None,
) -> LagrangianSDMState:
    """Assemble a persistent Lagrangian SDM state.

    ``grid`` is a ``spectral_les_plane.SpectralLESGrid``-like object with
    ``cfg.nx``, ``cfg.ny``, ``cfg.Lx``, ``cfg.Ly`` and ``cfg.Lz``. It is kept
    duck-typed to avoid importing the LES module from the microphysics package.
    """
    dtype = droplets.radius.dtype
    x = jnp.mod(jnp.asarray(x, dtype=dtype), jnp.asarray(grid.cfg.Lx, dtype=dtype))
    y = jnp.mod(jnp.asarray(y, dtype=dtype), jnp.asarray(grid.cfg.Ly, dtype=dtype))
    z = jnp.clip(jnp.asarray(z, dtype=dtype), 0.0, jnp.asarray(grid.cfg.Lz, dtype=dtype))
    if surface_precip is None:
        surface_precip = jnp.zeros((grid.cfg.ny, grid.cfg.nx), dtype=dtype)
    else:
        surface_precip = jnp.asarray(surface_precip, dtype=dtype)
    return LagrangianSDMState(
        droplets=droplets, x=x, y=y, z=z, key=key, surface_precip=surface_precip)


def initialize_lagrangian_sdm(
    key: jax.Array,
    grid,
    n_sd: int,
    number_concentration: float | jax.Array = 1.0e8,
    radius: float | jax.Array = 1.0e-6,
    solute_mass: float | jax.Array = 0.0,
    dtype=jnp.float64,
) -> LagrangianSDMState:
    """Seed a fixed-size super-droplet pool from scalar or vertical profiles.

    ``number_concentration`` is interpreted as real droplets per m³. If it is a
    vertical profile ``(nz,)``, each uniformly placed super-droplet receives the
    local concentration times the domain volume divided by ``n_sd``. ``radius``
    and ``solute_mass`` may also be scalars or ``(nz,)`` profiles sampled at the
    droplet's initial cell. This is a compact initializer for BOMEX/RICO-style
    aerosol or cloud profiles; callers needing importance sampling can build a
    ``SuperDropletState`` directly and pass it to :func:`make_lagrangian_sdm_state`.
    """
    if n_sd < 1:
        raise ValueError(f"n_sd must be >= 1, got {n_sd}")
    k_pos, k_next = random.split(key)
    kx, ky, kz = random.split(k_pos, 3)
    x = random.uniform(kx, (n_sd,), dtype=dtype) * jnp.asarray(grid.cfg.Lx, dtype)
    y = random.uniform(ky, (n_sd,), dtype=dtype) * jnp.asarray(grid.cfg.Ly, dtype)
    z = random.uniform(kz, (n_sd,), dtype=dtype) * jnp.asarray(grid.cfg.Lz, dtype)

    iz = jnp.clip(jnp.floor(z / grid.dz).astype(jnp.int32), 0, grid.cfg.nz - 1)

    def sample_profile(value):
        arr = jnp.asarray(value, dtype=dtype)
        if arr.ndim == 0:
            return jnp.full((n_sd,), arr, dtype=dtype)
        if arr.shape != (grid.cfg.nz,):
            raise ValueError(
                f"profile inputs must be scalar or shape ({grid.cfg.nz},), "
                f"got {arr.shape}")
        return arr[iz]

    n_local = sample_profile(number_concentration)
    radius_local = sample_profile(radius)
    solute_local = sample_profile(solute_mass)
    domain_volume = jnp.asarray(grid.cfg.Lx * grid.cfg.Ly * grid.cfg.Lz, dtype)
    multiplicity = n_local * domain_volume / jnp.asarray(n_sd, dtype)
    active = jnp.where(multiplicity > 0.0, 1.0, 0.0).astype(dtype)
    droplets = SuperDropletState(
        multiplicity=multiplicity,
        radius=radius_local,
        solute_mass=solute_local,
        active=active,
    )
    return make_lagrangian_sdm_state(droplets, x, y, z, k_next, grid)


def _broadcast_cell_field(value, grid, dtype):
    arr = jnp.asarray(value, dtype=dtype)
    shape = (grid.cfg.ny, grid.cfg.nx, grid.cfg.nz)
    if arr.ndim == 0:
        return jnp.full(shape, arr, dtype=dtype)
    if arr.ndim == 1:
        if arr.shape[0] != grid.cfg.nz:
            raise ValueError(
                f"vertical profile must have nz={grid.cfg.nz}, got {arr.shape}")
        return jnp.broadcast_to(arr[None, None, :], shape)
    if arr.shape != shape:
        raise ValueError(f"cell field must have shape {shape}, got {arr.shape}")
    return arr


def _particle_cell_indices(state: LagrangianSDMState, grid):
    dtype = state.x.dtype
    Lx = jnp.asarray(grid.cfg.Lx, dtype=dtype)
    Ly = jnp.asarray(grid.cfg.Ly, dtype=dtype)
    x = jnp.mod(state.x, Lx)
    y = jnp.mod(state.y, Ly)
    ix = jnp.floor(x / grid.dx).astype(jnp.int32) % grid.cfg.nx
    iy = jnp.floor(y / grid.dy).astype(jnp.int32) % grid.cfg.ny
    iz = jnp.clip(jnp.floor(state.z / grid.dz).astype(jnp.int32), 0, grid.cfg.nz - 1)
    cell_id = (iy * grid.cfg.nx + ix) * grid.cfg.nz + iz
    return ix, iy, iz, cell_id


def particle_cell_indices(state: LagrangianSDMState, grid):
    """Return ``(ix, iy, iz, cell_id)`` for every super-droplet."""
    return _particle_cell_indices(state, grid)


def _interp_trilinear(field, x, y, z, grid, *, z_stagger: str):
    dtype = field.dtype
    x = jnp.mod(x, jnp.asarray(grid.cfg.Lx, dtype=dtype))
    y = jnp.mod(y, jnp.asarray(grid.cfg.Ly, dtype=dtype))

    rx = x / grid.dx - 0.5
    ry = y / grid.dy - 0.5
    ix0_raw = jnp.floor(rx).astype(jnp.int32)
    iy0_raw = jnp.floor(ry).astype(jnp.int32)
    fx = rx - ix0_raw.astype(dtype)
    fy = ry - iy0_raw.astype(dtype)
    ix0 = ix0_raw % grid.cfg.nx
    iy0 = iy0_raw % grid.cfg.ny
    ix1 = (ix0 + 1) % grid.cfg.nx
    iy1 = (iy0 + 1) % grid.cfg.ny

    if z_stagger == "center":
        rz = jnp.clip(z / grid.dz - 0.5, 0.0, grid.cfg.nz - 1.0)
        nz_axis = grid.cfg.nz
    elif z_stagger == "face":
        rz = jnp.clip(z / grid.dz, 0.0, grid.cfg.nz)
        nz_axis = grid.cfg.nz + 1
    else:
        raise ValueError(f"unknown z_stagger {z_stagger!r}")
    iz0 = jnp.floor(rz).astype(jnp.int32)
    iz1 = jnp.minimum(iz0 + 1, nz_axis - 1)
    fz = rz - iz0.astype(dtype)

    c000 = field[iy0, ix0, iz0]
    c100 = field[iy0, ix1, iz0]
    c010 = field[iy1, ix0, iz0]
    c110 = field[iy1, ix1, iz0]
    c001 = field[iy0, ix0, iz1]
    c101 = field[iy0, ix1, iz1]
    c011 = field[iy1, ix0, iz1]
    c111 = field[iy1, ix1, iz1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def interpolate_wind(u, v, w, state: LagrangianSDMState, grid):
    """Tri-linearly interpolate resolved wind to every droplet position."""
    up = _interp_trilinear(u, state.x, state.y, state.z, grid, z_stagger="center")
    vp = _interp_trilinear(v, state.x, state.y, state.z, grid, z_stagger="center")
    wp = _interp_trilinear(w, state.x, state.y, state.z, grid, z_stagger="face")
    return up, vp, wp


def advect_step(
    state: LagrangianSDMState,
    u: jax.Array,
    v: jax.Array,
    w: jax.Array,
    grid,
    dt: float | jax.Array,
    cfg: SDMConfig,
    rho: float | jax.Array,
    p: float | jax.Array,
    T: float | jax.Array,
) -> tuple[LagrangianSDMState, jax.Array]:
    """Advect particles with resolved wind plus terminal sedimentation.

    The resolved wind is integrated with a midpoint (RK2) update. Horizontal
    boundaries are periodic. Particles crossing the lower wall are deactivated
    and their represented liquid mass is added to ``surface_precip`` in the
    surface cell [kg/m²]. The top wall is clamped.
    """
    dtype = state.x.dtype
    dt = jnp.asarray(dt, dtype=dtype)
    rho_grid = _broadcast_cell_field(rho, grid, dtype)
    p_grid = _broadcast_cell_field(p, grid, dtype)
    T_grid = _broadcast_cell_field(T, grid, dtype)

    up0, vp0, wp0 = interpolate_wind(u, v, w, state, grid)
    _, _, _, cell0 = _particle_cell_indices(state, grid)
    rho0 = rho_grid.reshape(-1)[cell0]
    p0 = p_grid.reshape(-1)[cell0]
    T0 = T_grid.reshape(-1)[cell0]
    vt0 = terminal_velocity(state.droplets.radius, rho0, p0, T0, cfg)

    active = state.droplets.active > 0.0
    xm = jnp.mod(state.x + 0.5 * dt * up0, jnp.asarray(grid.cfg.Lx, dtype=dtype))
    ym = jnp.mod(state.y + 0.5 * dt * vp0, jnp.asarray(grid.cfg.Ly, dtype=dtype))
    zm = jnp.clip(state.z + 0.5 * dt * (wp0 - vt0), 0.0,
                  jnp.asarray(grid.cfg.Lz, dtype=dtype))
    mid = state._replace(x=xm, y=ym, z=zm)
    upm, vpm, wpm = interpolate_wind(u, v, w, mid, grid)
    _, _, _, cellm = _particle_cell_indices(mid, grid)
    rhom = rho_grid.reshape(-1)[cellm]
    pm = p_grid.reshape(-1)[cellm]
    Tm = T_grid.reshape(-1)[cellm]
    vtm = terminal_velocity(state.droplets.radius, rhom, pm, Tm, cfg)

    x_trial = jnp.mod(state.x + dt * upm, jnp.asarray(grid.cfg.Lx, dtype=dtype))
    y_trial = jnp.mod(state.y + dt * vpm, jnp.asarray(grid.cfg.Ly, dtype=dtype))
    z_trial = state.z + dt * (wpm - vtm)
    crossed = active & (z_trial <= 0.0)
    z_trial = jnp.clip(z_trial, 0.0, jnp.asarray(grid.cfg.Lz, dtype=dtype))

    x_new = jnp.where(active, x_trial, state.x)
    y_new = jnp.where(active, y_trial, state.y)
    z_new = jnp.where(active, z_trial, state.z)

    ix_s = jnp.floor(x_new / grid.dx).astype(jnp.int32) % grid.cfg.nx
    iy_s = jnp.floor(y_new / grid.dy).astype(jnp.int32) % grid.cfg.ny
    surf_id = iy_s * grid.cfg.nx + ix_s
    area = jnp.asarray(grid.dx * grid.dy, dtype=dtype)
    dep = jnp.where(crossed, represented_water_mass(state.droplets) / area, 0.0)
    dprecip = jnp.zeros((grid.cfg.ny * grid.cfg.nx,), dtype=dtype).at[surf_id].add(dep)
    dprecip = dprecip.reshape((grid.cfg.ny, grid.cfg.nx))
    active_new = jnp.where(crossed, 0.0, state.droplets.active)
    droplets = state.droplets._replace(active=active_new)
    return state._replace(
        droplets=droplets,
        x=x_new,
        y=y_new,
        z=z_new,
        surface_precip=state.surface_precip + dprecip,
    ), dprecip


def diagnose_liquid_mixing_ratios(
    state: LagrangianSDMState,
    grid,
    rho: float | jax.Array,
    r_rain: float,
) -> tuple[jax.Array, jax.Array]:
    """Bin represented liquid to diagnostic ``(q_c, q_r)`` fields [kg/kg]."""
    dtype = state.x.dtype
    rho_grid = _broadcast_cell_field(rho, grid, dtype)
    _, _, _, cell_id = _particle_cell_indices(state, grid)
    mass = represented_water_mass(state.droplets)
    is_cloud = state.droplets.radius < r_rain
    n_cells = grid.cfg.ny * grid.cfg.nx * grid.cfg.nz
    cloud_mass = jnp.zeros((n_cells,), dtype=dtype).at[cell_id].add(
        jnp.where(is_cloud, mass, 0.0))
    rain_mass = jnp.zeros((n_cells,), dtype=dtype).at[cell_id].add(
        jnp.where(is_cloud, 0.0, mass))
    cell_air = rho_grid.reshape(-1) * jnp.asarray(
        grid.dx * grid.dy * grid.dz, dtype=dtype)
    qc = (cloud_mass / cell_air).reshape((grid.cfg.ny, grid.cfg.nx, grid.cfg.nz))
    qr = (rain_mass / cell_air).reshape((grid.cfg.ny, grid.cfg.nx, grid.cfg.nz))
    return qc, qr


def set_diagnostic_liquid_tracers(tracers, q_c, q_r):
    """Overwrite standard tracer slots 1/2 with binned SDM cloud/rain water."""
    if tracers.shape[-1] < 3:
        raise ValueError(
            "Lagrangian SDM needs tracer slots [0]=q_v, [1]=q_c, [2]=q_r; "
            f"got n_tracers={tracers.shape[-1]}.")
    return tracers.at[..., 1].set(q_c).at[..., 2].set(q_r)


def condensation_coupling_step(
    state: LagrangianSDMState,
    theta: jax.Array,
    tracers: jax.Array,
    grid,
    exner: jax.Array,
    p: jax.Array,
    rho: jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
) -> tuple[LagrangianSDMState, jax.Array, jax.Array, jax.Array]:
    """Grow/evaporate droplets and apply vapor/latent-heat back reaction.

    Returns ``(particle_state, theta_new, tracers_new, dq_liquid)`` where
    ``dq_liquid`` is the cellwise particle liquid change [kg/kg]. Positive
    values mean condensation.
    """
    if tracers.shape[-1] < 1:
        raise ValueError("Lagrangian SDM condensation needs tracer slot 0 = q_v.")
    dtype = theta.dtype
    exner_grid = _broadcast_cell_field(exner, grid, dtype)
    p_grid = _broadcast_cell_field(p, grid, dtype)
    rho_grid = _broadcast_cell_field(rho, grid, dtype)
    T_grid = theta * exner_grid
    qv_grid = tracers[..., 0]
    cell_volume = jnp.asarray(grid.dx * grid.dy * grid.dz, dtype=dtype)

    _, _, _, cell_id = _particle_cell_indices(state, grid)
    flat_T = T_grid.reshape(-1)
    flat_p = p_grid.reshape(-1)
    flat_qv = qv_grid.reshape(-1)
    flat_rho = rho_grid.reshape(-1)
    T_p = flat_T[cell_id]
    p_p = flat_p[cell_id]
    qv_p = flat_qv[cell_id]
    S_p = relative_humidity(T_p, p_p, qv_p)

    before = represented_water_mass(state.droplets)
    raw = integrate_radius(state.droplets, S_p, T_p, dt, cfg)
    after_raw = represented_water_mass(raw)
    delta_raw = after_raw - before

    n_cells = grid.cfg.ny * grid.cfg.nx * grid.cfg.nz
    pos_raw = jnp.maximum(delta_raw, 0.0)
    pos_cell = jnp.zeros((n_cells,), dtype=dtype).at[cell_id].add(pos_raw)
    max_condense = jnp.maximum(flat_qv, 0.0) * flat_rho * cell_volume
    pos_cell_safe = jnp.maximum(pos_cell, jnp.asarray(1.0e-30, dtype))
    scale_cell = jnp.where(
        pos_cell > 0.0, jnp.minimum(1.0, max_condense / pos_cell_safe), 1.0)
    scale = scale_cell[cell_id]
    delta = jnp.where(delta_raw > 0.0, delta_raw * scale, delta_raw)

    # Reconstruct radius from the represented mass after any vapor-donor clamp.
    xi = jnp.maximum(state.droplets.multiplicity, jnp.asarray(1.0e-300, dtype))
    m_drop = water_mass_per_droplet(state.droplets) + delta / xi
    m_drop = jnp.maximum(m_drop, 0.0)
    radius = jnp.cbrt(m_drop / (_FOUR_THIRDS_PI * constants.rho_water))
    radius = jnp.where(state.droplets.active > 0.0, radius, state.droplets.radius)
    droplets = state.droplets._replace(radius=radius)

    dq_liquid_flat = jnp.zeros((n_cells,), dtype=dtype).at[cell_id].add(delta)
    dq_liquid_flat = dq_liquid_flat / (flat_rho * cell_volume)
    qv_new_flat = flat_qv - dq_liquid_flat
    T_new_flat = flat_T + (constants.L_v / constants.c_pd) * dq_liquid_flat
    theta_new_flat = T_new_flat / exner_grid.reshape(-1)

    theta_new = theta_new_flat.reshape(theta.shape)
    qv_new = qv_new_flat.reshape(qv_grid.shape)
    tracers_new = tracers.at[..., 0].set(qv_new)
    dq_liquid = dq_liquid_flat.reshape(qv_grid.shape)
    return state._replace(droplets=droplets), theta_new, tracers_new, dq_liquid


def coalescence_cells_step(
    state: LagrangianSDMState,
    grid,
    rho: float | jax.Array,
    p: float | jax.Array,
    T: float | jax.Array,
    dt: float | jax.Array,
    cfg: SDMConfig,
) -> LagrangianSDMState:
    """Apply Shima collision-coalescence independently in every Eulerian cell.

    Active droplets are sorted by cell id once per call. In stochastic mode the
    global pool is first randomly permuted, then stable-sorted by cell so each
    segment has a random within-cell order. In deterministic mode no PRNG is
    consumed; stable sorting by cell preserves slot order inside a segment.
    Adjacent droplets inside each sorted active cell block form Shima candidate
    pairs. Cross-cell and inactive pairs are masked, and each valid pair uses
    the local active count ``n_local`` and local candidate count
    ``floor(n_local/2)`` in the Shima probability scale
    ``C(n_local,2)/floor(n_local/2)``.
    """
    dtype = state.x.dtype
    rho_grid = _broadcast_cell_field(rho, grid, dtype)
    p_grid = _broadcast_cell_field(p, grid, dtype)
    T_grid = _broadcast_cell_field(T, grid, dtype)
    flat_rho = rho_grid.reshape(-1)
    flat_p = p_grid.reshape(-1)
    flat_T = T_grid.reshape(-1)
    cell_volume = jnp.asarray(grid.dx * grid.dy * grid.dz, dtype=dtype)
    _, _, _, cell_id = _particle_cell_indices(state, grid)
    n_cells = grid.cfg.ny * grid.cfg.nx * grid.cfg.nz
    n_sd = state.droplets.radius.shape[0]
    if n_sd < 2:
        return state

    active_bool = (state.droplets.active > 0.0) & (state.droplets.multiplicity > 0.0)
    active_i = active_bool.astype(jnp.int32)
    n_local_i = jnp.zeros((n_cells,), dtype=jnp.int32).at[cell_id].add(active_i)

    # Inactive slots sort after every active cell block, so they cannot dilute
    # local pair formation. The stable sort preserves the random permutation
    # order within each cell.
    inactive_cell = jnp.asarray(n_cells, dtype=jnp.int32)
    sort_cell = jnp.where(active_bool, cell_id, inactive_cell)
    if cfg.collision_mode == "stochastic":
        key_next, key_pairs = random.split(state.key)
        k_perm, k_gamma = random.split(key_pairs)
        perm = random.permutation(k_perm, n_sd)
    elif cfg.collision_mode == "deterministic":
        key_next = state.key
        k_gamma = state.key
        perm = jnp.arange(n_sd, dtype=jnp.int32)
    else:
        raise ValueError(
            f"Unknown SDM collision_mode: {cfg.collision_mode!r} "
            "(expected 'stochastic' or 'deterministic')"
        )
    order = jnp.argsort(sort_cell[perm], stable=True)
    sorted_idx = perm[order]
    sorted_cell = sort_cell[sorted_idx]

    pos = jnp.arange(n_sd, dtype=jnp.int32)
    is_group_start = jnp.concatenate([
        jnp.ones((1,), dtype=bool),
        sorted_cell[1:] != sorted_cell[:-1],
    ])
    starts = jnp.where(is_group_start, pos, jnp.zeros_like(pos))
    group_start = lax.associative_scan(jnp.maximum, starts)
    is_group_stop = jnp.concatenate([
        sorted_cell[:-1] != sorted_cell[1:],
        jnp.ones((1,), dtype=bool),
    ])
    stops = jnp.where(is_group_stop, pos + 1, jnp.full_like(pos, n_sd))
    group_stop = lax.associative_scan(jnp.minimum, stops[::-1])[::-1]
    local_rank = pos - group_start

    next_pos = jnp.minimum(pos + 1, n_sd - 1)
    ia = sorted_idx
    ib = sorted_idx[next_pos]
    same_active_cell = (sorted_cell < n_cells) & (sorted_cell == sorted_cell[next_pos])
    even_local_rank = (local_rank % 2) == 0
    valid_pair = same_active_cell & even_local_rank & (pos < n_sd - 1)

    safe_cell = jnp.minimum(sorted_cell, n_cells - 1)
    n_local = n_local_i[safe_cell].astype(dtype)
    segment_count = jnp.maximum(group_stop - group_start, 0).astype(dtype)
    l_local = jnp.floor(segment_count / 2.0)
    pair_scaling = jnp.where(l_local > 0.0,
                             0.5 * n_local * (n_local - 1.0) / l_local,
                             0.0)

    if cfg.collision_mode == "deterministic":
        droplets = coalescence_step_pairs_deterministic(
            state.droplets,
            ia,
            ib,
            valid_pair.astype(dtype),
            pair_scaling,
            cell_volume,
            flat_rho[safe_cell],
            flat_p[safe_cell],
            flat_T[safe_cell],
            dt,
            cfg,
        )
    else:
        droplets = coalescence_step_pairs(
            state.droplets,
            ia,
            ib,
            valid_pair.astype(dtype),
            pair_scaling,
            cell_volume,
            flat_rho[safe_cell],
            flat_p[safe_cell],
            flat_T[safe_cell],
            dt,
            k_gamma,
            cfg,
        )
    return state._replace(droplets=droplets, key=key_next)


def total_water_mass(
    state: LagrangianSDMState,
    tracers: jax.Array,
    grid,
    rho: float | jax.Array,
) -> jax.Array:
    """Domain total water mass [kg]: vapor + airborne liquid + surface precip."""
    dtype = state.x.dtype
    rho_grid = _broadcast_cell_field(rho, grid, dtype)
    cell_volume = jnp.asarray(grid.dx * grid.dy * grid.dz, dtype=dtype)
    vapor = jnp.sum(tracers[..., 0] * rho_grid * cell_volume)
    airborne = jnp.sum(represented_water_mass(state.droplets))
    precip = jnp.sum(state.surface_precip) * jnp.asarray(grid.dx * grid.dy, dtype=dtype)
    return vapor + airborne + precip


def apply_lagrangian_sdm_to_les_state(
    les_state,
    sdm_state: LagrangianSDMState,
    grid,
    ref,
    dt: float | jax.Array,
    cfg: SDMConfig,
    *,
    do_condensation: bool = True,
    do_coalescence: bool = True,
) -> tuple[object, LagrangianSDMState, dict]:
    """Operator-split Lagrangian SDM update on a spectral LES state.

    ``les_state`` must carry ``theta`` and standard water tracers with at least
    slots ``[0]=q_v, [1]=q_c, [2]=q_r``. ``ref`` is a
    ``spectral_les_moist.SpectralRefState``-like object containing bottom-up
    ``p_c``, ``rho_c`` and ``exner_c`` profiles. The returned LES state has
    vapor and theta updated by condensation and q_c/q_r overwritten by particle
    diagnostics.
    """
    if les_state.theta is None or les_state.tracers is None:
        raise ValueError("Lagrangian SDM LES coupling needs theta and tracers.")
    if les_state.tracers.shape[-1] < 3:
        raise ValueError(
            "Lagrangian SDM LES coupling needs tracer slots q_v/q_c/q_r "
            f"(n_tracers >= 3); got {les_state.tracers.shape[-1]}.")

    exner = ref.exner_c
    p = ref.p_c
    rho = ref.rho_c
    T_grid = les_state.theta * jnp.asarray(exner, dtype=les_state.theta.dtype)[None, None, :]
    total_water_before = total_water_mass(sdm_state, les_state.tracers, grid, rho)

    sdm_state, dprecip = advect_step(
        sdm_state, les_state.u, les_state.v, les_state.w, grid, dt, cfg,
        rho=rho, p=p, T=T_grid)

    if do_coalescence:
        sdm_state = coalescence_cells_step(
            sdm_state, grid, rho=rho, p=p, T=T_grid, dt=dt, cfg=cfg)

    theta = les_state.theta
    tracers = les_state.tracers
    if do_condensation:
        sdm_state, theta, tracers, dq_liquid = condensation_coupling_step(
            sdm_state, theta, tracers, grid, exner, p, rho, dt, cfg)
    else:
        dq_liquid = jnp.zeros_like(tracers[..., 0])

    q_c, q_r = diagnose_liquid_mixing_ratios(sdm_state, grid, rho, cfg.r_rain)
    tracers = set_diagnostic_liquid_tracers(tracers, q_c, q_r)
    les_state = les_state._replace(theta=theta, tracers=tracers)
    total_water_after = total_water_mass(sdm_state, tracers, grid, rho)
    diagnostics = {
        "surface_precip_step": dprecip,
        "surface_precip": sdm_state.surface_precip,
        "q_c": q_c,
        "q_r": q_r,
        "dq_liquid": dq_liquid,
        "airborne_water": jnp.sum(represented_water_mass(sdm_state.droplets)),
        "total_water_before": total_water_before,
        "total_water": total_water_after,
        "total_water_error": total_water_after - total_water_before,
        "n_active": jnp.sum(sdm_state.droplets.active),
    }
    return les_state, sdm_state, diagnostics
