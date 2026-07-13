"""Arakawa C-grid Hydrostatic Primitive Equations on the latitude-longitude grid.

True staggered C-grid discretisation following Sadourny / MITgcm / NEMO
conventions:

  u at longitude interfaces:  shape (n_lat, n_lon+1, nlev)
  v at latitude interfaces:   shape (n_lat+1, n_lon, nlev)
  T, p_s, phis at cell centers: shape (n_lat, n_lon, nlev) / (n_lat, n_lon)

Key design choices:
- Vector-invariant momentum with Bernoulli function at cell centers.
- Sigma-coordinate pressure gradient: -grad(Φ + KE) - R_d T grad(ln p_s).
- Sadourny (1975) energy-conserving Coriolis from the ocean C-grid operators.
- Conservative C-grid divergence for continuity and sigma-dot diagnosis.
- Vertical advection using upwind differencing in sigma.
- Pole treatment: v = 0 at poles (wall BC), periodic longitude.

Operator reuse:
- Gradient, divergence, Coriolis, vector Laplacian from the ocean lat-lon
  C-grid operator module.
- Vertical coordinate utilities (sigma-dot, vertical advection, geopotential)
  from grids.vertical.
- PPM-compatible cell-centered gradients from operators_fv_latlon for
  temperature advection.

References
----------
- Simmons & Burridge (1981): An Energy and Angular-Momentum Conserving
  Vertical Finite-Difference Scheme and Hybrid Vertical Coordinates.
- Sadourny (1975): The dynamics of finite-difference models of the
  shallow water equations.
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core (sigma-coordinate
  hydrostatic PE sections).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState
from legoesm.grids.operators_latlon_cgrid import (
    gradient_x_cgrid,
    gradient_y_cgrid,
    divergence_cgrid,
    vector_laplacian_cgrid,
    laplacian_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface_halo,
    cell_to_cgrid_winds,
)
from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
    absolute_vorticity_coriolis,
)
from legoesm.core.operators_fv_latlon_3d import (
    cgrid_fv_scalar_advection_latlon_3d,
    cgrid_fv_flux_divergence_latlon_3d,
)
from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.polar_filter import (
    compute_polar_filter_mask,
    fourier_filter_3d,
    fourier_filter,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_sigma,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential,
    compute_geopotential_hybrid,
    vertical_advection,
    vertical_advection_hybrid,
    compute_pressure_velocity,
    compute_mass_flux_from_cumsum,
    compute_omega_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import (
    IntegrationMixin,
    refuse_unthreaded_stateful_physics,
)
from legoesm.core.cfl import pole_cell_dx, cfl_max_dt
from legoesm.core.conservation import (
    zero_mean_tendency,
    batch_global_area_sums,
    conservation_accumulator,
)
from legoesm.core.precision import cast_pytree
from legoesm.core.operators_fv_latlon import (
    fv_gradient_lon_3d as _fv_gradient_lon_3d,
    fv_gradient_lat_3d as _fv_gradient_lat_3d,
)

from legoesm.grids.halo_latlon import pad_halo_latlon_3d
from legoesm import constants
import inspect


# ==============================================================================
# State and Config
# ==============================================================================

class CGridLatLonHydrostaticState(NamedTuple):
    """Hydrostatic state on the lat-lon Arakawa C-grid.

    u      : (n_lat, n_lon+1, nlev)  -- zonal velocity at lon interfaces [m/s]
    v      : (n_lat+1, n_lon, nlev)  -- meridional velocity at lat interfaces [m/s]
    T      : (n_lat, n_lon, nlev)    -- temperature at cell centres [K]
    p_s    : (n_lat, n_lon)          -- surface pressure at cell centres [Pa]
    phis   : (n_lat, n_lon)          -- surface geopotential [m^2/s^2]  (static)
    tracers: dict mapping name → (n_lat, n_lon, nlev) arrays [various]
    """
    u: jax.Array
    v: jax.Array
    T: jax.Array
    p_s: jax.Array
    phis: jax.Array
    tracers: dict = {}


class CGridLatLonPrimitiveEquationConfig(NamedTuple):
    """Configuration for the C-grid lat-lon hydrostatic PE model.

    Time integrator options: "ssp_rk3", "ssp_rk34", "ssp_rk54", "rk4".

    pole_v_bc : tuple[bool, bool]
        (south, north) — whether to zero ``v`` at the south / north pole
        rows of the lat axis.  Default ``(True, True)`` is the serial /
        single-rank convention (wall BC at both global poles).  Under
        latitude-band MPI, the wrapper sets each flag to
        ``layout.<side>_rank is None`` so only the boundary ranks zero
        the actual global poles and interior ranks leave their band
        boundaries (which are interior v-faces shared with the
        neighbour rank) alone.  See
        :func:`legoesm.parallel.latlon_mpi.make_latlon_mpi_step`.
    """
    g: float = constants.g
    A_h: float = 0.0              # Laplacian viscosity [m^2/s]
    time_integrator: str = "ssp_rk3"
    fix_mass: bool = True
    anchor_mass_to_initial: bool = False  # Mirror cubed-sphere: anchor fixer to initial mass
    T_min: float = 50.0           # Temperature floor [K]
    p_floor: float = 100.0        # Pressure floor [Pa] for surface pressure positivity
    zero_mean_ps_tendency: bool = True
    use_ppm_transport: bool = True  # PPM scalar transport (vs cell-centered gradient)
    use_polar_filter: bool = False
    polar_filter_cutoff_deg: float = 60.0
    polar_filter_max_wave_speed: float = 300.0
    pole_v_bc: tuple = (True, True)  # (south_pole, north_pole) — see docstring
    pole_v_bc_offset: int = 0
    """Where the global pole rows live relative to the array ends.

    ``0`` (serial / single-rank): the actual pole rows sit at indices
    ``v[0]`` and ``v[-1]``, matching the legacy
    ``jnp.pad(v[1:-1], ((1, 1), ...))`` pattern.

    ``halo`` (lat-lon MPI on a state padded by ``halo`` rows on each
    side): the actual pole rows are ``halo`` cells *into* the padded
    array — ``v[halo]`` and ``v[-(halo+1)]``.  Zeroing the padded
    ends instead silently leaves the real poles non-zero, drives the
    next RK stage into a divergent state, and produces NaNs (caught
    by ``tests/parallel/test_latlon_mpi_step_serial.py`` on Stage-2
    smoke).
    """
    p_ceil: float = 2.0e6          # Surface-pressure ceiling [Pa] (~20-bar overflow guard for omega/p).
    # --- Top sponge (Rayleigh damping increasing toward the model lid, #836) ---
    # The hydrostatic lat-lon C-grid dycore otherwise has NO absorbing layer at
    # the rigid lid, so upward gravity-wave / convective energy reflects and
    # contaminates the upper levels (the nonhydrostatic cores already carry
    # ``sponge_profile``; the hydrostatic path dropped it).  ``sponge_coeff == 0``
    # (default) is OFF and byte-identical.  ``sponge_coeff > 0`` damps u/v toward
    # REST over the top ``sponge_width_m`` metres via the SHARED
    # ``compressible_euler.sponge_profile`` fed a log-pressure height proxy.
    # Appended after ``p_ceil`` (defaults => positional-ABI-safe for old callers).
    sponge_coeff: float = 0.0          # Rayleigh damping SCALE [1/s]: the exact lid
    #                                   value for shape='sin2'; 'sam_rational' peaks
    #                                   at sponge_coeff*100/101 (see sponge_profile)
    sponge_width_m: float = 10000.0    # sponge-layer depth below the top [m]
    sponge_shape: str = "sin2"         # "sin2" | "sam_rational" (see sponge_profile)
    sponge_scale_height_m: float = 7500.0  # log-pressure scale height for sigma->z.
    # Damp EDDIES only (deviations from the zonal mean).  A damp-to-rest
    # sponge acting on the zonal-mean jet exerts a net Coriolis torque that
    # drives poleward mass drift (measured on the latlon24 pilot: a 10/day
    # driver-level sponge piled zonal-mean p_s to ~1120 hPa at the polar
    # flanks by day 100 while draining the tropics).  Preserving the
    # zonal-mean momentum removes the torque while still absorbing the wave
    # energy (standard GCM sponge practice).  False = legacy damp-to-rest.
    # Last field to preserve positional ABI.
    sponge_eddy_only: bool = True


def _zero_v_at_pole(v, *, south: bool, north: bool, offset: int = 0):
    """Zero ``v`` at the south / north pole rows (wall BC).

    Default ``offset=0`` reproduces the legacy
    ``jnp.pad(v[1:-1], ((1, 1), (0, 0), (0, 0)))`` exactly — same
    single-Pad HLO, same bit-output.

    With ``offset>0`` the helper zeros ``v[offset]`` (south pole on an
    array padded by ``offset`` halo rows below) and ``v[-(offset+1)]``
    (north pole on an array padded by ``offset`` halo rows above).
    The halo rows themselves are left unchanged; they get stripped
    after the step by the MPI wrapper.

    Implementation note: for the asymmetric ``offset>0`` paths we use
    ``.at[...].set(0.0)`` rather than ``jnp.pad`` because the latter
    only models zero-pads at the array ends, not at an interior
    index.  ``.at`` lowers to a small ``Scatter`` on the lat axis —
    one HLO more than the legacy Pad, acceptable for the few
    per-step v-updates.
    """
    if south and north and offset == 0:
        # Legacy fast path — single Pad HLO, serial bit-identical.
        return jnp.pad(v[1:-1, :, :], ((1, 1), (0, 0), (0, 0)))
    out = v
    if south:
        out = out.at[offset].set(jnp.zeros_like(out[offset]))
    if north:
        n = out.shape[0]
        out = out.at[n - 1 - offset].set(jnp.zeros_like(out[n - 1 - offset]))
    return out


# interp_cell_to_uface and interp_cell_to_vface_halo are imported from
# legoesm.grids.operators_latlon_cgrid (shared with ocean).  The v-face
# variant is the backend-dispatched twin of ``interp_cell_to_vface``:
# bit-identical in serial, neighbour-averaged at MPI band cuts.


def _face_to_cell_u(u: jnp.ndarray) -> jnp.ndarray:
    """Average u from lon faces to cell centers.

    u : (n_lat, n_lon+1, nlev) → (n_lat, n_lon, nlev)
    """
    return 0.5 * (u[:, :-1] + u[:, 1:])


def _face_to_cell_v(v: jnp.ndarray) -> jnp.ndarray:
    """Average v from lat faces to cell centers.

    v : (n_lat+1, n_lon, nlev) → (n_lat, n_lon, nlev)
    """
    return 0.5 * (v[:-1] + v[1:])


# ==============================================================================
# Adapter: cell-centered HydrostaticState ↔ C-grid state
# ==============================================================================

def hydrostatic_to_cgrid(
    state: HydrostaticState,
    grid: LatLonGrid,
) -> CGridLatLonHydrostaticState:
    """Convert a cell-centered HydrostaticState to C-grid.

    Winds are interpolated from cell centers to faces.
    v is set to zero at pole boundaries.
    """
    u_cell = state.u.data
    v_cell = state.v.data if state.v is not None else jnp.zeros_like(u_cell)

    u_face, v_face = cell_to_cgrid_winds(u_cell, v_cell)

    tracers = {}
    if hasattr(state, 'tracers') and state.tracers is not None:
        tracers = {name: f.data for name, f in state.tracers.items()}

    return CGridLatLonHydrostaticState(
        u=u_face,
        v=v_face,
        T=state.T.data,
        p_s=state.p_s.data,
        phis=state.phis.data,
        tracers=tracers,
    )


def cgrid_to_hydrostatic(
    state: CGridLatLonHydrostaticState,
    grid: LatLonGrid,
) -> HydrostaticState:
    """Convert a C-grid state to cell-centered HydrostaticState.

    Winds are averaged from faces to cell centers.
    """
    u_cell = _face_to_cell_u(state.u)
    v_cell = _face_to_cell_v(state.v)

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    tracers = None
    if state.tracers:
        tracers = {
            name: Field(data=arr, name=name, dims=dims_3d, units="kg/kg")
            for name, arr in state.tracers.items()
        }

    return HydrostaticState(
        u=Field(data=u_cell, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=v_cell, name="v", dims=dims_3d, units="m/s"),
        T=Field(data=state.T, name="T", dims=dims_3d, units="K"),
        p_s=Field(data=state.p_s, name="p_s", dims=dims_2d, units="Pa"),
        phis=Field(data=state.phis, name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=tracers,
    )


# ==============================================================================
# Tendency computation
# ==============================================================================

def cgrid_latlon_hydrostatic_tendencies(
    state: CGridLatLonHydrostaticState,
    grid: LatLonGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    config: CGridLatLonPrimitiveEquationConfig = CGridLatLonPrimitiveEquationConfig(),
):
    """Compute hydrostatic PE tendencies on the lat-lon C-grid.

    Supports both pure sigma and hybrid sigma-pressure vertical coordinates.

    Parameters
    ----------
    state : CGridLatLonHydrostaticState
    grid : LatLonGrid
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    config : CGridLatLonPrimitiveEquationConfig

    Returns
    -------
    (du_dt, dv_dt, dT_dt, dp_s_dt, tracer_tends)
        Tendencies at (lon faces, lat faces, cell centers, cell centers,
        dict of cell-center tracer tendencies).
    """
    u, v, T, p_s, phis = state.u, state.v, state.T, state.p_s, state.phis
    tracers = state.tracers

    R_d = constants.R_d
    kappa = constants.kappa

    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    # Positivity protections
    T = jnp.maximum(T, config.T_min)
    p_s = jnp.clip(p_s, config.p_floor, config.p_ceil)

    # --- 1. Pressure at full levels ---
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)
    else:
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)

    # --- 2. Geopotential via hydrostatic integration ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = compute_geopotential(T, p_s, sigma_coord, phis)

    # --- 3. KE at cell centres from C-grid face velocities ---
    u_c = _face_to_cell_u(u)
    v_c = _face_to_cell_v(v)
    KE = 0.5 * (u_c**2 + v_c**2)

    # --- 4. Bernoulli function B = Φ + KE ---
    B = Phi + KE

    # --- 5/6. Bernoulli + ln(p_s) gradients (batched at faces) ---
    # ``B`` is (n_lat, n_lon, nlev) and ``ln_ps`` is (n_lat, n_lon).
    # ``gradient_*_cgrid`` treats any trailing axis as a passive batch
    # (the per-lat ``cos_lat`` / ``dx_u`` metric broadcasts cleanly), so
    # we promote ``ln_ps`` to a single-level tensor and concatenate
    # along the level axis.  Each gradient runs once on the
    # (n_lat, n_lon, nlev+1) tensor; ``ln_ps`` claims the trailing slot.
    # 4 gradient calls collapse to 2 (one batched x + one batched y).
    ln_ps = jnp.log(p_s)
    n_lat_g, n_lon_g, nlev_g = B.shape
    _Bln_stack = jnp.concatenate(
        [B, ln_ps[..., jnp.newaxis]], axis=-1,
    )  # (n_lat, n_lon, nlev+1)
    _dBln_dx = gradient_x_cgrid(_Bln_stack, grid)  # (n_lat, n_lon+1, nlev+1)
    _dBln_dy = gradient_y_cgrid(_Bln_stack, grid)  # (n_lat+1, n_lon, nlev+1)
    dB_dx = _dBln_dx[..., :nlev_g]
    dB_dy = _dBln_dy[..., :nlev_g]
    dln_dx = _dBln_dx[..., nlev_g]   # squeeze trailing-1 → (n_lat, n_lon+1)
    dln_dy = _dBln_dy[..., nlev_g]

    # Cell→face interps: the v-face (latitude) direction uses the
    # halo-aware variant so a band's end faces — which are interior
    # partition cuts under latitude-band MPI, not poles — receive the
    # serial interior average across the cut (AD-safe sendrecv) instead
    # of the legacy pole edge-copy.  Serial / single-rank behaviour is
    # bit-identical (the helper falls back to ``interp_cell_to_vface``).
    # The u-face (longitude) direction never needs a halo variant: lon
    # is periodic and fully rank-local under band decomposition.
    # Fused entry-level lat pads (audit lever O4; atm census probe
    # 8460424, fusion-completion round 8463739): ONE sendrecv pair per
    # cut carries every ENTRY-KNOWN field needing v-face-interp ghost
    # rows — T, u (consumed twice: curl circulation + the
    # absolute-vorticity 4-pt u->v-face average), the layer thickness
    # dp (hybrid: dA+dB*p_s from entry; sigma: p_s*dsigma built here),
    # and on the hybrid lane also hybrid_factor and p_s.  Only the
    # sigma_dot / mass-flux v-interps stay separate: they depend on
    # div(dp*v), which needs the padded dp first.  Census (np=2 LL32
    # dry): 9 -> 6 per-step exchanges on the sigma path.  Serial/local
    # backend: per-field jnp.pads, value-identical (unused ones are
    # dead-code-eliminated).
    from legoesm.grids.halo_latlon import pad_with_pole_bc_lat_multi
    if _hybrid:
        # (B*p_s/p) pressure-gradient correction factor — entry-known.
        hybrid_factor = (
            sigma_coord.B_full * p_s[..., jnp.newaxis] / p_full
        )
        _ps3 = p_s[..., jnp.newaxis]
        (_T_lat_pad, _u_lat_pad, _dp_lat_pad, _hf_lat_pad,
         _ps_lat_pad) = pad_with_pole_bc_lat_multi(
            (T, u, dp, hybrid_factor, _ps3), halo=1)
    else:
        # Sigma-coord layer thickness, hoisted from the continuity
        # branch below so its ghost rows ride the entry exchange.
        # dtype-pinned to p_s: under the default fp32 PrecisionPolicy
        # the sigma arrays are float32 while the state is float64, and
        # a stray f32 dp SPLITS the fused exchange into two dtype
        # groups (census 8463821: fused[3f/2g|f64,f64,f32]) — one
        # extra sendrecv pair per cut per stage.  astype is a no-op
        # when dtypes already match.
        dp = p_s[..., jnp.newaxis] * sigma_coord.dsigma.astype(p_s.dtype)
        _T_lat_pad, _u_lat_pad, _dp_lat_pad = pad_with_pole_bc_lat_multi(
            (T, u, dp), halo=1)

    T_u = interp_cell_to_uface(T)
    T_v = interp_cell_to_vface_halo(T, f_pad=_T_lat_pad)

    pg_corr_x = R_d * T_u * dln_dx[:, :, jnp.newaxis]
    pg_corr_y = R_d * T_v * dln_dy[:, :, jnp.newaxis]

    # Hybrid coordinate correction: in sigma coords grad_eta(ln p) = grad(ln p_s),
    # but in hybrid coords grad_eta(ln p) = (B*p_s/p) * grad(ln p_s).
    if _hybrid:
        hf_u = interp_cell_to_uface(hybrid_factor)
        hf_v = interp_cell_to_vface_halo(hybrid_factor, f_pad=_hf_lat_pad)
        pg_corr_x = pg_corr_x * hf_u
        pg_corr_y = pg_corr_y * hf_v

    # --- 7. Momentum tendencies ---
    du_dt = -(dB_dx + pg_corr_x)
    dv_dt = -(dB_dy + pg_corr_y)

    # --- 8. Coriolis using absolute vorticity (ζ+f) ---
    cor_u, cor_v = absolute_vorticity_coriolis(
        u, v, grid, u_lat_pad=_u_lat_pad,
    )
    du_dt = du_dt + cor_u
    dv_dt = dv_dt + cor_v

    # --- 9. Surface pressure tendency and vertical motion ---
    # Both branches use flux-form continuity: sum_k div(dp_k * v).
    # This differs from the advective form p_s * div(v) when p_s has
    # horizontal gradients (which is always the case in practice).

    # Iter-54: precompute the cumsum of ``div_dp`` once and reuse it
    # both for ``D_total_p = sum(div_dp)`` (the surface-pressure
    # tendency) and for the sigma_dot / mass_flux integration below.
    # Saves one cross-shard reduction per RK3 stage in each branch on
    # any horizontal sharding (lat-lon mesh).
    if _hybrid:
        # Hybrid closure: dp = dA + dB * p_s varies horizontally.
        dp_u = interp_cell_to_uface(dp)  # (n_lat, n_lon+1, nlev)
        dp_v = interp_cell_to_vface_halo(  # (n_lat+1, n_lon, nlev)
            dp, f_pad=_dp_lat_pad)
        div_dp = divergence_cgrid(dp_u * u, dp_v * v, grid)  # (n_lat, n_lon, nlev)
        _cumsum_dp = jnp.cumsum(div_dp, axis=-1)  # (n_lat, n_lon, nlev)
        D_total_p = _cumsum_dp[..., -1]
        dp_s_dt = -D_total_p / sigma_coord.B_range
    else:
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top
        # Flux-form: div(dp_k * v); dp = p_s * dsigma_k built at the
        # fused entry pad above so its ghost rows shared the entry
        # exchange.
        dp_u = interp_cell_to_uface(dp)  # (n_lat, n_lon+1, nlev)
        dp_v = interp_cell_to_vface_halo(  # (n_lat+1, n_lon, nlev)
            dp, f_pad=_dp_lat_pad)
        div_dp = divergence_cgrid(dp_u * u, dp_v * v, grid)  # (n_lat, n_lon, nlev)
        _cumsum_dp = jnp.cumsum(div_dp, axis=-1)  # (n_lat, n_lon, nlev)
        D_total_p = _cumsum_dp[..., -1]
        dp_s_dt = -D_total_p / sigma_range

    # Apply zero-mean correction only when the post-step mass fixer is OFF.
    # When fix_mass=True the mass fixer already corrects the global integral,
    # and applying both creates a double-correction artifact.
    if config.zero_mean_ps_tendency and not config.fix_mass:
        dp_s_dt = zero_mean_tendency(dp_s_dt, grid)

    # --- 10. Vertical advection ---
    # Compute vertical advection directly at face positions to avoid the
    # smoothing from a cell-center round-trip.  sigma_dot/mass_flux are
    # interpolated to u-face and v-face locations first.
    if _hybrid:
        # Build mass flux from the corrected div(dp*v) closure (div_dp), NOT
        # from div(v)*dp which is what compute_mass_flux_hybrid's INTERNAL
        # div_dp uses.  The integration + boundary closure is the shared
        # compute_mass_flux_from_cumsum; we feed it the flux-form cumsum
        # precomputed above (iter-54 reuse — no extra cross-shard reduction).
        mass_flux = compute_mass_flux_from_cumsum(
            _cumsum_dp, D_total_p[..., jnp.newaxis], sigma_coord)
        mf_u = interp_cell_to_uface(mass_flux)
        # mass_flux depends on div(dp*v) -> cannot join the entry pad.
        mf_v = interp_cell_to_vface_halo(mass_flux)
        ps_u = interp_cell_to_uface(p_s[..., jnp.newaxis])[..., 0]
        ps_v = interp_cell_to_vface_halo(
            p_s[..., jnp.newaxis], f_pad=_ps_lat_pad)[..., 0]
        du_dt = du_dt + vertical_advection_hybrid(u, mf_u, ps_u, sigma_coord)
        dv_dt = dv_dt + vertical_advection_hybrid(v, mf_v, ps_v, sigma_coord)
        vert_adv_T = vertical_advection_hybrid(T, mass_flux, p_s, sigma_coord)
    else:
        # Flux-form sigma_dot consistent with mass-flux continuity:
        # σ̇_{k+1/2} = [frac_k · D_total_p - cumsum_k(div(dp·v))] / p_s
        _frac = sigma_coord.fractional_sigma  # (nlev,)
        # Iter-54: reuse the cumsum precomputed for D_total_p above.
        sigma_dot_inner = (
            _frac * D_total_p[..., jnp.newaxis] - _cumsum_dp
        ) / (p_s[..., jnp.newaxis] + 1e-10)
        # Pad with zero on the top boundary; one HLO Pad op vs zeros
        # buffer + concatenate.
        _pad_axes_sd = ((0, 0),) * (sigma_dot_inner.ndim - 1) + ((1, 0),)
        sigma_dot = jnp.pad(sigma_dot_inner, _pad_axes_sd)
        sd_u = interp_cell_to_uface(sigma_dot)
        sd_v = interp_cell_to_vface_halo(sigma_dot)
        du_dt = du_dt + vertical_advection(u, sd_u, sigma_coord)
        dv_dt = dv_dt + vertical_advection(v, sd_v, sigma_coord)
        vert_adv_T = vertical_advection(T, sigma_dot, sigma_coord)

    # --- 11. Temperature equation ---
    if config.use_ppm_transport:
        # C-grid PPM advection of T (4th-order, shared operator)
        horiz_adv_T = cgrid_fv_scalar_advection_latlon_3d(T, u, v, grid)
    else:
        # Cell-centered gradient advection (fallback) — 3D-native variants
        # share one halo pad + PPM reconstruction across all levels.
        # Pre-pad T once so both gradient calls share the halo.
        _T_pad_h2 = pad_halo_latlon_3d(T, halo=2)
        # iter-169: use the imported aliases (lines 92-93) — bare
        # ``fv_gradient_lon_3d`` would F821 NameError at runtime.
        dT_dx = _fv_gradient_lon_3d(T, grid, padded=_T_pad_h2)
        dT_dy = _fv_gradient_lat_3d(T, grid, padded=_T_pad_h2)
        horiz_adv_T = -(u_c * dT_dx + v_c * dT_dy)

    # Adiabatic heating: κ T (ω/p + v·∇_η(ln p))
    # The v·∇_η(ln p) term is the horizontal pressure-gradient correction
    # to the thermodynamic equation (Simmons & Burridge 1981).
    if _hybrid:
        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt, sigma_coord)
    else:
        omega = compute_pressure_velocity(sigma_dot, p_s, dp_s_dt, sigma_coord)
    # Tiny epsilon prevents division by zero without suppressing physics
    # at the model top (the old p_floor=100 Pa clamp distorted heating
    # for all levels with p < 100 Pa).
    adiabatic = kappa * T * omega / (p_full + 1e-10)

    # v · grad(ln p_s) at cell centres (average face gradients to centres)
    dln_dx_cc = _face_to_cell_u(
        jnp.broadcast_to(dln_dx[:, :, jnp.newaxis], u.shape))
    dln_dy_cc = _face_to_cell_v(
        jnp.broadcast_to(dln_dy[:, :, jnp.newaxis], v.shape))
    v_dot_grad_lnps = u_c * dln_dx_cc + v_c * dln_dy_cc
    # In sigma coords: grad_eta(ln p) = grad(ln p_s).
    # In hybrid coords: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s).
    if _hybrid:
        v_dot_grad_lnps = v_dot_grad_lnps * (
            sigma_coord.B_full * p_s[..., jnp.newaxis] / (p_full + 1e-10))
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    dT_dt = horiz_adv_T + vert_adv_T + adiabatic

    # --- 12. Tracer transport (mass-weighted flux form) ---
    # Uses the same mass fluxes (dp_u*u, dp_v*v) as the continuity
    # equation for discrete consistency.  The mixing-ratio tendency is:
    #   dq/dt = [-div(dp·q·v) + q·div(dp·v)] / dp - vert_advection
    # This conserves ∫ q·dp·dA (tracer mass) to machine precision.
    #
    # All tracers share the same u, v, dp, div_dp.  Stack along a trailing
    # tracer axis and fold it into the level axis so the operator's
    # halo-pad + PPM reconstruction runs once instead of n_tracers times.
    # ``pad_halo_latlon_3d`` only touches axes 0 and 1, so the trailing
    # ``nlev * n_tracers`` axis is passively carried through.
    tracer_tends = {}
    if tracers:
        tracer_names = list(tracers.keys())
        n_tracers = len(tracer_names)
        tracer_stack = jnp.stack(
            [tracers[n] for n in tracer_names], axis=-1
        )  # (n_lat, n_lon, nlev, n_tracers)
        n_lat_t, n_lon_t, nlev_t, _ = tracer_stack.shape
        tracer_flat = tracer_stack.reshape(n_lat_t, n_lon_t, nlev_t * n_tracers)

        if config.use_ppm_transport:
            # Mass-weighted flux-form horizontal transport.
            u_mass = dp_u * u  # (n_lat, n_lon+1, nlev)
            v_mass = dp_v * v  # (n_lat+1, n_lon, nlev)
            if n_tracers == 1:
                u_mass_b, v_mass_b = u_mass, v_mass
            else:
                # ``tracer_flat`` reshape interleaves levels and tracers as
                # ``[lev0/trc0, lev0/trc1, ..., lev1/trc0, ...]`` — each level
                # has all tracers consecutive.  ``jnp.repeat`` builds a
                # matching velocity broadcast where every level value is
                # duplicated ``n_tracers`` times.  ``jnp.tile`` would instead
                # concatenate the whole array and mis-align tracer ↔ level.
                u_mass_b = jnp.repeat(u_mass, n_tracers, axis=-1)
                v_mass_b = jnp.repeat(v_mass, n_tracers, axis=-1)
            flux_flat = cgrid_fv_flux_divergence_latlon_3d(
                tracer_flat, u_mass_b, v_mass_b, grid)
            flux_stack = flux_flat.reshape(n_lat_t, n_lon_t, nlev_t, n_tracers)
            horiz_q_stack = (
                flux_stack + tracer_stack * div_dp[..., None]
            ) / (dp[..., None] + 1e-10)
        else:
            # Pre-pad the stacked tracer field once so both gradients
            # share the halo pad — saves one redundant pad_halo_latlon_3d
            # call per timestep.
            _q_pad_h2 = pad_halo_latlon_3d(tracer_flat, halo=2)
            # iter-169: aliased import (line 92-93).
            dq_dx_flat = _fv_gradient_lon_3d(tracer_flat, grid, padded=_q_pad_h2)
            dq_dy_flat = _fv_gradient_lat_3d(tracer_flat, grid, padded=_q_pad_h2)
            dq_dx_stack = dq_dx_flat.reshape(n_lat_t, n_lon_t, nlev_t, n_tracers)
            dq_dy_stack = dq_dy_flat.reshape(n_lat_t, n_lon_t, nlev_t, n_tracers)
            horiz_q_stack = -(
                u_c[..., None] * dq_dx_stack + v_c[..., None] * dq_dy_stack
            )

        # Vertical advection batched across all tracers — both
        # ``vertical_advection_hybrid`` and ``vertical_advection``
        # operate on ``axis=-1`` for the vertical, so move the tracer
        # axis to leading where the velocity-independent shared work
        # (``F_full`` / ``p_full`` for hybrid, ``F`` for sigma) is
        # computed *once* and the upwind ``jnp.diff(field, axis=-1)``
        # broadcasts across the (n_tracers,) axis.  Replaces a Python
        # for-loop that called the operator ``n_tracers`` times.
        # Same leading-axis batching as the (u, v) vertical advection
        # in CD-grid CE/PE (Loop 142).
        tracers_lead = jnp.moveaxis(
            tracer_stack, -1, 0,
        )  # (n_tracers, n_lat, n_lon, nlev)
        if _hybrid:
            vert_q_lead = vertical_advection_hybrid(
                tracers_lead, mass_flux, p_s, sigma_coord,
            )
        else:
            vert_q_lead = vertical_advection(
                tracers_lead, sigma_dot, sigma_coord,
            )
        for i, name in enumerate(tracer_names):
            tracer_tends[name] = horiz_q_stack[..., i] + vert_q_lead[i]

    # --- 13. Diffusion (optional) ---
    if config.A_h > 0.0:
        lap_u, lap_v = vector_laplacian_cgrid(u, v, grid)
        du_dt = du_dt + config.A_h * lap_u
        dv_dt = dv_dt + config.A_h * lap_v
        lap_T = laplacian_cgrid(T, grid)
        dT_dt = dT_dt + config.A_h * lap_T

    # --- 13b. Top sponge (Rayleigh damping increasing toward the lid, #836) ---
    # Absorb upward-propagating gravity-wave / convective energy that would else
    # reflect off the rigid model lid and contaminate the upper levels.  Gated on
    # config.sponge_coeff (0 -> OFF, byte-identical).  Reuse the SHARED
    # sponge_profile (no re-derivation): it ramps 0 -> sponge_coeff over the top
    # sponge_width_m metres, so feed it a log-pressure height proxy
    # z = -H_scale * ln(sigma_full), which increases UPWARD (small sigma = high
    # altitude).  SIGN: du/dt gets a -k*u term with k >= 0 -> damps u toward REST
    # (an absorbing sponge, matching the compressible core's -sponge*w); level-
    # only profile broadcasts over the horizontal (u/v have level as the last
    # axis).  OFF path adds nothing, so production stays bit-for-bit unchanged.
    if config.sponge_coeff > 0.0:
        from legoesm.atmosphere.dynamics.compressible_euler import (
            sponge_profile,
        )
        z_full = -config.sponge_scale_height_m * jnp.log(
            jnp.clip(sigma_coord.sigma_full, 1e-30, None))  # (nlev,), up = large z
        H_top = z_full[0]   # top level has the smallest sigma -> the largest z
        spge = sponge_profile(
            z_full, H_top, config.sponge_width_m, config.sponge_coeff,
            shape=config.sponge_shape,
        ).astype(du_dt.dtype)   # (nlev,), 0 below the sponge base
        if config.sponge_eddy_only:
            # Zonal-mean-preserving (eddy-only) damping: no net Coriolis
            # torque, no poleward mass drift (see the config docstring).
            # Zonal mean over the lon axis (axis 1); u lives on n_lon+1
            # interfaces whose periodic duplicate biases the mean by
            # O(1/n_lon) — irrelevant for a damping reference.
            u_zm = jnp.mean(u, axis=1, keepdims=True)
            v_zm = jnp.mean(v, axis=1, keepdims=True)
            du_dt = du_dt - spge * (u - u_zm)
            dv_dt = dv_dt - spge * (v - v_zm)
        else:
            du_dt = du_dt - spge * u
            dv_dt = dv_dt - spge * v

    # Enforce zero tendency at poles (wall BC) so that intermediate RK
    # stages never see nonzero v at poles feeding into divergence/Coriolis.
    # Rank-aware via config.pole_v_bc — under latitude-band MPI, only the
    # boundary ranks zero the actual global poles; interior ranks pass
    # through unchanged (their band-edge v-faces are shared with the
    # neighbour rank and kept consistent by halo exchange).
    #
    # Under single-program lat-band SPMD the SAME compiled body runs on every
    # band, so config.pole_v_bc is (True, True) on ALL bands — the static
    # _zero_v_at_pole would then zero every INTERIOR band cut's v-row (the
    # SPMD-blind bug: the cut v-tendency was wrong by ~1e-3 while interior
    # rows were machine-exact).  Select the physical-pole zeroing per band via
    # the traced axis_index masks; interior cuts pass through so their cut row
    # keeps the cross-band meridional v-tendency (function-scope import:
    # atmosphere -> core.parallel; returns None off the SPMD backend).
    from legoesm.parallel.latlon_spmd import spmd_pole_end_masks
    _spmd_pm = spmd_pole_end_masks()
    if _spmd_pm is not None:
        from legoesm.parallel.latlon_spmd import apply_pole_end_masks
        dv_dt = apply_pole_end_masks(
            dv_dt, _spmd_pm, offset=config.pole_v_bc_offset)
    else:
        dv_dt = _zero_v_at_pole(
            dv_dt,
            south=config.pole_v_bc[0],
            north=config.pole_v_bc[1],
            offset=config.pole_v_bc_offset,
        )

    return du_dt, dv_dt, dT_dt, dp_s_dt, tracer_tends


# ==============================================================================
# Model class
# ==============================================================================

class CGridLatLonPrimitiveEquationModel(IntegrationMixin):
    """C-grid hydrostatic PE model on the latitude-longitude grid.

    Features:
    - Compact-stencil C-grid operators (no checkerboard mode).
    - Sadourny (1975) energy-conserving Coriolis.
    - Simmons-Burridge geopotential integration.
    - Conservative C-grid divergence for continuity.
    - Wall BC at poles (v = 0), periodic longitude.
    - Adapter to/from cell-centered HydrostaticState for physics coupling.

    Parameters
    ----------
    grid : LatLonGrid
    sigma_coord : SigmaCoordinate
    config : CGridLatLonPrimitiveEquationConfig, optional
    """

    def __init__(
        self,
        grid: LatLonGrid,
        sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
        config: CGridLatLonPrimitiveEquationConfig | None = None,
        dt: float = 600.0,
    ):
        self.grid = grid
        self.sigma_coord = sigma_coord
        self.config = config or CGridLatLonPrimitiveEquationConfig()

        # Stash the constructor ``dt`` so callers downstream (notably
        # ``make_latlon_mpi_step`` rebuilding an MPI-aware model) can
        # recover it without falling back to the pole-cell ``_max_dt``
        # — Codex review Stage 3-E round 3 BLOCK caught the fallback
        # silently using ``_max_dt`` on direct-construction paths
        # where ``component_factory`` did not set ``effective_dt``.
        self.dt = float(dt)

        # Pole-cell CFL limit: dx_pole is the smallest cell on the grid.
        dx_pole = pole_cell_dx(grid)
        self._max_dt = cfl_max_dt(dx_pole, 300.0, cfl_number=0.8, ndim=1)

        # Precompute polar filter masks: one for cell-centered fields
        # (dT, dps, du after lon-trim, every tracer) and one for v-face
        # fields (dv).  The v-face mask is built against
        # ``grid.cos_lat_v`` + the half-cell-offset lat-interface
        # coordinates so the wavenumber cutoff matches the actual
        # v-face CFL — using the cell-centered mask on v-face indices
        # admits k modes the v-face CFL forbids (Codex review Stage
        # 3-E round 2 BLOCK #1).
        if self.config.use_polar_filter:
            self._polar_mask = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
            )
            self._polar_mask_v = compute_polar_filter_mask(
                grid, dt=dt,
                max_wave_speed=self.config.polar_filter_max_wave_speed,
                cutoff_lat_deg=self.config.polar_filter_cutoff_deg,
                is_v_face=True,
            )
        else:
            self._polar_mask = None
            self._polar_mask_v = None

        # Cache for the last C-grid output state.  Keyed on Python id()
        # of (u, v, T, p_s, phis, tracers) arrays in the HydrostaticState.
        # When the returned state is passed back unmodified, the cache
        # avoids a lossy cell→face re-projection of winds.  However, the
        # cache is invalidated whenever ANY field is replaced (e.g. by
        # driver friction/smoothing, checkpoint restore, or _replace()).
        # In practice this means the cache only helps consecutive
        # step_with_physics() calls that do NOT modify the returned state
        # between steps.
        self._cgrid_cache_key: tuple | None = None
        self._cgrid_cache: CGridLatLonHydrostaticState | None = None

        # Operator-split physics carry (issue #413, mirrors the MPAS and
        # CDGrid PEs): ``step(..., phys_state=...)`` stashes the updated
        # ``PhysicsState`` here for the caller to feed back next step,
        # while ``step`` keeps returning the state only.
        self._phys_state = None

        # Anchored mass target (lazy: filled on first step when
        # ``anchor_mass_to_initial`` is True). Mirrors the cubed-sphere
        # ``CDGridPrimitiveEquationModel`` pattern so the per-step
        # additive correction is taken against an unchanging fp64 scalar
        # instead of the (potentially fp32-rounded) previous-step mass.
        self._target_mass: jax.Array | None = None

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-18; see iter-4 SW twin)."""
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-19)."""
        self._target_mass = target_mass

    def compute_mass(self, state: CGridLatLonHydrostaticState, grid=None) -> jax.Array:
        """Compute total mass (for conservation fixer target).

        Iter-12: use the fp64 budget accumulator unconditionally.
        ``_accumulation_dtype`` is fp32 under the default storage
        policy, so a 16k-cell reduction leaked ~N·eps noise into the
        anchored target and blocked sub-fp32 conservation even after
        iter-2's anchor wiring.
        """
        acc = conservation_accumulator()
        _grid = self.grid if grid is None else grid
        return jnp.sum(state.p_s.astype(acc) * _grid.area.astype(acc))

    def tendencies(self, state: CGridLatLonHydrostaticState):
        return cgrid_latlon_hydrostatic_tendencies(
            state, self.grid, self.sigma_coord, self.config,
        )

    # ------------------------------------------------------------------
    # Internal C-grid stepping (raw-array CGridLatLonHydrostaticState)
    # ------------------------------------------------------------------

    def _call_physics(self, physics_fn, hs, phys_state=None, *,
                      grid=None, sigma_coord=None):
        """Call physics_fn with the correct contract and unwrap tuples.

        Supports both calling conventions used in the repo:
        - Legacy 3-arg: ``physics_fn(state, grid, sigma_coord)``
        - 1-arg closure: ``physics_fn(state)``

        When *phys_state* is supplied the 4-arg carry contract
        ``physics_fn(state, grid, sigma_coord, phys_state)`` is used
        (issue #413); a 3-arg-only physics_fn then fails loudly rather
        than silently dropping the carry.

        ``grid`` / ``sigma_coord`` default to ``self.*`` (serial). The lat-band
        SPMD body passes the BAND geometry so a lat-dependent physics (e.g.
        Held-Suarez ``T_eq``/``k_T``) sees the band's latitudes, not the global
        ones (a shape/value mismatch otherwise). Column-local physics is
        otherwise decomposition-invariant.

        Also unwraps ``(tendencies, aux)`` tuple returns from
        PhysicsModuleProtocol-style callables.
        """
        result = self._call_physics_raw(
            physics_fn, hs, phys_state, grid=grid, sigma_coord=sigma_coord)
        if type(result) is tuple:
            return result[0]
        return result

    def _call_physics_raw(self, physics_fn, hs, phys_state=None, *,
                          grid=None, sigma_coord=None):
        """As :meth:`_call_physics` but WITHOUT unwrapping the result —
        the carry-out evaluation needs ``result[1]``. ``grid`` / ``sigma_coord``
        default to ``self.*`` (serial / single-rank); the SPMD body passes the
        BAND grid so per-band physics is correct."""
        _grid = self.grid if grid is None else grid
        _sigma = self.sigma_coord if sigma_coord is None else sigma_coord
        if phys_state is not None:
            return physics_fn(hs, _grid, _sigma, phys_state)
        sig = inspect.signature(physics_fn)
        n_params = len(sig.parameters)
        if n_params >= 3:
            return physics_fn(hs, _grid, _sigma)
        return physics_fn(hs)

    @partial(jax.jit, static_argnums=(0, 4))
    def _step_cgrid(
        self,
        state: CGridLatLonHydrostaticState,
        dt: float,
        target_mass: jax.Array | None = None,
        physics_fn=None,
        phys_state=None,
    ) -> tuple:
        """Internal (JITTED): advance one step on C-grid state (raw arrays).

        Thin wrapper over :meth:`_step_cgrid_impl` with the model's own grid /
        sigma_coord / polar masks. The lat-band SPMD body calls
        ``_step_cgrid_impl`` directly with the BAND geometry (un-jitted, so the
        band grid stays a concrete value rather than a tracer).
        """
        return self._step_cgrid_impl(
            state, dt, target_mass, physics_fn, phys_state,
        )

    def _step_cgrid_impl(
        self,
        state: CGridLatLonHydrostaticState,
        dt: float,
        target_mass: jax.Array | None = None,
        physics_fn=None,
        phys_state=None,
        *,
        grid=None,
        sigma_coord=None,
        polar_mask=None,
        polar_mask_v=None,
        pole_v_bc_masks=None,
    ) -> tuple:
        """Advance one step on C-grid state (raw arrays). UN-jitted.

        Physics is evaluated inside each RK stage (matching the CDGrid
        PE contract), not as a post-step Euler update.  Every stage
        receives the STEP-INPUT ``phys_state``; the carry-out comes
        from one extra physics evaluation on the post-step state (see
        :meth:`step`).  Returns ``(state_new, phys_state_out)``.

        ``grid`` / ``sigma_coord`` / ``polar_mask`` / ``polar_mask_v`` default
        to ``self.*`` (serial / single-rank).  The lat-band SPMD wrapper passes
        the BAND geometry + band polar masks so the whole step runs on the band;
        un-jitted because a nested ``jax.jit`` would trace ``grid`` as a tracer
        and crash the operators' trace-time static ``if`` / pole constructions.
        """
        if grid is None:
            grid = self.grid
        if sigma_coord is None:
            sigma_coord = self.sigma_coord
        if polar_mask is None:
            polar_mask = self._polar_mask
        if polar_mask_v is None:
            polar_mask_v = self._polar_mask_v
        state_c = cast_pytree(state, None, "compute")

        def tendency_fn(s):
            du, dv, dT, dps, dq = cgrid_latlon_hydrostatic_tendencies(
                s, grid, sigma_coord, self.config,
            )

            # --- Physics coupling (inside RK stage) ---
            if physics_fn is not None:
                hs = cgrid_to_hydrostatic(s, grid)
                phys_tend = self._call_physics(
                    physics_fn, hs, phys_state,
                    grid=grid, sigma_coord=sigma_coord)

                dT = dT + phys_tend.dT_dt.data
                dps = dps + phys_tend.dp_s_dt.data

                du_phys = phys_tend.du_dt.data
                dv_phys = (phys_tend.dv_dt.data
                           if phys_tend.dv_dt is not None
                           else jnp.zeros_like(du_phys))
                # Cell→face coupling of physics wind tendencies.
                # v-face: halo-aware — under latitude-band MPI the
                # band's end rows are interior partition cuts (NOT
                # poles), so the legacy end-row copy of
                # ``interp_cell_to_vface`` would diverge from the
                # serial average 0.5*(dv_phys[j-1] + dv_phys[j]) at
                # the cut faces.  ``interp_cell_to_vface_halo`` pads
                # one lat row through the backend-dispatched
                # ``pad_with_pole_bc_lat`` (AD-safe ``_sendrecv_vjp``
                # sendrecv at cuts) and is bit-identical to the legacy
                # convention in serial / at true poles.
                # u-face: no halo needed — lon is periodic and fully
                # rank-local under band decomposition, so the legacy
                # wrap interp already matches serial row-by-row.
                du = du + interp_cell_to_uface(du_phys)
                dv = dv + interp_cell_to_vface_halo(dv_phys)

                # Physics tracer tendencies (only for tracers already in state;
                # introducing new tracer keys here would break the RK
                # integrator's pytree structure).
                if phys_tend.tracer_tendencies is not None:
                    for name, dq_field in phys_tend.tracer_tendencies.items():
                        if name in dq:
                            dq[name] = dq[name] + dq_field.data

            # Polar filter: damp high-frequency modes near poles for
            # EVERY transported quantity (dT, dps, du, dv, every dq).
            # Filtering only dT/dps/du leaves dv + tracers running at
            # full explicit resolution near the poles, so the lifted
            # equatorial-CFL dt (which only the filtered fields can
            # tolerate) would crash on the unfiltered transport.
            # Codex review of Stage 3-E BLOCK #1 + #2 caught this.
            if polar_mask is not None:
                dT = fourier_filter_3d(dT, grid, polar_mask)
                dps = fourier_filter(dps, grid, polar_mask)

                # u: lon-interface, shape (n_lat, n_lon+1, nlev).  Drop
                # the duplicated last lon column, filter, then restore
                # the periodicity column from the filtered first column.
                du_int = fourier_filter_3d(du[:, :-1, :], grid, polar_mask)
                du = jnp.concatenate([du_int, du_int[:, 0:1, :]], axis=1)

                # v: lat-interface, shape (n_lat+1, n_lon, nlev).  Use
                # the v-face mask (precomputed in __init__ against
                # cos_lat_v) so EVERY v-face row is filtered — not
                # only the first n_lat rows.  Codex review Stage 3-E
                # round 2 caught: under lat-band MPI, interior ranks'
                # ``dv[-1]`` is NOT a pole row (it's a shared v-face
                # with the northern neighbour) so re-appending it
                # unfiltered would leak an unfiltered perturbation
                # into v at each step.
                dv = fourier_filter_3d(dv, grid, polar_mask_v)

                # Tracers: each transported tracer has the same
                # (n_lat, n_lon, nlev) shape as dT, so the same mask
                # applies directly.  Without this loop, the lifted
                # equatorial-CFL dt would race the (un-filtered)
                # polar tracer advection past its CFL.
                if dq:
                    dq = {
                        name: fourier_filter_3d(dq_field, grid, polar_mask)
                        for name, dq_field in dq.items()
                    }

            return CGridLatLonHydrostaticState(
                u=du, v=dv, T=dT, p_s=dps,
                phis=jnp.zeros_like(s.phis),
                tracers=dq,
            )

        state_new = dispatch_integrator(
            state_c, tendency_fn, dt, self.config.time_integrator,
        )

        # Enforce v = 0 at poles via the rank-aware helper (matches the
        # tendency-side rewrite above).  Serial config has
        # ``pole_v_bc=(True, True)`` so this preserves bit-exact
        # behaviour with the pre-refactor single-Pad implementation.
        if pole_v_bc_masks is not None:
            # Lat-band SPMD: zero v at the PHYSICAL pole faces only (the south
            # band's bottom / north band's top), selected DATA-dependently per
            # band — an interior cut's end v-face is a shared interior face and
            # must NOT be walled.  ``_zero_v_at_pole``'s static ``if south`` /
            # fast-path cannot take traced masks, so apply the wall via
            # ``jnp.where`` over the same offset rows.
            _south_m, _north_m = pole_v_bc_masks
            _off = self.config.pole_v_bc_offset
            v_new = state_new.v
            _n = v_new.shape[0]
            v_new = jnp.where(
                _south_m, v_new.at[_off].set(jnp.zeros_like(v_new[_off])), v_new)
            v_new = jnp.where(
                _north_m,
                v_new.at[_n - 1 - _off].set(jnp.zeros_like(v_new[_n - 1 - _off])),
                v_new)
        else:
            # Enforce v = 0 at poles via the rank-aware helper.  Serial config
            # has ``pole_v_bc=(True, True)`` so this preserves bit-exact
            # behaviour with the pre-refactor single-Pad implementation.
            v_new = _zero_v_at_pole(
                state_new.v,
                south=self.config.pole_v_bc[0],
                north=self.config.pole_v_bc[1],
                offset=self.config.pole_v_bc_offset,
            )
        state_new = state_new._replace(v=v_new)

        # Safety rails: T floor, p_s floor, mass fixer (band grid + global
        # area denominator under SPMD).
        state_new = self._apply_safety_rails(
            state_new, target_mass, state, grid=grid, sigma_coord=sigma_coord)

        state_out = cast_pytree(state_new, None, "storage")

        # Operator-split physics carry (issue #413): one extra physics
        # evaluation on the POST-STEP state yields the carry-out
        # (tendencies discarded) — prognostic fields advance exactly
        # once per dt, consistent with the returned state.  RK-weight
        # combination of carries would corrupt replacement-semantics
        # values (e.g. the implicit TKE solve).  Never traced when no
        # carry is threaded (byte-identical legacy path).
        phys_state_out = phys_state
        if physics_fn is not None and phys_state is not None:
            _pr = self._call_physics_raw(
                physics_fn, cgrid_to_hydrostatic(state_out, grid),
                phys_state, grid=grid, sigma_coord=sigma_coord,
            )
            if type(_pr) is tuple and len(_pr) > 1:
                phys_state_out = _pr[1]

        return state_out, phys_state_out

    # ------------------------------------------------------------------
    # Public API — matches the driver contract
    # ------------------------------------------------------------------

    def _apply_safety_rails(
        self,
        state: CGridLatLonHydrostaticState,
        target_mass: jax.Array | None = None,
        pre_state: CGridLatLonHydrostaticState | None = None,
        *,
        grid=None,
        sigma_coord=None,
    ) -> CGridLatLonHydrostaticState:
        """Apply T_min floor, p_floor clamp, and mass fixer.

        Called after dynamics and again after physics to ensure safety
        invariants hold regardless of what physics tendencies produce.

        ``grid`` / ``sigma_coord`` default to ``self.*`` (serial / single-rank).
        The lat-band SPMD body passes the BAND grid (with a GLOBAL
        ``grid_total_area`` denominator kept global by the band slicer + the
        ``batch_global_area_sums`` "lat"-psum) so the mass fixer's numerator
        AND denominator stay global across bands.
        """
        if grid is None:
            grid = self.grid
        if sigma_coord is None:
            sigma_coord = self.sigma_coord
        # Temperature floor
        T_new = jnp.maximum(state.T, self.config.T_min)
        state = state._replace(T=T_new)

        # Surface pressure positivity
        p_s_new = jnp.maximum(state.p_s, self.config.p_floor)
        state = state._replace(p_s=p_s_new)

        # Conservation fixer for mass — fp64 budget accumulator
        # (iter-12 mirrors compute_mass; see docstring there).
        if self.config.fix_mass:
            acc = conservation_accumulator()
            # ``grid_total_area`` is a precomputed scalar on the grid;
            # avoids recomputing ``jnp.sum(area)`` every step (one
            # extra reduction in serial, one extra allreduce under
            # latlon SPMD sharding).
            total_area = grid.grid_total_area.astype(acc)
            if target_mass is not None:
                # Closure-constant target → only ``mass_new`` is reduced.
                area = grid.area.astype(acc)
                mass_target = target_mass
                mass_new = jnp.sum(state.p_s.astype(acc) * area)
            elif pre_state is not None:
                # Iter-57: batch the two area-weighted sums into a
                # single MPI allreduce / cross-shard reduction (the
                # cubed-sphere ``fix_mass_hydrostatic`` already does
                # this via ``batch_global_area_sums``).  Under lat-band SPMD
                # this psum's across the "lat" axis (the global numerator).
                mass_target, mass_new = batch_global_area_sums(
                    [pre_state.p_s, state.p_s], grid,
                )
            else:
                # Degenerate case: mass_target == mass_new → correction=0.
                # Skip the redundant second reduction.
                area = grid.area.astype(acc)
                mass_new = jnp.sum(state.p_s.astype(acc) * area)
                mass_target = mass_new
            correction = (mass_target - mass_new) / total_area

            # Preserve tracer mass: ∫ q·dp·dA must be invariant when the
            # mass fixer adjusts p_s.  Scale q by dp_pre / dp_post per
            # level.  For pure sigma dp = p_s·dσ so the ratio is just
            # p_s_pre/p_s_post, but for hybrid coords dp = dA + dB·p_s
            # and the ratio differs per level (upper levels with large dA
            # barely change).
            p_s_pre = state.p_s
            # Iter-2: keep ``correction`` at fp64 (no ``astype`` to
            # ``p_s_pre.dtype``) so the additive fixer matches the
            # cubed-sphere ``fix_ps_mass`` semantics. JAX promotes the
            # sum to fp64; ``cast_pytree(..., "storage")`` at the end
            # of ``_step_cgrid`` rounds back to storage dtype, but the
            # correction is applied before that single round-trip
            # instead of compounding fp32 quantization every step.
            p_s_post = jnp.maximum(
                p_s_pre + correction,
                self.config.p_floor,
            )
            if state.tracers:
                _hybrid = isinstance(
                    sigma_coord, HybridSigmaPressureCoordinate)
                if _hybrid:
                    dp_pre = dp_from_hybrid(sigma_coord, p_s_pre)
                    dp_post = dp_from_hybrid(sigma_coord, p_s_post)
                else:
                    dsigma = sigma_coord.dsigma
                    dp_pre = p_s_pre[..., jnp.newaxis] * dsigma
                    dp_post = p_s_post[..., jnp.newaxis] * dsigma
                ratio = dp_pre / (dp_post + 1e-10)  # (n_lat, n_lon, nlev)
                new_tracers = {
                    name: q * ratio
                    for name, q in state.tracers.items()
                }
                state = state._replace(tracers=new_tracers)

            state = state._replace(p_s=p_s_post)

        return state

    def step(
        self,
        state,
        dt: float,
        target_mass: jax.Array | None = None,
        physics_fn=None,
        phys_state=None,
    ):
        """Advance one time step.

        Pure with respect to the explicit state argument — always
        advances the supplied state, never a cached copy.

        Accepts either ``CGridLatLonHydrostaticState`` (native C-grid,
        raw arrays) or ``HydrostaticState`` (cell-centred, Field members).
        Returns the same type as the input.

        Physics is evaluated inside each RK stage (matching CDGrid PE),
        not as a post-step Euler update.

        Parameters
        ----------
        state : CGridLatLonHydrostaticState or HydrostaticState
        dt : float
        target_mass : jax.Array or None
            If provided, the mass fixer corrects to this target.
        physics_fn : callable or None
            Evaluated at each RK stage alongside dynamics (3-arg legacy
            or 1-arg closure, tuple returns unwrapped).
        phys_state : PhysicsState or None
            Operator-split physics carry (issue #413, mirrors the MPAS
            and CDGrid PEs).  Every RK stage's physics evaluation
            receives this STEP-INPUT carry; the carry-out comes from
            one extra physics evaluation on the post-step state and is
            stashed on ``self._phys_state`` for the caller to feed
            back next step.  ``None`` (default) is byte-identical to
            the legacy path.

        """
        refuse_unthreaded_stateful_physics(
            physics_fn, phys_state, where="lat-lon C-grid step()")
        # Anchor-to-initial: snapshot mass once outside JIT (mirrors
        # primitive_eq_cdgrid.step()).  Computed in fp64 via
        # ``compute_mass`` so it stays clean of the per-step
        # ``cast_pytree(..., "storage")`` round-trip and the runner-side
        # fp32 reduction noise that iter-1 cleaned out of the diagnostic.
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None
                and target_mass is None):
            if isinstance(state, CGridLatLonHydrostaticState):
                self._target_mass = self.compute_mass(state)
            else:
                # HS path: cell-centred Field state.  ``compute_mass``
                # expects a CGrid state, but the integral is the same
                # area-weighted sum of p_s.  Iter-12: fp64 budget acc.
                acc = conservation_accumulator()
                self._target_mass = jnp.sum(
                    state.p_s.data.astype(acc) * self.grid.area.astype(acc)
                )
        if (target_mass is None
                and self.config.fix_mass
                and self.config.anchor_mass_to_initial):
            target_mass = self._target_mass

        if isinstance(state, CGridLatLonHydrostaticState):
            state_new, self._phys_state = self._step_cgrid(
                state, dt, target_mass, physics_fn, phys_state)
            return state_new

        # HydrostaticState input: use cached face-staggered winds from
        # the previous step to avoid lossy cell→face re-projection.
        # Cache is keyed on identity (Python id()) of ALL state fields
        # including tracers.  Any field replacement invalidates.
        cache_key = self._hs_cache_key(state)
        if (self._cgrid_cache is not None
                and self._cgrid_cache_key == cache_key):
            cgrid_in = self._cgrid_cache
        else:
            cgrid_in = hydrostatic_to_cgrid(state, self.grid)

        cgrid_out, self._phys_state = self._step_cgrid(
            cgrid_in, dt, target_mass, physics_fn, phys_state)

        # Cache the output and convert to HydrostaticState
        hs_out = cgrid_to_hydrostatic(cgrid_out, self.grid)
        self._cgrid_cache = cgrid_out
        self._cgrid_cache_key = self._hs_cache_key(hs_out)
        return hs_out

    @staticmethod
    def _hs_cache_key(hs) -> tuple:
        """Build a cache key from all HydrostaticState array identities.

        Includes tracer names in the key so schema changes (renaming/
        adding/removing tracers) invalidate the cache.
        """
        key = [id(hs.u.data), id(hs.v.data),
               id(hs.T.data), id(hs.p_s.data),
               id(hs.phis.data)]
        if hs.tracers is not None:
            names = sorted(hs.tracers)
            key.append(tuple(names))  # schema part
            for name in names:
                f = hs.tracers[name]
                key.append(id(f.data if hasattr(f, 'data') else f))
        return tuple(key)

    def step_with_physics(self, state, dt, physics_fn=None,
                          phys_state=None):
        """Advance one step, optionally applying physics.

        This is the entry point used by ``ModelDriver``.  It accepts
        both ``HydrostaticState`` and ``CGridLatLonHydrostaticState``
        and returns the same type as the input.  ``physics_fn=None``
        runs dynamics only (no physics).  ``phys_state`` threads the
        operator-split physics carry (issue #413); the updated carry
        is stashed on ``self._phys_state``.
        """
        return self.step(state, dt, physics_fn=physics_fn,
                         phys_state=phys_state)
