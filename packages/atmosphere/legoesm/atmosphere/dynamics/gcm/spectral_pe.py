"""Spectral Primitive Equation Model using spherical harmonic transforms.

Solves the hydrostatic primitive equations on the sphere using the
vorticity-divergence formulation with pseudospectral (transform) method
in sigma-pressure coordinates.

Prognostic variables (in spectral space):
    vor_hat  : Relative vorticity SH coefficients, (n_sh, nlev)
    div_hat  : Divergence SH coefficients, (n_sh, nlev)
    T_hat    : Temperature SH coefficients, (n_sh, nlev)
    lnps_hat : Log(surface pressure) SH coefficients, (n_sh,)

Equations (vorticity-divergence form, Bourke 1972):
    d(vor)/dt  = -div((vor+f)*v) + curl(vert_adv)
    d(div)/dt  = curl((vor+f)*v) - lap(K + Phi + R_d*T*lnps) + div(vert_adv)
    d(T)/dt    = -div(T*v) + T*div(v) - sigma_dot*dT/dsigma + kappa*T*omega/p
    d(lnps)/dt = -integral((div + v·grad(lnps))*dsigma)   [flux form,
                 Hoskins & Simmons 1975; equals -(1/p_s)·integral(div(v·dp))]

References
----------
- Bourke, W. (1972). An Efficient, One-Level, Primitive-Equation Spectral
  Model. Monthly Weather Review, 100, 683-689.
- Hack, J. J. & Jakob, R. (1992). Description of a Global Shallow Water
  Model Based on the Spectral Transform Method. NCAR TN-343+STR.
- Hoskins, B. J. & Simmons, A. J. (1975). A multi-layer spectral model
  and the semi-implicit method. Quart. J. R. Met. Soc., 101, 637-655.
"""

from __future__ import annotations

import math
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.field import Field
from legoesm.parallel.metal import place_spectral_grid
from legoesm.grids.gaussian import (
    GaussianGrid,
    dealiasing_mask,
    sh_analysis,
    sh_synthesis,
    sh_analysis_3d,
    sh_synthesis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
    sh_analysis_oc2_dmu_3d,
    uv_from_vordiv_3d,
    spectral_hyperdiffusion_3d,
    sh_synthesis_H,
)
from legoesm.grids.vertical import (
    SigmaCoordinate,
    HybridSigmaPressureCoordinate,
    pressure_from_hybrid,
    dp_from_hybrid,
    compute_geopotential_hybrid,
    compute_mass_flux_from_cumsum,
    vertical_advection_hybrid,
    compute_omega_hybrid,
)
from legoesm.timestepping.dispatch import dispatch_integrator
from legoesm.timestepping.integration import (
    refuse_unthreaded_stateful_physics,
)
from legoesm.timestepping.semi_implicit import (
    euler_si_step,
    leapfrog_si_step,
    precompute_si_matrices,
    robert_asselin_filter,
    ssp_rk3_step_si,
)
from legoesm import constants

# Log-surface-pressure clamp bounds [ln Pa].  Pure host constants — use
# ``math.log`` (not ``jnp.log``) so this module imports without dispatching a
# device computation.  An eager ``float(jnp.log(...))`` here compiles a tiny
# HLO at import time, which (a) wastes a compile on every backend and (b)
# hard-crashes the Apple/Metal backend ("unknown attribute code") before any
# model is even constructed, taking down the whole physics/driver import chain.
_LNPS_MIN = math.log(100.0)
_LNPS_MAX = math.log(2.0e6)
# Softplus transition width [ln Pa] for the two-sided lnps clamp.  The
# UNSCALED softplus has a ~1 ln-unit transition zone: ln(1e5 Pa) sits only
# 3 ln-units below _LNPS_MAX, so softplus(-3) ≈ 0.049 subtracted 1e5 Pa
# down to 95333 Pa — a 4.8% bias deep in the physical interior.  The
# scaled form (same pattern as the T_min clamp) is identity to fp64
# precision more than ~10·scale inside the bounds.
_LNPS_CLIP_SCALE = 0.05
_COS_LAT_MIN = 1.0e-6


def soft_clip_lnps(lnps_raw: jax.Array) -> jax.Array:
    """Two-sided C∞ clamp of ln(p_s) to [_LNPS_MIN, _LNPS_MAX].

    Scaled softplus: identity to fp64 precision in the interior, smooth
    pull-up/pull-down only within ~10·_LNPS_CLIP_SCALE ln-units of the
    bounds.  Module-level so the interior-identity property is directly
    unit-testable (the unscaled version silently mapped 1e5 Pa → 95333 Pa).
    """
    return (lnps_raw
            + _LNPS_CLIP_SCALE * jax.nn.softplus((_LNPS_MIN - lnps_raw) / _LNPS_CLIP_SCALE)
            - _LNPS_CLIP_SCALE * jax.nn.softplus((lnps_raw - _LNPS_MAX) / _LNPS_CLIP_SCALE))

# Sentinel distinguishing "no forcing arg" (3-arg physics_fn) from a forcing
# value of None (4-arg) in SpectralPrimitiveEquationModel._make_tendency_fn.
_NO_FORCING = object()


# =============================================================================
# State and config
# =============================================================================

class SpectralHydrostaticState(NamedTuple):
    """State for the spectral hydrostatic primitive equations.

    3D spectral fields: shape (n_sh, nlev) complex128
    2D spectral fields: shape (n_sh,) complex128

    Tracers (optional): a ``dict[str, Field]`` mapping tracer name
    (``"q_v"``, ``"q_c"``, ``"q_r"``, ...) to a grid-space ``Field``
    of shape ``(n_lat, n_lon, nlev)``.  The spectral PE time-integration
    loop does NOT yet apply tracer tendencies — the field exists so
    physics bridges (radiation, convection, microphysics) can read
    ``q_v`` directly from the state without a duck-typed wrapper.
    Adding tracer advection / time-stepping is the dedicated "spectral
    PE tracers" follow-up.  Mirrors :class:`HydrostaticState.tracers`.
    """
    vor_hat: Field    # Spectral relative vorticity [1/s]
    div_hat: Field    # Spectral divergence [1/s]
    T_hat: Field      # Spectral temperature [K]
    lnps_hat: Field   # Spectral log(surface pressure) [-]
    phis_hat: Field   # Spectral surface geopotential [m^2/s^2] (static)
    tracers: dict | None = None  # name → grid-space Field (n_lat, n_lon, nlev)


_SPECTRAL_COEFF_NAMES = ("vor_hat", "div_hat", "T_hat", "lnps_hat", "phis_hat")


def reconstruct_spectral_state_from_npz(d, *, template: SpectralHydrostaticState | None = None):
    """Rebuild a :class:`SpectralHydrostaticState` + ``(step, day)`` from a saved
    spectral checkpoint dict ``d`` (the keys ``ModelDriver.save_checkpoint`` writes
    for a ``discretization='spectral'`` run: the five ``*_hat`` complex coefficient
    arrays + ``spectral_layout`` marker + ``step``/``day`` + optional grid-space
    ``tracer_names``/``trc_*``).

    Single canonical reconstruction used by BOTH the in-driver restart
    (``ModelDriver._load_checkpoint``, ``template=self.state`` → reuse the configured
    Field metadata + VALIDATE shapes + refuse to silently drop water) AND the
    standalone offline loader (``driver.restart.load_restart`` / the one-shot compare
    CLI, ``template=None`` → plain ``Field``-wrapped coefficients, since the spectral
    Fields carry no layout-specific metadata).  Raises if ``d`` is not a spectral
    checkpoint (missing the ``spectral_layout`` marker).
    """
    keys = d.files if hasattr(d, "files") else d
    if "spectral_layout" not in keys:
        raise ValueError(
            "reconstruct_spectral_state_from_npz: not a spectral checkpoint "
            "(missing the 'spectral_layout' marker)."
        )

    def _load_coeff(arr_in):
        """jnp.asarray with a LOUD x64 guard: complex128 coefficients silently
        downcast to complex64 when JAX x64 is disabled — the spectral transforms
        require x64 (Codex iter 92), so refuse rather than corrupt the restart."""
        src = np.asarray(arr_in)
        out = jnp.asarray(arr_in)
        if np.iscomplexobj(src) and src.dtype == np.complex128 and \
                out.dtype != jnp.complex128:
            raise TypeError(
                "spectral checkpoint is complex128 but JAX x64 is disabled — set "
                "JAX_ENABLE_X64=1 (spectral transforms require it); refusing to "
                "silently downcast the coefficients to complex64."
            )
        return out

    fields = {}
    for name in _SPECTRAL_COEFF_NAMES:
        arr = _load_coeff(d[name])
        if template is not None:
            cur = getattr(template, name)
            if arr.shape != cur.data.shape:
                raise ValueError(
                    f"spectral checkpoint {name} shape {tuple(arr.shape)} != "
                    f"configured state {tuple(cur.data.shape)} (resolution/nlev "
                    f"mismatch)."
                )
            fields[name] = cur.replace(data=arr)
        else:
            fields[name] = Field(arr)

    # tracer_names may be a NumPy string OR byte-string array (dtype-dependent) —
    # decode bytes so the trc_<name> lookup key is correct (Codex iter 92).
    def _name(n):
        return n.decode() if isinstance(n, bytes | np.bytes_) else str(n)

    tracers = None
    names = [_name(n) for n in d["tracer_names"]] if "tracer_names" in keys else None
    if names is not None:
        if template is not None:
            if template.tracers is None:
                raise ValueError(
                    f"spectral checkpoint carries tracers {names} but the configured "
                    "run has none — refusing to silently drop water."
                )
            if set(names) != set(template.tracers):
                raise ValueError(
                    f"spectral checkpoint tracers {sorted(names)} != configured "
                    f"{sorted(template.tracers)} — refusing to add/drop a tracer "
                    "across restart (a different run)."
                )
        tracers = {}
        for k in names:
            arr = _load_coeff(d[f"trc_{k}"])
            if template is not None:
                cur_t = template.tracers[k]
                if arr.shape != cur_t.data.shape:
                    raise ValueError(
                        f"spectral checkpoint tracer {k} shape {tuple(arr.shape)} != "
                        f"configured {tuple(cur_t.data.shape)} (resolution/nlev "
                        "mismatch)."
                    )
                tracers[k] = cur_t.replace(data=arr)
            else:
                tracers[k] = Field(arr)
    elif template is not None and template.tracers is not None:
        # The checkpoint carries NO tracers but the configured run is moist — a
        # bit-exact spectral save always writes tracer_names for a moist run, so
        # this is a dry-checkpoint-vs-moist-run mismatch; refuse to invent water
        # (the symmetric guard to the drop-water raise above) (Codex iter 92).
        raise ValueError(
            f"spectral checkpoint carries NO tracers but the configured run has "
            f"{sorted(template.tracers)} — refusing to silently invent water "
            "(a different run)."
        )

    state = (
        template._replace(tracers=tracers, **fields) if template is not None
        else SpectralHydrostaticState(tracers=tracers, **fields)
    )
    return state, int(d["step"]), float(d["day"])


class SpectralPEConfig(NamedTuple):
    """Configuration for spectral primitive equation model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_order: int = 2
    time_integrator: str = "ssp_rk3"  # "ssp_rk3", "ssp_rk34", or "ssp_rk54"
    semi_implicit: bool = False      # Use Hoskins-Simmons semi-implicit
    si_T_ref: float = 300.0         # Reference temperature for linearization [K]
    si_alpha: float = 0.5           # Implicitness (0.5 = Crank-Nicolson)
    si_substeps: int = 1            # Internal SI substeps per external model step
    si_hyperdiff_boost: float = 1.0  # Multiply hyperdiffusion in SI mode
    # Sponge layer (implicit multiplicative filter at model top)
    sponge_sigma: float = 0.1       # Sigma below which sponge is active (damps above)
    sponge_tau: float = 0.0         # E-folding time at model top [s] (0 = off)
    # Level-dependent hyperdiffusion scaling (stronger at low pressures)
    hyperdiff_pscale: float = 0.0   # Power-law exponent: nu_k = nu * (p_ref/p_k)^exp (0 = off)
    # Temperature floor (positivity protection)
    T_min: float = 50.0             # Minimum temperature [K]
    # Post-step spectral filter (damps highest wavenumbers)
    spectral_filter_order: int = 8   # Sharpness of spectral filter
    spectral_filter_strength: float = 0.0  # Retention at n=n_max (0=off, 0.01=aggressive)
    # Tendency truncation to prevent aliasing from cubic nonlinearities.
    dealiasing_fraction: float = 0.667  # 2/3 rule for cubic nonlinearities
    # Implicit (multiplicative) hyperdiffusion.  Applied as a post-step
    # filter: coeff_new = coeff_old * exp(-nu * [n(n+1)/a^2]^p * dt).
    # This is UNCONDITIONALLY STABLE, unlike explicit (tendency-based)
    # hyperdiffusion which is unstable with leapfrog time integration.
    # Set implicit_hyperdiff=True to use this instead of explicit.
    implicit_hyperdiff: bool = False
    # Pressure floor for adiabatic heating (limits 1/p at model top)
    p_floor: float = 10.0           # Pa; adiabatic uses max(p, p_floor) to prevent omega/p overflow
    # Robert-Asselin-Williams filter for leapfrog (controls computational mode)
    robert_asselin_coeff: float = 0.05  # Filter coefficient γ (0 = off, 0.05-0.1 typical)
    # Williams 2009 α parameter.  0.53 is conditionally stable and is
    # the recommended practical RAW choice; 0.5 conserves the
    # three-time-level mean exactly but is unconditionally unstable.
    # 1.0 recovers the original Robert-Asselin filter (3rd-order phase
    # error retained).
    robert_asselin_alpha: float = 0.53
    # Iter-3: optional initial-mass anchor for the spectral primitive
    # equations.  When ``fix_mass`` is on and
    # ``anchor_mass_to_initial`` is True, ``step()`` snapshots the
    # initial total dry mass (∫ p_s dA, fp64) on first call and rescales
    # ``lnps_hat[0]`` after each step so the global integral returns to
    # the snapshot.  Mirrors ``primitive_eq_cdgrid``/``primitive_eq_latlon_cgrid``
    # behaviour.  Disabled by default to preserve pre-iter-3 baseline
    # numerics for tests that intentionally measure drift.
    fix_mass: bool = False
    anchor_mass_to_initial: bool = False


# =============================================================================
# Internal vertical helpers (generic shapes, no cubed-sphere assumptions)
# =============================================================================

def _compute_geopotential_gaussian(T, p_s, sigma_coord, phis):
    """Simmons-Burridge geopotential on Gaussian grid.

    Same math as vertical.compute_geopotential but with generic
    broadcasting: T is (..., nlev), p_s is (...), phis is (...).
    """
    R_d = constants.R_d
    ln_ratio = sigma_coord.ln_ratio   # (nlev,)
    alpha = sigma_coord.alpha         # (nlev,)

    dPhi = R_d * T * ln_ratio         # broadcast: (..., nlev)

    dPhi_reversed = dPhi[..., ::-1]
    cumsum_reversed = jnp.cumsum(dPhi_reversed, axis=-1)
    cumsum = cumsum_reversed[..., ::-1]

    Phi_above = phis[..., None] + cumsum

    Phi_below = jnp.concatenate(
        [Phi_above[..., 1:], phis[..., None]], axis=-1,
    )

    Phi_full = Phi_below + alpha * R_d * T
    return Phi_full


def _compute_sigma_dot_gaussian(div_3d, sigma_coord):
    """Sigma-dot on arbitrary grid shape. div_3d is (..., nlev).

    Returns ``(sigma_dot, D_total)`` where ``D_total`` is the
    column-integrated divergence (``sum(div * dsigma, axis=-1,
    keepdims=True)``) — exposing it lets the caller reuse the value
    in the surface-pressure tendency without re-summing the column.
    """
    dsigma = sigma_coord.dsigma
    fractional_sigma = sigma_coord.fractional_sigma

    div_dsigma = div_3d * dsigma
    # ``cumsum`` already contains ``sum`` as its last entry — extract it
    # rather than computing the sum independently.  Under level-sharding
    # this drops the per-stage allreduce-equivalent cumsum-axis collective
    # from 2 to 1 (the prefix-cumsum + slicing the last index reuses the
    # same prefix-scan kernel).
    cumsum_div = jnp.cumsum(div_dsigma, axis=-1)
    D_total = cumsum_div[..., -1:]

    sigma_dot_inner = fractional_sigma * D_total - cumsum_div

    # Top BC: σ̇=0; bottom BC: zero by construction
    # (frac_sigma[-1]=1, cumsum_div[-1]=D_total → sigma_dot_inner[-1]=0).
    # Drop the (∼0) trailing element + pad with zeros on both ends in
    # one ``jnp.pad`` — replaces ``jnp.pad`` + scatter (2 HLO ops) with
    # slice + Pad (2 HLO ops) but eliminates the float roundoff in
    # sigma_dot[-1].
    pad_axes = ((0, 0),) * (sigma_dot_inner.ndim - 1) + ((1, 1),)
    sigma_dot = jnp.pad(sigma_dot_inner[..., :-1], pad_axes)
    return sigma_dot, D_total


def _vertical_advection_sigma_gaussian(field, sigma_dot, sigma_coord):
    """Vertical advection -sigma_dot * dfield/dsigma (upwind). Generic shapes.

    Top/bottom boundaries pad with a zero gradient; using ``jnp.pad``
    instead of ``concatenate([jnp.zeros(...), ...])`` lowers to a
    single XLA ``Pad`` op rather than allocating a fresh zero buffer
    every RHS evaluation (this helper runs 3-5× per outer step under
    SSP-RK).
    """
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])
    dsigma_bwd = sigma_coord.dsigma_full
    df_bwd = jnp.diff(field, axis=-1)
    diff = df_bwd / dsigma_bwd

    pad_axes = ((0, 0),) * (diff.ndim - 1)
    grad_bwd = jnp.pad(diff, (*pad_axes, (1, 0)))
    grad_fwd = jnp.pad(diff, (*pad_axes, (0, 1)))

    grad = jnp.where(sigma_dot_full > 0, grad_bwd, grad_fwd)
    return -sigma_dot_full * grad


def _compute_omega_gaussian(sigma_dot, p_s, dp_s_dt, sigma_coord):
    """Pressure velocity omega = sigma * dp_s/dt + p_s * sigma_dot_full."""
    sigma_full = sigma_coord.sigma_full
    sigma_dot_full = 0.5 * (sigma_dot[..., :-1] + sigma_dot[..., 1:])
    omega = sigma_full * dp_s_dt[..., None] + p_s[..., None] * sigma_dot_full
    return omega


def _tracer_advection_gaussian(
    q_grid: jnp.ndarray,
    u_cos: jnp.ndarray,
    v_cos: jnp.ndarray,
    div: jnp.ndarray,
    sigma_dot: jnp.ndarray,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    grid: GaussianGrid,
) -> jnp.ndarray:
    """Compute advective tendency ``∂q/∂t = -v · ∇q - σ̇ · ∂q/∂σ`` on grid.

    Uses the conservative + correction form (matching the T equation
    in :func:`spectral_pe_tendencies`):

        ∂q/∂t = -∇·(q v_h) + q · D - σ̇ · ∂q/∂σ

    Spectral horizontal divergence via the pole-safe ``oc2`` / ``dmu``
    operators; vertical advection via the existing upwind helper.

    Parameters
    ----------
    q_grid : jax.Array, shape (n_lat, n_lon, nlev)
        Tracer mixing ratio at full levels.
    u_cos, v_cos : jax.Array, shape (n_lat, n_lon, nlev)
        ``u·cos φ``, ``v·cos φ`` (pole-safe; matches the convention
        used elsewhere in the spectral PE RHS).
    div : jax.Array, shape (n_lat, n_lon, nlev)
        Horizontal divergence ``∇·v_h`` on grid.
    sigma_dot : jax.Array, shape (n_lat, n_lon, nlev+1)
        Sigma-dot at half levels (only used for the σ-coord branch).
        For hybrid coords this argument is ignored — the helper falls
        back to the same pseudospectral horizontal pathway and the
        caller drives vertical advection via
        ``vertical_advection_hybrid``.
    sigma_coord : SigmaCoordinate or HybridSigmaPressureCoordinate
    grid : GaussianGrid

    Returns
    -------
    jax.Array, shape (n_lat, n_lon, nlev)
        Tracer tendency ``∂q/∂t`` on grid (no physics added).
    """
    a = grid.radius
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a
    one_over_a = 1.0 / a

    # Horizontal flux ``q·v_h`` with cos-φ weighting absorbed; pair the
    # two analyses into a single oc2 / dmu batch (4 SH forwards → 2).
    flux_x = q_grid * u_cos      # = q · u · cos φ
    flux_y = q_grid * v_cos      # = q · v · cos φ
    n_lat_q, n_lon_q, nlev_q = q_grid.shape
    _flux_stack = jnp.stack([flux_x, flux_y], axis=-1)
    _flux_flat = _flux_stack.reshape(n_lat_q, n_lon_q, nlev_q * 2)
    # Iter-81: fused oc2+dmu shares the FFT + gather (was 2 separate
    # forward transforms on the same input).
    _oc2_raw, _dmu_raw = sh_analysis_oc2_dmu_3d(grid, _flux_flat)
    _oc2_pair = _oc2_raw.reshape(-1, nlev_q, 2)
    _dmu_pair = _dmu_raw.reshape(-1, nlev_q, 2)
    flux_x_oc2 = _oc2_pair[..., 0]
    flux_y_dmu = _dmu_pair[..., 1]

    # Spectral horizontal divergence of ``(q·u, q·v)``.
    flux_q_div_hat = im_over_a[:, None] * flux_x_oc2 - one_over_a * flux_y_dmu
    flux_q_div_grid = sh_synthesis_3d(grid, flux_q_div_hat)

    # Conservative advection + divergence-of-velocity correction:
    #   -∇·(q v) + q · ∇·v  ≡  -v · ∇q       (advective form)
    horiz_adv = -flux_q_div_grid + q_grid * div

    # Vertical advection.  Hybrid coords use the mass-flux helper from
    # the dycore; sigma coords use the existing upwind-stable helper.
    if isinstance(sigma_coord, HybridSigmaPressureCoordinate):
        # ``sigma_dot`` is ignored in this branch; the hybrid path is
        # handled by the caller post-return (it has access to mass_flux
        # and p_s).  Returning horizontal-only here keeps the helper
        # non-conditional on ``sigma_dot``.
        return horiz_adv

    vert_adv = _vertical_advection_sigma_gaussian(q_grid, sigma_dot, sigma_coord)
    return horiz_adv + vert_adv


# =============================================================================
# Tendency computation
# =============================================================================

def spectral_pe_tendencies(
    state: SpectralHydrostaticState,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate | HybridSigmaPressureCoordinate,
    config: SpectralPEConfig,
    physics_tendency: SpectralHydrostaticState | None = None,
) -> SpectralHydrostaticState:
    """Compute spectral tendencies for the hydrostatic PE.

    Uses the pseudospectral transform method:
    1. Transform prognostic fields to grid space
    2. Compute nonlinear products on grid
    3. Transform products to spectral space
    4. Assemble tendencies using spectral operators

    Returns tendencies in the same pytree structure as state (for SSP-RK3).
    """
    _hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)

    a = grid.radius
    R_d = constants.R_d
    kappa = constants.kappa

    # Dealiasing mask: zero wavenumbers above dealiasing_fraction * n_max
    # to prevent spectral aliasing from cubic nonlinearities.  Shared Orszag
    # 2/3-rule helper (canonical for spectral_pe/nh/sw — no inline re-derivation).
    if config.dealiasing_fraction > 0:
        _dealias = dealiasing_mask(grid, config.dealiasing_fraction)
        _dealias_3d = _dealias[:, None]  # (n_sh, 1) for 3D fields
    else:
        _dealias = None
        _dealias_3d = None

    # --- 1. Transform to grid space ---
    # Merge the (vor, div, T) 3D batch with the (lnps, phis, im·lnps)
    # 2D triplet via ``jnp.concatenate``: trailing axis = ``nlev*3 + 3``.
    # ``sh_synthesis_3d`` treats any trailing axis as a passive batch,
    # so different "level" sizes (nlev vs 1) combine cleanly into a
    # single ``segment_sum`` + IRFFT.  6 SH syntheses → 1.  Loop 182
    # extends Loop 181 (spectral ocean merge).
    n_sh, nlev = state.vor_hat.data.shape
    _hat_stack = jnp.stack(
        [state.vor_hat.data, state.div_hat.data, state.T_hat.data],
        axis=-1,
    )  # (n_sh, nlev, 3)
    _hat_flat = _hat_stack.reshape(n_sh, nlev * 3)
    # Append the three 2D fields as single-level slots.  ``im·lnps_hat``
    # is the spectral pre-multiply that yields ``∂(lnps)/∂λ`` on the
    # grid post-synthesis (Loop 147 trick).
    _ims_lnps = (1j * grid.ms) * state.lnps_hat.data  # (n_sh,)
    _all_hat_flat = jnp.concatenate(
        [
            _hat_flat,
            state.lnps_hat.data[:, jnp.newaxis],
            state.phis_hat.data[:, jnp.newaxis],
            _ims_lnps[:, jnp.newaxis],
        ],
        axis=-1,
    )  # (n_sh, nlev*3 + 3)
    _all_grid_flat = sh_synthesis_3d(grid, _all_hat_flat)
    _grid_stack = _all_grid_flat[..., : nlev * 3].reshape(
        grid.n_lat, grid.n_lon, nlev, 3,
    )
    vor = _grid_stack[..., 0]
    div = _grid_stack[..., 1]
    T = _grid_stack[..., 2]
    # Smooth positivity protection (C∞ differentiable, scaled softplus for ~0.07K bias)
    _sp_scale = 0.1
    T = T + _sp_scale * jax.nn.softplus((config.T_min - T) / _sp_scale)
    lnps_raw = _all_grid_flat[..., nlev * 3]
    phis = _all_grid_flat[..., nlev * 3 + 1]
    _dfdlon_lnps = _all_grid_flat[..., nlev * 3 + 2]
    # Smooth two-sided clip with zero bias in interior (C∞): scaled
    # softplus pulls up near the lower bound / down near the upper bound
    # and is identity to fp64 precision elsewhere (see _LNPS_CLIP_SCALE).
    lnps = soft_clip_lnps(lnps_raw)
    # (n_lat, n_lon)

    # --- 2. Velocities ---
    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )  # (n_lat, n_lon, nlev)
    cos_lat_3d = jnp.clip(grid.cos_lat[:, None, None], _COS_LAT_MIN, None)
    u = u_cos / cos_lat_3d
    v = v_cos / cos_lat_3d

    # --- 3. Pressure ---
    p_s = jnp.exp(lnps)
    if _hybrid:
        p_full = pressure_from_hybrid(sigma_coord, p_s)
        dp = dp_from_hybrid(sigma_coord, p_s)  # (n_lat, n_lon, nlev)
    else:
        p_full = p_s[..., None] * sigma_coord.sigma_full  # (n_lat, n_lon, nlev)

    # --- 4. Geopotential ---
    if _hybrid:
        Phi = compute_geopotential_hybrid(T, p_s, sigma_coord, phis)
    else:
        Phi = _compute_geopotential_gaussian(T, p_s, sigma_coord, phis)

    # --- 5. Kinetic energy (pole-safe via oc2 transform) ---
    # KE = (u²+v²)/2 = (u_cos²+v_cos²)/(2·cos²φ).  Computing KE on the
    # grid amplifies polar noise by 1/cos².  Instead we keep KE·cos²φ and
    # let sh_analysis_oc2_3d (which has 1/cos² in the Legendre matrix)
    # absorb the singularity.  Physical u, v are still needed below for
    # vertical advection and the adiabatic v·∇(lnps) term.
    KE_cos2 = 0.5 * (u_cos * u_cos + v_cos * v_cos)  # KE·cos²φ

    # --- 6. Absolute vorticity ---
    abs_vor = vor + grid.f[..., None]

    # --- 7. Vertical velocity ---
    # Flux-form continuity (Hoskins & Simmons 1975): the column integrand
    # is ∇·(v·dp_k) = dp_k·D + v·∇(dp_k), NOT the advective dp_k·D alone.
    # With dp = p_s·Δσ (sigma) the per-level correction is Δσ·p_s·(v·∇lnps);
    # with dp = ΔA + ΔB·p_s (hybrid) it is ΔB·p_s·(v·∇lnps).  Omitting it
    # produces the wrong local ∂p_s/∂t pattern wherever ∇p_s ≠ 0 (the
    # global mass fixer only restores the integral, not the pattern).
    # ∇lnps is synthesized here — before step 8 — and reused by the PGF
    # correction and the adiabatic v·∇lnps term below.
    cos_lat_2d = jnp.clip(grid.cos_lat[:, None], _COS_LAT_MIN, None)
    dlnps_dx = _dfdlon_lnps / (a * cos_lat_2d)
    dfdtheta_cos = sh_synthesis_H(grid, state.lnps_hat.data)
    dlnps_dy = -dfdtheta_cos / (a * cos_lat_2d)
    # Unscaled v·∇lnps (the hybrid thermodynamic term rescales its own copy).
    v_grad_lnps = u * dlnps_dx[..., None] + v * dlnps_dy[..., None]

    if _hybrid:
        # Flux-form mass-weighted divergence per level; its cumsum feeds
        # both the mass flux (shared boundary closure) and, via the last
        # entry, the surface-pressure tendency in step 8 — one cross-level
        # collective per RK3 stage under level-sharding, as before.
        div_dp_flux = div * dp + sigma_coord.dB * p_s[..., None] * v_grad_lnps
        _cumsum_dp = jnp.cumsum(div_dp_flux, axis=-1)
        _D_total_p_full = _cumsum_dp[..., -1:]
        mass_flux = compute_mass_flux_from_cumsum(
            _cumsum_dp, _D_total_p_full, sigma_coord,
        )
        sigma_dot = None   # hybrid path uses ``mass_flux`` instead
        _D_total_sigma_full = None
    else:
        # ``D_total_sigma_full`` is the (..., 1)-shaped column-sum that
        # ``_compute_sigma_dot_gaussian`` already produced — reuse it
        # below in the surface-pressure tendency rather than recomputing
        # ``jnp.sum(div * dsigma, axis=-1)``.  Saves one cross-level
        # collective per RK3 stage under level-sharding.
        sigma_dot, _D_total_sigma_full = _compute_sigma_dot_gaussian(
            div + v_grad_lnps, sigma_coord,
        )
        _D_total_p_full = None

    # --- 8. Surface pressure tendency ---
    if _hybrid:
        # ``_D_total_p_full`` has trailing-axis size 1; drop the
        # singleton to match the original ``jnp.sum(...)`` shape.
        D_total_p = _D_total_p_full[..., 0]
        dlnps_dt_grid = -D_total_p / (p_s * sigma_coord.B_range)
        dp_s_dt_grid = p_s * dlnps_dt_grid
    else:
        sigma_top = sigma_coord.sigma_half[0]
        sigma_range = 1.0 - sigma_top
        # ``D_total_sigma_full`` has trailing-axis size 1; drop the
        # singleton to match the original ``jnp.sum(...)`` shape.
        D_total = _D_total_sigma_full[..., 0]
        dlnps_dt_grid = -D_total / sigma_range
        dp_s_dt_grid = p_s * dlnps_dt_grid

    # --- 9. Spectral operators ---
    im_over_a = 1j * grid.ms.astype(jnp.float64) / a  # (n_sh,)
    one_over_a = 1.0 / a

    # --- 10. Vorticity fluxes: (zeta+f)*u*cos, (zeta+f)*v*cos ---
    # Batch the four SH analyses (oc2 on A_vor & B_vor, dmu on A_vor &
    # B_vor) into two: stack ``(A_vor, B_vor)`` along a trailing axis
    # and fold into the level dim so each SH-analysis variant runs once
    # on a thicker (n_lat, n_lon, nlev*2) tensor.  Both
    # ``sh_analysis_oc2_3d`` and ``sh_analysis_dmu_3d`` treat the
    # trailing axis as a passive batch (FFT on lon, weighted sum over
    # lat — neither touches the trailing axis), so the result is
    # identical to two separate calls.  4 SH forwards → 2.
    A_vor = abs_vor * u_cos   # (n_lat, n_lon, nlev)
    B_vor = abs_vor * v_cos
    _AB_stack = jnp.stack([A_vor, B_vor], axis=-1)  # (..., nlev, 2)
    n_lat_t, n_lon_t, nlev_t, _ = _AB_stack.shape
    _AB_flat = _AB_stack.reshape(n_lat_t, n_lon_t, nlev_t * 2)
    # Iter-81: fused oc2+dmu shares the FFT + gather.
    _oc2_AB_raw, _dmu_AB_raw = sh_analysis_oc2_dmu_3d(grid, _AB_flat)
    _oc2_AB = _oc2_AB_raw.reshape(-1, nlev_t, 2)
    _dmu_AB = _dmu_AB_raw.reshape(-1, nlev_t, 2)
    _A_oc2, _B_oc2 = _oc2_AB[..., 0], _oc2_AB[..., 1]
    _A_dmu, _B_dmu = _dmu_AB[..., 0], _dmu_AB[..., 1]

    # Spectral divergence of vorticity flux -> dvor/dt
    flux_vor_div = im_over_a[:, None] * _A_oc2 - one_over_a * _B_dmu  # (n_sh, nlev)

    # Spectral curl of vorticity flux -> ddiv/dt contribution
    flux_vor_curl = im_over_a[:, None] * _B_oc2 + one_over_a * _A_dmu

    # --- 11. Pressure gradient force (correct form, NOT Bourke E-variable) ---
    # The PGF divergence is: -∇²(K + Φ) - ∇·(R_d·T·∇lnps)
    # Split using T = T_ref + T':
    #   = -∇²(K + Φ) - R_d·T_ref·∇²(lnps) - ∇·(R_d·T'·∇(lnps))
    # The first two terms use spectral Laplacian (exact).
    # The third term is computed as a grid-point product + spectral divergence.
    #
    # NOTE: The Bourke (1972) E-variable form E = K + Φ + R_d·T·lnps
    # is NOT used because -∇²(R_d·T·lnps) ≠ -∇·(R_d·T·∇lnps).
    # The E-variable adds a spurious same-level T→D coupling
    # (-R_d·lnps_0·∇²T') that is unstable when combined with
    # adiabatic heating.
    T_ref = config.si_T_ref

    # ∇(lnps) on grid (dlnps_dx / dlnps_dy) was hoisted to step 7 — the
    # flux-form continuity needs it before the surface-pressure tendency;
    # the PGF correction below reuses the same arrays.

    # PGF correction: -∇·(R_d·T'·∇_eta(lnp)) computed as spectral div of grid product
    # In sigma coords: ∇_eta(ln p) = ∇(ln p_s).
    # In hybrid coords: ∇_eta(ln p) = (B*p_s/p) * ∇(ln p_s).
    T_prime_pgf = T - T_ref
    _pgf_dlnps_dx = dlnps_dx[..., None]
    _pgf_dlnps_dy = dlnps_dy[..., None]
    if _hybrid:
        B_full = sigma_coord.B_full  # (nlev,)
        _hf = B_full * p_s[..., None] / p_full  # (n_lat, n_lon, nlev)
        _pgf_dlnps_dx = _pgf_dlnps_dx * _hf
        _pgf_dlnps_dy = _pgf_dlnps_dy * _hf
    pgf_Fx_cos = R_d * T_prime_pgf * _pgf_dlnps_dx * cos_lat_3d
    pgf_Fy_cos = R_d * T_prime_pgf * _pgf_dlnps_dy * cos_lat_3d

    # --- 13. Temperature equation ---
    # Horizontal: dT/dt = -v·∇T = -div(T*v) + T*div(v)
    # Reference-temperature subtraction (Simmons & Burridge 1981):
    # For uniform T_ref, -div(T*v) + T*div = -div(T'*v) + T'*div
    # where T' = T - T_ref.  This eliminates the O(T_ref * ε) cancellation
    # error that otherwise destabilises the isothermal rest state.
    T_ref = config.si_T_ref
    T_prime = T - T_ref

    T_prime_u_cos = T_prime * u_cos
    T_prime_v_cos = T_prime * v_cos

    # Batch SH oc2: PGF needs oc2(Fx), the T equation needs oc2(T'u),
    # and ``KPhi_hat`` (used by the divergence Laplacian below) needs
    # oc2(KE_cos2).  All three are oc2 forward analyses on
    # ``(n_lat, n_lon, nlev)`` grid fields; stack along a trailing axis
    # and fold into level so the oc2 SH runs once on a thicker
    # ``(n_lat, n_lon, nlev*3)`` tensor — 3 separate oc2 forwards
    # collapse to 1.  ``Phi`` still needs the *plain* sh_analysis_3d
    # weights, so it stays in its own (single) call.  The dmu branch
    # remains a 2-input batch (PGF Fy + T'v).
    _pgfTK_oc2_stack = jnp.stack(
        [pgf_Fx_cos, T_prime_u_cos, KE_cos2], axis=-1,
    )  # (..., nlev, 3)
    _pgfT_dmu_stack = jnp.stack([pgf_Fy_cos, T_prime_v_cos], axis=-1)
    n_lat_p, n_lon_p, nlev_p, _ = _pgfTK_oc2_stack.shape
    _oc2_triple = sh_analysis_oc2_3d(
        grid, _pgfTK_oc2_stack.reshape(n_lat_p, n_lon_p, nlev_p * 3),
    ).reshape(-1, nlev_p, 3)
    _dmu_pair = sh_analysis_dmu_3d(
        grid, _pgfT_dmu_stack.reshape(n_lat_p, n_lon_p, nlev_p * 2),
    ).reshape(-1, nlev_p, 2)

    pgf_correction_hat = (
        im_over_a[:, None] * _oc2_triple[..., 0]
        - one_over_a * _dmu_pair[..., 0]
    )

    # ``KPhi_hat`` consumes the third slice of the oc2 batch above
    # (``oc2(KE_cos2)``) plus the *plain* SH analysis of geopotential
    # ``Phi`` (different SH weight matrix → cannot share the oc2 call).
    # The plain ``sh_analysis_3d(Phi)`` is batched with the temperature
    # tendency analysis below (Loop 145), so we keep ``KPhi_oc2`` here
    # and add the Phi contribution after the batch.
    KPhi_oc2 = _oc2_triple[..., 2]

    # --- 12. Horizontal tendencies ---
    dvor_hat = -flux_vor_div
    # Build ``ddiv_hat`` with the KE contribution to ``KPhi_hat``
    # already in place; the geopotential ``Phi`` contribution is added
    # after the batched sh_analysis_3d below (Loop 145).
    ddiv_hat = (
        flux_vor_curl
        - grid.lap[:, None] * KPhi_oc2
        - R_d * T_ref * grid.lap[:, None] * state.lnps_hat.data[:, None]
        - pgf_correction_hat
    )

    flux_T_div = (
        im_over_a[:, None] * _oc2_triple[..., 1]
        - one_over_a * _dmu_pair[..., 1]
    )

    T_prime_div = T_prime * div

    # Vertical advection of T
    if _hybrid:
        vert_adv_T = vertical_advection_hybrid(T, mass_flux, p_s, sigma_coord)
    else:
        vert_adv_T = _vertical_advection_sigma_gaussian(T, sigma_dot, sigma_coord)

    # Adiabatic heating: kappa * T * omega / p
    if _hybrid:
        omega = compute_omega_hybrid(mass_flux, p_s, dp_s_dt_grid, sigma_coord)
    else:
        omega = _compute_omega_gaussian(sigma_dot, p_s, dp_s_dt_grid, sigma_coord)
    p_adiab = jnp.maximum(p_full, config.p_floor) if config.p_floor > 0 else p_full
    adiabatic = kappa * T * omega / p_adiab

    # Material derivative correction: kappa * T * v . grad_eta(ln p)
    # In hybrid coords: grad_eta(ln p) = (B*p_s/p) * grad(ln p_s).
    # Reuses the unscaled v·∇lnps hoisted to step 7 (flux-form continuity).
    v_dot_grad_lnps = v_grad_lnps
    if _hybrid:
        v_dot_grad_lnps = v_dot_grad_lnps * (sigma_coord.B_full * p_s[..., None] / p_adiab)
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps

    # Combine the three grid-space contributions to dT/dt before the SH
    # forward transform.  ``sh_analysis_3d`` is linear, so
    # ``Σ_i sh_analysis_3d(f_i) = sh_analysis_3d(Σ_i f_i)`` — summing on
    # grid first replaces three SH analyses (each one ``segment_sum`` +
    # one FFT) with one.  ``vert_adv_T``, ``adiabatic``, and
    # ``T_prime_div`` all share the (n_lat, n_lon, nlev) grid shape.
    dT_grid_sum = T_prime_div + vert_adv_T + adiabatic

    # Batch the dT-grid SH analysis with the geopotential ``Phi``
    # analysis (Loop 145) AND the ``dlnps_dt_grid`` 2D analysis (Loop
    # 184).  All three are plain ``sh_analysis_3d`` calls on grid
    # fields with the same Legendre weight matrix; ``dlnps_dt_grid``
    # is 2D (n_lat, n_lon) so it joins as a single-level slot via
    # ``[..., None]`` + ``jnp.concatenate``.  Trailing axis =
    # ``nlev*2 + 1``.  3 plain SH analyses → 1.
    n_lat_T, n_lon_T, nlev_T = dT_grid_sum.shape
    _Tphi_stack = jnp.stack([dT_grid_sum, Phi], axis=-1)
    _Tphi_flat = _Tphi_stack.reshape(n_lat_T, n_lon_T, nlev_T * 2)
    _Tphi_lnps_input = jnp.concatenate(
        [_Tphi_flat, dlnps_dt_grid[..., jnp.newaxis]], axis=-1,
    )  # (n_lat, n_lon, nlev*2 + 1)
    _Tphi_lnps_hat = sh_analysis_3d(grid, _Tphi_lnps_input)
    _Tphi_hat = _Tphi_lnps_hat[:, : nlev_T * 2].reshape(-1, nlev_T, 2)
    _dT_grid_hat = _Tphi_hat[..., 0]
    _Phi_hat = _Tphi_hat[..., 1]
    _dlnps_hat_pre = _Tphi_lnps_hat[:, nlev_T * 2]
    dT_hat = -flux_T_div + _dT_grid_hat
    # Apply the deferred ``-∇²(Φ)`` contribution to the divergence
    # tendency now that ``Phi_hat`` is available from the batch.
    ddiv_hat = ddiv_hat - grid.lap[:, None] * _Phi_hat

    # --- 14. Vertical advection of momentum ---
    if _hybrid:
        vert_adv_u = vertical_advection_hybrid(u, mass_flux, p_s, sigma_coord)
        vert_adv_v = vertical_advection_hybrid(v, mass_flux, p_s, sigma_coord)
    else:
        vert_adv_u = _vertical_advection_sigma_gaussian(u, sigma_dot, sigma_coord)
        vert_adv_v = _vertical_advection_sigma_gaussian(v, sigma_dot, sigma_coord)

    # Convert to spectral vor/div contributions.  Same batching pattern
    # as the vorticity-flux SH analyses above: stack (vert_u_cos,
    # vert_v_cos) along a trailing axis and run each SH variant once on
    # a thicker (..., nlev*2) tensor — 4 SH forwards collapse to 2.
    vert_u_cos = vert_adv_u * grid.cos_lat[:, None, None]
    vert_v_cos = vert_adv_v * grid.cos_lat[:, None, None]
    _vert_uv_stack = jnp.stack([vert_u_cos, vert_v_cos], axis=-1)
    n_lat_v, n_lon_v, nlev_v, _ = _vert_uv_stack.shape
    _vert_uv_flat = _vert_uv_stack.reshape(n_lat_v, n_lon_v, nlev_v * 2)
    # Iter-81: fused oc2+dmu shares the FFT + gather.
    _vert_oc2_raw, _vert_dmu_raw = sh_analysis_oc2_dmu_3d(grid, _vert_uv_flat)
    _vert_oc2 = _vert_oc2_raw.reshape(-1, nlev_v, 2)
    _vert_dmu = _vert_dmu_raw.reshape(-1, nlev_v, 2)
    _vu_oc2, _vv_oc2 = _vert_oc2[..., 0], _vert_oc2[..., 1]
    _vu_dmu, _vv_dmu = _vert_dmu[..., 0], _vert_dmu[..., 1]

    # curl(vert_adv) -> dvor_hat
    vert_vor_tend = im_over_a[:, None] * _vv_oc2 + one_over_a * _vu_dmu
    # div(vert_adv) -> ddiv_hat
    vert_div_tend = im_over_a[:, None] * _vu_oc2 - one_over_a * _vv_dmu

    dvor_hat = dvor_hat + vert_vor_tend
    ddiv_hat = ddiv_hat + vert_div_tend

    # --- 15. Surface pressure tendency (spectral) ---
    # ``_dlnps_hat_pre`` was already computed alongside (dT_grid_sum,
    # Phi) via the batched ``sh_analysis_3d`` above (Loop 184); reuse
    # it instead of issuing a standalone 2D ``sh_analysis``.
    dlnps_hat = _dlnps_hat_pre

    # --- 16. Spectral hyperdiffusion ---
    hyperdiff_coeff = config.hyperdiff_coeff
    if config.semi_implicit and config.si_hyperdiff_boost != 1.0:
        hyperdiff_coeff = hyperdiff_coeff * config.si_hyperdiff_boost

    if hyperdiff_coeff > 0 and not config.implicit_hyperdiff:
        # Batch the three pointwise spectral hyperdiffusions (vor, div,
        # T) into one call by stacking the spectral coefficients along
        # a trailing axis.  ``spectral_hyperdiffusion_3d`` is just
        # ``damping[:, None] * coeffs`` — element-wise multiplication
        # with the trailing axis as a passive batch — so 3 separate
        # kernel launches collapse to 1 fused multiply on the thicker
        # tensor.  Same-axis pattern as Loops 93 / 97 for the SH
        # transforms above.
        n_sh_h, nlev_h = state.vor_hat.data.shape
        _vdT_hat = jnp.stack(
            [state.vor_hat.data, state.div_hat.data, state.T_hat.data], axis=-1,
        )  # (n_sh, nlev, 3)
        base_diff_stack = spectral_hyperdiffusion_3d(
            grid, _vdT_hat.reshape(n_sh_h, nlev_h * 3),
            hyperdiff_coeff, config.hyperdiff_order,
        ).reshape(n_sh_h, nlev_h, 3)

        if config.hyperdiff_pscale > 0:
            # Level-dependent scaling: (p_ref/p_k)^exponent
            # Stronger diffusion at low pressures (upper atmosphere)
            if _hybrid:
                sigma_full = sigma_coord.A_full + sigma_coord.B_full
            else:
                sigma_full = sigma_coord.sigma_full
            p_ref_sigma = sigma_full[-1]  # near-surface reference
            scale = (p_ref_sigma / jnp.clip(sigma_full, 1e-6, None)) ** config.hyperdiff_pscale
            # Apply scale on the level axis once for the stacked tensor.
            base_diff_stack = base_diff_stack * scale[None, :, None]

        dvor_hat = dvor_hat + base_diff_stack[..., 0]
        ddiv_hat = ddiv_hat + base_diff_stack[..., 1]
        dT_hat = dT_hat + base_diff_stack[..., 2]

    # --- 17. Add physics tendencies if provided ---
    if physics_tendency is not None:
        dvor_hat = dvor_hat + physics_tendency.vor_hat.data
        ddiv_hat = ddiv_hat + physics_tendency.div_hat.data
        dT_hat = dT_hat + physics_tendency.T_hat.data
        dlnps_hat = dlnps_hat + physics_tendency.lnps_hat.data

    # Apply dealiasing truncation to prevent aliasing instability
    if _dealias_3d is not None:
        dvor_hat = dvor_hat * _dealias_3d
        ddiv_hat = ddiv_hat * _dealias_3d
        dT_hat = dT_hat * _dealias_3d
        dlnps_hat = dlnps_hat * _dealias

    # --- 18. Tracer tendencies ---
    # When the input state carries a ``tracers`` dict the tendency
    # must mirror the same pytree structure for SSP-RK ``tree.map``.
    # Each tracer gets:
    #   ∂q/∂t = -∇·(q v_h) + q · D - σ̇ · ∂q/∂σ + (physics tendency)
    # where the spectral PE bridge / orchestrator may inject a grid
    # tendency via ``physics_tendency.tracers[name]``.
    #
    # Tracer values are duck-typed: callers may store ``Field`` objects
    # (with ``.data`` / ``.replace``) or raw JAX arrays.  We extract a
    # raw array for the math, then wrap the result back into the same
    # container so pytree leaves match.
    if state.tracers is None:
        tracers_tend = None
    else:
        is_hybrid = isinstance(sigma_coord, HybridSigmaPressureCoordinate)
        tracers_tend = {}
        for name, value in state.tracers.items():
            q_grid = value.data if hasattr(value, "data") else value
            # Compute advective tendency.  Sigma-coord branch handles
            # vertical advection internally; hybrid path returns
            # horizontal-only and we add vertical advection here using
            # the same ``vertical_advection_hybrid`` helper used for
            # T, u, v above (lines 604, 656-657).  Without this addition
            # tracers had NO vertical transport on the hybrid path —
            # iter-51 audit caught this CRITICAL bug.
            dq_dt_grid = _tracer_advection_gaussian(
                q_grid, u_cos, v_cos, div, sigma_dot,
                sigma_coord, grid,
            )
            if is_hybrid:
                vert_adv_q = vertical_advection_hybrid(
                    q_grid, mass_flux, p_s, sigma_coord
                )
                dq_dt_grid = dq_dt_grid + vert_adv_q
            # Add physics tendency for this tracer when the bridge
            # provided one.
            if (
                physics_tendency is not None
                and physics_tendency.tracers is not None
                and name in physics_tendency.tracers
            ):
                phys_v = physics_tendency.tracers[name]
                phys_grid = phys_v.data if hasattr(phys_v, "data") else phys_v
                dq_dt_grid = dq_dt_grid + phys_grid
            # Wrap back into the original container type so pytree
            # leaves match.
            if hasattr(value, "data") and hasattr(value, "replace"):
                tracers_tend[name] = value.replace(data=dq_dt_grid)
            else:
                tracers_tend[name] = dq_dt_grid

    return SpectralHydrostaticState(
        vor_hat=state.vor_hat.replace(data=dvor_hat),
        div_hat=state.div_hat.replace(data=ddiv_hat),
        T_hat=state.T_hat.replace(data=dT_hat),
        lnps_hat=state.lnps_hat.replace(data=dlnps_hat),
        phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        tracers=tracers_tend,
    )


def compute_spectral_filter(ls, n_max, order=8, cutoff_fraction=0.65):
    """Compute an exponential spectral filter.

    Applies exp(-alpha * (n/n_max)^order) where alpha is chosen so that
    the filter value at n_max equals cutoff_fraction.

    Parameters
    ----------
    ls : jax.Array, shape (n_sh,)
        Total wavenumber for each spectral coefficient.
    n_max : int
        Maximum wavenumber.
    order : int
        Filter order (higher = sharper cutoff).
    cutoff_fraction : float
        Filter value at n=n_max.

    Returns
    -------
    filter : jax.Array, shape (n_sh,)
        Multiplicative filter in [cutoff_fraction, 1].
    """
    alpha = -jnp.log(cutoff_fraction)
    ratio = ls / n_max
    return jnp.exp(-alpha * ratio**order)


def compute_sponge_factor(sigma_full, sponge_sigma, sponge_tau, dt):
    """Compute multiplicative sponge damping factor per level.

    Returns exp(-damping_rate * dt) where damping_rate uses a sin² profile
    ramping from zero at sponge_sigma to 1/sponge_tau at sigma=0.
    This is an implicit (unconditionally stable) sponge filter applied
    after each time step.

    Returns shape (nlev,) array of damping factors in [0, 1].
    """
    sponge_arg = jnp.clip(
        (sponge_sigma - sigma_full) / sponge_sigma, 0.0, 1.0,
    )
    damping_rate = jnp.sin(0.5 * jnp.pi * sponge_arg) ** 2 / sponge_tau
    return jnp.exp(-damping_rate * dt)


def apply_sponge_filter(state, sponge_factor, sponge_factor_T):
    """Apply multiplicative sponge damping to vor, div, and T' at top levels.

    Damps vor and div toward zero.  Damps T perturbations (m != 0 modes)
    toward the zonal mean so that the mean thermal structure is preserved
    but eddy T anomalies are suppressed.

    Parameters
    ----------
    state : SpectralHydrostaticState
    sponge_factor : jax.Array, shape (nlev,)
        Per-level damping factors in [0, 1].
    sponge_factor_T : jax.Array, shape (n_sh, nlev)
        Per-mode T damping factor.  Iter-70: precomputed in
        ``_ensure_sponge_factor`` to avoid the per-step
        ``jnp.where(is_zonal, 1.0, sf)`` cost.  Equals 1.0 at m=0
        and ``sponge_factor[None,:]`` elsewhere.
    """
    sf = sponge_factor[None, :]  # (1, nlev)

    vor_hat_damped = state.vor_hat.data * sf
    div_hat_damped = state.div_hat.data * sf
    T_hat_damped = state.T_hat.data * sponge_factor_T

    return state._replace(
        vor_hat=state.vor_hat.replace(data=vor_hat_damped),
        div_hat=state.div_hat.replace(data=div_hat_damped),
        T_hat=state.T_hat.replace(data=T_hat_damped),
    )


def apply_spectral_filter_to_state(state, spectral_filter):
    """Apply exponential spectral filter to all prognostic fields.

    Parameters
    ----------
    state : SpectralHydrostaticState
    spectral_filter : jax.Array, shape (n_sh,)
        Multiplicative filter per spectral coefficient.
    """
    sf_3d = spectral_filter[:, None]  # (n_sh, 1) for 3D fields
    sf_2d = spectral_filter           # (n_sh,) for 2D fields

    return state._replace(
        vor_hat=state.vor_hat.replace(data=state.vor_hat.data * sf_3d),
        div_hat=state.div_hat.replace(data=state.div_hat.data * sf_3d),
        T_hat=state.T_hat.replace(data=state.T_hat.data * sf_3d),
        lnps_hat=state.lnps_hat.replace(data=state.lnps_hat.data * sf_2d),
    )


def apply_filter_to_tracers(tracers, multiplicative_filter, grid):
    """Apply a per-SH-mode multiplicative filter to grid-space tracers.

    Tracers are stored on the model grid (shape ``(n_lat, n_lon, nlev)``)
    in :class:`SpectralHydrostaticState`, so a single SH round-trip per
    tracer per step is required: ``q_grid → q_hat → q_hat * filter →
    q_grid_filtered``.  The same diagonal filter is reused for the
    spectral exponential (de-aliasing) post-step damping AND the
    implicit hyperdiffusion damping — both are pure pointwise
    multiplications in spectral space, so they collapse into one filter
    applied via one transform pair.

    Parameters
    ----------
    tracers : dict[str, Field | jax.Array] or None
        Tracer dict with grid-space values (shape ``(n_lat, n_lon, nlev)``).
        ``Field`` and raw-array values are duck-typed.  ``None`` is a
        no-op (returns ``None``).
    multiplicative_filter : jax.Array, shape (n_sh,)
        Diagonal filter per spherical-harmonic coefficient.  Typically a
        product of ``spectral_filter`` and ``exp(-nu · eig · dt_eff)``.
    grid : GaussianGrid

    Returns
    -------
    dict or None
        Filtered tracers, container-type-preserving (Field stays Field,
        raw stays raw).
    """
    if tracers is None or multiplicative_filter is None:
        return tracers
    sf_3d = multiplicative_filter[:, None]  # (n_sh, 1) — broadcasts over level
    names = list(tracers.keys())
    if not names:
        return tracers
    # Iter-85: stack all tracers along the trailing axis and do ONE
    # SH analysis + ONE multiply + ONE SH synthesis.  Both
    # ``sh_analysis_3d`` and ``sh_synthesis_3d`` treat any trailing
    # axis as a passive batch — so 5 tracers (q_v, q_c, q_r, q_i, q_s)
    # collapse from 10 SH transforms (5 forward + 5 inverse) to 2.
    sample = tracers[names[0]]
    sample_data = sample.data if hasattr(sample, "data") else sample
    n_lat_q, n_lon_q, nlev_q = sample_data.shape
    n_tracers = len(names)
    # Track each tracer's dtype so we can cast back per-tracer at the end.
    raw = [
        (tracers[n].data if hasattr(tracers[n], "data") else tracers[n])
        for n in names
    ]
    # Promote all to the highest float dtype to avoid silent precision loss
    # in the SH transform (SH transforms internally promote to complex128).
    target_dtype = jnp.result_type(*[r.dtype for r in raw])
    stacked = jnp.stack(
        [r.astype(target_dtype) for r in raw], axis=-1,
    )  # (n_lat, n_lon, nlev, n_tracers)
    flat = stacked.reshape(n_lat_q, n_lon_q, nlev_q * n_tracers)
    hat_flat = sh_analysis_3d(grid, flat)  # (n_sh, nlev * n_tracers)
    hat_filtered_flat = hat_flat * sf_3d
    grid_filtered_flat = sh_synthesis_3d(grid, hat_filtered_flat)
    grid_filtered = grid_filtered_flat.reshape(
        n_lat_q, n_lon_q, nlev_q, n_tracers,
    )
    out = {}
    for i, name in enumerate(names):
        value = tracers[name]
        out_data = grid_filtered[..., i]
        if hasattr(value, "data") and hasattr(value, "replace"):
            out[name] = value.replace(data=out_data.astype(value.data.dtype))
        else:
            out[name] = out_data.astype(value.dtype)
    return out


# =============================================================================
# Model class
# =============================================================================

class SpectralPrimitiveEquationModel:
    """Spectral primitive equation model on the sphere.

    Parameters
    ----------
    grid : GaussianGrid
        Gaussian grid with precomputed SH transform matrices.
    sigma_coord : SigmaCoordinate
        Vertical sigma coordinate.
    config : SpectralPEConfig, optional
        Model configuration.
    """

    def __init__(
        self,
        grid: GaussianGrid,
        sigma_coord: SigmaCoordinate,
        config: SpectralPEConfig | None = None,
        *,
        allow_unsupported_backend: bool = False,
        legoesm_config=None,
    ):
        # Guard: spectral transforms require global data on a single rank.
        # Proper gather/scatter support is a large architectural change;
        # for now, raise a clear error so users know to switch dycores.
        _mpi_world = 1
        try:
            from mpi4py import MPI
            _mpi_world = MPI.COMM_WORLD.Get_size()
        except (ImportError, RuntimeError, OSError):
            # mpi4py installed but libmpi missing/incompatible → assume
            # single-rank. Distributed launches will surface real MPI
            # errors elsewhere.
            pass
        if _mpi_world > 1:
            raise RuntimeError(
                "Spectral PE dynamics do not yet support distributed (MPI) "
                "execution.  Spherical harmonic transforms require global "
                "data on every rank.  Use a single process or switch to a "
                "finite-volume dycore for MPI runs."
            )

        self.sigma_coord = sigma_coord
        cfg = config or SpectralPEConfig()

        # Force implicit hyperdiffusion for leapfrog integrators
        if cfg.time_integrator in ("leapfrog", "leapfrog_si") and not cfg.implicit_hyperdiff:
            import warnings
            warnings.warn(
                f"Explicit hyperdiffusion is unstable with {cfg.time_integrator} "
                f"time integration. Forcing implicit_hyperdiff=True.",
                stacklevel=2,
            )
            cfg = cfg._replace(implicit_hyperdiff=True)

        # Warn if dealiasing is off
        if cfg.dealiasing_fraction == 0.0:
            import warnings
            warnings.warn(
                "dealiasing_fraction=0.0: spectral aliasing from cubic "
                "nonlinearities is not suppressed. Set dealiasing_fraction=0.667 "
                "for production runs.",
                stacklevel=2,
            )

        self.config = cfg
        self._use_cpu_for_spectral = False
        self._cpu_device = None
        self._default_device = None
        self._si_data = None
        self._si_dt = None
        self._si_data_lf = None  # SI data for leapfrog (dt_eff = 2*dt)
        self._si_dt_lf = None
        self._sponge_factor = None
        self._sponge_factor_T = None  # iter-70: precomputed (n_sh, nlev) T factor
        self._sponge_dt = None
        # Leapfrog state management
        self._state_prev = None  # Previous time level for leapfrog
        # Precompute implicit hyperdiffusion filter (unconditionally stable)
        self._hyperdiff_filter = None
        self._hyperdiff_filter_div = None  # iter-69: precomputed hf**2
        self._hyperdiff_filter_dt = None
        # Tracer filter (combined spectral + implicit hyperdiff) is
        # precomputed lazily because it depends on dt.  ``None`` means
        # neither knob is active and we skip the SH round-trip on
        # tracers entirely.
        self._tracer_filter = None
        self._tracer_filter_dt = None
        # Iter-3: anchored mass target.  Lazily filled by ``step()`` on
        # first call when ``fix_mass`` and ``anchor_mass_to_initial`` are
        # both set.  Stored in fp64 so per-step corrections aren't
        # contaminated by storage round-trips.
        self._target_mass = None

        if legoesm_config is not None:
            allow_unsupported_backend = bool(
                legoesm_config.get(
                    "atmosphere.spectral.allow_unsupported", False
                )
            )

        # --- Metal detection MUST happen before any float64 computation ---
        placement = place_spectral_grid(
            grid, allow_unsupported=allow_unsupported_backend
        )
        self.grid = placement.grid
        self._use_cpu_for_spectral = placement.use_cpu_for_spectral
        self._cpu_device = placement.cpu_device
        self._default_device = placement.default_device

        # Precompute spectral filter (time-independent) — uses self.grid
        # which is now on CPU when Metal is active.
        self._spectral_filter = None
        if self.config.spectral_filter_strength > 0:
            self._spectral_filter = compute_spectral_filter(
                self.grid.ls,
                self.grid.n_max,
                order=self.config.spectral_filter_order,
                cutoff_fraction=self.config.spectral_filter_strength,
            )

        # State-truncation mask for the Orszag 2/3 rule (same fix as the
        # spectral-SW twin, 2026-07-12): the TENDENCIES of vor/div/T/lnps
        # are masked inside ``spectral_pe_tendencies``, which holds masked
        # modes CONSTANT — it cannot remove upper-third power already in
        # the state (mountainous ICs, restarts, user states).  Truncating
        # the state post-step enforces the band-limit the 2/3 rule
        # assumes; exact no-op for band-limited states.
        self._dealias_state = None
        if self.config.dealiasing_fraction > 0.0:
            self._dealias_state = dealiasing_mask(
                self.grid, self.config.dealiasing_fraction,
            )

        if self.config.si_substeps < 1:
            raise ValueError(
                f"si_substeps must be >= 1, got {self.config.si_substeps!r}",
            )

    def _ensure_si_data(self, dt: float):
        """Lazily precompute semi-implicit matrices and refresh when dt changes."""
        if not self.config.semi_implicit:
            return

        dt_si = float(dt) / float(self.config.si_substeps)
        if self._si_data is None or self._si_dt != dt_si:
            self._si_data = precompute_si_matrices(
                self.grid, self.sigma_coord,
                T_ref=self.config.si_T_ref,
                alpha=self.config.si_alpha,
                dt=dt_si,
            )
            self._si_dt = dt_si

    def _ensure_sponge_factor(self, dt: float):
        """Lazily precompute sponge damping factors and refresh when dt changes.

        Iter-70: precompute the per-mode T sponge factor as a single
        ``(n_sh, nlev)`` array (zonal m=0 modes preserved at 1.0,
        non-zonal modes get the sponge factor).  Avoids the per-step
        ``jnp.where(is_zonal, 1.0, sf)`` op in
        ``apply_sponge_filter``.
        """
        if self.config.sponge_tau <= 0:
            return
        if self._sponge_factor is not None and self._sponge_dt == dt:
            return

        _hybrid = isinstance(self.sigma_coord, HybridSigmaPressureCoordinate)
        if _hybrid:
            sigma_full = self.sigma_coord.A_full + self.sigma_coord.B_full
        else:
            sigma_full = self.sigma_coord.sigma_full

        self._sponge_factor = compute_sponge_factor(
            sigma_full, self.config.sponge_sigma, self.config.sponge_tau, dt,
        )
        # Per-(n_sh, nlev) factor for T: 1.0 at m=0, sf elsewhere.
        _is_zonal = (self.grid.ms == 0)[:, None]  # (n_sh, 1) bool
        self._sponge_factor_T = jnp.where(
            _is_zonal, 1.0, self._sponge_factor[None, :],
        )  # (n_sh, nlev)
        self._sponge_dt = dt

    def _ensure_hyperdiff_filter(self, dt: float):
        """Lazily precompute implicit hyperdiffusion filter and the
        ``hf**2`` divergence variant.
        """
        if not self.config.implicit_hyperdiff or self.config.hyperdiff_coeff <= 0:
            return
        if self._hyperdiff_filter is not None and self._hyperdiff_filter_dt == dt:
            return
        nu = self.config.hyperdiff_coeff
        order = self.config.hyperdiff_order
        # Eigenvalue: -[n(n+1)/a^2]^order
        eig = (self.grid.ls * (self.grid.ls + 1) / self.grid.radius ** 2) ** order
        # For leapfrog, effective dt is 2*dt
        integrator = self.config.time_integrator.lower()
        dt_eff = 2.0 * dt if 'leapfrog' in integrator else dt
        # Multiplicative filter: exp(-nu * eig * dt_eff)
        self._hyperdiff_filter = jnp.exp(-nu * eig * dt_eff)
        # Iter-69: precompute the squared variant used for divergence
        # (``div`` gets 2× stronger damping than ``vor`` / ``T``).
        # Avoids the per-step ``hf ** 2`` op and broadcasts cleanly.
        self._hyperdiff_filter_div = self._hyperdiff_filter ** 2
        self._hyperdiff_filter_dt = dt

    def _apply_implicit_hyperdiff(self, state):
        """Apply implicit (multiplicative) hyperdiffusion filter.

        Divergence gets 2x stronger damping than vorticity and temperature
        to preferentially suppress gravity wave noise from nonlinear
        baroclinic eddy breakdown (standard practice in operational GCMs).
        """
        if self._hyperdiff_filter is None:
            return state
        hf_3d = self._hyperdiff_filter[:, None]  # (n_sh, 1) for 3D fields
        hf_div_3d = self._hyperdiff_filter_div[:, None]
        return state._replace(
            vor_hat=state.vor_hat.replace(data=state.vor_hat.data * hf_3d),
            div_hat=state.div_hat.replace(data=state.div_hat.data * hf_div_3d),
            T_hat=state.T_hat.replace(data=state.T_hat.data * hf_3d),
            # lnps and phis are NOT diffused (mass conservation)
        )

    def _ensure_tracer_filter(self, dt: float):
        """Lazily precompute the combined post-step filter applied to
        tracers in grid space.

        The filter is the product of:

        * the spectral exponential filter (de-aliasing) — same factor
          used by ``apply_spectral_filter_to_state`` for vor/div/T;
        * the implicit hyperdiffusion factor ``exp(-nu · eig · dt_eff)``
          — same eigenvalue used by ``_ensure_hyperdiff_filter``.

        Both are diagonal in spectral space, so we collapse them into a
        single ``(n_sh,)`` multiplier applied per SH mode via one SH
        round-trip per tracer per step (see
        :func:`apply_filter_to_tracers`).

        Effective time step matches the integrator: ``dt_eff = 2·dt``
        for leapfrog (which spans 2 model dt per step), ``dt_eff = dt``
        for SSP-RK and semi-implicit RK.

        Stored as ``self._tracer_filter`` (or ``None`` when neither knob
        is active — the apply pathway then short-circuits).
        """
        if self._tracer_filter is not None and self._tracer_filter_dt == dt:
            return
        components = []
        if self._spectral_filter is not None:
            components.append(self._spectral_filter)
        if self.config.hyperdiff_coeff > 0:
            nu = self.config.hyperdiff_coeff
            order = self.config.hyperdiff_order
            eig = (
                self.grid.ls * (self.grid.ls + 1) / self.grid.radius ** 2
            ) ** order
            integrator = self.config.time_integrator.lower()
            dt_eff = 2.0 * dt if 'leapfrog' in integrator else dt
            components.append(jnp.exp(-nu * eig * dt_eff))
        if not components:
            self._tracer_filter = None
            self._tracer_filter_dt = dt
            return
        combined = components[0]
        for c in components[1:]:
            combined = combined * c
        self._tracer_filter = combined
        self._tracer_filter_dt = dt

    def _apply_tracer_filter(self, state):
        """Apply the precomputed tracer filter to ``state.tracers`` (no-op
        when filter or tracers are absent)."""
        if self._tracer_filter is None or state.tracers is None:
            return state
        return state._replace(
            tracers=apply_filter_to_tracers(
                state.tracers, self._tracer_filter, self.grid,
            )
        )

    def _ensure_si_data_leapfrog(self, dt: float):
        """Precompute SI matrices for leapfrog (dt_eff = 2*dt)."""
        dt_eff = 2.0 * float(dt)
        if self._si_data_lf is None or self._si_dt_lf != dt_eff:
            self._si_data_lf = precompute_si_matrices(
                self.grid, self.sigma_coord,
                T_ref=self.config.si_T_ref,
                alpha=self.config.si_alpha,
                dt=dt_eff,
            )
            self._si_dt_lf = dt_eff

    def _do_step(self, state, dt, tendency_fn,
                 si_data=None, sponge_factor=None, target_mass=None):
        """Core step: explicit RK3/RK54 or semi-implicit RK3, then sponge.

        ``si_data`` and ``sponge_factor`` are passed as **dynamic args**
        (not read off ``self``).  This is required because :func:`step`
        wraps this in a ``@jax.jit`` with ``static_argnums=(0, ...)`` —
        JAX caches on object identity for ``self``, which means a
        Python-level mutation of ``self._si_data`` after a dt change
        is invisible to the cache and the compiled function would
        keep using stale matrices (verified: ``maxdiff_after_dt_change_vs_fresh
        = 2.45e-05`` at dt=300 → 600).  Threading them as dynamic
        inputs lets XLA capture them as runtime tensors without
        recompilation.
        """
        if si_data is None:
            si_data = self._si_data
        if sponge_factor is None:
            sponge_factor = self._sponge_factor

        if self.config.semi_implicit:
            n_substeps = int(self.config.si_substeps)
            dt_si = dt / float(n_substeps)

            if n_substeps == 1:
                result = ssp_rk3_step_si(
                    state, tendency_fn, dt_si, si_data, self.grid,
                )
            else:
                def si_substep(_, s):
                    return ssp_rk3_step_si(
                        s, tendency_fn, dt_si, si_data, self.grid,
                    )
                result = jax.lax.fori_loop(0, n_substeps, si_substep, state)
        else:
            result = dispatch_integrator(
                state, tendency_fn, dt, self.config.time_integrator,
            )

        # Apply implicit sponge filter (unconditionally stable)
        if self._sponge_factor is not None:
            result = apply_sponge_filter(result, self._sponge_factor, self._sponge_factor_T)

        # Apply spectral filter (damps highest wavenumbers)
        if self._spectral_filter is not None:
            result = apply_spectral_filter_to_state(result, self._spectral_filter)

        # 2/3-rule STATE truncation (masked tendencies alone freeze, not
        # remove, pre-existing upper-third modes — see __init__ note).
        result = self._apply_state_truncation(result)

        # Implicit hyperdiffusion for the non-leapfrog integrators.
        # ``implicit_hyperdiff=True`` disables the explicit tendency term,
        # so without this multiplicative filter the SSP/RK paths ran with
        # NO vor/div/T diffusion at all (tracers still got theirs via
        # ``_apply_tracer_filter``).  ``step()`` precomputes the filter
        # via ``_ensure_hyperdiff_filter(dt)`` before entering the JIT —
        # same read-from-self pattern as ``_spectral_filter`` /
        # ``_tracer_filter`` above.
        if self.config.implicit_hyperdiff:
            result = self._apply_implicit_hyperdiff(result)

        # Apply combined spectral-filter + implicit-hyperdiff to tracers
        # via one SH round-trip per tracer (no-op when neither knob is
        # active or ``state.tracers is None``).
        result = self._apply_tracer_filter(result)

        # Iter-3: anchored-mass fixer.  ``target_mass`` is threaded in as
        # a TRACED argument (codex 2026-07-12): a closure-captured
        # ``self._target_mass`` is baked at trace time, so a later
        # ``set_target_mass()`` / ``reset_target_mass()`` + re-anchor was
        # silently ignored by the compiled step.  ``None`` (fixer off or
        # pre-snapshot) keeps the structure stable because ``step()``
        # snapshots BEFORE the first JIT call whenever anchoring is on.
        if (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and target_mass is not None):
            result = self._apply_mass_fixer(result, target_mass)

        return result

    def _apply_state_truncation(self, state):
        """Truncate the prognostic state to the de-aliased 2/3 band.

        Multiplies vor/div/T (3D) and lnps (2D) by the 0/1 dealiasing
        mask — the SAME fields whose tendencies are masked in
        ``spectral_pe_tendencies``.  ``phis_hat`` is static forcing and
        is never truncated; grid-space tracers are handled by the tracer
        filter.  No-op when ``dealiasing_fraction == 0``.
        """
        if self._dealias_state is None:
            return state
        m2 = self._dealias_state
        m3 = m2[:, None]
        return state._replace(
            vor_hat=state.vor_hat.replace(data=state.vor_hat.data * m3),
            div_hat=state.div_hat.replace(data=state.div_hat.data * m3),
            T_hat=state.T_hat.replace(data=state.T_hat.data * m3),
            lnps_hat=state.lnps_hat.replace(data=state.lnps_hat.data * m2),
        )

    def _apply_mass_fixer(self, state, target_mass=None):
        """Rescale ``lnps_hat[0]`` so the global integral matches ``_target_mass``.

        With the spectral basis ``(4π)``-normalised on the unit sphere
        (``sh_analysis(ones)[0] = sqrt(4π)``), adding a constant ``Δ`` to
        ``lnps`` in physical space is equivalent to adding ``Δ·sqrt(4π)``
        to ``lnps_hat[0]``.  We solve ``Δ = log(target / current)`` so
        that ``exp(lnps + Δ) = (target/current) · exp(lnps)`` and the
        integral is restored multiplicatively (gradients of ``p_s`` are
        preserved exactly — same property as the cubed-sphere/lat-lon
        ``fix_ps_mass`` additive uniform correction, just expressed in
        log-space because ``lnps`` is the prognostic variable).
        """
        if target_mass is None:
            target_mass = self._target_mass
        lnps_grid = sh_synthesis(self.grid, state.lnps_hat.data)
        p_s_grid = jnp.exp(lnps_grid)
        acc = jnp.float64
        mass_now = jnp.sum(
            p_s_grid.astype(acc) * self.grid.grid_area.astype(acc),
        )
        log_scale = jnp.log(target_mass / mass_now)
        # sqrt(4π) is the (0,0) coefficient of a constant=1 field under
        # the (4π)-normalised real-SH convention this module uses.
        sqrt_4pi = jnp.sqrt(jnp.asarray(4.0 * jnp.pi, dtype=acc))
        lnps_hat_new = state.lnps_hat.data.at[0].add(
            (log_scale * sqrt_4pi).astype(state.lnps_hat.data.dtype),
        )
        return state._replace(
            lnps_hat=state.lnps_hat.replace(data=lnps_hat_new),
        )

    def _maybe_snapshot_target_mass(self, state) -> None:
        """First-call anchored-mass snapshot, shared by step()/integrate().

        Snapshotting during an outer jit/grad trace would store a Tracer
        on ``self`` (leaks; the next eager step raises
        UnexpectedTracerError) — refuse with a usable remedy instead
        (codex 2026-07-12 rounds 2-3: BOTH entry points need this guard).
        """
        if not (self.config.fix_mass
                and self.config.anchor_mass_to_initial
                and self._target_mass is None):
            return
        if isinstance(state.lnps_hat.data, jax.core.Tracer):
            raise ValueError(
                "anchor_mass_to_initial cannot take its first-mass "
                "snapshot inside a jit/grad trace (it would store a "
                "tracer on the model). Call set_target_mass(<concrete "
                "fp64 mass>) or take one eager step() first."
            )
        self._target_mass = self._compute_initial_mass(state)

    def _compute_initial_mass(self, state):
        """Compute total dry mass ``∫ p_s dA`` in fp64 from a spectral state."""
        lnps_grid = sh_synthesis(self.grid, state.lnps_hat.data)
        p_s_grid = jnp.exp(lnps_grid)
        return jnp.sum(
            p_s_grid.astype(jnp.float64)
            * self.grid.grid_area.astype(jnp.float64),
        )

    def compute_mass(self, state) -> jax.Array:
        """Public alias of ``_compute_initial_mass`` (iter-22).

        Provides the same name as the cube / lat-lon / MPAS PE
        ``compute_mass(state)`` helpers — single API across all four
        hydrostatic dycores.
        """
        return self._compute_initial_mass(state)

    def reset_target_mass(self) -> None:
        """Clear the anchored mass target (iter-18; see iter-4 SW twin)."""
        self._target_mass = None

    def set_target_mass(self, target_mass) -> None:
        """Explicitly set the anchored mass target (iter-19)."""
        self._target_mass = target_mass

    def step(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
        forcing_data=None,
    ) -> SpectralHydrostaticState:
        """Advance one time step, optionally with physics forcing.

        Dispatches to leapfrog+SI or SSP-RK3 based on config.time_integrator.

        Parameters
        ----------
        forcing_data : pytree of jax.Array, optional
            Dynamic forcing data passed as a TRACED argument to
            ``physics_fn(state, grid, sigma_coord, forcing_data)``.
            When provided, the JIT cache is keyed by physics_fn
            identity (static) but the forcing data is treated as a
            dynamic argument — JAX retraces only if the pytree
            *structure* (not values) changes.  This avoids the
            stale-day pathology of closure-captured Python state
            (audit iter-74).  When None, falls back to the legacy
            3-arg ``physics_fn(state, grid, sigma_coord)`` API for
            backward compatibility.
        """
        # The spectral PE cannot thread a PhysicsState carry (transform
        # space has no per-column carry slot — the driver's _run_spectral
        # refuses stateful physics for the same reason).  Refuse a
        # ``make_physics`` output tagged _requires_phys_state here too so a
        # direct ``model.step(physics_fn=...)`` / step_with_physics loop
        # cannot silently reseed prognostic physics every step (#405/#413).
        refuse_unthreaded_stateful_physics(
            physics_fn, None, where="Spectral PE step()")
        # Iter-3: anchor mass on first call (outside JIT; the fp64 scalar
        # is then THREADED into the jitted step as a traced arg so a
        # later reset/set_target_mass is honored — codex 2026-07-12).
        self._maybe_snapshot_target_mass(state)

        integrator = self.config.time_integrator.lower()
        if integrator in ("leapfrog", "leapfrog_si"):
            return self._leapfrog_step(state, dt, physics_fn, forcing_data)

        self._ensure_si_data(dt)
        self._ensure_sponge_factor(dt)
        self._ensure_hyperdiff_filter(dt)
        self._ensure_tracer_filter(dt)
        if forcing_data is not None:
            return self._step_with_forcing_jit(
                state, dt, physics_fn, forcing_data, self._target_mass,
            )
        return self._step_jit(state, dt, physics_fn, self._target_mass)

    def _leapfrog_step(self, state, dt, physics_fn=None, forcing_data=None):
        """Leapfrog + SI step with Robert-Asselin filter + implicit diffusion.

        First call: forward Euler + SI (startup).
        Subsequent calls: leapfrog + SI + RA filter + implicit hyperdiffusion.

        ``forcing_data`` is threaded through to physics_fn as a TRACED
        pytree argument when provided (iter-95 extension to the iter-92
        forcing_data API).
        """
        self._ensure_sponge_factor(dt)
        self._ensure_hyperdiff_filter(dt)
        self._ensure_tracer_filter(dt)

        if self._state_prev is None:
            # --- First step: forward Euler + SI ---
            self._ensure_si_data(dt)  # SI matrices for dt
            # Also precompute leapfrog SI for next step (avoids stale jit)
            self._ensure_si_data_leapfrog(dt)
            if forcing_data is not None:
                result = self._euler_si_with_forcing_jit(
                    state, dt, physics_fn, forcing_data,
                )
            else:
                result = self._euler_si_jit(state, dt, physics_fn)
            # Apply sponge and spectral filter
            if self._sponge_factor is not None:
                result = apply_sponge_filter(result, self._sponge_factor, self._sponge_factor_T)
            if self._spectral_filter is not None:
                result = apply_spectral_filter_to_state(result, self._spectral_filter)
            # 2/3-rule state truncation (see _apply_state_truncation)
            result = self._apply_state_truncation(result)
            # Implicit hyperdiffusion (unconditionally stable)
            result = self._apply_implicit_hyperdiff(result)
            # Same combined filter applied to grid-space tracers
            result = self._apply_tracer_filter(result)
            # Iter-3: anchored mass fixer (leapfrog Euler-startup branch).
            if (self.config.fix_mass
                    and self.config.anchor_mass_to_initial
                    and self._target_mass is not None):
                result = self._apply_mass_fixer(result)
            # Store the TRUNCATED input as the leapfrog time-(n-1) level:
            # an unmasked _state_prev feeds its upper-third power straight
            # back through the next leapfrog combination + RA filter
            # (codex 2026-07-12 micro-review).
            self._state_prev = self._apply_state_truncation(state)
            return result
        else:
            # --- Leapfrog + SI ---
            self._ensure_si_data_leapfrog(dt)
            if forcing_data is not None:
                state_np1 = self._leapfrog_si_with_forcing_jit(
                    state, self._state_prev, dt, physics_fn, forcing_data,
                )
            else:
                state_np1 = self._leapfrog_si_jit(
                    state, self._state_prev, dt, physics_fn,
                )
            # Apply sponge and spectral filter
            if self._sponge_factor is not None:
                state_np1 = apply_sponge_filter(
                    state_np1, self._sponge_factor, self._sponge_factor_T,
                )
            if self._spectral_filter is not None:
                state_np1 = apply_spectral_filter_to_state(
                    state_np1, self._spectral_filter,
                )
            # 2/3-rule state truncation (see _apply_state_truncation)
            state_np1 = self._apply_state_truncation(state_np1)
            # Implicit hyperdiffusion (unconditionally stable with leapfrog)
            state_np1 = self._apply_implicit_hyperdiff(state_np1)
            # Same combined filter applied to grid-space tracers
            state_np1 = self._apply_tracer_filter(state_np1)
            # Robert-Asselin-Williams filter on time-n / time-(n+1) states.
            gamma = self.config.robert_asselin_coeff
            alpha = self.config.robert_asselin_alpha
            if gamma > 0:
                state_n_filtered, state_np1_filtered = robert_asselin_filter(
                    self._state_prev, state, state_np1, gamma, alpha=alpha,
                )
                # The RA mix re-injects O(γ) upper-third power from the
                # time-n / time-(n-1) states into BOTH outputs — truncate
                # them so the band-limit is exact on the returned state
                # AND on the stored _state_prev (codex 2026-07-12
                # micro-review: an unmasked _state_prev feeds the leaked
                # power back through every subsequent leapfrog step).
                state_n_filtered = self._apply_state_truncation(state_n_filtered)
                state_np1_filtered = self._apply_state_truncation(state_np1_filtered)
            else:
                state_n_filtered = self._apply_state_truncation(state)
                state_np1_filtered = state_np1
            # Iter-3: anchored mass fixer (leapfrog body).  Applied to
            # the time-(n+1) state AFTER the Robert-Asselin filter so
            # the computational mode is damped first, then mass is
            # restored exactly to the initial integral.
            if (self.config.fix_mass
                    and self.config.anchor_mass_to_initial
                    and self._target_mass is not None):
                state_np1_filtered = self._apply_mass_fixer(state_np1_filtered)
            self._state_prev = state_n_filtered
            return state_np1_filtered

    def step_with_physics(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
    ) -> SpectralHydrostaticState:
        """Backward-compatible wrapper for step() with physics."""
        return self.step(state, dt, physics_fn=physics_fn)

    def _make_tendency_fn(self, physics_fn, forcing_data=_NO_FORCING):
        """Build the SI/leapfrog tendency closure (one per step at trace time).

        Calls ``physics_fn`` with ``(s, grid, sigma_coord)`` — or the 4-arg
        ``(..., forcing_data)`` signature when ``forcing_data`` is supplied —
        then :func:`spectral_pe_tendencies`.  The ``_NO_FORCING`` sentinel
        (not ``None``) distinguishes "no forcing arg" from a forcing value of
        ``None``, so the 3-arg and 4-arg step methods keep their exact prior
        physics_fn call.  Replaces the byte-identical closure inlined in all
        seven step methods.
        """
        def tendency_fn(s):
            phys = None
            if physics_fn is not None:
                if forcing_data is _NO_FORCING:
                    _phys_result = physics_fn(s, self.grid, self.sigma_coord)
                else:
                    _phys_result = physics_fn(
                        s, self.grid, self.sigma_coord, forcing_data,
                    )
                phys = _phys_result[0] if type(_phys_result) is tuple else _phys_result
            return spectral_pe_tendencies(
                s, self.grid, self.sigma_coord, self.config, phys,
            )
        return tendency_fn

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _euler_si_jit(self, state, dt, physics_fn=None):
        """JIT-compiled Euler + SI step (leapfrog startup), optionally with physics.

        ``dt`` is STATIC (codex 2026-07-12): the body closure-captures the
        dt-dependent ``self._si_data`` — with a traced ``dt`` a dt change
        does not retrace, so the compiled step kept using the STALE SI
        matrices from the first dt.  A static ``dt`` keys the JIT cache on
        the value, so ``_ensure_si_data(dt)`` + retrace stay consistent.
        """
        tendency_fn = self._make_tendency_fn(physics_fn)
        from legoesm.timestepping.semi_implicit import euler_si_step
        return euler_si_step(state, tendency_fn, dt, self._si_data, self.grid)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _euler_si_with_forcing_jit(self, state, dt, physics_fn, forcing_data):
        """Iter-95: Euler + SI step with TRACED forcing_data threading.

        ``static_argnums=(0, 2, 3)``: ``self``, ``dt`` and ``physics_fn``
        are static (dt keys the cache — see ``_euler_si_jit``);
        ``forcing_data`` stays traced so per-step values never retrace.
        """
        tendency_fn = self._make_tendency_fn(physics_fn, forcing_data)
        return euler_si_step(state, tendency_fn, dt, self._si_data, self.grid)

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def _leapfrog_si_jit(self, state_n, state_nm1, dt, physics_fn=None):
        """JIT-compiled leapfrog + SI step, optionally with physics.

        ``dt`` static — closure-captures ``self._si_data_lf`` (see
        ``_euler_si_jit``).
        """
        tendency_fn = self._make_tendency_fn(physics_fn)
        return leapfrog_si_step(
            state_n, state_nm1, tendency_fn, dt, self._si_data_lf, self.grid,
        )

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def _leapfrog_si_with_forcing_jit(
        self, state_n, state_nm1, dt, physics_fn, forcing_data,
    ):
        """Iter-95: leapfrog + SI step with TRACED forcing_data threading.

        ``self``, ``dt``, ``physics_fn`` static (dt keys the cache — see
        ``_euler_si_jit``); ``forcing_data`` traced.
        """
        tendency_fn = self._make_tendency_fn(physics_fn, forcing_data)
        return leapfrog_si_step(
            state_n, state_nm1, tendency_fn, dt, self._si_data_lf, self.grid,
        )

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_jit(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
        target_mass=None,
    ) -> SpectralHydrostaticState:
        """JIT-compiled inner step (SI matrices already precomputed), optionally with physics.

        ``dt`` is a **static** arg — and, since codex 2026-07-12,
        ``static_argnums`` actually says so (it previously read
        ``(0, 3)`` while this docstring claimed dt-static, so a dt
        change silently reused the STALE SI matrices / sponge / hyperdiff
        / tracer filters captured at the first trace).  Iter-211 measured
        a ~60 % throughput regression on spectral T21 GPU when ``dt`` was
        traced (479 → 284 sps): static ``dt`` lets the dt-dependent
        filter/matrix closures constant-fold, and the JIT cache keyed on
        the dt value makes ``_ensure_*`` + retrace consistent.

        ``target_mass`` is TRACED (scalar or None): the anchored-mass
        target can be reset/re-anchored between calls without a stale
        closure capture (structure is stable — ``step()`` snapshots
        before the first JIT call whenever anchoring is on).
        """
        tendency_fn = self._make_tendency_fn(physics_fn)

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = self._do_step(state_cpu, dt, tendency_fn,
                                       target_mass=target_mass)
            return jax.device_put(result_cpu, self._default_device)

        return self._do_step(state, dt, tendency_fn, target_mass=target_mass)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_with_forcing_jit(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn,
        forcing_data,
        target_mass=None,
    ) -> SpectralHydrostaticState:
        """JIT-compiled step with TRACED ``forcing_data``.

        ``forcing_data`` is a non-static pytree — values can change
        between calls without triggering retrace (only structure
        changes do).  This solves the iter-74 ``_DayRef`` JIT-cache
        stale-day issue: callers pass ``day``, ``sst``, ``sic`` etc.
        as a JAX-array dict, and JAX retraces ONCE at first call but
        treats the values as dynamic for all subsequent calls.

        physics_fn is called with ``(state, grid, sigma_coord,
        forcing_data)`` — a 4-arg signature.  Existing 3-arg
        physics_fn implementations need to be extended.

        ``static_argnums=(0, 2, 3)``: ``self``, ``dt``, ``physics_fn``
        static (dt keys the cache so the dt-dependent filter/matrix
        closures stay fresh — see ``_step_jit``); ``forcing_data`` and
        ``target_mass`` traced.
        """
        tendency_fn = self._make_tendency_fn(physics_fn, forcing_data)

        if self._use_cpu_for_spectral:
            state_cpu = jax.device_put(state, self._cpu_device)
            result_cpu = self._do_step(state_cpu, dt, tendency_fn,
                                       target_mass=target_mass)
            return jax.device_put(result_cpu, self._default_device)

        return self._do_step(state, dt, tendency_fn, target_mass=target_mass)

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def _step_on_cpu(
        self,
        state: SpectralHydrostaticState,
        dt: float,
        physics_fn=None,
        target_mass=None,
    ) -> SpectralHydrostaticState:
        """Step on CPU without device transfers, optionally with physics.

        Same static-arg pattern as :func:`_step_jit` (``dt`` static,
        ``target_mass`` traced).
        """
        tendency_fn = self._make_tendency_fn(physics_fn)
        return self._do_step(state, dt, tendency_fn, target_mass=target_mass)

    def integrate(
        self,
        state: SpectralHydrostaticState,
        duration: float,
        dt: float,
        save_every: int = 1,
        physics_fn=None,
    ) -> tuple[SpectralHydrostaticState, list]:
        """Integrate forward for a given duration (Python loop).

        On Metal, batches CPU transfers: transfer state to CPU once,
        run all steps on CPU, then transfer results back to Metal.
        This avoids per-step CPU↔Metal round-trips.
        """
        n_steps = int(duration / dt)
        self._ensure_si_data(dt)
        self._ensure_sponge_factor(dt)
        self._ensure_hyperdiff_filter(dt)
        self._ensure_tracer_filter(dt)
        # Anchored-mass snapshot: step() does this itself on the direct
        # path, but the batched-CPU path calls _step_on_cpu directly —
        # snapshot here so BOTH paths anchor on the true initial state
        # (codex 2026-07-12: the batched path previously never anchored).
        self._maybe_snapshot_target_mass(state)

        if self._use_cpu_for_spectral:
            return self._integrate_on_cpu(state, n_steps, dt, save_every, physics_fn)

        trajectory = [state]
        for i in range(n_steps):
            state = self.step(state, dt, physics_fn=physics_fn)
            if (i + 1) % save_every == 0:
                trajectory.append(state)
        return state, trajectory

    def _integrate_on_cpu(self, state, n_steps, dt, save_every, physics_fn=None):
        """Batch integration on CPU: transfer once, not per step."""
        # The batch loop drives the generic SSP/SI ``_step_on_cpu`` only;
        # it has no two-time-level leapfrog state machine (``_state_prev``,
        # RA filter, Euler startup).  Silently stepping a leapfrog config
        # through it produces a DIFFERENT scheme than ``step()`` (measured
        # 1.2e-4 T_hat divergence in one step) — refuse loudly instead
        # (codex 2026-07-12 round 2; pre-existing gap).
        if self.config.time_integrator.lower() in ("leapfrog", "leapfrog_si"):
            raise NotImplementedError(
                "Batched-CPU (Metal) integrate() does not implement the "
                "leapfrog integrators; use per-step step() (which runs the "
                "leapfrog state machine) or an SSP/SI time_integrator."
            )
        state_cpu = jax.device_put(state, self._cpu_device)
        trajectory_cpu = [state_cpu]

        # Refresh dt-dependent matrices on the host once before stepping.
        # (integrate() already ensured the hyperdiff/tracer filters and
        # took the anchored-mass snapshot; direct callers get them here.)
        self._ensure_si_data(dt)
        self._ensure_sponge_factor(dt)
        self._ensure_hyperdiff_filter(dt)
        self._ensure_tracer_filter(dt)
        self._maybe_snapshot_target_mass(state_cpu)
        for i in range(n_steps):
            state_cpu = self._step_on_cpu(
                state_cpu, dt, physics_fn, self._target_mass,
            )
            if (i + 1) % save_every == 0:
                trajectory_cpu.append(state_cpu)

        # Transfer back to Metal
        state_out = jax.device_put(state_cpu, self._default_device)
        trajectory_out = [
            jax.device_put(s, self._default_device) for s in trajectory_cpu
        ]
        return state_out, trajectory_out


# =============================================================================
# Initialization helpers
# =============================================================================

def isothermal_rest_state_spectral(
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
    T_init: float = 300.0,
    p_s_init: float = constants.p_ref,
    phis: jnp.ndarray | None = None,
    perturbation_amplitude: float = 1.0,
    seed: int = 42,
    tracers: dict | None = None,
) -> SpectralHydrostaticState:
    """Create an isothermal rest-state initial condition in spectral space.

    All fields are at rest (zero winds) with uniform temperature and
    uniform surface pressure.  A small random temperature perturbation
    is added at the lowest level to break symmetry and trigger baroclinic
    instability, matching the cubed-sphere and lat-lon Held-Suarez inits.

    Parameters
    ----------
    grid : GaussianGrid
        Spectral/Gaussian grid.
    sigma_coord : SigmaCoordinate
        Vertical coordinate.
    T_init : float
        Initial temperature [K].
    p_s_init : float
        Initial surface pressure [Pa].
    phis : jnp.ndarray or None
        Surface geopotential [m^2/s^2], shape (n_lat, n_lon). If None,
        flat terrain is used. When provided, surface pressure is reduced
        hydrostatically: p_s = p_s_init * exp(-phis / (R_d * T_init)).
    perturbation_amplitude : float
        Amplitude of temperature perturbation [K] at the lowest level.
        Set to 0.0 to disable.
    seed : int
        Random seed for temperature perturbation.
    tracers : dict or None
        Optional initial tracer dict ``{name: Field | jax.Array}`` of
        grid-space mixing ratios with shape ``(n_lat, n_lon, nlev)``.
        Default ``None`` matches the dry pre-tracer pipeline.  Tracer
        values may be ``Field``-wrapped or raw JAX arrays — the dycore
        RHS duck-types both.
    """
    nlev = sigma_coord.n_levels
    n_sh = grid.n_sh

    dims_3d = ("spectral", "level")
    dims_2d = ("spectral",)

    # Zero winds -> zero vorticity and divergence
    vor_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)
    div_hat = jnp.zeros((n_sh, nlev), dtype=jnp.complex128)

    # Uniform temperature with perturbation at lowest level
    T_grid = jnp.full((grid.n_lat, grid.n_lon), T_init, dtype=jnp.float64)
    if perturbation_amplitude != 0.0:
        key = jax.random.PRNGKey(seed)
        noise = jax.random.normal(key, (grid.n_lat, grid.n_lon),
                                  dtype=jnp.float64)
        T_grid_pert = T_grid + perturbation_amplitude * noise
    else:
        T_grid_pert = T_grid

    # Spectral transform: perturbed field for lowest level, uniform elsewhere
    T_hat_uniform = sh_analysis(grid, T_grid)  # (n_sh,)
    T_hat = jnp.broadcast_to(T_hat_uniform[:, None], (n_sh, nlev)).copy()
    if perturbation_amplitude != 0.0:
        T_hat_pert = sh_analysis(grid, T_grid_pert)  # (n_sh,)
        T_hat = T_hat.at[:, -1].set(T_hat_pert)

    # Surface pressure (hydrostatic adjustment for topography)
    if phis is not None:
        phis_grid = jnp.asarray(phis, dtype=jnp.float64)
        p_s_grid = p_s_init * jnp.exp(-phis_grid / (constants.R_d * T_init))
        lnps_grid = jnp.log(p_s_grid)
    else:
        lnps_grid = jnp.full(
            (grid.n_lat, grid.n_lon), jnp.log(p_s_init), dtype=jnp.float64,
        )
        phis_grid = jnp.zeros((grid.n_lat, grid.n_lon), dtype=jnp.float64)

    lnps_hat = sh_analysis(grid, lnps_grid)

    # Topography in spectral space
    phis_hat_data = sh_analysis(grid, phis_grid)

    return SpectralHydrostaticState(
        vor_hat=Field(data=vor_hat, name="vor_hat", dims=dims_3d, units="1/s"),
        div_hat=Field(data=div_hat, name="div_hat", dims=dims_3d, units="1/s"),
        T_hat=Field(data=T_hat, name="T_hat", dims=dims_3d, units="K"),
        lnps_hat=Field(data=lnps_hat, name="lnps_hat", dims=dims_2d, units=""),
        phis_hat=Field(data=phis_hat_data, name="phis_hat", dims=dims_2d, units="m^2/s^2"),
        tracers=tracers,
    )


# =============================================================================
# Diagnostic utilities
# =============================================================================

def spectral_pe_to_grid(
    state: SpectralHydrostaticState,
    grid: GaussianGrid,
    sigma_coord: SigmaCoordinate,
) -> dict[str, jax.Array]:
    """Convert spectral PE state to grid-point fields for diagnostics.

    Returns
    -------
    dict with keys: 'u', 'v', 'T', 'vor', 'div', 'lnps', 'p_s', 'phis'
    """
    # Merge the (vor, div, T) 3D batch with the (lnps, phis) 2D pair
    # via ``jnp.concatenate`` — same exploit as Loop 182 in the tendency
    # block.  Trailing axis = ``nlev*3 + 2``.  5 SH syntheses → 1.
    n_sh_d, nlev_d = state.vor_hat.data.shape
    _vdT_stack = jnp.stack(
        [state.vor_hat.data, state.div_hat.data, state.T_hat.data],
        axis=-1,
    )  # (n_sh, nlev, 3)
    _vdT_flat = _vdT_stack.reshape(n_sh_d, nlev_d * 3)
    _all_diag_flat = jnp.concatenate(
        [
            _vdT_flat,
            state.lnps_hat.data[:, jnp.newaxis],
            state.phis_hat.data[:, jnp.newaxis],
        ],
        axis=-1,
    )  # (n_sh, nlev*3 + 2)
    _all_grid_diag = sh_synthesis_3d(grid, _all_diag_flat)
    _vdT_grid = _all_grid_diag[..., : nlev_d * 3].reshape(
        grid.n_lat, grid.n_lon, nlev_d, 3,
    )
    vor = _vdT_grid[..., 0]
    div = _vdT_grid[..., 1]
    T = _vdT_grid[..., 2]
    lnps = jnp.clip(_all_grid_diag[..., nlev_d * 3], _LNPS_MIN, _LNPS_MAX)
    phis = _all_grid_diag[..., nlev_d * 3 + 1]

    u_cos, v_cos = uv_from_vordiv_3d(
        grid, state.vor_hat.data, state.div_hat.data,
    )
    cos_lat_3d = jnp.clip(grid.cos_lat[:, None, None], _COS_LAT_MIN, None)
    u = u_cos / cos_lat_3d
    v = v_cos / cos_lat_3d

    return {
        'u': u, 'v': v, 'T': T,
        'vor': vor, 'div': div,
        'lnps': lnps, 'p_s': jnp.exp(lnps),
        'phis': phis,
    }
