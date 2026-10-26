"""Shared helpers for physics parameterization integration bridges.

Functions here are used by multiple physics packages (GWD, microphysics,
turbulence) to convert between prognostic model variables and the
column-physics inputs each scheme expects.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.core.operators_3d import fv_flux_divergence_3d
from legoesm.core.operators_fv_latlon_3d import fv_flux_divergence_latlon_3d
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.gaussian import sh_synthesis_3d, vordiv_from_uv_3d

from legoesm import constants

# ---------------------------------------------------------------------------
# AD-safe arithmetic helpers
# ---------------------------------------------------------------------------

def safe_divide(
    numerator: jnp.ndarray,
    denominator: jnp.ndarray,
    eps: float,
    fill: float = 0.0,
) -> jnp.ndarray:
    """Return ``numerator / denominator`` with AD-safe behaviour near zero.

    Equivalent in the forward path to::

        jnp.where(jnp.abs(denominator) > eps, numerator / denominator, fill)

    but written so the reverse-mode VJP never differentiates ``1/x`` or
    ``-a/x**2`` at tiny ``x``.  Use this anywhere both ``numerator`` and
    ``denominator`` can legitimately approach zero together (column
    mass-conservation rescalings, autoconversion/aggregation rates that
    vanish with their cloud field, geometric singularities at the poles
    or at the grid origin, etc.).

    The ``where``-before-divide ordering is required for AD safety; the
    common idiom ``numerator / jnp.clip(denominator, eps, None)`` is
    forward-equivalent for the unmasked branch but still differentiates
    the divide at the floor and emits ``-a / eps**2`` cotangents that
    overflow to ``inf``/``NaN`` — the very bug this helper exists to
    avoid.

    Forward outputs are bitwise identical to ``numerator / denominator``
    for any sample where ``|denominator| > eps``.  The mask uses
    ``|denominator|`` rather than ``denominator`` so that both polarities
    of the singularity are caught (e.g. wind shears near zero, signed
    metric quantities approaching zero from either side).

    Parameters
    ----------
    numerator : jnp.ndarray
        Dividend.
    denominator : jnp.ndarray
        Divisor.  May contain zeros or values with ``|x| ≤ eps``.
    eps : float
        Magnitude floor below which the divide is masked out.  Choose
        ``eps`` one or two decades above any prior ``clip`` floor so the
        forward stays bit-identical wherever the old code was sound.
    fill : float, optional
        Value substituted into the output where ``|denominator| ≤ eps``.
        Defaults to ``0.0`` — appropriate when ``numerator`` also
        vanishes with ``denominator`` (the common physical limit) or
        when downstream logic clips the result back to a valid range.
    """
    denom_abs = jnp.abs(denominator)
    mask = denom_abs > eps
    safe_denom = jnp.where(mask, denominator, jnp.asarray(1.0, denominator.dtype))
    return jnp.where(mask, numerator / safe_denom, jnp.asarray(fill, numerator.dtype))


# ---------------------------------------------------------------------------
# Height / thickness from hydrostatic balance
# ---------------------------------------------------------------------------

def compute_heights_from_sigma(T, p_half, q_v=None):
    """Approximate full- and half-level heights from hydrostatic balance.

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature at full levels [K].
    p_half : array (ncol, nlev+1)
        Pressure at half levels [Pa], TOA-first.
    q_v : array (ncol, nlev) or None, optional
        Water-vapour specific humidity [kg/kg].  When provided, the
        hypsometric integral uses the **virtual temperature**
        ``T_v = T · (1 + (1/ε − 1) · q_v)`` — ~1 % thicker layers in
        tropical moist columns.  Default ``None`` keeps the legacy
        dry-T behaviour for callers that don't have q_v handy.
        Audit cycle iter-35 finding F2.

    Returns
    -------
    z_full : array (ncol, nlev)
        Height at full levels [m].
    z_half : array (ncol, nlev+1)
        Height at half levels [m] (surface = 0).
    """
    dz = compute_layer_dz(T, p_half, q_v)

    # Integrate from surface upward.  Use ``jnp.pad`` to append the
    # surface (z=0) boundary instead of allocating a fresh
    # ``jnp.zeros((ncol, 1))`` and concatenating — single Pad HLO op
    # vs alloc + concat (this helper is invoked by GWD / microphysics
    # / turbulence integrations every physics step).
    dz_rev = dz[:, ::-1]
    z_half_cumsum = jnp.cumsum(dz_rev, axis=1)
    z_half_inner = z_half_cumsum[:, ::-1]
    z_half = jnp.pad(z_half_inner, ((0, 0), (0, 1)))
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    return z_full, z_half


def compute_layer_dz(T, p_half, q_v=None):
    """Approximate layer thicknesses from hydrostatic balance.

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature at full levels [K].
    p_half : array (ncol, nlev+1)
        Pressure at half levels [Pa], TOA-first.
    q_v : array (ncol, nlev) or None, optional
        Water-vapour specific humidity [kg/kg].  When provided, uses
        virtual temperature in the hypsometric integral — ~1 %
        thicker layers in moist columns.  Audit iter-35 F2.

    Returns
    -------
    dz : array (ncol, nlev)
        Layer thickness [m].
    """
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    T_eff = T if q_v is None else virtual_temperature(T, q_v)
    return jnp.abs(
        constants.R_d * T_eff * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    )


def half_to_full(x_half):
    """Interior-interface field ``(ncol, nlev-1)`` to full levels ``(ncol, nlev)``.

    Interior full levels take the mean of the two adjacent interfaces; the top
    and bottom full levels copy the nearest interface (one-sided).
    """
    mid = 0.5 * (x_half[:, :-1] + x_half[:, 1:])
    return jnp.concatenate([x_half[:, :1], mid, x_half[:, -1:]], axis=1)


def brunt_vaisala_n_full(T, p_full, z_full):
    """Brunt-Väisälä frequency ``N`` on full levels for column GWD schemes.

    Computes ``θ = T·(p_ref/p)^κ``, the half-level ``N`` from the potential-
    temperature gradient (``N² = (g/θ̄)·dθ/dz``, floored at 1e-8 s⁻² for
    AD/√ safety), then maps the ``nlev-1`` half-level values back to ``nlev``
    full levels by interior averaging with edge replication.  Shared by the
    Hines / Lindzen / McFarlane / prognostic-spectral GWD backends, which all
    consume only ``N`` on full levels (the layer-thickness floor of 1 m keeps
    ``dθ/dz`` finite on degenerate columns).

    Parameters
    ----------
    T : array (ncol, nlev)
        Temperature at full levels [K].
    p_full : array (ncol, nlev)
        Pressure at full levels [Pa].
    z_full : array (ncol, nlev)
        Geometric height at full levels [m].

    Returns
    -------
    N_full : array (ncol, nlev)
        Brunt-Väisälä frequency [s⁻¹] at full levels.
    """
    theta = T * (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    dz_full = jnp.abs(z_full[:, :-1] - z_full[:, 1:])
    dz_full = jnp.clip(dz_full, 1.0, None)
    dtheta_dz = (theta[:, :-1] - theta[:, 1:]) / dz_full
    theta_bar = 0.5 * (theta[:, :-1] + theta[:, 1:])
    N2_half = (constants.g / jnp.clip(theta_bar, 1.0, None)) * dtheta_dz
    N2_half = jnp.clip(N2_half, 1e-8, None)
    N_half = jnp.sqrt(N2_half)
    return jnp.concatenate([
        N_half[:, :1],
        0.5 * (N_half[:, :-1] + N_half[:, 1:]),
        N_half[:, -1:],
    ], axis=1)


# ---------------------------------------------------------------------------
# Density from ideal-gas law
# ---------------------------------------------------------------------------

def compute_rho(T, p_full, q_v=None):
    """Compute air density from the ideal gas law.

    Without ``q_v`` returns the dry-air density ``rho = p / (R_d · T)``.
    With ``q_v`` returns the moist density ``rho = p / (R_d · T_v)``
    where ``T_v = T · (1 + (1/ε − 1) · q_v)``.

    The moist form is consistent with ``diagnose_grid_w_from_omega``
    (which has always used T_v when ``q_v`` is provided) and
    avoids the ~1 % drift between dry-rho and moist-rho in tropical
    columns.  Default ``q_v=None`` preserves the legacy dry-T
    behaviour for callers that don't have q_v handy.  Audit cycle
    iter-35 finding F6.

    Parameters
    ----------
    T : array
        Temperature [K].
    p_full : array
        Pressure at full levels [Pa].
    q_v : array or None, optional
        Water-vapour specific humidity [kg/kg].  When provided, uses
        virtual temperature in the ideal-gas law.

    Returns
    -------
    array : Density [kg/m^3].
    """
    T_eff = T if q_v is None else virtual_temperature(T, q_v)
    return p_full / (constants.R_d * jnp.clip(T_eff, 1.0, None))


def virtual_temperature(T, q_v):
    """Compute virtual temperature ``T_v = T · (1 + (R_v/R_d - 1) · q_v)``.

    With ``constants.epsilon = R_d / R_v ≈ 0.622``, the virtual-T
    coefficient is ``1/ε - 1 ≈ 0.6078`` — the codebase historically
    used a rounded ``0.61`` literal in turbulence/PBL helpers, drifting
    by ~0.16 % from the canonical value.  This helper produces a
    consistent, derivation-correct ``T_v`` (audit cycle 1: turbulence
    static analysis B1).

    Parameters
    ----------
    T : array
        Air temperature [K].
    q_v : array
        Water-vapor mixing ratio [kg/kg].  The codebase uses ``q_v``
        and the saturation mixing ratio interchangeably (~1 % drift
        for typical tropospheric humidities; see ``thermo.py``).

    Returns
    -------
    array
        Virtual temperature [K], same shape as ``T``.
    """
    coeff = 1.0 / constants.epsilon - 1.0           # ≈ 0.6078
    return T * (1.0 + coeff * q_v)


def ke_dissipation_heating(u, v, du_dt, dv_dt, dt):
    """Heating [K/s] that returns as heat the kinetic energy a momentum
    tendency removes (frictional / dissipative heating), per level.

        dT/dt = -(du*(u + dt/2*du) + dv*(v + dt/2*dv)) / c_pd

    Discretely exact for the host update ``u += dt*du, v += dt*dv,
    T += dt*dT``: ``c_pd*dt*dT = -[(u+dt*du)^2 - u^2 + (v+dt*dv)^2 - v^2]/2``
    per unit mass, so the momentum tendency leaves the column
    ``sum((c_pd*T + KE) dp/g)`` unchanged level by level.  Sign: positive
    (warming) where the tendency removes kinetic energy.  The same closure as
    the orographic GWD port (``e3sm_cam.py``, ``use_discrete_ke_heating``) and
    CAM's ``diffusion_solver`` ``dtk``/``clubb_do_energyfix`` intent.
    """
    return -(du_dt * (u + 0.5 * dt * du_dt)
             + dv_dt * (v + 0.5 * dt * dv_dt)) / constants.c_pd


def column_energy_fixer_heating(residual, layer_mass, weight):
    """Heating [K/s] that removes a column energy residual (CAM ``se_dis``).

        dT/dt_k = -residual * w_k / (c_pd * sum_j layer_mass_j * w_j)

    ``residual`` [W/m2, + = energy created], ``layer_mass`` [kg/m2] (dp/g),
    ``weight`` >= 0 per level (where the fix may act; uniform per unit mass
    where it is 1).  Exact: ``sum(c_pd * dT/dt * layer_mass) == -residual``.
    CAM6 ``clubb_intr`` ``clubb_do_energyfix``: the imbalance spread evenly
    per unit mass over the depth where CLUBB is active.
    """
    return -(residual[..., None] * weight
             / (constants.c_pd * jnp.sum(layer_mass * weight, axis=-1,
                                         keepdims=True)))


def broadcast_column_param(value, like):
    """Broadcast a config coefficient over a column field's vertical dimension.

    Enables a scheme coefficient to be EITHER a scalar (production default) OR a
    per-column ``(ncol,)`` field (the LES-informed correction, see
    ``docs/COMPARE_REANALYSIS.md``) **without** changing the scheme body's
    arithmetic:

    * a scalar / 0-d ``value`` is returned unchanged (it already broadcasts
      against ``like`` — the production path stays byte-identical);
    * a 1-D ``(ncol,)`` ``value`` is reshaped to ``(ncol, 1, …)`` so it
      broadcasts over the trailing (vertical / other) axes of ``like`` (shape
      ``(ncol, nlev)`` etc.).

    Wrap a coefficient use as ``broadcast_column_param(cfg.coeff, X) * X`` in the
    scheme body; ``like`` is any per-column array whose leading axis is the
    column dimension.  Raises if a 1-D ``value`` length does not match
    ``like.shape[0]``.
    """
    value = jnp.asarray(value)
    like = jnp.asarray(like)
    if value.ndim == 0:
        return value
    if value.ndim == 1:
        if value.shape[0] != like.shape[0]:
            raise ValueError(
                f"broadcast_column_param: per-column value length "
                f"{value.shape[0]} != column count {like.shape[0]}."
            )
        return value.reshape((value.shape[0],) + (1,) * (like.ndim - 1))
    raise ValueError(
        f"broadcast_column_param: value must be scalar or 1-D (ncol,); got "
        f"shape {value.shape}."
    )


def exner_function(p):
    """Exner function ``Π = (p / p_ref)^κ`` (potential-temperature scaling).

    Canonical home for the Poisson-exponent power ``(p/p₀)^κ`` (``θ = T/Π``,
    ``T = θ·Π``) — use this instead of an inline ``(p/constants.p_ref) **
    constants.kappa`` so the formula lives in one place (CLAUDE.md "shared
    utilities — never re-derive"; the ``exner_potential_temperature`` ratchet).

    Parameters
    ----------
    p : array
        Pressure [Pa].

    Returns
    -------
    array
        Exner function [-], same shape as ``p``.
    """
    poisson_exponent = constants.kappa
    # AD-safe pressure floor: Π = (p/p₀)^κ with κ≈0.286<1 has
    # dΠ/dp ∝ p^(κ−1) → ∞ as p→0, so the REVERSE-mode gradient blows up
    # (NaN/Inf) at a p=0 top half-level — even though the forward value
    # (0) is finite.  This silently NaN'd the gradient of any
    # differentiable rollout that back-propagates through θ/Π (AIMIP
    # carry-based training: grad wrt p_s was non-finite, job 8533900).
    # Clip to 1 Pa (well below any real model level, so the forward is
    # bit-identical everywhere it matters); the clip's zero gradient
    # below the 1 Pa floor — times the finite (1 Pa)^(κ−1) — yields a
    # finite (0) gradient there instead of ∞. coeff-ok: 1 Pa AD floor.
    p_safe = jnp.clip(p, 1.0, None)
    return (p_safe / constants.p_ref) ** poisson_exponent


def exner_to_pressure(exner):
    """Inverse Exner: pressure ``p = p_ref · Π^(1/κ)`` from the Exner function ``Π``.

    The inverse of :func:`exner_function` (``Π = (p/p_ref)^κ``) — the canonical home
    for the inverse-Poisson recovery ``p = p_ref·Π^{1/κ}`` (e.g. converting a stored
    Exner reference back to a reference pressure), so the formula lives in one place
    (CLAUDE.md "shared utilities — never re-derive"; the
    ``exner_potential_temperature`` ratchet flags the ``Π^{1/κ}`` direction too).

    Parameters
    ----------
    exner : array
        Exner function ``Π`` [-].

    Returns
    -------
    array
        Pressure [Pa], same shape as ``exner``.
    """
    inverse_poisson_exponent = 1.0 / constants.kappa
    return constants.p_ref * exner ** inverse_poisson_exponent


def surface_buoyancy_flux(wtheta_s, wq_s, theta):
    """Kinematic surface buoyancy (virtual potential temperature) flux [K m/s].

    ``(w'theta_v')_s = (w'theta')_s + (R_v/R_d - 1) theta (w'q')_s`` with
    ``R_v/R_d - 1 = 1/epsilon - 1`` (E3SM ``zvir``); kinematic fluxes
    positive upward, ``theta`` the lowest-level potential temperature [K].
    """
    return wtheta_s + (1.0 / constants.epsilon - 1.0) * theta * wq_s


def buoyancy_coefficient(theta, *, gravity=None):
    """Buoyancy coefficient ``g / θ`` [m s⁻² K⁻¹] for buoyancy / N² terms.

    Canonical home for the gravity-over-potential-temperature factor that
    appears in buoyancy production (``(g/θ_v)·w'θ_v'``) and the Brunt-Väisälä
    frequency (``(g/θ)·∂θ/∂z``) — callers multiply the returned coefficient by
    their flux or gradient. Factors the ``g/θ`` re-derivation into one place
    (CLAUDE.md "shared utilities — never re-derive"; the
    ``buoyancy_term_g_over_theta`` ratchet). ``theta`` is the (virtual)
    potential temperature [K]; clip/floor it at the call site if needed.

    Parameters
    ----------
    theta : array
        (Virtual) potential temperature [K].
    gravity : float, optional
        Gravity to use [m/s²]; defaults to ``constants.g``. Only for callers
        that expose gravity as a knob (the LES intercomparison diagnostics do,
        for reproducing a reference run's value) — physics callers leave it.

    Returns
    -------
    array
        ``g / θ``, same shape as ``theta``.
    """
    return (constants.g if gravity is None else gravity) / theta


def brunt_vaisala_n_squared_from_gradient(theta, dtheta_dz):
    """Brunt-Väisälä frequency squared ``N² = g·∂θ/∂z/θ`` from an ALREADY-COMPUTED
    potential-temperature gradient + ``θ`` at the SAME levels.

    The canonical home for the N²-from-a-gradient form (the
    ``buoyancy_term_g_over_theta`` ratchet) — distinct from
    :func:`brunt_vaisala_n_full`, which itself builds ``θ`` + the gradient + the
    edge mapping from ``T``/``p``/``z`` for the GWD schemes.  Floor/clip ``θ`` at
    the CALL site as needed.  The arithmetic order ``g·∂θ/∂z/θ`` is PRESERVED (not
    ``(g/θ)·∂θ/∂z``) so a caller replacing an inline form stays BIT-IDENTICAL.

    Parameters
    ----------
    theta : array
        (Virtual) potential temperature [K] co-located with ``dtheta_dz``.
    dtheta_dz : array
        Potential-temperature vertical gradient ``∂θ/∂z`` [K/m], same shape.

    Returns
    -------
    array
        ``N²`` [s⁻²], same shape.
    """
    return constants.g * dtheta_dz / theta


# --- Deardorff (1980) stable-layer SGS length limit -------------------------
# SAM ``SGS_TKE/tke_full.f90`` dosmagor calibration. Stable mixing is suppressed
# by SHRINKING the master length, not by driving a stability factor to zero.
_DEARDORFF_STABLE_COEF = 0.76   # l = 0.76·√e/N (Deardorff 1980; PALM, ARPS, ERF)
_DEARDORFF_CK = 0.1             # SAM Ck
_DEARDORFF_SMIX_FLOOR = 0.1     # SAM floor on smix, as a fraction of the length
_DEARDORFF_CEE_A = 0.19         # Cee = Ce/0.7·(0.19 + 0.51·smix/l)
_DEARDORFF_CEE_B = 0.51
_DEARDORFF_CEE_DIV = 0.7
# Floor on the buoyancy-shutoff sqrt ARGUMENT [s⁻²]. This is the differentiable
# model's own requirement, NOT SAM's: d√x/dx = 0.5/√x, so an unfloored argument
# hands the optimizer 0.5/√1e-36 = 5e17 — finite, and useless. SAM never
# differentiates its closure, so it can leave the argument bare. The floor caps
# the slope at 0.5/√_STRAIN_FLOOR ≈ 1.6e5.
#
# It sits ONE order BELOW the caller's own strain floor (``S² = |∂V/∂z|² +
# 1e-10``): at zero shear and neutral N², ``S² − N²/Pr_t = 1e-10`` already
# exceeds this floor, so the floor does NOT bite and the ``deardorff`` option
# reduces EXACTLY to ``(c_s·l)²·√(S² − N²/Pr_t)`` there, byte-identical to the
# default. A floor ABOVE 1e-10 (an earlier 1e-9) silently changed the neutral
# strain from √1e-10 to √1e-9 — a √10 error the reviewer caught.
_STRAIN_FLOOR = 1.0e-11


def deardorff_stable_eddy_viscosity(strain_sq, n2, length, c_s, Pr_t):
    """Smagorinsky eddy viscosity with the Deardorff (1980) stable-length limit.

    ``K_m = √(Ck³/Cee)·smix²·√(S² − N²/Pr_t)`` with the master length shrunk
    under stable stratification to ``smix = clip(√(0.76·tk/(Ck·N)), 0.1·l, l)``
    (SAM ``dosmagor``).  Where ``N² ≤ 0`` (neutral/unstable) ``smix = l`` and
    ``√(Ck³/Cee) = c_s²`` EXACTLY, so this reduces identically to the plain
    ``(c_s·l)²·√(S² − N²/Pr_t)`` Smagorinsky-Lilly form.

    Sign convention: ``N² > 0`` is STABLE; the buoyancy term is SUBTRACTED from
    the shear so stratification suppresses mixing and ``N² < 0`` enhances it.

    Prandtl convention: this repo divides (``S² − N²/Pr_t``, equivalently
    Lilly's ``1 − Ri/Pr_t``).  SAM MULTIPLIES (``S² − Pr·N²``).  The two agree
    only at ``Pr_t = 1`` and invert across this scheme's tunable range
    ``Pr_t ∈ (0.33, 3)``, so the repo convention is used here and SAM's is NOT
    transplanted.

    Parameters
    ----------
    strain_sq : array
        RAW deformation ``S²`` [s⁻²] with NO buoyancy correction applied — the
        ``N²/Pr_t`` subtraction happens here.  Passing an already-corrected
        strain applies the correction twice and silently guts ``K_m``.
    n2 : array
        Brunt-Väisälä ``N²`` [s⁻²]; may be ≤ 0.
    length : array
        Master mixing length [m] (Blackadar in a column, grid ``Δ`` in an LES).
        MUST be strictly positive: ``smix/length`` is an unguarded divide, so a
        zero length gives 0/0 = NaN on both branches.  The single-column caller
        satisfies this because :func:`mixing_length` floors ``z`` at 1 m
        (``length ≥ 0.4 m``); a future LES caller passing a raw grid ``Δ`` must
        ensure the same.
    c_s, Pr_t : float
        Smagorinsky constant and turbulent Prandtl number.

    Returns
    -------
    array
        ``K_m`` [m²/s], same shape.
    """
    # Buoyancy-corrected strain, floored so the reverse-mode slope is BOUNDED
    # rather than merely finite.  Kept as one sqrt (never factored into
    # S·√(1−Ri/Pr_t)) so there is a single guarded argument.
    strain = jnp.sqrt(jnp.maximum(strain_sq - n2 / Pr_t, _STRAIN_FLOOR))

    # Deardorff stable length.  smix ∝ N^(-1/2) diverges as N² → 0⁺, but the
    # clip saturates at ``length`` there: the interior of the clip requires
    # 0.01 < 0.76·tk/(Ck·N·l²) < 1, which bounds N away from zero, so the
    # unbounded branch is never the selected one.  ``_STRAIN_FLOOR`` keeps
    # ``tk`` > 0, so the inner sqrt argument is strictly positive.
    tk = c_s ** 2 * length ** 2 * strain
    n_stable = jnp.sqrt(jnp.maximum(n2, _STRAIN_FLOOR))
    smix_raw = jnp.sqrt(_DEARDORFF_STABLE_COEF * tk
                        / (_DEARDORFF_CK * n_stable))
    smix = jnp.where(
        n2 > 0.0,
        jnp.clip(smix_raw, _DEARDORFF_SMIX_FLOOR * length, length),
        length,
    )

    # Ce = Ck³/c_s⁴ and Cee = Ce/0.7·(0.19 + 0.51·smix/l).  At smix = l this is
    # Cee = Ce, hence √(Ck³/Cee) = c_s² and the whole expression collapses to
    # the unmodified Smagorinsky form — the neutral/unstable identity above.
    ratio = smix / length
    cee = (_DEARDORFF_CK ** 3 / c_s ** 4) / _DEARDORFF_CEE_DIV * (
        _DEARDORFF_CEE_A + _DEARDORFF_CEE_B * ratio)
    return jnp.sqrt(_DEARDORFF_CK ** 3 / cee) * smix ** 2 * strain


def lilly_buoyancy_factor(Ri, Pr_t):
    """Lilly (1962) buoyancy stability factor ``√(max(0, 1 − Ri/Pr_t))`` that
    multiplies a strain-based Smagorinsky eddy viscosity to suppress mixing in
    stably stratified layers.

    ``f = 1`` at neutral (``Ri = 0``), ``> 1`` when unstable (``Ri < 0``,
    convective enhancement), and shuts mixing OFF at ``Ri ≥ Pr_t`` (Lilly's
    equilibrium result, with ``Pr_t`` playing the role of the critical
    Richardson number ``Ri_c``). Canonical home for the factor shared by the
    single-column Smagorinsky–Lilly PBL closure and the 3-D plane-LES SGS
    (CLAUDE.md "shared utilities — never re-derive").

    AD-safety: the ``max(0, ·)`` cutoff is written as a double-``jnp.where`` so
    the forward is exact AND the reverse-mode cotangent is finite at the
    ``Ri = Pr_t`` kink (a bare ``√(max(·, 0))`` leaks a ``0·∞`` NaN through the
    √' → ∞ at zero). The forward is continuous but NON-C¹ at the cutoff.

    Parameters
    ----------
    Ri : array
        Gradient Richardson number ``N²/|S|²`` (floor ``|S|²`` at the call site
        so ``Ri`` stays finite).
    Pr_t : float
        Turbulent Prandtl number, also the stable cutoff ``Ri_c``.

    Returns
    -------
    array
        Buoyancy factor in ``[0, ∞)``, same shape as ``Ri``.
    """
    buoy_arg = 1.0 - Ri / Pr_t
    buoy_safe = jnp.where(buoy_arg > 0.0, buoy_arg, 1.0)
    return jnp.where(buoy_arg > 0.0, jnp.sqrt(buoy_safe), 0.0)


def mixing_length(z, l_mix_max, z_floor=1.0):
    """Asymptotic master mixing length ``l = κz / (1 + κz/l_∞)`` (Blackadar 1962).

    The neutral surface-layer limit ``l → κz`` near the ground blends
    smoothly into the free-atmosphere asymptote ``l → l_mix_max`` aloft.
    Shared by the Louis, TKE, CLUBB-lite, Holtslag–Boville, EDMF and
    Smagorinsky–Lilly closures (audit: turbulence ``l_mix`` dedup — five
    schemes inlined this identical expression).

    Parameters
    ----------
    z : array
        Height above the surface [m] (half-level interface or full level).
    l_mix_max : float
        Free-atmosphere asymptotic mixing length ``l_∞`` [m].
    z_floor : float
        Lower clip on ``|z|`` [m] so ``l`` stays finite and its gradient
        well-defined at the surface (default 1.0 m).

    Returns
    -------
    array
        Mixing length [m], same shape as ``z``.
    """
    kappa = constants.kappa_vk
    z_abs = jnp.clip(jnp.abs(z), z_floor, None)
    return kappa * z_abs / (1.0 + kappa * z_abs / l_mix_max)


def louis_stability_functions(
    Ri, l_mix, dz, b, c, d, blend_sharpness, b_heat=None, z_full_pair=None,
):
    """Louis Richardson-number stability functions, smoothly blended.

    Canonical home for the Louis stability functions shared by the standalone
    Louis PBL scheme (``louis.py``) and the free-atmosphere local-Ri branch of
    YSU (``ysu.py``) (CLAUDE.md "no duplicate numerics"; colocated with the
    shared Blackadar :func:`mixing_length` the same closures use).

    Branch forms (momentum ``f_m`` with ``b``; heat ``f_h`` with ``b_heat``):

    * Stable   (Ri>0): ``f_m = 1/(1 + 2·b·Ri/sqrt(1+d·Ri))``,
      ``f_h = 1/(1 + 2·b_heat·Ri·sqrt(1+d·Ri))``.
    * Unstable (Ri<0): ``f = 1 - 2·b·Ri·u`` (heat: ``b_heat``) with
      ``u = 1/(1 + 3·b·c·l²·sqrt(|Ri|·G))``, where, when the full-level
      heights ``z_full_pair = (z_k, z_k+1)`` bracketing the interface are
      passed,
      ``G = (((z_k/z_k+1)^(1/3) - 1)/(z_k - z_k+1))³ / z_k+1``.

    blended with ``sigmoid(blend_sharpness · Ri)``; at the hard-switch limit
    (``blend_sharpness`` -> inf) this is the oracles' ``IF (Ri > 0)`` /
    ``IF (Ri < 0)`` branch selection (at Ri = 0 both branches equal 1).

    Oracle provenance (the running codes this reproduces; pinned in
    ``tests/atmosphere/hydrostatic/unit/test_louis_oracle.py``):
      * Stable branch, momentum AND heat: OpenIFS
        (github.com/ecmwf-ifs/openifs @ 8f6f722f732b92fed2c39af763d8fcd1ab15f573)
        ``arpifs/phys_ec/vdfexcu.F90`` lines 393-395::

            ZSCF    = SQRT(1.0_JPRB+ZCD*ZRI(JL))
            ZIFLTGM = 1.0_JPRB / (1.0_JPRB + 2.0_JPRB *ZCB * ZRI(JL)/ZSCF)
            ZIFLTGH = 1.0_JPRB / (1.0_JPRB + 2.0_JPRB *ZCB * ZRI(JL)*ZSCF)

        i.e. the heat function MULTIPLIES by the square root (IFS runs
        ``b = ZCB = 5``, ``d = ZCD = 1``, ``b_heat = b``).
      * Unstable branch (with the heights passed): ICON icon-2026.04-public
        (8597da45ef4b86323f3fb844caedc4ae5e1ffc01)
        ``src/atm_phy_schemes/mo_turb_vdiff_diag.f90`` lines 518-530 (the
        ECHAM/ICON "Louis 79" block): ``kh *= 1 - 3·cb·Ri·zucf``,
        ``km *= 1 - 2·cb·Ri·zucf``, ``zucf`` as ``u`` above with
        ``cons5 = 3·cb·cc`` (ICON runs ``cb = cc = 5``, i.e. ``b = c = 5``,
        ``b_heat = 1.5·b``).
      * The only addition to the oracle algebra is the pre-existing ``1e-10``
        inside ``sqrt(|Ri| + 1e-10)`` (keeps the reverse-mode derivative finite
        at Ri = 0, where the oracle never evaluates this branch); it moves
        ``f`` by a relative ~1e-11 on the oracle fixture.

    NOT-YET-FAITHFUL legacy path (``z_full_pair`` omitted;
    used ONLY by YSU's free-atmosphere local-K branch, whose own oracle is a
    separate scheme): the unstable geometry factor is ``1/(dz² + 1e-10)`` in
    place of ``sqrt(G)``.  This form is not the running ECHAM/ICON code (its
    unstable enhancement f_m was median 3.8x, up to 20x weaker on the tier-2
    audit columns).  YSU reads
    only ``f_m``, so the stable heat form does not reach it.

    Parameters
    ----------
    Ri : array
        Gradient Richardson number at interfaces.
    l_mix : array
        Mixing length at the same interfaces [m].
    dz : array
        Interface spacing [m] (same shape as ``Ri``); used only by the legacy
        unstable geometry.
    b, c, d : float
        Louis coefficients: ``b`` enters both branches, ``c`` the unstable
        denominator, ``d`` the stable square root.
    blend_sharpness : float
        Sigmoid sharpness of the stable/unstable blend [1/Ri].
    b_heat : float or None
        Heat-function coefficient (default ``b``).
    z_full_pair : (array, array) or None
        ``(z_k, z_k+1)``: full-level heights above the surface [m] of the
        layers above and below each interface, both > 0 (oracle geometry);
        ``None`` selects the legacy geometry.

    Returns
    -------
    (f_m, f_h) : tuple of arrays
        Momentum and heat stability functions (dimensionless, > 0).
    """
    if b_heat is None:
        b_heat = b

    # Unstable branch — denominator shared between momentum and heat.
    Ri_neg = jnp.minimum(Ri, 0.0)
    sqrt_abs_ri = jnp.sqrt(jnp.abs(Ri_neg) + 1e-10)
    if z_full_pair is None:
        geom = 1.0 / (dz ** 2 + 1e-10)  # legacy (YSU) geometry, see docstring
    else:
        # ICON mo_turb_vdiff_diag.f90:523-525 (pghf(jk) = z_k, pghf(jk+1) = z_k+1)
        z_full_upper, z_full_lower = z_full_pair
        geom = jnp.sqrt(
            (((z_full_upper / z_full_lower) ** (1.0 / 3.0) - 1.0)
             / (z_full_upper - z_full_lower)) ** 3
            / z_full_lower
        )
    denom_unstable = 1.0 + 3.0 * b * c * l_mix ** 2 * sqrt_abs_ri * geom
    f_unstable_m = 1.0 - 2.0 * b * Ri_neg / denom_unstable
    f_unstable_h = 1.0 - 2.0 * b_heat * Ri_neg / denom_unstable

    # Stable branch — OpenIFS vdfexcu.F90:393-395 (heat MULTIPLIES by ZSCF).
    Ri_pos = jnp.maximum(Ri, 0.0)
    sqrt_stable = jnp.sqrt(1.0 + d * Ri_pos)
    f_stable_m = 1.0 / (1.0 + 2.0 * b * Ri_pos / sqrt_stable)
    f_stable_h = 1.0 / (1.0 + 2.0 * b_heat * Ri_pos * sqrt_stable)

    # Smooth blending: sigmoid transitions from unstable to stable.
    blend = jax.nn.sigmoid(blend_sharpness * Ri)
    f_m = (1.0 - blend) * f_unstable_m + blend * f_stable_m
    f_h = (1.0 - blend) * f_unstable_h + blend * f_stable_h
    return f_m, f_h


# ---------------------------------------------------------------------------
# Tracer-pytree helpers (used to keep state ↔ tendency pytrees aligned
# for the spectral PE dycore RHS and orchestrator)
# ---------------------------------------------------------------------------

def zero_like_tracers(tracers):
    """Build a tracer dict whose values are zeros with the same shape
    and container type as the input.

    Tracer dict values may be ``Field`` objects (with ``.data`` and
    ``.replace``) or raw JAX arrays.  This helper duck-types both:

    * ``Field`` value → ``v.replace(data=jnp.zeros_like(v.data))``
    * raw array      → ``jnp.zeros_like(v)``

    When ``tracers`` is ``None`` we return ``None`` — preserving the
    "no tracers" pytree shape.

    The dycore RHS for spectral PE (``spectral_pe_tendencies``) and
    the spectral PE orchestrator both use this to keep their tendency
    pytrees structurally identical to the input state, otherwise
    ``jax.tree.map(state, tendency)`` in the SSP-RK steps trips on a
    dict-vs-None mismatch.
    """
    if tracers is None:
        return None
    out = {}
    for k, v in tracers.items():
        if hasattr(v, "data") and hasattr(v, "replace"):
            out[k] = v.replace(data=jnp.zeros_like(v.data))
        else:
            out[k] = jnp.zeros_like(v)
    return out


# ---------------------------------------------------------------------------
# Moisture-convergence diagnostic for Tiedtke / Bechtold closures
# ---------------------------------------------------------------------------

def moisture_convergence_supported(grid) -> bool:
    """True when :func:`compute_moisture_convergence` has an operator for *grid*.

    Mirrors that function's dispatch exactly (single-column, cubed
    sphere, Gaussian/spectral, lat-lon).  Callers that can degrade
    gracefully (e.g. the unified driver pipeline passing ``None`` MC so
    Tiedtke/Bechtold engage their built-in proxies) use this instead of
    catching the TypeError.
    """
    return (
        getattr(grid, "grid_n_columns", None) == 1
        or isinstance(grid, CubedSphereGrid)
        or (hasattr(grid, "n_max") and hasattr(grid, "Pnm"))
        or (hasattr(grid, "dlat") and hasattr(grid, "dlon"))
    )


def compute_moisture_convergence(
    q_v_grid: jnp.ndarray,
    u_grid: jnp.ndarray,
    v_grid: jnp.ndarray,
    grid,
) -> jnp.ndarray:
    """Per-level horizontal moisture convergence ``MC = -∇·(q_v * u)``.

    Used by the Tiedtke / Bechtold convection schemes as the deep
    closure's mass-flux driver.  Implementation reuses the dycore's
    finite-volume tracer-flux divergence operator
    (:func:`legoesm.core.operators_3d.fv_flux_divergence_3d` for cubed
    sphere; :func:`legoesm.core.operators_fv_latlon_3d.fv_flux_divergence_latlon_3d`
    for lat-lon) and negates the result.

    Parameters
    ----------
    q_v_grid : jax.Array
        Water-vapor specific humidity in grid-shape:

        * cubed sphere — ``(face, n, n, nlev)``
        * lat-lon C-grid — ``(n_lat, n_lon, nlev)``
    u_grid, v_grid : jax.Array
        Horizontal wind components, same grid-shape as ``q_v_grid``.
    grid : CubedSphereGrid or LatLonGrid
        Discretized grid metadata used by the underlying divergence
        operator.

    Returns
    -------
    jax.Array, shape (ncol, nlev)
        Column-flattened moisture convergence [kg/kg/s].  Positive
        values indicate net moisture inflow to the column.

    Notes
    -----
    Uses the FV-flux-divergence operator with the slope limiter
    *disabled* — we want a smooth, fully-differentiable diagnostic, and
    the limiter introduces non-smooth ``where``-style branching that
    would break ``jax.grad`` through the convection trigger.  Tracer
    advection in the dycore proper still uses the limiter; this is a
    closure diagnostic, not an advected quantity.
    """
    # Single-column model: no horizontal flux ⇒ MC ≡ 0. Detected by
    # ``grid_n_columns == 1`` so the SCM grid stays decoupled from the
    # heavyweight grid classes. Returning zeros lets schemes that use
    # MC as a trigger (Tiedtke, Bechtold) run in single-column mode with
    # closure parameters that depend only on local profile state.
    if getattr(grid, "grid_n_columns", None) == 1:
        ncol_flat = q_v_grid.reshape(-1, q_v_grid.shape[-1]).shape[0]
        return jnp.zeros((ncol_flat, q_v_grid.shape[-1]), dtype=q_v_grid.dtype)

    if isinstance(grid, CubedSphereGrid):
        # q_v_grid shape (6, n, n, nlev).  ``fv_flux_divergence_3d``
        # already returns the tendency form ``-div(q·V)`` (see its
        # docstring "dq/dt = -div(q*v)"), which IS the moisture
        # convergence — no extra sign flip needed.  Audit cycle
        # iter-35 finding F1 (CRITICAL): the previous implementation
        # double-negated, returning ``+div(q·V)`` (moisture
        # DIVERGENCE) for cubed-sphere and lat-lon while the spectral
        # branch was correct.  This silently inverted convective
        # forcing on cubed-sphere/lat-lon runs.
        mc = fv_flux_divergence_3d(
            q_v_grid, u_grid, v_grid, grid, limiter=False,
        )
        # Reshape to (ncol, nlev)
        face, n, _, nlev = q_v_grid.shape
        return mc.reshape(face * n * n, nlev)

    # Gaussian grid (spectral PE).  Compute the divergence of
    # (q_v u, q_v v) via the transform method: synthesize the grid
    # product, take the spectral divergence using the existing
    # ``vordiv_from_uv_3d`` helper (which uses the pole-safe oc2/dmu
    # operators), then synthesize the divergence back to grid.  This is
    # the standard transform pathway for spectral models and stays
    # smooth / differentiable.
    if hasattr(grid, "n_max") and hasattr(grid, "Pnm"):
        flux_x = q_v_grid * u_grid    # (n_lat, n_lon, nlev)
        flux_y = q_v_grid * v_grid
        # ``vordiv_from_uv_3d`` returns spectral (vor, div).  We only
        # need the divergence; the helper batches the SH analyses so
        # discarding the curl does not waste a forward transform.
        _, div_hat = vordiv_from_uv_3d(grid, flux_x, flux_y)
        div_grid = sh_synthesis_3d(grid, div_hat)
        n_lat, n_lon, nlev = q_v_grid.shape
        return -div_grid.reshape(n_lat * n_lon, nlev)

    # Try lat-lon — duck-typed by attribute presence so we don't
    # introduce an import dependency for users who never touch lat-lon.
    if hasattr(grid, "dlat") and hasattr(grid, "dlon"):
        # Same convention as cubed-sphere — the FV operator already
        # returns ``-div(q·V)`` = MC.  Audit F1.
        mc = fv_flux_divergence_latlon_3d(
            q_v_grid, u_grid, v_grid, grid, limiter=False,
        )
        n_lat, n_lon, nlev = q_v_grid.shape
        return mc.reshape(n_lat * n_lon, nlev)

    raise TypeError(
        f"compute_moisture_convergence: unsupported grid type "
        f"{type(grid).__name__!r}.  Supported: CubedSphereGrid, "
        f"GaussianGrid, LatLonGrid."
    )


# ---------------------------------------------------------------------------
# omega → w conversion (for KF's ``w_grid`` trigger from a hydrostatic dycore)
# ---------------------------------------------------------------------------

def diagnose_grid_w_from_omega(
    omega: jnp.ndarray,
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    q_v: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Convert pressure-velocity ``ω = dp/dt`` to grid-scale ``w = dz/dt``.

    Uses the hydrostatic identity ``w = -ω / (ρ g)`` with ``ρ = p / (R_d T_v)``.
    Virtual temperature ``T_v = T (1 + (R_v/R_d - 1) q_v) ≈ T (1 + 0.608 q_v)``
    is used when ``q_v`` is provided; otherwise dry-air ``ρ`` is used.

    The Kain-Fritsch scheme is the only convection backend that
    consumes ``w_grid`` (for its boundary-layer trigger; see
    :func:`legoesm.atmosphere.physics.convection.kain_fritsch.kain_fritsch_convection`).
    For the non-hydrostatic dycore the bridge already passes the native
    half-level-averaged ``state.w``.  The hydrostatic bridge derives
    ``ω`` on the fly from ``∇·v_h`` via
    :func:`legoesm.grids.vertical.compute_sigma_dot` /
    :func:`legoesm.grids.vertical.compute_pressure_velocity` (cubed
    sphere and lat-lon C-grid only) and feeds the result here.  The
    spectral-PE bridge feeds the divergence carried in ``fields["div"]``
    through the same continuity → ω pathway.

    Parameters
    ----------
    omega : jax.Array, shape (ncol, nlev)
        Pressure velocity ``dp/dt`` [Pa/s].  Positive ω = downward
        motion (sinking) in the standard sign convention.
    T : jax.Array, shape (ncol, nlev)
        Air temperature [K].
    p_full : jax.Array, shape (ncol, nlev)
        Full-level pressure [Pa].
    q_v : jax.Array or None, shape (ncol, nlev)
        Water-vapor specific humidity [kg/kg].  When provided we use
        virtual temperature ``T_v = T (1 + 0.608 q_v)`` for ``ρ``;
        otherwise dry-air density is used.

    Returns
    -------
    jax.Array, shape (ncol, nlev)
        Grid-scale vertical velocity ``w = dz/dt`` [m/s].  Positive
        upward.

    Notes
    -----
    The ``T_v`` correction is small at typical tropospheric humidities
    (≈ 1 % at 16 g/kg) but matters for tropical convection columns
    where the column-mean ``q_v`` is non-negligible.  Floor ``T`` at 1 K
    inside ``ρ`` to keep AD finite at numerical singularities.
    """
    R_d = constants.R_d
    g = constants.g
    if q_v is None:
        T_v = T
    else:
        # ε = R_d / R_v ≈ 0.622, so 1/ε - 1 ≈ 0.608.
        eps = R_d / constants.R_v
        T_v = T * (1.0 + (1.0 / eps - 1.0) * q_v)
    rho = p_full / (R_d * jnp.clip(T_v, 1.0, None))
    return -omega / jnp.clip(rho * g, 1e-3, None)  # coeff-ok: rho*g floor for w-from-omega
