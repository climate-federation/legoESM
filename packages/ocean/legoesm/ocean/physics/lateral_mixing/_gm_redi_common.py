"""Grid-agnostic helpers shared by cubed-sphere and lat-lon GM/Redi.

Functions in this module operate on ``(..., nlev)`` arrays and make no
reference to a specific grid type.  Both ``gm_redi.py`` (cubed-sphere)
and ``gm_redi_latlon_cgrid.py`` (lat-lon C-grid) import from here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.ocean.eos import (
    compute_buoyancy_frequency,
    compute_buoyancy_frequency_adiabatic,
    rho_0 as _RHO_0_DEFAULT,
)
from legoesm.ocean.physics.lateral_mixing.config import (
    TreguierConfig,
    VisbeckConfig,
)
from legoesm.ocean.vertical import OceanZStarCoordinate

EPS = float(jnp.finfo(jnp.float32).eps)  # ~1.19e-7

# Division / stable-stratification guard shared by ALL four GM/Redi + MLE
# variants (#518 item 11).  Used as ``1/max(x, EPS_DIV)`` and the negative
# stratification floor ``min(drho/dz, -EPS_DIV)``.  Unified to 1e-10 (the value
# the validated lat-lon GM/Redi + both MLE paths already used): it bounds the
# reciprocal at 1e10 so the fp64 backward pass cannot overflow into pathological
# gradients, and it is the stable-strat slope floor those paths are tuned to.
# This REPLACES gm_redi_mpas's prior 1e-30 (a copy-paste drift — its own code
# comment said it "mirrors lat-lon convention", yet lat-lon is 1e-10; the 1e-30
# stratification floor let near-neutral MPAS columns build ~1e10x larger raw
# slopes than lat-lon before tapering).  NOT byte-identical for gm_redi_mpas by
# design — this is the bugfix the issue asks for.
EPS_DIV = 1e-10


# ---------------------------------------------------------------------------
# DM95 slope tapering
# ---------------------------------------------------------------------------

# Adjoint-stabilization modes for the GM/Redi isoneutral operator
# (``GMRediConfig.adjoint_stabilization``).  The slope saturation (DM95
# taper; ±S_max clip on the in-situ path) bounds the PRIMAL fluxes but the
# TAPER does not bound the LINEARIZED operator: ``d(taper·S)/d(state)``
# exceeds the primal coefficient bound via (a) the taper-derivative term
# ``S·taper'`` in the transition band (≈1/(2·width_frac) excess) and (b) the
# UNCLIPPED neutral-slope tangent ``∂S/∂(∇ρ) ∝ 1/∂_zρ`` in weakly-stratified
# cells (the dominant path on the real ACC state).  The tangent/adjoint
# propagator then amplifies per step where the primal is stable, and
# long-horizon reverse-mode PARAMETER gradients grow exponentially
# (~×2-5/step on the Veros ACC recipe; probes + verdict in
# .physics-validator/gm_adjoint_stab/RESULTS.md, demo campaign in
# .physics-validator/diff_veros_demos/).  The fix is the standard
# differentiable-solver flux-limiter trick: treat the slope coefficient —
# or, finer, just the limiter (taper) — as a non-differentiated
# (Picard-frozen) coefficient via ``stop_gradient``:
#
# - ``"none"`` (default): exact AD, bit-identical legacy behaviour.
# - ``"stop_gradient_slopes"`` (RECOMMENDED): ``stop_gradient`` on the
#   slopes (and hence on the tapers computed from them) — the full
#   frozen-coefficient linearization.  Kills both mechanisms; full ACC
#   step |G| 78 → 1.02.  Keeps the tracer-flux linearization and the
#   κ sensitivity, drops only the density→tensor feedback.
# - ``"stop_gradient_taper"``: ``stop_gradient`` on the DM95 taper FACTORS
#   only.  Kills mechanism (a) only — measured INSUFFICIENT on the
#   faithful (neutral-slope) ACC stack.
#
# All options are primal-invisible; none is used unless explicitly
# selected (default-off).  See the ``GMRediConfig.adjoint_stabilization``
# field doc for the full measured verdict.
VALID_ADJOINT_STABILIZATION = frozenset(
    {"none", "stop_gradient_taper", "stop_gradient_slopes"}
)


# Default GM/Redi taper transition width (fraction of taper range).
_DEFAULT_TAPER_WIDTH_FRAC = 0.1

def validate_slope_limit(mode: str) -> None:
    """Fail-fast membership check for GMRediConfig.slope_limit."""
    if mode not in ("dm95_taper", "nemo_cap"):
        raise ValueError(
            f"Unknown GMRediConfig.slope_limit {mode!r}; expected "
            "'dm95_taper' or 'nemo_cap'")


def validate_adjoint_stabilization(mode: str) -> None:
    """Fail-fast on an unknown ``adjoint_stabilization`` literal.

    Static Python check (Dispatch Discipline): a typo must NOT silently
    fall through to the exact-AD path (it would mask the stabilization
    entirely and the long-horizon gradients would silently explode again).
    """
    if mode not in VALID_ADJOINT_STABILIZATION:
        raise ValueError(
            f"Unknown GMRediConfig.adjoint_stabilization={mode!r}; "
            f"expected one of {sorted(VALID_ADJOINT_STABILIZATION)}."
        )


def dm95_taper(
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    S_max: float,
    eps: float = EPS,
    transition_width_frac: float = _DEFAULT_TAPER_WIDTH_FRAC,
    stop_gradient_taper: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply Danabasoglu & McWilliams (1995) smooth slope tapering.

    Returns tapered ``(S_x, S_y, taper)`` where *taper* is a smooth
    factor in [0, 1] computed as::

        taper = 0.5 * (1 + tanh((S_max - |S|) / (width_frac * S_max)))

    Parameters
    ----------
    S_x, S_y : array (..., nlev-1)
        Raw (clipped but un-tapered) isopycnal slopes at interfaces.
    S_max : float
        Slope at which the taper crosses 0.5. Equivalent to Veros's
        ``iso_slopec``.
    eps : float
        Small constant for sqrt regularisation.
    transition_width_frac : float
        Tanh transition half-width as a fraction of ``S_max``.
        Default ``0.1`` matches the legoESM pre-2026 convention.
        Veros's ``iso_dslope`` parameter maps via
        ``transition_width_frac = iso_dslope / iso_slopec``.
    stop_gradient_taper : bool
        When True, the taper factor is wrapped in ``lax.stop_gradient``
        (the ``"stop_gradient_taper"`` adjoint-stabilization mode; see
        the module note above).  Primal bit-identical; static Python
        gate (no traced branching).

    Returns
    -------
    S_x_tapered, S_y_tapered, taper : same shapes as inputs.
    """
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + eps)
    taper = 0.5 * (1.0 + jnp.tanh(
        (S_max - S_mag) / (transition_width_frac * S_max + eps)
    ))
    if stop_gradient_taper:
        taper = jax.lax.stop_gradient(taper)
    return S_x * taper, S_y * taper, taper


def dm95_taper_scalar(
    S: jnp.ndarray,
    S_max: float,
    eps: float = EPS,
    transition_width_frac: float = _DEFAULT_TAPER_WIDTH_FRAC,
    stop_gradient_taper: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Single-component variant of :func:`dm95_taper`.

    Used by grids that carry a scalar slope along each face's own normal
    (e.g. MPAS/Voronoi edges).  Identical functional form, with ``|S|``
    replaced by ``|S_n|``. See :func:`dm95_taper` for parameter
    semantics (in particular ``transition_width_frac`` ↔ Veros's
    ``iso_dslope / iso_slopec`` and ``stop_gradient_taper`` ↔ the
    ``"stop_gradient_taper"`` adjoint-stabilization mode).

    Returns
    -------
    S_tapered, taper : same shape as ``S``.
    """
    taper = 0.5 * (1.0 + jnp.tanh(
        (S_max - jnp.abs(S)) / (transition_width_frac * S_max + eps)
    ))
    if stop_gradient_taper:
        taper = jax.lax.stop_gradient(taper)
    return S * taper, taper


# ---------------------------------------------------------------------------
# Hallberg (2013) mesoscale-eddy resolution function for kappa_GM
# ---------------------------------------------------------------------------

# |f| floor [s^-1] for the deformation-radius denominator ``L_d = c/|f|`` near
# the equator.  1e-5 s^-1 is |f| at ~4 deg latitude, capping ``L_d`` at
# ``c/1e-5 ~ 2e5 m`` (~200 km for c ~ 2 m/s) -- the equatorial first-baroclinic
# deformation-radius scale (Chelton et al. 1998, JPO 28, 433) -- so the
# resolution function stays physical (~coarse limit, f_res -> 1 on a coarse
# grid) across the equator instead of collapsing to 0 as |f| -> 0.  This is a
# PHYSICAL cap on the equatorial band width, deliberately far larger than the
# 1e-10 denominator-safety floors (_TREGUIER_F_MIN / eke._DENOM_FLOOR) used
# elsewhere in this stack -- a 1e-10 floor would let L_d blow up to ~2e13 m and
# spuriously switch GM OFF (f_res -> 0) in a wide equatorial band.
_RESFN_F_FLOOR_S = 1.0e-5


def gm_resolution_function(
    L_d: jnp.ndarray, dx: jnp.ndarray, gamma: float,
) -> jnp.ndarray:
    r"""Hallberg (2013) mesoscale-eddy resolution function.

    .. math::

        f_{res} = \frac{1}{1 + \left(L_d / (\gamma\,\Delta)\right)^2}

    with ``L_d`` the first-baroclinic deformation radius [m], ``dx`` = Delta the
    local grid spacing [m], and ``gamma`` (~2) the number of grid points per
    deformation radius at which GM is half-suppressed.

    Asymptotics (physically-correct sense; NOTE the MED-3 spec prose swapped the
    coarse/fine word-labels -- this FORMULA and the 1deg / (1/12)deg test are the
    authoritative statement):

    - ``Delta >> L_d`` (COARSE grid, eddies unresolved): ``L_d/(gamma*Delta) ->
      0`` so ``f_res -> 1`` (full GM).
    - ``Delta << L_d`` (FINE / eddy-resolving grid): ``L_d/(gamma*Delta) ->
      inf`` so ``f_res -> 0`` (GM off; the resolved eddies do the transport).

    ``f_res`` is bounded in ``(0, 1]`` for any finite ``L_d >= 0``, ``dx > 0``,
    ``gamma > 0`` — EXACTLY, in floating point.  The denominator
    ``gamma*Delta`` is floored at ``EPS`` so a degenerate zero-area (land)
    cell gives ``f_res -> 0+`` rather than a NaN, and the formula is
    evaluated in the overflow-free form ``s = denom/hypot(denom, L_d)``,
    ``f_res = s*s`` — mathematically identical to ``1/(1+ratio**2)`` but
    with no intermediate ``ratio**2`` that can overflow (the naive form
    returned exactly ``0.0`` for ``ratio > sqrt(float_max)``, and its VJP
    hit ``0*inf = NaN`` once ``ratio`` itself overflowed; codex MED-3 r2).
    ``s*s`` can still UNDERFLOW to ``0.0`` for astronomically large ratios,
    so the result is clamped from below at the dtype's smallest positive
    normal (``finfo.tiny``): the codomain is exactly ``[tiny, 1] ⊂ (0, 1]``
    for every finite input.

    Differentiability (codex MED-3 r2): smooth in ``L_d`` away from the
    clamp; PIECEWISE-smooth in ``dx`` and ``gamma`` — the ``EPS`` floor and
    the tiny-clamp are hard kinks at ``gamma*dx == EPS`` and ``f_res == tiny``
    (measure-zero thresholds; the same clip/floor convention as the DM95
    slope bounds and the Treguier ``clip``/``min`` chain).  Gradients are
    FINITE for every finite input — including ``dx = 0``, ``L_d`` up to
    ``float_max``, and at every kink (``hypot`` has bounded partials away
    from the origin, and ``denom >= EPS`` keeps it off the origin) — but not
    continuous across the thresholds.  NOT globally C^1; do not claim so.
    """
    denom = jnp.maximum(gamma * dx, EPS)
    # Overflow-free evaluation of 1/(1 + (L_d/denom)^2); see docstring.
    s = denom / jnp.hypot(denom, L_d)
    f_res = jnp.asarray(s * s)
    # Exact float lower bound (see docstring): s*s can underflow to 0.0;
    # clamp at the dtype's smallest positive normal so the codomain is
    # exactly [tiny, 1] ⊂ (0, 1] for every finite input.
    return jnp.maximum(f_res, jnp.finfo(f_res.dtype).tiny)


def gm_resolution_factor(
    f_coriolis: jnp.ndarray,
    dx: jnp.ndarray,
    gamma: float,
    c_bcl_ms: float,
) -> jnp.ndarray:
    """The Hallberg (2013) resolution factor ``f_res`` field itself.

    ``f_res = gm_resolution_function(L_d, dx, gamma)`` with the fixed-``c``
    deformation radius ``L_d = c_bcl_ms / max(|f|, _RESFN_F_FLOOR_S)`` — the
    SINGLE definition shared by :func:`gm_resolution_scaled_kappa` (the
    kappa_GM taper applied inside the GM/Redi tracer tendencies) and the
    lat-lon model step's EKE-budget coupling (which must scale the GM-derived
    eddy-energy production by the SAME factor the tracer flux sees; codex
    MED-3 r2 closure-consistency fix).  Never re-derive this composition.

    Piecewise-smooth in ``f_coriolis``: the equatorial floor
    ``max(|f|, _RESFN_F_FLOOR_S)`` has kinks at ``|f| = _RESFN_F_FLOOR_S``
    (the ``|f|=0`` kink of ``abs`` sits inside the floored region and is
    flattened away); gradients are finite everywhere.
    """
    f_abs = jnp.maximum(jnp.abs(f_coriolis), _RESFN_F_FLOOR_S)
    L_d = c_bcl_ms / f_abs
    return gm_resolution_function(L_d, dx, gamma)


def gm_resolution_scaled_kappa(
    kappa_GM,
    f_coriolis: jnp.ndarray,
    dx: jnp.ndarray,
    gamma: float,
    c_bcl_ms: float,
):
    """Scale the effective GM coefficient by the Hallberg (2013) resolution fn.

    ``kappa_GM_eff = f_res * kappa_GM`` with ``f_res`` from
    :func:`gm_resolution_function` and the first-baroclinic deformation radius

        ``L_d = c_bcl_ms / max(|f|, _RESFN_F_FLOOR_S)``      [m]

    a FIXED gravity-wave-speed bound (``c_bcl_ms`` [m/s]; Chelton et al. 1998
    give ``c1 ~ 2 m/s`` in the open ocean).  A fixed ``c`` is used -- rather than
    the flow-dependent ``int(N dz)/pi`` deformation radius the EKE / GEOMETRIC
    closures already build via :func:`eke.eke_deformation_radius` -- so the taper
    applies UNIFORMLY to every closure, INCLUDING the constant-kappa path, which
    never computes ``int(N dz)``.  The equatorial |f| floor caps ``L_d`` (see
    ``_RESFN_F_FLOOR_S``).

    Applies to GM ONLY; the caller leaves the Redi isopycnal diffusivity
    ``kappa_Redi`` unscaled -- NEMO ``ldf_eiv`` / MOM6 resolution-scaled
    ``KhTh`` scale the eddy-transport (bolus) coefficient, not the
    along-isopycnal tracer diffusion.  As ``f_res -> 0`` (eddy-resolving) the
    GM/Redi tensor therefore reduces to pure Redi isopycnal diffusion.

    ``kappa_GM`` may be a Python/JAX scalar, a horizontal field matching
    ``f_coriolis`` / ``dx``, or a depth-resolved field with one extra trailing
    (level) axis (the 3-D prognostic-EKE skew coefficient); ``f_res`` is
    broadcast over that trailing axis.  Returns the same kind (scalar in ->
    field out, since ``f_res`` is a field).

    Differentiability: piecewise-smooth (kinks at the ``|f|`` floor, the
    ``EPS`` denominator floor and the tiny-clamp — see
    :func:`gm_resolution_factor` / :func:`gm_resolution_function`); gradients
    are finite everywhere.
    """
    f_res = gm_resolution_factor(f_coriolis, dx, gamma, c_bcl_ms)
    # Rank check via jnp.ndim, NOT ``isinstance(kappa_GM, jnp.ndarray)``:
    # the isinstance test silently SKIPPED this reshape for NumPy arrays and
    # for tracer types that don't register as jnp.ndarray, so a 3-D kappa
    # under jit/grad (or from host code) hit ``(lat,lon,lev) * (lat,lon)``
    # and failed to broadcast (codex MED-3 r2).  jnp.ndim is rank-static and
    # works for Python scalars (0), NumPy arrays, JAX arrays and tracers.
    extra_axes = jnp.ndim(kappa_GM) - jnp.ndim(f_res)
    if extra_axes > 0:
        # Depth-resolved kappa (..., nlev-1): broadcast f_res over the trailing
        # (level) axes it lacks.
        f_res = f_res.reshape(f_res.shape + (1,) * extra_axes)
    return kappa_GM * f_res


# ---------------------------------------------------------------------------
# Vertical flux divergence with zero-flux BCs
# ---------------------------------------------------------------------------

def vertical_flux_divergence(
    F_z: jnp.ndarray,
    dz_actual: jnp.ndarray,
    eps: float = EPS,
) -> jnp.ndarray:
    """Compute vertical flux divergence at full levels.

    ``dq/dt[k] = (F_z[k-1/2] - F_z[k+1/2]) / dz[k]``

    with F_z = 0 at the surface and bottom boundaries (zero-flux BCs).

    Parameters
    ----------
    F_z : array (..., nlev-1)
        Vertical flux at interior interfaces.
    dz_actual : array (..., nlev)
        Layer thicknesses at full levels.

    Returns
    -------
    tendency : array (..., nlev)
    """
    # NOT routed through ``mixing.flux_divergence_zero_flux`` (#518 item 4): the
    # GM/Redi flux ``F_z`` is a rotated isoneutral tensor flux (off-diagonal
    # S_x·dq/dx terms), NOT a simple K·dq/dz, AND this site builds the surface
    # ghost via ``jnp.pad`` (``0 - F_z[0]``), which differs from the diffusion
    # sites' explicit ``-F_z[0]`` by a sign-of-zero when ``F_z[0]`` is +0.0.
    # Keeping the original pad form preserves byte identity here.
    pad_axes = ((0, 0),) * (F_z.ndim - 1)
    F_z_ext = jnp.pad(F_z, (*pad_axes, (1, 1)))
    return (F_z_ext[..., :-1] - F_z_ext[..., 1:]) / jnp.maximum(dz_actual, eps)


# ---------------------------------------------------------------------------
# Visbeck (1997) adaptive GM coefficient
# ---------------------------------------------------------------------------

def compute_visbeck_kappa_gm(
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    cfg: VisbeckConfig,
    rho_ref: float = _RHO_0_DEFAULT,
    *,
    T: jnp.ndarray | None = None,
    S: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    eos_fn=None,
) -> jnp.ndarray:
    """Visbeck (1997) adaptive GM coefficient.

    ``kappa(x, y) = alpha * L^2 * <N * |S|>_z`` with the depth-average
    weighted by the local interface thickness, optionally using the
    local first-baroclinic Rossby radius as the mixing length.

    Parameters
    ----------
    rho : (..., nlev) in-situ density.
    S_x, S_y : (..., nlev-1) tapered isopycnal slopes at interfaces.
    z_coord : OceanZStarCoordinate.
    jacobian : (...,) z* Jacobian at cell centres.
    f_coriolis : (...,) Coriolis parameter.
    cfg : VisbeckConfig.
    rho_ref : float
        Boussinesq reference density [kg/m^3].

    Returns
    -------
    kappa : (...,) horizontally-varying kappa_GM [m^2/s], clamped to
        the configured bounds.
    """
    sigma_bar, L, wet_col, _int_N_dz, _sigma_local, _dzh = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg, rho_ref,
        n2_mode=getattr(cfg, "n2_mode", "insitu"),
        n2_over_dzw=getattr(cfg, "n2_over_dzw", False),
        T=T, S=S, p_cell=p_cell, eos_fn=eos_fn,
    )
    # Apply the wet-column mask AFTER clipping — otherwise dry columns
    # get lifted to ``kappa_min`` rather than 0 (Codex review caught
    # this).  A dry column should contribute exactly zero diffusivity
    # so it cannot leak gradients through the GM/Redi tendencies.
    kappa = jnp.clip(cfg.alpha * L ** 2 * sigma_bar, cfg.kappa_min, cfg.kappa_max)
    return jnp.where(wet_col, kappa, 0.0)


# --- NEMO ldf_eiv fixed scheme constants (ldftra.F90, nn_aei_ijk_t=21) ---
# Fixed values hard-coded in the NEMO source (not namelist tunables); the ONE
# genuine tunable is the cap aei0 = rn_Ue*rn_Le (TreguierConfig.aei0).
_TREGUIER_RO_FACTOR = 0.4          # Ro = 0.4*(integral N dz)/|f|   (ldf_eiv "zRo = .4*zn/zfw")
_TREGUIER_RO_MIN_M = 2.0e3         # Rossby-radius clamp, lower [m]
_TREGUIER_RO_MAX_M = 4.0e4         # Rossby-radius clamp, upper [m]
_TREGUIER_F_MIN = 1.0e-10          # |f| floor in the Ro division  (ldf_eiv zfw MAX)
_TREGUIER_ZHW_OFFSET_M = 5.0       # zhw initialisation offset [m] (ldf_eiv "zhw(:,:) = 5.")
_TREGUIER_TAPER_LAT_DEG = 20.0     # tropical taper reference latitude (z1_f20)


def compute_treguier_kappa_gm(
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    cfg: TreguierConfig,
    rho_ref: float = _RHO_0_DEFAULT,
) -> jnp.ndarray:
    r"""Treguier et al. (1997) / Held-Larichev (1996) eddy-induced-velocity
    coefficient — faithful port of NEMO 5.0.1 ``ldftra.F90::ldf_eiv``
    (``nn_aei_ijk_t = 21``, the DINO **and** ORCA1 oracle setting):

    .. math::

        \kappa = \min\big(\;\min(1, |f/f_{20}|)\cdot Ro^2\,T^{-1},\; aei0\big)

    with (all reductions over the interior interfaces, weights = the
    interface thickness ``dz_half`` ≙ NEMO ``e3w``):

    - ``Ro = clip(0.4\cdot\int N\,dz / \max(|f|,10^{-10}),\ 2\,\mathrm{km},\ 40\,\mathrm{km})``
      (internal Rossby radius, ldf_eiv ``zRo``);
    - ``T^{-1} = \sqrt{\;\Sigma\,N^2(S_x^2+S_y^2)\,dz\;/\;(5 + \Sigma\,dz)\;}``
      — the inverse baroclinic-instability timescale from the isopycnal
      slopes (ldf_eiv ``zah``/``zhw``; the ``+5`` m is NEMO's ``zhw``
      initialisation offset, kept for bit-faithfulness);
    - the tropical taper ``min(1, |f/f_{20}|)`` with ``f_{20} =
      2\Omega\sin 20^\circ``;
    - the cap ``aei0 = rn_Ue\cdot rn_Le`` (the ONE namelist tunable —
      ``TreguierConfig.aei0``; DINO: 0.03·100 km = 3000 m²/s).

    NEMO evaluates this at the surface and copies it down the column; ours
    is the 2-D ``kappa_GM(x, y)`` consumed by the GM/Redi operator — the
    same depth-independent semantics.  Reuses the SHARED
    ``_eady_growth_and_length`` chain for N²/N/σ (no duplicate numerics);
    N² is the in-situ model N² (NEMO uses its native ``rn2b``).

    Returns the 2-D ``kappa_GM`` [m²/s], exactly 0 on dry columns.
    """
    # The shared helper needs a Visbeck-shaped cfg ONLY for its mixing-length
    # branch (L is discarded here — Treguier builds its own Rossby radius from
    # the returned integral N dz). Default VisbeckConfig() supplies those
    # length fields; none of its values reach the Treguier formula.
    sigma_bar, _L, wet_col, int_N_dz, sigma, dz_half = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jacobian, f_coriolis,
        _TREGUIER_LENGTH_STUB, rho_ref,
    )
    del sigma_bar, _L
    f_abs = jnp.maximum(jnp.abs(f_coriolis), _TREGUIER_F_MIN)
    ro = jnp.clip(_TREGUIER_RO_FACTOR * int_N_dz / f_abs,
                  _TREGUIER_RO_MIN_M, _TREGUIER_RO_MAX_M)
    # T^-1 from the slope-weighted N² integral: sigma = N|S| at interfaces,
    # so sigma^2·dz = N²·(S_x²+S_y²)·dz  (ldf_eiv zah accumulation).
    zah = jnp.sum(sigma ** 2 * dz_half, axis=-1)
    zhw = _TREGUIER_ZHW_OFFSET_M + jnp.sum(dz_half, axis=-1)
    t_inv = jnp.sqrt(zah / zhw)
    f20 = 2.0 * constants.Omega * jnp.sin(
        jnp.deg2rad(_TREGUIER_TAPER_LAT_DEG))
    taper = jnp.minimum(1.0, jnp.abs(f_coriolis) / f20)
    kappa = jnp.minimum(taper * ro ** 2 * t_inv, cfg.aei0)
    return jnp.where(wet_col, kappa, 0.0)


_TREGUIER_LENGTH_STUB = VisbeckConfig()


def _eady_growth_and_length(
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    cfg,
    rho_ref: float = _RHO_0_DEFAULT,
    *,
    n2_mode: str = "insitu",
    n2_over_dzw: bool = False,
    T: jnp.ndarray | None = None,
    S: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    eos_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Depth-averaged Eady growth rate ``sigma_bar = <N|S|>_z`` and mixing length
    ``L`` (first-baroclinic Rossby radius, or ``cfg.L_fixed``), the wet-column mask,
    and the column buoyancy integral ``int_N_dz = ∫N dz`` [m/s]. Shared by the
    Visbeck diagnostic ``kappa_GM`` (``alpha·L²·sigma_bar``) and the prognostic-EKE
    closure (``kappa_GM = c_k·L·√E``, production ``∝ sigma_bar²``, and — for the
    ``"rhines"`` eke_len — the deformation radius ``c1=int_N_dz/π``) so the
    N²/slope/length numerics live in ONE place. ``cfg`` is a VisbeckConfig (uses
    ``L_min``, ``L_max``, ``f_min``, ``use_rossby_radius``, ``L_fixed``).

    ``n2_mode`` selects the static-stability N²:

    - ``"insitu"`` (default, BIT-IDENTICAL legacy): N² from the in-situ density
      gradient (:func:`compute_buoyancy_frequency`). The extra ``T/S/p_cell/eos_fn``
      kwargs are ignored — the in-situ branch never touches them, so passing the
      defaults (None) is byte-identical to the pre-change code.
    - ``"adiabatic"``: N² by adiabatic parcel displacement to the upper cell's
      pressure (:func:`compute_buoyancy_frequency_adiabatic`; Veros EKE chain).
      Requires ``T``, ``S``, ``p_cell`` (cell-centre hydrostatic pressure [Pa]) and
      ``eos_fn`` to displace parcels through the EOS; raises if any is missing.

    The adiabatic N² is SIGNED (not clipped), but the downstream ``N = sqrt(max(N²,
    1e-30))`` floors it ≥ 0 exactly as the in-situ path does — statically unstable
    interfaces contribute N ≈ 0 (no Eady growth, tiny buoyancy integral), which is
    the physically correct Eady/Rossby behaviour (no growth on an unstable column).

    Returns ``(sigma_bar, L, wet_col, int_N_dz, sigma)`` — ``sigma`` is the LOCAL
    Eady growth ``N|S|`` at interfaces (n_lat, n_lon, nlev-1), used by the 3-D EKE
    source (depth-resolved ``P = kappa_GM(z)·sigma(z)^2``); ``sigma_bar`` is its
    depth average (the 2-D EKE source).
    """
    eps = EPS
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])

    # Local growth rate sigma_Eady ~ N * |S| at each interior interface.
    if n2_mode == "insitu":
        N2 = compute_buoyancy_frequency(
            rho, z_coord.dz_ref, jacobian, rho_ref=rho_ref, g=constants.g,
        )
    elif n2_mode == "adiabatic":
        if T is None or S is None or p_cell is None or eos_fn is None:
            raise ValueError(
                "_eady_growth_and_length: n2_mode='adiabatic' requires T, S, "
                "p_cell (cell-centre pressure [Pa]) and eos_fn to displace "
                "parcels through the EOS; one or more was None."
            )
        # ``n2_over_dzw`` (EKEConfig/VisbeckConfig opt-in): divide the
        # adiabatic density contrast by the ACTUAL centre spacing
        # ``dz_half_ref·J`` — the Veros ``dzw`` slot (thermodynamics.py:99) —
        # instead of the midpoint reconstruction. On a Veros u_centered
        # coordinate the two differ by up to ±50% per level (the same slot
        # fixed for the TKE chain by ``TKEConfig.veros_dz_slots``; the
        # EKE-side slot was the deferred remainder of that batch). Default
        # False ⇒ BIT-IDENTICAL legacy. Only the N² divisor changes: the
        # column-reduction weights (``dz_half`` above) keep the legacy
        # midpoint metric, exactly like the TKE fix's scoping.
        _dzw = (z_coord.dz_half_ref * jacobian[..., jnp.newaxis]
                if n2_over_dzw else None)
        N2 = compute_buoyancy_frequency_adiabatic(
            T, S, p_cell, z_coord.dz_ref, jacobian,
            eos_fn=eos_fn, rho_ref=rho_ref, g=constants.g,
            dz_half=_dzw,
        )
    else:
        raise ValueError(
            "_eady_growth_and_length: n2_mode must be 'insitu' or 'adiabatic', "
            f"got {n2_mode!r}."
        )
    # Use a small positive floor on N² before sqrt, NOT a hard zero.
    # ``sqrt(0)`` has an infinite gradient in JAX; combined with the
    # ``maximum(N²,0)`` mask whose gradient is zero on the unstable
    # side, the backward pass evaluates ``inf * 0`` and produces NaN.
    # ``maximum(N², 1e-30)`` keeps N tiny but positive in unstable
    # layers, so ``sqrt`` has a finite (but very large) derivative
    # which is then multiplied by zero from ``maximum``'s VJP — a
    # well-defined zero rather than NaN.  Forward effect is at most
    # ``sqrt(1e-30) ≈ 1e-15``, negligible.
    N = jnp.sqrt(jnp.maximum(N2, 1e-30))
    # Regularise sqrt at zero slope — 1e-30 avoids spurious |S| ~ 3e-4
    # that the float32 eps (~1.19e-7) would produce.
    S_mag = jnp.sqrt(S_x ** 2 + S_y ** 2 + 1e-30)
    sigma = N * S_mag

    # Fused column reduction of ``ones``, ``sigma`` and ``N`` (shared ``dz_half``
    # weight). ``int_N_dz = Σ N·dz_half = ∫N dz`` [m/s] is the column buoyancy
    # integral (= Veros's ``C_rossby·π``); it is returned so the prognostic-EKE
    # eke_len deformation radius (``c1 = int_N_dz/π``) reuses the shared N WITHOUT
    # re-deriving N²/N anywhere (no duplicate numerics). The N reduction is shared by
    # both length modes; the ``L_fixed`` branch ignores ``N_bar`` but the fused sum
    # is identical, so ``sigma_bar``/``L``/``wet_col`` stay bit-identical to the
    # pre-int_N_dz code.
    #
    # Dry-column safeguard: when ``dz_half`` is all zero (jacobian = 0 over land /
    # dry cells), ``w_total = 0`` and N² = 0/0 = NaN, so EVERY column-reduced
    # quantity is explicitly zeroed on dry columns via ``wet_col`` — ``sigma_bar``,
    # ``N_bar`` AND ``int_N_dz`` alike — so the wet mask propagates cleanly through
    # forward values and gradients. (A raw NaN ``int_N_dz`` would otherwise poison
    # the "rhines" eke_len + its VJP on any config with true land; bit-identical on
    # wet columns, where ``wet_col`` is True.)
    _stack = jnp.stack([jnp.ones_like(sigma), sigma, N], axis=-1)
    _col = jnp.sum(_stack * dz_half[..., None], axis=-2)
    w_total = _col[..., 0]
    w_safe = jnp.maximum(w_total, eps)
    wet_col = w_total > eps
    sigma_bar = jnp.where(wet_col, _col[..., 1] / w_safe, 0.0)
    int_N_dz = jnp.where(wet_col, _col[..., 2], 0.0)  # ∫N dz [m/s]; masked like sigma_bar
    if cfg.use_rossby_radius:
        N_bar = jnp.where(wet_col, int_N_dz / w_safe, 0.0)
        H_col = jnp.sum(dz_actual, axis=-1)
        f_safe = jnp.maximum(jnp.abs(f_coriolis), cfg.f_min)
        L = jnp.clip(N_bar * H_col / f_safe, cfg.L_min, cfg.L_max)
    else:
        L = jnp.full_like(sigma_bar, cfg.L_fixed)

    return sigma_bar, L, wet_col, int_N_dz, sigma, dz_half


def compute_geometric_column_integrals(
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    visbeck_cfg,
    rho_ref: float = _RHO_0_DEFAULT,
    *,
    n2_mode: str = "insitu",
    n2_over_dzw: bool = False,
    T: jnp.ndarray | None = None,
    S: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    eos_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Column integrals for the GEOMETRIC EKE closure (Torres et al. 2025,
    JAMES, doi:10.1029/2025MS005394), from the SHARED Eady machinery — no
    duplicate N²/slope numerics.

    With the local Eady rate ``sigma(z) = N|S|`` at the interior interfaces
    (the SAME tapered/clipped slopes + N the GM/Redi tendency uses — the
    paper likewise builds ``M⁴/N²`` "from the isoneutral slopes already
    computed by the model", Eq. 2, p. 4; the slope bound/taper here is the
    GM/Redi config's S_max/DM95 machinery rather than NEMO's hard 1/100 +
    mixed-layer ramp — the model's canonical slope treatment):

        int_sigma2_dz = ∫(N|S|)² dz = ∫M⁴/N² dz   [m/s²]  (B_C integrand,
                                                            Eq. 2)
        int_sigma_dz  = ∫ N|S|  dz  = ∫M²/N  dz   [m/s]   (kappa_gm
                                                            denominator, Eq. 6)
        int_N_dz      = ∫N dz                     [m/s]   (R_d, Appendix D)
        H_col         = Σ dz                      [m]     (column depth)

    plus the wet-column mask.  The interface measure is the SAME ``dz_half``
    weight ``_eady_growth_and_length`` uses for its column averages, so
    ``int_sigma_dz`` is numerically consistent with ``sigma_bar·Σdz_half``.
    All outputs are zeroed on dry columns (NaN-free, like the Eady outputs).

    Returns ``(int_sigma2_dz, int_sigma_dz, int_N_dz, H_col, wet_col)``.
    """
    _sigma_bar, _L, wet_col, int_N_dz, sigma, _dzh = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jacobian, f_coriolis, visbeck_cfg, rho_ref,
        n2_mode=n2_mode, n2_over_dzw=n2_over_dzw,
        T=T, S=S, p_cell=p_cell, eos_fn=eos_fn,
    )
    # Interface thickness weights — the same metric assembly as
    # _eady_growth_and_length (dz at centres from the z* Jacobian; half-sums
    # at the interior interfaces).
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    int_sigma_dz = jnp.where(wet_col, jnp.sum(sigma * dz_half, axis=-1), 0.0)
    int_sigma2_dz = jnp.where(
        wet_col, jnp.sum(sigma ** 2 * dz_half, axis=-1), 0.0)
    H_col = jnp.sum(dz_actual, axis=-1)
    return int_sigma2_dz, int_sigma_dz, int_N_dz, H_col, wet_col


def compute_eke_kappa_gm(
    E: jnp.ndarray,
    rho: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    f_coriolis: jnp.ndarray,
    visbeck_cfg,
    eke_cfg,
    rho_ref: float = _RHO_0_DEFAULT,
    *,
    beta: jnp.ndarray | None = None,
    depth_resolved: bool = False,
    T: jnp.ndarray | None = None,
    S: jnp.ndarray | None = None,
    p_cell: jnp.ndarray | None = None,
    eos_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Prognostic GM coefficient from the eddy-energy field ``E`` (Eden-Greatbatch).

    Reuses the SHARED Eady-rate/Rossby-length machinery (``_eady_growth_and_length``,
    using ``visbeck_cfg`` for the length params) and the EKE closure. Returns
    ``(kappa_GM, sigma_bar, L)``: ``kappa_GM = c_k·L·√E`` (2-D, masked to wet
    columns) for the GM/Redi tendency, and ``sigma_bar`` (depth-averaged Eady growth
    rate) + ``L`` (mixing length) for the EKE local source/sink
    (``eke_local_tendency``). Pure.

    The mixing length ``L`` follows ``eke_cfg.mixing_length_scheme``:

    - ``"rossby"`` (default) — ``L = max(L_rossby, l_min)``, the Visbeck
      first-baroclinic length (legoESM's pre-eke_len behaviour). ``beta`` unused.
    - ``"rhines"`` — Veros ``eke_len = max(l_min, min(eke_cross·L_rossby,
      eke_crhin·L_rhines))`` from the deformation radius ``c1=int_N_dz/π`` and the
      eddy-energy Rhines scale ``√(√E/β)``. Requires ``beta`` (df/dy); raises
      otherwise.

    Dispatch is on the static ``mixing_length_scheme`` Python string, so the
    ``if/elif/else`` runs at trace time (not on a traced value).
    """
    from legoesm.ocean.physics.lateral_mixing.eke import (
        eke_deformation_radius, eke_kappa_gm, eke_len_composite,
        eke_mixing_length, eke_rhines_length,
    )

    # The Eady/Rossby N² chain for the EKE closure is governed by the EKE
    # config's ``n2_mode`` (mirroring TKEConfig.n2_mode), NOT the Visbeck
    # config's — the Visbeck diagnostic and the prognostic EKE may opt in
    # independently. ``visbeck_cfg`` is still used for the LENGTH params
    # (L_min/L_max/f_min/use_rossby_radius/L_fixed).
    sigma_bar, L_rossby, wet_col, int_N_dz, sigma_local, _dzh = _eady_growth_and_length(
        rho, S_x, S_y, z_coord, jacobian, f_coriolis, visbeck_cfg, rho_ref,
        n2_mode=getattr(eke_cfg, "n2_mode", "insitu"),
        n2_over_dzw=getattr(eke_cfg, "n2_over_dzw", False),
        T=T, S=S, p_cell=p_cell, eos_fn=eos_fn,
    )
    scheme = eke_cfg.mixing_length_scheme
    if depth_resolved:
        # 3-D (depth-resolved) closure: E, L, kappa_GM at interior interfaces
        # (n_lat, n_lon, nlev-1), matching the LOCAL Eady growth ``sigma_local``.
        # The 2-D deformation radius / Rossby length broadcast over levels; only the
        # eddy Rhines scale (∝√(√E/β)) and ``kappa = c_k·L·√E`` carry the genuine 3-D
        # structure (E is 3-D). Returns (kappa(z), sigma(z), L(z)).
        if scheme == "rossby":
            L3 = eke_mixing_length(L_rossby, eke_cfg)[..., jnp.newaxis]
        elif scheme == "rhines":
            if beta is None:
                raise ValueError(
                    "compute_eke_kappa_gm(depth_resolved=True): "
                    "mixing_length_scheme='rhines' requires `beta` (df/dy)."
                )
            L3 = eke_len_composite(
                eke_deformation_radius(int_N_dz, f_coriolis, beta, eke_cfg)[..., jnp.newaxis],
                eke_rhines_length(E, beta[..., jnp.newaxis], eke_cfg),
                eke_cfg,
            )
        else:
            raise ValueError(
                "EKEConfig.mixing_length_scheme must be 'rossby' or 'rhines', got "
                f"{scheme!r}"
            )
        kappa3 = eke_kappa_gm(E, L3, eke_cfg)
        return jnp.where(wet_col[..., jnp.newaxis], kappa3, 0.0), sigma_local, L3
    if scheme == "rossby":
        L = eke_mixing_length(L_rossby, eke_cfg)
    elif scheme == "rhines":
        if beta is None:
            raise ValueError(
                "compute_eke_kappa_gm: mixing_length_scheme='rhines' requires "
                "`beta` (df/dy [1/(m·s)]); none was passed."
            )
        L = eke_len_composite(
            eke_deformation_radius(int_N_dz, f_coriolis, beta, eke_cfg),
            eke_rhines_length(E, beta, eke_cfg),
            eke_cfg,
        )
    else:
        raise ValueError(
            "EKEConfig.mixing_length_scheme must be 'rossby' or 'rhines', got "
            f"{scheme!r}"
        )
    kappa = eke_kappa_gm(E, L, eke_cfg)
    return jnp.where(wet_col, kappa, 0.0), sigma_bar, L
