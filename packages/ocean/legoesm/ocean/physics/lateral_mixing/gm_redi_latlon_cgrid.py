"""GM/Redi isopycnal mixing for the lat-lon Arakawa C-grid ocean.

Griffies (1998) skew-flux formulation with DM95 slope tapering,
optional Visbeck (1997) adaptive coefficient, and full land-mask
treatment via Neumann fill + face masks.

Grid conventions (LatLonGrid C-grid staggering):
- Scalars (rho, T, S): cell centers, shape (n_lat, n_lon, nlev)
- u-fluxes: lon interfaces, shape (n_lat, n_lon+1, nlev)
- v-fluxes: lat interfaces, shape (n_lat+1, n_lon, nlev)

References
----------
- Griffies, S. M. (1998). The Gent-McWilliams skew flux.
  J. Phys. Oceanogr., 28, 831-841.
- Danabasoglu, G. & McWilliams, J. C. (1995). J. Climate, 8, 2967-2987.
- Visbeck, M. et al. (1997). J. Phys. Oceanogr., 27, 381-402.
"""

from __future__ import annotations

import jax
from jax import lax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.source_rounding import nemo_source_round
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks_3d,
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    interp_cell_to_uface,
    interp_cell_to_vface,
    interp_wface_to_center,
    laplacian_cgrid,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import neumann_fill_cgrid
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.eos import (
    compute_hydrostatic_pressure,
    eos_density_derivatives,
    int_drhodTS_dynamic_enthalpy,
    make_eos_fn,
    nemo_bn2_live_ladders,
    nemo_r3t_stretch,
    NemoSEOSConfig,
    nemo_seos_prd_literal,
    rho_0 as _RHO_0,
)
from legoesm.ocean.vertical import (
    nemo_qco_live_face_thicknesses,
    nemo_qco_live_t_thickness,
)
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    EPS,
    EPS_DIV as _EPS_DIV,
    compute_eke_kappa_gm,
    compute_treguier_kappa_gm,
    validate_treguier_cfg,
    compute_visbeck_kappa_gm,
    dm95_taper,
    dm95_taper_scalar,
    gm_resolution_scaled_kappa,
    validate_adjoint_stabilization,
    validate_slope_limit,
    vertical_flux_divergence,
)
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.output import LateralMixingOutput
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate, OceanZStarCoordinate, compute_ocean_jacobian,
)

__physics_contract__ = {
    "summary": (
        "GM/Redi isopycnal mixing on the lat-lon Arakawa C-grid (Griffies 1998 "
        "skew-flux tensor): per-tracer adiabatic eddy advection + isoneutral "
        "diffusion via a single divergence of the combined face fluxes, with "
        "DM95 tapering, optional Visbeck kappa, and land face-mask / Neumann-"
        "fill treatment."
    ),
    "inputs": {
        "q": "degC or psu (tracer)", "S_x": "1 (isopycnal slope)",
        "S_y": "1 (isopycnal slope)", "mask": "1 (1=ocean)",
        "u_mask": "1 (u-face)", "v_mask": "1 (v-face)",
        "jacobian": "1 (z-star dimensionless)",
        "kappa_GM": "m^2/s", "kappa_Redi": "m^2/s",
    },
    "outputs": {"dq_dt": "degC/s or psu/s (tracer tendency)"},
    "sign_convention": (
        "kappa_GM, kappa_Redi >= 0; isopycnal slopes S_x,S_y tapered (DM95); GM "
        "skew flux adiabatic + Redi along-isopycnal down-gradient; fluxes are "
        "combined at faces and passed through ONE divergence_cgrid so the "
        "(mask-weighted) volume integral of the tracer is conserved; land faces "
        "carry no flux; z positive up."
    ),
    # Adiabatic tracer redistribution: conserves volume-integrated tracer.
    "conserves": ["tracer"],
    "differentiable": True,
    "reference": (
        "Griffies (1998) JPO 28, 831-841; Danabasoglu & McWilliams (1995) "
        "J. Climate 8, 2967-2987; Visbeck et al. (1997) JPO 27, 381-402"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_gm_redi_latlon_cgrid.py — sloping isopycnals "
        "give a slope-flattening tendency conserving the volume-integrated "
        "tracer; zero slope gives zero tendency; land faces carry no flux."
    ),
}

# Division-guard epsilon (shared _gm_redi_common.EPS_DIV = 1e-10, #518 item 11):
# larger than float32 machine eps to prevent intermediate blow-up in the
# backward pass (see plan §7, AD safety).
# Salinity floor [PSU] applied ONLY to the inputs of EOS *differentiation*
# (jax.grad): a sqrt(S)-bearing EOS (TEOS-10 gsw) has an unbounded ∂ρ/∂S at
# exactly S=0, so jax.grad returns NaN there — and NaN survives the
# downstream ``× mask`` (0·NaN = NaN).  S=0 occurs only on LAND cells the
# 3-pass horizontal Neumann fill could not reach (continental interiors) and
# on sub-seafloor dry cells (handled by _fill_dry_cells_columnwise); both
# have their derivative outputs masked/tapered away, so the floor never
# touches a physical value (wet salinity ≫ 1e-3 by the kbot salt==0 land
# rule).  Numerical safety floor, not a physical constant.
_S_GRAD_FLOOR = 1.0e-3


# Default GM/Redi taper transition width (fraction of taper range).
_DEFAULT_TAPER_WIDTH_FRAC = 0.1

# --- NEMO ldfslp mixed-layer slope ramp (zdfmxl.F90 + ldfslp.F90) ---
# Reference depth [m] for the mixed-layer-depth density criterion (NEMO's
# ``nlb10`` ~ 10 m level; zdfmxl integrates N^2 from here).
_NEMO_MLD_REF_DEPTH_M = 10.0
# Floor [m] on the mixed-layer depth used in the w-point slope ramp normaliser
# (ldfslp.F90:161 ``r1_hmlw = 1/MAX(hmlp - gdepw_top, 10.)``).
# --- NEMO ldfslp slope stability bound (ldfslp.F90:212-213) ---
_NEMO_HML_UV_FLOOR_M = 5.0   # [m] ldfslp:155-158 MAX(zhmlpt,...,5.) u/v ML floor
_NEMO_SLOPE_STAB_7E3 = 7.0e3  # [m] |S| <= e3/7e3 ("kxz max =< e1 e3/(pi^2 2 dt)")
_NEMO_HMLW_FLOOR_M = 10.0

def _kappa_is_interface_3d(kappa, nlev: int) -> bool:
    """True iff ``kappa`` is a depth-resolved *interface* (W-grid) diffusivity.

    The 3-D EKE closure (Stages 1-2) supplies ``kappa_GM`` / ``kappa_Redi`` at
    the ``nlev-1`` interior interfaces.  Distinguishing this from the legacy
    ``(n_lat, n_lon, 1)`` broadcast (a 2-D per-column kappa lifted with a
    trailing length-1 axis) is by the last-axis length:

    - last axis ``== nlev-1`` **and** ``nlev > 2`` ⇒ interface kappa (new path).
    - last axis ``== 1`` ⇒ broadcast (legacy path), so a ``(n,n,1)`` kappa is
      ALWAYS treated as a broadcast.

    The ``nlev > 2`` guard resolves the only ambiguous case: when ``nlev == 2``
    the interior-interface count ``nlev-1 == 1`` collides with the ``(n,n,1)``
    broadcast shape, so we never interpret a length-1 trailing axis as an
    interface field.  An interface kappa may therefore only be passed when
    ``nlev > 2`` (every real ocean config has ``nlev >= 10``).  A scalar or a
    ``(n_lat, n_lon)`` 2-D kappa is not an ndarray-with-ndim-3 and returns
    False here, taking the bit-identical broadcast path downstream.
    """
    return (
        isinstance(kappa, jnp.ndarray)
        and kappa.ndim == 3
        and kappa.shape[-1] == nlev - 1
        and nlev > 2
    )


def _kappa_center_uvw(kappa, nlev: int):
    """Resolve a kappa argument to its center / u-face / v-face / w-face forms.

    Returns ``(kappa_c, kappa_u, kappa_v, kappa_w)`` where:

    - **scalar / 2-D ``(n_lat, n_lon)``** (the legacy path) — the 2-D array is
      lifted to ``(n_lat, n_lon, 1)`` cell-centred and interpolated to u/v-faces
      exactly as before; ``kappa_w = kappa_c``.  A scalar passes through
      unchanged on all four outputs.  **Bit-identical to the prior code.**
    - **3-D interface ``(n_lat, n_lon, nlev-1)``** (the 3-D EKE path) — the
      interface kappa is the w-face value DIRECTLY (``kappa_w = kappa``; it
      already lives at the w-faces, so no lossy round-trip), while the
      horizontal forms come from a vertical interface→center interpolation
      (``interp_wface_to_center``) followed by the usual horizontal
      cell→u/v-face interpolation.

    Shared by the triad scheme, the centered scheme, and the K_33 getter so the
    three can never diverge.
    """
    if _kappa_is_interface_3d(kappa, nlev):
        kappa_w = kappa                              # (n_lat, n_lon, nlev-1)
        kappa_c = interp_wface_to_center(kappa)      # (n_lat, n_lon, nlev)
        kappa_u = interp_cell_to_uface(kappa_c)      # (n_lat, n_lon+1, nlev)
        kappa_v = interp_cell_to_vface(kappa_c)      # (n_lat+1, n_lon, nlev)
        return kappa_c, kappa_u, kappa_v, kappa_w
    if isinstance(kappa, jnp.ndarray) and kappa.ndim == 2:
        kappa_c = kappa[:, :, jnp.newaxis]           # (n_lat, n_lon, 1)
        kappa_u = interp_cell_to_uface(kappa_c)      # (n_lat, n_lon+1, 1)
        kappa_v = interp_cell_to_vface(kappa_c)      # (n_lat+1, n_lon, 1)
        return kappa_c, kappa_u, kappa_v, kappa_c
    # Scalar (or already a (n,n,1) broadcast array): pass through unchanged.
    return kappa, kappa, kappa, kappa


# =====================================================================
# Neutral (locally-referenced) density-gradient ingredients
# =====================================================================

_VALID_SLOPE_DENSITY = frozenset({"in_situ", "neutral"})


def _validate_slope_density(slope_density: str) -> None:
    """Fail-fast on an unknown ``slope_density`` literal (Dispatch Discipline).

    The in-situ / neutral selection is a binary ``if`` rather than a factory,
    but an unrecognised value must NOT silently fall through to the in-situ
    branch (a typo would then mask the neutral option entirely).
    """
    if slope_density not in _VALID_SLOPE_DENSITY:
        raise ValueError(
            f"Unknown GMRediConfig.slope_density={slope_density!r}; "
            f"expected one of {sorted(_VALID_SLOPE_DENSITY)}."
        )


def _fill_dry_cells_columnwise(q, z_coord):
    """Fill SUB-SEAFLOOR (dry) cells with the deepest overlying active value.

    Variable-bathymetry guard for EOS *differentiation*: with a partial-cell
    coordinate the cells below the seafloor carry the IC fill value (T=S=0).
    Density EVALUATION there is finite, but ``jax.grad`` of an EOS with a
    ``sqrt(S)`` term (TEOS-10 gsw, ``eos="veros_gsw"``) is NaN at exactly
    S=0 — and one NaN at a bottom face propagates through the B-triad slopes
    and the implicit-K33 column solve to the WHOLE column (global_4deg: every
    non-full-depth column went NaN on step 1).  Dry-cell derivative values
    are physically arbitrary — they are killed downstream by the wet-face
    indicators and the dm95 taper (exactly how Veros handles its salt=0 dry
    cells: its ANALYTIC ``gsw_drhodS`` is finite at 0 and ``maskW`` zeroes
    the flux) — but they must be FINITE for the masking to work.

    Thin no-op wrapper around the canonical
    :func:`legoesm.ocean.vertical.extrapolate_below_seafloor` (no duplicate
    numerics): bit-identical pass-through (zero traced ops) when ``z_coord``
    has no ``is_active`` (pure z-star, e.g. the flat-bottom ACC recipes).
    """
    if getattr(z_coord, "is_active", None) is None:
        return q
    from legoesm.ocean.vertical import extrapolate_below_seafloor
    return extrapolate_below_seafloor(q, z_coord)


def _per_level_face_acts(z_coord, grid, dtype):
    """Per-level u/v-face activity masks for the W-face triad numerators.

    ``(u_face_act, v_face_act)`` from ``compute_face_masks_3d`` on a
    partial-cell coordinate, or ``(None, None)`` for pure z-star (flat
    bottom) — the ``None`` path skips the numerator masking entirely
    (bit-identical legacy).  Shared by ``compute_isoneutral_K33_latlon`` and
    ``compute_realized_gm_skew_conversion`` so they kill the same coast/step
    triads the tracer-tendency assembly kills (Veros's per-level
    ``maskU``/``maskV``-masked gradients; see ``_w_triad_numerators``).
    """
    is_active = getattr(z_coord, "is_active", None)
    if is_active is None:
        return None, None
    um3, vm3 = compute_face_masks_3d(is_active, grid)
    return um3.astype(dtype), vm3.astype(dtype)


def _neutral_drho_derivs(T, S, mask, z_coord, jacobian, eos_fn, rho_0, g):
    """Cell-centred EOS partials ``(∂ρ/∂T, ∂ρ/∂S)`` at the LOCAL cell pressure.

    The ingredients of the locally-referenced *neutral* density gradient used
    by the ``slope_density="neutral"`` isoneutral slope build (Veros
    ``get_drhodT`` / ``get_drhodS`` at ``abs(zt)``;
    ``veros/core/isoneutral/isoneutral.py:40-41``).

    The LOCAL pressure is built EXACTLY as the in-situ EOS iteration
    (``iterate_eos_and_pressure_anomaly``) builds it — the *reference*
    hydrostatic pressure ``compute_hydrostatic_pressure(ρ, η=0, dz_ref, J=1)``
    — so the neutral derivatives reference the SAME cell pressure the in-situ
    ρ (and hence the in-situ slope) already uses (plan §"local pressure must
    be the SAME hydrostatic pressure the slope builder uses").  For a
    hydrostatic column this is ``ρ₀·g·z``, the legoESM analogue of Veros's
    ``abs(zt)``.  T, S are Neumann-filled so the partials on land take an
    ocean-neighbour value (the gradients across coastlines are still masked by
    the face masks downstream).

    Returns ``(drdT, drdS)`` each ``(n_lat, n_lon, nlev)`` at cell centres,
    masked to the wet domain.
    """
    T_filled = _fill_dry_cells_columnwise(neumann_fill_cgrid(T, mask), z_coord)
    S_filled = _fill_dry_cells_columnwise(neumann_fill_cgrid(S, mask), z_coord)
    # Off-the-sqrt(S)-singularity guard for the jax.grad evaluation below;
    # only unreachable-land cells (masked outputs) sit at S=0 (see
    # _S_GRAD_FLOOR).
    S_filled = jnp.maximum(S_filled, _S_GRAD_FLOOR)
    # In-situ density via the SAME 2-iteration EOS coupling the slope builder
    # uses, then the reference hydrostatic pressure (η=0, J=1) — identical to
    # iterate_eos_and_pressure_anomaly's internal p_hydro.  (The dry-cell
    # column fill changes ONLY sub-seafloor values: the hydrostatic integral
    # at a wet cell never reads cells below it, so wet-cell drdT/drdS are
    # bit-identical; see _fill_dry_cells_columnwise.)
    horiz_shape = T.shape[:-1]
    J_ref = jnp.ones(horiz_shape, dtype=T.dtype)
    eta_ref = jnp.zeros(horiz_shape, dtype=T.dtype)
    rho = eos_fn(T_filled, S_filled, jnp.zeros_like(T))
    for _ in range(2):
        p_hydro = compute_hydrostatic_pressure(
            rho, eta_ref, z_coord.dz_ref, J_ref, rho_0, g,
        )
        rho = eos_fn(T_filled, S_filled, p_hydro)
    drdT, drdS = eos_density_derivatives(eos_fn, T_filled, S_filled, p_hydro)
    return drdT * mask[:, :, jnp.newaxis], drdS * mask[:, :, jnp.newaxis]


def _slope_density_face_grads(
    rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
    eos_fn, rho_0, g,
):
    """Density gradients at the C-grid faces used to BUILD isoneutral slopes.

    Returns ``(drho_dx_u, drho_dy_v, drho_dz_w)`` — the eastward (u-face),
    northward (v-face) and vertical (w-face) density gradients with the
    stable-strat floor on ``drho_dz_w``.  Two modes:

    - ``"in_situ"`` (default, BIT-IDENTICAL): finite-difference the in-situ
      ``rho_filled``.
    - ``"neutral"``: the locally-referenced neutral form
      ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S``.  The horizontal face gradients use ``∂ρ/∂T``
      interpolated to the u/v-face (``interp_cell_to_uface`` / ``..._vface``)
      times the raw tracer face gradient; the vertical w-face gradient uses the
      UPPER-cell ``∂ρ/∂T`` (``drdT[:, :, :-1]``) — the validated probe form,
      matching Veros's ``drodzb`` with the cell's own derivative (the kr-sum
      over both triad levels is the documented metric-faithful follow-up).

    Shared by the centred slope builder (Visbeck/EKE diagnostics) so the two
    modes never diverge there.  The TRIAD / K_33 path uses
    :func:`_w_triad_slope_density_grads` instead (per-triad cell references).
    """
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]
    if slope_density == "neutral":
        drdT, drdS = _neutral_drho_derivs(
            T, S, mask, z_coord, jacobian, eos_fn, rho_0, g,
        )
        T_filled = neumann_fill_cgrid(T, mask)
        S_filled = neumann_fill_cgrid(S, mask)
        dTdx_u = gradient_x_cgrid(T_filled, grid)
        dSdx_u = gradient_x_cgrid(S_filled, grid)
        dTdy_v = gradient_y_cgrid(T_filled, grid)
        dSdy_v = gradient_y_cgrid(S_filled, grid)
        dTdz_w = (T_filled[:, :, :-1] - T_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        dSdz_w = (S_filled[:, :, :-1] - S_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        drdT_u = interp_cell_to_uface(drdT)
        drdS_u = interp_cell_to_uface(drdS)
        drdT_v = interp_cell_to_vface(drdT)
        drdS_v = interp_cell_to_vface(drdS)
        drho_dx_u = drdT_u * dTdx_u + drdS_u * dSdx_u
        drho_dy_v = drdT_v * dTdy_v + drdS_v * dSdy_v
        drdT_w = drdT[:, :, :-1]
        drdS_w = drdS[:, :, :-1]
        drho_dz_w = drdT_w * dTdz_w + drdS_w * dSdz_w
    else:
        drho_dx_u = gradient_x_cgrid(rho_filled, grid)
        drho_dy_v = gradient_y_cgrid(rho_filled, grid)
        drho_dz_w = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
    # Stable-strat floor: ∂_zρ must be negative (z UPWARD ⇒ ρ denser below).
    drho_dz_w = jnp.minimum(drho_dz_w, -_EPS_DIV)
    return drho_dx_u, drho_dy_v, drho_dz_w


# =====================================================================
# NEMO ldfslp mixed-layer slope ramp
# =====================================================================


def _nemo_mld_from_potential_density(T, S, mask, z_coord, eos_fn, rho_c,
                                     active_3d=None):
    """Mixed-layer depth [m] via NEMO's zdfmxl density criterion.

    NEMO (``zdfmxl.F90:95-104``) integrates the buoyancy frequency ``N^2`` from
    the ~10 m reference level and sets the mixed-layer w-level ``nmln`` where the
    cumulative ``integral(N^2 dz) = g/rho0 * Delta_rho_neutral`` first reaches
    ``g*rho_c/rho0`` — i.e. a NEUTRAL density difference of ``rho_c`` from the
    reference level.  We reproduce that with a POTENTIAL density (EOS referenced
    to the surface, ``p=0``), which removes the in-situ compressibility bias
    (~0.45 kg/m^3 over 100 m, ≫ ``rho_c``) that a raw in-situ column difference
    would carry — the textbook sigma_theta MLD criterion, matching NEMO's N^2
    integral over the mixed layer to within the EOS reference-pressure choice.

    Returns ``(hml, m_base)``: the MLD ``hml`` [m, positive] = w-interface depth
    at the ML base, and ``m_base`` the legoESM *interface* index (0..nlev-2) of
    that base (the deepest interface still in the mixed layer).  Both are
    ``(n_lat, n_lon)``.  AD-safe (no stop_gradient), but the MLD *level* ``m_base``
    is index-selected (argmax/searchsorted over the static depth ladder), so it is
    a quantized step function of T/S with ZERO gradient through the ramp
    normaliser ``hml``; ``grad`` reaches T/S through the below-ML base slope (and
    the unchanged below-ML slopes), which is genuinely differentiable and finite.
    (Mirrors NEMO's discrete ``nmln``; unlike the tramle MLD diagnostic, no
    stop_gradient is applied.)
    """
    # 3-D fill mask when available: sub-seafloor dry cells CARRY their IC
    # (MLF carry fix) and must never enter the MLD density/N^2 — NEMO gates
    # on 3-D tmask (zdfmxl/eosbn2); the 2-D surface mask leaks carried deep
    # values into the ML ramp at bathymetry steps (#1226 boundary-column
    # T anomaly, poison-probe verified).
    _fillm = mask if active_3d is None else active_3d
    T_filled = neumann_fill_cgrid(T, _fillm)
    S_filled = neumann_fill_cgrid(S, _fillm)
    rho_pot = eos_fn(T_filled, S_filled, jnp.zeros_like(T_filled))  # (...,nlev)
    nlev = rho_pot.shape[-1]
    dz_ref = z_coord.dz_ref
    z_iface = jnp.cumsum(dz_ref)                      # (nlev,) bottom-of-cell depths
    z_centers = z_iface - 0.5 * dz_ref               # (nlev,) cell-centre depths
    # Reference density at the nearest cell centre at/below ~10 m (NEMO nlb10).
    iref = jnp.clip(
        jnp.searchsorted(z_centers, jnp.asarray(_NEMO_MLD_REF_DEPTH_M, z_centers.dtype)),
        0, nlev - 1)
    rho_ref = jnp.take(rho_pot, iref, axis=-1)        # (n_lat, n_lon)
    z_ref = jnp.take(z_centers, iref)                 # scalar
    below_ref = (z_centers > z_ref).reshape((1, 1, nlev))
    exceed = (rho_pot > rho_ref[:, :, None] + rho_c) & below_ref
    has = jnp.any(exceed, axis=-1)                    # (n_lat, n_lon)
    first = jnp.argmax(exceed.astype(jnp.int32), axis=-1)   # first stratified cell
    first = jnp.where(has, first, nlev - 1)           # unstratified col -> deepest
    # ML-base w-interface = top of the first stratified cell = bottom of the last
    # mixed cell = interface index (first-1), clamped into [0, nlev-2].
    m_base = jnp.clip(first - 1, 0, nlev - 2)
    hml = jnp.take(z_iface, m_base)                   # (n_lat, n_lon)
    return hml, m_base


def _nemo_mld_from_n2_integral(T, S, mask, z_coord, eos_fn, rho_c, g, rho_0,
                               active_3d=None, jacobian=None,
                               n2_override=None, e3w_override=None,
                               eos_nemo_seos=None):
    """Mixed-layer depth [m] via NEMO's EXACT zdfmxl N^2-integral criterion.

    NEMO (``zdfmxl.F90:91-105``, 5.0.2) integrates the POSITIVE buoyancy
    frequency down from the ~10 m reference w-level ``nlb10`` and sets the
    mixed-layer w-level ``nmln`` at the shallowest level where the running
    integral first reaches an N^2 threshold::

        zN2_c   = grav * rho_c * r1_rho0                 (:95)
        hmlp   += MAX( rn2b(jk), 0 ) * e3w(jk)           (:98, jk>=nlb10)
        nmln    = shallowest jk with hmlp >= zN2_c       (:99)
        hmlp    = gdepw(nmln)                            (:104, the MLD)

    with ``rho_c = 0.01 kg/m^3`` (NEMO ``rho_c``) and ``r1_rho0 = 1/rho0``.
    This differs from the potential-density-difference sibling
    (:func:`_nemo_mld_from_potential_density`) by (a) using the in-situ
    locally-referenced (adiabatic) N^2 — NEMO ``rn2b`` from ``eosbn2``, i.e.
    the SAME ``compute_buoyancy_frequency_adiabatic`` the native ldfslp slopes
    already consume as ``pn2`` — and (b) the ``MAX(N^2, 0)`` clamp, which
    ignores statically-unstable inversions instead of letting a negative
    density step cancel the accumulated difference (so a convecting column
    mixes to its base, not to the first inversion).

    Returns ``(hml, m_base)`` with the SAME convention as
    :func:`_nemo_mld_from_potential_density` (drop-in interchangeable in the
    ramps): ``hml`` = ML-base w-interface depth [m, +], ``m_base`` the legoESM
    *interface* index (0..nlev-2) of that base.  AD-safe; ``m_base`` is
    index-selected (a quantized step of T/S, zero gradient through the ramp
    normaliser) exactly like the pot-density sibling.
    """
    from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
    # 3-D fill mask when available: sub-seafloor dry cells CARRY their IC
    # (MLF carry fix) and must never enter the MLD density/N^2 — NEMO gates
    # on 3-D tmask (zdfmxl/eosbn2); the 2-D surface mask leaks carried deep
    # values into the ML ramp at bathymetry steps (#1226 boundary-column
    # T anomaly, poison-probe verified).
    _fillm = mask if active_3d is None else active_3d
    T_filled = neumann_fill_cgrid(T, _fillm)
    S_filled = neumann_fill_cgrid(S, _fillm)
    nlev = T_filled.shape[-1]
    dtype = T_filled.dtype
    dz_ref = z_coord.dz_ref
    z_iface = jnp.cumsum(dz_ref)                      # (nlev,) bottom-of-cell depths
    z_centers = z_iface - 0.5 * dz_ref               # (nlev,) gdept
    # rn2b: adiabatic N^2 at the nlev-1 interior w-interfaces, referenced to the
    # UPPER cell pressure (bit-identical to the native-slope pn2 build, :580).
    p_cell = (jnp.asarray(rho_0, dtype) * jnp.asarray(g, dtype)
              * z_centers)[None, None, :] * jnp.ones_like(T_filled)
    J1 = jnp.ones(T_filled.shape[:-1], dtype=dtype)
    # NEMO integrates rn2b, i.e. the LINEARISED alpha/beta bn2 (eosbn2.F90),
    # not a parcel-displacement N^2.  Verified against NEMO's own dumped rn2b
    # on the DINO y5 state (#1226): compute_buoyancy_frequency_nemo_bn2 gives
    # corr 1.000000 / ratio 0.999970, the adiabatic form 0.999999 / 0.995994.
    # Under a 1.0 fidelity bar that 0.4% amplitude deviation moves the MLD
    # level in ~3% of columns, so use the NEMO form whenever the card runs the
    # S-EOS; fall back to the adiabatic N^2 for any other EOS (nemo_bn2 is
    # S-EOS-specific, being written in terms of the alpha/beta polynomial).
    _use_nemo_bn2 = getattr(z_coord, "t_depth_ref", None) is not None
    if _use_nemo_bn2:
        from legoesm.ocean.eos import (
            compute_buoyancy_frequency_nemo_bn2, nemo_e3w_from_live_gdept,
            NemoSEOSConfig,
        )
        _gdept = jnp.asarray(z_coord.t_depth_ref, dtype=dtype)
        # NEMO evaluates alpha/beta at the LIVE gdept(Kmm) = gdept_0*(1+r3t)
        # (eos_rab is called on the live grid).  The zrw weight is a RATIO of
        # depth differences, hence invariant under the column-uniform stretch;
        # only the alpha/beta pressure argument changes.  The 13 residual
        # knife-edge MLD columns sit ~7e-4 from threshold, where alpha's
        # ~1e-4 live-vs-static depth sensitivity has leverage (#1226).
        # gdepw = the TRUE w-interface depths (z_iface), NOT the midpoint of the
        # bracketing T-depths.  NEMO's zrw weight (eosbn2.F90:1459) is
        #     zrw = ( gdepw(k) - gdept(k) ) / ( gdept(k-1) - gdept(k) )
        # which is only 1/2 on a uniform ladder; on DINO's stretched grid gdept
        # is NOT centred between its interfaces, so the midpoint biases the
        # alpha/beta interpolation and puts ~4e-4 median error into N^2.
        _gdepw_int = z_iface[:-1]
        # #1226 bn2 live-e3w divisor (eosbn2.F90:1467, pn2 = ... / e3w(Kmm)):
        # NEMO's e3w(Kmm) = e3w_0*(1+r3t) under key_qco.  ``jacobian`` IS
        # (1+r3t) = (eta+H_bathy)/H_bathy ONLY on an OceanPartialCellCoordinate
        # -- on a pure z* coordinate it is (eta+H_bathy)/H_max instead (a
        # DIFFERENT quantity, compute_ocean_jacobian's own docstring), so
        # stretching by it there would be a mistranscription, not a fix.
        # Matches the identical gate in _nemo_wpoint_e3w_wmask_n2 above
        # (:787-788) -- same physical quantity, same guard, no new config
        # surface.
        if jacobian is not None and isinstance(z_coord, OceanPartialCellCoordinate):
            _J = jnp.asarray(jacobian, dtype)[..., None]     # (nlat,nlon,1)
            _gdept = _gdept[None, None, :] * _J
            _gdepw_int = _gdepw_int[None, None, :] * _J
            _stretch = jnp.asarray(jacobian, dtype)
        else:
            _stretch = None
        e3w = nemo_e3w_from_live_gdept(
            z_coord, _gdept, stretch=_stretch, interior=True)
        # A carried rn2b is already the compiled eosbn2 result.  Do not
        # evaluate and then discard a second local eosbn2 call: besides being
        # the wrong stage program, that dead call rejects DINO's dry/halo raw
        # e3w slots before the recorded operand can replace it.
        if n2_override is None:
            n2_int = compute_buoyancy_frequency_nemo_bn2(
                T_filled, S_filled, _gdept, _gdepw_int,
                (eos_nemo_seos if eos_nemo_seos is not None
                 else NemoSEOSConfig()), g=g,
                e3w_int=e3w)
    else:
        if n2_override is None:
            n2_int = compute_buoyancy_frequency_adiabatic(
                T_filled, S_filled, p_cell, dz_ref, J1, eos_fn=eos_fn,
                rho_ref=rho_0, g=g)                   # (...,nlev-1)
    # e3w(jk) for interface m = spacing between the bracketing T-centres.
    # It MUST be built from the SAME gdept ladder the N^2 was divided by, or the
    # exact e3w cancellation below is broken.  The faithful bn2 branch divides
    # by raw-mesh e3w_0*(1+r3t); the explicit legacy branch derives spacing
    # from t_depth_ref.  In either case this local ``e3w`` is the exact operand
    # selected above.  Rebuilding the multiplier from the arithmetic-midpoint
    # ladder cumsum(dz)-dz/2 left a residual and produced 14 mismatched DINO
    # MLD columns whose below-threshold decisions otherwise matched NEMO.
    if not _use_nemo_bn2:
        e3w = z_centers[1:] - z_centers[:-1]         # (nlev-1,)
    if n2_override is not None:
        n2_int = jnp.asarray(n2_override, dtype=dtype)
        if n2_int.shape[-1] == nlev:
            n2_int = n2_int[..., 1:]
        elif n2_int.shape[-1] != nlev - 1:
            raise ValueError(
                "n2_override must contain nlev or nlev-1 W levels, got "
                f"{n2_int.shape[-1]} for nlev={nlev}")
    if e3w_override is not None:
        e3w = jnp.asarray(e3w_override, dtype=dtype)
        if e3w.shape[-1] == nlev:
            e3w = e3w[..., 1:]
        elif e3w.shape[-1] != nlev - 1:
            raise ValueError(
                "e3w_override must contain nlev or nlev-1 W levels, got "
                f"{e3w.shape[-1]} for nlev={nlev}")
    # The MLD CRITERION is thickness-free, and that is not an approximation --
    # it is an identity in NEMO.  eosbn2 divides by the live e3w and zdfmxl
    # multiplies it straight back:
    #     rn2b(jk) = grav*( alpha*dT - beta*dS ) / e3w(jk,Kmm)   (eosbn2.F90:1467)
    #     hmlp    += MAX( rn2b(jk), 0 ) * e3w(jk,Kmm)            (zdfmxl.F90:98)
    #  => term    = MAX( grav*( alpha*dT - beta*dS ), 0 )
    # so e3w cancels EXACTLY and NO z-star stretch belongs in the integral
    # (measured: applying one flips zero levels).  Reference spacing is correct
    # here as long as n2_int is built on that SAME spacing, which it is.
    #
    # The DEPTH is a different matter: NEMO's hmlp = gdepw(nmln,Kmm) IS the live
    # w-depth and does carry (1+r3t), r3t=ssh/ht_0 (domqco.F90:160,
    # domzgr_substitute.h90:131,140 under DINO's key_qco+key_vco_3d build:
    # gdepw(i,j,k,t) = gdepw_3d(i,j,k)*(1+r3t(i,j,t)), no key_isf).  #1226
    # queue item 5 FIX: apply it.  ``jacobian`` (threaded into this function
    # since the live-e3w fix above) already IS that exact (1+r3t) factor on
    # an OceanPartialCellCoordinate -- ht_0 there is H_bathy (vertical.py
    # compute_ocean_jacobian: J=(eta+H_bathy)/H_bathy=1+eta/H_bathy=1+r3t) --
    # so no new quantity/config surface, just consuming what already flows
    # through.  Gated identically to the sibling live-e3w branches above
    # (:514, :796): a plain z*-coordinate's jacobian is (eta+H_bathy)/H_max,
    # a DIFFERENT quantity, so stretching hml by it there would be a
    # mistranscription. CONFIRMED against NEMO's own dumped hmlp on the DINO
    # y5 state (this function's own prior note, kept for provenance):
    # gdepw_0[nmln-1]*(1+ssh/H) reproduces it to max err 0.0 m over 9920 wet
    # columns -- this is exactly that substitution, now applied in the
    # return path instead of only in a verification note.
    # Reference w-level nlb10 = first interface at/below ~10 m; contributions
    # above it are excluded (the near-surface is mixed by definition), so the
    # MLD is floored at ~10 m exactly as NEMO's nmln>=nlb10 initialisation.
    m_arange = jnp.arange(nlev - 1)
    iref = jnp.clip(
        jnp.searchsorted(z_iface, jnp.asarray(_NEMO_MLD_REF_DEPTH_M, z_iface.dtype)),
        0, nlev - 2)
    contrib = jnp.where(
        (m_arange < iref).reshape((1, 1, nlev - 1)),
        jnp.zeros((), dtype),
        jnp.maximum(n2_int, jnp.zeros((), dtype))
        * jnp.broadcast_to(e3w, n2_int.shape))
    if active_3d is not None:
        # NEMO zdfmxl integrates nlb10..BOTTOM only (hard loop truncation on
        # 3-D tmask): zero the integrand at sub-seafloor interfaces. The
        # Neumann fill covers the FIRST dry cell at a step; deeper dry cells
        # (no wet lateral neighbour at that level) keep carried values whose
        # spurious N2 would otherwise pollute the cumulative integral
        # (#1226 boundary-column T anomaly, poison-gate verified).
        iface_wet = (jnp.asarray(active_3d)[..., :-1] > 0.5) \
            & (jnp.asarray(active_3d)[..., 1:] > 0.5)
        contrib = jnp.where(iface_wet, contrib, jnp.zeros((), dtype))
    cum = jnp.cumsum(contrib, axis=-1)               # integral(N^2 dz) from nlb10
    thresh = jnp.asarray(g * rho_c / rho_0, dtype)   # zN2_c = g*rho_c/rho0
    # NEMO (zdfmxl.F90:96-101) does NOT record the crossing level directly; it
    # advances nmln on every level that is STILL below threshold, bottom-capped:
    #
    #     DO jk = nlb10, jpkm1
    #        hmlp += MAX( rn2b(jk), 0 ) * e3w(jk,Kmm)
    #        IF( hmlp < zN2_c )   nmln = MIN( jk, mbkt ) + 1
    #
    # so nmln = MIN( last jk with cum < zN2_c , mbkt ) + 1.  A column that never
    # reaches the threshold therefore ends at mbkt+1 -- the ML reaches the
    # SEAFLOOR -- not at the deepest interface.  Omitting that cap put 339 of
    # DINO's 9920 wet columns 1-5 levels too deep (#1226, 96% of all the level
    # mismatches).  The integrand is MAX(N^2,0)*e3w >= 0, so cum is monotone and
    # {cum < zN2_c} is a PREFIX: that "last jk" is just the count of
    # sub-threshold levels, which vectorises without a scan.
    # NEMO's loop stops at jpkm1, i.e. lego interfaces m <= nlev-3 (jk = m+2).
    below = (cum[..., :nlev - 2] < thresh).astype(jnp.int32).sum(axis=-1)
    if active_3d is not None:
        # mbkt = deepest wet T-level (1-based) = count of wet cells in column.
        # NEMO guarantees mbkt <= jpkm1 (the bottom T-level is ALWAYS land,
        # tmask(:,:,jpk)=0), so nmln = mbkt+1 <= jpk and m_base <= nlev-2 -- the
        # clip below never binds on a NEMO-bridged column.  A hypothetical
        # all-wet column (mbkt = nlev) would want nmln = nlev+1, i.e. a w-level
        # BELOW the deepest interior interface, which this interface indexing
        # (m = 0..nlev-2) cannot represent; it saturates at the deepest
        # interface instead.  That is a representational limit, not a NEMO
        # mismatch, and it is unreachable for any oracle config.
        mbkt = (jnp.asarray(active_3d) > 0.5).astype(jnp.int32).sum(axis=-1)
    else:
        # No 3-D mask: assume NEMO's own invariant (deepest T-level is land)
        # rather than nlev, which would imply an all-wet column and hit the
        # saturation described above.
        mbkt = jnp.full(below.shape, nlev - 1, dtype=jnp.int32)
    # nmln = MIN(nlb10-1+below, mbkt) + 1 with nlb10 = iref+2, and the lego
    # interface index is m_base = nmln - 2 (verified convention, #1226).
    m_base = jnp.clip(jnp.minimum(1 + below, mbkt) - 1, 0, nlev - 2)
    # hmlp = gdepw(nmln,Kmm) (zdfmxl.F90:104).  z_iface[m] = gdepw_0(m+2) and
    # nmln = m_base+2, so this is gdepw_0(nmln).  Convention CERTIFIED against
    # NEMO's own dumped hmlp on the DINO y5 state: gdepw_0[nmln-1]*(1+ssh/H)
    # reproduces it to max err 0.0 m over 9920 wet columns.
    hml = jnp.take(z_iface, m_base)                   # (n_lat, n_lon), gdepw_0(nmln)
    if jacobian is not None and isinstance(z_coord, OceanPartialCellCoordinate):
        # Live-depth stretch (1+r3t) -- see the comment block above this
        # function's N^2-integral live-e3w gate for the full derivation;
        # same gate, same jacobian, no new quantity.
        hml = hml * jnp.asarray(jacobian, dtype)
    return hml, m_base


def _nemo_mld(criterion, T, S, mask, z_coord, eos_fn, rho_c, *,
              g=constants.g, rho_0=_RHO_0, active_3d=None, jacobian=None,
              n2_override=None, e3w_override=None, eos_nemo_seos=None):
    """Dispatch the NEMO zdfmxl mixed-layer depth by criterion (raise on typo).

    ``"rho_c"`` (default, byte-identical) = potential-density difference;
    ``"n2_integral"`` = NEMO's exact integral(N^2 dz) >= g*rho_c/rho0 criterion.
    Both return ``(hml, m_base)`` in the same convention, so the ldfslp slope
    ramps consume either transparently.
    """
    if criterion == "rho_c":
        return _nemo_mld_from_potential_density(
            T, S, mask, z_coord, eos_fn, rho_c, active_3d=active_3d)
    if criterion == "n2_integral":
        return _nemo_mld_from_n2_integral(
            T, S, mask, z_coord, eos_fn, rho_c, g, rho_0, active_3d=active_3d,
            jacobian=jacobian, n2_override=n2_override,
            e3w_override=e3w_override, eos_nemo_seos=eos_nemo_seos)
    raise ValueError(
        f"unknown GMRediConfig.mld_criterion {criterion!r}; "
        "expected 'rho_c' or 'n2_integral'.")


def _apply_nemo_mld_slope_ramp(S_x, S_y, T, S, mask, z_coord, eos_fn, rho_c,
                               mld_criterion="rho_c", *, active_3d=None,
                               g=constants.g, rho_0=_RHO_0, jacobian=None):
    """Linearly ramp interface slopes to 0 through the mixed layer (NEMO ldfslp).

    NEMO (``ldfslp.F90:284-297``, w-point branch): inside the mixed layer
    (``jk <= nmln``) the neutral slope is REPLACED by a linear profile

        wslp(k) = gdepw(k) / MAX(hmlp, 10) * wslp_base

    where ``wslp_base = wslp(nmln+1)`` is the computed slope at the first w-level
    BELOW the ML base (bounded and well-stratified — NOT the ML-base value, which
    still carries the surface weak-stratification blow-up) and ``hmlp`` the
    mixed-layer depth — a straight line from the below-ML slope down to 0 at the
    surface.  Below the ML the slope is unchanged.  (NEMO's ``wmask`` zeroes the
    ocean-floor w-face, ``ldfslp.F90:302-315``, but that face is NOT an element of
    this interior-interface array — legoESM carries the ``nlev-1`` interior
    interfaces only — so there is no bottom face to zero here.)

    Applied to the FINAL (tapered/clipped) legoESM slopes, so the interior /
    below-ML numerics — which already match NEMO — are untouched; only the
    mixed-layer interfaces are overwritten by the ramp.  ``rho_c`` [kg/m^3] is
    the MLD density criterion.  JIT/AD-safe (clip + where + take_along_axis).

    #1455 queue item 5: NEMO's ``wslp(k) = gdepw(k) / MAX(hmlp,10) * wslp_base``
    (ldfslp.F90:284-297) uses the LIVE ``gdepw(k)`` for BOTH the numerator
    profile depth and (via ``hmlp``) the denominator — the same live
    ``(1+r3t)`` stretch (domzgr_substitute.h90:131,140) on both.  ``_nemo_mld``
    now returns a LIVE-stretched ``hml`` on an ``OceanPartialCellCoordinate``
    (see ``_nemo_mld_from_n2_integral``'s own fix), so the profile depth
    ``z_iface`` used for the ``ramp``/``in_ml`` comparison here must carry the
    SAME stretch or the two sides of ``z_iface <= hml`` compare a static
    quantity against a live one — gated identically (isinstance check, no new
    config surface).
    """
    nlev_m1 = S_x.shape[-1]
    hml, m_base = _nemo_mld(
        mld_criterion, T, S, mask, z_coord, eos_fn, rho_c, g=g, rho_0=rho_0,
        active_3d=active_3d, jacobian=jacobian)
    z_iface = jnp.cumsum(z_coord.dz_ref)[:-1]         # (nlev-1,) interface depths
    z_iface_3d = z_iface[None, None, :]               # (1, 1, nlev-1), broadcastable
    if jacobian is not None and isinstance(z_coord, OceanPartialCellCoordinate):
        # (n_lat, n_lon, nlev-1) live-stretched profile depth.
        z_iface_3d = z_iface_3d * jnp.asarray(jacobian, dtype=z_iface.dtype)[:, :, None]
    # wslp_base = slope one interface BELOW the ML base (NEMO nmln+1).
    m_ref = jnp.clip(m_base + 1, 0, nlev_m1 - 1)
    Sx_base = jnp.take_along_axis(S_x, m_ref[:, :, None], axis=-1)  # (n_lat,n_lon,1)
    Sy_base = jnp.take_along_axis(S_y, m_ref[:, :, None], axis=-1)
    ramp = z_iface_3d / jnp.maximum(
        hml[:, :, None], _NEMO_HMLW_FLOOR_M)          # (n_lat, n_lon, nlev-1)
    in_ml = z_iface_3d <= hml[:, :, None]
    S_x = jnp.where(in_ml, ramp * Sx_base, S_x)
    S_y = jnp.where(in_ml, ramp * Sy_base, S_y)
    return S_x, S_y


def _shapiro_smooth_slopes(S_x, S_y, mask):
    """NEMO ldfslp horizontal Shapiro filter + coastal taper (ldfslp.F90:301-315).

    NEMO smooths the (already mixed-layer-flattened) interface slopes with a
    ``(1-2-1)⊗(1-2-1)`` nine-point binomial, divides by the fixed weight-sum 16,
    then multiplies by a coastal taper ``zcofw`` that SHRINKS the slope toward
    land — it is NOT a wet-renormalization.  Land neighbours enter the binomial
    sum as the masked zero they already are (``zwz = (...) * wmask``), and the
    fixed ``/16`` divisor plus ``zcofw`` make the slope decay near coasts:

        zcofw = tmask/16 · (uE + uW) · (vN + vS) · 0.25   (= tmask/16 in interior)
        wslp  = [ 9-pt binomial SUM of masked slope ] · zcofw

    The C-grid face masks are the products of adjacent T-mask cells (= NEMO
    ``umask``/``vmask`` for a flat-bottom GYRE/DINO domain).  Applied to the FINAL
    slopes after the ML ramp (NEMO's per-level order).  JIT/AD-safe: fixed-weight
    pad + weighted sum, no data-dependent control flow.
    """
    nlat, nlon = mask.shape
    m = mask[:, :, None]                                  # (n_lat, n_lon, 1)
    # C-grid face masks from the T-mask: a face is wet iff both bracketing
    # cells are wet (NEMO umask/vmask = product of adjacent tmask).
    # Lon neighbours are PERIODIC (roll): NEMO's smoother reads zwz/zww that
    # the slope loops computed over the HALO columns too (DO_2D(1,1,1,1),
    # ldfslp.F90:203,265) from lbc-filled inputs, so seam columns see true wrap
    # neighbours.  Zero-filling treated the seam as a wall (#1226 seam-column
    # slope deficit).  Lat stays zero-filled: closed in j.  INVARIANT: a domain
    # with a CLOSED lon boundary must carry land at the i-edge columns (all
    # current consumers do) -- a wet closed lon edge would wrap spuriously
    # here, as it already would in every roll stencil of this module.
    wE = mask * jnp.roll(mask, -1, axis=1)                # wet(i,j) & wet(i,j+1)
    wW = mask * jnp.roll(mask, +1, axis=1)                # wet(i,j) & wet(i,j-1)
    wN = mask * jnp.pad(mask, ((0, 1), (0, 0)))[1:, :]    # wet(i,j) & wet(i+1,j)
    wS = mask * jnp.pad(mask, ((1, 0), (0, 0)))[:-1, :]   # wet(i,j) & wet(i-1,j)
    zcofw = (m / 16.0) * (wE + wW)[:, :, None] * (wN + wS)[:, :, None] * 0.25  # coeff-ok: 16 = (1+2+1)^2 binomial weight sum (NEMO ldfslp z1_16)

    w = (1.0, 2.0, 1.0)                                   # 1-D binomial kernel
    def smooth(f):
        # Periodic lon ghost, zero lat ghost -- same seam fix as _shap above.
        fm = f * m                                        # masked
        fp = jnp.pad(fm, ((0, 0), (1, 1), (0, 0)), mode="wrap")
        fp = jnp.pad(fp, ((1, 1), (0, 0), (0, 0)))        # zero lat ghost
        acc = jnp.zeros_like(f)
        for a in range(3):
            for b in range(3):
                acc = acc + w[a] * w[b] * fp[a:a + nlat, b:b + nlon, :]
        return acc * zcofw

    return smooth(S_x), smooth(S_y)


# =====================================================================
# Isopycnal slope computation
# =====================================================================

def _nemo_wpoint_e3w_wmask_n2(rho, T, S, z_coord, eos_fn, rho_0, g, act,
                              slope_n2="adiabatic", jacobian=None,
                              eos_nemo_seos=None):
    """Shared W-point geometry + N² for the native ldfslp stencil.

    Factors the ``e3w``/``wmask3``/``pn2`` block common to
    :func:`compute_nemo_native_slopes` (the Redi/GM slope builder) and
    :func:`compute_treguier_kappa_gm_nemo_native` (NEMO's ``ldf_eiv``
    adaptive-κ, ``ldftra.F90:664-698``, which consumes the SAME ``wslpi``/
    ``wslpj``/``rn2b``/``e3w`` this module builds) — one N²/geometry
    construction, no duplicate numerics.

    Returns ``(e3w, wmask3, pn2)``: ``e3w`` (nlev,) NEMO w-level thickness;
    ``wmask3`` (n_lat,n_lon,nlev) W-point wet mask (``wmask3[...,0]`` = the
    surface T-mask, matching NEMO's ``wmask(:,:,1)=tmask(:,:,1)``); ``pn2``
    (n_lat,n_lon,nlev) locally-referenced (adiabatic) N² at W-points, NEMO
    indexing (``pn2[k]`` = top of cell k, ``pn2[0]=0``), masked by ``wmask3``
    at construction (eosbn2.F90:1467 convention — see the #1226 poison-gate
    note in :func:`compute_nemo_native_slopes`).
    """
    dtype = rho.dtype
    nlat, nlon, nlev = rho.shape
    dz = jnp.asarray(z_coord.dz_ref, dtype=dtype)
    # T-point depth: prefer the coordinate's OWN t_depth_ref, which the NEMO
    # bridge populates from NEMO's gdept_0.  Deriving it as the arithmetic
    # midpoint cumsum(dz)-dz/2 is NOT what NEMO does: gdept_0 is the ANALYTIC
    # mid-depth and differs from the arithmetic midpoint by up to 11.3 m on the
    # DINO ladder (#1226).  That error feeds e3w, and e3w weights every column
    # sum in ldf_eiv (zn, zah, zhw) -- so a wrong gdept mis-scales kappa_GM.
    _td = getattr(z_coord, "t_depth_ref", None)
    gdept = (jnp.asarray(_td, dtype=dtype) if _td is not None
             else jnp.cumsum(dz) - 0.5 * dz)
    # NEMO depth_to_e3: e3w(1) = 2*gdept(1); e3w(k) = gdept(k) - gdept(k-1).
    # Bit-identical ONLY for coordinates carrying no t_depth_ref (there the
    # arithmetic-midpoint fallback gives 2*gdept(1) == dz(1) identically).  On a
    # NEMO-bridged coordinate this DOES change answers, by design: measured on
    # the DINO twin, day-10 max|u| 0.6027 -> 0.6036 -- the old ladder's
    # gdept_1d is not the arithmetic midpoint either (5.28 m apart), so the
    # derived form was wrong in BOTH modes.
    from legoesm.ocean.eos import nemo_e3w_from_live_gdept
    # #1226 blocker 1: e3w(Kmm) = e3w_0*(1+r3t) is LIVE everywhere ldfslp.F90 /
    # ldftra.F90 read it (the :131 Time() macro applies to e3w unconditionally,
    # not just the rn2b division above) -- the slope-stability bound
    # (-7e3/e3w, ldfslp.F90:281-282) and the ldf_eiv column sums (zn/zah/zhw,
    # ldftra.F90:689,694-696) both consume this SAME e3w.  ``jacobian`` IS
    # (1+r3t) ONLY on an OceanPartialCellCoordinate -- see the matching gate
    # in compute_nemo_native_slopes above (a pure z* coordinate's jacobian is
    # (eta+H_bathy)/H_max, a different quantity).  jacobian=None or z*
    # coordinate -> stretch=1, BIT-IDENTICAL.
    _live_e3w = (jacobian is not None
                 and isinstance(z_coord, OceanPartialCellCoordinate))
    if _live_e3w:
        _stretch = jnp.asarray(jacobian, dtype)
        _gdept_live = gdept[None, None, :] * _stretch[..., None]
    else:
        _stretch = None
        _gdept_live = gdept
    e3w = nemo_e3w_from_live_gdept(
        z_coord, _gdept_live, stretch=_stretch, interior=False)

    wmask3 = act * jnp.roll(act, +1, axis=2)
    wmask3 = wmask3.at[:, :, 0].set(act[:, :, 0])

    # NEMO's ldf_slp consumes rn2b -- the LINEARISED alpha/beta bn2 of
    # eosbn2.F90:1455-1468, NOT a parcel-displacement N^2.  The two diverge with
    # pressure, so the adiabatic form biases the slopes progressively at depth.
    # Selectable so non-oracle recipes stay bit-identical (GMRediConfig.slope_n2).
    if slope_n2 == "nemo_bn2":
        from legoesm.ocean.eos import (
            compute_buoyancy_frequency_nemo_bn2, NemoSEOSConfig,
        )
        # zrw (eosbn2.F90:1459) weights the two T-point alpha/beta by the TRUE
        # w-interface depth gdepw, which is the gdept midpoint only on a uniform
        # ladder -- see the same fix in _nemo_mld_from_n2_integral.
        _gdepw_int = jnp.cumsum(dz)[:-1]
        # #1226: evaluate alpha/beta at the LIVE gdept(Kmm) = gdept_0*(1+r3t),
        # exactly as eosbn2.F90:1166/:1459 does -- NOT at the static ladder.
        # ``jacobian`` IS (1+r3t) under the _live_e3w gate above, so the live
        # ladders come for free without widening this signature; the same
        # pattern is already used in _nemo_mld_from_n2_integral.
        # Stretching BOTH gdept and gdepw keeps the zrw weight (a RATIO of
        # depth differences) invariant, and makes the e3w that
        # compute_buoyancy_frequency_nemo_bn2 derives internally as
        # diff(gdept) the LIVE e3w -- which is why the explicit
        # ``/ jacobian`` correction below is dropped WITH this change rather
        # than kept alongside it: keeping both would apply (1+r3t) twice.
        if _live_e3w:
            _Jn2 = jnp.asarray(jacobian, dtype)[..., None]
            _gdept_n2 = gdept[None, None, :] * _Jn2
            _gdepw_n2 = _gdepw_int[None, None, :] * _Jn2
        else:
            _gdept_n2, _gdepw_n2 = gdept, _gdepw_int
        n2_int = compute_buoyancy_frequency_nemo_bn2(
            T, S, _gdept_n2, _gdepw_n2,
            (eos_nemo_seos if eos_nemo_seos is not None
             else NemoSEOSConfig()), g=g,
            e3w_int=e3w[..., 1:])                           # (...,nlev-1)
        # HISTORICAL (superseded 2026-07-28, kept for provenance):
        # this branch used to divide n2_int by the jacobian --
        #     NEMO divides by the LIVE e3w(jk,Kmm) = e3w_0*(1+r3t)
        # (domzgr_substitute.h90:131); on an OceanPartialCellCoordinate
        # the (eta+H_bathy)/H_bathy Jacobian IS that (1+r3t) (see the
        # gate above -- NOT true on a pure z* coordinate, whose Jacobian
        # is (eta+H_bathy)/H_max instead).  Without this the reference
        # e3w leaves a ~1e-4 bias.  It does NOT cancel here (unlike in
        # the thickness-free MLD criterion).  Measured on the DINO y5
        # twin vs NEMO's dumped rn2b, with NEMO's g: median |rel|
        # 8.59e-05 -> 6.96e-06.
        #
        # SUPERSEDED 2026-07-28: this scalar correction fixed only the e3w
        # DENOMINATOR while alpha/beta stayed at STATIC depths, leaving
        # pn2 at err_norm 3.460e-07 -- quantitatively the whole of zbw's
        # 3.467e-07 floor.  The live ladders above now carry (1+r3t) into
        # BOTH the alpha/beta depths and the internal diff(gdept) e3w, so
        # this division would double-count.  Measured: zbw 3.467e-07 ->
        # 9.369e-16, statistically identical to substituting NEMO's own
        # dumped rn2b (9.304e-16).
    elif slope_n2 == "adiabatic":
        from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
        p_cell = (jnp.asarray(rho_0, dtype) * jnp.asarray(g, dtype)
                  * gdept)[None, None, :] * jnp.ones_like(rho)
        J1 = jnp.ones((nlat, nlon), dtype=dtype)
        n2_int = compute_buoyancy_frequency_adiabatic(
            T, S, p_cell, z_coord.dz_ref, J1, eos_fn=eos_fn,
            # SAME pair the pressure three lines above was built from. Left to
            # the library defaults these disagree with it on any card that
            # pins its own constants (#1627).
            rho_ref=rho_0, g=g)                                   # (...,nlev-1)
    else:
        raise ValueError(
            f"unknown GMRediConfig.slope_n2 {slope_n2!r}; "
            "expected 'adiabatic' or 'nemo_bn2'.")
    pn2 = jnp.concatenate([jnp.zeros((nlat, nlon, 1), dtype=dtype),
                           n2_int.astype(dtype)], axis=-1)        # (...,nlev)
    pn2 = pn2 * wmask3
    return e3w, wmask3, pn2


def _nemo_ml_anchor_index(first: jnp.ndarray, act: jnp.ndarray,
                          nlev: int) -> jnp.ndarray:
    """The w-slope ML-ramp anchor index (ldfslp.F90 ``jk = nmln + 1``).

    ``first`` is the 0-based ``nmln`` (see :func:`compute_nemo_native_slopes`
    docstring); the anchor reads the interior slope one w-level BELOW the
    mixed-layer base.  NEMO's own ``nmln`` is bottom-capped per column
    (zdfmxl.F90:99, ``nmln = MIN(jk,mbkt) + 1``), so the anchor ``nmln+1`` can
    reach ``mbkt+2`` -- past the last WET w-level (``wmask=0`` there) when the
    ML reaches the seafloor.  That's deliberate: NEMO's ``zwslpi_hml``
    recurrence stores the (masked-zero) value at that dry level as the
    "anchor" for the whole ramp in that column.

    A prior version clamped only against the GLOBAL array size
    (``jnp.clip(first + 1, 1, nlev - 1)``), with no PER-COLUMN ceiling. That
    is fine wherever the column reaches deeper than ``nlev - 3`` (the common
    case for the DINO oracle grid, where level ``nlev - 1`` is a universal
    dry sentinel every column shares), but for a genuinely shallow column
    sitting in a domain with much deeper neighbours, an unstratified
    (never-crosses-MLD-threshold) shallow column's ``first`` saturates at the
    GLOBAL ``nlev - 1`` regardless of that column's own bottom -- i.e. the
    old clamp silently assumed every column shares one dry bottom level, an
    assumption real (non-uniform-depth) bathymetry does not satisfy (#1226:
    442/9920 DINO wet columns read past their own ``bottom_wet_k`` under the
    old formula, though inert there since DINO happens to carry a universal
    dry sentinel level).  Clamp against the column's own bottom (``mbkt`` =
    count of active T-cells) instead, matching NEMO's per-column
    ``MIN(...,mbkt)`` semantics exactly.
    """
    mbkt = jnp.sum(act > 0.5, axis=-1).astype(jnp.int32)          # (nlat,nlon)
    return jnp.clip(first + 1, 1, jnp.minimum(mbkt + 1, nlev - 1))


def _nemo_qco_live_slope_face_thicknesses(
    eta: jnp.ndarray,
    z_coord: OceanPartialCellCoordinate,
    e3u_0: jnp.ndarray,
    e3v_0: jnp.ndarray,
    umask3: jnp.ndarray,
    vmask3: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compatibility wrapper around the shared QCO operand builder."""
    return nemo_qco_live_face_thicknesses(
        eta, z_coord, e3u_0, e3v_0, umask3, vmask3,
    )


def _nemo_qco_live_slope_depths(
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    z_coord: OceanPartialCellCoordinate,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Raw NEMO T/W depth ladders times NOW ``1 + ssh*r1_ht_0``."""
    gdept_0 = getattr(z_coord, "nemo_gdept_0", None)
    gdepw_0 = getattr(z_coord, "nemo_gdepw_0", None)
    if gdept_0 is None or gdepw_0 is None:
        raise ValueError(
            "slope_depth_evaluation='nemo_qco_live_literal' requires raw "
            "NEMO gdept_0 and gdepw_0 fields")
    stretch = nemo_r3t_stretch(
        z_coord, jnp.asarray(eta, dtype=dtype),
        jnp.asarray(H_bathy, dtype=dtype), evaluation="nemo_reciprocal")
    stretch = lax.optimization_barrier(stretch)
    return (
        lax.optimization_barrier(
            jnp.asarray(gdept_0, dtype=dtype) * stretch[..., None]),
        lax.optimization_barrier(
            jnp.asarray(gdepw_0, dtype=dtype) * stretch[..., None]),
        stretch,
    )


def _nemo_literal_slope_face_depth(
    gdept_live: jnp.ndarray,
    e3face_live: jnp.ndarray,
    *,
    axis: int,
) -> jnp.ndarray:
    """DINO no-ice-shelf ldfslp ``zdep{u,v}`` source association."""
    dtype = gdept_live.dtype
    pair = lax.optimization_barrier(
        gdept_live + jnp.roll(gdept_live, -1, axis=axis))
    inner = lax.optimization_barrier(pair - e3face_live[..., :1])
    return lax.optimization_barrier(jnp.asarray(0.5, dtype=dtype) * inner)


def compute_nemo_native_slopes(
    rho: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    grid: LatLonGrid,
    cfg: GMRediConfig,
    eos_fn,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    active_3d: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    eta: jnp.ndarray | None = None,
    H_bathy: jnp.ndarray | None = None,
    prd_jacobian: jnp.ndarray | None = None,
    prd_TS_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    prd_override: jnp.ndarray | None = None,
    pn2_override: jnp.ndarray | None = None,
    e3w_override: jnp.ndarray | None = None, return_diagnostics: bool = False,
    nmln_override: jnp.ndarray | None = None, eos_nemo_seos=None,
):
    """NEMO ldfslp native four-position isopycnal slopes (uslp, vslp, wslpi,
    wslpj) — a direct transcription of ``ldfslp.F90`` (ldf_slp, NEMO 5.0.2)
    for the z-coordinate flat-bottom oracle configs.

    Conventions (NEMO, 0-based here):
      * ``uslp[j,i,k]``: slope at the EAST u-face of cell i, tracer level k;
        ``vslp`` the north v-face analogue.
      * ``wslpi/wslpj[j,i,k]``: slope at the w-point at the TOP of cell k
        (``wslp[...,0] = 0`` — the sea surface); this is exactly the indexing
        ``nemo_iso_lap_tracer_tendency_latlon_cgrid`` consumes natively
        (its below-cell-k flux uses ``roll(wslpi,-1)``).
      * SIGN: NEMO's — ``slp = zau/(zbu-eps)`` with the DENOMINATOR bounded
        NEGATIVE, i.e. ``slp = -dx(rho)/|drho_dz|``. No dispatch negation.

    Faithful pieces (line refs are ldfslp.F90):
      * zgru/zgrv masked horizontal rhd-gradients (:167-171, :180-184);
      * zdzr = -(1/g)(prd+1)(pn2(k)+pn2(k+1))(1-0.5 tmask(k+1)) (:188-195);
      * the DOUBLE slope bound via the denominator (:212-213, :281-282):
        |S| <= rn_slpmax AND |S| <= e3/7e3 (_NEMO_SLOPE_STAB_7E3);
      * mixed-layer linear ramp: uslp anchors to the interior slope AT the
        ML-base level iku=MAX(nmln_i, nmln_{i+1}) scaled by depth/hml_u
        (:216-238); wslp anchors at nmln+1 scaled by gdepw/hml_w (:285-297);
      * the w-point gradient uses WET-FACE-COUNT normalisation
        zci = MAX(sum of 4 umask, eps) (:271-278);
      * horizontal Shapiro 1/16 + coastal decrease, native mask factors
        (:242-257 u/v; :301-315 w).

    Scope: the current horizontally uniform DINO full-step grid represented by
    ``OceanPartialCellCoordinate``, with live z-star depths and selectable
    live-QCO face thicknesses on the two literal oracle cards.  Generic cards
    retain the historical static-face construction.  Partial-step and
    ice-shelf geometry remain out of scope (risfdep=0, mikt=1).
    """
    zeps = 1.0e-20
    dtype = rho.dtype
    nlat, nlon, nlev = rho.shape
    dz = jnp.asarray(z_coord.dz_ref, dtype=dtype)                # (nlev,)
    # T-point depth: prefer the coordinate's own t_depth_ref (NEMO gdept_0),
    # matching _nemo_wpoint_e3w_wmask_n2's ladder -- the arithmetic midpoint
    # differs from NEMO's analytic gdept_0 by up to 11.3 m on DINO (#1226).
    # gdepw_top (top-of-cell-k interface depth) is unaffected: NEMO derives it
    # from cumsum(e3t_0) (e3_to_depth_1d, depth_e3.F90:125-130), which for a
    # full-step config already equals cumsum(dz_ref) -- the reference ladder
    # legoESM already carries independent of t_depth_ref.
    _td = getattr(z_coord, "t_depth_ref", None)
    gdept = (jnp.asarray(_td, dtype=dtype) if _td is not None
             else jnp.cumsum(dz) - 0.5 * dz)                     # cell centres
    gdepw_top = jnp.cumsum(dz) - dz                              # top-of-cell depth
    # #1226 blocker 1: NEMO's ldf_slp (ldfslp.F90:143 zhmlpt, :226-229 zdepu/
    # zdepv, :289 zck) reads all THREE off the LIVE gdept(Kmm)/gdepw(Kmm) =
    # gdept_0*(1+r3t) / gdepw_0*(1+r3t) (domzgr_substitute.h90:131,139 Time()
    # macro -- a pure per-column multiplicative stretch, no vertical
    # dependence).  ``jacobian`` IS that (1+r3t) factor ONLY on an
    # OceanPartialCellCoordinate (vertical.py compute_ocean_jacobian:
    # (eta+H_bathy)/H_bathy, the LOCAL column depth).  On a pure
    # OceanZStarCoordinate the SAME function returns (eta+H_bathy)/H_max (the
    # GLOBAL max depth) -- a DIFFERENT quantity (nemo_bn2_live_ladders'
    # docstring: off by median 1.1e-1) that is not even ~1 at eta=0 over
    # sloping bathymetry, so applying it here would corrupt e3w on every
    # z-star fixture (caught by test_dispatch_prefers_nemo_native_over_
    # generic_treguier: NaN from a stretched e3w going negative on a shallow
    # column).  Gate strictly on the coordinate type, matching the existing
    # isinstance(z_coord, OceanPartialCellCoordinate) convention this file
    # already uses for pgf_scheme="nemo_sco" et al.  jacobian=None or a pure
    # z* coordinate -> stretch=1, BIT-IDENTICAL.
    _stretch2d = (jnp.asarray(jacobian, dtype=dtype)
                  if (jacobian is not None
                      and isinstance(z_coord, OceanPartialCellCoordinate))
                  else None)
    _depth_mode = getattr(
        cfg, "slope_depth_evaluation", "legacy_jacobian_t_surface")
    if _depth_mode not in (
            "legacy_jacobian_t_surface", "nemo_qco_live_literal"):
        raise ValueError(
            "unknown GMRediConfig.slope_depth_evaluation "
            f"{_depth_mode!r}; expected 'legacy_jacobian_t_surface' or "
            "'nemo_qco_live_literal'")
    _live_gdept = None
    _live_gdepw = None
    if _depth_mode == "nemo_qco_live_literal":
        if eta is None or H_bathy is None:
            raise ValueError(
                "slope_depth_evaluation='nemo_qco_live_literal' requires "
                "NOW sea-surface height and local bathymetry")
        _live_gdept, _live_gdepw, _live_stretch = _nemo_qco_live_slope_depths(
            eta, H_bathy, z_coord, dtype)

    from legoesm.grids.latlon import ensure_geometry
    geom = ensure_geometry(grid)
    e1u = jnp.asarray(geom.dx_u[:, 1:], dtype=dtype)             # east face of cell i
    e2v = jnp.asarray(geom.dy_v[1:, :], dtype=dtype)
    e1t = jnp.asarray(geom.dx_T, dtype=dtype)
    e2t = jnp.asarray(geom.dy_T, dtype=dtype)

    ones_z = jnp.ones((1, 1, nlev), dtype=dtype)
    act = (mask[:, :, None] * ones_z if active_3d is None
           else active_3d.astype(dtype))
    umask3 = u_mask[:, 1:, None] * act * jnp.roll(act, -1, axis=1)
    vmask3 = v_mask[1:, :, None] * act * jnp.roll(act, -1, axis=0)

    _prd_mode = getattr(cfg, "slope_prd_evaluation", "density_roundtrip")
    if prd_override is not None:
        prd = jnp.asarray(prd_override, dtype=dtype)
    elif _prd_mode == "density_roundtrip":
        prd = rho / jnp.asarray(rho_0, dtype=dtype) - 1.0        # legacy
    elif _prd_mode == "nemo_literal":
        if not isinstance(z_coord, OceanPartialCellCoordinate):
            raise ValueError(
                "slope_prd_evaluation='nemo_literal' requires the NEMO "
                "partial-cell coordinate carrying t_depth_ref")
        _raw_gdept = getattr(z_coord, "nemo_gdept_0", None)
        if _raw_gdept is None:
            raise ValueError(
                "slope_prd_evaluation='nemo_literal' requires raw "
                "z_coord.nemo_gdept_0; an averaged 1-D ladder changes the "
                "last bits before horizontal differencing")
        _gdept_prd = jnp.asarray(_raw_gdept, dtype=dtype)
        _prd_stretch = (_stretch2d if prd_jacobian is None
                        else jnp.asarray(prd_jacobian, dtype=dtype))
        if _prd_stretch is not None:
            _gdept_prd = _gdept_prd * _prd_stretch[..., jnp.newaxis]
        _prd_T, _prd_S = (T, S) if prd_TS_override is None else prd_TS_override
        _seos = (eos_nemo_seos if eos_nemo_seos is not None
                 else NemoSEOSConfig(rho0=rho_0))
        prd = nemo_seos_prd_literal(
            _prd_T, _prd_S, _gdept_prd, _seos) * act
    else:
        raise ValueError(
            "unknown GMRediConfig.slope_prd_evaluation "
            f"{_prd_mode!r}; expected 'density_roundtrip' or 'nemo_literal'")

    # Shared W-point e3w / wmask3 / pn2 (NEMO ldf_eiv reuses exactly this
    # geometry + N² — factored so the adaptive-κ path below stays bit-
    # consistent with the slopes it is coupled to; no duplicate numerics).
    e3w, wmask3, pn2 = _nemo_wpoint_e3w_wmask_n2(
        rho, T, S, z_coord, eos_fn, rho_0, g, act,
        slope_n2=getattr(cfg, 'slope_n2', 'adiabatic'), jacobian=jacobian,
        eos_nemo_seos=eos_nemo_seos)
    _e3w_surface = e3w[..., :1]
    if pn2_override is not None:
        pn2 = jnp.asarray(pn2_override, dtype=dtype)
        # The step-entry eosbn2 bundle stores NEMO levels 2:jpk (nlev-1),
        # while ldf_slp's local rn2b array includes the prescribed zero
        # surface W slot.  Restore that exact slot before any full-level
        # prd/rn2 arithmetic.
        if pn2.shape[-1] == nlev - 1:
            pn2 = jnp.concatenate([jnp.zeros_like(pn2[..., :1]), pn2], axis=-1)
        elif pn2.shape[-1] != nlev:
            raise ValueError(
                "pn2_override must contain nlev or nlev-1 W levels, got "
                f"{pn2.shape[-1]} for nlev={nlev}")
    if e3w_override is not None:
        e3w = jnp.asarray(e3w_override, dtype=dtype)
        # The step-entry bundle stores the interior W interfaces (NEMO
        # levels 2:jpk), whereas ldf_slp's local array also has the unused
        # surface slot.  Restore its finite live-geometry value: a zero
        # placeholder would be primal-inert after the prescribed surface-slope
        # overwrite, but the vectorized limiter evaluates 1/e3w first and
        # reverse mode would see 0*inf -> NaN.
        if e3w.shape[-1] == nlev - 1:
            surface_shape = e3w.shape[:-1] + (1,)
            surface_e3w = jnp.broadcast_to(_e3w_surface, surface_shape)
            e3w = jnp.concatenate([surface_e3w, e3w], axis=-1)
        elif e3w.shape[-1] != nlev:
            raise ValueError(
                "e3w_override must contain nlev or nlev-1 W levels, got "
                f"{e3w.shape[-1]} for nlev={nlev}")
    pn2_kp1 = jnp.concatenate([pn2[:, :, 1:],
                               jnp.zeros((nlat, nlon, 1), dtype=dtype)], axis=-1)

    # masked horizontal rhd gradients (east/north faces of cell i/j)
    zgru = umask3 * (jnp.roll(prd, -1, axis=1) - prd)
    zgrv = vmask3 * (jnp.roll(prd, -1, axis=0) - prd)

    # zdzr at T-points (:188-195); the (1 - 0.5*tmask(k+1)) factor averages
    # over the wet w-levels bracketing level k.
    act_kp1 = jnp.concatenate([act[:, :, 1:],
                               jnp.zeros((nlat, nlon, 1), dtype=dtype)], axis=-1)
    zdzr = (-1.0 / jnp.asarray(g, dtype)) * (prd + 1.0) \
        * (pn2 + pn2_kp1) * (1.0 - 0.5 * act_kp1)

    slpmax = jnp.asarray(cfg.S_max, dtype=dtype)
    z1_slpmax = 1.0 / slpmax

    # ML indices per column (NEMO nmln, via the shared zdfmxl-criterion
    # helper: ``first`` = first stratified cell = 0-based nmln).
    hml, m_base = _nemo_mld(
        cfg.mld_criterion, T, S, mask, z_coord, eos_fn, cfg.mld_rho_c,
        g=g, rho_0=rho_0, active_3d=active_3d, jacobian=jacobian,
        n2_override=pn2_override, e3w_override=e3w_override,
        eos_nemo_seos=eos_nemo_seos)
    first = jnp.clip(m_base + 1, 1, nlev - 1) if nmln_override is None else jnp.asarray(nmln_override, dtype=jnp.int32)
    # zhmlpt = gdept(nmln-1,Kmm) = depth of the last T-point inside the ML
    # (ldfslp.F90:143) -- live gdept, so the static per-level gather is
    # stretched by the SAME per-column (1+r3t) factor afterward (stretch has
    # no level dependence, so gather-then-stretch == stretch-then-gather).
    if _depth_mode == "nemo_qco_live_literal":
        zhmlpt = jnp.take_along_axis(
            _live_gdept, jnp.clip(first - 1, 0, nlev - 1)[..., None],
            axis=-1)[..., 0] * mask
    else:
        zhmlpt = jnp.take(gdept, jnp.clip(first - 1, 0, nlev - 1)) * mask
        if _stretch2d is not None:
            zhmlpt = zhmlpt * _stretch2d

    kidx = jnp.arange(nlev)[None, None, :]

    _metric_mode = getattr(cfg, "slope_metric_evaluation", "division")
    if _metric_mode not in ("division", "nemo_reciprocal"):
        raise ValueError(
            "unknown GMRediConfig.slope_metric_evaluation "
            f"{_metric_mode!r}; expected 'division' or 'nemo_reciprocal'")

    def _uv_slp(zg, zb_pair, e1_face, e3_face, iku, r1_hml, zdep_face, msk3):
        """Shared u/v-slope assembly (:206-238 without the Shapiro)."""
        if _metric_mode == "nemo_reciprocal":
            # domhgr.F90:140 stores r1_e1u/r1_e2v; ldfslp.F90:242-243 then
            # multiplies. Keep both rounding boundaries visible to XLA.
            r1_face = jax.lax.optimization_barrier(
                jnp.asarray(1.0, dtype=dtype) / e1_face)
            zau = jax.lax.optimization_barrier(zg * r1_face[:, :, None])
        else:
            zau = zg / e1_face[:, :, None]
        zbu = jnp.minimum(
            zb_pair,
            jnp.minimum(-z1_slpmax * jnp.abs(zau),
                        (-_NEMO_SLOPE_STAB_7E3 / e3_face) * jnp.abs(zau)))
        s_int = zau / (zbu - zeps)
        anchor = jnp.take_along_axis(s_int, iku[:, :, None], axis=-1)[:, :, 0]
        anchor = anchor * r1_hml
        in_ml = kidx < iku[:, :, None]
        return jnp.where(in_ml, zdep_face * anchor[:, :, None], s_int) * msk3

    # --- uslp ---
    zb_u = 0.5 * (zdzr + jnp.roll(zdzr, -1, axis=1))
    iku = jnp.maximum(first, jnp.roll(first, -1, axis=1))
    r1_hmlu = 1.0 / jnp.maximum(
        jnp.maximum(zhmlpt, jnp.roll(zhmlpt, -1, axis=1)),
        jnp.asarray(_NEMO_HML_UV_FLOOR_M, dtype))
    # zdepu/zdepv ~ gdept(...,Kmm) (ldfslp.F90:261-266, live).  NEMO takes the
    # U-FACE / V-FACE AVERAGE of the two bracketing T-column depths:
    #     zdepu = 0.5*( (gdept(i,j,k) + gdept(i+1,j,k)) - e3u(i,j,miku,Kmm) )
    #     zdepv = 0.5*( (gdept(i,j,k) + gdept(i,j+1,k)) - e3v(i,j,mikv,Kmm) )
    # (risfdep == 0, no ice shelf in DINO).  Using the bare T-point ladder for
    # BOTH -- and in particular passing zdepu to the v-slope, which averages
    # over the wrong axis entirely -- was a transcription defect (#1226).
    # Stretch per column FIRST, then face-average, so each column carries its
    # own (1+r3t) exactly as NEMO's live gdept does.
    if _depth_mode == "nemo_qco_live_literal":
        _gd_col = _live_gdept
    else:
        _gd_col = gdept[None, None, :] * jnp.ones_like(zgru)
        if _stretch2d is not None:
            _gd_col = _gd_col * _stretch2d[:, :, None]
        _e3_top = 0.5 * dz[0]
        # axis=1 is i/lon, axis=0 is j/lat. Preserve legacy association.
        if _stretch2d is not None:
            _e3_top = _e3_top * _stretch2d[:, :, None]
    # NEMO's slope stability bound is -7e3/e3u(ji,jj,jk,Kmm)*|zau| (ldfslp.F90
    # :133-134) and it uses the U-FACE / V-FACE thickness, NOT the cell value.
    # At a staircase / partial-cell topography step the face thickness is the
    # MIN of the two adjacent cells and is therefore MUCH smaller than e3t, so
    # NEMO clamps the slope far harder exactly there -- at the topography that
    # sets form stress and the sill. Using e3t (and the same array for BOTH the
    # u- and v-slope) left legoESM's wslpi at corr 0.9585 vs NEMO's own dumped
    # field, a PATTERN error concentrated at topography which the psi vertical
    # difference then amplified into a 0.77 correlation on the eiv transport
    # (#1226). h_partial carries the staircase; dz_ref does not.
    _hp = getattr(z_coord, "h_partial", None)
    if _hp is not None:
        _h3 = jnp.asarray(_hp, dtype=dtype)
        _floor = jnp.asarray(1.0e-10, dtype=dtype)
        e3u_k = jnp.maximum(jnp.minimum(_h3, jnp.roll(_h3, -1, axis=1)), _floor)
        e3v_k = jnp.maximum(jnp.minimum(_h3, jnp.roll(_h3, -1, axis=0)), _floor)
    else:                                    # z-star / flat: e3u = e3v = e3t
        e3u_k = dz[None, None, :]
        e3v_k = dz[None, None, :]
    _face_e3_mode = getattr(
        cfg, "slope_face_thickness_evaluation", "static_face")
    if _face_e3_mode not in ("static_face", "nemo_qco_live"):
        raise ValueError(
            "unknown GMRediConfig.slope_face_thickness_evaluation "
            f"{_face_e3_mode!r}; expected 'static_face' or 'nemo_qco_live'")
    if _face_e3_mode == "nemo_qco_live":
        if eta is None:
            raise ValueError(
                "slope_face_thickness_evaluation='nemo_qco_live' requires "
                "the NOW sea-surface height")
        e3u_k, e3v_k = _nemo_qco_live_slope_face_thicknesses(
            eta, z_coord, e3u_k, e3v_k, umask3, vmask3)
    if _depth_mode == "nemo_qco_live_literal":
        zdepu = _nemo_literal_slope_face_depth(
            _gd_col, e3u_k, axis=1)
        zdepv = _nemo_literal_slope_face_depth(
            _gd_col, e3v_k, axis=0)
    else:
        zdepu = 0.5 * (_gd_col + jnp.roll(_gd_col, -1, axis=1)) - _e3_top
        zdepv = 0.5 * (_gd_col + jnp.roll(_gd_col, -1, axis=0)) - _e3_top
    uslp = _uv_slp(zgru, zb_u, e1u, e3u_k, iku, r1_hmlu, zdepu, umask3)

    # --- vslp ---
    zb_v = 0.5 * (zdzr + jnp.roll(zdzr, -1, axis=0))
    ikv = jnp.maximum(first, jnp.roll(first, -1, axis=0))
    r1_hmlv = 1.0 / jnp.maximum(
        jnp.maximum(zhmlpt, jnp.roll(zhmlpt, -1, axis=0)),
        jnp.asarray(_NEMO_HML_UV_FLOOR_M, dtype))
    vslp = _uv_slp(zgrv, zb_v, e2v, e3v_k, ikv, r1_hmlv, zdepv, vmask3)

    # --- wslpi / wslpj (:265-297) ---
    zgru_im1 = jnp.roll(zgru, +1, axis=1)
    zgrv_jm1 = jnp.roll(zgrv, +1, axis=0)
    um_im1 = jnp.roll(umask3, +1, axis=1)
    vm_jm1 = jnp.roll(vmask3, +1, axis=0)

    def _km1(a):   # value at level k-1, zero-padded at the surface row
        return jnp.concatenate(
            [jnp.zeros((nlat, nlon, 1), dtype=dtype), a[:, :, :-1]], axis=-1)

    zci = jnp.maximum(um_im1 + umask3 + _km1(um_im1) + _km1(umask3), zeps) \
        * e1t[:, :, None]
    zcj = jnp.maximum(vm_jm1 + vmask3 + _km1(vm_jm1) + _km1(vmask3), zeps) \
        * e2t[:, :, None]
    # Preserve compiled ldfslp.f90:294-297's explicit pairings.  The j expression is
    # intentionally cross-paired (jm1@iik + local@iikm1, then the converse),
    # not a left-associated sum of four faces.  The barriers make those source
    # parentheses survive XLA lowering; this matters after the exact prd carry,
    # where the remaining differences are a few ULP of a near-zero gradient.
    # Round 42 exposed a few-ULP association change on non-identity cards.
    # Until the owner decides its global scope, select NEMO's association only
    # on the existing DINO literal path and GYRE's raw-live-depth path.
    _nemo_identity_association = (
        _prd_mode == "nemo_literal"
        or _depth_mode == "nemo_qco_live_literal"
    )
    if _nemo_identity_association:
        _zai_now = lax.optimization_barrier(zgru_im1 + zgru)
        _zai_before = lax.optimization_barrier(_km1(zgru_im1) + _km1(zgru))
        zai = (lax.optimization_barrier(_zai_now + _zai_before) / zci) * wmask3
        _zaj_cross1 = lax.optimization_barrier(zgrv_jm1 + _km1(zgrv))
        _zaj_cross2 = lax.optimization_barrier(_km1(zgrv_jm1) + zgrv)
        zaj = (lax.optimization_barrier(_zaj_cross1 + _zaj_cross2) / zcj) * wmask3
    else:
        zai = (zgru_im1 + zgru + _km1(zgru_im1) + _km1(zgru)) / zci * wmask3
        zaj = (zgrv_jm1 + zgrv + _km1(zgrv_jm1) + _km1(zgrv)) / zcj * wmask3
    zbw = (-0.5 / jnp.asarray(g, dtype)) * pn2 * (prd + _km1(prd) + 2.0)
    # e3w is (nlev,) (static ladder, jacobian=None) or (nlat,nlon,nlev) (live,
    # jacobian passed) -- shape is trace-time-static, branch is safe under JIT.
    e3w_k = e3w[None, None, :] if e3w.ndim == 1 else e3w
    zbi = jnp.minimum(zbw, jnp.minimum(-z1_slpmax * jnp.abs(zai),
                                       (-_NEMO_SLOPE_STAB_7E3 / e3w_k) * jnp.abs(zai)))
    zbj = jnp.minimum(zbw, jnp.minimum(-z1_slpmax * jnp.abs(zaj),
                                       (-_NEMO_SLOPE_STAB_7E3 / e3w_k) * jnp.abs(zaj)))
    swi_int = zai / (zbi - zeps)
    swj_int = zaj / (zbj - zeps)
    # ML ramp (:284-297): in-ML for w-level jk <= nmln (1-based) — in the
    # 0-based top-of-cell-k indexing (jk = k+1): k <= first; anchor at
    # jk = nmln+1 => k = first+1 (the first w-level BELOW the ML base).
    kanc = _nemo_ml_anchor_index(first, act, nlev)
    r1_hmlw = 1.0 / jnp.maximum(hml, jnp.asarray(_NEMO_HMLW_FLOOR_M, dtype))
    anc_i = jnp.take_along_axis(swi_int, kanc[:, :, None], axis=-1)[:, :, 0] * r1_hmlw
    anc_j = jnp.take_along_axis(swj_int, kanc[:, :, None], axis=-1)[:, :, 0] * r1_hmlw
    # zck = gdepw(jk,Kmm) - gdepw(mikt,Kmm) (ldfslp.F90:289); mikt=0 (no ice
    # shelf) and gdepw_top[0]=0, so this is the live gdepw_top -- same
    # per-column stretch as zhmlpt/zdepu above.
    if _depth_mode == "nemo_qco_live_literal":
        zck = _live_gdepw - _live_gdepw[..., :1]
    else:
        zck = gdepw_top[None, None, :]
        if _stretch2d is not None:
            zck = zck * _stretch2d[:, :, None]
    in_ml_w = kidx < kanc[:, :, None]
    # Compiled ldfslp.f90:303-312 writes the integer zfk selector as two multiplied
    # arms followed by one addition and the final wmask.  This association is
    # part of the NEMO-native operator, independent of how prd was produced.
    if _nemo_identity_association:
        zfk = (~in_ml_w).astype(dtype)
        _outside_i = lax.optimization_barrier(zfk * swi_int)
        _outside_j = lax.optimization_barrier(zfk * swj_int)
        _inside_depth = lax.optimization_barrier((1.0 - zfk) * zck)
        _inside_i = lax.optimization_barrier(
            _inside_depth * anc_i[:, :, None])
        _inside_j = lax.optimization_barrier(
            _inside_depth * anc_j[:, :, None])
        wslpi = lax.optimization_barrier(
            lax.optimization_barrier(_outside_i + _inside_i) * wmask3)
        wslpj = lax.optimization_barrier(
            lax.optimization_barrier(_outside_j + _inside_j) * wmask3)
    else:
        wslpi = jnp.where(in_ml_w, zck * anc_i[:, :, None], swi_int) * wmask3
        wslpj = jnp.where(in_ml_w, zck * anc_j[:, :, None], swj_int) * wmask3
    wslpi = wslpi.at[:, :, 0].set(0.0)
    wslpj = wslpj.at[:, :, 0].set(0.0)

    _uslp_raw, _vslp_raw = uslp, vslp  # pre-Shapiro diagnostic rows
    def _shap(f, cof, literal_factors=None):
        # Lon (axis 1) ghost cells are PERIODIC: NEMO's slope loops compute
        # zwz/zww over the halo columns as well (DO_2D(1,1,1,1),
        # ldfslp.F90:203,265) from lbc-filled inputs (DINO ldIperio=.TRUE.),
        # so the i-edge columns see their true wrap neighbours.  Zero-filling
        # here treated the seam as a closed wall and deflated the two seam
        # columns' slopes to ~0.75x (squaring to the 0.55 zah deficit, #1226)
        # -- while every other stencil in this function already wraps via
        # jnp.roll.  Lat (axis 0) stays zero-filled: the channel is closed in
        # j on both sides, matching NEMO's masked halo there.
        fp = jnp.pad(f, ((0, 0), (1, 1), (0, 0)), mode="wrap")
        fp = jnp.pad(fp, ((1, 1), (0, 0), (0, 0)))
        if literal_factors is None:
            w = (1.0, 2.0, 1.0)
            acc = jnp.zeros_like(f)
            for a in range(3):
                for b in range(3):
                    acc = acc + w[a] * w[b] * fp[a:a + nlat, b:b + nlon, :]
            return acc * cof / 16.0  # coeff-ok: NEMO's 1/16 Shapiro weight

        # ldfslp.F90:348-367 writes this association explicitly for
        # reproducibility.  Keep its corner pairs, cardinal pairs, and
        # left-to-right coefficient product intact on the literal path.
        nw_ne = lax.optimization_barrier(
            fp[:nlat, :nlon, :] + fp[:nlat, 2:, :])
        sw_se = lax.optimization_barrier(
            fp[2:, :nlon, :] + fp[2:, 2:, :])
        corners = lax.optimization_barrier(nw_ne + sw_se)
        n_w = lax.optimization_barrier(
            fp[:nlat, 1:nlon + 1, :] + fp[1:nlat + 1, :nlon, :])
        e_s = lax.optimization_barrier(
            fp[1:nlat + 1, 2:, :] + fp[2:, 1:nlon + 1, :])
        cardinals = lax.optimization_barrier(n_w + e_s)
        acc = lax.optimization_barrier(
            corners + 2.0 * cardinals
            + 4.0 * fp[1:nlat + 1, 1:nlon + 1, :])
        # 1/16 normalises the (1,2,1)x(1,2,1) nine-point stencil assembled
        # just above: corners + 2*cardinals + 4*centre sums to 16.
        zcof = jnp.asarray(1.0 / 16.0, dtype=dtype)  # coeff-ok: stencil weight
        for factor in literal_factors:
            zcof = lax.optimization_barrier(zcof * factor)
        return lax.optimization_barrier(acc * zcof)

    def _kp1m(a):  # mask at level k+1, zero at the bottom
        return jnp.concatenate(
            [a[:, :, 1:], jnp.zeros((nlat, nlon, 1), dtype=dtype)], axis=-1)

    _u_lat = jnp.roll(umask3, -1, axis=0) + jnp.roll(umask3, +1, axis=0)
    _u_vert = umask3 + _kp1m(umask3)
    _v_lon = jnp.roll(vmask3, -1, axis=1) + jnp.roll(vmask3, +1, axis=1)
    _v_vert = vmask3 + _kp1m(vmask3)
    _w_u = umask3 + um_im1
    _w_v = vmask3 + vm_jm1
    cof_u = 0.25 * _u_lat * _u_vert
    cof_v = 0.25 * _v_lon * _v_vert
    cof_w = 0.25 * wmask3 * _w_u * _w_v
    if _nemo_identity_association:
        half = jnp.asarray(0.5, dtype=dtype)
        quarter = jnp.asarray(0.25, dtype=dtype)
        uslp = _shap(uslp, cof_u, (_u_lat, half, _u_vert, half))
        vslp = _shap(vslp, cof_v, (_v_lon, half, _v_vert, half))
        wslpi = _shap(wslpi, cof_w, (wmask3, _w_u, _w_v, quarter))
        wslpj = _shap(wslpj, cof_w, (wmask3, _w_u, _w_v, quarter))
    else:
        uslp = _shap(uslp, cof_u)
        vslp = _shap(vslp, cof_v)
        wslpi = _shap(wslpi, cof_w)
        wslpj = _shap(wslpj, cof_w)
    # ldfslp.F90:210 executes jk=jpkm1..2; U/V level 1 is never assigned and
    # enters ldftra as its initialized zero.  The vectorized transcription
    # otherwise evaluates that extra surface level.  Keep legacy cards byte-
    # identical and prescribe the literal DINO slot only.
    if _nemo_identity_association:
        uslp = uslp.at[..., 0].set(0.0)
        vslp = vslp.at[..., 0].set(0.0)
    return ((uslp, vslp, wslpi, wslpj, _nemo_native_slope_diagnostics(locals())) if return_diagnostics else (uslp, vslp, wslpi, wslpj))


def _nemo_treguier_left_reductions(zn_term, zah_term, ze3w, zhw_offset):
    """Source-ordered ``ldf_eiv`` column accumulators (jk=1..jpk)."""
    def _left_body(jk, carry):
        _zn, _zah, _zhw = carry
        return (_zn + zn_term[..., jk],
                _zah + zah_term[..., jk],
                _zhw + ze3w[..., jk])

    _zero = jnp.zeros_like(zn_term[..., 0])
    return jax.lax.fori_loop(
        0, zn_term.shape[-1], _left_body,
        (_zero, _zero, _zero + jnp.asarray(
            zhw_offset, dtype=zn_term.dtype)))


@jax.custom_jvp
def _nemo_sqrt_nonnegative_forward_exact(value):
    """NEMO ``SQRT(MAX(value,0))`` forward with a finite zero tangent."""
    return jnp.sqrt(jnp.maximum(value, jnp.zeros_like(value)))


@_nemo_sqrt_nonnegative_forward_exact.defjvp
def _nemo_sqrt_nonnegative_forward_exact_jvp(primals, tangents):
    (value,), (value_dot,) = primals, tangents
    result = _nemo_sqrt_nonnegative_forward_exact(value)
    safe_result = jnp.where(result > 0.0, result, jnp.ones_like(result))
    result_dot = jnp.where(
        value > 0.0, 0.5 * value_dot / safe_result,
        jnp.zeros_like(value_dot))
    return result, result_dot


def compute_treguier_kappa_gm_nemo_native(
    rho: jnp.ndarray,
    T: jnp.ndarray,
    S: jnp.ndarray,
    wslpi: jnp.ndarray,
    wslpj: jnp.ndarray,
    mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    grid: LatLonGrid,
    f_coriolis: jnp.ndarray,
    cfg,
    eos_fn,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    active_3d: jnp.ndarray | None = None,
    jacobian: jnp.ndarray | None = None,
    return_diagnostics: bool = False,
    slope_n2: str = "adiabatic",
    omega: float = constants.Omega,
    vertical_reduction_evaluation: str = "tree",
    sqrt_evaluation: str = "guarded_floor",
    pn2_override: jnp.ndarray | None = None,
    e3w_override: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Treguier et al. (1997) adaptive κ_GM (NEMO ``ldftra.F90::ldf_eiv``,
    ``nn_aei_ijk_t=21``, the non-triad ``ln_traldf_triad=.FALSE.`` ELSE
    branch, :684-698) fed the SAME W-point slopes (``wslpi``/``wslpj``) that
    drive the Redi/GM tendency operator itself — NOT the simplified
    cell-centred ``compute_isopycnal_slopes_latlon_cgrid`` output that
    :func:`compute_treguier_kappa_gm` (``_gm_redi_common.py``) consumes.

    #1317: with ``slope_positions="nemo_native"``, the Redi/GM flux and
    the implicit K33 both already read ``wslpi``/``wslpj`` from
    :func:`compute_nemo_native_slopes` — but ``kappa_GM`` (the coefficient
    that then FEEDS that same flux's bolus term) was being built from a
    DIFFERENT slope field (cell-centred gradients averaged to interior
    interfaces only, DM95/nemo_cap tapered — no mixed-layer ramp, no
    surface w-point). NEMO's ``ldf_eiv`` sums over the SAME ``wslpi``/
    ``wslpj``/``rn2b`` arrays ``ldf_slp`` just built, over the FULL water
    column ``jk=1,jpk`` (including the surface w-level, ``e3w(1)`` — a real
    half-cell contribution, NOT the interior-only interfaces the generic
    Visbeck/EKE path integrates). Verified (#1317 diagnostic, DINO day-0
    twin): the two κ_GM fields have corr=0.28, mean 646 vs 196 m²/s — a
    real, non-negligible formulation gap, not roundoff.

    Method (``ldftra.F90:664-706``, exact — see also
    :data:`_gm_redi_common.TREGUIER_RO_FACTOR` &c. for the shared tunables
    reused here):

    .. math::

        \kappa = \min\big(\;\min(1, |f/f_{20}|)\cdot Ro^2\,T^{-1},\; aei0\big)

    - ``zn2 = max(rn2b, 0)``; ``zn = Σ_jk sqrt(zn2)·e3w(jk)`` — **UNMASKED**
      by ``wmask`` (ldftra.F90:689, matches NEMO exactly: pn2 itself is
      already wmask-zeroed at construction, ldfslp convention, so the
      product is effectively masked without an explicit second factor);
    - ``Ro = clip(0.4·zn/max(|f|,1e-10), 2 km, 40 km)``;
    - ``zah = Σ_jk zn2·(wslpi²+wslpj²)·e3w(jk)·wmask(jk)``,
      ``zhw = 5 + Σ_jk e3w(jk)·wmask(jk)`` (the ``ldf_eiv`` ``zhw(:,:)=5.``
      initialisation offset, ``_gm_redi_common.TREGUIER_ZHW_OFFSET_M``);
      ``T⁻¹ = sqrt(zah/zhw)``;
    - tropical taper ``min(1, |f|/f₂₀)``, ``f₂₀ = 2Ω sin(20°)``;
    - cap at ``cfg.aei0`` (the one namelist tunable, ``rn_Ue·rn_Le``).

    ``wslpi``/``wslpj`` are the FINAL (Shapiro-smoothed, ML-ramped)
    :func:`compute_nemo_native_slopes` output — exactly what
    ``ldf_slp`` hands to ``ldf_eiv`` in the SAME call window
    (``stpmlf.F90:196-203``: both consume the Nbb-level ``wslpi``/``wslpj``/
    ``rn2b`` ``ldf_slp`` just computed).

    ``omega`` MUST be the SAME Earth rotation rate that built ``f_coriolis``
    (``grid.f``) -- it feeds ``f20 = 2*omega*sin(20deg)``, the tropical-taper
    reference used as ``min(1, |f_coriolis|/f20)``. A mismatched ``omega``
    here would NOT cancel in that ratio and would reintroduce the amplitude
    bias this parameter exists to remove (#1226: legoESM's canonical
    ``constants.Omega`` is a rounded 4-sig-fig version of the physical
    Earth rotation rate; the relative gap against NEMO's own full-precision
    value enters ``zRo = 0.4*zn/|f|`` linearly and ``zaeiw = zRo^2*T^-1``
    quadratically -- confirmed by feeding NEMO's own dumped
    zn/zah/zhw/wslpi/wslpj through this exact formula with NEMO's omega vs
    legoESM's default: corr stayed 1.0 both ways but the per-cell relative
    bias dropped to machine precision under NEMO's omega).

    Returns the 2-D ``kappa_GM`` [m²/s], zero on dry columns.
    """
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        TREGUIER_RO_FACTOR, TREGUIER_RO_MIN_M, TREGUIER_RO_MAX_M,
        TREGUIER_F_MIN, TREGUIER_ZHW_OFFSET_M, TREGUIER_TAPER_LAT_DEG,
    )
    dtype = rho.dtype
    ones_z = jnp.ones((1, 1, rho.shape[-1]), dtype=dtype)
    act = (mask[:, :, None] * ones_z if active_3d is None
           else active_3d.astype(dtype))
    e3w, wmask3, pn2 = _nemo_wpoint_e3w_wmask_n2(
        rho, T, S, z_coord, eos_fn, rho_0, g, act,
        slope_n2=slope_n2, jacobian=jacobian)
    _e3w_surface = e3w[..., :1]
    nlev = rho.shape[-1]
    if pn2_override is not None:
        pn2 = jnp.asarray(pn2_override, dtype=dtype)
        if pn2.shape[-1] == nlev - 1:
            pn2 = jnp.concatenate([jnp.zeros_like(pn2[..., :1]), pn2], axis=-1)
        elif pn2.shape[-1] != nlev:
            raise ValueError(
                "pn2_override must contain nlev or nlev-1 W levels, got "
                f"{pn2.shape[-1]} for nlev={nlev}")
    if e3w_override is not None:
        e3w = jnp.asarray(e3w_override, dtype=dtype)
        if e3w.shape[-1] == nlev - 1:
            surface_shape = e3w.shape[:-1] + (1,)
            surface_e3w = jnp.broadcast_to(_e3w_surface, surface_shape)
            e3w = jnp.concatenate([surface_e3w, e3w], axis=-1)
        elif e3w.shape[-1] != nlev:
            raise ValueError(
                "e3w_override must contain nlev or nlev-1 W levels, got "
                f"{e3w.shape[-1]} for nlev={nlev}")
    e3w_3d = jnp.broadcast_to(e3w, rho.shape)

    # Floor at 1e-30 (not a hard 0) before sqrt: sqrt(0) has an infinite
    # gradient in JAX, which combined with maximum(pn2,0)'s zero-gradient
    # on the clipped side evaluates the backward pass as inf*0 -> NaN on
    # every dry/masked cell (and, via the wslpi/wslpj roll stencils'
    # meridional wrap at the channel's non-periodic north/south rows,
    # poisons those rows' gradient too). Forward effect is sqrt(1e-30)
    # ~ 1e-15, negligible -- exactly the guard compute_treguier_kappa_gm
    # (_gm_redi_common.py) already uses for the same reason.
    zn2 = jnp.maximum(pn2, 0.0)
    if sqrt_evaluation == "nemo_forward_exact":
        sqrt_zn2 = _nemo_sqrt_nonnegative_forward_exact(pn2)
    elif sqrt_evaluation == "guarded_floor":
        sqrt_zn2 = jnp.sqrt(jnp.maximum(zn2, 1e-30))
    else:
        raise ValueError(
            "GMRediConfig.treguier_sqrt_evaluation must be "
            f"'guarded_floor' or 'nemo_forward_exact', got {sqrt_evaluation!r}")
    _reduction = vertical_reduction_evaluation
    if _reduction == "nemo_left":
        # ldftra.F90:665,687-696: the three 2-D work arrays are initialized
        # once, then each jk contributes exactly once in surface-to-bottom
        # source order. A tree reduction changes the last bits of all three
        # operands and those bits survive the Rossby-radius/timescale chain.
        _zn_term = sqrt_zn2 * e3w_3d
        _ze3w = e3w_3d * wmask3
        _zah_term = zn2 * (wslpi ** 2 + wslpj ** 2) * _ze3w

        zn, zah, zhw = _nemo_treguier_left_reductions(
            _zn_term, _zah_term, _ze3w, TREGUIER_ZHW_OFFSET_M)
    elif _reduction == "tree":
        # Preserve the generic/off statements byte-for-byte.
        zn = jnp.sum(sqrt_zn2 * e3w_3d, axis=-1)  # :689, unmasked term
        ze3w = e3w_3d * wmask3
        zah = jnp.sum(zn2 * (wslpi ** 2 + wslpj ** 2) * ze3w, axis=-1)  # :694-695
        zhw = TREGUIER_ZHW_OFFSET_M + jnp.sum(ze3w, axis=-1)           # :665,696
    else:
        raise ValueError(
            "GMRediConfig.treguier_vertical_reduction_evaluation must be "
            f"'tree' or 'nemo_left', got {_reduction!r}")

    f_abs = jnp.maximum(jnp.abs(f_coriolis), TREGUIER_F_MIN)
    ro = jnp.clip(TREGUIER_RO_FACTOR * zn / f_abs,
                  TREGUIER_RO_MIN_M, TREGUIER_RO_MAX_M)
    # Same sqrt(0)-VJP guard as zn above: zah is EXACTLY 0 wherever wslpi=
    # wslpj=0 (dry columns, or a genuinely flat/unstratified wet column —
    # unlike compute_treguier_kappa_gm's sigma, which always carries a
    # 1e-30 floor baked into S_mag, wslpi/wslpj here are the raw
    # ldfslp-native slopes and CAN be exact zero); floor before sqrt.
    t_inv = jnp.sqrt(jnp.maximum(zah, 1e-30) / jnp.maximum(zhw, _EPS_DIV))
    f20 = 2.0 * omega * jnp.sin(jnp.deg2rad(TREGUIER_TAPER_LAT_DEG))
    taper = jnp.minimum(1.0, jnp.abs(f_coriolis) / f20)
    kappa = jnp.minimum(taper * ro ** 2 * t_inv, cfg.aei0)
    # Same equatorial-taper floor as the generic path
    # (``_gm_redi_common.compute_treguier_kappa_gm``): the NEMO-native branch
    # runs the IDENTICAL min(1,|f/f20|) taper, so it needs the IDENTICAL floor
    # -- otherwise ``TreguierConfig.kappa_min`` is silently inert on exactly the
    # nemo_iso_lap+nemo_native (most NEMO-faithful) configuration.  Clamped to
    # the cap (same reason as the generic path: the invariant must hold even
    # when a trained/traced aei0 disables the Python-level validator) and
    # applied BEFORE the wet mask so dry columns still return exactly 0;
    # default kappa_min=0.0 keeps this byte-identical.
    kappa = jnp.maximum(kappa, jnp.minimum(cfg.kappa_min, cfg.aei0))
    kappa = jnp.where(mask > 0.5, kappa, 0.0)
    if return_diagnostics:
        # The intermediates, named as in NEMO ldf_eiv (ldftra.F90:664-707), so
        # an oracle deficit in the final coefficient can be localised to ONE
        # term against NEMO's own dumped zn/zah/zhw/zRo/zaeiw (#1226).
        return kappa, {"zn": zn, "zah": zah, "zhw": zhw, "zRo": ro,
                       "zaeiw": kappa}
    return kappa


def nemo_kappa_gm_to_faces(kappa_t, u_surf_mask, v_surf_mask):
    """NEMO's T-point -> U/V-face average of the eiv coefficient.

    Transcribes ``ldftra.F90:716-717`` (NEMO 5.0.2)::

        zaeiu(ji,jj) = 0.5 * ( zaeiw(ji,jj) + zaeiw(ji+1,jj) ) * ssumask(ji,jj)
        zaeiv(ji,jj) = 0.5 * ( zaeiw(ji,jj) + zaeiw(ji,jj+1) ) * ssvmask(ji,jj)

    with the subsequent ``lbc_lnk`` supplying the periodic wrap that
    ``jnp.roll`` provides here.  NEMO's ``paeiu``/``paeiv`` are THESE face
    fields broadcast in depth and 3-D-masked -- the T-point ``zaeiw`` never
    reaches the tendency.  Exposed publicly so oracle comparisons measure the
    ACTUAL NEMO quantity instead of re-implementing this average inline (a
    reconstruction that previously corrupted the two periodic wrap columns and
    confounded the aeiu comparison, #1226).

    Parameters: ``kappa_t`` (n_lat, n_lon) T-point coefficient;
    ``u_surf_mask`` (n_lat, n_lon) EAST-face surface mask aligned with
    ``kappa_t`` columns; ``v_surf_mask`` (n_lat, n_lon) NORTH-face analogue.
    Returns ``(kappa_u, kappa_v)``, each (n_lat, n_lon).
    """
    ax_y, ax_x = 0, 1
    kappa_u = 0.5 * (kappa_t + jnp.roll(kappa_t, -1, ax_x)) * u_surf_mask
    kappa_v = 0.5 * (kappa_t + jnp.roll(kappa_t, -1, ax_y)) * v_surf_mask
    return kappa_u, kappa_v


def compute_isopycnal_slopes_latlon_cgrid(
    rho: jnp.ndarray,
    mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    cfg: GMRediConfig,
    *,
    T: jnp.ndarray | None = None,
    S: jnp.ndarray | None = None,
    eos_fn=None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute tapered isopycnal slopes at vertical interfaces.

    Parameters
    ----------
    rho : (n_lat, n_lon, nlev)
        In-situ density at cell centers.
    mask : (n_lat, n_lon)
        Ocean mask (1 = ocean, 0 = land).
    z_coord : OceanZStarCoordinate
    jacobian : (n_lat, n_lon)
        z-star Jacobian (eta + H) / H.
    grid : LatLonGrid
    cfg : GMRediConfig
    T, S : (n_lat, n_lon, nlev) or None
        Tracer fields, REQUIRED when ``cfg.slope_density == "neutral"`` so the
        neutral density gradient ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` can be built; ignored
        for the default ``"in_situ"`` mode (then only ``rho`` is used).
    eos_fn : callable or None
        EOS ``fn(T, S, p) -> ρ`` (needed only for the neutral mode; the local
        cell pressure / EOS partials are built from it).
    rho_0, g : float
        Reference density / gravity for the local hydrostatic pressure.

    Returns
    -------
    S_x, S_y : (n_lat, n_lon, nlev-1)
        Tapered isopycnal slopes at interfaces.
    taper : (n_lat, n_lon, nlev-1)
        DM95 taper factor in [0, 1].
    """
    slope_density = getattr(cfg, "slope_density", "in_situ")
    _validate_slope_density(slope_density)
    if slope_density == "neutral" and (T is None or S is None or eos_fn is None):
        raise ValueError(
            "compute_isopycnal_slopes_latlon_cgrid: slope_density='neutral' "
            "requires T, S and eos_fn to build the neutral density gradient."
        )
    # Neumann-fill density to prevent garbage gradients at coastlines.
    rho_filled = neumann_fill_cgrid(rho, mask)

    # Face density gradients (in-situ FD of rho, or the neutral
    # ∂ρ/∂T·∇T+∂ρ/∂S·∇S form) — shared with the centred K_33 / triad builders.
    drho_dx_u, drho_dy_v, drho_dz_safe = _slope_density_face_grads(
        rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
        eos_fn, rho_0, g,
    )

    # Average face gradients to cell centers.
    # u-face j is between cell (j-1) and cell j, so cell j's gradient
    # is the average of face j (west) and face j+1 (east).
    drho_dx = 0.5 * (drho_dx_u[:, :-1, :] + drho_dx_u[:, 1:, :])  # (n_lat, n_lon, nlev)
    # v-face i is between cell (i-1) and cell i, so cell i's gradient
    # is the average of face i (south) and face i+1 (north).
    drho_dy = 0.5 * (drho_dy_v[:-1, :, :] + drho_dy_v[1:, :, :])  # (n_lat, n_lon, nlev)

    # Average cell-center horizontal gradients from full levels to interfaces.
    drho_dx_half = 0.5 * (drho_dx[:, :, :-1] + drho_dx[:, :, 1:])  # (n_lat, n_lon, nlev-1)
    drho_dy_half = 0.5 * (drho_dy[:, :, :-1] + drho_dy[:, :, 1:])

    # drho_dz_safe already carries the stable-strat floor (min(0,·)-eps) for
    # both modes (applied inside _slope_density_face_grads).

    # --- Slopes ---
    # in_situ clips to ±S_max (legoESM safety); neutral leaves the slope
    # UNCLIPPED and lets the DM95 taper suppress steep slopes (Veros never
    # clips — see _w_triad_slopes_tapers).  dm95_taper returns S·taper, which is
    # bounded for |S| → ∞ (taper decays faster than S grows).
    S_x_raw = -drho_dx_half / drho_dz_safe
    S_y_raw = -drho_dy_half / drho_dz_safe
    if slope_density != "neutral":
        S_x_raw = jnp.clip(S_x_raw, -cfg.S_max, cfg.S_max)
        S_y_raw = jnp.clip(S_y_raw, -cfg.S_max, cfg.S_max)

    # Adjoint stabilization (primal-invisible; see _gm_redi_common note):
    # "stop_gradient_slopes" freezes the raw slopes (the tapers computed from
    # them then carry no gradient either); "stop_gradient_taper" freezes only
    # the DM95 taper factor inside dm95_taper.  Static Python gating.
    adj_stab = getattr(cfg, "adjoint_stabilization", "none")
    validate_adjoint_stabilization(adj_stab)
    if adj_stab == "stop_gradient_slopes":
        S_x_raw = jax.lax.stop_gradient(S_x_raw)
        S_y_raw = jax.lax.stop_gradient(S_y_raw)

    if getattr(cfg, "slope_limit", "dm95_taper") == "nemo_cap":
        # NEMO ldfslp convention (ldfslp.F90:212-213): the slope is HARD-CAPPED
        # and the flux keeps diffusing ALONG the capped direction at steep
        # fronts — the taper is 1 (the DM95 taper would send the flux to ZERO
        # exactly at the ML-base outcrops, killing the subduction pathway that
        # moves surface heat into the permanent thermocline; plan §G). NEMO
        # applies TWO caps via the denominator bound
        # zbu = MIN(zbu, -|zau|/rn_slpmax, -7e3/e3u*|zau|):
        #   |S| <= rn_slpmax  AND  |S| <= e3/7000
        # (the second is the numerical-stability bound "kxz max = ah slope max
        # =< e1 e3/(pi**2 2 dt)", hardcoded 7.e+3 in NEMO — binding only in
        # thin near-surface cells: e3=10 m => cap 1.4e-3 < rn_slpmax).
        # dz_half_ref IS 0.5*(dz_ref[:-1]+dz_ref[1:]) (review DRY note).
        # NB NEMO caps per slope position (e3u for uslp, e3w for wslp); the
        # single w-spacing here is exact only while the cap binds in the
        # near-surface uniform cells — revisit for a stretched-interior
        # oracle using nemo_cap.
        _dz_iface = jnp.asarray(z_coord.dz_half_ref)
        _cap = jnp.minimum(
            jnp.asarray(cfg.S_max, dtype=S_x_raw.dtype),
            (_dz_iface / _NEMO_SLOPE_STAB_7E3).astype(S_x_raw.dtype),
        )[None, None, :]
        S_x_t = jnp.clip(S_x_raw, -_cap, _cap)
        S_y_t = jnp.clip(S_y_raw, -_cap, _cap)
        taper = jnp.ones_like(S_x_t)
    else:
        # DM95 tapering via shared helper (identical formula across grids).
        S_x_t, S_y_t, taper = dm95_taper(
            S_x_raw, S_y_raw, cfg.S_max, EPS, cfg.taper_width_frac,
            stop_gradient_taper=(adj_stab == "stop_gradient_taper"),
        )

    # NEMO ldfslp mixed-layer slope ramp (default OFF => byte-identical).
    if getattr(cfg, "nemo_mld_slope_ramp", False):
        if T is None or S is None or eos_fn is None:
            raise ValueError(
                "compute_isopycnal_slopes_latlon_cgrid: nemo_mld_slope_ramp=True "
                "requires T, S and eos_fn to build the mixed-layer-depth ramp."
            )
        S_x_t, S_y_t = _apply_nemo_mld_slope_ramp(
            S_x_t, S_y_t, T, S, mask, z_coord, eos_fn, cfg.mld_rho_c,
            cfg.mld_criterion, g=g, rho_0=rho_0,
            active_3d=getattr(z_coord, "is_active", None),
            jacobian=jacobian,
        )

    # NEMO ldfslp horizontal Shapiro smoother (default OFF => byte-identical).
    # ponytail: the 2D T-mask stands in for NEMO's per-level umask/vmask/wmask;
    # exact only for a flat-bottom domain (GYRE/DINO). Pass 3D masks when
    # partial-cell topography lands.
    if cfg.nemo_slope_shapiro:
        S_x_t, S_y_t = _shapiro_smooth_slopes(S_x_t, S_y_t, mask)
    return S_x_t, S_y_t, taper


# =====================================================================
# Tracer tendency (single tracer)
# =====================================================================

def gm_redi_tracer_tendency_latlon_cgrid(
    q: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    kappa_GM,
    kappa_Redi,
) -> jnp.ndarray:
    """GM+Redi tendency for a single tracer on the lat-lon C-grid.

    Uses the Griffies (1998) small-slope tensor.  Diagonal and
    off-diagonal horizontal fluxes are combined at faces and passed
    through a single ``divergence_cgrid`` call for conservation.

    Parameters
    ----------
    q : (n_lat, n_lon, nlev)
        Tracer field at cell centers.
    S_x, S_y : (n_lat, n_lon, nlev-1)
        Tapered isopycnal slopes at interfaces.
    mask : (n_lat, n_lon)
        Ocean mask.
    u_mask : (n_lat, n_lon+1)
        u-face mask (1 if both adjacent cells are ocean).
    v_mask : (n_lat+1, n_lon)
        v-face mask.
    z_coord : OceanZStarCoordinate
    jacobian : (n_lat, n_lon)
    grid : LatLonGrid
    kappa_GM : float, (n_lat, n_lon), or (n_lat, n_lon, nlev-1)
        GM transport coefficient [m^2/s].  Scalar, per-column 2-D, or a 3-D
        **interface** field (the 3-D EKE closure, used directly at the nlev-1
        interfaces where the centered scheme evaluates all fluxes).
    kappa_Redi : float, (n_lat, n_lon), or (n_lat, n_lon, nlev-1)
        Redi isopycnal diffusivity [m^2/s].  Same three forms as ``kappa_GM``.

    Returns
    -------
    tendency : (n_lat, n_lon, nlev)
    """
    nlev = q.shape[-1]
    dz_actual = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev)
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]  # (n_lat, n_lon, nlev-1)

    # The centered scheme evaluates ALL fluxes (F_x/F_y/F_z) at the nlev-1
    # interior interfaces, so kappa is needed at the interfaces (the W-grid).
    # Three cases (see ``_kappa_is_interface_3d``):
    #   * scalar              → pass through unchanged (bit-identical).
    #   * 2-D (n_lat, n_lon)  → broadcast over interfaces via a trailing axis
    #                            (the prognostic-EKE-2D / Visbeck / K_iso=K_gm
    #                            per-column override) — bit-identical to before.
    #   * 3-D (…, nlev-1)     → the 3-D EKE interface kappa, used DIRECTLY (it
    #                            already lives at the interfaces).
    if _kappa_is_interface_3d(kappa_GM, nlev):
        kappa_GM_b = kappa_GM
    elif isinstance(kappa_GM, jnp.ndarray) and kappa_GM.ndim == 2:
        kappa_GM_b = kappa_GM[:, :, jnp.newaxis]
    else:
        kappa_GM_b = kappa_GM
    if _kappa_is_interface_3d(kappa_Redi, nlev):
        kappa_Redi_b = kappa_Redi
    elif isinstance(kappa_Redi, jnp.ndarray) and kappa_Redi.ndim == 2:
        kappa_Redi_b = kappa_Redi[:, :, jnp.newaxis]
    else:
        kappa_Redi_b = kappa_Redi

    # --- Neumann-fill tracer before computing gradients ---
    q_filled = neumann_fill_cgrid(q, mask)

    # --- Horizontal tracer gradients at faces (3D-native) ---
    dq_dx_u = gradient_x_cgrid(q_filled, grid)  # (n_lat, n_lon+1, nlev) at u-faces
    dq_dy_v = gradient_y_cgrid(q_filled, grid)  # (n_lat+1, n_lon, nlev) at v-faces

    # --- Vertical tracer gradient at interfaces ---
    dq_dz_half = (q_filled[:, :, :-1] - q_filled[:, :, 1:]) / jnp.maximum(dz_half, _EPS_DIV)

    # ================================================================
    # Horizontal fluxes — computed at INTERFACES for exact cancellation
    # ================================================================
    # F_x = kappa_Redi * dq/dx + (kappa_Redi - kappa_GM) * S_x * dq/dz
    # F_y = kappa_Redi * dq/dy + (kappa_Redi - kappa_GM) * S_y * dq/dz
    #
    # KEY: Both the diagonal (kR * dq/dx) and off-diagonal (kR-kG)*S*dq/dz
    # terms are evaluated at INTERFACE levels before averaging to full
    # levels.  This ensures exact cancellation (dq/dx + S_x*dq/dz = 0)
    # when q is constant along isopycnals (e.g., linear EOS with T as
    # tracer).  The previous approach evaluated the diagonal at full
    # levels and the off-diagonal at interfaces, breaking the
    # cancellation and producing spurious cross-isopycnal diffusion.

    # Average horizontal tracer gradients from full levels to interfaces.
    # u-face gradient → cell center → interface
    dq_dx_center = 0.5 * (dq_dx_u[:, :-1, :] + dq_dx_u[:, 1:, :])  # (n_lat, n_lon, nlev)
    dq_dy_center = 0.5 * (dq_dy_v[:-1, :, :] + dq_dy_v[1:, :, :])
    dq_dx_half = 0.5 * (dq_dx_center[:, :, :-1] + dq_dx_center[:, :, 1:])  # (n_lat, n_lon, nlev-1)
    dq_dy_half = 0.5 * (dq_dy_center[:, :, :-1] + dq_dy_center[:, :, 1:])

    # Total horizontal Redi flux at interfaces (exact cancellation here).
    F_x_half = (kappa_Redi_b * dq_dx_half
                + (kappa_Redi_b - kappa_GM_b) * S_x * dq_dz_half)  # (n_lat, n_lon, nlev-1)
    F_y_half = (kappa_Redi_b * dq_dy_half
                + (kappa_Redi_b - kappa_GM_b) * S_y * dq_dz_half)

    # Average interface fluxes to full levels (zero-pad at surface/bottom).
    z_pad = jnp.zeros((*F_x_half.shape[:2], 1), dtype=F_x_half.dtype)
    F_x_full = 0.5 * (
        jnp.concatenate([z_pad, F_x_half], axis=-1)
        + jnp.concatenate([F_x_half, z_pad], axis=-1)
    )  # (n_lat, n_lon, nlev)
    F_y_full = 0.5 * (
        jnp.concatenate([z_pad, F_y_half], axis=-1)
        + jnp.concatenate([F_y_half, z_pad], axis=-1)
    )

    # Interpolate cell-center fluxes to faces for divergence.
    F_x_u = interp_cell_to_uface(F_x_full)  # (n_lat, n_lon+1, nlev)
    F_y_v = interp_cell_to_vface(F_y_full)   # (n_lat+1, n_lon, nlev)

    # Apply face masks (zero flux through land boundaries).
    F_x_u = F_x_u * u_mask[:, :, jnp.newaxis]
    F_y_v = F_y_v * v_mask[:, :, jnp.newaxis]

    # Single conservative FV divergence.
    dq_h = divergence_cgrid(F_x_u, F_y_v, grid)

    # ================================================================
    # Vertical flux at interfaces
    # ================================================================
    # F_z = (kR + kG) * (S_x*dq/dx_center + S_y*dq/dy_center) + kR * S^2 * dq/dz
    # dq_dx_half, dq_dy_half already computed above (reused here).

    S2_half = S_x ** 2 + S_y ** 2
    F_z = ((kappa_Redi_b + kappa_GM_b) * (S_x * dq_dx_half + S_y * dq_dy_half)
           + kappa_Redi_b * S2_half * dq_dz_half)

    # Vertical flux divergence via shared helper (zero-flux BCs at surface/bottom).
    dq_vert = vertical_flux_divergence(F_z, dz_actual, EPS)

    # ================================================================
    # Total tendency, masked
    # ================================================================
    tendency = (dq_h + dq_vert) * mask[:, :, jnp.newaxis]
    return tendency


def nemo_eiv_bolus_transport(
    kappa_GM,
    wslpi_kp1: jnp.ndarray,
    wslpj_kp1: jnp.ndarray,
    e2u: jnp.ndarray,
    e1v: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    act: jnp.ndarray,
    act_below: jnp.ndarray,
    out_shape: tuple,
    dtype,
    kappa_face_average: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """NEMO ``ldf_eiv_trp_MLF`` eddy-induced (GM bolus) TRANSPORT, curl form.

    Returns the discrete eddy-induced transport ``(u_eiv, v_eiv, w_eiv_kp1)``
    [m^3/s] as the CURL of the bolus streamfunction

        ψ_uw(iface below cell k) = -e2u · mi(wslpi_{k+1}) · mk(aeiu) · wumask
        ψ_vw(iface below cell j) = -e1v · mj(wslpj_{k+1}) · mk(aeiu) · wvmask

    with the cell-level horizontal transport ``u_eiv = ψ_below − ψ_above`` and
    the interface-below-cell vertical transport
    ``w_eiv_kp1 = Δi(ψ_uw) + Δj(ψ_vw)``.  Being a discrete curl, the field is
    divergence-free by construction (the 3-D divergence telescopes to zero),
    so advecting a tracer by it conserves the column/domain integral exactly
    regardless of the ψ masking.

    This is the SINGLE definition of the bolus transport.  The centred
    in-operator GM path (below) consumes it as an explicit 2nd-order flux
    ``zfu -= u_eiv·t_u``; the ``through_fct`` path exports the same
    ``(u_eiv, v_eiv, w_eiv)`` to the model step, which adds it to the
    advecting mass flux so the bolus passes through the monotone FCT limiter
    (NEMO ``traadv``: the eiv velocity is added to the advecting velocity
    BEFORE the tracer scheme).  Axes: ``(lat=jj, lon=ji, lev=jk)``.
    """
    ax_y, ax_x, ax_z = 0, 1, 2
    if isinstance(kappa_GM, jnp.ndarray) and kappa_GM.ndim == 3:
        aeiu = jnp.broadcast_to(kappa_GM, out_shape)
    elif isinstance(kappa_GM, jnp.ndarray) and kappa_GM.ndim == 2:
        aeiu = jnp.broadcast_to(kappa_GM[:, :, jnp.newaxis], out_shape)
    else:
        aeiu = jnp.broadcast_to(jnp.asarray(kappa_GM, dtype=dtype), out_shape)
    # NEMO averages kappa onto EACH FACE separately before building psi
    # (ldftra.F90:715-718):
    #     zaeiu(ji,jj) = 0.5*( zaeiw(ji,jj) + zaeiw(ji+1,jj) ) * ssumask
    #     zaeiv(ji,jj) = 0.5*( zaeiw(ji,jj) + zaeiw(ji,jj+1) ) * ssvmask
    # Reusing the cell-centred kappa for both faces (the legacy default) is
    # EXACT only for a constant kappa; the Treguier kappa is spatially 2-D, so
    # it leaves a half-cell offset in the bolus transport.  Opt-in so the
    # legacy path stays bit-identical.
    if kappa_face_average:
        aeiu_u = 0.5 * (aeiu + jnp.roll(aeiu, -1, ax_x))       # -> u-face
        aeiu_v = 0.5 * (aeiu + jnp.roll(aeiu, -1, ax_y))       # -> v-face
    else:
        aeiu_u = aeiu_v = aeiu
    aeiu_if_u = 0.5 * (aeiu_u + jnp.roll(aeiu_u, -1, ax_z))   # mk() at iface below k
    aeiu_if_v = 0.5 * (aeiu_v + jnp.roll(aeiu_v, -1, ax_z))
    wslpi_u = 0.5 * (wslpi_kp1 + jnp.roll(wslpi_kp1, -1, ax_x))  # mi(wslpi) -> u-face
    wslpj_v = 0.5 * (wslpj_kp1 + jnp.roll(wslpj_kp1, -1, ax_y))
    act_kp1 = act_below
    wumask_uw = (u_mask[:, 1:, jnp.newaxis] * act * jnp.roll(act, -1, ax_x)
                 * act_kp1 * jnp.roll(act_kp1, -1, ax_x))
    wvmask_vw = (v_mask[1:, :, jnp.newaxis] * act * jnp.roll(act, -1, ax_y)
                 * act_kp1 * jnp.roll(act_kp1, -1, ax_y))
    psi_uw = -(e2u[:, :, jnp.newaxis] * wslpi_u * aeiu_if_u * wumask_uw)
    psi_vw = -(e1v[:, :, jnp.newaxis] * wslpj_v * aeiu_if_v * wvmask_vw)
    psi_uw_top = jnp.roll(psi_uw, +1, ax_z).at[:, :, 0].set(0.0)
    psi_vw_top = jnp.roll(psi_vw, +1, ax_z).at[:, :, 0].set(0.0)
    u_eiv = psi_uw - psi_uw_top
    v_eiv = psi_vw - psi_vw_top
    w_eiv_kp1 = ((psi_uw - jnp.roll(psi_uw, +1, ax_x))
                 + (psi_vw - jnp.roll(psi_vw, +1, ax_y)))
    return u_eiv, v_eiv, w_eiv_kp1


def nemo_iso_face_masks(u_mask, v_mask, act):
    """NEMO-convention 3-D face masks from the 2-D walls + 3-D wet mask.

    ``umask[j,i,k]`` = east u-face of cell i wet at level k (wall open AND
    both bracketing cells wet); ``vmask`` the north analogue;
    ``wmask(k) = tmask(k)·tmask(k-1)``, ``wmask(0)=tmask(0)``.

    Shared by the explicit ``nemo_iso_lap`` operator and the implicit K33
    (traldf_iso_a33) so the two sides of the explicit/implicit split build
    their stencils from IDENTICAL masks (#1226).
    """
    umask = u_mask[:, 1:, jnp.newaxis] * act * jnp.roll(act, -1, axis=1)
    vmask = v_mask[1:, :, jnp.newaxis] * act * jnp.roll(act, -1, axis=0)
    wmask = act * jnp.roll(act, +1, axis=2)
    wmask = wmask.at[:, :, 0].set(act[:, :, 0])
    return umask, vmask, wmask


def nemo_iso_w_kappa_sums(aht, umask, vmask, aht_v=None):
    """Masked 4-point kappa sums + wet counts for the traldf_iso w-point
    kappa average, in the a33 "above" convention: level pair (k-1, k),
    faces (i-1, i) / (j-1, j).

    NEMO masks ``ahtu/ahtv`` at build (``ldftra.F90:365``), so its raw
    4-sum is a WET-ONLY sum — transcribed here as masked-kappa-sum, with
    the wet count returned separately because the two consumers apply
    DIFFERENT wmask indices to the normalization (transcription detail):

      * a33 / implicit K33 (``traldf_iso_a33``): faces (k-1,k) with
        ``zmsku = wmask(k)/MAX(count,1)`` — use these fields directly.
      * explicit flux at the interface BELOW cell k
        (``traldf_iso_scheme.h90:109``): faces (k,k+1) with
        ``zmsku = wmask(k)/MAX(count,1)`` — i.e. ``roll(sum/count, -1)``
        in the level axis but wmask NOT rolled.

    ONE sum/count implementation shared by both so the explicit/implicit
    split can never diverge (#1226).  ``aht_v`` (default ``None`` -> reuse
    ``aht``) is NEMO's independently-evaluated ``ahtv`` (nn_aht_ijk_t=20:
    ``ahtv(ji,jj) = zUfac*MAX(e1v,e2v)**inn`` at the v-point, NOT a T-point
    field averaged onto the v-face) — every other closure (Visbeck/EKE/
    Treguier/GEOMETRIC) genuinely IS a T-point quantity face-broadcast the
    same way for u and v, so they leave ``aht_v=None`` and stay
    bit-identical.  Returns ``(ksum_u, cnt_u, ksum_v, cnt_v)``, all
    (n_lat, n_lon, nlev).
    """
    ax_y, ax_x, ax_z = 0, 1, 2
    up = lambda a: jnp.roll(a, +1, ax_z)     # level k-1 view
    aht_v_ = aht if aht_v is None else aht_v
    # Summation ORDER matters (FP non-associativity): use NEMO a33's literal
    # order  ahtu(i,k-1) + ahtu(i-1,k) + ahtu(i-1,k-1) + ahtu(i,k)  — which,
    # rolled to the explicit flux's (k,k+1) pair, reproduces the operator's
    # pre-#1226 order  aht(k) + ah_im1(k+1) + ah_im1(k) + aht(k+1)  exactly
    # (interior masks are 1.0 and x*1.0 is exact), keeping the all-wet
    # interior byte-identical (codex round-2 #1).
    um_im1 = jnp.roll(umask, +1, ax_x)
    ah_im1 = jnp.roll(aht, +1, ax_x)
    A_u, B_u = aht * umask, ah_im1 * um_im1
    cnt_u = up(umask) + um_im1 + up(um_im1) + umask
    ksum_u = up(A_u) + B_u + up(B_u) + A_u
    vm_jm1 = jnp.roll(vmask, +1, ax_y)
    ah_jm1 = jnp.roll(aht_v_, +1, ax_y)
    A_v, B_v = aht_v_ * vmask, ah_jm1 * vm_jm1
    cnt_v = up(vmask) + vm_jm1 + up(vm_jm1) + vmask
    ksum_v = up(A_v) + B_v + up(B_v) + A_v
    return ksum_u, cnt_u, ksum_v, cnt_v


def nemo_iso_a33_e3w(z_coord, e3t, jacobian, dtype):
    """NEMO's ``e3w(:,:,:,Kmm)`` in ``traldf_iso_a33``'s "above" convention.

    ``traldf_iso.f90:831-833`` squares ``e3w_3d(ji,jj,jk)*(1+r3t(ji,jj,Kmm))``
    for ``akz`` and ``:285`` divides the explicit A33 flux by the same object
    one level down, with (``domzgr_substitute.h90:131``, ``:108``)

        e3w(i,j,k,t) = e3w_0(i,j,k) * (1 + r3t(i,j,t))
        e3w_0(k)     = gdept_0(k) - gdept_0(k-1)

    the T-POINT DEPTH DIFFERENCE, which on a stretched ladder is NOT the
    interface midpoint ``0.5*(e3t_k + e3t_{k-1})``.  Resolved through the
    SINGLE shared resolver the implicit tracer and momentum solves already
    use; that resolver returns ``None`` only for a coordinate whose T points
    ARE the midpoints, and raises rather than silently substituting for one
    where they are not — so the midpoint arm below is reached only where it
    is the same object.

    ONE implementation, called by BOTH sides of the explicit/implicit A33
    split (the MSC block of
    :func:`nemo_iso_lap_tracer_tendency_latlon_cgrid` and
    :func:`compute_isoneutral_K33_latlon`), so the two ``akz`` can never
    disagree — the same reason ``nemo_iso_a33`` itself is shared (#1226).
    """
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w0_reference
    raw = nemo_e3w0_reference(z_coord)
    if raw is None:
        e3w = 0.5 * (jnp.roll(e3t, +1, 2) + e3t)
        # surface w-point (unused downstream: wslp(0) = 0)
        return e3w.at[:, :, 0].set(e3t[:, :, 0])
    # jacobian is the (1 + r3t) stretch e3t itself already carries.
    # No trailing-axis slice: nemo_e3w0_reference already REFUSES a field
    # whose trailing size is not n_levels, so a slice here could only mask a
    # real mismatch (diff review finding 8).
    return (jnp.asarray(raw, dtype=dtype)
            * jnp.asarray(jacobian, dtype=dtype)[:, :, jnp.newaxis])


def nemo_iso_a33(aht, umask, vmask, wmask, wslpi, wslpj,
                 e1u_c, e2v_c, e3w2, dt=None, msc: bool = False, aht_v=None,
                 evaluation: str = "normalized_square"):
    """``traldf_iso_a33`` in the "above" (k-1,k) convention: the a33 element
    of the rotated tensor and its explicit/implicit split.

    Returns ``(ah_wslp2, akz)`` at the w-point at the TOP of cell k:

      * ``msc=False`` (``ln_traldf_msc=F``): ``akz = ah_wslp2`` — the FULL
        diagonal goes implicit and the explicit A33 flux coefficient
        ``ah_wslp2 - akz`` is zero.
      * ``msc=True`` (``ln_traldf_msc=T`` — the DINO namelist): NEMO's Method
        of Stabilizing Correction, verbatim (``traldf_iso.F90:314-333``):
        ``akz_h = 0.25·Σ4( ahtu/e1u² + ahtv/e2v² )`` (per-face metric, level
        pair (k-1,k); NEMO ships the 0.25 form — the ``!!gm BUG?`` note about
        zmsku is NOT in the executed code, so it is NOT transcribed), then
        ``akz = MAX( dt·(akz_h + ah_wslp2/e3w²) − ½, 0 )·e3w²/dt`` — the
        implicit part, leaving the explicit remainder ``ah_wslp2 − akz``
        bounded by the ½ vertical-CFL limit.

    ``aht`` contributions are face-masked (NEMO masks aht at build,
    ``ldftra.F90:365``).  ``e1u_c``/``e2v_c``: (n_lat, n_lon) east-face /
    north-face metrics of cell (j,i) (the operator's ``e1u``/``e2v``);
    ``e3w2``: squared w-thickness at the top-of-cell-k w-point.  ONE
    implementation consumed by the explicit operator (rolled to its (k,k+1)
    flux convention) and the implicit-K33 getter, so the split cannot
    diverge (#1226).  ``aht_v`` (default ``None`` -> reuse ``aht``): see
    ``nemo_iso_w_kappa_sums``.
    """
    if evaluation not in ("normalized_square", "nemo_literal"):
        raise ValueError(
            "nemo_iso_a33 evaluation must be 'normalized_square' or "
            f"'nemo_literal', got {evaluation!r}")
    ax_y, ax_x, ax_z = 0, 1, 2
    up = lambda a: jnp.roll(a, +1, ax_z)
    aht_v_ = aht if aht_v is None else aht_v
    ksum_u, cnt_u, ksum_v, cnt_v = nemo_iso_w_kappa_sums(aht, umask, vmask, aht_v=aht_v)
    zahu_w = ksum_u * (wmask / jnp.maximum(cnt_u, 1.0))
    zahv_w = ksum_v * (wmask / jnp.maximum(cnt_v, 1.0))
    if evaluation == "nemo_literal":
        # traldf_iso.F90:296-297: preserve Fortran's left association.
        ah_wslp2 = (zahu_w * wslpi) * wslpi + (zahv_w * wslpj) * wslpj
    else:
        ah_wslp2 = zahu_w * wslpi ** 2 + zahv_w * wslpj ** 2
    if not msc:
        return ah_wslp2, ah_wslp2
    if dt is None:
        raise ValueError(
            "nemo_iso_a33: msc=True (ln_traldf_msc) requires dt (rDt) for "
            "the akz stability threshold.")
    inv_e1u2 = (1.0 / (e1u_c ** 2))[:, :, jnp.newaxis]
    inv_e2v2 = (1.0 / (e2v_c ** 2))[:, :, jnp.newaxis]
    inv_e1u2_im1 = jnp.roll(inv_e1u2, +1, ax_x)
    inv_e2v2_jm1 = jnp.roll(inv_e2v2, +1, ax_y)
    ahu = aht * umask
    ahu_im1 = jnp.roll(ahu, +1, ax_x)
    ahv = aht_v_ * vmask
    ahv_jm1 = jnp.roll(ahv, +1, ax_y)
    # a33 msc akz_h, level pair (k, k-1) per face, per-face metric, x0.25.
    akz_h = 0.25 * (
        (ahu + up(ahu)) * inv_e1u2
        + (ahu_im1 + up(ahu_im1)) * inv_e1u2_im1
        + (ahv + up(ahv)) * inv_e2v2
        + (ahv_jm1 + up(ahv_jm1)) * inv_e2v2_jm1
    )
    zcoef0 = dt * (akz_h + ah_wslp2 / e3w2)
    akz = jnp.maximum(zcoef0 - 0.5, 0.0) * e3w2 / dt
    return ah_wslp2, akz


def nemo_iso_lap_tracer_tendency_latlon_cgrid(
    q: jnp.ndarray,
    S_x: jnp.ndarray,
    S_y: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    kappa_Redi,
    active_3d: jnp.ndarray | None = None,
    native_slopes: tuple | None = None,
    msc_stabilize: bool = False,
    dt: float | None = None,
    kappa_GM=None,
    gm_bolus_advection: str = "centred",
    gm_bolus_kappa_face_average: bool = False,
    return_bolus: bool = False,
    kappa_Redi_v=None,
    msc_e3w_override: jnp.ndarray | None = None,
    face_thickness_u: jnp.ndarray | None = None,
    face_thickness_v: jnp.ndarray | None = None,
    bolus_native_slopes: tuple | None = None,
    vertical_skew_evaluation: str = "normalized_sums",
    a33_evaluation: str = "normalized_square",
    return_diagnostics: bool = False,
    return_operand_diagnostics: bool = False,
    divisor_thickness: jnp.ndarray | None = None,
    closed_bottom_wmask: bool = True,
    horizontal_flux_evaluation: str = "vectorized",
    area_reciprocal: jnp.ndarray | None = None,
    area_reciprocal_evaluation: str = "vectorized",
    final_update_evaluation: str = "masked",
    rhs_accumulator: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """NEMO ``traldf_iso`` (``#define iso_lap``) iso-neutral Laplacian Redi
    tracer tendency on the lat-lon C-grid.

    When ``kappa_GM`` is supplied (NEMO ``ln_ldfeiv``), the Gent-McWilliams
    eddy-induced (bolus) transport is added as an advective flux — a faithful
    port of NEMO's ``ldf_eiv_trp_MLF`` streamfunction form
    (``ψ_uw = -¼·e2u·mi(wslpi)·mk(aeiu)·wumask``; eiv transport = curl(ψ); the
    tracer is centered-advected by it).  The bolus streamfunction is divergence-
    free by construction, so the GM tendency conserves the column/domain tracer
    integral (verified by ``test_nemo_iso_lap_gm_conserves``).

    This is a *faithful* port of NEMO 5.0.2's standard rotated-Laplacian
    iso-neutral operator (``cfgs/*/WORK/traldf_iso_scheme.h90``), verified
    term-by-term against NEMO's dumped ``ttrd_ldf`` (GYRE oracle, T corr
    0.9997 fed NEMO's own slopes; 0.96 fed legoESM's centered slopes — the
    ``ISO_LAP_OPERATOR_FINDINGS`` proof).  It is the *skew / off-diagonal*
    iso-neutral part: with ``ln_traldf_msc=F`` NEMO's explicit K33 diagonal
    term is zero (K33 goes to the implicit vertical solve), so this operator
    reproduces exactly the explicit ``ttrd_ldf`` NEMO dumps.

    Pure Redi (``κ_GM`` is NOT part of ``traldf_iso``); the dispatcher raises
    if ``kappa_GM != 0`` is requested with this scheme.

    Convention (matches NEMO's native stencil, remapped to legoESM axes):
    ``axis0 = lat = NEMO jj``, ``axis1 = lon = NEMO ji``, ``axis2 = lev = jk``
    (k increases downward, k=0 surface — same index ordering as NEMO).  Fluxes
    are assembled as ``zfu`` (east face of cell i), ``zfv`` (north face of
    cell j), ``zfw`` (w-interface below cell k), and the tendency is the
    ``+``-signed 3-D flux divergence ``(Δi zfu + Δj zfv + Δk zfw)/(e1e2t·e3t)``
    — diffusion, so the mask-weighted volume integral of ``q`` is conserved.

    Parameters
    ----------
    q : (n_lat, n_lon, nlev)
        Tracer at cell centres.
    S_x, S_y : (n_lat, n_lon, nlev-1)
        Tapered iso-neutral interface slopes (cell-centre w-interfaces).
        v1 maps this single interface field onto NEMO's four native slope
        positions (uslp/vslp at tracer levels, wslpi/wslpj at w-levels) —
        the mode-(b) placement that scored corr 0.96.
    mask : (n_lat, n_lon) surface ocean mask (1=ocean).
    u_mask : (n_lat, n_lon+1) u-face mask; v_mask : (n_lat+1, n_lon) v-face.
    z_coord : OceanZStarCoordinate
    jacobian : (n_lat, n_lon) z-star Jacobian.
    grid : LatLonGrid (or LatLonCGridGeometry) — supplies e1/e2 metrics.
    kappa_Redi : float or (n_lat, n_lon[, nlev]) iso-neutral diffusivity
        [m^2/s] (NEMO ``ahtu=ahtv``).
    kappa_Redi_v : same shape options as ``kappa_Redi``, or ``None``.
        NEMO's independently-evaluated ``ahtv`` (nn_aht_ijk_t=20:
        ``ahtv(ji,jj)=zUfac*MAX(e1v,e2v)**inn`` at the v-point) when it is
        NOT simply ``ahtu`` broadcast onto both faces.  ``None`` (default,
        every closure except the static lat-scaling override) reuses
        ``kappa_Redi`` for the v-face too — bit-identical to before.
    active_3d : (n_lat, n_lon, nlev) or None
        Per-cell wet mask (1=water, 0=below seafloor).  Supplies NEMO's
        vertical ``tmask`` extent so the sub-seafloor dry level (which
        carries a garbage 0 tracer) cannot leak a spurious across-floor
        vertical gradient into the deepest wet cell.  ``None`` ⇒ assume
        every level of a wet column is water (full-depth flat bottom) —
        only correct when ``q`` has no below-bathymetry levels.
    closed_bottom_wmask : bool
        ``True`` by default for every lat-lon C-grid caller. NEMO's
        ``traldf_iso`` closes the W mask below the deepest wet tracer cell
        before forming the horizontal/vertical tensor pair; ``False`` exists
        only as a fidelity discriminator.

    Returns
    -------
    tendency : (n_lat, n_lon, nlev)  [tracer-units / s], land-masked.

    Assumptions / limitations (v1)
    ------------------------------
    - Flat-bottom / z-coordinate: horizontal walls come from the 2-D
      ``u_mask``/``v_mask``; the vertical (bottom) extent comes from
      ``active_3d`` (built by the dispatcher from ``H_bathy``+``z_coord``).
      Uniform-depth columns assumed (cell-i active ⟺ cell-(i±1) active) —
      no interior partial-cell steps.
    - Single interface slope field placed at all four NEMO slope positions
      (mode-(b) approximation; corr 0.96, amplitude ~1.35).  The documented
      follow-up is native four-position slopes to tighten amplitude toward 1.
    """
    # ponytail: v1 places legoESM's single interface slope field (S_x,S_y) at
    # both the u/v-point (tracer level) and w-point NEMO positions — the
    # mode-(b) 0.96 placement.  Native four-position slopes (uslp/vslp@tracer,
    # wslpi/wslpj@w) are the documented follow-up to push amplitude toward 1.0.
    nlev = q.shape[-1]
    dtype = q.dtype
    if return_operand_diagnostics and not return_diagnostics:
        raise ValueError(
            "return_operand_diagnostics=True requires return_diagnostics=True")
    if vertical_skew_evaluation not in ("normalized_sums", "nemo_literal"):
        raise ValueError(
            "vertical_skew_evaluation must be 'normalized_sums' or "
            f"'nemo_literal', got {vertical_skew_evaluation!r}")
    if a33_evaluation not in ("normalized_square", "nemo_literal"):
        raise ValueError(
            "a33_evaluation must be 'normalized_square' or 'nemo_literal', "
            f"got {a33_evaluation!r}")
    if horizontal_flux_evaluation not in (
            "vectorized", "nemo_pairwise", "nemo_literal",
            "nemo_literal_differences", "nemo_literal_horizontal_sum",
            "nemo_literal_flux_sum", "nemo_literal_divergence"):
        raise ValueError(
            "horizontal_flux_evaluation must be 'vectorized', "
            "'nemo_pairwise', 'nemo_literal', "
            "'nemo_literal_differences', 'nemo_literal_horizontal_sum', "
            "'nemo_literal_flux_sum' or 'nemo_literal_divergence', got "
            f"{horizontal_flux_evaluation!r}")
    if area_reciprocal_evaluation not in ("vectorized", "nemo_stored"):
        raise ValueError(
            "area_reciprocal_evaluation must be 'vectorized' or "
            f"'nemo_stored', got {area_reciprocal_evaluation!r}")
    if (area_reciprocal is not None
            and area_reciprocal_evaluation != "vectorized"):
        raise ValueError(
            "area_reciprocal and a non-vectorized "
            "area_reciprocal_evaluation are mutually exclusive")
    if final_update_evaluation not in (
            "masked", "unmasked", "nemo_rhs_increment"):
        raise ValueError(
            "final_update_evaluation must be 'masked', 'unmasked' or "
            f"'nemo_rhs_increment', got {final_update_evaluation!r}")
    if ((rhs_accumulator is None)
            != (final_update_evaluation != "nemo_rhs_increment")):
        raise ValueError(
            "rhs_accumulator must be supplied exactly when "
            "final_update_evaluation='nemo_rhs_increment'")
    ones_z = jnp.ones((1, 1, nlev), dtype=dtype)
    if (face_thickness_u is None) != (face_thickness_v is None):
        raise ValueError(
            "face_thickness_u and face_thickness_v must be supplied together")

    # --- Metrics (n_lat, n_lon).  NEMO e1u/e2u/e1v/e2v are co-located at the
    # u/v-point of cell i/j; map legoESM's (n_lon+1)/(n_lat+1) face metrics to
    # the east-face-of-cell-i / north-face-of-cell-j cell arrays.  ensure_geometry
    # promotes a bare LatLonGrid to the C-grid geometry carrying dx_T/dx_u/… ---
    from legoesm.grids.latlon import ensure_geometry
    geom = ensure_geometry(grid)
    e1t, e2t = geom.dx_T, geom.dy_T                  # (n_lat, n_lon)
    e1u, e2u = geom.dx_u[:, 1:], geom.dy_u[:, 1:]    # east face of cell i
    e1v, e2v = geom.dx_v[1:, :], geom.dy_v[1:, :]    # north face of cell j
    # e3 at tracer level (flat zco: e3u=e3v=e3t=dz_ref·jacobian).
    e3t = z_coord.dz_ref[None, None, :] * jacobian[:, :, jnp.newaxis]  # (n_lat,n_lon,nlev)

    # --- Cell wet mask (NEMO tmask): full-depth flat bottom if not supplied.
    if active_3d is None:
        act = mask[:, :, jnp.newaxis] * ones_z
    else:
        act = active_3d.astype(dtype)
    # --- Cell face masks in NEMO convention.  NEMO umask(i)=east u-point of
    # T(i)=legoESM u-face i+1 = tmask(i)·tmask(i+1): wet iff the horizontal wall
    # is open (u_mask) AND BOTH adjacent cells are wet at level k.  Requiring the
    # neighbour's activity (roll of ``act``) is byte-identical on a flat bottom
    # (act column-uniform ⇒ act(i)==act(i+1)) but prevents a silent flux leak
    # into a dry cell at a lateral bathymetry step (a topographic column next to
    # a shallower one) — the "stale face mask → mass leak" footgun.
    # Shared with the implicit-K33 side (#1226): one mask construction.
    umask, vmask, wmask = nemo_iso_face_masks(u_mask, v_mask, act)

    # --- Diffusivity as a 3-D field (NEMO ahtu=ahtv=aht).  Broadcast a
    # scalar / 2-D per-column / 3-D interface kappa to (n_lat,n_lon,nlev). ---
    if isinstance(kappa_Redi, jnp.ndarray) and kappa_Redi.ndim == 3:
        aht = jnp.broadcast_to(kappa_Redi, q.shape)
    elif isinstance(kappa_Redi, jnp.ndarray) and kappa_Redi.ndim == 2:
        aht = jnp.broadcast_to(kappa_Redi[:, :, jnp.newaxis], q.shape)
    else:
        aht = jnp.broadcast_to(jnp.asarray(kappa_Redi, dtype=dtype), q.shape)
    # NEMO's ahtv is NOT ahtu broadcast onto the v-face (nn_aht_ijk_t=20
    # evaluates ahtu/ahtv independently at their own U/V points); kappa_Redi_v
    # supplies that distinct v-face value when the caller has one (only the
    # static lat-scaling override, below), else reuse aht (bit-identical).
    if kappa_Redi_v is None:
        aht_v = aht
    elif isinstance(kappa_Redi_v, jnp.ndarray) and kappa_Redi_v.ndim == 3:
        aht_v = jnp.broadcast_to(kappa_Redi_v, q.shape)
    elif isinstance(kappa_Redi_v, jnp.ndarray) and kappa_Redi_v.ndim == 2:
        aht_v = jnp.broadcast_to(kappa_Redi_v[:, :, jnp.newaxis], q.shape)
    else:
        aht_v = jnp.broadcast_to(jnp.asarray(kappa_Redi_v, dtype=dtype), q.shape)
    # NEMO masks the diffusivity ONCE, at build:
    #   ldftra.f90:433-434   ahtu(:,:,1:jpkm1) = ahtu(:,:,1:jpkm1) * umask(...)
    #                        ahtv(:,:,1:jpkm1) = ahtv(:,:,1:jpkm1) * vmask(...)
    # so every consumer sees a face-masked coefficient.  That masking is
    # LOAD-BEARING on the horizontal flux: NEMO's uslp is a 16-point Shapiro
    # smear of the umask-ed raw slope (ldfslp.f90:288 masks zwz, :298 smears
    # it), so uslp is generally NONZERO on a closed u-face, and only
    # ahtu = 0 stops traldf_iso.f90:242 emitting the zA13 term there.
    # Applying it here makes the operator's four consumers (zfu, zfv, the
    # w-point kappa sums and akz_h) read ONE masked field; the latter three
    # already re-applied the same 0/1 mask, so this is bit-identical for them.
    aht = aht * umask
    aht_v = aht_v * vmask

    # --- Slope positions.  native_slopes = the ldfslp four-position fields
    # (uslp/vslp at tracer levels, wslpi/wslpj at top-of-cell w-points, NEMO
    # sign convention, from compute_nemo_native_slopes) — the exact traldf_iso
    # stencil.  None => mode-(b): the single interface field at all four
    # positions (corr 0.96, amplitude ~1.35 — the documented v1). ---
    if native_slopes is not None:
        uslp, vslp, wslpi, wslpj = native_slopes
    else:
        # Interface k (between cells k,k+1) -> native tracer level k; deepest
        # native level = 0 (no interface below the bottom cell).
        zpad = jnp.zeros((*S_x.shape[:2], 1), dtype=dtype)
        uslp = jnp.concatenate([S_x, zpad], axis=2)  # (n_lat, n_lon, nlev)
        vslp = jnp.concatenate([S_y, zpad], axis=2)
        wslpi, wslpj = uslp, vslp
    if bolus_native_slopes is None:
        bolus_wslpi, bolus_wslpj = wslpi, wslpj
    else:
        _, _, bolus_wslpi, bolus_wslpj = bolus_native_slopes

    ax_y, ax_x, ax_z = 0, 1, 2

    # ---- masked tracer gradients (traldf_iso_scheme.h90 top block) ----
    zdit = (jnp.roll(q, -1, ax_x) - q) * umask       # (T(i+1)-T(i))·umask
    zdjt = (jnp.roll(q, -1, ax_y) - q) * vmask
    zdkt = (jnp.roll(q, +1, ax_z) - q) * wmask       # (T(k-1)-T(k))·wmask
    zdkt = zdkt.at[:, :, 0].set(0.0)                 # surface w-level = 0

    # ================= HORIZONTAL fluxes (A11 + A13, A22 + A23) =============
    if face_thickness_u is None:
        e3u_flux = e3t
        e3v_flux = e3t
    else:
        e3u_flux = jnp.asarray(face_thickness_u, dtype=dtype)
        e3v_flux = jnp.asarray(face_thickness_v, dtype=dtype)
        if e3u_flux.shape != q.shape or e3v_flux.shape != q.shape:
            raise ValueError(
                "face thickness overrides must have tracer shape "
                f"{q.shape}; got {e3u_flux.shape} and {e3v_flux.shape}")
    zA11 = (e2u / e1u)[:, :, jnp.newaxis] * e3u_flux
    zA22 = (e1v / e2v)[:, :, jnp.newaxis] * e3v_flux
    # zmsku = 1/max(Σ4 wmask around the u-face vertical pair, 1)
    wm_ip1 = jnp.roll(wmask, -1, ax_x)
    wm_kp1 = jnp.roll(wmask, -1, ax_z)
    if closed_bottom_wmask:
        # NEMO's deepest tracer level reads the closed jpk W level, not the
        # surface W level.  This private discriminator prevents the periodic
        # vertical roll from wrapping that surface mask onto the floor
        # (traldf_iso.f90:246-249).
        wm_kp1 = wm_kp1.at[:, :, -1].set(0.0)
    wm_ip1_kp1 = jnp.roll(wm_kp1, -1, ax_x)
    zmsku_h = 1.0 / jnp.maximum(wm_ip1 + wm_kp1 + wm_ip1_kp1 + wmask, 1.0)
    wm_jp1 = jnp.roll(wmask, -1, ax_y)
    wm_jp1_kp1 = jnp.roll(wm_kp1, -1, ax_y)
    zmskv_h = 1.0 / jnp.maximum(wm_jp1 + wm_kp1 + wm_jp1_kp1 + wmask, 1.0)

    zA13 = -e2u[:, :, jnp.newaxis] * uslp * zmsku_h
    zA23 = -e1v[:, :, jnp.newaxis] * vslp * zmskv_h

    # 4-pt vertical-gradient average around the u-face / v-face
    zdkt_kp1 = jnp.roll(zdkt, -1, ax_z)
    literal_flux = horizontal_flux_evaluation.startswith("nemo_literal")
    if horizontal_flux_evaluation == "nemo_pairwise" or literal_flux:
        # traldf_iso.f90:254-259: NEMO requires the two explicit pairs for
        # halo/fold compatibility.  Keep this private selector until the
        # production-step discriminator closes the complete statement.
        avg4_u = ((jnp.roll(zdkt, -1, ax_x) + zdkt_kp1)
                  + (jnp.roll(zdkt_kp1, -1, ax_x) + zdkt))
        avg4_v = ((jnp.roll(zdkt, -1, ax_y) + zdkt_kp1)
                  + (jnp.roll(zdkt_kp1, -1, ax_y) + zdkt))
    else:
        avg4_u = (jnp.roll(zdkt, -1, ax_x) + zdkt_kp1
                  + jnp.roll(zdkt_kp1, -1, ax_x) + zdkt)
        avg4_v = (jnp.roll(zdkt, -1, ax_y) + zdkt_kp1
                  + jnp.roll(zdkt_kp1, -1, ax_y) + zdkt)

    if literal_flux:
        # Materialize every source operation in traldf_iso.f90:254-259.  Bare
        # parentheses are reassociated by the production XLA closure; the
        # shared identity helper retains each gfortran rounding boundary.
        diag_u = nemo_source_round(zA11 * zdit)
        cross_u = nemo_source_round(zA13 * nemo_source_round(avg4_u))
        diag_v = nemo_source_round(zA22 * zdjt)
        cross_v = nemo_source_round(zA23 * nemo_source_round(avg4_v))
        zfu = nemo_source_round(
            aht * nemo_source_round(diag_u + cross_u))
        zfv = nemo_source_round(
            aht_v * nemo_source_round(diag_v + cross_v))
    else:
        zfu = aht * (zA11 * zdit + zA13 * avg4_u)
        zfv = aht_v * (zA22 * zdjt + zA23 * avg4_v)

    # ================= VERTICAL flux zfw at w-level jk+1 (A31 + A32) ========
    # Shared a33 kappa sums (#1226): faces (k,k+1) here = the "above"
    # sums at k+1 (roll -1); the wmask factor stays AT k — NEMO's
    # scheme.h90:109 zmsku uses wmask(jk) with the (jk,jk+1) face pair
    # (transcription detail; NOT a pure shift of the a33 stencil).
    # NEMO masks aht at build (ldftra:365), so the shared masked-sum /
    # wet-count IS scheme.h90's (masked 4-sum)·zmsku — the previous
    # inline version summed UNMASKED kappa (4k/N at an N-wet-face wall
    # vs the K33's k: the residual split mismatch).
    if vertical_skew_evaluation == "nemo_literal":
        # traldf_iso_scheme.h90:109-120, in source association.  This cannot
        # reuse the normalized shared K33 sums: NEMO forms the four Kmm face
        # values as two explicit pairs, applies zmsk, then applies zmsk again
        # in zA31/zA32.  The distinction is one-ULP class but is amplified by
        # the O(1e4) metric and gradient factors in the vertical skew flux.
        def _pair4(a, b, c, d):
            return (a + b) + (c + d)

        um_w = jnp.roll(umask, +1, ax_x)
        um_kp1 = jnp.roll(umask, -1, ax_z)
        um_w_kp1 = jnp.roll(um_kp1, +1, ax_x)
        vm_s = jnp.roll(vmask, +1, ax_y)
        vm_kp1 = jnp.roll(vmask, -1, ax_z)
        vm_s_kp1 = jnp.roll(vm_kp1, +1, ax_y)
        zmsku_w = wmask / jnp.maximum(
            _pair4(umask, um_w_kp1, um_w, um_kp1), 1.0)
        zmskv_w = wmask / jnp.maximum(
            _pair4(vmask, vm_s_kp1, vm_s, vm_kp1), 1.0)
        aht_masked = aht * umask
        ahtv_masked = aht_v * vmask
        aht_w = jnp.roll(aht_masked, +1, ax_x)
        aht_kp1 = jnp.roll(aht_masked, -1, ax_z)
        aht_w_kp1 = jnp.roll(aht_kp1, +1, ax_x)
        ahtv_s = jnp.roll(ahtv_masked, +1, ax_y)
        ahtv_kp1 = jnp.roll(ahtv_masked, -1, ax_z)
        ahtv_s_kp1 = jnp.roll(ahtv_kp1, +1, ax_y)
        zahu_w = _pair4(
            aht_masked, aht_w_kp1, aht_w, aht_kp1) * zmsku_w
        zahv_w = _pair4(
            ahtv_masked, ahtv_s_kp1, ahtv_s, ahtv_kp1) * zmskv_w
    else:
        _ksum_u, _cnt_u, _ksum_v, _cnt_v = nemo_iso_w_kappa_sums(
            aht, umask, vmask, aht_v=aht_v)
        zmsku_w = wmask / jnp.maximum(jnp.roll(_cnt_u, -1, ax_z), 1.0)
        zmskv_w = wmask / jnp.maximum(jnp.roll(_cnt_v, -1, ax_z), 1.0)
        zahu_w = jnp.roll(_ksum_u, -1, ax_z) * zmsku_w
        zahv_w = jnp.roll(_ksum_v, -1, ax_z) * zmskv_w

    wslpi_kp1 = jnp.roll(wslpi, -1, ax_z)            # wslpi(jk+1)
    wslpj_kp1 = jnp.roll(wslpj, -1, ax_z)
    bolus_wslpi_kp1 = jnp.roll(bolus_wslpi, -1, ax_z)
    bolus_wslpj_kp1 = jnp.roll(bolus_wslpj, -1, ax_z)
    zA31 = -zahu_w * e2t[:, :, jnp.newaxis] * zmsku_w * wslpi_kp1
    zA32 = -zahv_w * e1t[:, :, jnp.newaxis] * zmskv_w * wslpj_kp1

    # 4-pt horizontal-gradient average around the w-point (i, jk+1)
    zdit_kp1 = jnp.roll(zdit, -1, ax_z)
    zdjt_kp1 = jnp.roll(zdjt, -1, ax_z)
    if vertical_skew_evaluation == "nemo_literal":
        avg4_wi = ((zdit + jnp.roll(zdit_kp1, +1, ax_x))
                   + (jnp.roll(zdit, +1, ax_x) + zdit_kp1))
        avg4_wj = ((zdjt + jnp.roll(zdjt_kp1, +1, ax_y))
                   + (jnp.roll(zdjt, +1, ax_y) + zdjt_kp1))
    else:
        avg4_wi = (zdit + jnp.roll(zdit_kp1, +1, ax_x)
                   + jnp.roll(zdit, +1, ax_x) + zdit_kp1)
        avg4_wj = (zdjt + jnp.roll(zdjt_kp1, +1, ax_y)
                   + jnp.roll(zdjt, +1, ax_y) + zdjt_kp1)
    zfw_kp1 = zA31 * avg4_wi + zA32 * avg4_wj        # flux at interface BELOW cell k
    zfw_skew_current = zfw_kp1
    zfw_a33_current = jnp.zeros_like(zfw_kp1)
    # A33 explicit vertical-diagonal part.  With ln_traldf_msc=F the FULL K33
    # (ah_wslp2) goes to the implicit vertical solve → explicit coeff (ah_wslp2 −
    # akz) = 0 (akz≡ah_wslp2), so this block is skipped and the operator is
    # bit-identical.  With ln_traldf_msc=T (DINO) NEMO's Method of Stabilizing
    # Correction (traldf_iso_a33 / traldf_iso_scheme.h90:285) puts a stabilized
    # part explicit:  zfw += e1e2t/e3w · (ah_wslp2 − akz) · (T(k) − T(k+1)) at the
    # interface below cell k, with (all at that interface k+1):
    #   ah_wslp2  = zahu_w·wslpi² + zahv_w·wslpj²                  (the K33 diag)
    #   akz_h     = ¼ Σ4( aht/e1u² + aht/e2v² )                   (horiz stabiliser)
    #   akz       = MAX( dt·(akz_h + ah_wslp2/e3w²) − ½, 0 )·e3w²/dt   (impl part)
    # so the explicit coeff is bounded by the ½ vertical-CFL limit at steep slopes.
    if msc_stabilize:
        if dt is None:
            raise ValueError(
                "nemo_iso_lap_tracer_tendency_latlon_cgrid: msc_stabilize=True "
                "(ln_traldf_msc) requires dt (rDt) for the akz stability threshold."
            )
        # Shared a33 (#1226): compute (ah_wslp2, akz) once in the a33 "above"
        # convention via nemo_iso_a33 — the SAME function the implicit-K33
        # getter calls — then roll to this flux's (k,k+1) pair.  NEMO reads
        # akz(ji,jj,jk+1) from the a33 arrays here (scheme.h90:128), so the
        # roll IS the faithful indexing.  akz_h now uses face-MASKED aht
        # (NEMO's ahtu is masked at build; the previous inline version
        # summed raw aht — same wall deviation class as the zahu_w fix).
        # z*-scaled w-thickness.  The optional operand is diagnostic-only at
        # this layer: None preserves the historical T-thickness average
        # byte-for-byte, while a caller that already carries NEMO's live Kmm
        # e3w can replay both of the exact source uses below (akz's e3w**2 and
        # the explicit-A33 1/e3w post-factor) without changing any other Redi
        # operand.  Production dispatch remains None until its registered
        # stage-22/23 bracket owns the substitution.
        if msc_e3w_override is None:
            # traldf_iso.f90:285 / :831-833, through the ONE resolver the
            # implicit K33 side calls too, so the split cannot diverge.
            e3w_ab = nemo_iso_a33_e3w(z_coord, e3t, jacobian, dtype)
        else:
            e3w_ab = jnp.asarray(msc_e3w_override, dtype=dtype)
            if e3w_ab.shape != q.shape:
                raise ValueError(
                    "msc_e3w_override must have the full tracer shape "
                    f"{q.shape}, got {e3w_ab.shape}")
        _ahw_ab, _akz_ab = nemo_iso_a33(
            aht, umask, vmask, wmask, wslpi, wslpj,
            e1u, e2v, e3w_ab ** 2, dt=dt, msc=True, aht_v=aht_v,
            evaluation=a33_evaluation)
        ah_wslp2 = jnp.roll(_ahw_ab, -1, ax_z)
        akz = jnp.roll(_akz_ab, -1, ax_z)
        e3w_kp1 = jnp.roll(e3w_ab, -1, ax_z)
        # e1e2t/e3w · (ah_wslp2 − akz) · (T(k)−T(k+1)); zdkt_kp1 already carries
        # wmask(k+1) and the (T(k)−T(k+1)) difference.  Applied at EVERY interface
        # below cell k: the flux below the top cell (k=0 → NEMO w-level 2) IS a real
        # NEMO A33 flux (a33 loops jk=2..jpkm1, and the below-top-cell flux uses
        # ah_wslp2(jk+1=2)); the true SURFACE (NEMO w-level 1) is zfw_top[0], zeroed
        # below.  The near-surface ML k1/k2 over-diffusion is a pre-existing ML-ramp
        # SLOPE mismatch (native wslp match NEMO corr 0.99 below the ML, ~0.3 in it),
        # not this A33 term — masking it here would drop a legitimate NEMO flux.
        zfw_a33_current = (e1t * e2t)[:, :, jnp.newaxis] / e3w_kp1 * (
            ah_wslp2 - akz) * zdkt_kp1
        zfw_kp1 = zfw_kp1 + zfw_a33_current
    # Sea-floor / bottom no-flux BC: the interface below cell k carries flux
    # only if cell k+1 is water.  This zeros the flux crossing into the floor
    # (already ~0 there via the padded bottom slope) AND, critically, kills the
    # periodic z-roll wrap at k=nlev-1 when the column is full-depth (no dry
    # level) — so the vertical divergence telescopes to zero (conservation).
    act_below = jnp.concatenate(
        [act[:, :, 1:], jnp.zeros_like(act[:, :, :1])], axis=2)  # act[k+1], 0 at bottom
    zfw_kp1 = zfw_kp1 * act_below

    # ================= GM eddy-induced (bolus) transport (NEMO ldf_eiv_trp_MLF) =
    # NEMO adds the eiv velocity to the advecting velocity; equivalently we add
    # the eiv ADVECTIVE tracer flux (transport · centred tracer) to the Redi
    # flux and take one divergence.  The transport is the CURL of the bolus
    # streamfunction ψ, so its discrete divergence telescopes to zero exactly
    # (tracer conservation is structural, independent of ψ's masking).
    #   ψ_uw(iface below cell k) = -e2u · mi(wslpi_kp1) · mk(aeiu) · wumask   [m³/s]
    # SIGN: this code's tendency = +div(flux); advection dT/dt = -div(u·T), so
    # the eiv flux enters the flux arrays with a MINUS sign.
    bolus_transport = None
    if kappa_GM is not None:
        # SINGLE bolus-transport definition (curl of ψ): shared by the centred
        # in-operator path AND the through_fct export.  Masking notes: ψ at the
        # interface BELOW cell k is nonzero ONLY where the FOUR cells around the
        # u/v-w point are wet AND the horizontal wall is open (no bolus into a dry
        # cell at a topographic step — exact conservation on varying bathymetry);
        # ``act_below`` (k+1 activity, zeroed at the floor) is used NOT roll(act,
        # -1,z) so a full-depth column does not wrap a surface slope onto the sea
        # floor (NEMO forces wslpi(jpk)=0, wumask(...,jpk)=0 at the deepest iface).
        u_eiv, v_eiv, w_eiv_kp1 = nemo_eiv_bolus_transport(
            kappa_GM, bolus_wslpi_kp1, bolus_wslpj_kp1, e2u, e1v,
            u_mask, v_mask, act, act_below, q.shape, dtype,
            kappa_face_average=gm_bolus_kappa_face_average,
        )
        bolus_transport = (u_eiv, v_eiv, w_eiv_kp1)
        if gm_bolus_advection == "centred":
            # In-operator GM: add the eiv ADVECTIVE tracer flux (transport ·
            # centred tracer) to the Redi flux.  LIMITATIONS (fidelity, not
            # conservation): the bolus flux is 2nd-order CENTRED (dispersive at
            # sharp fronts, leans on the co-located Redi K to damp 2Δx noise) and
            # aeiu_if reuses the cell-centred κ_GM for both faces (exact for
            # constant κ_GM; a half-cell offset for a spatially-2-D κ_GM).
            t_u = 0.5 * (q + jnp.roll(q, -1, ax_x))
            t_v = 0.5 * (q + jnp.roll(q, -1, ax_y))
            t_w_kp1 = 0.5 * (q + jnp.roll(q, -1, ax_z))           # tracer at iface below k
            zfu = zfu - u_eiv * t_u
            zfv = zfv - v_eiv * t_v
            zfw_kp1 = zfw_kp1 - w_eiv_kp1 * t_w_kp1 * act_below
        elif gm_bolus_advection == "through_fct":
            # NEMO traadv: the eiv transport is added to the ADVECTING velocity
            # (exported via ``return_bolus``) and passed through the monotone FCT
            # tracer scheme by the model step — NOT added here (no double count).
            pass
        else:
            raise ValueError(
                "nemo_iso_lap_tracer_tendency_latlon_cgrid: unknown "
                f"gm_bolus_advection={gm_bolus_advection!r}; expected "
                "'centred' or 'through_fct'.")

    # ================= 3-D DIVERGENCE (added to RHS with + sign) =============
    zfw_top = jnp.roll(zfw_kp1, +1, ax_z)            # flux at interface ABOVE cell k
    zfw_top = zfw_top.at[:, :, 0].set(0.0)           # surface flux = 0
    if area_reciprocal is not None:
        r1_e1e2t = jnp.asarray(area_reciprocal, dtype=dtype)
        if r1_e1e2t.shape != e1t.shape:
            raise ValueError(
                "area_reciprocal must have the tracer-cell horizontal "
                f"shape {e1t.shape}, got {r1_e1e2t.shape}")
    elif area_reciprocal_evaluation == "nemo_stored":
        # domhgr.f90:155 stores e1e2t first, then its reciprocal. Retain both
        # assignments as compiled source-rounding boundaries for the private
        # production-step discriminator.
        r1_e1e2t = nemo_source_round(
            1.0 / nemo_source_round(e1t * e2t))
    else:
        r1_e1e2t = 1.0 / (e1t * e2t)
    e3t_divisor = e3t if divisor_thickness is None else jnp.asarray(
        divisor_thickness, dtype=dtype)
    if e3t_divisor.shape != q.shape:
        raise ValueError(
            "divisor_thickness must have the full tracer shape "
            f"{q.shape}, got {e3t_divisor.shape}")
    divergence_mode = horizontal_flux_evaluation.removeprefix(
        "nemo_literal_")
    if divergence_mode in (
            "differences", "horizontal_sum", "flux_sum", "divergence"):
        # traldf_iso.f90:306-310/:327-331.  These cumulative private modes
        # retain one more compiled boundary at a time after the horizontal
        # fluxes have become exact.  The deepest level uses zfw alone rather
        # than subtracting the identically-zero flux below the model floor.
        du = nemo_source_round(zfu - jnp.roll(zfu, +1, ax_x))
        dv = nemo_source_round(zfv - jnp.roll(zfv, +1, ax_y))
        dw = nemo_source_round(zfw_top - zfw_kp1)
        dw = dw.at[:, :, -1].set(zfw_top[:, :, -1])
        hdiv = du + dv
        if divergence_mode in ("horizontal_sum", "flux_sum", "divergence"):
            hdiv = nemo_source_round(hdiv)
        total_div = hdiv + dw
        if divergence_mode in ("flux_sum", "divergence"):
            total_div = nemo_source_round(total_div)
        scaled_div = total_div * r1_e1e2t[:, :, jnp.newaxis]
        if divergence_mode == "divergence":
            scaled_div = nemo_source_round(scaled_div)
            tend = nemo_source_round(scaled_div / e3t_divisor)
        else:
            tend = scaled_div / e3t_divisor
    else:
        hdiv = ((zfu - jnp.roll(zfu, +1, ax_x))
                + (zfv - jnp.roll(zfv, +1, ax_y)))
        vdiv = zfw_top - zfw_kp1
        tend = ((hdiv + vdiv) * r1_e1e2t[:, :, jnp.newaxis]
                / e3t_divisor)
    # Mask by the 3-D cell wet mask (NEMO tmask), not just the 2-D surface mask,
    # so sub-seafloor dry levels of a wet column are zeroed too (byte-identical
    # on flat bottom, where those levels already carry zero divergence).
    if final_update_evaluation != "unmasked":
        tend = tend * act
    if final_update_evaluation == "nemo_rhs_increment":
        rhs_before = jnp.asarray(rhs_accumulator, dtype=dtype)
        if rhs_before.shape != q.shape:
            raise ValueError(
                "rhs_accumulator must have the full tracer shape "
                f"{q.shape}, got {rhs_before.shape}")
        # traldf_iso.f90:306-310/:327-331 updates Krhs in place.  The Round
        # 235 private discriminator returns the exact increment subsequently
        # written by the oracle (rhs_after-rhs_before), preserving both
        # compiled assignment boundaries without changing any production arm.
        tend = nemo_source_round(
            nemo_source_round(rhs_before + tend) - rhs_before)
    if return_diagnostics:
        diagnostics = {
            "zfu": zfu,
            "zfv": zfv,
            "zfw_kp1": zfw_kp1,
        }
        if return_operand_diagnostics:
            diagnostics.update({
                "q": q, "tmask": act, "umask": umask, "vmask": vmask,
                "wmask": wmask, "ahtu": aht, "ahtv": aht_v,
                "uslp": uslp, "vslp": vslp, "wslpi": wslpi,
                "wslpj": wslpj, "e3t": e3t_divisor, "e3u_flux": e3u_flux,
                "e3v_flux": e3v_flux, "dit": zdit, "djt": zdjt,
                "dkt": zdkt, "A11": zA11, "A22": zA22, "A13": zA13,
                "A23": zA23, "hmsku": zmsku_h, "hmskv": zmskv_h,
                "vmsku": zmsku_w, "vmskv": zmskv_w, "ahu_w": zahu_w,
                "ahv_w": zahv_w, "A31": zA31, "A32": zA32,
                "zfw_top": zfw_top, "tendency": tend,
                "r1_e1e2t": r1_e1e2t,
            })
            diagnostics["zfu_operands"] = {
                # NOTE: face-MASKED, as NEMO's own ahtu is (ldftra.f90:433).
                "ahtu": aht,
                "e1u": e1u,
                "e2u": e2u,
                "e3t": e3t,
                "e3u_flux": e3u_flux,
                "uslp": uslp,
                "wmask": wmask,
                "zmsku": zmsku_h,
                "zdit": zdit,
                "zdkt": zdkt,
                "avg4_u": avg4_u,
            }
            if msc_stabilize:
                diagnostics["zfw_operands"] = {
                    "ahtu": aht,
                    "ahtv": aht_v,
                    "wslpi": wslpi,
                    "wslpj": wslpj,
                    "bolus_wslpi": bolus_wslpi,
                    "bolus_wslpj": bolus_wslpj,
                    "zA31": zA31,
                    "zA32": zA32,
                    "zdit": zdit,
                    "zdjt": zdjt,
                    "wmask": wmask,
                    "ah_wslp2": ah_wslp2,
                    "akz": akz,
                    "ah_wslp2_above": _ahw_ab, "akz_above": _akz_ab,
                    "e1e2t": e1t * e2t,
                    "e3w_kp1": e3w_kp1,
                    "qdiff_kp1": q - jnp.roll(q, -1, axis=2),
                    "act_below": act_below,
                    "skew_current": zfw_skew_current,
                    "a33_current": zfw_a33_current,
                }
        if return_bolus:
            return tend, bolus_transport, diagnostics
        return tend, diagnostics
    if return_bolus:
        return tend, bolus_transport
    return tend


# =====================================================================
# Triad slope discretisation (Griffies, Gnanadesikan et al. 1998)
# =====================================================================
#
# The centered scheme above evaluates slopes at vertical interfaces and
# tracer gradients at faces, then averages the two onto interfaces.
# This averaging breaks the algebraic identity
#
#     dq/dx + S_x * dq/dz = 0           when q = f(rho),
#
# leaving a small Redi residual that accumulates through dynamical
# feedback over long integrations.
#
# The triad scheme decomposes the flux at each face into four
# quarter-cell triads.  Each triad uses the SAME three rho/q values for
# both its slope and its tracer gradients, so the cancellation above
# holds *exactly* for every triad — the Redi tendency is zero to
# machine precision when q is constant along isopycnals.
#
# Convention: ``z`` increases UPWARD, ``k = 0`` is the surface, and
# ``k+1`` is the level below ``k``.  ``drho_dz_w[k]`` therefore is the
# (negative) finite difference ``(rho[k] - rho[k+1]) / dz_half`` at the
# w-face between full levels ``k`` and ``k+1``.
#
# u-face triads at face index ``j_face`` (between cells ``j_w`` and
# ``j_e``) and full level ``k`` share the horizontal pair
# ``(j_w, k)-(j_e, k)`` — this gives ``drho_dx_uface(j_face, k)``,
# common to all four triads — and differ only in the vertical pair:
#
#     T1 (W,B): (j_w, k) - (j_w, k+1)   uses drho_dz_w[j_w, k]
#     T2 (W,A): (j_w, k-1) - (j_w, k)   uses drho_dz_w[j_w, k-1]
#     T3 (E,B): (j_e, k) - (j_e, k+1)   uses drho_dz_w[j_e, k]
#     T4 (E,A): (j_e, k-1) - (j_e, k)   uses drho_dz_w[j_e, k-1]
#
# Triads "above" the top level (k = 0 ⇒ T2/T4) and "below" the bottom
# level (k = nlev-1 ⇒ T1/T3) are not defined; their contributions are
# masked out and the per-face average is normalised by the number of
# valid triads, preserving the diagonal Redi flux at full strength
# while still cancelling the off-diagonal exactly when q = f(rho).


def _to_uface_west(field: jnp.ndarray) -> jnp.ndarray:
    """Lift a cell-centre field to the *west* neighbour of every u-face.

    For a ``(n_lat, n_lon, ...)`` array, returns ``(n_lat, n_lon+1, ...)``
    with ``out[:, j_face, ...] = field[:, (j_face - 1) mod n_lon, ...]``
    using periodic wrap in longitude.
    """
    rolled = jnp.roll(field, 1, axis=1)
    return jnp.concatenate([rolled, rolled[:, 0:1]], axis=1)


def _to_uface_east(field: jnp.ndarray) -> jnp.ndarray:
    """Lift a cell-centre field to the *east* neighbour of every u-face.

    Returns ``out[:, j_face, ...] = field[:, j_face mod n_lon, ...]``.
    """
    return jnp.concatenate([field, field[:, 0:1]], axis=1)


def _to_vface_south(field: jnp.ndarray) -> jnp.ndarray:
    """Lift a cell-centre field to the *south* neighbour of every v-face.

    No periodic wrap in latitude — wall BCs at the poles.  The pole
    sentinels are inert because the v-face mask zeroes the flux there.
    """
    return jnp.concatenate([field[0:1], field], axis=0)


def _to_vface_north(field: jnp.ndarray) -> jnp.ndarray:
    """Lift a cell-centre field to the *north* neighbour of every v-face."""
    return jnp.concatenate([field, field[-1:]], axis=0)



def _w_face_slope_density_inputs(
    rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
    eos_fn, rho_0, g,
):
    """Build the W-face slope-density inputs for :func:`_w_triad_slopes_tapers`.

    Returns a dict of keyword arguments — ``drho_dx_u``, ``drho_dy_v``,
    ``drho_dz_w`` and (for the neutral mode) ``drdT_w``, ``drdS_w``, ``dTdx_u``,
    ``dSdx_u``, ``dTdy_v``, ``dSdy_v`` — so the K_33 getter and the realized
    GM-skew conversion (both W-face-only consumers) build the slopes through the
    SAME path as the triad ``F_z`` and never diverge.

    ``"in_situ"`` (BIT-IDENTICAL): the in-situ face gradients + floored vertical
    gradient (exactly the prior inline code).  ``"neutral"``: the locally-
    referenced ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` ingredients (W-cell ``drdT_w`` + tracer
    face gradients), with the floor applied to the neutral ``drho_dz_w``.

    kr-sum (Veros ``isoneutral.py:176-198``): each w-face triad references one
    of the TWO cells adjacent to the face, and Veros builds BOTH ``drodxb`` and
    ``drodzb`` from that reference cell's own EOS derivatives (the ``kr`` loop).
    The neutral dict therefore carries both the UPPER-cell (``drdT_w``/
    ``drho_dz_w``, the A-triads) and the LOWER-cell (``drdT_wb``/
    ``drho_dz_w_b``, the B-triads) variants.
    """
    _validate_slope_density(slope_density)
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]
    if slope_density == "neutral":
        drdT_c, drdS_c = _neutral_drho_derivs(
            T, S, mask, z_coord, jacobian, eos_fn, rho_0, g,
        )
        T_filled = neumann_fill_cgrid(T, mask)
        S_filled = neumann_fill_cgrid(S, mask)
        dTdx_u = gradient_x_cgrid(T_filled, grid)
        dSdx_u = gradient_x_cgrid(S_filled, grid)
        dTdy_v = gradient_y_cgrid(T_filled, grid)
        dSdy_v = gradient_y_cgrid(S_filled, grid)
        dTdz_w = (T_filled[:, :, :-1] - T_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        dSdz_w = (S_filled[:, :, :-1] - S_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        drdT_w = drdT_c[:, :, :-1]
        drdS_w = drdS_c[:, :, :-1]
        drdT_wb = drdT_c[:, :, 1:]
        drdS_wb = drdS_c[:, :, 1:]
        drho_dz_w = jnp.minimum(drdT_w * dTdz_w + drdS_w * dSdz_w, -_EPS_DIV)
        drho_dz_w_b = jnp.minimum(
            drdT_wb * dTdz_w + drdS_wb * dSdz_w, -_EPS_DIV)
        return dict(
            drho_dx_u=None, drho_dy_v=None, drho_dz_w=drho_dz_w,
            slope_density="neutral", drdT_w=drdT_w, drdS_w=drdS_w,
            drdT_wb=drdT_wb, drdS_wb=drdS_wb, drho_dz_w_b=drho_dz_w_b,
            dTdx_u=dTdx_u, dSdx_u=dSdx_u, dTdy_v=dTdy_v, dSdy_v=dSdy_v,
        )
    drho_dx_u = gradient_x_cgrid(rho_filled, grid)
    drho_dy_v = gradient_y_cgrid(rho_filled, grid)
    drho_dz_raw = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(
        dz_half, _EPS_DIV)
    drho_dz_w = jnp.minimum(drho_dz_raw, -_EPS_DIV)
    return dict(
        drho_dx_u=drho_dx_u, drho_dy_v=drho_dy_v, drho_dz_w=drho_dz_w,
        slope_density="in_situ",
    )


def _w_triad_numerators(slope_density, drho_dx_u, drho_dy_v,
                        drdT_w, drdS_w, dTdx_u, dSdx_u, dTdy_v, dSdy_v,
                        n_lat, n_lon, drdT_wb=None, drdS_wb=None,
                        u_face_act=None, v_face_act=None):
    """The 8 W-face triad horizontal density-gradient *numerators* ``-∇_hρ``.

    Returns ``(nx_W, nx_E, nx_Wb, nx_Eb, ny_S, ny_N, ny_Sb, ny_Nb)`` where the
    A-suffix (W/E/S/N) is the upper-level (k) horizontal gradient and the
    b-suffix is the lower-level (k+1) one, all at the ``nlev-1`` w-faces.

    ``u_face_act`` / ``v_face_act`` (optional, ``(n_lat, n_lon+1, nlev)`` /
    ``(n_lat+1, n_lon, nlev)``) are the PER-LEVEL face-activity masks
    (``compute_face_masks_3d``).  When given, each triad's horizontal numerator
    is multiplied by the activity of ITS face at ITS level — Veros's mechanism
    exactly (its ``dTdx``/``dTdy`` carry ``maskU``/``maskV`` per level,
    ``isoneutral.py:64-95``, so a triad whose horizontal pair crosses a CLOSED
    face — a coastline or a topographic step — has slope 0 and contributes
    nothing to F_z / K_33 / the realized conversions).  Without this, the
    Neumann-filled gradients leak a spurious nonzero triad flux across closed
    faces (global_4deg tier-2 isolation: F_z rms 1.9-2.3× Veros at coast/step
    faces, corr 0.62-0.69, vs 1.02-1.04 / 0.98 interior).  ``None`` (pure
    z-star / flat bottom) skips the multiply — bit-identical legacy path.

    - ``"in_situ"`` (BIT-IDENTICAL): the corresponding slices of the in-situ
      face gradient ``drho_dx_u`` / ``drho_dy_v`` (exactly the slices the prior
      inline code used).
    - ``"neutral"``: the locally-referenced ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` form, with
      the PER-TRIAD reference cell's own EOS derivatives (Veros's ``kr`` loop,
      ``isoneutral.py:176-198``): the A-triads (upper-level tracer gradients)
      use the UPPER cell's ``drdT_w``/``drdS_w``; the B-triads (lower-level
      gradients) use the LOWER cell's ``drdT_wb``/``drdS_wb`` — Veros pairs
      ``drdT[..., kr]`` with ``dTdx[..., kr]`` for both ``drodxb`` and
      ``drodzb``.  (Before the kr-sum refinement all 8 used the upper cell.)
    """
    if u_face_act is not None:
        mxW = u_face_act[:, :n_lon, :-1]
        mxE = u_face_act[:, 1:n_lon + 1, :-1]
        mxWb = u_face_act[:, :n_lon, 1:]
        mxEb = u_face_act[:, 1:n_lon + 1, 1:]
    else:
        mxW = mxE = mxWb = mxEb = None
    if v_face_act is not None:
        myS = v_face_act[:n_lat, :, :-1]
        myN = v_face_act[1:n_lat + 1, :, :-1]
        mySb = v_face_act[:n_lat, :, 1:]
        myNb = v_face_act[1:n_lat + 1, :, 1:]
    else:
        myS = myN = mySb = myNb = None
    if slope_density == "neutral":
        if drdT_wb is None or drdS_wb is None:
            raise ValueError(
                "_w_triad_numerators: slope_density='neutral' requires the "
                "lower-cell drdT_wb/drdS_wb (the kr-sum pairing); got None. "
                "Build inputs via _w_face_slope_density_inputs.")
        nx_W = drdT_w * dTdx_u[:, :n_lon, :-1] + drdS_w * dSdx_u[:, :n_lon, :-1]
        nx_E = (drdT_w * dTdx_u[:, 1:n_lon + 1, :-1]
                + drdS_w * dSdx_u[:, 1:n_lon + 1, :-1])
        nx_Wb = (drdT_wb * dTdx_u[:, :n_lon, 1:]
                 + drdS_wb * dSdx_u[:, :n_lon, 1:])
        nx_Eb = (drdT_wb * dTdx_u[:, 1:n_lon + 1, 1:]
                 + drdS_wb * dSdx_u[:, 1:n_lon + 1, 1:])
        ny_S = drdT_w * dTdy_v[:n_lat, :, :-1] + drdS_w * dSdy_v[:n_lat, :, :-1]
        ny_N = (drdT_w * dTdy_v[1:n_lat + 1, :, :-1]
                + drdS_w * dSdy_v[1:n_lat + 1, :, :-1])
        ny_Sb = (drdT_wb * dTdy_v[:n_lat, :, 1:]
                 + drdS_wb * dSdy_v[:n_lat, :, 1:])
        ny_Nb = (drdT_wb * dTdy_v[1:n_lat + 1, :, 1:]
                 + drdS_wb * dSdy_v[1:n_lat + 1, :, 1:])
    else:
        nx_W = drho_dx_u[:, :n_lon, :-1]
        nx_E = drho_dx_u[:, 1:n_lon + 1, :-1]
        nx_Wb = drho_dx_u[:, :n_lon, 1:]
        nx_Eb = drho_dx_u[:, 1:n_lon + 1, 1:]
        ny_S = drho_dy_v[:n_lat, :, :-1]
        ny_N = drho_dy_v[1:n_lat + 1, :, :-1]
        ny_Sb = drho_dy_v[:n_lat, :, 1:]
        ny_Nb = drho_dy_v[1:n_lat + 1, :, 1:]
    if mxW is not None:
        nx_W = nx_W * mxW
        nx_E = nx_E * mxE
        nx_Wb = nx_Wb * mxWb
        nx_Eb = nx_Eb * mxEb
    if myS is not None:
        ny_S = ny_S * myS
        ny_N = ny_N * myN
        ny_Sb = ny_Sb * mySb
        ny_Nb = ny_Nb * myNb
    return nx_W, nx_E, nx_Wb, nx_Eb, ny_S, ny_N, ny_Sb, ny_Nb


def _w_triad_slopes_tapers(drho_dx_u, drho_dy_v, drho_dz_w, n_lat, n_lon,
                           S_max, taper_width_frac,
                           slope_density="in_situ",
                           drdT_w=None, drdS_w=None,
                           drdT_wb=None, drdS_wb=None, drho_dz_w_b=None,
                           dTdx_u=None, dSdx_u=None, dTdy_v=None, dSdy_v=None,
                           u_face_act=None, v_face_act=None,
                           adjoint_stabilization="none",
                           slope_limit="dm95_taper"):
    """W-face (vertical-flux) triad isopycnal slopes + DM95 tapers.

    Shared by the explicit ``F_z`` assembly (in
    ``gm_redi_tracer_tendency_triads_latlon_cgrid``), the implicit-K_33
    diffusivity getter (``compute_isoneutral_K33_latlon``), and the realized
    GM-skew EKE source so the slope numerics can never diverge.  ``A`` = upper
    level k, ``B`` = lower level k+1; the x-triads use the west/east u-faces,
    the y-triads the south/north v-faces.  Returns the 8 per-triad slopes
    followed by their 8 DM95 tapers, each at the w-faces (n_lat, n_lon, nlev-1).

    ``slope_density`` selects the density gradient that builds the slopes:
    ``"in_situ"`` (default, BIT-IDENTICAL) slices ``drho_dx_u`` / ``drho_dy_v``
    and CLIPS the slope to ±S_max before the DM95 taper (the legoESM safety
    convention); ``"neutral"`` builds ``∂ρ/∂T·∇T + ∂ρ/∂S·∇S`` from the W-cell
    ``drdT_w`` / ``drdS_w`` and the raw tracer face gradients, and does NOT clip
    — Veros never clips the slope, it relies on the DM95 taper (``dm_taper``) to
    suppress steep slopes (``taper → 0`` as ``|S| → ∞``).  The neutral slope is
    ~4× steeper than the in-situ one (compressibility removed from ``∂_zρ``), so
    it routinely exceeds S_max; clipping it would saturate the taper at its
    S_max value (0.5) and inflate K_33 ~10× over Veros — the unclipped taper is
    what makes the neutral K_33 track Veros (ratio ~1.0 at the thermocline).
    """
    validate_adjoint_stabilization(adjoint_stabilization)
    validate_slope_limit(slope_limit)
    (nx_W, nx_E, nx_Wb, nx_Eb, ny_S, ny_N, ny_Sb, ny_Nb) = _w_triad_numerators(
        slope_density, drho_dx_u, drho_dy_v, drdT_w, drdS_w,
        dTdx_u, dSdx_u, dTdy_v, dSdy_v, n_lat, n_lon,
        drdT_wb=drdT_wb, drdS_wb=drdS_wb,
        u_face_act=u_face_act, v_face_act=v_face_act)
    # nemo_cap: ALWAYS cap the slope at ±S_max (NEMO ldfslp rn_slpmax)
    # and skip the DM95 taper — the flux keeps diffusing along the
    # capped direction at steep fronts instead of dying.
    clip = (slope_density != "neutral") or (slope_limit == "nemo_cap")
    # kr-sum: the B-triads divide by the LOWER cell's drodzb (Veros pairs the
    # same kr-cell derivatives in numerator and denominator).  in_situ has no
    # per-cell derivative, so both levels share the single face denominator
    # (a neutral call without the b-variant already raised in the numerators).
    drho_dz_w_B = drho_dz_w_b if drho_dz_w_b is not None else drho_dz_w
    # Adjoint stabilization (primal-invisible; see _gm_redi_common note):
    # freeze the per-triad slopes ("stop_gradient_slopes" — the tapers built
    # from frozen slopes then carry no gradient) or only the tapers
    # ("stop_gradient_taper").  Static Python gating.
    _sg_slopes = adjoint_stabilization == "stop_gradient_slopes"
    _sg_taper = adjoint_stabilization == "stop_gradient_taper"

    def _slope(num, dz=drho_dz_w):
        # Sign-preserving denominator floor: a fully-dry / unstratified
        # column has drho_dz == 0 exactly (0/0 -> NaN poisons the flux even
        # through masked branches). Floor at the STABLE-limit sign (drho_dz
        # < 0 for stable stratification, see the dz_half note above); wet
        # stratified columns (|dz| >> 1e-20) are bit-identical. Mirrors
        # NEMO's ldfslp MIN(zbu, -eps·|zau|) denominator capping.
        dz = jnp.where(jnp.abs(dz) > 1e-20, dz, -1e-20)
        s = -num / dz
        s = jnp.clip(s, -S_max, S_max) if clip else s
        return jax.lax.stop_gradient(s) if _sg_slopes else s

    S_Wx1 = _slope(nx_W)                    # W,A
    S_Wx2 = _slope(nx_E)                    # E,A
    S_Wx3 = _slope(nx_Wb, drho_dz_w_B)      # W,B
    S_Wx4 = _slope(nx_Eb, drho_dz_w_B)      # E,B
    S_Wy1 = _slope(ny_S)                    # S,A
    S_Wy2 = _slope(ny_N)                    # N,A
    S_Wy3 = _slope(ny_Sb, drho_dz_w_B)      # S,B
    S_Wy4 = _slope(ny_Nb, drho_dz_w_B)      # N,B
    if slope_limit == "nemo_cap":
        tw = lambda s: jnp.ones_like(s)
    else:
        tw = lambda s: dm95_taper_scalar(
            s, S_max, transition_width_frac=taper_width_frac,
            stop_gradient_taper=_sg_taper)[1]
    return (S_Wx1, S_Wx2, S_Wx3, S_Wx4, S_Wy1, S_Wy2, S_Wy3, S_Wy4,
            tw(S_Wx1), tw(S_Wx2), tw(S_Wx3), tw(S_Wx4),
            tw(S_Wy1), tw(S_Wy2), tw(S_Wy3), tw(S_Wy4))


def gm_redi_tracer_tendency_triads_latlon_cgrid(
    q: jnp.ndarray,
    rho: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
    grid: LatLonGrid,
    kappa_GM,
    kappa_Redi,
    S_max: float,
    taper_width_frac: float = _DEFAULT_TAPER_WIDTH_FRAC,
    implicit_K33: bool = False,
    K_iso_steep: float = 0.0,
    slope_density: str = "in_situ",
    slope_limit: str = "dm95_taper",
    T_tracer: jnp.ndarray | None = None,
    S_tracer: jnp.ndarray | None = None,
    eos_fn=None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    return_fluxes: bool = False,
    double_diag_kappa=None,
    double_diag_steep: float = 0.0,
    veros_triad_weights: bool = False,
    adjoint_stabilization: str = "none",
) -> jnp.ndarray:
    """Triad-based GM+Redi tracer tendency on the lat-lon C-grid.

    ``adjoint_stabilization`` (default "none" = exact AD, bit-identical):
    reverse-mode gradient stabilization via ``stop_gradient`` on the DM95
    tapers ("stop_gradient_taper") or the per-triad slopes
    ("stop_gradient_slopes") — PRIMAL-INVISIBLE by construction; see the
    ``GMRediConfig`` field doc and the note in ``_gm_redi_common``.

    ``double_diag_kappa`` (default None = off, bit-identical): when given (the
    ``GMRediConfig.double_redi_diagonal`` option), the horizontal DIAGONAL flux
    ``κ·taper·∇_h q`` (with its ``double_diag_steep`` K_iso_steep floor) is
    added ONE extra time to ``F_x``/``F_y`` using this kappa — reproducing
    Veros's double-counted ``K_11``/``K_22`` diagonal (its iso AND skew passes
    each add the precomputed diagonal; see the ``GMRediConfig`` field doc).
    Pass the Redi (K_iso) kappa here — Veros builds K_11/K_22 from ``K_iso``
    regardless of which pass re-adds them.

    ``veros_triad_weights`` (default False = bit-identical 1/N_valid legacy):
    weight each u/v-face triad by ``dzw(pair)/(4·dzt)`` with NO boundary
    renormalization, Veros's convention — see the ``GMRediConfig`` field doc.

    When ``return_fluxes`` is True (default False ⇒ unchanged single-array
    return), returns ``(tendency, F_x_u, F_y_v, F_z)`` — the assembled, masked
    isopycnal-tensor face fluxes of ``q`` (east-face ``F_x_u``
    ``(n_lat, n_lon+1, nlev)``, north-face ``F_y_v`` ``(n_lat+1, n_lon, nlev)``,
    top-face ``F_z`` ``(n_lat, n_lon, nlev-1)``).  These are the EXACT quantities
    the divergence consumes, so the realized signed APE conversions
    (``-P_diss_skew`` / ``-P_diss_iso``) are built by contracting them with the
    density-gradient field WITHOUT re-assembling any flux (no duplicate
    numerics).  Passing ``kappa_Redi=0`` gives the SKEW-only fluxes (Veros
    ``K_iso=0, K_skew=K_gm``); passing ``kappa_GM=0`` gives the ISO-only fluxes
    (Veros ``K_iso=K_iso, K_skew=0``).  ``F_z`` carries the explicit off-diagonal
    (and, when ``implicit_K33=False``, the K_33 diagonal) exactly as the
    tendency does.

    Implements the Griffies, Gnanadesikan, Pacanowski, Larichev,
    Dukowicz & Smith (1998) triad decomposition of the small-slope
    isopycnal-tensor fluxes.  Each face flux is built from 4 triads
    (w-face) or up to 4 triads (u/v-face); each triad uses one shared
    horizontal density-and-tracer pair and one shared vertical pair, so
    the algebraic identity ``dq/dx + S_x dq/dz = 0`` (and its z-flux
    counterpart) holds *per triad* when ``q = f(rho)`` — the Redi
    tendency is zero to machine precision.

    Parameters
    ----------
    q, rho : (n_lat, n_lon, nlev)
        Tracer and in-situ density at cell centres.  Both are
        Neumann-filled internally — pass the raw fields exactly as
        ``compute_isopycnal_slopes_latlon_cgrid`` expects them.
    mask, u_mask, v_mask : ocean / face masks.
    z_coord, jacobian, grid : geometry.
    kappa_GM : float, (n_lat, n_lon), or (n_lat, n_lon, nlev-1)
        GM bolus coefficient [m^2/s].  Scalar (constant), per-column 2-D
        (Visbeck), or a depth-resolved 3-D **interface** field (the 3-D EKE
        closure): the interface kappa lives at the ``nlev-1`` interior
        interfaces (W-grid) and is used DIRECTLY for the vertical flux, with a
        vertical interface→center interpolation feeding the horizontal flux.
        Dispatch is by ``_kappa_is_interface_3d`` (only when ``nlev > 2``).
    kappa_Redi : float, (n_lat, n_lon), or (n_lat, n_lon, nlev-1)
        Redi isopycnal diffusivity [m^2/s].  Same three forms as ``kappa_GM``.
        Per-column or 3-D for the K_iso=K_gm coupling (pass kappa_Redi ==
        kappa_GM): then the (kappa_Redi-kappa_GM) horizontal off-diagonal
        cancels, as in Veros enable_eke_isopycnal_diffusion.  A scalar keeps the
        path bit-identical.
    S_max : float
        Slope cap for clipping and DM95 taper.

    Returns
    -------
    tendency : (n_lat, n_lon, nlev)
    """
    _validate_slope_density(slope_density)
    validate_adjoint_stabilization(adjoint_stabilization)
    # Adjoint stabilization (primal-invisible; static Python gating).
    _sg_slopes = adjoint_stabilization == "stop_gradient_slopes"
    _sg_taper = adjoint_stabilization == "stop_gradient_taper"
    n_lat, n_lon, nlev = q.shape
    dz_actual = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]
    dz_half = z_coord.dz_half_ref * jacobian[:, :, jnp.newaxis]

    # Per-level face/interface activity (partial-cell coords): Veros masks the
    # triad fluxes with the 3-D maskU/maskV (Ai_ez/Ai_nz) and maskW
    # (Ai_bx/Ai_by, the W-face triads).  With only the 2-D masks, fluxes cross
    # CLOSED faces below the shallower neighbour's seafloor and through the
    # seafloor W-face; the wet-cell side of those fluxes is real tendency while
    # the dry-cell side is discarded by the model's active_3d guard — a NET
    # TRACER LEAK (global_4deg: −6%/yr volume salt with GM on; machine-zero
    # without GM).  Pure z-star: masks are all-ones ⇒ bit-identical.
    _is_active_3d = getattr(z_coord, "is_active", None)
    if _is_active_3d is not None:
        _um3, _vm3 = compute_face_masks_3d(_is_active_3d, grid)
        u_mask_lvl = _um3.astype(q.dtype)                 # (n_lat, n_lon+1, nlev)
        v_mask_lvl = _vm3.astype(q.dtype)                 # (n_lat+1, n_lon, nlev)
        w_mask_if = jnp.asarray(
            _is_active_3d, dtype=q.dtype)[..., 1:]        # (n_lat, n_lon, nlev-1)
    else:
        u_mask_lvl = jnp.ones((n_lat, n_lon + 1, 1), dtype=q.dtype)
        v_mask_lvl = jnp.ones((n_lat + 1, n_lon, 1), dtype=q.dtype)
        w_mask_if = jnp.ones((n_lat, n_lon, 1), dtype=q.dtype)

    # PER-TRIAD activity (variable bathymetry only; None ⇒ legacy bit-identical
    # path with zero added ops).  Veros kills triads PER LEVEL: a u/v-face
    # triad whose VERTICAL pair crosses the seafloor has maskW-masked dTdz ⇒
    # slope → ±∞ ⇒ dm_taper → 0 (isoneutral.py:117-130), and a W-face triad
    # whose HORIZONTAL pair crosses a closed face has maskU/maskV-masked dTdx
    # ⇒ slope 0 ⇒ zero contribution (isoneutral.py:182-217).  legoESM's
    # Neumann-filled gradients otherwise leak spurious triad fluxes across
    # coasts and topographic steps (tier-2 isolation: F_z 1.9-2.3× Veros at
    # coast/step W-faces vs 1.02-1.04 interior).  Mechanisms mirrored here:
    # u/v-face triads → multiply the per-triad TAPER by the vertical-pair
    # interface activity (off-diagonal dies; the K_iso_steep diagonal floor
    # survives at full strength, exactly Veros's max(K_iso_steep, K·taper));
    # W-face triads → mask the horizontal slope NUMERATORS (threaded into
    # _w_triad_slopes_tapers below).
    if _is_active_3d is not None:
        _zer1 = jnp.zeros((n_lat, n_lon, 1), dtype=q.dtype)
        _act_below = jnp.concatenate([w_mask_if, _zer1], axis=-1)
        _act_above = jnp.concatenate([_zer1, w_mask_if], axis=-1)
        act_T1 = _to_uface_west(_act_below)
        act_T2 = _to_uface_west(_act_above)
        act_T3 = _to_uface_east(_act_below)
        act_T4 = _to_uface_east(_act_above)
        act_V1 = _to_vface_south(_act_below)
        act_V2 = _to_vface_south(_act_above)
        act_V3 = _to_vface_north(_act_below)
        act_V4 = _to_vface_north(_act_above)
        _u_face_act = u_mask_lvl
        _v_face_act = v_mask_lvl
    else:
        act_T1 = act_T2 = act_T3 = act_T4 = None
        act_V1 = act_V2 = act_V3 = act_V4 = None
        _u_face_act = None
        _v_face_act = None

    # Resolve kappa to its center / u-face / v-face / w-face forms (shared
    # helper, identical dispatch for kappa_GM, kappa_Redi, and the K_33 getter):
    #   * scalar / 2-D (n_lat, n_lon)  → cell-centred broadcast + horizontal
    #     interp to faces, ``kappa_w = kappa_c`` — BIT-IDENTICAL to the prior
    #     code (the K_iso=K_gm coupling passes kappa_Redi == kappa_GM, so the
    #     (kappa_Redi-kappa_GM) horizontal off-diagonal still cancels, as in
    #     Veros enable_eke_isopycnal_diffusion).
    #   * 3-D interface (…, nlev-1)    → the 3-D EKE interface kappa: it is the
    #     w-face value DIRECTLY (the faithful placement for the vertical flux —
    #     no lossy round-trip), while the u/v-face horizontal forms come from a
    #     vertical interface→center interp then the usual cell→face interp.
    # The w-face triads share the SAME drho_dz_w(k+1/2) at a w-face, so a single
    # interface-level kappa_w is the correct per-w-face coefficient.
    kappa_GM_c, kappa_GM_u, kappa_GM_v, kappa_GM_w = _kappa_center_uvw(kappa_GM, nlev)
    kappa_Redi_c, kappa_Redi_u, kappa_Redi_v, kappa_Redi_w = _kappa_center_uvw(
        kappa_Redi, nlev)
    if double_diag_kappa is not None:
        # Veros double-counted K_11/K_22 diagonal (see docstring): u/v-face
        # forms of the kappa used for the EXTRA diagonal add below.
        _, _ddk_u, _ddk_v, _ = _kappa_center_uvw(double_diag_kappa, nlev)

    # Neumann-fill BOTH rho and q so the gradients across coastlines do
    # not pick up jumps between ocean and land sentinel values.  This
    # is essential for the per-triad cancellation when q = f(rho):
    # without it, a land cell that has T = 0 sentinel produces a
    # spurious vertical drho/dz = 0 in the land column, which clips the
    # slope to S_max and breaks the algebraic identity.
    rho_filled = neumann_fill_cgrid(rho, mask)
    q_filled = neumann_fill_cgrid(q, mask)

    # -----------------------------------------------------------------
    # 1. Density and tracer gradients on the native C-grid
    # -----------------------------------------------------------------
    dq_dx_u = gradient_x_cgrid(q_filled, grid)
    dq_dy_v = gradient_y_cgrid(q_filled, grid)
    dq_dz_w = (q_filled[:, :, :-1] - q_filled[:, :, 1:]) / jnp.maximum(
        dz_half, _EPS_DIV
    )

    # Density gradients that BUILD the slopes.  For the per-triad u/v-face
    # slopes, the horizontal density gradient references the TRIAD's cell (Veros
    # K_11 ``drodxe`` uses ``drdT[1+ip]`` — west cell for the W triads, east for
    # the E triads); the W-face triads use the W-cell's own derivative (Veros
    # K_33 uses ``drdT[2:-2]`` for all triads).  So we build the neutral
    # vertical w-face gradient with each cell's own drdT (so the cell-shifted
    # ``_to_uface_west/east`` already select the right per-triad drho_dz), and
    # the neutral horizontal numerators per reference cell below.
    if slope_density == "neutral":
        if T_tracer is None or S_tracer is None or eos_fn is None:
            raise ValueError(
                "gm_redi_tracer_tendency_triads_latlon_cgrid: "
                "slope_density='neutral' requires T_tracer, S_tracer, eos_fn."
            )
        drdT_c, drdS_c = _neutral_drho_derivs(
            T_tracer, S_tracer, mask, z_coord, jacobian, eos_fn, rho_0, g,
        )
        T_filled = neumann_fill_cgrid(T_tracer, mask)
        S_filled = neumann_fill_cgrid(S_tracer, mask)
        dTdx_u = gradient_x_cgrid(T_filled, grid)
        dSdx_u = gradient_x_cgrid(S_filled, grid)
        dTdy_v = gradient_y_cgrid(T_filled, grid)
        dSdy_v = gradient_y_cgrid(S_filled, grid)
        dTdz_w = (T_filled[:, :, :-1] - T_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        dSdz_w = (S_filled[:, :, :-1] - S_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV)
        # Per-cell neutral vertical gradient at the w-face, floored for stable
        # strat — BOTH adjacent-cell variants (the Veros kr loop): the upper
        # cell's (A-triads / faces BELOW their reference cell) and the lower
        # cell's (B-triads / faces ABOVE their reference cell).  The west/east
        # (south/north) shifts below pick the per-triad reference COLUMN; the
        # up/down variant picks the reference CELL within the column, so every
        # triad pairs its own cell's EOS derivatives in numerator AND
        # denominator (Veros K_11 ``drodze`` / K_33 ``drodzb``).
        drdT_w = drdT_c[:, :, :-1]
        drdS_w = drdS_c[:, :, :-1]
        drdT_wb = drdT_c[:, :, 1:]
        drdS_wb = drdS_c[:, :, 1:]
        drho_dz_w = jnp.minimum(drdT_w * dTdz_w + drdS_w * dSdz_w, -_EPS_DIV)
        drho_dz_w_b = jnp.minimum(
            drdT_wb * dTdz_w + drdS_wb * dSdz_w, -_EPS_DIV)
        # Per-reference-cell neutral horizontal gradients at the u/v-faces, used
        # by the u/v-face slopes (S_T*/S_V*).  drdT lifted to the west/east
        # (south/north) neighbour of each u-face (v-face).
        drdT_uw = _to_uface_west(drdT_c)   # (n_lat, n_lon+1, nlev)
        drdS_uw = _to_uface_west(drdS_c)
        drdT_ue = _to_uface_east(drdT_c)
        drdS_ue = _to_uface_east(drdS_c)
        drho_dx_u_west = drdT_uw * dTdx_u + drdS_uw * dSdx_u   # W-triad numerator
        drho_dx_u_east = drdT_ue * dTdx_u + drdS_ue * dSdx_u   # E-triad numerator
        drdT_vs = _to_vface_south(drdT_c)  # (n_lat+1, n_lon, nlev)
        drdS_vs = _to_vface_south(drdS_c)
        drdT_vn = _to_vface_north(drdT_c)
        drdS_vn = _to_vface_north(drdS_c)
        drho_dy_v_south = drdT_vs * dTdy_v + drdS_vs * dSdy_v
        drho_dy_v_north = drdT_vn * dTdy_v + drdS_vn * dSdy_v
        # Single drho_dx_u / drho_dy_v are unused in the neutral path (the W-face
        # builder takes the tracer gradients + drdT_w directly); set to None so
        # any accidental in-situ slice raises.
        drho_dx_u = None
        drho_dy_v = None
    else:
        drho_dx_u = gradient_x_cgrid(rho_filled, grid)        # (n_lat, n_lon+1, nlev)
        drho_dy_v = gradient_y_cgrid(rho_filled, grid)        # (n_lat+1, n_lon, nlev)
        drho_dz_raw = (rho_filled[:, :, :-1] - rho_filled[:, :, 1:]) / jnp.maximum(
            dz_half, _EPS_DIV
        )
        # Stable-strat floor: drho_dz must be negative (z UPWARD ⇒ rho denser below).
        drho_dz_w = jnp.minimum(drho_dz_raw, -_EPS_DIV)        # (n_lat, n_lon, nlev-1)
        # In-situ: the same single face gradient feeds all triads at a face.
        drho_dx_u_west = drho_dx_u
        drho_dx_u_east = drho_dx_u
        drho_dy_v_south = drho_dy_v
        drho_dy_v_north = drho_dy_v
        drdT_w = None
        drdS_w = None
        drdT_wb = None
        drdS_wb = None
        # In-situ has no per-cell EOS derivative; both adjacent cells share
        # the single face gradient (bit-identical to the prior code).
        drho_dz_w_b = drho_dz_w
        dTdx_u = dSdx_u = dTdy_v = dSdy_v = None

    # -----------------------------------------------------------------
    # 2. Pad drho_dz_w / dq_dz_w along the level axis so that boundary
    #    triads can be expressed by a single jnp.where / multiplication.
    #    "below" array at level k = drho_dz_w at w-face (k+1/2);
    #    "above" array at level k = drho_dz_w at w-face (k-1/2).
    #
    # kr pairing (neutral): a triad referencing cell k is the UPPER cell of
    # its below-face (k+1/2) ⇒ ``below_lev`` uses the upper-cell variant
    # ``drho_dz_w``; it is the LOWER cell of its above-face (k-1/2) ⇒
    # ``above_lev`` uses the lower-cell variant ``drho_dz_w_b`` (Veros
    # ``drodze``/``drodzn`` build each face gradient from the reference
    # cell's own drdT — isoneutral.py kr/ki loops).  in_situ: identical
    # arrays, bit-identical.
    # -----------------------------------------------------------------
    sentinel_rho = jnp.full(
        (n_lat, n_lon, 1), -_EPS_DIV, dtype=drho_dz_w.dtype,
    )
    drho_dz_below_lev = jnp.concatenate([drho_dz_w, sentinel_rho], axis=-1)
    drho_dz_above_lev = jnp.concatenate([sentinel_rho, drho_dz_w_b], axis=-1)

    sentinel_q = jnp.zeros((n_lat, n_lon, 1), dtype=dq_dz_w.dtype)
    dq_dz_below_lev = jnp.concatenate([dq_dz_w, sentinel_q], axis=-1)
    dq_dz_above_lev = jnp.concatenate([sentinel_q, dq_dz_w], axis=-1)

    valid_below = jnp.concatenate([
        jnp.ones((n_lat, n_lon, nlev - 1), dtype=drho_dz_w.dtype),
        jnp.zeros((n_lat, n_lon, 1), dtype=drho_dz_w.dtype),
    ], axis=-1)
    valid_above = jnp.concatenate([
        jnp.zeros((n_lat, n_lon, 1), dtype=drho_dz_w.dtype),
        jnp.ones((n_lat, n_lon, nlev - 1), dtype=drho_dz_w.dtype),
    ], axis=-1)

    # -----------------------------------------------------------------
    # 3. U-FACE triads — flux F_x at (n_lat, n_lon+1, nlev)
    #
    # IMPORTANT: the DM95 taper is applied to the *whole* per-triad
    # flux contribution, NOT to the raw slope.  Tapering the slope
    # before the cancellation
    #
    #     F_x^(m) = K_R · dq/dx + (K_R − K_GM) · taper · S · dq/dz^(m)
    #
    # would produce a residual ``K_R · (1 − taper) · dq/dx`` for
    # ``q = f(ρ)``, because the diagonal stays at full κ_R while the
    # off-diagonal is reduced by ``taper``.  Even at slopes well below
    # S_max the DM95 ``tanh`` saturates to ``1 − O(exp)``, so this
    # residual is *much* larger than float64 round-off.  Tapering the
    # whole triad — diagonal + off-diagonal together — preserves the
    # algebraic identity ``F_x^(m) = 0`` for ``q = f(ρ)`` and just
    # damps the genuine Redi flux when slopes saturate.
    # -----------------------------------------------------------------
    drho_dz_T1 = _to_uface_west(drho_dz_below_lev)
    drho_dz_T2 = _to_uface_west(drho_dz_above_lev)
    drho_dz_T3 = _to_uface_east(drho_dz_below_lev)
    drho_dz_T4 = _to_uface_east(drho_dz_above_lev)

    dq_dz_T1 = _to_uface_west(dq_dz_below_lev)
    dq_dz_T2 = _to_uface_west(dq_dz_above_lev)
    dq_dz_T3 = _to_uface_east(dq_dz_below_lev)
    dq_dz_T4 = _to_uface_east(dq_dz_above_lev)

    valid_T1 = _to_uface_west(valid_below)
    valid_T2 = _to_uface_west(valid_above)
    valid_T3 = _to_uface_east(valid_below)
    valid_T4 = _to_uface_east(valid_above)

    # Raw slopes — DO NOT taper here; taper goes on the whole flux.
    # T1/T2 reference the WEST cell, T3/T4 the EAST cell (Veros K_11 drodxe uses
    # drdT[1+ip]); in-situ both branches share the single face gradient so this
    # is bit-identical there.  in_situ clips to ±S_max (legoESM safety); neutral
    # leaves the slope UNCLIPPED (Veros relies on the DM95 taper — see
    # _w_triad_slopes_tapers) since neutral slopes routinely exceed S_max.
    _clip_slope = (slope_density != "neutral") or (slope_limit == "nemo_cap")

    def _uvslope(num, dz):
        # Sign-preserving denominator floor (same rationale as _slope above:
        # dry/unstratified columns give exact 0/0 -> NaN; wet stratified
        # columns bit-identical; NEMO ldfslp denominator-capping analogue).
        dz = jnp.where(jnp.abs(dz) > 1e-20, dz, -1e-20)
        s = -num / dz
        s = jnp.clip(s, -S_max, S_max) if _clip_slope else s
        # Adjoint stabilization: frozen-coefficient slopes (primal-invisible).
        return jax.lax.stop_gradient(s) if _sg_slopes else s

    S_T1 = _uvslope(drho_dx_u_west, drho_dz_T1)
    S_T2 = _uvslope(drho_dx_u_west, drho_dz_T2)
    S_T3 = _uvslope(drho_dx_u_east, drho_dz_T3)
    S_T4 = _uvslope(drho_dx_u_east, drho_dz_T4)

    if slope_limit == "nemo_cap":
        _tw_uv = lambda s: jnp.ones_like(s)
    else:
        _tw_uv = lambda s: dm95_taper_scalar(
            s, S_max, transition_width_frac=taper_width_frac,
            stop_gradient_taper=_sg_taper)[1]
    taper_T1 = _tw_uv(S_T1)
    taper_T2 = _tw_uv(S_T2)
    taper_T3 = _tw_uv(S_T3)
    taper_T4 = _tw_uv(S_T4)
    if act_T1 is not None:
        # Kill triads whose vertical pair crosses the seafloor (Veros taper→0
        # via the maskW-ed dTdz; the in_situ slope-clip would otherwise pin
        # the junk slope at S_max where the taper bottoms at 0.5).
        taper_T1 = taper_T1 * act_T1
        taper_T2 = taper_T2 * act_T2
        taper_T3 = taper_T3 * act_T3
        taper_T4 = taper_T4 * act_T4

    if veros_triad_weights:
        # Veros triad weights dzw(pair)/(4·dzt) (GMRediConfig field doc): 1-D
        # reference profiles broadcast over the horizontal axes.  The "below"
        # pair uses the centre spacing under the level (0 at the global
        # bottom — Veros's K_11 sumz updates [ki:] only, excluding the bottom
        # cell's below triads entirely, isoneutral.py:123-129); the "above"
        # pair uses the spacing above, with the SURFACE half-cell slot
        # dzw_sfc = 2·dzt[0] − dzw[1] at the top (u_centered_grid line 21).
        # Edge off-diagonal death: multiply the tapers by valid_* (Veros:
        # dTdz=0 ⇒ slope→∞ ⇒ dm_taper→0); the K_iso_steep diagonal deficit
        # then contributes steep·w for dead triads, exactly Veros's
        # max(K_iso_steep, K·taper)·dzw.
        _dzr_w = jnp.asarray(z_coord.dz_ref, dtype=q.dtype)
        _dzh_w = jnp.asarray(z_coord.dz_half_ref, dtype=q.dtype)
        _dzw_sfc = 2.0 * _dzr_w[:1] - _dzh_w[:1]
        _w_below = (jnp.concatenate([_dzh_w, jnp.zeros((1,), dtype=q.dtype)])
                    / (4.0 * _dzr_w))[jnp.newaxis, jnp.newaxis, :]
        _w_above = (jnp.concatenate([_dzw_sfc, _dzh_w])
                    / (4.0 * _dzr_w))[jnp.newaxis, jnp.newaxis, :]
        w_T1 = w_T3 = _w_below
        w_T2 = w_T4 = _w_above
        taper_T1 = taper_T1 * valid_T1
        taper_T2 = taper_T2 * valid_T2
        taper_T3 = taper_T3 * valid_T3
        taper_T4 = taper_T4 * valid_T4
    else:
        N_valid_u = valid_T1 + valid_T2 + valid_T3 + valid_T4
        N_valid_u_safe = jnp.maximum(N_valid_u, 1.0)
        w_T1 = valid_T1 / N_valid_u_safe
        w_T2 = valid_T2 / N_valid_u_safe
        w_T3 = valid_T3 / N_valid_u_safe
        w_T4 = valid_T4 / N_valid_u_safe

    # Per-triad full flux (cancels exactly when q = f(ρ)).
    flux_T1 = kappa_Redi_u * dq_dx_u + (kappa_Redi_u - kappa_GM_u) * S_T1 * dq_dz_T1
    flux_T2 = kappa_Redi_u * dq_dx_u + (kappa_Redi_u - kappa_GM_u) * S_T2 * dq_dz_T2
    flux_T3 = kappa_Redi_u * dq_dx_u + (kappa_Redi_u - kappa_GM_u) * S_T3 * dq_dz_T3
    flux_T4 = kappa_Redi_u * dq_dx_u + (kappa_Redi_u - kappa_GM_u) * S_T4 * dq_dz_T4

    F_x_u = (w_T1 * taper_T1 * flux_T1
           + w_T2 * taper_T2 * flux_T2
           + w_T3 * taper_T3 * flux_T3
           + w_T4 * taper_T4 * flux_T4)
    if K_iso_steep > 0.0:
        # Veros K_11 steep-slope floor (isoneutral.py:128): lift the along-
        # isopycnal diffusivity to K_iso_steep where DM95 tapered it below,
        # reverting to horizontal diffusion at steep slopes.  Additive deficit on
        # the diagonal (dq/dx) only; the skew is untouched.  K_iso_steep=0 ⇒ no-op.
        F_x_u = F_x_u + dq_dx_u * (
            w_T1 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_u * taper_T1)
            + w_T2 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_u * taper_T2)
            + w_T3 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_u * taper_T3)
            + w_T4 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_u * taper_T4))
    if double_diag_kappa is not None:
        # Veros double-counted diagonal: + K_11·dq/dx once more (taper-weighted
        # diagonal incl. its steep floor; see the GMRediConfig field doc).
        F_x_u = F_x_u + dq_dx_u * (
            w_T1 * taper_T1 + w_T2 * taper_T2
            + w_T3 * taper_T3 + w_T4 * taper_T4) * _ddk_u
        if double_diag_steep > 0.0:
            F_x_u = F_x_u + dq_dx_u * (
                w_T1 * jnp.maximum(0.0, double_diag_steep - _ddk_u * taper_T1)
                + w_T2 * jnp.maximum(0.0, double_diag_steep - _ddk_u * taper_T2)
                + w_T3 * jnp.maximum(0.0, double_diag_steep - _ddk_u * taper_T3)
                + w_T4 * jnp.maximum(0.0, double_diag_steep - _ddk_u * taper_T4))
    F_x_u = F_x_u * u_mask[:, :, jnp.newaxis] * u_mask_lvl

    # -----------------------------------------------------------------
    # 4. V-FACE triads — flux F_y at (n_lat+1, n_lon, nlev)
    # -----------------------------------------------------------------
    drho_dz_V1 = _to_vface_south(drho_dz_below_lev)
    drho_dz_V2 = _to_vface_south(drho_dz_above_lev)
    drho_dz_V3 = _to_vface_north(drho_dz_below_lev)
    drho_dz_V4 = _to_vface_north(drho_dz_above_lev)

    dq_dz_V1 = _to_vface_south(dq_dz_below_lev)
    dq_dz_V2 = _to_vface_south(dq_dz_above_lev)
    dq_dz_V3 = _to_vface_north(dq_dz_below_lev)
    dq_dz_V4 = _to_vface_north(dq_dz_above_lev)

    valid_V1 = _to_vface_south(valid_below)
    valid_V2 = _to_vface_south(valid_above)
    valid_V3 = _to_vface_north(valid_below)
    valid_V4 = _to_vface_north(valid_above)

    # At pole v-faces drho_dy_v = 0 ⇒ all S_V* = 0 ⇒ flux = K_R·dq_dy_v = 0
    # (gradient_y_cgrid sets dq_dy_v = 0 there); v_mask zeros the result.
    # V1/V2 reference the SOUTH cell, V3/V4 the NORTH cell (Veros K_22 drodyb);
    # in-situ both share the single face gradient (bit-identical).  neutral
    # leaves the slope unclipped (see S_T* above / _w_triad_slopes_tapers).
    S_V1 = _uvslope(drho_dy_v_south, drho_dz_V1)
    S_V2 = _uvslope(drho_dy_v_south, drho_dz_V2)
    S_V3 = _uvslope(drho_dy_v_north, drho_dz_V3)
    S_V4 = _uvslope(drho_dy_v_north, drho_dz_V4)

    taper_V1 = _tw_uv(S_V1)
    taper_V2 = _tw_uv(S_V2)
    taper_V3 = _tw_uv(S_V3)
    taper_V4 = _tw_uv(S_V4)
    if act_V1 is not None:
        # Vertical-pair seafloor kill, as for the u-face triads above.
        taper_V1 = taper_V1 * act_V1
        taper_V2 = taper_V2 * act_V2
        taper_V3 = taper_V3 * act_V3
        taper_V4 = taper_V4 * act_V4

    if veros_triad_weights:
        # Veros dzw(pair)/(4·dzt) weights + edge taper kill, as for the
        # u-face triads above (same 1-D profiles, broadcast).
        w_V1 = w_V3 = _w_below
        w_V2 = w_V4 = _w_above
        taper_V1 = taper_V1 * valid_V1
        taper_V2 = taper_V2 * valid_V2
        taper_V3 = taper_V3 * valid_V3
        taper_V4 = taper_V4 * valid_V4
    else:
        N_valid_v = valid_V1 + valid_V2 + valid_V3 + valid_V4
        N_valid_v_safe = jnp.maximum(N_valid_v, 1.0)
        w_V1 = valid_V1 / N_valid_v_safe
        w_V2 = valid_V2 / N_valid_v_safe
        w_V3 = valid_V3 / N_valid_v_safe
        w_V4 = valid_V4 / N_valid_v_safe

    flux_V1 = kappa_Redi_v * dq_dy_v + (kappa_Redi_v - kappa_GM_v) * S_V1 * dq_dz_V1
    flux_V2 = kappa_Redi_v * dq_dy_v + (kappa_Redi_v - kappa_GM_v) * S_V2 * dq_dz_V2
    flux_V3 = kappa_Redi_v * dq_dy_v + (kappa_Redi_v - kappa_GM_v) * S_V3 * dq_dz_V3
    flux_V4 = kappa_Redi_v * dq_dy_v + (kappa_Redi_v - kappa_GM_v) * S_V4 * dq_dz_V4

    F_y_v = (w_V1 * taper_V1 * flux_V1
           + w_V2 * taper_V2 * flux_V2
           + w_V3 * taper_V3 * flux_V3
           + w_V4 * taper_V4 * flux_V4)
    if K_iso_steep > 0.0:
        # Veros K_22 steep-slope floor (isoneutral.py:165), as for F_x_u above.
        F_y_v = F_y_v + dq_dy_v * (
            w_V1 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_v * taper_V1)
            + w_V2 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_v * taper_V2)
            + w_V3 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_v * taper_V3)
            + w_V4 * jnp.maximum(0.0, K_iso_steep - kappa_Redi_v * taper_V4))
    if double_diag_kappa is not None:
        # Veros double-counted diagonal: + K_22·dq/dy once more (see F_x_u).
        F_y_v = F_y_v + dq_dy_v * (
            w_V1 * taper_V1 + w_V2 * taper_V2
            + w_V3 * taper_V3 + w_V4 * taper_V4) * _ddk_v
        if double_diag_steep > 0.0:
            F_y_v = F_y_v + dq_dy_v * (
                w_V1 * jnp.maximum(0.0, double_diag_steep - _ddk_v * taper_V1)
                + w_V2 * jnp.maximum(0.0, double_diag_steep - _ddk_v * taper_V2)
                + w_V3 * jnp.maximum(0.0, double_diag_steep - _ddk_v * taper_V3)
                + w_V4 * jnp.maximum(0.0, double_diag_steep - _ddk_v * taper_V4))
    F_y_v = F_y_v * v_mask[:, :, jnp.newaxis] * v_mask_lvl

    # Horizontal divergence (single conservative call).
    dq_h = divergence_cgrid(F_x_u, F_y_v, grid)

    # -----------------------------------------------------------------
    # 5. W-FACE triads — flux F_z at (n_lat, n_lon, nlev-1)
    #
    # All eight triads at a w-face share the same drho_dz_w(k+1/2)
    # because the vertical pair (k, k+1) is fixed; they differ in their
    # horizontal pair.  The 4 x-triads use drho_dx_u at u-face j or j+1
    # and at level k (above the w-face) or k+1 (below).  The 4 y-triads
    # use drho_dy_v at v-face i or i+1 and at level k or k+1.
    # -----------------------------------------------------------------
    # Tracer gradients at the 8 triad positions ("A" = level k, "B" = k+1;
    # west/east u-faces and south/north v-faces).
    dq_dx_west_A = dq_dx_u[:, :n_lon, :-1]
    dq_dx_west_B = dq_dx_u[:, :n_lon, 1:]
    dq_dx_east_A = dq_dx_u[:, 1:n_lon + 1, :-1]
    dq_dx_east_B = dq_dx_u[:, 1:n_lon + 1, 1:]
    dq_dy_south_A = dq_dy_v[:n_lat, :, :-1]
    dq_dy_south_B = dq_dy_v[:n_lat, :, 1:]
    dq_dy_north_A = dq_dy_v[1:n_lat + 1, :, :-1]
    dq_dy_north_B = dq_dy_v[1:n_lat + 1, :, 1:]

    # rho-based per-triad slopes + DM95 tapers (shared with the K_33 getter so
    # the explicit F_z and the implicit K_33 use bit-identical slopes/tapers).
    # Neutral mode passes the W-cell drdT_w/drdS_w + tracer face gradients so the
    # W-face slopes use the same ∂ρ/∂T·∇T+∂ρ/∂S·∇S form Veros K_33 uses.
    (S_Wx1, S_Wx2, S_Wx3, S_Wx4, S_Wy1, S_Wy2, S_Wy3, S_Wy4,
     taper_Wx1, taper_Wx2, taper_Wx3, taper_Wx4,
     taper_Wy1, taper_Wy2, taper_Wy3, taper_Wy4) = _w_triad_slopes_tapers(
        drho_dx_u, drho_dy_v, drho_dz_w, n_lat, n_lon, S_max, taper_width_frac,
        slope_density=slope_density, slope_limit=slope_limit,
        drdT_w=drdT_w, drdS_w=drdS_w,
        drdT_wb=drdT_wb, drdS_wb=drdS_wb,
        drho_dz_w_b=(drho_dz_w_b if slope_density == "neutral" else None),
        dTdx_u=dTdx_u, dSdx_u=dSdx_u, dTdy_v=dTdy_v, dSdy_v=dSdy_v,
        u_face_act=_u_face_act, v_face_act=_v_face_act,
        adjoint_stabilization=adjoint_stabilization)

    # Per-triad vertical flux.  The off-diagonal skew (kR+kG)·S·dq/dx is ALWAYS
    # explicit.  The DIAGONAL K_33 term (kR·S²·dq/dz — the "enhanced vertical
    # mixing ∝ S²") is added here only when NOT ``implicit_K33``; otherwise it is
    # applied implicitly by the model step (Veros core/isoneutral/diffusion.py),
    # with ``compute_isoneutral_K33_latlon`` supplying the matching diffusivity.
    # The per-triad algebraic cancellation for q = f(ρ) is preserved either way
    # (each term cancels independently; taper × averaging keeps the exact zero).
    flux_Wx1 = (kappa_Redi_w + kappa_GM_w) * S_Wx1 * dq_dx_west_A
    flux_Wx2 = (kappa_Redi_w + kappa_GM_w) * S_Wx2 * dq_dx_east_A
    flux_Wx3 = (kappa_Redi_w + kappa_GM_w) * S_Wx3 * dq_dx_west_B
    flux_Wx4 = (kappa_Redi_w + kappa_GM_w) * S_Wx4 * dq_dx_east_B
    flux_Wy1 = (kappa_Redi_w + kappa_GM_w) * S_Wy1 * dq_dy_south_A
    flux_Wy2 = (kappa_Redi_w + kappa_GM_w) * S_Wy2 * dq_dy_north_A
    flux_Wy3 = (kappa_Redi_w + kappa_GM_w) * S_Wy3 * dq_dy_south_B
    flux_Wy4 = (kappa_Redi_w + kappa_GM_w) * S_Wy4 * dq_dy_north_B
    if not implicit_K33:
        flux_Wx1 = flux_Wx1 + kappa_Redi_w * S_Wx1 ** 2 * dq_dz_w
        flux_Wx2 = flux_Wx2 + kappa_Redi_w * S_Wx2 ** 2 * dq_dz_w
        flux_Wx3 = flux_Wx3 + kappa_Redi_w * S_Wx3 ** 2 * dq_dz_w
        flux_Wx4 = flux_Wx4 + kappa_Redi_w * S_Wx4 ** 2 * dq_dz_w
        flux_Wy1 = flux_Wy1 + kappa_Redi_w * S_Wy1 ** 2 * dq_dz_w
        flux_Wy2 = flux_Wy2 + kappa_Redi_w * S_Wy2 ** 2 * dq_dz_w
        flux_Wy3 = flux_Wy3 + kappa_Redi_w * S_Wy3 ** 2 * dq_dz_w
        flux_Wy4 = flux_Wy4 + kappa_Redi_w * S_Wy4 ** 2 * dq_dz_w

    F_z = 0.25 * (taper_Wx1 * flux_Wx1 + taper_Wx2 * flux_Wx2
                 + taper_Wx3 * flux_Wx3 + taper_Wx4 * flux_Wx4
                 + taper_Wy1 * flux_Wy1 + taper_Wy2 * flux_Wy2
                 + taper_Wy3 * flux_Wy3 + taper_Wy4 * flux_Wy4)
    # Veros maskW: no W-face flux through the seafloor (see the per-level
    # face-activity note above) — required for tracer conservation.
    F_z = F_z * w_mask_if

    dq_vert = vertical_flux_divergence(F_z, dz_actual, EPS)

    tendency = (dq_h + dq_vert) * mask[:, :, jnp.newaxis]
    if return_fluxes:
        # F_z is masked here so the contracted realized-conversion sees the same
        # wet-only flux the divergence does (F_x_u / F_y_v are already u/v-mask
        # multiplied above).
        return tendency, F_x_u, F_y_v, F_z * mask[:, :, jnp.newaxis]
    return tendency


# =====================================================================
# Top-level orchestrator (public API matching call site)
# =====================================================================

def gm_redi_density_and_jacobian(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    *,
    eos: str = "wright",
    eos_linear=None,
    eos_nemo_seos=None,
    mask: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    eos_depth: str = "insitu",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Shared GM/Redi in-situ density (2-iteration EOS coupling) + z* Jacobian.

    Both :func:`gm_redi_tracer_tendency_latlon` and
    :func:`compute_isoneutral_K33_latlon` need EXACTLY this ``rho`` (from the
    same 2-iteration EOS coupling) and ``jacobian`` from the SAME
    ``(T, S, eta, H_bathy)``.  When ``implicit_K33=True`` (the Veros/ACC
    production recipe) the model step calls BOTH per step, so each was
    recomputing the expensive 3-D EOS coupling independently — hoist it here,
    compute once, and pass via the ``density_jacobian`` argument (scaling
    review 2026-06-13 lever #3).  Pure; returns ``(rho, jacobian)``.

    eos_depth : str, default ``"insitu"``
        Depth convention fed to the EOS pressure term, matching
        :func:`legoesm.ocean.eos.compute_ocean_rho` /
        :func:`legoesm.ocean.dynamics.ocean_tendency_common.iterate_eos_and_pressure_anomaly`.
        ``"insitu"`` (default, BYTE-IDENTICAL): the 2-pass in-situ hydrostatic
        pressure integral.  ``"geometric"``: ``p = rho0*g*gdept`` from
        ``z_coord.t_depth_ref`` (NEMO ``gdept_1d``) — matches NEMO's
        ``eos_insitu``/S-EOS, which is written directly in terms of the
        geometric T-depth, not a self-consistent hydrostatic integral.  #1226:
        the mismatch left a depth-growing ``O(1e-6)`` density bias between the
        two conventions that is invisible in column-integrated diagnostics
        (``aeiu`` corr 1.000000) but dominates the genuinely tiny (``O(1e-9)``
        near the seafloor) raw isopycnal-slope density GRADIENT the GM eiv
        transport differences between adjacent columns — enough to flip its
        sign at the deepest active level and deflate the eiv transport ratio
        (u 0.994, v 0.982) even though every upstream ``kappa``/slope
        aggregate had already been verified.  ``eos_linear`` MUST be built
        with ``rho0=rho_0`` for ``"geometric"`` to cancel exactly (mirrors
        ``ocean_pe_latlon_cgrid.py``'s ``_eos_mk_kw`` pattern) — done here via
        ``make_eos_fn(eos, eos_linear, rho0=rho_0)``.
    """
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    _eos_mk_kw = {"rho0": rho_0} if eos_depth == "geometric" else {}
    eos_fn = make_eos_fn(
        eos, eos_linear, eos_nemo_seos=eos_nemo_seos, **_eos_mk_kw)
    fill_fn = lambda field: neumann_fill_cgrid(field, mask)
    # NEMO's eos_insitu evaluates at the LIVE gdept(Knn) = gdept_0*(1+r3t),
    # r3t = ssh/ht_0 (eosbn2.F90:541 `zh = gdept(ji,jj,jk,Knn)`), NOT the static
    # reference ladder.  Feeding the static one omitted a stretch of up to
    # 1.206 m on DINO and put a depth-STRUCTURED 2.559e-6 into prd -- the
    # residual floor inherited by all four ldf_slp rows (#1226).  Substituting
    # NEMO's own gdept collapsed prd to 1.804e-11, i.e. this term owned the
    # whole residual.  nemo_bn2_live_ladders is the canonical helper the
    # eos_rab/bn2 consumers already use (it takes H_bathy directly; do NOT use
    # compute_ocean_jacobian here -- that is (eta+H_bathy)/H_max, a different
    # quantity, off by median 1.1e-1 vs 2.5e-8) and it applies the same
    # t_depth_ref-or-|z_full_ref| fallback this previously did inline.
    # iterate_eos_and_pressure_anomaly's p_eos = rho0*g*gdept multiply is
    # shape-general, so a (nlat, nlon, nlev) depth needs no change there.
    # GATING mirrors the PGF sibling (ocean_pe_latlon_cgrid.py:1289-1294) so the
    # two never disagree on the EOS depth within one timestep: BIT-IDENTICAL to
    # the previous behaviour when t_depth_ref is None (no fidelity ladder — i.e.
    # every non-NEMO-bridged recipe), live stretch only when it is carried.
    # nemo_bn2_live_ladders itself honours linear_free_surface (key_linssh: the
    # column never stretches, so r3t == 0 and gdept(Kmm) == gdept_0).
    _geo_depth = (
        (jnp.abs(z_coord.z_full_ref)
         if getattr(z_coord, "t_depth_ref", None) is None
         else nemo_bn2_live_ladders(z_coord, eta, H_bathy)[0])
        if eos_depth == "geometric" else None
    )
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2,
        eos_depth=eos_depth, eos_geometric_depth_1d=_geo_depth,
    )
    return rho, jacobian


def _nemo_native_active_3d(
    mask: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    H_bathy: jnp.ndarray,
    dtype,
) -> jnp.ndarray:
    """3-D active (wet) mask for the ``compute_nemo_native_slopes`` family.

    #1226 (this row): prefer ``z_coord.is_active`` (``OceanPartialCellCoordinate``,
    ``vertical.py:514`` — an EXACT per-column integer index compare,
    ``k <= bottom_level``) over re-deriving a ``top-interface-depth < H_bathy``
    FLOAT comparison here.  The float form is what three ``nemo_iso_lap``
    call sites in this module (the Treguier-κ branch, the ``nemo_native``
    tendency branch, and the K33 branch) used to duplicate independently —
    and it is NOT robust: ``H_bathy`` (from the fidelity bridge, per-column
    ``cumsum(e3t_0)`` at the column's own ``bottom_level``) and
    ``cumsum(z_coord.dz_ref)`` (the GLOBAL representative ladder) are two
    independently-rounded quantities that can differ by a few ULPs. For a
    full-step column whose bottom lands exactly on a level interface this
    tie breaks the WRONG way often enough to matter: measured on the DINO Y5
    RUN_GDB twin (kt=57601), it spuriously marked the deepest level ACTIVE at
    ~1861/10348 T-columns, giving ``compute_nemo_native_slopes``'s ``zaj``
    stage a bottom-level (k=34 of 36) correlation of 0.98 against NEMO's own
    dumped ``eiv_dump_zgrv_iik.bin`` (vs 1.0 at every other level) — the
    FIRST deviating stage in the ldf_slp family's five-row debt
    (wslpi/wslpj/uslp/vslp/ldf_eiv-aeiu). Switching to ``is_active`` closes
    that level to corr 0.999999, matching the rest of the column. This is
    the SAME idiom :func:`compute_isopycnal_slopes_latlon_cgrid` already uses
    at its ``nemo_mld_slope_ramp`` branch (``getattr(z_coord, "is_active",
    None)``) — a pure z-star coordinate has no ``is_active`` attribute at
    all (no partial cells => no dry interior/bottom cells to mis-mask), so
    the float fallback there is inert, not wrong.
    """
    is_active = getattr(z_coord, "is_active", None)
    if is_active is not None:
        return (mask[:, :, jnp.newaxis] > 0.5).astype(dtype) * is_active.astype(dtype)
    z_top = jnp.cumsum(z_coord.dz_ref) - z_coord.dz_ref
    return (
        (mask[:, :, jnp.newaxis] > 0.5)
        & (z_top[jnp.newaxis, jnp.newaxis, :] < H_bathy[:, :, jnp.newaxis])
    ).astype(dtype)


# NEMO 5.0.1 src/OCE/LDF/ldftra.F90:427: fixed fraction, not a namelist knob.
NEMO21_REDI_FLOOR_FRACTION = 0.2


def validate_redi_coefficient(cfg):
    """Reject unsupported or inert Redi selections before tracing physics."""
    if cfg.redi_coefficient not in ("constant", "nemo21"):
        raise ValueError(f"unknown redi_coefficient {cfg.redi_coefficient!r}")
    if cfg.redi_coefficient == "constant":
        if cfg.redi_f_f is not None:
            raise ValueError("redi_f_f requires redi_coefficient='nemo21'")
        return
    if (cfg.slope_scheme != "nemo_iso_lap"
            or cfg.slope_positions != "nemo_native"):
        raise ValueError("nemo21 requires nemo_iso_lap and nemo_native slopes")
    if not cfg.treguier.enabled or cfg.visbeck.enabled or cfg.eke is not None:
        raise ValueError("nemo21 requires Treguier alone (no Visbeck/EKE)")
    validate_treguier_cfg(cfg.treguier)
    if cfg.treguier.kappa_min != 0:
        raise ValueError("nemo21 requires unfloored GM: gm_kappa_min=0")
    if cfg.resolution_function or cfg.kappa_redi_lat_scaling:
        raise ValueError("nemo21 conflicts with resolution/latitude scaling")
    if cfg.redi_f_f is None:
        raise ValueError("nemo21 requires exact mesh redi_f_f (no averaged-f fallback)")
    if not isinstance(cfg.redi_aht0, jax.core.Tracer):
        import math
        if not math.isfinite(float(cfg.redi_aht0)) or cfg.redi_aht0 <= 0:
            raise ValueError("redi_aht0 must be finite and positive")


def nemo21_redi_from_gm(kappa_t, f_t, f_f, u_mask, v_mask, aht0, omega, *, grid=None):
    """NEMO 5.0.1 ldftra:419-441, AFTER masked GM face averaging.

    Inputs/outputs are cell-indexed east/north faces, m2/s, nonnegative.
    f_t and f_f are signed Coriolis at T and NE F points, respectively.
    The ABS makes the enhancement hemisphere-symmetric. Depth masking is
    applied by the existing explicit tensor and implicit K33 consumers.
    """
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        TREGUIER_TAPER_LAT_DEG,
    )
    if f_f.shape != kappa_t.shape or f_t.shape != kappa_t.shape:
        raise ValueError("nemo21 Coriolis fields must match the local T grid")
    if grid is None:
        ku, kv = nemo_kappa_gm_to_faces(kappa_t, u_mask, v_mask)
    else:
        # Same two-point mean, with the existing scalar halo/fold handling.
        ku = interp_cell_to_uface(kappa_t)[:, 1:] * u_mask
        kv = interp_cell_to_vface(kappa_t, grid)[1:, :] * v_mask
    floor = NEMO21_REDI_FLOOR_FRACTION * aht0
    increment = aht0 - floor
    f20 = 2.0 * omega * jnp.sin(jnp.deg2rad(TREGUIER_TAPER_LAT_DEG))
    ku = (jnp.maximum(floor, ku)
          + (1.0 - jnp.minimum(1.0, jnp.abs(f_t) / f20)) * increment)
    kv = (jnp.maximum(floor, kv)
          + (1.0 - jnp.minimum(1.0, jnp.abs(f_f) / f20)) * increment)
    return ku * u_mask, kv * v_mask


def native_treguier_kappa_for_state(
    rho, T, S, mask, u_mask, v_mask, z_coord, grid, cfg, eos_fn,
    jacobian, eta, H_bathy, prd_jacobian, prd_TS, pn2, e3w,
    f_coriolis, rho_0, g, omega,
):
    """Shared native GM diagnosis for tracer fluxes and the Redi K33 split."""
    _act_kgm = _nemo_native_active_3d(mask, z_coord, H_bathy, T.dtype)
    _uslp_kgm, _vslp_kgm, _wslpi_kgm, _wslpj_kgm = compute_nemo_native_slopes(
        rho, T, S, mask, u_mask, v_mask, z_coord, grid, cfg, eos_fn,
        jacobian=jacobian, eta=eta, H_bathy=H_bathy,
        prd_jacobian=prd_jacobian,
        prd_TS_override=prd_TS,
        pn2_override=pn2,
        e3w_override=e3w,
        rho_0=rho_0, g=g, active_3d=_act_kgm,
    )
    kappa_GM = compute_treguier_kappa_gm_nemo_native(
        rho, T, S, _wslpi_kgm, _wslpj_kgm, mask, z_coord, grid,
        f_coriolis, cfg.treguier, eos_fn, rho_0=rho_0, g=g, active_3d=_act_kgm,
        # slope_n2 lives on the PARENT GMRediConfig, not on cfg.treguier.
        # Passing cfg.treguier alone made getattr(cfg,'slope_n2',..) silently
        # fall back to 'adiabatic' while the slopes two lines above ran
        # 'nemo_bn2' -- an internally INCONSISTENT kappa that carried a
        # 1.6% aeiu deficit (#1226).  Explicit params, no fallback.
        slope_n2=getattr(cfg, "slope_n2", "adiabatic"),
        jacobian=jacobian,
        omega=omega,
        vertical_reduction_evaluation=getattr(
            cfg, "treguier_vertical_reduction_evaluation", "tree"),
        sqrt_evaluation=getattr(
            cfg, "treguier_sqrt_evaluation", "guarded_floor"),
        pn2_override=pn2,
        e3w_override=e3w,
    )
    return kappa_GM


def gm_redi_tracer_tendency_latlon(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    *,
    eos: str = "wright",
    eos_linear=None,
    eos_nemo_seos=None,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    f_coriolis: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    omega: float = constants.Omega,
    kappa_gm_override: jnp.ndarray | None = None,
    kappa_redi_override: jnp.ndarray | None = None,
    kappa_redi_v_override: jnp.ndarray | None = None,
    density_jacobian: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    native_prd_jacobian: jnp.ndarray | None = None,
    native_prd_TS: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    native_slope_pn2: jnp.ndarray | None = None,
    native_slope_e3w: jnp.ndarray | None = None,
    native_slope_eta: jnp.ndarray | None = None,
    native_kappa_slope_eta: jnp.ndarray | None = None,
    native_bolus_slope_eta: jnp.ndarray | None = None,
    redi_flux_eta: jnp.ndarray | None = None,
    redi_divisor_eta: jnp.ndarray | None = None,
    dt: float | None = None,
    return_bolus_transport: bool = False,
    return_redi_diagnostics: bool = False, return_redi_slope_diagnostics: bool = False, native_slope_nmln_override: jnp.ndarray | None = None,
    redi_face_thickness_override: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    redi_divisor_thickness_override: jnp.ndarray | None = None,
    redi_closed_bottom_wmask_override: bool | None = None,
    redi_horizontal_flux_evaluation_override: str | None = None,
    redi_area_reciprocal_override: jnp.ndarray | None = None,
    redi_area_reciprocal_evaluation_override: str | None = None,
    redi_final_update_evaluation_override: str | None = None,
    redi_rhs_accumulator_override: jnp.ndarray | None = None,
    eos_depth: str = "insitu",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-level GM/Redi for lat-lon C-grid.  ``kappa_redi_v_override``:
    NEMO's independently-evaluated ahtv (nn_aht_ijk_t=20 static lat-scaling
    case only); ``None`` (every other closure) reuses ``kappa_redi_override``
    for the v-face too, bit-identical to before.

    Computes density, isopycnal slopes, optional Visbeck coefficient,
    then returns tracer tendencies for T and S.

    Parameters
    ----------
    T, S : (n_lat, n_lon, nlev)
    eta : (n_lat, n_lon)
        Sea-surface height.
    redi_flux_eta : (n_lat, n_lon) or None
        Optional carried Kmm sea-surface height used only to build
        traldf_iso's live-QCO e3u/e3v flux faces. NEMO builds the lateral
        tracer flux after the dynamics update but indexes these faces at Kmm,
        so the MLF model step supplies its step-entry eta here. ``None`` keeps
        same-level callers byte-identical.
    redi_divisor_eta : (n_lat, n_lon) or None
        Kmm sea-surface height consumed by ``traldf_iso``'s live T-thickness
        divisor. On WS-RK3 this is the stage-2 output entering stage 3, not
        the step-entry height used by the once-per-step slope calculation.
    H_bathy : (n_lat, n_lon)
        Bottom depth (positive).
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    cfg : GMRediConfig
    eos : str
        Equation of state ("wright" or "linear").
    eos_linear : LinearEOSConfig or None
    mask : (n_lat, n_lon) ocean mask
    u_mask : (n_lat, n_lon+1) u-face mask
    v_mask : (n_lat+1, n_lon) v-face mask
    f_coriolis : (n_lat, n_lon) Coriolis parameter
    omega : Earth rotation rate [rad/s]. MUST match the value used to build
        ``grid``/``f_coriolis`` (see :func:`compute_treguier_kappa_gm_nemo_native`)
        — only the Treguier ``gm_kappa_scheme="treguier"`` path reads it, for
        the ``f20`` tropical-taper reference (#1226).
    eos_depth : str, default ``"insitu"``
        Forwarded to :func:`gm_redi_density_and_jacobian` (see its docstring)
        when ``density_jacobian`` is not already hoisted.  Ignored when
        ``density_jacobian`` is provided (the caller already fixed the
        convention).

    Returns
    -------
    dT_dt, dS_dt : (n_lat, n_lon, nlev)
    """
    validate_redi_coefficient(cfg)
    if cfg.redi_coefficient == "nemo21" and (
            kappa_gm_override is not None or kappa_redi_override is not None
            or kappa_redi_v_override is not None):
        raise ValueError("nemo21 conflicts with runtime coefficient overrides")
    # Default masks: all ocean.
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    if u_mask is None:
        u_mask = jnp.ones((T.shape[0], T.shape[1] + 1), dtype=T.dtype)
    if v_mask is None:
        v_mask = jnp.ones((T.shape[0] + 1, T.shape[1]), dtype=T.dtype)

    # Jacobian + density via 2-iteration EOS coupling.  Hoistable: when the
    # model step also computes K_33 (implicit_K33), it passes the SAME
    # (rho, jacobian) here so the expensive 3-D EOS coupling runs once
    # (scaling review lever #3).  density_jacobian=None => compute inline,
    # bit-identical to the pre-hoist path.  Reuse invariant: ONLY
    # (rho, jacobian) are hoisted; the function still receives the current
    # eta/H_bathy and any other use of them (e.g. the Visbeck
    # n2_mode="adiabatic" partial-cell h_actual/pressure path) recomputes
    # independently from those same current values, so a precomputed
    # density_jacobian never makes them stale.
    # #1226: eos_fn feeds compute_nemo_native_slopes's N^2 (_nemo_wpoint_e3w_
    # wmask_n2) and the Visbeck neutral-gradient mode below — build it with
    # the SAME rho0-cancelling convention as gm_redi_density_and_jacobian's
    # rho, or the two disagree by the eos_depth="geometric" vs "insitu"
    # residual (a depth-growing O(1e-6) bias that dominates the genuinely
    # tiny near-seafloor density gradient and sign-flips the eiv transport;
    # see gm_redi_density_and_jacobian's docstring).
    _eos_mk_kw = {"rho0": rho_0} if eos_depth == "geometric" else {}
    _native_eta = eta if native_slope_eta is None else native_slope_eta
    _kappa_native_eta = (
        eta if native_kappa_slope_eta is None else native_kappa_slope_eta)
    eos_fn = make_eos_fn(
        eos, eos_linear, eos_nemo_seos=eos_nemo_seos, **_eos_mk_kw)
    if density_jacobian is None:
        rho, jacobian = gm_redi_density_and_jacobian(
            T, S, eta, H_bathy, grid, z_coord,
            eos=eos, eos_linear=eos_linear, eos_nemo_seos=eos_nemo_seos,
            mask=mask, rho_0=rho_0, g=g,
            eos_depth=eos_depth,
        )
    else:
        rho, jacobian = density_jacobian
    # Native ldf_slp is allowed to consume a different time level of z-star
    # geometry than the later tracer-volume operator.  NEMO stp_MLF evaluates
    # eos(ts,Nbb) and ldf_slp on gdept(Nbb), while tra_ldf subsequently applies
    # those carried slopes in the current step.  None keeps every legacy path
    # byte-identical.
    _native_prd_J = jacobian if native_prd_jacobian is None else native_prd_jacobian

    slope_density = getattr(cfg, "slope_density", "in_situ")

    # Centred interface slopes — used for Visbeck (only ⟨N|S|⟩_z is
    # needed; the cancellation property of triads is irrelevant there).
    # T, S, eos_fn are passed so the neutral mode can build the
    # locally-referenced gradient; ignored for the default in-situ mode.
    S_x, S_y, _taper = compute_isopycnal_slopes_latlon_cgrid(
        rho, mask, z_coord, jacobian, grid, cfg,
        T=T, S=S, eos_fn=eos_fn, rho_0=rho_0, g=g,
    )

    # GM coefficient. Precedence: prognostic-EKE override (computed by the step
    # from the evolving eddy-energy field) > Visbeck diagnostic > constant.
    _treg = getattr(cfg, "treguier", None)
    if _treg is not None and _treg.enabled and cfg.visbeck.enabled:
        raise ValueError(
            "GMRediConfig: visbeck.enabled and treguier.enabled are mutually "
            "exclusive adaptive-kappa diagnostics — enable exactly one.")
    if _treg is not None and _treg.enabled and kappa_gm_override is not None:
        # The override (prognostic EKE, built by the step whenever
        # ``gm_redi.eke is not None``) is consumed BEFORE Treguier below, so
        # this combination would silently run the EKE coefficient -- and its
        # own [0, kappa_max] clip, NOT the Treguier taper/cap/floor -- while
        # the user believes the selected NEMO ldf_eiv scheme is active.
        raise ValueError(
            "GMRediConfig: treguier.enabled with a prognostic-EKE kappa_GM "
            "override (gm_redi.eke) — the EKE override takes precedence and "
            "the Treguier coefficient would never reach the operator. Enable "
            "exactly one of eke / treguier.")
    if _treg is not None:
        # Concrete-value check only (skipped for a trained/traced aei0); the
        # kappa_min <= aei0 invariant itself is enforced in the kernels.
        validate_treguier_cfg(_treg)
    if kappa_gm_override is not None:
        kappa_GM = kappa_gm_override
    elif _treg is not None and _treg.enabled:
        # Treguier-1997 / NEMO nn_aei_ijk_t=21 adaptive κ (the oracle scaling).
        if f_coriolis is None:
            f_coriolis = jnp.broadcast_to(grid.f, mask.shape)
        # #1317: with slope_positions="nemo_native" the Redi/GM flux (and the
        # bolus this kappa_GM feeds) is built from compute_nemo_native_slopes's
        # wslpi/wslpj (the true ldfslp W-point slopes, incl. the mixed-layer
        # linear ramp) — NEMO's own ldf_eiv (ldftra.F90:664-706) sums over
        # those SAME arrays. Feeding the generic S_x/S_y (cell-centred,
        # interior-interfaces-only, DM95/nemo_cap tapered — no ML ramp, no
        # surface w-point) produces a materially different κ_GM (verified:
        # corr=0.28, mean 646 vs 196 m²/s on the DINO day-0 twin) — the
        # dominant source of the #1317 ADVECTION-bucket (bolus-inclusive)
        # residual. Only meaningful under nemo_iso_lap+nemo_native (the sole
        # config that builds wslpi/wslpj at all); every other slope_scheme/
        # slope_positions combination keeps the byte-identical generic path.
        if (getattr(cfg, "slope_scheme", "triads") == "nemo_iso_lap"
                and getattr(cfg, "slope_positions", "mode_b") == "nemo_native"):
            kappa_GM = native_treguier_kappa_for_state(
                rho, T, S, mask, u_mask, v_mask, z_coord, grid, cfg, eos_fn,
                jacobian, _kappa_native_eta, H_bathy, _native_prd_J,
                native_prd_TS, native_slope_pn2, native_slope_e3w,
                f_coriolis, rho_0, g, omega,
            )
        else:
            kappa_GM = compute_treguier_kappa_gm(
                rho, S_x, S_y, z_coord, jacobian, f_coriolis, _treg,
                rho_ref=rho_0, g=g,
                omega=omega,
            )
    elif cfg.visbeck.enabled:
        if f_coriolis is None:
            # Use the grid's Coriolis field (f = 2·Ω·sin(lat), already built
            # with the grid's pinned rotation rate) rather than re-deriving from
            # the module constant — keeps Visbeck consistent with the pinned Ω
            # (e.g. a Veros recipe) and avoids an inline constants.Omega read.
            f_coriolis = jnp.broadcast_to(grid.f, mask.shape)
        # Adiabatic Eady-N² for the Visbeck diagnostic (default insitu ⇒
        # kwargs None ⇒ byte-identical). Build the cell-centre pressure +
        # pass T/S/EOS only when the Visbeck config opts in.
        _T_vb = _S_vb = _p_vb = _eos_vb = None
        if getattr(cfg.visbeck, "n2_mode", "insitu") == "adiabatic":
            from legoesm.ocean.eos import compute_hydrostatic_pressure
            from legoesm.ocean.vertical import (
                OceanPartialCellCoordinate, compute_layer_thickness,
            )
            _h_actual = None
            if isinstance(z_coord, OceanPartialCellCoordinate):
                _h_actual = compute_layer_thickness(eta, H_bathy, z_coord)
            _p_vb = compute_hydrostatic_pressure(
                rho, eta, z_coord.dz_ref, jacobian, rho_0, g,
                h_actual=_h_actual,
            )
            _T_vb, _S_vb, _eos_vb = T, S, eos_fn
        kappa_GM = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg.visbeck,
            # The SAME pair the pressure above was built from. Threading the
            # density and defaulting the gravity is how this path kept a
            # library-gravity buoyancy frequency under a configured pressure.
            rho_ref=rho_0, g=g,
            T=_T_vb, S=_S_vb, p_cell=_p_vb, eos_fn=_eos_vb,
        )
    else:
        kappa_GM = cfg.kappa_GM

    # Hallberg (2013) resolution taper of the GM coefficient (default off =>
    # byte-identical). Static Python gate on the config bool (feature-gating
    # exception; no traced branching). Scales whatever the closure produced
    # (override / Treguier / Visbeck / constant); the Redi diffusivity
    # (kappa_Redi_eff below) is intentionally left unscaled -- GM-only.
    # BUDGET COUPLING: when a prognostic EKE/GEOMETRIC closure supplies the
    # override, the model step scales the GM-derived EKE production by the
    # SAME f_res (gm_resolution_factor -- one definition), so the eddy-energy
    # budget sees the conversion this scaled kappa actually performs (codex
    # MED-3 r2; see the resolution_function note in config.py).
    if getattr(cfg, "resolution_function", False):
        if f_coriolis is None:
            f_coriolis = jnp.broadcast_to(grid.f, mask.shape)
        kappa_GM = gm_resolution_scaled_kappa(
            kappa_GM, f_coriolis, jnp.sqrt(grid.area),
            cfg.resfn_gamma, cfg.resfn_cbcl_ms,
        )

    # Redi isopycnal diffusivity. K_iso = K_gm (prognostic) when the override is
    # supplied (Veros enable_eke_isopycnal_diffusion -> the step passes
    # kappa_redi_override = kappa_gm_override); else the constant cfg.kappa_Redi.
    kappa_Redi_eff = cfg.kappa_Redi if kappa_redi_override is None else kappa_redi_override
    # v-face analogue: None unless the static lat-scaling override supplied a
    # genuinely distinct ahtv (see nemo_iso_lap_tracer_tendency_latlon_cgrid).
    kappa_Redi_v_eff = kappa_redi_v_override
    if cfg.redi_coefficient == "nemo21":
        kappa_Redi_eff, kappa_Redi_v_eff = nemo21_redi_from_gm(
            kappa_GM, jnp.broadcast_to(grid.f, mask.shape), cfg.redi_f_f,
            u_mask[:, 1:], v_mask[1:, :], cfg.redi_aht0, omega, grid=grid)


    scheme = getattr(cfg, "slope_scheme", "triads")
    # Guard (codex r5-r7): msc_stabilize (ln_traldf_msc) is implemented ONLY
    # on the nemo_iso_lap scheme with native slopes and an implicit K33 —
    # anywhere else the flag would be silently ignored (or its akz portion
    # silently dropped).  Validate at FN ENTRY on the static config so every
    # scheme branch is covered.
    if getattr(cfg, "msc_stabilize", False):
        if scheme != "nemo_iso_lap":
            raise ValueError(
                "GMRediConfig: msc_stabilize=True (ln_traldf_msc) requires "
                f"slope_scheme='nemo_iso_lap'; with {scheme!r} the flag "
                "would be silently ignored.")
        if not cfg.implicit_K33:
            raise ValueError(
                "GMRediConfig: msc_stabilize=True (ln_traldf_msc) requires "
                "implicit_K33=True — the capped akz must be applied by the "
                "implicit vertical solve; without it the akz part of the "
                "a33 diagonal is silently dropped.")
        if getattr(cfg, "slope_positions", "mode_b") != "nemo_native":
            raise ValueError(
                "GMRediConfig: msc_stabilize=True (ln_traldf_msc) requires "
                "slope_positions='nemo_native' — the MSC split is "
                "implemented on the native ldfslp stencil only.")
    if return_bolus_transport and scheme != "nemo_iso_lap":
        raise ValueError(
            "gm_redi_tracer_tendency_latlon(return_bolus_transport=True) is only "
            f"supported by slope_scheme='nemo_iso_lap', got {scheme!r}.")
    if return_redi_diagnostics and (
            scheme != "nemo_iso_lap" or return_bolus_transport):
        raise ValueError(
            "return_redi_diagnostics requires the nemo_iso_lap tracer "
            "operator without a bolus-transport return")
    _slope_limit = getattr(cfg, "slope_limit", "dm95_taper")
    validate_slope_limit(_slope_limit)
    # nemo_cap is wired for BOTH the triads and the centered/nemo_iso_lap
    # slope paths (2026-07-16: the centered path previously kept the DM95
    # taper, which kills the flux at steep ML-base outcrops where NEMO's
    # cap keeps pumping — the subduction pathway; plan §G).
    _double_diag = bool(getattr(cfg, "double_redi_diagonal", False))
    _vtw = bool(getattr(cfg, "veros_triad_weights", False))
    _adj_stab = getattr(cfg, "adjoint_stabilization", "none")
    validate_adjoint_stabilization(_adj_stab)
    if (_double_diag or _vtw) and scheme != "triads":
        raise ValueError(
            "GMRediConfig.double_redi_diagonal / veros_triad_weights (the "
            "Veros-faithful triad options) are only supported by "
            "slope_scheme='triads'.")
    _ddk = kappa_Redi_eff if _double_diag else None
    if scheme == "triads":
        dT_dt = gm_redi_tracer_tendency_triads_latlon_cgrid(
            T, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, kappa_Redi_eff, cfg.S_max,
            cfg.taper_width_frac, cfg.implicit_K33, cfg.K_iso_steep,
            slope_density=slope_density, slope_limit=_slope_limit,
            T_tracer=T, S_tracer=S,
            eos_fn=eos_fn, rho_0=rho_0, g=g,
            double_diag_kappa=_ddk, double_diag_steep=cfg.K_iso_steep,
            veros_triad_weights=_vtw,
            adjoint_stabilization=_adj_stab,
        )
        dS_dt = gm_redi_tracer_tendency_triads_latlon_cgrid(
            S, rho, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, kappa_Redi_eff, cfg.S_max,
            cfg.taper_width_frac, cfg.implicit_K33, cfg.K_iso_steep,
            slope_density=slope_density, slope_limit=_slope_limit,
            T_tracer=T, S_tracer=S,
            eos_fn=eos_fn, rho_0=rho_0, g=g,
            double_diag_kappa=_ddk, double_diag_steep=cfg.K_iso_steep,
            veros_triad_weights=_vtw,
            adjoint_stabilization=_adj_stab,
        )
    elif scheme == "centered":
        dT_dt = gm_redi_tracer_tendency_latlon_cgrid(
            T, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, kappa_Redi_eff,
        )
        dS_dt = gm_redi_tracer_tendency_latlon_cgrid(
            S, S_x, S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_GM, kappa_Redi_eff,
        )
        # --- Near-surface horizontal diffusion complement ---
        # In the mixed layer, DM95 tapers Redi to zero, leaving no
        # horizontal tracer mixing.  Following Ferrari et al. (2008,
        # J. Climate, 21, 2770-2789), add horizontal diffusion with
        # coefficient kappa_Redi in the boundary layer so total
        # diffusivity is always kappa_Redi.
        #
        # Use a fixed depth proxy rather than the DM95 taper,
        # because the taper-based complement was ineffective (taper ≈ 1
        # where the 2Δy feedback operates).  When KPP is active, this
        # should be replaced with the KPP-diagnosed boundary layer depth.
        if cfg.surface_complement:
            z_full = z_coord.z_full_ref  # (nlev,) — negative depths
            complement = jnp.where(
                jnp.abs(z_full) < cfg.surface_complement_depth, 1.0, 0.0
            )  # (nlev,) — broadcast over (n_lat, n_lon)
            for q_field, tend_ref in [(T, 'dT_dt'), (S, 'dS_dt')]:
                q_filled = neumann_fill_cgrid(q_field, mask)
                dq_dx_u = gradient_x_cgrid(q_filled, grid) * u_mask[:, :, jnp.newaxis]
                dq_dy_v = gradient_y_cgrid(q_filled, grid) * v_mask[:, :, jnp.newaxis]
                # complement is (nlev,) — broadcasts over spatial dims. This
                # boundary-layer term deliberately uses the CONSTANT cfg.kappa_Redi,
                # NOT kappa_Redi_eff: the K_iso=K_gm override applies to the
                # isopycnal-tensor fluxes (above); the Ferrari (2008) surface
                # complement is a separate fixed-diffusivity term. Scoped limitation
                # for a centered + surface_complement + prognostic-override config
                # (making it override-aware needs cell->u/v-face interp of the
                # array kappa); INERT for ACC, which uses slope_scheme="triads".
                F_x = cfg.kappa_Redi * complement * dq_dx_u
                F_y = cfg.kappa_Redi * complement * dq_dy_v
                dq_complement = divergence_cgrid(F_x, F_y, grid) * mask[:, :, jnp.newaxis]
                if tend_ref == 'dT_dt':
                    dT_dt = dT_dt + dq_complement
                else:
                    dS_dt = dS_dt + dq_complement
    elif scheme == "nemo_iso_lap":
        # GM bolus advection form (NEMO ldf_eiv_trp): "centred" (default,
        # bit-identical) applies the bolus as a 2nd-order centred flux INSIDE the
        # iso operator; "through_fct" exports the bolus transport so the model
        # step adds it to the advecting mass flux → the bolus passes through the
        # monotone FCT limiter (NEMO traadv).  Validated on the static config here
        # (fn entry) so an unknown value fails loudly even when kappa_GM=0.
        _gm_bolus = getattr(cfg, "gm_bolus_advection", "centred")
        # NEMO face-averages kappa onto U/V before building psi
        # (ldftra.F90:716-718).  Selected by the oracle card; default False
        # keeps legacy runs bit-identical.  Read DIRECTLY (no getattr default):
        # a wrong config object must raise, not silently disable -- the same
        # silent-fallback pattern hid the slope_n2 bug for a day (#1226).
        _gm_kfa = cfg.gm_bolus_kappa_face_average
        if _gm_bolus not in ("centred", "through_fct"):
            raise ValueError(
                "GMRediConfig.gm_bolus_advection must be 'centred' or "
                f"'through_fct', got {_gm_bolus!r}.")
        if return_bolus_transport and _gm_bolus != "through_fct":
            raise ValueError(
                "gm_redi_tracer_tendency_latlon(return_bolus_transport=True) "
                "requires gm_bolus_advection='through_fct' (else the bolus is "
                "already applied in-operator — exporting it would double-count).")
        # NEMO traldf_iso (iso_lap) rotated-Laplacian iso-neutral Redi, plus the
        # ln_ldfeiv GM bolus (ldf_eiv_trp_MLF) when kappa_GM != 0 — the faithful
        # NEMO isoneutral-Redi + GM combination (the Madec discretization used by
        # the DINO / nemo_dino_kamm oracle).  kappa_GM flows into the tendency's
        # streamfunction bolus; kappa_GM=0 recovers pure Redi bit-for-bit.
        # 3-D wet mask (NEMO tmask): a cell is water iff its column is wet
        # (2-D mask) AND its level is active (see _nemo_native_active_3d --
        # prefers z_coord.is_active's exact per-column integer bottom-level
        # compare over a float top-depth-vs-H_bathy tie, #1226).
        _active_3d = _nemo_native_active_3d(mask, z_coord, H_bathy, T.dtype)
        _flux_face_mode = cfg.redi_flux_face_thickness_evaluation
        if _flux_face_mode not in ("tpoint_jacobian", "nemo_qco_live"):
            raise ValueError(
                "unknown GMRediConfig.redi_flux_face_thickness_evaluation "
                f"{_flux_face_mode!r}; expected 'tpoint_jacobian' or "
                "'nemo_qco_live'")
        _divisor_mode = cfg.redi_divisor_thickness_evaluation
        if _divisor_mode not in ("reference_jacobian", "nemo_qco_live"):
            raise ValueError(
                "unknown GMRediConfig.redi_divisor_thickness_evaluation "
                f"{_divisor_mode!r}; expected 'reference_jacobian' or "
                "'nemo_qco_live'")
        _flux_e3u = None
        _flux_e3v = None
        _divisor_e3t = redi_divisor_thickness_override
        _needs_raw_e3t = (
            _flux_face_mode == "nemo_qco_live"
            or (_divisor_mode == "nemo_qco_live" and _divisor_e3t is None))
        _e3t0 = None
        if _needs_raw_e3t:
            if eta is None:
                raise ValueError(
                    "live Redi face or divisor thickness requires NOW "
                    "sea-surface height")
            _e3t0 = getattr(z_coord, "nemo_e3t_0", None)
            if _e3t0 is None:
                raise ValueError(
                    "live Redi face or divisor thickness requires raw NEMO "
                    "e3t_0")
            _e3t0 = jnp.asarray(_e3t0, dtype=T.dtype)[..., :T.shape[-1]]
        if _flux_face_mode == "nemo_qco_live":
            _umask3, _vmask3, _ = nemo_iso_face_masks(
                u_mask, v_mask, _active_3d)
            _flux_eta = eta if redi_flux_eta is None else redi_flux_eta
            _flux_e3u, _flux_e3v = nemo_qco_live_face_thicknesses(
                _flux_eta, z_coord, _e3t0, _e3t0, _umask3, _vmask3)
        if _divisor_mode == "nemo_qco_live" and _divisor_e3t is None:
            # traldf_iso.f90:306-310/:327-331 divides the flux divergence by
            # e3t(Kmm). NEMO's r3t divisor uses ht_0, the source-ordered sum
            # of e3t_0*tmask (domain.f90:193-212), independently of which
            # horizontal face-thickness construction the card selects.
            _ht0 = jnp.zeros_like(eta, dtype=T.dtype)
            for _jk in range(T.shape[-1]):
                _ht0 = nemo_source_round(
                    _ht0 + nemo_source_round(
                        _e3t0[..., _jk] * _active_3d[..., _jk]))
            # Preserve the pre-split operand precedence: cards that do not
            # name a distinct divisor SSH inherit the independently routed
            # Kmm face SSH, and only then the dispatcher's primary SSH.
            _divisor_eta = (
                redi_divisor_eta
                if redi_divisor_eta is not None
                else redi_flux_eta if redi_flux_eta is not None else eta)
            _divisor_e3t = nemo_qco_live_t_thickness(
                _divisor_eta, _ht0, z_coord, T.dtype, e3t_0=_e3t0)
        _closed_bottom_wmask = (
            True if redi_closed_bottom_wmask_override is None
            else redi_closed_bottom_wmask_override)
        _positions = getattr(cfg, "slope_positions", "mode_b")
        if _positions not in ("mode_b", "nemo_native"):
            raise ValueError(
                "Unknown GMRediConfig.slope_positions scheme: must be one of "
                f"('mode_b', 'nemo_native'), got {_positions!r}")
        if _positions == "nemo_native":
            # ldfslp native four-position slopes: NEMO sign convention and
            # NEMO's own limiters (double cap, ML ramp, Shapiro) built in —
            # NO dispatch negation, exact traldf_iso stencil (amplitude 1.0).
            _nat = compute_nemo_native_slopes(
                rho, T, S, mask, u_mask, v_mask, z_coord, grid, cfg,
                eos_fn, rho_0=rho_0, g=g, active_3d=_active_3d,
                jacobian=jacobian, eta=_native_eta, H_bathy=H_bathy,
                prd_jacobian=_native_prd_J,
                prd_TS_override=native_prd_TS,
                pn2_override=native_slope_pn2,
                e3w_override=native_slope_e3w,
                return_diagnostics=return_redi_slope_diagnostics,
                nmln_override=native_slope_nmln_override,
                eos_nemo_seos=eos_nemo_seos)
            _slope_diagnostics = _nat[4] if return_redi_slope_diagnostics else None; _nat = _nat[:4]; _bolus_nat = None
            if native_bolus_slope_eta is not None:
                _bolus_nat = compute_nemo_native_slopes(
                    rho, T, S, mask, u_mask, v_mask, z_coord, grid, cfg,
                    eos_fn, rho_0=rho_0, g=g, active_3d=_active_3d,
                    jacobian=jacobian, eta=native_bolus_slope_eta,
                    H_bathy=H_bathy, prd_jacobian=_native_prd_J,
                    prd_TS_override=native_prd_TS,
                    pn2_override=native_slope_pn2,
                    e3w_override=native_slope_e3w,
                    eos_nemo_seos=eos_nemo_seos)
            _w_stage = getattr(
                cfg, "redi_w_slope_stage_evaluation", "redi_tuple")
            if _w_stage == "redi_tuple":
                _redi_nat = _nat
            elif _w_stage == "nemo_post_slope_pair":
                if _bolus_nat is None:
                    raise ValueError(
                        "redi_w_slope_stage_evaluation="
                        "'nemo_post_slope_pair' requires "
                        "native_bolus_slope_eta")
                # NEMO ldf_slp completes before ldf_tra consumes its native
                # four-position fields. Carry the post-stage W pair into the
                # vertical Redi tensor while retaining the certified Kmm U/V
                # pair used by the horizontal fluxes.
                _redi_nat = (
                    _nat[0], _nat[1], _bolus_nat[2], _bolus_nat[3])
            else:
                raise ValueError(
                    "Unknown GMRediConfig.redi_w_slope_stage_evaluation: "
                    f"{_w_stage!r}")
            _msc = getattr(cfg, "msc_stabilize", False)
            _skew_eval = cfg.redi_vertical_skew_evaluation
            _a33_eval = cfg.redi_a33_evaluation
            _bolus = None
            if redi_face_thickness_override is not None:
                _flux_e3u, _flux_e3v = redi_face_thickness_override
            _dT = nemo_iso_lap_tracer_tendency_latlon_cgrid(
                T, S_x, S_y, mask, u_mask, v_mask,
                z_coord, jacobian, grid, kappa_Redi_eff, _active_3d,
                native_slopes=_redi_nat, msc_stabilize=_msc, dt=dt,
                bolus_native_slopes=_bolus_nat,
                kappa_GM=kappa_GM, gm_bolus_advection=_gm_bolus,
                gm_bolus_kappa_face_average=_gm_kfa,
                return_bolus=return_bolus_transport,
                kappa_Redi_v=kappa_Redi_v_eff,
                face_thickness_u=_flux_e3u,
                face_thickness_v=_flux_e3v,
                vertical_skew_evaluation=_skew_eval,
                a33_evaluation=_a33_eval,
                return_diagnostics=return_redi_diagnostics,
                return_operand_diagnostics=return_redi_diagnostics,
                divisor_thickness=_divisor_e3t,
                closed_bottom_wmask=_closed_bottom_wmask,
                horizontal_flux_evaluation=(
                    redi_horizontal_flux_evaluation_override
                    if redi_horizontal_flux_evaluation_override is not None
                    else "vectorized"),
                area_reciprocal=redi_area_reciprocal_override,
                area_reciprocal_evaluation=(
                    redi_area_reciprocal_evaluation_override
                    if redi_area_reciprocal_evaluation_override is not None
                    else "vectorized"),
                final_update_evaluation=(
                    redi_final_update_evaluation_override
                    if redi_final_update_evaluation_override is not None
                    else "masked"),
                rhs_accumulator=redi_rhs_accumulator_override)
            if return_bolus_transport:
                dT_dt, _bolus = _dT
            elif return_redi_diagnostics:
                dT_dt, _redi_diagnostics = _dT; _redi_diagnostics.update({"slope": _slope_diagnostics} if return_redi_slope_diagnostics else {})
            else:
                dT_dt = _dT
            dS_dt = nemo_iso_lap_tracer_tendency_latlon_cgrid(
                S, S_x, S_y, mask, u_mask, v_mask,
                z_coord, jacobian, grid, kappa_Redi_eff, _active_3d,
                native_slopes=_redi_nat, msc_stabilize=_msc, dt=dt,
                bolus_native_slopes=_bolus_nat,
                kappa_GM=kappa_GM, gm_bolus_advection=_gm_bolus,
                gm_bolus_kappa_face_average=_gm_kfa,
                kappa_Redi_v=kappa_Redi_v_eff,
                face_thickness_u=_flux_e3u,
                face_thickness_v=_flux_e3v,
                vertical_skew_evaluation=_skew_eval,
                a33_evaluation=_a33_eval,
                divisor_thickness=_divisor_e3t,
                closed_bottom_wmask=_closed_bottom_wmask,
                horizontal_flux_evaluation=(
                    redi_horizontal_flux_evaluation_override
                    if redi_horizontal_flux_evaluation_override is not None
                    else "vectorized"),
                area_reciprocal=redi_area_reciprocal_override,
                area_reciprocal_evaluation=(
                    redi_area_reciprocal_evaluation_override
                    if redi_area_reciprocal_evaluation_override is not None
                    else "vectorized"))
            if return_bolus_transport:
                return dT_dt, dS_dt, _bolus
            if return_redi_diagnostics:
                return dT_dt, dS_dt, _redi_diagnostics
            return dT_dt, dS_dt
        # SLOPE SIGN CONVENTION (2026-07-17 winter ttrd_ldf certificate):
        # the producer computes S = -grad_h(rho)/drho_dz with drho_dz floored
        # NEGATIVE => S = +d(rho)/dx / |drho_dz|. NEMO ldfslp computes
        # slp = zau/(zbu-eps) with zbu bounded NEGATIVE => -dx(rho)/|drho_dz|
        # — the OPPOSITE sign. This operator was certified against NEMO
        # traldf_iso CONSUMING NEMO-convention slopes (0.9997 fed uslp/wslpi),
        # so the producer's slopes must be NEGATED here. Un-negated, the
        # off-diagonal (subduction) fluxes run BACKWARD: on NEMO's Jan-yr5
        # state the 200-430 m band read -1.0e-7 K/s vs NEMO ttrd_ldf +5.2e-8
        # (slope corr vs wslpi_stg: -0.995); negated: +6.8e-8 (mode-b amp).
        # The diagonal K33 term uses S^2 (sign-immune). GM bolus (kappa_GM) uses
        # the SAME negated (NEMO-convention) slopes as the Redi, so its
        # streamfunction sign follows NEMO's ldf_eiv_trp.
        _bolus = None
        _dT = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            T, -S_x, -S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_Redi_eff, _active_3d,
            kappa_GM=kappa_GM, gm_bolus_advection=_gm_bolus,
                gm_bolus_kappa_face_average=_gm_kfa,
            return_bolus=return_bolus_transport,
            kappa_Redi_v=kappa_Redi_v_eff,
            face_thickness_u=_flux_e3u,
            face_thickness_v=_flux_e3v,
            divisor_thickness=_divisor_e3t,
            closed_bottom_wmask=_closed_bottom_wmask,
        )
        if return_bolus_transport:
            dT_dt, _bolus = _dT
        else:
            dT_dt = _dT
        dS_dt = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            S, -S_x, -S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_Redi_eff, _active_3d,
            kappa_GM=kappa_GM, gm_bolus_advection=_gm_bolus,
            gm_bolus_kappa_face_average=_gm_kfa,
            kappa_Redi_v=kappa_Redi_v_eff,
            face_thickness_u=_flux_e3u,
            face_thickness_v=_flux_e3v,
            divisor_thickness=_divisor_e3t,
            closed_bottom_wmask=_closed_bottom_wmask,
        )
        if return_bolus_transport:
            return dT_dt, dS_dt, _bolus
    else:
        raise ValueError(
            f"Unknown GMRediConfig.slope_scheme={scheme!r}; "
            f"expected 'triads', 'centered', or 'nemo_iso_lap'."
        )

    return dT_dt, dS_dt


def compute_isoneutral_K33_latlon(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    *,
    eos: str = "wright",
    eos_linear=None,
    eos_nemo_seos=None,
    mask: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    omega: float = constants.Omega,
    kappa_redi_override: jnp.ndarray | None = None,
    kappa_redi_v_override: jnp.ndarray | None = None,
    density_jacobian: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    native_prd_jacobian: jnp.ndarray | None = None,
    native_prd_TS: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    native_slope_pn2: jnp.ndarray | None = None,
    native_slope_e3w: jnp.ndarray | None = None,
    native_slope_eta: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    dt: float | None = None,
    eos_depth: str = "insitu",
) -> jnp.ndarray:
    """Vertical isoneutral diffusivity K_33 at w-faces, for the implicit solve.

    ``K_33 = 0.25 · kappa_Redi · Σ_8 taper_W · S_W²``  →  (n_lat, n_lon, nlev-1), ≥ 0.

    This is EXACTLY the coefficient of dq/dz in the explicit triad ``F_z`` diagonal
    term that ``gm_redi_tracer_tendency_triads_latlon_cgrid`` drops when
    ``implicit_K33=True`` (same rho via the 2-iteration EOS coupling, same shared
    ``_w_triad_slopes_tapers``, same ``kappa_Redi`` w-face placement via
    ``_kappa_center_uvw`` — so the explicit-drop and the implicit-add are
    consistent; ``kappa_Redi`` may be a scalar, a 2-D per-column array, or a 3-D
    interface field, just as in the triad ``F_z``).  The lat-lon C-grid model
    step adds it to the implicit vertical-diffusion ``K_v`` (Veros
    core/isoneutral/diffusion.py:154, ``delta = dt/dzw · K_33``), applying the
    vertical isoneutral diffusion backward-Euler-implicitly as in Veros rather than
    explicitly.  Density-independent of the tracer, so one call serves both T and S.
    """
    validate_redi_coefficient(cfg)
    if cfg.redi_coefficient == "nemo21" and (
            kappa_redi_override is not None or kappa_redi_v_override is not None):
        raise ValueError("nemo21 conflicts with runtime coefficient overrides")
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    # Jacobian + density (2-iteration EOS coupling).  When the model step
    # already computed these for the GM/Redi tracer tendency from the SAME
    # (T,S,eta,H_bathy) (implicit_K33 path), reuse them via density_jacobian
    # so the expensive 3-D EOS coupling is not run twice; None => compute
    # inline, bit-identical (scaling review lever #3).
    _eos_mk_kw = {"rho0": rho_0} if eos_depth == "geometric" else {}
    eos_fn = make_eos_fn(
        eos, eos_linear, eos_nemo_seos=eos_nemo_seos, **_eos_mk_kw)
    if density_jacobian is None:
        rho, jacobian = gm_redi_density_and_jacobian(
            T, S, eta, H_bathy, grid, z_coord,
            eos=eos, eos_linear=eos_linear, eos_nemo_seos=eos_nemo_seos,
            mask=mask, rho_0=rho_0, g=g,
            eos_depth=eos_depth,
        )
    else:
        rho, jacobian = density_jacobian
    if getattr(cfg, "slope_positions", "mode_b") == "nemo_native":
        # Native ldfslp slopes: K33 = NEMO's ah_wslp2 (traldf_iso_a33,
        # ln_traldf_msc=F => akz = ah_wslp2, the FULL implicit a33):
        #
        #   zmsku    = wmask / MAX(umask(i,k-1)+umask(i-1,k)
        #                          +umask(i-1,k-1)+umask(i,k), 1)
        #   zahu_w   = (ahtu(i,k-1)+ahtu(i-1,k)+ahtu(i-1,k-1)+ahtu(i,k))*zmsku
        #   ah_wslp2 = zahu_w*wslpi^2 + zahv_w*wslpj^2        (w-points 2:jpkm1)
        #
        # — the mask-NORMALIZED 4-point ahtu/ahtv average onto the w-point
        # (NOT the centre kappa), with the 3-D (staircase-aware) umask/vmask,
        # and the SAME wslpi/wslpj arrays the explicit operator's A31/A32
        # off-diagonal fluxes difference.  Using the SAME slope fields on both
        # sides of the explicit/implicit split is what keeps the rotated
        # tensor PSD: a K33 built from a DIFFERENT slope discretization
        # under-covers the dropped diagonal wherever its |S| is smaller and
        # the net vertical diffusivity goes NEGATIVE — a kappa-scaled local
        # tracer runaway (#1226; subcritical at kappa=200, runaway at
        # NEMO-strength 1501*cos(phi)).
        #
        # The slopes+masks here are built from bit-identical inputs to the
        # tendency dispatcher's nemo_native branch (same rho via
        # density_jacobian, same active_3d construction from H_bathy, same
        # u_mask/v_mask when threaded by the model step), so the two
        # compute_nemo_native_slopes calls return bit-identical arrays.
        from legoesm.ocean.eos import make_eos_fn as _mk
        _eosfn = _mk(eos, eos_linear, **_eos_mk_kw)
        if density_jacobian is not None:
            _rho, _J = density_jacobian
        else:
            _rho, _J = gm_redi_density_and_jacobian(
                T, S, eta, H_bathy, grid, z_coord,
                eos=eos, eos_linear=eos_linear, mask=mask, rho_0=rho_0, g=g,
                eos_depth=eos_depth)
        _m = mask if mask is not None else jnp.ones(T.shape[:2], T.dtype)
        # 3-D wet mask (NEMO tmask) — SAME construction as the tendency
        # dispatcher's nemo_iso_lap branch (_nemo_native_active_3d, #1226).
        _act = _nemo_native_active_3d(_m, z_coord, H_bathy, T.dtype)
        # 2-D wall masks: threaded from the model step (staircase walls);
        # None => interior-open walls derived from the cell mask (the flat
        # GYRE oracle behaviour, unchanged).
        if u_mask is None:
            _um = jnp.zeros((T.shape[0], T.shape[1] + 1), T.dtype)
            _um = _um.at[:, 1:-1].set(_m[:, :-1] * _m[:, 1:])
        else:
            _um = u_mask
        if v_mask is None:
            _vm = jnp.zeros((T.shape[0] + 1, T.shape[1]), T.dtype)
            _vm = _vm.at[1:-1, :].set(_m[:-1, :] * _m[1:, :])
        else:
            _vm = v_mask
        _native_prd_J = _J if native_prd_jacobian is None else native_prd_jacobian
        _native_eta = eta if native_slope_eta is None else native_slope_eta
        _, _, _wi, _wj = compute_nemo_native_slopes(
            _rho, T, S, _m, _um, _vm, z_coord, grid, cfg, _eosfn,
            jacobian=_J, eta=_native_eta, H_bathy=H_bathy,
            prd_jacobian=_native_prd_J,
            prd_TS_override=native_prd_TS,
            pn2_override=native_slope_pn2,
            e3w_override=native_slope_e3w,
            rho_0=rho_0, g=g, active_3d=_act,
            eos_nemo_seos=eos_nemo_seos)
        if cfg.redi_coefficient == "nemo21":
            _kgm = native_treguier_kappa_for_state(
                _rho, T, S, _m, _um, _vm, z_coord, grid, cfg, _eosfn,
                _J, eta, H_bathy, _native_prd_J,
                native_prd_TS, native_slope_pn2, native_slope_e3w,
                jnp.broadcast_to(grid.f, _m.shape), rho_0, g, omega)
            kappa_redi_override, kappa_redi_v_override = nemo21_redi_from_gm(
                _kgm, jnp.broadcast_to(grid.f, _m.shape), cfg.redi_f_f,
                _um[:, 1:], _vm[1:, :], cfg.redi_aht0, omega, grid=grid)
        _kap = cfg.kappa_Redi if kappa_redi_override is None else kappa_redi_override
        # Center kappa broadcast IDENTICAL to the explicit operator's own
        # ``aht`` block, then the SHARED mask + a33 kappa-sum helpers — one
        # construction on both sides of the explicit/implicit split, so the
        # coefficients cannot diverge at walls or with nonuniform kappa.
        if isinstance(_kap, jnp.ndarray) and _kap.ndim == 3:
            _aht = jnp.broadcast_to(_kap, T.shape)
        elif isinstance(_kap, jnp.ndarray) and _kap.ndim == 2:
            _aht = jnp.broadcast_to(_kap[:, :, jnp.newaxis], T.shape)
        else:
            _aht = jnp.broadcast_to(jnp.asarray(_kap, T.dtype), T.shape)
        # NEMO's ahtv (nn_aht_ijk_t=20) is independently-evaluated at the
        # v-point, not ahtu broadcast onto the v-face; kappa_redi_v_override
        # supplies that distinct value (static lat-scaling case only) so the
        # implicit a33 stays consistent with the explicit operator's aht_v.
        if kappa_redi_v_override is None:
            _aht_v = _aht
        elif isinstance(kappa_redi_v_override, jnp.ndarray) and kappa_redi_v_override.ndim == 3:
            _aht_v = jnp.broadcast_to(kappa_redi_v_override, T.shape)
        elif isinstance(kappa_redi_v_override, jnp.ndarray) and kappa_redi_v_override.ndim == 2:
            _aht_v = jnp.broadcast_to(kappa_redi_v_override[:, :, jnp.newaxis], T.shape)
        else:
            _aht_v = jnp.broadcast_to(jnp.asarray(kappa_redi_v_override, T.dtype), T.shape)
        _um3, _vm3, _wm3 = nemo_iso_face_masks(_um, _vm, _act)
        # Shared a33 (#1226): the SAME nemo_iso_a33 the explicit operator's
        # MSC block consumes.  msc=False (ln_traldf_msc=F): akz = ah_wslp2,
        # the full diagonal implicit.  msc=True (ln_traldf_msc=T — the DINO
        # namelist): akz is the CAPPED implicit part; the explicit operator
        # carries the (ah_wslp2 - akz) remainder, so the implicit solve must
        # receive akz — returning full ah_wslp2 here would double-count the
        # remainder.
        from legoesm.grids.latlon import ensure_geometry as _eg
        _geom = _eg(grid)
        _e1u_c = _geom.dx_u[:, 1:]
        _e2v_c = _geom.dy_v[1:, :]
        # z*-scaled thickness with the SAME jacobian as the operator's e3t
        # (from the shared density_jacobian thread).
        _e3t = z_coord.dz_ref[None, None, :] * _J[:, :, jnp.newaxis]
        _msc = bool(getattr(cfg, "msc_stabilize", False))
        # Same e3w object as the explicit A33 flux (traldf_iso.f90:831-833):
        # the two sides of the split share one resolver by construction.
        # Resolved ONLY when ln_traldf_msc is on, matching the explicit side's
        # own guard: with msc=F, akz = ah_wslp2 and traldf_iso.f90:88-91 never
        # reads e3w at all, so a coordinate the resolver would REFUSE must not
        # be refused for a value nothing consumes.
        _e3w = (nemo_iso_a33_e3w(z_coord, _e3t, _J, T.dtype) if _msc
                else _e3t)
        _, _akz = nemo_iso_a33(
            _aht, _um3, _vm3, _wm3, _wi, _wj,
            _e1u_c, _e2v_c, _e3w ** 2, dt=dt, msc=_msc, aht_v=_aht_v,
            evaluation=cfg.redi_a33_evaluation)
        return _akz[:, :, 1:]                              # interfaces 0..nlev-2
    kappa_Redi = cfg.kappa_Redi if kappa_redi_override is None else kappa_redi_override
    nlev = T.shape[-1]
    # K_33 is evaluated at the nlev-1 w-faces, so kappa_Redi is needed there.
    # The shared dispatch (``_kappa_center_uvw``) puts a 3-D interface kappa on
    # the w-faces DIRECTLY (== the 4th return) and a scalar / 2-D per-column
    # kappa as its cell-centred broadcast (bit-identical to the prior inline
    # 2-D handling).  Take only the w-face form here.
    kappa_Redi_w = _kappa_center_uvw(kappa_Redi, nlev)[3]
    n_lat, n_lon = mask.shape
    rho_filled = neumann_fill_cgrid(rho, mask)
    slope_density = getattr(cfg, "slope_density", "in_situ")
    w_inputs = _w_face_slope_density_inputs(
        rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
        eos_fn, rho_0, g,
    )
    _ufa, _vfa = _per_level_face_acts(z_coord, grid, T.dtype)
    (S_Wx1, S_Wx2, S_Wx3, S_Wx4, S_Wy1, S_Wy2, S_Wy3, S_Wy4,
     tWx1, tWx2, tWx3, tWx4, tWy1, tWy2, tWy3, tWy4) = _w_triad_slopes_tapers(
        n_lat=n_lat, n_lon=n_lon, S_max=cfg.S_max,
        taper_width_frac=cfg.taper_width_frac,
        u_face_act=_ufa, v_face_act=_vfa,
        adjoint_stabilization=getattr(cfg, "adjoint_stabilization", "none"),
        slope_limit=getattr(cfg, "slope_limit", "dm95_taper"),
        **w_inputs)
    K_33 = 0.25 * kappa_Redi_w * (
        tWx1 * S_Wx1 ** 2 + tWx2 * S_Wx2 ** 2 + tWx3 * S_Wx3 ** 2 + tWx4 * S_Wx4 ** 2
        + tWy1 * S_Wy1 ** 2 + tWy2 * S_Wy2 ** 2 + tWy3 * S_Wy3 ** 2 + tWy4 * S_Wy4 ** 2)
    return K_33 * mask[:, :, jnp.newaxis]


# =====================================================================
# EKE SOURCE augmentation (Veros K_diss_h + realized -P_diss_skew)
# =====================================================================

def harmonic_lateral_kediss_eke_source(
    visc_u: jnp.ndarray,
    visc_v: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
    mask: jnp.ndarray,
    kdiss_h_cell: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Mean-KE removal by the harmonic lateral viscosity, routed to the EKE source.

    Returns the rate at which the harmonic lateral viscosity removes mean kinetic
    energy, as a NON-NEGATIVE EKE source [m²/s³] at the ``nlev-1`` interior
    interfaces (the W-grid) — legoESM's analogue of Veros's ``K_diss_h``
    (``veros/core/friction.py`` ``harmonic_friction`` → ``calc_diss_u``/``calc_diss_v``,
    fed into the EKE forcing at ``veros/core/eke.py:110``).

    TWO config-selectable discretisations (``EKEConfig.kdiss_h_flux_form``):

    **Flux form (FAITHFUL, positive-definite; ``kdiss_h_flux_form=True``, ACC recipe).**
    ``kdiss_h_cell`` is the pre-computed cell-centre positive-definite dissipation
    density [m²/s³], MATCHED to the selected lateral-viscosity operator (each is the
    exact KE-removal of its own operator, so it pairs with whichever viscosity the
    config selects):
      * VECTOR Laplacian → ``A_h·(div² + <ζ²>)`` — the Helmholtz KE-removal
        ``-∫u·∇²_vec u = ∫(div²+|ζ|²) ≥ 0`` — from
        :func:`...latlon_cgrid_operators.vector_laplacian_dissipation_cgrid`.
      * FLUX-DIVERGENCE → the componentwise ``A_h·|∇u|² = 0.5·Σ(Δu·flux)``
        (Veros ``calc_diss_u/v``) from
        :func:`...latlon_cgrid_operators.flux_divergence_viscosity_cgrid`.
    This branch only averages that density to the ``nlev-1`` interior interfaces (the
    SAME W-grid mapping Veros's ``dissipation_on_wgrid`` uses for the interior).  It is
    ``≥ 0`` EVERYWHERE by construction so NO clamp is applied, and (probe-verified on
    the spun-up ACC state) its domain integral equals the mean KE actually removed by
    ``A_h`` to ~0.3% and matches Veros's captured ``K_diss_h`` to ~7%.  (The
    flux-divergence operator IS Veros's componentwise form exactly; the vector-Laplacian
    form replaces the squared one-sided gradients with squared div/curl because that
    operator carries the metric/curvature coupling the componentwise form omits.)

    **Dynamical form (DEFAULT, ``kdiss_h_flux_form=False``; ``kdiss_h_cell=None``).**
    Uses the KE-tendency ``-u·(A_h∇²_vec u) - v·(A_h∇²_vec v)`` from the supplied
    ``visc_u``/``visc_v`` momentum tendencies, averaged faces→centres→interfaces and
    CLAMPED ``≥ 0``.  This form is NOT positive-definite (it carries the transport
    divergence ``-∇·(A_h u∇u)`` ⇒ ~35% of wet cells negative); the per-cell clamp
    truncates those, OVER-CREDITING the domain-integrated KE dissipation by ~11–20%
    on the ACC channel (probe-measured; see
    .physics-validator/eke_source/probe_kdiss_conservation.py).  Retained as the
    default so existing runs stay BIT-IDENTICAL; the ACC recipe opts into the flux
    form for the exact, clamp-free Veros analogue.

    Either way the energy added to EKE is the mean KE removed by ``A_h``: the sink
    already happens in the momentum tendency (``du_dt`` carries ``A_h∇²_vec u``), and
    this term credits that lost KE to the eddy field (Veros's mean→eddy pathway).

    Parameters
    ----------
    visc_u : (n_lat, n_lon+1, nlev) — harmonic-viscosity u-tendency ``A_h∇²u`` [m/s²]
        (used only by the dynamical-form / ``kdiss_h_cell=None`` path).
    visc_v : (n_lat+1, n_lon, nlev) — harmonic-viscosity v-tendency ``A_h∇²v`` [m/s²]
        (dynamical-form path only).
    u : (n_lat, n_lon+1, nlev) — zonal velocity at u-faces [m/s] (dynamical-form path).
    v : (n_lat+1, n_lon, nlev) — meridional velocity at v-faces [m/s] (dynamical path).
    grid : LatLonGrid (unused metric-wise — the C-grid averaging is index-based; kept
        for signature parity with the other EKE source builders).
    mask : (n_lat, n_lon) — ocean mask (1 = ocean).
    kdiss_h_cell : (n_lat, n_lon, nlev) — OPTIONAL pre-computed positive-definite
        cell-centre dissipation density [m²/s³], matched to the lateral-viscosity
        operator (``A_h(div²+<ζ²>)`` vector Laplacian / ``A_h|∇u|²`` flux-divergence).
        When provided (the flux-form path), it is mapped to the W-grid with NO clamp;
        ``visc_*``, ``u``, ``v`` are then unused.  ``None`` (default) = dynamical form.

    Returns
    -------
    K_diss_h : (n_lat, n_lon, nlev-1) — non-negative EKE source [m²/s³] at interfaces.
    """
    if kdiss_h_cell is not None:
        # FAITHFUL FLUX FORM: kdiss_h_cell is already the positive-definite per-cell
        # dissipation density ≥ 0 matched to the operator (A_h·(div²+<ζ²>) from
        # vector_laplacian_dissipation_cgrid, or A_h·|∇u|² from
        # flux_divergence_viscosity_cgrid) — built from the SAME velocities / A_h
        # scaling the applied viscous tendency uses.  Average the full-level (nlev) cell
        # density to the nlev-1 interior interfaces — the same interior W-grid
        # mapping Veros's dissipation_on_wgrid uses (0.5·(c[:-1]+c[1:])).  No clamp:
        # the density is ≥ 0 everywhere by construction, so the W-grid average is too.
        diss_cell = kdiss_h_cell * mask[:, :, jnp.newaxis]
        K_diss_h_w = 0.5 * (diss_cell[:, :, :-1] + diss_cell[:, :, 1:])
        return K_diss_h_w * mask[:, :, jnp.newaxis]

    # DYNAMICAL FORM (default): per-face KE-dissipation rate -u·(A_h∇²u) [m²/s³ on
    # the face], averaged faces→centres→interfaces and clamped ≥ 0.  NB: the clamp
    # is NOT inactive — -u·A_h∇²u is locally negative in transport regions, so the
    # clamp over-credits the column-integrated KE dissipation by ~11-20% vs the
    # positive-definite flux form above (set kdiss_h_flux_form=True for the exact,
    # clamp-free Veros analogue).
    return _kediss_from_momentum_tendency(visc_u, visc_v, u, v, mask, clamp=True)


def _kediss_from_momentum_tendency(
    tend_u: jnp.ndarray,
    tend_v: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray,
    *,
    clamp: bool = True,
) -> jnp.ndarray:
    """KE removal ``-u·tend_u - v·tend_v`` mapped faces→centres→W-grid [m²/s³].

    Shared C-grid mapping for the mean-KE that a face-grid momentum tendency
    (``A_h∇²u`` for K_diss_h, ``-r·u`` for K_diss_bot) extracts from the flow,
    routed to the ``nlev-1`` interior interfaces (the W-grid where TKE/EKE live).
    The per-face dissipation rate ``-u·tend_u`` is averaged to cell centres
    (inverse of the cell→face interpolation) then to the interior interfaces
    (``0.5·(c[:-1]+c[1:])`` — Veros's ``dissipation_on_wgrid`` interior mapping).
    With ``clamp=True`` the result is floored ≥ 0 to make it a pure source (the
    dynamical form is not positive-definite); the caller passes the already
    face-masked tendencies.
    """
    p_u = -u * tend_u                                   # (n_lat, n_lon+1, nlev)
    p_v = -v * tend_v                                   # (n_lat+1, n_lon, nlev)
    diss_cell = 0.5 * (p_u[:, :-1, :] + p_u[:, 1:, :])  # (n_lat, n_lon, nlev)
    diss_cell = diss_cell + 0.5 * (p_v[:-1, :, :] + p_v[1:, :, :])
    diss_cell = diss_cell * mask[:, :, jnp.newaxis]
    K_diss_w = 0.5 * (diss_cell[:, :, :-1] + diss_cell[:, :, 1:])
    if clamp:
        K_diss_w = jnp.maximum(K_diss_w, 0.0)
    return K_diss_w * mask[:, :, jnp.newaxis]


def bottom_drag_kediss_tke_source(
    drag_u: jnp.ndarray,
    drag_v: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Bottom-drag KE extraction → prognostic-TKE source ``K_diss_bot`` [m²/s³].

    legoESM's analogue of Veros ``K_diss_bot`` (``veros/core/friction.py``
    ``linear_bottom_friction``: ``diss = r_bot·u²`` at the bottom level, mapped
    via ``calc_diss_u/v`` to the T-grid). Here the bottom-drag MOMENTUM tendency
    ``drag_u = -r_eff·u/dz_bot`` (already face-masked, applied at the seafloor /
    BBL band by ``_bc_bottom_drag``) gives the per-face KE-removal rate
    ``-u·drag_u = r_eff·u²/dz_bot ≥ 0`` (drag always opposes the flow), which is
    averaged faces→centres→interior interfaces (the W-grid where the prognostic
    TKE lives). No clamp is needed — the term is ≥ 0 by construction (drag·flow),
    but the shared mapping applies a ≥ 0 floor harmlessly for safety.

    Returns ``(n_lat, n_lon, nlev-1)`` ≥ 0 at the interior interfaces.
    """
    return _kediss_from_momentum_tendency(drag_u, drag_v, u, v, mask, clamp=True)


def compute_realized_gm_skew_conversion(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    kappa_gm_w: jnp.ndarray,
    *,
    eos: str = "wright",
    eos_linear=None,
    mask: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
) -> jnp.ndarray:
    """Realized GM-skew buoyancy conversion ``-P_diss_skew`` as an EKE source [m²/s³].

    The Gent-McWilliams skew (bolus) flux releases mean available potential energy
    into eddy energy at the rate ``-P_diss_skew = -(g/ρ₀)·∇₃ρ·F_skew`` (Veros
    ``veros/core/isoneutral/diffusion.py:234-281``), where ``F_skew = κ_GM·S·∂ρ/∂z``
    is the per-triad GM skew flux of density.  Using ``S = -∇_h ρ / ∂_z ρ`` per
    triad and contracting with ``∇₃ρ`` gives the closed W-grid form

        -P_diss_skew(z) = κ_GM(z) · (g/ρ₀)·|∂ρ/∂z|_w · ( 0.25·Σ_8 taper·S² )
                        = κ_GM(z) · N²_w · <S²>_triad,                        (≥ 0)

    i.e. the GM coefficient times the true local buoyancy frequency ``N²_w =
    (g/ρ₀)|∂ρ/∂z|`` times the per-triad slope-variance ``<S²>_triad = 0.25·Σ taper·S²``
    (which is exactly ``K_33/κ_Redi`` from :func:`compute_isoneutral_K33_latlon`).

    This is the *realized* conversion — it differs from the *parameterized* EKE
    production ``κ_GM·σ²`` with ``σ = <N|S|>`` (a squared slope AVERAGE) because the
    per-triad slope VARIANCE ``<S²> ≥ <S>²`` (Jensen): the realized form keeps the
    discrete slope variance the GM tracer flux actually transports, which the
    pre-averaged Visbeck σ under-counts.  It reuses the SHARED per-triad W-face
    slopes/tapers (``_w_triad_slopes_tapers``) — the SAME slopes the GM/Redi skew
    flux and the implicit ``K_33`` use — so no slope numerics are duplicated.

    Sign / positivity: ``N²_w ≥ 0`` (stable-strat floor ``|∂ρ/∂z| ≥ _EPS_DIV``),
    ``S² ≥ 0``, ``taper ≥ 0``, ``κ_GM ≥ 0`` ⇒ the source is ≥ 0 everywhere (slumping
    isopycnals release mean APE into EKE). The energy comes from the mean APE that
    the GM skew flux flattens — the same flux already applied to the tracer
    tendency; documenting the routing: this credits that released APE to EKE.

    Parameters
    ----------
    T, S, eta, H_bathy, grid, z_coord, cfg, eos/eos_linear/mask/rho_0/g : as in
        :func:`compute_isoneutral_K33_latlon` (same rho via the 2-iteration EOS
        coupling, same shared slopes).
    kappa_gm_w : (n_lat, n_lon, nlev-1) — the GM coefficient ``κ_GM(z)`` at the
        interior W-faces (the prognostic-EKE 3-D override; the SAME kappa fed to the
        GM tracer flux), so the released APE is consistent with the applied skew flux.

    Returns
    -------
    P_skew : (n_lat, n_lon, nlev-1) — non-negative realized GM-skew EKE source [m²/s³].
    """
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: neumann_fill_cgrid(field, mask)
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2,
    )
    n_lat, n_lon = mask.shape
    rho_filled = neumann_fill_cgrid(rho, mask)
    slope_density = getattr(cfg, "slope_density", "in_situ")
    w_inputs = _w_face_slope_density_inputs(
        rho_filled, T, S, mask, z_coord, jacobian, grid, slope_density,
        eos_fn, rho_0, g,
    )
    drho_dz_w = w_inputs["drho_dz_w"]      # (n_lat, n_lon, nlev-1)
    _ufa, _vfa = _per_level_face_acts(z_coord, grid, T.dtype)
    (S_Wx1, S_Wx2, S_Wx3, S_Wx4, S_Wy1, S_Wy2, S_Wy3, S_Wy4,
     tWx1, tWx2, tWx3, tWx4, tWy1, tWy2, tWy3, tWy4) = _w_triad_slopes_tapers(
        n_lat=n_lat, n_lon=n_lon, S_max=cfg.S_max,
        taper_width_frac=cfg.taper_width_frac,
        u_face_act=_ufa, v_face_act=_vfa,
        adjoint_stabilization=getattr(cfg, "adjoint_stabilization", "none"),
        slope_limit=getattr(cfg, "slope_limit", "dm95_taper"),
        **w_inputs)
    # Per-triad slope variance <S²>_triad = 0.25·Σ taper·S²  (= K_33/κ_Redi).
    S2_triad = 0.25 * (
        tWx1 * S_Wx1 ** 2 + tWx2 * S_Wx2 ** 2 + tWx3 * S_Wx3 ** 2 + tWx4 * S_Wx4 ** 2
        + tWy1 * S_Wy1 ** 2 + tWy2 * S_Wy2 ** 2 + tWy3 * S_Wy3 ** 2 + tWy4 * S_Wy4 ** 2)
    # True local buoyancy frequency at the W-faces: N²_w = (g/ρ₀)|∂ρ/∂z| ≥ 0.
    # In the neutral mode this is the locally-referenced N² (compressibility
    # bias removed), consistent with the neutral slope variance above.
    N2_w = (g / rho_0) * jnp.abs(drho_dz_w)
    P_skew = jnp.maximum(kappa_gm_w, 0.0) * N2_w * S2_triad
    return jnp.maximum(P_skew, 0.0) * mask[:, :, jnp.newaxis]


def _dynamic_enthalpy_dissipation_wgrid(
    int_drhodX,
    F_x_u,
    F_y_v,
    F_z,
    grid,
    dz_cell,
    mask,
    g,
    rho_0,
    *,
    K_33=None,
    dq_dz_w=None,
):
    r"""Veros dynamic-enthalpy dissipation of one tracer's isopycnal flux, on the
    W-grid [m²/s³].  This is ``P_diss_skew`` / ``P_diss_iso`` for ONE tracer
    (the caller sums the T and S contributions).

    Reproduces Veros's two-part construction
    (``veros/core/isoneutral/diffusion.py:234-279`` +
    ``veros/core/diffusion.py:compute_dissipation`` / ``dissipation_on_wgrid``):

    HORIZONTAL (cell-centred, then averaged to the interior W-faces):

        diss_h = 0.5·g/ρ₀·( ∇_x(int_drhodX)·F_x  +  ∇_y(int_drhodX)·F_y ),

    the contraction of the dynamic-enthalpy horizontal gradient with the isopycnal
    horizontal flux of the tracer, averaged over the two faces straddling each
    cell (Veros's ``flux_east[C]`` + ``flux_east[W]`` stencil), then mapped to the
    interior W-faces by ``0.5·(c[k]+c[k+1])`` (Veros ``dissipation_on_wgrid``).

    VERTICAL (already at the W-faces):

        diss_v = +g/ρ₀·∂_z(int_drhodX)·( F_z [+ K_33·∂_z q] ),

    where ``∂_z(int_drhodX) = (int[k]-int[k+1])/dz_w`` (Veros ``fxa``) and ``F_z``
    is the explicit vertical isopycnal flux (the off-diagonal K31/K32 skew/iso
    term).  When the K_33 vertical isoneutral diagonal is treated IMPLICITLY
    (``implicit_K33=True``, the ACC recipe), ``F_z`` carries ONLY the off-diagonal,
    so the diagonal dissipation ``K_33·∂_z q`` is supplied separately via
    ``K_33`` + ``dq_dz_w`` (Veros adds it explicitly to ``P_diss_iso``,
    diffusion.py:274-278); pass ``K_33=None`` to omit it (the skew has no K_33).

    SIGN: Veros uses ``-g/ρ₀`` for the vertical term because its ``flux_top``
    convention is OPPOSITE legoESM's ``F_z`` (z-upward flux); the single global
    sign flip is absorbed by the ``+g/ρ₀`` here, validated against Veros's captured
    ``P_diss_skew`` field (per-cell sign structure identical, spatial corr 0.95,
    integral within 1 % on the ACC equilibrium).  The horizontal term keeps the
    same sign as Veros (``F_x_u``/``F_y_v`` share Veros's ``flux_east``/``flux_north``
    sign convention — both store ``+K·∇q``).

    Returns ``P_diss_X`` (n_lat, n_lon, nlev-1) [m²/s³]; the EKE SOURCE is
    ``-P_diss_X``.  ``int_drhodX``, ``F_*`` are this tracer's fields.
    """
    n_lat, n_lon, nlev = int_drhodX.shape
    # ---- HORIZONTAL: 0.5·g/ρ₀·(∇int·F) averaged to cell centres ----
    dX_dx_u = gradient_x_cgrid(int_drhodX, grid)        # (n_lat, n_lon+1, nlev)
    dX_dy_v = gradient_y_cgrid(int_drhodX, grid)        # (n_lat+1, n_lon, nlev)
    fx = dX_dx_u * F_x_u
    fy = dX_dy_v * F_y_v
    diss_h_cell = 0.5 * (g / rho_0) * (
        (fx[:, :-1, :] + fx[:, 1:, :]) + (fy[:-1, :, :] + fy[1:, :, :])
    ) * mask[:, :, jnp.newaxis]                          # (n_lat, n_lon, nlev)
    # interior W-grid mapping (Veros dissipation_on_wgrid interior 0.5·(c[:-1]+c[1:])).
    P_h_w = 0.5 * (diss_h_cell[:, :, :-1] + diss_h_cell[:, :, 1:])
    # ---- VERTICAL: +g/ρ₀·fxa·(F_z [+ K_33·dq/dz]) ----
    dz_w = 0.5 * (dz_cell[..., :-1] + dz_cell[..., 1:])  # (n_lat, n_lon, nlev-1)
    fxa = (int_drhodX[:, :, :-1] - int_drhodX[:, :, 1:]) / jnp.maximum(dz_w, _EPS_DIV)
    Fz_full = F_z
    if K_33 is not None and dq_dz_w is not None:
        Fz_full = Fz_full + K_33 * dq_dz_w
    P_v_w = (g / rho_0) * fxa * Fz_full
    return (P_h_w + P_v_w) * mask[:, :, jnp.newaxis]


def compute_realized_signed_conversions(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    kappa_gm_w: jnp.ndarray,
    *,
    want_skew: bool = True,
    want_iso: bool = False,
    kappa_redi_w: jnp.ndarray | None = None,
    eos: str = "wright",
    eos_linear=None,
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    dt: float | None = None,
):
    r"""Realized SIGNED GM-skew (``-P_diss_skew``) and Redi (``-P_diss_iso``) EKE
    sources [m²/s³] at the interior W-faces — the literal Veros energy conversions.

    Unlike :func:`compute_realized_gm_skew_conversion` (the positive-definite
    parameterized ``κ_GM·N²·⟨S²⟩``), these are the SIGNED dynamic-enthalpy
    dissipations Veros computes (``veros/core/isoneutral/diffusion.py``):

    - ``-P_diss_skew`` (``want_skew``) — the energy the GM SKEW (bolus) flux
      EXTRACTS from the mean APE and gives to the eddies (the GM brake).  Built
      from the SKEW-only isopycnal flux (``kappa_Redi=0``, ``kappa_GM=kappa_gm_w``)
      contracted with the dynamic-enthalpy gradient.  ``≥ 0`` wherever isopycnals
      slump (the ACC equilibrium has it ``> 0`` in every wet cell, matching Veros),
      but it is NOT clamped — the caller (``gm_source_mode="realized_signed"``)
      folds any locally-negative value semi-implicitly (it is then a local sink,
      like dissipation), keeping ``E ≥ e_min`` by construction.

    - ``-P_diss_iso`` (``want_iso``) — the energy the Redi (isopycnal-diffusive)
      flux DISSIPATES out of the mean APE (a SINK of EKE in Veros's budget, mostly
      negative).  Built from the ISO-only flux (``kappa_GM=0``,
      ``kappa_Redi=kappa_redi_w`` — the ``K_iso=K_gm`` coupling supplies
      ``kappa_redi_w = kappa_gm_w``) PLUS the implicit K_33 vertical diagonal
      dissipation (when ``cfg.implicit_K33``), exactly as Veros adds it explicitly
      to ``P_diss_iso``.  Subtracted from the EKE source by the caller
      (``source_p_diss_iso=True``).

    Both reuse the EXISTING GM/Redi flux assembly
    (:func:`gm_redi_tracer_tendency_triads_latlon_cgrid` with the appropriate
    kappa zeroed and ``return_fluxes=True``) — NO flux numerics are duplicated.
    The dynamic-enthalpy contraction is the shared
    :func:`_dynamic_enthalpy_dissipation_wgrid`, summed over the T and S fluxes.

    Returns ``(neg_P_diss_skew, neg_P_diss_iso)``; each is ``None`` when its
    ``want_*`` flag is False.  Shapes ``(n_lat, n_lon, nlev-1)``.
    """
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    if u_mask is None:
        u_mask = jnp.ones((T.shape[0], T.shape[1] + 1), dtype=T.dtype)
    if v_mask is None:
        v_mask = jnp.ones((T.shape[0] + 1, T.shape[1]), dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: neumann_fill_cgrid(field, mask)
    rho, _rp, _pp = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2,
    )
    rho_filled = neumann_fill_cgrid(rho, mask)
    slope_density = getattr(cfg, "slope_density", "in_situ")
    dz_cell = z_coord.dz_ref * jacobian[:, :, jnp.newaxis]

    # Dynamic-enthalpy integrands int_drhodT/S (Veros int_drhodT/S) at cell centres.
    z_full = jnp.asarray(z_coord.z_full_ref)[jnp.newaxis, jnp.newaxis, :]
    # Dry-cell column fill: int_drhodTS differentiates the EOS via jax.grad —
    # NaN at the sub-seafloor S=0 cells under the gsw EOS (see
    # _fill_dry_cells_columnwise); wet-cell values are unchanged.
    int_drhodT, int_drhodS = int_drhodTS_dynamic_enthalpy(
        eos_fn,
        _fill_dry_cells_columnwise(neumann_fill_cgrid(T, mask), z_coord),
        jnp.maximum(
            _fill_dry_cells_columnwise(neumann_fill_cgrid(S, mask), z_coord),
            _S_GRAD_FLOOR),
        z_full, rho_0, g,
    )
    int_drhodT = int_drhodT * mask[:, :, jnp.newaxis]
    int_drhodS = int_drhodS * mask[:, :, jnp.newaxis]

    # Zeroing one kappa selects the skew-only (kappa_Redi=0) or iso-only
    # (kappa_GM=0) flux.  A scalar 0.0 passes through _kappa_center_uvw unchanged
    # (its skew/diagonal contributions vanish), so the other (3-D interface)
    # kappa keeps its faithful w-face placement.
    zero_kappa = jnp.asarray(0.0, dtype=kappa_gm_w.dtype) \
        if isinstance(kappa_gm_w, jnp.ndarray) else 0.0

    def _fluxes(q, kappa_GM_arg, kappa_Redi_arg, K_iso_steep=None,
                double_diag_kappa=None):
        # The K_iso_steep floor ``dq·max(0, K_iso_steep − kappa_Redi·taper)`` is
        # NONLINEAR in kappa_Redi: with kappa_Redi=0 it degenerates to a full
        # K_iso_steep horizontal diffusion. It is a REDI-diagonal term, never a
        # GM term, so the skew-only call must zero it or the decomposition
        # F(kR,kG)=F(kR,0)+F(0,kG) breaks (~25% skew contamination — caught by
        # the adversarial review's kappa-decomposition probe).
        _steep = cfg.K_iso_steep if K_iso_steep is None else K_iso_steep
        _, Fx, Fy, Fz = gm_redi_tracer_tendency_triads_latlon_cgrid(
            q, rho_filled, mask, u_mask, v_mask, z_coord, jacobian, grid,
            kappa_GM_arg, kappa_Redi_arg, cfg.S_max, cfg.taper_width_frac,
            cfg.implicit_K33, _steep, slope_density=slope_density,
            slope_limit=getattr(cfg, "slope_limit", "dm95_taper"),
            T_tracer=T, S_tracer=S, eos_fn=eos_fn, rho_0=rho_0, g=g,
            return_fluxes=True,
            double_diag_kappa=double_diag_kappa,
            double_diag_steep=cfg.K_iso_steep,
            veros_triad_weights=bool(
                getattr(cfg, "veros_triad_weights", False)),
            adjoint_stabilization=getattr(
                cfg, "adjoint_stabilization", "none"),
        )
        return Fx, Fy, Fz

    # Veros's SKEW-pass fluxes carry the precomputed K_11/K_22 (K_iso-based)
    # diagonal (diffusion.py:40-47 — unconditional), so its P_diss_skew
    # includes that diagonal's dissipation.  With the double_redi_diagonal
    # option the skew-only fluxes here carry the same extra diagonal (built
    # from the Redi kappa + its steep floor), keeping the realized signed
    # skew bookkeeping Veros-faithful.  Off (default): bit-identical.
    _skew_ddk = None
    if bool(getattr(cfg, "double_redi_diagonal", False)):
        _skew_ddk = kappa_redi_w if kappa_redi_w is not None else cfg.kappa_Redi

    neg_skew = None
    if want_skew:
        # SKEW-only fluxes: kappa_Redi=0, kappa_GM=kappa_gm_w (Veros K_iso=0,
        # K_skew=K_gm) — with the Redi-diagonal K_iso_steep floor ZEROED.
        FxT, FyT, FzT = _fluxes(T, kappa_gm_w, zero_kappa, K_iso_steep=0.0,
                                double_diag_kappa=_skew_ddk)
        FxS, FyS, FzS = _fluxes(S, kappa_gm_w, zero_kappa, K_iso_steep=0.0,
                                double_diag_kappa=_skew_ddk)
        P_skew = (
            _dynamic_enthalpy_dissipation_wgrid(
                int_drhodT, FxT, FyT, FzT, grid, dz_cell, mask, g, rho_0)
            + _dynamic_enthalpy_dissipation_wgrid(
                int_drhodS, FxS, FyS, FzS, grid, dz_cell, mask, g, rho_0)
        )
        neg_skew = -P_skew * mask[:, :, jnp.newaxis]

    neg_iso = None
    if want_iso:
        kappa_redi = kappa_redi_w if kappa_redi_w is not None else kappa_gm_w
        # ISO-only fluxes: kappa_GM=0, kappa_Redi=kappa_redi (Veros K_iso,K_skew=0).
        FxTi, FyTi, FzTi = _fluxes(T, zero_kappa, kappa_redi)
        FxSi, FySi, FzSi = _fluxes(S, zero_kappa, kappa_redi)
        K33 = dq_dzT = dq_dzS = None
        if cfg.implicit_K33:
            # K_33 vertical isoneutral diagonal (dropped from the explicit F_z when
            # implicit) — its dynamic-enthalpy dissipation is part of Veros's
            # P_diss_iso (diffusion.py:274-278).  K_33 = 0.25·kappa_Redi·Σ taper·S².
            K33 = compute_isoneutral_K33_latlon(
                T, S, eta, H_bathy, grid, z_coord, cfg, eos=eos,
                eos_linear=eos_linear, mask=mask, rho_0=rho_0, g=g,
                kappa_redi_override=kappa_redi,
                # #1226: same wall masks as this function's flux path.
                u_mask=u_mask, v_mask=v_mask, dt=dt,
            )
            Tf = neumann_fill_cgrid(T, mask)
            Sf = neumann_fill_cgrid(S, mask)
            dz_half = 0.5 * (dz_cell[..., :-1] + dz_cell[..., 1:])
            dq_dzT = (Tf[:, :, :-1] - Tf[:, :, 1:]) / jnp.maximum(dz_half, _EPS_DIV)
            dq_dzS = (Sf[:, :, :-1] - Sf[:, :, 1:]) / jnp.maximum(dz_half, _EPS_DIV)
        P_iso = (
            _dynamic_enthalpy_dissipation_wgrid(
                int_drhodT, FxTi, FyTi, FzTi, grid, dz_cell, mask, g, rho_0,
                K_33=K33, dq_dz_w=dq_dzT)
            + _dynamic_enthalpy_dissipation_wgrid(
                int_drhodS, FxSi, FySi, FzSi, grid, dz_cell, mask, g, rho_0,
                K_33=K33, dq_dz_w=dq_dzS)
        )
        neg_iso = -P_iso * mask[:, :, jnp.newaxis]

    return neg_skew, neg_iso


# =====================================================================
# LateralMixingOutput wrapper (for future factory integration)
# =====================================================================

def gm_redi_lateral_mixing_latlon(
    T: jnp.ndarray,
    S: jnp.ndarray,
    eta: jnp.ndarray,
    H_bathy: jnp.ndarray,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    cfg: GMRediConfig,
    **kwargs,
) -> LateralMixingOutput:
    """GM/Redi returning ``LateralMixingOutput`` (factory-compatible API).

    Wraps ``gm_redi_tracer_tendency_latlon`` with zero momentum tendencies.
    """
    dT_dt, dS_dt = gm_redi_tracer_tendency_latlon(
        T, S, eta, H_bathy, grid, z_coord, cfg, **kwargs,
    )
    return LateralMixingOutput(
        du_dt=jnp.zeros_like(dT_dt[:, :1, :]),  # placeholder shape
        dv_dt=jnp.zeros_like(dT_dt[:1, :, :]),
        dT_dt=dT_dt,
        dS_dt=dS_dt,
    )


def eke_horizontal_transport(E, U_bar, V_bar, grid, eke_cfg, mask, u_mask, v_mask):
    """Conservative 2-D EKE transport tendency [m^2/s^3]: flux-form advection of the
    eddy-energy field ``E`` by the depth-mean flow (upwind) + lateral diffusion
    (``k_iso``). Returns ``dE/dt|transport``.

    Conserves the area-integral of E by construction: the advective flux divergence
    telescopes (periodic in lon; v-flux = 0 at the N/S walls) and the lateral
    diffusion is flux-form (``laplacian_cgrid`` = div of grad, no-flux walls).
    Reuses ``divergence_cgrid`` + ``laplacian_cgrid`` — no duplicate numerics.

    Parameters
    ----------
    E : (n_lat, n_lon) eddy kinetic energy.
    U_bar : (n_lat, n_lon+1) depth-mean zonal velocity at u-faces.
    V_bar : (n_lat+1, n_lon) depth-mean meridional velocity at v-faces (0 at poles).
    grid, eke_cfg (k_iso), mask/u_mask/v_mask.
    """
    # E upwinded to u-faces by U_bar sign (periodic in lon).
    E_west = jnp.roll(E, 1, axis=1)                          # E[:, j-1]
    E_uface_core = jnp.where(U_bar[:, :-1] > 0.0, E_west, E)  # upwind
    E_uface = jnp.concatenate([E_uface_core, E_uface_core[:, 0:1]], axis=1)
    # E upwinded to interior v-faces by V_bar sign; poles are walls (V_bar=0).
    E_vface_int = jnp.where(V_bar[1:-1, :] > 0.0, E[:-1, :], E[1:, :])
    zero_row = jnp.zeros((1, E.shape[1]), dtype=E.dtype)
    E_vface = jnp.concatenate([zero_row, E_vface_int, zero_row], axis=0)
    # Flux-form advection (conservative).
    flux_u = E_uface * U_bar * u_mask
    flux_v = E_vface * V_bar * v_mask
    adv = -divergence_cgrid(flux_u, flux_v, grid)
    # Lateral diffusion (conservative).
    diff = eke_cfg.k_iso * laplacian_cgrid(E, grid, mask=mask)
    return (adv + diff) * mask


def eke_3d_horizontal_transport(E, U, V, grid, eke_cfg, mask, u_mask, v_mask):
    """Conservative 3-D EKE transport tendency [m^2/s^3] at the interior interfaces
    (the ``eke_3d=True`` path): flux-form advection of the eddy-energy field ``E``
    by the flow at EACH interface level (upwind) + per-level lateral diffusion
    (``k_iso``). Returns ``dE/dt|transport`` with shape ``(n_lat, n_lon, nlev-1)``.

    This is the depth-resolved generalisation of the 2-D
    :func:`eke_horizontal_transport`: ``E`` is advected by the LOCAL flow ``u(z),
    v(z)`` at each interface (Veros advects ``vs.eke`` by the W-grid velocities),
    NOT by the depth-mean flow. The conservative flux-form, upwind face values,
    and the reuse of ``divergence_cgrid`` (3-D-native) + ``laplacian_cgrid``
    (3-D-native, per level) are identical to the 2-D path — only the arrays carry an
    extra level axis, so the per-level numerics are bit-identical to the 2-D scheme
    at each level and no numerics are duplicated.

    Conservation: at every interface level the advective flux divergence telescopes
    (periodic in lon; v-flux = 0 at the N/S walls) and the lateral diffusion is
    flux-form (div of grad, no-flux walls), so the volume-integral
    ``Σ E·area·dz`` is conserved by construction (each level's area-integral is).

    NOTE (parity with Veros): Veros uses ``max(500, K_gm)·∇E`` for the lateral
    diffusivity of EKE (``veros/core/eke.py:173,183``); legoESM keeps the validated
    2-D path's constant ``eke_cfg.k_iso·∇E`` here for continuity (a depth-/flow-
    independent lateral diffusivity). The W-grid VERTICAL advection of E (Veros's
    ``adv_flux_top`` / ``flux_top`` Adams-Bashforth term) is NOT included here — see
    the module/stage notes; it is a documented known omission (the dominant 3-D
    effects — depth-resolved source/sink, implicit vertical diffusion, and per-level
    horizontal advection + lateral diffusion — are all present).

    Parameters
    ----------
    E : (n_lat, n_lon, nlev-1) eddy kinetic energy at interior interfaces.
    U : (n_lat, n_lon+1, nlev-1) zonal velocity at u-faces, per interface level.
    V : (n_lat+1, n_lon, nlev-1) meridional velocity at v-faces (0 at poles).
    grid, eke_cfg (k_iso), mask/u_mask/v_mask (2-D; broadcast over levels).
    """
    # E upwinded to u-faces by U sign at each level (periodic in lon).
    E_west = jnp.roll(E, 1, axis=1)                          # E[:, j-1, :]
    E_uface_core = jnp.where(U[:, :-1, :] > 0.0, E_west, E)   # upwind, (n_lat, n_lon, nlev-1)
    E_uface = jnp.concatenate([E_uface_core, E_uface_core[:, 0:1, :]], axis=1)
    # E upwinded to interior v-faces by V sign; poles are walls (V=0).
    E_vface_int = jnp.where(V[1:-1, :, :] > 0.0, E[:-1, :, :], E[1:, :, :])
    zero_row = jnp.zeros((1, E.shape[1], E.shape[2]), dtype=E.dtype)
    E_vface = jnp.concatenate([zero_row, E_vface_int, zero_row], axis=0)
    # Flux-form advection (conservative); face masks broadcast over the level axis.
    flux_u = E_uface * U * u_mask[:, :, jnp.newaxis]
    flux_v = E_vface * V * v_mask[:, :, jnp.newaxis]
    adv = -divergence_cgrid(flux_u, flux_v, grid)            # 3-D-native
    # Lateral diffusion (conservative), per level (laplacian_cgrid is 3-D-native).
    diff = eke_cfg.k_iso * laplacian_cgrid(E, grid, mask=mask)
    return (adv + diff) * mask[:, :, jnp.newaxis]


def eke_3d_vertical_diffusion(E, A_v_profile, dz_w, dz_half_w, dt, eke_cfg):
    """Backward-Euler implicit vertical diffusion of the 3-D eddy-energy field ``E``
    (the ``eke_3d=True`` path), with diffusivity ``K = alpha_eke · A_v`` — Veros's
    ``delta = dt/dzt · 0.5(kappaM[k]+kappaM[k+1]) · alpha_eke`` (``veros/core/eke.py``
    :134-142), here as a clean reuse of the shared
    :func:`implicit_vertical_diffusion_ocean` (no duplicate tridiagonal numerics).

    ``E`` lives on the interior interfaces (the W-grid, ``M = nlev-1`` levels), so the
    implicit diffusion operates over those ``M`` levels with zero-flux BCs at the top
    and bottom of the W-grid column. The caller (the model step, a later build stage)
    supplies the W-grid metrics ``dz_w`` (M layer thicknesses) + ``dz_half_w`` (M-1
    spacings between adjacent W-levels) and the vertical viscosity ``A_v_profile`` at
    the ``M-1`` interior W-interfaces. ``alpha_eke`` scales it (Veros's vertical-
    friction factor). Unconditionally stable + AD-safe (the underlying Thomas solve
    is differentiable).

    NOTE: Veros's EKE dissipation ``c_int = c_eps·√E/eke_len`` is folded into the
    SAME tridiagonal implicit solve (its ``b_tri`` diagonal). legoESM keeps the
    validated split treatment: the local source/sink (incl. the semi-implicit
    dissipation, :func:`legoesm.ocean.physics.lateral_mixing.eke.eke_apply_local_source`)
    is applied separately by the step; this function applies ONLY the vertical
    diffusion. The two operator-split sub-steps are each unconditionally stable.

    Parameters
    ----------
    E : (..., M) eddy kinetic energy on the W-grid (M = nlev-1 interior interfaces).
    A_v_profile : (..., M-1) or float — vertical viscosity at the interior W-grid
        interfaces (>= 0). Scaled by ``alpha_eke``. A scalar/profile is fine.
    dz_w : (..., M) or (M,) — W-grid layer thicknesses [m].
    dz_half_w : (..., M-1) or (M-1,) — spacing between adjacent W-levels [m].
    dt : float — time step [s].
    eke_cfg : EKEConfig (uses ``alpha_eke``).
    """
    from legoesm.ocean.physics.vertical_mixing import (
        implicit_vertical_diffusion_ocean,
    )
    K = eke_cfg.alpha_eke * jnp.asarray(A_v_profile)
    return implicit_vertical_diffusion_ocean(E, K, dz_w, dz_half_w, dt)


def _eke_stage1_fields(
    T, S, eta, H_bathy, grid, z_coord, cfg, *,
    eos="wright", eos_linear=None, mask=None, rho_0=_RHO_0, g=constants.g,
    omega=constants.Omega, r_earth=constants.R_earth,
):
    """Stage-1 fields shared by the prognostic-EKE closures (Eden-Greatbatch
    ``compute_eke_step_kappa`` and GEOMETRIC ``compute_geometric_step_kappa``):
    density + isopycnal slopes via the SAME shared helpers GM/Redi uses
    internally (a redundant recompute -- correct; compute-once is a future
    optimization), the Coriolis field, the analytic beta = 2*Omega*cos(phi)/R,
    and -- only when ``cfg.eke.n2_mode == "adiabatic"`` -- the cell-centre
    hydrostatic pressure + EOS handles for the adiabatic N^2 (otherwise None,
    byte-identical default).  Pure code motion from ``compute_eke_step_kappa``
    (bit-identical numerics).

    Returns ``(mask, jacobian, rho, S_x, S_y, f_coriolis, beta, T_eos, S_eos,
    p_cell, eos_for_n2)``.
    """
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: neumann_fill_cgrid(field, mask)
    rho, _rp, _pp = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2,
    )
    S_x, S_y, _taper = compute_isopycnal_slopes_latlon_cgrid(
        rho, mask, z_coord, jacobian, grid, cfg,
        T=T, S=S, eos_fn=eos_fn, rho_0=rho_0, g=g,
    )
    f_coriolis = jnp.broadcast_to(grid.f, mask.shape)
    # beta = df/dy = 2*Omega*cos(phi)/R (analytic; grid.cos_lat is cos(phi)).
    # Broadcast (n_lat,) -> (n_lat, n_lon) to match f. Used only by the
    # "rhines" eke_len scheme (the GEOMETRIC closure ignores it).
    beta = jnp.broadcast_to(
        (2.0 * omega * grid.cos_lat / r_earth)[:, None], mask.shape,
    )
    # Adiabatic static-stability N^2 (Veros EKE chain) needs the cell-centre
    # hydrostatic pressure + the same EOS as the dynamical core. Only built
    # when the EKE config opts in (``n2_mode="adiabatic"``) so the default
    # ("insitu") path is byte-identical (kwargs stay None).
    T_eos = S_eos = p_cell = eos_for_n2 = None
    if getattr(cfg.eke, "n2_mode", "insitu") == "adiabatic":
        from legoesm.ocean.vertical import (
            OceanPartialCellCoordinate, compute_layer_thickness,
        )
        # Partial-cell thickness when applicable (matches k_profiles.py's
        # adiabatic-N^2 path), else compute_hydrostatic_pressure falls back to
        # dz_ref * jacobian internally.
        h_actual = None
        if isinstance(z_coord, OceanPartialCellCoordinate):
            h_actual = compute_layer_thickness(eta, H_bathy, z_coord)
        p_cell = compute_hydrostatic_pressure(
            rho, eta, z_coord.dz_ref, jacobian, rho_0, g, h_actual=h_actual,
        )
        T_eos, S_eos, eos_for_n2 = T, S, eos_fn
    return (mask, jacobian, rho, S_x, S_y, f_coriolis, beta,
            T_eos, S_eos, p_cell, eos_for_n2)


def compute_eke_step_kappa(
    T, S, eta, H_bathy, eke, grid, z_coord, cfg, *,
    eos="wright", eos_linear=None, mask=None, rho_0=_RHO_0, g=constants.g,
    omega=constants.Omega, r_earth=constants.R_earth,
    depth_resolved=False,
):
    """Prognostic GM coefficient + Eady growth rate + mixing length from the
    eddy-energy field, for the EKE-active model step. Returns ``(kappa_GM,
    sigma, L)``: ``kappa_GM`` is the override fed into the GM/Redi tracer
    tendency; ``sigma``/``L`` drive the EKE local source/sink.

    Two shapes, selected by the static Python ``depth_resolved`` flag (the
    ``EKEConfig.eke_3d`` switch in the model step):

    - ``depth_resolved=False`` (default, the 2-D path): ``eke`` is 2-D
      ``(n_lat, n_lon)`` and the return is the 2-D ``(kappa_GM, sigma_bar, L)``
      — ``kappa_GM`` the 2-D GM override, ``sigma_bar`` the depth-AVERAGED Eady
      growth ``<N|S|>_z``, ``L`` the 2-D mixing length. Bit-identical to before.
    - ``depth_resolved=True`` (the 3-D ``eke_3d`` path): ``eke`` is 3-D
      ``(n_lat, n_lon, nlev-1)`` at the interior interfaces (W-grid) and the
      return is the depth-resolved ``(kappa_GM(z), sigma(z), L(z))`` all at the
      ``nlev-1`` interfaces — the 3-D GM-skew override AND the per-level Eady
      growth / mixing length the depth-resolved EKE source/sink consumes
      (:func:`legoesm.ocean.physics.lateral_mixing.eke.eke_3d_local_tendency`).

    Recomputes rho + isopycnal slopes with the SAME shared helpers GM/Redi uses
    internally (a redundant recompute — correct; compute-once is a future
    optimization), then ``compute_eke_kappa_gm`` (which reuses the shared
    Eady-length machinery and itself dispatches on ``depth_resolved``). ``cfg``
    is the GMRediConfig (uses ``cfg.visbeck`` for the Rossby-length params and
    ``cfg.eke`` for the closure params).

    For ``cfg.eke.mixing_length_scheme == "rhines"`` the eke_len needs ``β =
    df/dy``; it is computed analytically as ``2Ω·cosφ/R`` (exact on the sphere
    where ``f = 2Ω sinφ``; equals Veros's discrete ``df/dy`` to O(dφ²)). ``Ω``/``R``
    come from the model constants (``omega``/``r_earth``; Veros-pinned in the ACC
    recipe), never literals. The ``"rossby"`` scheme ignores ``β``.
    """
    (mask, jacobian, rho, S_x, S_y, f_coriolis, beta,
     T_eos, S_eos, p_cell, eos_for_n2) = _eke_stage1_fields(
        T, S, eta, H_bathy, grid, z_coord, cfg,
        eos=eos, eos_linear=eos_linear, mask=mask, rho_0=rho_0, g=g,
        omega=omega, r_earth=r_earth,
    )
    return compute_eke_kappa_gm(
        eke, rho, S_x, S_y, z_coord, jacobian, f_coriolis,
        cfg.visbeck, cfg.eke, rho_ref=rho_0, g=g, beta=beta,
        depth_resolved=depth_resolved,
        T=T_eos, S=S_eos, p_cell=p_cell, eos_fn=eos_for_n2,
    )


def compute_geometric_step_kappa(
    T, S, eta, H_bathy, eke_int, grid, z_coord, cfg, *,
    eos="wright", eos_linear=None, mask=None, rho_0=_RHO_0, g=constants.g,
    omega=constants.Omega, r_earth=constants.R_earth,
):
    """GEOMETRIC (Torres et al. 2025, JAMES, doi:10.1029/2025MS005394) eddy
    coefficients + EKE-budget pieces for the EKE-active model step.

    The prognostic field is the DEPTH-INTEGRATED ``eke_int = ∫EKE dz``
    [m³/s²] (paper Eq. 1).  Pass ``eke_int=None`` for a cold start: the
    paper's depth-proportional initial condition ``e0_per_depth·H`` is built
    from the column depth (Appendix E, p. 35; the static Python None-branch
    mirrors the EG path's ``state.eke is None`` fill).

    Reuses the SHARED stage-1 fields (``_eke_stage1_fields``: rho + tapered
    isopycnal slopes via the same helpers the GM/Redi tendency uses) and the
    shared column integrals (``compute_geometric_column_integrals``), then
    the pure GEOMETRIC formulas from ``eke.py``:

      kappa_gm  = alpha·∫E dz / max(∫M²/N dz, mn_floor)        (Eq. 6, 2-D)
      kappa_n   = Gamma·min(R_d, l_mix_max)·√(2·∫E dz/H)       (Eq. 7, 2-D;
                  ``None`` unless ``geometric.kappa_n_coupling``)
      B_C       = kappa_gm·∫M⁴/N² dz                           (Eq. 2) [m³/s³]
      L_eff     = R_d·√H  (the Eq.-4 dissipation folded as the
                  ``eke_apply_local_source`` implicit rate
                  ``c_eps_geometric·√I/L_eff = C_eps·√(I/H)/R_d``)
      R_d       = clip(rossby_factor·∫N dz/|f|, 2-40 km)       (Appendix D)

    Both kappas are wet-masked AFTER clipping (dry columns contribute exactly
    zero diffusivity — the same convention as ``compute_visbeck_kappa_gm``).

    Returns ``(eke_int, kappa_gm, kappa_n_or_None, production_bc, L_eff,
    dz_actual)`` — the first five 2-D ``(n_lat, n_lon)``; ``dz_actual``
    ``(n_lat, n_lon, nlev)`` is the cell-centre thickness measure
    (``dz_ref·jacobian``) the step feeds to
    :func:`geometric_barotropic_production`, so the B_T (Eq. 3) depth
    integral uses the SAME measure as the other column integrals.
    """
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_geometric_column_integrals,
    )
    from legoesm.ocean.physics.lateral_mixing.eke import (
        geometric_dissipation_length,
        geometric_kappa_gm,
        geometric_kappa_n,
        geometric_rossby_radius,
    )

    geom = cfg.eke.geometric
    (mask, jacobian, rho, S_x, S_y, f_coriolis, _beta,
     T_eos, S_eos, p_cell, eos_for_n2) = _eke_stage1_fields(
        T, S, eta, H_bathy, grid, z_coord, cfg,
        eos=eos, eos_linear=eos_linear, mask=mask, rho_0=rho_0, g=g,
        omega=omega, r_earth=r_earth,
    )
    int_sigma2_dz, int_sigma_dz, int_N_dz, H_col, wet_col = (
        compute_geometric_column_integrals(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg.visbeck,
            rho_ref=rho_0, n2_mode=getattr(cfg.eke, "n2_mode", "insitu"),
            n2_over_dzw=getattr(cfg.eke, "n2_over_dzw", False),
            T=T_eos, S=S_eos, p_cell=p_cell, eos_fn=eos_for_n2,
        ))
    if eke_int is None:
        # Cold start: ∫EKE dz = e0_per_depth·H (Torres et al. 2025 App. E,
        # p. 35: "10⁻⁶·h in m³/s²"), zero on land.
        eke_int = geom.e0_per_depth * H_col * mask
    r_d = geometric_rossby_radius(int_N_dz, f_coriolis, geom)
    kappa_gm = geometric_kappa_gm(eke_int, int_sigma_dz, geom)
    kappa_gm = jnp.where(wet_col, kappa_gm, 0.0)
    kappa_n = None
    if geom.kappa_n_coupling:
        kappa_n = geometric_kappa_n(eke_int, H_col, r_d, geom)
        kappa_n = jnp.where(wet_col, kappa_n, 0.0)
    production_bc = kappa_gm * int_sigma2_dz          # Eq. 2 [m³/s³], ≥ 0
    L_eff = geometric_dissipation_length(r_d, H_col)
    # Cell-centre thickness measure for the B_T depth integral (the same
    # metric assembly as compute_geometric_column_integrals).
    dz_actual = z_coord.dz_ref * jacobian[..., jnp.newaxis]
    return eke_int, kappa_gm, kappa_n, production_bc, L_eff, dz_actual


def geometric_barotropic_production(
    u, v, grid, kappa_u, dz_actual, mask, u_mask, v_mask, *, z_coord=None,
):
    """GEOMETRIC barotropic EKE production ``B_T = ∫kappa_u·|∇h u_h|² dz``
    (Torres et al. 2025, Eq. 3, p. 4; kappa_u = 1500 m²/s calibrated,
    Appendix E p. 38) at cell centres [m³/s³], ≥ 0 by construction.

    ``kappa_u·|∇h u_h|²`` is EXACTLY the positive-definite component-wise
    flux-form dissipation density the K_diss_h machinery already provides
    (``flux_divergence_viscosity_cgrid(want_dissipation=True)``, Veros
    ``calc_diss_u``/``calc_diss_v`` analogue) with the viscosity replaced by
    the eddy momentum diffusivity ``kappa_u`` and NO cos-power scaling
    (kappa_u is constant in the paper) — reused, not re-derived.  The unused
    viscous tendencies it also returns are discarded (a small constant-factor
    overhead, once per step).  NOTE: that operator raises on tripolar grids;
    the GEOMETRIC closure inherits the restriction (regular lat-lon only).

    On a PARTIAL-CELL coordinate (``z_coord`` given) the gradients are
    masked with the PER-LEVEL 3-D face masks (``compute_face_masks_3d``) and
    per-level-masked velocities — the SAME fix as the K_diss_h flux-form
    source (ocean_pe_latlon_cgrid): 2-D-only masks treat a face that is
    closed at depth (topographic step) as a u=0 wall, crediting a spurious
    no-slip shear ``|∇u|²`` to the EKE source (probe-measured +37% of the
    production-path EKE source on the global_4deg yr-3 state).  Full-cell
    coordinates: 3-D masks are all-ones ⇒ identical to the 2-D path.

    Parameters
    ----------
    u, v : 3-D face velocities (n_lat, n_lon+1, nlev) / (n_lat+1, n_lon, nlev).
    kappa_u : float (or traced scalar) — eddy momentum diffusivity [m²/s].
    dz_actual : (n_lat, n_lon, nlev) cell-centre layer thicknesses [m]
        (dz_ref·jacobian — the same measure as the other column integrals).
    mask / u_mask / v_mask : 2-D wet masks (centres / u-faces / v-faces).
    z_coord : optional vertical coordinate; when an
        ``OceanPartialCellCoordinate``, its ``is_active`` drives the 3-D
        face masks (free-slip at topographic steps).

    Returns
    -------
    B_T : (n_lat, n_lon) ≥ 0 [m³/s³].
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        flux_divergence_viscosity_cgrid,
    )
    from legoesm.ocean.vertical import OceanPartialCellCoordinate

    if isinstance(z_coord, OceanPartialCellCoordinate):
        um3, vm3 = compute_face_masks_3d(z_coord.is_active, grid)
        um, vm = um3.astype(u.dtype), vm3.astype(v.dtype)
        u_eff, v_eff = u * um, v * vm
    else:
        um, vm = u_mask, v_mask
        u_eff, v_eff = u, v
    _vu, _vv, diss = flux_divergence_viscosity_cgrid(
        u_eff, v_eff, grid, kappa_u, cos_power=0,
        mask=mask, u_mask=um, v_mask=vm, want_dissipation=True,
    )
    # Depth integral with the cell-centre thickness measure; diss is already
    # centre-masked by the operator, the dz product re-applies the 2-D mask.
    return jnp.sum(diss * dz_actual, axis=-1) * mask


def _nemo_native_slope_diagnostics(scope):
    """Expose the native-slope causal rows only through the test hook."""
    dtype = scope["dtype"]
    zgru, zgrv = scope["zgru"], scope["zgrv"]
    if scope["_metric_mode"] == "nemo_reciprocal":
        r1_e1u = lax.optimization_barrier(
            jnp.asarray(1.0, dtype=dtype) / scope["e1u"])
        r1_e2v = lax.optimization_barrier(
            jnp.asarray(1.0, dtype=dtype) / scope["e2v"])
        zau = lax.optimization_barrier(zgru * r1_e1u[..., None])
        zav = lax.optimization_barrier(zgrv * r1_e2v[..., None])
    else:
        r1_e1u = jnp.asarray(1.0, dtype=dtype) / scope["e1u"]
        r1_e2v = jnp.asarray(1.0, dtype=dtype) / scope["e2v"]
        zau = zgru / scope["e1u"][..., None]
        zav = zgrv / scope["e2v"][..., None]

    zbu_raw, zbv_raw = scope["zb_u"], scope["zb_v"]
    zbu = jnp.minimum(
        zbu_raw,
        jnp.minimum(-scope["z1_slpmax"] * jnp.abs(zau),
                    (-_NEMO_SLOPE_STAB_7E3 / scope["e3u_k"])
                    * jnp.abs(zau)))
    zbv = jnp.minimum(
        zbv_raw,
        jnp.minimum(-scope["z1_slpmax"] * jnp.abs(zav),
                    (-_NEMO_SLOPE_STAB_7E3 / scope["e3v_k"])
                    * jnp.abs(zav)))
    s_int_u = zau / (zbu - scope["zeps"])
    s_int_v = zav / (zbv - scope["zeps"])
    iku, ikv, kidx = scope["iku"], scope["ikv"], scope["kidx"]
    anchor_u = (jnp.take_along_axis(
        s_int_u, iku[..., None], axis=-1)[..., 0] * scope["r1_hmlu"])
    anchor_v = (jnp.take_along_axis(
        s_int_v, ikv[..., None], axis=-1)[..., 0] * scope["r1_hmlv"])
    valid = kidx > 0
    pre_u = jnp.where((kidx < iku[..., None]) & valid,
                      anchor_u[..., None], 0.0)
    pre_v = jnp.where((kidx < ikv[..., None]) & valid,
                      anchor_v[..., None], 0.0)
    post_u = jnp.where((kidx <= iku[..., None]) & valid,
                       anchor_u[..., None], 0.0)
    post_v = jnp.where((kidx <= ikv[..., None]) & valid,
                       anchor_v[..., None], 0.0)
    zfi = (kidx >= iku[..., None]).astype(dtype) * valid
    zfj = (kidx >= ikv[..., None]).astype(dtype) * valid
    zmli = (kidx == iku[..., None]).astype(dtype) * valid
    zmlj = (kidx == ikv[..., None]).astype(dtype) * valid

    stretch = scope.get("_live_stretch")
    if stretch is None:
        stretch = jnp.ones_like(scope["mask"], dtype=dtype)
    r3t = stretch - jnp.asarray(1.0, dtype=dtype)
    hp = getattr(scope["z_coord"], "h_partial", None)
    if hp is None:
        base_u = scope["dz"][None, None, :]
        base_v = base_u
    else:
        h3 = jnp.asarray(hp, dtype=dtype)
        floor = jnp.asarray(1.0e-10, dtype=dtype)
        base_u = jnp.maximum(jnp.minimum(h3, jnp.roll(h3, -1, axis=1)), floor)
        base_v = jnp.maximum(jnp.minimum(h3, jnp.roll(h3, -1, axis=0)), floor)
    r3u = jnp.where(scope["umask3"][..., 0] != 0.0,
                    scope["e3u_k"][..., 0] / base_u[..., 0] - 1.0, 0.0)
    r3v = jnp.where(scope["vmask3"][..., 0] != 0.0,
                    scope["e3v_k"][..., 0] / base_v[..., 0] - 1.0, 0.0)
    ones = jnp.ones_like(scope["first"], dtype=dtype)
    gdept_0 = jnp.asarray(
        getattr(scope["z_coord"], "nemo_gdept_0", scope["gdept"]),
        dtype=dtype)
    gdepw_0 = jnp.asarray(
        getattr(scope["z_coord"], "nemo_gdepw_0", scope["gdepw_top"]),
        dtype=dtype)
    gdept_1d = gdept_0[0, 0] if gdept_0.ndim == 3 else gdept_0
    gdepw_1d = gdepw_0[0, 0] if gdepw_0.ndim == 3 else gdepw_0
    e3w = scope["e3w"]
    e3w_1d = e3w[0, 0] / stretch[0, 0] if e3w.ndim == 3 else e3w

    return {
        "prd": scope["prd"], "pn2": scope["pn2"],
        "tmask": scope["act"], "umask": scope["umask3"],
        "vmask": scope["vmask3"], "wmask": scope["wmask3"],
        "e3u_live": scope["e3u_k"], "e3v_live": scope["e3v_k"],
        "zgru": zgru, "zgrv": zgrv, "zdzr": scope["zdzr"],
        "zau": zau, "zav": zav, "zbu_raw": zbu_raw,
        "zbv_raw": zbv_raw, "zbu_limited": zbu,
        "zbv_limited": zbv, "zfi": zfi, "zfj": zfj,
        "zmli": zmli, "zmlj": zmlj, "zdepu": scope["zdepu"],
        "zdepv": scope["zdepv"], "zwz": scope["_uslp_raw"],
        "zww": scope["_vslp_raw"], "zuslp_pre": pre_u,
        "zvslp_pre": pre_v, "zuslp_post": post_u,
        "zvslp_post": post_v, "uslp": scope["uslp"],
        "vslp": scope["vslp"], "wslpi": scope["wslpi"], "wslpj": scope["wslpj"], "r3t_Kmm": r3t,
        "r3u_Kmm": r3u, "r3v_Kmm": r3v,
        "zhmlpt": scope["zhmlpt"], "r1_hmlu": scope["r1_hmlu"],
        "r1_hmlv": scope["r1_hmlv"],
        "r1_hmlw": 1.0 / jnp.maximum(scope["hml"], 10.0),
        "hmlp": scope["hml"], "ssmask": scope["mask"],
        "r1_e1u": r1_e1u, "r1_e2v": r1_e2v,
        "nmln": (scope["first"] + 1).astype(dtype),
        "miku": ones, "mikv": ones, "mikt": ones,
        "iku": (iku + 1).astype(dtype), "ikv": (ikv + 1).astype(dtype),
        "gdept_1d": gdept_1d, "gdepw_1d": gdepw_1d,
        "e3w_1d": e3w_1d,
    }
