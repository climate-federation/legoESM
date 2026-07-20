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
import jax.numpy as jnp

from legoesm import constants
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
    rho_0 as _RHO_0,
)
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    EPS,
    EPS_DIV as _EPS_DIV,
    compute_eke_kappa_gm,
    compute_treguier_kappa_gm,
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
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_ocean_jacobian

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


def _nemo_mld_from_potential_density(T, S, mask, z_coord, eos_fn, rho_c):
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
    T_filled = neumann_fill_cgrid(T, mask)
    S_filled = neumann_fill_cgrid(S, mask)
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


def _nemo_mld_from_n2_integral(T, S, mask, z_coord, eos_fn, rho_c, g, rho_0):
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
    T_filled = neumann_fill_cgrid(T, mask)
    S_filled = neumann_fill_cgrid(S, mask)
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
    n2_int = compute_buoyancy_frequency_adiabatic(
        T_filled, S_filled, p_cell, dz_ref, J1, eos_fn=eos_fn,
        rho_ref=rho_0, g=g)                           # (...,nlev-1)
    # e3w(jk) for interface m = spacing between the bracketing T-centres.
    e3w = z_centers[1:] - z_centers[:-1]             # (nlev-1,)
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
        jnp.maximum(n2_int, jnp.zeros((), dtype)) * e3w[None, None, :])
    cum = jnp.cumsum(contrib, axis=-1)               # integral(N^2 dz) from nlb10
    thresh = jnp.asarray(g * rho_c / rho_0, dtype)   # zN2_c = g*rho_c/rho0
    reached = cum >= thresh                           # (...,nlev-1)
    has = jnp.any(reached, axis=-1)
    m_base = jnp.argmax(reached.astype(jnp.int32), axis=-1)   # shallowest crossing
    m_base = jnp.clip(jnp.where(has, m_base, nlev - 2), 0, nlev - 2)
    hml = jnp.take(z_iface, m_base)                   # (n_lat, n_lon)
    return hml, m_base


def _nemo_mld(criterion, T, S, mask, z_coord, eos_fn, rho_c, *,
              g=constants.g, rho_0=_RHO_0):
    """Dispatch the NEMO zdfmxl mixed-layer depth by criterion (raise on typo).

    ``"rho_c"`` (default, byte-identical) = potential-density difference;
    ``"n2_integral"`` = NEMO's exact integral(N^2 dz) >= g*rho_c/rho0 criterion.
    Both return ``(hml, m_base)`` in the same convention, so the ldfslp slope
    ramps consume either transparently.
    """
    if criterion == "rho_c":
        return _nemo_mld_from_potential_density(T, S, mask, z_coord, eos_fn, rho_c)
    if criterion == "n2_integral":
        return _nemo_mld_from_n2_integral(
            T, S, mask, z_coord, eos_fn, rho_c, g, rho_0)
    raise ValueError(
        f"unknown GMRediConfig.mld_criterion {criterion!r}; "
        "expected 'rho_c' or 'n2_integral'.")


def _apply_nemo_mld_slope_ramp(S_x, S_y, T, S, mask, z_coord, eos_fn, rho_c,
                               mld_criterion="rho_c", *,
                               g=constants.g, rho_0=_RHO_0):
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

    ponytail: the MLD depth ``hml`` is built from reference thicknesses
    (``z_coord.dz_ref``), ignoring z-star ``eta``/jacobian stretching — negligible
    for the flat-bottom, ``eta~0`` oracle configs (GYRE/DINO) this targets; carry
    the jacobian when a stretched/topography oracle needs it.
    """
    nlev_m1 = S_x.shape[-1]
    hml, m_base = _nemo_mld(
        mld_criterion, T, S, mask, z_coord, eos_fn, rho_c, g=g, rho_0=rho_0)
    z_iface = jnp.cumsum(z_coord.dz_ref)[:-1]         # (nlev-1,) interface depths
    # wslp_base = slope one interface BELOW the ML base (NEMO nmln+1).
    m_ref = jnp.clip(m_base + 1, 0, nlev_m1 - 1)
    Sx_base = jnp.take_along_axis(S_x, m_ref[:, :, None], axis=-1)  # (n_lat,n_lon,1)
    Sy_base = jnp.take_along_axis(S_y, m_ref[:, :, None], axis=-1)
    ramp = z_iface[None, None, :] / jnp.maximum(
        hml[:, :, None], _NEMO_HMLW_FLOOR_M)          # (n_lat, n_lon, nlev-1)
    in_ml = z_iface[None, None, :] <= hml[:, :, None]
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
    wE = mask * jnp.pad(mask, ((0, 0), (0, 1)))[:, 1:]    # wet(i,j) & wet(i,j+1)
    wW = mask * jnp.pad(mask, ((0, 0), (1, 0)))[:, :-1]   # wet(i,j) & wet(i,j-1)
    wN = mask * jnp.pad(mask, ((0, 1), (0, 0)))[1:, :]    # wet(i,j) & wet(i+1,j)
    wS = mask * jnp.pad(mask, ((1, 0), (0, 0)))[:-1, :]   # wet(i,j) & wet(i-1,j)
    zcofw = (m / 16.0) * (wE + wW)[:, :, None] * (wN + wS)[:, :, None] * 0.25  # coeff-ok: 16 = (1+2+1)^2 binomial weight sum (NEMO ldfslp z1_16)

    w = (1.0, 2.0, 1.0)                                   # 1-D binomial kernel
    def smooth(f):
        fp = jnp.pad(f * m, ((1, 1), (1, 1), (0, 0)))    # masked, zero ghost
        acc = jnp.zeros_like(f)
        for a in range(3):
            for b in range(3):
                acc = acc + w[a] * w[b] * fp[a:a + nlat, b:b + nlon, :]
        return acc * zcofw

    return smooth(S_x), smooth(S_y)


# =====================================================================
# Isopycnal slope computation
# =====================================================================

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

    Scope (v1, matches the operator's documented assumptions): reference
    geometry (dz_ref ladder; z-star eta/J stretching ignored — linssh/flat
    oracle configs), no ice shelves (risfdep=0, mikt=1), no partial cells.
    """
    zeps = 1.0e-20
    dtype = rho.dtype
    nlat, nlon, nlev = rho.shape
    dz = jnp.asarray(z_coord.dz_ref, dtype=dtype)                # (nlev,)
    gdept = jnp.cumsum(dz) - 0.5 * dz                            # cell centres
    gdepw_top = jnp.cumsum(dz) - dz                              # top-of-cell depth
    e3w = jnp.concatenate([dz[:1], gdept[1:] - gdept[:-1]])      # NEMO e3w(k)

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
    wmask3 = act * jnp.roll(act, +1, axis=2)
    wmask3 = wmask3.at[:, :, 0].set(act[:, :, 0])

    prd = rho / jnp.asarray(rho_0, dtype=dtype) - 1.0            # NEMO rhd

    # pn2: locally-referenced (adiabatic) N^2 at w-points, NEMO indexing
    # (pn2[k] at the TOP of cell k; pn2[0]=0). eos.compute_buoyancy_frequency_
    # adiabatic returns the nlev-1 interior interfaces (lego interface m =
    # NEMO w-level m+1).
    from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
    p_cell = (jnp.asarray(rho_0, dtype) * jnp.asarray(g, dtype)
              * gdept)[None, None, :] * jnp.ones_like(rho)
    J1 = jnp.ones((nlat, nlon), dtype=dtype)
    n2_int = compute_buoyancy_frequency_adiabatic(
        T, S, p_cell, z_coord.dz_ref, J1, eos_fn=eos_fn)          # (...,nlev-1)
    pn2 = jnp.concatenate([jnp.zeros((nlat, nlon, 1), dtype=dtype),
                           n2_int.astype(dtype)], axis=-1)        # (...,nlev)
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
        g=g, rho_0=rho_0)
    first = jnp.clip(m_base + 1, 1, nlev - 1)                    # (nlat,nlon) int
    # zhmlpt = gdept(nmln-1) = depth of the last T-point inside the ML
    zhmlpt = jnp.take(gdept, jnp.clip(first - 1, 0, nlev - 1)) * mask

    kidx = jnp.arange(nlev)[None, None, :]

    def _uv_slp(zg, zb_pair, e1_face, e3_face, iku, r1_hml, zdep_face, msk3):
        """Shared u/v-slope assembly (:206-238 without the Shapiro)."""
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
    zdepu = (gdept - 0.5 * dz[0])[None, None, :] * jnp.ones_like(zgru)
    e3u_k = dz[None, None, :]                                     # flat: e3u=e3t
    uslp = _uv_slp(zgru, zb_u, e1u, e3u_k, iku, r1_hmlu, zdepu, umask3)

    # --- vslp ---
    zb_v = 0.5 * (zdzr + jnp.roll(zdzr, -1, axis=0))
    ikv = jnp.maximum(first, jnp.roll(first, -1, axis=0))
    r1_hmlv = 1.0 / jnp.maximum(
        jnp.maximum(zhmlpt, jnp.roll(zhmlpt, -1, axis=0)),
        jnp.asarray(_NEMO_HML_UV_FLOOR_M, dtype))
    vslp = _uv_slp(zgrv, zb_v, e2v, e3u_k, ikv, r1_hmlv, zdepu, vmask3)

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
    zai = (zgru_im1 + zgru + _km1(zgru_im1) + _km1(zgru)) / zci * wmask3
    zaj = (zgrv_jm1 + zgrv + _km1(zgrv_jm1) + _km1(zgrv)) / zcj * wmask3
    zbw = (-0.5 / jnp.asarray(g, dtype)) * pn2 * (prd + _km1(prd) + 2.0)
    e3w_k = e3w[None, None, :]
    zbi = jnp.minimum(zbw, jnp.minimum(-z1_slpmax * jnp.abs(zai),
                                       (-_NEMO_SLOPE_STAB_7E3 / e3w_k) * jnp.abs(zai)))
    zbj = jnp.minimum(zbw, jnp.minimum(-z1_slpmax * jnp.abs(zaj),
                                       (-_NEMO_SLOPE_STAB_7E3 / e3w_k) * jnp.abs(zaj)))
    swi_int = zai / (zbi - zeps)
    swj_int = zaj / (zbj - zeps)
    # ML ramp (:284-297): in-ML for w-level jk <= nmln (1-based) — in the
    # 0-based top-of-cell-k indexing (jk = k+1): k <= first; anchor at
    # jk = nmln+1 => k = first+1 (the first w-level BELOW the ML base).
    kanc = jnp.clip(first + 1, 1, nlev - 1)
    r1_hmlw = 1.0 / jnp.maximum(hml, jnp.asarray(_NEMO_HMLW_FLOOR_M, dtype))
    anc_i = jnp.take_along_axis(swi_int, kanc[:, :, None], axis=-1)[:, :, 0] * r1_hmlw
    anc_j = jnp.take_along_axis(swj_int, kanc[:, :, None], axis=-1)[:, :, 0] * r1_hmlw
    zck = gdepw_top[None, None, :]
    in_ml_w = kidx < kanc[:, :, None]
    wslpi = jnp.where(in_ml_w, zck * anc_i[:, :, None], swi_int) * wmask3
    wslpj = jnp.where(in_ml_w, zck * anc_j[:, :, None], swj_int) * wmask3
    wslpi = wslpi.at[:, :, 0].set(0.0)
    wslpj = wslpj.at[:, :, 0].set(0.0)

    # --- Shapiro 1/16 + coastal decrease, native mask factors ---
    def _shap(f, cof):
        fp = jnp.pad(f, ((1, 1), (1, 1), (0, 0)))
        w = (1.0, 2.0, 1.0)
        acc = jnp.zeros_like(f)
        for a in range(3):
            for b in range(3):
                acc = acc + w[a] * w[b] * fp[a:a + nlat, b:b + nlon, :]
        return acc * cof / 16.0  # coeff-ok: 16 = (1+2+1)^2 binomial weight sum (NEMO ldfslp z1_16), same as L486

    def _kp1m(a):  # mask at level k+1, zero at the bottom
        return jnp.concatenate(
            [a[:, :, 1:], jnp.zeros((nlat, nlon, 1), dtype=dtype)], axis=-1)

    cof_u = 0.25 * (jnp.roll(umask3, -1, axis=0) + jnp.roll(umask3, +1, axis=0)) \
        * (umask3 + _kp1m(umask3))
    cof_v = 0.25 * (jnp.roll(vmask3, -1, axis=1) + jnp.roll(vmask3, +1, axis=1)) \
        * (vmask3 + _kp1m(vmask3))
    cof_w = 0.25 * wmask3 * (umask3 + um_im1) * (vmask3 + vm_jm1)
    uslp = _shap(uslp, cof_u)
    vslp = _shap(vslp, cof_v)
    wslpi = _shap(wslpi, cof_w)
    wslpj = _shap(wslpj, cof_w)
    return uslp, vslp, wslpi, wslpj


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
    aeiu_if = 0.5 * (aeiu + jnp.roll(aeiu, -1, ax_z))         # mk(aeiu) at iface below k
    wslpi_u = 0.5 * (wslpi_kp1 + jnp.roll(wslpi_kp1, -1, ax_x))  # mi(wslpi) -> u-face
    wslpj_v = 0.5 * (wslpj_kp1 + jnp.roll(wslpj_kp1, -1, ax_y))
    act_kp1 = act_below
    wumask_uw = (u_mask[:, 1:, jnp.newaxis] * act * jnp.roll(act, -1, ax_x)
                 * act_kp1 * jnp.roll(act_kp1, -1, ax_x))
    wvmask_vw = (v_mask[1:, :, jnp.newaxis] * act * jnp.roll(act, -1, ax_y)
                 * act_kp1 * jnp.roll(act_kp1, -1, ax_y))
    psi_uw = -(e2u[:, :, jnp.newaxis] * wslpi_u * aeiu_if * wumask_uw)
    psi_vw = -(e1v[:, :, jnp.newaxis] * wslpj_v * aeiu_if * wvmask_vw)
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


def nemo_iso_w_kappa_sums(aht, umask, vmask):
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
    split can never diverge (#1226).  Returns
    ``(ksum_u, cnt_u, ksum_v, cnt_v)``, all (n_lat, n_lon, nlev).
    """
    ax_y, ax_x, ax_z = 0, 1, 2
    up = lambda a: jnp.roll(a, +1, ax_z)     # level k-1 view
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
    ah_jm1 = jnp.roll(aht, +1, ax_y)
    A_v, B_v = aht * vmask, ah_jm1 * vm_jm1
    cnt_v = up(vmask) + vm_jm1 + up(vm_jm1) + vmask
    ksum_v = up(A_v) + B_v + up(B_v) + A_v
    return ksum_u, cnt_u, ksum_v, cnt_v


def nemo_iso_a33(aht, umask, vmask, wmask, wslpi, wslpj,
                 e1u_c, e2v_c, e3w2, dt=None, msc: bool = False):
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
    diverge (#1226).
    """
    ax_y, ax_x, ax_z = 0, 1, 2
    up = lambda a: jnp.roll(a, +1, ax_z)
    ksum_u, cnt_u, ksum_v, cnt_v = nemo_iso_w_kappa_sums(aht, umask, vmask)
    zahu_w = ksum_u * (wmask / jnp.maximum(cnt_u, 1.0))
    zahv_w = ksum_v * (wmask / jnp.maximum(cnt_v, 1.0))
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
    ahv = aht * vmask
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
    return_bolus: bool = False,
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
    active_3d : (n_lat, n_lon, nlev) or None
        Per-cell wet mask (1=water, 0=below seafloor).  Supplies NEMO's
        vertical ``tmask`` extent so the sub-seafloor dry level (which
        carries a garbage 0 tracer) cannot leak a spurious across-floor
        vertical gradient into the deepest wet cell.  ``None`` ⇒ assume
        every level of a wet column is water (full-depth flat bottom) —
        only correct when ``q`` has no below-bathymetry levels.

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
    ones_z = jnp.ones((1, 1, nlev), dtype=dtype)

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

    ax_y, ax_x, ax_z = 0, 1, 2

    # ---- masked tracer gradients (traldf_iso_scheme.h90 top block) ----
    zdit = (jnp.roll(q, -1, ax_x) - q) * umask       # (T(i+1)-T(i))·umask
    zdjt = (jnp.roll(q, -1, ax_y) - q) * vmask
    zdkt = (jnp.roll(q, +1, ax_z) - q) * wmask       # (T(k-1)-T(k))·wmask
    zdkt = zdkt.at[:, :, 0].set(0.0)                 # surface w-level = 0

    # ================= HORIZONTAL fluxes (A11 + A13, A22 + A23) =============
    zA11 = (e2u / e1u)[:, :, jnp.newaxis] * e3t
    zA22 = (e1v / e2v)[:, :, jnp.newaxis] * e3t
    # zmsku = 1/max(Σ4 wmask around the u-face vertical pair, 1)
    wm_ip1 = jnp.roll(wmask, -1, ax_x)
    wm_kp1 = jnp.roll(wmask, -1, ax_z)
    wm_ip1_kp1 = jnp.roll(wm_ip1, -1, ax_z)
    zmsku_h = 1.0 / jnp.maximum(wm_ip1 + wm_kp1 + wm_ip1_kp1 + wmask, 1.0)
    wm_jp1 = jnp.roll(wmask, -1, ax_y)
    wm_jp1_kp1 = jnp.roll(wm_jp1, -1, ax_z)
    zmskv_h = 1.0 / jnp.maximum(wm_jp1 + wm_kp1 + wm_jp1_kp1 + wmask, 1.0)

    zA13 = -e2u[:, :, jnp.newaxis] * uslp * zmsku_h
    zA23 = -e1v[:, :, jnp.newaxis] * vslp * zmskv_h

    # 4-pt vertical-gradient average around the u-face / v-face
    zdkt_kp1 = jnp.roll(zdkt, -1, ax_z)
    avg4_u = (jnp.roll(zdkt, -1, ax_x) + zdkt_kp1
              + jnp.roll(zdkt_kp1, -1, ax_x) + zdkt)
    avg4_v = (jnp.roll(zdkt, -1, ax_y) + zdkt_kp1
              + jnp.roll(zdkt_kp1, -1, ax_y) + zdkt)

    zfu = aht * (zA11 * zdit + zA13 * avg4_u)
    zfv = aht * (zA22 * zdjt + zA23 * avg4_v)

    # ================= VERTICAL flux zfw at w-level jk+1 (A31 + A32) ========
    # Shared a33 kappa sums (#1226): faces (k,k+1) here = the "above"
    # sums at k+1 (roll -1); the wmask factor stays AT k — NEMO's
    # scheme.h90:109 zmsku uses wmask(jk) with the (jk,jk+1) face pair
    # (transcription detail; NOT a pure shift of the a33 stencil).
    # NEMO masks aht at build (ldftra:365), so the shared masked-sum /
    # wet-count IS scheme.h90's (masked 4-sum)·zmsku — the previous
    # inline version summed UNMASKED kappa (4k/N at an N-wet-face wall
    # vs the K33's k: the residual split mismatch).
    _ksum_u, _cnt_u, _ksum_v, _cnt_v = nemo_iso_w_kappa_sums(
        aht, umask, vmask)
    zmsku_w = wmask / jnp.maximum(jnp.roll(_cnt_u, -1, ax_z), 1.0)
    zmskv_w = wmask / jnp.maximum(jnp.roll(_cnt_v, -1, ax_z), 1.0)
    zahu_w = jnp.roll(_ksum_u, -1, ax_z) * zmsku_w
    zahv_w = jnp.roll(_ksum_v, -1, ax_z) * zmskv_w

    wslpi_kp1 = jnp.roll(wslpi, -1, ax_z)            # wslpi(jk+1)
    wslpj_kp1 = jnp.roll(wslpj, -1, ax_z)
    zA31 = -zahu_w * e2t[:, :, jnp.newaxis] * zmsku_w * wslpi_kp1
    zA32 = -zahv_w * e1t[:, :, jnp.newaxis] * zmskv_w * wslpj_kp1

    # 4-pt horizontal-gradient average around the w-point (i, jk+1)
    zdit_kp1 = jnp.roll(zdit, -1, ax_z)
    zdjt_kp1 = jnp.roll(zdjt, -1, ax_z)
    avg4_wi = (zdit + jnp.roll(zdit_kp1, +1, ax_x)
               + jnp.roll(zdit, +1, ax_x) + zdit_kp1)
    avg4_wj = (zdjt + jnp.roll(zdjt_kp1, +1, ax_y)
               + jnp.roll(zdjt, +1, ax_y) + zdjt_kp1)
    zfw_kp1 = zA31 * avg4_wi + zA32 * avg4_wj        # flux at interface BELOW cell k
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
        # z*-scaled w-thickness.  APPROX: the T-thickness average, not NEMO's
        # analytic e3w_0·(1+r3t) (from gdepw) — a few-% difference on the stretched
        # grid that feeds the flux magnitude + the akz threshold (accepted; exact
        # fidelity would use the coordinate's e3w_0).
        e3w_ab = 0.5 * (jnp.roll(e3t, +1, ax_z) + e3t)
        e3w_ab = e3w_ab.at[:, :, 0].set(e3t[:, :, 0])   # surface w (unused: wslp(0)=0)
        _ahw_ab, _akz_ab = nemo_iso_a33(
            aht, umask, vmask, wmask, wslpi, wslpj,
            e1u, e2v, e3w_ab ** 2, dt=dt, msc=True)
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
        zfw_kp1 = zfw_kp1 + (e1t * e2t)[:, :, jnp.newaxis] / e3w_kp1 * (
            ah_wslp2 - akz) * zdkt_kp1
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
            kappa_GM, wslpi_kp1, wslpj_kp1, e2u, e1v,
            u_mask, v_mask, act, act_below, q.shape, dtype,
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
    hdiv = (zfu - jnp.roll(zfu, +1, ax_x)) + (zfv - jnp.roll(zfv, +1, ax_y))
    zfw_top = jnp.roll(zfw_kp1, +1, ax_z)            # flux at interface ABOVE cell k
    zfw_top = zfw_top.at[:, :, 0].set(0.0)           # surface flux = 0
    vdiv = zfw_top - zfw_kp1

    r1_e1e2t = 1.0 / (e1t * e2t)
    tend = (hdiv + vdiv) * r1_e1e2t[:, :, jnp.newaxis] / e3t
    # Mask by the 3-D cell wet mask (NEMO tmask), not just the 2-D surface mask,
    # so sub-seafloor dry levels of a wet column are zeroed too (byte-identical
    # on flat bottom, where those levels already carry zero divergence).
    if return_bolus:
        return tend * act, bolus_transport
    return tend * act


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
    mask: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
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
    """
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    jacobian = compute_ocean_jacobian(eta, H_bathy, z_coord)
    eos_fn = make_eos_fn(eos, eos_linear)
    fill_fn = lambda field: neumann_fill_cgrid(field, mask)
    rho, _rho_prime, _p_prime = iterate_eos_and_pressure_anomaly(
        T, S, mask, fill_fn, eos_fn, z_coord.dz_ref, rho_0, g, n_iter=2,
    )
    return rho, jacobian


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
    mask: jnp.ndarray | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    f_coriolis: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    kappa_gm_override: jnp.ndarray | None = None,
    kappa_redi_override: jnp.ndarray | None = None,
    density_jacobian: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    dt: float | None = None,
    return_bolus_transport: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Top-level GM/Redi for lat-lon C-grid.

    Computes density, isopycnal slopes, optional Visbeck coefficient,
    then returns tracer tendencies for T and S.

    Parameters
    ----------
    T, S : (n_lat, n_lon, nlev)
    eta : (n_lat, n_lon)
        Sea-surface height.
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

    Returns
    -------
    dT_dt, dS_dt : (n_lat, n_lon, nlev)
    """
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
    eos_fn = make_eos_fn(eos, eos_linear)
    if density_jacobian is None:
        rho, jacobian = gm_redi_density_and_jacobian(
            T, S, eta, H_bathy, grid, z_coord,
            eos=eos, eos_linear=eos_linear, mask=mask, rho_0=rho_0, g=g,
        )
    else:
        rho, jacobian = density_jacobian

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
    if kappa_gm_override is not None:
        kappa_GM = kappa_gm_override
    elif _treg is not None and _treg.enabled:
        # Treguier-1997 / NEMO nn_aei_ijk_t=21 adaptive κ (the oracle scaling).
        if f_coriolis is None:
            f_coriolis = jnp.broadcast_to(grid.f, mask.shape)
        kappa_GM = compute_treguier_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, _treg,
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
                rho, eta, z_coord.dz_ref, jacobian, rho_0, h_actual=_h_actual,
            )
            _T_vb, _S_vb, _eos_vb = T, S, eos_fn
        kappa_GM = compute_visbeck_kappa_gm(
            rho, S_x, S_y, z_coord, jacobian, f_coriolis, cfg.visbeck,
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

    scheme = getattr(cfg, "slope_scheme", "triads")
    if return_bolus_transport and scheme != "nemo_iso_lap":
        raise ValueError(
            "gm_redi_tracer_tendency_latlon(return_bolus_transport=True) is only "
            f"supported by slope_scheme='nemo_iso_lap', got {scheme!r}.")
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
        # (2-D mask) AND its TOP-interface reference depth is above the
        # bathymetry (cell has some water). Supplies the vertical bottom extent
        # so the sub-seafloor dry level (garbage 0 tracer) cannot leak an
        # across-floor vertical gradient. Top-interface test (vs bottom) leaves
        # a full-dz margin, so it is robust to cumsum roundoff.
        _z_top = jnp.cumsum(z_coord.dz_ref) - z_coord.dz_ref   # (nlev,) top-iface depth
        _active_3d = (
            (mask[:, :, jnp.newaxis] > 0.5)
            & (_z_top[jnp.newaxis, jnp.newaxis, :]
               < H_bathy[:, :, jnp.newaxis])
        ).astype(T.dtype)
        _positions = getattr(cfg, "slope_positions", "mode_b")
        if _positions not in ("mode_b", "nemo_native"):
            raise ValueError(
                "Unknown GMRediConfig.slope_positions scheme: must be one of "
                f"('mode_b', 'nemo_native'), got {_positions!r}")
        # Guards (codex r5/r6, hoisted to scheme entry so EVERY sub-branch is
        # covered): MSC's capped akz goes to the IMPLICIT solve — with
        # implicit_K33=False nobody applies it and the akz portion of the
        # diagonal silently vanishes (unstable AND unfaithful); and MSC is
        # only implemented on the native-slope stencil — with mode_b the
        # flag would be silently ignored.
        if getattr(cfg, "msc_stabilize", False):
            if not cfg.implicit_K33:
                raise ValueError(
                    "GMRediConfig: msc_stabilize=True (ln_traldf_msc) "
                    "requires implicit_K33=True — the capped akz must be "
                    "applied by the implicit vertical solve; without it the "
                    "akz part of the a33 diagonal is silently dropped.")
            if _positions != "nemo_native":
                raise ValueError(
                    "GMRediConfig: msc_stabilize=True (ln_traldf_msc) "
                    "requires slope_positions='nemo_native' — the MSC split "
                    "is implemented on the native ldfslp stencil only; with "
                    f"slope_positions={_positions!r} the flag would be "
                    "silently ignored.")
        if _positions == "nemo_native":
            # ldfslp native four-position slopes: NEMO sign convention and
            # NEMO's own limiters (double cap, ML ramp, Shapiro) built in —
            # NO dispatch negation, exact traldf_iso stencil (amplitude 1.0).
            _nat = compute_nemo_native_slopes(
                rho, T, S, mask, u_mask, v_mask, z_coord, grid, cfg,
                eos_fn, rho_0=rho_0, g=g, active_3d=_active_3d)
            _msc = getattr(cfg, "msc_stabilize", False)
            _bolus = None
            _dT = nemo_iso_lap_tracer_tendency_latlon_cgrid(
                T, S_x, S_y, mask, u_mask, v_mask,
                z_coord, jacobian, grid, kappa_Redi_eff, _active_3d,
                native_slopes=_nat, msc_stabilize=_msc, dt=dt,
                kappa_GM=kappa_GM, gm_bolus_advection=_gm_bolus,
                return_bolus=return_bolus_transport)
            if return_bolus_transport:
                dT_dt, _bolus = _dT
            else:
                dT_dt = _dT
            dS_dt = nemo_iso_lap_tracer_tendency_latlon_cgrid(
                S, S_x, S_y, mask, u_mask, v_mask,
                z_coord, jacobian, grid, kappa_Redi_eff, _active_3d,
                native_slopes=_nat, msc_stabilize=_msc, dt=dt,
                kappa_GM=kappa_GM, gm_bolus_advection=_gm_bolus)
            if return_bolus_transport:
                return dT_dt, dS_dt, _bolus
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
            return_bolus=return_bolus_transport,
        )
        if return_bolus_transport:
            dT_dt, _bolus = _dT
        else:
            dT_dt = _dT
        dS_dt = nemo_iso_lap_tracer_tendency_latlon_cgrid(
            S, -S_x, -S_y, mask, u_mask, v_mask,
            z_coord, jacobian, grid, kappa_Redi_eff, _active_3d,
            kappa_GM=kappa_GM, gm_bolus_advection=_gm_bolus,
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
    mask: jnp.ndarray | None = None,
    rho_0: float = _RHO_0,
    g: float = constants.g,
    kappa_redi_override: jnp.ndarray | None = None,
    density_jacobian: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    u_mask: jnp.ndarray | None = None,
    v_mask: jnp.ndarray | None = None,
    dt: float | None = None,
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
    if mask is None:
        mask = jnp.ones(T.shape[:2], dtype=T.dtype)
    # Jacobian + density (2-iteration EOS coupling).  When the model step
    # already computed these for the GM/Redi tracer tendency from the SAME
    # (T,S,eta,H_bathy) (implicit_K33 path), reuse them via density_jacobian
    # so the expensive 3-D EOS coupling is not run twice; None => compute
    # inline, bit-identical (scaling review lever #3).
    eos_fn = make_eos_fn(eos, eos_linear)
    if density_jacobian is None:
        rho, jacobian = gm_redi_density_and_jacobian(
            T, S, eta, H_bathy, grid, z_coord,
            eos=eos, eos_linear=eos_linear, mask=mask, rho_0=rho_0, g=g,
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
        _eosfn = _mk(eos, eos_linear)
        if density_jacobian is not None:
            _rho, _J = density_jacobian
        else:
            _rho, _J = gm_redi_density_and_jacobian(
                T, S, eta, H_bathy, grid, z_coord,
                eos=eos, eos_linear=eos_linear, mask=mask, rho_0=rho_0, g=g)
        _m = mask if mask is not None else jnp.ones(T.shape[:2], T.dtype)
        # 3-D wet mask (NEMO tmask) — SAME construction as the tendency
        # dispatcher's nemo_iso_lap branch (top-interface depth vs bathymetry).
        _z_top = jnp.cumsum(z_coord.dz_ref) - z_coord.dz_ref
        _act = (
            (_m[:, :, jnp.newaxis] > 0.5)
            & (_z_top[jnp.newaxis, jnp.newaxis, :]
               < H_bathy[:, :, jnp.newaxis])
        ).astype(T.dtype)
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
        _, _, _wi, _wj = compute_nemo_native_slopes(
            _rho, T, S, _m, _um, _vm, z_coord, grid, cfg, _eosfn,
            rho_0=rho_0, g=g, active_3d=_act)
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
        _e3w = 0.5 * (jnp.roll(_e3t, +1, 2) + _e3t)
        _e3w = _e3w.at[:, :, 0].set(_e3t[:, :, 0])
        _msc = bool(getattr(cfg, "msc_stabilize", False))
        _, _akz = nemo_iso_a33(
            _aht, _um3, _vm3, _wm3, _wi, _wj,
            _e1u_c, _e2v_c, _e3w ** 2, dt=dt, msc=_msc)
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
            rho, eta, z_coord.dz_ref, jacobian, rho_0, h_actual=h_actual,
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
        cfg.visbeck, cfg.eke, rho_ref=rho_0, beta=beta,
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
