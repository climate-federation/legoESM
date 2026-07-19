"""Compute vertical mixing K_v / A_v profiles for the implicit solver.

The vertical-mixing and convection schemes already compute K_v / A_v at
cell-center interfaces alongside their explicit tendencies.  When the
host model runs in *implicit* vertical-mixing mode the explicit
tendencies are suppressed (``apply_diffusion=False`` on each scheme)
and the dynamics step needs the K_v / A_v profiles separately so it can
apply them via :func:`implicit_vertical_diffusion_ocean`.

This module provides a thin helper that re-runs only the K-computation
parts of each scheme (the costly part of an implicit step is the
tridiagonal solve, not the K computation, so the duplication is small)
and returns the combined ``(K_v_total, A_v_total)`` at interfaces.

The combination rule across schemes is *sum* (each scheme's
contribution is taken on top of the others), bounded above by
``KPPConfig.K_max`` when KPP is active.  Background floors from each
scheme's config are already included in their respective ``K_v`` /
``A_v`` output, so a separate ``A_v_floor`` is added by the caller only
to enforce ``LatLonCGridOceanConfig.A_v`` and ``K_v`` as additional
floors.  Exception: with the ``constant`` scheme in ``lat_dependent``
mode the Gregg (2003) latitude background REPLACES the constant
background, so those caller floors are suppressed (not double-added).
"""

from __future__ import annotations


import jax.numpy as jnp

from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.eos import (
    compute_ocean_rho as _compute_rho,
)
from legoesm.ocean.vertical import (
    OceanZStarCoordinate, compute_ocean_jacobian,
)
from legoesm.ocean.physics.convection.config import OceanConvectionConfig
from legoesm.ocean.physics.vertical_mixing._shared import (
    surface_buoyancy_flux,
    compute_N2,
    latitude_background_diffusivity,
)

# Floor on the constant diffusivity K_v when deriving the background Prandtl
# ratio A_v/K_v for the latitude-dependent viscosity (avoids /0 if K_v -> 0).
_KV_PRANDTL_FLOOR = 1e-30

__physics_contract__ = {
    "summary": (
        "Assemble the total vertical (K_v, A_v) at interior interfaces for the "
        "implicit solver by SUMMING the active closures (vertical-mixing scheme "
        "+ convection + internal-wave mixing), capped at KPP K_max; no tendency "
        "is applied here."
    ),
    "inputs": {
        "state.u": "m/s", "state.v": "m/s", "state.T": "degC", "state.S": "psu",
        "surface_forcing.tau_x": "N/m^2", "surface_forcing.q_net": "W/m^2",
    },
    "outputs": {
        "K_v": "m^2/s", "A_v": "m^2/s", "tke_new": "m^2/s^2",
    },
    "sign_convention": (
        "K_v, A_v >= 0 at interior interfaces; combined by SUM across the active "
        "closures then min-capped at KPP K_max (when KPP is active); zeroed at "
        "non-wet (sub-seafloor) interfaces; z positive up. No flux is applied "
        "here — the implicit solver applies the diffusion and closes the budget; "
        "an unknown scheme raises ValueError."
    ),
    # Pure diffusivity/viscosity producer: nothing conserved here; the budget
    # closes in the implicit diffusion solver.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Composition of Large-McWilliams-Doney (1994) KPP, Pacanowski-Philander "
        "(1981), Gaspar (1990)/Burchard (2002) TKE and de Lavergne et al. (2020) "
        "internal-wave mixing; budget closed by the implicit solver"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_vmix_k_profiles_direct.py — the summed profile "
        "equals the explicit schemes' K_v/A_v; K/A zeroed at dry interfaces; "
        "an unknown vertical_mixing scheme raises ValueError."
    ),
}


def compute_vertical_K_profiles(
    state,
    z_coord: "OceanZStarCoordinate",
    surface_forcing,
    physics_config,
    A_v_background: float = 0.0,
    K_v_background: float = 0.0,
    eos_fn=None,
    *,
    tke_old=None,
    dt_tke: float | None = None,
    tke_source=None,
    return_tke: bool = False,
    lat_deg=None,
    iwm_fields=None,
    n2_tracers=None,
) -> (
    tuple[jnp.ndarray, jnp.ndarray]
    | tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
):
    """Compute total ``(K_v, A_v)`` at interior interfaces for an implicit solve.

    Parameters
    ----------
    state
        Ocean state with at least ``u, v, T, S, eta, H_bathy``.
    z_coord
        Vertical coordinate.
    surface_forcing
        ``OceanSurfaceForcing`` (or None).  Required for KPP boundary
        layer diagnosis.
    physics_config
        ``OceanPhysicsConfig`` controlling which schemes contribute.
    A_v_background, K_v_background
        Optional additional floors added uniformly to all interfaces
        (typically ``LatLonCGridOceanConfig.A_v`` / ``K_v``).  IGNORED
        (treated as zero) when the ``constant`` scheme runs with
        ``lat_dependent=True``: the Gregg (2003) latitude background
        REPLACES the constant background, so adding the model-level floor
        on top would shift the documented range ``[K_bg_eq, K_bg_pole]``
        and break the configured Prandtl ratio ``A_v/K_v``.
    eos_fn
        Optional EOS ``fn(T, S, p) -> rho`` (e.g. the recipe's
        ``veros_nonlin2``). When None, the schemes' density (and the TKE
        static-stability N²) default to Wright 1997 — bit-identical with
        the historical behaviour. Passing the model's EOS makes the TKE
        N² (and, for ``n2_mode="adiabatic"``, the convective trigger)
        consistent with the dynamical core.
    tke_old, dt_tke, tke_source, return_tke
        PROGNOSTIC TKE carry (Veros enable_tke prognostic form). Consulted
        ONLY when the vmix scheme is ``"tke"`` AND
        ``vertical_mixing.tke.prognostic=True``. ``tke_old`` is the carried
        TKE at the interior interfaces ``(..., nlev-1)`` (or ``None`` for the
        first step / cold start); ``dt_tke`` is the TKE step ``dt`` (= the
        MOMENTUM timestep ``dt_mom``, Veros tke.py:137); ``tke_source`` is an
        optional additive energy-recycling source ``forc`` [m²/s³] at the
        interior interfaces (``eke_diss_iw`` + ``K_diss_bot``, Veros
        integrate_tke forc). ``return_tke=True`` makes this function return the
        3-tuple ``(K_v, A_v, tke_new)`` so the caller can carry ``tke_new``
        back onto the state; otherwise the 2-tuple ``(K_v, A_v)`` is returned
        (BIT-IDENTICAL legacy contract).

    Returns
    -------
    (K_v, A_v) : tuple of arrays
        Each of shape ``state.T.data.shape[:-1] + (nlev - 1,)``, on
        interior interfaces, in m²/s.
    (K_v, A_v, tke_new) : when ``return_tke=True`` — ``tke_new`` is the updated
        prognostic TKE field ``(..., nlev-1)`` when the active scheme is the
        prognostic TKE closure, else ``None``. Under
        ``TKEConfig.buoyancy_timing="post_mixing_veros"`` the third slot is
        instead a :class:`...tke.TKEPostMixingContext` (phase-1 kappa from the
        carried TKE; the model step advances the TKE AFTER the tracer solve).
    """
    T = state.T.data
    nlev = T.shape[-1]
    interface_shape = T.shape[:-1] + (nlev - 1,)
    dtype = T.dtype

    # ---- Variable-bathymetry (partial-cell) dry-cell guard ----
    # Same pattern as the MPAS vmix bridge (mpas_integration.py:238-296):
    # extend the deepest ACTIVE T/S/u/v downward so the schemes' N²/shear see
    # a neutral, quiescent sub-seafloor instead of the T=S=0 IC fill (which
    # reads as a huge fake instability at the seafloor interface → spurious
    # convective K_M mixing the bottom wet cell with rock), and zero the
    # returned K/A (and prognostic TKE) at NON-WET interfaces — no flux
    # through the seafloor, Veros's maskW semantics.  No-op (bit-identical)
    # for pure z-star coords (no ``is_active``), e.g. the flat-bottom ACC.
    _is_active = getattr(z_coord, "is_active", None)
    if _is_active is not None:
        from legoesm.ocean.vertical import extrapolate_below_seafloor
        _T_f = extrapolate_below_seafloor(state.T.data, z_coord)
        _S_f = extrapolate_below_seafloor(state.S.data, z_coord)
        state = state._replace(T=state.T.replace(data=_T_f),
                               S=state.S.replace(data=_S_f))
        # The before-advection N² tracers (TKEConfig.n2_before_advection)
        # get the SAME sub-seafloor extrapolation so the deep interface sees
        # a neutral fill, not the T=S=0 rock IC (partial cells). Python-static
        # (n2_tracers is None ⇒ untouched ⇒ BIT-IDENTICAL flat-bottom no-op).
        if n2_tracers is not None:
            n2_tracers = (
                extrapolate_below_seafloor(n2_tracers[0], z_coord),
                extrapolate_below_seafloor(n2_tracers[1], z_coord),
            )
        # u/v only when already cell-centred (the lat-lon model passes the
        # centred cc_state; staggered shapes have no cell is_active match).
        if state.u.data.shape[:-1] == state.T.data.shape[:-1]:
            _u_f = extrapolate_below_seafloor(state.u.data, z_coord)
            _v_f = extrapolate_below_seafloor(state.v.data, z_coord)
            state = state._replace(u=state.u.replace(data=_u_f),
                                   v=state.v.replace(data=_v_f))
        # Interface k sits between cells k and k+1: wet iff cell k+1 active.
        _wet_if = jnp.asarray(_is_active, dtype=dtype)[..., 1:]
    else:
        _wet_if = None

    # Start with the configured background floors.  These are scalar
    # floats; broadcast to interface shape.
    #
    # EXCEPTION (MED-2, codex batch2): when the ``constant`` scheme runs with
    # ``lat_dependent=True`` the Gregg (2003) latitude background REPLACES the
    # constant background ENTIRELY — the scheme branch already substitutes
    # ``cfg.K_v``/``cfg.A_v``, and the model-level caller floors
    # (``LatLonCGridOceanConfig.K_v``/``A_v``) must be suppressed here too.
    # Adding them on top would (a) shift the documented final range
    # ``[K_bg_eq, K_bg_pole]`` to ``[K_bg_eq + K_v_background, K_bg_pole +
    # K_v_background]`` (defaults: [1.1e-4, 2e-4] instead of [1e-5, 1e-4])
    # and (b) break the configured Prandtl ratio ``A_v/K_v`` whenever the
    # caller's fallback ratio differs.  Static config bool -> Python gate
    # (feature-gating doctrine, not jnp.where); every other scheme and
    # ``lat_dependent=False`` keep the additive floors BIT-IDENTICALLY.
    vmix = physics_config.vertical_mixing
    if (vmix.scheme == "constant"
            and getattr(vmix.constant, "lat_dependent", False)):
        K_v_background = 0.0
        A_v_background = 0.0
    K_v_total = jnp.full(interface_shape, K_v_background, dtype=dtype)
    A_v_total = jnp.full(interface_shape, A_v_background, dtype=dtype)

    tke_new = None
    if vmix.scheme != "none":
        K_vmix, A_vmix, tke_new = _vmix_K_profiles(
            state, z_coord, surface_forcing, vmix, physics_config.constants,
            eos_fn=eos_fn,
            tke_old=tke_old, dt_tke=dt_tke, tke_source=tke_source,
            lat_deg=lat_deg, n2_tracers=n2_tracers)
        K_v_total = K_v_total + K_vmix
        A_v_total = A_v_total + A_vmix

    conv = physics_config.convection
    if conv.scheme == "enhanced_diffusion":
        # Fail closed: nu_conv/nu_bg cannot be honoured under KPP (the A_conv
        # momentum term is gated off below to avoid double-counting KPP's
        # own interior convective viscosity).  Reject rather than silently
        # ignore — mirrors the guard in combined.make_ocean_physics.
        _ed = conv.enhanced_diffusion
        if vmix.scheme == "kpp" and (_ed.nu_conv != 0.0 or _ed.nu_bg != 0.0):
            raise ValueError(
                "EnhancedDiffusionConfig convective momentum viscosity "
                "(nu_conv/nu_bg) cannot be combined with KPP vertical mixing: "
                "KPP already enhances interior momentum where N²<0, so "
                "applying nu_* on top would double-count and is suppressed. "
                "Set EnhancedDiffusionConfig(nu_conv=0.0, nu_bg=0.0) to let "
                "KPP own convective momentum, or choose a non-KPP "
                "vertical_mixing scheme."
            )
        K_conv, A_conv = _enhanced_diffusion_K(state, z_coord, conv,
                                               eos_fn=eos_fn,
                                               before_tracers=n2_tracers)
        # Convection enhances tracer diffusivity (convective_κz).
        K_v_total = K_v_total + K_conv
        # Momentum gets the independent convective viscosity (convective_νz
        # = ``nu_conv``).  When KPP is on, the KPP interior already enhances
        # momentum for the same N²<0 instability (A_interior includes its
        # own K_conv), so adding here would double-count — gate it off.
        # When KPP is off, apply A_conv so the explicit/implicit equivalence
        # holds for the constant + convection composition.
        if vmix.scheme != "kpp":
            A_v_total = A_v_total + A_conv

    # Clip to KPP K_max when KPP is the vertical mixing scheme, matching
    # the explicit path's saturation behavior.  Otherwise leave the sum
    # uncapped (the schemes' own configs already include sensible
    # backgrounds, and any caller-supplied floor is small).
    if vmix.scheme == "kpp":
        K_max = vmix.kpp.K_max
        K_v_total = jnp.minimum(K_v_total, K_max)
        A_v_total = jnp.minimum(A_v_total, K_max)

    # Internal wave-driven mixing (NEMO zdfiwm, de Lavergne 2020) —
    # ADDITIVE on top of the closure, AFTER the KPP saturation clip so the
    # wave contribution keeps its own [k_min, k_max] bounds (NEMO's
    # zdfphy order: the closure runs first, zdf_iwm then ADDS onto
    # avt/avs/avm with no combined cap).  Contributes to BOTH tracer
    # diffusivity and momentum viscosity.  The wet-interface mask below
    # zeroes it at the seafloor (NEMO's wmask factor).
    iwm_cfg = getattr(vmix, "iwm", None)
    if iwm_cfg is not None and iwm_cfg.enabled:
        if iwm_cfg.tsdiff:
            # avs = avt * ratio needs a SEPARATE salinity diffusivity
            # channel through the implicit tracer solve; the lat-lon
            # solve shares one K between T and S.  Fail loud rather than
            # silently ignoring the requested differential mixing.  (The
            # ORCA1 oracle runs ln_tsdiff = .false., so the faithful
            # comparison path is unaffected.)
            raise ValueError(
                "IWMConfig.tsdiff=True (differential T/S wave-driven "
                "mixing) is not supported on the shared-K implicit tracer "
                "solve; set tsdiff=False (the ORCA1 oracle value).")
        K_iwm = iwm_K_profile(
            state, z_coord, physics_config, iwm_cfg,
            eos_fn=eos_fn, iwm_fields=iwm_fields)
        K_v_total = K_v_total + K_iwm
        A_v_total = A_v_total + K_iwm

    # Zero K/A (and the prognostic TKE) at non-wet interfaces (partial-cell
    # coords only; see the dry-cell guard above).  The implicit solve then
    # has no flux through the seafloor — wet cells can never exchange with
    # rock cells regardless of what the schemes produced below the bottom.
    if _wet_if is not None:
        K_v_total = K_v_total * _wet_if
        A_v_total = A_v_total * _wet_if
        if tke_new is not None and isinstance(tke_new, jnp.ndarray):
            # Post-mixing TKE returns a TKEPostMixingContext in this slot
            # (phase 1; no tke array to mask yet) — the model step masks the
            # post-solve tke_new with the same wet-interface guard.
            tke_new = tke_new * _wet_if

    if return_tke:
        return K_v_total, A_v_total, tke_new
    return K_v_total, A_v_total


# ---------------------------------------------------------------------------
# Per-scheme K computation helpers (interior interface shape).
# ---------------------------------------------------------------------------


def _surface_buoyancy_flux(surface_forcing, state, constants_config,
                           eos_fn=None):
    """Surface buoyancy flux ``B_f`` [m^2/s^3] (>0 destabilising) + the
    kinematic surface T / S fluxes, from the surface heat (``q_net``) +
    freshwater / salt forcing and the EOS thermal-expansion / haline-contraction
    coefficients.

    ``eos_fn`` (``None`` ⇒ Wright, bit-identical) sets the surface α/β so a
    non-Wright EOS (e.g. ``nemo_seos``) drives the boundary-layer buoyancy
    forcing consistently with the interior ρ/N² used by KPP/CATKE.

    Shared by the KPP boundary-layer diagnosis and the CATKE convective length
    so the surface buoyancy forcing lives in ONE place (same sign convention:
    surface cooling / brine rejection -> ``B_f > 0`` -> convection).

    Returns ``(B_f, Q_sfc_T, Q_sfc_S)``; each may be ``None`` when its forcing
    channel is absent (``B_f`` is ``None`` only when BOTH heat and
    freshwater/salt are absent).
    """
    sf = surface_forcing
    q_net = getattr(sf, "q_net", None) if sf else None
    fw = getattr(sf, "freshwater", None) if sf else None
    salt = getattr(sf, "salt_flux", None) if sf else None
    # Grid-agnostic kernel (#518 item 1).  Lat-lon convention: the real
    # salt-mass flux feeds BOTH the surface buoyancy and the non-local
    # Q_sfc_S (``real_salt_in_qs=True``).
    return surface_buoyancy_flux(
        q_net, fw, salt,
        state.T.data[..., 0], state.S.data[..., 0],
        g=constants_config.g,
        rho_0=constants_config.rho_0,
        c_sw=constants_config.c_sw,
        real_salt_in_qs=True,
        eos_fn=eos_fn,
    )


def _vmix_K_profiles(state, z_coord, surface_forcing, vmix_cfg,
                     constants_config=ConstantsConfig(), eos_fn=None,
                     *, tke_old=None, dt_tke=None, tke_source=None,
                     lat_deg=None, n2_tracers=None):
    """Re-compute K_v, A_v at interfaces for the chosen vmix scheme.

    For ``constant`` / ``richardson`` this duplicates only the K
    computation (cheap).  For ``kpp`` this also re-runs the boundary
    layer diagnosis, which is somewhat more expensive but still much
    cheaper than the tridiagonal solve it enables.

    ``eos_fn`` (optional) overrides the density EOS used to compute N²
    (default Wright 1997 -> bit-identical legacy).

    ``tke_old`` / ``dt_tke`` / ``tke_source`` drive the PROGNOSTIC TKE carry
    (``tke`` scheme + ``prognostic=True``); see
    :func:`compute_vertical_K_profiles`.

    Returns ``(K_v, A_v, tke_new)`` — ``tke_new`` is the updated prognostic TKE
    field for the prognostic ``tke`` scheme, else ``None``.
    """
    scheme = vmix_cfg.scheme
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)

    if scheme == "constant":
        cfg = vmix_cfg.constant
        nlev = state.T.data.shape[-1]
        shape = state.T.data.shape[:-1] + (nlev - 1,)
        dtype = state.T.data.dtype
        if getattr(cfg, "lat_dependent", False):
            # Latitude-dependent internal-wave background (Gregg 2003 / CVMix
            # bkgnd): REPLACE the spatially-constant K_v/A_v floor with the
            # latitude/stratification-scaled field.  Needs the column latitude
            # and N^2, so compute rho + N^2 here (the constant branch otherwise
            # skips the EOS).  K_v, A_v >= 0; z positive up.
            if lat_deg is None:
                raise ValueError(
                    "ConstantVerticalMixingConfig.lat_dependent=True requires "
                    "lat_deg (column latitudes in degrees) to be threaded to "
                    "compute_vertical_K_profiles; got None.")
            rho = _compute_rho(state, z_coord, J, eos_fn=eos_fn)
            dz_half = z_coord.dz_half_ref * J[..., jnp.newaxis]
            N2 = compute_N2(
                rho, dz_half, constants_config.rho_0,
                g=constants_config.g, n2_mode="insitu")
            K_v = latitude_background_diffusivity(lat_deg, N2, cfg)
            # Momentum viscosity carries the SAME latitude scaling, preserving
            # the configured background Prandtl ratio A_v/K_v (trace-safe floor
            # on K_v so the ratio is finite even if K_v -> 0).
            prandtl = cfg.A_v / jnp.maximum(
                jnp.asarray(cfg.K_v, dtype), _KV_PRANDTL_FLOOR)
            A_v = K_v * prandtl
            return K_v, A_v, None
        K_v = jnp.full(shape, cfg.K_v, dtype=dtype)
        A_v = jnp.full(shape, cfg.A_v, dtype=dtype)
        return K_v, A_v, None

    rho = _compute_rho(state, z_coord, J, eos_fn=eos_fn)

    if scheme == "richardson":
        from legoesm.ocean.physics.vertical_mixing.richardson import (
            richardson_vertical_mixing,
        )
        # Adiabatic PP81 N² (Veros parcel displacement) needs the cell-centre
        # hydrostatic pressure + the same EOS as the dynamical core. Only
        # computed when the config opts in (n2_mode="adiabatic") so the default
        # in-situ path is unchanged (mirrors the tke branch below).
        rich_p_cell = None
        if getattr(vmix_cfg.richardson, "n2_mode", "insitu") == "adiabatic":
            from legoesm.ocean.eos import (
                compute_hydrostatic_pressure, maybe_partial_h_actual,
            )
            rich_h_actual = maybe_partial_h_actual(state, z_coord)
            rich_p_cell = compute_hydrostatic_pressure(
                rho, state.eta.data, z_coord.dz_ref, J,
                constants_config.rho_0, h_actual=rich_h_actual,
            )
        out = richardson_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, vmix_cfg.richardson,
            apply_diffusion=False,
            p_cell=rich_p_cell, eos_fn=eos_fn,
        )
        return out.K_v, out.A_v, None

    if scheme == "tke":
        from legoesm.ocean.physics.vertical_mixing.tke import (
            tke_vertical_mixing,
        )
        # Interpolate u, v to cell centres for the closure on C-grid;
        # on cubed-sphere they are already at cell centres.
        u_data = state.u.data
        v_data = state.v.data
        T_data = state.T.data
        S_data = state.S.data
        if u_data.shape[1] != T_data.shape[1]:
            u_data = 0.5 * (u_data[:, :-1, :] + u_data[:, 1:, :])
            v_data = 0.5 * (v_data[:-1, :, :] + v_data[1:, :, :])
        # Before-advection (Nnow) T/S for the diffusivity-stage N²
        # (TKEConfig.n2_before_advection). None ⇒ the closure uses the
        # post-advection T_data/S_data ⇒ BIT-IDENTICAL.
        T_n2, S_n2 = (n2_tracers if n2_tracers is not None else (None, None))
        dz_half = jnp.broadcast_to(
            z_coord.dz_half_ref * J[..., jnp.newaxis],
            T_data.shape[:-1] + (z_coord.n_levels - 1,),
        )
        tau_x = (getattr(surface_forcing, "tau_x", None)
                 if surface_forcing is not None else None)
        tau_y = (getattr(surface_forcing, "tau_y", None)
                 if surface_forcing is not None else None)
        # NEMO taum channel: stress-modulus override for the TKE surface
        # input (None -> computed from the components inside tke.py).
        taum_sf = (getattr(surface_forcing, "taum", None)
                   if surface_forcing is not None else None)
        # Adiabatic static-stability N² (Veros parcel displacement) needs
        # the cell-centre hydrostatic pressure + the same EOS as the
        # dynamical core. Only computed when the TKE config opts in
        # (``n2_mode="adiabatic"``) so the default path is unchanged.
        p_cell = None
        if getattr(vmix_cfg.tke, "n2_mode", "insitu") == "adiabatic":
            from legoesm.ocean.eos import compute_hydrostatic_pressure
            from legoesm.ocean.eos import maybe_partial_h_actual
            h_actual = maybe_partial_h_actual(state, z_coord)
            p_cell = compute_hydrostatic_pressure(
                rho, state.eta.data, z_coord.dz_ref, J,
                constants_config.rho_0, h_actual=h_actual,
            )
        # NEMO bn2 trigger (n2_mode="nemo_bn2"): the geometric depth ladders
        # (gdept / interior gdepw); ignored by every other n2_mode.
        _bn2_t_depth = _bn2_w_depth = None
        if getattr(vmix_cfg.tke, "n2_mode", "insitu") == "nemo_bn2":
            from legoesm.ocean.eos import nemo_bn2_depth_ladders
            _bn2_t_depth, _bn2_w_depth = nemo_bn2_depth_ladders(z_coord)
        tke_cfg = vmix_cfg.tke
        prognostic = bool(getattr(tke_cfg, "prognostic", False))
        # Veros metric slots (TKEConfig.veros_dz_slots): the surface-flux
        # injection volume is Veros's surface W half-volume 0.5·dzw_top
        # (tke.py:225) = the distance from z=0 down to the top cell centre,
        # scaled by the z-star Jacobian like every other thickness. On a
        # Veros u_centered coordinate -z_full_ref[0] IS 0.5·dzw_top exactly
        # (dzw_top = 2·dzt_top - dzw[-2] = -2·zt_top, numerics.py:21).
        dz_surface = None
        if getattr(tke_cfg, "veros_dz_slots", False):
            dz_surface = (-z_coord.z_full_ref[0]) * J
        # Veros tke_mxl_choice=1 distance-to-boundary cap (tke.py:43-47):
        # the buoyancy mixing length may not exceed the distance to the
        # surface / seafloor. Computed once from the STATIC reference geometry
        # (interior interface heights + centre spacing) and the per-column
        # ocean depth (H_bathy = Veros ``ht``). Only choice=1 needs it;
        # choice=2 is bounded by the MITgcm/OPA recursion in
        # _veros_buoyancy_length. Without it the choice=1 length overflows to
        # +inf where N²→0 at a convecting surface (the global_1deg blowup).
        # Shared by the prognostic (post- and pre-mixing) and the Mode-B
        # diagnostic paths below.
        _mxl1_cap = None
        if getattr(tke_cfg, "tke_mxl_choice", 2) == 1:
            from legoesm.ocean.physics.vertical_mixing.tke import (
                veros_mxl_choice1_boundary_cap,
            )
            _mxl1_cap = veros_mxl_choice1_boundary_cap(
                z_coord.z_half_ref[1:-1], z_coord.dz_half_ref,
                state.H_bathy.data,
            )
        # Under-ice attenuation of the wave-driven TKE sources (NEMO nn_eice;
        # ``TKEConfig.eice``).  The lc/etau kernels apply ``(1 - ice_frac)``
        # internally, so the mode maps onto an EFFECTIVE ice fraction:
        #   0 (default, bit-identical): no attenuation — ice_frac stays None;
        #   1: eff = fi              -> kernel factor (1-fi)        (nn_eice=1);
        #   3: eff = min(4*fi, 1)    -> kernel factor max(0,1-4*fi) (nn_eice=3,
        #      the ORCA1 namelist choice — wave TKE fully killed at fi>=0.25).
        # Unknown values raise (dispatch hardening; static config value).
        _eice = int(getattr(tke_cfg, "eice", 0))
        if _eice not in (0, 1, 3):
            raise ValueError(
                f"Unknown TKEConfig.eice={_eice!r}; expected 0 (no under-ice "
                "attenuation), 1 ((1-fi)) or 3 (max(0,1-4*fi), NEMO nn_eice=3) "
                "on the lc/etau TKE sources.")
        _tke_ice_fr = None
        if _eice != 0 and surface_forcing is not None:
            _fi = getattr(surface_forcing, "ice_concentration", None)
            if _fi is not None:
                _tke_ice_fr = (_fi if _eice == 1
                               else jnp.minimum(4.0 * _fi, 1.0))
        if prognostic:
            # PROGNOSTIC mode (Veros enable_tke): ONE backward-Euler step per
            # model step, seeded from the carried ``tke_old``, with dt = the
            # MOMENTUM timestep ``dt_tke`` (= dt_mom; Veros tke.py:137 dt_tke =
            # dt_mom even though TKE advances once per tracer step). The updated
            # TKE is returned to the caller (``tke_new``) to carry on the state.
            # ``tke_source`` is the additive energy-recycling forc (eke_diss_iw
            # + K_diss_bot) routed in by the model step (Stage 2), at the
            # interior interfaces — None ⇒ no recycled sources.
            if dt_tke is None:
                raise ValueError(
                    "prognostic TKE (vertical_mixing.tke.prognostic=True) "
                    "requires dt_tke (the momentum timestep dt_mom) to be passed "
                    "to compute_vertical_K_profiles."
                )
            _tke_seed = tke_old
            if _tke_seed is None:
                # Cold start (state.tke not seeded): background floor, one step.
                leading = T_data.shape[:-1]
                _tke_seed = jnp.full(
                    leading + (z_coord.n_levels - 1,),
                    tke_cfg.tke_background, dtype=T_data.dtype,
                )
            if (getattr(tke_cfg, "buoyancy_timing", "pre_mixing")
                    == "post_mixing_veros"):
                # POST-MIXING Veros step order (buoyancy_timing=
                # "post_mixing_veros"): phase 1 only — K_M/K_H for the
                # tracer + momentum solves from the CARRIED tke (Veros
                # set_tke_diffusivities, tke[tau]). The TKE field is NOT
                # advanced here; the model step runs
                # ``tke_integrate_post_mixing`` AFTER the implicit tracer
                # solve on the POST-mixing N². The third slot returns the
                # :class:`TKEPostMixingContext` (the phase-1 ingredients)
                # instead of a tke array.
                from legoesm.ocean.physics.vertical_mixing.tke import (
                    tke_set_diffusivities,
                )
                K_M_old, K_H_old, _tke_ctx = tke_set_diffusivities(
                    u_data, v_data, T_data, S_data, rho, dz_half,
                    tke_old=_tke_seed,
                    tau_x_surface=tau_x, tau_y_surface=tau_y,
                taum_surface=taum_sf,
                    cfg=tke_cfg,
                    rho_0=constants_config.rho_0, g=constants_config.g,
                    p_cell=p_cell, dz_ref=z_coord.dz_ref, jacobian=J,
                    eos_fn=eos_fn, z_interface=z_coord.z_half_ref[1:-1],
                    dz_surface=dz_surface, boundary_cap=_mxl1_cap,
                    T_n2=T_n2, S_n2=S_n2,
                    ice_frac=_tke_ice_fr,
                )
                return K_H_old, K_M_old, _tke_ctx
            tke_out = tke_vertical_mixing(
                u_data, v_data, T_data, S_data, rho, dz_half,
                tke_old=_tke_seed,
                tau_x_surface=tau_x, tau_y_surface=tau_y,
                taum_surface=taum_sf,
                dt=dt_tke, cfg=tke_cfg,
                rho_0=constants_config.rho_0, g=constants_config.g,
                n_iterations=1,
                p_cell=p_cell, dz_ref=z_coord.dz_ref, jacobian=J, eos_fn=eos_fn,
                z_interface=z_coord.z_half_ref[1:-1],
                external_source=tke_source,
                dz_surface=dz_surface, boundary_cap=_mxl1_cap,
                lat_deg=lat_deg,
                T_n2=T_n2, S_n2=S_n2,
                t_depth=_bn2_t_depth, w_depth=_bn2_w_depth,
                ice_frac=_tke_ice_fr,
            )
            return tke_out.K_H, tke_out.K_M, tke_out.tke_new
        # Mode B (DIAGNOSTIC / quasi-steady, default): ``tke_old=None`` seeds at
        # background and 3 iterations of the same backward-Euler step bring TKE
        # to within ~few % of the prognostic equilibrium for typical ocean
        # shear / stratification. No TKE field is carried.
        _DIAGNOSTIC_DT = 86400.0   # long dt drives implicit solve to equilibrium
        tke_out = tke_vertical_mixing(
            u_data, v_data, T_data, S_data, rho, dz_half,
            tke_old=None,
            tau_x_surface=tau_x, tau_y_surface=tau_y,
                taum_surface=taum_sf,
            dt=_DIAGNOSTIC_DT, cfg=tke_cfg,
            rho_0=constants_config.rho_0, g=constants_config.g,
            n_iterations=3,
            p_cell=p_cell, dz_ref=z_coord.dz_ref, jacobian=J, eos_fn=eos_fn,
            # Interior interface depths (nlev-1) for the Bryan-Lewis kappaH
            # floor (Veros enable_kappaH_profile); z_half_ref is negative
            # downward, interior interfaces drop the surface (k=0) + bottom.
            z_interface=z_coord.z_half_ref[1:-1],
            dz_surface=dz_surface, boundary_cap=_mxl1_cap,
            lat_deg=lat_deg,
            T_n2=T_n2, S_n2=S_n2,
            t_depth=_bn2_t_depth, w_depth=_bn2_w_depth,
            ice_frac=_tke_ice_fr,
        )
        return tke_out.K_H, tke_out.K_M, None

    if scheme == "catke":
        from legoesm.ocean.physics.vertical_mixing.catke import (
            catke_vertical_mixing,
        )
        catke_cfg = vmix_cfg.catke
        # CATKE is ALWAYS prognostic (one backward-Euler TKE step per model
        # step, dt = dt_mom; the updated TKE is carried on the state).
        if dt_tke is None:
            raise ValueError(
                "CATKE (vertical_mixing.scheme='catke') is prognostic and "
                "requires dt_tke (the momentum timestep dt_mom) to be passed to "
                "compute_vertical_K_profiles."
            )
        # Cell-centre velocities (interp from C-grid faces if needed).
        u_data = state.u.data
        v_data = state.v.data
        T_data = state.T.data
        S_data = state.S.data
        if u_data.shape[1] != T_data.shape[1]:
            u_data = 0.5 * (u_data[:, :-1, :] + u_data[:, 1:, :])
            v_data = 0.5 * (v_data[:-1, :, :] + v_data[1:, :, :])
        dz_half = jnp.broadcast_to(
            z_coord.dz_half_ref * J[..., jnp.newaxis],
            T_data.shape[:-1] + (z_coord.n_levels - 1,),
        )
        # Interface geometry from the reference coordinate (interior interfaces,
        # length nlev-1). Depth below surface (>0) and height above the bottom.
        depth_iface = -z_coord.z_half_ref[1:-1]
        H_col = state.H_bathy.data
        hab_iface = jnp.maximum(H_col[..., jnp.newaxis] - depth_iface, 0.0)
        # Surface buoyancy flux Jb (shared helper) + friction velocity u_star.
        Jb, _, _ = _surface_buoyancy_flux(
            surface_forcing, state, constants_config, eos_fn=eos_fn)
        if Jb is None:
            Jb = jnp.zeros(T_data.shape[:-1], dtype=T_data.dtype)
        tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
        tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
        if tau_x is None and tau_y is None:
            u_star = jnp.zeros(T_data.shape[:-1], dtype=T_data.dtype)
        else:
            tx = tau_x if tau_x is not None else jnp.zeros_like(T_data[..., 0])
            ty = tau_y if tau_y is not None else jnp.zeros_like(T_data[..., 0])
            u_star = jnp.sqrt(
                jnp.sqrt(tx * tx + ty * ty) / constants_config.rho_0)
        _seed = tke_old
        if _seed is None:
            _seed = jnp.full(
                T_data.shape[:-1] + (z_coord.n_levels - 1,),
                catke_cfg.minimum_tke, dtype=T_data.dtype,
            )
        K_u, K_c, tke_new = catke_vertical_mixing(
            u_data, v_data, T_data, S_data, rho, dz_half,
            depth_iface, hab_iface, H_col,
            tke_old=_seed, Jb=Jb, u_star=u_star, dt=dt_tke, cfg=catke_cfg,
            rho_0=constants_config.rho_0, g=constants_config.g,
        )
        # Return (K_v = tracer = K_c, A_v = momentum = K_u, tke_new).
        return K_c, K_u, tke_new

    if scheme == "kpp":
        from legoesm.ocean.physics.vertical_mixing.kpp import (
            kpp_vertical_mixing,
        )
        # Reconstruct surface kinematic fluxes the same way the
        # integration factory does so the boundary-layer depth here
        # matches the depth used in the explicit physics call.
        tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
        tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
        # Surface buoyancy flux + kinematic T/S fluxes (shared with CATKE).
        B_f, Q_sfc_T, Q_sfc_S = _surface_buoyancy_flux(
            surface_forcing, state, constants_config, eos_fn=eos_fn)

        out = kpp_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, state.eta.data, z_coord, J, vmix_cfg.kpp,
            tau_x=tau_x, tau_y=tau_y, B_f=B_f,
            Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
            apply_diffusion=False, eos_fn=eos_fn,
        )
        return out.K_v, out.A_v, None

    if scheme == "none":
        # No background closure here (handled by the caller); zero K_v/A_v.
        nlev = state.T.data.shape[-1]
        shape = state.T.data.shape[:-1] + (nlev - 1,)
        dtype = state.T.data.dtype
        return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype), None

    # Dispatch hardening: an unknown scheme must NOT silently fall through to a
    # zero-mixing "be defensive" return (that disables vertical mixing on a typo,
    # masking the error).  ``scheme`` is the static config value, so raising at
    # function entry is jit-safe (this is the same defense used by the sibling
    # factories — see CLAUDE.md "Dispatch").
    from legoesm.ocean.physics.vertical_mixing.config import (
        VALID_VERTICAL_MIXING_SCHEMES,
    )
    raise ValueError(
        f"unknown vertical_mixing.scheme={scheme!r}; expected one of "
        f"{sorted(VALID_VERTICAL_MIXING_SCHEMES)}"
    )


def _enhanced_diffusion_K(state, z_coord, conv_cfg: OceanConvectionConfig,
                          eos_fn=None, before_tracers=None):
    """``(K_v, A_v)`` fields used by the ``enhanced_diffusion`` scheme.

    Returns the convective tracer diffusivity (``convective_κz``) and the
    independent momentum viscosity (``convective_νz``) at interfaces,
    bit-identical to the explicit ``enhanced_diffusion_convection`` path.

    ``eos_fn`` (optional) overrides the density EOS used for the convective
    N² trigger so it matches the dynamical core — the SAME contract the
    vmix branch honours.  ``None`` -> Wright 1997 (bit-identical legacy).
    Without threading it here the documented "convective trigger consistent
    with the dynamical core" promise of :func:`compute_vertical_K_profiles`
    was silently violated for non-Wright EOSs (codex review, finding #3).
    """
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    cfg = conv_cfg.enhanced_diffusion
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    rho = _compute_rho(state, z_coord, J, eos_fn=eos_fn)
    # Adiabatic N² trigger (cfg.n2_mode == "adiabatic") needs the cell-centre
    # hydrostatic pressure + the model EOS; computed only when opted in so the
    # default in-situ path is bit-identical (mirrors the richardson/tke branches
    # in _vmix_K_profiles).
    ed_p_cell = None
    if getattr(cfg, "n2_mode", "insitu") == "adiabatic":
        from legoesm.ocean.eos import (
            compute_hydrostatic_pressure, maybe_partial_h_actual,
        )
        ed_h_actual = maybe_partial_h_actual(state, z_coord)
        ed_p_cell = compute_hydrostatic_pressure(
            rho, state.eta.data, z_coord.dz_ref, J,
            ConstantsConfig().rho_0, h_actual=ed_h_actual,
        )
    # NEMO bn2 trigger (n2_mode="nemo_bn2"): geometric depth ladders
    # (gdept / interior gdepw); ignored by every other n2_mode.
    ed_t_depth = ed_w_depth = None
    if getattr(cfg, "n2_mode", "insitu") == "nemo_bn2":
        from legoesm.ocean.eos import nemo_bn2_depth_ladders
        ed_t_depth, ed_w_depth = nemo_bn2_depth_ladders(z_coord)
    # Shared, AD-safe helper — bit-for-bit identical to the explicit
    # ``enhanced_diffusion_convection`` path (no duplicated numerics).
    # Returns the full K / A (including the scheme's own backgrounds);
    # summing across schemes here is the *same* operation as the explicit
    # path: ``div(K1·∇T) + div(K2·∇T) = div((K1+K2)·∇T)``.
    K, A, _ = convective_K_A_flag(
        rho, z_coord.dz_ref, J, cfg,
        T=state.T.data, S=state.S.data, p_cell=ed_p_cell, eos_fn=eos_fn,
        t_depth=ed_t_depth, w_depth=ed_w_depth,
    )
    if getattr(cfg, "two_level_trigger", False) and before_tracers is not None:
        # NEMO zdfevd MIN(rn2, rn2b): evaluate the trigger on the BEFORE
        # tracers too and take the elementwise max of the coefficients —
        # equivalent to the min-N² trigger for the hard-threshold path.
        # Prevents per-step ON/OFF flicker of the convective coefficient in
        # marginal columns (a grid-scale noise source; plan §G).
        T_b, S_b = before_tracers
        state_b = state._replace(T=state.T.replace(data=T_b),
                                 S=state.S.replace(data=S_b))
        rho_b = _compute_rho(state_b, z_coord, J, eos_fn=eos_fn)
        ed_p_cell_b = None
        if getattr(cfg, "n2_mode", "insitu") == "adiabatic":
            from legoesm.ocean.eos import (
                compute_hydrostatic_pressure, maybe_partial_h_actual,
            )
            ed_h_b = maybe_partial_h_actual(state_b, z_coord)
            ed_p_cell_b = compute_hydrostatic_pressure(
                rho_b, state_b.eta.data, z_coord.dz_ref, J,
                ConstantsConfig().rho_0, h_actual=ed_h_b,
            )
        K_b, A_b, _ = convective_K_A_flag(
            rho_b, z_coord.dz_ref, J, cfg,
            T=T_b, S=S_b, p_cell=ed_p_cell_b, eos_fn=eos_fn,
            t_depth=ed_t_depth, w_depth=ed_w_depth,
        )
        K = jnp.maximum(K, K_b)
        A = jnp.maximum(A, A_b)
    return K, A


def iwm_K_profile(state, z_coord, physics_config, iwm_cfg, *,
                   eos_fn=None, iwm_fields=None):
    """Internal wave-driven diffusivity at interior interfaces (zdfiwm).

    Assembles the column geometry (NEMO gdept / e3w / ht analogues) and
    the interface N², then delegates the physics to
    :func:`..internal_wave_mixing.compute_iwm_diffusivity` (single-owner
    numerics).  ``iwm_fields`` is an :class:`..internal_wave_mixing.
    IWMForcing` of static 2-D maps (the de Lavergne product regridded to
    the model grid); ``None`` falls back to the uniform constant-power
    maps built from the config scalars.

    Geometry note: on an :class:`OceanPartialCellCoordinate` the depths
    and spacings come from the partial thicknesses (``h_partial·J`` —
    exactly NEMO's partial-aware e3t/gdept construction); on a pure
    z-star coordinate they are the reference geometry stretched by the
    Jacobian.  ``N²`` uses the shared ``compute_N2`` in-situ mode
    (clipped >= 0) — the same construction as NEMO's ``MAX(0, rn2)``
    usage in every zdfiwm structure function.
    """
    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
        compute_iwm_diffusivity, uniform_iwm_forcing,
    )
    from legoesm.ocean.vertical import OceanPartialCellCoordinate

    constants_config = physics_config.constants
    T = state.T.data
    dtype = T.dtype
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    rho = _compute_rho(state, z_coord, J, eos_fn=eos_fn)

    if isinstance(z_coord, OceanPartialCellCoordinate):
        # Partial-aware geometry: actual per-cell thickness (0 below the
        # seafloor), centre depths from its cumsum (NEMO gdept), column
        # depth = wet-column sum (NEMO ht).
        h_act = z_coord.h_partial * J[..., jnp.newaxis]
        depth_cell = jnp.cumsum(h_act, axis=-1) - 0.5 * h_act
        H_col = jnp.sum(h_act, axis=-1)
        dz_w = 0.5 * (h_act[..., :-1] + h_act[..., 1:])
    else:
        depth_cell = -z_coord.z_full_ref * J[..., jnp.newaxis]
        H_col = -z_coord.z_half_ref[-1] * J
        dz_w = z_coord.dz_half_ref * J[..., jnp.newaxis]
    depth_cell = depth_cell.astype(dtype)
    dz_w = dz_w.astype(dtype)

    N2 = compute_N2(
        rho, dz_w, constants_config.rho_0, g=constants_config.g,
        n2_mode="insitu",
    )

    if iwm_fields is None:
        iwm_fields = uniform_iwm_forcing(iwm_cfg, H_col.shape, dtype=dtype)

    K_iwm, _ratio = compute_iwm_diffusivity(
        iwm_fields, depth_cell, dz_w, H_col, N2,
        cfg=iwm_cfg, rho_0=constants_config.rho_0,
    )
    # NEMO applies wmask inside zdf_iwm; here the caller's wet-interface
    # guard (compute_vertical_K_profiles tail) zeroes non-wet interfaces,
    # and dry COLUMNS (H = 0) already produce the k_min floor which the
    # land mask removes in the tracer/momentum solves.
    return K_iwm.astype(dtype)


def ddm_K_profile(state, z_coord, physics_config, ddm_cfg, *, eos_fn):
    """Double-diffusive ``(avt_ddm, avs_ddm)`` at interior interfaces (zdfddm).

    Mirrors :func:`iwm_K_profile` geometry, forms the density ratio
    ``R_rho = (alpha dT/dz)/(beta dS/dz)`` from the SELECTED EOS's locally-
    referenced derivatives (``eos_density_derivatives`` — alpha=-drho_dT/rho,
    beta=drho_dS/rho, single-owner α/β; never a re-derived linear pair) and the
    interface T/S gradients, and delegates the regime physics to
    :func:`..double_diffusion.compute_ddm_diffusivity` (single-owner numerics).
    ``N^2`` uses the shared insitu :func:`compute_N2` (clipped >= 0), so
    statically unstable interfaces return zero (the convection scheme owns
    them).  Returns the RAW interior-interface contribution; the caller applies
    the same wet-interface mask it applies to ``K_v``.  ``eos_fn`` is REQUIRED
    (the model threads its own EOS, same as iwm).
    """
    from legoesm.ocean.eos import eos_density_derivatives
    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
    from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
        compute_ddm_diffusivity,
    )
    from legoesm.ocean.vertical import (
        OceanPartialCellCoordinate, extrapolate_below_seafloor,
    )

    cc = physics_config.constants
    # Extrapolate T/S into the below-seafloor cells BEFORE any EOS / gradient
    # math (same guard as compute_vertical_K_profiles): partial-cell dry cells
    # carry T=S=0, which would give spurious R_rho / poison reverse-mode grads
    # (0*NaN) even though the caller's wet-interface mask zeroes the RESULT
    # later — too late for the nonlinear EOS/gradient ops here (codex r2).
    # No-op (bit-identical) for pure z-star coords (no ``is_active``).
    if getattr(z_coord, "is_active", None) is not None:
        state = state._replace(
            T=state.T.replace(data=extrapolate_below_seafloor(
                state.T.data, z_coord)),
            S=state.S.replace(data=extrapolate_below_seafloor(
                state.S.data, z_coord)),
        )
    T = state.T.data
    S = state.S.data
    dtype = T.dtype
    _eps = jnp.asarray(jnp.finfo(jnp.float32).eps, dtype)
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    rho = _compute_rho(state, z_coord, J, eos_fn=eos_fn)

    # Interface spacing dz_w + interface depth (for the local reference
    # pressure), identical construction to iwm_K_profile.
    if isinstance(z_coord, OceanPartialCellCoordinate):
        h_act = z_coord.h_partial * J[..., jnp.newaxis]
        dz_w = 0.5 * (h_act[..., :-1] + h_act[..., 1:])
        depth_if = jnp.cumsum(h_act, axis=-1)[..., :-1]
    else:
        dz_w = z_coord.dz_half_ref * J[..., jnp.newaxis]
        # Interior interface depths from the REFERENCE half-levels (z up:
        # z_half_ref[1:-1] are the nlev-1 interior interfaces), stretched by J
        # — NOT cumsum(dz_half), which drifts from the true depth on a
        # non-uniform grid (codex r2).  Matches iwm_K_profile's use of the
        # reference levels rather than a running sum.
        depth_if = -z_coord.z_half_ref[1:-1] * J[..., jnp.newaxis]
    dz_w = jnp.maximum(dz_w.astype(dtype), _eps)

    # Local reference pressure at the interface [Pa] (hydrostatic proxy, same
    # as the EOS pressure path); alpha/beta from the interface-averaged T,S.
    p_if = (cc.rho_0 * cc.g * depth_if).astype(dtype)
    T_if = 0.5 * (T[..., :-1] + T[..., 1:])
    S_if = 0.5 * (S[..., :-1] + S[..., 1:])
    rho_if = jnp.maximum(0.5 * (rho[..., :-1] + rho[..., 1:]), _eps)
    drho_dT, drho_dS = eos_density_derivatives(eos_fn, T_if, S_if, p_if)
    alpha_if = -drho_dT / rho_if       # thermal expansion (>0 typical)
    beta_if = drho_dS / rho_if         # haline contraction (>0 typical)

    # Interface gradients (z up: upper cell minus lower cell over dz_w).
    dT_dz = (T[..., :-1] - T[..., 1:]) / dz_w
    dS_dz = (S[..., :-1] - S[..., 1:]) / dz_w
    alpha_dTdz = alpha_if * dT_dz
    beta_dSdz = beta_if * dS_dz

    N2 = compute_N2(rho, dz_w, cc.rho_0, g=cc.g, n2_mode="insitu")

    avt_ddm, avs_ddm = compute_ddm_diffusivity(N2, alpha_dTdz, beta_dSdz, ddm_cfg)
    return avt_ddm.astype(dtype), avs_ddm.astype(dtype)
