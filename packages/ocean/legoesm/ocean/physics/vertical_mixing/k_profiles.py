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
floors.
"""

from __future__ import annotations


import jax.numpy as jnp

from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.eos import (
    compute_ocean_rho as _compute_rho,
    thermal_expansion_coeff,
    haline_contraction_coeff,
)
from legoesm.ocean.vertical import (
    OceanZStarCoordinate, compute_ocean_jacobian,
)
from legoesm.ocean.physics.convection.config import OceanConvectionConfig


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
        (typically ``LatLonCGridOceanConfig.A_v`` / ``K_v``).
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
        prognostic TKE closure, else ``None``.
    """
    T = state.T.data
    nlev = T.shape[-1]
    interface_shape = T.shape[:-1] + (nlev - 1,)
    dtype = T.dtype

    # Start with the configured background floors.  These are scalar
    # floats; broadcast to interface shape.
    K_v_total = jnp.full(interface_shape, K_v_background, dtype=dtype)
    A_v_total = jnp.full(interface_shape, A_v_background, dtype=dtype)

    tke_new = None
    vmix = physics_config.vertical_mixing
    if vmix.scheme != "none":
        K_vmix, A_vmix, tke_new = _vmix_K_profiles(
            state, z_coord, surface_forcing, vmix, physics_config.constants,
            eos_fn=eos_fn,
            tke_old=tke_old, dt_tke=dt_tke, tke_source=tke_source)
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
        K_conv, A_conv = _enhanced_diffusion_K(state, z_coord, conv)
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

    if return_tke:
        return K_v_total, A_v_total, tke_new
    return K_v_total, A_v_total


# ---------------------------------------------------------------------------
# Per-scheme K computation helpers (interior interface shape).
# ---------------------------------------------------------------------------


def _vmix_K_profiles(state, z_coord, surface_forcing, vmix_cfg,
                     constants_config=ConstantsConfig(), eos_fn=None,
                     *, tke_old=None, dt_tke=None, tke_source=None):
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
        K_v = jnp.full(shape, cfg.K_v, dtype=dtype)
        A_v = jnp.full(shape, cfg.A_v, dtype=dtype)
        return K_v, A_v, None

    rho = _compute_rho(state, z_coord, J, eos_fn=eos_fn)

    if scheme == "richardson":
        from legoesm.ocean.physics.vertical_mixing.richardson import (
            richardson_vertical_mixing,
        )
        out = richardson_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, z_coord, J, vmix_cfg.richardson,
            apply_diffusion=False,
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
        dz_half = jnp.broadcast_to(
            z_coord.dz_half_ref * J[..., jnp.newaxis],
            T_data.shape[:-1] + (z_coord.n_levels - 1,),
        )
        tau_x = (getattr(surface_forcing, "tau_x", None)
                 if surface_forcing is not None else None)
        tau_y = (getattr(surface_forcing, "tau_y", None)
                 if surface_forcing is not None else None)
        # Adiabatic static-stability N² (Veros parcel displacement) needs
        # the cell-centre hydrostatic pressure + the same EOS as the
        # dynamical core. Only computed when the TKE config opts in
        # (``n2_mode="adiabatic"``) so the default path is unchanged.
        p_cell = None
        if getattr(vmix_cfg.tke, "n2_mode", "insitu") == "adiabatic":
            from legoesm.ocean.eos import compute_hydrostatic_pressure
            from legoesm.ocean.eos import _maybe_partial_h_actual
            h_actual = _maybe_partial_h_actual(state, z_coord)
            p_cell = compute_hydrostatic_pressure(
                rho, state.eta.data, z_coord.dz_ref, J,
                constants_config.rho_0, h_actual=h_actual,
            )
        tke_cfg = vmix_cfg.tke
        prognostic = bool(getattr(tke_cfg, "prognostic", False))
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
            tke_out = tke_vertical_mixing(
                u_data, v_data, T_data, S_data, rho, dz_half,
                tke_old=_tke_seed,
                tau_x_surface=tau_x, tau_y_surface=tau_y,
                dt=dt_tke, cfg=tke_cfg,
                rho_0=constants_config.rho_0, g=constants_config.g,
                n_iterations=1,
                p_cell=p_cell, dz_ref=z_coord.dz_ref, jacobian=J, eos_fn=eos_fn,
                z_interface=z_coord.z_half_ref[1:-1],
                external_source=tke_source,
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
            dt=_DIAGNOSTIC_DT, cfg=tke_cfg,
            rho_0=constants_config.rho_0, g=constants_config.g,
            n_iterations=3,
            p_cell=p_cell, dz_ref=z_coord.dz_ref, jacobian=J, eos_fn=eos_fn,
            # Interior interface depths (nlev-1) for the Bryan-Lewis kappaH
            # floor (Veros enable_kappaH_profile); z_half_ref is negative
            # downward, interior interfaces drop the surface (k=0) + bottom.
            z_interface=z_coord.z_half_ref[1:-1],
        )
        return tke_out.K_H, tke_out.K_M, None

    if scheme == "kpp":
        from legoesm.ocean.physics.vertical_mixing.kpp import (
            kpp_vertical_mixing,
        )
        # Reconstruct surface kinematic fluxes the same way the
        # integration factory does so the boundary-layer depth here
        # matches the depth used in the explicit physics call.
        tau_x = getattr(surface_forcing, "tau_x", None) if surface_forcing else None
        tau_y = getattr(surface_forcing, "tau_y", None) if surface_forcing else None
        q_net = getattr(surface_forcing, "q_net", None) if surface_forcing else None
        fw = getattr(surface_forcing, "freshwater", None) if surface_forcing else None
        salt = getattr(surface_forcing, "salt_flux", None) if surface_forcing else None

        Q_sfc_T = None
        B_f = None
        if q_net is not None:
            Q_sfc_T = q_net / (constants_config.rho_0 * constants_config.c_sw)
            T_sfc = state.T.data[..., 0]
            S_sfc = state.S.data[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            alpha = thermal_expansion_coeff(T_sfc, S_sfc, p_sfc)
            B_f = -constants_config.g * alpha * Q_sfc_T

        Q_sfc_S = None
        if fw is not None or salt is not None:
            S_sfc = state.S.data[..., 0]
            T_sfc = state.T.data[..., 0]
            p_sfc = jnp.zeros_like(T_sfc)
            beta = haline_contraction_coeff(T_sfc, S_sfc, p_sfc)
            # Kinematic surface salt flux [PSU·m/s]: freshwater dilution
            # (-S*fw/rho, fw>0 in -> stabilizing) PLUS a REAL salt-mass flux
            # (+salt*1e3/rho, salt>0 in -> destabilizing brine rejection).
            Q_sfc_S = jnp.zeros_like(S_sfc)
            if fw is not None:
                Q_sfc_S = Q_sfc_S - S_sfc * fw / constants_config.rho_0
            if salt is not None:
                Q_sfc_S = Q_sfc_S + salt * 1.0e3 / constants_config.rho_0
            B_salt = constants_config.g * beta * Q_sfc_S
            B_f = B_salt if B_f is None else (B_f + B_salt)

        out = kpp_vertical_mixing(
            state.u.data, state.v.data, state.T.data, state.S.data,
            rho, state.eta.data, z_coord, J, vmix_cfg.kpp,
            tau_x=tau_x, tau_y=tau_y, B_f=B_f,
            Q_sfc_T=Q_sfc_T, Q_sfc_S=Q_sfc_S,
            apply_diffusion=False,
        )
        return out.K_v, out.A_v, None

    # "none" — handled by the caller, but be defensive.
    nlev = state.T.data.shape[-1]
    shape = state.T.data.shape[:-1] + (nlev - 1,)
    dtype = state.T.data.dtype
    return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype), None


def _enhanced_diffusion_K(state, z_coord, conv_cfg: OceanConvectionConfig):
    """``(K_v, A_v)`` fields used by the ``enhanced_diffusion`` scheme.

    Returns the convective tracer diffusivity (``convective_κz``) and the
    independent momentum viscosity (``convective_νz``) at interfaces,
    bit-identical to the explicit ``enhanced_diffusion_convection`` path.
    """
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    cfg = conv_cfg.enhanced_diffusion
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    rho = _compute_rho(state, z_coord, J)
    # Shared, AD-safe helper — bit-for-bit identical to the explicit
    # ``enhanced_diffusion_convection`` path (no duplicated numerics).
    # Returns the full K / A (including the scheme's own backgrounds);
    # summing across schemes here is the *same* operation as the explicit
    # path: ``div(K1·∇T) + div(K2·∇T) = div((K1+K2)·∇T)``.
    K, A, _ = convective_K_A_flag(rho, z_coord.dz_ref, J, cfg)
    return K, A
