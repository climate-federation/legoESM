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

from typing import Tuple

import jax
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
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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

    Returns
    -------
    (K_v, A_v) : tuple of arrays
        Each of shape ``state.T.data.shape[:-1] + (nlev - 1,)``, on
        interior interfaces, in m²/s.
    """
    T = state.T.data
    S = state.S.data
    nlev = T.shape[-1]
    interface_shape = T.shape[:-1] + (nlev - 1,)
    dtype = T.dtype

    # Start with the configured background floors.  These are scalar
    # floats; broadcast to interface shape.
    K_v_total = jnp.full(interface_shape, K_v_background, dtype=dtype)
    A_v_total = jnp.full(interface_shape, A_v_background, dtype=dtype)

    vmix = physics_config.vertical_mixing
    if vmix.scheme != "none":
        K_vmix, A_vmix = _vmix_K_profiles(
            state, z_coord, surface_forcing, vmix, physics_config.constants,
            eos_fn=eos_fn)
        K_v_total = K_v_total + K_vmix
        A_v_total = A_v_total + A_vmix

    conv = physics_config.convection
    if conv.scheme == "enhanced_diffusion":
        K_conv = _enhanced_diffusion_K(state, z_coord, conv)
        # Convection enhances tracer diffusivity (and indirectly momentum,
        # since the static instability is shared with KPP's K_conv).
        K_v_total = K_v_total + K_conv
        # For momentum, the convective enhancement is also applied in KPP
        # interior (A_interior includes K_conv).  When KPP is on, that
        # contribution is already in A_vmix.  When KPP is off, we still
        # apply K_conv to momentum so the explicit/implicit equivalence
        # holds for the constant + convection composition.
        if vmix.scheme != "kpp":
            A_v_total = A_v_total + K_conv

    # Clip to KPP K_max when KPP is the vertical mixing scheme, matching
    # the explicit path's saturation behavior.  Otherwise leave the sum
    # uncapped (the schemes' own configs already include sensible
    # backgrounds, and any caller-supplied floor is small).
    if vmix.scheme == "kpp":
        K_max = vmix.kpp.K_max
        K_v_total = jnp.minimum(K_v_total, K_max)
        A_v_total = jnp.minimum(A_v_total, K_max)

    return K_v_total, A_v_total


# ---------------------------------------------------------------------------
# Per-scheme K computation helpers (interior interface shape).
# ---------------------------------------------------------------------------


def _vmix_K_profiles(state, z_coord, surface_forcing, vmix_cfg,
                     constants_config=ConstantsConfig(), eos_fn=None):
    """Re-compute K_v, A_v at interfaces for the chosen vmix scheme.

    For ``constant`` / ``richardson`` this duplicates only the K
    computation (cheap).  For ``kpp`` this also re-runs the boundary
    layer diagnosis, which is somewhat more expensive but still much
    cheaper than the tridiagonal solve it enables.

    ``eos_fn`` (optional) overrides the density EOS used to compute N²
    (default Wright 1997 -> bit-identical legacy).
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
        return K_v, A_v

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
        return out.K_v, out.A_v

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
        # Use Mode B (diagnostic / quasi-steady) iteration: ``tke_old=None``
        # seeds at background and 3 iterations of the same backward-Euler
        # step bring TKE to within ~few % of the prognostic equilibrium
        # for typical ocean shear / stratification. True prognostic mode
        # (TKE carried across timesteps via ``state.tke``) is a future
        # upgrade tracked in the Phase G audit doc.
        _DIAGNOSTIC_DT = 86400.0   # long dt drives implicit solve to equilibrium
        tke_out = tke_vertical_mixing(
            u_data, v_data, T_data, S_data, rho, dz_half,
            tke_old=None,
            tau_x_surface=tau_x, tau_y_surface=tau_y,
            dt=_DIAGNOSTIC_DT, cfg=vmix_cfg.tke,
            rho_0=constants_config.rho_0, g=constants_config.g,
            n_iterations=3,
            p_cell=p_cell, dz_ref=z_coord.dz_ref, jacobian=J, eos_fn=eos_fn,
            # Interior interface depths (nlev-1) for the Bryan-Lewis kappaH
            # floor (Veros enable_kappaH_profile); z_half_ref is negative
            # downward, interior interfaces drop the surface (k=0) + bottom.
            z_interface=z_coord.z_half_ref[1:-1],
        )
        return tke_out.K_H, tke_out.K_M

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
        return out.K_v, out.A_v

    # "none" — handled by the caller, but be defensive.
    nlev = state.T.data.shape[-1]
    shape = state.T.data.shape[:-1] + (nlev - 1,)
    dtype = state.T.data.dtype
    return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype)


def _enhanced_diffusion_K(state, z_coord, conv_cfg: OceanConvectionConfig):
    """Diffusivity field used by the ``enhanced_diffusion`` convection scheme."""
    cfg = conv_cfg.enhanced_diffusion
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z_coord)
    rho = _compute_rho(state, z_coord, J)
    from legoesm.ocean.eos import compute_buoyancy_frequency
    N2 = compute_buoyancy_frequency(rho, z_coord.dz_ref, J)
    if cfg.smooth_transition:
        K = cfg.K_bg + (cfg.K_conv - cfg.K_bg) * jax.nn.sigmoid(
            -N2 * cfg.sigmoid_sharpness)
    else:
        K = jnp.where(N2 < 0.0, cfg.K_conv, cfg.K_bg)
    # Return the full K (including the scheme's own K_bg).  Summing
    # across schemes here is the *same* operation as the explicit path:
    # ``div(K1·∇T) + div(K2·∇T) = div((K1+K2)·∇T)``.
    return K
