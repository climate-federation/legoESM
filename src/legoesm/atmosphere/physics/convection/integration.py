"""Model integration bridge for convection.

Provides `make_convection_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    HydrostaticTendencies,
    NonHydrostaticState,
    NonHydrostaticTendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    HeightCoordinate,
    SigmaCoordinate,
    TerrainMetric,
    pressure_from_sigma,
)
from legoesm import constants

from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import dca_convection
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.mass_flux import (
    edmf_convection,
    mass_flux_convection,
)
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection,
)
from legoesm.atmosphere.physics.convection.kain_fritsch import (
    kain_fritsch_convection,
)
from legoesm.atmosphere.physics.convection.emanuel import (
    emanuel_convection,
)
from legoesm.atmosphere.physics.convection.tiedtke import (
    tiedtke_convection,
)
from legoesm.atmosphere.physics.convection.bechtold import (
    bechtold_convection,
)
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


def _get_convection_fn(config: ConvectionConfig):
    """Select the convection backend based on config.scheme.

    Returns
    -------
    scheme_name : str
        Name of the scheme.
    conv_fn : callable or None
        Backend convection function.
    scheme_config : NamedTuple or None
        Scheme-specific configuration.
    """
    if config.scheme == "sbm":
        return "sbm", sbm_convection, config.sbm
    elif config.scheme == "dca":
        return "dca", dca_convection, config.dca
    elif config.scheme == "kuo":
        return "kuo", kuo_convection, config.kuo
    elif config.scheme == "mass_flux":
        return "mass_flux", mass_flux_convection, config.mass_flux
    elif config.scheme == "edmf":
        return "edmf", edmf_convection, config.edmf
    elif config.scheme == "zhang_mcfarlane":
        return "zhang_mcfarlane", zhang_mcfarlane_convection, config.zhang_mcfarlane
    elif config.scheme == "kain_fritsch":
        return "kain_fritsch", kain_fritsch_convection, config.kain_fritsch
    elif config.scheme == "emanuel":
        return "emanuel", emanuel_convection, config.emanuel
    elif config.scheme == "tiedtke":
        return "tiedtke", tiedtke_convection, config.tiedtke
    elif config.scheme == "bechtold":
        return "bechtold", bechtold_convection, config.bechtold
    elif config.scheme == "none":
        return "none", None, None
    else:
        raise ValueError(f"Unknown convection scheme: {config.scheme!r}")


def make_convection_physics(
    convection_config: ConvectionConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,
) -> Callable:
    """Create a physics function for convection matching a model's signature.

    Parameters
    ----------
    convection_config : ConvectionConfig
        Convection configuration (selects SBM, DCA, Kuo, mass_flux,
        EDMF, or none).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    dt : float
        Model time step [s]. Needed for relaxation timescale.

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    if model_type == "hydrostatic":
        return _make_hydrostatic_convection(convection_config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_convection(convection_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_convection(convection_config, dt)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord, phys_state=None)
               -> (HydrostaticTendencies, conv_prog_profile_new | None)

    When *phys_state* is passed, the convective prognostic profile is
    read from ``phys_state.conv_prog_profile`` (shape ``(ncol, nlev)``)
    and the updated profile is returned as the second element of the
    result tuple.  Scalar-carrying schemes (``mass_flux``, ``edmf``)
    pack their scalar at ``[:, -1]`` (cloud-base proxy) with zeros
    aloft; profile-carrying schemes (Tiedtke, Bechtold, added in later
    PRs) use the full profile.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    # Scalar-carrying schemes that pack their state at [:, -1].
    is_scalar_prognostic = scheme_name in ("mass_flux", "edmf")
    # Profile-carrying schemes (full conv_prog_profile is meaningful).
    # ZM is technically diagnostic but uses [:, -1] as a M_b carry for
    # implicit relaxation; we route it through the profile-aware path
    # because the leaf signature accepts u, v for CMT.  KF is also
    # diagnostic but routed through the profile path so the bridge can
    # plumb the ``w_grid`` argument for its trigger.
    is_profile_prognostic = scheme_name in (
        "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
    )
    is_cmt_capable = scheme_name in ("zhang_mcfarlane", "tiedtke", "bechtold")
    is_w_grid_consumer = scheme_name in ("kain_fritsch",)
    is_stochastic = scheme_name in ("bechtold",)

    prog_key = None
    prog_init = None
    if is_scalar_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(
        state: HydrostaticState,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        phys_state=None,
    ):
        T = state.T.data          # (6, n, n, nlev)
        p_s = state.p_s.data      # (6, n, n)

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        # Pressure at full and half levels
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

        # Reshape to columns: (6,n,n,...) -> (ncol, ...)
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Pin defaulted allocations to the state precision so x64-default
        # zeros do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _ps_dtype = state.p_s.data.dtype
        # Extract water vapor from tracers if available; else assume dry.
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        # Wind columns for CMT-capable schemes.
        if is_cmt_capable:
            u_col = state.u.data.reshape(ncol, nlev)
            v_col = (
                state.v.data.reshape(ncol, nlev)
                if state.v is not None
                else jnp.zeros_like(u_col)
            )
        else:
            u_col = None
            v_col = None

        # Grid-scale w proxy for w-consuming schemes (Kain-Fritsch).
        # The hydrostatic dycore does not expose ``omega`` (or
        # equivalently ``sigma_dot``) at this layer; we pass zeros and
        # let the trigger be driven by ``parcel_perturb_T`` alone.  See
        # TODO(kf-w-grid) for the eventual omega→w wiring.
        if is_w_grid_consumer:
            w_grid_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        else:
            w_grid_col = None

        conv_prog_out = None
        if conv_fn is None:
            # "none" scheme: return zero tendencies
            dT_dt = jnp.zeros(shape_3d, dtype=_state_dtype)
            conv_out = None
        elif is_scalar_prognostic:
            # Scalar-carrying schemes pack at [:, -1]; slice to recover
            # the per-column scalar.  Falls back to scheme default when
            # the carry is the wrong shape (warm start, scheme switch).
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile[:, -1]
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            # Pack the updated scalar back into the (ncol, nlev) profile.
            conv_prog_out = jnp.zeros(
                (ncol, nlev), dtype=_state_dtype
            ).at[:, -1].set(prog_new)
            dT_dt = conv_out.dT_dt.reshape(shape_3d)
        elif is_profile_prognostic:
            # Profile-carrying schemes (ZM, KF, Emanuel, Tiedtke,
            # Bechtold) take and return the full
            # ``conv_prog_profile`` directly.  Per-scheme kwarg
            # plumbing handles CMT (u, v), the KF trigger (w_grid),
            # and Bechtold's stochastic state (conv_stoch_state +
            # prng_key).
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile
            else:
                prog_in = jnp.zeros((ncol, nlev), dtype=_state_dtype)

            if is_stochastic:
                # Bechtold: also threads conv_stoch_state and prng_key.
                if phys_state is not None and (
                    phys_state.conv_stoch_state.shape == (ncol,)
                ):
                    stoch_in = phys_state.conv_stoch_state
                else:
                    stoch_in = jnp.zeros((ncol,), dtype=_state_dtype)
                # TODO(bechtold-prng): route a per-step PRNG key from
                # the orchestrator.  For now we pass ``None`` which
                # falls back to deterministic behavior in the leaf.
                conv_out, prog_new_profile, stoch_new = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    conv_stoch_state=stoch_in,
                    prng_key=None,
                    dt=dt, config=scheme_config,
                )
                # Multi-field carry update — return as dict so the
                # orchestrator can ``update`` both PhysicsState slots.
                conv_prog_out = {
                    "conv_prog_profile": prog_new_profile,
                    "conv_stoch_state": stoch_new,
                }
            elif is_cmt_capable:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            elif is_w_grid_consumer:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    w_grid=w_grid_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            else:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            dT_dt = conv_out.dT_dt.reshape(shape_3d)
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )
            dT_dt = conv_out.dT_dt.reshape(shape_3d)

        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        # Propagate tracer tendencies from convection backend.
        # Convective detrained condensate (``dq_c_conv_dt``) feeds the
        # cloud-water tracer; the dynamical core's tracer registry
        # picks it up by name (``q_c``) and applies it alongside the
        # microphysics tendency on the next step. Models without a
        # ``q_c`` tracer simply ignore the entry.
        tracer_tends = None
        if conv_fn is not None:
            dq_v_dt = conv_out.dq_v_dt.reshape(shape_3d)
            dq_c_conv_dt = conv_out.dq_c_conv_dt.reshape(shape_3d)
            tracer_tends = {
                "q_v": Field(
                    data=dq_v_dt, name="dq_v_dt_conv",
                    dims=dims_3d, units="kg/kg/s",
                ),
                "q_c": Field(
                    data=dq_c_conv_dt, name="dq_c_conv_dt",
                    dims=dims_3d, units="kg/kg/s",
                ),
            }

        # Convective momentum transport (CMT): use the scheme's optional
        # ``du_dt_conv``/``dv_dt_conv`` when present (Zhang-McFarlane,
        # Tiedtke, Bechtold).  Schemes that do not produce CMT (the
        # existing five plus Kain-Fritsch and Emanuel) leave these as
        # ``None`` and the bridge zero-fills.
        if conv_fn is not None and conv_out.du_dt_conv is not None:
            du_dt = conv_out.du_dt_conv.reshape(shape_3d)
        else:
            du_dt = jnp.zeros(shape_3d, dtype=_state_dtype)
        if conv_fn is not None and conv_out.dv_dt_conv is not None:
            dv_dt = conv_out.dv_dt_conv.reshape(shape_3d)
        else:
            dv_dt = jnp.zeros(shape_3d, dtype=_state_dtype)

        tendencies = HydrostaticTendencies(
            du_dt=Field(
                data=du_dt, name="du_dt_conv",
                dims=dims_3d, units="m/s^2",
            ),
            dv_dt=Field(
                data=dv_dt, name="dv_dt_conv",
                dims=dims_3d, units="m/s^2",
            ),
            dT_dt=Field(
                data=dT_dt, name="dT_dt_conv",
                dims=dims_3d, units="K/s",
            ),
            dp_s_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_conv",
                dims=dims_2d, units="Pa/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_conv",
                dims=dims_2d, units="m^2/s^3",
            ),
            tracer_tendencies=tracer_tends,
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric, phys_state=None)
               -> (NonHydrostaticTendencies, conv_prog_profile_new | None)

    When *phys_state* is passed, the convective prognostic profile is
    read from ``phys_state.conv_prog_profile`` (shape ``(ncol, nlev)``)
    and the updated profile is returned as the second element of the
    result tuple.  See the hydrostatic bridge for the slice/pack
    convention for scalar-carrying schemes.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    is_scalar_prognostic = scheme_name in ("mass_flux", "edmf")
    is_profile_prognostic = scheme_name in (
        "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
    )
    is_cmt_capable = scheme_name in ("zhang_mcfarlane", "tiedtke", "bechtold")
    is_w_grid_consumer = scheme_name in ("kain_fritsch",)
    is_stochastic = scheme_name in ("bechtold",)
    prog_key = None
    prog_init = None
    if is_scalar_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        phys_state=None,
    ) -> NonHydrostaticTendencies:
        theta_p = state.theta_prime.data   # (6, n, n, nlev)
        rho_p = state.rho_prime.data       # (6, n, n, nlev)
        tracers = state.tracers.data       # (6, n, n, nlev, n_tracers)

        # Reference profiles
        theta_0 = height_coord.theta_ref
        rho_0 = height_coord.rho_ref

        # Total fields
        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p,
            rho_0 + rho_p,
        )

        # Temperature and pressure
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape
        shape_w = state.w.data.shape
        shape_2d = state.phis.data.shape
        n_tracers = tracers.shape[-1] if tracers.ndim >= 5 else 0

        # Interface pressure from evolving column state (not fixed reference).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Pin defaulted allocations to the state precision so x64 zeros
        # do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype
        # Extract q_v from tracers if available
        if n_tracers > 0:
            q_v_col = tracers[..., 0].reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        if conv_fn is None:
            return NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_conv", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_conv", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_conv", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_conv", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_conv", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_conv", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_conv", dims=dims_tr, units="1/s"),
            )

        # Wind columns for CMT-capable schemes.
        if is_cmt_capable:
            u_col = state.u.data.reshape(ncol, nlev)
            v_col = state.v.data.reshape(ncol, nlev)
        else:
            u_col = None
            v_col = None

        # Grid-scale w for w-consuming schemes (KF).  ``state.w`` lives
        # at half levels — interpolate to full-level centers.
        if is_w_grid_consumer:
            w_data = state.w.data.reshape(ncol, nlev + 1)
            w_grid_col = 0.5 * (w_data[:, :-1] + w_data[:, 1:])
        else:
            w_grid_col = None

        conv_prog_out = None
        if is_scalar_prognostic:
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile[:, -1]
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            conv_prog_out = jnp.zeros(
                (ncol, nlev), dtype=_state_dtype
            ).at[:, -1].set(prog_new)
        elif is_profile_prognostic:
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile
            else:
                prog_in = jnp.zeros((ncol, nlev), dtype=_state_dtype)

            if is_stochastic:
                if phys_state is not None and (
                    phys_state.conv_stoch_state.shape == (ncol,)
                ):
                    stoch_in = phys_state.conv_stoch_state
                else:
                    stoch_in = jnp.zeros((ncol,), dtype=_state_dtype)
                conv_out, prog_new_profile, stoch_new = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    conv_stoch_state=stoch_in,
                    prng_key=None,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = {
                    "conv_prog_profile": prog_new_profile,
                    "conv_stoch_state": stoch_new,
                }
            elif is_cmt_capable:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            elif is_w_grid_consumer:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    w_grid=w_grid_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            else:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner)
        dT_dt = conv_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        # Tracer tendencies. The non-hydrostatic state's tracer ordering
        # is documented on ``NonHydrostaticState`` in
        # ``src/legoesm/core/state.py``:
        #   moist runs → tracers[..., 0] = q_vapor,
        #                tracers[..., 1] = q_cloud,
        #                tracers[..., 2] = q_rain.
        # Slot 0 (``q_v``) carries the convection vapor tendency; slot
        # 1 (``q_c``) carries the convective detrained-condensate
        # source so that microphysics processes it through
        # autoconversion / sedimentation / evaporation rather than the
        # previous instant-fall assumption (Option C). Models with
        # ``n_tracers < 2`` (dry or vapor-only runs) silently omit the
        # ``q_c`` write — there is no slot to receive it.
        dtracers = jnp.zeros_like(tracers)
        if n_tracers > 0:
            dq_v_dt = conv_out.dq_v_dt.reshape(shape_3d)
            dtracers = dtracers.at[..., 0].set(dq_v_dt)
        if n_tracers > 1:
            dq_c_conv_dt = conv_out.dq_c_conv_dt.reshape(shape_3d)
            dtracers = dtracers.at[..., 1].set(dq_c_conv_dt)

        # CMT plumbing — see hydrostatic bridge for rationale.
        if conv_out.du_dt_conv is not None:
            du_dt_data = conv_out.du_dt_conv.reshape(shape_3d)
        else:
            du_dt_data = jnp.zeros(shape_3d, dtype=_state_dtype)
        if conv_out.dv_dt_conv is not None:
            dv_dt_data = conv_out.dv_dt_conv.reshape(shape_3d)
        else:
            dv_dt_data = jnp.zeros(shape_3d, dtype=_state_dtype)

        tendencies = NonHydrostaticTendencies(
            du_dt=Field(data=du_dt_data, name="du_dt_conv", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt_data, name="dv_dt_conv", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_conv", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_conv", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_conv", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_conv", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_conv", dims=dims_tr, units="1/s"),
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_convection(
    convection_config: ConvectionConfig,
    dt: float,
) -> Callable:
    """Create convection physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None, phys_state=None)
               -> (SpectralHydrostaticState, conv_prog_profile_new | None)

    When *phys_state* is passed, the convective prognostic profile is
    read from ``phys_state.conv_prog_profile`` (shape ``(ncol, nlev)``)
    and the updated profile is returned as the second element of the
    result tuple.  See the hydrostatic bridge for the slice/pack
    convention.
    """
    scheme_name, conv_fn, scheme_config = _get_convection_fn(convection_config)
    is_scalar_prognostic = scheme_name in ("mass_flux", "edmf")
    is_profile_prognostic = scheme_name in (
        "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
    )
    is_cmt_capable = scheme_name in ("zhang_mcfarlane", "tiedtke", "bechtold")
    is_w_grid_consumer = scheme_name in ("kain_fritsch",)
    is_stochastic = scheme_name in ("bechtold",)
    prog_key = None
    prog_init = None
    if is_scalar_prognostic:
        if scheme_name == "mass_flux":
            prog_key, prog_init = "M_c", scheme_config.M_c_init
        else:  # edmf
            prog_key, prog_init = "a_u", scheme_config.a_u_init

    def physics_fn(state, grid, sigma_coord, grid_fields=None, phys_state=None):
        from legoesm.atmosphere.dynamics.spectral_pe import (
            SpectralHydrostaticState,
            spectral_pe_to_grid,
        )
        from legoesm.grids.gaussian import sh_analysis_3d

        # 1. Transform spectral state to grid space
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T = fields['T']         # (n_lat, n_lon, nlev)
        p_s = fields['p_s']     # (n_lat, n_lon)

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        # Pressure at full and half levels
        sigma_full = sigma_coord.sigma_full
        sigma_half = sigma_coord.sigma_half
        p_full = p_s[..., None] * sigma_full
        p_half = p_s[..., None] * sigma_half

        # Reshape to columns
        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        # Pin the column-physics dtype to the gridded state precision so
        # we never silently flow x64 zeros into the column path.
        _state_dtype = T.dtype
        # Extract water vapor if spectral state carries tracers.
        if hasattr(state, "tracers") and state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        # Spectral PE columns for CMT-capable schemes.  We pass zeros
        # for u, v because CMT can't currently round-trip through the
        # spectral state (see TODO(spectral-pe-cmt) below).  Passing
        # zeros makes the leaf's CMT contribution identically zero,
        # which is consistent with the spectral PE bridge's existing
        # zero-fill for ``vor_hat`` and ``div_hat`` tendencies.
        if is_cmt_capable:
            u_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
            v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        else:
            u_col = None
            v_col = None

        # Spectral PE has no native ``w`` field — pass zeros to KF and
        # let its trigger run on ``parcel_perturb_T`` alone.  See
        # TODO(kf-w-grid) — eventual hookup to the dycore's vertical
        # mass-flux diagnostic.
        if is_w_grid_consumer:
            w_grid_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        else:
            w_grid_col = None

        conv_prog_out = None
        if conv_fn is None:
            dT_dt = jnp.zeros_like(T)
        elif is_scalar_prognostic:
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile[:, -1]
            else:
                prog_in = jnp.full(ncol, prog_init, dtype=_state_dtype)

            conv_out, prog_new = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                **{prog_key: prog_in},
                dt=dt, config=scheme_config,
            )
            conv_prog_out = jnp.zeros(
                (ncol, nlev), dtype=_state_dtype
            ).at[:, -1].set(prog_new)
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
        elif is_profile_prognostic:
            if phys_state is not None and (
                phys_state.conv_prog_profile.shape == (ncol, nlev)
            ):
                prog_in = phys_state.conv_prog_profile
            else:
                prog_in = jnp.zeros((ncol, nlev), dtype=_state_dtype)
            if is_stochastic:
                if phys_state is not None and (
                    phys_state.conv_stoch_state.shape == (ncol,)
                ):
                    stoch_in = phys_state.conv_stoch_state
                else:
                    stoch_in = jnp.zeros((ncol,), dtype=_state_dtype)
                conv_out, prog_new_profile, stoch_new = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    conv_stoch_state=stoch_in,
                    prng_key=None,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = {
                    "conv_prog_profile": prog_new_profile,
                    "conv_stoch_state": stoch_new,
                }
            elif is_w_grid_consumer:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    w_grid=w_grid_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            else:
                conv_out, prog_new_profile = conv_fn(
                    T=T_col, q_v=q_v_col,
                    p_full=p_full_col, p_half=p_half_col,
                    u=u_col, v=v_col,
                    conv_prog_profile=prog_in,
                    dt=dt, config=scheme_config,
                )
                conv_prog_out = prog_new_profile
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)
        else:
            conv_out = conv_fn(
                T=T_col, q_v=q_v_col,
                p_full=p_full_col, p_half=p_half_col,
                dt=dt, config=scheme_config,
            )
            dT_dt = conv_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Transform T tendency to spectral space.
        # NOTE: this dispatcher already drops ``conv_out.dq_v_dt`` and
        # (post-Option-C) also drops ``conv_out.dq_c_conv_dt`` — the
        # spectral PE state surfaced here doesn't carry tracer
        # tendencies. This is a pre-existing limitation: spectral PE
        # runs effectively dry through the convection coupling.
        # TODO(option-c): when spectral PE gains tracer-tendency
        # plumbing, route ``conv_out.dq_v_dt`` and
        # ``conv_out.dq_c_conv_dt`` here so the convective vapor sink
        # and cloud-water source are no longer silently discarded.
        # TODO(spectral-pe-cmt): convective momentum transport from
        # ``conv_out.du_dt_conv`` / ``dv_dt_conv`` is also dropped here.
        # CMT-capable schemes (Zhang-McFarlane, Tiedtke, Bechtold) emit
        # them in grid space; routing through to spectral vor/div
        # tendencies requires the same SH-analysis pipeline used for
        # ``dT_dt``.  Deferred — consistent with the existing tracer
        # drop above.
        dT_hat = sh_analysis_3d(grid, dT_dt)

        # No wind or surface pressure tendencies from convection
        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        tendencies = SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=zero_3d),
            div_hat=state.div_hat.replace(data=zero_3d),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )
        return tendencies, conv_prog_out

    def reset_state():
        pass

    physics_fn.reset_state = reset_state
    return physics_fn
