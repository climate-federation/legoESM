"""Model integration bridge for microphysics.

Provides `make_microphysics_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.state import (
    HydrostaticState,
    HydrostaticTendencies,
    MPASNonHydrostaticState,
    MPASNonHydrostaticTendencies,
    NonHydrostaticState,
    NonHydrostaticTendencies,
    PlaneNonHydrostaticState,
    PlaneNonHydrostaticTendencies,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.vertical import (
    HeightCoordinate,
    SigmaCoordinate,
    TerrainMetric,
)
from legoesm import constants

from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.atmosphere.dynamics.shared.tracer_positivity import clip_positive
from legoesm.atmosphere.physics._shared import zero_like_tracers
from legoesm.grids.gaussian import sh_analysis_3d
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.clouds.cloud_fraction import cam6_ice_stratus_fraction
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
from legoesm.atmosphere.physics.microphysics.fast_sbm.column import (
    fast_sbm_microphysics,
)
from legoesm.atmosphere.physics.microphysics.sdm import sdm_microphysics
from legoesm.atmosphere.physics.microphysics.ml_emulator import (
    ml_microphysics,
    MicrophysicsEmulator,
)
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


def _get_microphysics_fn(config: MicrophysicsConfig):
    """Select the microphysics backend based on config.scheme.

    Returns
    -------
    scheme_name : str
    micro_fn : callable or None
    scheme_config : NamedTuple or None
    """
    if config.scheme == "kessler":
        return "kessler", kessler_microphysics, config.kessler
    elif config.scheme == "sundqvist":
        return "sundqvist", sundqvist_microphysics, config.sundqvist
    elif config.scheme == "seifert_beheng":
        return "seifert_beheng", seifert_beheng_microphysics, config.seifert_beheng
    elif config.scheme == "morrison":
        return "morrison", morrison_microphysics, config.morrison
    elif config.scheme == "thompson":
        return "thompson", thompson_microphysics, config.thompson
    elif config.scheme == "p3":
        return "p3", p3_microphysics, config.p3
    elif config.scheme == "sdm":
        return "sdm", sdm_microphysics, config.sdm
    elif config.scheme == "fast_sbm":
        return "fast_sbm", fast_sbm_microphysics, config.fast_sbm
    elif config.scheme == "ml_emulator":
        return "ml_emulator", ml_microphysics, config.ml_emulator
    elif config.scheme == "none":
        return "none", None, None
    else:
        raise ValueError(f"Unknown microphysics scheme: {config.scheme!r}")


def get_microphysics_fn(config: MicrophysicsConfig):
    """Public scheme dispatch: ``config.scheme`` → ``(scheme_name, micro_fn,
    scheme_config)``. The swappable per-scheme tendency interface — used by the
    grid adapters here AND external dycores (e.g. the spectral plane LES) so no
    caller imports the private ``_get_microphysics_fn``."""
    return _get_microphysics_fn(config)


# --- Droplet-number storage convention (one place, every lane) -------------
# The dycores advect every tracer with a MASS-MIXING-RATIO operator: the
# quantity is invariant following a parcel.  A per-VOLUME number density is
# not — number conservation in a material volume carries a divergence term the
# operator drops — so a per-volume N_c/N_r advected that way drifts against its
# own mass, and the diagnosed droplet size q*rho/N goes wrong by the density
# change along the trajectory.
#
# So cloud and rain number are STORED and TRANSPORTED per MASS [1/kg], exactly
# as ice number always has been, and converted to the per-VOLUME [1/m^3] the
# microphysics formulas and the radiation effective-radius coupling expect only
# at the bridge, where rho is already in hand.  Ratios of two per-mass scalars
# are then transport-invariant, which is the property the size diagnosis needs.
#
# The alternative — freezing N_c/N_r in place — was tried and is worse: it
# leaves the mass moving while the number stays, so the size error is O(1)
# rather than bounded by the density change.
_PER_MASS_NUMBER_SPECIES = ("N_c", "N_r")


def number_per_mass_to_per_volume(n_per_mass, rho):
    """Stored [1/kg] -> the [1/m^3] the microphysics formulas expect."""
    return n_per_mass * rho


def number_per_volume_to_per_mass(n_per_volume, rho):
    """Microphysics [1/m^3] -> the [1/kg] that transports like a mixing ratio.

    EXACT for a tendency, not an approximation.  What the schemes return is a
    process RATE — activation, collection, evaporation — with no dilution term
    in it, and the air mass of a parcel does not change, so dividing by rho is
    the whole conversion.  Writing the chain rule
    ``d(N/rho)/dt = (dN/dt)/rho - (N/rho^2) drho/dt`` and keeping the second
    term would DOUBLE-COUNT dilution: that term is what cancels the ``-N div(u)``
    the material derivative of a per-volume density carries, and the rate does
    not contain it.  The dycore's mixing-ratio advection carries the density
    change instead (GLM review, 2026-08-14).

    The residual is ordinary operator splitting — the rate is evaluated at the
    start-of-step density — worth about half of ``w*dt/H``: ~1% for stratiform
    ascent at a 1800 s step, and ~15% inside a deep convective core at 300 s,
    where the answer is a shorter physics step, not a chain-rule term.
    """
    return n_per_volume / jnp.maximum(rho, 1.0e-12)


def _min_tracer_slots_for_config(scheme_name: str, scheme_config=None) -> int:
    """Minimum tracer slots for a scheme, including opt-in config features."""
    if (scheme_name == "sdm"
            and getattr(scheme_config, "column_do_coalescence", False)):
        # q_v, q_c, q_r, q_i, q_s, q_g, N_c, N_r. The default SDM column
        # adapter remains condensation-only and needs only q_v/q_c; the
        # reconstructed-box coalescence path writes rain mass and cloud/rain
        # number tendencies, so slot 7 must exist.
        return 8
    return _PLANE_MIN_TRACER_SLOTS[scheme_name]


def min_tracer_slots(scheme_name: str, scheme_config=None) -> int:
    """Public lookup of the minimum tracer-slot count a scheme writes (standard
    slot layout, see ``_PLANE_MIN_TRACER_SLOTS``)."""
    return _min_tracer_slots_for_config(scheme_name, scheme_config)


from legoesm.atmosphere.physics._shared import (
    compute_layer_dz as _compute_heights_from_sigma,
    compute_rho as _compute_rho,
)


def make_microphysics_physics(
    microphysics_config: MicrophysicsConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,  # coeff-ok: default physics timestep [s]
    cloud_config=None,
) -> Callable:
    """Create a physics function for microphysics matching a model's signature.

    Parameters
    ----------
    microphysics_config : MicrophysicsConfig
        Microphysics configuration (selects scheme).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    dt : float
        Model time step [s].
    cloud_config : CloudConfig, optional
        The run's cloud config (scheme ``cam6_clubb``); read only by Morrison
        ``warm_rain_incloud`` for the CAM6 ice-stratus fraction ``aist``.

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    # ``model_type="mpas"`` reuses the hydrostatic factory: the
    # ``_make_hydrostatic_microphysics`` bridge reshapes
    # ``(*shape_2d, nlev)`` to ``(ncol, nlev)`` and never references
    # grid lat/lon — works identically for cubed-sphere ``(face, n, n)``,
    # lat-lon ``(n_lat, n_lon)``, and MPAS Voronoi ``(nCells,)``.
    if model_type in ("hydrostatic", "mpas"):
        return _make_hydrostatic_microphysics(microphysics_config, dt,
                                              cloud_config=cloud_config,
                                              model_type=model_type)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_microphysics(microphysics_config, dt)
    elif model_type == "plane":
        return _make_plane_microphysics(microphysics_config, dt)
    elif model_type == "mpas_nh":
        return _make_mpas_nh_microphysics(microphysics_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_microphysics(microphysics_config, dt)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'plane', "
            f"'mpas_nh', 'spectral_pe', 'mpas'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
    cloud_config=None,
    model_type: str = "hydrostatic",
) -> Callable:
    """Create microphysics physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord) -> HydrostaticTendencies
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    # Aerosol-CCN specified droplet number (Andreae 2009 AOD->CCN): active
    # only for a specified-Nc scheme (predict_Nc=False) that exposes the
    # ``nc_from_aerosol`` switch (currently Morrison).  When active the
    # per-step traced ``forcing["aerosol_od"]`` fills ``hydrometeors.N_c``
    # so the warm-rain KK2000 ``Nc^-1.79`` autoconversion (second indirect
    # effect) sees the aerosol-driven number instead of the constant Nc_0 —
    # the same fill the coupled (cube/lat-lon) physics_pipeline does.  This
    # is the MPAS / hydrostatic combined-physics half of that wiring.
    _nc_from_aerosol = bool(
        getattr(scheme_config, "nc_from_aerosol", False)
        and not getattr(scheme_config, "predict_Nc", False)
    )

    _incloud = bool(getattr(scheme_config, "warm_rain_incloud", False))
    if _incloud and getattr(cloud_config, "scheme", None) != "cam6_clubb":
        raise ValueError(
            "warm_rain_incloud=True uses CAM6's ast = max(alst, aist); the "
            "ice-stratus fraction needs the cam6_clubb cloud config, got "
            f"{getattr(cloud_config, 'scheme', None)!r}.")
    if _incloud and model_type != "mpas":
        # Checked at build time, not the first step: aist's tropopause
        # switch needs MPAS cell latitudes (mesh.latCell).
        raise ValueError(
            "warm_rain_incloud=True is wired on the MPAS lane only (aist "
            f"needs mesh.latCell); got model_type={model_type!r}.")

    def physics_fn(
        state: HydrostaticState,
        grid,
        sigma_coord: SigmaCoordinate,
        forcing=None,
        phys_state=None,
    ) -> HydrostaticTendencies:
        T = state.T.data
        p_s = state.p_s.data

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        # Derive Field metadata from the input state so the returned
        # tendencies match the underlying grid: cubed-sphere uses
        # ("face","x","y",...), lat-lon uses ("lat","lon",...), and
        # MPAS uses ("nCells",...).
        dims_3d = state.T.dims
        dims_2d = state.p_s.dims
        u_shape = state.u.data.shape
        u_dims = state.u.dims
        v_dims = state.v.dims if state.v is not None else None
        v_shape = state.v.data.shape if state.v is not None else None

        # Pin defaulted allocations to the state precision so we never
        # silently flow x64 zeros into the column physics path.
        _state_dtype = T.dtype

        def _zero_dv_dt():
            """``None`` for MPAS (no v), Field of zeros otherwise."""
            if state.v is None:
                return None
            return Field(
                data=jnp.zeros(v_shape, dtype=_state_dtype),
                name="dv_dt_micro", dims=v_dims, units="m/s^2",
            )

        if micro_fn is None:
            return HydrostaticTendencies(
                du_dt=Field(
                    data=jnp.zeros(u_shape, dtype=_state_dtype),
                    name="du_dt_micro", dims=u_dims, units="m/s^2",
                ),
                dv_dt=_zero_dv_dt(),
                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dT_dt_micro", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_micro", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
            )

        # Pressure at full and half levels
        p_full = sigma_coord.pressure_at_full(p_s)
        p_half = sigma_coord.pressure_at_half(p_s)

        # Reshape to columns generically across cubed-sphere
        # ``shape_2d=(6,n,n)``, lat-lon ``(n_lat,n_lon)``, and MPAS
        # ``(nCells,)``.
        ncol = 1
        for s in shape_2d:
            ncol *= int(s)
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        # Helper: extract a tracer from the tracer dict, returning a
        # column-reshaped (ncol, nlev) array clipped to non-negative.
        def _get_tracer(name):
            if state.tracers is not None and name in state.tracers:
                raw = state.tracers[name]
                data = raw.data if hasattr(raw, "data") else raw
                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
            return jnp.zeros((ncol, nlev), dtype=_state_dtype)

        # Extract water vapor from tracers if available; else assume dry.
        q_v_col = _get_tracer("q_v")

        # Moist (virtual-temperature) density and thickness — the helpers
        # apply T_v = T(1+0.608 q_v); a dry rho overestimates density (and the
        # SDM column's reconstructed droplet mass) by ~0.6·q_v.
        rho = _compute_rho(T_col, p_full_col, q_v_col)
        dz = _compute_heights_from_sigma(T_col, p_half_col, q_v_col)

        # Extract actual hydrometeor state from tracers (fall back to zero
        # for any species not present in the tracer registry).
        hydrometeors = HydrometeorState(
            q_c=_get_tracer("q_c"),
            q_r=_get_tracer("q_r"),
            q_i=_get_tracer("q_i"),
            q_s=_get_tracer("q_s"),
            q_g=_get_tracer("q_g"),
            # stored per MASS, used per VOLUME — see _PER_MASS_NUMBER_SPECIES
            N_c=number_per_mass_to_per_volume(_get_tracer("N_c"), rho),
            N_r=number_per_mass_to_per_volume(_get_tracer("N_r"), rho),
            N_i=_get_tracer("N_i"),
        )

        # Aerosol-CCN specified-Nc fill.  Override the (dead-zeros, never
        # evolved under predict_Nc=False) N_c carry with the per-column
        # Andreae (2009) AOD->CCN diagnostic.  The override is unconditional
        # on the N_c carry VALUE (the moisture registry always allocates an
        # N_c slot for Morrison), mirroring the coupled physics_pipeline.
        # Fail fast at trace time if the coupling is configured but no
        # aerosol field was threaded — silently feeding zero N_c (-> Nc_0
        # fallback) is exactly the silent no-op the cube path guards against.
        if _nc_from_aerosol:
            # Route through the activation DISPATCH: "proxy" (default) is
            # byte-identical to the Andreae (2009) ``specified_nc_field``;
            # "arg" switches to Abdul-Razzak & Ghan (2000) modal activation
            # driven by the local (T, p) and a characteristic updraft.
            from legoesm.atmosphere.physics.microphysics.arg_activation import (  # noqa: E501
                ActivationConfig,
                activated_nc_field,
            )
            _activation_cfg = getattr(
                scheme_config, "activation", ActivationConfig())
            _aer_od = forcing.get("aerosol_od") if forcing is not None else None
            # Optional PROGNOSTIC aerosol number [1/m^3]: when a driver stepping
            # the prognostic-aerosol tracer places it in ``forcing`` it drives
            # ARG activation instead of the prescribed config modes (Part-2 ->
            # Part-1 feed). Ignored by the "proxy" scheme (=> byte-identical).
            _aer_num = (
                forcing.get("aerosol_number") if forcing is not None else None)
            if _activation_cfg.scheme == "proxy" and _aer_od is None:
                raise ValueError(
                    "nc_from_aerosol=True but no 'aerosol_od' was passed to "
                    "the microphysics physics_fn via forcing — enable "
                    "external aerosol forcing (--aerosol-forcing external) "
                    "or disable --aerosol-ccn."
                )
            hydrometeors = hydrometeors._replace(
                N_c=activated_nc_field(
                    _activation_cfg, (ncol, nlev),
                    aerosol_od=None if _aer_od is None else jnp.asarray(_aer_od),
                    T=T_col, p=p_full_col,
                    aerosol_number=(
                        None if _aer_num is None else jnp.asarray(_aer_num)),
                    ccn_aod=(forcing.get("aerosol_ccn_aod")
                             if forcing is not None else None),
                ),
            )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half_col, rho, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            _kw = {}
            if _incloud:
                # CLUBB's PDF cloud fraction, written into the carry by the
                # turbulence sub-step that precedes this one in the macmic
                # loop (CAM6: clubb_tend_cam, then MG2 on ast).
                _cf = (None if phys_state is None
                       else getattr(phys_state, "cloud_fraction", None))
                if _cf is None:
                    raise ValueError(
                        "warm_rain_incloud=True but no cloud_fraction carry "
                        "reached the microphysics (needs CLUBB turbulence).")
                if not hasattr(grid, "latCell"):
                    raise ValueError(
                        "warm_rain_incloud=True: the aist tropopause switch "
                        "needs cell latitudes (MPAS mesh latCell).")
                # CAM6 micro_mg_cam.F90:1809-1810 liqcldf = ast, with
                # clubb_intr.F90:2575 ast = max(alst, aist); aist is
                # cldfrc2m aist_vector on the post-CLUBB state (:2556),
                # which is this sub-step's input state.
                _aist = cam6_ice_stratus_fraction(
                    q_v_col, T_col, p_full_col, hydrometeors.q_i,
                    jnp.asarray(grid.latCell).reshape(ncol), cloud_config,
                    p_half_col[:, :-1])
                _kw["cloud_fraction"] = jnp.maximum(
                    jnp.clip(_cf.reshape(ncol, nlev), 0.0, 1.0), _aist)
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half_col, rho, dz, dt, scheme_config, **_kw,
            )

        dT_dt = micro_out.dT_dt.reshape(shape_3d)

        # Propagate tracer tendencies from microphysics backend
        tracer_tends = {
            "q_v": Field(data=micro_out.dq_v_dt.reshape(shape_3d),
                         name="dq_v_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_c": Field(data=micro_out.dq_c_dt.reshape(shape_3d),
                         name="dq_c_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_r": Field(data=micro_out.dq_r_dt.reshape(shape_3d),
                         name="dq_r_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_i": Field(data=micro_out.dq_i_dt.reshape(shape_3d),
                         name="dq_i_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_s": Field(data=micro_out.dq_s_dt.reshape(shape_3d),
                         name="dq_s_dt_micro", dims=dims_3d, units="kg/kg/s"),
            "q_g": Field(data=micro_out.dq_g_dt.reshape(shape_3d),
                         name="dq_g_dt_micro", dims=dims_3d, units="kg/kg/s"),
        }

        # Double-moment number-concentration tendencies.  Emitted only for the
        # prognostic number tracers the dycore actually carries (mirrors the
        # _get_tracer input side); single-moment runs carry none, so the keys
        # are absent.  Previously dropped here, so morrison/seifert_beheng/
        # thompson number concentrations were never advected on the hydrostatic
        # dycore even though the backend computes their tendencies (the
        # nonhydrostatic/plane/MPAS/spectral adapters all propagate them).
        # Units follow MicrophysicsOutput: N_c/N_r are per-VOLUME [1/(m^3 s)]
        # (Seifert-Beheng), N_i is per-MASS [1/(kg s)] (Morrison/Thompson).
        _num_tends = {
            # Converted back to the per-MASS storage convention the dycores
            # transport (see _PER_MASS_NUMBER_SPECIES).
            "N_c": (number_per_volume_to_per_mass(micro_out.dN_c_dt, rho),
                    "1/(kg s)"),
            "N_r": (number_per_volume_to_per_mass(micro_out.dN_r_dt, rho),
                    "1/(kg s)"),
            "N_i": (micro_out.dN_i_dt, "1/(kg s)"),
        }
        for _nname, (_ntend, _nunits) in _num_tends.items():
            if state.tracers is not None and _nname in state.tracers:
                tracer_tends[_nname] = Field(
                    data=_ntend.reshape(shape_3d),
                    name=f"d{_nname}_dt_micro", dims=dims_3d, units=_nunits,
                )

        return HydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(u_shape, dtype=_state_dtype),
                name="du_dt_micro", dims=u_dims, units="m/s^2",
            ),
            dv_dt=_zero_dv_dt(),
            dT_dt=Field(data=dT_dt, name="dT_dt_micro", dims=dims_3d, units="K/s"),
            dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_micro", dims=dims_2d, units="Pa/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
            tracer_tendencies=tracer_tends,
            # Surface precip [kg/m^2/s, +into surface] the scheme already
            # computes (MicrophysicsOutput.precipitation); carried on the
            # tendency so the lean MPAS coupled loop can export it (the RK
            # integrator ignores this diagnostic field, so it is inert to
            # dynamics — byte-identical where unread).
            precip=Field(
                data=micro_out.precipitation.reshape(shape_2d).astype(p_s.dtype),
                name="precip_micro", dims=dims_2d, units="kg/m^2/s"),
            sed_substeps_required=(
                None if micro_out.sed_substeps_required is None else Field(
                    data=micro_out.sed_substeps_required.reshape(shape_2d),
                    name="sed_substeps_required", dims=dims_2d, units="1")),
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    # Advertise the per-step traced ``forcing`` dependency to the
    # combined-physics accumulator ONLY when the aerosol-CCN fill is active
    # (it reads ``forcing["aerosol_od"]``).  Left unset otherwise so a run
    # without --aerosol-ccn is byte-identical (the accumulator calls the fn
    # with the legacy 3-arg signature).
    if _nc_from_aerosol:
        physics_fn._wants_forcing = True
    if _incloud:
        physics_fn._wants_phys_state_ro = True
    return physics_fn


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
) -> Callable:
    """Create microphysics physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric) -> NonHydrostaticTendencies
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> NonHydrostaticTendencies:
        theta_p = state.theta_prime.data
        rho_p = state.rho_prime.data
        tracers = state.tracers.data

        theta_0 = height_coord.theta_ref
        rho_0 = height_coord.rho_ref

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p,
            rho_0 + rho_p,
        )

        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape
        shape_w = state.w.data.shape
        shape_2d = state.phis.data.shape
        n_tracers = tracers.shape[-1] if tracers.ndim >= 5 else 0

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        # Pin defaulted allocations to the state precision so x64 zeros
        # do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype
        if micro_fn is None:
            return NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_micro", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_micro", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
            )

        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]

        # Terrain-aware layer thickness and interface pressure.
        z_half_3d = terrain_metric.z_half_3d
        dz = jnp.abs(z_half_3d[..., :-1] - z_half_3d[..., 1:]).reshape(ncol, nlev)
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=z_half_3d,
        ).reshape(ncol, nlev + 1)

        # Reshape to columns
        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        rho_col = rho_total.reshape(ncol, nlev)

        # Map tracers -> HydrometeorState
        # [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i
        def _get_tracer(idx):
            # Positivity clip on the READ (codex CRM-dycore review): a
            # non-positivity-preserving advection (weno5) can leave q<0,
            # which microphysics reads as a spurious source (negative q_v
            # injects energy on condensation). Guard here so the physics
            # never sees it, matching the name-keyed bridges above.
            if n_tracers > idx:
                return clip_positive(tracers[..., idx]).reshape(ncol, nlev)
            return jnp.zeros((ncol, nlev), dtype=_state_dtype)

        q_v_col = _get_tracer(0)
        hydrometeors = HydrometeorState(
            q_c=_get_tracer(1),
            q_r=_get_tracer(2),
            q_i=_get_tracer(3),
            q_s=_get_tracer(4),
            q_g=_get_tracer(5),
            N_c=number_per_mass_to_per_volume(_get_tracer(6), rho_col),
            N_r=number_per_mass_to_per_volume(_get_tracer(7), rho_col),
            N_i=_get_tracer(8),
        )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt, scheme_config,
            )

        # Convert dT/dt -> dtheta'/dt using local Exner (T = theta * exner).
        dT_dt = micro_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        # Map output fields -> dtracers_dt
        dtracers = jnp.zeros_like(tracers)
        # Tracer mapping: 0=q_v, 1=q_c, 2=q_r, 3=q_i, 4=q_s, 5=q_g, 6=N_c, 7=N_r, 8=N_i
        tend_fields = [
            micro_out.dq_v_dt, micro_out.dq_c_dt, micro_out.dq_r_dt,
            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
            number_per_volume_to_per_mass(micro_out.dN_c_dt, rho_col),
            number_per_volume_to_per_mass(micro_out.dN_r_dt, rho_col),
            micro_out.dN_i_dt,
        ]
        for idx, field in enumerate(tend_fields):
            if n_tracers > idx:
                dtracers = dtracers.at[..., idx].set(field.reshape(shape_3d))

        return NonHydrostaticTendencies(
            du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_micro", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_micro", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_micro", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_micro", dims=dims_tr, units="1/s"),
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Plane (doubly-periodic Cartesian CRM)
# ===========================================================================

# Minimum tracer-slot count each scheme requires to receive its full set of
# tendency outputs without silent truncation. Slot layout is fixed:
# [0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r,
# [8]=N_i. A scheme that writes any tendency at slot ``k`` requires
# ``n_tracers >= k + 1`` or it silently discards that tendency.
#
# Codex review 2026-05-24: previously the plane adapter accepted any
# n_tracers and silently dropped any tendency beyond the supplied slot
# count. That hid scheme/state-shape mismatches (e.g., Morrison on a
# 3-tracer plane state). Now we raise at factory-build time.
_PLANE_MIN_TRACER_SLOTS = {
    "kessler": 3,           # q_v, q_c, q_r
    "sundqvist": 3,         # q_v, q_c, q_r
    "seifert_beheng": 9,    # q_{v,c,r,i,s,g} + N_{c,r,i}
    "morrison": 9,          # q_{v,c,r,i,s,g} + N_{c,r,i}
    "thompson": 9,          # q_{v,c,r,i,s,g} + N_{c,r,i}
    "p3": 9,                # q_{v,c,r,i} + q_rim(s) + B_rim(g) + N_{c,r,i}
    "sdm": 2,               # q_v, q_c by default; opt-in coalescence needs 8
    "fast_sbm": 9,          # q_v,q_c,q_r + N_c,N_r live; ice slots zero
    "ml_emulator": 9,       # generic full layout
    "none": 0,              # no-op
}


def _make_plane_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
) -> Callable:
    """Create microphysics physics_fn for the plane CompressibleEuler model.

    Mirrors :func:`_make_nonhydrostatic_microphysics` but for
    ``PlaneNonHydrostaticState`` (shape ``(ny, nx, nlev)``).

    Signature: ``(state, grid, height_coord, terrain_metric) ->
    PlaneNonHydrostaticTendencies``.

    Raises ``ValueError`` on the first call if ``state.tracers``
    carries fewer slots than the selected scheme writes — see
    ``_PLANE_MIN_TRACER_SLOTS`` for the per-scheme minimum.
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(
        microphysics_config
    )
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]
    _min_slots = _min_tracer_slots_for_config(scheme_name, scheme_config)

    def physics_fn(
        state: PlaneNonHydrostaticState,
        grid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> PlaneNonHydrostaticTendencies:
        theta_p = state.theta_prime.data
        rho_p = state.rho_prime.data
        tracers = state.tracers.data

        theta_0 = height_coord.theta_ref
        rho_0 = height_coord.rho_ref
        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p, rho_0 + rho_p,
        )
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape          # (ny, nx, nlev)
        shape_w = state.w.data.shape      # (ny, nx, nlev+1)
        shape_2d = state.phis.data.shape  # (ny, nx)
        ny, nx = shape_2d
        ncol = ny * nx
        n_tracers = tracers.shape[-1] if tracers.ndim >= 4 else 0

        # Codex review 2026-05-24: hard-fail when the state cannot
        # carry every tendency the scheme writes. Silent truncation
        # (the prior behaviour) hid Morrison-on-3-tracer-state
        # regressions where ice/snow/graupel/number tendencies were
        # dropped without warning.
        if n_tracers < _min_slots:
            raise ValueError(
                f"microphysics scheme {scheme_name!r} writes up to "
                f"{_min_slots} tracer-slot tendencies (slot layout: "
                f"[0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, "
                f"[5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i), but state "
                f"carries only {n_tracers} tracer slots. Allocate "
                f"the state with at least {_min_slots} tracers or "
                f"choose a scheme with fewer requirements (e.g., "
                f"kessler / sundqvist need 3)."
            )

        # Match the PlaneNonHydrostaticState field convention
        # (("y","x","z"), ("y","x","z_half"), ...) so summed tendencies
        # across surface_flux + radiation + microphysics carry
        # consistent metadata for the dycore composer.
        dims_3d = ("y", "x", "z")
        dims_w = ("y", "x", "z_half")
        dims_2d = ("y", "x")
        dims_tr = ("y", "x", "z", "tracer")

        _sd = T.dtype
        _pd = state.phis.data.dtype

        if micro_fn is None:
            return PlaneNonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd),
                            name="du_dt_micro", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd),
                            name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_sd),
                            name="dw_dt_micro", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(
                    data=jnp.zeros(shape_3d, dtype=_sd),
                    name="dtheta_prime_dt_micro", dims=dims_3d, units="K/s",
                ),
                drho_prime_dt=Field(
                    data=jnp.zeros(shape_3d, dtype=_sd),
                    name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s",
                ),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_pd),
                               name="dphis_dt_micro", dims=dims_2d,
                               units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers),
                                  name="dtracers_dt_micro", dims=dims_tr,
                                  units="1/s"),
            )

        z_half_3d = terrain_metric.z_half_3d
        dz = jnp.abs(
            z_half_3d[..., :-1] - z_half_3d[..., 1:]
        ).reshape(ncol, nlev)
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p, rho_full=rho_total, z_half=z_half_3d,
        ).reshape(ncol, nlev + 1)

        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        rho_col = rho_total.reshape(ncol, nlev)

        def _get_tracer(idx):
            # Positivity clip on the READ (codex CRM-dycore review) — see
            # the companion bridge above: microphysics must never read the
            # q<0 a non-positivity-preserving advection (weno5) can leave.
            if n_tracers > idx:
                return clip_positive(tracers[..., idx]).reshape(ncol, nlev)
            return jnp.zeros((ncol, nlev), dtype=_sd)

        q_v_col = _get_tracer(0)
        hydrometeors = HydrometeorState(
            q_c=_get_tracer(1), q_r=_get_tracer(2), q_i=_get_tracer(3),
            q_s=_get_tracer(4), q_g=_get_tracer(5),
            N_c=number_per_mass_to_per_volume(_get_tracer(6), rho_col),
            N_r=number_per_mass_to_per_volume(_get_tracer(7), rho_col),
            N_i=_get_tracer(8),
            # Slot [9] = prognostic snow number ⇒ double-moment snow; absent
            # (≤9 slots) ⇒ None ⇒ single-moment snow (Morrison falls back).
            N_s=(_get_tracer(9) if n_tracers > 9 else None),
            # Slot [10] = prognostic graupel number ⇒ double-moment graupel.
            N_g=(_get_tracer(10) if n_tracers > 10 else None),
        )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt, scheme_config,
            )

        dT_dt = micro_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dtracers = jnp.zeros_like(tracers)
        tend_fields = [
            micro_out.dq_v_dt, micro_out.dq_c_dt, micro_out.dq_r_dt,
            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
            number_per_volume_to_per_mass(micro_out.dN_c_dt, rho_col),
            number_per_volume_to_per_mass(micro_out.dN_r_dt, rho_col),
            micro_out.dN_i_dt,
            # Slot [9] = snow-number tendency (double-moment snow); None for
            # single-moment schemes ⇒ skipped below.
            micro_out.dN_s_dt,
            # Slot [10] = graupel-number tendency (double-moment graupel).
            micro_out.dN_g_dt,
        ]
        for idx, field in enumerate(tend_fields):
            if field is not None and n_tracers > idx:
                dtracers = dtracers.at[..., idx].set(field.reshape(shape_3d))

        return PlaneNonHydrostaticTendencies(
            du_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd),
                        name="du_dt_micro", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_sd),
                        name="dv_dt_micro", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_sd),
                        name="dw_dt_micro", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(
                data=dtheta_prime_dt, name="dtheta_prime_dt_micro",
                dims=dims_3d, units="K/s",
            ),
            drho_prime_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_sd),
                name="drho_prime_dt_micro", dims=dims_3d, units="kg/m^3/s",
            ),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_pd),
                           name="dphis_dt_micro", dims=dims_2d,
                           units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_micro",
                              dims=dims_tr, units="1/s"),
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# MPAS Voronoi non-hydrostatic compressible Euler
# ===========================================================================

# Minimum tracer-slot count per scheme. Identical layout to
# ``_PLANE_MIN_TRACER_SLOTS`` (q_v, q_c, q_r, q_i, q_s, q_g, N_c, N_r,
# N_i). Plane and MPAS NH share the layout because the tracer column
# axis is the same.
_MPAS_NH_MIN_TRACER_SLOTS = dict(_PLANE_MIN_TRACER_SLOTS)


def _make_mpas_nh_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
) -> Callable:
    """Create microphysics physics_fn for the MPAS NH dycore.

    Mirrors :func:`_make_nonhydrostatic_microphysics` but for
    ``MPASNonHydrostaticState`` (cell-centred quantities on Voronoi
    cells, shape ``(nCells, nlev)``).

    Signature: ``(state, mesh, height_coord, terrain_metric) ->
    MPASNonHydrostaticTendencies``.

    Raises ``ValueError`` on the first call if the state carries
    fewer tracer slots than the selected scheme writes — see
    ``_MPAS_NH_MIN_TRACER_SLOTS`` for the per-scheme minimum.
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(
        microphysics_config
    )
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]
    _min_slots = _min_tracer_slots_for_config(scheme_name, scheme_config)

    def physics_fn(
        state: MPASNonHydrostaticState,
        mesh,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
    ) -> MPASNonHydrostaticTendencies:
        theta_p = state.theta_prime.data
        rho_p = state.rho_prime.data
        tracers = state.tracers.data       # (nCells, nlev, n_tracers)

        theta_0 = height_coord.theta_ref
        rho_0 = height_coord.rho_ref
        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p, rho_0 + rho_p,
        )
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_cell_3d = (mesh.nCells, nlev)
        shape_edge_3d = state.u.data.shape
        shape_w = state.w.data.shape
        shape_2d = (mesh.nCells,)
        ncol = mesh.nCells
        n_tracers = tracers.shape[-1] if tracers.ndim >= 3 else 0

        if n_tracers < _min_slots:
            raise ValueError(
                f"microphysics scheme {scheme_name!r} writes up to "
                f"{_min_slots} tracer-slot tendencies (slot layout: "
                f"[0]=q_v, [1]=q_c, [2]=q_r, [3]=q_i, [4]=q_s, "
                f"[5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i), but MPAS NH "
                f"state carries only {n_tracers} tracer slots. "
                f"Allocate at least {_min_slots} tracers or pick a "
                f"scheme with fewer requirements (kessler / sundqvist "
                f"need 3)."
            )

        dims_cell = ("nCells", "nlev")
        dims_edge = ("nEdges", "nlev")
        dims_w = ("nCells", "nlev_half")
        dims_2d = ("nCells",)
        dims_tr = ("nCells", "nlev", "tracer")
        _sd = T.dtype
        _pd = state.phis.data.dtype

        if micro_fn is None:
            return MPASNonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_edge_3d, dtype=_sd),
                            name="du_dt_micro", dims=dims_edge,
                            units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_sd),
                            name="dw_dt_micro", dims=dims_w,
                            units="m/s^2"),
                dtheta_prime_dt=Field(
                    data=jnp.zeros(shape_cell_3d, dtype=_sd),
                    name="dtheta_prime_dt_micro",
                    dims=dims_cell, units="K/s",
                ),
                drho_prime_dt=Field(
                    data=jnp.zeros(shape_cell_3d, dtype=_sd),
                    name="drho_prime_dt_micro",
                    dims=dims_cell, units="kg/m^3/s",
                ),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_pd),
                               name="dphis_dt_micro", dims=dims_2d,
                               units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers),
                                  name="dtracers_dt_micro", dims=dims_tr,
                                  units="1/s"),
            )

        # Terrain-aware layer thickness + half-level pressure.
        z_half_3d = terrain_metric.z_half_3d         # (nCells, nlev+1)
        dz = jnp.abs(z_half_3d[..., :-1] - z_half_3d[..., 1:])
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p, rho_full=rho_total, z_half=z_half_3d,
        )

        T_col = T
        p_full_col = p
        rho_col = rho_total

        def _get_tracer(idx):
            # Positivity clip on the READ (codex CRM-dycore review): MPAS-NH
            # tracer transport (centered edge averages + vertical advection)
            # is not positivity-preserving either, so microphysics must never
            # read a q<0 — matching the plane / NH / spectral bridges.
            if n_tracers > idx:
                return clip_positive(tracers[..., idx])
            return jnp.zeros((ncol, nlev), dtype=_sd)

        q_v_col = _get_tracer(0)
        hydrometeors = HydrometeorState(
            q_c=_get_tracer(1), q_r=_get_tracer(2), q_i=_get_tracer(3),
            q_s=_get_tracer(4), q_g=_get_tracer(5),
            N_c=number_per_mass_to_per_volume(_get_tracer(6), rho_col),
            N_r=number_per_mass_to_per_volume(_get_tracer(7), rho_col),
            N_i=_get_tracer(8),
        )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half, rho_col, dz, dt, scheme_config,
            )

        dT_dt = micro_out.dT_dt
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dtracers = jnp.zeros_like(tracers)
        tend_fields = [
            micro_out.dq_v_dt, micro_out.dq_c_dt, micro_out.dq_r_dt,
            micro_out.dq_i_dt, micro_out.dq_s_dt, micro_out.dq_g_dt,
            number_per_volume_to_per_mass(micro_out.dN_c_dt, rho_col),
            number_per_volume_to_per_mass(micro_out.dN_r_dt, rho_col),
            micro_out.dN_i_dt,
        ]
        for idx, field in enumerate(tend_fields):
            if n_tracers > idx:
                dtracers = dtracers.at[..., idx].set(field)

        return MPASNonHydrostaticTendencies(
            du_dt=Field(data=jnp.zeros(shape_edge_3d, dtype=_sd),
                        name="du_dt_micro", dims=dims_edge,
                        units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_sd),
                        name="dw_dt_micro", dims=dims_w,
                        units="m/s^2"),
            dtheta_prime_dt=Field(
                data=dtheta_prime_dt, name="dtheta_prime_dt_micro",
                dims=dims_cell, units="K/s",
            ),
            drho_prime_dt=Field(
                data=jnp.zeros(shape_cell_3d, dtype=_sd),
                name="drho_prime_dt_micro",
                dims=dims_cell, units="kg/m^3/s",
            ),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_pd),
                           name="dphis_dt_micro", dims=dims_2d,
                           units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_micro",
                              dims=dims_tr, units="1/s"),
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_microphysics(
    microphysics_config: MicrophysicsConfig,
    dt: float,
) -> Callable:
    """Create microphysics physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None) -> SpectralHydrostaticState

    The bridge pulls ``q_v`` and the full hydrometeor state out of
    ``state.tracers`` (when present), runs the column microphysics
    backend, and returns a ``SpectralHydrostaticState`` whose ``T_hat``
    carries the spectral latent-heating tendency *and* whose ``tracers``
    dict carries grid-space ``dq_v_dt`` / ``dq_c_dt`` / ``dq_r_dt`` /
    etc.  The dycore RHS (``spectral_pe_tendencies``) adds these tracer
    tendencies to its own advective tendencies during the SSP-RK stages.
    """
    scheme_name, micro_fn, scheme_config = _get_microphysics_fn(microphysics_config)
    is_ml = scheme_name == "ml_emulator"
    _ml_model_cache = [None]

    # Tracer key → MicrophysicsOutput attribute name.  Mirrors the
    # ``HydrometeorState`` field layout in ``microphysics/output.py``
    # plus ``q_v``.  The dycore RHS only flows tendencies for keys that
    # exist on the input ``state.tracers``; missing keys are silently
    # dropped (no carry to write into).
    _TRACER_TEND_MAP = {
        "q_v": "dq_v_dt",
        "q_c": "dq_c_dt",
        "q_r": "dq_r_dt",
        "q_i": "dq_i_dt",
        "q_s": "dq_s_dt",
        "q_g": "dq_g_dt",
        "N_c": "dN_c_dt",
        "N_r": "dN_r_dt",
        "N_i": "dN_i_dt",
    }

    def physics_fn(state, grid, sigma_coord, grid_fields=None):
        # Transform spectral state to grid space
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        T = fields['T']
        p_s = fields['p_s']

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)
        # Pin the column-physics dtype to the gridded state precision so
        # we do not silently flow x64 zeros into the column path.
        _state_dtype = T.dtype

        if micro_fn is None:
            # Mirror the input tracer pytree shape with zeros so the
            # orchestrator's accumulator and the dycore RHS see a
            # consistent tendency structure even when microphysics is
            # disabled.
            return SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
                tracers=zero_like_tracers(state.tracers),
            )

        # Pressure at full and half levels
        p_full = sigma_coord.pressure_at_full(p_s)
        p_half = sigma_coord.pressure_at_half(p_s)

        # Reshape to columns
        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Pull tracer fields out of ``state.tracers`` and reshape to the
        # column-physics ``(ncol, nlev)`` layout.  Backend microphysics
        # schemes assume non-negative mixing ratios, so clip on the way
        # in (matches the hydrostatic bridge's ``_get_tracer``).
        def _get_tracer(name):
            if state.tracers is not None and name in state.tracers:
                raw = state.tracers[name]
                data = raw.data if hasattr(raw, "data") else raw
                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
            return jnp.zeros((ncol, nlev), dtype=_state_dtype)

        q_v_col = _get_tracer("q_v")

        # Moist (virtual-temperature) density/thickness — see the hydrostatic
        # bridge note above.
        rho = _compute_rho(T_col, p_full_col, q_v_col)
        dz = _compute_heights_from_sigma(T_col, p_half_col, q_v_col)

        hydrometeors = HydrometeorState(
            q_c=_get_tracer("q_c"),
            q_r=_get_tracer("q_r"),
            q_i=_get_tracer("q_i"),
            q_s=_get_tracer("q_s"),
            q_g=_get_tracer("q_g"),
            # stored per MASS, used per VOLUME — see _PER_MASS_NUMBER_SPECIES
            N_c=number_per_mass_to_per_volume(_get_tracer("N_c"), rho),
            N_r=number_per_mass_to_per_volume(_get_tracer("N_r"), rho),
            N_i=_get_tracer("N_i"),
        )

        if is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = MicrophysicsEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output, key=key,
                )
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half_col, rho, dz, dt,
                scheme_config, _ml_model_cache[0],
            )
        else:
            micro_out = micro_fn(
                T_col, q_v_col, hydrometeors,
                p_full_col, p_half_col, rho, dz, dt, scheme_config,
            )

        dT_dt = micro_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Transform T tendency to spectral space
        dT_hat = sh_analysis_3d(grid, dT_dt)

        # Build the tracer tendency dict in grid-space ``(n_lat, n_lon,
        # nlev)`` layout, matching ``SpectralHydrostaticState.tracers``.
        # Wrap each tendency back into the same container type as the
        # input state's tracer (``Field`` vs raw ``jax.Array``) so the
        # SSP-RK ``tree.map`` pytree leaves line up.  Untouched tracer
        # keys are mirrored as zeros via ``zero_like_tracers``.
        tracers_tend = None
        if state.tracers is not None:
            tt = {}
            for name, attr in _TRACER_TEND_MAP.items():
                if name not in state.tracers:
                    continue
                template = state.tracers[name]
                _tend_col = getattr(micro_out, attr)
                if name in _PER_MASS_NUMBER_SPECIES:
                    # Back to the per-MASS storage the dycore transports.
                    _tend_col = number_per_volume_to_per_mass(_tend_col, rho)
                tend_grid = _tend_col.reshape(
                    n_lat, n_lon, nlev,
                )
                if hasattr(template, "data") and hasattr(template, "replace"):
                    tt[name] = template.replace(
                        data=tend_grid.astype(template.data.dtype),
                    )
                else:
                    tt[name] = tend_grid.astype(template.dtype)
            # Mirror any untouched tracer keys (e.g. a passive scalar
            # the user attached) as zeros so the orchestrator's
            # accumulator and the dycore RHS see a complete pytree.
            zeros = zero_like_tracers(state.tracers)
            if zeros is not None:
                for k, zv in zeros.items():
                    tt.setdefault(k, zv)
            tracers_tend = tt

        return SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=zero_3d),
            div_hat=state.div_hat.replace(data=zero_3d),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            tracers=tracers_tend,
        )

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn
