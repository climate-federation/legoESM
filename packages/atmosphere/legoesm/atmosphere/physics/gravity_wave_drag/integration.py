"""Model integration bridge for gravity wave drag.

Provides `make_gwd_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

GWD produces momentum tendencies (du_dt, dv_dt) and temperature
tendencies (dT_dt). For the spectral PE dycore, wind tendencies
are projected to spectral vorticity/divergence.

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

from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.dynamics.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import (
    sh_analysis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
)
from legoesm.atmosphere.physics.gravity_wave_drag.rayleigh import rayleigh_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.lindzen import lindzen_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import mcfarlane_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.hines import hines_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
    prognostic_spectral_gwd,
)
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import e3sm_cam_gwd
from legoesm.atmosphere.physics.gravity_wave_drag.ml_emulator import (
    ml_gwd,
    GWDEmulator,
)
from legoesm.atmosphere.physics.gravity_wave_drag.output import GWDOutput
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


def get_gwd_fn(config: GravityWaveDragConfig):
    """Select the GWD backend based on config.scheme."""
    if config.scheme == "rayleigh":
        return "rayleigh", rayleigh_gwd, config.rayleigh
    elif config.scheme == "lindzen":
        return "lindzen", lindzen_gwd, config.lindzen
    elif config.scheme == "mcfarlane":
        return "mcfarlane", mcfarlane_gwd, config.mcfarlane
    elif config.scheme == "hines":
        return "hines", hines_gwd, config.hines
    elif config.scheme == "prognostic_spectral":
        return "prognostic_spectral", prognostic_spectral_gwd, config.prognostic_spectral
    elif config.scheme == "e3sm_cam":
        return "e3sm_cam", e3sm_cam_gwd, config.e3sm_cam
    elif config.scheme == "ml_emulator":
        return "ml_emulator", ml_gwd, config.ml_emulator
    elif config.scheme == "none":
        return "none", None, None
    elif "+" in config.scheme:
        # Composite: a ``+``-joined list of GWD sources whose tendencies are
        # summed (e.g. ``mcfarlane+prognostic_spectral`` = orographic +
        # non-orographic, issue #834).  ``_combined_gwd`` fans out to each
        # part's single-scheme backend and adds their du/dv/dT; the full
        # ``config`` is returned as the "scheme config" because the executor
        # needs every part's sub-config (and reads ``config.scheme`` to know
        # the parts).  Validate the composition HERE (factory chokepoint,
        # independent of ExperimentConfig.validate_strict) so an invalid
        # composite fails loudly instead of hitting a low-level TypeError or
        # silently corrupting the spectrum carry inside the executor.
        _validate_gwd_composite(config.scheme)
        return config.scheme, _combined_gwd, config
    else:
        raise ValueError(f"Unknown GWD scheme: {config.scheme!r}")


# ---------------------------------------------------------------------------
# Composite (multi-source) GWD — issue #834
# ---------------------------------------------------------------------------
#
# Orographic (lindzen/mcfarlane) and non-orographic (rayleigh/hines/
# prognostic_spectral) drag parameterize DISTINCT gravity-wave populations
# launched by different sources (flow over topography vs convection/fronts),
# so as body forces on the same column their momentum and thermal tendencies
# add linearly.  A ``+``-joined ``gravity_wave_drag`` string composes them.
#
# At most ONE stateful part (``prognostic_spectral``, which carries a
# wave-action spectrum through ``phys_state.gwd_spectrum``) may appear; its
# spectrum threads through ``spectrum_in``/``spectrum_out``.  The stateless
# parts hold no per-step state.  ``e3sm_cam`` / ``ml_emulator`` are NOT
# composable here (they need extra per-column source fields / a network
# module the composite signature does not carry).
_GWD_COMPOSABLE_STATELESS = ("rayleigh", "lindzen", "mcfarlane", "hines")
_GWD_COMPOSABLE_STATEFUL = ("prognostic_spectral",)
_GWD_COMPOSABLE = _GWD_COMPOSABLE_STATELESS + _GWD_COMPOSABLE_STATEFUL
# Orographic parts accept the optional per-column subgrid-topo stddev.
_GWD_OROGRAPHIC_PARTS = ("lindzen", "mcfarlane")


def gwd_carries_spectrum(scheme: str) -> bool:
    """True if a GWD scheme string threads a prognostic wave-action spectrum.

    Accepts a single scheme (``"prognostic_spectral"``) or a ``+``-composite
    (``"mcfarlane+prognostic_spectral"``).  Shared by the driver's
    prognostic-carry plumbing (``physics_pipeline``) and the spectrum seeding
    (``init_physics_state``) so the "does this scheme carry a spectrum?"
    predicate is defined in exactly one place.
    """
    return any(p in _GWD_COMPOSABLE_STATEFUL for p in scheme.split("+"))


def _validate_gwd_composite(scheme: str) -> None:
    """Raise ValueError unless ``scheme`` is a well-formed GWD composite.

    Factory-level dispatch hardening (independent of
    ``ExperimentConfig.validate_strict``): every part must be composable and at
    most one stateful source (``prognostic_spectral``) may appear (its
    wave-action spectrum is a single carry).  Without this guard the executor
    would fan out to a non-composable backend with the wrong signature
    (``ml_emulator`` needs a network module; ``e3sm_cam`` needs extra source
    fields) — a low-level ``TypeError`` — or run two spectral sources from the
    same ``spectrum_in`` and return only the last ``spectrum_out``, silently
    corrupting the carry.
    """
    parts = scheme.split("+")
    bad = [p for p in parts if p not in _GWD_COMPOSABLE]
    if bad:
        raise ValueError(
            f"Non-composable GWD part(s) {bad} in composite {scheme!r}; "
            f"composable sources are {_GWD_COMPOSABLE} "
            f"(e3sm_cam / ml_emulator are not composable)."
        )
    n_stateful = sum(p in _GWD_COMPOSABLE_STATEFUL for p in parts)
    if n_stateful > 1:
        raise ValueError(
            f"A GWD composite may contain at most one stateful source "
            f"{_GWD_COMPOSABLE_STATEFUL} (its wave-action spectrum is a single "
            f"carry), got {n_stateful} in {scheme!r}."
        )


def _combined_gwd(
    u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt,
    config, spectrum_in, h_topo_col=None,
):
    """Sum the tendencies of a ``+``-composite GWD scheme (issue #834).

    ``config`` is the full :class:`GravityWaveDragConfig`; its ``scheme``
    field (e.g. ``"mcfarlane+prognostic_spectral"``) names the parts and its
    sub-configs supply each backend's parameters.  Returns
    ``(GWDOutput, spectrum_out)`` mirroring ``prognostic_spectral_gwd`` so the
    composite is a drop-in for the prognostic call path.  ``spectrum_out`` is
    the advanced spectrum when a ``prognostic_spectral`` part is present, else
    ``spectrum_in`` unchanged (harmless passthrough for a stateless
    composite).
    """
    parts = config.scheme.split("+")
    du = dv = dT = eps = None
    spectrum_out = spectrum_in
    for part in parts:
        # Per-part single-scheme dispatch reuses get_gwd_fn (which raises on an
        # unknown part) → no duplicated scheme→backend table.
        _name, fn, sub_cfg = get_gwd_fn(config._replace(scheme=part))
        if part in _GWD_COMPOSABLE_STATEFUL:
            out, spectrum_out = fn(
                u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt,
                sub_cfg, spectrum_in,
            )
        elif part in _GWD_OROGRAPHIC_PARTS:
            out = fn(
                u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt,
                sub_cfg, h_topo_col=h_topo_col,
            )
        else:
            out = fn(
                u, v, T, p_full, p_half, z_full, z_half, rho, lat, dt,
                sub_cfg,
            )
        du = out.du_dt if du is None else du + out.du_dt
        dv = out.dv_dt if dv is None else dv + out.dv_dt
        dT = out.dT_dt if dT is None else dT + out.dT_dt
        eps = out.eps_gwd if eps is None else eps + out.eps_gwd
    return GWDOutput(du_dt=du, dv_dt=dv, dT_dt=dT, eps_gwd=eps), spectrum_out


def _combined_spec_in(config, phys_state, ncol):
    """Read or seed the wave-action spectrum for a spectrum-carrying composite.

    Mirrors the pure-``prognostic_spectral`` spec-in logic in each factory,
    reading the spectrum sub-config from ``config.prognostic_spectral`` (the
    composite's full config).  Reseeds to ``launch_flux`` when absent or the
    column count changed.
    """
    pcfg = config.prognostic_spectral
    shape = (ncol, pcfg.n_azimuths, pcfg.n_wavenumbers)
    if phys_state is not None:
        spec_in = phys_state.gwd_spectrum
        if spec_in.shape[0] != ncol:
            spec_in = jnp.full(shape, pcfg.launch_flux)
        return spec_in
    return jnp.full(shape, pcfg.launch_flux)


from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma as _compute_heights_from_sigma,
    compute_rho as _compute_rho,
)


def make_gwd_physics(
    gwd_config: GravityWaveDragConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,  # coeff-ok: default physics timestep [s]
) -> Callable:
    """Create a physics function for GWD matching a model's signature.

    Parameters
    ----------
    gwd_config : GravityWaveDragConfig
        GWD configuration (selects scheme).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    dt : float
        Model time step [s].

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    if model_type == "hydrostatic":
        return _make_hydrostatic_gwd(gwd_config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_gwd(gwd_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_gwd(gwd_config, dt)
    elif model_type == "mpas":
        return _make_mpas_gwd(gwd_config, dt)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_gwd(
    gwd_config: GravityWaveDragConfig,
    dt: float,
) -> Callable:
    """Create GWD physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord, phys_state=None)
               -> HydrostaticTendencies

    When *phys_state* is passed, the GWD wave action spectrum is read
    from ``phys_state.gwd_spectrum`` and the updated spectrum is
    returned as the second element of the result tuple.
    """
    scheme_name, gwd_fn, scheme_config = get_gwd_fn(gwd_config)
    is_prognostic = scheme_name == "prognostic_spectral"
    is_ml = scheme_name == "ml_emulator"
    # ``e3sm_cam`` shares the orographic launch signature (accepts the
    # optional per-column ``h_topo_col`` subgrid orographic stddev keyword);
    # its config selects the orographic vs frontal/convective source internally.
    #
    # NOTE: the ``e3sm_cam`` frontal source needs a per-column frontogenesis
    # function ``frontgf_col`` and the convective (Beres) source needs a
    # convective heating profile ``netdt_col`` (and, for bit-faithfulness, the
    # offline ``mfcc`` table).  This GWD factory's ``(state, grid, sigma)``
    # signature does not currently carry those fields, so from here only the
    # orographic source is driven; the frontal/convective sources are exercised
    # through the public ``e3sm_cam_gwd(..., frontgf_col=, netdt_col=,
    # mfcc_table=)`` entry point (wiring the convective-heating coupling through
    # the physics pipeline is a separate integration task and must not reach
    # into the convection package from here).
    is_orographic = scheme_name in ("lindzen", "mcfarlane", "e3sm_cam")
    # Composite (issue #834): a ``+``-joined scheme runs several sources and
    # sums their tendencies; ``combined_spectrum`` marks the sub-case that also
    # threads the prognostic wave-action spectrum.
    is_combined = "+" in scheme_name
    combined_spectrum = is_combined and gwd_carries_spectrum(scheme_name)
    _ml_model_cache = [None]

    def physics_fn(
        state: HydrostaticState,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        phys_state=None,
    ):
        gwd_spectrum_out = None
        T = state.T.data
        u = state.u.data
        v = state.v.data
        p_s = state.p_s.data

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")
        # Pin defaulted allocations to the state precision so x64 zeros
        # do not silently flow into the column physics path.
        _state_dtype = T.dtype
        _ps_dtype = p_s.dtype

        if gwd_fn is None:
            tendencies = HydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dT_dt_gwd", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_gwd", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
            )
            return tendencies, gwd_spectrum_out

        # Read q_v from tracers (if present) so the moist forms of the
        # hydrostatic-height and ideal-gas-density helpers are used —
        # GWD stress depends on N (Brunt-Väisälä) and ρ which drift by
        # ~1 % when computed dry in moist tropical columns.  Audit
        # 2026-05-12 MEDIUM #8.
        q_v_col = None
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col, q_v=q_v_col)
        rho = _compute_rho(T_col, p_full_col, q_v=q_v_col)

        # Latitude: use grid.lat_face if available, else zeros
        lat = _get_lat_hydrostatic(grid, ncol)

        if is_combined:
            # Composite GWD (#834): sum McFarlane (orographic) + the
            # non-orographic source(s); thread the spectrum when present.
            h_topo_col = _extract_subgrid_topo_stddev(grid, ncol)
            spec_in = (
                _combined_spec_in(scheme_config, phys_state, ncol)
                if combined_spectrum else None
            )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config, spec_in,
                h_topo_col=h_topo_col,
            )
            if combined_spectrum:
                gwd_spectrum_out = spec_new
        elif is_prognostic:
            sc = scheme_config
            # NOTE: ``spec_in`` (and the prognostic spectrum more
            # broadly) is intentionally allocated at the JAX default
            # float dtype rather than ``_state_dtype``.  The internal
            # propagation in ``prognostic_spectral_gwd`` builds
            # ``tau_sat`` from sigma-coord-derived quantities at
            # compute precision; pinning ``spec_in`` to storage
            # precision would force a carry-input/output dtype
            # mismatch in the lax.scan body.  Keep the spectrum at
            # compute precision end-to-end.
            if phys_state is not None:
                spec_in = phys_state.gwd_spectrum
                if spec_in.shape[0] != ncol:
                    spec_in = jnp.full(
                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux,
                    )
            else:
                spec_in = jnp.full(
                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux,
                )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, sc, spec_in,
            )
            gwd_spectrum_out = spec_new
        elif is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = GWDEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output,
                    key=key,
                )
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
                _ml_model_cache[0],
            )
        elif is_orographic:
            h_topo_col = _extract_subgrid_topo_stddev(grid, ncol)
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
                h_topo_col=h_topo_col,
            )
        else:
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
            )

        du_dt = gwd_out.du_dt.reshape(shape_3d)
        dv_dt = gwd_out.dv_dt.reshape(shape_3d)
        dT_dt = gwd_out.dT_dt.reshape(shape_3d)

        tendencies = HydrostaticTendencies(
            du_dt=Field(data=du_dt, name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
            dT_dt=Field(data=dT_dt, name="dT_dt_gwd", dims=dims_3d, units="K/s"),
            dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dp_s_dt_gwd", dims=dims_2d, units="Pa/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_ps_dtype), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
        )
        return tendencies, gwd_spectrum_out

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


def _extract_subgrid_topo_stddev(grid, ncol):
    """Return a per-column subgrid orographic stddev [m], or ``None``.

    The orographic GWD schemes (Lindzen, McFarlane) accept an optional
    per-column ``h_topo_col`` driving the launch stress
    ``tau_0 ∝ h_topo²``.  Previously the schemes used a single global
    ``config.h_topo = 500 m`` everywhere — mountainous and oceanic
    columns alike — which audit 2026-05-12 MEDIUM #9 flagged as a
    physical idealization.  This helper looks for a per-column
    standard deviation of the resolved or subgrid topography stored
    on the grid (canonical attribute name
    ``subgrid_topo_stddev``).  Users wanting realistic orographic
    forcing should set this attribute from a GMTED2010-style subgrid
    statistics dataset before constructing the model driver.  When the
    attribute is absent we return ``None`` so the schemes fall back to
    their scalar ``config.h_topo`` (legacy behaviour).
    """
    raw = getattr(grid, "subgrid_topo_stddev", None)
    if raw is None:
        return None
    return jnp.asarray(raw).reshape(-1)[:ncol]


def _make_mpas_gwd(
    gwd_config: GravityWaveDragConfig,
    dt: float,
) -> Callable:
    """Create GWD physics_fn for MPASPrimitiveEquationModel.

    Mirrors the turbulence MPAS bridge: reconstruct cell-centered winds
    from the prognostic edge-normal velocity (Perot 2000), run the
    column GWD backend on cell quantities, project the cell-centered
    wind tendencies back to edge-normal form via ``angleEdge``.
    Audit 2026-05-12 finding MEDIUM #10.

    Prognostic and ML GWD variants are unsupported on MPAS for now
    (they require additional pytree carries not yet wired through the
    MPAS PE driver); the dispatch falls back to a clear NotImplementedError
    at runtime if such a scheme is selected.
    """
    scheme_name, gwd_fn, scheme_config = get_gwd_fn(gwd_config)
    is_prognostic = scheme_name == "prognostic_spectral"
    is_ml = scheme_name == "ml_emulator"
    # ``e3sm_cam`` shares the orographic launch signature (accepts the
    # optional per-column ``h_topo_col`` subgrid orographic stddev keyword);
    # its config selects the orographic vs frontal source internally.
    is_orographic = scheme_name in ("lindzen", "mcfarlane", "e3sm_cam")
    # Composite (issue #834): a stateless ``+``-composite (e.g.
    # ``hines+mcfarlane``) runs on MPAS by summing its parts; a
    # spectrum-carrying composite does NOT (the wave-action spectrum is not
    # threaded through the MPAS PE driver pytree — same limitation as pure
    # ``prognostic_spectral`` below).
    is_combined = "+" in scheme_name
    combined_spectrum = is_combined and gwd_carries_spectrum(scheme_name)

    def physics_fn(state, mesh, sigma_coord, phys_state=None):
        from legoesm.grids.voronoi import reconstruct_cell_velocity

        gwd_spectrum_out = None
        if gwd_fn is None:
            zero_edges = jnp.zeros_like(state.u.data)
            zero_cells = jnp.zeros_like(state.T.data)
            zero_ps = jnp.zeros_like(state.p_s.data)
            tendencies = HydrostaticTendencies(
                du_dt=state.u.replace(data=zero_edges),
                dv_dt=None,
                dT_dt=state.T.replace(data=zero_cells),
                dp_s_dt=state.p_s.replace(data=zero_ps),
                dphis_dt=state.phis.replace(data=zero_ps),
            )
            return tendencies, gwd_spectrum_out

        if is_prognostic or is_ml or combined_spectrum:
            raise NotImplementedError(
                f"GWD scheme {scheme_name!r} on MPAS Voronoi mesh "
                "needs the prognostic spectrum / ML model state to "
                "be threaded through the MPAS driver pytree.  Use a "
                "diagnostic scheme (rayleigh, lindzen, mcfarlane, "
                "hines) — or a stateless '+'-composite of those — on "
                "MPAS until that wiring lands."
            )

        u_edge = state.u.data
        T = state.T.data
        p_s = state.p_s.data
        nlev = sigma_coord.n_levels
        nCells = T.shape[0]

        u_cell, v_cell = reconstruct_cell_velocity(u_edge, mesh)

        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)
        T_col = T.reshape(nCells, nlev)
        u_col = u_cell.reshape(nCells, nlev)
        v_col = v_cell.reshape(nCells, nlev)
        p_full_col = p_full.reshape(nCells, nlev)
        p_half_col = p_half.reshape(nCells, nlev + 1)

        q_v_col = None
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(nCells, nlev)

        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col, q_v=q_v_col)
        rho = _compute_rho(T_col, p_full_col, q_v=q_v_col)

        lat = jnp.asarray(mesh.latCell)

        if is_combined:
            # Stateless composite only (spectrum-carrying case raised above).
            # Orographic parts use the per-cell subgrid-topo stddev.
            h_topo_col = _extract_subgrid_topo_stddev(mesh, nCells)
            gwd_out, _ = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config, None,
                h_topo_col=h_topo_col,
            )
        elif is_orographic:
            h_topo_col = _extract_subgrid_topo_stddev(mesh, nCells)
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
                h_topo_col=h_topo_col,
            )
        else:
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
            )

        # ``mesh.cellsOnEdge`` has shape ``(2, nEdges)``; slice axis 0
        # to get per-edge cell-index vectors.
        du_cell = gwd_out.du_dt
        dv_cell = gwd_out.dv_dt
        c0 = mesh.cellsOnEdge[0]
        c1 = mesh.cellsOnEdge[1]
        du_e_east = 0.5 * (du_cell[c0] + du_cell[c1])
        dv_e_north = 0.5 * (dv_cell[c0] + dv_cell[c1])
        angle = mesh.angleEdge[:, None]
        du_edge_normal = du_e_east * jnp.cos(angle) + dv_e_north * jnp.sin(angle)
        dT_cell = gwd_out.dT_dt

        zero_ps = jnp.zeros_like(p_s)
        tendencies = HydrostaticTendencies(
            du_dt=state.u.replace(data=du_edge_normal, name="du_dt_gwd"),
            dv_dt=None,
            dT_dt=state.T.replace(data=dT_cell, name="dT_dt_gwd"),
            dp_s_dt=state.p_s.replace(data=zero_ps, name="dp_s_dt_gwd"),
            dphis_dt=state.phis.replace(data=zero_ps, name="dphis_dt_gwd"),
        )
        return tendencies, gwd_spectrum_out

    def reset_state():
        return None

    physics_fn.reset_state = reset_state
    return physics_fn


def _get_lat_hydrostatic(grid, ncol):
    """Extract latitude array for hydrostatic columns."""
    return jnp.asarray(grid.grid_lat).reshape(-1)[:ncol]


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_gwd(
    gwd_config: GravityWaveDragConfig,
    dt: float,
) -> Callable:
    """Create GWD physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric, phys_state=None)
               -> NonHydrostaticTendencies
    """
    scheme_name, gwd_fn, scheme_config = get_gwd_fn(gwd_config)
    is_prognostic = scheme_name == "prognostic_spectral"
    is_ml = scheme_name == "ml_emulator"
    # Composite (issue #834): sum multiple sources; thread the spectrum when a
    # prognostic_spectral part is present.
    is_combined = "+" in scheme_name
    combined_spectrum = is_combined and gwd_carries_spectrum(scheme_name)
    _ml_model_cache = [None]

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        phys_state=None,
    ):
        gwd_spectrum_out = None
        theta_p = state.theta_prime.data
        rho_p = state.rho_prime.data
        u_data = state.u.data
        v_data = state.v.data
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

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        dims_tr = ("face", "x", "y", "level", "tracer")

        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        # Pin defaulted allocations to the state precision so x64 zeros
        # do not silently widen the NH GWD tendency struct.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype

        if gwd_fn is None:
            tendencies = NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_gwd", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_gwd", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_gwd", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_gwd", dims=dims_tr, units="1/s"),
            )
            return tendencies, gwd_spectrum_out

        # Terrain-aware heights and interface pressure.
        z_full = terrain_metric.z_full_3d.reshape(ncol, nlev)
        z_half = terrain_metric.z_half_3d.reshape(ncol, nlev + 1)
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        ).reshape(ncol, nlev + 1)

        T_col = T.reshape(ncol, nlev)
        u_col = u_data.reshape(ncol, nlev)
        v_col = v_data.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        rho_col = rho_total.reshape(ncol, nlev)

        lat = _get_lat_hydrostatic(grid, ncol)

        if is_combined:
            # Composite GWD (#834): sum orographic + non-orographic; thread the
            # spectrum when present.  NH drives orographic parts with the
            # scalar ``config.h_topo`` (h_topo_col=None), matching this
            # factory's single-scheme orographic path.
            spec_in = (
                _combined_spec_in(scheme_config, phys_state, ncol)
                if combined_spectrum else None
            )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half,
                z_full, z_half, rho_col, lat, dt, scheme_config, spec_in,
            )
            if combined_spectrum:
                gwd_spectrum_out = spec_new
        elif is_prognostic:
            sc = scheme_config
            if phys_state is not None:
                spec_in = phys_state.gwd_spectrum
                if spec_in.shape[0] != ncol:
                    spec_in = jnp.full(
                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                    )
            else:
                spec_in = jnp.full(
                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half,
                z_full, z_half, rho_col, lat, dt, sc, spec_in,
            )
            gwd_spectrum_out = spec_new
        elif is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = GWDEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output,
                    key=key,
                )
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half,
                z_full, z_half, rho_col, lat, dt, scheme_config,
                _ml_model_cache[0],
            )
        else:
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half,
                z_full, z_half, rho_col, lat, dt, scheme_config,
            )

        du_dt = gwd_out.du_dt.reshape(shape_3d)
        dv_dt = gwd_out.dv_dt.reshape(shape_3d)
        dT_dt = gwd_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        tendencies = NonHydrostaticTendencies(
            du_dt=Field(data=du_dt, name="du_dt_gwd", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt_gwd", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_gwd", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_gwd", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_gwd", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_gwd", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_gwd", dims=dims_tr, units="1/s"),
        )
        return tendencies, gwd_spectrum_out

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_gwd(
    gwd_config: GravityWaveDragConfig,
    dt: float,
) -> Callable:
    """Create GWD physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None, phys_state=None)
               -> SpectralHydrostaticState
    """
    scheme_name, gwd_fn, scheme_config = get_gwd_fn(gwd_config)
    is_prognostic = scheme_name == "prognostic_spectral"
    is_ml = scheme_name == "ml_emulator"
    # Composite (issue #834): sum multiple sources; thread the spectrum when a
    # prognostic_spectral part is present.
    is_combined = "+" in scheme_name
    combined_spectrum = is_combined and gwd_carries_spectrum(scheme_name)
    _ml_model_cache = [None]

    def physics_fn(state, grid, sigma_coord, grid_fields=None, phys_state=None):
        gwd_spectrum_out = None
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        u = fields['u']
        v = fields['v']
        T = fields['T']
        p_s = fields['p_s']

        nlev = sigma_coord.n_levels
        n_lat, n_lon = p_s.shape

        sigma_full = sigma_coord.sigma_full
        sigma_half = sigma_coord.sigma_half
        p_full = p_s[..., None] * sigma_full
        p_half = p_s[..., None] * sigma_half

        ncol = n_lat * n_lon
        T_col = T.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        if gwd_fn is None:
            tendencies = SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            )
            return tendencies, gwd_spectrum_out

        # Read q_v from tracers when available so moist forms feed
        # N and ρ inside GWD (audit 2026-05-12 MEDIUM #8).
        q_v_col_gwd = None
        if hasattr(state, "tracers") and state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            # Tracer pytree carries gridded data; reshape to columns.
            q_v_col_gwd = _qv_data.reshape(ncol, nlev)
        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col, q_v=q_v_col_gwd)
        rho = _compute_rho(T_col, p_full_col, q_v=q_v_col_gwd)

        # Latitude from Gaussian grid
        lat = jnp.broadcast_to(grid.lat[:, None], (n_lat, n_lon)).reshape(ncol)

        if is_combined:
            # Composite GWD (#834): sum orographic + non-orographic; thread the
            # spectrum when present.  Orographic parts use the scalar
            # ``config.h_topo`` (h_topo_col=None), matching this factory's
            # single-scheme orographic path.
            spec_in = (
                _combined_spec_in(scheme_config, phys_state, ncol)
                if combined_spectrum else None
            )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config, spec_in,
            )
            if combined_spectrum:
                gwd_spectrum_out = spec_new
        elif is_prognostic:
            sc = scheme_config
            if phys_state is not None:
                spec_in = phys_state.gwd_spectrum
                if spec_in.shape[0] != ncol:
                    spec_in = jnp.full(
                        (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                    )
            else:
                spec_in = jnp.full(
                    (ncol, sc.n_azimuths, sc.n_wavenumbers), sc.launch_flux
                )
            gwd_out, spec_new = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, sc, spec_in,
            )
            gwd_spectrum_out = spec_new
        elif is_ml:
            if _ml_model_cache[0] is None:
                key = jax.random.PRNGKey(scheme_config.seed)
                _ml_model_cache[0] = GWDEmulator(
                    scheme_config.n_input, scheme_config.n_hidden,
                    scheme_config.n_layers, scheme_config.n_output,
                    key=key,
                )
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
                _ml_model_cache[0],
            )
        else:
            gwd_out = gwd_fn(
                u_col, v_col, T_col, p_full_col, p_half_col,
                z_full, z_half, rho, lat, dt, scheme_config,
            )

        du_dt = gwd_out.du_dt.reshape(n_lat, n_lon, nlev)
        dv_dt = gwd_out.dv_dt.reshape(n_lat, n_lon, nlev)
        dT_dt = gwd_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Project wind tendencies to spectral vorticity/divergence
        a = grid.radius
        im_over_a = 1j * grid.ms.astype(jnp.float64) / a
        one_over_a = 1.0 / a

        cos_lat_3d = grid.cos_lat[:, None, None]
        du_cos = du_dt * cos_lat_3d
        dv_cos = dv_dt * cos_lat_3d

        dvor_hat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
            + one_over_a * sh_analysis_dmu_3d(grid, du_cos)
        )
        ddiv_hat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
            - one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
        )

        dT_hat = sh_analysis_3d(grid, dT_dt)

        tendencies = SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=dvor_hat),
            div_hat=state.div_hat.replace(data=ddiv_hat),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )
        return tendencies, gwd_spectrum_out

    def reset_state():
        _ml_model_cache[0] = None

    physics_fn.reset_state = reset_state
    return physics_fn
