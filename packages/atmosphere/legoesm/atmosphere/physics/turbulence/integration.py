"""Model integration bridge for turbulence.

Provides `make_turbulence_physics()`, a factory that returns a physics
function matching each dynamical core's `step_with_physics` signature.

Key difference from radiation/convection: turbulence produces momentum
tendencies (du_dt, dv_dt) that are nonzero. For the spectral PE dycore,
these must be projected to spectral vorticity/divergence tendencies.

Supported model types:
- "hydrostatic"  : PrimitiveEquationModel (sigma coordinates)
- "nonhydrostatic": CompressibleEulerModel (z* coordinates)
- "spectral_pe"  : SpectralPEModel (Gaussian grid + sigma coordinates)
"""

from __future__ import annotations

import inspect
from typing import Callable, NamedTuple

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

from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralHydrostaticState,
    spectral_pe_to_grid,
)
from legoesm.grids.gaussian import (
    sh_analysis_3d,
    sh_analysis_oc2_3d,
    sh_analysis_dmu_3d,
)
from legoesm.atmosphere.physics.turbulence.smagorinsky import smagorinsky_turbulence
from legoesm.atmosphere.physics.turbulence.louis import louis_turbulence
from legoesm.atmosphere.physics.turbulence.tke import tke_turbulence
from legoesm.atmosphere.physics.turbulence.mynn25 import mynn25_turbulence
from legoesm.atmosphere.physics.turbulence.clubb import clubb_turbulence
from legoesm.atmosphere.physics.turbulence.clubb_lite import clubb_lite_turbulence
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    holtslag_boville_turbulence,
)
from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence
from legoesm.atmosphere.physics.turbulence.edmf import edmf_turbulence
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import (
    pressure_from_eos,
    reconstruct_half_level_pressure_hydrostatic,
    sanitize_theta_rho,
)


class TurbulenceSchemeTraits(NamedTuple):
    """Static carry-plumbing traits for a turbulence scheme.

    Single source of truth for "which schemes carry a prognostic
    energy field, and in which ``PhysicsState`` slot" — consumed by
    ``init_physics_state`` (seeding), the combined-physics dispatcher,
    the driver physics pipeline, and the driver's stateful-physics
    guard, so the four call paths cannot drift (issue #405: the guard
    originally omitted ``clubb_lite`` because the set was re-derived
    by hand).

    Attributes
    ----------
    carries_energy : bool
        True when the scheme threads a prognostic turbulent-energy
        carry between steps (kernel takes the energy array as its 5th
        positional argument and returns ``(TurbulenceOutput, energy_new)``).
    energy_field : str or None
        ``PhysicsState`` field holding the carry: ``"tke"`` for the
        MY-2.5 family (tke / clubb_lite / edmf), ``"qke"`` for
        MYNN-2.5 (``q² = 2·TKE`` — distinct slot so a restart-time
        scheme switch cannot feed the wrong moment as energy).
        ``None`` for diagnostic schemes.
    """
    carries_energy: bool
    energy_field: str | None


_ENERGY_FIELD_BY_SCHEME = {
    "tke": "tke",
    "clubb_lite": "tke",
    "clubb": "tke",
    "edmf": "tke",
    "mynn25": "qke",
}


def turbulence_scheme_traits(scheme_name: str) -> TurbulenceSchemeTraits:
    """Return the static carry traits for *scheme_name*.

    Unknown schemes report ``carries_energy=False`` (mirrors
    ``convection_scheme_traits``); :func:`get_turbulence_fn` is the
    authority that rejects unknown scheme names.
    """
    field = _ENERGY_FIELD_BY_SCHEME.get(scheme_name)
    return TurbulenceSchemeTraits(
        carries_energy=field is not None,
        energy_field=field,
    )


def materialize_sub_config(config: TurbulenceConfig) -> TurbulenceConfig:
    """Return *config* with the ACTIVE scheme's sub-config materialized.

    ``TurbulenceConfig.clubb`` defaults to ``None`` and dispatch substitutes a
    fresh ``CLUBBConfig()`` -- so ``None`` does NOT mean "no config", it means
    "the default one".  Anything reasoning about what the atmosphere will
    actually run has to make that substitution the same way, or it reasons about
    a config the model never uses.

    That gap was a real bug: ``apply_surface_flux_config`` bailed out on the
    ``None`` sub-config, so ``surface_bulk_scheme="coare3"`` was SILENTLY
    IGNORED under ``turbulence="clubb"`` (the run used CLUBB's own default
    constant surface layer), and the air-sea guard could not see the resulting
    split against a COARE3 ocean tile (codex).  Both call sites now route
    through this one function so they cannot drift.

    ``scheme="none"`` keeps ``None``: it genuinely has no sub-config.  Every
    other scheme already carries a materialized sub-config by default.
    """
    if config.scheme == "clubb" and config.clubb is None:
        # Deferred: clubb.py imports turbulence/config.py, so a top-level
        # import here would close a cycle.
        from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
        return config._replace(clubb=CLUBBConfig())
    return config


def get_turbulence_fn(config: TurbulenceConfig):
    """Select the turbulence backend based on config.scheme."""
    if config.scheme == "smagorinsky":
        return "smagorinsky", smagorinsky_turbulence, config.smagorinsky
    elif config.scheme == "louis":
        return "louis", louis_turbulence, config.louis
    elif config.scheme == "tke":
        return "tke", tke_turbulence, config.tke
    elif config.scheme == "mynn25":
        return "mynn25", mynn25_turbulence, config.mynn25
    elif config.scheme == "clubb_lite":
        return "clubb_lite", clubb_lite_turbulence, config.clubb_lite
    elif config.scheme == "clubb":
        clubb_cfg = materialize_sub_config(config).clubb
        if getattr(clubb_cfg, "prognostic", False):
            from legoesm.atmosphere.physics.turbulence.clubb import (
                clubb_turbulence_prognostic,
            )
            return "clubb", clubb_turbulence_prognostic, clubb_cfg
        return "clubb", clubb_turbulence, clubb_cfg
    elif config.scheme == "holtslag_boville":
        return "holtslag_boville", holtslag_boville_turbulence, config.holtslag_boville
    elif config.scheme == "ysu":
        return "ysu", ysu_turbulence, config.ysu
    elif config.scheme == "edmf":
        return "edmf", edmf_turbulence, config.edmf
    elif config.scheme == "none":
        return "none", None, None
    else:
        raise ValueError(f"Unknown turbulence scheme: {config.scheme!r}")


def turbulence_carry_field(scheme_name: str, scheme_config) -> str:
    """PhysicsState field that carries this turbulence scheme's prognostic state.

    ``mynn25`` → ``qke``; prognostic CLUBB (``scheme="clubb"`` with
    ``CLUBBConfig.prognostic=True``) → ``clubb_moments`` (the packed
    CLUBBMomentState); every other TKE-carrying scheme → ``tke``. Used by both the
    per-model physics_fn (to READ the carry) and ``combined.py`` (to STORE it),
    so the read/write slots always agree.
    """
    if scheme_name == "mynn25":
        return "qke"
    if scheme_name == "clubb" and getattr(scheme_config, "prognostic", False):
        return "clubb_moments"
    return "tke"


def _read_turb_carry(phys_state, carry_field, ncol, nlev, scheme_config, dtype):
    """Read (or freshly seed) the turbulence carry from ``phys_state``.

    Returns the array passed as the carry arg to ``turb_fn`` — ``(ncol, nlev)``
    TKE/qke, or the packed CLUBB moments ``(ncol, 15, nlev+1)`` for prognostic
    CLUBB. Re-seeds when ``phys_state`` is absent or the stored slot has the wrong
    shape (e.g. a minimal placeholder from a non-clubb init / scheme switch)."""
    if carry_field == "clubb_moments":
        from legoesm.atmosphere.physics.turbulence.clubb import (
            init_clubb_moments,
            pack_clubb_moments,
        )
        expected = (ncol, 15, nlev + 1)
        stored = getattr(phys_state, "clubb_moments", None) if phys_state else None
        if stored is not None and stored.shape == expected:
            return stored
        if stored is not None:
            # A wrong-shape carry (e.g. the minimal (ncol,1,1) placeholder) means
            # PhysicsState was NOT initialised for prognostic CLUBB. Fail fast at
            # trace time rather than silently RESIZE the carry inside the JIT step
            # (which would recompile next step) or silently re-seed the moments.
            raise ValueError(
                f"prognostic CLUBB expects PhysicsState.clubb_moments of shape "
                f"{expected}, got {tuple(stored.shape)} — call init_physics_state "
                f"with TurbulenceConfig(scheme='clubb', clubb=CLUBBConfig("
                f"prognostic=True)) so the carry is seeded at the right shape.")
        # No phys_state at all (a one-off, non-looped physics_fn call): seed fresh.
        return pack_clubb_moments(
            init_clubb_moments(ncol, nlev, scheme_config, dtype=dtype))
    floor = jnp.full((ncol, nlev), scheme_config.tke_min, dtype=dtype)
    if phys_state is None:
        return floor
    carry = getattr(phys_state, carry_field)
    return carry if carry.shape == (ncol, nlev) else floor


from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma as _compute_heights_from_sigma,
    compute_rho as _compute_rho,
    exner_function as _exner_function,
)


def _resolve_T_sfc(T_col, phys_state):
    """Pick the surface temperature seen by the bulk-flux call.

    Default convention (preserved bit-for-bit by 3-D runs): ``T_sfc ==
    T_col[:, -1]`` — the lowest air temperature stands in for the
    surface skin temperature.  The SCM driver may override this on a
    per-column basis by writing ``phys_state.surface_T_sfc_override``;
    the override uses the finite ``NO_SFC_T_OVERRIDE`` sentinel (a large
    negative value below any physical surface temperature) for "fall
    back" — keeping the state finite (#911).  Legacy checkpoints written
    with the old ``NaN`` sentinel still resolve to the fallback here
    (``NaN > min`` is False), so restarts stay backward-compatible.

    This is what gives ``SCMForcing(prescribe="T_s")`` a non-zero
    bulk-flux gradient when paired with a turbulence scheme: anchoring
    only ``T[..., -1]`` to the prescribed value would collapse
    ``T_sfc − T[..., -1]`` to zero and silently suppress the sensible
    heat flux (Phase B codex iter-1 high finding).
    """
    from legoesm.atmosphere.physics.physics_state import (
        SFC_T_OVERRIDE_VALID_MIN,
    )
    fallback = T_col[:, -1]
    if phys_state is None:
        return fallback
    override = getattr(phys_state, "surface_T_sfc_override", None)
    if override is None:
        return fallback
    # A physical override exceeds the threshold; the sentinel (and any legacy
    # NaN) does not -> fall back.  ``NaN > x`` is False, so old checkpoints work.
    return jnp.where(override > SFC_T_OVERRIDE_VALID_MIN, override, fallback)


def _carry_update_with_cloud_fraction(carry_field, carry_val, turb_out):
    """Package a turbulence scheme's carry for combined.py's ``phys_updates``.

    A moist higher-order closure (CLUBB) diagnoses a sub-grid PDF cloud fraction
    (``turb_out.cloud_fraction``, shape ``(ncol, nlev)``).  When present we hand
    back a MULTI-FIELD dict — combined.py's ``isinstance(field_val, dict)``
    branch merges every key into ``PhysicsState`` — so radiation
    (``cloud_scheme="clubb"``) can read ``phys_state.cloud_fraction`` instead of
    the RH-diagnosed grid-scale one.  The scheme's prognostic carry
    (``carry_field`` -> ``carry_val``: tke / qke / clubb_moments) is co-located
    under its own key so it still evolves across the step.

    Schemes with no PDF cloud closure leave ``cloud_fraction=None`` -> we return
    the plain ``carry_val``, byte-identical to the pre-existing (bare value, not
    dict) contract.  combined.py only merges the dict when ``field_name`` (the
    registered carry field) is not None, which holds for every cf-producing
    scheme today (CLUBB carries ``tke``); a diagnostic scheme that ever set cf
    with ``carry_field=None`` would need combined.py's ``field_name is not None``
    guard relaxed for the dict branch first.
    """
    if turb_out.cloud_fraction is None:
        return carry_val
    updates = {"cloud_fraction": turb_out.cloud_fraction}
    if carry_field is not None and carry_val is not None:
        updates[carry_field] = carry_val
    return updates


def make_turbulence_physics(
    turbulence_config: TurbulenceConfig,
    model_type: str = "hydrostatic",
    dt: float = 300.0,  # coeff-ok: default physics timestep [s]
    f_land=None,
    land_beta: float = 1.0,
) -> Callable:
    """Create a physics function for turbulence matching a model's signature.

    Parameters
    ----------
    turbulence_config : TurbulenceConfig
        Turbulence configuration (selects scheme).
    model_type : str
        One of "hydrostatic", "nonhydrostatic", "spectral_pe".
    dt : float
        Model time step [s].
    f_land : array or None
        Static per-cell land fraction in [0, 1] for the MPAS land surface
        boundary (baked as a closure constant).  MPAS-only: the FV /
        spectral pipelines carry their own land tile, so passing it for any
        other ``model_type`` raises rather than being silently inert.
    land_beta : float
        Land evaporation efficiency in [0, 1] applied on the ``f_land``
        fraction of the surface humidity (MPAS-only, with ``f_land``).
        ``1.0`` (default) keeps the saturated wet surface, byte-identical.

    Returns
    -------
    Callable
        Physics function with the correct signature for the model.
    """
    if model_type != "mpas" and (f_land is not None or land_beta != 1.0):
        raise ValueError(
            "f_land/land_beta are the MPAS land surface boundary knobs; the "
            f"{model_type!r} pipeline has its own land tile (they would be "
            "silently inert here). Drop them or use model_type='mpas'."
        )
    if model_type == "hydrostatic":
        return _make_hydrostatic_turbulence(turbulence_config, dt)
    elif model_type == "nonhydrostatic":
        return _make_nonhydrostatic_turbulence(turbulence_config, dt)
    elif model_type == "spectral_pe":
        return _make_spectral_pe_turbulence(turbulence_config, dt)
    elif model_type == "mpas":
        # UNBLOCKED: ``MPASPrimitiveEquationModel.step`` now (a) carries
        # ``state.tracers`` (q_v advected by the dycore — Phase B) and (b)
        # applies physics OPERATOR-SPLIT once per dt with a ``PhysicsState``
        # carry threaded in/out, so the prognostic TKE field and the q_v
        # diffusion tendency are no longer dropped.  ``_make_mpas_turbulence``
        # (Perot edge→cell reconstruction + column backend + cell→edge
        # projection) is the implementation; it reads the prescribed surface
        # temperature from the per-step ``forcing["T_sfc"]`` (AMIP SST) when
        # supplied, so the surface sensible/latent fluxes are SST-driven.
        _mpas_turb_fn = _make_mpas_turbulence(
            turbulence_config, dt, f_land=f_land, land_beta=land_beta)
        # Forcing-aware: the combined-physics dispatcher forwards
        # ``forcing["T_sfc"]`` to fns advertising this (mirrors radiation).
        _mpas_turb_fn._wants_forcing = True
        return _mpas_turb_fn
    else:
        raise ValueError(
            f"Unknown model_type: {model_type!r}. "
            f"Choose from 'hydrostatic', 'nonhydrostatic', 'spectral_pe', 'mpas'."
        )


# ===========================================================================
# Hydrostatic PE
# ===========================================================================

def _make_hydrostatic_turbulence(
    turbulence_config: TurbulenceConfig,
    dt: float,
) -> Callable:
    """Create turbulence physics_fn for PrimitiveEquationModel.

    Signature: (state, grid, sigma_coord, phys_state=None) -> HydrostaticTendencies

    When *phys_state* (a ``PhysicsState``) is passed, TKE is read from
    ``phys_state.tke`` and the updated TKE is returned as the second
    element of the result tuple.

    Turbulence produces nonzero du_dt, dv_dt (unlike convection/radiation).
    When tracers are available, q_v is read from ``state.tracers["q_v"]``
    and the moisture tendency ``dq_v_dt`` is returned via ``tracer_tendencies``.
    """
    scheme_name, turb_fn, scheme_config = get_turbulence_fn(turbulence_config)
    needs_tke = turbulence_scheme_traits(scheme_name).carries_energy
    carry_field = turbulence_carry_field(scheme_name, scheme_config)

    def physics_fn(
        state: HydrostaticState,
        grid: CubedSphereGrid,
        sigma_coord: SigmaCoordinate,
        phys_state=None,
    ):
        tke_out = None
        T = state.T.data          # (6, n, n, nlev)
        u = state.u.data
        v = state.v.data
        p_s = state.p_s.data      # (6, n, n)

        nlev = sigma_coord.n_levels
        shape_3d = T.shape
        shape_2d = p_s.shape

        # Pressure
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)

        # Reshape to columns
        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        T_col = T.reshape(ncol, nlev)
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)

        # Extract water vapor from tracers if available; else assume dry.
        # Pin all defaulted allocations to the state precision so we never
        # silently promote an x64 zero into a float32 column path (which
        # poisons downstream scan carries with mixed-precision dtypes).
        _state_dtype = T.dtype
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(ncol, nlev)
        else:
            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        dims_3d = ("face", "x", "y", "level")
        dims_2d = ("face", "x", "y")

        if turb_fn is None:
            return HydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_turb", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_turb", dims=dims_3d, units="m/s^2"),
                dT_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dT_dt_turb", dims=dims_3d, units="K/s"),
                dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_turb", dims=dims_2d, units="Pa/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_turb", dims=dims_2d, units="m^2/s^3"),
            ), tke_out

        # Heights and density.  Use the moist (virtual-temperature)
        # forms when q_v is available so that hydrostatic layer
        # thicknesses and ρ in the diffusion solver are consistent
        # with the actual column moisture (audit 2026-05-12
        # MEDIUM #8 — dry forms drift ~1 % in tropical moist
        # columns and produce inconsistent K · ∂φ/∂z fluxes vs the
        # mass-weighted surface BC).
        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col, q_v=q_v_col)
        rho = _compute_rho(T_col, p_full_col, q_v=q_v_col)

        # Surface conditions
        T_sfc = _resolve_T_sfc(T_col, phys_state)
        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])

        if needs_tke:
            # Read (or seed) the prognostic carry from PhysicsState — tke/qke, or
            # the packed CLUBB moments (ncol,15,nlev+1) for prognostic clubb.
            tke_in = _read_turb_carry(
                phys_state, carry_field, ncol, nlev, scheme_config, _state_dtype)

            turb_out, tke_new = turb_fn(
                u_col, v_col, T_col, q_v_col, tke_in,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )
            tke_out = tke_new
        else:
            # No land-flux hand-over on this lane: the structured-grid driver
            # has no forcing channel carrying the surface scheme's own
            # turbulent fluxes, so the scheme computes its own from T_sfc/q_sfc.
            turb_out = turb_fn(
                u_col, v_col, T_col, q_v_col,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )

        du_dt = turb_out.du_dt.reshape(shape_3d)
        dv_dt = turb_out.dv_dt.reshape(shape_3d)
        dT_dt = turb_out.dT_dt.reshape(shape_3d)

        # Propagate moisture tendency from turbulence backend
        tracer_tends = {
            "q_v": Field(
                data=turb_out.dq_v_dt.reshape(shape_3d),
                name="dq_v_dt_turb",
                dims=dims_3d, units="kg/kg/s",
            ),
        }

        tendencies = HydrostaticTendencies(
            du_dt=Field(data=du_dt, name="du_dt_turb", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt_turb", dims=dims_3d, units="m/s^2"),
            dT_dt=Field(data=dT_dt, name="dT_dt_turb", dims=dims_3d, units="K/s"),
            dp_s_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dp_s_dt_turb", dims=dims_2d, units="Pa/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=p_s.dtype), name="dphis_dt_turb", dims=dims_2d, units="m^2/s^3"),
            tracer_tendencies=tracer_tends,
        )
        return tendencies, _carry_update_with_cloud_fraction(
            carry_field, tke_out, turb_out)

    def reset_state():
        return None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# MPAS Voronoi (hydrostatic primitive eqn)
# ===========================================================================

def _make_mpas_turbulence(
    turbulence_config: TurbulenceConfig,
    dt: float,
    f_land=None,
    land_beta: float = 1.0,
) -> Callable:
    """Create turbulence physics_fn for MPASPrimitiveEquationModel.

    Signature: ``(state, mesh, sigma_coord, phys_state=None) -> (HydrostaticTendencies, tke_out)``.

    MPAS stores the prognostic horizontal velocity as the edge-normal
    component ``state.u`` of shape ``(nEdges, nlev)`` with ``state.v is None``.
    Column physics needs cell-centered ``u``/``v`` to compute shear and
    eddy diffusivities.  We use the Perot (2000) area-weighted
    edge→cell reconstruction (:func:`legoesm.grids.voronoi.reconstruct_cell_velocity`)
    to recover ``(u_east, v_north)`` at cells, run the existing
    column turbulence backend, then convert the cell-centered wind
    tendencies back to edge-normal tendencies via
    ``du_normal = du_east * cosθ + dv_north * sinθ`` where θ is
    ``mesh.angleEdge``.  The averaging cells-flanking-edge is the
    canonical MPAS C-grid projection; round-trip on a uniform field
    is the identity to within floating-point error.

    Audit 2026-05-12 finding MEDIUM #10.
    """
    scheme_name, turb_fn, scheme_config = get_turbulence_fn(turbulence_config)
    needs_tke = turbulence_scheme_traits(scheme_name).carries_energy
    carry_field = turbulence_carry_field(scheme_name, scheme_config)
    # Which schemes can be HANDED a surface flux instead of computing their
    # own?  Only the non-TKE kernels that declare the keyword: louis and
    # diagnostic clubb today.  smagorinsky, holtslag_boville and ysu do not,
    # and the TKE-carrying kernels are called on a branch that never forwards
    # it.  Resolved once here, from the static kernel, so land coupling with an
    # unsupported scheme fails at build time with a scheme name rather than as
    # a TypeError inside a traced column.
    _accepts_surface_flux = (
        (not needs_tke)
        and turb_fn is not None
        and "surface_flux" in inspect.signature(turb_fn).parameters
    )

    def physics_fn(state, mesh, sigma_coord, phys_state=None, forcing=None):
        from legoesm.grids.voronoi import reconstruct_cell_velocity

        tke_out = None
        if turb_fn is None:
            # Build zero-tendency in MPAS layout (edge-centric u).
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
            return tendencies, tke_out

        u_edge = state.u.data       # (nEdges, nlev)
        T = state.T.data            # (nCells, nlev)
        p_s = state.p_s.data        # (nCells,)
        nlev = sigma_coord.n_levels
        nCells = T.shape[0]

        # Edge → cell wind reconstruction (Perot 2000).
        u_cell, v_cell = reconstruct_cell_velocity(u_edge, mesh)

        # Pressures
        p_full = pressure_from_sigma(sigma_coord.sigma_full, p_s)  # (nCells, nlev)
        p_half = pressure_from_sigma(sigma_coord.sigma_half, p_s)  # (nCells, nlev+1)

        # Column-format inputs (already 1D × nlev, so reshape is a no-op).
        T_col = T.reshape(nCells, nlev)
        u_col = u_cell.reshape(nCells, nlev)
        v_col = v_cell.reshape(nCells, nlev)
        p_full_col = p_full.reshape(nCells, nlev)
        p_half_col = p_half.reshape(nCells, nlev + 1)

        _state_dtype = T.dtype
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
            q_v_col = _qv_data.reshape(nCells, nlev)
        else:
            q_v_col = jnp.zeros((nCells, nlev), dtype=_state_dtype)

        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col, q_v=q_v_col)
        rho = _compute_rho(T_col, p_full_col, q_v=q_v_col)

        # Surface temperature for the bulk fluxes: the prescribed SST from the
        # per-step traced ``forcing["T_sfc"]`` (AMIP path) when supplied, so
        # the sensible/latent surface fluxes are SST-driven and consistent
        # with the radiation surface boundary; else the SCM/phys-carry value.
        if forcing is not None and forcing.get("T_sfc") is not None:
            T_sfc = jnp.asarray(forcing["T_sfc"]).reshape(nCells)
        else:
            T_sfc = _resolve_T_sfc(T_col, phys_state)
        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])
        # MPAS land surface boundary: throttle the LAND fraction's surface
        # humidity gradient by a soil-moisture availability beta instead of
        # the saturated infinite-swamp value the nearest-ocean SST fill
        # otherwise implies.  Two sources, traced wins:
        #   1. ``forcing["beta_land"]`` — TRACED per-cell root-zone beta_soil
        #      from the interactive multilayer land (#1312 phase 2b; one-step
        #      lag, updated by the driver loop each step without retrace).
        #   2. the STATIC ``land_beta`` build-time knob (mpas_land_beta).
        # Both gates are static Python ``if``s (dict-key membership is part
        # of the forcing pytree structure; the knob is a build-time closure
        # const — the JAX feature-gating exception): defaults keep these
        # branches out of the trace entirely, byte-identical to before.
        # ``forcing["q_sfc_land"]`` — the land scheme's SOLVED boundary
        # humidity (for the two-leaf canopy the canopy-air humidity out of the
        # stomatal + soil + aerodynamic resistance network), used DIRECTLY as
        # the land fraction's surface humidity.  Review killed the first
        # attempt at this handoff, which round-tripped the humidity through an
        # effective beta: the inversion and this reconstruction anchored their
        # saturation at different pressures and different-lag skin
        # temperatures (0.3-2.5 % of q_sat before lag error), and the [0,1]
        # clip could only ever SHRINK the flux -- truncating legitimate
        # super-saturation sources and zeroing evening-transition dew.  Passing
        # the humidity itself has no inversion, no anchor mismatch and no
        # clip.  Blended by land fraction; ocean/ice keep saturation at SST.
        _qsfc_traced = (forcing.get("q_sfc_land")
                        if forcing is not None else None)
        if _qsfc_traced is not None:
            if f_land is None:
                raise ValueError(
                    "forcing['q_sfc_land'] (traced land surface humidity) "
                    "requires the land fraction to be threaded into the "
                    "turbulence factory (make_physics f_land=...)."
                )
            _f_land_col = jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
            q_sfc = ((1.0 - _f_land_col) * q_sfc
                     + _f_land_col * jnp.asarray(
                         _qsfc_traced, dtype=q_sfc.dtype).reshape(nCells))
        _beta_traced = (forcing.get("beta_land")
                        if forcing is not None else None)
        if _qsfc_traced is not None:
            pass                     # the solved humidity supersedes beta
        elif _beta_traced is not None:
            if f_land is None:
                raise ValueError(
                    "forcing['beta_land'] (traced per-cell beta_soil) "
                    "requires the land fraction to be threaded into the "
                    "turbulence factory (make_physics f_land=...); the "
                    "driver must pass f_land whenever mpas_land_beta_soil "
                    "is enabled."
                )
            from legoesm.atmosphere.physics.turbulence.surface_layer import (
                beta_limited_surface_humidity,
            )
            _f_land_col = jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
            q_sfc = beta_limited_surface_humidity(
                q_sfc, q_v_col[:, -1], _f_land_col,
                jnp.asarray(_beta_traced, dtype=q_sfc.dtype).reshape(nCells))
        elif f_land is not None and land_beta != 1.0:
            from legoesm.atmosphere.physics.turbulence.surface_layer import (
                beta_limited_surface_humidity,
            )
            _f_land_col = jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
            q_sfc = beta_limited_surface_humidity(
                q_sfc, q_v_col[:, -1], _f_land_col, land_beta)

        # ``forcing["shflx_land"]`` / ``forcing["lhflx_land"]`` — the land
        # scheme's OWN turbulent fluxes, blended by land fraction into the
        # surface flux the BL scheme consumes.  This exists because handing
        # over the canopy's HUMIDITY was measured insufficient: the canopy
        # solved its flux against ITS aerodynamic resistance, so its boundary
        # humidity sits close to the air by construction, and the atmosphere
        # re-applying its own resistance to that already-collapsed gradient
        # delivered ~a tenth of the canopy's flux (Amazon latent heat 78 W/m2
        # offline -> 7 coupled, land 10 K cold in 30 days).  A flux is what
        # the land solved; a flux is what crosses the boundary.  Momentum and
        # the ocean/ice fraction keep the scheme's own bulk computation.
        _shf_land = (forcing.get("shflx_land") if forcing is not None else None)
        _surface_flux = None
        if _shf_land is not None:
            if f_land is None:
                raise ValueError(
                    "forcing['shflx_land'] requires f_land in the turbulence "
                    "factory (make_physics f_land=...).")
            if not _accepts_surface_flux:
                raise ValueError(
                    f"turbulence scheme {scheme_name!r} cannot be handed the "
                    "land surface fluxes: it computes its own from T_sfc/q_sfc "
                    "and takes no 'surface_flux' argument. Silently dropping "
                    "them would run the advertised land coupling with a "
                    "surface flux the land model never solved. Use 'louis' or "
                    "'clubb', or teach this scheme the argument.")
            from legoesm.atmosphere.physics.turbulence.surface_layer import (
                compute_surface_fluxes,
            )
            _fl = jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
            _tx, _ty, _sh, _lh, _us = compute_surface_fluxes(
                u_col[:, -1], v_col[:, -1], T_col[:, -1], q_v_col[:, -1],
                T_sfc, q_sfc, rho[:, -1], scheme_config.surface,
            )
            _lh_land = jnp.asarray(
                forcing["lhflx_land"], dtype=q_sfc.dtype).reshape(nCells)
            _sh_land = jnp.asarray(_shf_land, dtype=q_sfc.dtype).reshape(nCells)
            _surface_flux = (
                _tx, _ty,
                (1.0 - _fl) * _sh + _fl * _sh_land,
                (1.0 - _fl) * _lh + _fl * _lh_land,
                _us,
            )

        if needs_tke:
            tke_in = _read_turb_carry(
                phys_state, carry_field, nCells, nlev, scheme_config, _state_dtype)
            turb_out, tke_new = turb_fn(
                u_col, v_col, T_col, q_v_col, tke_in,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )
            tke_out = tke_new
        else:
            turb_out = turb_fn(
                u_col, v_col, T_col, q_v_col,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
                **({"surface_flux": _surface_flux}
                   if _surface_flux is not None else {}),
            )

        # Cell → edge tendency projection.  Average the cell tendencies
        # of the two cells flanking each edge, then project onto the
        # edge normal via ``angleEdge`` (eastward = 0).
        # ``mesh.cellsOnEdge`` has shape ``(2, nEdges)`` in the MPAS
        # layout: slice along axis 0 to get the per-edge cell-index
        # vectors (``cellsOnEdge[0]`` and ``cellsOnEdge[1]``).
        du_cell = turb_out.du_dt  # (nCells, nlev)
        dv_cell = turb_out.dv_dt
        c0 = mesh.cellsOnEdge[0]  # (nEdges,)
        c1 = mesh.cellsOnEdge[1]  # (nEdges,)
        du_e_east = 0.5 * (du_cell[c0] + du_cell[c1])
        dv_e_north = 0.5 * (dv_cell[c0] + dv_cell[c1])
        angle = mesh.angleEdge[:, None]
        du_edge_normal = du_e_east * jnp.cos(angle) + dv_e_north * jnp.sin(angle)

        dT_cell = turb_out.dT_dt

        # Moisture tendency stays at cells (tracer pytree).
        tracer_tends = {}
        if state.tracers is not None and "q_v" in state.tracers:
            _qv_raw = state.tracers["q_v"]
            _name = "dq_v_dt_turb"
            if hasattr(_qv_raw, "replace"):
                tracer_tends["q_v"] = _qv_raw.replace(
                    data=turb_out.dq_v_dt.reshape(_qv_raw.data.shape),
                    name=_name,
                )
            else:
                tracer_tends["q_v"] = turb_out.dq_v_dt.reshape(_qv_raw.shape)

        zero_ps = jnp.zeros_like(p_s)
        # Surface turbulent fluxes for the CMOR hfss/hfls feed [W/m^2,
        # positive upward — the schemes' own shflx/lhflx sign, which is
        # already the CMOR convention]. None-guarded: a scheme without
        # surface fluxes (or turbulence "none") simply leaves the fields
        # unset, byte-identical to the pre-export tendency.
        _shf = getattr(turb_out, "shflx", None)
        _lhf = getattr(turb_out, "lhflx", None)
        tendencies = HydrostaticTendencies(
            du_dt=state.u.replace(data=du_edge_normal, name="du_dt_turb"),
            dv_dt=None,
            dT_dt=state.T.replace(data=dT_cell, name="dT_dt_turb"),
            dp_s_dt=state.p_s.replace(data=zero_ps, name="dp_s_dt_turb"),
            dphis_dt=state.phis.replace(data=zero_ps, name="dphis_dt_turb"),
            tracer_tendencies=tracer_tends if tracer_tends else None,
            # units="W/m^2": p_s.replace would otherwise INHERIT p_s's "Pa"
            # (Field.replace keeps self.units), mislabelling the flux Field.
            shflx_sfc=None if _shf is None else state.p_s.replace(
                data=_shf.reshape(p_s.shape), name="shflx_sfc_turb",
                units="W/m^2"),
            lhflx_sfc=None if _lhf is None else state.p_s.replace(
                data=_lhf.reshape(p_s.shape), name="lhflx_sfc_turb",
                units="W/m^2"),
        )
        return tendencies, _carry_update_with_cloud_fraction(
            carry_field, tke_out, turb_out)

    def reset_state():
        return None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Non-hydrostatic Compressible Euler
# ===========================================================================

def _make_nonhydrostatic_turbulence(
    turbulence_config: TurbulenceConfig,
    dt: float,
) -> Callable:
    """Create turbulence physics_fn for CompressibleEulerModel.

    Signature: (state, grid, height_coord, terrain_metric, phys_state=None)
               -> (NonHydrostaticTendencies, tke_out)

    When *phys_state* (a ``PhysicsState``) is passed, TKE is read from
    ``phys_state.tke`` and the updated TKE is returned as the second
    element of the result tuple.
    """
    scheme_name, turb_fn, scheme_config = get_turbulence_fn(turbulence_config)
    needs_tke = turbulence_scheme_traits(scheme_name).carries_energy
    carry_field = turbulence_carry_field(scheme_name, scheme_config)
    if carry_field in ("qke", "clubb_moments"):
        # Phase C codex iter-3 high: the nonhydrostatic CD-grid dynamics
        # driver drops the returned ``PhysicsState`` after every
        # physics call (see ``slow_tendency_fn`` in
        # ``compressible_euler_cdgrid.py``), so an evolved prognostic carry
        # (qke, or the prognostic-CLUBB moments) would silently re-initialise
        # on every step.  Fail fast at factory time until phys_state is
        # threaded through the nonhydrostatic step path (out of Phase C scope).
        _what = "MYNN-2.5" if carry_field == "qke" else "prognostic CLUBB"
        raise NotImplementedError(
            f"{_what} turbulence requires a dynamics driver that "
            "persists PhysicsState across steps.  The current "
            "nonhydrostatic CD-grid driver discards the returned "
            f"phys_state, which would silently re-initialise the {carry_field} "
            "carry on every step.  Use ``model_type='hydrostatic'`` "
            f"for {_what} (MPAS turbulence is not yet wired up); tracking issue: thread "
            "PhysicsState through the nonhydrostatic step path."
        )

    def physics_fn(
        state: NonHydrostaticState,
        grid: CubedSphereGrid,
        height_coord: HeightCoordinate,
        terrain_metric: TerrainMetric,
        phys_state=None,
    ):
        tke_out = None
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
        # Exner Pi = (p/p_ref)^kappa via the canonical helper (adds the 1 Pa
        # AD-safe pressure floor; identical forward for any physical p).
        exner = _exner_function(p)
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

        ncol = shape_2d[0] * shape_2d[1] * shape_2d[2]
        # Mirror the hydrostatic bridge: pin defaulted allocations to the
        # state precision so x64-default zeros do not poison the column
        # path.
        _state_dtype = T.dtype
        _phis_dtype = state.phis.data.dtype

        if turb_fn is None:
            return NonHydrostaticTendencies(
                du_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="du_dt_turb", dims=dims_3d, units="m/s^2"),
                dv_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dv_dt_turb", dims=dims_3d, units="m/s^2"),
                dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_turb", dims=dims_w, units="m/s^2"),
                dtheta_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="dtheta_prime_dt_turb", dims=dims_3d, units="K/s"),
                drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_turb", dims=dims_3d, units="kg/m^3/s"),
                dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_turb", dims=dims_2d, units="m^2/s^3"),
                dtracers_dt=Field(data=jnp.zeros_like(tracers), name="dtracers_dt_turb", dims=dims_tr, units="1/s"),
            ), tke_out

        # Use terrain-aware heights for NH columns.
        z_full = terrain_metric.z_full_3d.reshape(ncol, nlev)
        z_half = terrain_metric.z_half_3d.reshape(ncol, nlev + 1)

        # Interface pressure from evolving column state (not fixed reference).
        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p,
            rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        ).reshape(ncol, nlev + 1)

        # Reshape to columns
        T_col = T.reshape(ncol, nlev)
        u_col = u_data.reshape(ncol, nlev)
        v_col = v_data.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        rho_col = rho_total.reshape(ncol, nlev)

        q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
        if n_tracers > 0:
            q_v_col = tracers[..., 0].reshape(ncol, nlev)

        T_sfc = _resolve_T_sfc(T_col, phys_state)
        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])

        if needs_tke:
            tke_in = _read_turb_carry(
                phys_state, carry_field, ncol, nlev, scheme_config, _state_dtype)
            turb_out, tke_new = turb_fn(
                u_col, v_col, T_col, q_v_col, tke_in,
                p_full_col, p_half, z_full, z_half,
                T_sfc, q_sfc, rho_col, dt, scheme_config,
            )
            tke_out = tke_new
        else:
            turb_out = turb_fn(
                u_col, v_col, T_col, q_v_col,
                p_full_col, p_half, z_full, z_half,
                T_sfc, q_sfc, rho_col, dt, scheme_config,
            )

        du_dt = turb_out.du_dt.reshape(shape_3d)
        dv_dt = turb_out.dv_dt.reshape(shape_3d)
        dT_dt = turb_out.dT_dt.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)

        dtracers = jnp.zeros_like(tracers)
        if n_tracers > 0:
            dq_v_dt = turb_out.dq_v_dt.reshape(shape_3d)
            dtracers = dtracers.at[..., 0].set(dq_v_dt)

        tendencies = NonHydrostaticTendencies(
            du_dt=Field(data=du_dt, name="du_dt_turb", dims=dims_3d, units="m/s^2"),
            dv_dt=Field(data=dv_dt, name="dv_dt_turb", dims=dims_3d, units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=_state_dtype), name="dw_dt_turb", dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=dtheta_prime_dt, name="dtheta_prime_dt_turb", dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=_state_dtype), name="drho_prime_dt_turb", dims=dims_3d, units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=_phis_dtype), name="dphis_dt_turb", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=dtracers, name="dtracers_dt_turb", dims=dims_tr, units="1/s"),
        )
        return tendencies, _carry_update_with_cloud_fraction(
            carry_field, tke_out, turb_out)

    def reset_state():
        return None

    physics_fn.reset_state = reset_state
    return physics_fn


# ===========================================================================
# Spectral PE
# ===========================================================================

def _make_spectral_pe_turbulence(
    turbulence_config: TurbulenceConfig,
    dt: float,
) -> Callable:
    """Create turbulence physics_fn for SpectralPEModel.

    Signature: (state, grid, sigma_coord, grid_fields=None, phys_state=None)
               -> (SpectralHydrostaticState, tke_out)

    When *phys_state* (a ``PhysicsState``) is passed, TKE is read from
    ``phys_state.tke`` and the updated TKE is returned as the second
    element of the result tuple.
    """
    scheme_name, turb_fn, scheme_config = get_turbulence_fn(turbulence_config)
    needs_tke = turbulence_scheme_traits(scheme_name).carries_energy
    carry_field = turbulence_carry_field(scheme_name, scheme_config)
    if carry_field in ("qke", "clubb_moments"):
        # Phase C codex iter-3 high: spectral PE dynamics drops the
        # returned ``PhysicsState`` (see spectral_pe.py:1556-1557), so an
        # evolved prognostic carry (qke, or the prognostic-CLUBB moments)
        # would silently re-initialise on every step.  Fail fast until
        # phys_state is threaded through the spectral PE step.
        _what = "MYNN-2.5" if carry_field == "qke" else "prognostic CLUBB"
        raise NotImplementedError(
            f"{_what} turbulence requires a dynamics driver that "
            "persists PhysicsState across steps.  The current "
            "spectral PE driver discards the returned phys_state, "
            f"which would silently re-initialise the {carry_field} carry on "
            f"every step.  Use ``model_type='hydrostatic'`` for {_what} (MPAS "
            "turbulence is not yet wired up); tracking issue: thread "
            "PhysicsState through the spectral PE step path."
        )

    def physics_fn(state, grid, sigma_coord, grid_fields=None, phys_state=None):
        tke_out = None

        # Transform spectral state to grid space (or reuse precomputed fields).
        fields = grid_fields
        if fields is None:
            fields = spectral_pe_to_grid(state, grid, sigma_coord)
        u = fields['u']         # (n_lat, n_lon, nlev)
        v = fields['v']
        T = fields['T']
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
        u_col = u.reshape(ncol, nlev)
        v_col = v.reshape(ncol, nlev)
        p_full_col = p_full.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        # Pin the column-physics dtype to the gridded state precision so
        # we never silently flow x64 zeros into the column path.
        _state_dtype = T.dtype
        q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)

        zero_3d = jnp.zeros_like(state.vor_hat.data)
        zero_2d = jnp.zeros_like(state.lnps_hat.data)

        if turb_fn is None:
            return SpectralHydrostaticState(
                vor_hat=state.vor_hat.replace(data=zero_3d),
                div_hat=state.div_hat.replace(data=zero_3d),
                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
                lnps_hat=state.lnps_hat.replace(data=zero_2d),
                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
            ), tke_out

        # Heights and density (moist form — audit 2026-05-12 MEDIUM #8).
        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col, q_v=q_v_col)
        rho = _compute_rho(T_col, p_full_col, q_v=q_v_col)

        T_sfc = _resolve_T_sfc(T_col, phys_state)
        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])

        if needs_tke:
            tke_in = _read_turb_carry(
                phys_state, carry_field, ncol, nlev, scheme_config, _state_dtype)
            turb_out, tke_new = turb_fn(
                u_col, v_col, T_col, q_v_col, tke_in,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )
            tke_out = tke_new
        else:
            # No land-flux hand-over on this lane (see the hydrostatic closure).
            turb_out = turb_fn(
                u_col, v_col, T_col, q_v_col,
                p_full_col, p_half_col, z_full, z_half,
                T_sfc, q_sfc, rho, dt, scheme_config,
            )

        # Reshape tendencies to grid space
        du_dt = turb_out.du_dt.reshape(n_lat, n_lon, nlev)
        dv_dt = turb_out.dv_dt.reshape(n_lat, n_lon, nlev)
        dT_dt = turb_out.dT_dt.reshape(n_lat, n_lon, nlev)

        # Project wind tendencies to spectral vorticity/divergence
        # (same pattern as held_suarez_forcing_spectral)
        a = grid.radius
        im_over_a = 1j * grid.ms.astype(jnp.float64) / a
        one_over_a = 1.0 / a

        cos_lat_3d = grid.cos_lat[:, None, None]  # (n_lat, 1, 1)
        du_cos = du_dt * cos_lat_3d
        dv_cos = dv_dt * cos_lat_3d

        # curl(du, dv) -> vorticity tendency
        dvor_hat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
            + one_over_a * sh_analysis_dmu_3d(grid, du_cos)
        )

        # div(du, dv) -> divergence tendency
        ddiv_hat = (
            im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
            - one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
        )

        # Temperature tendency to spectral
        dT_hat = sh_analysis_3d(grid, dT_dt)

        tendencies = SpectralHydrostaticState(
            vor_hat=state.vor_hat.replace(data=dvor_hat),
            div_hat=state.div_hat.replace(data=ddiv_hat),
            T_hat=state.T_hat.replace(data=dT_hat),
            lnps_hat=state.lnps_hat.replace(data=zero_2d),
            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
        )
        return tendencies, _carry_update_with_cloud_fraction(
            carry_field, tke_out, turb_out)

    def reset_state():
        return None

    physics_fn.reset_state = reset_state
    return physics_fn
