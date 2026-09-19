"""prescribed_flow: the IN-MODEL vertical-physics isolation lever.

``LatLonCGridOceanConfig.prescribed_flow`` pins the circulation INSIDE
``_step_impl`` — where the flux-form tracer mass fluxes are built — so T/S
evolve by vertical physics alone (mode="zero": mass fluxes and the diagnosed w
vanish; no advection at all) or under the held step-entry flow (mode="frozen").
A host-loop post-step reset can NOT do this (the intra-step tracer advection
would still use the post-momentum/barotropic flow — the codex-rejected v1).

Contract pinned here, through REAL ``model.step`` calls:
  * "zero":   u/v/eta identically zero after every step; the horizontal
              center-of-mass of a localized T anomaly does not move (no
              advection); T still changes in the vertical (physics active).
  * "frozen": u/v/eta after N steps == the entry values exactly.
  * default None: byte-identical to a config that never sets the field
              (the untraced-default guarantee).
  * unknown value / implicit_unsplit combination: ValueError at construction.
  * parameterized ADVECTIVE tracer transports are rejected at construction
              (they bypass the pinned mass-flux block): GM/Redi — model-level
              ``gm_redi`` (incl. prognostic EKE, which rides on it) AND the
              physics-pipeline ``lateral_mixing.scheme="gm_redi"`` — and the
              Fox-Kemper MLE (``physics.mle``).
  * outer AB2: a stale nonzero u_incr_prev/v_incr_prev carry is neither
              consumed nor re-written (stored carries are zeros).
  * inner tracer AB2: a stale nonzero T/S_flux_div_prev carry is neither
              consumed (no ab2_blend leak of pre-lever advection) nor
              re-written (stored carries are zeros under "zero").
  * ew_cyclic_overlap: slaves TRACERS only under the lever — it runs after
              the final re-pin and must not mutate the frozen u/v/eta.
  * config tail: ``prescribed_flow`` is the appended LAST NamedTuple field,
              so legacy positional construction is unshifted.
  * CLI gates (run_omip_core2.validate_prescribed_flow_args): non-C-grid
              grids and the --spinup-drag combination are rejected, and
              --no-gm-redi is REQUIRED (both supported builders ship GM ON).

Run in the fp64 precision policy.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

_DT = 1800.0
N_LAT, N_LON, NLEV = 8, 12, 5
H_MAX = 1000.0


@pytest.fixture(autouse=True)
def _fp64():
    from legoesm.core.precision import set_policy, get_policy, PrecisionPolicy
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(prev)


def _grid_z():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    return (create_latlon_grid(n_lat=N_LAT, n_lon=N_LON),
            create_ocean_z_star(n_levels=NLEV, H_max=H_MAX))


def _model(grid, z_coord, **cfg_kw):
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    cfg_kw.setdefault("implicit_vertical_mixing", True)
    cfg_kw.setdefault("A_v", 1.0e-3)
    cfg_kw.setdefault("K_v", 1.0e-4)
    cfg = LatLonCGridOceanConfig.from_flat(
        n_barotropic_substeps=8, enable_runtime_checks=False, **cfg_kw)
    return LatLonCGridOceanModel(grid, z_coord, cfg)


def _rest_state(grid, z_coord):
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    # land_lat_threshold=80 with 22.5-deg rows (centers at +/-78.75 max)
    # -> ALL-OCEAN grid: uniform-area-free COM bookkeeping, no mask terms.
    return rest_state_latlon_cgrid_ocean(grid, z_coord, H_max=H_MAX,
                                         land_lat_threshold=80.0)


def _with_T_blob(state):
    """Horizontally-localized warm anomaly (top 2 levels of a 2x2 patch)."""
    T = np.asarray(state.T.data).copy()
    T[2:4, 3:5, :2] += 2.0
    return state._replace(T=state.T.replace(data=jnp.asarray(T)))


def _with_flow(state, grid):
    """Nonzero sheared jet + eta pattern (test_ab2_scope-style seed)."""
    lat = np.degrees(np.asarray(grid.lat))
    shear = np.linspace(1.0, 0.1, NLEV)[None, None, :]
    u = 0.2 * np.cos(np.radians(lat))[:, None, None] * shear
    u = np.broadcast_to(u, state.u.data.shape).copy()
    u *= np.asarray(state.u_mask.data)[..., None]
    u[:, -1] = u[:, 0]                       # periodic wrap column
    v = np.full(state.v.data.shape, 0.05) * shear
    v *= np.asarray(state.v_mask.data)[..., None]
    lon = np.degrees(np.asarray(grid.lon))
    eta = 0.02 * np.sin(np.radians(2.0 * lon))[None, :] * np.ones((N_LAT, 1))
    eta *= np.asarray(state.land_mask.data)
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
    )


def _forcing():
    """Wind stress + net surface heating: momentum forcing must be discarded
    by the pin; the heat forcing must still reach T (vertical physics)."""
    from legoesm.ocean.state import OceanSurfaceForcing
    return OceanSurfaceForcing(
        tau_x=0.1 * jnp.ones((N_LAT, N_LON)),
        tau_y=0.05 * jnp.ones((N_LAT, N_LON)),
        q_net=200.0 * jnp.ones((N_LAT, N_LON)),
    )


def _anomaly_com(state, z_coord):
    """Horizontal center-of-mass (index space) of the column-integrated T
    anomaly, measured against a blob-free reference column (0, 0).  With zero
    advection and horizontally-uniform forcing, every column evolves by the
    SAME per-column vertical operator, so the anomaly support — and its COM —
    is exactly invariant."""
    T = np.asarray(state.T.data)
    d = T - T[0:1, 0:1, :]                       # anomaly vs reference column
    dz = np.asarray(z_coord.dz_ref)[None, None, :]
    A = np.abs((d * dz).sum(axis=-1))            # column-integrated anomaly
    tot = A.sum()
    assert tot > 1e-12, "anomaly vanished — the test lost its tracer signal"
    ii, jj = np.meshgrid(np.arange(N_LAT), np.arange(N_LON), indexing="ij")
    return np.array([(A * ii).sum() / tot, (A * jj).sum() / tot])


# ------------------------------------------------------------------ "zero"

def test_zero_mode_pins_flow_and_kills_advection():
    grid, z = _grid_z()
    state = _with_T_blob(_rest_state(grid, z))
    model = _model(grid, z, prescribed_flow="zero")
    sf = _forcing()
    T0 = np.asarray(state.T.data).copy()
    com0 = _anomaly_com(state, z)

    s = state
    for n in range(5):
        s = model.step(s, _DT, surface_forcing=sf)
        # Flow identically zero after EVERY step (wind stress discarded).
        assert float(jnp.max(jnp.abs(s.u.data))) == 0.0, f"u nonzero, step {n}"
        assert float(jnp.max(jnp.abs(s.v.data))) == 0.0, f"v nonzero, step {n}"
        assert float(jnp.max(jnp.abs(s.eta.data))) == 0.0, \
            f"eta nonzero, step {n}"
        # The diagnosed w is built from the (zero) mass fluxes.
        assert float(jnp.max(jnp.abs(s.w.data))) == 0.0, f"w nonzero, step {n}"

    # No advection: the anomaly's horizontal center-of-mass did not move.
    com5 = _anomaly_com(s, z)
    np.testing.assert_allclose(com5, com0, atol=1e-9)

    # Vertical physics ACTIVE: surface heating + implicit K_v changed T in the
    # vertical (top layer warmed much more than the bottom layer moved).
    dT = np.asarray(s.T.data) - T0
    assert np.max(np.abs(dT[..., 0])) > 1.0e-6, "surface T never changed"
    assert np.max(np.abs(dT[..., 0])) > 10.0 * np.max(np.abs(dT[..., -1])), \
        "T change is not vertically structured (physics suspect)"


# ---------------------------------------------------------------- "frozen"

def test_frozen_mode_holds_entry_flow_exactly():
    grid, z = _grid_z()
    state = _with_flow(_with_T_blob(_rest_state(grid, z)), grid)
    model = _model(grid, z, prescribed_flow="frozen")
    sf = _forcing()
    u0 = np.asarray(state.u.data).copy()
    v0 = np.asarray(state.v.data).copy()
    eta0 = np.asarray(state.eta.data).copy()
    T0 = np.asarray(state.T.data).copy()

    s = state
    for n in range(5):
        s = model.step(s, _DT, surface_forcing=sf)
        np.testing.assert_array_equal(
            np.asarray(s.u.data), u0, err_msg=f"u drifted at step {n}")
        np.testing.assert_array_equal(
            np.asarray(s.v.data), v0, err_msg=f"v drifted at step {n}")
        np.testing.assert_array_equal(
            np.asarray(s.eta.data), eta0, err_msg=f"eta drifted at step {n}")

    # T still evolves (held-flow advection + surface heating).
    assert np.max(np.abs(np.asarray(s.T.data) - T0)) > 1.0e-6


# ------------------------------------------------- default None byte-identity

def test_default_none_is_byte_identical():
    """A config that never sets prescribed_flow and one that sets it to None
    explicitly are the SAME static gate: multi-step states are byte-equal
    (the untraced-default guarantee)."""
    grid, z = _grid_z()
    state = _with_flow(_with_T_blob(_rest_state(grid, z)), grid)
    m_default = _model(grid, z)
    m_none = _model(grid, z, prescribed_flow=None)
    sf = _forcing()

    sa, sb = state, state
    for _ in range(3):
        sa = m_default.step(sa, _DT, surface_forcing=sf)
        sb = m_none.step(sb, _DT, surface_forcing=sf)
    for f in ("u", "v", "T", "S", "eta", "w"):
        np.testing.assert_array_equal(
            np.asarray(getattr(sa, f).data), np.asarray(getattr(sb, f).data),
            err_msg=f"default-vs-explicit-None {f} not byte-identical")


# ------------------------------------------------------------- construction

def test_unknown_mode_raises_at_construction():
    grid, z = _grid_z()
    with pytest.raises(ValueError, match="prescribed_flow"):
        _model(grid, z, prescribed_flow="held")


def test_implicit_unsplit_rejected_at_construction():
    grid, z = _grid_z()
    with pytest.raises(ValueError, match="implicit_unsplit"):
        _model(grid, z, prescribed_flow="zero",
               barotropic_solver="implicit_unsplit", outer_integrator="ab2")


def test_positional_construction_unshifted_by_tail_field():
    """Appending ``prescribed_flow`` at the NamedTuple TAIL must not shift any
    positional slot.  No purely-positional multi-arg construction exists in
    the repo (grepped packages/scripts/tests: every call site is keyword or
    ``from_flat``), so this stands in for the legacy-positional-caller
    contract the config docstring promises ("Appended at the NamedTuple tail
    so positional construction for legacy callers is preserved")."""
    from legoesm.ocean.state import (
        DynBottomDragConfig,
        LatLonCGridOceanConfig,
        LateralViscosityConfig,
    )
    f = LatLonCGridOceanConfig._fields
    # Later append-only additions extend the tail without moving this lever.
    # Pin its neighbours and the complete post-lever suffix explicitly.
    prescribed_i = f.index("prescribed_flow")
    assert f[prescribed_i - 2:prescribed_i + 1] == (
        "tidal_forcing", "backscatter", "prescribed_flow")
    assert f[prescribed_i + 1:] == (
        "freshwater_salinity", "zdf_drag_in_matrix",
        "zdf_baroclinic_only", "barotropic_drag_substep",
        "barotropic_forcing_centred", "store_mass_flux", "store_salt_flux",
        "zdf_implicit_solver_evaluation")
    # Head layout unchanged — the slots any positional caller binds first.
    assert f[:9] == (
        "metric_convention", "vface_zonal_metric_evaluation",
        "coriolis_placement", "lateral_viscosity", "bottom_drag", "K_h",
        "K_bih", "A_v", "K_v")
    # Positional construction of the head lands each value where expected
    # (values chosen distinguishable from every default).
    cfg = LatLonCGridOceanConfig(
        "exact", "nemo_vpoint", "cell_average",
        LateralViscosityConfig(A_h=777.0), DynBottomDragConfig(), 55.0)
    assert cfg.lateral_viscosity.A_h == 777.0
    assert cfg.K_h == 55.0
    assert cfg.K_bih == 0.0                 # first defaulted slot after the 6
    assert cfg.prescribed_flow is None      # tail default untouched


def _isolationbreaking_physics(mle=None, lateral_scheme="none"):
    """OceanPhysicsConfig with every module 'none' except the args under test
    (mirrors build_tripole's explicit-disable construction so nothing else in
    the pipeline can raise first)."""
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        VerticalMixingConfig,
    )
    return OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        lateral_mixing=LateralMixingConfig(scheme=lateral_scheme),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
        shortwave_penetration=None,
        mle=mle,
    )


def test_gm_redi_rejected_under_lever():
    """Model-level GM/Redi (config.gm_redi) is parameterized ADVECTION that
    bypasses the pinned mass-flux block -> ValueError at construction.  The
    prognostic-EKE closure rides ON gm_redi (GMRediConfig.eke) so the same
    rejection covers it — asserted with an EKE-carrying config too."""
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.lateral_mixing.eke import EKEConfig
    grid, z = _grid_z()
    with pytest.raises(ValueError, match="GM/Redi"):
        _model(grid, z, prescribed_flow="zero", gm_redi=GMRediConfig())
    with pytest.raises(ValueError, match="GM/Redi"):
        _model(grid, z, prescribed_flow="frozen",
               gm_redi=GMRediConfig(eke=EKEConfig()))
    # Lever OFF -> the same GM/Redi config constructs fine (lever-scoped gate).
    _model(grid, z, gm_redi=GMRediConfig())


def test_pipeline_gm_redi_rejected_under_lever():
    """GM/Redi expressed through the physics pipeline
    (physics.lateral_mixing.scheme='gm_redi') is the SAME bolus transport,
    injected via physics_fn -> ValueError at construction."""
    grid, z = _grid_z()
    with pytest.raises(ValueError, match="lateral_mixing"):
        _model(grid, z, prescribed_flow="zero",
               physics=_isolationbreaking_physics(lateral_scheme="gm_redi"))


def test_mle_rejected_under_lever():
    """Fox-Kemper MLE (physics.mle) is a bolus tracer transport injected via
    physics_fn before the splice -> ValueError at construction; the identical
    config constructs fine with the lever off."""
    from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig
    grid, z = _grid_z()
    with pytest.raises(ValueError, match="MLE"):
        _model(grid, z, prescribed_flow="zero",
               physics=_isolationbreaking_physics(mle=MLEConfig()))
    with pytest.raises(ValueError, match="MLE"):
        _model(grid, z, prescribed_flow="frozen",
               physics=_isolationbreaking_physics(mle=MLEConfig()))
    _model(grid, z, physics=_isolationbreaking_physics(mle=MLEConfig()))


# ------------------------------------------------------- outer-AB2 carries

def test_ab2_outer_neutralizes_stale_momentum_carries():
    """Under outer_integrator='ab2' + the lever, a stale NONZERO
    u_incr_prev/v_incr_prev is neither consumed (u stays pinned exactly) nor
    re-written (the stored carries are zeros) — codex finding 2.  The TRACER
    AB2 carries stay live (T/S physics increments are real under the lever)."""
    from legoesm.core.field import Field
    grid, z = _grid_z()
    state = _with_flow(_with_T_blob(_rest_state(grid, z)), grid)
    # Poison the momentum carries with discarded-momentum "history".
    state = state._replace(
        u_incr_prev=Field(data=0.7 * jnp.ones_like(state.u.data),
                          name="u_incr_prev", dims=state.u.dims,
                          units=state.u.units),
        v_incr_prev=Field(data=-0.4 * jnp.ones_like(state.v.data),
                          name="v_incr_prev", dims=state.v.dims,
                          units=state.v.units),
    )
    model = _model(grid, z, prescribed_flow="frozen", outer_integrator="ab2")
    sf = _forcing()
    u0 = np.asarray(state.u.data).copy()
    v0 = np.asarray(state.v.data).copy()
    eta0 = np.asarray(state.eta.data).copy()

    s = state
    for n in range(4):
        s = model.step(s, _DT, surface_forcing=sf)
        np.testing.assert_array_equal(
            np.asarray(s.u.data), u0, err_msg=f"u unpinned at step {n} "
            "(stale AB2 carry re-injected?)")
        np.testing.assert_array_equal(
            np.asarray(s.v.data), v0, err_msg=f"v unpinned at step {n}")
        np.testing.assert_array_equal(
            np.asarray(s.eta.data), eta0, err_msg=f"eta unpinned at step {n}")
        assert float(jnp.max(jnp.abs(s.u_incr_prev.data))) == 0.0, \
            "u_incr_prev not neutralized"
        assert float(jnp.max(jnp.abs(s.v_incr_prev.data))) == 0.0, \
            "v_incr_prev not neutralized"
    # Tracer AB2 carry is REAL (physics increments nonzero).
    assert float(jnp.max(jnp.abs(s.T_incr_prev.data))) > 0.0


# ------------------------------------------------- inner tracer-AB2 carries

def test_tracer_ab2_no_stale_advection_leak():
    """tracer_time_integrator='ab2' + prescribed_flow='zero' from a state with
    POISONED nonzero T/S_flux_div_prev (a pre-lever-regime restart): the stale
    advective carries are neither consumed — T/S/u/v/eta/w evolution is
    byte-identical to the euler-integrator zero-mode run (without the fix,
    ab2_blend(0, fd_prev, eps) = -(0.5+eps)*fd_prev is NONZERO and leaks
    stale advection) — nor re-poisoned (the stored carries are exactly zeros
    after every step, scan-stable but never read)."""
    from legoesm.core.field import Field
    grid, z = _grid_z()
    base = _with_T_blob(_rest_state(grid, z))
    poison = 1.0e-5 * jnp.ones_like(base.T.data)
    poisoned = base._replace(
        T_flux_div_prev=Field(data=poison, name="T_flux_div_prev",
                              dims=("lat", "lon", "level"), units="m/s"),
        S_flux_div_prev=Field(data=-0.5 * poison, name="S_flux_div_prev",
                              dims=("lat", "lon", "level"), units="m/s"),
    )
    m_ab2 = _model(grid, z, prescribed_flow="zero",
                   tracer_time_integrator="ab2")
    m_eul = _model(grid, z, prescribed_flow="zero",
                   tracer_time_integrator="euler")
    sf = _forcing()

    sa, sb = poisoned, base
    for n in range(3):
        sa = m_ab2.step(sa, _DT, surface_forcing=sf)
        sb = m_eul.step(sb, _DT, surface_forcing=sf)
        # Stored advective flux-div carry is ZEROS (never stale again).
        assert float(jnp.max(jnp.abs(sa.T_flux_div_prev.data))) == 0.0, \
            f"T_flux_div_prev carry not neutralized at step {n}"
        assert float(jnp.max(jnp.abs(sa.S_flux_div_prev.data))) == 0.0, \
            f"S_flux_div_prev carry not neutralized at step {n}"
        # No stale-advection leak: identical to the euler zero-mode run
        # (the pinned advection term is the same exact zero in both graphs).
        for fld in ("T", "S", "u", "v", "eta", "w"):
            np.testing.assert_array_equal(
                np.asarray(getattr(sa, fld).data),
                np.asarray(getattr(sb, fld).data),
                err_msg=f"{fld} diverged at step {n} (stale advection leak?)")


# ------------------------------------------------------- ew_cyclic_overlap

def test_ew_overlap_does_not_unpin_frozen_flow():
    """config.ew_cyclic_overlap runs AFTER the final re-pin; under the lever
    it must slave TRACERS only.  Entry u/v/eta halo columns are POISONED so
    they violate the overlap identity (col0 != col_{nx-2}): the full overlap
    would rewrite them every step — the exact-hold contract catches any
    mutation.  (Geometrically the overlap is an ORCA-grid construct; here it
    is exercised mechanically on the regular grid, which is exactly the
    mutation path under test.)"""
    grid, z = _grid_z()
    state = _with_flow(_with_T_blob(_rest_state(grid, z)), grid)
    u = np.asarray(state.u.data).copy()
    v = np.asarray(state.v.data).copy()
    eta = np.asarray(state.eta.data).copy()
    u[:, 0] += 0.031          # violate u overlap identity (col0 vs nx-2)
    v[:, 0] += 0.017
    eta[:, 0] += 0.004
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(u)),
        v=state.v.replace(data=jnp.asarray(v)),
        eta=state.eta.replace(data=jnp.asarray(eta)),
    )
    model = _model(grid, z, prescribed_flow="frozen", ew_cyclic_overlap=True)
    sf = _forcing()

    s = state
    for n in range(3):
        s = model.step(s, _DT, surface_forcing=sf)
        np.testing.assert_array_equal(
            np.asarray(s.u.data), u,
            err_msg=f"u mutated by ew overlap at step {n}")
        np.testing.assert_array_equal(
            np.asarray(s.v.data), v,
            err_msg=f"v mutated by ew overlap at step {n}")
        np.testing.assert_array_equal(
            np.asarray(s.eta.data), eta,
            err_msg=f"eta mutated by ew overlap at step {n}")
    # Tracers ARE still slaved (the overlap ran, T/S-only).
    T = np.asarray(s.T.data)
    np.testing.assert_array_equal(T[:, 0], T[:, N_LON - 2],
                                  err_msg="T overlap not applied")


# ---------------------------------------------------------------- CLI gates

def test_cli_gate_rejects_non_cgrid_grids():
    from scripts.run.run_omip_core2 import validate_prescribed_flow_args
    for bad in ("cubed_sphere", "mpas"):
        with pytest.raises(SystemExit, match="lat-lon C-grid"):
            validate_prescribed_flow_args("zero", bad, 0.0)


def test_cli_gate_rejects_spinup_drag_combo():
    from scripts.run.run_omip_core2 import validate_prescribed_flow_args
    with pytest.raises(SystemExit, match="spinup-drag"):
        validate_prescribed_flow_args("frozen", "tripole", 5.0)


def test_cli_gate_requires_no_gm_redi():
    """Both supported builders ship GM/Redi ON unconditionally (tripole
    recipe gm_redi=True; latlon_bathy _DEFAULT_BATHY_GM_REDI), and the model
    constructor rejects prescribed_flow+GM/Redi — so the arg gate demands the
    explicit --no-gm-redi BEFORE the expensive grid/mesh/IC build."""
    from scripts.run.run_omip_core2 import validate_prescribed_flow_args
    with pytest.raises(SystemExit, match="no-gm-redi"):
        validate_prescribed_flow_args("zero", "tripole", 0.0)
    with pytest.raises(SystemExit, match="no-gm-redi"):
        validate_prescribed_flow_args("frozen", "latlon_bathy", 0.0,
                                      no_gm_redi=False)


def test_cli_gate_accepts_supported_combinations():
    from scripts.run.run_omip_core2 import validate_prescribed_flow_args
    validate_prescribed_flow_args("zero", "tripole", 0.0, no_gm_redi=True)
    validate_prescribed_flow_args("frozen", "latlon_bathy", 0.0,
                                  no_gm_redi=True)
    # None disables every gate (any grid / drag / GM combination is fine).
    validate_prescribed_flow_args(None, "mpas", 5.0)
