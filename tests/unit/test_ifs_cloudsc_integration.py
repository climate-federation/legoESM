"""IFS 49r1 CLOUDSC wiring (votes vote_1103 VC2-VC21): tracer registry,
config refusals, the build-time refusals of the combined physics, the
CUMASTRN -> CLOUDSC hand-off in the convection bridge and the FV3-duo column
lane's slot contract for the extra tracer."""
from __future__ import annotations

import types

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.tracers import make_full_moisture_registry
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.physics_pipeline import (
    moisture_registry_for, registry_scheme_config,
    validate_microphysics_tracer_slots)

_MPAS = dict(grid=GridConfig(grid_type="mpas", resolution=3, nlev=8),
             dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas"))
_CUBE = dict(grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=8),
             dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"))
_IFS_CONV = dict(convection="bechtold", bechtold_use_ifs_ascent=True)


def _errors(cfg):
    try:
        cfg.validate_strict()
    except ValueError as exc:
        return str(exc)
    return ""


# --- registry ---------------------------------------------------------------

def test_registry_appends_the_cloud_fraction_only_for_ifs_cloudsc():
    full = make_full_moisture_registry().names
    ifs = moisture_registry_for("ifs_cloudsc")
    assert ifs.names == full + ("cloud_fraction_ifs",)
    info = ifs.tracers[-1]
    assert (info.units, info.positive_definite, info.transported) == ("1", True, True)
    validate_microphysics_tracer_slots("ifs_cloudsc", ifs.n_tracers, context="t")
    with pytest.raises(ValueError, match="too few tracer slots"):
        validate_microphysics_tracer_slots("ifs_cloudsc", len(full), context="t")
    for scheme in ("none", "kessler", "sundqvist", "morrison", "thompson", "p3"):
        names = moisture_registry_for(
            scheme, registry_scheme_config(ExperimentConfig(microphysics=scheme))).names
        assert "cloud_fraction_ifs" not in names, scheme


def test_cloud_fraction_is_outside_the_water_ledgers():
    from legoesm.core.conservation import is_borrow_eligible_tracer
    from legoesm.diagnostics.process_ledger import LEDGER_WATER_SPECIES
    assert "cloud_fraction_ifs" not in LEDGER_WATER_SPECIES
    assert not is_borrow_eligible_tracer("cloud_fraction_ifs")   # plain floor


# --- config validation ------------------------------------------------------

def test_validate_strict_accepts_ifs_cloudsc_on_mpas_with_ifs_convection():
    ExperimentConfig(microphysics="ifs_cloudsc", cld_macmic_num_steps=1,
                     **_IFS_CONV, **_MPAS).validate_strict()


def test_validate_strict_refuses_ifs_cloudsc_sub_cycled():
    msg = _errors(ExperimentConfig(microphysics="ifs_cloudsc", cld_macmic_num_steps=3,
                                   **_IFS_CONV, **_MPAS))
    assert "needs cld_macmic_num_steps=1" in msg, msg


@pytest.mark.parametrize("conv", [dict(convection="zhang_mcfarlane"),
                                  dict(convection="bechtold"),
                                  dict(convection="none")])
def test_validate_strict_refuses_ifs_cloudsc_without_cumastrn(conv):
    msg = _errors(ExperimentConfig(microphysics="ifs_cloudsc", **conv, **_MPAS))
    assert "needs the faithful IFS convection" in msg, msg


def test_validate_strict_refuses_ifs_cloudsc_on_the_compiled_cube_lane():
    msg = _errors(ExperimentConfig(microphysics="ifs_cloudsc", **_IFS_CONV, **_CUBE))
    assert "runs only on the MPAS column lane" in msg, msg
    assert "needs the faithful IFS convection" not in msg


# --- combined physics build -------------------------------------------------

def _pcfg(conv="bechtold", ifs=True):
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.convection.config import (
        BechtoldConfig, ConvectionConfig)
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    return PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(
            scheme=conv, bechtold=BechtoldConfig(use_ifs_ascent=ifs, dx_m=1.0e5)),
        microphysics=MicrophysicsConfig(scheme="ifs_cloudsc"))


@pytest.mark.parametrize("kw,match", [
    (dict(model_type="hydrostatic", cld_macmic_num_steps=1), "MPAS column lane only"),
    (dict(model_type="mpas", cld_macmic_num_steps=2), "cld_macmic_num_steps=1"),
])
def test_make_physics_refusals(kw, match):
    from legoesm.atmosphere.physics.combined import make_physics
    with pytest.raises(ValueError, match=match):
        make_physics(_pcfg(), dt=600.0, **kw)


@pytest.mark.parametrize("conv,ifs", [("bechtold", False), ("zhang_mcfarlane", False)])
def test_make_physics_refuses_non_cumastrn_convection(conv, ifs):
    from legoesm.atmosphere.physics.combined import make_physics
    with pytest.raises(ValueError, match="use_ifs_ascent=True"):
        make_physics(_pcfg(conv, ifs), model_type="mpas", dt=600.0,
                     cld_macmic_num_steps=1)


def test_radiation_refuses_two_cloud_fraction_sources():
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.radiation.integration import make_radiation_physics
    with pytest.raises(ValueError, match="two cloud-fraction sources"):
        make_radiation_physics(RadiationConfig(scheme="gray"), "mpas",
                               use_clubb_cloud_fraction=True,
                               cloud_fraction_tracer="cloud_fraction_ifs")
    with pytest.raises(ValueError, match="MPAS lane only"):
        make_radiation_physics(RadiationConfig(scheme="gray"), "hydrostatic",
                               cloud_fraction_tracer="cloud_fraction_ifs")


# --- convection bridge: the hand-off replaces the host booking --------------

def _conv_bridge(defer, ncol=3, nlev=6):
    from legoesm.atmosphere.physics.convection import integration as ci
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics.ifs_cloudsc import CloudscConvection
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate

    T = jnp.broadcast_to(jnp.linspace(220.0, 298.0, nlev)[None, :], (ncol, nlev))
    q = jnp.broadcast_to(jnp.linspace(1e-5, 0.017, nlev)[None, :], (ncol, nlev))
    z = jnp.zeros((ncol, nlev))
    names = moisture_registry_for("ifs_cloudsc").names
    tracers = {"q_v": Field(q), **{n: Field(z) for n in names if n != "q_v"}}
    state = HydrostaticState(
        u=Field(z), T=Field(T), p_s=Field(jnp.full((ncol,), 1.0e5)),
        phis=Field(jnp.zeros((ncol,))), v=Field(z), tracers=tracers)
    grid = types.SimpleNamespace(grid_shape_2d=(ncol,), land_frac=np.zeros(ncol))
    a = lambda v: jnp.full((ncol, nlev), v)  # noqa: E731
    raw = CloudscConvection(plu=a(1e-3), plude=a(2e-4), pludeli=jnp.zeros((ncol, nlev, 4)),
                            psnde=jnp.zeros((ncol, nlev, 2)), pmfu=a(0.01), pmfd=a(-0.005),
                            ldcum=jnp.ones((ncol,), bool), ktype=jnp.ones((ncol,), int))
    real = ci.bechtold_convection

    def spy(*args, **kw):
        out, m_u, stoch = real(*args, **kw)
        return out._replace(dq_r_conv_dt=None, dq_c_conv_dt=a(3e-7),
                            dq_r_detr_dt=a(2e-7), dq_s_detr_dt=a(1e-7),
                            ifs_cloudsc_conv=raw), m_u, stoch

    ci.bechtold_convection = spy
    try:
        fn = ci.make_convection_physics(
            ConvectionConfig(scheme="bechtold"), "mpas", 600.0,
            defer_to_cloudsc=defer)
        tend, upd = fn(state, grid, create_sigma_coordinate(nlev), phys_state=None)
    finally:
        ci.bechtold_convection = real
    return tend.tracer_tendencies, upd, raw


def test_convection_hands_raw_fields_to_cloudsc_instead_of_booking_them():
    tt_off, upd_off, _ = _conv_bridge(False)
    tt_on, upd_on, raw = _conv_bridge(True)
    # without the hand-off (every other microphysics): the host books it
    np.testing.assert_allclose(np.asarray(tt_off["q_c"].data), 3e-7, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(tt_off["q_r"].data), 2e-7, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(tt_off["q_s"].data), 1e-7, rtol=1e-12)
    assert "ifs_cloudsc_conv" not in upd_off
    # with it: nothing booked, the raw CUMASTRN fields published
    assert float(jnp.max(jnp.abs(tt_on["q_c"].data))) == 0.0
    assert "q_r" not in tt_on and "q_s" not in tt_on
    assert upd_on["ifs_cloudsc_conv"] is raw
    np.testing.assert_array_equal(np.asarray(tt_on["q_v"].data),
                                  np.asarray(tt_off["q_v"].data))


def test_convection_hand_off_refuses_a_scheme_without_raw_fields():
    """The legacy (non-CUMASTRN) Bechtold publishes no raw fields."""
    with pytest.raises(ValueError, match="raw CUMASTRN fields"):
        _conv_bridge_noraw()


def _conv_bridge_noraw(ncol=2, nlev=6):
    from legoesm.atmosphere.physics.convection import integration as ci
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState
    from legoesm.grids.vertical import create_sigma_coordinate
    T = jnp.broadcast_to(jnp.linspace(220.0, 298.0, nlev)[None, :], (ncol, nlev))
    z = jnp.zeros((ncol, nlev))
    names = moisture_registry_for("ifs_cloudsc").names
    state = HydrostaticState(
        u=Field(z), T=Field(T), p_s=Field(jnp.full((ncol,), 1.0e5)),
        phis=Field(jnp.zeros((ncol,))), v=Field(z),
        tracers={n: Field(z) for n in names})
    grid = types.SimpleNamespace(grid_shape_2d=(ncol,), land_frac=np.zeros(ncol))
    fn = ci.make_convection_physics(ConvectionConfig(scheme="bechtold"), "mpas",
                                    600.0, defer_to_cloudsc=True)
    fn(state, grid, create_sigma_coordinate(nlev), phys_state=None)


# --- FV3-duo column lane: the registry names are passenger slots ------------

def test_fv3_duo_column_takes_the_ifs_registry_as_nwat6_plus_passengers():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import FV3DuoColumnModel
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    dyn = FV3DuoDynamicsModel(create_fv3_duo_grid(12, 3),
                              FV3DuoConfig(km=5, n_split=2, moist=True))
    names = moisture_registry_for("ifs_cloudsc").names
    col = FV3DuoColumnModel(dyn, tracer_names=names)
    assert col.nwat == 6
    assert col.tracer_names[6:] == ("N_c", "N_r", "N_i", "cloud_fraction_ifs")
    # the column view labels the dimensionless fraction "1", not kg/kg (VC41)
    state, _ = col.column_view(dyn.dcmip16_initial_state(n_tracers=len(names)))
    assert state.tracers["cloud_fraction_ifs"].units == "1"
    assert state.tracers["q_v"].units == "kg/kg"


def test_cloudsc_bridge_refuses_a_state_missing_a_water_species():
    from legoesm.atmosphere.physics.microphysics.integration import (
        _cloudsc_lane_kwargs)
    names = [n for n in moisture_registry_for("ifs_cloudsc").names if n != "q_s"]
    state = types.SimpleNamespace(tracers={n: None for n in names})
    with pytest.raises(ValueError, match="missing \\['q_s'\\]"):
        _cloudsc_lane_kwargs(state, None, 1, None, None)


def test_advected_cloud_fraction_replaces_the_diagnosed_one_over_the_whole_column():
    """VC24: on the IFS lane the override is not cut at the CLUBB 700 hPa gate."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import compute_cloud_properties
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.radiation.integration import (
        full_column_cloud_fraction_override)
    rc = full_column_cloud_fraction_override(RadiationConfig(scheme="gray",
                                                             cloud_scheme="resolved"))
    nlev = 8
    p = jnp.linspace(2.0e4, 1.0e5, nlev)[None, :]
    T = jnp.linspace(220.0, 295.0, nlev)[None, :]
    q = jnp.full_like(T, 1.0e-3)
    dp = jnp.full_like(T, 1.0e4)
    a = jnp.linspace(0.1, 0.8, nlev)[None, :]
    qc = jnp.full_like(T, 1.0e-5)
    cp = compute_cloud_properties(T, p, q, dp, rc.cloud_config, q_cloud=qc,
                                  q_ice=jnp.zeros_like(T), cloud_fraction_override=a)
    np.testing.assert_allclose(np.asarray(cp.cloud_fraction), np.asarray(a), rtol=0, atol=0)
    assert full_column_cloud_fraction_override(RadiationConfig(cloud_scheme="none")) == \
        RadiationConfig(cloud_scheme="none")


def test_mpas_radiation_applies_the_full_column_override_for_the_tracer(monkeypatch):
    """VC24 call site: building MPAS radiation with the advected tracer routes
    its config through the full-column override (and only then)."""
    from legoesm.atmosphere.physics.radiation import integration as ri
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    seen = []
    real = ri.full_column_cloud_fraction_override
    monkeypatch.setattr(ri, "full_column_cloud_fraction_override",
                        lambda rc: seen.append(rc) or real(rc))
    ri.make_radiation_physics(RadiationConfig(scheme="gray"), "mpas")
    assert seen == []
    ri.make_radiation_physics(RadiationConfig(scheme="gray"), "mpas",
                              cloud_fraction_tracer="cloud_fraction_ifs")
    assert len(seen) == 1
