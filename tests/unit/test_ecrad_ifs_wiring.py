"""radiation='ecrad_ifs' wiring: scheme literal, validate_strict, factory
refusals, the physics-state carry and the run_amip flag (votes L1-L28,
S1-S6)."""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.atmosphere.physics.combined import PhysicsConfig  # noqa: E402
from legoesm.atmosphere.physics.physics_state import (  # noqa: E402
    init_physics_state, update_physics_state)
from legoesm.atmosphere.physics.radiation import integration as rint  # noqa: E402
from legoesm.atmosphere.physics.radiation.config import RadiationConfig  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs.config import (  # noqa: E402
    EcradIFSConfig, EcradIFSSources)
from legoesm.atmosphere.physics.radiation.ecrad_ifs.lane import carry_width  # noqa: E402
from legoesm.driver.config import (  # noqa: E402
    VALID_RADIATION, DycoreConfig, ExperimentConfig, GridConfig)

_RC = RadiationConfig(scheme="ecrad_ifs", diurnal_cycle=True)
_SRC = EcradIFSSources(solar_irradiance=1361.0, gas_mmr={}, o3_mmr=np.zeros((1, 1)))


def _strict(**kw):
    try:
        ExperimentConfig(**kw).validate_strict()
    except ValueError as e:
        return str(e)
    return ""


def test_literal_and_defaults_are_49r1():
    assert "ecrad_ifs" in VALID_RADIATION
    c = EcradIFSConfig()
    assert (c.ccn_land, c.ccn_sea, c.sharpness) == (900.0, 50.0, 100.0)
    assert RadiationConfig().ecrad_ifs == c


def test_validate_strict_refuses_off_lane_and_aerosols():
    msg = _strict(radiation="ecrad_ifs")                     # default: not MPAS
    assert "radiation='ecrad_ifs' runs only on the MPAS column lane" in msg
    mpas = dict(grid=GridConfig(grid_type="mpas", resolution=3, nlev=8),
                dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas"))
    assert "MPAS column lane" not in _strict(radiation="ecrad_ifs", **mpas)
    msg = _strict(radiation="ecrad_ifs", aerosol_forcing="external", **mpas)
    assert "aerosols OFF" in msg


def test_dispatch_raises():
    with pytest.raises(ValueError, match="ecrad_ifs"):
        rint._get_radiation_fn(_RC)


@pytest.mark.parametrize("kw,match", [
    (dict(model_type="mpas", cloud_fraction_tracer="cloud_fraction_ifs"), "A15"),
    (dict(model_type="hydrostatic", ecrad_ifs_sources=_SRC,
          cloud_fraction_tracer=None), "MPAS column lane"),
    (dict(model_type="mpas", ecrad_ifs_sources=_SRC), "cloud_fraction_ifs"),
])
def test_factory_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        rint.make_radiation_physics(_RC, **kw)


def test_factory_needs_instantaneous_zenith():
    with pytest.raises(ValueError, match="diurnal_cycle"):
        rint.make_radiation_physics(
            _RC._replace(diurnal_cycle=False), "mpas",
            cloud_fraction_tracer="cloud_fraction_ifs", ecrad_ifs_sources=_SRC)


def test_sources_with_another_scheme_raise():
    with pytest.raises(ValueError, match="ecrad_ifs_sources"):
        rint.make_radiation_physics(RadiationConfig(scheme="gray"), "mpas",
                                    ecrad_ifs_sources=_SRC)


def test_factory_builds_and_marks_the_carry_writer():
    for full in (True, False):
        fn = rint.make_radiation_physics(
            _RC, "mpas", cloud_fraction_tracer="cloud_fraction_ifs",
            ecrad_ifs_sources=_SRC, ecrad_ifs_full_call=full)
        assert fn._writes_phys_state is True and fn._wants_forcing is True


def test_physics_state_carry_only_for_ecrad():
    ncol, nlev = 5, 7
    ps = init_physics_state(ncol, nlev, PhysicsConfig(radiation=_RC))
    assert ps.ecrad_carry.shape == (ncol, carry_width(nlev))
    assert carry_width(nlev) == 5 * (nlev + 1) + 13
    new = np.ones_like(np.asarray(ps.ecrad_carry))
    ps2 = update_physics_state(ps, {"ecrad_carry": new})
    assert np.array_equal(np.asarray(ps2.ecrad_carry), new)
    ps3 = update_physics_state(ps2, {})
    assert np.array_equal(np.asarray(ps3.ecrad_carry), new)        # carried
    other = init_physics_state(ncol, nlev, PhysicsConfig(radiation=RadiationConfig()))
    assert other.ecrad_carry is None


def test_run_amip_flag_round_trip():
    from scripts.run import run_amip
    p = run_amip.build_arg_parser()
    a = p.parse_args(["--radiation", "ecrad_ifs"])
    assert a.radiation == "ecrad_ifs"
    assert run_amip.build_config_from_args(a).radiation == "ecrad_ifs"


def test_exported_segment_advances_the_seed_clock():
    """mpas_segment advances forcing['model_seconds'] per step from the
    export's clock (codex r1 P2): a held value would reuse one McICA seed."""
    from legoesm.driver.mpas_diff_step import mpas_segment
    seen = []

    class _M:
        def mass_target(self, s):
            return None

        def step_pure(self, s, dt, physics_fn, forcing, phys_state, target_mass):
            seen.append(forcing.get("model_seconds"))
            return s, phys_state, None, None

    exp = dict(model=_M(), dt=60.0, forcing={"model_seconds": -1.0},
               calendar=lambda k: (1.0, 60.0 * k), model_seconds=lambda k: 60.0 * k + 7.0,
               subcycle_rad=False, rad_update_steps=1, hold_phys=False,
               phys_update_steps=1, physics_fn=None, physics_fn_norad=None,
               physics_fn_held=None, poststep=lambda s: s)
    mpas_segment(exp, 0.0, None, 3, step0=2)
    assert [float(x) for x in seen] == [127.0, 187.0, 247.0]
    seen.clear()
    exp["forcing"] = {"day_of_year": 1.0}   # runs without the key get none
    mpas_segment(exp, 0.0, None, 1)
    assert seen == [None]
