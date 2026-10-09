"""SMT-5 (Decision 107): NEMO's tra_dmp on the shared WS-RK3 tracer step.

tradmp.f90:190-195  Krhs += resto * (zts_dta - ts(Kbb)), nn_zdmp = 0
dtatsd.f90:212,261,308  fld_read target, raw copy, times tmask
fldread.f90:244-246  two-record time interpolation
stprk3_stg.f90:526,529,538  stage 3 only, after tra_ldf, before tra_zdf
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.dynamics.ocean_tendency_common import (
    NEMOTracerDamping,
    nemo_tra_dmp_rates,
)
from legoesm.ocean.forcing.nemo_fld_read import (
    nemo_clim_monthly_record_centres,
    nemo_fld_time_interpolate,
)

DAY = 86400


def test_record_centres_are_nemos_integer_month_midpoints():
    centres, index = nemo_clim_monthly_record_centres(1)
    # previous year's 12, then year 1's 12 (365-day calendar)
    assert centres.dtype == np.int64 and len(centres) == 24
    assert centres[11] == -15.5 * DAY and index[11] == 11      # Dec, year 0
    assert centres[12] == 15.5 * DAY and index[12] == 0        # Jan, year 1
    assert centres[13] == (31 + 14) * DAY and index[13] == 1   # Feb (28 d)
    with pytest.raises(ValueError):
        nemo_clim_monthly_record_centres(1, nleapy=1)


def test_time_weight_reproduces_orca2s_printed_kt1_records():
    # ORCA2 rung-1 ocean.output: "kt = 1 (0.0625 days) records b/a: 0012/0001
    # (days -15.5000/15.5000)", rn_Dt = 10800 -> isecsbc = ndt05 = 5400 s.
    # The ORCA2 card pins that weight as 249/496.
    centres, index = nemo_clim_monthly_record_centres(1)
    records = np.arange(12, dtype=np.float64).reshape(12, 1)
    got = float(np.asarray(nemo_fld_time_interpolate(records, index, centres,
                                                     5400.0))[0])
    a = np.float64(249.0 / 496.0)
    assert got == (np.float64(1.0) - a) * 11.0 + a * 0.0


def test_time_outside_the_records_is_nan_not_a_clamped_record():
    centres, index = nemo_clim_monthly_record_centres(1)
    records = np.ones((12, 2))
    late = float(centres[-1]) + 1.0
    assert np.all(np.isnan(np.asarray(
        nemo_fld_time_interpolate(records, index, centres, late))))


def _synthetic_column(nk=7):
    rng = np.random.default_rng(107)
    target = 10.0 + rng.standard_normal((1, 1, nk))
    records = np.repeat(target[None], 12, axis=0)          # Decision 107d
    tmask = np.ones((1, 1, nk))
    tmask[..., -2:] = 0.0                                   # land below
    resto = tmask / 86400.0                                 # Decision 107b
    centres, index = nemo_clim_monthly_record_centres(1)
    damping = NEMOTracerDamping(resto, records, records + 25.0, centres,
                                index, 1440.0)
    T_bb = target + 1.0e-3 * rng.standard_normal(target.shape)
    S_bb = target + 25.0 - 2.0e-3
    return damping, T_bb, S_bb, tmask


@pytest.mark.parametrize("t_seconds", [0.0, 2880.0 * 37, 2880.0 * 2999])
def test_damping_rate_is_the_tradmp_statement_bit_for_bit(t_seconds):
    damping, T_bb, S_bb, tmask = _synthetic_column()
    dT, dS = nemo_tra_dmp_rates(damping, T_bb, S_bb, tmask, t_seconds)
    centres = damping.record_centres_s
    isec = damping.isecsbc_at_t0_s + t_seconds
    ia = int(np.searchsorted(centres, isec, side="left"))
    a = (isec - centres[ia - 1]) / (centres[ia] - centres[ia - 1])
    for got, rec, tb in ((dT, damping.target_T, T_bb),
                         (dS, damping.target_S, S_bb)):
        dta = ((1.0 - a) * rec[damping.record_index[ia - 1]]
               + a * rec[damping.record_index[ia]]) * tmask
        want = damping.resto * (dta - tb)
        np.testing.assert_array_equal(np.asarray(got), want)
    assert np.all(np.asarray(dT)[..., -2:] == 0.0)


def test_damping_refuses_a_step_without_model_time():
    damping, T_bb, S_bb, tmask = _synthetic_column()
    with pytest.raises(ValueError, match="t_seconds"):
        nemo_tra_dmp_rates(damping, T_bb, S_bb, tmask, None)


# ---- the card and the production placement --------------------------------

def _write_nemo_like_inputs(card4, root):
    """Files in the layout vortex_smt5_target_dump.F90 writes (jpk levels).

    The TARGET is offset from the initial state (+0.25 K, +0.01) so the
    damping increment is nonzero: a plant on a zero increment proves nothing.
    """
    import netCDF4  # noqa: N813

    active = np.asarray(card4.recipe.z_coord.is_active, dtype=np.float64)
    nj, ni, nk = active.shape
    for name, var, field in (
            ("data_1m_potential_temperature_nomask.nc", "votemper",
             np.asarray(card4.recipe.initial_state.T.data) + 0.25),
            ("data_1m_salinity_nomask.nc", "vosaline",
             np.asarray(card4.recipe.initial_state.S.data) + 0.01)):
        zyx = np.zeros((nk + 1, nj, ni))
        zyx[:nk] = np.moveaxis(field * active, -1, 0)
        with netCDF4.Dataset(root / name, "w") as ds:
            for dim, size in (("x", ni), ("y", nj), ("z", nk + 1)):
                ds.createDimension(dim, size)
            ds.createDimension("time_counter", None)
            v = ds.createVariable(var, "f8", ("time_counter", "z", "y", "x"))
            v[:] = np.repeat(zyx[None], 12, axis=0)
    resto = np.zeros((nk + 1, nj, ni))
    resto[:nk] = np.moveaxis(active, -1, 0) * (1.0 / 86400.0)
    with netCDF4.Dataset(root / "resto.nc", "w") as ds:
        for dim, size in (("x", ni), ("y", nj), ("z", nk + 1)):
            ds.createDimension(dim, size)
        ds.createVariable("resto", "f8", ("z", "y", "x"))[:] = resto


@pytest.fixture(scope="module")
def smt5_card(tmp_path_factory):
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    root = tmp_path_factory.mktemp("smt5_inputs")
    card4 = build_nemo_testcase_card("VORTEX_SMT4_VEC-zps")
    _write_nemo_like_inputs(card4, root)
    return card4, build_nemo_testcase_card("VORTEX_SMT5_VEC-zps",
                                           deck_root=root)


def test_smt5_card_is_smt4_plus_the_damping_alone(smt5_card):
    card4, card5 = smt5_card
    cfg4, cfg5 = card4.recipe.model_config, card5.recipe.model_config
    assert cfg4.nemo_tracer_damping is None
    assert isinstance(cfg5.nemo_tracer_damping, NEMOTracerDamping)
    assert cfg5._replace(nemo_tracer_damping=None) == cfg4


def test_the_validator_keeps_the_damping_on_the_smt5_card_alone(smt5_card):
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
        validate_nemo_testcase_card,
    )

    card4, card5 = smt5_card
    leaked = card4._replace(recipe=card4.recipe._replace(
        model_config=card4.recipe.model_config._replace(
            nemo_tracer_damping=card5.recipe.model_config.nemo_tracer_damping)))
    with pytest.raises(ValueError, match="SMT5"):
        validate_nemo_testcase_card(leaked)
    dropped = card5._replace(recipe=card5.recipe._replace(
        model_config=card5.recipe.model_config._replace(
            nemo_tracer_damping=None)))
    with pytest.raises(ValueError, match="SMT5"):
        validate_nemo_testcase_card(dropped)
    with pytest.raises(ValueError, match="deck_root"):
        build_nemo_testcase_card("VORTEX_SMT5_VEC-zps")


def _stage3_trace(card, plant):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    hooks = _NEMOWSRK3TestHooks(expose_stage3_tracer_damping=True,
                                tracer_damping_stage1_plant=plant)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config,
                                  _nemo_ws_test_hooks=hooks)
    return model.step(card.recipe.initial_state, dt=card.dt_s, t_seconds=0.0)


def assert_stage3_carries_tradmp(card, trace):
    """The gate: stage 3's consumed RHS = before + resto*(T_dta - T_Kbb)."""
    dmp = card.recipe.model_config.nemo_tracer_damping
    active = np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)
    a = (1440.0 - dmp.record_centres_s[11]) / (
        dmp.record_centres_s[12] - dmp.record_centres_s[11])
    dT, dS, bT, bS, cT, cS = (np.asarray(x) for x in trace[1])
    state = card.recipe.initial_state
    for got, rec, tb, before, consumed in (
            (dT, dmp.target_T, state.T.data, bT, cT),
            (dS, dmp.target_S, state.S.data, bS, cS)):
        dta = ((1.0 - a) * rec[11] + a * rec[0]) * active
        np.testing.assert_array_equal(got, dmp.resto * (dta - np.asarray(tb)))
        np.testing.assert_array_equal(consumed, before + got)


def test_stage3_rhs_is_the_tradmp_statement_and_a_stage1_plant_fires(
        smt5_card):
    card4, card5 = smt5_card
    production = _stage3_trace(card5, plant=False)
    assert_stage3_carries_tradmp(card5, production)
    assert np.min(np.abs(np.asarray(production[1][0])[
        np.asarray(card5.recipe.z_coord.is_active)])) > 0.0
    planted = _stage3_trace(card5, plant=True)
    with pytest.raises(AssertionError):
        assert_stage3_carries_tradmp(card5, planted)
    # Stage 1 never sees the production damping: the stage-1 tracer equals
    # the SMT-4 card's bit for bit, and the plant moves it.
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    def stage1_T(card, plant, t_seconds):
        hooks = _NEMOWSRK3TestHooks(expose_tracer_stage=1,
                                    tracer_damping_stage1_plant=plant)
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        return np.asarray(model.step(card.recipe.initial_state, dt=card.dt_s,
                                     t_seconds=t_seconds).T.data)

    smt4 = stage1_T(card4, False, None)
    np.testing.assert_array_equal(stage1_T(card5, False, 0.0), smt4)
    assert not np.array_equal(stage1_T(card5, True, 0.0), smt4)
