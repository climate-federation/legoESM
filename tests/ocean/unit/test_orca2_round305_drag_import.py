"""ORCA2 round 305: G5 top drag + G7 bottom drag imported from the grid ladder.

The VORTEX-based statement tests are gridladder's (tests/ocean/fidelity/
test_nemo_testcase_g7_orca2drg.py at 9aad93036), ported unchanged; the
bottom-only selector tests are this round's.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
SELECTORS = ("implicit_ice_ocean_top_drag", "nemo_bottom_drag_written_statements")


def test_defaults_are_off_and_no_orca2_card_selects_them():
    from legoesm.ocean.state import LatLonCGridOceanConfig, OceanSurfaceForcing

    assert all(LatLonCGridOceanConfig._field_defaults[n] is False for n in SELECTORS)
    assert OceanSurfaceForcing._field_defaults["ice_ocean_top_drag_rCdu"] is None
    if not DECK.is_dir():
        pytest.skip("ORCA2 input deck is absent")
    pytest.importorskip("netCDF4")
    cfg = recipe.build_orca2_zps_card(DECK).recipe.model_config
    assert not any(getattr(cfg, n) for n in SELECTORS)



@pytest.fixture(scope="module")
def vortex():
    return recipe.build_nemo_testcase_card("VORTEX-zco")


def _field(card, scale, seed):
    ny, nx = card.recipe.initial_state.eta.data.shape
    return jnp.asarray(-scale * np.random.default_rng(seed).random((ny, nx)))


def test_combined_coefficient_is_nemos_one_statement(vortex):
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        nemo_ice_ocean_top_drag_coeff_faces,
    )
    t, b = _field(vortex, 1.5e-3, 5), _field(vortex, 4e-4, 7)
    cu, cv, su, sv, _ = jax.jit(lambda x, y: nemo_ice_ocean_top_drag_coeff_faces(
        x, vortex.recipe.grid, rCdU_bot=y))(t, b)
    t, b = np.asarray(t), np.asarray(b)
    # dynspg_ts.f90:1431-1432: r1_2*((bot(i+1)+bot(i)) + (top(i+1)+top(i)))
    np.testing.assert_array_equal(np.asarray(cu)[:, 1:-1], 0.5 * (
        (b[:, 1:] + b[:, :-1]) + (t[:, 1:] + t[:, :-1])))
    np.testing.assert_array_equal(np.asarray(cv)[1:-1, :], 0.5 * (
        (b[1:, :] + b[:-1, :]) + (t[1:, :] + t[:-1, :])))
    # the returned pairs stay the top pair alone (residual / dyn_zdf read it)
    np.testing.assert_array_equal(np.asarray(su)[:, 1:-1], t[:, 1:] + t[:, :-1])
    np.testing.assert_array_equal(np.asarray(sv)[1:-1, :], t[1:, :] + t[:-1, :])


@pytest.mark.parametrize("change", [
    dict(zdf_drag_in_matrix=True), dict(barotropic_drag_substep=True)])
def test_guard_refuses_partial_top_plus_bottom(vortex, change):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    cfg = recipe._nemo_linear_bottom_drag(vortex.recipe.model_config)._replace(
        implicit_ice_ocean_top_drag=True)
    cfg = cfg._replace(**{"zdf_drag_in_matrix": False,
                          "barotropic_drag_substep": False, **change})
    with pytest.raises(ValueError, match="requires"):
        LatLonCGridOceanModel(vortex.recipe.grid, vortex.recipe.z_coord, cfg)


def test_guard_refuses_quadratic_top_plus_bottom(vortex):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    cfg = recipe._nemo_linear_bottom_drag(vortex.recipe.model_config)
    cfg = cfg._replace(implicit_ice_ocean_top_drag=True, bottom_drag=cfg.bottom_drag._replace(
        bottom_drag_scheme="nemo_quadratic"))
    with pytest.raises(ValueError, match="full linear bundle"):
        LatLonCGridOceanModel(vortex.recipe.grid, vortex.recipe.z_coord, cfg)


def test_substeps_receive_nemos_bottom_tfield(vortex, monkeypatch):
    # dynspg_ts.f90:1431-1432 on the substep path: the face helper must get
    # rCdU_bot = -(rn_Cd0*ssmask)*rn_Uc0 (zdfdrg.f90:276), -0 on dry columns.
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OceanSurfaceForcing
    real, seen = pe.nemo_ice_ocean_top_drag_coeff_faces, []

    def spy(*args, rCdU_bot=None, **kw):
        if rCdU_bot is not None:
            jax.debug.callback(lambda v: seen.append(np.asarray(v)), rCdU_bot)
        return real(*args, rCdU_bot=rCdU_bot, **kw)

    monkeypatch.setattr(pe, "nemo_ice_ocean_top_drag_coeff_faces", spy)
    cfg = recipe._nemo_linear_bottom_drag(vortex.recipe.model_config)._replace(
        implicit_ice_ocean_top_drag=True)
    model = LatLonCGridOceanModel(vortex.recipe.grid, vortex.recipe.z_coord, cfg)
    s0 = vortex.recipe.initial_state
    zero = jnp.zeros_like(s0.eta.data)
    out = model.step(s0, dt=vortex.dt_s, surface_forcing=OceanSurfaceForcing(
        tau_x=zero, tau_y=zero, ice_ocean_top_drag_rCdu=_field(vortex, 1.5e-3, 5)))
    assert np.all(np.isfinite(np.asarray(out.eta.data)))
    assert seen, "the substeps never passed rCdU_bot"
    wet = np.asarray(vortex.recipe.z_coord.bottom_level) >= 0
    want = np.where(wet, -(recipe._SMT2_RN_CD0 * recipe._SMT2_RN_UC0), -0.0)
    for got in seen:
        got = np.asarray(got)
        np.testing.assert_array_equal(got, want)
        assert np.array_equal(np.signbit(got), np.signbit(want))


def test_residual_increment_is_nemos_statement():
    # dynspg_ts.f90:1462: pu_RHSi + (r1_hu_0/(1+r3u)) * r1_2*(pair) * zu_i
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_drg_residual_increment
    rng = np.random.default_rng(3)
    r1h, cdu, uk, ub = (rng.random((5, 7)) for _ in range(4))
    got = np.asarray(nemo_drg_residual_increment(
        jnp.asarray(r1h), jnp.asarray(-4e-4 * cdu), jnp.asarray(uk), jnp.asarray(ub)))
    np.testing.assert_array_equal(got, (r1h * (-4e-4 * cdu)) * (uk - ub))


def test_residual_add_rounds_the_increment_under_jit():
    # dynspg_ts.f90:1462 rounds the product before pu_RHSi adds it; under jit
    # with traced inputs XLA may otherwise contract F + a*b into one FMA.
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_drg_residual_increment
    rng = np.random.default_rng(13)
    r1h, uk, ub = (rng.standard_normal((64, 64)) for _ in range(3))
    cdu, f = -4e-4 * rng.random((64, 64)), 4e-4 * rng.standard_normal((64, 64))
    got = jax.jit(lambda f, *a: nemo_source_round(f + nemo_drg_residual_increment(*a)))(
        *map(jnp.asarray, (f, r1h, cdu, uk, ub)))
    np.testing.assert_array_equal(np.asarray(got), f + (r1h * cdu) * (uk - ub))


def test_bottom_residual_reads_the_stored_reciprocal_depth(vortex, monkeypatch):
    # dynspg_ts.f90:1462: the bottom residual divides by the STORED
    # r1_hu_0/(1+r3u(Kmm)), the same factor as G5's certified top residual
    # (:1487), never the live column depth; sea level is off rest so they differ.
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OceanSurfaceForcing
    real, seen = pe.nemo_drg_residual_increment, []

    def spy(r1_h, cdu, u_k, u_b):
        jax.debug.callback(lambda r, c: seen.append((np.asarray(r), np.asarray(c))),
                           r1_h, cdu)
        return real(r1_h, cdu, u_k, u_b)

    monkeypatch.setattr(pe, "nemo_drg_residual_increment", spy)
    cfg = recipe._nemo_linear_bottom_drag(vortex.recipe.model_config)._replace(
        implicit_ice_ocean_top_drag=True, barotropic_wind_nemo_stored_depth=True)
    model = LatLonCGridOceanModel(vortex.recipe.grid, vortex.recipe.z_coord, cfg)
    s0 = vortex.recipe.initial_state
    eta = s0.eta.data + 0.5 * jnp.asarray(np.random.default_rng(17).random(
        s0.eta.data.shape)) * (s0.H_bathy.data > 0)
    zero = jnp.zeros_like(s0.eta.data)
    model.step(s0._replace(eta=s0.eta.replace(data=eta)), dt=vortex.dt_s,
               surface_forcing=OceanSurfaceForcing(
                   tau_x=zero, tau_y=zero, ice_ocean_top_drag_rCdu=_field(vortex, 1.5e-3, 5)))
    rate = -(recipe._SMT2_RN_CD0 * recipe._SMT2_RN_UC0)
    bottom = {r.shape: r for r, c in seen if np.any(c == rate)}
    top = {r.shape: r for r, c in seen if not np.any(c == rate)}
    assert len(bottom) == len(top) == 2, (len(bottom), len(top))
    for shape, r in bottom.items():
        np.testing.assert_array_equal(r, top[shape])


def test_bottom_residual_reads_carried_barotropic_velocity(vortex, monkeypatch):
    # dynspg_ts.f90:1447-1450: zu_i = puu(mbku,Kmm) - puu_b(Kmm), the CARRIED
    # barotropic velocity, never a depth mean rebuilt from the 3-D field.
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OceanSurfaceForcing
    real, seen = pe.nemo_drg_residual_increment, []

    def spy(r1_h, cdu, u_k, u_b):
        jax.debug.callback(lambda c, b: seen.append((np.asarray(c), np.asarray(b))),
                           cdu, u_b)
        return real(r1_h, cdu, u_k, u_b)

    monkeypatch.setattr(pe, "nemo_drg_residual_increment", spy)
    cfg = recipe._nemo_linear_bottom_drag(vortex.recipe.model_config)._replace(
        implicit_ice_ocean_top_drag=True)
    model = LatLonCGridOceanModel(vortex.recipe.grid, vortex.recipe.z_coord, cfg)
    s0 = vortex.recipe.initial_state
    rng = np.random.default_rng(11)
    ub = jnp.asarray(0.01 * rng.random(s0.uu_b.data.shape))
    vb = jnp.asarray(0.01 * rng.random(s0.vv_b.data.shape))
    st = s0._replace(uu_b=s0.uu_b.replace(data=ub), vv_b=s0.vv_b.replace(data=vb))
    zero = jnp.zeros_like(s0.eta.data)
    model.step(st, dt=vortex.dt_s, surface_forcing=OceanSurfaceForcing(
        tau_x=zero, tau_y=zero, ice_ocean_top_drag_rCdu=zero))
    rate = -(recipe._SMT2_RN_CD0 * recipe._SMT2_RN_UC0)
    bottom = [(c, b) for c, b in seen if np.any(c == rate)]
    assert len(bottom) == 2, "the bottom residual did not run NEMO's statement"
    by_shape = {b.shape: b for _, b in bottom}
    np.testing.assert_array_equal(by_shape[ub.shape], np.asarray(ub))
    np.testing.assert_array_equal(by_shape[vb.shape], np.asarray(vb))


def test_dynzdf_drag_rhs_is_nemos_statement():
    # dynzdf.f90:166: zDt_2 * (rCdU_bot(ji+1)+rCdU_bot(ji)) * uu_b / e3u(iku)
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_dynzdf_drag_rhs
    rng = np.random.default_rng(5)
    pair, ub, e3 = -8e-4 * rng.random((4, 6)), rng.random((4, 6, 3)), 1 + rng.random((4, 6, 3))
    got = np.asarray(nemo_dynzdf_drag_rhs(5400.0, jnp.asarray(pair), jnp.asarray(ub),
                                          jnp.asarray(e3)))
    np.testing.assert_array_equal(got, ((5400.0 * pair)[..., None] * ub) / e3)


def test_dynzdf_bottom_rhs_runs_nemos_statement(vortex, monkeypatch):
    # G7's dyn_zdf bottom term goes through the written statement with the
    # bottom pair rCdU_bot(ji+1)+rCdU_bot(ji) = 2 * -(rn_Cd0*rn_Uc0) on wet faces.
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.state import OceanSurfaceForcing
    real, seen = pe.nemo_dynzdf_drag_rhs, []

    def spy(zdt_2, pair_sum, u_b, e3):
        jax.debug.callback(lambda p: seen.append(np.asarray(p)), pair_sum)
        return real(zdt_2, pair_sum, u_b, e3)

    monkeypatch.setattr(pe, "nemo_dynzdf_drag_rhs", spy)
    cfg = recipe._nemo_linear_bottom_drag(vortex.recipe.model_config)._replace(
        implicit_ice_ocean_top_drag=True)
    model = LatLonCGridOceanModel(vortex.recipe.grid, vortex.recipe.z_coord, cfg)
    s0 = vortex.recipe.initial_state
    zero = jnp.zeros_like(s0.eta.data)
    model.step(s0, dt=vortex.dt_s, surface_forcing=OceanSurfaceForcing(
        tau_x=zero, tau_y=zero, ice_ocean_top_drag_rCdu=zero))
    pair = 2 * -(recipe._SMT2_RN_CD0 * recipe._SMT2_RN_UC0)
    assert sum(np.any(p == pair) for p in seen) >= 2, \
        "dyn_zdf's bottom drag RHS did not run NEMO's statement"


@pytest.mark.parametrize("change", [
    dict(zdf_drag_in_matrix=False), dict(barotropic_drag_substep=False),
    dict(zdf_baroclinic_only=False), "quadratic"])
def test_bottom_only_selector_refuses_without_the_linear_bundle(vortex, change):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    cfg = recipe._nemo_linear_bottom_drag(vortex.recipe.model_config)._replace(
        nemo_bottom_drag_written_statements=True)
    if change == "quadratic":
        cfg = cfg._replace(bottom_drag=cfg.bottom_drag._replace(
            bottom_drag_scheme="nemo_quadratic"))
    else:
        cfg = cfg._replace(**change)
    with pytest.raises(ValueError, match="requires"):
        LatLonCGridOceanModel(vortex.recipe.grid, vortex.recipe.z_coord, cfg)


@pytest.mark.parametrize("bottom_only", [True, False])
def test_bottom_only_selector_runs_both_g7_statements_without_ice(
        vortex, monkeypatch, bottom_only):
    # nn_ice=0 (zdfdrg.f90:380): no top-drag input, yet the bottom residual
    # (dynspg_ts.f90:1462-1463, carried uu_b) and the dyn_zdf bottom RHS
    # (dynzdf.f90:166-169) must both run NEMO's statements.  Selector off ->
    # neither statement helper runs (the control for this test).
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    real_r, real_z, res, zdf = (pe.nemo_drg_residual_increment,
                                pe.nemo_dynzdf_drag_rhs, [], [])

    def spy_r(r1_h, cdu, u_k, u_b):
        jax.debug.callback(lambda b: res.append(np.asarray(b)), u_b)
        return real_r(r1_h, cdu, u_k, u_b)

    def spy_z(zdt_2, pair_sum, u_b, e3):
        jax.debug.callback(lambda p: zdf.append(np.asarray(p)), pair_sum)
        return real_z(zdt_2, pair_sum, u_b, e3)

    monkeypatch.setattr(pe, "nemo_drg_residual_increment", spy_r)
    monkeypatch.setattr(pe, "nemo_dynzdf_drag_rhs", spy_z)
    cfg = recipe._nemo_linear_bottom_drag(vortex.recipe.model_config)._replace(
        nemo_bottom_drag_written_statements=bottom_only)
    model = LatLonCGridOceanModel(vortex.recipe.grid, vortex.recipe.z_coord, cfg)
    s0 = vortex.recipe.initial_state
    rng = np.random.default_rng(23)
    ub = jnp.asarray(0.01 * rng.random(s0.uu_b.data.shape))
    vb = jnp.asarray(0.01 * rng.random(s0.vv_b.data.shape))
    model.step(s0._replace(uu_b=s0.uu_b.replace(data=ub),
                           vv_b=s0.vv_b.replace(data=vb)), dt=vortex.dt_s)
    if not bottom_only:
        assert not res and not zdf
        return
    by_shape = {b.shape: b for b in res}
    np.testing.assert_array_equal(by_shape[ub.shape], np.asarray(ub))
    np.testing.assert_array_equal(by_shape[vb.shape], np.asarray(vb))
    pair = 2 * -(recipe._SMT2_RN_CD0 * recipe._SMT2_RN_UC0)
    assert sum(np.any(p == pair) for p in zdf) >= 2
