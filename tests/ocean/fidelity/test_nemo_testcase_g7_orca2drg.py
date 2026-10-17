"""Grid-ladder G7: certified G5 plus ORCA2's linear bottom drag."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G5 = CASES / "nemo_testcase_g5_orca2ice"
G7 = CASES / "nemo_testcase_g7_orca2drg"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
ORCA2 = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/EXP00")
CARD_FILES = ("ORCA_R2_zps_domcfg.nc", "data_1m_potential_temperature_nomask.nc",
              "data_1m_salinity_nomask.nc")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 inputs")
needs_nemo = pytest.mark.skipif(not ORCA2.is_dir(), reason="no NEMO tree")
SWITCH = {"namdrg.ln_drg_off": (True, False), "namdrg.ln_lin": (False, True)}


@pytest.fixture(autouse=True, scope="module")
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _values(path):
    return recipe._orca2g1_namelist_values(path)


def _root(tmp_path, rung):
    root = tmp_path / rung.name
    root.mkdir()
    for name in ("namelist_cfg", "namelist_ice_cfg"):
        shutil.copy(rung / name, root / name)
    (root / "domain_cfg.nc").symlink_to(INPUTS / CARD_FILES[0])
    for name in CARD_FILES[1:]:
        (root / name).symlink_to(INPUTS / name)
    return root


def test_g7_deck_is_g5_plus_namdrg_only():
    g5, g7 = _values(G5 / "namelist_cfg"), _values(G7 / "namelist_cfg")
    changed = {key: (g5.get(key), g7.get(key)) for key in set(g5) | set(g7)
               if g5.get(key) != g7.get(key)}
    assert changed == SWITCH
    assert (G7 / "namelist_ice_cfg").read_bytes() == (G5 / "namelist_ice_cfg").read_bytes()
    pin = re.search(r"G7_DECK_SHA=(\w+)", (G7 / "run.sh").read_text()).group(1)
    assert hashlib.sha256((G7 / "namelist_cfg").read_bytes()).hexdigest() == pin


@needs_nemo
def test_g7_drag_lines_are_orca2s():
    orca2 = _values(ORCA2 / "namelist_cfg")
    assert orca2["namdrg.ln_lin"] is True and _values(G7 / "namelist_cfg")["namdrg.ln_lin"]
    # ORCA2 leaves every companion to namelist_ref; so does G7, and the card
    # transcribes namelist_ref's bottom values.
    assert not [k for k in orca2 if k.startswith(recipe._ORCA2G_DRG_UNSTATED)]
    ref = _values(ORCA2 / "namelist_ref")
    assert (ref["namdrg_bot.rn_cd0"], ref["namdrg_bot.rn_uc0"]) == (
        recipe._SMT2_RN_CD0, recipe._SMT2_RN_UC0)
    assert ref["namdrg_bot.ln_boost"] is False and ref["namdrg.ln_drgimp"] is True


@pytest.mark.parametrize("old,new,match", [
    ("   ln_non_lin  = .false.", "   ln_non_lin  = .true.", "ln_non_lin"),
    ("   ln_lin      = .true. ", "   ln_lin      = .false.", "ln_drg_off"),
    ("   ln_loglayer = .false.", "   ln_loglayer = .false.\n   ln_drgimp = .false.",
     "ln_drgimp"),
])
def test_g7_reader_refuses_planted_drag_line(tmp_path, old, new, match):
    root = tmp_path / "deck"
    root.mkdir()
    text = (G7 / "namelist_cfg").read_text()
    assert text.count(old) == 1
    (root / "namelist_cfg").write_text(text.replace(old, new))
    with pytest.raises(ValueError, match=match):
        recipe.read_orca2g1_deck(root, "ORCA2G7-drg")


@needs_inputs
def test_g7_card_is_g5_plus_the_linear_drag_bundle(tmp_path):
    g5_root, g7_root = _root(tmp_path, G5), _root(tmp_path, G7)
    g5 = recipe.build_orca2g5_ice_card(g5_root)
    g7 = recipe.build_nemo_testcase_card("ORCA2G7-drg", deck_root=g7_root)
    c5 = g5.recipe.model_config
    assert g7.recipe.model_config == recipe._nemo_linear_bottom_drag(c5)
    assert c5.bottom_drag.bottom_drag_scheme == "legacy" and not c5.zdf_drag_in_matrix
    assert g7.recipe.model_config.implicit_ice_ocean_top_drag
    for field in ("T", "S", "u", "v", "uu_b", "vv_b", "eta"):
        assert np.array_equal(np.asarray(getattr(g7.recipe.initial_state, field).data),
                              np.asarray(getattr(g5.recipe.initial_state, field).data))
    assert g7.unmeasured_features == recipe.ORCA2G7_UNMEASURED
    assert recipe._orca2g1_selector_failures(g7) == []
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g7, g7_root)
    with pytest.raises(ValueError, match="ln_lin"):
        recipe.build_orca2g5_ice_card(g7_root)
    with pytest.raises(ValueError, match="ln_lin"):
        recipe.build_orca2g7_drg_card(g5_root)


def test_run_script_reuses_g5_binaries_smoke_first_and_checks_the_drag_log():
    run = G7 / "run.sh"
    assert subprocess.run(["bash", "-n", str(run)]).returncode == 0
    text = run.read_text()
    assert "makenemo" not in text.split("set -Eeuo pipefail", 1)[1]
    executions = re.findall(r'^run_one "([^\"]+)" "\$(\w+)"', text, re.M)
    assert executions == [
        ("$TARGET/BLD/bin/nemo.exe", "SMOKE"),
        ("$REF_TARGET/BLD/bin/nemo.exe", "REF"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_A"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_B"),
        ("$TARGET/BLD/bin/nemo.exe", "TWIN_C"),
    ]
    assert "linear BOTTOM friction (constant coef = Cd0*Uc0 =    4.0000000000000002E-004 )" in text
    assert "ln_drgimp += +T$" in text and "ln_drgice_imp += *T$" in text
    assert "g5_records.py\" admit" in text
    assert "REF_BIN_SHA=e8ebf4ae" in text and "REC_BIN_SHA=41cd9a63" in text
    assert "round41/oracle_g7_r41" in text and "G6" not in text


def test_g2_family_ladder_routes_g7_through_the_ice_record_path():
    sys.path.insert(0, str(CASES))
    spec = importlib.util.spec_from_file_location(
        "g7_g2_ladder", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    assert "ORCA2G7-drg" in ladder.CASES
    assert "ORCA2G7-drg" in ladder.ICE_CASES and "ORCA2G7-drg" in ladder.FORCED


# --- fast synthetic controls on the input-free VORTEX card -------------------

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
