"""Grid-ladder G9: certified G5 plus ORCA2's tracer advection."""

from __future__ import annotations

import hashlib
import importlib.util
import pickle
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G5 = CASES / "nemo_testcase_g5_orca2ice"
G8 = CASES / "nemo_testcase_g8_orca2ldf"
G9 = CASES / "nemo_testcase_g9_orca2tadv"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
ORCA2 = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/EXP00")
ROUNDS = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds")
G8_STAGE1 = ROUNDS / "round45/oracle_g8_r45/twin_a/oracle_rkstage1_transport_operands_kt00000001.bin"
G5_RECORD = ROUNDS / "round31/oracle_g5_r31/twin_a"
CARD_FILES = ("ORCA_R2_zps_domcfg.nc", "data_1m_potential_temperature_nomask.nc",
              "data_1m_salinity_nomask.nc")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 inputs")
needs_nemo = pytest.mark.skipif(not ORCA2.is_dir(), reason="no NEMO tree")
SWITCH = {"namtra_adv.ln_traadv_off": (True, False),
          "namtra_adv.ln_traadv_fct": (False, True),
          "namtra_adv.nn_fct_h": (None, 2),
          "namtra_adv.nn_fct_v": (None, 2)}


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


def test_g9_deck_is_g5_plus_namtra_adv_only():
    g5, g9 = _values(G5 / "namelist_cfg"), _values(G9 / "namelist_cfg")
    changed = {key: (g5.get(key), g9.get(key)) for key in set(g5) | set(g9)
               if g5.get(key) != g9.get(key)}
    assert changed == SWITCH
    assert (G9 / "namelist_ice_cfg").read_bytes() == (G5 / "namelist_ice_cfg").read_bytes()
    pin = re.search(r"G9_DECK_SHA=(\w+)", (G9 / "run.sh").read_text()).group(1)
    assert hashlib.sha256((G9 / "namelist_cfg").read_bytes()).hexdigest() == pin


@needs_nemo
def test_g9_tadv_lines_resolve_to_orca2s():
    orca2, g9 = _values(ORCA2 / "namelist_cfg"), _values(G9 / "namelist_cfg")
    ref = _values(ORCA2 / "namelist_ref")
    keys = [k for k in ref if k.startswith("namtra_adv.")]
    assert len(keys) == 13
    for key in keys:
        assert g9.get(key, ref[key]) == orca2.get(key, ref[key]), key
    assert not [k for k in orca2 if k.startswith(recipe._ORCA2G_TADV_UNSTATED)]
    assert ref["namtra_adv.nn_fct_imp"] == 1          # inert: ln_zad_Aimp = F
    assert g9["namzdf.ln_zad_aimp"] is False


@pytest.mark.parametrize("old,new,match", [
    ("      nn_fct_h   =  2 ", "      nn_fct_h   =  4 ", "nn_fct_h"),
    ("      nn_fct_v   =  2 ", "      nn_fct_v   =  4 ", "nn_fct_v"),
    ("      nn_fct_v   =  2 ", "      nn_fct_v   =  2\n      nn_fct_imp =  2", "nn_fct_imp"),
    ("   ln_traadv_fct = .true. ", "   ln_traadv_fct = .false.", "without FCT"),
    ("   ln_traadv_cen = .false.", "   ln_traadv_cen = .true. ", "ln_traadv_cen"),
])
def test_g9_reader_refuses_planted_tadv_line(tmp_path, old, new, match):
    root = tmp_path / "deck"
    root.mkdir()
    text = (G9 / "namelist_cfg").read_text()
    assert text.count(old) == 1
    (root / "namelist_cfg").write_text(text.replace(old, new))
    with pytest.raises(ValueError, match=match):
        recipe.read_orca2g1_deck(root, "ORCA2G9-tadv")


@needs_inputs
def test_g9_card_is_g5_plus_fct2_and_the_folded_transport_vmask(tmp_path):
    g5_root, g9_root = _root(tmp_path, G5), _root(tmp_path, G9)
    g5 = recipe.build_orca2g5_ice_card(g5_root)
    g9 = recipe.build_nemo_testcase_card("ORCA2G9-tadv", deck_root=g9_root)
    c5, c9 = g5.recipe.model_config, g9.recipe.model_config
    assert [f for f in c5._fields if getattr(c5, f) != getattr(c9, f)] == [
        "tracer_advection", "rk3_stage_transport_raw_vmask", "rk3_stage_transport_fold_exchanges"]
    assert (c5.tracer_advection, c9.tracer_advection) == ("none", "fct2")
    assert c9.rk3_stage_transport_raw_vmask is c9.rk3_stage_transport_fold_exchanges is True
    assert c5.rk3_stage_transport_raw_vmask is c5.rk3_stage_transport_fold_exchanges is False
    assert c9.momentum_advection == "flux_form" and c9.lateral_viscosity.A_h == 0.0
    for name in ("z_coord", "grid", "initial_state"):
        a, b = getattr(g5.recipe, name), getattr(g9.recipe, name)
        assert pickle.dumps(a) == pickle.dumps(b), name
    assert g9.unmeasured_features == recipe.ORCA2G9_UNMEASURED
    assert recipe._orca2g1_selector_failures(g9) == []
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g9, g9_root)
    with pytest.raises(ValueError, match="ln_traadv"):
        recipe.build_orca2g5_ice_card(g9_root)
    with pytest.raises(ValueError, match="ln_traadv"):
        recipe.build_orca2g9_tadv_card(g5_root)
    # the selector check is not vacuous: each G9 selection planted back fails it
    for field, value, row in (("tracer_advection", "none", "ln_traadv_fct"),
                              ("rk3_stage_transport_raw_vmask", False, "stage transport vmask"),
                              ("rk3_stage_transport_fold_exchanges", False,
                               "stage transport fold exchanges")):
        planted = g9._replace(recipe=g9.recipe._replace(model_config=c9._replace(**{field: value})))
        assert recipe._orca2g1_selector_failures(planted) == [row]


def test_only_g9_selects_tracer_advection():
    assert [c for c, on in recipe._ORCA2G_TADV.items() if on] == ["ORCA2G9-tadv"]
    assert set(recipe._ORCA2G_TADV) == set(recipe._ORCA2G_JPK)
    for table in (recipe._ORCA2G_ZDF_FACES, recipe._ORCA2G_TRA_ZDF_ROUNDED,
                  recipe._ORCA2G_UE_RHS_ASCENDING, recipe._ORCA2G_ZDF_STAGE_R3U,
                  recipe._ORCA2G_WIND_STORED_DEPTH, recipe._ORCA2G_TRA_STAGE_ORDER,
                  recipe._ORCA2G_TSD_INIT, recipe._ORCA2G_TSD_DMP, recipe._ORCA2G_SBC,
                  recipe._ORCA2G_DYNADV_VEC, recipe._ORCA2G_DRG, recipe._ORCA2G_LDF):
        assert table["ORCA2G9-tadv"] == table["ORCA2G5-ice"]
    assert "ORCA2G9-tadv" in recipe._ORCA2G_ICE and recipe._ORCA2G_JPK["ORCA2G9-tadv"] == 31
    for deck in sorted(CASES.glob("nemo_testcase_g*_orca2*/namelist_cfg")):
        values = _values(deck)
        want = recipe._orca2g_tadv_switch_set(values)
        assert all(values[k] == v for k, v in want.items()), deck
        assert values["namtra_adv.ln_traadv_fct"] is (deck.parent == G9)
    hits = [path.name for path in (REPO / "packages").rglob("*.py")
            for line in path.read_text().splitlines()
            if "rk3_stage_transport_raw_vmask=" in line or "rk3_stage_transport_fold_exchanges=" in line]
    assert hits == ["nemo_testcase_recipe.py"]


def test_run_script_reuses_g5_binaries_and_checks_the_log():
    run = G9 / "run.sh"
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
    assert "ln_traadv_fct = +T$" in text and "ln_traadv_OFF = +F$" in text
    assert "Horizontal order +: +2 +Vertical +order +: +2" in text
    assert "ln_dynldf_lap = +F$" in text and "ln_drg_OFF += +T$" in text
    assert "eddy_viscosity" not in text and "G8" not in text
    assert "g5_records.py\" admit" in text
    assert "REF_BIN_SHA=e8ebf4ae" in text and "REC_BIN_SHA=41cd9a63" in text
    assert "round46/oracle_g9_r46" in text
    g8 = (G8 / "run.sh").read_text().splitlines()
    extra = [line for line in text.splitlines() if line not in g8]
    assert len(extra) < 40, extra


def test_g2_family_ladder_routes_g9_through_the_ice_record_path():
    sys.path.insert(0, str(CASES))
    spec = importlib.util.spec_from_file_location(
        "g9_g2_ladder", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    assert "ORCA2G9-tadv" in ladder.CASES
    assert "ORCA2G9-tadv" in ladder.ICE_CASES and "ORCA2G9-tadv" in ladder.FORCED


def _nemo_stage1_vmask(path):
    """``vmask`` of NEMO's stage-1 transport operand record (stprk3_stg.f90:288-314)."""
    with path.open("rb") as handle:
        assert handle.read(16).decode("ascii").rstrip() == "NEMO_L2_TRPOP_2"
        _, _, _, _, nx, ny, nz, _ = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    n2, n3 = nx * ny, nx * ny * nz
    offset = 4 * n2 + 6 * n3              # e2u e3u uu zub umask zFu e1v e3v vv zvb
    vmask = values[offset:offset + n3].reshape((nx, ny, nz), order="F")
    return vmask[2:-2, 2:-2].transpose(1, 0, 2)


@needs_inputs
@pytest.mark.skipif(not G8_STAGE1.is_file(), reason="no round-45 G8 record")
def test_g9_transport_vmask_is_nemos_stage1_vmask_and_compact_misses_the_pivot_row(tmp_path):
    """NEMO's vmask after its fold exchange (dommsk.f90:232), as read by the
    stage transport (stprk3_stg.f90:283), from the record binary's own dump.
    The mask is a domain operand, identical on every G rung."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands
    card = recipe.build_nemo_testcase_card("ORCA2G9-tadv", deck_root=_root(tmp_path, G9))
    zc, grid = card.recipe.z_coord, card.recipe.grid
    cu, cv = compute_face_masks_3d(zc.is_active, grid)
    cv = np.asarray(cv, np.float64)
    raw = nemo_qco_resolved_mesh_operands(
        zc, grid, jnp.asarray(cu, jnp.float64), jnp.asarray(cv), jnp.float64,
        zc.is_active.shape[-1]).vmask3
    nemo = _nemo_stage1_vmask(G8_STAGE1)[..., :cv.shape[-1]]
    resolved = np.asarray(raw)                      # card V rows 1.. = NEMO rows 0..
    assert resolved.shape == nemo.shape
    assert np.array_equal(resolved, nemo)
    compact = cv[1:]
    differ = compact != nemo
    assert int(differ.sum()) == 1319 and set(np.nonzero(differ)[0]) == {nemo.shape[0] - 1}


@needs_inputs
@pytest.mark.skipif(not (G5_RECORD / "oracle_ocean_surface_input_kt00000001.bin").is_file(),
                    reason="no round-31 G5 record")
def test_g9_transport_vmask_reaches_the_step_on_the_fold_band_only(tmp_path):
    """One kt-1 step of the G9 card on G5's recorded surface input, with and
    without the selection: the stage transport is where it acts, so the two
    differ, and only in the fold band (last four T rows)."""
    sys.path.insert(0, str(CASES))
    import nemo_testcase_g2_orca2bathy_ladder as ladder
    root = tmp_path / "g9root"
    root.mkdir()
    for path in G5_RECORD.iterdir():
        if path.name != "namelist_cfg":
            (root / path.name).symlink_to(path)
    shutil.copy(G9 / "namelist_cfg", root / "namelist_cfg")

    def step(selected, out):
        original = recipe.build_orca2g1_flat_card

        def build(deck_root, case="ORCA2G1-flat"):
            card = original(deck_root, case)
            cfg = card.recipe.model_config._replace(rk3_stage_transport_raw_vmask=selected)
            return card._replace(recipe=card.recipe._replace(model_config=cfg))

        patch = pytest.MonkeyPatch()
        patch.setattr(recipe, "build_orca2g1_flat_card", build)
        patch.setattr(recipe, "_orca2g1_selector_failures", lambda card: [])
        try:
            ladder.run(root, arm="independent", kt_from=1, kt_to=1, case="ORCA2G9-tadv",
                       state_in=None, state_out=out, allow_dirty=True)
        finally:
            patch.undo()
        state = pickle.loads(out.read_bytes())["state"]
        return {k: np.asarray(getattr(getattr(state, k), "data", getattr(state, k)))
                for k in ("T", "S", "v")}

    on, off = step(True, tmp_path / "on.pkl"), step(False, tmp_path / "off.pkl")
    n_rows = on["T"].shape[0]
    for name in ("T", "S", "v"):
        assert np.isfinite(on[name]).all()
        rows = set(np.nonzero(on[name] != off[name])[0].tolist())
        assert rows and min(rows) >= n_rows - 4, (name, sorted(rows))


G9_RECORD = ROUNDS / "round46/oracle_g9_r46/twin_a"


def _nemo_stage_zfv(stage):
    """NEMO's zFv as tra_adv receives it at kt 1, stage 1 or 2 (stprk3_stg.f90:283;
    the record binary's tracer-operand stream, written before tra_adv)."""
    path = G9_RECORD / f"oracle_rktracer_operands_kt00000001_s{stage}.bin"
    with path.open("rb") as handle:
        assert handle.read(16).decode("ascii").rstrip() == "NEMO_L2_RKTRA_1"
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    assert header[2] == stage
    nx, ny, nz = header[7:10]
    n3 = nx * ny * nz
    zfv = values[3 * n3:4 * n3].reshape((nx, ny, nz), order="F")   # T S zFu zFv
    return zfv.transpose(1, 0, 2)[2:-2, 2:-2]


@needs_inputs
@pytest.mark.skipif(not (G9_RECORD / "oracle_rktracer_operands_kt00000001_s2.bin").is_file(),
                    reason="no round-46 G9 record")
def test_g9_stage_transport_on_the_pivot_row_is_nemos(tmp_path):
    """kt 1 of the jitted G9 step from the card's initial state (state traced, x64),
    on NEMO's recorded surface input: the stage-1 and stage-2 V transports the
    tracer update reads equal NEMO's on the T-pivot row, on every level and wet
    or dry face (bit-identical but for signed zeros).  It needs vn_adv after its fold exchange
    (dynspg_ts.f90:844) and r3v after its fold exchange (stprk3_stg.f90:169);
    stage 1 is bit-identical everywhere."""
    import jax
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    card = recipe.build_nemo_testcase_card("ORCA2G9-tadv", deck_root=G9_RECORD)
    cfg, zc, s0 = card.recipe.model_config, card.recipe.z_coord, card.recipe.initial_state
    assert cfg.rk3_stage_transport_fold_exchanges is True
    fw, sf = g5.nemo_supplied_forcing(card, G9_RECORD, 1)
    nk = int(zc.n_levels)
    for stage in (1, 2):
        model = LatLonCGridOceanModel(
            card.recipe.grid, zc, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_tracer_transport_stage=stage))
        out = jax.device_get(model.step(s0, dt=card.dt_s, freshwater=fw, surface_forcing=sf))
        card_zfv = np.asarray(out.v.data)[1:]          # card V rows 1.. = NEMO rows 0..
        nemo = _nemo_stage_zfv(stage)[..., :nk]
        assert card_zfv.shape == nemo.shape
        assert np.count_nonzero(nemo[-1]) == 1319, stage
        pivot = card_zfv[-1] != nemo[-1]          # stage 2: 375 zeros differ only in sign
        assert not pivot.any(), (stage, int(pivot.sum()))
        if stage == 1:
            assert np.array_equal(card_zfv.view(np.uint64), nemo.view(np.uint64))
