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
    assert {f for f in c5._fields if getattr(c5, f) != getattr(c9, f)} == {
        "tracer_advection", "rk3_stage_transport_raw_vmask", "rk3_stage_transport_fold_exchanges",
        "fct_fold_exchange", "rk3_end_of_stage_pivot_fold", "rk3_stage_wzv_nemo_clock_pair",
        "rk3_stage3_tracer_krhs_nemo_order",
        "rk3_stage_tracer_wzv_stored_kmm_ratios", "rk3_stage_transport_stored_r3uv"}
    assert (c5.rk3_stage3_tracer_krhs_nemo_order, c9.rk3_stage3_tracer_krhs_nemo_order) == (False, True)
    assert (c5.rk3_end_of_stage_pivot_fold, c9.rk3_end_of_stage_pivot_fold) == (False, True)
    assert (c5.rk3_stage_wzv_nemo_clock_pair, c9.rk3_stage_wzv_nemo_clock_pair) == (False, True)
    assert (c5.rk3_stage_tracer_wzv_stored_kmm_ratios,
            c9.rk3_stage_tracer_wzv_stored_kmm_ratios) == (False, True)
    assert (c5.rk3_stage_transport_stored_r3uv,
            c9.rk3_stage_transport_stored_r3uv) == (False, True)
    assert (c5.tracer_advection, c9.tracer_advection) == ("none", "fct2")
    assert c9.rk3_stage_transport_raw_vmask is c9.rk3_stage_transport_fold_exchanges is True
    assert c5.rk3_stage_transport_raw_vmask is c5.rk3_stage_transport_fold_exchanges is False
    assert (c5.fct_fold_exchange, c9.fct_fold_exchange) == (False, True)
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


def test_only_g9_and_its_l3_composition_select_tracer_advection():
    assert [c for c, on in recipe._ORCA2G_TADV.items() if on] == [
        "ORCA2G9-tadv", "ORCA2L3-eiv", "ORCA2L5-mle"]
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
G9_R52_ZRHS = (ROUNDS / "round52/oracle_g9_r52/twin_walk/"
               "oracle_g9_trazdf_zrhs_kt00000001.bin")


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


@needs_inputs
@pytest.mark.skipif(not (G9_RECORD / "oracle_stage_kt00000001_s3.bin").is_file(),
                    reason="no round-46 G9 record")
def test_g9_stage3_fct_reads_the_fold_on_the_pivot_rows(tmp_path):
    """kt 1 of the jitted G9 step from the card's initial state (state traced, x64),
    on NEMO's recorded surface input: after stage 3, T and S on the three rows
    under the T-pivot fold are within the bar of NEMO's, every level.  It needs
    the FCT's faces north of the pivot row as the fold image (traadv_fct.f90:311
    and the halo upstream fluxes and limiter bound around it); without them the
    pivot rows are off by ~1e-4 K / PSU."""
    import jax
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    card = recipe.build_nemo_testcase_card("ORCA2G9-tadv", deck_root=G9_RECORD)
    cfg, zc, s0 = card.recipe.model_config, card.recipe.z_coord, card.recipe.initial_state
    assert cfg.fct_fold_exchange is True
    fw, sf = g5.nemo_supplied_forcing(card, G9_RECORD, 1)
    nk = int(zc.n_levels)
    nemo = g5.read_step(G9_RECORD, 1, [G9_RECORD], nk)
    out = jax.device_get(LatLonCGridOceanModel(card.recipe.grid, zc, cfg).step(
        s0, dt=card.dt_s, freshwater=fw, surface_forcing=sf))
    wet = np.asarray(zc.is_active, dtype=bool)
    for name, key in (("T", "3_tn_all_aa"), ("S", "3_sn_all_aa")):
        ref, got = nemo[key], np.asarray(getattr(out, name).data)
        assert ref.shape == got.shape
        err = np.where(wet, np.abs(got - ref), 0.0) / np.abs(np.where(wet, ref, 0.0)).max()
        assert err[-3:].max() <= 1.0e-15, (name, float(err[-3:].max()))


@needs_inputs
@pytest.mark.skipif(not (G9_RECORD / "oracle_stage_kt00000001_s3.bin").is_file(),
                    reason="no round-46 G9 record")
def test_g9_fct_pivot_row_right_half_is_the_image_of_its_left_half(tmp_path):
    """The lbc_lnk at traadv_fct.f90:311 also overwrites the right half of the stored
    pivot row: zta_up1 (T, +1) and ztFu (U, -1) become the image of the left half of
    the same row (lbcnfd.f90:624-628, :669-673).  On a self-mirrored pivot row that
    is a no-op, so the input here is made asymmetric: the card's kt-1 stage-3 FCT
    inputs with the pivot row's right half of both tracer levels scaled by 1+1e-6.
    Jitted FCT, tracers traced, x64.  The expected values are NEMO's loop indices,
    written out here, not the model's helper."""
    import jax
    import jax.numpy as jnp
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5
    from legoesm.ocean.advection import (
        NEMO_FCT_STENCIL_TRACE_FIELDS, NEMO_FCT_TRACE_FIELDS, fct_tracer_advection)
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    card = recipe.build_nemo_testcase_card("ORCA2G9-tadv", deck_root=G9_RECORD)
    cfg, zc, s0, grid = (card.recipe.model_config, card.recipe.z_coord,
                         card.recipe.initial_state, card.recipe.grid)
    fw, sf = g5.nemo_supplied_forcing(card, G9_RECORD, 1)
    obs = LatLonCGridOceanModel(
        # the FCT-input side output refuses the stage-3 Krhs path; the inputs sit upstream of it
        grid, zc, cfg._replace(rk3_stage3_tracer_krhs_nemo_order=False),
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_stage3_fct_inputs=True)
    ).step(s0, dt=card.dt_s, freshwater=fw, surface_forcing=sf).mass_flux_w
    n = grid.n_lon
    right = jnp.zeros_like(obs.tracer_a).at[-1, n // 2 + 1:].set(1.0e-6)
    now, before = obs.tracer_a * (1.0 + right), obs.tracer_a_before * (1.0 + right)
    assert float(jnp.abs(now[-1] - now[-1][np.r_[0, n - 1:0:-1]]).max()) > 0.0

    def fct(now, before, trace):
        return fct_tracer_advection(
            now, obs.mass_flux_u, obs.mass_flux_v, obs.w_explicit, obs.thickness, grid,
            card.dt_s, high_order="centred2", tracer_before=before,
            active_mask=obs.active_mask, low_order_predictor="nemo_rk3_two_step",
            base_thickness=obs.base_thickness, after_thickness=obs.after_thickness,
            implicit_w=obs.w_implicit, nemo_fold_exchange=True, **{trace: True})[2]

    anti_u = np.asarray(dict(zip(NEMO_FCT_TRACE_FIELDS, jax.jit(
        lambda a, b: fct(a, b, "return_nemo_trace"))(now, before)))["anti_pre_u"])[-1]
    paft = np.asarray(dict(zip(NEMO_FCT_STENCIL_TRACE_FIELDS, jax.jit(
        lambda a, b: fct(a, b, "return_nemo_stencil_trace"))(now, before)))["paft"])[-1]
    # T: native i1 = N/2 + ji <- i2 = N/2 - ji, ji = 1 .. N/2-1 (lbcnfd.f90:624-628).
    ji = np.arange(1, n // 2)
    assert np.array_equal(paft[n // 2 + ji], paft[n // 2 - ji])
    # U (compact column = native U + 1, column 0 the periodic closure):
    # native N/2 - 1 + ji <- N/2 - ji, ji = 1 .. N/2, sign -1 (lbcnfd.f90:669-673).
    ji = np.arange(1, n // 2 + 1)
    assert np.array_equal(anti_u[n // 2 + ji], -anti_u[n // 2 - ji + 1])
    assert np.array_equal(anti_u[0], anti_u[n])


def _pivot_mirror_breaks(t_like, u_like, wet_top):
    """Mirror breaks on the stored T-pivot row, indices from NEMO's T-pivot loops
    (lbcnfd.f90:624-628: native T i <-> (N - i) mod N, sign +1; :669-673: native
    U i <-> N - 1 - i, sign -1).  u_like is the card's compact U (column 0 is the
    periodic closure, columns 1: the native row)."""
    n = t_like[0].shape[1]
    it = (-np.arange(n)) % n
    iu = n - 1 - np.arange(n)
    out = []
    for a in t_like:
        r = np.asarray(a)[-1]
        out.append(int(((r != r[it]) & wet_top & wet_top[it]).sum()))
    r = np.asarray(u_like)[-1, 1:]
    out.append(int((r != -r[iu]).sum()))
    return out


@needs_inputs
@pytest.mark.skipif(not (G9_RECORD / "oracle_stage_kt00000001_s3.bin").is_file(),
                    reason="no round-46 G9 record")
def test_g9_every_stage_ends_with_a_self_mirrored_pivot_row(tmp_path):
    """stprk3_stg.f90:795-796: after every stage NEMO's lbc_lnk rewrites the right
    half of the stored T-pivot row of uu, T and S from its left half (vv is the
    existing top-V-row image).  The entry here is made asymmetric (pivot row's right
    half: T, S scaled by 1+1e-6, u given 1e-3 on wet faces), so only the fold can
    restore the mirror.  Jitted step, state traced, x64; stages 1, 2 via the
    exposure hooks, stage 3 the returned state."""
    import jax
    import jax.numpy as jnp
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    card = recipe.build_nemo_testcase_card("ORCA2G9-tadv", deck_root=G9_RECORD)
    cfg, zc, s0 = card.recipe.model_config, card.recipe.z_coord, card.recipe.initial_state
    assert cfg.rk3_end_of_stage_pivot_fold is True
    fw, sf = g5.nemo_supplied_forcing(card, G9_RECORD, 1)
    wet = np.asarray(zc.is_active, dtype=bool)[-1]
    n = wet.shape[0]
    half = np.arange(n) > n // 2
    scale = np.where(half[:, None] & wet, 1.0 + 1.0e-6, 1.0)
    um = np.asarray(s0.u_mask.data, dtype=float)[-1, 1:]
    du = np.zeros(np.asarray(s0.u.data).shape)
    du[-1, 1:] = np.where((np.arange(n) >= n // 2)[:, None], 1.0e-3, 0.0) * um[..., None]
    entry = s0._replace(
        T=s0.T.replace(data=jnp.asarray(s0.T.data).at[-1].multiply(scale)),
        S=s0.S.replace(data=jnp.asarray(s0.S.data).at[-1].multiply(scale)),
        u=s0.u.replace(data=jnp.asarray(s0.u.data) + du))
    assert min(_pivot_mirror_breaks((entry.T.data, entry.S.data), entry.u.data, wet)) > 0
    for stage in (1, 2, 3):
        hooks = (_NEMOWSRK3TestHooks(expose_momentum_stage=stage, expose_tracer_stage=stage)
                 if stage < 3 else _NEMOWSRK3TestHooks())
        out = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, zc, cfg, _nemo_ws_test_hooks=hooks).step(
                entry, dt=card.dt_s, freshwater=fw, surface_forcing=sf))
        assert _pivot_mirror_breaks((out.T.data, out.S.data), out.u.data, wet) == [0, 0, 0], stage


@needs_inputs
@pytest.mark.skipif(not (G9_RECORD / "oracle_rktracer_operands_kt00000001_s2.bin").is_file(),
                    reason="no round-46 G9 record")
def test_g9_stage_transport_wzv_reads_nemos_stage_clock_pair(tmp_path):
    """sshwzv.f90:297-298 at stages 1 and 2 reads that stage's rDt (rn_Dt/3, rn_Dt/2;
    stprk3_stg.f90:134-135, :188-189) and r3t(Kaa) as the stored blend of r3t(Kbb)
    and r3ta (:177, :221).  kt 1, jitted step, state and forcing traced, x64,
    against NEMO's recorded zFu, zFv, zFw: stage-1 zFw equals NEMO's on every cell;
    stage-2 zFw equals NEMO's on every column whose four face zFu/zFv equal
    NEMO's (those horizontal transports differ upstream of wzv), and
    every stage-2 zFu, zFv, zFw cell is at the bar (|card - NEMO| / max|NEMO| <= 1e-15)."""
    import jax
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    card = recipe.build_nemo_testcase_card("ORCA2G9-tadv", deck_root=G9_RECORD)
    cfg, zc, s0 = card.recipe.model_config, card.recipe.z_coord, card.recipe.initial_state
    assert cfg.rk3_stage_wzv_nemo_clock_pair is True
    fw, sf = g5.nemo_supplied_forcing(card, G9_RECORD, 1)
    nk = int(zc.n_levels)
    for stage in (1, 2):
        # record fields: T_rhs0, S_rhs0, zFu, zFv, zFw, ...
        with open(G9_RECORD / f"oracle_rktracer_operands_kt00000001_s{stage}.bin", "rb") as h:
            h.read(16)
            hd = struct.unpack("=11i", h.read(44))
            v = np.fromfile(h, np.float64)
        nx, ny, nz = hd[7:10]
        n3 = nx * ny * nz
        ref = {name: v[i * n3:(i + 1) * n3].reshape((nx, ny, nz), order="F")
               .transpose(1, 0, 2)[2:-2, 2:-2, :nk]
               for i, name in ((2, "zFu"), (3, "zFv"), (4, "zFw"))}
        model = LatLonCGridOceanModel(
            card.recipe.grid, zc, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_tracer_transport_stage=stage))
        out = jax.device_get(jax.jit(lambda s, f, g: model.step(
            s, dt=card.dt_s, freshwater=f, surface_forcing=g))(s0, fw, sf))
        # the card carries one extra west U face / south V face
        got = {"zFu": np.asarray(out.u.data)[:, 1:], "zFv": np.asarray(out.v.data)[1:],
               "zFw": np.asarray(out.T.data)}
        for name in got:
            assert got[name].shape == ref[name].shape, name
            err = np.abs(got[name] - ref[name]).max() / np.abs(ref[name]).max()
            assert err <= 1e-15, (stage, name, err)
        off_u = (got["zFu"] != ref["zFu"]).any(-1)
        off_v = (got["zFv"] != ref["zFv"]).any(-1)
        if stage == 1:
            assert not off_u.any() and not off_v.any()
        # wzv's stencil (sshwzv.f90:297-298): a column reads its own east/west zFu
        # and north/south zFv
        dirty = off_u | np.roll(off_u, 1, 1) | off_v | np.roll(off_v, 1, 0)
        off_w = (got["zFw"] != ref["zFw"]) & ~dirty[..., None]
        assert not off_w.any(), (stage, int(off_w.sum()))


@needs_inputs
@pytest.mark.skipif(not (G9_RECORD / "oracle_rktracer_stage3_kt00000001.bin").is_file()
                    or not G9_R52_ZRHS.is_file(), reason="no round-46/r52 G9 record")
def test_g9_stage3_tracer_content_is_nemos_trazdf_zrhs(tmp_path):
    """trazdf.f90:284-289: zrhs = e3t(Kbb)*T(Kbb) + (p2dt*e3t(Kmm))*Krhs, Krhs the
    FCT rate plus sbc, runoff, qsr in call order (stprk3_stg.f90:604-717).  kt 1,
    jitted ordinary step, state and forcing traced, x64.  NEMO's round-52
    content record is reconstructed bit-exactly from the passivity-identical
    round-46 RHS operands, and the card's ordinary final T/S is bit-exact on
    every wet level against the record's downstream solution.  Removing only
    the G9 stored-Kmm tracer-WZV wiring restores the scored 105 T / 96 S
    level-1 residue; removing only the G9 stored r3u/r3v stage-transport
    wiring (stprk3_stg.f90:221-223 read at :270-283) restores the full-depth
    residue at (y 59, x 138, level 12)."""
    import jax
    import netCDF4
    sys.path[:0] = [str(CASES), str(CASES / "nemo_testcase_g5_orca2ice")]
    import g5_records as g5
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    card = recipe.build_nemo_testcase_card("ORCA2G9-tadv", deck_root=G9_RECORD)
    cfg, zc, s0 = card.recipe.model_config, card.recipe.z_coord, card.recipe.initial_state
    assert cfg.rk3_stage3_tracer_krhs_nemo_order is True
    fw, sf = g5.nemo_supplied_forcing(card, G9_RECORD, 1)
    nk = int(zc.n_levels)
    wet = np.asarray(zc.is_active, bool)
    # RKTR3: 16 3-D fields (zero, adv, sbc, qsr, ldf, Kbb, Kmm, Kaa; T then S), r3t x3
    with open(G9_RECORD / "oracle_rktracer_stage3_kt00000001.bin", "rb") as h:
        assert h.read(16).decode().strip() == "NEMO_L2_RKTR3_1"
        hd = struct.unpack("=11i", h.read(44))
        v = np.fromfile(h, np.float64)
    nx, ny, nz = hd[7:10]
    n3, n2 = nx * ny * nz, nx * ny
    f3 = lambda i: v[i * n3:(i + 1) * n3].reshape((nx, ny, nz), order="F").transpose(1, 0, 2)[2:-2, 2:-2, :nk]  # noqa: E731
    f2 = lambda i: v[16 * n3 + i * n2:16 * n3 + (i + 1) * n2].reshape((nx, ny), order="F").T[2:-2, 2:-2, None]  # noqa: E731
    with G9_R52_ZRHS.open("rb") as h:
        assert h.read(16).decode() == "NEMO_G9_ZRHS_1  "
        r52 = struct.unpack("=13i", h.read(52))
        assert r52 == (1, 1, 1, 2, 3, 3, nx, ny, nz, 2, 64, 4, 10)
        assert h.read(16).decode().rstrip() == "zrhs_content"
        r52_content = np.fromfile(h, np.float64, count=2 * n3).reshape(
            (nx, ny, nz, 2), order="F").transpose(1, 0, 2, 3)[
                2:-2, 2:-2, :nk]
        assert h.read(16).decode().rstrip() == "matrix_diags"
        h.seek(4 * n3 * 8, 1)
        assert h.read(16).decode().rstrip() == "forward_state"
        h.seek(2 * n3 * 8, 1)
        assert h.read(16).decode().rstrip() == "solution"
        r52_solution = np.fromfile(h, np.float64, count=2 * n3).reshape(
            (nx, ny, nz, 2), order="F").transpose(1, 0, 2, 3)[
                2:-2, 2:-2, :nk]
    mesh = netCDF4.Dataset(G9_RECORD / "mesh_mask.nc")
    e3t0 = np.asarray(mesh["e3t_0"][0]).transpose(1, 2, 0)[..., :nk]
    tmask = np.asarray(mesh["tmask"][0]).transpose(1, 2, 0)[..., :nk].astype(float)
    mesh.close()
    e3bb, e3mm = e3t0 * (1.0 + f2(0) * tmask), e3t0 * (1.0 + f2(1) * tmask)
    dt = float(card.dt_s)

    assert cfg.rk3_stage_transport_stored_r3uv is True

    def step(**off):
        model = LatLonCGridOceanModel(card.recipe.grid, zc, cfg._replace(**off))
        return jax.device_get(jax.jit(lambda s, f, g: model.step(
            s, dt=card.dt_s, freshwater=f, surface_forcing=g))(s0, fw, sf))

    selected = step()
    no_wzv = step(rk3_stage_tracer_wzv_stored_kmm_ratios=False)
    no_r3uv = step(rk3_stage_transport_stored_r3uv=False)
    for name, i in (("T", 0), ("S", 1)):
        zrhs = e3bb * f3(10 + i) + (dt * e3mm) * f3(6 + i)   # Kbb, after-qsr (= after-ldf) Krhs
        assert np.array_equal(zrhs[wet], r52_content[..., i][wet]), name
        ref = r52_solution[..., i]
        assert np.array_equal(ref[wet], f3(14 + i)[wet]), name
        got = np.asarray(getattr(selected, name).data)
        assert np.array_equal(got[wet], ref[wet]), name
        wet1 = wet[..., 0]
        old = np.asarray(getattr(no_wzv, name).data)[..., 0]
        assert int(np.count_nonzero(old[wet1] != ref[..., 0][wet1])) == {
            "T": 105, "S": 96}[name]
        old = np.asarray(getattr(no_r3uv, name).data)
        assert np.argwhere((old != ref) & wet).tolist() == {
            "T": [[59, 138, 11]], "S": []}[name]


def test_g9_stage3_krhs_and_content_are_nemos_statements(monkeypatch):
    """stprk3_stg.f90:604 zeroes Krhs; the FCT rate (:637), then qns/sfx and runoff
    (trasbc.f90:307-308, :323) and qsr (:717) add to it in call order; trazdf.f90:288-289
    forms zrhs = e3t(Kbb)*T(Kbb) + (p2dt*e3t(Kmm))*Krhs.  Synthetic traced operands
    (magnitudes spread so association shows), jitted, x64, against a numpy transcription
    of those lines: bit-equal on every cell."""
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from legoesm.grids.latlon import create_latlon_grid
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as mm

    rng = np.random.default_rng(49)
    shape = (24, 24, 3)
    T, S = rng.uniform(-2.0, 30.0, shape), rng.uniform(33.0, 37.0, shape)
    e3bb, e3mm = rng.uniform(9.9, 10.1, shape), rng.uniform(9.9, 10.1, shape)
    adv_T, adv_S = rng.normal(scale=1e-6, size=shape), rng.normal(scale=1e-7, size=shape)
    terms_T = [rng.normal(scale=s, size=shape) for s in (3e-5, 1e-9, 2e-6)]  # sbc, rnf, qsr
    terms_S = [rng.normal(scale=s, size=shape) for s in (4e-6, 1e-9)]        # sbc, rnf
    dt = 10800.0
    base = create_latlon_grid(24, 24, dtype=jnp.float64)
    grid = SimpleNamespace(**base._asdict(), dy_u=jnp.ones((24, 25)),
                           dx_v=jnp.ones((25, 24)), area_T=jnp.asarray(base.area))
    seen = []

    def content(T, S, e3bb, e3mm, adv_T, adv_S, terms_T, terms_S, dt):
        def fct_pair(a, b, scheme, *args, **kw):
            seen.append(kw.get("return_fct_nemo_rhs"))
            z = jnp.zeros_like(a)
            return (z, z, adv_T), (z, z, adv_S)
        monkeypatch.setattr(mm, "compute_advection_flux_div_pair", fct_pair)
        h, mf_u, mf_v = jnp.ones_like(T), jnp.ones(shape[:1] + (25, 3)), jnp.ones((25,) + shape[1:])
        w = jnp.ones(shape[:2] + (4,))
        stage_geometry = (mf_u, mf_v, w, h, mf_u, mf_v, None, mf_u, mf_v)
        return mm._nemo_ws_rk3_tracer_pair_step(
            T, S, "fct2", mf_u, mf_v, w, h, h, mf_u, mf_v,
            grid, dt, h, stage3_content_written_order=(e3bb, e3mm),
            stage3_krhs_terms=(tuple(terms_T), tuple(terms_S)),
            stage_transport_geometry=(stage_geometry,) * 3,
            return_final_content=True)[2:4]

    got = jax.device_get(jax.jit(content)(T, S, e3bb, e3mm, adv_T, adv_S,
                                          terms_T, terms_S, jnp.float64(dt)))
    assert seen == [True]
    for g, tr, adv, terms in ((got[0], T, adv_T, terms_T), (got[1], S, adv_S, terms_S)):
        krhs = np.zeros(shape)                    # stprk3_stg.f90:604
        krhs = krhs + adv                         # :637
        for term in terms:                        # trasbc :307-308, :323; qsr :717
            krhs = krhs + term
        zrhs = e3bb * tr + (dt * e3mm) * krhs     # trazdf.f90:288-289
        np.testing.assert_array_equal(np.asarray(g), zrhs)


def test_g9_stage3_fct_divides_by_stored_kmm_thickness(monkeypatch):
    """traadv_fct.f90:538 reads stored e3t(Kmm), whose r3t is assigned at
    stprk3_stg.f90:221.  Synthetic traced x64 operands make the divisor
    observable in every cell; restoring the old live-stage divisor fails
    exactly."""
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from legoesm.grids.latlon import create_latlon_grid
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as mm

    shape = (4, 4, 2)
    base = create_latlon_grid(4, 4, dtype=jnp.float64)
    grid = SimpleNamespace(**base._asdict(), dy_u=jnp.ones((4, 5)),
                           dx_v=jnp.ones((5, 4)), area_T=jnp.asarray(base.area))

    def fct_pair(a, b, scheme, *args, **kwargs):
        assert scheme == "fct2" and kwargs["return_fct_nemo_rhs"] is True
        zero = jnp.zeros_like(a)
        reciprocal_divisor = jnp.reciprocal(args[3])
        return ((zero, zero, reciprocal_divisor),
                (zero, zero, reciprocal_divisor))

    monkeypatch.setattr(mm, "compute_advection_flux_div_pair", fct_pair)
    tracer = jnp.arange(np.prod(shape), dtype=jnp.float64).reshape(shape) + 1.0
    e3bb = jnp.linspace(9.8, 10.2, np.prod(shape), dtype=jnp.float64).reshape(shape)
    e3mm = jnp.linspace(10.3, 10.9, np.prod(shape), dtype=jnp.float64).reshape(shape)
    live = jnp.linspace(11.1, 11.7, np.prod(shape), dtype=jnp.float64).reshape(shape)
    mf_u = jnp.ones((4, 5, 2), dtype=jnp.float64)
    mf_v = jnp.ones((5, 4, 2), dtype=jnp.float64)
    w = jnp.ones((4, 4, 3), dtype=jnp.float64)
    dt = jnp.asarray(10800.0, dtype=jnp.float64)

    def run(tr, stored, live_arg, dt_arg):
        geom = (mf_u, mf_v, w, live_arg, mf_u, mf_v, None, mf_u, mf_v)
        return mm._nemo_ws_rk3_tracer_pair_step(
            tr, tr, "fct2", mf_u, mf_v, w, e3bb, stored, mf_u, mf_v,
            grid, dt_arg, jnp.ones_like(tr),
            stage3_content_written_order=(e3bb, stored),
            stage3_krhs_terms=((), ()), stage_transport_geometry=(geom,) * 3,
            resume=(2, tr, tr), return_final_content=True)[2]

    got = np.asarray(jax.jit(run)(tracer, e3mm, live, dt))
    expected = np.add(
        np.multiply(np.asarray(e3bb), np.asarray(tracer)),
        np.multiply(np.multiply(np.float64(10800.0), np.asarray(e3mm)),
                    np.reciprocal(np.asarray(e3mm))))
    old_divisor = np.add(
        np.multiply(np.asarray(e3bb), np.asarray(tracer)),
        np.multiply(np.multiply(np.float64(10800.0), np.asarray(e3mm)),
                    np.reciprocal(np.asarray(live))))
    np.testing.assert_array_equal(got, expected)
    assert np.count_nonzero(got != old_divisor) == got.size
