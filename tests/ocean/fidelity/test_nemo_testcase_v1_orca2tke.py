"""Vertical-mixing lane V1: certified G5 plus ORCA2's TKE closure, one switch."""

from __future__ import annotations

import difflib
import re
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G5 = CASES / "nemo_testcase_g5_orca2ice"
V1 = CASES / "nemo_testcase_v1_orca2tke"
RUN = V1 / "run.sh"
INPUTS = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0")
NEMO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2")
ORCA2_DECK = NEMO / "cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg"
G5_STPRK3 = NEMO / "cfgs/ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/MY_SRC/stprk3.F90"
CARD_FILES = ("ORCA_R2_zps_domcfg.nc", "data_1m_potential_temperature_nomask.nc",
              "data_1m_salinity_nomask.nc")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 inputs")
needs_nemo = pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
TKE_KEYS = {"namzdf_tke.nn_mxl": 3, "namzdf_tke.nn_etau": 1,
            "namzdf_tke.ln_mxhsw": False, "namzdf_tke.nn_mxlice": 2}


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _values(path):
    return recipe._orca2g1_namelist_values(path)


def _root(base: Path, rung: Path, name: str | None = None) -> Path:
    root = base / (name or rung.name)
    root.mkdir()
    for deck in ("namelist_cfg", "namelist_ice_cfg"):
        shutil.copy(rung / deck, root / deck)
    (root / "domain_cfg.nc").symlink_to(INPUTS / CARD_FILES[0])
    for name in CARD_FILES[1:]:
        (root / name).symlink_to(INPUTS / name)
    return root


def test_v1_deck_is_g5_plus_tke_only():
    g5, v1 = _values(G5 / "namelist_cfg"), _values(V1 / "namelist_cfg")
    changed = {key: (g5.get(key), v1.get(key)) for key in set(g5) | set(v1)
               if g5.get(key) != v1.get(key)}
    assert changed == {"namzdf.ln_zdfcst": (True, False),
                       "namzdf.ln_zdftke": (False, True),
                       **{k: (None, v) for k, v in TKE_KEYS.items()}}
    assert (G5 / "namelist_ice_cfg").read_bytes() == (V1 / "namelist_ice_cfg").read_bytes()
    assert recipe._ORCA2V1_TKE_FIELDS == TKE_KEYS


@needs_nemo
def test_v1_tke_lines_are_orca2s_own():
    orca2 = _values(ORCA2_DECK)
    assert {k: v for k, v in orca2.items() if k.startswith("namzdf_tke.")} == TKE_KEYS
    assert orca2["namzdf.ln_zdftke"] is True and "namzdf.ln_zdfcst" not in orca2


def test_reader_refuses_tke_on_the_wrong_rung_and_a_planted_line(tmp_path):
    with pytest.raises(ValueError, match="namzdf"):
        recipe.read_orca2g1_deck(V1, "ORCA2G5-ice")
    with pytest.raises(ValueError, match="namzdf"):
        recipe.read_orca2g1_deck(G5, "ORCA2V1-tke")
    planted = tmp_path / "planted"
    planted.mkdir()
    text = (V1 / "namelist_cfg").read_text()
    assert text.count("nn_mxlice = 2") == 1
    (planted / "namelist_cfg").write_text(text.replace("nn_mxlice = 2", "nn_mxlice = 0"))
    with pytest.raises(ValueError, match="namzdf_tke"):
        recipe.read_orca2g1_deck(planted, "ORCA2V1-tke")
    assert recipe.read_orca2g1_deck(V1, "ORCA2V1-tke") == recipe.read_orca2g1_deck(
        G5, "ORCA2G5-ice")


def _flat(x, pre=""):
    if hasattr(x, "_asdict"):
        out = {}
        for k, v in x._asdict().items():
            out.update(_flat(v, f"{pre}{k}."))
        return out
    return {pre[:-1]: x}


def _differs(a, b) -> bool:
    if hasattr(a, "shape") or hasattr(b, "shape"):
        return not np.array_equal(np.asarray(a), np.asarray(b))
    return a != b


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("v1")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        g5 = recipe.build_orca2g5_ice_card(_root(base, G5))
        v1_root = _root(base, V1)
        v1 = recipe.build_orca2v1_tke_card(v1_root)
    finally:
        set_policy(before)
    return g5, v1, v1_root


@needs_inputs
def test_v1_card_is_g5_plus_the_tke_selection(cards):
    g5, v1, _ = cards
    a, b = _flat(g5.recipe.model_config), _flat(v1.recipe.model_config)
    moved = {k for k in set(a) | set(b) if _differs(a.get(k), b.get(k))}
    assert moved == {"physics.vertical_mixing.scheme",
                     "physics.vertical_mixing.vmix_background_mode",
                     "physics.vertical_mixing.tke.etau_mode",
                     "physics.vertical_mixing.tke.etau_htau_mode"}
    vm = v1.recipe.model_config.physics.vertical_mixing
    t = vm.tke
    # Rule 10: the numbers the closure will use, as instantiated.
    assert (vm.scheme, vm.vmix_background_mode, vm.iwm.enabled, vm.ddm.enabled) == (
        "tke", "nemo_max_floor", False, False)
    assert (t.tke_background, t.tke_surface_min, t.nemo_derived_mxl_min,
            t.nemo_mxl0_rmxl_min_overwrite, t.tke_mxl_choice) == (1e-6, 1e-4, True, True, 3)
    assert (t.lc, t.lc_coeff, t.etau_mode, t.etau_htau_mode, t.etau_frac, t.eice,
            t.bottom_tke_bc) == (True, 0.15, "below_ml", "latitude", 0.05, 1, False)
    assert (t.kappaM_min, t.kappaH_min, t.c_k, t.c_eps) == (1.2e-4, 1.2e-5, 0.1, 0.7)
    assert v1.recipe.model_config.physics.convection.scheme == "none"   # ln_zdfevd=F
    assert v1.unmeasured_features == recipe.ORCA2V1_UNMEASURED
    assert "tke_nn_mxlice2_under_ice_mixing_length" in v1.unmeasured_features
    assert (v1.surface_boundary_condition, v1.surface_input_operator) == (
        g5.surface_boundary_condition, g5.surface_input_operator)
    assert (v1.dt_s, v1.n_steps) == (g5.dt_s, g5.n_steps)


@needs_inputs
def test_v1_initial_state_is_g5_plus_nemo_tke_rest_seeds(cards):
    g5, v1, _ = cards
    sa, sb = g5.recipe.initial_state, v1.recipe.initial_state
    moved = set()
    for name in sa._fields:
        x, y = getattr(sa, name), getattr(sb, name)
        if _differs(getattr(x, "data", x), getattr(y, "data", y)):
            moved.add(name)
    assert moved == {"tke", "tke_avm", "tke_avt", "tke_dissl", "tke_avm_surface"}
    wet_w = np.asarray(v1.recipe.z_coord.is_active, dtype=np.float64)[..., 1:]
    assert np.array_equal(np.asarray(sb.tke.data), 1e-6 * wet_w)
    assert np.array_equal(np.asarray(sb.tke_avm.data), 1.2e-4 * wet_w)
    assert np.array_equal(np.asarray(sb.tke_avt.data), 1.2e-5 * wet_w)
    assert np.array_equal(np.asarray(sb.tke_dissl.data), 1e-12 * wet_w)


@needs_inputs
def test_v1_selector_check_fires_on_each_planted_tke_line(cards):
    _, v1, _ = cards
    assert recipe._orca2g1_selector_failures(v1) == []
    cfg = v1.recipe.model_config
    vm = cfg.physics.vertical_mixing
    for plant in ({"etau_mode": "none"}, {"eice": 0}, {"bottom_tke_bc": True},
                  {"tke_background": 1e-10}, {"lc": False}):
        bad = vm._replace(tke=vm.tke._replace(**plant))
        card = v1._replace(recipe=v1.recipe._replace(model_config=cfg._replace(
            physics=cfg.physics._replace(vertical_mixing=bad))))
        assert recipe._orca2g1_selector_failures(card) == ["ln_zdftke"], plant
    for plant in ({"scheme": "constant"}, {"vmix_background_mode": "additive"}):
        card = v1._replace(recipe=v1.recipe._replace(model_config=cfg._replace(
            physics=cfg.physics._replace(vertical_mixing=vm._replace(**plant)))))
        assert "ln_zdftke" in recipe._orca2g1_selector_failures(card), plant


def test_run_script_is_fail_closed_smoke_first_and_never_uses_step1_restart():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    assert text.index('run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE"') < text.index(
        'run_one "$REF_BIN" "$REF" 100')
    assert "list_b() { pad10 $(seq 2 2" in text and "pad10 $(seq 3 2" in text
    # writer-free reference = G5's pinned binary, never rebuilt
    assert "ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe" in text
    assert text.count("makenemo -r") == 1
    resume = text[text.index('if [[ "$mode" == --resume ]]'):text.index("else\n", text.index(
        'if [[ "$mode" == --resume ]]'))]
    assert "admit_smoke_in_place" in resume and "prepare_resume_target" in resume
    assert 'run_one "$TARGET/BLD/bin/nemo.exe" "$SMOKE"' not in resume


@needs_nemo
def test_hmi_patch_is_additive_write_only_and_refuses_a_moved_anchor(tmp_path):
    out = tmp_path / "stprk3.F90"
    ok = subprocess.run([str(RUN), "--patch-stprk3", str(G5_STPRK3), str(out)],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    old, new = G5_STPRK3.read_text().splitlines(), out.read_text().splitlines()
    ops = difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes()
    assert {tag for tag, *_ in ops} == {"equal", "insert"}          # additive only
    added = [line for tag, _, _, j1, j2 in ops if tag == "insert" for line in new[j1:j2]]
    assert len(new) - len(old) == len(added) == 21
    assert not any(re.match(r"\s*hm_i\s*[(=]", line) for line in added)   # never assigned
    assert any("SIZE(hm_i,1), SIZE(hm_i,2)" in line for line in added)
    call = new.index(next(line for line in new if "CALL v1_dump_hm_i( kstp )" in line))
    assert "CALL l4_dump_ocean_surface_input( kstp, Nbb )" in new[call - 1]
    assert "CALL sbc        ( kstp, Nbb, Nbb )" in new[call - 2]
    moved = tmp_path / "moved.F90"
    moved.write_text(G5_STPRK3.read_text().replace(
        "CALL l4_dump_ocean_surface_input( kstp, Nbb )", "CALL l4_dump_ocean_surface_input(kstp,Nbb)"))
    bad = subprocess.run([str(RUN), "--patch-stprk3", str(moved), str(tmp_path / "x.F90")],
                         capture_output=True, text=True)
    assert bad.returncode != 0 and "anchors" in bad.stderr


def _hmi_frame(
    path: Path, kt: int, values: np.ndarray, shape: tuple[int, int] = (4, 3)
) -> None:
    head = b"NEMO_V1_HMI_1".ljust(16) + struct.pack("=5i", 1, kt, *shape, 64)
    path.write_bytes(head + values.astype("<f8").tobytes())


def test_hmi_frame_gate_passes_good_frames_and_refuses_plants(tmp_path):
    for kt in (1, 2):
        _hmi_frame(tmp_path / f"oracle_v1_hm_i_kt{kt:08d}.bin", kt, np.full(12, 0.5))
    run = lambda: subprocess.run([str(RUN), "--check-hmi-frames", str(tmp_path), "2"],  # noqa: E731
                                 capture_output=True, text=True)
    assert run().returncode == 0
    frame = tmp_path / "oracle_v1_hm_i_kt00000002.bin"
    for values in (np.full(11, 0.5), np.array([np.nan] + [0.5] * 11), np.full(12, -1.0)):
        _hmi_frame(frame, 2, values)
        assert run().returncode != 0
    _hmi_frame(frame, 1, np.full(12, 0.5))       # wrong kt in the header
    assert run().returncode != 0


def test_hmi_frame_gate_refuses_the_legacy_writer_shape_and_bad_legacy_plant(tmp_path):
    values = np.full(180 * 148, 0.5)
    for kt in (1, 2):
        _hmi_frame(tmp_path / f"oracle_v1_hm_i_kt{kt:08d}.bin", kt, values,
                   shape=(184, 152))
    strict = subprocess.run([str(RUN), "--check-hmi-frames", str(tmp_path), "2"],
                            capture_output=True, text=True)
    assert strict.returncode != 0 and "26640" in strict.stderr
    legacy = lambda: subprocess.run(  # noqa: E731
        [str(RUN), "--check-legacy-hmi-frames", str(tmp_path), "2"],
        capture_output=True, text=True)
    assert legacy().returncode == 0
    # Plant: even the narrow round-1 admission refuses a different halo header.
    _hmi_frame(tmp_path / "oracle_v1_hm_i_kt00000002.bin", 2, values, shape=(184, 151))
    assert legacy().returncode != 0


def test_additive_guard_refuses_a_removed_or_changed_line(tmp_path):
    (tmp_path / "o").write_text("a\nb\nc\n")
    (tmp_path / "ok").write_text("a\nX\nb\nc\n")
    (tmp_path / "bad").write_text("a\nc\nZ\n")
    check = lambda new: subprocess.run(  # noqa: E731
        [str(RUN), "--check-additive", str(tmp_path / "o"), str(tmp_path / new)],
        capture_output=True, text=True).returncode
    assert check("ok") == 0
    assert check("bad") != 0
