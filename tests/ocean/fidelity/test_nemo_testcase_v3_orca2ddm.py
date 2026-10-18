"""Vertical-mixing lane V3: certified V2 plus ORCA2's double-diffusive mixing."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import jax
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES, CASES, INPUTS, ORCA2_DECK, V1, _differs, _flat, _root, _values,
)

V2 = CASES / "nemo_testcase_v2_orca2evd"
V3 = CASES / "nemo_testcase_v3_orca2ddm"
RUN = V3 / "run.sh"
DDM_KEYS = {"namzdf.ln_zdfddm": True, "namzdf.rn_avts": 1.0e-4, "namzdf.rn_hsbfr": 1.6}
V2_FRAME = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_vmix_rounds/"
                "round12/oracle_v2_r12/twin_a")
needs_inputs = pytest.mark.skipif(
    not all((INPUTS / name).is_file() for name in CARD_FILES), reason="no ORCA2 inputs")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def test_v3_deck_is_v2_plus_orca2s_ddm_lines_only():
    v2, v3 = _values(V2 / "namelist_cfg"), _values(V3 / "namelist_cfg")
    changed = {k: (v2.get(k), v3.get(k)) for k in set(v2) | set(v3) if v2.get(k) != v3.get(k)}
    assert changed == {"namzdf.ln_zdfddm": (False, True),
                       "namzdf.rn_avts": (None, 1.0e-4), "namzdf.rn_hsbfr": (None, 1.6)}
    assert (V2 / "namelist_ice_cfg").read_bytes() == (V3 / "namelist_ice_cfg").read_bytes()
    assert recipe._ORCA2V3_DDM_FIELDS == DDM_KEYS


@pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
def test_v3_ddm_lines_are_orca2s_own():
    orca2 = _values(ORCA2_DECK)
    assert {k: orca2[k] for k in DDM_KEYS} == DDM_KEYS


def test_reader_refuses_ddm_on_the_wrong_rung_and_a_planted_line(tmp_path):
    with pytest.raises(ValueError, match="ln_zdfddm"):
        recipe.read_orca2g1_deck(V3, "ORCA2V2-evd")
    with pytest.raises(ValueError, match="ln_zdfddm"):
        recipe.read_orca2g1_deck(V2, "ORCA2V3-ddm")
    planted = tmp_path / "planted"
    planted.mkdir()
    text = (V3 / "namelist_cfg").read_text()
    assert text.count("rn_hsbfr =    1.6") == 1
    (planted / "namelist_cfg").write_text(text.replace("rn_hsbfr =    1.6", "rn_hsbfr =    2.0"))
    with pytest.raises(ValueError, match="rn_hsbfr"):
        recipe.read_orca2g1_deck(planted, "ORCA2V3-ddm")
    assert recipe.read_orca2g1_deck(V3, "ORCA2V3-ddm") == recipe.read_orca2g1_deck(
        V2, "ORCA2V2-evd")


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("v3")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        v2 = recipe.build_orca2v2_evd_card(_root(base, V2))
        v3 = recipe.build_orca2v3_ddm_card(_root(base, V3))
    finally:
        set_policy(before)
    return v2, v3


@needs_inputs
def test_v3_card_is_v2_plus_the_ddm_selection_only(cards):
    v2, v3 = cards
    a, b = _flat(v2.recipe.model_config), _flat(v3.recipe.model_config)
    moved = {k for k in set(a) | set(b) if _differs(a.get(k), b.get(k))}
    assert moved == {"physics.vertical_mixing.ddm.enabled",
                     "physics.vertical_mixing.ddm.form",
                     "physics.convection.enhanced_diffusion.trigger_step_entry_bn2"}
    assert v3.recipe.model_config.physics.convection.enhanced_diffusion.trigger_step_entry_bn2
    assert not v2.recipe.model_config.physics.convection.enhanced_diffusion.trigger_step_entry_bn2
    ddm = v3.recipe.model_config.physics.vertical_mixing.ddm
    assert (ddm.enabled, ddm.form, ddm.rn_avts, ddm.rn_hsbfr) == (
        True, "nemo_literal", 1.0e-4, 1.6)
    assert not v2.recipe.model_config.physics.vertical_mixing.ddm.enabled
    s2, s3 = v2.recipe.initial_state, v3.recipe.initial_state
    assert not [n for n in s2._fields if _differs(
        getattr(getattr(s2, n), "data", getattr(s2, n)),
        getattr(getattr(s3, n), "data", getattr(s3, n)))]
    assert (v3.dt_s, v3.n_steps, v3.unmeasured_features) == (
        v2.dt_s, v2.n_steps, v2.unmeasured_features)


@needs_inputs
def test_v3_selector_fires_on_each_planted_ddm_field(cards):
    v2, v3 = cards
    assert recipe._orca2g1_selector_failures(v3) == []
    assert recipe._orca2g1_selector_failures(v2) == []
    for card, plant, row in (
            (v3, {"form": "hybrid"}, "ln_zdfddm"), (v3, {"enabled": False}, "ln_zdfddm"),
            (v3, {"rn_avts": 2e-4}, "ln_zdfddm"), (v3, {"rn_hsbfr": 2.0}, "ln_zdfddm"),
            (v2, {"enabled": True}, "ln_zdfddm=F")):
        cfg = card.recipe.model_config
        vm = cfg.physics.vertical_mixing
        bad = card._replace(recipe=card.recipe._replace(model_config=cfg._replace(
            physics=cfg.physics._replace(vertical_mixing=vm._replace(
                ddm=vm.ddm._replace(**plant))))))
        assert recipe._orca2g1_selector_failures(bad) == [row], plant
    for card, flag in ((v3, False), (v2, True)):
        cfg = card.recipe.model_config
        conv = cfg.physics.convection
        bad = card._replace(recipe=card.recipe._replace(model_config=cfg._replace(
            physics=cfg.physics._replace(convection=conv._replace(
                enhanced_diffusion=conv.enhanced_diffusion._replace(
                    trigger_step_entry_bn2=flag))))))
        assert recipe._orca2g1_selector_failures(bad) == ["ln_zdfevd"], flag


def test_run_script_reuses_v1_binaries_unbuilt_and_is_fail_closed():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    assert "makenemo" not in text
    assert "ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/bin/nemo.exe" in text
    assert "ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe" in text
    for name in ("zdfddm.f90", "trazdf.f90", "zdf_oce.f90", "zdfddm.o",
                 "use double diffusive mixing: avs /= avt", "check_ddm_frame \"$TWIN_A\""):
        assert name in text, name


@pytest.mark.skipif(not (V2_FRAME / "oracle_zdf_entry_kt00000001.bin").is_file(),
                    reason="no V2 record")
def test_ddm_frame_check_refuses_v2s_record():
    """V2 runs without DDM, so its kt-1 frame has avs == avt everywhere."""
    out = subprocess.run([str(RUN), "--check-ddm-frame", str(V2_FRAME)],
                         capture_output=True, text=True)
    assert out.returncode == 66, out
    assert "V3_DDM_FRAME unequal_avs_avt=0 of 825840" in out.stdout


def kt2_restarts_differ(before: Path, after: Path) -> bool:
    """Control: the rung moves NEMO's kt-2 T, so its record is not the last rung's."""
    import netCDF4
    import numpy as np

    kt2 = "twin_b/ORCA2G1_00000002_restart.nc"
    with netCDF4.Dataset(before / kt2) as a, netCDF4.Dataset(after / kt2) as b:
        return bool(np.count_nonzero(np.asarray(a["tn"][:]) != np.asarray(b["tn"][:])))


def assert_record_bit_for_bit(twins, case: str, *, kt_to: int) -> dict:
    """kt-1 consumed avt/avs/avm (wet and dry) and every independent-arm field
    kt 1..kt_to equal NEMO's record; returns the kt-1 frame for the caller's
    rung-specific checks."""
    sys.path.insert(0, str(CASES))
    spec = importlib.util.spec_from_file_location(
        "orca2_ladder", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    frame = ladder.run(twins[0], arm="ddm_kt1", kt_from=1, kt_to=1,
                       restart_roots=twins, allow_dirty=True, case=case)
    for f in ("avt", "avs", "avm"):
        assert frame["rows"][f]["n_unequal_wet"] == 0, f
        assert frame["rows"][f]["summary"]["dry_n_unequal"] == 0, f
    out = ladder.run(twins[0], arm="independent", kt_from=1, kt_to=kt_to,
                     restart_roots=twins, allow_dirty=True, case=case)
    assert [k["kt"] for k in out["per_kt"]] == list(range(1, kt_to + 1))
    for k in out["per_kt"]:
        bad = [r["name"] for r in k["rows"] if r["n_unequal"]]
        for name, lv in (k["all_levels"] or {}).items():
            s = lv.get("summary")
            if (lv["n_unequal"] if s is None else
                    s["full"]["n_unequal"] or s["partial"]["n_unequal"]
                    or s["dry_n_unequal"]):
                bad.append(name)
        assert not bad, f"kt {k['kt']} unequal: {bad}"
    return frame


V3_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_vmix_rounds/round13/oracle_v3_r13")
V3_TWINS = [V3_RECORD / t for t in ("twin_a", "twin_b", "twin_c")]


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
@pytest.mark.skipif(not all(t.is_dir() for t in V3_TWINS)
                    or not (V2_FRAME.parent / "twin_b").is_dir(),
                    reason="the operator's V2/V3 records are not on this machine")
def test_v3_kt1_3_every_field_and_kt1_avt_avs_avm_bit_for_bit():
    """kt 2 needs zdf_ddm's true division by rn_hsbfr under jit; kt 3 (one
    column) needs EVD's trigger on the step-entry rn2/rn2b."""
    assert kt2_restarts_differ(V2_FRAME.parent, V3_RECORD)   # DDM moves kt-2 T
    census = assert_record_bit_for_bit(V3_TWINS, "ORCA2V3-ddm", kt_to=3)["census"]
    assert census["nemo"] == census["card"]
    assert census["nemo"]["avs_gt_avt"] > 0 and census["nemo"]["avs_lt_avt"] > 0
