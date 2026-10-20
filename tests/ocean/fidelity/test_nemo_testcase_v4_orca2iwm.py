"""Vertical-mixing lane V4: certified V3 plus ORCA2's internal-wave mixing."""

from __future__ import annotations

import subprocess
from pathlib import Path

import jax
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES, CASES, INPUTS, ORCA2_DECK, _differs, _flat, _root, _values,
)
from tests.ocean.fidelity.test_nemo_testcase_v3_orca2ddm import (
    V3_RECORD, V3_TWINS, assert_record_bit_for_bit, kt2_restarts_differ,
)

V3 = CASES / "nemo_testcase_v3_orca2ddm"
V4 = CASES / "nemo_testcase_v4_orca2iwm"
RUN = V4 / "run.sh"
IWM_FILE = INPUTS / "zdfiwm_forcing_orca2.nc"
IWM_KEYS = {"namzdf.ln_zdfiwm": True, "namzdf_iwm.ln_mevar": False,
            "namzdf_iwm.ln_tsdiff": True, "namzdf_iwm.cn_dir": "'./'"}


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def test_v4_deck_is_v3_plus_orca2s_iwm_lines_only():
    v3, v4 = _values(V3 / "namelist_cfg"), _values(V4 / "namelist_cfg")
    changed = {k: (v3.get(k), v4.get(k)) for k in set(v3) | set(v4) if v3.get(k) != v4.get(k)}
    assert changed.pop("namzdf.ln_zdfiwm") == (False, True)
    assert {k: v for k, (old, v) in changed.items() if old is None} == recipe._ORCA2V4_IWM_LINES
    assert all(old is None for old, _ in changed.values())
    assert (V3 / "namelist_ice_cfg").read_bytes() == (V4 / "namelist_ice_cfg").read_bytes()
    assert {**recipe._ORCA2V4_IWM_FIELDS, **recipe._ORCA2V4_IWM_LINES}.items() >= IWM_KEYS.items()


@pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
def test_v4_iwm_lines_are_orca2s_own():
    orca2 = _values(ORCA2_DECK)
    assert orca2["namzdf.ln_zdfiwm"] is True
    assert {k: v for k, v in orca2.items() if k.startswith("namzdf_iwm.")} == (
        recipe._ORCA2V4_IWM_LINES)


def test_reader_refuses_iwm_on_the_wrong_rung_and_a_planted_line(tmp_path):
    with pytest.raises(ValueError, match="ln_zdfiwm|namzdf_iwm"):
        recipe.read_orca2g1_deck(V4, "ORCA2V3-ddm")
    with pytest.raises(ValueError, match="ln_zdfiwm|namzdf_iwm"):
        recipe.read_orca2g1_deck(V3, "ORCA2V4-iwm")
    planted = tmp_path / "planted"
    planted.mkdir()
    text = (V4 / "namelist_cfg").read_text()
    old = "ln_tsdiff   = .true."
    assert text.count(old) == 1
    (planted / "namelist_cfg").write_text(text.replace(old, "ln_tsdiff   = .false."))
    with pytest.raises(ValueError, match="namzdf_iwm"):
        recipe.read_orca2g1_deck(planted, "ORCA2V4-iwm")
    assert recipe.read_orca2g1_deck(V4, "ORCA2V4-iwm") == recipe.read_orca2g1_deck(
        V3, "ORCA2V3-ddm")


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in (*CARD_FILES, IWM_FILE.name)):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("v4")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        v3 = recipe.build_orca2v3_ddm_card(_root(base, V3))
        root4 = _root(base, V4)
        (root4 / IWM_FILE.name).symlink_to(IWM_FILE)
        v4 = recipe.build_orca2v4_iwm_card(root4)
    finally:
        set_policy(before)
    return v3, v4


def test_v4_card_is_v3_plus_the_iwm_selection_and_its_tke_coupling_only(cards):
    v3, v4 = cards
    a, b = _flat(v3.recipe.model_config), _flat(v4.recipe.model_config)
    moved = {k for k in set(a) | set(b) if _differs(a.get(k), b.get(k))}
    vm = "physics.vertical_mixing."
    assert moved == {vm + "iwm.enabled", vm + "iwm.form", vm + "iwm.tsdiff",
                     vm + "iwm.require_forcing_maps", vm + "tke.tke_background",
                     vm + "tke.nemo_derived_mxl_min", vm + "tke.mxl_min",
                     vm + "tke.kappaM_min", vm + "tke.kappaH_min"}
    i, t = v4.recipe.model_config.physics.vertical_mixing.iwm, \
        v4.recipe.model_config.physics.vertical_mixing.tke
    assert (i.enabled, i.form, i.mevar, i.tsdiff, i.nu_molecular, i.k_min, i.k_max) == (
        True, "nemo_literal", False, True, 1.4e-6, 1.4e-7, 1.0e-2)
    assert (t.tke_background, t.mxl_min, t.kappaM_min, t.kappaH_min) == (
        1.0e-10, 1.0e-3, 1.4e-6, 1.0e-10)
    assert t.nemo_mxl0_rmxl_min_overwrite and not t.nemo_derived_mxl_min
    s3, s4 = v3.recipe.initial_state, v4.recipe.initial_state
    assert [n for n in s3._fields if _differs(
        getattr(getattr(s3, n), "data", getattr(s3, n)),
        getattr(getattr(s4, n), "data", getattr(s4, n)))] == ["tke"]
    en3, en4 = np.asarray(s3.tke.data), np.asarray(s4.tke.data)
    assert np.array_equal(en4, en3 * 1.0e-4)              # rn_emin 1e-6 -> 1e-10
    assert v3.recipe.iwm_forcing is None
    f = v4.recipe.iwm_forcing
    assert float(np.min(f.hbot)) > 0.0 and float(np.max(f.hcri_inv)) <= 0.01 + 1e-15
    assert (v4.dt_s, v4.n_steps, v4.unmeasured_features) == (
        v3.dt_s, v3.n_steps, v3.unmeasured_features)


def test_v4_selector_fires_on_each_planted_iwm_field(cards):
    v3, v4 = cards
    assert recipe._orca2g1_selector_failures(v4) == []
    assert recipe._orca2g1_selector_failures(v3) == []
    for plant, rows in (({"form": "generic"}, ["ln_zdfiwm"]),
                        ({"tsdiff": False}, ["ln_zdfiwm"]),
                        ({"mevar": True}, ["ln_zdfiwm"]),
                        ({"k_max": 1.0e-3}, ["ln_zdfiwm"]),
                        ({"enabled": False}, ["ln_zdfiwm", "ln_zdftke"])):
        cfg = v4.recipe.model_config
        vm = cfg.physics.vertical_mixing
        bad = v4._replace(recipe=v4.recipe._replace(model_config=cfg._replace(
            physics=cfg.physics._replace(vertical_mixing=vm._replace(
                iwm=vm.iwm._replace(**plant))))))
        assert recipe._orca2g1_selector_failures(bad) == rows, plant
    for plant, rows in (({"tke_background": 1.0e-6}, ["ln_zdftke"]),
                        ({"mxl_min": 1.0e-8}, ["ln_zdfiwm"]),
                        ({"kappaH_min": 1.2e-5}, ["ln_zdfiwm"])):
        cfg = v4.recipe.model_config
        vm = cfg.physics.vertical_mixing
        bad = v4._replace(recipe=v4.recipe._replace(model_config=cfg._replace(
            physics=cfg.physics._replace(vertical_mixing=vm._replace(
                tke=vm.tke._replace(**plant))))))
        assert recipe._orca2g1_selector_failures(bad) == rows, plant
    no_maps = v4._replace(recipe=v4.recipe._replace(iwm_forcing=None))
    assert recipe._orca2g1_selector_failures(no_maps) == ["ln_zdfiwm"]


def test_run_script_reuses_v1_binaries_unbuilt_and_is_fail_closed():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    assert "makenemo" not in text
    assert "ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/bin/nemo.exe" in text
    assert "ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe" in text
    for name in ("zdfiwm.f90", "zdfiwm.o", "--only-section=$sec", "IWM_FILE_SHA",
                 "zdf_iwm_init : internal wave-driven mixing",
                 "force   rn_emin = 1.e-10 and rmxl_min = 1.e-3",
                 'check_iwm_frame "$TWIN_A"', '--check-ddm-frame "$TWIN_A"'):
        assert name in text, name


V4_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_vmix_rounds/round14/oracle_v4_r14")
V4_TWINS = [V4_RECORD / t for t in ("twin_a", "twin_b", "twin_c")]


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
@pytest.mark.skipif(not all(t.is_dir() for t in (*V3_TWINS, *V4_TWINS)),
                    reason="the operator's V3/V4 records are not on this machine")
def test_v4_kt1_2_every_field_and_kt1_avt_avs_avm_bit_for_bit_and_iwm_acts():
    """kt 1 needs zdf_iwm's true divisions by hbot_iwm under jit."""
    assert kt2_restarts_differ(V3_RECORD, V4_RECORD)   # IWM moves kt-2 T
    iwm = assert_record_bit_for_bit(V4_TWINS, "ORCA2V4-iwm", kt_to=2)["iwm_active"]
    for f in ("avt", "avs", "avm"):
        assert iwm[f]["nemo_v4_ne_v3"] > 0 and iwm[f]["same_cells"], (f, iwm[f])
