"""Vertical-mixing lane V5: certified V4 plus ORCA2's nn_havtb = 1."""

from __future__ import annotations

import subprocess

import netCDF4
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES, CASES, INPUTS, ORCA2_DECK, _differs, _flat, _root, _values,
)

V4 = CASES / "nemo_testcase_v4_orca2iwm"
V5 = CASES / "nemo_testcase_v5_orca2havtb"
RUN = V5 / "run.sh"
IWM_FILE = INPUTS / "zdfiwm_forcing_orca2.nc"


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def test_v5_deck_is_v4_plus_orca2s_nn_havtb_only():
    v4, v5 = _values(V4 / "namelist_cfg"), _values(V5 / "namelist_cfg")
    changed = {k: (v4.get(k), v5.get(k)) for k in set(v4) | set(v5) if v4.get(k) != v5.get(k)}
    assert changed == {"namzdf.nn_havtb": (0, 1)}
    assert (V4 / "namelist_ice_cfg").read_bytes() == (V5 / "namelist_ice_cfg").read_bytes()


@pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
def test_v5_nn_havtb_is_orca2s_own():
    assert _values(ORCA2_DECK)["namzdf.nn_havtb"] == recipe._ORCA2V5_HAVTB_FIELDS[
        "namzdf.nn_havtb"]


def test_reader_refuses_nn_havtb_on_the_wrong_rung():
    with pytest.raises(ValueError, match="nn_havtb"):
        recipe.read_orca2g1_deck(V5, "ORCA2V4-iwm")
    with pytest.raises(ValueError, match="nn_havtb"):
        recipe.read_orca2g1_deck(V4, "ORCA2V5-havtb")
    assert recipe.read_orca2g1_deck(V5, "ORCA2V5-havtb") == recipe.read_orca2g1_deck(
        V4, "ORCA2V4-iwm")


def test_havtb_shape_is_nemos_three_where_statements():
    # zdfphy.f90:215-224, scalar doubles: 1-0.09*(gphit+15), 0.1, 0.1+0.09*(gphit-5)
    lat = np.array([-90.0, -15.0001, -15.0, -10.3, -5.0001, -5.0, 0.0, 4.9999,
                    5.0, 9.7, 14.9999, 15.0, 60.0])
    want = []
    for g in lat.tolist():
        if -15.0 <= g < -5.0:
            want.append(1.0 - 0.09 * (g + 15.0))
        elif -5.0 <= g < 5.0:
            want.append(0.1)
        elif 5.0 <= g < 15.0:
            want.append(0.1 + 0.09 * (g - 5.0))
        else:
            want.append(1.0)
    got = recipe.nemo_havtb_shape(lat)
    assert got.tolist() == want
    assert (got[[0, 1, 11, 12]] == 1.0).all() and got[2] == 1.0 and got[5] == 0.1


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in (*CARD_FILES, IWM_FILE.name)):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("v5")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        out = []
        for rung, build in ((V4, recipe.build_orca2v4_iwm_card),
                            (V5, recipe.build_orca2v5_havtb_card)):
            root = _root(base, rung)
            (root / IWM_FILE.name).symlink_to(IWM_FILE)
            out.append((build(root), root))
    finally:
        set_policy(before)
    return out


def test_v5_card_is_v4_plus_the_shaped_avt_k_seed_only(cards):
    (v4, _), (v5, root5) = cards
    a, b = _flat(v4.recipe.model_config), _flat(v5.recipe.model_config)
    assert {k for k in set(a) | set(b) if _differs(a.get(k), b.get(k))} == set()
    s4, s5 = v4.recipe.initial_state, v5.recipe.initial_state
    assert [n for n in s4._fields if _differs(
        getattr(getattr(s4, n), "data", getattr(s4, n)),
        getattr(getattr(s5, n), "data", getattr(s5, n)))] == ["tke_avt"]
    with netCDF4.Dataset(root5 / "domain_cfg.nc") as ds:
        gphit = np.squeeze(np.asarray(ds.variables["gphit"][:], dtype=np.float64))
    avt4, avt5 = np.asarray(s4.tke_avt.data), np.asarray(s5.tke_avt.data)
    wet = avt4 > 0.0
    assert np.array_equal(avt4, 1.2e-5 * wet)
    assert np.array_equal(avt5, (recipe.nemo_havtb_shape(gphit)[..., None] * 1.2e-5) * wet)
    band = ((-15.0 <= gphit) & (gphit < 15.0))[..., None]
    moved = avt5 != avt4
    assert moved.any() and not (moved & ~band).any()
    assert np.array_equal(moved, band & wet)
    assert (v5.dt_s, v5.n_steps, v5.unmeasured_features) == (
        v4.dt_s, v4.n_steps, v4.unmeasured_features)
    assert recipe._orca2g1_selector_failures(v5) == []


def test_validator_refuses_v5_with_v4s_seed_and_v4_with_v5s(cards):
    (v4, root4), (v5, root5) = cards
    swap5 = v5._replace(recipe=v5.recipe._replace(initial_state=v4.recipe.initial_state))
    with pytest.raises(ValueError, match="nn_havtb"):
        recipe.validate_orca2g1_flat_card(swap5, root5)
    swap4 = v4._replace(recipe=v4.recipe._replace(initial_state=v5.recipe.initial_state))
    with pytest.raises(ValueError, match="nn_havtb"):
        recipe.validate_orca2g1_flat_card(swap4, root4)


def test_run_script_reuses_v1_binaries_unbuilt_and_is_fail_closed():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    assert "makenemo" not in text
    assert "ORCA2V1_OMIP_L4_R1FULLENTRY/BLD/bin/nemo.exe" in text
    assert "ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe" in text
    for name in ("zdfphy.o zdfiwm.o", "--only-section=$sec", "V5_DECK_SHA", "V4_FRAME_SHA",
                 "nn_havtb += +1$", 'check_havtb_frame "$TWIN_A"',
                 '"$V4_TOOL" --check-iwm-frame "$TWIN_A"', '--check-ddm-frame "$TWIN_A"'):
        assert name in text, name
