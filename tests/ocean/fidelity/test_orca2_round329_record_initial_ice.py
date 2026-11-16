"""ORCA2 round 329 (DECISION 157): the rung-10 comparison card's initial ice.

The comparison card lowers SSH by the record's own SI3 initial ice, read from
the record's ocean.output and fed to the card's one initial-ice/SSH function;
the shipped card keeps ORCA2's five-category ponds-on ice.  A record whose
settings cannot be read refuses.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    nemo_testcase_l4_orca2_round329_recordice_d122_gate as gate,
)

RECORD = ladder.RUNG10_RECORD_RUN
needs_record = pytest.mark.skipif(not RECORD.is_dir(),
                                  reason="rung-10 record not on this machine")


@pytest.fixture(scope="module")
def orca2():
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return recipe.build_orca2_zps_card(RECORD)


def _record_copy(tmp_path, edit):
    text = (RECORD / "ocean.output").read_text(encoding="utf-8")
    (tmp_path / "ocean.output").write_text(edit(text), encoding="utf-8")
    ladder.record_initial_ice.cache_clear()
    return tmp_path


@needs_record
def test_record_initial_ice_is_read_from_its_ocean_output():
    ladder.record_initial_ice.cache_clear()
    ice, jpl = ladder.record_initial_ice(RECORD)
    assert jpl == 1
    assert ice == recipe.ORCA2InitialIce(
        thres_sst=2.0, ice_concentration=(0.997, 0.8),
        ice_thickness_m=(2.5, 1.0), snow_thickness_m=(0.2, 0.1),
        pond_concentration=(0.0, 0.0), pond_depth_m=(0.0, 0.0))


@needs_record
@pytest.mark.parametrize("plant,match", [
    ("missing", "unreadable"),
    ("drop_ati", "rn_ati_ini 0 times"),
    ("duplicate_hti", "rn_hti_ini 2 times"),
    ("file_ice", "analytic initial ice"),
])
def test_unreadable_record_settings_refuse(tmp_path, monkeypatch, plant, match):
    if plant == "missing":
        root = tmp_path
    elif plant == "drop_ati":
        root = _record_copy(tmp_path, lambda t: t.replace("rn_ati_ini", "rn_ati_xxx"))
    elif plant == "duplicate_hti":
        root = _record_copy(tmp_path, lambda t: t + "\n rn_hti_ini = 1.0 1.0\n")
    else:
        root = _record_copy(tmp_path, lambda t: t.replace(
            "nn_iceini_file =            0", "nn_iceini_file =            1"))
    ladder.record_initial_ice.cache_clear()
    with pytest.raises(ladder.GateError, match=match):
        ladder.record_initial_ice(root)
    ladder.record_initial_ice.cache_clear()


@needs_record
def test_ponds_on_record_feeds_pond_area_times_ice_concentration(tmp_path):
    root = _record_copy(tmp_path, lambda t: t.replace(
        "ln_pnd         =  F", "ln_pnd         =  T"))
    ice, _ = ladder.record_initial_ice(root)
    ladder.record_initial_ice.cache_clear()
    assert ice.pond_concentration == (0.0 * 0.997, 0.2 * 0.8)
    assert ice.pond_depth_m == (0.0, 0.05)


@needs_record
def test_shipped_card_keeps_deck_ice_and_comparison_matches_nemo_entry(orca2):
    state = orca2.recipe.initial_state
    z = orca2.recipe.z_coord
    wet = np.asarray(z.is_active)[..., 0]
    shipped = np.asarray(state.eta.data)
    # The shipped card's five-category ponds-on adjustment (round 327).
    assert np.unique(shipped[wet]).tolist() == [-0.09343505872684969]
    rebuilt = recipe.orca2_initial_ice_ssh(
        np.asarray(state.T.data), np.asarray(state.S.data), np.asarray(z.is_active),
        np.asarray(orca2.recipe.grid.f_T), np.asarray(z.nemo_e1e2t),
        recipe.ORCA2_DECK_INITIAL_ICE)
    assert np.array_equal(rebuilt.view(np.uint64), shipped.view(np.uint64))

    ladder.record_initial_ice.cache_clear()
    cmp_card = ladder.comparison_card(orca2)
    got = np.asarray(cmp_card.recipe.initial_state.eta.data)
    nemo = ladder.assemble_state_fields(RECORD, 1, stage=None)["ssh"]
    assert np.array_equal(got.view(np.uint64), nemo.view(np.uint64))
    assert not np.array_equal(shipped, nemo)
    # Only SSH moves; the control arm is round 326's comparison card.
    for name in ("T", "S", "u", "v"):
        assert getattr(cmp_card.recipe.initial_state, name) is getattr(state, name)
    control = gate.selected_card(orca2, arm="control", rung="rung10")
    assert control.recipe.initial_state is state
    assert control.recipe.model_config == cmp_card.recipe.model_config
    for rung in gate.RUNGS[:-1]:
        assert gate.selected_card(orca2, arm="recordice", rung=rung) is orca2


def test_cards_without_si3_initial_ice_keep_their_ssh():
    card = recipe.build_nemo_testcase_card("GYRE-zco")
    got = ladder.comparison_card(card)
    assert got.recipe.initial_state is card.recipe.initial_state
