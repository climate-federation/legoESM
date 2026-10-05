"""ORCA2's initial-state hand alterations — the compiled branch, cell by cell.

NEMO's initial-condition reader applies configuration-specific alterations to
the time-interpolated temperature and salinity BEFORE the land mask when the
resolved configuration is ORCA at index 2 with data damping on:
``dtatsd.f90:218-253`` in the round-5 ORCA2 build.  Omitting them left 1,283
temperature and 720 salinity cells unequal against NEMO's own step-1 state.

The index arithmetic is the place a silent wrong-number defect enters, so
these tests pin the exact boxes, the exact levels and the exact values rather
than a count.  Every assertion is written so that a one-cell or one-level
shift makes it fail.
"""
from __future__ import annotations

import numpy as np
import pytest


def _recipe():
    from legoesm.ocean.fidelity import nemo_testcase_recipe
    return nemo_testcase_recipe


# Inner one-based boxes, independently restated here from the compiled source
# so a change to the module's table cannot silently agree with itself.
ALBORAN_J = slice(100, 109)     # inner j 101..109
ALBORAN_I = slice(139, 154)     # inner i 140..154
RED_SEA_J = slice(86, 96)       # inner j  87..96
RED_SEA_I = slice(146, 159)     # inner i 147..159


def _blank():
    return (np.zeros((148, 180, 30), dtype=np.float64),
            np.zeros((148, 180, 30), dtype=np.float64))


def test_alboran_offsets_land_on_exactly_the_compiled_levels():
    temperature, salinity = _blank()
    _recipe().apply_orca2_hand_alterations(temperature, salinity)

    box_t = temperature[ALBORAN_J, ALBORAN_I]
    assert np.all(box_t[..., 12] == -0.20)
    assert np.all(box_t[..., 13:15] == -0.35)
    assert np.all(box_t[..., 15:25] == -0.40)
    assert np.all(box_t[..., :12] == 0.0)
    assert np.all(box_t[..., 25:] == 0.0)

    box_s = salinity[ALBORAN_J, ALBORAN_I]
    assert np.all(box_s[..., 12] == -0.15)
    assert np.all(box_s[..., 13:15] == -0.25)
    assert np.all(box_s[..., 15:17] == -0.30)
    assert np.all(box_s[..., 17:25] == -0.35)
    assert np.all(box_s[..., :12] == 0.0)
    assert np.all(box_s[..., 25:] == 0.0)


def test_red_sea_temperatures_are_assigned_not_incremented():
    temperature, salinity = _blank()
    temperature[...] = 99.0
    _recipe().apply_orca2_hand_alterations(temperature, salinity)

    box = temperature[RED_SEA_J, RED_SEA_I]
    assert np.all(box[..., 3:10] == 7.0)
    assert np.all(box[..., 10:13] == 6.5)
    assert np.all(box[..., 13:20] == 6.0)
    # untouched levels keep the pre-existing value, proving assignment scope
    assert np.all(box[..., :3] == 99.0)
    assert np.all(box[..., 20:] == 99.0)
    # salinity has no Red Sea statement at all
    assert np.all(salinity[RED_SEA_J, RED_SEA_I] == 0.0)


def test_nothing_outside_the_two_boxes_moves():
    temperature, salinity = _blank()
    _recipe().apply_orca2_hand_alterations(temperature, salinity)

    touched = np.zeros((148, 180), dtype=bool)
    touched[ALBORAN_J, ALBORAN_I] = True
    touched[RED_SEA_J, RED_SEA_I] = True
    assert np.all(temperature[~touched] == 0.0)
    assert np.all(salinity[~touched] == 0.0)
    # and the boxes really are disjoint in j, as the source order implies
    assert not (set(range(100, 109)) & set(range(86, 96)))


def test_the_two_boxes_have_the_compiled_extent():
    module = _recipe()
    assert module._ORCA2_ALBORAN_BOX == (101, 109, 140, 154)
    assert module._ORCA2_RED_SEA_BOX == (87, 96, 147, 159)
    temperature, salinity = _blank()
    module.apply_orca2_hand_alterations(temperature, salinity)
    altered = np.any(temperature != 0.0, axis=-1)
    rows, columns = np.nonzero(altered)
    assert rows.min() == 86 and rows.max() == 108
    assert columns.min() == 139 and columns.max() == 158
    # 9x15 Alboran + 10x13 Red Sea, disjoint
    assert int(altered.sum()) == 9 * 15 + 10 * 13


def test_a_shifted_box_is_detectable(monkeypatch):
    """Synthetic-violation check: the assertions above CAN fail."""
    module = _recipe()
    monkeypatch.setattr(module, "_ORCA2_ALBORAN_BOX", (102, 110, 140, 154))
    temperature, salinity = _blank()
    module.apply_orca2_hand_alterations(temperature, salinity)
    with pytest.raises(AssertionError):
        assert np.all(temperature[ALBORAN_J, ALBORAN_I][..., 12] == -0.20)


def test_the_card_builder_takes_the_alterations_by_default():
    import inspect
    source = inspect.getsource(_recipe().build_orca2_initial_ts)
    assert "apply_hand_alterations: bool = True" in source
    assert "apply_orca2_hand_alterations(temperature, salinity)" in source
    # masked AFTER the alterations, as the z/zps branch does
    assert source.index("apply_orca2_hand_alterations") < source.index(
        "np.where(tmask, temperature, 0.0)")
