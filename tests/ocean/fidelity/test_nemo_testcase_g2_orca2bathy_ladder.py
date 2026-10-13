"""Grid ladder rung G2, round 2: the level-aware ladder, its offline replay of
NEMO's vertical momentum solve, and the landed dyn_zdf divisors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

CASES = Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/testcases"
RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
              "round10/oracle_g2_r10/p3_a_100step")
needs_record = pytest.mark.skipif(
    not (RECORD / "oracle_g1step_kt00000001.bin").is_file(),
    reason="the operator's G2 record is not on this machine")


@pytest.fixture(scope="module")
def ladder():
    sys.path.insert(0, str(CASES))
    spec = importlib.util.spec_from_file_location(
        "g2_ladder", CASES / "nemo_testcase_g2_orca2bathy_ladder.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _classes():
    full = np.zeros((2, 3, 4), bool)
    full[:, :, :2] = True
    partial = np.zeros_like(full)
    partial[0, 0, 2] = partial[1, 2, 3] = True
    return {"full": full, "partial": partial}


def test_level_rows_split_wet_cells_by_depth_class(ladder):
    cls = _classes()
    ref = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    cand = ref.copy()
    cand[0, 0, 2] += 3.0            # a partial cell on level 3
    cand[1, 1, 0] += 1.0            # a full cell on level 1
    cand[1, 1, 3] += 9.0            # a dry cell on level 4
    rows = ladder.level_rows(ref, cand, cls)
    assert [r["level"] for r in rows] == [1, 2, 3, 4]
    assert rows[2]["partial"] == {"n_cells": 1, "n_unequal": 1, "max_abs": 3.0,
                                  "relative": 3.0 / 23.0, "ref_max_abs": 2.0}
    assert rows[0]["full"]["n_unequal"] == 1 and rows[0]["full"]["max_abs"] == 1.0
    assert rows[3]["dry_n_unequal"] == 1 and rows[3]["partial"]["n_unequal"] == 0
    summary = ladder.class_summary(rows)
    assert summary["status"] == "DEBT" and summary["dry_n_unequal"] == 1
    assert summary["partial"]["worst_level"] == 3 and summary["full"]["n_cells"] == 12
    assert ladder.class_summary(ladder.level_rows(ref, ref, cls))["status"] == "AT-BAR"
    # the plant moves one partial cell and must turn the summary red
    planted = ladder.class_summary(ladder.level_rows(ref, ref, cls, plant=True))
    assert planted["status"] == "DEBT" and planted["partial"]["n_unequal"] == 1


def test_levels_last_refuses_a_value_below_the_cards_levels(ladder):
    a = np.zeros((4, 2, 3))
    a[:3] = 1.0
    assert ladder.levels_last(a, 3).shape == (2, 3, 3)
    a[3, 0, 0] = 5.0
    with pytest.raises(Exception, match="below the card's last level"):
        ladder.levels_last(a, 3)


def _column(nk=5, bottom=4, partial=0.3):
    """One wet U face on a 1 x 1 domain: ``bottom`` wet levels, the last a
    partial cell."""
    e3 = np.array([10.0, 12.0, 15.0, 20.0, 30.0])[None, None, :nk].copy()
    e3[..., bottom - 1] *= partial
    mask = np.zeros((1, 1, nk))
    mask[..., :bottom] = 1.0
    e3w = np.array([5.0, 11.0, 13.5, 17.5, 25.0])
    h0 = (e3 * mask).sum(-1)
    return dict(e3_0=e3, mask=mask, avm_pair=np.full((1, 1, nk), 2.4e-4) * mask,
                r3_aa=np.full((1, 1), 2.0e-5), r3_mm=np.full((1, 1), 1.0e-5),
                r1_h0=1.0 / h0, e3w_1d=e3w, dt=10800.0)


def test_replay_is_nemos_solve_and_restores_the_barotropic_velocity(ladder):
    g = _column()
    explicit = np.array([1.0, 0.8, 0.5, 0.1, 0.0])[None, None, :] * 1e-3
    ub = np.array([[4.0e-4]])
    solved, after = ladder.nemo_dynzdf_replay(explicit, ub, **g)
    # the same matrix, dense: (I - dt d/dz A d/dz)(u - ub) = explicit - ub
    nk, wet = 5, 4
    e3a = g["e3_0"][0, 0] * (1.0 + 2.0e-5)
    e3w = g["e3w_1d"] * (1.0 + 1.0e-5)
    m = np.eye(nk)
    for k in range(wet):
        for nb in (k - 1, k + 1):
            if 0 <= nb < wet:
                c = 10800.0 * 1.2e-4 / (e3a[k] * e3w[max(k, nb)])
                m[k, k] += c
                m[k, nb] -= c
    want = np.linalg.solve(m, (explicit[0, 0] - ub[0, 0]) * g["mask"][0, 0])
    np.testing.assert_allclose(solved[0, 0], want, rtol=1e-12, atol=1e-18)
    assert not np.any(solved[0, 0, wet:]) and not np.any(after[0, 0, wet:])
    # stprk3_stg.f90:413-417: the rest-thickness mean of the result is uu_b
    mean = (g["e3_0"] * after).sum(-1) * g["r1_h0"]
    np.testing.assert_allclose(mean, ub, rtol=1e-13)
    # zero viscosity: the solve changes nothing
    g0 = dict(g, avm_pair=np.zeros_like(g["avm_pair"]))
    solved0, _ = ladder.nemo_dynzdf_replay(explicit, ub, **g0)
    np.testing.assert_array_equal(solved0, (explicit - ub[..., None]) * g["mask"])
    # the interface stretch is a real operand: the variants differ
    other, _ = ladder.nemo_dynzdf_replay(explicit, ub, **g, interface="none")
    assert np.any(other != solved)


def test_the_option_is_off_by_default_and_selected_by_g2_only():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert LatLonCGridOceanConfig._field_defaults[
        "zdf_momentum_nemo_face_thickness"] is False
    assert recipe._ORCA2G_ZDF_FACES == {"ORCA2G1-flat": False, "ORCA2G2-zps": True}
    assert recipe.ORCA2G2_UNMEASURED == ("multi_level_partial_cell_step",)


@needs_record
def test_geometry_is_bit_identical_on_every_level_and_the_plant_fires(ladder):
    out = ladder.geometry(RECORD)
    assert (out["n_rows"], out["n_bit_identical"], out["card_levels"]) == (36, 36, 30)
    assert out["nemo_nonzero_below_card_levels"]["fmask"] == 7      # the straits
    assert all(len(r["shape"]) == 3 for r in out["rows"] if "e3" in r["row"])
    planted = ladder.geometry(RECORD, "operand fmask")
    red = [r["row"] for r in planted["rows"] if not r["bit_identical"]]
    assert red == ["operand fmask"] and not planted["all_bit_identical"]


@needs_record
def test_depth_classes_come_from_nemos_mesh(ladder):
    cls = ladder.depth_classes(RECORD / "mesh_mask.nc", 30)
    t = cls["t"]
    assert not np.any(t["full"] & t["partial"])
    assert int((t["full"] | t["partial"])[..., 0].sum()) == 16433     # wet columns
    assert int(t["partial"][..., 0].sum()) == 0                       # never level 1
    assert int(t["partial"].sum()) == 15554        # columns ending in a partial cell
    assert int(cls["u"]["partial"].sum()) > 0 and int(cls["u"]["full"][..., 29].sum()) == 0


@needs_record
@pytest.mark.slow
def test_step_one_velocity_is_at_the_bar_only_with_nemos_dyn_zdf_divisors(ladder, tmp_path):
    """The round-2 landing and its before arm (non-vacuity): one step from the
    card's initial state against NEMO's level-1 record."""
    after = ladder.run(RECORD, arm="independent", kt_from=1, kt_to=1, allow_dirty=True)
    rows = {r["name"].rsplit(".", 1)[1]: r for r in after["per_kt"][0]["rows"]}
    assert all(r["status"] == "AT-BAR" for r in rows.values())
    assert rows["u"]["max_abs"] < 1e-17 and rows["v"]["max_abs"] < 1e-17
    before = ladder.run(RECORD, arm="independent", kt_from=1, kt_to=1,
                        allow_dirty=True, without_zdf_faces=True)
    rows = {r["name"].rsplit(".", 1)[1]: r for r in before["per_kt"][0]["rows"]}
    assert rows["u"]["status"] == "DEBT" and rows["u"]["max_abs"] > 1e-10
    assert rows["v"]["status"] == "DEBT" and rows["ssh"]["status"] == "AT-BAR"
