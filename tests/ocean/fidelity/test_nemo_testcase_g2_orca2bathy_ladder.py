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
    # a dry cell alone is DEBT too: NEMO holds zero there and so must the card
    dry = ref.copy()
    dry[1, 1, 3] += 9.0
    only_dry = ladder.class_summary(ladder.level_rows(ref, dry, cls))
    assert only_dry["status"] == "DEBT" and only_dry["full"]["n_unequal"] == 0
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
    assert LatLonCGridOceanConfig._field_defaults[
        "zdf_tracer_nemo_rounded_thickness"] is False       # round 3
    assert recipe._ORCA2G_ZDF_FACES == {"ORCA2G1-flat": False, "ORCA2G1b-vec": False, "ORCA2G2-zps": True, "ORCA2G3-ts": True,
                                          "ORCA2G4-sbc": True}
    assert recipe.ORCA2G2_UNMEASURED == ("multi_level_partial_cell_step",)
    # no other builder assigns it: one assignment in the whole package, the
    # shared G1/G2 builder's, keyed by the rung table (the resolved pair of
    # cards is asserted in test_nemo_testcase_g2_orca2bathy.py)
    hits = [(path.name, n) for path in (CASES.parents[3] / "packages").rglob("*.py")
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if "zdf_momentum_nemo_face_thickness=" in line]
    assert [name for name, _ in hits] == ["nemo_testcase_recipe.py"]
    # round 3's option: its own rung table, one assignment in the package
    assert recipe._ORCA2G_TRA_ZDF_ROUNDED == {"ORCA2G1-flat": False, "ORCA2G1b-vec": False, "ORCA2G2-zps": True, "ORCA2G3-ts": True,
                                          "ORCA2G4-sbc": True}
    mine = [(path.name, line) for path in (CASES.parents[3] / "packages").rglob("*.py")
            for line in path.read_text().splitlines()
            if "zdf_tracer_nemo_rounded_thickness=" in line]
    assert [name for name, _ in mine] == ["nemo_testcase_recipe.py"]
    assert "_ORCA2G_TRA_ZDF_ROUNDED[case]" in mine[0][1]
    # round 4's option: its own rung table, one assignment in the package
    assert LatLonCGridOceanConfig._field_defaults[
        "barotropic_rhs_ascending_level_sum"] is False
    assert recipe._ORCA2G_UE_RHS_ASCENDING == {"ORCA2G1-flat": False, "ORCA2G1b-vec": False, "ORCA2G2-zps": True, "ORCA2G3-ts": True,
                                          "ORCA2G4-sbc": True}
    sums = [(path.name, line) for path in (CASES.parents[3] / "packages").rglob("*.py")
            for line in path.read_text().splitlines()
            if "barotropic_rhs_ascending_level_sum=" in line]
    assert [name for name, _ in sums] == ["nemo_testcase_recipe.py"]
    assert "_ORCA2G_UE_RHS_ASCENDING[case]" in sums[0][1]
    # each appended LAST in its round: no older field changed position
    # G3 round 3's option: its own rung table (G3 only), one assignment
    assert LatLonCGridOceanConfig._field_defaults[
        "zdf_momentum_stage_r3u_nemo_literal"] is False
    assert recipe._ORCA2G_ZDF_STAGE_R3U == {"ORCA2G1-flat": False, "ORCA2G1b-vec": False, "ORCA2G2-zps": False, "ORCA2G3-ts": True,
                                              "ORCA2G4-sbc": True}
    r3u = [(path.name, line) for path in (CASES.parents[3] / "packages").rglob("*.py")
           for line in path.read_text().splitlines()
           if "zdf_momentum_stage_r3u_nemo_literal=" in line]
    assert [name for name, _ in r3u] == ["nemo_testcase_recipe.py"]
    assert "_ORCA2G_ZDF_STAGE_R3U[case]" in r3u[0][1]
    assert LatLonCGridOceanConfig._fields[-10:-2] == (
        "zdf_tracer_nemo_rounded_thickness", "barotropic_rhs_ascending_level_sum",
        "zdf_momentum_stage_r3u_nemo_literal", "barotropic_wind_nemo_stored_depth",
        "surface_stress_nemo_fold_vector_sign", "zdf_surface_stress_nemo_face_thickness",
        "een_nemo_relative_vorticity", "rk3_stage_tracer_written_order")
    assert LatLonCGridOceanConfig._fields[-2:] == (
        "rk3_vector_rhs_nemo_order", "rk3_stage3_heat_nemo_written")
    assert "_ORCA2G_ZDF_FACES[case]" in (
        CASES.parents[3] / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py"
    ).read_text().splitlines()[hits[0][1] - 1]


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
def test_two_steps_are_near_the_bar_only_with_nemos_dyn_zdf_divisors(ladder):
    """The round-2 landing and its before arm (non-vacuity).  Step 1 against
    NEMO's level-1 record; step 2 against NEMO's restart on ALL levels, split
    by depth class, so the partial-cell faces at depth are scored too."""
    twin_b = RECORD.parent / "p3_b_10step"
    after = ladder.run(RECORD, arm="independent", kt_from=1, kt_to=2,
                       restart_roots=(twin_b,), allow_dirty=True)
    rows = {r["name"].rsplit(".", 1)[1]: r for r in after["per_kt"][0]["rows"]}
    assert all(r["status"] == "AT-BAR" for r in rows.values())
    assert rows["u"]["max_abs"] < 1e-17 and rows["v"]["max_abs"] < 1e-17
    deep = after["per_kt"][1]["all_levels"]
    assert deep["u"]["summary"]["partial"]["n_cells"] > 10000
    for f in ("u", "v"):
        assert deep[f]["summary"]["partial"]["max_abs"] < 1e-12
        assert deep[f]["summary"]["full"]["max_abs"] < 1e-12
        assert deep[f]["summary"]["dry_n_unequal"] == 0
    before = ladder.run(RECORD, arm="independent", kt_from=1, kt_to=2,
                        restart_roots=(twin_b,), allow_dirty=True,
                        without_zdf_faces=True)
    rows = {r["name"].rsplit(".", 1)[1]: r for r in before["per_kt"][0]["rows"]}
    assert rows["u"]["status"] == "DEBT" and rows["u"]["max_abs"] > 1e-10
    assert rows["v"]["status"] == "DEBT" and rows["ssh"]["status"] == "AT-BAR"
    deep = before["per_kt"][1]["all_levels"]
    for f in ("u", "v"):        # the partial-cell faces carry the largest error
        assert deep[f]["summary"]["partial"]["max_abs"] > 1e-7
        assert (deep[f]["summary"]["partial"]["max_abs"]
                > 10 * deep[f]["summary"]["full"]["max_abs"])


def test_tracer_replay_is_nemos_solve(ladder):
    """The offline replay against a dense solve of the same matrix, on a
    column with a partial bottom cell and dry levels below it."""
    g = _column()
    nk, tmask, e3 = 5, g["mask"], g["e3_0"]
    before = np.array([20.0, 19.5, 18.0, 15.0, 7.0])[None, None, :] * tmask
    kw = dict(tmask=tmask, e3t_0=e3, e3w_1d=np.append(g["e3w_1d"], 30.0), avt0=1.2e-5,
              r3t_bb=np.full((1, 1), 1e-5), r3t_mm=np.full((1, 1), 2e-5),
              r3t_aa=np.full((1, 1), 3e-5), dt=10800.0)
    got = ladder.nemo_trazdf_replay(before, **kw)[0, 0]
    wet = int(tmask.sum())
    a = np.zeros((wet, wet))
    for k in range(wet):
        up = 10800.0 * 1.2e-5 / (kw["e3w_1d"][k] * (1 + 2e-5)) if k else 0.0
        dn = 10800.0 * 1.2e-5 / (kw["e3w_1d"][k + 1] * (1 + 2e-5)) if k < wet - 1 else 0.0
        a[k, k] = e3[0, 0, k] * (1 + 3e-5) + up + dn
        if k:
            a[k, k - 1] = -up
        if k < wet - 1:
            a[k, k + 1] = -dn
    want = np.linalg.solve(a, e3[0, 0, :wet] * (1 + 1e-5) * before[0, 0, :wet])
    np.testing.assert_allclose(got[:wet], want, rtol=1e-13)
    assert np.all(got[wet:] == 0.0)
    # content is conserved by the solve (zero flux at both ends)
    np.testing.assert_allclose((got * e3[0, 0] * (1 + 3e-5)).sum(),
                               (before[0, 0] * e3[0, 0] * (1 + 1e-5)).sum(), rtol=1e-13)
    # a uniform column stays uniform up to rounding: no diffusion to do
    flat = ladder.nemo_trazdf_replay(20.0 * tmask, **dict(kw, r3t_bb=kw["r3t_aa"]))
    np.testing.assert_allclose(flat[0, 0, :wet], 20.0, rtol=1e-14)


@needs_record
@pytest.mark.slow
def test_tracers_are_nemos_bit_for_bit_only_with_the_rounded_diagonal(ladder):
    """The round-3 landing and its before arm.  The replay must be NEMO's on
    level 1 (0 unequal) before the card is compared with it on all levels."""
    after = ladder.run(RECORD, arm="trazdf_replay", allow_dirty=True)
    before = ladder.run(RECORD, arm="trazdf_replay", allow_dirty=True,
                        without_tracer_rounding=True)
    for f in ("T", "S"):
        a, b = (x["per_kt"][0]["trazdf_replay"][f] for x in (after, before))
        assert a["level1_replay_vs_nemo"]["n_unequal"] == 0
        assert b["level1_replay_vs_nemo"]["n_unequal"] == 0
        assert a["level1_card_vs_nemo"]["n_unequal"] == 0
        assert sum(a["unequal_by_level"]) == 0
        assert a["all_levels_card_vs_replay"]["dry_n_unequal"] == 0
        assert b["level1_card_vs_nemo"]["n_unequal"] > 500
        assert sum(b["unequal_by_level"]) > 10000 and min(b["unequal_by_level"]) > 100


@needs_record
@pytest.mark.slow
def test_step_two_is_at_the_bar_on_all_levels_only_with_the_rounded_diagonal(ladder):
    twin_b = RECORD.parent / "p3_b_10step"
    for flag, at_bar in ((False, True), (True, False)):
        out = ladder.run(RECORD, arm="independent", kt_from=1, kt_to=2,
                         restart_roots=(twin_b,), allow_dirty=True,
                         without_tracer_rounding=flag)
        assert out["zdf_tracer_nemo_rounded_thickness"] is (not flag)
        deep = out["per_kt"][1]["all_levels"]
        assert all(deep[f]["summary"]["status"] == "AT-BAR"
                   for f in ("u", "v", "T", "S")) is at_bar
        rows = {r["name"].rsplit(".", 1)[1]: r for r in out["per_kt"][1]["rows"]}
        assert (rows["ssh"]["status"] == "AT-BAR") is at_bar
        if at_bar:
            assert deep["T"]["summary"]["full"]["n_unequal"] == 0
            assert deep["S"]["summary"]["partial"]["n_unequal"] == 0
            assert deep["u"]["summary"]["full"]["max_abs"] < 1e-15
        else:
            assert deep["u"]["summary"]["full"]["max_abs"] > 1e-14


@needs_record
@pytest.mark.slow
def test_the_traced_solver_loop_is_nemos_at_step_one(ladder):
    out = ladder.run(RECORD, arm="substeps", allow_dirty=True)
    assert out["n_substeps"] == 65 and not out["scalars_unequal"]
    unequal = [r for r in out["rows"] if not r["bit_identical"]]
    # NEMO's histories are not set at step 1; and one cell of the fold pivot
    # row holds a number of order 1e-112 at the last substep
    assert {r["operand"] for r in unequal} == {"entry.sshb", "entry.sshbb", "spg.v"}
    tiny = [r for r in unequal if r["operand"] == "spg.v"]
    assert [(r["substep"], r["n_unequal"]) for r in tiny] == [(65, 1)]
    assert tiny[0]["max_abs"] < 1e-100 and tiny[0]["regions"]["fold_pivot_row"]["n_unequal"] == 1
    assert all(v["traced_vs_nemo_unequal"] == 0 for v in out["trace_passive"].values())


def test_ascending_level_sum_is_fortrans_order():
    """stp2d.f90:178-179's SUM accumulates in ascending k; the helper must
    give that order bit for bit.  (A lone ``jnp.sum`` compiles ascending too;
    XLA vectorises it only inside the full step, so the option's effect is
    pinned by the record test's before arm below, not here.)"""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.barotropic_common import ascending_level_sum
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 30)) * np.exp(rng.uniform(-20, 0, (400, 30)))
    want = np.zeros(400)
    for k in range(30):
        want = want + x[:, k]
    got = np.asarray(jax.jit(ascending_level_sum)(jnp.asarray(x)))
    assert np.array_equal(got, want)
    # the order is real: a 4-lane split of the same values differs somewhere
    lanes = [x[:, k::4].sum(axis=1) for k in range(4)]
    assert np.count_nonzero(((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) != want) > 0


@needs_record
@pytest.mark.slow
def test_step_one_is_nemos_bit_for_bit_only_with_the_ascending_rhs_sum(ladder):
    """The round-4 landing and its before arm: every scored field of step 1."""
    for flag in (False, True):
        out = ladder.run(RECORD, arm="independent", kt_from=1, kt_to=1,
                         allow_dirty=True, without_rhs_ascending_sum=flag)
        assert out["barotropic_rhs_ascending_level_sum"] is (not flag)
        rows = {r["name"].rsplit(".", 1)[1]: r for r in out["per_kt"][0]["rows"]}
        assert set(rows) == {"ssh", "uu_b", "vv_b", "u", "v", "T", "S"}
        if flag:
            assert rows["ssh"]["n_unequal"] > 500 and rows["uu_b"]["n_unequal"] > 500
        else:
            assert all(r["n_unequal"] == 0 for r in rows.values())


def test_ue_rhs_replay_is_an_ascending_masked_mean(ladder):
    rng = np.random.default_rng(1)
    rhs = rng.standard_normal((3, 4, 5)) * np.exp(rng.uniform(-30, 0, (3, 4, 5)))
    e3 = rng.uniform(10, 500, (3, 4, 5))
    mask = np.ones((3, 4, 5))
    mask[0, 0] = 0.0                      # a dry column
    mask[1, 1, 3:] = 0.0                  # a shallow one
    got = ladder.nemo_ue_rhs_replay(rhs, e3, mask)
    assert got[0, 0] == 0.0
    acc, h0 = 0.0, 0.0
    for k in range(5):
        acc, h0 = acc + e3[1, 1, k] * rhs[1, 1, k] * mask[1, 1, k], h0 + e3[1, 1, k] * mask[1, 1, k]
    assert got[1, 1] == acc * (1.0 / (h0 + 1.0 - 1.0))
    np.testing.assert_allclose(ladder.nemo_ue_rhs_replay(rhs, e3, mask, lanes=4), got, rtol=1e-12)


@needs_record
@pytest.mark.slow
def test_the_rhs_frame_is_nemos_on_wet_faces_and_its_depth_mean_is_ascending(ladder):
    """R2-P3 / G-P6: the card's 3-D RHS after stp_2D equals NEMO's frame on
    every wet face; NEMO's frame alone carries the unmasked sco pressure
    gradient on dry faces (dynhpg.f90:393-394).  The depth mean is the
    ascending replay's, and a 4-lane order would not be."""
    out = ladder.run(RECORD, arm="rhs_frame", allow_dirty=True)
    for f in ("u", "v"):
        r = out["per_kt"][0]["rhs_frame"][f]
        assert r["level1_wet_faces"] > 15000 and r["level1_wet_unequal"] == 0
        assert r["level1_dry_card_nonzero"] == 0 and r["level1_dry_unequal"] > 1000
        assert r["ascending_replay_vs_card_unequal"] == 0
        assert r["depth_mean_faces"] == r["level1_wet_faces"]   # fold row included
        assert r["lane4_replay_vs_card_unequal"] > 50


@needs_record
@pytest.mark.slow
def test_the_100_step_record_is_bit_identical_on_every_field(ladder):
    """G2 certification (round 4): INDEPENDENT, 100 steps, level 1 and the
    2-D fields every step, all levels by depth class where NEMO wrote a
    restart, and the six barotropic histories every tenth step."""
    twin_b = RECORD.parent / "p3_b_10step"
    out = ladder.run(RECORD, arm="independent", kt_from=1, kt_to=100,
                     restart_roots=(twin_b,), allow_dirty=True)
    assert len(out["per_kt"]) == 100
    assert all(r["n_unequal"] == 0 for p in out["per_kt"] for r in p["rows"])
    deep = [p["all_levels"] for p in out["per_kt"] if p["all_levels"]]
    assert [p["kt"] for p in out["per_kt"] if p["all_levels"]] == [
        2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    assert all(v["summary"][c]["n_unequal"] == 0 and v["summary"]["dry_n_unequal"] == 0
               for d in deep for v in d.values() for c in ("full", "partial"))
    assert len(out["history_rows"]) == 60
    assert all(r["n_unequal"] == 0 for r in out["history_rows"])
