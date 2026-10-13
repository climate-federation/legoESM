"""The G1 ladder harness and the card's fold-image V row.

Synthetic tests run everywhere.  RECORD tests need the admitted G1 record and
are skipped off this machine; each one has a plant that must turn it red.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

_PATH = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
         "testcases/nemo_testcase_g1_orca2grid_ladder.py")
_spec = importlib.util.spec_from_file_location("g1_orca2grid_ladder", _PATH)
ladder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ladder)

RECORD = ladder.DEFAULT_ROOT
needs_record = pytest.mark.skipif(
    not (RECORD / "oracle_g1step_kt00000010.bin").is_file(),
    reason="the G1 record is not admitted on this machine")


def _mask():
    wet = np.ones((12, 16), dtype=float)
    wet[0] = 0.0                    # south wall
    wet[5:7, 6:9] = 0.0             # an island
    return wet


def test_regions_partition_the_domain_once_and_name_the_fold_rows():
    reg = ladder.regions(_mask())
    total = sum(reg[k].astype(int) for k in ("fold", "seam", "south", "interior"))
    assert np.all(total == 1)
    assert np.array_equal(np.where(reg["fold"].any(1))[0], [9, 10, 11])
    assert np.array_equal(np.where(reg["seam"].any(0))[0], [0, 1, 14, 15])
    assert not reg["south"][:, :2].any() and reg["south"][:3, 2:14].all()
    assert np.array_equal(reg["interior_coast"] | reg["interior_open"], reg["interior"])
    assert not (reg["interior_coast"] & reg["interior_open"]).any()
    # the island, its wet rim and nothing two cells away
    assert reg["interior_coast"][5, 7] and reg["interior_coast"][4, 7]
    assert reg["interior_open"][8, 4] and not reg["interior_coast"][8, 4]


def test_row_scores_by_region_and_its_plant_leaves_the_bar():
    reg = ladder.regions(_mask())
    ref = np.linspace(0.0, 1.0, 12 * 16).reshape(12, 16)
    same = ladder.row("x", ref, ref.copy(), reg)
    assert same["bit_identical"] and same["status"] == "AT-BAR"
    cand = ref.copy()
    cand[10, 5] += 1.0e-9           # one fold cell
    moved = ladder.row("x", ref, cand, reg)
    assert not moved["bit_identical"] and moved["status"] == "DEBT"
    assert moved["regions"]["fold"]["n_unequal"] == 1
    assert all(moved["regions"][k]["n_unequal"] == 0
               for k in ("seam", "south", "interior"))
    assert ladder.row("x", ref, ref.copy(), reg, plant=True)["status"] == "DEBT"


def test_redundant_face_records():
    a = np.arange(12.0).reshape(3, 4)
    assert np.array_equal(ladder.pad_u(a)[:, 0], a[:, -1])
    assert ladder.pad_u(a).shape == (3, 5) and ladder.pad_v(a).shape == (4, 4)
    assert not ladder.pad_v(a)[0].any()


class _Fold:
    is_active = True
    vector_sign_v = -1.0

    def __init__(self, n):
        self.perm_v = (n - np.arange(n)) % n


class _Grid:
    def __init__(self, fold):
        self.fold = fold


def test_fold_image_v_row_is_the_signed_mirror_of_the_row_below():
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _een_fold_image_v_row
    n, rows = 6, 4
    rng = np.random.default_rng(0)
    v = jnp.asarray(rng.normal(size=(rows + 1, n, 1)))
    h_v = jnp.asarray(rng.uniform(1.0, 2.0, size=(rows + 1, n, 1)))
    mask = np.ones((rows + 1, n, 1))
    mask[-1] = 0.0                  # the compact mask keeps the image row dry
    mask[-2, 2] = 0.0               # one dry face below the fold
    mask = jnp.asarray(mask)
    h_u = u = u_mask = object()
    off = _een_fold_image_v_row(False, _Grid(_Fold(n)), h_v, v, h_u, u, u_mask, mask)
    assert off[0] is h_v and off[1] is v and off[5] is mask     # untouched
    h2, v2, _, _, _, m2 = _een_fold_image_v_row(
        True, _Grid(_Fold(n)), h_v, v, h_u, u, u_mask, mask)
    perm = (n - np.arange(n)) % n
    wet = np.asarray(mask)[-2][perm]
    assert np.array_equal(np.asarray(m2)[-1], wet)
    assert np.array_equal(np.asarray(v2)[-1], -np.asarray(v)[-2][perm] * wet)
    assert np.array_equal(np.asarray(h2)[-1], np.asarray(h_v)[-2][perm])
    assert np.array_equal(np.asarray(v2)[:-1], np.asarray(v)[:-1])  # rows below
    assert float(np.asarray(v2)[-1, (n - 2) % n, 0]) == 0.0          # dry image
    with pytest.raises(ValueError, match="active north fold"):
        _een_fold_image_v_row(True, _Grid(None), h_v, v, h_u, u, u_mask, mask)


def test_only_the_g1_card_selects_the_fold_image_row():
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert LatLonCGridOceanConfig._field_defaults["een_fold_image_v_row"] is False
    src = (Path(__file__).resolve().parents[3]
           / "packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py").read_text()
    assert src.count("een_fold_image_v_row=True") == 1
    assert src.index("een_fold_image_v_row=True") > src.index(
        "def build_orca2g1_flat_card")


@needs_record
def test_geometry_gate_is_36_of_36_and_its_plant_names_its_row():
    pytest.importorskip("netCDF4")
    clean = ladder.geometry(RECORD)
    assert clean["n_rows"] == 36 and clean["all_bit_identical"]
    planted = ladder.geometry(RECORD, plant="operand fmask")
    assert [r["row"] for r in planted["rows"] if not r["bit_identical"]] == [
        "operand fmask"]
    with pytest.raises(Exception, match="unknown geometry plant"):
        ladder.geometry(RECORD, plant="not a row")


@needs_record
def test_offline_vor_een_replay_reproduces_nemo_and_needs_the_image_row():
    """NEMO's compiled statement, replayed from NEMO's own fields, is NEMO's
    record bit for bit; zeroing the fold-image V row breaks the pivot row and
    leaves the row below alone."""
    pytest.importorskip("netCDF4")
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=RECORD)
    fe3 = np.asarray(card.recipe.z_coord.nemo_een_barotropic.fe3mask)[..., 0]
    nemo = ladder.read_step(RECORD, 10)["b_uu_rhs_k1"]
    full = ladder.vor_een_u_replay(RECORD, 10, fe3, zero_image_row=False)
    cut = ladder.vor_een_u_replay(RECORD, 10, fe3, zero_image_row=True)
    for name, j in (("pivot_row", -1), ("row_below", -2)):
        wet = full["wet"][name]
        assert wet.sum() > 60 and np.any(nemo[j][wet])
        assert np.array_equal(full[name][wet], nemo[j][wet]), name
    wet = full["wet"]["pivot_row"]
    err = np.max(np.abs(cut["pivot_row"][wet] - nemo[-1][wet]))
    assert err > 0.5 * np.max(np.abs(nemo[-1][wet]))
    wet = full["wet"]["row_below"]
    assert np.array_equal(cut["row_below"][wet], nemo[-2][wet])


@needs_record
@pytest.mark.parametrize("before_arm, exact", [(False, True), (True, False)])
def test_card_pivot_row_coriolis_matches_nemo_only_with_the_image_row(
        before_arm, exact):
    """The landed statement and its plant: the card's stp_2D right-hand side
    on the pivot row equals NEMO's to rounding; switching the card's
    een_fold_image_v_row off puts it out by order one."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD, arm="vor_fold_replay", kt_max=10, allow_dirty=True,
                     before_arm=before_arm)
    assert out["een_fold_image_v_row"] is (not before_arm)
    last = {r["row"]: r for r in out["per_kt"][-1]["replay_rows"]}
    pivot = last["pivot_row"]["card_vs_NEMO_record"]
    assert pivot["n_wet"] == 68 and pivot["ref_max_abs"] > 0.0
    assert last["pivot_row"]["replay_as_compiled_vs_NEMO_record"]["n_unequal"] == 0
    assert last["row_below"]["card_vs_NEMO_record"]["relative"] < 1.0e-15
    if exact:
        assert pivot["relative"] < 1.0e-15
    else:
        assert pivot["relative"] > 0.5


def test_fold_top_v_operands_are_untouched_unless_a_row_is_given():
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bt
    v_mask = jnp.asarray(np.arange(12.0).reshape(4, 3))
    same, flag = bt._nemo_fold_top_v_operands(v_mask, False, None)
    assert same is v_mask and flag is False
    top = jnp.asarray([1.0, 0.0, 1.0])
    moved, flag = bt._nemo_fold_top_v_operands(v_mask, False, top)
    assert flag is True
    assert np.array_equal(np.asarray(moved)[-1], [1.0, 0.0, 1.0])
    assert np.array_equal(np.asarray(moved)[:-1], np.asarray(v_mask)[:-1])


def test_only_the_g1_card_selects_the_fold_row_v_face_depth():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import BarotropicConfig
    assert BarotropicConfig._field_defaults["barotropic_fold_top_v_face_ssh"] is False
    assert BarotropicConfig._fields[-1] == "barotropic_fold_top_v_face_ssh"
    src = Path(recipe.__file__).read_text()
    assert src.count("barotropic_fold_top_v_face_ssh=True") == 1
    start = src.index("def build_orca2g1_flat_card")
    assert src.index("barotropic_fold_top_v_face_ssh=True") > start


@needs_record
@pytest.mark.parametrize("selected", [True, False])
def test_solver_v_face_sea_level_on_the_fold_row_is_nemos_only_when_selected(
        selected):
    """The landed statement and its plant, at the operand: NEMO's row is
    NONZERO; the card equals it to rounding with the option and holds zero
    without it.  The row below is exact to rounding in both arms."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD, arm="fold_v_depth", kt_max=10, allow_dirty=True,
                     without_fold_v_depth=not selected)
    assert out["barotropic_fold_top_v_face_ssh"] is selected
    rows = {r["row"]: r for r in out["per_kt"][-1]["operand_rows"]}
    top, below = rows["fold_image_v_row"], rows["row_below"]
    assert top["n_nemo_nonzero"] > 60 and top["nemo_max_abs"] > 0.0
    assert below["relative"] < 1.0e-15
    if selected:
        assert top["relative"] < 1.0e-15
    else:
        assert top["card_max_abs"] == 0.0 and top["relative"] == 1.0


@needs_record
@pytest.mark.parametrize("selected, bound", [(True, 1.0e-13), (False, None)])
def test_one_step_from_nemos_restart_needs_the_fold_row_v_face_depth(
        selected, bound):
    """Whole-step effect and plant: one step from NEMO's restart at kt 20,
    fold-band sea level relative to the band's own maximum."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD, arm="restart_entry", kt_max=21, allow_dirty=True,
                     without_fold_v_depth=not selected)
    ssh = next(r for r in out["per_kt"][-1]["rows"] if r["name"].endswith(".ssh"))
    fold = ssh["regions"]["fold"]
    assert out["per_kt"][-1]["kt"] == 21 and fold["ref_max_abs"] > 0.0
    rel = fold["max_abs"] / fold["ref_max_abs"]
    if selected:
        assert rel < 1.0e-14
    else:
        assert rel > 1.0e-13



def test_land_distance_counts_cells_to_the_nearest_dry_cell():
    wet = _mask() > 0.5
    dist = ladder.land_distance(wet)
    assert dist[5, 7] == 0 and dist[0, 3] == 0          # island, south wall
    assert dist[4, 7] == 1 and dist[3, 7] == 2 and dist[1, 0] == 1
    assert dist[11, 0] == 6                 # island corner (6, 6): 5 rows, 6 columns
    assert dist.max() < 40


@needs_record
def test_interior_event_at_step_17_is_far_from_any_boundary():
    """Census facts of the first sea-level cell over the bar.  Plant: a cell
    counted as over the bar must really differ (the bar itself is the gate).
    This arm's one-step re-run keeps the card's own T and S, which is what
    carries the event (round 4: see the tracer-carry test below)."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD, arm="interior_events", kt_max=17, allow_dirty=True)
    assert all(k["n_over_bar"] == 0 for k in out["per_kt"][:16])
    ev = out["per_kt"][16]
    assert ev["kt"] == 17 and ev["n_over_bar"] == 2 and ev["cell"] == [72, 70]
    assert ev["region"] == "interior_open" and ev["cells_to_land"] > 10
    assert ev["rows_to_fold"] > 50 and abs(ev["entry_error_at_cell"]) < 1.0e-16
    assert abs(ev["error"]) > 5.0e-15
    # NEMO's dynamic entry with the card's own T, S reproduces it
    assert abs(ev["one_step_from_nemo_entry_error_at_cell"]) > 0.9 * abs(ev["error"])



# ------------------------------------------------------------------ round 4
RECORD4 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
               "round3/oracle_g1_r3/p3_a_100step")
needs_record4 = pytest.mark.skipif(
    not (RECORD4 / "oracle_g1step_kt00000050.bin").is_file(),
    reason="the 50-stage-step G1 record is not admitted on this machine")


def test_only_the_g1_card_selects_nemos_stage1_sea_level():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert LatLonCGridOceanConfig._field_defaults["rk3_stage1_ssh_nemo_literal"] is False
    assert LatLonCGridOceanConfig._fields[-1] == "rk3_stage1_ssh_nemo_literal"
    src = Path(recipe.__file__).read_text()
    assert src.count("rk3_stage1_ssh_nemo_literal=True") == 1
    assert src.index("rk3_stage1_ssh_nemo_literal=True") > src.index(
        "def build_orca2g1_flat_card")


def test_stage_steps_are_read_off_the_record(tmp_path):
    cr = ladder._checker()
    for kt, size in ((1, 64), (2, 64), (3, 64), (4, 8), (5, 8)):
        cr.step_path(tmp_path, kt).write_bytes(b"x" * size)
    assert ladder.stage_steps(tmp_path) == 3


@needs_record4
def test_offline_stage_replay_reproduces_nemo_and_separates_the_two_orders():
    """NEMO's stage statements from its own frames: 0 unequal in NEMO's order
    (the replay is a transliteration check), and the legacy stage-1 order
    differs from NEMO's record in the last place at thousands of cells."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD4, arm="stage_replay", kt_min=17, kt_max=17, allow_dirty=True)
    rows = out["per_kt"][0]["replay"]
    for name, r in rows.items():
        if "nemo" in r:
            assert r["nemo"]["n_unequal"] == 0, name
            assert r["nemo"]["ref_max_abs"] > 0.0, name
    legacy = rows["stage1 ssh (:148)"]["card_order"]
    assert legacy["n_unequal"] > 1000 and legacy["max_abs"] < 1.0e-16


@needs_record4
@pytest.mark.parametrize("selected", [True, False])
def test_stage1_sea_level_is_nemos_bit_for_bit_only_when_selected(selected):
    """The landed statement and its plant: given NEMO's handoff, the card's
    stage-1 sea level has no unequal cell with the option and thousands
    without it."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD4, arm="stages", kt_min=3, kt_max=3, allow_dirty=True,
                     without_stage1_ssh=not selected)
    assert out["rk3_stage1_ssh_nemo_literal"] is selected
    ssh = next(r for r in out["per_kt"][0]["stages"][0]["rows"]
               if r["name"].endswith(".ssh"))
    assert ssh["ref_max_abs"] > 0.0
    assert (ssh["n_unequal"] == 0) if selected else (ssh["n_unequal"] > 500)


@needs_record4
@pytest.mark.parametrize("ts_from_chain", [False, True])
def test_step_17_event_is_carried_by_the_entry_tracers(ts_from_chain):
    """One step from NEMO's step-17 entry: at the bar with NEMO's T and S,
    over it at the event cell with the card's own (2 and 3 last-place units
    off there).  Each arm is the other's plant."""
    pytest.importorskip("netCDF4")
    ladder.CELL[:] = [72, 70]
    try:
        out = ladder.run(RECORD4, arm="given_entry", kt_min=17, kt_max=17,
                         allow_dirty=True, ts_from_chain=ts_from_chain)
    finally:
        ladder.CELL[:] = []
    assert out["ts_from_chain_measurement_arm"] is ts_from_chain
    ssh = next(r for r in out["per_kt"][0]["rows"] if r["name"].endswith(".ssh"))
    if ts_from_chain:
        assert ssh["status"] != "AT-BAR" and abs(ssh["at_cell"]["error"]) > 5.0e-15
        assert ssh["max_abs_cell"] == [72, 70]
    else:
        assert ssh["status"] == "AT-BAR" and ssh["max_abs"] < 1.0e-16
