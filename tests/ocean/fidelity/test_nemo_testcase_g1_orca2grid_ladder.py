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

