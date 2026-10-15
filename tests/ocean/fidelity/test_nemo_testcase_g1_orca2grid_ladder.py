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

def _run_in_order(fields, names):
    """``names`` are all fields, in this order (a pin later appended or
    inserted fields cannot move)."""
    it = iter(fields)
    return all(name in it for name in names)


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
    assert _run_in_order(BarotropicConfig._fields, ("barotropic_fold_top_v_face_ssh",))   # round 8 appended one
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
@pytest.mark.parametrize("off, first", [((), None), (("without_ratio_order",), 21)])
def test_no_interior_event_through_step_30_with_the_round8_statements(off, first):
    """Census of sea-level cells over the bar.  Rounds 4 to 7 had a first
    event at step 17, 23, then 21 (cell (76, 34), carried by the chain's T
    and S).  With round 8's two statements there is none through step 30
    (the first is at step 57 of the 100-step record).  Plant: with the
    solver's ratios divided by the face area the step-21 event is back."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD, arm="interior_events", kt_max=30, allow_dirty=True,
                     **{k: True for k in off})
    over = [k["kt"] for k in out["per_kt"] if k["n_over_bar"]]
    assert (over == []) if first is None else (over[0] == first)



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
    assert _run_in_order(LatLonCGridOceanConfig._fields, ("rk3_stage1_ssh_nemo_literal",))   # round 9 appended three
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
def test_step_21_is_at_the_bar_with_the_chains_own_tracers(ts_from_chain):
    """One step from NEMO's entry of step 21.  Through round 7 the card's own
    T and S put one cell over the bar there (the tracer carrier).  Round 8's
    stored T-point ratio makes the chain's T and S NEMO's bit for bit, so
    both arms are at the bar.  The option's own plant is the step-19 test."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD4, arm="given_entry", kt_min=21, kt_max=21,
                     allow_dirty=True, ts_from_chain=ts_from_chain)
    assert out["ts_from_chain_measurement_arm"] is ts_from_chain
    ssh = next(r for r in out["per_kt"][0]["rows"] if r["name"].endswith(".ssh"))
    assert ssh["status"] == "AT-BAR" and ssh["max_abs"] < 1.0e-16


# --- round 5: the solver's substep loop --------------------------------------

RECORD5 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
               "round4/oracle_g1_r4/p3_a_100step")
needs_record5 = pytest.mark.skipif(
    not (RECORD5 / "oracle_g1spgts_kt00000017.bin").is_file(),
    reason="the G1 substep record is not admitted on this machine")
ROUND5_OPTIONS = ("barotropic_flux_update_nemo_literal",
                  "barotropic_substep_dt_nemo_literal",
                  "barotropic_entry_depth_nemo_literal")


def test_only_the_g1_card_selects_the_three_solver_statements():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import BarotropicConfig
    assert _run_in_order(BarotropicConfig._fields, ROUND5_OPTIONS)
    src = Path(recipe.__file__).read_text()
    start = src.index("def build_orca2g1_flat_card")
    for name in ROUND5_OPTIONS:
        assert BarotropicConfig._field_defaults[name] is False
        assert src.count(f"{name}=True") == 1
        assert src.index(f"{name}=True") > start
    if RECORD.is_dir():      # the built card carries them, validator included
        cfg = recipe.build_nemo_testcase_card(
            "ORCA2G1-flat", deck_root=RECORD).recipe.model_config
        assert all(getattr(cfg.barotropic, name) is True for name in ROUND5_OPTIONS)


def _solver_options(**on):
    from types import SimpleNamespace

    from legoesm.ocean.state import BarotropicConfig
    return SimpleNamespace(barotropic=BarotropicConfig()._replace(**on))


def test_literal_flux_update_is_the_written_order_and_dry_faces_are_zero():
    """dynspg_ts.f90:557-568 against the same arithmetic in numpy, one
    operation at a time; a dry face is 0 / 1, never a division by zero."""
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as B
    rng = np.random.default_rng(5)
    shape = (6, 9)
    vel, pgf, trd, frc = (rng.normal(scale=1e-3, size=shape) for _ in range(4))
    h_in, h_pgf, h_mid, h_kmm, h_exit = (100.0 + rng.normal(scale=1e-2, size=shape)
                                         for _ in range(5))
    wet = np.ones(shape)
    wet[2, 3] = 0.0
    h_exit[2, 3] = 0.0
    dt = 10800.0 / 13
    args = [jnp.asarray(a) for a in (vel, h_in, h_pgf, h_mid, h_kmm, h_exit, pgf, trd, frc)]
    cfg = _solver_options(barotropic_flux_update_nemo_literal=True)
    got = np.asarray(B._nemo_flux_form_literal_velocity_update(
        *args, dt, jnp.asarray(wet), cfg, 0))
    want = (h_in * vel + dt * (h_pgf * pgf + h_mid * trd + h_kmm * frc)) * (
        wet / (h_exit + 1.0 - wet))
    assert np.array_equal(got, want) and got[2, 3] == 0.0
    divided = (h_in * vel + dt * (h_pgf * pgf + h_mid * trd + h_kmm * frc)) / np.where(
        wet > 0, h_exit, 1.0) * wet
    assert np.count_nonzero(divided != want) > 0        # the other order differs
    # the entry depth is read at the first substep only, and only when selected
    both = _solver_options(barotropic_flux_update_nemo_literal=True,
                           barotropic_entry_depth_nemo_literal=True)
    first = np.asarray(B._nemo_flux_form_literal_velocity_update(
        *args, dt, jnp.asarray(wet), both, 0))
    later = np.asarray(B._nemo_flux_form_literal_velocity_update(
        *args, dt, jnp.asarray(wet), both, 1))
    assert np.array_equal(later, want)
    want_first = (h_kmm * vel + dt * (h_pgf * pgf + h_mid * trd + h_kmm * frc)) * (
        wet / (h_exit + 1.0 - wet))
    assert np.array_equal(first, want_first) and not np.array_equal(first, want)


def test_entry_face_depth_is_rest_depth_times_one_plus_ratio_only_when_selected():
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as B
    h0u, h0v = jnp.asarray([[100.0, 0.0]]), jnp.asarray([[100.0], [50.0]])
    r3u, r3v = jnp.asarray([[3.0e-5, 0.0]]), jnp.asarray([[1.0e-5], [-2.0e-5]])
    summed = ("sum_u", "sum_v")
    off = B._nemo_entry_face_depths(_solver_options(), summed, (h0u, h0v), (None, None, r3u, r3v))
    assert off is summed
    hu, hv = B._nemo_entry_face_depths(
        _solver_options(barotropic_flux_update_nemo_literal=True,
                        barotropic_entry_depth_nemo_literal=True),
        summed, (h0u, h0v), (None, None, r3u, r3v))
    assert np.array_equal(np.asarray(hu), np.asarray(h0u) * (1.0 + np.asarray(r3u)))
    assert np.array_equal(np.asarray(hv), np.asarray(h0v) * (1.0 + np.asarray(r3v)))


def test_solver_statement_options_refuse_a_path_that_does_not_read_them():
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as B
    from legoesm.ocean.state import LatLonCGridOceanConfig
    base = LatLonCGridOceanConfig()
    assert not B.nemo_flux_form_update_active(base)
    for on in ({"barotropic_flux_update_nemo_literal": True},
               {"barotropic_entry_depth_nemo_literal": True}):
        cfg = base._replace(barotropic=base.barotropic._replace(**on))
        with pytest.raises(ValueError, match="nemo_literal=True"):
            B._nemo_fold_top_v_wet(cfg, None, jnp.float64)
    assert B._nemo_fold_top_v_wet(base, None, jnp.float64) is None


def test_guarded_divisor_gives_the_true_quotient_under_jit():
    """rDt_e = rn_Dt / nn_e (dynspg_ts.f90:886): with the shared rounding
    guard on the divisor the compiled quotient is Python's, for G1's pair."""
    import jax
    from legoesm.core.source_rounding import nemo_source_round
    if not jax.config.jax_enable_x64:
        pytest.skip("needs x64")
    guarded = jax.jit(lambda dt: dt / nemo_source_round(jnp.asarray(13.0)))
    assert float(guarded(jnp.asarray(10800.0))) == 10800.0 / 13


def test_substep_row_scores_wet_points_by_region_and_its_plant_fires():
    wet = _mask() > 0
    reg = ladder.regions(_mask())
    ref = np.linspace(1.0, 2.0, wet.size).reshape(wet.shape)
    cand = ref.copy()
    cand[0, 5] += 1.0                       # a dry cell: not scored
    r = ladder.substep_row("x", ref, cand, wet, reg)
    assert r["bit_identical"] and r["n_unequal"] == 0 and r["n_wet"] == int(wet.sum())
    cand[11, 4] += 1.0e-9                   # the pivot row
    r = ladder.substep_row("x", ref, cand, wet, reg)
    assert not r["bit_identical"] and r["max_abs_cell"] == [11, 4]
    assert r["regions"]["fold_pivot_row"]["n_unequal"] == 1
    assert r["regions"]["interior"]["n_unequal"] == 0
    planted = ladder.substep_row("x", ref, ref, wet, reg, plant=True)
    assert planted["regions"]["interior"]["n_unequal"] == 1


@needs_record5
def test_offline_flux_update_replay_reproduces_nemo_and_separates_the_orders():
    """NEMO's velocity update from NEMO's substep record alone: 0 unequal as
    compiled at all 13 substeps (the instrument's own control); dividing by
    the exit depth, or the entry depth as a sum, do not reproduce it."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD5, arm="flux_update_replay", kt_min=17, kt_max=17,
                     allow_dirty=True)
    assert len(out["replay_rows"]) == 26
    assert all(r["nemo_order"]["ref_max_abs"] > 0.0 for r in out["replay_rows"])
    assert out["totals"]["nemo_order"]["n_unequal"] == 0
    assert out["totals"]["divide_by_exit_depth"]["n_unequal"] > 10000
    assert out["totals"]["entry_depth_as_sum"]["n_unequal"] > 1000


@needs_record5
@pytest.mark.parametrize("off, floor", [
    (None, 0), ("without_entry_depth", 1), ("without_flux_literal", 30),
    ("without_substep_dt", 30)])
def test_step_1_substep_loop_is_nemos_bit_for_bit_only_with_all_three(off, floor):
    """The three landed statements and their plants: from NEMO's step-1
    entry every velocity-update row of the 13 substeps and the loop's exit
    are bit-identical with the card; each option switched off alone leaves
    unequal faces in the first substep's update."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD5, arm="substeps", kt_min=1, kt_max=1, allow_dirty=True,
                     **({off: True} if off else {}))
    assert out["n_substeps"] == 13 and not out["scalars_unequal"]
    assert out["trace_passive"]["uu_b"] and out["trace_passive"]["vv_b"]
    update = [r for r in out["rows"] if r["operand"] in ("new.u", "new.v")]
    assert len(update) == 26 and all(r["ref_max_abs"] > 0.0 for r in update)
    first = sum(r["n_unequal"] for r in update if r["substep"] == 1)
    if off is None:
        assert all(r["bit_identical"] for r in update)
        assert all(r["bit_identical"] for r in out["exit_rows"])
        # the one unequal row is the cold-start history frame, which the first
        # substep's weights (1, 0, 0) and the compiled swap never read
        assert out["first_unequal"]["name"] == "i000.entry.sshb"
        assert out["record_scalars"]["i000_entry_sc"][4] == 1.0
    else:
        assert first >= floor and first > 0


# --- round 6: the fold pivot row of the coefficients; the stage velocity -----

ROUND6_SOLVER = ("barotropic_een_ratio_fold_exchange",
                 "barotropic_een_fe3mask_pivot_row")
ROUND6_STAGE = ("rk3_stage1_rhs_nemo_literal", "rk3_stage_velocity_written_order")


def test_only_the_g1_card_selects_the_round6_statements():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import BarotropicConfig, LatLonCGridOceanConfig
    assert _run_in_order(BarotropicConfig._fields, ROUND6_SOLVER)
    assert _run_in_order(LatLonCGridOceanConfig._fields, ROUND6_STAGE)   # rounds 7 and 9 appended seven
    src = Path(recipe.__file__).read_text()
    start = src.index("def build_orca2g1_flat_card")
    for cls, names in ((BarotropicConfig, ROUND6_SOLVER),
                       (LatLonCGridOceanConfig, ROUND6_STAGE)):
        for name in names:
            assert cls._field_defaults[name] is False
            assert src.count(f"{name}=True") == 1
            assert src.index(f"{name}=True") > start
    if RECORD.is_dir():
        cfg = recipe.build_nemo_testcase_card(
            "ORCA2G1-flat", deck_root=RECORD).recipe.model_config
        assert all(getattr(cfg.barotropic, n) is True for n in ROUND6_SOLVER)
        assert all(getattr(cfg, n) is True for n in ROUND6_STAGE)
        assert cfg.barotropic.barotropic_entry_depth_nemo_literal is True  # D119 pending


def test_fold_pivot_row_halves_are_refused_one_without_the_other():
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as B
    assert B.nemo_een_fold_pivot_row(_solver_options()) == (False, False)
    assert B.nemo_een_fold_pivot_row(_solver_options(
        **{n: True for n in ROUND6_SOLVER})) == (True, True)
    for name in ROUND6_SOLVER:
        with pytest.raises(ValueError, match="one fold unit"):
            B.nemo_een_fold_pivot_row(_solver_options(**{name: True}))


def test_ratio_fold_exchange_refuses_a_grid_without_a_stored_pivot_row():
    from types import SimpleNamespace

    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as B
    a = jnp.ones((4, 6))
    assert B.nemo_een_ratio_fold_exchange(a, a, a, None, False) == (a, a, a)
    with pytest.raises(ValueError, match="T-pivot"):
        B.nemo_een_ratio_fold_exchange(a, a, a, SimpleNamespace(fold=None), True)


@needs_record5
def test_ratio_fold_exchange_writes_nemos_three_rules_and_nothing_else():
    """lbcnfd.f90:639-748 at T pivot, sign +1: V and F pivot rows are the
    mirrored row below; the U pivot row's second half its mirrored first."""
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as B
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    grid = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=RECORD5).recipe.grid
    rng = np.random.default_rng(6)
    u, v, f = (rng.normal(size=(148, 180)) for _ in range(3))
    eu, ev, ef = (np.asarray(x) for x in B.nemo_een_ratio_fold_exchange(
        jnp.asarray(u), jnp.asarray(v), jnp.asarray(f), grid, True))
    n = np.arange(180)
    assert np.array_equal(ev[-1], v[-2][(180 - n) % 180])
    assert np.array_equal(ef[-1], f[-2][179 - n])
    assert np.array_equal(eu[-1, 90:], u[-1][179 - n][90:])
    assert np.array_equal(eu[-1, :90], u[-1, :90])
    for got, src in ((eu, u), (ev, v), (ef, f)):
        assert np.array_equal(got[:-1], src[:-1])


@needs_record5
def test_fold_pair_closes_the_coefficients_and_each_half_alone_does_not():
    """The offline leaf table at step 41.  Plants: either half off leaves
    unequal pivot-row faces; the instrument's own control is the ratio in
    NEMO's written order, 0 unequal on every row."""
    out = ladder.run(RECORD5, arm="ratio_fold_replay", kt_min=41, kt_max=41,
                     allow_dirty=True)
    assert out["nemo_halo_ssh_equals_mirrored_row_below"] and out["L3_equals_L2_bitwise"]
    rows = {(r["variant"], r["ratio"]): r for r in out["ratio_rows"]}
    for ratio in ("r3u", "r3v", "r3f"):
        control = rows[("L4_exchange_only_times_inverse_area", ratio)]
        assert all(control[b]["n_unequal"] == 0 for b in (
            "pivot_row", "row_below", "all_rows_below"))
    for ratio in ("r3v", "r3f"):                    # the bare stencil: a magnitude
        bare = rows[("L0_stencil_cyclic_wrap", ratio)]["pivot_row"]
        assert bare["n_unequal"] > 60 and bare["max_abs"] > 0.3 * bare["nemo_max_abs"]
    worst = {}
    for r in out["coefficient_rows"]:
        key = (r["ratio_exchange"], r["fe3mask_pivot_row"])
        worst[key] = max(worst.get(key, 0), r["n_unequal"])
    assert worst[(True, True)] == 0
    assert worst[(True, False)] >= 5 and worst[(False, True)] >= 60
    assert worst[(False, False)] >= 60


def test_fused_multiply_add_rounds_once():
    x = 1.0 + 2.0 ** -30
    assert x * x - 1.0 == 2.0 ** -29                      # the product rounded first
    got = ladder.fused_multiply_add(np.array([x, 0.0]), np.array([x, 3.0]),
                                    np.array([-1.0, 0.0]))
    assert got[0] == 2.0 ** -29 + 2.0 ** -60 and got[1] == 0.0
    with pytest.raises(Exception, match="non-finite"):
        ladder.fused_multiply_add(np.array([0.0]), np.array([np.inf]), np.array([1.0]))


def test_written_order_stage_update_and_correction_match_numpy_under_jit():
    """stprk3_stg.f90:373-378 and :413-418, each product and sum rounded once."""
    import jax
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as M
    from legoesm.ocean.dynamics.barotropic_common import rk3_stage_barotropic_correction
    if not jax.config.jax_enable_x64:
        pytest.skip("needs x64")
    rng = np.random.default_rng(66)
    shape = (64, 64, 1)
    ub, rhs = rng.normal(scale=1e-2, size=shape), rng.normal(scale=1e-7, size=shape)
    qb, qn, qa = (1.0 + rng.normal(scale=1e-4, size=shape) for _ in range(3))
    dt = 3600.0
    mask = np.ones(shape)
    mask[3, 4] = 0.0
    raw = np.asarray(jax.jit(M._rk3_stage_update_written_order)(
        *(jnp.asarray(a) for a in (ub, rhs)), dt,
        *(jnp.asarray(a) for a in (qb, qn, qa, mask))))
    want_raw = ((qb * ub) + ((dt * qn) * rhs)) / qa * mask
    assert np.array_equal(raw, want_raw) and raw[3, 4, 0] == 0.0
    same = np.asarray(M.rk3_stage_velocity_update(
        jnp.asarray(ub), jnp.asarray(rhs), dt, jnp.asarray(mask), vector_form=False,
        qco_before=jnp.asarray(qb), qco_now=jnp.asarray(qn), qco_after=jnp.asarray(qa),
        written_order=True))
    assert np.array_equal(same, want_raw)
    h = 100.0 + rng.normal(scale=1e-2, size=shape)
    r1 = (1.0 / h[..., 0]) * mask[..., 0]
    target = rng.normal(scale=1e-2, size=shape[:2])
    args = [jnp.asarray(a) for a in (want_raw, target, h * mask, r1, mask)]
    got = np.asarray(jax.jit(lambda *a: rk3_stage_barotropic_correction(
        *a, written_order=True))(*args))
    zub = target - (want_raw * (h * mask))[..., 0] * r1
    want = want_raw + zub[..., None] * mask              # :417-418 as written
    assert np.array_equal(got, want) and got[3, 4, 0] == 0.0
    # the association is the statement's: an unmasked dry input stays as it is
    # (uu + zub*0), where (uu + zub)*0 would return zero
    dry = np.array(want_raw)
    dry[3, 4, 0] = 7.0
    kept = np.asarray(rk3_stage_barotropic_correction(
        jnp.asarray(dry), *args[1:], written_order=True))
    assert kept[3, 4, 0] == 7.0
    loose = np.asarray(rk3_stage_barotropic_correction(*args))     # default path runs
    assert np.allclose(loose, want, rtol=0, atol=1e-17)


def test_written_order_vector_stage_update_matches_numpy_under_jit():
    """stprk3_stg.f90:366-370: multiply, add, and mask round separately."""
    import jax
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as M
    if not jax.config.jax_enable_x64:
        pytest.skip("needs x64")
    rng = np.random.default_rng(7)
    shape = (96, 96, 1)
    before = rng.normal(scale=1e-2, size=shape)
    rhs = rng.normal(scale=1e-7, size=shape)
    mask = np.ones(shape)
    dt = np.float64(3600.0)

    # Independent one-source-operation-at-a-time NumPy transcription.
    expected = np.multiply(dt, rhs)
    expected = np.add(before, expected)
    expected = np.multiply(expected, mask)

    def compiled(written_order):
        return jax.jit(lambda b, r, m: M.rk3_stage_velocity_update(
            b, r, dt, m, vector_form=True,
            written_order=written_order))(
            jnp.asarray(before), jnp.asarray(rhs), jnp.asarray(mask))

    got = np.asarray(compiled(True))
    contracted = np.asarray(compiled(False))
    assert np.array_equal(got, expected)
    assert np.count_nonzero(contracted != expected) == 465


@needs_record5
@pytest.mark.parametrize("off, floor", [((), 0), (("without_stage1_rhs",), 1),
                                        (("without_written_order",), 1)])
def test_stage1_velocity_is_nemos_at_step_1_and_each_option_is_needed(off, floor):
    """GIVEN-NEMO-ENTRY, stage 1 of step 1.  With both options the card's
    stage-1 u and v equal NEMO's frame; either one off leaves unequal faces
    (the plants).  The replay's own control: NEMO's written order on NEMO's
    operands reproduces NEMO's frame."""
    out = ladder.run(RECORD5, arm="stages", kt_min=1, kt_max=1, allow_dirty=True,
                     **{name: True for name in off})
    stage1 = out["per_kt"][0]["stages"][0]
    unequal = sum(r["n_unequal"] for r in stage1["rows"]
                  if r["name"].split(".")[-1] in ("u", "v"))
    assert (unequal == 0) if not off else (unequal >= floor)
    for r in stage1["momentum_replay"]["rows"]:
        if r.get("order") == "written" and r["after_ratio"] == "nemo_recorded_ratio":
            assert r["vs_nemo"]["n_unequal"] == 0
    if not off:
        for r in stage1["momentum_replay"]["card_operand_rows"]:
            assert r["card_rhs_vs_nemo_rhs"]["n_unequal"] == 0
            assert r["card_rhs_vs_nemo_rhs"]["ref_max_abs"] > 1e-9   # not a zero row


def test_stage1_rhs_substitution_arm_is_refused_outside_the_stage_table():
    with pytest.raises(Exception, match="stages table"):
        ladder.run(RECORD, arm="geometry", stage1_rhs_from_nemo=True, allow_dirty=True)


# --- round 7: the vorticity order, stage 3's residual, the stage F ratio, ----
# --- the stored top V row -----------------------------------------------------

ROUND7 = ("een_vorticity_written_order", "zdf_barotropic_removal_nemo_literal",
          "rk3_fold_exchange_top_v_row", "rk3_stage_r3f_nemo_literal")


def test_only_the_g1_card_selects_the_round7_statements():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert _run_in_order(LatLonCGridOceanConfig._fields, ROUND7)     # round 9 appended three
    src = Path(recipe.__file__).read_text()
    start = src.index("def build_orca2g1_flat_card")
    for name in ROUND7:
        assert LatLonCGridOceanConfig._field_defaults[name] is False
        assert name not in src[:start], name      # no other card selects it
        assert f"{name}=True" in src[start:] and f"cfg.{name} is True" in src[start:]


def _triad_inputs(seed=7, ny=6, nx=8):
    rng = np.random.default_rng(seed)
    pos = lambda *shape: jnp.asarray(rng.uniform(0.5, 2.0, size=shape))  # noqa: E731
    return dict(
        zeta=jnp.zeros((ny + 1, nx + 1, 1)), h_vtx=pos(ny + 1, nx + 1, 1) * 100.0,
        h_v=pos(ny + 1, nx, 1) * 100.0, v=jnp.asarray(rng.normal(size=(ny + 1, nx, 1))),
        h_u=pos(ny, nx + 1, 1) * 100.0, u=jnp.asarray(rng.normal(size=(ny, nx + 1, 1))),
        u_mask_3d=jnp.ones((ny, nx + 1, 1)), v_mask_3d=jnp.ones((ny + 1, nx, 1)),
        vtx_mask=jnp.ones((ny + 1, nx + 1)), f_vtx=pos(ny + 1, nx + 1) * 1.0e-4,
        q_boundary="nemo_live",
        metric_widths=(pos(ny, nx + 1) * 1e5, pos(ny + 1, nx) * 1e5,
                       pos(ny, nx + 1) * 1e5, pos(ny + 1, nx) * 1e5))


def test_written_order_vorticity_is_nemos_statement_and_real_equal_to_the_default():
    """dynvor.f90:720, :727, :766-767, :772-775, :779-784 transcribed in numpy
    at one interior U face and one interior V face; the default order is the
    same real number and not the same bits everywhere."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import pv_flux_al81_partial_cell
    kw = _triad_inputs()
    u0, v0 = (np.asarray(x) for x in pv_flux_al81_partial_cell(**kw))
    u1, v1 = (np.asarray(x) for x in pv_flux_al81_partial_cell(**kw, written_order=True))
    np.testing.assert_allclose(u1, u0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(v1, v0, rtol=1e-12, atol=0)
    assert np.any(u1 != u0) and np.any(v1 != v0)
    n = {k: np.asarray(v) for k, v in kw.items() if k not in ("q_boundary", "metric_widths")}
    e1u, e1v, e2u, e2v = (np.asarray(x) for x in kw["metric_widths"])
    zwz = n["f_vtx"] * (1.0 / n["h_vtx"][..., 0])          # vertex [j, i] = SW of cell (j, i)
    zwx = e2u * n["h_u"][..., 0] * n["u"][..., 0]
    zwy = e1v * n["h_v"][..., 0] * n["v"][..., 0]

    def triads(j, i):                                       # the cell's four corners
        sw, se, nw, ne = zwz[j, i], zwz[j, i + 1], zwz[j + 1, i], zwz[j + 1, i + 1]
        return {"ne": nw + ne + se, "nw": sw + nw + ne, "se": ne + se + sw, "sw": se + sw + nw}
    j, i = 2, 3                                             # U face i: west cell i-1, east cell i
    w, e = triads(j, i - 1), triads(j, i)
    zua = (1.0 / 12.0) * (1.0 / e1u[j, i]) * (
        (w["ne"] * zwy[j + 1, i - 1] + e["nw"] * zwy[j + 1, i])
        + (w["se"] * zwy[j, i - 1] + e["sw"] * zwy[j, i]))
    assert u1[j, i, 0] == zua
    s_, n_ = triads(j - 1, i), triads(j, i)                 # V face j: south cell j-1, north cell j
    zva = -((1.0 / 12.0) * (1.0 / e2v[j, i]) * (
        (n_["sw"] * zwx[j, i] + n_["se"] * zwx[j, i + 1])
        + (s_["nw"] * zwx[j - 1, i] + s_["ne"] * zwx[j - 1, i + 1])))
    assert v1[j, i, 0] == zva
    with pytest.raises(ValueError, match="metric_widths"):
        pv_flux_al81_partial_cell(**{**kw, "metric_widths": None}, written_order=True)


def test_zdf_residual_option_needs_the_stage_correction():
    from types import SimpleNamespace as NS

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as M
    ok = NS(zdf_barotropic_removal_nemo_literal=True, momentum_time_integrator="rk3_ws",
            zdf_baroclinic_only=True, barotropic=NS(nemo_stage_mean_imposition=True))
    assert M._zdf_residual_nemo_literal(ok) is True
    assert M._zdf_residual_nemo_literal(NS(zdf_barotropic_removal_nemo_literal=False)) is False
    for bad in (dict(momentum_time_integrator="euler"), dict(zdf_baroclinic_only=False),
                dict(barotropic=NS(nemo_stage_mean_imposition=False))):
        with pytest.raises(ValueError, match="stage correction"):
            M._zdf_residual_nemo_literal(NS(**{**vars(ok), **bad}))


@needs_record5
def test_top_v_row_exchange_reproduces_nemos_own_stored_row():
    """stprk3_stg.f90:572-573.  NEMO's after frame at step 100 handed to the
    helper with its top row zeroed: the helper must write NEMO's own top row
    back, bit for bit, and touch no other row.  Plant: the wrong sign."""
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as M
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    ladder._fp64()      # the card's geometry follows the precision policy
    card = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=RECORD5)
    grid = card.recipe.grid
    vm = np.asarray(M.compute_face_masks_3d(card.recipe.z_coord.is_active, grid)[1],
                    dtype=np.float64)
    nemo = ladder.pad_v(ladder.read_step(RECORD5, 100)["f_vv_k1_bb"])[..., None]
    assert np.count_nonzero(nemo[-1]) > 50                  # the row carries signal
    cut = nemo.copy()
    cut[-1] = 0.0
    got = np.asarray(M._rk3_fold_exchange_top_v(jnp.asarray(cut), grid, jnp.asarray(vm)))
    assert np.array_equal(got, nemo)
    assert not np.array_equal(-got[-1], nemo[-1])           # plant: the sign matters
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="north fold"):
        M._rk3_fold_exchange_top_v(jnp.asarray(cut), SimpleNamespace(fold=None), vm)


@needs_record5
def test_stage_f_ratio_is_nemos_interpolation_of_the_ratios():
    """stprk3_stg.f90:239 at step 2: the stage-3 F thickness from the
    interpolated ratios equals NEMO's at every F point; from the ratio of
    the interpolated sea level it differs at one (the plant)."""
    from types import SimpleNamespace as NS

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as M
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.vertical import nemo_qco_live_vorticity_e3f_cgrid as e3f_of
    ladder._fp64()      # the card's geometry follows the precision policy
    card = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=RECORD5)
    zc, grid = card.recipe.z_coord, card.recipe.grid
    fe3 = np.asarray(zc.nemo_een_barotropic.fe3mask)[..., 0]
    g = ladder.read_step(RECORD5, 2)
    rep = ladder.rhs_term_replay(RECORD5, 2, fe3, stage=3)
    wet = np.isfinite(rep["zwz"]) & (rep["e3f"] != 0)
    pair = (jnp.asarray(g["e_ssh_bb"]), jnp.asarray(g["b_ssh_aa"]))
    on = NS(rk3_stage_r3f_nemo_literal=True)
    assert M._nemo_ws_stage_r3f(NS(rk3_stage_r3f_nemo_literal=False), 3, None, zc, grid) is None
    assert M._nemo_ws_stage_r3f(on, 1, None, zc, grid) is None
    r3f = M._nemo_ws_stage_r3f(on, 3, lambda: pair, zc, grid)
    eta = jnp.asarray(g["2_ssh_aa"])
    lit = np.asarray(e3f_of(eta, zc, jnp.float64, grid=grid, r3f_native=r3f))[1:, 1:, 0]
    own = np.asarray(e3f_of(eta, zc, jnp.float64, grid=grid))[1:, 1:, 0]
    assert np.count_nonzero((lit - rep["e3f"])[wet]) == 0
    assert np.count_nonzero((own - rep["e3f"])[wet]) == 1


def _terms(out, key):
    return [r[key]["n_unequal"] for r in out["per_kt"][0]["rows"]]


@needs_record5
@pytest.mark.parametrize("off, floor", [((), 0), (("without_vor_order",), 300)])
def test_step_2_right_hand_side_is_nemos_only_in_the_written_order(off, floor):
    """The offline replay of dynhpg.f90:380-394 and vor_een against NEMO's
    recorded right-hand side is the control (0 unequal).  The card's pressure
    gradient is bit-equal either way; its vorticity term only in NEMO's
    order.  Plant: the option off leaves hundreds of unequal faces.  This G1
    flat-bottom card has zero KEG, so it cannot test the HPG/KEG split."""
    out = ladder.run(RECORD5, arm="rhs_terms", kt_min=2, kt_max=2, allow_dirty=True,
                     **{k: True for k in off})
    assert _terms(out, "control_replay_vs_nemo") == [0, 0]
    assert _terms(out, "card_pressure_gradient_vs_replay") == [0, 0]
    for key in ("card_vorticity_vs_replay", "card_total_vs_nemo"):
        got = _terms(out, key)
        assert (got == [0, 0]) if not floor else min(got) >= floor, (key, got)


@needs_record5
def test_right_hand_side_control_fires_on_a_planted_cell():
    out = ladder.run(RECORD5, arm="rhs_terms", kt_min=2, kt_max=2, allow_dirty=True,
                     plant="score")
    assert _terms(out, "control_replay_vs_nemo") == [1, 1]


G1B_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
    "round26/g1b_merged")
needs_g1b_record = pytest.mark.skipif(
    not (G1B_RECORD / "oracle_g1step_kt00000002.bin").is_file(),
    reason="the admitted G1b record is not on this machine")


@needs_g1b_record
def test_g1b_nonzero_hpg_keg_split_and_plant(monkeypatch):
    """The split closure and its plant both run on nonzero HPG and KEG."""
    monkeypatch.setattr(ladder, "CASE", "ORCA2G1b-vec")
    out = ladder.run(G1B_RECORD, arm="rhs_terms", kt_min=2, kt_max=2,
                     allow_dirty=True)
    for row in out["per_kt"][0]["rows"]:
        assert row["card_term_max_abs"]["hpg"] > 0.0
        assert row["card_term_max_abs"]["keg"] > 0.0
    split = _terms(out, "control_split_sum_vs_combined")
    assert split == [5, 0]
    assert _terms(out, "card_hpg_vs_replay") == [0, 0]
    assert _terms(out, "card_keg_vs_replay") == [0, 0]

    planted = ladder.run(G1B_RECORD, arm="rhs_terms", kt_min=2, kt_max=2,
                         allow_dirty=True, plant="rhs_split")
    assert _terms(planted, "control_split_sum_vs_combined") == [n + 1 for n in split]
    assert _terms(planted, "card_hpg_vs_replay") == [1, 1]


@needs_g1b_record
def test_g1b_stage1_replay_selects_vector_velocity_statement(monkeypatch):
    """The G1b stage table must replay the executed IF arm, not QCO ELSE."""
    monkeypatch.setattr(ladder, "CASE", "ORCA2G1b-vec")
    out = ladder.run(G1B_RECORD, arm="stages", kt_min=2, kt_max=2,
                     allow_dirty=True)
    replay = out["per_kt"][0]["stages"][0]["momentum_replay"]
    assert {r["after_ratio"] for r in replay["rows"]} == {
        "not_read_vector_form", "card_live_row"}
    for row in replay["card_operand_rows"]:
        assert row["card_raw_vs_written_update_of_card_rhs"]["n_unequal"] == 0

    off = ladder.run(G1B_RECORD, arm="stages", kt_min=2, kt_max=2,
                     allow_dirty=True, without_written_order=True)
    replay_off = off["per_kt"][0]["stages"][0]["momentum_replay"]
    assert sum(r["card_raw_vs_written_update_of_card_rhs"]["n_unequal"]
               for r in replay_off["card_operand_rows"]) >= 1


@needs_record5
@pytest.mark.parametrize("off, stage, key, floor", [
    ((), None, None, 0),
    (("without_zdf_residual",), 3, "card_vs_replay", 100),
    (("without_stage_r3f",), 3, "card_rhs_vs_replay_rhs", 1)])
def test_step_2_stages_are_nemos_and_each_stage_3_option_is_needed(off, stage, key, floor):
    """Stages 2 and 3 replayed offline from NEMO's frames are 0 unequal
    against NEMO's stage frames (the control).  The card equals the replay on
    every wet face; each option off (the plants) leaves unequal faces."""
    out = ladder.run(RECORD5, arm="stages", kt_min=2, kt_max=2, allow_dirty=True,
                     **{k: True for k in off})
    for st in out["per_kt"][0]["stages"][1:]:
        for r in st["stage_replay"]["rows"]:
            assert r["control_replay_vs_nemo"]["n_unequal"] == 0
            if not floor:
                assert all(v["n_unequal"] == 0 for v in r.values() if isinstance(v, dict))
    if floor:
        rows = out["per_kt"][0]["stages"][stage - 1]["stage_replay"]["rows"]
        assert max(r[key]["n_unequal"] for r in rows) >= floor


@needs_record5
@pytest.mark.parametrize("off, want", [((), 0), (("without_fold_exchange_v",), 5)])
def test_every_stage_frame_holds_nemos_top_v_row_only_with_the_exchange(off, want):
    """The PRODUCTION call (review finding): at step 5 the sea-level signal
    has reached the fold, and NEMO's stage frames hold the fold image on the
    top V row after every stage (stprk3_stg.f90:572-573).  The card's stage
    velocities equal them on the whole domain; with the option off (the
    plant) the wet top-row faces that carry signal, five or six, are unequal
    and nothing else is."""
    out = ladder.run(RECORD5, arm="stages", kt_min=5, kt_max=5, allow_dirty=True,
                     **{k: True for k in off})
    for st in out["per_kt"][0]["stages"]:
        v = next(r for r in st["rows"] if r["name"].endswith(".v"))
        got = (v["n_unequal"], v["regions"]["fold_pivot_row"]["n_unequal"])
        # off: 5 faces at stages 1 and 2, 6 at stage 3, all on the top row
        assert (got == (0, 0)) if not want else (got[0] == got[1] >= want), (
            st["stage"], got)


# --- round 8: the solver's ratio order; the stored T-point ratio --------------

ROUND8_SOLVER = "barotropic_een_ratio_written_order"
ROUND8_COORDINATE = "nemo_r3t_stored_product"


def test_only_the_g1_card_selects_the_round8_statements():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import BarotropicConfig
    from legoesm.ocean.vertical import OceanPartialCellCoordinate
    assert _run_in_order(BarotropicConfig._fields, (ROUND8_SOLVER,))
    assert _run_in_order(OceanPartialCellCoordinate._fields, (ROUND8_COORDINATE,))
    assert BarotropicConfig._field_defaults[ROUND8_SOLVER] is False
    assert OceanPartialCellCoordinate._field_defaults[ROUND8_COORDINATE] is False
    src = Path(recipe.__file__).read_text()
    start = src.index("def build_orca2g1_flat_card")
    for name in (ROUND8_SOLVER, ROUND8_COORDINATE):
        assert src.count(f"{name}=True") == 1
        assert src.index(f"{name}=True") > start
    if RECORD5.is_dir():
        card = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=RECORD5)
        assert card.recipe.model_config.barotropic.barotropic_een_ratio_written_order is True
        assert card.recipe.z_coord.nemo_r3t_stored_product is True


def test_ratio_written_order_multiplies_by_the_reciprocal_area():
    """domqco.f90:266-269, :281-285: (lead * inverse depth) * (1 / area).
    Off, the caller's divided ratios come back untouched; on, the two orders
    differ somewhere (or the option would be vacuous)."""
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as B
    rng = np.random.default_rng(8)
    lead, r1_h, area = (rng.uniform(0.5, 2.0, size=(3, 40, 50)) * scale
                        for scale in (1.0e7, 1.0e-2, 1.0e10))
    divided = tuple(jnp.asarray(lead[k] * r1_h[k] / area[k]) for k in range(3))
    args = (divided, tuple(jnp.asarray(x) for x in lead),
            tuple(jnp.asarray(x) for x in r1_h), tuple(jnp.asarray(x) for x in area))
    for selected in ((), (False,)):
        assert B._nemo_een_ratio_written_order(*args, selected) is divided
    written = B._nemo_een_ratio_written_order(*args, (True,))
    for k in range(3):
        assert np.array_equal(np.asarray(written[k]), (lead[k] * r1_h[k]) * (1.0 / area[k]))
    assert any(np.any(np.asarray(written[k]) != np.asarray(divided[k])) for k in range(3))


def test_stored_r3t_product_is_rounded_before_one_is_added():
    """domqco.f90:257 stores ssh * r1_ht_0; 1 + r3t then reads the stored
    value.  Operands are chosen where a fused multiply-add of the same three
    numbers rounds differently, so a contracted sum cannot pass."""
    import jax
    from legoesm.ocean import eos, vertical
    rng = np.random.default_rng(88)
    eta = rng.uniform(-0.5, 0.5, size=20000)
    r1 = np.full_like(eta, 0.01)
    written = 1.0 + eta * r1
    fused = ladder.fused_multiply_add(eta, r1, 1.0)
    pick = np.nonzero(written != fused)[0][:8]
    assert pick.size >= 1                       # the operands discriminate
    eta = jnp.asarray(eta[pick].reshape(-1, 1))
    depth = jnp.full(eta.shape, 100.0)

    zc = vertical.OceanPartialCellCoordinate(*([None] * 9), nemo_r3t_stored_product=True)
    assert vertical._nemo_r3t_product(eta, zc._replace(nemo_r3t_stored_product=False)) is eta
    jac = jax.jit(lambda e: vertical.compute_ocean_jacobian(e, depth, zc))(eta)
    assert np.array_equal(np.asarray(jac).ravel(), written[pick])
    stretch = jax.jit(lambda e: eos.nemo_r3t_stretch(
        zc, e, depth, evaluation="nemo_reciprocal"))(eta)
    assert np.array_equal(np.asarray(stretch).ravel(), written[pick])
    # review finding: a coordinate passed THROUGH jit carries the flag as a
    # traced leaf; the identity test reads it as off and must not raise
    traced = jax.jit(lambda c: vertical.compute_ocean_jacobian(eta, depth, c))(zc)
    assert np.allclose(np.asarray(traced).ravel(), written[pick], rtol=1.0e-15, atol=0.0)


@needs_record5
def test_ratio_order_replay_controls_and_the_step_3_coefficients():
    """Offline.  NEMO's ratios in the written order: 0 unequal against its
    record.  The V-face coefficients replayed from NEMO's recorded ratios:
    0 unequal against NEMO's recorded coefficients (step 2).  At step 3 the
    production builder differs on three entries with the option off and on
    none with it on.  The score plant turns the step-2 control red."""
    pytest.importorskip("netCDF4")
    out = ladder.run(RECORD5, arm="ratio_order_replay", kt_min=2, kt_max=3,
                     allow_dirty=True)
    for k in out["per_kt"]:
        assert all(r["written_order_vs_nemo"] == 0 for r in k["ratio_rows"])
        assert any(r["divided_vs_nemo"] > 100 for r in k["ratio_rows"])
    step2, step3 = (k["coefficient_rows"] for k in out["per_kt"])
    assert all(r["replay_vs_nemo_recorded"] == 0 for r in step2)
    assert sum(r["builder_vs_replay"] for r in step3 if not r["written_order_option"]) == 3
    assert sum(r["builder_vs_replay"] for r in step3 if r["written_order_option"]) == 0
    assert ladder.main(["--arm", "ratio_order_replay", "--root", str(RECORD5),
                        "--kt-min", "2", "--kt-max", "2", "--plant", "score",
                        "--allow-dirty"]) == 1


@needs_record5
@pytest.mark.parametrize("off, want", [((), 0), (("without_ratio_order",), 1)])
def test_the_chain_is_nemos_through_step_4_only_with_the_ratio_order(off, want):
    """INDEPENDENT chain: every scored field bit-identical through step 4;
    with the option off (the plant) one barotropic V face after step 3."""
    out = ladder.run(RECORD5, arm="independent", kt_max=4, allow_dirty=True,
                     **{k: True for k in off})
    unequal = {(k["kt"], r["name"].split(".")[-1]): r["n_unequal"]
               for k in out["per_kt"] for r in k["rows"]}
    assert unequal[(3, "vv_b")] == want
    if not want:
        assert not any(unequal.values())


@needs_record5
@pytest.mark.parametrize("off, want", [((), 0), (("without_r3t_product",), 1)])
def test_the_step_19_tracer_cell_needs_the_stored_ratio(off, want):
    """INDEPENDENT chain to step 20: T and S are NEMO's bit for bit, and
    step 20 is bit-identical on every field; with the coordinate flag off
    (the plant) one T cell and one S cell differ after step 19, at
    inner-domain (58, 60)."""
    out = ladder.run(RECORD5, arm="independent", kt_max=20, allow_dirty=True,
                     **{k: True for k in off})
    rows = {(k["kt"], r["name"].split(".")[-1]): r
            for k in out["per_kt"] for r in k["rows"]}
    for f in ("T", "S"):
        assert rows[(19, f)]["n_unequal"] == want
        if want:
            assert rows[(19, f)]["max_abs_cell"] == [58, 60]
    if not want:
        assert not any(r["n_unequal"] for (kt, _), r in rows.items() if kt >= 12)


# ---------------------------------------------------------------- round 9
ROUND9 = ("rk3_stage_r3t_nemo_literal", "nemo_eos_gdept_rounded_product",
          "rk3_stage_r3uv_nemo_literal")


def test_only_the_g1_card_selects_the_round9_statements():
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    from legoesm.ocean.state import LatLonCGridOceanConfig
    assert _run_in_order(LatLonCGridOceanConfig._fields, ROUND9)
    src = Path(recipe.__file__).read_text()
    start = src.index("def build_orca2g1_flat_card")
    for name in ROUND9:
        assert LatLonCGridOceanConfig._field_defaults[name] is False
        assert name not in src[:start], name      # no other card selects it
        assert f"{name}=True" in src[start:] and f"cfg.{name} is True" in src[start:]


def test_stage_ratio_is_nemos_written_interpolation_under_jit():
    """stprk3_stg.f90:171-173 and :215-217: two rounded products and a sum
    at N+1/3, a rounded sum and a product at N+1/2; any other level raises."""
    import jax
    from types import SimpleNamespace as NS

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as M
    rng = np.random.default_rng(9)
    b, a = rng.normal(size=4000) * 1e-3, rng.normal(size=4000) * 1e-3
    third = np.asarray(jax.jit(lambda x, y: M._nemo_stage_ratio(1, x, y))(
        jnp.asarray(b), jnp.asarray(a)))
    half = np.asarray(jax.jit(lambda x, y: M._nemo_stage_ratio(2, x, y))(
        jnp.asarray(b), jnp.asarray(a)))
    assert np.array_equal(third, (2.0 / 3.0) * b + (1.0 / 3.0) * a)
    assert np.array_equal(half, 0.5 * (b + a))
    with pytest.raises(ValueError, match="1 or 2"):
        M._nemo_stage_ratio(3, b, a)
    on, off = NS(rk3_stage_r3uv_nemo_literal=True), NS(rk3_stage_r3uv_nemo_literal=False)
    assert M._nemo_ws_stage_pair(off, 2, (b, a)) is None
    assert M._nemo_ws_stage_pair(on, 0, (b, a)) is None
    assert M._nemo_ws_stage_pair(on, 2, (b, a))[0] == 2
    assert M._nemo_ws_stage_r3uv(None, None, None) is None


def test_stage_t_stretch_is_one_plus_the_mean_of_the_stored_ratios():
    """stprk3_stg.f90:215 (stage 3 reads it) and :171 (stage 2): the handed
    stretch is NEMO's written form; off, at stage 1, or without the
    pressure gradient that reads it, nothing is handed."""
    from types import SimpleNamespace as NS

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as M
    rng = np.random.default_rng(15)
    eb, ea = rng.normal(size=(40, 50)) * 0.3, rng.normal(size=(40, 50)) * 0.3
    depth, zc = jnp.full(eb.shape, 100.0), NS(linear_free_surface=False)
    r1 = 1.0 / 100.0
    on = NS(rk3_stage_r3t_nemo_literal=True, pgf_scheme="nemo_sco")
    pair = lambda: (jnp.asarray(eb), jnp.asarray(ea))  # noqa: E731
    s3 = np.asarray(M._nemo_ws_stage_r3t_stretch(on, 3, pair, zc, depth))
    s2 = np.asarray(M._nemo_ws_stage_r3t_stretch(on, 2, pair, zc, depth))
    assert np.array_equal(s3, 1.0 + 0.5 * (eb * r1 + ea * r1))
    assert np.array_equal(s2, 1.0 + ((2.0 / 3.0) * (eb * r1) + (1.0 / 3.0) * (ea * r1)))
    own = 1.0 + (0.5 * (eb + ea)) * r1          # the ratio of the mean sea level
    assert 0 < np.count_nonzero(s3 != own) < eb.size // 2    # last-place twin
    assert M._nemo_ws_stage_r3t_stretch(on, 1, pair, zc, depth) is None
    assert M._nemo_ws_stage_r3t_stretch(
        NS(rk3_stage_r3t_nemo_literal=False), 3, pair, zc, depth) is None
    with pytest.raises(ValueError, match="nemo_sco"):
        M._nemo_ws_stage_r3t_stretch(
            NS(rk3_stage_r3t_nemo_literal=True, pgf_scheme="adcroft"), 3, pair, zc, depth)


def test_unequal_census_separates_flushed_subnormals_from_other_cells():
    ref = np.array([0.0, 1.0e-310, -3.0e-320, 1.0e-300, 2.0, 5.0e-309])
    cand = np.array([0.0, 0.0, 0.0, 1.1e-300, 2.0, 5.0e-309])
    c = ladder.unequal_census(ref, cand)
    assert (c["n_unequal"], c["nemo_subnormal_card_zero"], c["other"]) == (3, 2, 1)
    assert (c["nemo_subnormal_anywhere"], c["card_subnormal_anywhere"]) == (3, 1)
    assert c["unequal_nemo_abs_min"] == 3.0e-320 and c["unequal_nemo_abs_max"] == 1.0e-300
    assert ladder.unequal_census(ref, ref)["unequal_nemo_abs_min"] is None


def test_xla_flushes_subnormals_and_numpy_keeps_them_on_this_host():
    """The registered environment row: the card's executable returns zero
    for a subnormal result and reads a subnormal operand as zero; numpy in
    the same process does neither.  Red here means the row must be re-read."""
    env = ladder.run(RECORD, arm="subnormal_environment", allow_dirty=True)
    assert env["numpy_keeps"] and env["numpy_before_xla"] == env["numpy_after_xla"]
    assert env["xla_flushes_result"] and env["xla_flushes_operand"]
    assert env["xla_compiled_n4"] == [0.0, 0.0]


@needs_record5
def test_density_depth_is_the_rounded_product_only_with_the_option():
    """eosbn2.f90:324: ``(gdept_1d * (1 + r3t)) * r1_Z0``.  Compiled as one
    graph, the helper's depth times the scale equals the written order on
    every wet cell with the option; bare (the plant), the two constants are
    folded and cells differ."""
    import jax
    from types import SimpleNamespace as NS

    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as P
    from legoesm.ocean.eos import nemo_bn2_depth_ladders
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    ladder._fp64()
    card = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=RECORD5)
    zc, H = card.recipe.z_coord, card.recipe.initial_state.H_bathy.data
    g = ladder.read_step(RECORD5, 40)
    wet = np.asarray(zc.is_active)[..., 0] > 0.5
    ssh, scale = g["2_ssh_aa"], 1.0e-4
    gdept = np.asarray(nemo_bn2_depth_ladders(zc)[0], dtype=np.float64)
    written = ((gdept * (1.0 + ssh / 100.0)[..., None]) * scale)[..., 0]

    def unequal(selected, stretch=None):
        cfg = NS(nemo_eos_gdept_rounded_product=selected)
        zh = jax.jit(lambda e: P._nemo_eos_gdept(cfg, zc, e, H, stretch) * scale)(
            jnp.asarray(ssh))
        return int(np.count_nonzero((np.asarray(zh)[..., 0] - written)[wet]))
    assert unequal(True) == 0
    assert unequal(False) > 1000
    handed = jnp.asarray(1.0 + ssh / 100.0)     # a handed stretch replaces the ratio
    assert unequal(True, handed) == 0


@needs_record5
def test_stage_operands_at_steps_5_and_40_name_the_round9_cells():
    """OFFLINE, NEMO's frames only: the face whose 1 + r3u differs between
    NEMO's interpolated ratio and the ratio of the stage's sea level (step 5,
    stage 3, U face (77, 46)); the density depth cell (step 40, stage 3) that
    does not move the density anomaly."""
    out = ladder.run(RECORD5, arm="stage_operand_replay", kt_min=5, kt_max=5,
                     allow_dirty=True)["per_kt"][0]
    s3 = [r for r in out["face_ratio_rows"] if r["stage"] == 3][0]
    assert s3["u"]["one_plus_n_unequal"] == 1 and s3["u"]["faces"] == [[77, 46]]
    assert s3["v"]["one_plus_n_unequal"] == 0 and s3["u"]["ratio_n_unequal"] > 1000
    out = ladder.run(RECORD5, arm="stage_operand_replay", kt_min=40, kt_max=40,
                     allow_dirty=True)["per_kt"][0]
    d3 = [r for r in out["rows"] if r["stage"] == 3][0]["card_ladder"]
    assert (d3["depth_n_unequal"], d3["rhd_n_unequal"]) == (1, 0)


R9_FIRST_OVER = {(): None, ("without_stage_r3t",): 76, ("without_eos_gdept",): 63,
                 ("without_stage_r3uv",): None}


@needs_record5
@pytest.mark.parametrize("off", list(R9_FIRST_OVER))
def test_the_100_step_record_is_at_the_bar_and_needs_each_round9_statement(off):
    """INDEPENDENT chain, 100 steps.  The card: no row over the bar, no
    unequal cell after step 11, and steps 8 to 11 hold only the registered
    tiny-number class (NEMO subnormal, the card zero, and their descendants
    below 1e-280).  Each option off is the plant: the sea level leaves the
    bar at the pinned step, or (the U and V ratios) cells differ past step 11
    while staying at the bar."""
    out = ladder.run(RECORD5, arm="independent", kt_max=100, allow_dirty=True,
                     **{k: True for k in off})
    rows = [(k["kt"], r) for k in out["per_kt"] for r in k["rows"]]
    over = [kt for kt, r in rows if r["status"] != "AT-BAR"]
    late = [kt for kt, r in rows if kt > 11 and r["n_unequal"]]
    assert (min(over) if over else None) == R9_FIRST_OVER[off]
    if off:
        assert late                               # the plant moved cells
        return
    assert not late
    early = [(kt, r) for kt, r in rows if r["n_unequal"]]
    assert {kt for kt, _ in early} == {8, 9, 10, 11}
    assert all(r["max_abs"] < 1.0e-280 for _, r in early)
    assert all(r["unequal_census"]["card_subnormal_anywhere"] == 0 for _, r in rows)
    first = [r for kt, r in early if kt == 8]
    assert all(r["unequal_census"]["other"] == 0 for r in first)


@needs_record5
@pytest.mark.parametrize("off, want", [((), []), (("without_stage_r3uv",), [[77, 46], [77, 47]])])
def test_step_5_stage_3_vorticity_faces_need_the_interpolated_u_ratio(off, want):
    """GIVEN-NEMO-ENTRY, stage 3 of step 5, split by term (the model's own
    operator hook; the terms' sum against the exposed right-hand side is the
    split's control): the pressure gradient is NEMO's either way; the
    vorticity term differs on two V faces without the option (the plant)."""
    out = ladder.run(RECORD5, arm="stages", kt_min=5, kt_max=5, allow_dirty=True,
                     **{k: True for k in off})
    v = [r for r in out["per_kt"][0]["stages"][2]["stage_replay"]["rows"]
         if r["face"] == "v"][0]
    assert v["control_card_terms_sum_vs_card_rhs"]["n_unequal"] == 0
    assert v["control_replay_vs_nemo"]["n_unequal"] == 0
    assert (v["unequal_faces_hpg"], v["unequal_faces_vor"]) == ([], want)
