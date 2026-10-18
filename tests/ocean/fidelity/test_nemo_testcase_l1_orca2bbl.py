"""Lane B rung L1: certified G5 plus ORCA2's bottom boundary layer (ln_trabbl)."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import jax
import netCDF4
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES, CASES, INPUTS, ORCA2_DECK, _differs, _flat, _root, _values,
)
from tests.ocean.fidelity.test_nemo_testcase_v3_orca2ddm import (
    assert_record_bit_for_bit, kt2_restarts_differ,
)

G5 = CASES / "nemo_testcase_g5_orca2ice"
L1 = CASES / "nemo_testcase_l1_orca2bbl"
RUN = L1 / "run.sh"
BBL = {"nambbl.ln_trabbl": (False, True), "nambbl.nn_bbl_ldf": (None, 1),
       "nambbl.nn_bbl_adv": (None, 0), "nambbl.rn_ahtbbl": (None, 1000.0),
       "nambbl.rn_gambbl": (None, 10.0)}


def _checker():
    spec = importlib.util.spec_from_file_location("l1_records", L1 / "l1_records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def test_l1_deck_is_g5_plus_orca2s_five_nambbl_lines_only():
    g5, l1 = _values(G5 / "namelist_cfg"), _values(L1 / "namelist_cfg")
    changed = {k: (g5.get(k), l1.get(k)) for k in set(g5) | set(l1) if g5.get(k) != l1.get(k)}
    assert changed == BBL
    assert (G5 / "namelist_ice_cfg").read_bytes() == (L1 / "namelist_ice_cfg").read_bytes()


@pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
def test_l1_nambbl_lines_are_orca2s_own():
    orca2 = _values(ORCA2_DECK)
    assert {k: orca2[k] for k in recipe._ORCA2L1_BBL_FIELDS} == recipe._ORCA2L1_BBL_FIELDS
    for k, v in recipe._ORCA2L1_BBL_FIELDS.items():
        assert type(orca2[k]) is type(v), k


def test_reader_refuses_bbl_on_the_wrong_rung_and_a_planted_line(tmp_path):
    with pytest.raises(ValueError, match="nambbl"):
        recipe.read_orca2g1_deck(L1, "ORCA2G5-ice")
    with pytest.raises(ValueError, match="nambbl"):
        recipe.read_orca2g1_deck(G5, "ORCA2L1-bbl")
    planted = tmp_path / "planted"
    planted.mkdir()
    text = (G5 / "namelist_cfg").read_text().replace(
        "   ln_trabbl   = .false.\n", "   ln_trabbl   = .false.\n   rn_ahtbbl   = 1000.\n")
    (planted / "namelist_cfg").write_text(text)
    with pytest.raises(ValueError, match="nambbl"):
        recipe.read_orca2g1_deck(planted, "ORCA2G5-ice")
    assert recipe.read_orca2g1_deck(L1, "ORCA2L1-bbl") == recipe.read_orca2g1_deck(
        G5, "ORCA2G5-ice")


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("l1")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        return [(build(root), root) for build, root in (
            (recipe.build_orca2g5_ice_card, _root(base, G5)),
            (recipe.build_orca2l1_bbl_card, _root(base, L1)))]
    finally:
        set_policy(before)


def test_l1_card_is_g5_plus_the_diffusive_bbl_selection_only(cards):
    (g5, _), (l1, _) = cards
    a, b = _flat(g5.recipe.model_config), _flat(l1.recipe.model_config)
    changed = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if _differs(a.get(k), b.get(k))}
    assert changed == {"bbl_diffusive_option": (0, 1), "bbl_aht_m2_s": (0.0, 1000.0),
                       "bbl_gamma_s": (0.0, 10.0), "bbl_nemo_stage3_krhs": (False, True),
                       "bbl_nemo_pivot_ghost": (False, True),
                       "rk3_fold_exchange_tracer_pivot_row": (False, True)}
    assert l1.recipe.model_config.bbl_adv_option == 0
    s5, s1 = g5.recipe.initial_state, l1.recipe.initial_state
    assert [n for n in s5._fields if _differs(
        getattr(getattr(s5, n), "data", getattr(s5, n)),
        getattr(getattr(s1, n), "data", getattr(s1, n)))] == []
    assert (l1.bbl_adv_option, l1.bbl_diffusive_option, l1.bbl_aht_m2_s, l1.bbl_gamma_s) == (
        0, 1, 1000.0, 10.0)
    assert (l1.dt_s, l1.n_steps, l1.unmeasured_features, l1.surface_boundary_condition) == (
        g5.dt_s, g5.n_steps, g5.unmeasured_features, g5.surface_boundary_condition)
    assert recipe._orca2g1_selector_failures(l1) == []
    assert recipe._orca2g1_selector_failures(g5) == []


def test_selector_refuses_l1_without_bbl_and_g5_with_it(cards):
    (g5, _), (l1, _) = cards
    off = l1.recipe.model_config._replace(bbl_diffusive_option=0)
    assert "ln_trabbl" in recipe._orca2g1_selector_failures(
        l1._replace(recipe=l1.recipe._replace(model_config=off)))
    for flag in ("bbl_nemo_stage3_krhs", "bbl_nemo_pivot_ghost",
                 "rk3_fold_exchange_tracer_pivot_row"):
        cut = l1.recipe.model_config._replace(**{flag: False})
        assert "ln_trabbl" in recipe._orca2g1_selector_failures(
            l1._replace(recipe=l1.recipe._replace(model_config=cut))), flag
    meta = l1._replace(bbl_aht_m2_s=500.0)        # card metadata disagreeing with config
    assert "ln_trabbl" in recipe._orca2g1_selector_failures(meta)
    on = g5.recipe.model_config._replace(bbl_diffusive_option=1, bbl_aht_m2_s=1000.0)
    assert "ln_trabbl=F" in recipe._orca2g1_selector_failures(
        g5._replace(recipe=g5.recipe._replace(model_config=on)))


def test_bbl_footprint_takes_step_faces_cyclic_and_fold_rows_only():
    mod = _checker()
    mb = np.array([[0, 3, 3, 3, 5],
                   [3, 3, 3, 3, 3],
                   [3, 3, 3, 3, 3],
                   [4, 3, 3, 3, 3],
                   [3, 3, 3, 3, 3],
                   [3, 3, 3, 3, 3]])
    fp = mod.bbl_footprint(mb)
    assert fp[0].tolist() == [False, False, False, True, True]   # land never; step 3|5
    assert fp[1].tolist() == [False] * 4 + [True]                 # under the 5 (v face)
    assert fp[2].tolist() == [True] + [False] * 4                 # above the 4 (v face)
    assert fp[3].tolist() == [True, True, False, False, True]     # 4 vs 3, cyclic neighbour
    assert fp[4:].all()                                           # fold rows wholesale


def test_bbl_entry_check_passes_only_bbl_shaped_changes(monkeypatch):
    mod = _checker()
    mb = np.full((6, 5), 3)
    mb[2, 2] = 4                       # one bottom step: columns around (2, 2)
    base = {"step": 2, "T": np.ones((6, 5, 4)), "S": np.ones((6, 5, 4)),
            "u": np.zeros((6, 5, 4)), "v": np.zeros((6, 5, 4)), "ssh": np.zeros((6, 5))}

    def run(edit):
        cand = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        edit(cand)
        monkeypatch.setattr(mod, "read_entry",
                            lambda path, case: cand if case == "ORCA2L1-bbl" else base)
        return mod.check_bbl_entry("l1", "g5", mb)

    assert run(lambda c: None)["pass"] is False                         # BBL inert
    assert run(lambda c: c["T"].__setitem__((2, 1, 3), 2.0))["pass"] is True
    assert run(lambda c: c["S"].__setitem__((0, 0, 0), 2.0))["pass"] is False   # outside
    res = run(lambda c: (c["T"].__setitem__((2, 2, 0), 2.0), c["u"].__setitem__((0, 0, 0), 1.0)))
    assert res["pass"] is False and res["u_unequal"] == 1                   # dynamics moved
    assert run(lambda c: (c["T"].__setitem__((2, 2, 0), 2.0),
                          c["ssh"].__setitem__((0, 0), 1e-9)))["pass"] is False


def test_run_script_reuses_g5_binaries_unbuilt_and_is_fail_closed():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    assert "makenemo" not in text
    assert "ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/bin/nemo.exe" in text
    assert "ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe" in text
    assert "__L1_DECK_SHA__" not in text
    for name in ("for m in trabbl.o trabbc.o", "--only-section=$sec", "L1_DECK_SHA", "G5_ENTRY_SHA",
                 "nn_bbl_ldf = +1$", "nn_bbl_adv = +0$", "rn_ahtbbl += +1000", 'check_bbl_entry "$SMOKE"',
                 'check_bbl_entry "$TWIN_A"', "ln_trabbc += +F$", "commit the exact acquisition tool"):
        assert name in text, name


G5_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_rounds/round31/oracle_g5_r31")
L1_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                 "gridladder_vmix_rounds/round16/oracle_l1_r16")
L1_TWINS = [L1_RECORD / t for t in ("twin_a", "twin_b", "twin_c")]


@pytest.mark.skipif(not jax.config.x64_enabled, reason="binary64 receipt")
@pytest.mark.skipif(not all(t.is_dir() for t in (*L1_TWINS, G5_RECORD / "twin_b")),
                    reason="the operator's G5/L1 records are not on this machine")
def test_l1_kt1_4_every_field_bit_for_bit_and_bbl_acts_at_stage3_in_footprint_only():
    """NEMO runs bbl/tra_bbl at stage 3 only (stprk3_stg.f90:560, :739); kt 4
    is the first step the BBL depth gdept_1d*(1+r3t(Kmm)) decides a gate."""
    assert kt2_restarts_differ(G5_RECORD, L1_RECORD)   # the BBL moves kt-2 T
    with netCDF4.Dataset(_checker().DOMAIN) as ds:
        mb = np.squeeze(np.asarray(ds.variables["bottom_level"][:]))
    stages = _checker().check_bbl_stages(L1_TWINS[0], G5_RECORD / "twin_a", mb)
    assert stages["pass"], stages
    assert assert_record_bit_for_bit(L1_TWINS, "ORCA2L1-bbl", kt_to=4, frame=False) is None


def test_tracer_pivot_row_exchange_copies_the_image_half_from_its_mirror():
    """lbcnfd.f90:620-628: points ipi/2+2..ipi-ihls of the T-pivot row take
    the mirror's value (sign +1); the source half and every other row stay."""
    from types import SimpleNamespace

    import jax.numpy as jnp

    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as M
    n = 8
    perm = np.array([0] + [n - i for i in range(1, n)])          # 0 and n/2 self-mirrored
    grid = SimpleNamespace(fold=SimpleNamespace(is_active=True, pivot_row_stored=True,
                                                perm_T=perm))
    x = np.arange(3 * n * 2, dtype=np.float64).reshape(3, n, 2)
    got = np.asarray(M._rk3_fold_exchange_tracer_pivot_row(jnp.asarray(x), grid))
    want = x.copy()
    want[-1, n // 2 + 1:] = x[-1, perm[n // 2 + 1:]]
    assert np.array_equal(got, want)
    assert np.array_equal(got[-1, : n // 2 + 1], x[-1, : n // 2 + 1])
    assert np.array_equal(got[-1], got[-1, perm])                # the row is now symmetric
    for fold in (None, SimpleNamespace(is_active=True, pivot_row_stored=False, perm_T=perm)):
        with pytest.raises(ValueError, match="T-pivot"):
            M._rk3_fold_exchange_tracer_pivot_row(jnp.asarray(x), SimpleNamespace(fold=fold))


def test_bbl_north_neighbour_of_the_pivot_row_is_the_mirror_of_the_row_below():
    """lbcnfd.f90:588-590: the ghost north of the stored T-pivot row takes
    row ipj-2*ihls+jj-1 (the row below the pivot) through the fold map."""
    from types import SimpleNamespace

    import jax.numpy as jnp

    from legoesm.ocean.physics import bbl_adv
    n = 8
    perm = np.array([0] + [n - i for i in range(1, n)])
    grid = SimpleNamespace(fold=SimpleNamespace(is_active=True, pivot_row_stored=True,
                                                perm_T=perm))
    x = np.arange(4 * n, dtype=np.float64).reshape(4, n)
    got = np.asarray(jax.jit(lambda v: bbl_adv._north_cell(v, grid, True))(jnp.asarray(x)))
    assert np.array_equal(got[:-1], x[1:])
    assert np.array_equal(got[-1], x[-2, perm])
    assert not np.array_equal(got[-1], x[-1, perm])           # not the pivot row's own mirror
    legacy = np.asarray(bbl_adv._north_cell(jnp.asarray(x), grid))
    assert np.array_equal(legacy[-1], x[-1, perm])
    with pytest.raises(ValueError, match="pivot"):
        bbl_adv._north_cell(jnp.asarray(x), SimpleNamespace(fold=SimpleNamespace(
            is_active=True, pivot_row_stored=False, perm_T=perm)), True)
