"""Lane B L2: certified G5 plus ORCA2's standard isoneutral diffusion."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

from tests.ocean.fidelity.test_nemo_testcase_v1_orca2tke import (
    CARD_FILES,
    CASES,
    G5,
    INPUTS,
    ORCA2_DECK,
    _differs,
    _flat,
    _root,
    _values,
)

L2 = CASES / "nemo_testcase_l2_orca2iso"
RUN = L2 / "run.sh"


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DECK = _module("l2_deck", L2 / "l2_deck.py")
RECORDS = _module("l2_records", L2 / "l2_records.py")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def _l2_root(base: Path) -> Path:
    root = _root(base, G5, "l2")
    (root / "namelist_cfg").write_text(DECK.render_text(
        (root / "namelist_cfg").read_text()))
    return root


def test_l2_deck_is_g5_plus_the_complete_resolved_namtra_ldf_group_only(tmp_path):
    root = _l2_root(tmp_path)
    g5, l2 = _values(G5 / "namelist_cfg"), _values(root / "namelist_cfg")
    changed = {k: (g5.get(k), l2.get(k)) for k in set(g5) | set(l2)
               if g5.get(k) != l2.get(k)}
    assert changed == {k: (g5.get(k), value)
                       for k, value in recipe._ORCA2L2_LDF_FIELDS.items()
                       if g5.get(k) != value}
    assert {k: v for k, v in l2.items() if k.startswith("namtra_ldf.")} == (
        recipe._ORCA2L2_LDF_FIELDS)
    assert l2["namtra_eiv.ln_ldfeiv"] is False
    with pytest.raises(ValueError, match="certified G5"):
        DECK.render_text(DECK.render_text((G5 / "namelist_cfg").read_text()))


@pytest.mark.skipif(not ORCA2_DECK.is_file(), reason="no NEMO tree")
def test_l2_namtra_ldf_values_resolve_from_orca2_cfg_plus_reference():
    cfg = _values(ORCA2_DECK)
    ref = _values(ORCA2_DECK.with_name("namelist_ref"))
    resolved = {key: cfg.get(key, ref[key]) for key in recipe._ORCA2L2_LDF_FIELDS}
    assert resolved == recipe._ORCA2L2_LDF_FIELDS
    # Full ORCA2 enables EIV; L2 deliberately does not copy that later rung.
    assert cfg["namtra_eiv.ln_ldfeiv"] is True


def test_reader_refuses_l2_group_on_g5_and_off_group_on_l2(tmp_path):
    root = _l2_root(tmp_path)
    with pytest.raises(ValueError, match="namtra_ldf"):
        recipe.read_orca2g1_deck(root, "ORCA2G5-ice")
    with pytest.raises(ValueError, match="namtra_ldf"):
        recipe.read_orca2g1_deck(G5, "ORCA2L2-iso")
    text = (root / "namelist_cfg").read_text()
    (root / "namelist_cfg").write_text(text.replace("rn_sw_triad     = 1", "rn_sw_triad     = 0"))
    with pytest.raises(ValueError, match="namtra_ldf"):
        recipe.read_orca2g1_deck(root, "ORCA2L2-iso")


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    if not all((INPUTS / name).is_file() for name in CARD_FILES):
        pytest.skip("no ORCA2 inputs")
    base = tmp_path_factory.mktemp("l2")
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        return (
            recipe.build_orca2g5_ice_card(_root(base, G5, "g5")),
            recipe.build_orca2l2_iso_card(_l2_root(base)),
        )
    finally:
        set_policy(before)


def test_l2_card_is_g5_plus_shared_isoneutral_config_only(cards):
    g5, l2 = cards
    a, b = _flat(g5.recipe.model_config), _flat(l2.recipe.model_config)
    changed = {k for k in set(a) | set(b) if _differs(a.get(k), b.get(k))}
    assert changed and all(k == "gm_redi" or k.startswith("gm_redi.") for k in changed)
    assert recipe._orca2l2_isoneutral_selected(l2.recipe.model_config.gm_redi)
    assert l2.recipe.model_config.gm_redi == recipe._orca2_isoneutral_gm_redi(
        0.5 * recipe._SMT3_RN_UD * recipe._SMT3_RN_LD)
    assert l2.recipe.model_config.gm_redi.kappa_GM == 0.0
    assert l2.recipe.model_config.tracer_advection == "none"
    assert l2.recipe.model_config.bbl_diffusive_option == 0
    s5, s2 = g5.recipe.initial_state, l2.recipe.initial_state
    assert [name for name in s5._fields if _differs(
        getattr(getattr(s5, name), "data", getattr(s5, name)),
        getattr(getattr(s2, name), "data", getattr(s2, name)))] == []
    assert recipe._orca2g1_selector_failures(l2) == []
    assert recipe._orca2g1_selector_failures(g5) == []


def test_selector_refuses_missing_msc_metric_mode_or_nonzero_eiv(cards):
    _, l2 = cards
    gm = l2.recipe.model_config.gm_redi
    for change in ({"msc_stabilize": False},
                   {"kappa_redi_horizontal_evaluation": "cosine_scaled"},
                   {"kappa_GM": 1.0}):
        bad = l2.recipe.model_config._replace(gm_redi=gm._replace(**change))
        failures = recipe._orca2g1_selector_failures(
            l2._replace(recipe=l2.recipe._replace(model_config=bad)))
        assert "ln_traldf_lap+iso+msc" in failures or "ln_ldfeiv=F" in failures


def test_shared_constructor_keeps_vortex_smt3_selection_unchanged():
    smt3 = recipe.build_vortex_smt_zps_card("vector", "smt3")
    assert smt3.recipe.model_config.gm_redi == recipe._orca2_isoneutral_gm_redi(270.0)


def test_acting_checker_requires_wet_ldf_and_tracer_only_entry(monkeypatch, tmp_path):
    shape = (2, 3, 2)
    zero = np.zeros(shape)
    base_stage = {f"{where}_{tr}": zero.copy()
                  for where in ("before_ldf", "after_ldf", "Kaa")
                  for tr in ("T", "S")}
    l2_stage = {k: v.copy() for k, v in base_stage.items()}
    l2_stage["after_ldf_T"][0, 1, 0] = 1.0
    l2_stage["Kaa_T"][0, 1, 0] = 1.0
    entry0 = {"T": zero.copy(), "S": zero.copy(), "u": zero.copy(),
              "v": zero.copy(), "ssh": np.zeros(shape[:2])}
    entry2 = {k: v.copy() for k, v in entry0.items()}
    entry2["T"][0, 1, 0] = 1.0
    monkeypatch.setattr(RECORDS, "read_tracer_stage3",
                        lambda path: l2_stage if "l2" in str(path) else base_stage)
    monkeypatch.setattr(RECORDS, "read_entry",
                        lambda path, case: entry2 if case == "ORCA2L2-iso" else entry0)

    class Var:
        def __getitem__(self, key):
            return np.array([[2, 2, 0], [2, 2, 2]])

    class Dataset:
        variables = {"bottom_level": Var()}
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(RECORDS.netCDF4, "Dataset", lambda path: Dataset())
    result = RECORDS.check_l2(Path("l2"), Path("g5"), tmp_path / "domain")
    assert result["pass"] and result["ldf_wet_nonzero"] == 1
    l2_stage["after_ldf_T"][0, 2, 0] = 1.0                 # dry-column plant
    assert not RECORDS.check_l2(Path("l2"), Path("g5"), tmp_path / "domain")["pass"]


def test_run_script_reuses_g5_binaries_and_is_fail_closed():
    text = RUN.read_text()
    assert subprocess.run(["bash", "-n", str(RUN)]).returncode == 0
    assert "makenemo" not in text
    for token in (
        "ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/bin/nemo.exe",
        "ORCA2G5_OMIP_L4_R31REF_FIX1/BLD/bin/nemo.exe",
        "for m in ldfslp.o ldftra.o ldfc1d_c2d.o traldf.o traldf_iso.o trazdf.o",
        "L2_DECK_SHA", "G5_STAGE_SHA", 'check_acting "$SMOKE"',
        'check_acting "$TWIN_A"', "ln_traldf_msc += +T$", "ln_ldfeiv += +F$",
        "commit the exact acquisition tool before running it",
    ):
        assert token in text, token
