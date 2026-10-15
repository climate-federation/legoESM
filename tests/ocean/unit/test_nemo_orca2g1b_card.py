"""The ORCA2G1b card (grid ladder rung G1b): G1 plus ORCA2's vector form.

* FILE-FREE: the committed G1b deck is G1's with exactly the &namdyn_adv
  lines ORCA2 states; the reader refuses each moved or misplaced switch.
* ON G1'S ADMITTED DOMAIN FILE (skipped off this machine): the G1b card and
  the G1 card differ in the momentum-advection selectors and the three
  flux-form-only options NEMO's vector deck does not execute, and in
  nothing else; each card refuses the other rung's deck.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

ROOT = Path(__file__).parents[3]
CASES = ROOT / "scripts/validate/ocean_fidelity/testcases"
G1B = CASES / "nemo_testcase_g1b_orca2vec/namelist_cfg"
G1 = CASES / "nemo_testcase_g1_orca2grid/namelist_cfg"
ORCA2 = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg")
DOMAIN = Path(os.environ.get(
    "LEGOESM_ORCA2G1_DOMAIN",
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round4/"
    "oracle_g1_r4/domaincfg/domain_cfg.nc"))


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(before)


def _deck(tmp_path: Path, source: Path, old: str | None = None, new: str = "") -> Path:
    text = source.read_text()
    if old is not None:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    (tmp_path / "namelist_cfg").write_text(text)
    return tmp_path


def test_g1b_deck_is_g1_plus_the_namdyn_adv_lines():
    a = recipe._orca2g1_namelist_values(G1)
    b = recipe._orca2g1_namelist_values(G1B)
    moved = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert moved == {"namdyn_adv.ln_dynadv_off", "namdyn_adv.ln_dynadv_vec",
                     "namdyn_adv.nn_dynkeg"}
    assert (b["namdyn_adv.ln_dynadv_off"], b["namdyn_adv.ln_dynadv_vec"],
            b["namdyn_adv.nn_dynkeg"]) == (False, True, 0)


@pytest.mark.skipif(not ORCA2.is_file(), reason="ORCA2's deck is not on this machine")
def test_g1b_namdyn_adv_is_orca2s():
    orca2 = recipe._orca2g1_namelist_values(ORCA2)
    b = recipe._orca2g1_namelist_values(G1B)
    assert orca2["namdyn_adv.ln_dynadv_vec"] is True and orca2["namdyn_adv.nn_dynkeg"] == 0
    assert {k: v for k, v in orca2.items() if k.startswith("namdyn_adv.")} == {
        "namdyn_adv.ln_dynadv_vec": b["namdyn_adv.ln_dynadv_vec"],
        "namdyn_adv.nn_dynkeg": b["namdyn_adv.nn_dynkeg"]}
    # the other namelist_ref-defaulted choices ORCA2 leaves alone, G1b states
    assert "namzdf.ln_zad_aimp" not in orca2 and b["namzdf.ln_zad_aimp"] is False
    assert orca2["namdyn_vor.ln_dynvor_een"] is True and b["namdyn_vor.ln_dynvor_een"] is True


def test_g1b_deck_reads(tmp_path):
    deck = recipe.read_orca2g1_deck(_deck(tmp_path, G1B))
    assert deck.rn_Dt_s == 10800.0 and deck.nn_itend == 100


@pytest.mark.parametrize("old,new,match", [
    ("   ln_dynadv_OFF  = .false.", "   ln_dynadv_OFF  = .true.", "ln_dynadv_off"),
    ("      nn_dynkeg   = 0 ", "      nn_dynkeg   = 1 ", "nn_dynkeg"),
    ("      nn_dynkeg   = 0 ", "      !", "does not state namdyn_adv.nn_dynkeg"),
])
def test_g1b_reader_refuses_moved_switches(tmp_path, old, new, match):
    with pytest.raises(ValueError, match=re.escape(match)):
        recipe.read_orca2g1_deck(_deck(tmp_path, G1B, old, new))


def test_nn_dynkeg_without_the_vector_form_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nn_dynkeg without the vector form"):
        recipe.read_orca2g1_deck(_deck(
            tmp_path, G1, "   ln_dynadv_vec  = .false.\n",
            "   ln_dynadv_vec  = .false.\n   nn_dynkeg = 0\n"))


@pytest.fixture(scope="module")
def roots(tmp_path_factory):
    pytest.importorskip("netCDF4")
    if not DOMAIN.is_file():
        pytest.skip(f"G1's admitted domain file is not on this machine: {DOMAIN}")
    out = {}
    for name, deck in (("g1", G1), ("g1b", G1B)):
        root = tmp_path_factory.mktemp(name)
        (root / "namelist_cfg").write_text(deck.read_text())
        (root / "domain_cfg.nc").symlink_to(DOMAIN)
        out[name] = root
    return out


def test_g1b_card_differs_from_g1_only_by_nemos_vector_branch(roots):
    a = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=roots["g1"])
    b = recipe.build_nemo_testcase_card("ORCA2G1b-vec", deck_root=roots["g1b"])
    ca, cb = a.recipe.model_config, b.recipe.model_config
    moved = {f: (getattr(ca, f), getattr(cb, f)) for f in ca._fields
             if repr(getattr(ca, f)) != repr(getattr(cb, f)) and f != "barotropic"}
    assert moved == {
        "momentum_advection": ("flux_form", "vector_invariant"),
        "vorticity_scheme": ("een_planetary", "een_total"),
        "momentum_flux_scheme": ("none", "upwind"),
        "vertical_momentum_scheme": ("none", "nemo_advective"),
        "rk3_stage1_rhs_nemo_literal": (True, False),   # flux-form only (state.py)
        "een_nemo_relative_vorticity": (False, True),   # dynvor.f90:744-748
        "rk3_vector_rhs_nemo_order": (False, True),     # stp2d.f90:138, :144, :161
    }
    ba, bb = ca.barotropic, cb.barotropic
    moved_bt = {f for f in ba._fields if repr(getattr(ba, f)) != repr(getattr(bb, f))}
    assert moved_bt == {"barotropic_flux_update_nemo_literal",
                        "barotropic_entry_depth_nemo_literal",
                        "barotropic_literal_fold_unit_vector_form"}
    assert bb.barotropic_literal_fold_unit_vector_form is True
    assert (bb.barotropic_flux_update_nemo_literal,
            bb.barotropic_entry_depth_nemo_literal) == (False, False)
    assert cb.ke_gradient_scheme == "c2" and not cb.adaptive_implicit_vertadv
    assert b.unmeasured_features == recipe.ORCA2G1B_UNMEASURED
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(b, roots["g1b"])
    # recipe state, geometry and time step: the same objects' values
    assert (a.dt_s, a.n_steps) == (b.dt_s, b.n_steps)
    assert repr(a.recipe.initial_state.eta.data.tolist()) == repr(
        b.recipe.initial_state.eta.data.tolist())


@pytest.mark.parametrize("case,deck", [("ORCA2G1b-vec", "g1"), ("ORCA2G1-flat", "g1b")])
def test_each_rung_refuses_the_other_deck(roots, case, deck):
    with pytest.raises(ValueError, match="ln_dynadv_vec"):
        recipe.build_nemo_testcase_card(case, deck_root=roots[deck])


def test_g1b_selector_plants_are_refused(roots):
    card = recipe.build_nemo_testcase_card("ORCA2G1b-vec", deck_root=roots["g1b"])
    cfg = card.recipe.model_config
    for change in ({"vorticity_scheme": "een_planetary"},
                   {"momentum_advection": "flux_form"},
                   {"rk3_stage1_rhs_nemo_literal": True},
                   {"barotropic": cfg.barotropic._replace(
                       barotropic_flux_update_nemo_literal=True)},
                   {"een_nemo_relative_vorticity": False},
                   {"rk3_vector_rhs_nemo_order": False},
                   {"barotropic": cfg.barotropic._replace(
                       barotropic_literal_fold_unit_vector_form=False)}):
        planted = card._replace(recipe=card.recipe._replace(
            model_config=cfg._replace(**change)))
        with pytest.raises(ValueError, match="disagrees with its deck"):
            recipe.validate_orca2g1_flat_card(planted, roots["g1b"])


def test_g1b_fold_unit_predicate_admits_vector_form_only_when_selected(roots):
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_literal_external_mode_active)
    cfg = recipe.build_nemo_testcase_card(
        "ORCA2G1b-vec", deck_root=roots["g1b"]).recipe.model_config
    assert nemo_literal_external_mode_active(cfg)
    off = cfg._replace(barotropic=cfg.barotropic._replace(
        barotropic_literal_fold_unit_vector_form=False))
    assert not nemo_literal_external_mode_active(off)     # plant: the old gate
    g1 = recipe.build_nemo_testcase_card(
        "ORCA2G1-flat", deck_root=roots["g1"]).recipe.model_config
    assert nemo_literal_external_mode_active(g1)          # flux form unchanged


def test_nemo_f_vertex_area_is_e1f_times_e2f(roots):
    import numpy as np
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import _nemo_f_vertex_area
    card = recipe.build_nemo_testcase_card("ORCA2G1b-vec", deck_root=roots["g1b"])
    grid, z = card.recipe.grid, card.recipe.z_coord
    raw = z.nemo_een_barotropic
    area = np.asarray(_nemo_f_vertex_area(grid, z))
    e1e2f = np.asarray(raw.e1f) * np.asarray(raw.e2f)
    assert np.array_equal(area[1:, 1:], e1e2f)
    assert np.array_equal(area[1:, 0], e1e2f[:, -1])       # cyclic west column
    assert np.array_equal(area[0], np.asarray(grid.area_q)[0])
    # the grid's own F area is not NEMO's on this mesh (the defect it fixes)
    assert not np.array_equal(np.asarray(grid.area_q)[1:, 1:], e1e2f)
    with pytest.raises(ValueError, match="e1f and e2f"):
        _nemo_f_vertex_area(grid, z._replace(nemo_een_barotropic=None))


def test_g1b_step_routes_the_vector_rhs_order_into_every_stage(roots, monkeypatch):
    """stp2d.f90:138/:144/:161 and stprk3_stg.f90 stages 2-3: the card's
    option reaches every stage's tendencies call; off, none of them."""
    from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as om
    card = recipe.build_nemo_testcase_card("ORCA2G1b-vec", deck_root=roots["g1b"])
    seen = []
    real = om.LatLonCGridOceanModel.tendencies

    def spy(self, *a, **kw):
        seen.append(kw.get("nemo_operator_association", False))
        return real(self, *a, **kw)

    monkeypatch.setattr(om.LatLonCGridOceanModel, "tendencies", spy)
    cfg = card.recipe.model_config
    for on in (True, False):
        seen.clear()
        m = om.LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                     cfg._replace(rk3_vector_rhs_nemo_order=on))
        m.step(card.recipe.initial_state, dt=card.dt_s)
        assert len(seen) >= 3
        assert all(bool(s) is on for s in seen), seen
