"""Grid ladder rung G2 (DECISION 117): the certified G1 card and deck with ONE
switch, the domain (ORCA2's shipped 31-level partial-cell file).

Deck and acquisition-script contracts need no data.  The card tests read
ORCA2's shipped domain file and skip when it is absent.  Nothing here compares
with a NEMO record: none exists yet.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

REPO = Path(__file__).resolve().parents[3]
CASES = REPO / "scripts/validate/ocean_fidelity/testcases"
G2_DIR = CASES / "nemo_testcase_g2_orca2bathy"
G1_DIR = CASES / "nemo_testcase_g1_orca2grid"
SHIPPED = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/"
               "ORCA_R2_zps_domcfg.nc")
SHIPPED_SHA = "7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839"
G1_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/"
                 "round4/oracle_g1_r4/p3_a_100step")
needs_shipped = pytest.mark.skipif(not SHIPPED.is_file(), reason="no shipped ORCA2 domain file")


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(before)


def test_g2_deck_assigns_exactly_what_the_certified_g1_deck_assigns(tmp_path):
    g1 = recipe._orca2g1_namelist_values(G1_DIR / "namelist_cfg")
    g2 = recipe._orca2g1_namelist_values(G2_DIR / "namelist_cfg")
    assert g2 == g1 and len(g1) == 129
    assert recipe.read_orca2g1_deck(G2_DIR) == recipe.read_orca2g1_deck(G1_DIR)
    # plant: one moved assignment is seen
    moved = tmp_path / "namelist_cfg"
    moved.write_text((G2_DIR / "namelist_cfg").read_text().replace(
        "rn_avm0     =   1.2e-4", "rn_avm0     =   1.3e-4"))
    assert recipe._orca2g1_namelist_values(moved) != g1


def test_run_sh_reuses_the_round4_binaries_and_states_the_one_switch():
    text = (G2_DIR / "run.sh").read_text()
    assert subprocess.run(["bash", "-n", str(G2_DIR / "run.sh")]).returncode == 0
    assert SHIPPED_SHA in text and str(SHIPPED) in text
    code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r"makenemo|maketools|\bmake\b", code)      # no build
    # admission judges provenance and the frames' field lists, not only the checker
    assert "was not run with its pinned binary" in code
    assert "a stage step lacks a field the writer writes" in code
    assert "cmp -s <(assignments \"$DECK\") <(assignments \"$G1_DECK\")" in text
    manifest = G1_RECORD.parent / "manifest/binaries.sha256"
    if manifest.is_file():      # the pinned binaries are the certified record's
        pinned = dict(re.findall(r"readonly (REF|RUN)_SHA=([0-9a-f]{64})", text))
        recorded = {("REF" if "R4REF" in line else "RUN"): line.split()[0]
                    for line in manifest.read_text().splitlines()}
        assert pinned == recorded


def _strait_free_copy(tmp_path: Path) -> Path:
    import netCDF4
    path = tmp_path / "no_strait.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("x", 3)
        ds.createVariable("e1t", "f8", ("x",))[:] = 1.0
    return path


def test_strait_shlat_is_the_files_or_no_override(tmp_path):
    none = recipe._orca2g_strait_shlat(_strait_free_copy(tmp_path), (2, 3))
    assert none.shape == (2, 3) and (none == -1.0).all()
    if SHIPPED.is_file():
        got = recipe._orca2g_strait_shlat(SHIPPED, (148, 180))
        assert sorted(np.unique(got).tolist()) == [-1.0, 0.5, 1.0]


@pytest.fixture(scope="module")
def g2_root(tmp_path_factory):
    if not SHIPPED.is_file():
        pytest.skip("no shipped ORCA2 domain file")
    assert hashlib.sha256(SHIPPED.read_bytes()).hexdigest() == SHIPPED_SHA
    root = tmp_path_factory.mktemp("g2_deck")
    (root / "namelist_cfg").write_text((G2_DIR / "namelist_cfg").read_text())
    (root / "domain_cfg.nc").symlink_to(SHIPPED)
    return root


@pytest.fixture(scope="module")
def g2_card(g2_root):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return recipe.build_nemo_testcase_card("ORCA2G2-zps", deck_root=g2_root)


@needs_shipped
def test_g2_card_is_the_g1_card_on_thirty_partial_cell_levels(g2_card, g2_root):
    cfg, state = g2_card.recipe.model_config, g2_card.recipe.initial_state
    assert g2_card.case == "ORCA2G2-zps"
    assert (g2_card.dt_s, g2_card.n_steps) == (10800.0, 100)
    assert np.shape(state.T.data) == (148, 180, 30)
    assert recipe._orca2g1_selector_failures(g2_card) == []     # every G1 selector
    # NEMO's own ORCA2 log: nn_e = 65, Courant number 0.7980 (same file, same rn_Dt)
    assert cfg.barotropic.n_barotropic_substeps == 65
    assert g2_card.unmeasured_features == recipe.ORCA2G2_UNMEASURED
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(g2_card, g2_root)
    eta = np.asarray(state.eta.data)
    assert eta[73, 50] == 0.1 and int((eta != 0).sum()) == 201
    h = np.asarray(g2_card.recipe.z_coord.h_partial)
    bottom = np.asarray(g2_card.recipe.z_coord.bottom_level)
    assert int(bottom.max()) == 29 and float(h.sum(-1)[73, 50]) == pytest.approx(5312.0)
    assert float(h[h > 0].min()) < 2.0          # ORCA2's thinnest partial cell


@needs_shipped
def test_substep_count_needs_the_masked_column_depth(g2_root):
    deck = recipe.read_orca2g1_deck(g2_root)
    dom = recipe._orca2g1_domain(g2_root / "domain_cfg.nc", 31)
    assert recipe._orca2g1_auto_substeps(dom, deck) == 65
    # plant: the pre-G2 form summed e3t_0 below the bottom as well
    unmasked = dict(dom, bottom_level=np.where(dom["bottom_level"] > 0, 30, 0))
    assert recipe._orca2g1_auto_substeps(unmasked, deck) != 65


@needs_shipped
def test_g2_refuses_planted_disagreements(g2_card, g2_root):
    state = g2_card.recipe.initial_state
    deep = np.asarray(state.T.data).copy()
    deep[73, 50, 20] += 1.0e-3              # a level the one-level check never saw
    with pytest.raises(ValueError, match="uniform_T"):
        recipe.validate_orca2g1_flat_card(g2_card._replace(recipe=g2_card.recipe._replace(
            initial_state=state._replace(T=state.T.replace(data=deep)))), g2_root)
    thick = np.asarray(g2_card.recipe.z_coord.h_partial).copy()
    thick[73, 50, 29] += 1.0
    with pytest.raises(ValueError, match="cell_thickness"):
        recipe.validate_orca2g1_flat_card(g2_card._replace(recipe=g2_card.recipe._replace(
            z_coord=g2_card.recipe.z_coord._replace(h_partial=thick))), g2_root)
    cfg = g2_card.recipe.model_config
    with pytest.raises(ValueError, match="stage r3u/r3v"):
        recipe.validate_orca2g1_flat_card(g2_card._replace(recipe=g2_card.recipe._replace(
            model_config=cfg._replace(rk3_stage_r3uv_nemo_literal=False))), g2_root)
    with pytest.raises(ValueError, match="jpk = 2"):
        recipe.validate_orca2g1_flat_card(g2_card._replace(case="ORCA2G1-flat"), g2_root)


@needs_shipped
def test_each_rung_refuses_the_other_rungs_domain(g2_root):
    with pytest.raises(ValueError, match="jpk = 2"):
        recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=g2_root)
    if (G1_RECORD / "domain_cfg.nc").is_file():
        with pytest.raises(ValueError, match="jpk = 31"):
            recipe.build_nemo_testcase_card("ORCA2G2-zps", deck_root=G1_RECORD)
    with pytest.raises(ValueError, match="requires an explicit deck_root"):
        recipe.build_nemo_testcase_card("ORCA2G2-zps")


@needs_shipped
@pytest.mark.skipif(not (G1_RECORD / "domain_cfg.nc").is_file(), reason="no G1 record")
def test_g2_differs_from_g1_only_by_what_the_domain_resolves(g2_card):
    g1 = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=G1_RECORD)
    a, b = g2_card.recipe.model_config, g1.recipe.model_config
    assert [f for f in a._fields if repr(getattr(a, f)) != repr(getattr(b, f))] == ["barotropic"]
    assert [f for f in a.barotropic._fields
            if getattr(a.barotropic, f) != getattr(b.barotropic, f)] == ["n_barotropic_substeps"]
    assert b.barotropic.n_barotropic_substeps == 13            # G1 unchanged
    assert np.array_equal(np.asarray(g2_card.recipe.initial_state.eta.data),
                          np.asarray(g1.recipe.initial_state.eta.data))
    assert np.array_equal(np.asarray(g2_card.recipe.land_mask), np.asarray(g1.recipe.land_mask))
    assert (g2_card.dt_s, g2_card.n_steps) == (g1.dt_s, g1.n_steps)
