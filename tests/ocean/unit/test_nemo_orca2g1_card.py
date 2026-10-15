"""The ORCA2G1 card (grid ladder rung G1): deck, bump, selectors, geometry.

Three layers:

* FILE-FREE: the committed deck parses to the stated values, every switch the
  card relies on is refused when moved, and the bump is the committed Fortran
  routine's.
* STAND-IN (skipped off this machine): the builder and its validator run on a
  one-level domain synthesised from ORCA2's shipped domain file.  The stand-in
  is NOT NEMO's DOMAINcfg output and proves nothing about geometry identity.
* RECORD (skipped until the G1 acquisition is admitted): the card's geometry
  against NEMO's own mesh_mask.nc, array by array.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

ROOT = Path(__file__).parents[3]
CASE = ROOT / "scripts/validate/ocean_fidelity/testcases/nemo_testcase_g1_orca2grid"
SHIPPED = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0/"
               "ORCA_R2_zps_domcfg.nc")
RECORD = Path(os.environ.get(
    "LEGOESM_ORCA2G1_RECORD",
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round1/"
    "oracle_g1_r1/p3_a_100step"))


@pytest.fixture(autouse=True)
def _fp64():
    before = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(before)


def _deck_copy(tmp_path: Path, edit=None) -> Path:
    text = (CASE / "namelist_cfg").read_text()
    if edit is not None:
        edited = edit(text)
        assert edited != text, "the plant did not change the deck"
        text = edited
    (tmp_path / "namelist_cfg").write_text(text)
    return tmp_path


def test_committed_deck_states_the_g1_values(tmp_path):
    deck = recipe.read_orca2g1_deck(_deck_copy(tmp_path))
    assert deck == recipe.ORCA2G1Deck(
        rn_Dt_s=10800.0, nn_itend=100, nn_fsbc=2, rn_bt_cmax=0.8, rn_bt_alpha=0.09,
        rn_avm0_m2_s=1.2e-4, rn_avt0_m2_s=1.2e-5)


def _flip(name: str):
    group, key = name.split(".")

    def edit(text: str) -> str:
        out, inside, hits = [], False, 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip().lower()
            if stripped.startswith("&"):
                inside = stripped[1:].split()[0] == group
            match = re.match(r"(\s*)([A-Za-z0-9_]+)(\s*=\s*)(\S+)(.*)", line, re.S)
            if inside and match and match.group(2).lower() == key:
                value = match.group(4).lower()
                new = {".true.": ".false.", ".false.": ".true."}.get(value)
                if new is None:
                    new = "7" if "." not in value else "7.5"
                line = "".join((match.group(1), match.group(2), match.group(3),
                                new, match.group(5)))
                hits += 1
            out.append(line)
        assert hits == 1, f"{name} is stated {hits} times"
        return "".join(out)

    return edit


@pytest.mark.parametrize("name", sorted(recipe._ORCA2G1_REQUIRED_SWITCHES))
def test_every_required_switch_is_refused_when_moved(tmp_path, name):
    with pytest.raises(ValueError, match=re.escape(name)):
        recipe.read_orca2g1_deck(_deck_copy(tmp_path, _flip(name)))


def test_missing_and_duplicated_switches_are_refused(tmp_path):
    drop = lambda text: text.replace("   ln_dynadv_OFF  = .true.", "")  # noqa: E731
    with pytest.raises(ValueError, match="does not state namdyn_adv.ln_dynadv_off"):
        recipe.read_orca2g1_deck(_deck_copy(tmp_path, drop))
    twice = lambda text: text.replace(  # noqa: E731
        "   rn_bt_alpha   = 0.09", "   rn_bt_alpha   = 0.09\n      rn_bt_alpha   = 0.07")
    with pytest.raises(ValueError, match="assigned twice"):
        recipe.read_orca2g1_deck(_deck_copy(tmp_path, twice))
    odd = lambda text: text.replace("nn_itend    =     100", "nn_itend    =     101")  # noqa: E731
    with pytest.raises(ValueError, match="multiple of nn_fsbc"):
        recipe.read_orca2g1_deck(_deck_copy(tmp_path, odd))


def test_bump_constants_are_the_committed_fortran_routines():
    source = (CASE / "usrdef_istate.F90").read_text()
    ji = int(re.search(r"np_g1_ji0\s*=\s*(\d+)", source).group(1))
    jj = int(re.search(r"np_g1_jj0\s*=\s*(\d+)", source).group(1))
    assert (ji, jj) == (recipe._ORCA2G1_BUMP_JI, recipe._ORCA2G1_BUMP_JJ)
    assert "pts(:,:,:,jp_tem) = 20._wp" in source
    assert "pts(:,:,:,jp_sal) = 30._wp" in source
    assert "zmax = MAXVAL( zdist ) / 20._wp" in source
    assert ("0.1 * COS( zdist(ji,jj) / zmax * rpi * 0.5_wp ) * ptmask(ji,jj,1)"
            in source)


def test_bump_formula_on_a_regular_grid():
    lon = (80.0 + 2.0 * np.arange(180) + 180.0) % 360.0 - 180.0
    lat = -74.0 + np.arange(148.0)
    glamt, gphit = np.meshgrid(lon, lat)
    wet = np.ones(glamt.shape, dtype=bool)
    j0, i0 = recipe._ORCA2G1_BUMP_JJ - 1, recipe._ORCA2G1_BUMP_JI - 1
    wet[j0 + 1, i0] = False
    ssh = recipe.orca2g1_initial_ssh(glamt, gphit, wet)
    assert ssh[j0, i0] == 0.1
    assert ssh[j0 + 1, i0] == 0.0 and ssh[j0 - 1, i0] > 0.0      # land is masked
    # The centre sits on the dateline: the wrapped longitude difference makes
    # the bump symmetric across it instead of 360 degrees wide.
    assert ssh[j0, i0 + 2] == ssh[j0, i0 - 2] > 0.0
    dlam = glamt - glamt[j0, i0]
    dlam = dlam - 360.0 * np.round(dlam / 360.0)
    dist = np.sqrt(dlam ** 2 + (gphit - gphit[j0, i0]) ** 2)
    assert np.all(ssh[dist > dist.max() / 20.0] == 0.0)
    assert 0 < np.count_nonzero(ssh) < ssh.size // 20


@pytest.fixture(scope="module")
def standin_root(tmp_path_factory):
    netCDF4 = pytest.importorskip("netCDF4")
    if not SHIPPED.is_file():
        pytest.skip(f"ORCA2's shipped domain file is not on this machine: {SHIPPED}")
    root = tmp_path_factory.mktemp("g1_standin")
    shutil.copy(CASE / "namelist_cfg", root / "namelist_cfg")
    with netCDF4.Dataset(SHIPPED) as src, netCDF4.Dataset(
            root / "domain_cfg.nc", "w", format="NETCDF4_CLASSIC") as dst:
        dst.createDimension("y", 148)
        dst.createDimension("x", 180)
        dst.createDimension("z", 2)
        for name in ("glamt", "glamu", "glamv", "glamf", "gphit", "gphiu",
                     "gphiv", "gphif", "e1t", "e1u", "e1v", "e1f", "e2t",
                     "e2u", "e2v", "e2f", "ff_f", "ff_t"):
            dst.createVariable(name, "f8", ("y", "x"))[:] = src[name][:]
        for name in ("e3t_1d", "e3w_1d"):
            dst.createVariable(name, "f8", ("z",))[:] = 100.0
        for name in ("e3t_0", "e3u_0", "e3v_0", "e3f_0"):
            dst.createVariable(name, "f8", ("z", "y", "x"))[:] = 100.0
        wet = (np.asarray(src["bottom_level"][:]) > 0).astype("i4")
        dst.createVariable("bottom_level", "i4", ("y", "x"))[:] = wet
    return root


def test_standin_card_builds_and_states_every_selector(standin_root):
    set_policy(PrecisionPolicy.fp64())
    card = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=standin_root)
    cfg = card.recipe.model_config
    state = card.recipe.initial_state
    assert (card.case, card.dt_s, card.n_steps) == ("ORCA2G1-flat", 10800.0, 100)
    assert np.asarray(state.T.data).shape == (148, 180, 1)
    assert np.asarray(state.T.data).dtype == np.float64
    assert np.asarray(card.recipe.z_coord.dz_ref).dtype == np.float64
    assert float(np.max(np.asarray(card.recipe.z_coord.h_partial))) == 100.0
    assert cfg.barotropic.nemo_barotropic_filter_alpha == 0.09
    # CEILING(10800 / 0.8 * 9.4035e-4) = CEILING(12.69) (dynspg_ts.f90:1062)
    assert cfg.barotropic.n_barotropic_substeps == 13
    assert (cfg.momentum_flux_scheme, cfg.vertical_momentum_scheme,
            cfg.tracer_advection) == ("none", "none", "none")
    assert np.count_nonzero(np.asarray(state.eta.data)) == 201
    assert cfg.physics.convection.scheme == "none"
    recipe.validate_orca2g1_flat_card(card, standin_root)
    with pytest.raises(ValueError, match="not execution-ready"):
        recipe.validate_orca2g1_flat_card_for_execution(card, standin_root)
    # The shared table does not list this card: it must refuse, not guess.
    with pytest.raises(ValueError, match="unknown NEMO testcase card"):
        recipe.validate_nemo_testcase_card(card)

    def moved(**changes):
        return card._replace(recipe=card.recipe._replace(
            model_config=cfg._replace(**changes)))

    plants = {
        "ln_dynadv_OFF": moved(momentum_flux_scheme="up3"),
        "ln_traadv_OFF": moved(tracer_advection="fct2"),
        "ln_dynvor_een": moved(vorticity_scheme="een"),
        "ln_traldf_OFF": moved(K_h=1.0),
        "ln_hpg_sco": moved(pgf_scheme="adcroft"),
        "ln_zdfevd=F": moved(physics=cfg.physics._replace(
            convection=cfg.physics.convection._replace(
                scheme="enhanced_diffusion"))),
        "nn_e3f_typ=0": moved(een_e3f_scheme="nemo_sum"),
        "ln_eos80": moved(eos="nemo_seos"),
        "barotropic_identity": moved(barotropic=cfg.barotropic._replace(
            barotropic_pgf_evaluation="vectorized")),
        "nn_e": moved(barotropic=cfg.barotropic._replace(n_barotropic_substeps=65)),
        "rn_bt_alpha": moved(barotropic=cfg.barotropic._replace(
            nemo_barotropic_filter_alpha=0.07)),
        "rn_Dt": card._replace(dt_s=5400.0),
        "no_ice": card._replace(icebergs_enabled=True),
        "bump": card._replace(recipe=card.recipe._replace(
            initial_state=state._replace(eta=state.eta.replace(
                data=np.asarray(state.eta.data) * 2.0)))),
        "initial_T": card._replace(recipe=card.recipe._replace(
            initial_state=state._replace(T=state.T.replace(
                data=np.asarray(state.T.data) + 1.0)))),
    }
    for name, planted in plants.items():
        with pytest.raises(ValueError, match=re.escape(name)):
            recipe.validate_orca2g1_flat_card(planted, standin_root)


def test_external_courant_number_at_the_deck_time_step(standin_root):
    """dynspg_ts.f90:1049-1080 in NEMO's own order, not the card's helper."""
    dom = recipe._orca2g1_domain(standin_root / "domain_cfg.nc")
    deck = recipe.read_orca2g1_deck(standin_root)
    ht_0 = 100.0 * (dom["bottom_level"] > 0)
    zcu = np.sqrt(float(recipe.NEMO_CONSTANTS_CONFIG.g) * ht_0
                  * (dom["e1t"] ** -2 + dom["e2t"] ** -2))
    j, i = np.unravel_index(np.argmax(zcu), zcu.shape)
    nn_e = int(np.ceil(deck.rn_Dt_s / deck.rn_bt_cmax * zcu.max()))
    courant = zcu.max() * deck.rn_Dt_s / nn_e
    assert (j + 1, i + 1) == (2, 50)                  # (jj, ji), Antarctic coast
    assert nn_e == recipe._orca2g1_auto_substeps(dom, deck) == 13
    np.testing.assert_allclose(zcu.max(), 9.4035e-4, rtol=1e-4)   # 1/s
    np.testing.assert_allclose(courant, 0.7812, rtol=1e-4)
    assert courant < deck.rn_bt_cmax < 0.9            # NEMO's stop is at 0.9


def test_single_level_column_needs_the_explicit_opt_in():
    from legoesm.ocean.vertical import create_z_star_from_thicknesses
    with pytest.raises(ValueError, match=">= 2 thicknesses"):
        create_z_star_from_thicknesses(np.array([100.0]))
    one = create_z_star_from_thicknesses(np.array([100.0]), allow_single_level=True)
    assert np.asarray(one.dz_ref).shape == (1,)


def test_geometry_identity_against_nemo_mesh_mask():
    """Every mesh array the card carries equals NEMO's mesh file (36 rows).

    Round 1 listed 17 names and asserted 19; the list and the count now live
    in the round-2 gate, which this test runs on the admitted record.
    """
    pytest.importorskip("netCDF4")
    if not (RECORD / "mesh_mask.nc").is_file():
        pytest.skip(f"the G1 record is not admitted on this machine ({RECORD})")
    import importlib.util
    path = (Path(__file__).resolve().parents[3] / "scripts/validate/ocean_fidelity/"
            "testcases/nemo_testcase_g1_orca2grid_ladder.py")
    spec = importlib.util.spec_from_file_location("g1_ladder_for_card_test", path)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    result = gate.geometry(RECORD)
    assert result["n_rows"] == 36
    assert result["all_bit_identical"], [
        r for r in result["rows"] if not r["bit_identical"]]
