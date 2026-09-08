"""The moist-oracle deck checkers, which decide whether a certification
number means anything.

``full_step_oracle_parity.py --moist`` reports a residual against a real
Fortran run.  Two preconditions make that number attributable: the
oracle deck's physics must not have touched the state, and the deck must
actually be moist.  Both are read out of the deck's own ``input.nml``,
so a parser slip does not raise -- it silently blesses the wrong deck
and manufactures a confident wrong certification.

Those functions previously had NO tests (GLM MINOR, job 9442724), and
one claim about them was only half-demonstrated: the dry deck is refused
by ``check_moist_deck`` on ``adiabatic``, which never exercised
``check_physics_is_inert``'s firing path at all.  Every case below is
synthetic so each checker's REFUSE path is reached on its own.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SRC = (pathlib.Path(__file__).resolve().parents[2]
        / "scripts/validate/fv3_native/full_step_oracle_parity.py")


def _mod():
    spec = importlib.util.spec_from_file_location("_fsop", _SRC)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:          # argparse at import time, if any
        pass
    return m


M = _mod()

_CLEAN = """
 &fv_core_nml
   fv_sg_adj = -1
   consv_te = 0.0
   adiabatic = .false.
 /
"""


def _deck(tmp_path, text):
    d = tmp_path / "run"
    d.mkdir()
    (d / "input.nml").write_text(text)
    return str(d)


# ---------------------------------------------------------------- parser
def test_last_assignment_wins_not_first():
    """These decks carry a commented alternative and then the live
    value; taking the first match reads the wrong one."""
    t = "fv_sg_adj = -1\nfv_sg_adj = 1\n"
    assert M._nml_int(t, "fv_sg_adj") == 1
    t = "do_Held_Suarez = .false.\ndo_Held_Suarez = .true.\n"
    assert M._nml_logical(t, "do_held_suarez") is True
    t = "consv_te = 0.0\nconsv_te = 2.5\n"
    assert M._nml_real(t, "consv_te") == 2.5


def test_commented_assignments_are_ignored():
    """``!`` starts a comment. A first-match regex would read the
    commented-out safe value and miss the live one."""
    assert M._nml_real("!consv_te = 0.0\nconsv_te = 1.5\n",
                       "consv_te") == 1.5
    assert M._nml_int("  ! fv_sg_adj = -1\n  fv_sg_adj = 3\n",
                      "fv_sg_adj") == 3
    assert M._nml_logical("!adiabatic = .true.\nadiabatic = .false.\n",
                          "adiabatic") is False


def test_absent_keys_are_none_not_a_default():
    """A missing key must be distinguishable from a present one: the
    checkers refuse on None rather than assuming the safe value."""
    assert M._nml_int("", "fv_sg_adj") is None
    assert M._nml_real("", "consv_te") is None
    assert M._nml_logical("", "adiabatic") is None


def test_fortran_d_exponent_is_read_as_a_real():
    assert M._nml_real("consv_te = 1.0d0\n", "consv_te") == 1.0


# ------------------------------------------------- check_physics_is_inert
@pytest.mark.parametrize("switch", sorted(M._PHYSICS_SWITCHES))
def test_every_listed_physics_switch_refuses(tmp_path, switch):
    """Each switch clears ``no_tendency`` (fv_phys.F90) and so lets
    ``fv_update_phys`` write the state. One case per switch, so the
    list cannot rot to a subset that is actually enforced."""
    d = _deck(tmp_path, _CLEAN + f"\n {switch} = .true.\n")
    with pytest.raises(SystemExit, match=switch):
        M.check_physics_is_inert(d)


def test_subgrid_z_refuses(tmp_path):
    d = _deck(tmp_path, "consv_te = 0.\nadiabatic = .false.\n"
                        "fv_sg_adj = 1\n")
    with pytest.raises(SystemExit, match="fv_sg_adj"):
        M.check_physics_is_inert(d)


def test_unset_fv_sg_adj_refuses_rather_than_defaulting(tmp_path):
    d = _deck(tmp_path, "consv_te = 0.\nadiabatic = .false.\n")
    with pytest.raises(SystemExit, match="fv_sg_adj"):
        M.check_physics_is_inert(d)


def test_a_clean_deck_passes_physics(tmp_path):
    """ANTI-VACUITY: the refusals above must not be a checker that
    refuses everything."""
    M.check_physics_is_inert(_deck(tmp_path, _CLEAN))


# ----------------------------------------------------- check_moist_deck
def test_a_dry_deck_is_refused(tmp_path):
    """``zvir`` is nowhere in the namelist -- atmosphere.F90:156-161
    derives it from ``adiabatic``. A deck resolving ``.true.`` ran DRY,
    and scoring the moist port against it would measure the coupling
    itself as the error."""
    d = _deck(tmp_path, "fv_sg_adj = -1\nconsv_te = 0.\n"
                        "adiabatic = .true.\n")
    with pytest.raises(SystemExit, match="adiabatic"):
        M.check_moist_deck(d)


def test_a_deck_without_adiabatic_at_all_is_refused(tmp_path):
    d = _deck(tmp_path, "fv_sg_adj = -1\nconsv_te = 0.\n")
    with pytest.raises(SystemExit, match="adiabatic"):
        M.check_moist_deck(d)


@pytest.mark.parametrize("te", ["1.0", "-0.5", "1.0d0"])
def test_a_nonzero_consv_te_is_refused(tmp_path, te):
    d = _deck(tmp_path, f"fv_sg_adj = -1\nadiabatic = .false.\n"
                        f"consv_te = {te}\n")
    with pytest.raises(SystemExit, match="consv_te"):
        M.check_moist_deck(d)


def test_a_clean_moist_deck_passes(tmp_path):
    M.check_moist_deck(_deck(tmp_path, _CLEAN))


# ------------------------------------------------- the REAL pinned decks
def test_the_pinned_decks_classify_as_the_campaign_claims():
    """The two checkers on the actual oracle runs: the moist decks pass
    both, and the DRY deck is refused as moist. Skipped rather than
    failed off-cluster, since the pinned tree is a machine path."""
    root = pathlib.Path(M.ORACLE_ROOT)
    if not root.exists():
        pytest.skip(f"pinned oracle tree not present at {root}")
    for name in ("run_hydro_1step_moist_gfs",
                 "run_hydro_zerostep_moist_gfs"):
        M.check_physics_is_inert(str(root / name))
        M.check_moist_deck(str(root / name))
    dry = str(root / "run_hydro_1step_gfs")
    M.check_physics_is_inert(dry)          # the dry deck IS physics-free
    with pytest.raises(SystemExit, match="adiabatic"):
        M.check_moist_deck(dry)
