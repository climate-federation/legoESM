"""``run_dycoms_les.py`` drives TWO stratocumulus decks, not one.

DYCOMS_RF01 and ASTEX209 both set ``dolongwave = .true., doradsimple = .true.``
so both are driven by the same Stevens (2005) simple longwave with the same
hardcoded constants. What differs is per deck, and getting any of it wrong is
a silent mis-forcing rather than an error:

* ASTEX sets ``docoriolis = .false.``, so its LES has no Coriolis at all.
* The SUBSIDENCE divergence is not the longwave's. gSAM hardcodes
  ``f0 = 3.75e-6`` in rad_simple's clear-sky term for every deck, while the
  subsidence comes from the lsf file. They coincide for DYCOMS and differ for
  ASTEX.

These tests pin that the DYCOMS defaults are unchanged and that ASTEX gets its
own values.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "run" / "run_dycoms_les.py")


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("_strato_les", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_strato_les"] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(drv, argv):
    old = sys.argv
    sys.argv = ["run_dycoms_les.py", *argv]
    try:
        return drv.parse_args()
    finally:
        sys.argv = old


def test_dycoms_defaults_are_unchanged(drv):
    """The historical hardcoded values, now reached through the case table."""
    a = _args(drv, [])
    assert a.case == "dycoms"
    assert a.f_cor == pytest.approx(0.376e-4)
    assert a.subsidence_divergence_s == pytest.approx(3.75e-6)
    assert a.Lz == pytest.approx(1500.0)
    assert a.n_c_m3 == pytest.approx(140.0e6)
    assert "DYCOMS_RF01" in str(a.case_dir)


def test_astex_gets_its_own_forcing(drv):
    a = _args(drv, ["--case", "astex"])
    assert a.f_cor == 0.0                      # docoriolis = .false.
    assert a.subsidence_divergence_s == pytest.approx(5.0e-6)
    assert a.Lz == pytest.approx(2500.0)
    assert "ASTEX209" in str(a.case_dir)


def test_subsidence_divergence_is_not_the_longwave_divergence(drv):
    """The conflation this separation exists to prevent.

    They are equal for DYCOMS, which is exactly why substituting one for the
    other would have gone unnoticed until ASTEX ran.
    """
    dy = _args(drv, [])
    ax = _args(drv, ["--case", "astex"])
    lw_div = drv._SIMPLE_LW.divergence_s
    assert dy.subsidence_divergence_s == pytest.approx(lw_div)
    assert ax.subsidence_divergence_s != pytest.approx(lw_div)
    # the longwave constant is deck-independent: one config serves both
    assert drv._SIMPLE_LW.divergence_s == pytest.approx(3.75e-6)


def test_explicit_flags_override_the_case_defaults(drv):
    a = _args(drv, ["--case", "astex", "--Lz", "1800"])
    assert a.Lz == pytest.approx(1800.0)
    assert a.subsidence_divergence_s == pytest.approx(5.0e-6)


def test_unknown_case_is_rejected(drv):
    """Dispatch hardening: a typo must not silently run DYCOMS."""
    with pytest.raises(SystemExit):
        _args(drv, ["--case", "dycomsII"])


def test_every_registered_case_has_a_deck_on_disk(drv):
    """A case in the table with no deck would fail only at run time."""
    root = Path(__file__).resolve().parents[2] / "data" / "les_cases"
    if not root.is_dir():
        pytest.skip("no local les_cases cache")
    for name, spec in drv._STRATOCUMULUS_CASES.items():
        deck = root / spec["gsam_dir"]
        assert deck.is_dir(), f"{name}: no deck at {deck}"
        for f in ("snd", "lsf", "sfc", "prm"):
            assert (deck / f).exists(), f"{name}: deck is missing {f}"
