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

import numpy as np
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


# --- DYCOMS-II RF02 ---------------------------------------------------------

def test_rf02_gets_its_own_forcing(drv):
    """RF02 is the DRIZZLING flight of the same campaign, not a relabelled RF01.

    Every number here is read off ``data/les_cases/DYCOMS_RF02`` rather than
    inherited: the deck's prm carries no ``fcor`` (RF01's does), so gSAM
    derives f from ``latitude0 = 31.5``; its lsf gives w = -0.00598 m/s at
    1595 m, i.e. D = 3.75e-6; and its ``&MICRO_DRIZZLE Nc0`` is 55 cm^-3
    against RF01's 140.
    """
    from legoesm import constants

    a = _args(drv, ["--case", "rf02"])
    assert "DYCOMS_RF02" in str(a.case_dir)
    assert a.f_cor == pytest.approx(
        2.0 * constants.Omega * float(np.sin(np.deg2rad(31.5))))
    assert a.subsidence_divergence_s == pytest.approx(3.75e-6)
    assert a.Lz == pytest.approx(1500.0)
    assert a.n_c_m3 == pytest.approx(55.0e6)
    assert a.case_label == "rf02"


def test_rf02_does_not_inherit_rf01_coriolis(drv):
    """The guard is not vacuous: the two flights really do differ, by 2.03x."""
    rf01 = _args(drv, ["--case", "dycoms"])
    rf02 = _args(drv, ["--case", "rf02"])
    assert not np.isclose(rf01.f_cor, rf02.f_cor, rtol=0.05)
    assert rf02.f_cor / rf01.f_cor == pytest.approx(2.03, abs=0.02)


def test_perturbation_is_seeded_below_the_case_inversion(drv):
    """RF02's inversion is at 795 m; RF01's literal 840 m would seed noise in
    air the case does not perturb."""
    assert _args(drv, ["--case", "rf02"]).perturb_z_m == pytest.approx(795.0)
    # unchanged for the decks that were already running
    assert _args(drv, []).perturb_z_m == pytest.approx(840.0)
    assert _args(drv, ["--case", "astex"]).perturb_z_m == pytest.approx(840.0)


def test_les_and_scm_arms_agree_on_rf02_coriolis(drv):
    """The LES driver and its SCM twin compute f separately; a drift between
    them is a rotation mismatch the tuner would charge to turbulence."""
    from legoesm.atmosphere.forcing.scm.sam_case_scm import SAM_SCM_CASES
    for case in ("dycoms", "rf02", "astex"):
        les = _args(drv, ["--case", case]).f_cor
        assert les == pytest.approx(SAM_SCM_CASES[case].les_f_c, rel=1e-12), case


def test_every_stratocumulus_case_has_an_scm_twin(drv):
    """A deck the LES can run but the SCM cannot is a case with no comparison."""
    from legoesm.atmosphere.forcing.scm.sam_case_scm import SAM_SCM_CASES
    for name, spec in drv._STRATOCUMULUS_CASES.items():
        assert name in SAM_SCM_CASES, f"{name} has no SCM twin"
        assert SAM_SCM_CASES[name].gsam_dir == spec["gsam_dir"]


def test_rf02_droplet_concentration_actually_drives_its_drizzle(drv):
    """RF02's defining feature is drizzle, and the only knob the case sets for
    it is the droplet concentration.

    Controlled A/B on the SAME RF02 initial column with the SAME Morrison
    scheme, one field changed: dropping N_c from RF01's 140 cm^-3 to RF02's 55
    must RAISE the rain production, by roughly the KK2000-type N_c^-1.79 of the
    autoconversion (2.55^1.79 = 5.4x). If the case's N_c were inert -- wired in
    but not reaching the scheme -- the two arms would be identical here.
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.les.spectral_les_moist import (
        make_les_microphysics_fn,
    )
    from legoesm.atmosphere.physics.microphysics.config import (
        MicrophysicsConfig, MorrisonConfig,
    )

    args = _args(drv, ["--case", "rf02", "--nx", "4", "--ny", "4",
                       "--nz", "48", "--Lx", "200", "--Ly", "200", "--f32"])
    if not Path(args.case_dir).is_dir():
        pytest.skip("DYCOMS_RF02 gSAM deck not cached")
    g, st, ref, _ = drv.build(args, jnp.float32)
    assert float(jnp.max(st.tracers[..., 1])) > 1.0e-4, (
        "RF02's sub-inversion layer must be saturated in the initial state, or "
        "there is no cloud for the drizzle comparison to act on")

    rain = {}
    for n_c in (55.0e6, 140.0e6):
        cfg = MicrophysicsConfig(
            scheme="morrison",
            morrison=MorrisonConfig(morrison_flavor="sam", Nc_0=n_c))
        micro = make_les_microphysics_fn(cfg, ref, g.dz, 0.5)
        _, dtr, _ = micro(st.theta, st.tracers)
        rain[n_c] = float(jnp.max(jnp.abs(dtr[..., 2])))   # slot 2 = q_r

    assert rain[55.0e6] > 0.0, "RF02 must produce rain; it is a drizzling case"
    ratio = rain[55.0e6] / rain[140.0e6]
    assert 3.0 < ratio < 9.0, (
        f"N_c must drive the drizzle: 55 vs 140 cm^-3 gave {ratio:.2f}x, "
        "expected the ~5x of an N_c^-1.79 autoconversion")
