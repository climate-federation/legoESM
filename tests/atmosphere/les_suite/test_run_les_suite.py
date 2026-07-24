"""CPU dispatch tests for the LES-suite emission driver (no GPU LES integration).

The end-to-end emit (running the spectral core) is GPU-gated and exercised
separately; here we lock the registry dispatch + not-wired-regime hardening.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "run"))


def _driver():
    import run_les_suite  # noqa: PLC0415
    return run_les_suite


def test_unknown_case_raises():
    m = _driver()
    with pytest.raises(Exception):
        m.main(["--case", "does_not_exist"])


def test_not_wired_regime_raises_systemexit():
    # a stratocumulus/moist case is registered but its emission is not wired yet;
    # the driver must refuse (SystemExit), not emit a wrong-regime artifact.
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "dycoms_rf01_sc"])


def test_wired_regimes_contains_dry_convective():
    m = _driver()
    assert "dry_convective" in m._WIRED_REGIMES


# --- SGS selection for the D7 σ_LES spread ------------------------------------
def test_sgs_les_config_maps_each_variant():
    m = _driver()
    assert m._sgs_les_config("lasd") == {
        "smagorinsky_dynamic": True, "sgs_model": "smagorinsky"}
    assert m._sgs_les_config("smagorinsky") == {
        "smagorinsky_dynamic": False, "sgs_model": "smagorinsky"}
    assert m._sgs_les_config("vreman") == {
        "smagorinsky_dynamic": False, "sgs_model": "vreman"}


def test_sgs_les_config_unknown_raises():
    m = _driver()
    with pytest.raises(SystemExit):
        m._sgs_les_config("amd")  # not wired into the emit reconstruction → hard error


def test_sgs_not_in_case_variants_raises():
    # --sgs must be one of the case's declared sgs_variants; a stray one is a hard
    # error, not a silent emission of an unclaimed closure.
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "cbl_nieuwstadt", "--sgs", "amd"])


# --- sheared CBL (U_g axis) ---------------------------------------------------
def test_coriolis_f_from_latitude():
    m = _driver()
    import numpy as np

    from legoesm import constants
    # f = 2Ω sin(φ); GABLS1 latitude ~73° gives ~1.39e-4
    assert m._coriolis_f(0.0) == 0.0
    assert m._coriolis_f(90.0) == pytest.approx(2.0 * constants.Omega)
    assert m._coriolis_f(73.0) == pytest.approx(
        2.0 * constants.Omega * np.sin(np.radians(73.0)))


def test_build_cbl_sheared_initialises_geostrophic_wind():
    import argparse

    import jax.numpy as jnp
    import numpy as np
    m = _driver()
    from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
    if not list_cases():
        register_default_catalog()
    case = get_case("cbl_nieuwstadt")
    args = argparse.Namespace(theta0=300.0, z0=0.1, pr_sgs=1.0, gamma=0.008,
                              zi0=800.0, q0=None)
    # free convection: u=0
    g0, st0, _ = m._build_cbl(case, args, jnp.float32, sgs="lasd", u_geo_mag=0.0)
    assert float(np.mean(np.asarray(st0.u))) == pytest.approx(0.0, abs=1e-6)
    # sheared: u initialised to U_g everywhere
    g8, st8, _ = m._build_cbl(case, args, jnp.float32, sgs="lasd", u_geo_mag=8.0)
    assert float(np.mean(np.asarray(st8.u))) == pytest.approx(8.0, abs=1e-5)
    assert float(np.mean(np.asarray(st8.v))) == pytest.approx(0.0, abs=1e-6)


def test_negative_ug_rejected():
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "cbl_nieuwstadt", "--Ug", "-1.0"])


def test_nonfinite_ug_rejected():
    # a NaN comparison is False, so `--Ug nan` would slip past `< 0` → must be caught
    m = _driver()
    with pytest.raises(SystemExit):
        m.main(["--case", "cbl_nieuwstadt", "--Ug", "nan"])


def test_bad_latitude_rejected_for_sheared():
    # these raise at input validation BEFORE the GPU emit (Ug valid, lat invalid)
    m = _driver()
    for lat in ("100", "-91", "nan"):
        with pytest.raises(SystemExit):
            m.main(["--case", "cbl_nieuwstadt", "--Ug", "8", "--lat", lat])


def test_build_cbl_selects_sgs_in_config():
    # _build_cbl only builds the config + IC (no integration) → CPU-safe. Each SGS
    # variant must land in the SpectralLESConfig the core will integrate.
    import argparse

    m = _driver()
    from legoesm.atmosphere.les_suite import get_case, list_cases, register_default_catalog
    if not list_cases():
        register_default_catalog()
    case = get_case("cbl_nieuwstadt")
    args = argparse.Namespace(
        theta0=300.0, z0=0.1, pr_sgs=1.0, gamma=0.008, zi0=800.0, q0=None)
    import jax.numpy as jnp
    for sgs, dyn, model in (("lasd", True, "smagorinsky"),
                            ("smagorinsky", False, "smagorinsky"),
                            ("vreman", False, "vreman")):
        g, _st, _q0 = m._build_cbl(case, args, jnp.float32, sgs=sgs)
        assert bool(g.cfg.smagorinsky_dynamic) is dyn
        assert g.cfg.sgs_model == model


# --- frame scheduling (codex round-2 regression) ------------------------------
def test_frame_schedule_production_evenly_spaced():
    m = _driver()
    # 7200 steps (dt=1, T=7200), 12 frames → 11 post-IC snapshots, last = n_steps
    rec = m.frame_step_schedule(7200, 12)
    assert len(rec) == 11
    assert rec == sorted(set(rec))           # strictly increasing + distinct
    assert rec[-1] == 7200                    # final frame at T
    assert all(1 <= k <= 7200 for k in rec)


def test_frame_schedule_distinct_and_capped_when_dt_coarse():
    m = _driver()
    # only 3 steps but 6 frames requested → at most 3 distinct post-IC snapshots
    rec = m.frame_step_schedule(3, 6)
    assert rec == [1, 2, 3]                    # distinct, no duplicates/drops
    assert rec == sorted(set(rec))


def test_frame_schedule_last_is_nsteps():
    m = _driver()
    for n_steps, frames in [(100, 5), (7, 4), (14400, 12), (5, 10)]:
        rec = m.frame_step_schedule(n_steps, frames)
        assert rec[-1] == n_steps
        assert rec == sorted(set(rec))
