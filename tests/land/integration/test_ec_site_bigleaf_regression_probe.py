"""Smoke + invariant test for the EC-site big-leaf regression probe.

Guards the probe used to check the PR-#897 big-leaf FvCB adoption for regressions
(``scripts/validate/ec_site_bigleaf_regression_probe.py``).  Asserts the physical
invariants the probe exists to verify — finite, GPP>=0, monotone light response,
sane magnitude — plus that the two-leaf canopy kernels return finite values.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")

_PROBE = (pathlib.Path(__file__).resolve().parents[3]
          / "scripts" / "validate" / "ec_site_bigleaf_regression_probe.py")


def _load():
    spec = importlib.util.spec_from_file_location("ec_bigleaf_probe", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lai_fixture_present_and_drives_sites():
    """The probe's per-site inputs come from the committed provenance fixture, not
    hard-coded constants — a missing/empty fixture must fail loudly."""
    mod = _load()
    assert mod._LAI_CSV.exists(), "LAI provenance fixture must be committed"
    sites = mod.load_sites()
    assert sites == mod.SITES and len(sites) == 4
    for name, sp in sites.items():
        assert len(sp["lai"]) == 2 and sp["lai"][1] >= sp["lai"][0] > 0.0
        assert 250.0 < sp["T_air"] < 320.0 and 0.0 < sp["q"] < 0.05
    with pytest.raises(FileNotFoundError):
        mod.load_sites(pathlib.Path("/nonexistent/ec_lai.csv"))


def test_bigleaf_grid_invariants():
    mod = _load()
    rows = mod.bigleaf_grid()
    assert len(rows) == len(mod.SITES) * 2 * len(mod.SW_LEVELS) * len(mod.BETA_LEVELS)
    gpp = np.array([r["gpp"] for r in rows])
    beta = np.array([r["beta_eff"] for r in rows])
    # finite, non-negative GPP, effective beta a valid [0, 1] fraction
    assert np.all(np.isfinite(gpp))
    assert np.all(gpp >= -1e-12)
    assert np.all(np.isfinite(beta)) and np.all(beta >= -1e-9) and np.all(beta <= 1.0 + 1e-9)
    # sane magnitude: peak GPP in a physical range (gC/m2/s -> umol/m2/s via /12e-6)
    assert (gpp.max() / 12e-6) < 60.0
    # monotone light response: GPP non-decreasing with SW at fixed site/LAI/beta
    for site in mod.SITES:
        for lai in mod.SITES[site]["lai"]:
            lai_r = round(float(lai), 3)
            for beta_s in mod.BETA_LEVELS:
                g = [next(r["gpp"] for r in rows
                          if r["site"] == site and r["lai"] == lai_r
                          and r["sw"] == sw and r["beta_soil"] == beta_s)
                     for sw in mod.SW_LEVELS]
                assert all(g[i + 1] >= g[i] - 1e-9 for i in range(len(g) - 1)), (
                    f"non-monotone light response at {site} LAI={lai_r} beta={beta_s}")


def test_production_grid_exercises_land_params_and_c4_blend():
    """The probe must drive the production land_params override + C3/C4 blend (not only
    the land_params=None default branch): fC4=1 (C4) must change GPP vs fC4=0 (C3)."""
    mod = _load()
    assert mod._has_fc4(), "post-#897 tree must expose LandSurfaceParams.fC4"
    rows = mod.bigleaf_production_grid()
    assert len(rows) == 2 * len(mod.SITES)
    gpp = np.array([r["gpp"] for r in rows])
    assert np.all(np.isfinite(gpp)) and np.all(gpp >= -1e-12)
    for site in mod.SITES:
        g_c3 = next(r["gpp"] for r in rows if r["site"] == site and r["fc4"] == 0.0)
        g_c4 = next(r["gpp"] for r in rows if r["site"] == site and r["fc4"] == 1.0)
        assert not np.isclose(g_c3, g_c4), (
            f"{site}: fC4 does not change GPP -> C3/C4 blend not exercised")


def test_canopy_kernel_grid_finite():
    mod = _load()
    ker = mod.canopy_kernel_grid()
    assert set(ker) == {"c3", "c4"}
    for k, vals in ker.items():
        arr = np.array(vals)
        assert arr.shape == (4,)
        assert np.all(np.isfinite(arr)) and np.all(arr >= -1e-9)


_GOLDEN = (pathlib.Path(__file__).resolve().parents[3]
           / "scripts" / "validate" / "ec_site_regression_golden.json")


def test_matches_committed_golden():
    """Pin the probe outputs to a committed HEAD golden so a future numerics change that
    silently alters the documented GPP shifts (median -17.5%, per-site magnitudes, C3/C4
    blend, canopy kernels) fails LOUDLY instead of staying green on broad invariants.

    Tolerance rtol=1e-3 catches any material (>0.1%) physics drift while absorbing
    cross-platform float noise in the FvCB kernels; atol handles near-zero night points.
    """
    import json
    mod = _load()
    gold = json.loads(_GOLDEN.read_text())
    # big-leaf grid (land_params=None) — align by key, compare GPP + beta
    def _key(r):
        return (r["site"], r["lai"], r["sw"], r["beta_soil"])
    now = {_key(r): r for r in mod.bigleaf_grid()}
    g = {_key(r): r for r in gold["bigleaf"]}
    assert set(now) == set(g)
    for k in g:
        np.testing.assert_allclose(now[k]["gpp"], g[k]["gpp"], rtol=1e-3, atol=1e-10)
        np.testing.assert_allclose(now[k]["beta_eff"], g[k]["beta_eff"], rtol=1e-3, atol=1e-6)
    # production grid (land_params override + C3/C4 blend)
    npd = {(r["site"], r["fc4"]): r["gpp"] for r in mod.bigleaf_production_grid()}
    gpd = {(r["site"], r["fc4"]): r["gpp"] for r in gold["bigleaf_production"]}
    assert set(npd) == set(gpd)
    for k in gpd:
        np.testing.assert_allclose(npd[k], gpd[k], rtol=1e-3, atol=1e-10)
    # canopy kernels — the byte-identity surface; tight tolerance
    for kern in ("c3", "c4"):
        np.testing.assert_allclose(mod.canopy_kernel_grid()[kern],
                                   gold["canopy_kernel"][kern], rtol=1e-6, atol=1e-9)
