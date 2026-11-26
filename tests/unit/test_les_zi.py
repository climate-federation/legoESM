"""Known-answer checks for the LES z/zi test (les_zi.py)."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
import les_zi as lz  # noqa: E402


@pytest.fixture(autouse=True)
def baseline():
    lz.cf.register(lz.BASE, "J3", lz.J3_V5, "test")


def _profile(path, sgs_share):
    zf = np.arange(0, 2000, 10.0)
    zm = zf + 5.0
    ustar, q, th0 = 0.4, 0.2, 300.0
    U = 0.8 * ustar / lz.KAPPA * np.log(zm / 0.1)
    V = 0.6 * ustar / lz.KAPPA * np.log(zm / 0.1)  # turned wind: speed shear still u*/(kappa z)
    tot = -ustar**2 * (1 - zf / 2000)
    wt = q * (1.2 * (zf - 830) ** 2 / 830**2 - 0.2)
    sgs = tot * sgs_share(zf)
    rows = np.c_[zm, U, V, th0 - 0.002 * np.log(zm), zf, tot - sgs, sgs, 0 * zf, 0 * zf, wt, 0 * zf]
    np.savetxt(path, rows, header="h", comments="", delimiter="\t")


def test_load_drops_the_near_wall_sgs_block_and_uses_surface_scales(tmp_path):
    _profile(tmp_path / "p.txt", lambda z: np.where(z < 35, 0.8, 0.1))
    d = lz.load(str(tmp_path / "p.txt"))
    assert d["ustar"] == pytest.approx(0.4) and d["zi"] == pytest.approx(830.0)
    assert d["z"].min() == pytest.approx(40.0) and d["z"].max() <= 0.1 * d["zi"]
    # neutral-log wind: phi_m = z/(z+...)-free form gives kappa z dU/dz / u* ~ 1 at the flux height
    assert np.allclose(d["phi_m"], 1.0, atol=0.02)
    assert d["L"] == pytest.approx(-0.4**3 * (300.0 - 0.002 * np.log(5.0)) / (lz.KAPPA * lz.constants.g * 0.2), rel=1e-4) and d["L"] < 0
    # theta = th0 - 0.002 ln z: dtheta/dz = -0.002/z, theta* = -q/u* = -0.5: phi_h = kappa 0.002 / 0.5 > 0
    assert np.allclose(d["phi_h"], lz.KAPPA * 0.002 / 0.5, rtol=0.02)


def test_detached_sgs_spike_does_not_drop_the_profile_below_it(tmp_path):
    _profile(tmp_path / "p.txt", lambda z: np.where(z < 35, 0.8, np.where(np.abs(z - 60) < 1, 0.9, 0.1)))
    d = lz.load(str(tmp_path / "p.txt"))
    assert d["z"].min() == pytest.approx(40.0) and d["levels_dropped"] == 3


def test_cross_case_recovers_a_height_term_at_fixed_zeta():
    cases = {}
    rng = np.random.default_rng(0)
    for i, (L, zi) in enumerate([(-30, 900), (-60, 1100), (-120, 800), (-250, 1300), (-400, 1000)]):
        z = np.geomspace(20, 100, 15)
        r = 0.2 * np.log(z) - 0.1 * np.log(zi) + rng.normal(0, 1e-4, z.size)
        cases[str(i)] = {"z": z, "zeta": z / L, "zi": zi, "L": L, "r_m": r}
    c = lz.cross_case(cases, "m")
    assert c["rank"] == 2 and c["b_z"] == pytest.approx(0.2, abs=0.03) and c["b_zi"] == pytest.approx(-0.1, abs=0.05)


def test_cross_case_zero_for_stability_only_error_and_rank_deficient_without_zi_spread():
    cases = {}
    for i, (L, zi) in enumerate([(-30, 900), (-60, 1100), (-120, 800), (-250, 1300)]):
        z = np.geomspace(20, 100, 15)
        cases[str(i)] = {"z": z, "zeta": z / L, "zi": zi, "L": L, "r_m": 0.3 * np.log(np.abs(z / L)) + 0.1 * (z / L) ** 2}
    c = lz.cross_case(cases, "m")
    assert abs(c["b_z"]) < 1e-3 and abs(c["b_zi"]) < 1e-3  # piecewise-linear interpolation residual only; error depending on z/L only cannot appear as height
    for d in cases.values():
        d["zi"] = 1000.0
    assert lz.cross_case(cases, "m")["rank"] == 1
