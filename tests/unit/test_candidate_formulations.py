"""Known-answer checks for the fitted co-scientist candidates (benchmark v3)."""
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "scripts/validate/most_revisit"))
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import candidate_formulations as cf  # noqa: E402
import standard_formulations as sf  # noqa: E402
from test_flux_gradient_sites import site  # noqa: E402

J3 = [5.06, 3.44, 2.13, 0.95]  # Kansas variant quoted in coscientist/cs_refl2_J3_prompt.txt


def test_j3_values_slopes_and_heat_continuity():
    m, h = cf.make("J3", J3)
    assert float(m(0.7)) == pytest.approx(1 + 3.44 * 0.7 + (5.06 - 3.44) * 0.7 / (1 + 0.7 / 2.13), rel=1e-12)
    assert float(jax.grad(m)(1e-9)) == pytest.approx(5.06, rel=1e-6)            # a0 at 0+
    assert float(jax.grad(m)(-1e-9)) == pytest.approx(4.0, rel=1e-6)            # Businger-Dyer at 0-
    assert float(h(1e-12)) == pytest.approx(0.95) and float(h(-1e-12)) == pytest.approx(0.95)


def test_j17_quoted_value_and_slope():
    m, _ = cf.make("J17", [1.0])
    assert float(m(-0.7)) == pytest.approx(0.512, abs=5e-4)                     # cs_refl2_J17_prompt: lambda 1, x 0.7
    assert float(jax.grad(m)(-1e-9)) == pytest.approx(4 + 1 / 12, rel=1e-6)    # 4 + 1/(12 lambda)
    assert float(jax.grad(m)(1e-9)) == pytest.approx(5.0, rel=1e-6)


@pytest.mark.parametrize("a", [-8.0, 8.0])
def test_j5_constraint_keeps_phi_monotone(a):
    m, _ = cf.make("J5", [a, 0.0])
    z = np.r_[np.linspace(1e-4, 9.99, 2000), 2 - np.sqrt(2), 2 + np.sqrt(2)]  # include the exact extrema
    slope = np.asarray(jax.vmap(jax.grad(m))(jnp.asarray(z)))
    assert slope.min() >= -1e-9
    assert slope.min() < 1e-2 * max(1.0, abs(a))   # the bound is tight, not just loose


@pytest.mark.parametrize("form,truth", [("J3", [6.0, 2.0, 0.8, 0.9]), ("J17", [0.7]), ("J5", [3.0, 2.0])])
def test_fit_recovers_parameters(form, truth):
    rng = np.random.default_rng(0)
    m, h = cf.make(form, truth)
    sites = []
    for _ in range(2):
        out = {}
        for kind, f in (("m", m), ("h", h)):
            z = np.r_[rng.uniform(0.01, 5, 3000), -rng.uniform(0.01, 5, 3000)]
            out[kind] = (z, np.asarray(jax.vmap(f)(jnp.asarray(z))) * np.exp(rng.normal(0, 0.05, z.size)))
        sites.append(out)
    p = cf.fit(form, sites, 30)
    for name, t in zip(cf.FORMS[form][0], truth):
        assert p[name] == pytest.approx(t, rel=0.1), name


def test_day_parity_splits_samples():
    ds = site().assign_coords(time=1800.0 * (np.arange(200) + 1))
    days = np.floor(ds.time.values / 86400) % 2
    full = sf.site_pairs(ds, 0.1, False, 50.0)[0]["m"][0]["keep"]
    for parity in (0, 1):
        keep = sf.site_pairs(ds, 0.1, False, 50.0, day_parity=parity)[0]["m"][0]["keep"]
        assert np.array_equal(keep, full & (days == parity))


def test_strict_reading_flags():
    cf.register("J3_test", "J3", J3, "test")
    try:
        r = cf.strict("J3_test", "h")
        assert r["pass"] and r["monotone"]["stable"] and r["monotone"]["unstable"]
    finally:
        sf.SCHEMES.pop("J3_test")
    z = cf.strict("zeng1998_clm", "m")
    assert not z["monotone"]["unstable"] and not z["pass"]   # free-convection phi_m rises again with -z/L


def test_offset_uses_only_its_own_split():
    """Perturbing even-day neutral temperature differences must not change odd-day heat samples."""
    n = 400
    base = site(h=-200.0)  # strong enough that |z/L| >= 0.1 keeps heat samples
    ds = xr.concat([base, base], dim="time").assign_coords(time=1800.0 * (np.arange(n) + 1))
    even = (np.floor(ds.time.values / 86400) % 2 == 0)
    neutral = even | (np.arange(n) % 4 == 0)   # neutral rows mostly on even days, so a pooled median follows them
    ds["h"].values[:] = np.where(neutral, 2.0, -200.0)[:, None]
    ds["ustar"].values[:] = 0.5
    odd_a = sf.site_pairs(ds, 0.1, True, 50.0, day_parity=1)[0]["h"]
    ds2 = ds.copy(deep=True)
    ds2["temp"].values[even, 1] += 0.7  # .values: boolean indexing on the DataArray would modify a copy
    odd_b = sf.site_pairs(ds2, 0.1, True, 50.0, day_parity=1)[0]["h"]
    assert odd_a and len(odd_a) == len(odd_b) and sum(int(p["keep"].sum()) for p in odd_a) > 30
    for a, b in zip(odd_a, odd_b):
        assert np.allclose(a["phi"][a["keep"]], b["phi"][b["keep"]], equal_nan=True)


def test_global_weights_equal_per_site():
    one = [("h", np.full(100, 0.5), np.full(100, 2.0))]
    two = [("h", np.full(50, 0.5), np.full(50, 2.0)), ("h", np.full(50, -0.5), np.full(50, 0.5))]
    r = cf._residuals("J3", J3, [one, two])
    w1, w2 = np.sum(r[:100] ** 2), np.sum(r[100:] ** 2)
    m, h = cf.make("J3", J3)
    e1 = np.log(2.0 / float(h(0.5))) ** 2
    e2 = 0.5 * np.log(2.0 / float(h(0.5))) ** 2 + 0.5 * np.log(0.5 / float(h(-0.5))) ** 2
    assert w1 == pytest.approx(e1) and w2 == pytest.approx(e2)


def test_nonfinite_model_is_penalised_not_ignored():
    r = cf._residuals("J5", [0.0, 0.0], [[("m", np.array([0.5]), np.array([-1.0]))]])  # log of negative ratio
    assert abs(r[0]) == cf.FAIL_RESIDUAL


def test_failed_optimizer_is_not_a_fit(monkeypatch):
    class R:
        success, x = False, np.array([1.0])
    monkeypatch.setattr(cf, "least_squares", lambda *a, **k: R())
    z = -np.linspace(0.1, 3, 100)
    assert cf.fit("J17", [{"m": (z, np.ones(100)), "h": (z, np.ones(100))}], 30) is None


def test_repeated_zeta_marked_unidentified():
    z = np.full(500, 0.5)
    m, h = cf.make("J3", J3)
    p = cf.fit("J3", [{"m": (z, np.full(500, float(m(0.5)))), "h": (np.r_[z, -z], np.r_[np.full(500, 2.0), np.full(500, 0.6)])}], 30)
    assert p is not None and not p["diagnostics"]["momentum"]["identified"]


@pytest.mark.parametrize("power,finite", [(0.999, False), (1.1, True)])
def test_finite_slope_detector(power, finite):
    f = lambda z: 1.0 + jnp.sign(z) * jnp.abs(z) ** power  # noqa: E731
    sf.SCHEMES["planted"] = (f, f, sf.KAPPA, "", "")
    try:
        assert cf.strict("planted", "m")["finite_slope_at_0"] is finite
    finally:
        sf.SCHEMES.pop("planted")


def test_single_parameter_at_bound_is_flagged():
    z = -np.linspace(0.1, 3.0, 300)
    m, _ = cf.make("J17", [0.01])  # truth below the lower bound 0.05, where phi is sensitive, so the optimum sits on it
    p = cf.fit("J17", [{"m": (z, np.asarray(m(jnp.asarray(z)))), "h": (z, np.ones(300))}], 30)
    d = p["diagnostics"]["momentum"]
    assert d["constrained"] and d["at_bound"] == [True] and np.isnan(d["se"][0]) and p["lambda"] == pytest.approx(0.05)


def test_registered_dyer_heat_keeps_dyer_branch_checks():
    cf.register("J17_test", "J17", [1.0], "test")
    try:
        assert cf.strict("J17_test", "h")["c1_elsewhere"] == cf.strict("dyer1974", "h")["c1_elsewhere"]
    finally:
        sf.SCHEMES.pop("J17_test"); sf.BRANCH_POINTS.pop("J17_test")
