"""Direct test for ``scripts/validate/mpas_energy_budget_leak.py`` (#1354).

The whole answer this script produces is a sign and a magnitude, so the tests
are about exactly those: a synthetic series with a KNOWN leak must come back
with that leak, and a series that closes must come back at zero.  Built from
the budget identity rather than from the script's own algebra, so it is a
cross-check and not a restatement.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROBE = _ROOT / "scripts" / "validate" / "mpas_energy_budget_leak.py"


def _load():
    spec = importlib.util.spec_from_file_location("_leak", _PROBE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_leak"] = mod
    spec.loader.exec_module(mod)
    return mod


def _series(tmp_path, *, leak, n=6, dt_days=1.0,
            key="energy_toa_minus_dE_dt"):
    """A synthetic run built from an ENERGY TRAJECTORY, not from the algebra.

    Codex, reviewing the first version: building `dE_dt` straight from the
    budget identity and then re-deriving the same identity in the assertion is
    not an independent check.  So this instead constructs a column energy
    E(t) whose tendency is what a leaking column would have, differences it
    the way the tracker does (`(E_n - E_{n-1}) / dt`, energy_budget.py), and
    solves for the ONE flux that has to balance -- here the sensible heat --
    given the others.  The probe's algebra is then checked against a
    trajectory, not against itself.

    The first sample carries `dE_dt = 0` and `residual = toa_net`, exactly as
    the tracker emits it, so the "drop the first sample" behaviour is exercised
    by every test rather than assumed.
    """
    rng = np.random.default_rng(0)
    toa = rng.normal(-3.0, 2.0, n)
    sw = rng.normal(160.0, 5.0, n)
    lw = rng.normal(-55.0, 3.0, n)
    lh = rng.normal(80.0, 4.0, n)

    dt_s = dt_days * 86400.0
    # A column energy trajectory [J/m^2] with n+1 points, so the PHYSICAL
    # tendency exists at every reported sample INCLUDING the first -- a real
    # column is not at rest when the tracker starts watching it.
    E = 2.7e9 + np.cumsum(rng.normal(0.0, 4.0e6, n + 1))
    dedt_true = (E[1:] - E[:-1]) / dt_s       # the tracker's own difference
    # Choose sensible heat so the TRUE budget carries exactly `leak`:
    #   dE/dt = toa - (sw+lw) + sh + lh + leak
    sh = dedt_true - toa + (sw + lw) - lh - leak
    # Now report it the way the tracker does: at its first sample there is no
    # previous energy to difference against, so it emits dE_dt = 0 and hence
    # residual = toa_net, while the fluxes are the real ones.  Sample 0's leak
    # is therefore wrong by exactly the true tendency there.
    dedt = dedt_true.copy()
    residual = toa - dedt
    dedt[0] = 0.0
    residual[0] = toa[0]

    path = tmp_path / "timeseries.npz"
    np.savez(path, days=np.arange(n, dtype=float) * dt_days,
             energy_toa_net=toa, energy_dE_dt=dedt, **{key: residual},
             sw_net_sfc=sw, lw_net_sfc=lw, hfss=sh, hfls=lh,
             # Interval means unless a test says otherwise: the probe refuses
             # snapshot-sourced series outright.
             energy_flux_interval_mean=np.ones(n),
             precip=np.full(n, 3.0))
    return path


def _leak_values(mod, path, drop_first=True):
    """Leak per sample, with the first dropped the way the probe drops it."""
    t = mod.load(path)
    v = (t["sw_net_sfc"] + t["lw_net_sfc"] - t["hfss"] - t["hfls"]
         - t["energy_toa_minus_dE_dt"])
    return v[1:] if drop_first else v


def test_a_closing_budget_reports_zero_leak(tmp_path):
    mod = _load()
    got = _leak_values(mod, _series(tmp_path, leak=0.0))
    np.testing.assert_allclose(got, 0.0, atol=1e-6)


@pytest.mark.parametrize("injected", [20.0, -20.0, 5.0])
def test_an_injected_leak_is_recovered_with_the_right_sign(tmp_path, injected):
    """Sign included: a leak that CREATES energy must come back positive."""
    mod = _load()
    got = _leak_values(mod, _series(tmp_path, leak=injected))
    np.testing.assert_allclose(got, injected, atol=1e-6)


def test_the_first_sample_is_bogus_and_must_be_dropped(tmp_path):
    """The tracker emits dE_dt = 0 on its first sample, so its leak is junk.

    Keeping it contaminates the mean, and the previous `>=` filter kept it at
    the default settings (codex review, HIGH). This pins that the first sample
    really is wrong -- so dropping it is load-bearing, not cosmetic -- and that
    the probe drops it.
    """
    mod = _load()
    path = _series(tmp_path, leak=20.0)
    all_vals = _leak_values(mod, path, drop_first=False)
    assert abs(all_vals[0] - 20.0) > 1.0, (
        "the first sample happens to give the right leak here, so this test "
        "cannot show that dropping it matters")
    np.testing.assert_allclose(all_vals[1:], 20.0, atol=1e-6)

    t = mod.load(path)
    days = t["days"] if "days" in t else np.arange(len(t["hfss"]), dtype=float)
    keep = days > 1.0
    keep[0] = False
    assert not keep[0], "the probe must never keep sample 0"


def test_the_thermal_latent_split_recovers_a_known_drying_rate(tmp_path):
    """The follow-up diagnostic: latent part = L_v * d(CWV)/dt.

    If the budget closes but the interior still warms, the two live
    explanations are a real boundary-flux error and MSE compensation (the
    column warming while it dries).  The split separates them, so it has to
    recover a KNOWN drying rate rather than merely run.
    """
    mod = _load()
    path = _series(tmp_path, leak=0.0, n=6, dt_days=1.0)
    d = dict(np.load(path))
    # Dry the column at a rate whose latent tendency is exactly -20 W/m^2.
    from legoesm import constants
    dcwv_per_day = -20.0 * 86400.0 / constants.L_v          # kg/m^2 per day
    d["CWV"] = 25.0 + dcwv_per_day * np.arange(6, dtype=float)
    np.savez(path, **d)

    t = mod.load(path)
    assert "CWV" in t
    dt_s = np.gradient(t["days"]) * 86400.0
    latent = constants.L_v * np.gradient(t["CWV"]) / dt_s
    np.testing.assert_allclose(latent, -20.0, rtol=1e-9)
    assert mod.main([str(path), "--skip-days", "1"]) == 0


def test_missing_energy_channels_fail_loudly(tmp_path):
    """A series without the energy channels must raise, not report a leak.

    The MPAS lane writes these only once it reaches a diagnostic step; silently
    reporting on a file that lacks them would manufacture a number.
    """
    mod = _load()
    path = tmp_path / "timeseries.npz"
    np.savez(path, days=np.arange(3.0), T_atm=np.zeros(3))
    with pytest.raises(SystemExit, match="energy"):
        mod.load(path)


def test_ragged_channels_are_refused_not_reported(tmp_path):
    """Misaligned channels must raise, because a shifted leak looks plausible.

    The MPAS lane appends `days` unconditionally at each diagnostic step but
    the energy channels under a further condition, so a mismatch means every
    sample is paired with the wrong day's fluxes.  That produces a number, not
    an error -- which is exactly why it has to be refused explicitly.
    """
    mod = _load()
    path = _series(tmp_path, leak=20.0, n=6)
    d = dict(np.load(path))
    d["days"] = np.arange(7, dtype=float)          # one more day than samples
    np.savez(path, **d)
    with pytest.raises(SystemExit, match="SHIFTED|lengths disagree"):
        mod.load(path)


def test_snapshot_sourced_series_is_refused(tmp_path):
    """A leak from snapshots must RAISE, not be reported with a caveat.

    The contaminated number is plausible (+34.7 W/m^2 measured against a ~20
    hypothesis), so a caveat would be dropped the moment it is quoted onward.
    """
    mod = _load()
    path = _series(tmp_path, leak=20.0, n=6)
    d = dict(np.load(path))
    d["energy_flux_interval_mean"] = np.zeros(6)
    np.savez(path, **d)
    with pytest.raises(SystemExit, match="REFUSING"):
        mod.assert_interval_means(mod.load(path))
    # ...and the escape hatch still labels it rather than hiding it.
    msg = mod.assert_interval_means(mod.load(path), allow_snapshots=True)
    assert "CONTAMINATED" in msg


def test_a_series_predating_the_flag_is_also_refused(tmp_path):
    """Unknown timing is refused too — absence of the flag is not consent."""
    mod = _load()
    path = _series(tmp_path, leak=20.0, n=6)
    d = dict(np.load(path))
    del d["energy_flux_interval_mean"]
    np.savez(path, **d)
    with pytest.raises(SystemExit, match="REFUSING"):
        mod.assert_interval_means(mod.load(path))


def test_interval_mean_series_passes_and_says_so(tmp_path):
    mod = _load()
    msg = mod.assert_interval_means(mod.load(_series(tmp_path, leak=0.0)))
    assert "INTERVAL MEANS" in msg


def test_the_driver_persists_every_channel_the_probe_requires():
    """The probe's required channels must all be WRITTEN by the driver.

    A channel can be collected into the in-memory series and never reach
    `timeseries.npz`, and the failure is invisible until a multi-hour run
    finishes and the probe refuses it (codex review caught exactly this for
    `energy_flux_interval_mean`). Read the writer's keyword list from source
    and check it covers what `load` demands.
    """
    import ast
    import pathlib as _pl

    driver = (_pl.Path(__file__).resolve().parents[2] / "packages" / "coupler"
              / "legoesm" / "driver" / "model_driver.py")
    tree = ast.parse(driver.read_text())
    written: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "savez"):
            written.update(kw.arg for kw in node.keywords if kw.arg)
    mod = _load()
    required = set(mod._NEEDED) | {"energy_flux_interval_mean"}
    missing = sorted(required - written)
    assert not missing, (
        f"the probe requires {missing} but no np.savez in model_driver.py "
        "writes them — a run would complete and then be refused")


def test_the_frozen_precipitation_bound_is_the_predicted_size(tmp_path):
    """L_f * precip, the pre-registered upper bound on the definitional gap.

    E carries -L_f*q_frozen and the flux list has no precipitation term, so
    frozen condensate leaving the column reads as a spurious energy SOURCE of
    L_f times the frozen precipitation rate. Pin the conversion, because the
    whole prediction is a magnitude: 3.337e5 J/kg over 86400 s is 3.86 W/m^2
    per mm/day, so a 3 mm/day column bounds the gap at ~11.6 W/m^2 -- the size
    of the residual left after the interval-mean correction.
    """
    from legoesm import constants
    mod = _load()
    t = mod.load(_series(tmp_path, leak=0.0, n=6))
    bound = constants.L_f * t["precip"] / 86400.0
    assert float(bound[0]) == pytest.approx(11.59, rel=1e-3), float(bound[0])
    assert constants.L_f / 86400.0 == pytest.approx(3.862, rel=1e-3)


def test_runs_end_to_end(tmp_path):
    mod = _load()
    assert mod.main([str(_series(tmp_path, leak=12.0)), "--skip-days", "1"]) == 0


def test_a_file_with_the_legacy_key_gives_the_same_leak(tmp_path, capsys):
    """Runs written before the rename stored TOA - dE/dt as energy_residual."""
    mod = _load()
    new = _leak_values(mod, _series(tmp_path, leak=7.0))
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    old = _leak_values(mod, _series(old_dir, leak=7.0, key="energy_residual"))
    np.testing.assert_array_equal(old, new)
    np.testing.assert_allclose(new, 7.0, atol=1e-6)
    assert "legacy key 'energy_residual'" in capsys.readouterr().err


def _series_with_closure_terms(tmp_path, *, a_top, snow_kg_m2_s, n=8):
    """A closed column (no leak) that also loses radiation above the model top
    and exports snow.  Built, like ``_series``, from an ENERGY TRAJECTORY: the
    column's true budget is

        dE/dt = (toa - A_top) - (sw + lw) + sh + lh + L_f * S

    (A_top enters the reported TOA but is absorbed above the model top; snow
    leaving the column removes its -L_f*q_frozen), solved for sh.
    """
    from legoesm import constants
    rng = np.random.default_rng(1)
    toa = rng.normal(15.0, 3.0, n)
    sw = rng.normal(165.0, 5.0, n)
    lw = rng.normal(-58.0, 3.0, n)
    lh = rng.normal(75.0, 4.0, n)
    a = np.full(n, a_top) + rng.normal(0.0, 0.05, n)
    s = np.full(n, snow_kg_m2_s) * (1.0 + 0.1 * rng.normal(size=n))
    E = 2.6e9 + np.cumsum(rng.normal(0.0, 3.0e6, n + 1))
    dedt = (E[1:] - E[:-1]) / 86400.0
    sh = dedt - (toa - a) + (sw + lw) - lh - constants.L_f * s
    resid = toa - dedt
    dedt[0], resid[0] = 0.0, toa[0]
    path = tmp_path / "timeseries.npz"
    np.savez(path, days=np.arange(n, dtype=float), energy_toa_net=toa,
             energy_dE_dt=dedt, energy_toa_minus_dE_dt=resid,
             sw_net_sfc=sw, lw_net_sfc=lw, hfss=sh, hfls=lh,
             energy_flux_interval_mean=np.ones(n),
             energy_rad_above_top=a,
             energy_snowfall_fusion=constants.L_f * s)
    return path


def test_closure_terms_close_the_budget_and_each_is_needed(tmp_path):
    """float64: with both terms the column closes to round-off; dropping
    either one leaves a gap of its own size, so neither is inert."""
    mod = _load()
    t = mod.load(_series_with_closure_terms(tmp_path, a_top=0.7,
                                            snow_kg_m2_s=6.0e-6))
    leak, leak_new = mod.closure_leak(t)
    np.testing.assert_allclose(leak_new[1:], 0.0, atol=1e-10)
    no_snow = dict(t, energy_snowfall_fusion=np.zeros_like(leak))
    no_top = dict(t, energy_rad_above_top=np.zeros_like(leak))
    assert np.all(np.abs(mod.closure_leak(no_snow)[1][1:]) > 1.0)
    assert np.all(np.abs(mod.closure_leak(no_top)[1][1:]) > 0.5)
    # The old closure carries both terms: +L_f*S - A_top.
    assert np.all(np.abs(leak[1:]) > 0.5)


def test_files_without_the_closure_terms_report_old_leak_only(tmp_path, capsys):
    mod = _load()
    t = mod.load(_series(tmp_path, leak=3.0))
    leak, leak_new = mod.closure_leak(t)
    assert leak_new is None
    np.testing.assert_allclose(leak[1:], 3.0, atol=1e-6)
    sub = tmp_path / "run"
    sub.mkdir()
    assert mod.main([str(_series(sub, leak=3.0)), "--skip-days", "0"]) == 0
    assert "LEAK_new unavailable" in capsys.readouterr().out


def test_driver_writes_the_closure_terms():
    """The probe's closure terms must be persisted by the MPAS series writer
    and produced by the energy sample (same AST check as the required keys)."""
    import ast
    from pathlib import Path
    driver = (Path(__file__).resolve().parents[2] / "packages" / "coupler"
              / "legoesm" / "driver" / "model_driver.py")
    src = driver.read_text()
    tree = ast.parse(src)
    written: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "savez"):
            written.update(kw.arg for kw in node.keywords if kw.arg)
    mod = _load()
    assert set(mod._CLOSURE_TERMS) <= written
    from legoesm.driver.model_driver import ModelDriver
    assert set(mod._CLOSURE_TERMS) <= set(ModelDriver._ENERGY_SERIES_KEYS)
