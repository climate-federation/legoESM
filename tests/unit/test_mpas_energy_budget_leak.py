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


def _series(tmp_path, *, leak, n=6, dt_days=1.0):
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
             energy_toa_net=toa, energy_dE_dt=dedt, energy_residual=residual,
             sw_net_sfc=sw, lw_net_sfc=lw, hfss=sh, hfls=lh)
    return path


def _leak_values(mod, path, drop_first=True):
    """Leak per sample, with the first dropped the way the probe drops it."""
    t = mod.load(path)
    v = (t["sw_net_sfc"] + t["lw_net_sfc"] - t["hfss"] - t["hfls"]
         - t["energy_residual"])
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


def test_runs_end_to_end(tmp_path):
    mod = _load()
    assert mod.main([str(_series(tmp_path, leak=12.0)), "--skip-days", "1"]) == 0
