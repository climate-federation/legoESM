"""Conservation diagnostics on the lightweight (MPAS / spectral) lane.

Regression cover for the 2026-08-04 output audit finding: ``_run_mpas``
populated only state-derived timeseries channels, so ``timeseries.npz`` carried
all-NaN ``energy_residual`` / ``moisture_residual`` (0/155 finite samples on a
completed 365-day AMIP chain).  Because ``validate_amip_run.py`` skipped a NaN
residual, BOTH conservation checks were silently inert and the run reported
"ALL CHECKS PASSED".  Separately the writer OVERWROTE ``timeseries.npz`` per
restart link, so only the final link's samples survived.

Covers:
- ``_merge_timeseries_chunks``: restart links concatenate (not clobber), later
  links win on overlapping days, schema drift stays index-aligned;
- ``ModelDriver._mpas_timeseries_extras``: fixed key set, real residuals from
  the shared budget trackers, NaN (never a fabricated zero) when a channel is
  unmeasurable;
- ``validate_amip_run``: an unmeasured residual FAILS instead of skipping.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.driver.model_driver import (  # noqa: E402
    ModelDriver,
    _merge_timeseries_chunks,
)

_VALIDATOR = (Path(__file__).resolve().parents[2]
              / "scripts" / "validate" / "validate_amip_run.py")

_CHANNELS = ("T_low", "sst", "sic", "precip", "sw_up_toa", "lw_up_toa",
             "sw_net_sfc", "lw_net_sfc", "energy_residual",
             "moisture_residual")


# ----------------------------------------------------------------------
# _merge_timeseries_chunks — restart chaining
# ----------------------------------------------------------------------

def _write_chunk(d: Path, days, **cols):
    days = np.asarray(days, dtype=np.float64)
    payload = {"days": days}
    for k, v in cols.items():
        payload[k] = np.asarray(v, dtype=np.float64)
    p = d / f"chunk_day{days[0]:09.2f}.npz"
    np.savez(p, **payload)
    return p


def test_merge_concatenates_restart_links(tmp_path):
    """Two links must produce one continuous record, not the last link only."""
    _write_chunk(tmp_path, [1.0, 2.0, 3.0], T_atm=[280.0, 281.0, 282.0])
    _write_chunk(tmp_path, [4.0, 5.0], T_atm=[283.0, 284.0])

    out = _merge_timeseries_chunks(sorted(tmp_path.glob("chunk_day*.npz")))

    np.testing.assert_array_equal(out["days"], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(
        out["T_atm"], [280.0, 281.0, 282.0, 283.0, 284.0])


def test_merge_later_link_wins_on_overlap(tmp_path):
    """A crash-restart from an older checkpoint re-runs days; the LATER link
    is the trajectory the chain continued from, so it must win."""
    _write_chunk(tmp_path, [1.0, 2.0, 3.0], T_atm=[280.0, 281.0, 282.0])
    _write_chunk(tmp_path, [2.0, 3.0, 4.0], T_atm=[291.0, 292.0, 293.0])

    out = _merge_timeseries_chunks(sorted(tmp_path.glob("chunk_day*.npz")))

    np.testing.assert_array_equal(out["days"], [1.0, 2.0, 3.0, 4.0])
    # day 1 from link A; days 2-4 from link B (not A's 281/282).
    np.testing.assert_allclose(out["T_atm"], [280.0, 291.0, 292.0, 293.0])


def test_merge_days_are_sorted_and_monotonic(tmp_path):
    _write_chunk(tmp_path, [10.0, 11.0], T_atm=[1.0, 2.0])
    _write_chunk(tmp_path, [1.0, 2.0], T_atm=[3.0, 4.0])
    out = _merge_timeseries_chunks(sorted(tmp_path.glob("chunk_day*.npz")))
    assert np.all(np.diff(out["days"]) > 0), out["days"]


def test_merge_fills_missing_key_with_nan_keeping_alignment(tmp_path):
    """A chunk from an older code version lacks the new channel; the column
    must be NaN over that span, never shortened (which would misalign days)."""
    _write_chunk(tmp_path, [1.0, 2.0], T_atm=[280.0, 281.0])
    _write_chunk(tmp_path, [3.0], T_atm=[282.0], energy_residual=[2.5])

    out = _merge_timeseries_chunks(sorted(tmp_path.glob("chunk_day*.npz")))

    assert out["energy_residual"].size == out["days"].size == 3
    assert np.all(np.isnan(out["energy_residual"][:2]))
    assert out["energy_residual"][2] == pytest.approx(2.5)


def test_merge_empty_input_returns_empty_dict(tmp_path):
    assert _merge_timeseries_chunks([]) == {}


def test_merge_skips_corrupt_chunk(tmp_path):
    _write_chunk(tmp_path, [1.0], T_atm=[280.0])
    bad = tmp_path / "chunk_day000002.00.npz"
    bad.write_bytes(b"not an npz")
    out = _merge_timeseries_chunks(sorted(tmp_path.glob("chunk_day*.npz")))
    np.testing.assert_array_equal(out["days"], [1.0])


# ----------------------------------------------------------------------
# _mpas_timeseries_extras — the channels themselves
# ----------------------------------------------------------------------

class _Sigma:
    """Minimal pure-sigma vertical coordinate for the extras path."""

    def __init__(self, nlev):
        self.dsigma = jnp.full((nlev,), 1.0 / nlev)
        self.sigma_full = (jnp.arange(nlev) + 0.5) / nlev

    def layer_thickness_dp(self, p_s):
        return jnp.asarray(p_s)[..., None] * self.dsigma

    def pressure_at_full(self, p_s):
        return jnp.asarray(p_s)[..., None] * self.sigma_full


def _fake_driver(*, n_cells=8, nlev=4, with_fluxes=True, moist=True,
                 q_scale=1.0):
    """A duck-typed stand-in exposing exactly what ``_mpas_timeseries_extras``
    reads.  Avoids building a real MPAS mesh + dycore for a host-side
    diagnostic."""
    T = jnp.full((n_cells, nlev), 280.0)
    p_s = jnp.full((n_cells,), constants.p_ref)
    phis = jnp.zeros((n_cells,))
    tracers = ({"q_v": SimpleNamespace(data=jnp.full((n_cells, nlev),
                                                     0.005 * q_scale))}
               if moist else None)
    state = SimpleNamespace(
        T=SimpleNamespace(data=T),
        p_s=SimpleNamespace(data=p_s),
        phis=SimpleNamespace(data=phis),
        u=SimpleNamespace(data=jnp.zeros((n_cells, nlev))),
        tracers=tracers,
    )
    # sfc_diag slot contract: 0 sw_net_sfc, 1 lw_net_sfc, 2 precip,
    # 3 lw_up_toa, 4 sw_up_toa, 5 sw_down_toa, 6 hfss, 7 hfls.
    def _f(v):
        return SimpleNamespace(data=jnp.full((n_cells,), v))
    sfc = ([_f(150.0), _f(-60.0), _f(3.0 / 86400.0), _f(240.0), _f(100.0),
            _f(340.0), _f(20.0), _f(80.0)] if with_fluxes else None)

    from legoesm.diagnostics.energy_budget import (
        EnergyBudgetTracker, MoistureBudgetTracker,
    )
    drv = SimpleNamespace(
        state=state,
        sigma=_Sigma(nlev),
        grid=None,
        model=SimpleNamespace(_sfc_diag=sfc),
        _mpas_sfc_accum=None,
        get_sst_sic=lambda day: (jnp.full((n_cells,), 290.0),
                                 jnp.zeros((n_cells,))),
        _mpas_energy_tracker=EnergyBudgetTracker(),
        _mpas_moisture_tracker=MoistureBudgetTracker(),
    )
    # ``_mpas_sfc_slot`` is module-level, so the extras method reads the slots
    # off ``drv.model._sfc_diag`` directly — nothing to bind or stub here.
    # Cell-collocated winds: the real lane reconstructs edge-normal u via
    # Perot.  Zero wind makes the reconstruction a no-op we can stub.
    drv._perot = (jnp.zeros((n_cells, nlev)), jnp.zeros((n_cells, nlev)))
    return drv


def _extras(drv, day, monkeypatch):
    """Call the real method against the fake driver, stubbing only the
    Voronoi wind reconstruction (needs a real mesh)."""
    import legoesm.grids.voronoi as voro
    monkeypatch.setattr(voro, "reconstruct_cell_velocity",
                        lambda u, grid: drv._perot, raising=False)
    return ModelDriver._mpas_timeseries_extras(drv, day)


def test_extras_returns_fixed_key_set(monkeypatch):
    """Lists must stay index-aligned with ``days`` -> every key every call."""
    drv = _fake_driver()
    out = _extras(drv, 1.0, monkeypatch)
    assert set(out) == set(_CHANNELS)


def test_extras_populates_the_channels_that_were_all_nan(monkeypatch):
    """The exact 8 non-residual channels the audit found 0/155 finite."""
    drv = _fake_driver()
    out = _extras(drv, 1.0, monkeypatch)
    for k in ("T_low", "sst", "sic", "precip", "sw_up_toa", "lw_up_toa",
              "sw_net_sfc", "lw_net_sfc"):
        assert np.isfinite(out[k]), f"{k} still NaN"
    assert out["T_low"] == pytest.approx(280.0)
    assert out["sst"] == pytest.approx(290.0)
    assert out["precip"] == pytest.approx(3.0)          # kg/m2/s -> mm/day
    assert out["sw_up_toa"] == pytest.approx(100.0)
    assert out["lw_up_toa"] == pytest.approx(240.0)


def test_extras_first_sample_residuals_are_finite(monkeypatch):
    """The trackers report 0.0 (cold start), not NaN — a NaN would now be a
    validator failure."""
    drv = _fake_driver()
    out = _extras(drv, 1.0, monkeypatch)
    assert np.isfinite(out["energy_residual"])
    assert np.isfinite(out["moisture_residual"])


def test_extras_energy_residual_tracks_toa_imbalance(monkeypatch):
    """Second sample on an UNCHANGED state: dE/dt = 0, so the residual must
    equal the prescribed TOA net = sw_down - sw_up - lw_up = 340-100-240 = 0.
    Re-run with a deliberately imbalanced TOA to prove it is not hardwired."""
    drv = _fake_driver()
    _extras(drv, 1.0, monkeypatch)
    out = _extras(drv, 2.0, monkeypatch)
    assert out["energy_residual"] == pytest.approx(0.0, abs=1e-9)

    # Now break the balance: lw_up_toa 240 -> 200 gives TOA net = +40 W/m2.
    drv2 = _fake_driver()
    drv2.model._sfc_diag[3] = SimpleNamespace(
        data=jnp.full((8,), 200.0))
    _extras(drv2, 1.0, monkeypatch)
    out2 = _extras(drv2, 2.0, monkeypatch)
    assert out2["energy_residual"] == pytest.approx(40.0, abs=1e-6)


def test_extras_moisture_residual_tracks_e_minus_p(monkeypatch):
    """Steady column (dW/dt = 0): residual = E - P.  E = hfls/L_v*86400 with
    hfls = 80 W/m2; P = 3 mm/day."""
    drv = _fake_driver()
    _extras(drv, 1.0, monkeypatch)
    out = _extras(drv, 2.0, monkeypatch)
    expected_E = 80.0 / constants.L_v * 86400.0
    assert out["moisture_residual"] == pytest.approx(expected_E - 3.0,
                                                     abs=1e-6)


def test_extras_moisture_residual_sees_a_storage_change(monkeypatch):
    """A column that GAINS vapour between samples must move the residual by
    -dW/dt — proves dW/dt is really in the closure, not dropped."""
    drv = _fake_driver()
    _extras(drv, 1.0, monkeypatch)
    steady = _fake_driver()
    _extras(steady, 1.0, monkeypatch)
    out_steady = _extras(steady, 2.0, monkeypatch)
    # Double the vapour before the second sample.
    drv.state.tracers["q_v"] = SimpleNamespace(
        data=drv.state.tracers["q_v"].data * 2.0)
    out_wet = _extras(drv, 2.0, monkeypatch)
    assert out_wet["moisture_residual"] < out_steady["moisture_residual"] - 1.0


def test_extras_residual_is_nan_without_radiation(monkeypatch):
    """No flux slots -> the closure is unmeasurable.  It must be NaN, NOT a
    fabricated 0.0 that would read as perfect conservation."""
    drv = _fake_driver(with_fluxes=False)
    out = _extras(drv, 1.0, monkeypatch)
    assert np.isnan(out["energy_residual"])
    assert np.isnan(out["moisture_residual"])
    assert np.isnan(out["sw_up_toa"])


def test_extras_moisture_residual_is_nan_on_a_dry_run(monkeypatch):
    drv = _fake_driver(moist=False)
    out = _extras(drv, 1.0, monkeypatch)
    assert np.isnan(out["moisture_residual"])
    assert np.isnan(out["energy_residual"])   # needs q_v for column energy
    assert np.isfinite(out["sw_up_toa"])      # fluxes still published


def test_extras_never_raises_on_a_broken_state(monkeypatch):
    """A diagnostic failure must not abort the run; it leaves NaN, which the
    validator now treats as a FAILURE."""
    drv = _fake_driver()
    drv.state = None
    out = ModelDriver._mpas_timeseries_extras(drv, 1.0)
    assert set(out) == set(_CHANNELS)
    assert all(np.isnan(v) for v in out.values())


# ----------------------------------------------------------------------
# validate_amip_run — fail loud
# ----------------------------------------------------------------------

def _run_validator(run_dir: Path):
    p = subprocess.run(
        [sys.executable, str(_VALIDATOR), str(run_dir)],
        capture_output=True, text=True,
    )
    return p.returncode, p.stdout + p.stderr


def _make_run(tmp_path: Path, *, energy, moisture, radiation="rrtmg",
              cwv=25.0, n=10):
    d = tmp_path / "run"
    d.mkdir(exist_ok=True)
    ones = np.ones(n)
    np.savez(
        d / "timeseries.npz",
        days=np.arange(1.0, n + 1.0),
        T_atm=280.0 * ones,
        max_wind=40.0 * ones,
        dry_mass_ps=constants.p_ref * ones,
        CWV=cwv * ones,
        energy_residual=np.asarray(energy, dtype=np.float64) * ones,
        moisture_residual=np.asarray(moisture, dtype=np.float64) * ones,
    )
    (d / "results.txt").write_text(
        "legoESM AMIP run\n"
        "Grid: mpas 5 / L30, dt=75.0s, 10 days\n"
        f"Radiation: {radiation}\n"
        "Status: COMPLETED\n\n"
        "Wall time: 1.0s\n\n"
    )
    return d


def test_validator_fails_on_all_nan_energy_residual(tmp_path):
    """THE regression: this is the cc_on/cc_off shape.  It used to print
    'ALL CHECKS PASSED'."""
    d = _make_run(tmp_path, energy=np.nan, moisture=np.nan)
    rc, out = _run_validator(d)
    assert rc == 1, out
    assert "NEVER MEASURED" in out
    assert "toa_residual_unmeasured" in out
    assert "moisture_residual_unmeasured" in out
    assert "ALL CHECKS PASSED" not in out


def test_validator_passes_when_residuals_are_measured_and_small(tmp_path):
    """Non-vacuity control: the same run with real, small residuals passes —
    so the new check is not just failing everything."""
    d = _make_run(tmp_path, energy=1.5, moisture=0.2)
    rc, out = _run_validator(d)
    assert rc == 0, out
    assert "NEVER MEASURED" not in out


def test_validator_fails_on_a_large_measured_energy_residual(tmp_path):
    """The check must still catch the thing it was always meant to catch."""
    d = _make_run(tmp_path, energy=5000.0, moisture=0.2)
    rc, out = _run_validator(d)
    assert rc == 1, out
    assert "toa_residual" in out


def test_validator_skips_residual_only_without_radiation(tmp_path):
    """A genuinely radiation-free run may legitimately have no closure."""
    d = _make_run(tmp_path, energy=np.nan, moisture=0.2, radiation="none")
    rc, out = _run_validator(d)
    assert rc == 0, out
    assert "no radiation" in out


def _make_run_without_residual_channels(tmp_path, name, results_txt):
    d = tmp_path / name
    d.mkdir()
    np.savez(
        d / "timeseries.npz",
        days=np.arange(1.0, 6.0),
        T_atm=280.0 * np.ones(5),
        max_wind=40.0 * np.ones(5),
        dry_mass_ps=constants.p_ref * np.ones(5),
        CWV=25.0 * np.ones(5),
    )
    (d / "results.txt").write_text(results_txt)
    return d


def test_validator_fails_when_residual_channel_is_absent(tmp_path):
    """A run that DECLARES radiation but ships no residual channel is
    unmeasured, exactly like an all-NaN one."""
    d = _make_run_without_residual_channels(
        tmp_path, "run2",
        "Grid: mpas 5 / L30\nRadiation: rrtmg\nStatus: COMPLETED\n")
    rc, out = _run_validator(d)
    assert rc == 1, out
    assert "ABSENT" in out
    assert "NEVER MEASURED" in out


def test_validator_skips_absent_channel_when_config_undeterminable(tmp_path):
    """A minimal/foreign file with no ``Radiation:`` line cannot be judged.
    It must SKIP with an explicit "cannot evaluate" — an honest non-answer,
    not a silent pass and not a false failure."""
    d = _make_run_without_residual_channels(
        tmp_path, "run3", "Status: COMPLETED\n")
    rc, out = _run_validator(d)
    assert rc == 0, out
    assert "cannot evaluate" in out
    assert "NEVER MEASURED" not in out
