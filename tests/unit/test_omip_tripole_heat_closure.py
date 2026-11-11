"""Known-answer control for the tripole heat-closure gate.

A synthetic two-snapshot pair whose heat-content change is exactly the
applied surface + geothermal heat must close to rounding; a doubled q_net
must leave the planted residual; a step-count mismatch must be fatal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "validate" / "ocean_fidelity"))

from omip_tripole_heat_closure import (  # noqa: E402
    build_z_coord, close_windows, step_month_weights)

RHO, CP = 1025.0, 3994.0
NY, NX, H = 4, 5, 1000.0
DT_WIN = 5 * 86400.0
Q, GEO = 30.0, 0.1          # W/m2 surface, W/m2 seafloor


def _zc():
    from legoesm.ocean.vertical import create_ocean_z_star
    return create_ocean_z_star(n_levels=3, H_max=H, dz_surface=200.0, dz_deep=400.0)


def _snap(day, step, T, q=None):
    s = {"T": np.full((NY, NX, 3), T), "eta": np.zeros((NY, NX)),
         "H_bathy": np.full((NY, NX), H), "land_mask": np.ones((NY, NX)),
         "cell_area": np.full((NY, NX), 1.0e10), "time_days": np.float64(day),
         "step": np.float64(step)}
    if q is not None:
        s.update(q_net_mean=np.full((NY, NX), q),
                 flux_mean_n_steps=np.float64(2880),
                 flux_mean_window_s=np.float64(DT_WIN))
    return s


def _pair(q_saved):
    dT = (Q + GEO) * DT_WIN / (RHO * CP * H)     # exact column warming
    a = _snap(5.0, 2880, 10.0)
    b = _snap(10.0, 5760, 10.0 + dT, q=q_saved)
    return [a, b]


def _close(snaps):
    return close_windows(snaps, lambda s: _zc(), RHO, CP, GEO, None, 3.34e5)


def test_known_answer_closes_to_rounding():
    [row] = _close(_pair(Q))
    assert abs(row["residual_Wm2"]) < 1e-9 * (Q + GEO)
    assert row["Q_geothermal_W"] == pytest.approx(GEO * NY * NX * 1.0e10, rel=1e-15)


def test_doubled_qnet_leaves_the_planted_residual():
    [row] = _close(_pair(2.0 * Q))
    assert row["residual_Wm2"] == pytest.approx(-Q, rel=1e-9)


def test_step_count_mismatch_is_fatal():
    snaps = _pair(Q)
    snaps[1]["flux_mean_n_steps"] = np.float64(2879)
    with pytest.raises(SystemExit, match="accumulator holds"):
        _close(snaps)


def test_missing_flux_means_is_fatal():
    snaps = _pair(Q)
    del snaps[1]["q_net_mean"]
    with pytest.raises(SystemExit, match="flux-accumulate"):
        _close(snaps)


def test_partial_bottom_and_nonzero_eta_close_through_the_run_geometry():
    """Thickness via the run's constructors (e3t ladder -> partial bottom cell,
    H=800 in a 1000 m ladder -> eta stretch) against the analytic form
    h_k = h_partial_k * (H + eta) / H, with eta and every level's T changing."""
    e3t = np.array([100.0, 300.0, 600.0])
    h_partial = np.array([100.0, 300.0, 400.0])
    Hb = 800.0
    a = _snap(5.0, 2880, 10.0)
    b = _snap(10.0, 5760, 10.0, q=Q)
    a["T"][...] = np.array([12.0, 8.0, 3.0])
    b["T"][...] = np.array([12.3, 8.1, 3.7])          # bottom cell moves most
    for s, eta in ((a, 0.5), (b, -0.2)):
        s["H_bathy"][:] = Hb
        s["eta"][:] = eta
    def content(s):
        h = h_partial * (Hb + s["eta"][0, 0]) / Hb
        return RHO * CP * float((s["T"][0, 0] * h).sum())
    q_exact = (content(b) - content(a)) / DT_WIN       # per m2 of column
    b["q_net_mean"][:] = q_exact
    zc = lambda s: build_z_coord(e3t, None, s["H_bathy"])[1]  # noqa: E731
    [row] = close_windows([a, b], zc, RHO, CP, 0.0, None, 3.34e5)
    assert abs(row["residual_Wm2"]) < 1e-9 * abs(q_exact)


def test_duration_must_match_the_snapshot_span():
    snaps = _pair(Q)
    snaps[1]["flux_mean_window_s"] = np.float64(DT_WIN - 1800.0)
    with pytest.raises(SystemExit, match="snapshot span"):
        _close(snaps)


def test_isf_month_follows_the_driver_rule():
    """Driver: month of step s = searchsorted(month_end, s*dt/86400, right);
    a step ending exactly on day 31 already belongs to February."""
    w = step_month_weights(31 * 48 - 2, 31 * 48 + 2, 1800.0)
    assert w[0] == pytest.approx(0.25) and w[1] == pytest.approx(0.75)


def _run_dir(tmp_path, q_saved, extra_flags="", z_iface=(100.0, 400.0)):
    """Minimal run directory the CLI reads: manifest, vertical file, run.log,
    two day snapshots on a 3-level 1000 m ladder (flat bottom, eta = 0)."""
    import json

    import xarray as xr
    e3t = np.array([100.0, 300.0, 600.0])
    xr.Dataset({"e3t_1d": ("z", e3t)}).to_netcdf(tmp_path / "domain_cfg.nc")
    cmd = ("scripts/run/run_omip_core2.py --grid tripole --partial-cell "
           "--nemo-vertical --snapshot-every-days 5 --geothermal "
           f"--geothermal-flux-wm2 {GEO} " + extra_flags)
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "run": {"command_line": cmd}, "legoESM": {"commit": "x"},
        "config": {"resolved_config": {"runtime_config": {
            "constants": {"rho_0": RHO, "c_sw": CP}}}}}))
    (tmp_path / "run.log").write_text("")
    a, b = _pair(q_saved)
    for tag, s in (("day0005", a), ("day0010", b)):
        s["z_interface_ref"] = np.asarray(z_iface)
        np.savez(tmp_path / f"snapshot_{tag}.npz", **s)
    return tmp_path


def _cli(run_dir, gate):
    from omip_tripole_heat_closure import main
    return main(["--run-dir", str(run_dir), "--domain-cfg",
                 str(run_dir / "domain_cfg.nc"), "--max-residual-wm2", str(gate)])


def test_cli_passes_a_closing_run(tmp_path):
    assert _cli(_run_dir(tmp_path, Q), 1e-6) == 0


def test_cli_fails_a_planted_residual(tmp_path):
    assert _cli(_run_dir(tmp_path, 2.0 * Q), 1e-6) == 1


def test_cli_refuses_an_unclosed_heat_source(tmp_path):
    with pytest.raises(SystemExit, match="freeze-floor"):
        _cli(_run_dir(tmp_path, Q, extra_flags="--freeze-floor"), 1e-6)


def test_cli_refuses_a_geometry_mismatch(tmp_path):
    with pytest.raises(SystemExit, match="z_interface_ref"):
        _cli(_run_dir(tmp_path, Q, z_iface=(100.0, 450.0)), 1e-6)


def test_cyclic_overlap_columns_are_counted_once():
    from omip_tripole_heat_closure import drop_cyclic_overlap
    a, b = _pair(Q)
    for s in (a, b):
        s["T"][:, -1] = s["T"][:, 1]
        s["T"][:, 0] = s["T"][:, -2]
    b["T"][:, 0] += 5.0                     # a change seen only in a halo copy
    [row] = _close([drop_cyclic_overlap(a), drop_cyclic_overlap(b)])
    assert abs(row["residual_Wm2"]) < 1e-9 * (Q + GEO)


def _salty_pair(ice_gain_kg_per_m2):
    """Ocean S drops by exactly the salt that moves into 1 m of ice."""
    from legoesm import constants
    a, b = _pair(Q)
    for s in (a, b):
        s["S"] = np.full_like(s["T"], 35.0)
        s["ice_concentration"] = np.full((NY, NX), 0.5)
        s["ice_thickness"] = np.full((NY, NX), 1.0)
        s["ice_salinity"] = np.full((NY, NX), 4.0)
    dS_ice = 1.0e3 * ice_gain_kg_per_m2 / (constants.rho_ice * 0.5 * 1.0)
    b["ice_salinity"] += dS_ice
    b["S"] -= 1.0e3 * ice_gain_kg_per_m2 / (RHO * H)
    return a, b


def test_salt_moving_into_ice_is_conserved():
    [row] = _close(list(_salty_pair(0.2)))
    assert abs(row["salt_rel_change"]) < 1e-14


def test_salt_created_in_ice_is_flagged():
    a, b = _salty_pair(0.2)
    b["ice_salinity"] += 1.0                  # ice salt with no ocean debit
    [row] = _close([a, b])
    assert row["salt_rel_change"] > 1e-6
