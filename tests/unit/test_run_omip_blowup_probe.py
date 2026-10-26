"""BLOWUP locator + memory probe in the OMIP driver (probe tooling, not physics)."""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_omip_probe_mod", _ROOT / "scripts" / "run" / "run_omip.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = ["run_omip.py"]
    spec.loader.exec_module(mod)
    return mod


def _field(a):
    return types.SimpleNamespace(data=np.asarray(a))


def test_report_nonfinite_names_the_cell(capsys):
    run_omip = _load()
    n_lat, n_lon, nz = 8, 6, 3
    T = np.full((n_lat, n_lon, nz), 10.0)
    T[5, 2, 1] = np.nan            # one bad wet cell in the northern half
    T[0, 0, 0] = np.inf            # on LAND: must be ignored by the mask
    mask = np.ones((n_lat, n_lon))
    mask[0, 0] = 0.0
    state = types.SimpleNamespace(T=_field(T), eta=_field(np.zeros((n_lat, n_lon))),
                                  land_mask=_field(mask),
                                  H_bathy=_field(np.full((n_lat, n_lon), 42.0)))
    grid = types.SimpleNamespace(lat_T=np.deg2rad(np.tile(np.linspace(-70, 70, n_lat)[:, None], (1, n_lon))),
                                 lon_T=np.deg2rad(np.tile(np.linspace(0, 300, n_lon)[None, :], (n_lat, 1))))
    run_omip._report_nonfinite(state, "latlon", grid)
    out = capsys.readouterr().out
    assert "nonfinite T cells=1" in out
    assert "(row=5, col=2, lev=1)" in out
    assert "lat=30.00 lon=120.00" in out and "depth=42.0m" in out
    assert "rows with nonfinite: 5..5 (n=1 of 8)" in out


def test_memprobe_is_silent_unless_enabled(capsys, monkeypatch):
    run_omip = _load()
    monkeypatch.delenv("LEGOESM_OMIP_MEMPROBE", raising=False)
    run_omip._memprobe("x")
    assert capsys.readouterr().out == ""
    import jax.numpy as jnp
    keep = jnp.ones((64, 64))       # a live array the probe must list
    monkeypatch.setenv("LEGOESM_OMIP_MEMPROBE", "1")
    run_omip._memprobe("after setup")
    out = capsys.readouterr().out
    assert "MEMPROBE after setup" in out and "top5:" in out and "(64, 64)" in out
    del keep


def test_zero_surface_fluxes_flag_round_trips():
    run_omip = _load()
    args = run_omip.parse_args(["--jra55-zero-surface-fluxes"])
    assert args.jra55_zero_surface_fluxes is True
    assert run_omip.parse_args([]).jra55_zero_surface_fluxes is False
