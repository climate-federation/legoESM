"""Transient solar threading on the SPECTRAL (gaussian) lane (2026-10-06).

The spectral full-physics loop built its daily forcing dict with ozone /
aerosol / GHG but never the solar channel, and the spectral radiation wrapper
never read forcing["tsi"] / forcing["solar_spectral_fraction"]: a
real solar file was INERT on this grid while the MPAS and compiled lanes
consumed it (tests/unit/test_mpas_solar_threading.py pins those).  Same
contract, same two checks, reader monkeypatched (offline, no NetCDF):

1. NEUTRALITY: solar_source="file" returning EXACTLY the configured
   S_0 equals solar_source="constant" (rtol 1e-15).
2. EFFECT: a halved TSI changes the integrated state.

Gray radiation + Louis turbulence so the loop takes the full-physics branch
(the dry gray legacy branch carries no forcing dict by design).
"""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
    OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

RES, NLEV, DT = 8, 8, 600.0
FOUR_STEPS_DAYS = 2401.0 / 86400.0


def _build_driver(tmpdir: str, solar_source: str,
                  turbulence: str = "louis", S_0: float = 1361.0) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=RES, nlev=NLEV),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="spectral",
                            dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=FOUR_STEPS_DAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence=turbulence, precision="fp64",
        distributed=False,
        solar_source=solar_source, S_0=S_0,
        solar_file=("synthetic.nc" if solar_source == "file" else ""),
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def _grid_t(d: ModelDriver) -> np.ndarray:
    return np.asarray(spectral_pe_to_grid(d.state, d.grid, d.sigma)["T"])


def _patch_solar(monkeypatch, tsi_fn):
    import legoesm.forcing.external as ext
    monkeypatch.setattr(
        ext, "get_solar_forcing_at_time",
        lambda config, day: {"tsi": float(tsi_fn(config)),
                             "solar_fraction_by_gpt": None})


# ("none", 1350.0): dry gray branch with an overridden S_0 -- the file
# returning cfg.S_0 must be neutral there too (codex/GLM 2026-10-08).
@pytest.mark.parametrize("turbulence,s0", [("louis", 1361.0), ("none", 1350.0)])
def test_file_tsi_equal_to_s0_matches_constant(monkeypatch, tmp_path,
                                               turbulence, s0):
    d_const = _build_driver(str(tmp_path / "const"), "constant", turbulence, s0)
    assert d_const.run() == "COMPLETED"
    _patch_solar(monkeypatch, lambda config: config.S_0)
    d_file = _build_driver(str(tmp_path / "file"), "file", turbulence, s0)
    assert d_file.run() == "COMPLETED"
    # The traced ratio tsi/S_0 == 1.0 is multiplied in, not constant-folded;
    # x*1.0 is exact, but the extra op changes XLA's fusion/reassociation of
    # the surrounding arithmetic and can move a few ulp (measured 1.1e-13 K
    # at ~300 K = 2 ulp, one cell of 3136, on both this lane and the MPAS
    # lane on Ginsburg CPUs); contract: no physical change, rtol 1e-15.
    np.testing.assert_allclose(
        _grid_t(d_file), _grid_t(d_const), rtol=1e-15, atol=0.0,
        err_msg="file-mode TSI == S_0 must equal constant mode (rtol 1e-15)")


# "none" = the dry gray legacy branch (host-side daily_mean_insolation, no
# forcing dict); it ignored the solar file until 2026-10-08.
@pytest.mark.parametrize("turbulence", ["louis", "none"])
def test_halved_tsi_changes_the_state(monkeypatch, tmp_path, turbulence):
    d_const = _build_driver(str(tmp_path / "const"), "constant", turbulence)
    assert d_const.run() == "COMPLETED"
    _patch_solar(monkeypatch, lambda config: 0.5 * config.S_0)
    d_half = _build_driver(str(tmp_path / "half"), "file", turbulence)
    assert d_half.run() == "COMPLETED"
    t_half, t_const = _grid_t(d_half), _grid_t(d_const)
    assert np.isfinite(t_half).all()
    assert np.max(np.abs(t_half - t_const)) > 0.0, (
        "halving the transient TSI left the spectral state bit-identical — "
        "forcing['tsi'] is not reaching the spectral radiation")
    # signed: half the sun must leave the column COOLER, not merely different
    # (an inverted ratio would also move the state)
    assert t_half.mean() < t_const.mean()
