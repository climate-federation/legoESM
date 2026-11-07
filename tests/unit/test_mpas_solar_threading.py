"""Transient solar threading on the MPAS lane (2026-07-23 port).

The MPAS standalone radiation previously integrated with the configured
constant ``S_0`` — the CMIP6 solar file (TSI + 14-band spectral) was the one
inert channel of the six-channel AMIP deck.  The port samples the file DAILY
(same canonical-day convention as SST/ozone, restart-exact) and threads a
traced ``forcing["tsi"]`` (+ ``solar_spectral_fraction`` for rrtmg/rrtmgp
under ``spectral_file``) into the radiation physics.

Contract pinned here, with the reader monkeypatched (offline, no NetCDF):

1. NEUTRALITY: ``solar_source="file"`` with the file returning EXACTLY the
   configured ``S_0`` equals ``solar_source="constant"`` (rtol 1e-15) — the
   scaling path adds nothing when the transient value matches the constant.
2. EFFECT: a halved TSI changes the integrated state — the traced value
   demonstrably reaches the radiative heating (not silently dropped).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

MPAS_RES, MPAS_NLEV, DT = 3, 8, 300.0   # icosahedral level 3 = 642 cells
FOUR_STEPS_DAYS = 1201.0 / 86400.0


def _build_driver(tmpdir: str, solar_source: str) -> ModelDriver:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=MPAS_RES,
                        nlev=MPAS_NLEV, vertical_coord="hybrid"),
        dycore=DycoreConfig(discretization="mpas", dt=DT),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=FOUR_STEPS_DAYS, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", precision="fp64",
        distributed=False,
        solar_source=solar_source,
        solar_file=("synthetic.nc" if solar_source == "file" else ""),
    )
    d = ModelDriver(cfg, output_dir=tmpdir)
    d.setup()
    return d


def _patch_solar(monkeypatch, tsi_fn):
    """Patch the file reader with a synthetic TSI (called with the driver's
    SolarConfig at the canonical forcing day)."""
    import legoesm.forcing.external as ext
    monkeypatch.setattr(
        ext, "get_solar_forcing_at_time",
        lambda config, day: {"tsi": float(tsi_fn(config)),
                             "solar_fraction_by_gpt": None})


def test_file_tsi_equal_to_s0_matches_constant(monkeypatch, tmp_path):
    d_const = _build_driver(str(tmp_path / "const"), "constant")
    assert d_const.run() == "COMPLETED"

    _patch_solar(monkeypatch, lambda config: config.S_0)
    d_file = _build_driver(str(tmp_path / "file"), "file")
    assert d_file.run() == "COMPLETED"

    # tsi/S_0 == 1.0 is a traced multiply, not constant-folded: exact, but it
    # changes XLA fusion and moved one cell of 5136 by 2 ulp (1.1e-13 K) on
    # Ginsburg CPUs (jobs 10254740, 10288139).  Contract: no physical change.
    np.testing.assert_allclose(
        np.asarray(d_file.state.T.data), np.asarray(d_const.state.T.data),
        rtol=1e-15, atol=0.0,
        err_msg="file-mode TSI == S_0 must equal constant mode (rtol 1e-15)")


def test_halved_tsi_changes_the_state(monkeypatch, tmp_path):
    d_const = _build_driver(str(tmp_path / "const"), "constant")
    assert d_const.run() == "COMPLETED"

    _patch_solar(monkeypatch, lambda config: 0.5 * config.S_0)
    d_half = _build_driver(str(tmp_path / "half"), "file")
    assert d_half.run() == "COMPLETED"

    dT = np.max(np.abs(np.asarray(d_half.state.T.data)
                       - np.asarray(d_const.state.T.data)))
    assert np.isfinite(np.asarray(d_half.state.T.data)).all()
    assert dT > 0.0, (
        "halving the transient TSI left the state bit-identical — "
        "forcing['tsi'] is not reaching the radiation")
