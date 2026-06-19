"""Unit tests for :meth:`ModelDriver.static_topography_phis`.

The side-effect-free topography probe: it exposes the model's OWN static surface
geopotential ``phis = g·z_s`` (the consistent source for the orographic
LES-forcing term) WITHOUT running the full ``setup()`` — so a launch-time probe
writes no run manifest / output directory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.driver.config import (  # noqa: E402
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.model_driver import ModelDriver  # noqa: E402


def _cfg(topography: str) -> ExperimentConfig:
    return ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=3),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic"),
        topography=topography,
        radiation="gray",
        days=1,
    )


def test_static_topography_phis_flat_is_zeros_and_writes_nothing(tmp_path):
    """A flat model yields an all-zero phis of the grid 2-D shape — and the probe
    performs NO filesystem writes (the output dir / run manifest are NOT created),
    so a launch-time orographic probe leaves no phantom run directory."""
    out = tmp_path / "run"
    driver = ModelDriver(_cfg("flat"), output_dir=str(out))
    phis = driver.static_topography_phis()
    assert tuple(phis.shape) == tuple(driver.grid.grid_shape_2d)
    assert bool(jnp.all(phis == 0))            # flat → identically zero
    assert not out.exists()                    # NO mkdir / manifest / config writes


def test_static_topography_phis_gaussian_is_nonzero(tmp_path):
    """A gaussian-mountain model yields a non-zero phis (real terrain), so the
    caller's flat-detection keeps the orographic term ON."""
    driver = ModelDriver(_cfg("gaussian"), output_dir=str(tmp_path / "run"))
    phis = driver.static_topography_phis()
    assert bool(jnp.any(phis != 0))            # gaussian mountain → real terrain


def test_static_topography_phis_idempotent(tmp_path):
    """Repeated calls return the SAME cached field (the build runs once); a driver
    that already ran setup() would likewise return its built _phis_data."""
    driver = ModelDriver(_cfg("flat"), output_dir=str(tmp_path / "run"))
    first = driver.static_topography_phis()
    second = driver.static_topography_phis()
    assert first is second
