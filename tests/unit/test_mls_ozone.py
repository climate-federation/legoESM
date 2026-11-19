"""SAM MLS ozone (gSAM tracesini layer means) — idealized and dispatch tests.

The oracle match itself is tests/atmosphere/test_ozone_mls_oracle.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.ozone_mls import (
    mls_ozone_layer_vmr,
    _MLS_P_HPA,
    _MLS_O3_VMR,
)
from legoesm.atmosphere.physics.radiation.integration import _compute_ozone_vmr
from legoesm.atmosphere.physics.radiation.config import OzoneProfileConfig


jax.config.update("jax_enable_x64", True)


def _column_path(p_half):
    """Total O3 path (VMR x hPa) from the surface to gSAM's extra-layer top."""
    lay, above = mls_ozone_layer_vmr(jnp.asarray(p_half)[None, :])
    ph = np.asarray(p_half) / 100.0
    top = min(1.0e-4, ph[0] / 8.0)
    return float(np.sum(np.asarray(lay[0]) * np.diff(ph)) + float(above[0]) * (ph[0] - top))


def test_column_path_is_grid_invariant():
    """tracesini's design property: same bottom and top, any interior grid,
    same total path (to round-off)."""
    a = np.array([50.0, 3.0e3, 2.0e4, 5.0e4, 8.0e4, 1.0e5])
    b = np.concatenate([[50.0], np.geomspace(200.0, 9.9e4, 23), [1.0e5]])
    pa, pb = _column_path(a), _column_path(b)
    assert abs(pa - pb) <= 1.0e-10 * abs(pa)


def test_layer_on_a_constant_stretch_takes_the_node_value():
    """A layer entirely below the sounding bottom gets the bottom node VMR."""
    ph = jnp.asarray([[1.0e4, 5.0e4, 1.06e5, 1.08e5]])
    lay, _ = mls_ozone_layer_vmr(ph)
    assert float(lay[0, -1]) == pytest.approx(_MLS_O3_VMR[0], rel=1e-12)


def test_lower_stratosphere_not_overestimated():
    """A thin layer around 52.46 hPa returns the MLS 1.49 ppm, NOT the built-in
    Gaussian's ~4.9 ppm."""
    p = _MLS_P_HPA[15] * 100.0
    lay, _ = mls_ozone_layer_vmr(jnp.asarray([[1.0e3, p - 1.0, p + 1.0, 1.0e5]]))
    assert float(lay[0, 1]) == pytest.approx(_MLS_O3_VMR[15], rel=1e-3)


def test_dispatch_through_compute_ozone_vmr():
    """source='mls' routes through the radiation ozone dispatcher with the
    interfaces; without them it refuses; an unknown source still raises."""
    p_half = jnp.asarray([[1.0e3, 1.0e4, 3.0e4, 7.0e4]])
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    lat = jnp.asarray([0.0])
    out = _compute_ozone_vmr(p_full, lat, OzoneProfileConfig(source="mls"), p_half=p_half)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(mls_ozone_layer_vmr(p_half)[0]))
    with pytest.raises(ValueError, match="needs p_half"):
        _compute_ozone_vmr(p_full, lat, OzoneProfileConfig(source="mls"))
    with pytest.raises(ValueError, match="Unknown OzoneProfileConfig.source"):
        _compute_ozone_vmr(p_full, lat, OzoneProfileConfig(source="typo"))


def test_cd_grid_lane_uses_layer_means():
    """The driver's inline-ozone precompute (cd-grid / lat-lon pipeline) gives
    the gSAM layer means on the model layers and gSAM's own extra layer above
    the model top to the overhead layer."""
    from types import SimpleNamespace

    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.grids.vertical import make_cam6_l32_levels

    fake = SimpleNamespace(config=ExperimentConfig(radiation="rrtmgp", ozone_source="mls"))
    ModelDriver._setup_external_forcing(fake)
    fake.sigma = make_cam6_l32_levels()
    ncol = 3
    p_s = jnp.asarray([1.0e5, 9.5e4, 7.0e4])
    lat = jnp.zeros(ncol)
    fake.state = SimpleNamespace()
    fake._owned_face_ids = None
    fake._gather_spmd_tree_to_host = lambda x: x
    fake._owned_temp = lambda: ModelDriver._owned_temp(fake)
    fake.radiation_ghg_vmr = lambda day: ModelDriver.radiation_ghg_vmr(fake, day)
    fake._grid_lon = jnp.zeros_like(lat)
    fake._owned_lon = lambda: ModelDriver._owned_lon(fake)
    o3, _, _ = ModelDriver._precompute_external_forcing(fake, 15.5, p_s, lat)
    p_half = fake.sigma.pressure_at_half(p_s).reshape(ncol, -1)
    np.testing.assert_array_equal(np.asarray(o3), np.asarray(mls_ozone_layer_vmr(p_half)[0]))
    assert float(jnp.min(p_half[:, 0])) > 0.0
    above = np.asarray(mls_ozone_layer_vmr(p_half)[1])
    assert np.max(np.abs(above - np.asarray(o3)[:, 0]) / above) > 1e-3   # not the top layer
    np.testing.assert_allclose(np.asarray(fake._o3_top_vmr), above, rtol=1e-12)


def test_zero_pressure_gradient_outside_the_table():
    """Layers lying wholly below the sounding bottom take its node value, so
    their VMR does not depend on the interfaces: the
    reverse-mode derivative is zero to round-off (the old pointwise test's
    property, on the layer means)."""
    def f(ph):
        lay, _ = mls_ozone_layer_vmr(ph)
        return jnp.sum(lay[:, 1:])          # the two layers below 1053.63 hPa
    ph = jnp.asarray([[1.0e4, 1.06e5, 1.08e5, 1.10e5]])
    g = np.asarray(jax.grad(f)(ph))
    assert np.all(np.isfinite(g))
    assert np.max(np.abs(g)) <= 1.0e-12 * _MLS_O3_VMR[0] / 1.0e3   # per Pa
