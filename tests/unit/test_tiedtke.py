"""Host adapter of the WRF v4.8.0 Tiedtke port (``tiedtke.tiedtke_convection``).

The port itself is certified against the running Fortran in
tests/atmosphere/test_tiedtke_wrf_oracle.py.  These tests pin the MAPPING the
adapter adds on the same oracle fixture columns: the land weight (exactly the
oracle's land mask at land fraction 0 / 1, ocean for None), the output fields
onto ConvectionOutput, and the surface precipitation emitted in the lowest
layer (its column integral is exactly the oracle precipitation).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.tiedtke import (
    tiedtke_convection, tiedtke_wrf,
)

_FIX = (Path(__file__).parents[1] / "atmosphere" / "oracle_fixtures"
        / "tiedtke_wrf.npz")
_IN = ("T", "q_v", "u", "v", "p_full", "p_half", "geopotential", "omega",
       "dq_v_dt_dyn", "dq_v_dt_pbl", "surface_moisture_flux", "sigma")


@pytest.fixture(scope="module")
def case():
    f = dict(np.load(_FIX))
    a = {k: jnp.asarray(f[f"F2/in/{k}"], jnp.float64) for k in _IN}
    land = np.asarray(f["F2/in/land"])
    assert land.any() and (~land).any()  # both surface types present
    return a, land, json.loads(str(f["meta"]))["dt"]


def _adapter(a, dt, land_frac):
    return tiedtke_convection(
        a["T"], a["q_v"], a["p_full"], a["p_half"], a["u"], a["v"], dt,
        geopotential=a["geopotential"], omega=a["omega"],
        dq_v_dt_dyn=a["dq_v_dt_dyn"], dq_v_dt_pbl=a["dq_v_dt_pbl"],
        surface_moisture_flux=a["surface_moisture_flux"],
        land_frac=land_frac, sigma=a["sigma"])


def _port(a, dt, land):
    return tiedtke_wrf(**{k: a[k] for k in _IN}, land=jnp.asarray(land),
                       dt=dt)


def test_adapter_maps_port_outputs_exactly(case):
    """Land fraction 0/1 reproduces the oracle land mask BITWISE, and every
    ConvectionOutput field is the port's field."""
    a, land, dt = case
    out, mfu = _adapter(a, dt, jnp.asarray(land, jnp.float64))
    ref = _port(a, dt, land)
    assert int(np.asarray(ref.ldcum).sum()) >= 10
    for ours, theirs in (
            (out.dT_dt, ref.dT_dt), (out.dq_v_dt, ref.dq_v_dt),
            (out.dq_c_conv_dt, ref.dq_c_dt), (out.dq_i_conv_dt, ref.dq_i_dt),
            (out.du_dt_conv, ref.du_dt), (out.dv_dt_conv, ref.dv_dt),
            (out.precip_frozen_sfc, ref.precip_snow), (mfu, ref.mfu)):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    np.testing.assert_array_equal(np.asarray(out.convective_mask),
                                  np.asarray(ref.ldcum, np.float64))
    np.testing.assert_array_equal(np.asarray(out.cape), 0.0)


def test_land_weight_changes_answer_and_none_is_ocean(case):
    """The land weight is read (all-land differs from the mixed fixture on its
    ocean columns) and land_frac=None is the all-ocean oracle, bitwise."""
    a, land, dt = case
    ocean = np.zeros_like(land)
    out_none, _ = _adapter(a, dt, None)
    np.testing.assert_array_equal(np.asarray(out_none.dT_dt),
                                  np.asarray(_port(a, dt, ocean).dT_dt))
    out_land, _ = _adapter(a, dt, jnp.ones(land.shape, jnp.float64))
    np.testing.assert_array_equal(np.asarray(out_land.dT_dt),
                                  np.asarray(_port(a, dt, ~ocean).dT_dt))
    assert not np.array_equal(np.asarray(out_land.dT_dt),
                              np.asarray(out_none.dT_dt))


def test_surface_precip_is_lowest_layer_and_integrates_exactly(case):
    a, land, dt = case
    out, _ = _adapter(a, dt, jnp.asarray(land, jnp.float64))
    ref = _port(a, dt, land)
    dq_r = np.asarray(out.dq_r_conv_dt)
    np.testing.assert_array_equal(dq_r[:, :-1], 0.0)
    dp = np.diff(np.asarray(a["p_half"]), axis=1)
    col = np.sum(dq_r * dp, axis=1) / constants.g
    np.testing.assert_allclose(col, np.asarray(ref.precip), rtol=1e-13,
                               atol=0.0)
    assert np.max(np.asarray(ref.precip)) > 0.0


def test_adapter_jit_matches_eager(case):
    a, land, dt = case
    lf = jnp.asarray(land, jnp.float64)
    eager, _ = _adapter(a, dt, lf)
    jitted, _ = jax.jit(lambda aa, l: _adapter(aa, dt, l))(a, lf)
    for k in ("dT_dt", "dq_v_dt", "dq_r_conv_dt", "du_dt_conv"):
        # the oracle test's metric and tolerance: max|diff| / max|ref|
        x, ref = np.asarray(getattr(jitted, k)), np.asarray(getattr(eager, k))
        assert np.max(np.abs(x - ref)) <= 1e-10 * np.max(np.abs(ref)), k


def test_adapter_gradient_wrt_land_fraction_finite(case):
    """The smooth land weight carries a finite gradient (the fixture's land
    fractions sit at 0/1, where it is ~0; at 0.5 it is O(sharpness))."""
    a, land, dt = case

    def loss(lf):
        out, _ = _adapter(a, dt, lf)
        return jnp.sum(out.dT_dt)

    g = jax.grad(loss)(jnp.full(land.shape, 0.5, jnp.float64))
    assert np.all(np.isfinite(np.asarray(g)))
