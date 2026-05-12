"""FV3_3D iter 706: prt_height_fv3 port.

Faithful JAX port of FV3 ``prt_height`` (tools/fv_diagnostics.F90:
4413-4460).  Composition of iter-684 ``get_height_given_pressure_fv3``
+ iter-705 ``prt_gb_nh_sh_fv3``.

Tests
-----

1. ``test_prt_height_isothermal_500hPa``.
2. ``test_prt_height_band_dict_keys``.
3. ``test_prt_height_lat_band_uniform``.
4. ``test_prt_height_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import prt_height_fv3


def _build_isothermal_column(n_cells, km, T, p_s=1.0e5):
    """Hydrostatic isothermal column: pe = p_s · exp(-k · dz / H),
    H = R_d·T/g.
    Layer thickness delz = -dz (top-down, km layers from p_top to p_s).
    For test simplicity: uniform delz and pe values per cell.
    """
    H = constants.R_d * T / constants.g
    dz = 200.0
    delz = jnp.full((n_cells, km), -dz)
    # pe at km+1 interfaces: pe[k] = p_s · exp(-(km-k)·dz / H)
    # k=km bottom (pe=p_s); k=0 top
    k_idx = jnp.arange(km + 1)
    pe_col = p_s * jnp.exp(-(km - k_idx) * dz / H)
    peln_col = jnp.log(pe_col)
    peln = jnp.broadcast_to(peln_col[None, :], (n_cells, km + 1))
    return delz, peln


def test_prt_height_isothermal_500hPa():
    """Isothermal column, p_500 = 500 hPa → height ≈ H·ln(p_s/p_500)."""
    km = 30
    n_cells = 50
    T = 280.0
    p_s = 1.0e5
    p_target = 5.0e4
    delz, peln = _build_isothermal_column(n_cells, km, T, p_s)
    phis = jnp.zeros((n_cells,))
    area = jnp.ones((n_cells,))
    lat = jnp.linspace(-jnp.pi / 2 * 0.9, jnp.pi / 2 * 0.9, n_cells)
    out = prt_height_fv3(p_target, phis, delz, peln, area, lat)
    H = constants.R_d * T / constants.g
    expected = H * jnp.log(p_s / p_target)
    # All band means should be ~ expected (uniform column)
    assert abs(out["gb"] - float(expected)) / float(expected) < 0.01


def test_prt_height_band_dict_keys():
    """Output dict has gb/nh/sh/eq keys."""
    km = 10
    n_cells = 20
    delz, peln = _build_isothermal_column(n_cells, km, 280.0)
    phis = jnp.zeros((n_cells,))
    area = jnp.ones((n_cells,))
    lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_cells)
    out = prt_height_fv3(5.0e4, phis, delz, peln, area, lat)
    assert set(out.keys()) == {"gb", "nh", "sh", "eq"}


def test_prt_height_lat_band_uniform():
    """Same column at all cells → all bands return same height."""
    km = 30
    n_cells = 100
    delz, peln = _build_isothermal_column(n_cells, km, 290.0)
    phis = jnp.zeros((n_cells,))
    area = jnp.ones((n_cells,))
    # Cells spanning full latitude range
    lat = jnp.linspace(-jnp.pi / 2 * 0.95, jnp.pi / 2 * 0.95, n_cells)
    out = prt_height_fv3(5.0e4, phis, delz, peln, area, lat)
    # All band means should match (uniform column)
    assert abs(out["gb"] - out["nh"]) < 1.0   # tolerance 1 m
    assert abs(out["gb"] - out["sh"]) < 1.0
    assert abs(out["gb"] - out["eq"]) < 1.0


def test_prt_height_finite():
    """Random column data → finite output."""
    rng = np.random.default_rng(seed=706)
    km = 30
    n_cells = 100
    delz, peln = _build_isothermal_column(n_cells, km, 280.0)
    phis = jnp.asarray(rng.uniform(0.0, 5000.0, size=(n_cells,))) * constants.g
    area = jnp.asarray(rng.uniform(0.5, 2.0, size=(n_cells,)))
    lat = jnp.asarray(rng.uniform(-jnp.pi / 2 + 0.01, jnp.pi / 2 - 0.01,
                                  size=(n_cells,)))
    out = prt_height_fv3(5.0e4, phis, delz, peln, area, lat)
    for k in ("gb", "nh", "sh", "eq"):
        assert np.isfinite(out[k]) or out[k] == -1.0
