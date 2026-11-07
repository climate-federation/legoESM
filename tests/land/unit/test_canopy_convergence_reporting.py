"""A failed canopy root solve must be visible on the slab lane and in the mosaic.

* The slab lane does not hold an unconverged column (the multilayer lane does),
  so ``step_land_with_diagnostics`` must hand the canopy's per-column
  ``converged`` flag back to the caller, and ``step_land`` must stay the same
  3-tuple it always was.
* The patch mosaic reduces ``converged`` with ``all`` — an area-weighted boolean
  reported a column with one failed patch as e.g. 0.6 "converged".

Run under ``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

import legoesm.land.slab_land as slab
from legoesm.land.config import LandConfig
from legoesm.land.surface_scheme import SurfaceFluxOutput, TwoLeafCanopyConfig
from legoesm.land.surface_scheme.patch_mosaic import _area_weight

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "integration"))
from test_canopy_carbon_coupling import _make_forcing, _make_slab_state  # noqa: E402

_PATTERN = jnp.asarray([True, False, True, False])


def _slab_inputs():
    cfg = LandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=30))
    return (_make_slab_state((4,)), _make_forcing(4, sw_down=700.0, cos_zenith=0.8),
            cfg)


def test_slab_reports_the_canopy_converged_flag(monkeypatch):
    real = slab.compute_two_leaf_canopy_fluxes

    def forced(*a, **k):
        return real(*a, **k)._replace(converged=_PATTERN)

    state, forcing, cfg = _slab_inputs()
    kw = dict(U_min=1.0, dt=1800.0, lat=jnp.zeros(4), doy=180.0)
    clean = slab.step_land_with_diagnostics(state, forcing, cfg, **kw)
    monkeypatch.setattr(slab, "compute_two_leaf_canopy_fluxes", forced)
    out = slab.step_land_with_diagnostics(state, forcing, cfg, **kw)
    assert len(out) == 4
    np.testing.assert_array_equal(np.asarray(out[3].converged), np.asarray(_PATTERN))
    # Reporting only: a failed column's fluxes and state are NOT changed.
    for a, b in zip(jax.tree.leaves(clean[:2]), jax.tree.leaves(out[:2])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_step_land_is_the_first_three_of_the_diagnostic_variant():
    state, forcing, cfg = _slab_inputs()
    kw = dict(U_min=1.0, dt=1800.0, lat=jnp.zeros(4), doy=180.0)
    plain = slab.step_land(state, forcing, cfg, **kw)
    diag = slab.step_land_with_diagnostics(state, forcing, cfg, **kw)
    assert len(plain) == 3
    for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(diag[:3])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert diag[3].converged is not None and diag[3].converged.shape == (4,)


def _patch_output(converged):
    n = len(converged)
    z = jnp.zeros(n)
    return SurfaceFluxOutput(
        shflx=z, lhflx=z, tau_x=z, tau_y=z, sw_net=z, lw_net=z, lw_up=z, G_soil=z,
        T_surface=jnp.full(n, 300.0), q_surface=z, albedo=jnp.full(n, 0.2),
        emissivity=jnp.full(n, 0.97), z0=jnp.full(n, 0.1),
        converged=jnp.asarray(converged))


def test_mosaic_column_converged_only_when_every_patch_converged():
    fr = jnp.asarray([0.4, 0.6])
    one_failed = _area_weight(_patch_output([True, False]), fr)
    assert one_failed.converged.dtype == jnp.bool_
    assert not bool(one_failed.converged[0])
    both = _area_weight(_patch_output([True, True]), fr)
    assert bool(both.converged[0])
