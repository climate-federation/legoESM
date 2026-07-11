"""Unit test for the #931 Held-Suarez Rayleigh surface-drag double-count fix.

``ModelDriver._create_friction`` precomputes ``self._fric_decay``, a per-level
multiplier applied to ``u, v`` every timestep.  The boundary-layer / free-tropo
Rayleigh drag it builds is the Held-Suarez DRY-CORE surrogate for surface
friction.  When a REAL turbulence scheme (e.g. ``louis``) is active it already
applies the physical surface stress as the BL bottom BC, so the Rayleigh term
double-counts surface drag and must be gated to an EXACT no-op (decay = 1.0).
It stays active ONLY when no BL scheme owns surface momentum
(``turbulence="none"`` -- which includes the pure Held-Suarez dry core, run at
the ``turbulence="none"`` default).  It is NOT kept merely because
``held_suarez_forcing`` is set: HS is additive to the physics pipeline, so a
``held_suarez_forcing`` + ``louis`` config would apply the Louis surface stress
AND this Rayleigh drag -- the double-count (codex #931).

We exercise the method directly on a lightweight ``self`` stub (a real
``ExperimentConfig`` + real ``SigmaCoordinate``) so the true field access and
formula run without constructing a full grid/state driver.
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
from legoesm.driver.model_driver import ModelDriver
from legoesm.grids.vertical import create_sigma_coordinate

_NLEV = 8
_DT = 600.0


def _make_stub(*, turbulence: str, held_suarez_forcing: bool):
    """Minimal ``self`` for the unbound ``_create_friction``.

    ``_create_friction`` reads only ``self.config``, ``self.sigma`` and
    ``self._hyperdiff``; it writes ``self._fric_decay`` (+ derived attrs).  A
    ``SimpleNamespace`` carrying a real config/sigma exercises the production
    formula without a full driver construction.
    """
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=8, nlev=_NLEV),
        dycore=DycoreConfig(
            dt=_DT, model_type="hydrostatic", discretization="finite_volume"
        ),
        turbulence=turbulence,
        held_suarez_forcing=held_suarez_forcing,
    )
    stub = types.SimpleNamespace(
        config=cfg,
        sigma=create_sigma_coordinate(_NLEV),
        _hyperdiff=1.0e15,
    )
    return stub, cfg


def _expected_hs_decay(cfg, sigma_full):
    """The always-on HS Rayleigh profile the drag applied before the fix."""
    k_f_max = cfg.k_BL_max_per_day / 86400.0
    k_free = cfg.k_free_per_day / 86400.0
    k_f = k_free + k_f_max * jnp.maximum(
        0.0, (sigma_full - cfg.sigma_b) / (1.0 - cfg.sigma_b)
    )
    return jnp.exp(-k_f * cfg.dycore.dt)


def test_rayleigh_drag_noop_with_real_turbulence():
    """louis owns surface momentum, HS off, sponge off -> decay is exactly 1."""
    stub, cfg = _make_stub(turbulence="louis", held_suarez_forcing=False)
    ModelDriver._create_friction(stub)
    # Guard: no top-of-atmosphere sponge (#836) contribution in this config.
    assert cfg.sponge_enabled is False
    np.testing.assert_array_equal(np.asarray(stub._fric_decay), np.ones(_NLEV))


def test_rayleigh_drag_gated_off_under_hs_when_louis_owns_momentum():
    """held_suarez_forcing=True + louis -> Rayleigh drag STILL gated off (#931).

    HS is ADDITIVE to the physics pipeline, so a held_suarez + louis config
    applies the Louis surface stress; keeping the HS Rayleigh surrogate too
    would re-introduce the exact double-count this fix removes.  The gate keys
    off momentum ownership (turbulence == "none") ONLY, never
    held_suarez_forcing -- so Louis being active gates the drag to an exact
    no-op regardless of HS.  (The HS thermal Newtonian relaxation is separate
    and unaffected.)  A prior revision asserted the opposite (drag KEPT here),
    which enshrined the double-count -- codex #931 caught it.
    """
    stub, cfg = _make_stub(turbulence="louis", held_suarez_forcing=True)
    ModelDriver._create_friction(stub)
    assert cfg.sponge_enabled is False
    np.testing.assert_array_equal(np.asarray(stub._fric_decay), np.ones(_NLEV))


def test_rayleigh_drag_active_for_pure_held_suarez_core():
    """The REAL Held-Suarez config (turbulence="none" -- the config default and
    what the HS test matrix sets) keeps its defining Rayleigh friction: no BL
    scheme owns momentum, so the surrogate drag is the intended SOLE friction.
    Proves the #931 gate does not strip HS's own friction when HS is run
    correctly (dry, no turbulence scheme)."""
    stub, cfg = _make_stub(turbulence="none", held_suarez_forcing=True)
    ModelDriver._create_friction(stub)
    expected = _expected_hs_decay(cfg, stub.sigma.sigma_full)
    np.testing.assert_allclose(
        np.asarray(stub._fric_decay), np.asarray(expected), rtol=1e-6, atol=0.0
    )
    # Non-vacuous: the retained drag genuinely damps somewhere (decay < 1).
    assert float(jnp.min(expected)) < 1.0


def test_rayleigh_drag_active_when_no_turbulence():
    """turbulence="none" (no BL scheme owns momentum) keeps the surrogate drag."""
    stub, cfg = _make_stub(turbulence="none", held_suarez_forcing=False)
    ModelDriver._create_friction(stub)
    expected = _expected_hs_decay(cfg, stub.sigma.sigma_full)
    np.testing.assert_allclose(
        np.asarray(stub._fric_decay), np.asarray(expected), rtol=1e-6, atol=0.0
    )
    assert float(jnp.min(expected)) < 1.0
