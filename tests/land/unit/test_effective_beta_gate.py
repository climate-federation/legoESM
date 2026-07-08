"""compute_effective_beta enabled-gate: stomata OFF returns the bare soil-
moisture beta (no physiological control); stomata ON applies stomatal
conductance limitation to land ET.  Guards the run_coupled --stomata wiring."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import LandConfig
from legoesm.land.stomata_utils import compute_effective_beta


def _forcing(co2_ppmv=400.0):
    shape = (4, 4)
    z = jnp.zeros(shape)
    return AtmToSurface(
        sw_down=jnp.full(shape, 600.0), lw_down=jnp.full(shape, 350.0),
        precip_total=z, precip_snow=z,
        T_lowest=jnp.full(shape, 295.0), q_lowest=jnp.full(shape, 8e-3),
        u_lowest=jnp.full(shape, 2.0), v_lowest=z,
        p_lowest=jnp.full(shape, 1.0e5), p_surface=jnp.full(shape, 1.0e5),
        rho_lowest=jnp.full(shape, 1.15), cos_zenith=jnp.full(shape, 0.7),
        co2_ppmv=jnp.array(co2_ppmv), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(0.0),
    )


def test_stomata_off_returns_bare_soil_beta():
    """enabled=False => beta == beta_soil (no stomatal/physiological control)."""
    cfg = LandConfig()
    assert cfg.stomata.enabled is False               # default
    beta_soil = jnp.full((4, 4), 0.7)
    beta, gpp = compute_effective_beta(
        jnp.full((4, 4), 298.0), _forcing(), beta_soil, cfg,
        carbon_state=None, dt=3600.0)
    np.testing.assert_allclose(np.asarray(beta), np.asarray(beta_soil))
    assert gpp is None


def test_stomata_on_applies_conductance_limit():
    """enabled=True (Jarvis, no carbon) => beta is stomatally limited
    (<= beta_soil) and DIFFERS from the bare soil beta => ET is physiologically
    controlled, not the no-stomata case."""
    cfg = LandConfig()._replace(
        stomata=LandConfig().stomata._replace(enabled=True))
    beta_soil = jnp.full((4, 4), 0.7)
    beta, _ = compute_effective_beta(
        jnp.full((4, 4), 298.0), _forcing(), beta_soil, cfg,
        carbon_state=None, dt=3600.0)
    beta = np.asarray(beta)
    assert np.all(np.isfinite(beta))
    assert np.all(beta <= np.asarray(beta_soil) + 1e-9)   # stomata can only limit
    assert not np.allclose(beta, np.asarray(beta_soil))   # genuinely active
