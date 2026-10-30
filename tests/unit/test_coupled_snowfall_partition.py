"""Coupled surface snowfall = the atmosphere's own frozen flux (review
2026-10-10 F29 remainder).

``CoupledESMDriver._build_atm_forcing`` and ``EarthSystemDriver.
_build_atm_forcing`` used to split the segment-mean precipitation by a
lowest-level temperature ramp (``snow_fraction(T_low)``).  They now hand the
surface the segment-mean FROZEN flux the atmosphere lane publishes
(``_carry_aux['seg_precip_frozen']``), bounded by the total -- the rule the
MPAS land already follows -- and refuse a lane that publishes precipitation
without it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.unit.test_coupled_insolation_seam import (
    _coupled_driver, _earth_system_driver,
)

jax.config.update("jax_enable_x64", True)


def _drivers(tmp_path):
    return (_coupled_driver(None), _earth_system_driver(None, tmp_path))


def test_surface_snow_is_the_published_frozen_flux(tmp_path):
    for drv in _drivers(tmp_path):
        ps = drv._atm.state.p_s.data
        rng = np.random.default_rng(0)
        P = jnp.asarray(rng.uniform(1e-5, 1e-4, size=ps.shape))
        frozen = 0.37 * P
        frozen = frozen.reshape(-1).at[0].set(3.0 * P.reshape(-1)[0]
                                              ).reshape(ps.shape)
        drv._atm._carry_aux.update(
            {"seg_precip": P, "seg_precip_frozen": frozen})
        f = drv._build_atm_forcing(0.0)
        np.testing.assert_allclose(np.asarray(f.precip_total), np.asarray(P),
                                   rtol=1e-12)
        np.testing.assert_allclose(np.asarray(f.precip_snow),
                                   np.minimum(np.asarray(frozen), np.asarray(P)),
                                   rtol=1e-12)


def test_precip_without_frozen_part_is_refused(tmp_path):
    for drv in _drivers(tmp_path):
        # a precipitating physics config (the strict check is config-gated:
        # CoupledESMDriver reads atm_config, EarthSystemDriver config)
        for name in ("atm_config", "config"):
            cfg = getattr(drv, name, None)
            if hasattr(cfg, "microphysics"):
                setattr(drv, name, cfg._replace(microphysics="kessler"))
        ps = drv._atm.state.p_s.data
        drv._atm._carry_aux.update({"seg_precip": jnp.full_like(ps, 1e-5)})
        drv._atm._carry_aux.pop("seg_precip_frozen", None)
        with pytest.raises(RuntimeError, match="seg_precip_frozen"):
            drv._build_atm_forcing(0.0)
