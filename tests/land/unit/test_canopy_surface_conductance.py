"""Two-leaf canopy Robin-BC surface conductance (Phase 2a boreal-NaN fix).

The canopy must return a finite, non-negative ``surface_conductance`` (the
linearised lambda = -dG_soil/dTs) so ``multilayer_land``'s final soil-thermal
solve uses the semi-implicit (Robin) surface BC that damps the surface-T
feedback.  If this silently reverts to ``None`` the EXPLICIT (Neumann) BC returns
and cold boreal/Arctic cells diverge to NaN again — so pin it here.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.land.config import MultiLayerLandConfig
from legoesm.land.surface_scheme import TwoLeafCanopyConfig
from legoesm.land.multilayer_land import (
    init_multilayer_land_state,
    step_multilayer_land_with_diagnostics,
)


def _forcing(ncol, T_lowest, sw_down):
    return AtmToSurface(
        T_lowest=jnp.full(ncol, T_lowest), q_lowest=jnp.full(ncol, 0.004),
        u_lowest=jnp.full(ncol, 2.0), v_lowest=jnp.full(ncol, 0.5),
        p_lowest=jnp.full(ncol, 98000.0), p_surface=jnp.full(ncol, 101325.0),
        rho_lowest=jnp.full(ncol, 1.25), sw_down=jnp.full(ncol, sw_down),
        lw_down=jnp.full(ncol, 250.0), cos_zenith=jnp.full(ncol, 0.4),
        precip_total=jnp.zeros(ncol), precip_snow=jnp.zeros(ncol),
        co2_ppmv=jnp.full(ncol, 420.0), has_radiation=True, has_precipitation=False)


def _run(T_init, T_lowest, sw_down, ncol=2):
    cfg = MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig(max_iters=25))
    state = init_multilayer_land_state(ncol, cfg, T_init=T_init)
    _, _, _, surf_out = step_multilayer_land_with_diagnostics(
        state, _forcing(ncol, T_lowest, sw_down), cfg, U_min=1.0, dt=1800.0,
        lat=jnp.zeros(ncol), doy=15.0)
    return surf_out


def test_canopy_returns_finite_nonnegative_surface_conductance():
    surf = _run(T_init=290.0, T_lowest=295.0, sw_down=600.0)
    sc = np.asarray(surf.surface_conductance)
    assert surf.surface_conductance is not None
    assert np.all(np.isfinite(sc))
    assert np.all(sc >= 0.0)                    # lambda only ADDS damping to the soil solve
    assert np.all(sc > 1.0)                     # sensible term rhoa*Cp/rah dominates -> O(10) W/m2/K


def test_surface_conductance_finite_and_nonneg_in_cold_regime():
    """The regime that motivated the fix: a cold surface / cold air, low sun.
    lambda must stay finite and non-negative (it can only stabilise)."""
    surf = _run(T_init=245.0, T_lowest=238.0, sw_down=80.0)
    sc = np.asarray(surf.surface_conductance)
    assert np.all(np.isfinite(sc)) and np.all(sc >= 0.0)
    # the analytic LW floor 4 eps_s sigma Ts^3 alone is a few W/m2/K at ~245 K
    assert np.all(sc > 0.0)
