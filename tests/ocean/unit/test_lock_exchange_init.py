"""Regression guards for the legoESM lock-exchange initialization helper.

Two findings caught during the 2026-05-17 alignment with the Veros peer
setup that this test pins:

1. ``LockExchangeConfig.H_max`` defaults to Petersen 2015 Fig. 5 (20 m).
   Prior default of 500 m drifted from the canonical benchmark scale and
   broke like-for-like comparison.
2. ``run_ocean_test_matrix._init_lock_exchange`` previously checked
   ``lon < 0`` to pick the cold half of the basin. legoESM lat-lon grids
   use ``lon in [0, 2pi]`` (radians) so that branch was never taken; every
   cell ended up at ``T_warm = 30``. The fix uses the median longitude
   of the actual grid coordinate as the front position.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "matrix"))  # for `import run_ocean_test_matrix`

from legoesm.ocean.experiments.lock_exchange import (  # noqa: E402
    EXPERIMENT_CONFIG,
    LockExchangeConfig,
)


def test_config_default_H_max_is_petersen_fig5():
    cfg = LockExchangeConfig()
    assert cfg.H_max == 20.0, (
        "LockExchangeConfig.H_max must equal Petersen 2015 Fig. 5 (20 m). "
        "Drifting from this value silently misaligns the legoESM run vs the "
        "Veros peer setup in src/legoesm/ocean/fidelity/veros_configs/"
        "lock_exchange.py."
    )


def test_config_default_temperatures_match_petersen_fig5():
    cfg = LockExchangeConfig()
    assert cfg.T_cold_C == 5.0
    assert cfg.T_warm_C == 30.0


def test_special_config_H_max_matches_dataclass():
    """Two ``H_max`` sources should not drift apart."""
    assert EXPERIMENT_CONFIG["special_config"]["H_max"] == 20.0


def test_init_lock_exchange_handles_zero_to_two_pi_longitude():
    """Synthetic state on a grid with ``lon in [0, 2pi]`` must produce a
    half-and-half T split, not a uniform-warm initialization."""

    import jax.numpy as jnp
    import run_ocean_test_matrix as m

    class FakeField:
        def __init__(self, data):
            self.data = jnp.asarray(data)

    class FakeState:
        def __init__(self, T, mask):
            self.T = FakeField(T)
            self.land_mask = FakeField(mask)
            self._replacements = {}

        def _replace(self, **kwargs):
            self._replacements.update(kwargs)
            new = FakeState(
                np.asarray(kwargs["T"].data) if "T" in kwargs
                else np.asarray(self.T.data),
                np.asarray(self.land_mask.data),
            )
            return new

    class FakeGrid:
        def __init__(self, n_lat, n_lon):
            # lon in [0, 2pi] (matches legoESM lat-lon convention). The
            # matrix's ``_get_cell_latlon_rad`` for the ``latlon`` grid
            # expects 1-D lat / lon arrays and meshgrids them itself.
            self.lon = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
            self.lat = np.linspace(-np.pi / 3, np.pi / 3, n_lat)

    n_lat, n_lon, nlev = 4, 8, 3
    grid = FakeGrid(n_lat, n_lon)
    T = np.full((n_lat, n_lon, nlev), 15.0)
    mask = np.ones((n_lat, n_lon), dtype=np.float32)
    state = FakeState(T, mask)

    new_state = m._init_lock_exchange(state, "latlon", grid, None)
    T_new = np.asarray(new_state.T.data)

    assert T_new.min() == 5.0, f"expected cold side T=5, got {T_new.min()}"
    assert T_new.max() == 30.0, f"expected warm side T=30, got {T_new.max()}"
    cold_cells = int(np.sum(T_new == 5.0))
    warm_cells = int(np.sum(T_new == 30.0))
    assert cold_cells > 0 and warm_cells > 0, (
        "lock-exchange initial condition must produce both cold and warm "
        "cells; bug regression: lon < 0 check on [0, 2pi] grid leaves "
        "every cell at T_warm"
    )


def _fake_latlon(n_lat=4, n_lon=8, nlev=3, lon_west=0.0, lon_east=2.0 * np.pi):
    """The fakes above, as a helper: a lat-lon state on a chosen lon range."""
    import jax.numpy as jnp

    class FakeField:
        def __init__(self, data):
            self.data = jnp.asarray(data)

    class FakeState:
        def __init__(self, T, mask):
            self.T = FakeField(T)
            self.land_mask = FakeField(mask)

        def _replace(self, **kw):
            return FakeState(np.asarray(kw["T"].data),
                             np.asarray(self.land_mask.data))

    class FakeGrid:
        def __init__(self):
            self.lon = np.linspace(lon_west, lon_east, n_lon, endpoint=False)
            self.lat = np.linspace(-np.pi / 3, np.pi / 3, n_lat)

    return (FakeState(np.full((n_lat, n_lon, nlev), 15.0),
                      np.ones((n_lat, n_lon), dtype=np.float32)),
            FakeGrid())


def test_a_domain_holding_one_water_mass_is_refused():
    """The class-catching check, not only the case that tripped it.

    A case whose initial condition does not contain the phenomenon being
    measured passes every gate silently: a near-uniform box cannot undershoot
    a bound and has nothing to mix. The regional Petersen channel ran that way
    for months -- its domain starts at the prime meridian, where the front
    sits by default, so one wet column of sixty-six was cold. The initialiser
    now refuses it instead of reporting a pass.
    """
    import pytest
    import run_ocean_test_matrix as m

    # A channel lying entirely east of the front: no lock to release.
    state, grid = _fake_latlon(lon_west=0.0, lon_east=np.radians(0.576))
    with pytest.raises(ValueError, match="only one water mass"):
        m._init_lock_exchange(state, "latlon", grid, None,
                              lx_config=LockExchangeConfig(front_longitude=0.0))

    # The same channel with the front at its midpoint is accepted.
    state, grid = _fake_latlon(lon_west=0.0, lon_east=np.radians(0.576))
    out = m._init_lock_exchange(
        state, "latlon", grid, None,
        lx_config=LockExchangeConfig(front_longitude=0.288))
    T = np.asarray(out.T.data)
    assert T.min() == 5.0 and T.max() == 30.0
