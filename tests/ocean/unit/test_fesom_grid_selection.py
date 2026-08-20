"""Selection-registry wiring tests for the ``fesom`` grid type.

Each test is constructed to fail if the branch it exercises is removed:
```
- ``test_fesom_in_global_grid_types``           → GLOBAL_GRID_TYPES edit
- ``test_create_grid_fesom_returns_grid``        → create_grid fesom branch
- ``test_create_grid_fesom_rejects_resolution``  → inner resolution guard
- ``test_create_grid_unknown_grid_type_raises``  → trailing ValueError (regression)
- ``test_supported_extents_fesom``               → GLOBAL_GRID_TYPES edit (drives extents)
- ``test_operator_family_fesom``                 → _OPERATOR_FAMILY entry
- ``test_lock_exchange_fesom_ic_has_front``      → lock_exchange fesom branch
- ``test_lock_exchange_unknown_grid_type_raises``→ trailing ValueError (regression)
```
"""
from __future__ import annotations

import numpy as np
import pytest

# The adapter is an OPTIONAL dependency: skip rather than error when absent.
pytest.importorskip("fesom_jax")

from legoesm.grids.factory import GLOBAL_GRID_TYPES, create_grid
from legoesm.grids.capability import supported_extents, operator_family
from legoesm.ocean.dynamics.ocean_model_fesom import (
    FesomOceanGrid,
    FesomOceanState,
)
from legoesm.ocean.experiments.lock_exchange import (
    LockExchangeConfig,
    create_initial_conditions,
)

# Must match the value chosen in
# legoesm.grids.capability._OPERATOR_FAMILY["fesom"].
FESOM_OPERATOR_FAMILY = "fesom_operators"


def test_fesom_in_global_grid_types():
    assert "fesom" in GLOBAL_GRID_TYPES


def test_create_grid_fesom_returns_grid():
    grid = create_grid("fesom", H_max=20.0, nlev=20)
    assert isinstance(grid, FesomOceanGrid)
    # Per-node control-volume areas are a 1-D array over mesh nodes.
    assert np.asarray(grid.area).ndim == 1


def test_create_grid_fesom_rejects_resolution():
    # match= is load-bearing: if the fesom branch is removed entirely, the
    # function falls through to the "Unknown grid_type" ValueError whose
    # message lacks "mesh-file-backed", so this test still fails.
    with pytest.raises(ValueError, match="mesh-file-backed"):
        create_grid("fesom", resolution=48, H_max=20.0, nlev=20)


def test_create_grid_unknown_grid_type_raises():
    with pytest.raises(ValueError, match="Unknown grid_type"):
        create_grid("nonsense")


def test_supported_extents_fesom():
    assert supported_extents("fesom") == frozenset({"global", "column"})


def test_operator_family_fesom():
    assert operator_family("fesom") == FESOM_OPERATOR_FAMILY


def test_lock_exchange_fesom_ic_has_front():
    grid = create_grid("fesom", H_max=20.0, nlev=20)
    config = LockExchangeConfig()
    state = create_initial_conditions("fesom", grid, "z", config)
    assert isinstance(state, FesomOceanState)
    # The lock-exchange IC must contain BOTH the cold and the warm value, i.e.
    # the front is real. If the fesom branch is removed, create_initial_conditions
    # raises "Unknown grid type" and this assertion never runs (test fails).
    T = np.asarray(state.T.data)
    unique = np.unique(np.round(T, decimals=4))
    assert unique.size >= 2, "FESOM lock-exchange T must contain a real front"


def test_lock_exchange_unknown_grid_type_raises():
    with pytest.raises(ValueError, match="Unknown grid type"):
        create_initial_conditions("nonsense", None, "z", LockExchangeConfig())
