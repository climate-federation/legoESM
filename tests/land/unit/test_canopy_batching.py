"""Batching of the two-leaf canopy Newton solve
(surface_scheme.two_leaf_canopy.canopy_solve_batching): one batch on GPU and on
column-sharded CPU, sorted groups of 64 on a single CPU device.  That the two
layouts give the same answer is pinned by
tests/land/integration/test_run_lmip_biophys.py::
test_column_sharded_run_matches_the_unsharded_run (1 CPU device = sorted
groups, 4 CPU devices = one batch per device)."""
from __future__ import annotations

import jax
import pytest

import legoesm.land.surface_scheme.two_leaf_canopy as tlc


@pytest.mark.parametrize("backend,ndev,sharded,expected", [
    ("gpu", 1, False, (None, False)),
    ("gpu", 4, True, (None, False)),
    ("cpu", 4, True, (None, False)),
    ("cpu", 1, False, (64, True)),
    # A CPU host exposing several devices to a run that does NOT shard keeps
    # the grouped, sorted solve (the device count is not the signal).
    ("cpu", 4, False, (64, True)),
])
def test_batching_policy(monkeypatch, backend, ndev, sharded, expected):
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(jax, "device_count", lambda *a, **k: ndev)
    assert tlc.canopy_solve_batching(sharded) == expected


def test_canopy_config_is_unsharded_by_default():
    from legoesm.land.canopy.config import CanopyConfig
    assert CanopyConfig().column_sharded is False
    assert CanopyConfig(column_sharded=True).validate().column_sharded is True
