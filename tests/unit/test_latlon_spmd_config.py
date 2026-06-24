"""validate_strict gating for the single-process multi-device lat-lon SPMD
flag (config.enable_latlon_spmd). Cheap (no devices, no JIT) — exercises only
the config validation guards added with the A1 driver wiring. GridConfig /
OutputConfig / ExperimentConfig are all NamedTuples -> use ``._replace``."""
from __future__ import annotations

import pytest

from legoesm.driver.config import ExperimentConfig


def _base_latlon(**over):
    """A minimal lat-lon ExperimentConfig with enable_latlon_spmd on. diag /
    checkpoint off (the SPMD path's writers are a follow-up)."""
    cfg = ExperimentConfig()
    fields = dict(
        enable_latlon_spmd=True, distributed=False, days=1,
        grid=cfg.grid._replace(grid_type="latlon"),
        output=cfg.output._replace(diag_days=0, checkpoint_days=0),
    )
    fields.update(over)  # caller overrides (e.g. distributed=True)
    return cfg._replace(**fields)


def test_validate_accepts_latlon_spmd_dynamics_only():
    """A lat-lon, single-process config validates (the n_lat % n_devices check
    is deferred to runtime against the built grid)."""
    _base_latlon(n_devices=4).validate_strict()  # must not raise


def test_validate_rejects_latlon_spmd_on_cubed_sphere():
    cfg = _base_latlon(n_devices=4)
    cfg = cfg._replace(grid=cfg.grid._replace(grid_type="cubed_sphere"))
    with pytest.raises(ValueError, match="grid_type='latlon'"):
        cfg.validate_strict()


def test_validate_rejects_latlon_spmd_with_distributed():
    with pytest.raises(ValueError, match="mutually exclusive with distributed"):
        _base_latlon(n_devices=4, distributed=True).validate_strict()
