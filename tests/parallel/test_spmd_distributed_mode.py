"""distributed_mode='spmd' plumbing (codex-1: production CS driver SPMD).

Single-process lanes only: config validation membership, the
mixed-stack grid refusals, and the single-process spmd mesh shape.
The multi-process end-to-end smoke lives in the gate sbatch
(scripts/tmp/_probe_driver_cs_spmd.py) — jax.distributed needs a real
multi-process launch.
"""
from __future__ import annotations

import pytest


def _cfg(grid_type="cubed_sphere", **kw):
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    return ExperimentConfig(
        grid=GridConfig(
            grid_type=grid_type, resolution=8, nlev=4,
            vertical_coord="hybrid", p_top_Pa=200.0, stretching=2.0,
        ),
        dycore=DycoreConfig(discretization="centered", dt=600.0),
        output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
        days=1, dataset="analytical", radiation="gray",
        convection="none", turbulence="none", microphysics="none",
        cloud_scheme="none", gravity_wave_drag="none",
        **kw,
    )


def test_validate_strict_accepts_spmd_cubed_sphere():
    _cfg(distributed=True, distributed_mode="spmd").validate_strict()


def test_validate_strict_rejects_unknown_mode():
    with pytest.raises(ValueError, match="distributed_mode"):
        _cfg(distributed_mode="bogus").validate_strict()


def test_validate_strict_rejects_spmd_on_latlon():
    with pytest.raises(ValueError, match="spmd"):
        _cfg(grid_type="latlon", distributed=True,
             distributed_mode="spmd").validate_strict()


def test_validate_strict_allows_spmd_field_when_not_distributed():
    # The grid restriction binds only when distributed=True (the field
    # is inert otherwise).
    _cfg(grid_type="latlon", distributed=False,
         distributed_mode="spmd").validate_strict()


def test_setup_devices_rejects_spmd_non_cubed_sphere():
    from legoesm.runtime.devices import setup_devices
    with pytest.raises(ValueError, match="spmd"):
        setup_devices(distributed=True, distributed_mode="spmd",
                      grid_type="latlon", grid_n=32)


def test_bootstrap_rejects_unknown_mode():
    from legoesm.runtime import bootstrap
    with pytest.raises(ValueError, match="distributed_mode"):
        bootstrap(distributed_mode="bogus")


def test_validate_strict_rejects_spmd_checkpointing():
    from legoesm.driver.config import OutputConfig
    cfg = _cfg(distributed=True, distributed_mode="spmd")
    cfg = cfg._replace(output=OutputConfig(
        output_dir="", diag_days=0, checkpoint_days=5))
    with pytest.raises(ValueError, match="checkpoint"):
        cfg.validate_strict()


def test_validate_strict_rejects_spmd_diagnostics():
    from legoesm.driver.config import OutputConfig
    cfg = _cfg(distributed=True, distributed_mode="spmd")
    cfg = cfg._replace(output=OutputConfig(
        output_dir="", diag_days=2, checkpoint_days=0))
    with pytest.raises(ValueError, match="diag"):
        cfg.validate_strict()


def test_setup_devices_spmd_rejects_n_devices_mismatch():
    from legoesm.runtime.devices import setup_devices
    with pytest.raises(ValueError, match="GLOBAL"):
        setup_devices(distributed=True, distributed_mode="spmd",
                      grid_type="cubed_sphere", n_devices=999)


def test_setup_devices_spmd_single_process_mesh():
    """Single process (no SLURM/OMPI env): spmd mode builds the mesh over
    the local device set and keeps is_distributed=False (the driver's
    SPMD shard branch, never the mpi4jax replicated branch)."""
    import jax
    from legoesm.runtime.devices import setup_devices

    cfg = setup_devices(distributed=True, distributed_mode="spmd",
                        grid_type="cubed_sphere")
    assert cfg.is_distributed is False
    assert cfg.n_devices == len(jax.devices())
    assert cfg.mesh is not None or cfg.n_devices == 1
