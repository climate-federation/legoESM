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


def test_validate_strict_accepts_spmd_checkpointing():
    """cs_spmd step 5a: checkpointing is supported under spmd — the
    save_checkpoint tail gathers the sharded state to a host replica on
    every process and writes root-only.  validate_strict must accept it."""
    from legoesm.driver.config import OutputConfig
    cfg = _cfg(distributed=True, distributed_mode="spmd")
    cfg = cfg._replace(output=OutputConfig(
        output_dir="", diag_days=0, checkpoint_days=5))
    cfg.validate_strict()


def test_validate_strict_accepts_spmd_diagnostics_perf_mode():
    """cs_spmd step 5b: perf-mode diagnostics (scalar SPMD-global
    reductions + root-gated flushes) are supported under spmd."""
    from legoesm.driver.config import OutputConfig
    cfg = _cfg(distributed=True, distributed_mode="spmd")
    cfg = cfg._replace(output=OutputConfig(
        output_dir="", diag_days=2, checkpoint_days=0))
    cfg.validate_strict()


def test_validate_strict_accepts_spmd_full_collect():
    """cs_spmd step 5c: the full collect() gathers sharded fields to host
    replicas on every process, so cmip_output and
    diagnostics_perf_mode='never' are both legal under spmd now."""
    from legoesm.driver.config import OutputConfig
    cfg = _cfg(distributed=True, distributed_mode="spmd")
    cfg._replace(output=OutputConfig(
        output_dir="", diag_days=2, checkpoint_days=0,
        cmip_output=True)).validate_strict()
    cfg._replace(output=OutputConfig(
        output_dir="", diag_days=2, checkpoint_days=0,
        diagnostics_perf_mode="never")).validate_strict()


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


# --- _spmd_barrier_on_root_error (root-write failure rendezvous) ----------
# Hermetic single-process tests of the lockstep-error helper (codex LOW:
# the real 2-process failure injection needs an mpirun lane; these pin the
# helper's logic — native re-raise off-SPMD, collective + peer-flag raise
# on-SPMD — without a launcher).

def _barrier_stub(spmd: bool):
    from legoesm.driver.model_driver import ModelDriver

    class _Stub:
        def _is_spmd_multiprocess(self):
            return spmd
        _spmd_barrier_on_root_error = ModelDriver._spmd_barrier_on_root_error

    return _Stub()


def test_spmd_barrier_off_spmd_is_native_raise():
    s = _barrier_stub(spmd=False)
    s._spmd_barrier_on_root_error(None)  # no-op
    with pytest.raises(ValueError, match="boom"):
        s._spmd_barrier_on_root_error(ValueError("boom"))


def test_spmd_barrier_peer_failure_raises_lockstep(monkeypatch):
    """A peer's error flag (allgather'd) raises HERE, so this process never
    sails into the next collective while the peer is dead."""
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    calls = []

    def _fake_allgather(x):
        calls.append(x)
        return jnp.asarray([0.0, 1.0])  # some peer flagged failure

    monkeypatch.setattr(multihost_utils, "process_allgather", _fake_allgather)
    with pytest.raises(RuntimeError, match="lockstep"):
        _barrier_stub(spmd=True)._spmd_barrier_on_root_error(None)
    assert len(calls) == 1, "the collective must run on every process"


def test_spmd_barrier_own_error_reraises_after_collective(monkeypatch):
    """The failing process still DISPATCHES the collective (peers need the
    flag), then re-raises its own error."""
    import jax.numpy as jnp
    from jax.experimental import multihost_utils

    calls = []

    def _fake_allgather(x):
        calls.append(float(x))
        return jnp.asarray([1.0, 0.0])

    monkeypatch.setattr(multihost_utils, "process_allgather", _fake_allgather)
    with pytest.raises(ValueError, match="root boom"):
        _barrier_stub(spmd=True)._spmd_barrier_on_root_error(
            ValueError("root boom"))
    assert calls == [1.0], "own failure must be flagged to peers"
