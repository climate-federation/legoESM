"""Serial-path unit tests for ``legoesm.parallel.early_init``.

The multi-node branch needs a real MPI launch (covered by ``tests/distributed``);
here we pin the *serial / single-task* contract, which is what every non-MPI
import of an entry-point script hits: ``maybe_init_jax_distributed()`` must be a
cheap no-op (no mpi4py / jax import, no ``jax.distributed.initialize``) and
return ``False`` when the environment reports a single task.
"""

from __future__ import annotations

import legoesm.parallel.early_init as early_init


_TASK_ENV_VARS = ("SLURM_NTASKS", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE")


def _clear_task_env(monkeypatch):
    for var in _TASK_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_no_env_is_serial_noop(monkeypatch):
    # No launcher env at all -> treated as a single task -> no-op.
    _clear_task_env(monkeypatch)
    assert early_init.maybe_init_jax_distributed() is False


def test_ntasks_one_is_serial_noop(monkeypatch):
    _clear_task_env(monkeypatch)
    monkeypatch.setenv("SLURM_NTASKS", "1")
    assert early_init.maybe_init_jax_distributed() is False


def test_returns_before_importing_jax(monkeypatch):
    # The serial path must not even import jax: force an ImportError if it tries.
    _clear_task_env(monkeypatch)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "1")
    import builtins

    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name in ("jax", "mpi4py") or name.startswith(("jax.", "mpi4py.")):
            raise AssertionError(f"serial path must not import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    assert early_init.maybe_init_jax_distributed() is False


def test_already_initialized_is_noop_without_jax_probe(monkeypatch):
    # #693: idempotency must NOT probe jax.process_count() (that inits the XLA
    # backend, breaking jax.distributed.initialize). When _INITIALIZED is set,
    # return False without importing jax/mpi4py even under a multi-task env.
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setattr(early_init, "_INITIALIZED", True)
    import builtins

    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name in ("jax", "mpi4py") or name.startswith(("jax.", "mpi4py.")):
            raise AssertionError(f"idempotent path must not import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    assert early_init.maybe_init_jax_distributed() is False
