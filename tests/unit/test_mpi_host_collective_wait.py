"""Host mpi4py collectives must wait for queued mpi4jax work first.

mpi4jax ops run on the XLA thread after async dispatch, on the same
COMM_WORLD the driver's host collectives use, so a host collective issued
while one is in flight can cross-match it across ranks.  Single rank here:
the tests pin ORDER (wait before the first host collective), not the race.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

import legoesm.driver.model_driver as md  # noqa: E402
from tests.unit.test_mpas_global_diag import _make_stub  # noqa: E402


def test_global_diag_waits_before_its_first_allreduce(monkeypatch):
    events = []

    class _SpyComm(MPI.Intracomm):
        def Allreduce(self, *a, **k):
            events.append("Allreduce")
            return super().Allreduce(*a, **k)

    real_barrier = jax.effects_barrier

    def _barrier():
        events.append("effects_barrier")
        real_barrier()

    monkeypatch.setattr(jax, "effects_barrier", _barrier)
    monkeypatch.setattr(MPI, "COMM_WORLD", _SpyComm(MPI.COMM_WORLD))
    drv, _, _ = _make_stub()
    jnp = jax.numpy
    drv._mpas_global_diag(jnp.full((6, 3), 280.0), jnp.full((6,), 1.0e5),
                          jnp.zeros((5, 3)), None)

    assert "Allreduce" in events, "spy never saw the host collective"
    assert "effects_barrier" in events[:events.index("Allreduce")], events


class _Waited(Exception):
    pass


def test_save_checkpoint_waits_before_anything_else(monkeypatch):
    """Every save_checkpoint branch can reach a host collective (Barrier,
    gather), so the wait is its first statement: a bare driver must hit it
    before touching any attribute."""
    def _wait():
        raise _Waited

    monkeypatch.setattr(md, "_wait_for_mpi_device_work", _wait)
    drv = md.ModelDriver.__new__(md.ModelDriver)
    with pytest.raises(_Waited):
        drv.save_checkpoint(0, 0.0)


def test_a_failed_wait_aborts_the_job_under_mpi(monkeypatch):
    """An async error surfaced by the wait may hit one rank only; its peers
    are about to enter a host collective, so the job must abort, not hang."""
    aborted = []

    class _SpyComm(MPI.Intracomm):
        def Get_size(self):
            return 2

        def Abort(self, code=0):
            aborted.append(code)

    def _fail():
        raise RuntimeError("synthetic async failure")

    monkeypatch.setattr(jax, "effects_barrier", _fail)
    monkeypatch.setattr(MPI, "COMM_WORLD", _SpyComm(MPI.COMM_WORLD))
    with pytest.raises(RuntimeError, match="synthetic"):
        md._wait_for_mpi_device_work()
    assert aborted == [1]
