"""``_record_final_state_digest`` must gather sharded state before hashing.

Under multi-controller SPMD (#693/#749) the final state leaves span
non-addressable devices; hashing them raises and the 2x3-GPU receipt run
(job 26037824) logged "Could not record final state digest", leaving
``legoesm reproduce --check`` without a reference.

Fix under test: the SPMD branch gathers state/tracers/carry through
``_gather_spmd_tree_to_host`` (collective — every process reaches it via
``run()``) and gates the manifest write to process 0.  The serial / mpi4jax
lanes are byte-unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import legoesm.driver.model_driver as model_driver
import legoesm.driver.restart as restart
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.restart import RUN_MANIFEST_FILENAME


@pytest.fixture()
def digest_spies(monkeypatch):
    """Spy out the digest/record helpers (function-scope imports re-resolve
    module attributes at call time, so patching the module works)."""
    digest_spy = MagicMock(return_value="deadbeef")
    record_spy = MagicMock()
    monkeypatch.setattr(restart, "pytree_state_digest", digest_spy)
    monkeypatch.setattr(restart, "record_state_digest", record_spy)
    return digest_spy, record_spy


def _fake_driver(tmp_path, spmd):
    fake = MagicMock()
    fake._mpi_rank = None
    fake._output_dir = tmp_path
    fake._is_spmd_multiprocess = lambda: spmd
    fake.state, fake.tracers, fake._carry_aux = ("S",), ("T",), ("C",)
    # Tag-wrap so the test can assert the DIGEST saw the gathered trees.
    fake._gather_spmd_tree_to_host = MagicMock(
        side_effect=lambda tree: ("gathered", tree))
    (tmp_path / RUN_MANIFEST_FILENAME).write_text("{}")
    return fake


def test_spmd_process0_gathers_then_writes(tmp_path, digest_spies, monkeypatch):
    digest_spy, record_spy = digest_spies
    monkeypatch.setattr(model_driver.jax, "process_index", lambda: 0)
    fake = _fake_driver(tmp_path, spmd=True)

    ModelDriver._record_final_state_digest(fake)

    assert fake._gather_spmd_tree_to_host.call_count == 3
    digest_spy.assert_called_once_with(
        ("gathered", ("S",)), ("gathered", ("T",)), ("gathered", ("C",)))
    record_spy.assert_called_once_with(
        tmp_path / RUN_MANIFEST_FILENAME, "deadbeef")


def test_spmd_nonroot_gathers_collectively_but_never_writes(
        tmp_path, digest_spies, monkeypatch):
    # Non-root MUST still reach the gather (it is a collective — skipping it
    # on rank>0 would deadlock process 0) and MUST NOT write the manifest.
    digest_spy, record_spy = digest_spies
    monkeypatch.setattr(model_driver.jax, "process_index", lambda: 2)
    fake = _fake_driver(tmp_path, spmd=True)

    ModelDriver._record_final_state_digest(fake)

    assert fake._gather_spmd_tree_to_host.call_count == 3
    digest_spy.assert_not_called()
    record_spy.assert_not_called()


def test_serial_lane_is_unchanged(tmp_path, digest_spies):
    # Non-SPMD: no gather, digest over the ORIGINAL trees (byte-identical
    # to the pre-fix behaviour).
    digest_spy, record_spy = digest_spies
    fake = _fake_driver(tmp_path, spmd=False)

    ModelDriver._record_final_state_digest(fake)

    fake._gather_spmd_tree_to_host.assert_not_called()
    digest_spy.assert_called_once_with(("S",), ("T",), ("C",))
    record_spy.assert_called_once()
