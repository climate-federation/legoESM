"""A stale checkpoint must say what to do, not raise a tree traceback.

Removing the nine gray-radiation knobs on 2026-08-12 took
``AIMIPClassicalParams`` from 59 trainable entries to 50, so every checkpoint
written before that date fails to load. The owner's decision is to retrain
rather than migrate — so the only requirement is that the failure be legible.
Before this wrapper it surfaced as ``RuntimeError: Deserialised leaf at path
... has changed shape`` from inside equinox's ``tree_map_with_path``.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("JAX_ENABLE_X64", "1")

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from legoesm.ml.checkpoint_io import (  # noqa: E402
    CheckpointStructureError,
    load_checkpoint_or_fail,
)


class _Params(eqx.Module):
    """Stand-in for AIMIPClassicalParams: a dict of named scalar leaves."""

    raw_values: dict


def _params(n: int) -> _Params:
    return _Params(raw_values={f"k{i}": jnp.zeros(()) for i in range(n)})


def test_a_stale_checkpoint_names_the_file_and_the_remedy(tmp_path):
    path = tmp_path / "params.eqx"
    eqx.tree_serialise_leaves(path, _params(59))

    with pytest.raises(CheckpointStructureError) as excinfo:
        load_checkpoint_or_fail(path, _params(50), what="the classical variant")

    msg = str(excinfo.value)
    assert str(path) in msg
    assert "50" in msg                      # what the current skeleton wants
    assert "the classical variant" in msg   # which load failed
    assert "retrain" in msg.lower()         # what to do about it
    # This case is the DANGEROUS one: equinox reads leaves positionally, so
    # 59 scalars into a 50-scalar skeleton raises nothing at all — it loads a
    # prefix of a stale checkpoint and trains on it. The guard catches it by
    # the bytes left unread, so there is no equinox exception to chain here.
    assert "silently" in msg


def test_the_equinox_complaint_is_preserved_when_there_is_one(tmp_path):
    """The other failure mode — a real deserialise error — must keep its
    cause, so the original message is still available for debugging."""
    path = tmp_path / "m.eqx"
    eqx.tree_serialise_leaves(path, _Params(raw_values={"a": jnp.zeros(4)}))

    with pytest.raises(CheckpointStructureError) as excinfo:
        load_checkpoint_or_fail(path, _Params(raw_values={"a": jnp.zeros(7)}))
    assert excinfo.value.__cause__ is not None


def test_a_matching_checkpoint_still_loads(tmp_path):
    """The wrapper must not turn a good load into an error — without this the
    test above passes on a function that always raises."""
    path = tmp_path / "params.eqx"
    original = _Params(raw_values={"a": jnp.asarray(1.5), "b": jnp.asarray(2.5)})
    eqx.tree_serialise_leaves(path, original)

    loaded = load_checkpoint_or_fail(path, _params(2))
    assert float(loaded.raw_values["k0"]) == pytest.approx(1.5)
    assert float(loaded.raw_values["k1"]) == pytest.approx(2.5)


def test_a_shape_change_is_also_caught(tmp_path):
    """Same leaf count, different shape — the other way a stale checkpoint
    fails, and the one the classical arm actually hit."""
    path = tmp_path / "m.eqx"
    eqx.tree_serialise_leaves(path, _Params(raw_values={"a": jnp.zeros(4)}))

    with pytest.raises(CheckpointStructureError):
        load_checkpoint_or_fail(path, _Params(raw_values={"a": jnp.zeros(7)}))


def test_a_missing_file_is_not_reported_as_a_structure_mismatch(tmp_path):
    """Unrelated failures must propagate: telling someone to retrain when they
    merely typed the path wrong sends them off for hours."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint_or_fail(tmp_path / "absent.eqx", _params(2))


def test_the_real_classical_skeleton_has_fifty_knobs():
    """Ties the message's numbers to the code. If the parameter set changes
    again, this fails and the message gets updated with it."""
    from legoesm.training.aimip_params import AIMIPClassicalParams

    assert len(AIMIPClassicalParams.from_defaults().raw_values) == 50


def test_a_truncated_file_is_also_actionable(tmp_path):
    """The other direction: fewer leaves on disk than the skeleton wants. The
    guard must convert equinox's read error too, not only the surplus case."""
    path = tmp_path / "short.eqx"
    eqx.tree_serialise_leaves(path, _params(2))

    with pytest.raises(CheckpointStructureError) as excinfo:
        load_checkpoint_or_fail(path, _params(40), what="a truncated file")
    assert "retrain" in str(excinfo.value).lower()


def test_a_suffixless_path_still_resolves(tmp_path):
    """equinox maps ``foo`` to ``foo.eqx``; opening the handle ourselves must
    not break that (codex)."""
    eqx.tree_serialise_leaves(tmp_path / "m.eqx", _params(2))
    loaded = load_checkpoint_or_fail(tmp_path / "m", _params(2))
    assert len(loaded.raw_values) == 2
