"""Checkpoint loading that fails with an actionable message.

`equinox.tree_deserialise_leaves` streams leaves positionally (each written
with `np.save`; there is no name or count header). A checkpoint whose pytree
no longer matches the current skeleton therefore surfaces as a bare
``RuntimeError`` from equinox's internal ``_assert_same_impl``, raised inside a
``tree_map_with_path`` — a traceback that says a leaf "changed shape" and
nothing about which model or what to do.

That is what a 2026-08-11 checkpoint now hits: the gray-radiation knobs left
``AIMIPClassicalParams`` on 2026-08-12, so its parameter set went from 59
entries to 50. Those checkpoints are abandoned by decision of the model owner —
retrain, no migration shim — and this module makes the failure say so.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


# Deliberate, owner-approved layout changes; their checkpoints are not
# migrated, by design.
_KNOWN_LAYOUT_BREAKS = (
    "Most likely the checkpoint predates one of these deliberate layout "
    "changes:\n"
    "  - 2026-08-12: gray-radiation knobs removed (AIMIPClassicalParams went "
    "from 59 trainable entries to 50);\n"
    "  - 2026-10-15: AIMIP spatial surface fields the selected schemes never "
    "read (e.g. z0 under the constant bulk law) are no longer built, so a "
    "checkpoint that still holds them has extra leaves.\n"
    "Those checkpoints are not migrated by design — retrain from scratch "
    "with the current code.")


class CheckpointStructureError(RuntimeError):
    """A checkpoint's pytree does not match the current model skeleton."""


# Sidecar record of a checkpoint's AIMIP knobs (user 2026-10-28, R1): which
# knobs were frozen and the range each was trained with.
KNOB_RECORD_FORMAT = "legoesm-aimip-knobs"
KNOB_RECORD_VERSION = 1


def knob_record_path(path) -> Path:
    """``epoch_0003.eqx`` -> ``epoch_0003.knobs.json``."""
    return checkpoint_path(path).with_suffix(".knobs.json")


def knob_record(model) -> dict | None:
    """The knob record of the AIMIP parameter module(s) in ``model``, or None
    when there is none.  Optimizer-moment copies (None / masked slots at
    frozen knobs) are skipped; two different records in one tree raise."""
    import jax.tree_util as jtu

    has = lambda x: hasattr(x, "knob_record")  # noqa: E731
    recs = []
    jtu.tree_map(lambda x: recs.append(x.knob_record()) if has(x) else None,
                 model, is_leaf=has)
    recs = [r for r in recs if r is not None]
    if not recs:
        return None
    if any(r != recs[0] for r in recs[1:]):
        raise ValueError("checkpoint holds AIMIP parameter modules with "
                         "different frozen sets or ranges; no single record")
    return {"format": KNOB_RECORD_FORMAT, "version": KNOB_RECORD_VERSION,
            **recs[0]}


def write_knob_record(model, path) -> None:
    """Write ``model``'s knob record beside checkpoint ``path`` (atomic), or
    remove a stale one when the model has no AIMIP knobs."""
    import json
    import os

    rec, side = knob_record(model), knob_record_path(path)
    if rec is None:
        side.unlink(missing_ok=True)
        return
    tmp = side.with_name(f"{side.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(rec, sort_keys=True))
    os.replace(tmp, side)


def read_knob_record(path) -> dict | None:
    """The record beside ``path``; None for a checkpoint written without one
    (legacy).  An unknown format or version raises."""
    import json

    side = knob_record_path(path)
    if not side.exists():
        return None
    rec = json.loads(side.read_text())
    if (rec.get("format") != KNOB_RECORD_FORMAT
            or rec.get("version") != KNOB_RECORD_VERSION):
        raise CheckpointStructureError(
            f"{side}: unknown knob record format/version "
            f"{rec.get('format')!r}/{rec.get('version')!r}")
    return rec


def check_knob_record(path, skeleton) -> None:
    """For loaders that call equinox directly: raise when the knob record
    beside ``path`` does not fit the AIMIP modules in ``skeleton`` (ranges).
    No record: nothing to check."""
    import jax.tree_util as jtu

    rec = read_knob_record(path)
    if rec is None:
        return
    has = lambda x: hasattr(x, "checkpoint_load_spec")  # noqa: E731
    jtu.tree_map(lambda x: x.checkpoint_load_spec(rec) if has(x) else None,
                 skeleton, is_leaf=has)


def checkpoint_path(path) -> Path:
    """The file :func:`load_checkpoint_or_fail` opens for ``path``.

    equinox resolves a suffix-less path to ``.eqx`` (_serialisation._with_suffix).
    """
    path = Path(path)
    return path.with_suffix(".eqx") if path.suffix == "" else path


def load_checkpoint_or_fail(path, skeleton: Any, *, what: str = "") -> Any:
    """Deserialise ``path`` into ``skeleton``, or raise a message a human can act on.

    Parameters
    ----------
    path : str or pathlib.Path
        Checkpoint file (``epoch_NNNN.eqx`` / ``params.eqx``).
    skeleton : Any
        The untrained pytree the checkpoint must match leaf for leaf.
    what : str, optional
        What is being loaded (e.g. ``"classical params"``), quoted back in the
        error so the reader knows which of several checkpoints failed.

    Returns
    -------
    Any
        The deserialised pytree.

    Raises
    ------
    CheckpointStructureError
        The checkpoint does not match ``skeleton``. Never a partial load: the
        exception replaces the result, it does not accompany one.
    FileNotFoundError
        Propagated unchanged — a missing file is not a structure mismatch.

    Notes
    -----
    Only the structure-mismatch failures equinox raises (``RuntimeError`` from
    its leaf-by-leaf type/shape/dtype assertions, plus the truncation errors of
    a short file) are converted. Anything else — OOM, a device error, a bug in
    a custom filter spec — propagates untouched, so this wrapper cannot hide an
    unrelated fault.
    """
    import equinox as eqx
    import jax.tree_util as jtu

    # Opening the handle ourselves would otherwise break equinox's suffix rule (codex).
    path = checkpoint_path(path)
    try:
        # Deserialise from an OPEN HANDLE, not the path, so the trailing bytes
        # can be checked below. equinox reads leaves POSITIONALLY and stops at
        # the skeleton's last one: a checkpoint with extra leaves whose shapes
        # happen to line up loads SILENTLY and wrong, which is exactly what a
        # 59-knob file does against the 50-knob skeleton when the dropped
        # entries sit at the end.
        record = read_knob_record(path)
        has_spec = lambda x: hasattr(x, "checkpoint_load_spec")  # noqa: E731
        spec = jtu.tree_map(
            lambda x: (x.checkpoint_load_spec(record) if has_spec(x)
                       else eqx.default_deserialise_filter_spec),
            skeleton, is_leaf=has_spec)
        with open(path, "rb") as handle:
            loaded = eqx.tree_deserialise_leaves(handle, skeleton, spec)
            leftover = handle.read(1)
        if leftover:
            n_expected = len(jtu.tree_leaves(skeleton))
            label = f" for {what}" if what else ""
            raise CheckpointStructureError(
                f"checkpoint {path} has MORE data than the current parameter "
                f"set{label} consumes: the skeleton took {n_expected} array "
                f"leaves and the file still has bytes left, so the load would "
                f"have silently used a prefix of a stale checkpoint.\n"
                + _KNOWN_LAYOUT_BREAKS
            )
        return loaded
    except CheckpointStructureError:
        raise
    except (RuntimeError, EOFError, ValueError) as exc:
        n_expected = len(jtu.tree_leaves(skeleton))
        label = f" for {what}" if what else ""
        root = exc   # equinox re-raises a filter-spec error as a bare leaf path
        while root.__cause__ is not None or root.__context__ is not None:
            root = root.__cause__ or root.__context__
        cause = "" if root is exc else f" ({type(root).__name__}: {root})"
        raise CheckpointStructureError(
            f"checkpoint {path} does not match the current parameter set"
            f"{label}: the skeleton has {n_expected} array leaves, and equinox "
            f"rejected the file with: {exc}{cause}\n"
            + _KNOWN_LAYOUT_BREAKS
        ) from exc
