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


class CheckpointStructureError(RuntimeError):
    """A checkpoint's pytree does not match the current model skeleton."""


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

    path = Path(path)
    # equinox resolves a suffix-less path to ``.eqx`` (_serialisation._with_suffix);
    # opening the handle ourselves would otherwise break that (codex).
    if path.suffix == "":
        path = path.with_suffix(".eqx")
    try:
        # Deserialise from an OPEN HANDLE, not the path, so the trailing bytes
        # can be checked below. equinox reads leaves POSITIONALLY and stops at
        # the skeleton's last one: a checkpoint with extra leaves whose shapes
        # happen to line up loads SILENTLY and wrong, which is exactly what a
        # 59-knob file does against the 50-knob skeleton when the dropped
        # entries sit at the end.
        with open(path, "rb") as handle:
            loaded = eqx.tree_deserialise_leaves(handle, skeleton)
            leftover = handle.read(1)
        if leftover:
            n_expected = len(jtu.tree_leaves(skeleton))
            label = f" for {what}" if what else ""
            raise CheckpointStructureError(
                f"checkpoint {path} has MORE data than the current parameter "
                f"set{label} consumes: the skeleton took {n_expected} array "
                f"leaves and the file still has bytes left, so the load would "
                f"have silently used a prefix of a stale checkpoint.\n"
                f"Most likely it predates the 2026-08-12 removal of the "
                f"gray-radiation knobs (AIMIPClassicalParams went from 59 "
                f"trainable entries to 50). Those checkpoints are not migrated "
                f"by design — retrain from scratch with the current code."
            )
        return loaded
    except CheckpointStructureError:
        raise
    except (RuntimeError, EOFError, ValueError) as exc:
        n_expected = len(jtu.tree_leaves(skeleton))
        label = f" for {what}" if what else ""
        raise CheckpointStructureError(
            f"checkpoint {path} does not match the current parameter set"
            f"{label}: the skeleton has {n_expected} array leaves, and equinox "
            f"rejected the file with: {exc}\n"
            f"Most likely the checkpoint predates the 2026-08-12 removal of the "
            f"gray-radiation knobs (AIMIPClassicalParams went from 59 trainable "
            f"entries to 50). Those checkpoints are not migrated by design — "
            f"retrain from scratch with the current code."
        ) from exc
