"""Resumable LES driver state (``--checkpoint`` / ``--wall-limit-s``).

A run that reaches its wall-clock limit writes everything it needs to continue
and exits :data:`EXIT_CHECKPOINTED` (EX_TEMPFAIL); the job script resubmits and
the next job resumes from the file. A resumed run is bit-identical to an
uninterrupted one: arrays are stored exactly, and a checkpoint carries a
fingerprint of the run (options, resolved inputs, source) so it is never resumed
under a different configuration or edited code.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import legoesm

EXIT_CHECKPOINTED = 75


def to_host(tree):
    """The pytree with every array leaf copied to a host NumPy array (exact)."""
    return jax.tree_util.tree_map(np.asarray, tree)


def run_fingerprint(record: dict, inputs, source_files) -> str:
    """One stable string for "the same run": ``record`` (options and resolved
    configuration), a digest of the resolved ``inputs`` pytree (initial state,
    forcing) and a digest of the source (``source_files`` plus every ``legoesm``
    .py file)."""
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(inputs):
        a = np.ascontiguousarray(np.asarray(leaf))
        h.update(str((a.dtype, a.shape)).encode())
        h.update(a.tobytes())
    src = hashlib.sha256()
    for f in sorted(Path(p).resolve() for p in source_files):
        src.update(f.name.encode())
        src.update(f.read_bytes())
    for root in sorted(legoesm.__path__):
        for f in sorted(Path(root).rglob("*.py")):
            src.update(str(f.relative_to(root)).encode())
            src.update(f.read_bytes())
    runtime = dict(python=platform.python_version(), numpy=np.__version__,
                   jax=jax.__version__, jaxlib=_jaxlib_version(),
                   backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
                   xla_flags=os.environ.get("XLA_FLAGS", ""))
    return json.dumps(dict(record=record, inputs=h.hexdigest(), source=src.hexdigest(),
                           runtime=runtime), sort_keys=True, default=str)


def _jaxlib_version() -> str:
    import jaxlib  # noqa: PLC0415
    return jaxlib.__version__


def save_checkpoint(path: Path, payload: dict, fingerprint: str) -> None:
    """Write ``payload`` (arrays copied to the host) atomically: temp file, rename."""
    path = Path(path)
    body = pickle.dumps(to_host(payload), protocol=pickle.HIGHEST_PROTOCOL)
    blob = {"fingerprint": fingerprint, "sha256": hashlib.sha256(body).hexdigest(),
            "payload": body}
    tmp = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with open(tmp, "wb") as fh:
        pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def load_checkpoint(path: Path, fingerprint: str) -> dict:
    """The payload :func:`save_checkpoint` wrote; a checkpoint of a different run
    is refused, never silently continued."""
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    if blob["fingerprint"] != fingerprint:
        raise SystemExit(f"checkpoint {path} is from a different run configuration:\n"
                         f"  checkpoint: {blob['fingerprint']}\n  this run:   {fingerprint}")
    if hashlib.sha256(blob["payload"]).hexdigest() != blob["sha256"]:
        raise SystemExit(f"checkpoint {path} is corrupt (payload digest mismatch)")
    return pickle.loads(blob["payload"])


def add_checkpoint_args(p) -> None:
    """``--checkpoint`` / ``--wall-limit-s`` on a driver's argument parser."""
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="resumable-state file: resumed from when it exists, written "
                        "at every recorded frame and at --wall-limit-s, removed once "
                        "the run finishes")
    p.add_argument("--wall-limit-s", type=float, default=None,
                   help="stop after this many wall-clock seconds of stepping, write "
                        f"--checkpoint and exit {EXIT_CHECKPOINTED} (the job script "
                        "resubmits)")


def checkpoint_fingerprint(args, inputs, *source_files) -> str:
    """A driver run's identity: every parsed option except where the checkpoint
    lives and when to stop, the resolved IC + forcing ``inputs``, and the source
    (the driver's own files plus the package)."""
    record = {k: v for k, v in vars(args).items()
              if k not in ("checkpoint", "wall_limit_s")}
    return run_fingerprint(record, inputs, source_files)


def restore_state(saved):
    """A state pytree from a checkpoint payload, arrays exactly as written."""
    return jax.tree_util.tree_map(
        lambda a: jnp.asarray(a) if isinstance(a, np.ndarray) else a, saved)
