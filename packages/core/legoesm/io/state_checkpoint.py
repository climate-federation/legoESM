"""Generic state checkpoint/restart for any model component.

Works for any NamedTuple whose fields are :class:`~legoesm.core.field.Field`
objects or plain ``jax.Array`` / ``numpy.ndarray``.  Unlike
:mod:`legoesm.driver.restart` (atmosphere-specific, driver-level), this module
knows nothing about ``T``, ``u``, ``v``, or ``q_v`` — it iterates over
``state._fields`` generically, and stays in the legoesm-core substrate.

Shares the pure SHA-256 state digest with the rest of the substrate via
:mod:`legoesm.io.state_digest`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax
import numpy as np

from legoesm.core.field import Field
from legoesm.io.state_digest import compute_state_digest


# ---------------------------------------------------------------------------
# Platform helpers (duplicated from restart.py to avoid private imports)
# ---------------------------------------------------------------------------

def _get_jax_version() -> str:
    try:
        import jax
        return jax.__version__
    except Exception:
        return "unknown"


def _get_jax_x64() -> bool:
    try:
        import jax
        return jax.config.x64_enabled  # type: ignore[attr-defined]
    except Exception:
        return False


def _get_platform_tag() -> str:
    return f"{platform.system().lower()}-{platform.machine()}"


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_arrays(state) -> dict[str, np.ndarray]:
    """Extract numpy arrays from any NamedTuple of Fields / arrays.

    For ``Field`` attributes, uses ``.data``.
    For raw ``jax.Array`` / ``numpy.ndarray``, uses the value directly.
    ``None`` fields are skipped.

    Pulls all device arrays in a single batched ``jax.device_get`` so
    transfers can overlap on the GPU runtime instead of serialising
    one-per-leaf at every checkpoint write.
    """
    names: list[str] = []
    values = []
    for name in state._fields:
        val = getattr(state, name)
        if val is None:
            continue
        names.append(name)
        values.append(val.data if isinstance(val, Field) else val)
    if not values:
        return {}
    host = jax.device_get(values)
    return {n: np.asarray(v) for n, v in zip(names, host)}


def _extract_field_metadata(state) -> dict[str, dict[str, Any]]:
    """Capture Field metadata (dims, units, staggering) for reconstruction."""
    meta: dict[str, dict[str, Any]] = {}
    for name in state._fields:
        val = getattr(state, name)
        if val is None:
            meta[name] = {"is_none": True}
        elif isinstance(val, Field):
            meta[name] = {
                "is_field": True,
                "name": val.name,
                "dims": list(val.dims),
                "units": val.units,
                "long_name": val.long_name,
                "staggering": val.staggering,
            }
        else:
            meta[name] = {"is_field": False}
    return meta


def _meta_path(checkpoint_path: Path) -> Path:
    """Companion ``.meta.json`` path for an ``.npz`` checkpoint."""
    p = Path(checkpoint_path)
    stem = p.stem if p.suffix == ".npz" else p.name
    return p.with_name(stem + ".state_meta.json")


def _config_hash(config) -> str:
    """SHA-256 of a NamedTuple config serialized as sorted JSON.

    Works for any NamedTuple with ``_asdict()``.  Falls back to str()
    for non-serializable values.
    """
    if config is None:
        return ""

    def _default(obj):
        if hasattr(obj, "_asdict"):
            return obj._asdict()
        return str(obj)

    text = json.dumps(config._asdict(), sort_keys=True, default=_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_state_checkpoint(
    path: str | Path,
    state,
    step: int,
    elapsed_time_s: float,
    dt: float,
    config=None,
    *,
    prognostic_fields: set[str] | None = None,
    model_version: str | None = None,
) -> None:
    """Save any state NamedTuple to ``.npz`` + companion metadata JSON.

    Parameters
    ----------
    path : Path-like
        Output ``.npz`` file path.
    state
        Any ``NamedTuple`` whose fields are ``Field`` or ``jax.Array``.
    step : int
        Current time-step index.
    elapsed_time_s : float
        Elapsed model time in seconds.
    dt : float
        Time-step size in seconds (recorded for CFL diagnostics on reload).
    config : NamedTuple, optional
        Run configuration; hashed for validation on reload.
    prognostic_fields : set of str, optional
        If given, only these fields are saved.  By default all fields
        are saved.
    model_version : str, optional
        Version string recorded in metadata.  When ``None`` (default) the
        installed package version is used (single-sourced from pyproject).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_version is None:
        from legoesm._version import __version__ as model_version

    arrays = _extract_arrays(state)
    field_meta = _extract_field_metadata(state)

    if prognostic_fields is not None:
        arrays = {k: v for k, v in arrays.items() if k in prognostic_fields}

    # Save arrays — np.savez appends .npz if not present
    np.savez(path, **arrays)
    # Resolve the actual path (np.savez may have appended .npz)
    actual_path = path if path.suffix == ".npz" else path.with_suffix(
        path.suffix + ".npz" if path.suffix else ".npz"
    )
    if not actual_path.exists():
        actual_path = path  # fallback

    # Compute integrity digest (over saved arrays only)
    digest = compute_state_digest(arrays)
    cfg_hash = _config_hash(config)

    # Assemble metadata
    meta = {
        "model_version": model_version,
        "creation_time": datetime.now(timezone.utc).isoformat(),
        "platform": _get_platform_tag(),
        "jax_version": _get_jax_version(),
        "jax_x64_enabled": _get_jax_x64(),
        "numpy_version": np.__version__,
        "config_hash": cfg_hash,
        "state_digest": digest,
        "state_type": type(state).__qualname__,
        "field_names": list(state._fields),
        "saved_fields": sorted(arrays.keys()),
        "field_metadata": field_meta,
        "field_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "field_dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "step": step,
        "elapsed_time_s": elapsed_time_s,
        "dt": dt,
        "git_hash": _get_git_hash(),
    }

    # Atomic write: write to temp file then rename to avoid partial metadata
    meta_file = _meta_path(actual_path)
    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=meta_file.parent, suffix=".tmp", prefix=".state_meta_"
    )
    try:
        with open(tmp_fd, "w") as f:
            json.dump(meta, f, indent=2)
        Path(tmp_name).replace(meta_file)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_state_checkpoint(
    path: str | Path,
    state_template,
    *,
    config=None,
    strict: bool = True,
    restore_static: bool = False,
    static_fields: set[str] | None = None,
) -> tuple:
    """Load state from checkpoint.

    Parameters
    ----------
    path : Path-like
        Path to ``.npz`` checkpoint.
    state_template
        An existing state instance used as the reconstruction template.
        Field metadata (dims, units, staggering) and any static fields
        are taken from this template.
    config : NamedTuple, optional
        If provided with *strict*, the config hash is verified.
    strict : bool
        Enable validation checks (default True).
    restore_static : bool
        If True, load static fields from file instead of template.
    static_fields : set of str, optional
        Fields to take from *state_template* instead of the file.
        Ignored when *restore_static* is True.

    Returns
    -------
    tuple
        ``(state, step, elapsed_time_s, metadata_dict)``
    """
    import jax.numpy as jnp

    path = Path(path)
    if not path.exists():
        # numpy may have appended .npz
        if path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load metadata
    meta_file = _meta_path(path)
    metadata: dict | None = None
    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)
    elif strict:
        warnings.warn(
            f"No companion metadata at {meta_file}; skipping validation.",
            stacklevel=2,
        )

    # Validate before loading heavy arrays
    if strict and metadata is not None:
        issues = _validate_metadata(metadata, state_template, config)
        for issue in issues:
            if issue.startswith("ERROR"):
                raise ValueError(issue)
            else:
                warnings.warn(issue, stacklevel=2)

    # Load arrays (cast to active precision)
    from legoesm.core.precision import get_policy
    storage_dtype = get_policy().storage

    static_fields = static_fields or set()

    # Reconstruct state — use context manager to close NPZ file handle
    field_meta = metadata.get("field_metadata", {}) if metadata else {}
    kwargs = {}
    loaded_arrays: dict[str, np.ndarray] = {}

    with np.load(path) as npz:
        for name in state_template._fields:
            template_val = getattr(state_template, name)

            # Decide whether to use template or loaded data
            use_template = (
                (not restore_static and name in static_fields)
                or name not in npz
            )

            if use_template:
                kwargs[name] = template_val
                continue

            raw = npz[name]
            loaded_arrays[name] = raw
            arr = jnp.array(raw, dtype=storage_dtype)

            # Reconstruct Field if the original was a Field
            if isinstance(template_val, Field):
                fm = field_meta.get(name, {})
                kwargs[name] = Field(
                    data=arr,
                    name=fm.get("name", template_val.name),
                    dims=tuple(fm.get("dims", template_val.dims)),
                    units=fm.get("units", template_val.units),
                    long_name=fm.get("long_name", template_val.long_name),
                    staggering=fm.get("staggering", template_val.staggering),
                )
            elif template_val is None:
                # Slot was None in the template but exists in the checkpoint.
                # If the checkpoint's field_meta says it was a Field (e.g. the
                # prognostic tke carry), reconstruct the Field — grafting the
                # raw array would break every ``.data`` consumer downstream
                # (codex MED 2026-07-27). No field_meta => it was a raw-array
                # slot; keep the raw array (old behaviour).
                fm = field_meta.get(name)
                if fm:
                    kwargs[name] = Field(
                        data=arr,
                        name=fm.get("name", name),
                        dims=tuple(fm.get("dims", ())),
                        units=fm.get("units", ""),
                        long_name=fm.get("long_name", ""),
                        staggering=fm.get("staggering", "cell"),
                    )
                else:
                    kwargs[name] = arr
            else:
                kwargs[name] = arr

    # Verify digest after loading — only when all saved fields were loaded
    # (skipped static fields change the set, so digest won't match)
    if strict and metadata is not None and loaded_arrays:
        saved_fields = set(metadata.get("saved_fields", []))
        if saved_fields == set(loaded_arrays.keys()):
            saved_digest = metadata.get("state_digest", "")
            if saved_digest:
                current_digest = compute_state_digest(loaded_arrays)
                if current_digest != saved_digest:
                    raise ValueError(
                        "State digest mismatch: loaded arrays do not match "
                        "the SHA-256 digest recorded at save time. The "
                        "checkpoint may be corrupted."
                    )

    state = type(state_template)(**kwargs)

    step = metadata.get("step", 0) if metadata else 0
    elapsed = metadata.get("elapsed_time_s", 0.0) if metadata else 0.0

    return state, step, elapsed, metadata


def validate_state_checkpoint(
    path: str | Path,
    state_template,
) -> list[str]:
    """Validate checkpoint compatibility without loading arrays.

    Returns a list of issues (empty if compatible).
    Each issue is prefixed with ``ERROR:`` (fatal) or ``WARNING:``
    (non-fatal).
    """
    path = Path(path)
    meta_file = _meta_path(path)

    if not meta_file.exists():
        return ["ERROR: no companion metadata file found"]

    with open(meta_file) as f:
        metadata = json.load(f)

    return _validate_metadata(metadata, state_template, config=None)


# ---------------------------------------------------------------------------
# Validation internals
# ---------------------------------------------------------------------------

def _validate_metadata(
    metadata: dict,
    state_template,
    config,
) -> list[str]:
    """Check metadata against a state template. Returns list of issues."""
    issues: list[str] = []

    # JAX x64 mode
    if _get_jax_x64() != metadata.get("jax_x64_enabled"):
        saved = "enabled" if metadata.get("jax_x64_enabled") else "disabled"
        current = "enabled" if _get_jax_x64() else "disabled"
        issues.append(
            f"WARNING: JAX x64 mismatch: saved with x64={saved}, "
            f"current x64={current}."
        )

    # Config hash
    if config is not None and metadata.get("config_hash"):
        current_hash = _config_hash(config)
        if current_hash != metadata["config_hash"]:
            issues.append(
                "WARNING: config hash mismatch — run configuration "
                "differs from checkpoint."
            )

    # Field name compatibility
    saved_fields = set(metadata.get("saved_fields", []))
    template_fields = set(state_template._fields)
    missing = saved_fields - template_fields
    if missing:
        issues.append(
            f"WARNING: checkpoint has fields not in template: {sorted(missing)}"
        )

    # Shape compatibility
    saved_shapes = metadata.get("field_shapes", {})
    for name in state_template._fields:
        val = getattr(state_template, name)
        if val is None or name not in saved_shapes:
            continue
        template_shape = val.shape if isinstance(val, Field) else (
            val.shape if hasattr(val, "shape") else None
        )
        if template_shape is not None:
            saved_shape = tuple(saved_shapes[name])
            if template_shape != saved_shape:
                issues.append(
                    f"ERROR: shape mismatch for '{name}': "
                    f"template {template_shape} vs checkpoint {saved_shape}"
                )

    return issues
