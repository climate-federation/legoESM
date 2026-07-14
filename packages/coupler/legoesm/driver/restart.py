"""Enhanced restart system with reproducibility checks.

Wraps the existing checkpoint I/O in ``legoesm.forcing.amip_config`` with
SHA-256 integrity digests, platform/JAX metadata, and cross-restart
reproducibility comparison.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import jax
import numpy as np

from legoesm.forcing.amip_config import save_checkpoint

# Pure state-digest helpers live in the legoesm-core substrate (io.state_digest)
# so io.state_checkpoint can share them without importing this driver-level
# module (federation carve, Step 3).  Re-exported here for back-compat callers
# of ``legoesm.driver.restart.{compute_state_digest,pytree_state_digest}``.
from legoesm.io.state_digest import compute_state_digest, pytree_state_digest

# Lazy imports to avoid circular dependency:
#   io.restart → driver.config → driver.__init__ → model_driver → io.restart
# We import ExperimentConfig and its serializer at function level instead.
_ExperimentConfig = None
_experiment_config_to_dict = None


def _get_experiment_config_type():
    global _ExperimentConfig, _experiment_config_to_dict
    if _ExperimentConfig is None:
        from legoesm.driver.config import (
            ExperimentConfig as _EC,
            experiment_config_to_dict as _ectd,
        )
        _ExperimentConfig = _EC
        _experiment_config_to_dict = _ectd
    return _ExperimentConfig, _experiment_config_to_dict


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RestartMetadata(NamedTuple):
    """Metadata written alongside every restart file."""

    model_version: str
    creation_time: str          # ISO 8601
    platform: str               # e.g. "darwin-arm64"
    jax_version: str
    jax_x64_enabled: bool
    numpy_version: str
    config_hash: str            # SHA-256 of JSON-serialized config
    state_digest: str           # SHA-256 of concatenated state array bytes
    resolution: int
    nlev: int
    step: int
    day: float
    git_hash: str               # empty string if not in a git repo
    # Storage dtype the digest was computed at (e.g. "float32").  Empty for
    # legacy checkpoints written before this field existed; the validator then
    # falls back to probing the active policy dtype and its fp32/fp64 sibling
    # (a cross-precision restart re-hashes losslessly at the SAVED dtype).
    storage_dtype: str = ""


class ReproducibilityReport(NamedTuple):
    """Result of comparing two restart files."""

    identical: bool
    differences: list           # list[str] — human-readable difference descriptions
    state_digest_match: bool
    config_match: bool


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

# compute_state_digest / pytree_state_digest now live in io.state_digest
# (imported above) — single source in the legoesm-core substrate.


def _config_to_dict_any(config) -> dict:
    """Convert any config type to a dict for serialization.

    Accepts ``ExperimentConfig`` (canonical) or ``AMIPExperimentConfig``
    (legacy).  Returns a flat dict in both cases.
    """
    EC, ectd = _get_experiment_config_type()
    if isinstance(config, EC):
        return ectd(config)
    # Legacy AMIPExperimentConfig or anything with _asdict. Deferred driver.config
    # import (canonical codec) — restart routes through the cycle-prone driver
    # package, so a top-level import is unsafe.
    from legoesm.driver.config import config_to_dict
    return config_to_dict(config)


# ---------------------------------------------------------------------------
# Config-kind dispatch (atmosphere ExperimentConfig vs ocean runtime configs).
# The ocean runtime configs are arbitrarily-nested NamedTuples with no bespoke
# (grid/dycore/output) codec, so they use the recursive tagged codec in
# ``legoesm.ocean.config``.  ``config_kind`` is recorded in the manifest so a
# reader rebuilds with the matching deserializer; it defaults to "atmosphere"
# for manifests written before this field existed (back-compat).
# ---------------------------------------------------------------------------

def detect_config_kind(config) -> str:
    """Classify *config* as ``"ocean"`` or ``"atmosphere"`` for codec dispatch.

    Module-name based (no import side effects): every ocean runtime config
    NamedTuple (``LatLonCGridOceanConfig`` / ``OceanConfig`` /
    ``SpectralOceanConfig`` / ``MPASOceanConfig``) lives under ``legoesm.ocean``,
    while ``ExperimentConfig`` / ``AMIPExperimentConfig`` do not — so an
    atmosphere manifest never triggers an ocean import.
    """
    module = type(config).__module__ or ""
    if module == "legoesm.ocean" or module.startswith("legoesm.ocean."):
        return "ocean"
    return "atmosphere"


def _serialize_config(config, kind: str) -> dict:
    if kind == "ocean":
        from legoesm.ocean.config import ocean_config_to_dict
        return ocean_config_to_dict(config)
    return _config_to_dict_any(config)


def _deserialize_config(resolved: dict, kind: str):
    if kind == "ocean":
        from legoesm.ocean.config import ocean_config_from_dict
        return ocean_config_from_dict(resolved)
    from legoesm.driver.config import experiment_config_from_dict
    return experiment_config_from_dict(resolved)


def compute_config_hash(config, kind: str | None = None) -> str:
    """SHA-256 of the JSON-serialized *config* (kind-aware).

    Atmosphere configs keep the exact pre-existing serialization
    (``_config_to_dict_any``) so previously-recorded hashes are unchanged.
    """
    kind = kind or detect_config_kind(config)
    text = json.dumps(_serialize_config(config, kind), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
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


def _get_git_ref() -> str:
    """Current branch/ref name (empty string if detached or not a git repo)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            ref = result.stdout.strip()
            return "" if ref == "HEAD" else ref  # "HEAD" => detached
    except Exception:
        pass
    return ""


def _get_git_dirty() -> bool:
    """True if the working tree has uncommitted changes (False if unknown)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return False


def _state_arrays_from_checkpoint_args(state, q_v, q_c=None, q_r=None,
                                       carry_aux=None) -> dict[str, np.ndarray]:
    """Build a dict of numpy arrays mirroring save_checkpoint layout.

    Pulls every device array in a single ``jax.device_get`` so the
    runtime can pipeline the device→host transfers in parallel.
    The previous per-leaf ``np.asarray`` chain forced N serial
    transfers, blocking the GPU at every restart write.
    """
    names = ["T", "u", "v", "p_s", "phis", "q_v"]
    values = [
        state.T.data, state.u.data, state.v.data,
        state.p_s.data, state.phis.data, q_v,
    ]
    if q_c is not None:
        names.append("q_c"); values.append(q_c)
    if q_r is not None:
        names.append("q_r"); values.append(q_r)
    if carry_aux:
        for k, v in carry_aux.items():
            names.append(f"carry_{k}"); values.append(v)
    host = jax.device_get(values)
    return {n: np.asarray(v) for n, v in zip(names, host)}


def _meta_path(checkpoint_path: Path) -> Path:
    """Return the companion metadata path for a checkpoint file."""
    p = Path(checkpoint_path)
    # Strip .npz (and handle the case where numpy appends .npz automatically)
    stem = p.stem if p.suffix == ".npz" else p.name
    return p.with_name(stem + ".meta.json")


# ---------------------------------------------------------------------------
# Run manifest (Stage A1 reproducibility spine)
# ---------------------------------------------------------------------------
# A run manifest is written ONCE at the start of a run (driver setup / cmd_run),
# capturing everything needed to reconstruct it — independent of any checkpoint.
# It is the legoESM half of the unified provenance document (master plan §8.2);
# the section layout and field names align with the legoESM-ocean-runners
# ``experiment.tag`` so that tag is a valid [legoESM]+[reproducibility] subset:
#
#   [legoESM]          ref, commit                         (env layer)
#   [reproducibility]  runner_tag, python_version, jax_version, patches
#   [config]           resolved_config (full dict)         (config layer)
#   [result]           state_digest, rng_seeds, dataset_provenance,
#                      model_weights_provenance            (in-/post-run layer)
#
# ``legoesm reproduce`` (a follow-up) reads either a bare tag (env-only) or a
# full manifest (env + config + bit-digest ``--check``).
#
# Bit-identical replay holds per platform/precision/backend; the caveats (GPU
# reduction order, JIT cache, fp32 vs x64) are catalogued in
# docs/architecture/portability_gpu_mpi_precision.md ("Nondeterminism sources").

# Version policy: bump ONLY on a breaking shape change (removed/renamed field,
# changed meaning). validate_run_manifest requires the EXACT version, so a bump
# invalidates resume-into-existing-dir for every older manifest — populating an
# already-present optional field (e.g. dataset_provenance) is NOT a bump.
RUN_MANIFEST_SCHEMA_VERSION = 1
RUN_MANIFEST_FILENAME = "run_manifest.json"


def _json_safe(obj):
    """Recursively coerce a manifest value tree to JSON-native types.

    Preserves ``str``/``bool``/``int``/``float``/``None``; maps NumPy scalars to
    Python scalars, ``Path`` to ``str``, and arrays to lists; recurses through
    dicts/lists/tuples.  Raises ``TypeError`` on anything else so a *lossy*
    manifest is never written silently — the caller (best-effort CLI wrapper)
    warns instead.  This keeps the dict returned by :func:`build_run_manifest`
    byte-for-byte consistent with what :func:`write_run_manifest` serialises
    (no hidden ``default=str`` stringification that would make read-back differ,
    e.g. an integer seed silently becoming a string).
    """
    # NumPy scalar checks first: np.float64 is a subclass of float, so it would
    # otherwise pass the python-scalar check and reach json.dump unconverted.
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        # Non-NumPy array-like (a JAX ArrayImpl config leaf): summarise as
        # shape/dtype/range + content digest instead of raising -- a raw device
        # array crashed the whole provenance write (ocean runoff map).  NumPy
        # ndarrays above keep their existing full-list encoding.  The ocean
        # codec already summarises its own arrays; this covers other config
        # kinds reaching _json_safe directly.
        import hashlib as _hashlib
        arr = np.asarray(obj)
        if arr.ndim == 0:
            return _json_safe(arr.item())
        _c = np.ascontiguousarray(arr)
        return {"__array_summary__": {
            "shape": list(arr.shape), "dtype": str(arr.dtype),
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "sha256": _hashlib.sha256(
                str(arr.shape).encode() + str(arr.dtype).encode()
                + _c.tobytes()).hexdigest(),
        }}
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        # Require string keys rather than coercing with str(k): coercion would
        # silently rewrite an unsupported key (e.g. an object) and could collapse
        # distinct keys (1 and "1") into one — exactly the lossy behaviour this
        # normaliser exists to prevent.  JSON keys are strings anyway.
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"run-manifest dict key must be str, got "
                    f"{type(k).__name__!r} ({k!r})"
                )
            out[k] = _json_safe(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    raise TypeError(
        f"run-manifest value of type {type(obj).__name__!r} is not "
        f"JSON-serializable (value: {obj!r})"
    )


def dataset_provenance_entry(
    path,
    *,
    dataset_id: str | None = None,
    sha256: str | None = None,
    compute_sha256: bool = False,
) -> dict:
    """One ``[result].dataset_provenance`` entry for an input dataset.

    Records identity plus cheap integrity facts for the resolved path:
    ``{id, path, exists, size_bytes, mtime_utc, sha256}``. A checksum is
    recorded only when already known (pass it through from
    ``config/data_catalog.yaml``) or on explicit ``compute_sha256=True`` —
    hashing is opt-in because forcing files are routinely multi-GB and a run
    start must not stall on them. For a directory store (Zarr) size/sha256
    stay ``None``; resolved path + mtime still pin identity. Pure-Python,
    ``_json_safe``-clean, never raises on a missing path (``exists=False`` is
    itself provenance worth recording).
    """
    p = Path(path).expanduser()
    exists = p.exists()
    size_bytes = None
    mtime_utc = None
    if exists:
        try:
            stat = p.stat()
            mtime_utc = datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat()
            if p.is_file():
                size_bytes = int(stat.st_size)
        except OSError:
            pass  # stat raced a concurrent delete — keep exists, drop details
    digest = sha256
    if digest is None and compute_sha256 and exists and p.is_file():
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
    return {
        "id": dataset_id if dataset_id is not None else p.name,
        "path": str(p.resolve()),
        "exists": bool(exists),
        "size_bytes": size_bytes,
        "mtime_utc": mtime_utc,
        "sha256": digest,
    }


def build_run_manifest(
    config,
    *,
    command_line: str | None = None,
    runner_tag: str = "",
    patches: list | None = None,
    rng_seeds: dict | None = None,
    dataset_provenance: list | None = None,
    model_weights_provenance=None,
    state_digest: str | None = None,
    config_kind: str | None = None,
) -> dict:
    """Assemble the run-manifest dict (pure; does no I/O).

    Parameters
    ----------
    config
        An ``ExperimentConfig`` (canonical), legacy ``AMIPExperimentConfig``, or
        an ocean runtime config (``LatLonCGridOceanConfig`` / ``OceanConfig`` /
        ``SpectralOceanConfig``) — serialized into ``[config].resolved_config``
        so the run can be rebuilt.
    command_line
        The invoking command; defaults to ``" ".join(sys.argv)``.
    state_digest
        Filled after N steps by the reproduce/checkpoint path; ``None`` at start.
    config_kind
        ``"atmosphere"`` or ``"ocean"`` — selects the (de)serializer.  Defaults
        to auto-detection from the config type.
    """
    from legoesm._version import __version__

    kind = config_kind or detect_config_kind(config)

    raw = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "legoESM": {
            "ref": _get_git_ref(),
            "commit": _get_git_hash(),
        },
        "reproducibility": {
            "runner_tag": runner_tag,
            "python_version": platform.python_version(),
            "jax_version": _get_jax_version(),
            "jax_x64_enabled": _get_jax_x64(),
            "numpy_version": np.__version__,
            "legoesm_version": __version__,
            "platform": _get_platform_tag(),
            "git_dirty": _get_git_dirty(),
            "patches": list(patches) if patches else [],
        },
        "config": {
            "config_kind": kind,
            "resolved_config": _serialize_config(config, kind),
            "config_hash": compute_config_hash(config, kind),
        },
        "result": {
            "state_digest": state_digest,
            "rng_seeds": dict(rng_seeds) if rng_seeds else {},
            "dataset_provenance": list(dataset_provenance) if dataset_provenance else [],
            "model_weights_provenance": model_weights_provenance,
        },
        "run": {
            "command_line": (
                command_line if command_line is not None else " ".join(sys.argv)
            ),
            "creation_time": datetime.now(timezone.utc).isoformat(),
        },
    }
    # Normalise to JSON-native types up front so the returned dict matches the
    # serialised file exactly (and unsupported provenance values fail loudly).
    return _json_safe(raw)


def write_run_manifest(directory, config, *, exclusive: bool = False, **kwargs) -> Path:
    """Write ``run_manifest.json`` into *directory* and return its path.

    Call this at every run start (driver ``setup()`` / ``cmd_run``), not only at
    checkpoint time, so an interrupted or crashed run is still reconstructible.
    Extra keyword arguments are forwarded to :func:`build_run_manifest`.

    The content is always written atomically (temp file then link/rename), so a
    reader never observes a half-written manifest.  With ``exclusive=True`` the
    create is also *atomic-exclusive* (``os.link``): it raises ``FileExistsError``
    if the manifest already exists, which closes the check-then-create race when
    two processes start into the same output directory — only one wins the
    create, the other takes the read-and-validate path.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(config, **kwargs)  # already JSON-normalised
    path = directory / RUN_MANIFEST_FILENAME
    # PID-unique temp so concurrent writers never clobber each other's scratch.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    try:
        if exclusive:
            os.link(tmp, path)   # atomic; raises FileExistsError if path exists
        else:
            os.replace(tmp, path)  # atomic overwrite
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def read_run_manifest(path) -> dict:
    """Load a run manifest written by :func:`write_run_manifest`."""
    path = Path(path)
    if path.is_dir():
        path = path / RUN_MANIFEST_FILENAME
    with open(path) as f:
        return json.load(f)


_REQUIRED_MANIFEST_SECTIONS = ("legoESM", "reproducibility", "config", "result", "run")


def validate_run_manifest(manifest: dict) -> None:
    """Raise ``ValueError`` unless *manifest* is complete and reconstructable.

    Checks the exact schema version, the presence of every provenance section,
    and — the part that actually matters — that ``config.resolved_config``
    rebuilds an ``ExperimentConfig`` whose hash equals the recorded
    ``config.config_hash``.  That last check means a manifest cannot merely
    *look* well-formed (right keys) while carrying a config that can't be
    reconstructed or whose hash was tampered/corrupted.  Shared by the driver's
    write-once guard and by ``reproduce`` (a follow-up).
    """
    if not isinstance(manifest, dict):
        raise ValueError("run manifest is not a JSON object")
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported run-manifest schema_version "
            f"{manifest.get('schema_version')!r} "
            f"(expected {RUN_MANIFEST_SCHEMA_VERSION})"
        )
    # Every provenance section must be present AND a real object (a ``null`` or
    # missing section would leave the run unreconstructable while passing a mere
    # name-presence check).
    for section in _REQUIRED_MANIFEST_SECTIONS:
        if not isinstance(manifest.get(section), dict):
            raise ValueError(f"run manifest section [{section}] missing or not an object")
    # Required keys within sections.  Values may be empty strings for legitimate
    # reasons (e.g. commit/ref == "" outside a git repo), so check presence, not
    # truthiness.
    for key in ("ref", "commit"):
        if key not in manifest["legoESM"]:
            raise ValueError(f"run manifest [legoESM] missing {key!r}")
    for key in ("python_version", "jax_version"):
        if key not in manifest["reproducibility"]:
            raise ValueError(f"run manifest [reproducibility] missing {key!r}")
    for key in ("command_line", "creation_time"):
        if key not in manifest["run"]:
            raise ValueError(f"run manifest [run] missing {key!r}")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("run manifest [config] is not an object")
    config_hash = config.get("config_hash")
    if not isinstance(config_hash, str) or not config_hash:
        raise ValueError("run manifest [config].config_hash missing or empty")
    resolved = config.get("resolved_config")
    if not isinstance(resolved, dict) or not resolved:
        raise ValueError("run manifest [config].resolved_config missing or empty")
    # Reconstructability + integrity: the resolved config must rebuild and its
    # hash must match what was recorded.  ``config_kind`` selects the codec;
    # it defaults to "atmosphere" so manifests written before the field existed
    # still validate (back-compat).
    kind = config.get("config_kind", "atmosphere")
    rebuilt = _deserialize_config(resolved, kind)
    if compute_config_hash(rebuilt, kind) != config_hash:
        raise ValueError(
            "run manifest config_hash does not match its resolved_config "
            "(corrupt or tampered provenance)"
        )


def record_state_digest(manifest_path, state_digest: str) -> Path:
    """Record the post-run final ``state_digest`` into an existing manifest.

    The run-start manifest is otherwise immutable; this is the single sanctioned
    post-run update.  It fills ``result.state_digest`` (``None`` at run start) so
    that ``legoesm reproduce --check`` has a reference to compare a rerun's final
    state against.  The manifest is validated, then rewritten atomically with the
    digest set — provenance (config/env) is never touched, only the result.
    """
    manifest_path = Path(manifest_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / RUN_MANIFEST_FILENAME
    manifest = read_run_manifest(manifest_path)
    validate_run_manifest(manifest)
    manifest["result"]["state_digest"] = state_digest
    tmp = manifest_path.with_name(f"{manifest_path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(_json_safe(manifest), f, indent=2, sort_keys=True)
    tmp.replace(manifest_path)
    return manifest_path


def recorded_state_digest(manifest: dict) -> str:
    """Return the recorded final ``state_digest``, or raise if the run never set it.

    A missing digest means the *original* run did not complete (or predates digest
    recording), so there is nothing to reproduce against — a clear error beats a
    false ``reproduce --check`` pass.
    """
    digest = manifest.get("result", {}).get("state_digest")
    if not digest:
        raise ValueError(
            "run manifest has no recorded result.state_digest to check against; "
            "the original run may not have completed."
        )
    return digest


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_restart(
    path,
    state,
    q_v,
    step: int,
    day: float,
    config,
    *,
    diag_accumulators=None,
    q_c=None,
    q_r=None,
    carry_aux=None,
    model_version: str | None = None,
    backend: str = "npz",
) -> None:
    """Save a restart checkpoint together with reproducibility metadata.

    Accepts either ``ExperimentConfig`` (canonical) or
    ``AMIPExperimentConfig`` (legacy).  ExperimentConfig is serialized
    directly using its native JSON format.  Legacy AMIPExperimentConfig
    is still accepted for backward compatibility.

    Delegates array persistence to :func:`save_checkpoint` (npz) or
    :func:`save_checkpoint_zarr` (zarr) and writes a companion
    ``.meta.json`` alongside the checkpoint.

    Parameters
    ----------
    backend : str
        ``"npz"`` (default) or ``"zarr"``.
    model_version : str, optional
        Version string recorded in metadata.  When ``None`` (default) the
        installed package version is used (single-sourced from pyproject).
    """
    path = Path(path)
    if model_version is None:
        from legoesm._version import __version__ as model_version

    # 1. Delegate to appropriate backend
    if backend == "zarr":
        from legoesm.driver.checkpoint import save_checkpoint_zarr

        save_checkpoint_zarr(
            path,
            state,
            q_v,
            step,
            day,
            config,
            q_c=q_c,
            q_r=q_r,
            diag_accumulators=diag_accumulators,
        )
    elif backend == "npz":
        save_checkpoint(
            path,
            state,
            q_v,
            step,
            day,
            config,
            diag_accumulators=diag_accumulators,
            q_c=q_c,
            q_r=q_r,
            carry_aux=carry_aux,
        )
    else:
        raise ValueError(
            f"Unknown checkpoint backend {backend!r}; expected 'npz' or 'zarr'."
        )

    # 2. Compute integrity hashes.
    # Cast to storage_dtype before hashing so the digest matches what
    # load_checkpoint produces (it casts every array to storage_dtype on
    # reload, so a mixed-precision state would otherwise produce a
    # different hash on save vs load).
    from legoesm.core.precision import get_policy
    _sd = get_policy().storage
    state_arrays = {
        k: np.asarray(v, dtype=_sd)
        for k, v in _state_arrays_from_checkpoint_args(state, q_v, q_c, q_r).items()
    }
    digest = compute_state_digest(state_arrays)
    cfg_hash = compute_config_hash(config)

    # 3. Infer resolution / nlev from state
    # state.T.data has shape (6, n, n, nlev) for cubed-sphere
    t_shape = np.asarray(state.T.data).shape
    resolution = t_shape[1] if len(t_shape) >= 3 else 0
    nlev = t_shape[-1] if len(t_shape) >= 2 else 0

    # 4. Assemble metadata
    meta = RestartMetadata(
        model_version=model_version,
        creation_time=datetime.now(timezone.utc).isoformat(),
        platform=_get_platform_tag(),
        jax_version=_get_jax_version(),
        jax_x64_enabled=_get_jax_x64(),
        numpy_version=np.__version__,
        config_hash=cfg_hash,
        state_digest=digest,
        resolution=resolution,
        nlev=nlev,
        step=step,
        day=day,
        git_hash=_get_git_hash(),
        storage_dtype=np.dtype(_sd).name,
    )

    # 5. Write companion JSON
    meta_file = _meta_path(path)
    with open(meta_file, "w") as f:
        json.dump(meta._asdict(), f, indent=2)


def load_restart(
    path,
    grid,
    sigma,
    *,
    config=None,
    strict: bool = True,
):
    """Load a restart checkpoint and validate metadata.

    Parameters
    ----------
    path : Path-like
        Path to the ``.npz`` checkpoint file.
    grid : CubedSphereGrid
        Grid object (must match checkpoint resolution).
    sigma : SigmaCoordinate or HybridSigmaPressureCoordinate
        Vertical coordinate (must match checkpoint nlev).
    config : ExperimentConfig or AMIPExperimentConfig, optional
        If provided and *strict* is True the saved config hash is compared
        against this config.  Accepts either type.
    strict : bool
        When True (default) several validation checks are applied; failures
        raise :class:`ValueError` (for hard mismatches) or emit warnings.

    Returns
    -------
    tuple
        ``(state, q_v, step, day, loaded_config, diag_accumulators, q_c, q_r,
        metadata, carry_aux)`` where *metadata* is a :class:`RestartMetadata` (or ``None`` if the
        companion ``.meta.json`` is absent).
    """
    path = Path(path)

    # 0. Spectral checkpoint (iter 92): a discretization='spectral' run's
    #    ``ModelDriver.save_checkpoint`` writes the five ``*_hat`` coefficient arrays
    #    + a ``spectral_layout`` marker via ``np.savez`` — NOT the grid layout
    #    ``load_checkpoint_auto`` reads.  Reconstruct the ``SpectralHydrostaticState``
    #    directly (the SAME canonical helper the in-driver restart uses, template=None
    #    ⇒ plain Field coefficients) so an offline caller (the one-shot compare CLI,
    #    which then synthesizes grid winds via ``grid_winds_from_spectral``) can load
    #    it.  The spectral save carries no config / diag / q_c/q_r / carry_aux and no
    #    companion ``.meta.json`` → those are ``None``; q_v is the grid-space tracer.
    if path.is_file() and path.suffix == ".npz":
        with np.load(path) as d:
            if "spectral_layout" in d.files:
                from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
                    reconstruct_spectral_state_from_npz,
                )
                # strict (default): validate the coefficient shapes against the
                # configured grid (n_sh = grid.lap) + sigma (nlev) BEFORE
                # reconstructing — the template=None path can't, and the downstream
                # spectral→grid synthesis would otherwise fail with a less clear
                # error.  Skipped only when grid/sigma are absent (unit harness); a
                # SUPPLIED non-spectral grid (no .lap) is a config error → raise
                # rather than silently skip (Codex iter 92).
                if strict and grid is not None and sigma is not None:
                    lap = getattr(grid, "lap", None)
                    if lap is None:
                        raise ValueError(
                            f"spectral checkpoint loaded with a non-spectral grid "
                            f"({type(grid).__name__} has no .lap) — --grid-type must "
                            "be spectral/gaussian for a spectral restart."
                        )
                    n_sh = int(np.asarray(lap).shape[0])
                    nlev = int(np.asarray(sigma.sigma_full).shape[0])
                    expected = {
                        "vor_hat": (n_sh, nlev), "div_hat": (n_sh, nlev),
                        "T_hat": (n_sh, nlev), "lnps_hat": (n_sh,),
                        "phis_hat": (n_sh,),
                    }
                    for nm, sh in expected.items():
                        got = tuple(np.asarray(d[nm]).shape)
                        if got != sh:
                            raise ValueError(
                                f"spectral restart {nm} shape {got} != expected {sh} "
                                f"for the configured grid (n_sh={n_sh}) + sigma "
                                f"(nlev={nlev}); --grid-type/--resolution/--nlev must "
                                "match the run."
                            )
                state, step, day = reconstruct_spectral_state_from_npz(d)
                q_v = (
                    state.tracers["q_v"].data
                    if state.tracers is not None and "q_v" in state.tracers
                    else None
                )
                return state, q_v, step, day, None, None, None, None, None, None

    # 1. Delegate to auto-detecting loader (handles both .npz and .zarr).
    #    load_checkpoint_auto returns ExperimentConfig regardless of
    #    whether the checkpoint used the legacy AMIP or new format.
    from legoesm.driver.checkpoint import load_checkpoint_auto

    state, q_v, step, day, loaded_config, diag_accumulators, q_c, q_r, carry_aux = (
        load_checkpoint_auto(path, grid, sigma)
    )

    # 2. Try to read companion metadata
    meta_file = _meta_path(path)
    metadata: RestartMetadata | None = None

    if meta_file.exists():
        with open(meta_file) as f:
            raw = json.load(f)
        metadata = RestartMetadata(**raw)

        if strict:
            _validate_metadata(metadata, state, q_v, q_c, q_r, config)
    else:
        if strict:
            warnings.warn(
                f"No companion metadata file found at {meta_file}; "
                "skipping reproducibility checks.",
                stacklevel=2,
            )

    return state, q_v, step, day, loaded_config, diag_accumulators, q_c, q_r, metadata, carry_aux


def _validate_metadata(
    metadata: RestartMetadata,
    state,
    q_v,
    q_c,
    q_r,
    config,
) -> None:
    """Run strict validation checks on loaded restart metadata."""
    # JAX x64 mode
    if _get_jax_x64() != metadata.jax_x64_enabled:
        warnings.warn(
            f"JAX x64 mode mismatch: restart was saved with "
            f"x64={'enabled' if metadata.jax_x64_enabled else 'disabled'}, "
            f"but current session has "
            f"x64={'enabled' if _get_jax_x64() else 'disabled'}. "
            f"Loaded arrays have been cast to the active storage dtype.",
            stacklevel=3,
        )

    # Config hash comparison (only when caller provides a config)
    if config is not None:
        current_hash = compute_config_hash(config)
        if current_hash != metadata.config_hash:
            warnings.warn(
                "Config hash mismatch: the provided config differs from "
                "the config used when this restart was saved.",
                stacklevel=3,
            )

    # Resolution / nlev hard checks
    t_shape = np.asarray(state.T.data).shape
    loaded_res = t_shape[1] if len(t_shape) >= 3 else 0
    loaded_nlev = t_shape[-1] if len(t_shape) >= 2 else 0

    if loaded_res != metadata.resolution:
        raise ValueError(
            f"Resolution mismatch: restart metadata says resolution="
            f"{metadata.resolution} but loaded state has n={loaded_res}."
        )
    if loaded_nlev != metadata.nlev:
        raise ValueError(
            f"Vertical level mismatch: restart metadata says nlev="
            f"{metadata.nlev} but loaded state has nlev={loaded_nlev}."
        )

    # State digest verification (cross-precision aware).
    from legoesm.core.precision import get_policy
    raw_arrays = _state_arrays_from_checkpoint_args(state, q_v, q_c, q_r)
    _verify_state_digest(raw_arrays, metadata, np.dtype(get_policy().storage))


def _verify_state_digest(raw_arrays, metadata, active_dtype) -> None:
    """Verify ``metadata.state_digest`` against ``raw_arrays``.

    The digest was computed at SAVE time at the then-active storage dtype.
    Loading under a different precision policy up-casts the arrays, so hashing
    at the ACTIVE dtype spuriously failed every cross-precision restart (e.g.
    an fp32-written checkpoint resumed under ``--precision fp64``).  Re-casting
    the loaded arrays back to the SAVED dtype is lossless for an up-cast, so
    the digest can still be verified exactly.

    - ``metadata.storage_dtype`` recorded (new checkpoints): hash at that dtype.
    - Legacy metadata (field empty): try the active dtype, then its
      fp32/fp64 sibling; a sibling match is a verified cross-precision restart
      (warn, accept).
    - A LOSSY path (saved wider than loaded, e.g. fp64 checkpoint under fp32)
      cannot be verified — the load itself discarded bits; warn, don't raise.
    - No candidate matches at the saved/candidate dtypes: corruption -> raise.
    """
    def _digest_at(dt) -> str:
        return compute_state_digest(
            {k: np.asarray(v, dtype=dt) for k, v in raw_arrays.items()})

    active_dtype = np.dtype(active_dtype)
    if metadata.storage_dtype:
        saved_dtype = np.dtype(metadata.storage_dtype)
        if saved_dtype.itemsize > active_dtype.itemsize:
            warnings.warn(
                f"Cross-precision restart: checkpoint saved at "
                f"{saved_dtype.name} but loaded under {active_dtype.name}; the "
                "load discarded precision, so the state digest cannot be "
                "verified.",
                stacklevel=4,
            )
            return
        if _digest_at(saved_dtype) == metadata.state_digest:
            if saved_dtype != active_dtype:
                warnings.warn(
                    f"Cross-precision restart: digest verified at the saved "
                    f"dtype {saved_dtype.name}; arrays were up-cast to "
                    f"{active_dtype.name}.",
                    stacklevel=4,
                )
            return
    else:
        # Legacy metadata: dtype unknown.  Probe the active dtype, then its
        # fp32/fp64 sibling (the only storage dtypes the policy system uses).
        if _digest_at(active_dtype) == metadata.state_digest:
            return
        sibling = np.dtype(
            np.float64 if active_dtype == np.float32 else np.float32)
        if sibling.itemsize <= active_dtype.itemsize and \
                _digest_at(sibling) == metadata.state_digest:
            warnings.warn(
                f"Cross-precision restart (legacy metadata): digest verified "
                f"at {sibling.name}; arrays were up-cast to "
                f"{active_dtype.name}.",
                stacklevel=4,
            )
            return
    raise ValueError(
        "State digest mismatch: the loaded arrays do not match the "
        "SHA-256 digest recorded at save time. The checkpoint may be "
        "corrupted or was modified after saving."
    )


# ---------------------------------------------------------------------------
# Reproducibility comparison
# ---------------------------------------------------------------------------

def verify_reproducibility(path_a, path_b) -> ReproducibilityReport:
    """Compare the metadata of two restart files.

    This does **not** load the full checkpoint arrays — it only reads the
    companion ``.meta.json`` files and compares their digests and platform
    information.

    Parameters
    ----------
    path_a, path_b : Path-like
        Paths to two ``.npz`` checkpoint files.

    Returns
    -------
    ReproducibilityReport
    """
    meta_a = _load_meta(path_a)
    meta_b = _load_meta(path_b)

    differences: list[str] = []

    # State digest
    state_match = meta_a.state_digest == meta_b.state_digest
    if not state_match:
        differences.append(
            f"state_digest differs: {meta_a.state_digest[:16]}... vs "
            f"{meta_b.state_digest[:16]}..."
        )

    # Config hash
    config_match = meta_a.config_hash == meta_b.config_hash
    if not config_match:
        differences.append(
            f"config_hash differs: {meta_a.config_hash[:16]}... vs "
            f"{meta_b.config_hash[:16]}..."
        )

    # Platform
    if meta_a.platform != meta_b.platform:
        differences.append(
            f"platform differs: {meta_a.platform} vs {meta_b.platform}"
        )

    # JAX version
    if meta_a.jax_version != meta_b.jax_version:
        differences.append(
            f"jax_version differs: {meta_a.jax_version} vs {meta_b.jax_version}"
        )

    # x64 mode
    if meta_a.jax_x64_enabled != meta_b.jax_x64_enabled:
        differences.append(
            f"jax_x64_enabled differs: {meta_a.jax_x64_enabled} vs "
            f"{meta_b.jax_x64_enabled}"
        )

    # Resolution / nlev
    if meta_a.resolution != meta_b.resolution:
        differences.append(
            f"resolution differs: {meta_a.resolution} vs {meta_b.resolution}"
        )
    if meta_a.nlev != meta_b.nlev:
        differences.append(
            f"nlev differs: {meta_a.nlev} vs {meta_b.nlev}"
        )

    identical = len(differences) == 0

    return ReproducibilityReport(
        identical=identical,
        differences=differences,
        state_digest_match=state_match,
        config_match=config_match,
    )


def _load_meta(checkpoint_path) -> RestartMetadata:
    """Load the companion metadata for a checkpoint."""
    meta_file = _meta_path(Path(checkpoint_path))
    if not meta_file.exists():
        raise FileNotFoundError(
            f"No companion metadata file found at {meta_file}. "
            "Cannot perform reproducibility comparison."
        )
    with open(meta_file) as f:
        raw = json.load(f)
    return RestartMetadata(**raw)
