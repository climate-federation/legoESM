"""Enhanced restart system with reproducibility checks.

Wraps the existing checkpoint I/O in ``legoesm.forcing.amip_config`` with
SHA-256 integrity digests, platform/JAX metadata, and cross-restart
reproducibility comparison.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np

from legoesm.forcing.amip_config import (
    config_to_dict as _amip_config_to_dict,
    load_checkpoint,
    save_checkpoint,
)

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


class ReproducibilityReport(NamedTuple):
    """Result of comparing two restart files."""

    identical: bool
    differences: list           # list[str] — human-readable difference descriptions
    state_digest_match: bool
    config_match: bool


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------

def compute_state_digest(state_arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 of the concatenated raw bytes of *state_arrays*.

    Keys are sorted so that the digest is independent of insertion order.
    """
    h = hashlib.sha256()
    for key in sorted(state_arrays.keys()):
        arr = np.asarray(state_arrays[key])
        h.update(arr.tobytes())
    return h.hexdigest()


def _config_to_dict_any(config) -> dict:
    """Convert any config type to a dict for serialization.

    Accepts ``ExperimentConfig`` (canonical) or ``AMIPExperimentConfig``
    (legacy).  Returns a flat dict in both cases.
    """
    EC, ectd = _get_experiment_config_type()
    if isinstance(config, EC):
        return ectd(config)
    # Legacy AMIPExperimentConfig or anything with _asdict
    return _amip_config_to_dict(config)


def config_to_dict(config) -> dict:
    """Public alias — serialize any config type to a dict.

    Kept for backward compatibility with code that imports
    ``config_to_dict`` from this module.
    """
    return _config_to_dict_any(config)


def compute_config_hash(config) -> str:
    """SHA-256 of the JSON-serialized *config*."""
    text = json.dumps(_config_to_dict_any(config), sort_keys=True)
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


def _state_arrays_from_checkpoint_args(state, q_v, q_c=None, q_r=None,
                                       carry_aux=None) -> dict[str, np.ndarray]:
    """Build a dict of numpy arrays mirroring save_checkpoint layout."""
    arrays: dict[str, np.ndarray] = {
        "T": np.asarray(state.T.data),
        "u": np.asarray(state.u.data),
        "v": np.asarray(state.v.data),
        "p_s": np.asarray(state.p_s.data),
        "phis": np.asarray(state.phis.data),
        "q_v": np.asarray(q_v),
    }
    if q_c is not None:
        arrays["q_c"] = np.asarray(q_c)
    if q_r is not None:
        arrays["q_r"] = np.asarray(q_r)
    if carry_aux:
        for k, v in carry_aux.items():
            arrays[f"carry_{k}"] = np.asarray(v)
    return arrays


def _meta_path(checkpoint_path: Path) -> Path:
    """Return the companion metadata path for a checkpoint file."""
    p = Path(checkpoint_path)
    # Strip .npz (and handle the case where numpy appends .npz automatically)
    stem = p.stem if p.suffix == ".npz" else p.name
    return p.with_name(stem + ".meta.json")


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def _ensure_amip_config(config):
    """Convert ExperimentConfig to AMIPExperimentConfig if needed.

    The npz/zarr checkpoint formats still use the AMIP wire format
    for backward compatibility.  This helper keeps the conversion
    at the serialization boundary.
    """
    EC, _ = _get_experiment_config_type()
    if isinstance(config, EC):
        return config.to_amip_config()
    return config


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
    model_version: str = "0.1.0",
    backend: str = "npz",
) -> None:
    """Save a restart checkpoint together with reproducibility metadata.

    Accepts either ``ExperimentConfig`` (canonical) or
    ``AMIPExperimentConfig`` (legacy).  Conversion to the AMIP wire
    format happens at this serialization boundary.

    Delegates array persistence to :func:`save_checkpoint` (npz) or
    :func:`save_checkpoint_zarr` (zarr) and writes a companion
    ``.meta.json`` alongside the checkpoint.

    Parameters
    ----------
    backend : str
        ``"npz"`` (default) or ``"zarr"``.
    """
    path = Path(path)

    # Convert ExperimentConfig → AMIP wire format at the boundary
    wire_config = _ensure_amip_config(config)

    # 1. Delegate to appropriate backend
    if backend == "zarr":
        from legoesm.io.checkpoint import save_checkpoint_zarr

        save_checkpoint_zarr(
            path,
            state,
            q_v,
            step,
            day,
            wire_config,
            q_c=q_c,
            q_r=q_r,
            diag_accumulators=diag_accumulators,
        )
    else:
        save_checkpoint(
            path,
            state,
            q_v,
            step,
            day,
            wire_config,
            diag_accumulators=diag_accumulators,
            q_c=q_c,
            q_r=q_r,
            carry_aux=carry_aux,
        )

    # 2. Compute integrity hashes
    state_arrays = _state_arrays_from_checkpoint_args(state, q_v, q_c, q_r)
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
        ``(state, q_v, step, day, loaded_config, diag_accumulators, q_c, q_r, metadata)``
        where *metadata* is a :class:`RestartMetadata` (or ``None`` if the
        companion ``.meta.json`` is absent).
    """
    path = Path(path)

    # 1. Delegate to auto-detecting loader (handles both .npz and .zarr)
    from legoesm.io.checkpoint import load_checkpoint_auto

    state, q_v, step, day, loaded_amip_config, diag_accumulators, q_c, q_r, carry_aux = (
        load_checkpoint_auto(path, grid, sigma)
    )

    # Upconvert AMIP wire format → canonical ExperimentConfig
    EC, _ = _get_experiment_config_type()
    loaded_config = EC.from_amip_config(loaded_amip_config)

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

    # State digest verification
    state_arrays = _state_arrays_from_checkpoint_args(state, q_v, q_c, q_r)
    current_digest = compute_state_digest(state_arrays)
    if current_digest != metadata.state_digest:
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
