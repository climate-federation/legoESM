"""In-process driver for Veros reference runs (team-ocean/veros).

The runner is the bridge that lets the ocean fidelity layer compare legoESM
outputs against a peer JAX primitive-equation ocean model. It

* lazily imports ``veros`` so ``import legoesm.ocean.fidelity`` stays cheap
  and so missing-Veros installs fail with an actionable message;
* dispatches a case name to one of the per-case factories under
  :mod:`legoesm.ocean.fidelity.veros_configs`;
* caches the result NetCDF/pickle under
  ``$LEGOESM_OCEAN_FIDELITY_CACHE/veros/<case>/<key>/`` keyed by the
  inputs (case, resolution, runlen, ``veros.__version__``);
* returns a :class:`VerosResult` ready to feed
  :mod:`legoesm.ocean.fidelity.regrid_veros` and ``compare_field``.

The runner does NOT regrid; that is the job of ``regrid_veros``. It also
does NOT issue a network download for asset-backed setups (e.g.
``global_overturning``) — first-run downloads happen inside Veros itself
and are gated by the user via the per-case factory's ``download_assets``
flag.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from legoesm.ocean.fidelity import cache as _cache

VerosCase = Literal[
    "acc_channel",
    "global_overturning",
    "lock_exchange",
    "overflow",
    "eady_uniform",
    "dino",
]

# v2: the cache key now includes the requested ``capture_vars`` set. v1 caches
# omitted it, so a snapshot captured with one variable set could be silently
# reused for a request needing MORE variables -> a capture-incomplete result
# (degenerate report). Bumping invalidates those v1 caches.
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class VerosResult:
    """Result bundle returned by :func:`run_veros`.

    ``variables`` is a dict of numpy arrays keyed by Veros variable name
    (``u``, ``v``, ``temp``, ``salt``, ``psi``, etc.). When the runner only
    captures the final state, each array has its native Veros shape
    (e.g. ``(nx, ny, nz, n_tau)``). When multiple snapshots are captured,
    the first axis is time.

    ``grid_metadata`` carries enough information for downstream regridding:
    cell centres / edges (``xt``, ``yt``, ``zt``) and the land mask
    (``maskT``).

    ``provenance`` records the case name, Veros version, run wall time, and
    the cache key — useful to spot stale caches.
    """

    case_name: str
    times_s: np.ndarray
    variables: dict[str, np.ndarray]
    grid_metadata: dict[str, Any]
    provenance: dict[str, Any]


class VerosRunError(RuntimeError):
    """Raised when a Veros run aborts before producing a result."""


# Registry of per-case factories. Populated lazily by
# :mod:`legoesm.ocean.fidelity.veros_configs` on first ``run_veros`` call,
# so importing the runner module does not pull in Veros.
_CASE_FACTORIES: dict[str, Callable[..., Any]] = {}


def _register_case_factories() -> None:
    """Import and register per-case Veros setup factories (lazy)."""
    if _CASE_FACTORIES:
        return
    from legoesm.ocean.fidelity import veros_configs

    for name in veros_configs.AVAILABLE_CASES:
        _CASE_FACTORIES[name] = veros_configs.AVAILABLE_CASES[name]


def _require_veros():
    """Lazy import ``veros``, raising a clean ImportError if absent."""
    try:
        import veros  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "Veros is not installed. Install it via `pip install -e ../Veros` "
            "(or `pip install -e legoesm[ocean-fidelity]` once that extra "
            "lands) before calling run_veros()."
        ) from exc
    import veros as v
    return v


def available_cases() -> tuple[str, ...]:
    """Return the tuple of registered per-case Veros names."""
    _register_case_factories()
    return tuple(sorted(_CASE_FACTORIES))


def _cache_key(case_name: str, *, runlen_s: float, identifier: str,
               capture_vars: tuple[str, ...] | None = None,
               extra: dict[str, Any] | None = None) -> str:
    v = _require_veros()
    # Normalise the capture set (sorted + deduped) so the key is order-
    # independent — {"u","v"} and {"v","u"} share a cache (same content) — but a
    # DIFFERENT set (e.g. adding "eke") yields a different key and forces a
    # recompute rather than silently reusing a capture-incomplete snapshot.
    cv = DEFAULT_CAPTURE_VARS if capture_vars is None else capture_vars
    payload = {
        "schema_version": SCHEMA_VERSION,
        "case_name": case_name,
        "runlen_s": float(runlen_s),
        "identifier": identifier,
        "veros_version": str(v.__version__),
        "capture_vars": sorted(set(cv)),
        "extra": dict(extra) if extra else {},
    }
    return _cache.hash_key(payload)


def _result_path_for(key: str, case_name: str) -> Path:
    return _cache.sub("veros") / case_name / key


def _load_cached(case_dir: Path) -> VerosResult | None:
    pickle_path = case_dir / "result.pkl"
    if not pickle_path.exists():
        return None
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def _save_result(case_dir: Path, key: str, result: VerosResult) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    with open(case_dir / "result.pkl", "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    _cache.write_json(case_dir / "key.json", {
        "case_name": result.case_name,
        "key": key,
        "provenance": result.provenance,
    })


def _extract_state(setup, *, capture_vars: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Pull a known set of variables out of a finished Veros setup state."""
    out: dict[str, np.ndarray] = {}
    variables = setup.state.variables
    for name in capture_vars:
        try:
            value = getattr(variables, name, None)
        except RuntimeError:
            # Veros RAISES RuntimeError when accessing an INACTIVE variable
            # (e.g. ``kappa_gm`` when that EKE diagnostic is off) rather than
            # returning None, so a plain getattr-default does not shield it.
            # Skip it gracefully -- otherwise requesting one inactive capture
            # var aborts the whole (already-integrated, expensive) run.
            continue
        if value is None:
            continue
        out[name] = np.asarray(value)
    return out


def _extract_grid_metadata(setup) -> dict[str, Any]:
    out: dict[str, Any] = {}
    variables = setup.state.variables
    for name in ("xt", "xu", "yt", "yu", "zt", "zw", "dxt", "dyt", "dzt",
                 "maskT", "maskU", "maskV"):
        value = getattr(variables, name, None)
        if value is not None:
            out[name] = np.asarray(value)
    settings = setup.state.settings
    for name in ("nx", "ny", "nz", "x_origin", "y_origin", "coord_degree",
                 "enable_cyclic_x", "dt_mom", "dt_tracer"):
        value = getattr(settings, name, None)
        if value is not None:
            out[name] = value
    return out


DEFAULT_CAPTURE_VARS: tuple[str, ...] = (
    "u", "v", "w", "temp", "salt", "rho", "psi", "surface_taux", "surface_tauy",
)


@contextlib.contextmanager
def _veros_io_dir(case_dir: Path, *, force_overwrite: bool):
    """Chdir into ``case_dir`` so Veros writes its diagnostic NetCDFs there.

    Veros writes output files to ``cwd`` by default. Without this context
    manager, repeated runs collide (``OSError: ... exists``) and outputs
    pollute the working tree. ``force_overwrite=True`` is also set so a
    cache-hit re-run does not fail on its own previous NetCDFs — and is
    pushed via the ``VEROS_FORCE_OVERWRITE`` env var because
    ``veros.runtime_settings`` is locked once any veros core module is
    imported.
    """
    orig_cwd = os.getcwd()
    orig_env = os.environ.get("VEROS_FORCE_OVERWRITE")
    case_dir.mkdir(parents=True, exist_ok=True)
    if force_overwrite:
        # Veros refuses to overwrite existing diagnostic NetCDFs / HDF5
        # restarts and the ``VEROS_FORCE_OVERWRITE`` env var is only read
        # during initial veros-core import, which has already happened by
        # the time we get here. Pre-clear the files instead.
        for stale in case_dir.glob("*.nc"):
            stale.unlink()
        for stale in case_dir.glob("*.h5"):
            stale.unlink()
    try:
        os.chdir(case_dir)
        if force_overwrite:
            os.environ["VEROS_FORCE_OVERWRITE"] = "1"
        yield
    finally:
        os.chdir(orig_cwd)
        if orig_env is None:
            os.environ.pop("VEROS_FORCE_OVERWRITE", None)
        else:
            os.environ["VEROS_FORCE_OVERWRITE"] = orig_env


def run_veros(
    case_name: str,
    *,
    runlen_s: float | None = None,
    identifier: str | None = None,
    force_recompute: bool = False,
    cache_dir: Path | None = None,
    capture_vars: tuple[str, ...] = DEFAULT_CAPTURE_VARS,
    factory_kwargs: dict[str, Any] | None = None,
) -> VerosResult:
    """Run a registered Veros case, returning a cached :class:`VerosResult`.

    Parameters
    ----------
    case_name : str
        One of :func:`available_cases`.
    runlen_s : float, optional
        Override the per-case factory's default integration length (seconds).
    identifier : str, optional
        Veros-internal identifier string; defaults to ``case_name``.
    force_recompute : bool, default False
        Ignore any cached result and re-run.
    cache_dir : Path, optional
        Override the global ocean-fidelity cache root for this call.
    capture_vars : tuple of str, default :data:`DEFAULT_CAPTURE_VARS`
        Names of ``state.variables`` attributes to copy into the result.
    factory_kwargs : dict, optional
        Forwarded to the per-case factory function.
    """
    _register_case_factories()
    if case_name not in _CASE_FACTORIES:
        raise ValueError(
            f"Unknown Veros case {case_name!r}; available: "
            f"{available_cases()}"
        )
    _require_veros()

    factory = _CASE_FACTORIES[case_name]
    ident = identifier or case_name
    kwargs = dict(factory_kwargs or {})

    # Pre-resolve runlen from the factory's stashed target so the cache key
    # can be computed before we touch the disk (Veros opens its output
    # NetCDFs during ``.setup()``, so we have to chdir into the cache
    # directory BEFORE then). Factories store the desired runlen on
    # ``setup._legoesm_target_runlen_s`` if they wrap a built-in setup.
    setup = factory(**kwargs)
    if runlen_s is None:
        runlen_s = float(
            getattr(setup, "_legoesm_target_runlen_s", 86400.0)
        )

    key = _cache_key(case_name, runlen_s=runlen_s, identifier=ident,
                     capture_vars=capture_vars, extra=kwargs)
    case_dir = (cache_dir / "veros" / case_name / key
                if cache_dir is not None
                else _result_path_for(key, case_name))

    if not force_recompute:
        cached = _load_cached(case_dir)
        if cached is not None:
            return cached

    case_dir.mkdir(parents=True, exist_ok=True)
    wall_t0 = time.time()
    try:
        with _veros_io_dir(case_dir, force_overwrite=True):
            setup.setup()
            # Veros locks ``state.settings`` after ``.setup()``; unlock to
            # override runlen + identifier before run().
            with setup.state.settings.unlock():
                setup.state.settings.runlen = float(runlen_s)
                setup.state.settings.identifier = ident
            setup.run()
    except Exception as exc:
        raise VerosRunError(
            f"Veros run failed for case {case_name!r} (key {key}): {exc}"
        ) from exc
    wall_s = time.time() - wall_t0

    import veros as v
    captured = _extract_state(setup, capture_vars=capture_vars)
    # Requested vars absent from the finished state (inactive in this Veros
    # config, e.g. ``kappa_gm`` when EKE is off). Recorded so a degenerate
    # capture surfaces in provenance instead of producing a silently empty
    # report downstream.
    missing_capture_vars = [n for n in capture_vars if n not in captured]
    result = VerosResult(
        case_name=case_name,
        times_s=np.asarray([runlen_s]),  # final snapshot only in v1
        variables=captured,
        grid_metadata=_extract_grid_metadata(setup),
        provenance={
            "veros_version": str(v.__version__),
            "runlen_s": float(runlen_s),
            "identifier": ident,
            "key": key,
            "wall_seconds": wall_s,
            "schema_version": SCHEMA_VERSION,
            "factory_kwargs": dict(kwargs),
            "capture_vars": list(capture_vars),
            "missing_capture_vars": missing_capture_vars,
        },
    )
    _save_result(case_dir, key, result)
    return result
