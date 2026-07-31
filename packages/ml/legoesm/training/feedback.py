"""Assemble and apply the LES-informed feedback to the next AMIP/CMIP config.

Stage 7 of ``docs/COMPARE_REANALYSIS.md`` (the feedback **application**): chain
the worst-column diagnoses into a spatially-varying parameter field and splice
it into a target scheme ``*Config`` as a per-column (traced-in-loss) leaf.

* :func:`build_parameter_field` — one hardened entry point over the two
  generalization strategies in :mod:`legoesm.training.parameter_field`
  (``"static"`` lat/lon scatter vs ``"environment"`` kernel regression);
  ``raise``\\ s on an unknown strategy (no silent default — CLAUDE.md dispatch
  hardening).
* :func:`apply_column_parameter_field` — flatten the grid-shaped field to the
  per-column ``(ncol,)`` layout the physics expects, validate its length, and
  splice it into the scheme config via
  :func:`legoesm.core.param_overrides.apply_param_overrides` (which raises on
  an unknown field).  Applied **inside the loss** so the substituted leaf is
  traced; production keeps the static scalar default (SegmentForcing doctrine).

The diagnosed coefficient itself comes from
:mod:`legoesm.atmosphere.dynamics.les.les_closure_diagnosis`; this module is the
grid-agnostic glue that turns it into the config the GCM runs with.

NOTE on the consuming scheme: promoting a *specific* production coefficient to a
per-column field also requires the scheme body to broadcast a ``(ncol,)`` array
over its column dimension (and a ``shape``-keyed ``__param_spec__`` entry).  That
per-scheme physics change is validated separately; this module provides the
mechanism and its shape/dispatch guards.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
from legoesm.core.param_overrides import apply_param_overrides
from legoesm.training.parameter_field import (
    environment_kernel_field,
    scatter_column_field,
)

_STRATEGIES = ("static", "environment")


def build_parameter_field(
    strategy: str,
    *,
    grid_shape: tuple[int, ...],
    background: float = 0.0,
    # static-strategy inputs
    flat_indices: jax.Array | None = None,
    values: jax.Array | None = None,
    valid: jax.Array | None = None,
    # environment-strategy inputs
    grid_env: jax.Array | None = None,
    sample_env: jax.Array | None = None,
    sample_values: jax.Array | None = None,
    length_scales: jax.Array | None = None,
) -> jax.Array:
    """Assemble the grid-shaped feedback field via the chosen strategy.

    ``strategy="static"`` → :func:`scatter_column_field` (needs ``flat_indices``
    + ``values``); ``strategy="environment"`` → :func:`environment_kernel_field`
    (needs ``grid_env`` + ``sample_env`` + ``sample_values`` + ``length_scales``)
    reshaped to ``grid_shape``.  An unknown strategy or missing required input
    raises ``ValueError`` (dispatch hardening — never a silent no-op).
    """
    if strategy == "static":
        if flat_indices is None or values is None:
            raise ValueError(
                "strategy='static' requires flat_indices and values."
            )
        env_only = [
            n for n, v in (
                ("grid_env", grid_env), ("sample_env", sample_env),
                ("sample_values", sample_values), ("length_scales", length_scales),
            ) if v is not None
        ]
        if env_only:
            raise ValueError(
                f"strategy='static' got environment-only input(s) {env_only}; "
                f"these are silently ignored — pass them only with "
                f"strategy='environment'."
            )
        return scatter_column_field(
            grid_shape, flat_indices, values,
            background=background, valid=valid,
        )
    if strategy == "environment":
        missing = [
            n for n, v in (
                ("grid_env", grid_env), ("sample_env", sample_env),
                ("sample_values", sample_values), ("length_scales", length_scales),
            ) if v is None
        ]
        if missing:
            raise ValueError(
                f"strategy='environment' requires {missing}."
            )
        static_only = [
            n for n, v in (("flat_indices", flat_indices), ("values", values))
            if v is not None
        ]
        if static_only:
            raise ValueError(
                f"strategy='environment' got static-only input(s) {static_only};"
                f" these are silently ignored — pass them only with "
                f"strategy='static'."
            )
        flat = environment_kernel_field(
            grid_env, sample_env, sample_values,
            length_scales=length_scales, background=background, valid=valid,
        )
        ncol = 1
        for d in grid_shape:
            ncol *= int(d)
        if flat.ndim != 1 or flat.shape[0] != ncol:
            raise ValueError(
                f"environment field shape {flat.shape} is not a 1-D length-"
                f"{ncol} column vector (prod(grid_shape)); grid_env rows must "
                f"match the grid columns."
            )
        return flat.reshape(grid_shape)
    raise ValueError(
        f"Unknown feedback strategy {strategy!r}; choose from {_STRATEGIES}."
    )


def _find_param_spec(config_obj) -> dict | None:
    """Return the config's ``__param_spec__`` dict (class attr or its module)."""
    spec = getattr(type(config_obj), "__param_spec__", None)
    if isinstance(spec, dict):
        return spec
    mod = sys.modules.get(type(config_obj).__module__)
    spec = getattr(mod, "__param_spec__", None) if mod is not None else None
    return spec if isinstance(spec, dict) else None


def _field_spec_entry(spec: dict, config_obj, field_name: str) -> dict | None:
    """Return the per-field spec dict, handling both spec layouts.

    Canonical (module-level) layout is nested by class:
    ``{ClassName: {"scheme_key": ..., "params": {field: {...}}}}``.  A flat
    ``{field: {...}}`` layout (e.g. a class-attached spec) is also accepted.
    """
    cls_entry = spec.get(type(config_obj).__name__)
    if isinstance(cls_entry, dict) and isinstance(cls_entry.get("params"), dict):
        entry = cls_entry["params"].get(field_name)
    else:
        entry = spec.get(field_name)
    return entry if isinstance(entry, dict) else None


def param_field_bounds(config_obj, field_name: str) -> tuple[float, float] | None:
    """The ``(lo, hi)`` physical bounds for ``field_name`` from the config's
    ``__param_spec__``, or ``None`` if the field/spec/bounds are absent.

    Used to clamp an LES-diagnosed feedback coefficient to its registered
    calibratable range (the param-hygiene bounds, CLAUDE.md) so a degenerate
    diagnosis cannot inject an unphysical / destabilizing value.  Only scalar
    ``(lo, hi)`` bounds are returned; a tuple-valued (per-element) bound is
    treated as absent here (the column field is a single coefficient broadcast
    over the vertical).
    """
    spec = _find_param_spec(config_obj)
    if spec is None:
        return None
    entry = _field_spec_entry(spec, config_obj, field_name)
    if not (isinstance(entry, dict) and "bounds" in entry):
        return None
    lo, hi = entry["bounds"]
    try:
        return (float(lo), float(hi))
    except (TypeError, ValueError):
        # TODO: a per-element (tuple) bound on a future array-shaped promotable
        # param lands here and silently skips clamping; extend to a per-element
        # clip when such a param is registered (none today — all scalar bounds).
        return None


def _check_field_promoted(config_obj, field_name, promoted_fields) -> None:
    """Authorize ``field_name`` as a column-promoted parameter, else raise.

    Overwriting a *scalar* parameter with a per-column field would silently
    change a scheme's contract (``apply_param_overrides`` only rejects unknown
    *names*, not wrong *shapes*).  Authorization is either an explicit
    ``promoted_fields`` allowlist or a non-``None`` ``shape`` key in the config's
    ``__param_spec__`` (the canonical column-promotion declaration).
    """
    if promoted_fields is not None:
        if field_name not in promoted_fields:
            raise ValueError(
                f"{field_name!r} is not in promoted_fields "
                f"{sorted(promoted_fields)}; refusing to overwrite an "
                f"undeclared (scalar) parameter with a per-column field."
            )
        return
    spec = _find_param_spec(config_obj)
    if spec is None:
        raise ValueError(
            f"cannot verify {field_name!r} is column-promoted: "
            f"{type(config_obj).__name__} exposes no __param_spec__ and no "
            f"promoted_fields was passed. Pass promoted_fields={{...}} to "
            f"authorize the per-column override."
        )
    entry = _field_spec_entry(spec, config_obj, field_name)
    if not (isinstance(entry, dict) and entry.get("shape") is not None):
        raise ValueError(
            f"{field_name!r} is not declared column-promoted (a non-None "
            f"'shape' in __param_spec__); refusing to overwrite a scalar "
            f"parameter with a per-column field."
        )


def apply_column_parameter_field(
    config_obj,
    field_name: str,
    field: jax.Array,
    *,
    expected_ncol: int | None = None,
    promoted_fields: set[str] | frozenset[str] | None = None,
) -> object:
    """Splice a per-column parameter ``field`` into a scheme ``*Config``.

    ``field`` is grid-shaped or already ``(ncol,)``; it is flattened to the
    per-column vector the physics consumes.  When ``expected_ncol`` is given the
    length is validated (a mismatched field fails loudly rather than mis-applying
    to the wrong grid).  ``field_name`` MUST be authorized as column-promoted —
    via an explicit ``promoted_fields`` allowlist or a ``shape``-keyed
    ``__param_spec__`` entry — so a per-column field can never silently overwrite
    a scalar parameter.  Splicing goes through :func:`apply_param_overrides`
    (which also raises on an unknown field name).  Call this inside the loss to
    keep the substituted leaf traced; production leaves the scalar default.
    """
    arr = jnp.asarray(field)
    if arr.ndim == 0:
        raise ValueError(
            "apply_column_parameter_field: field is a scalar/0-d; pass a "
            "per-column array (grid-shaped or (ncol,)). A scalar belongs in the "
            "config default, not the feedback field."
        )
    flat = arr.reshape(-1)
    if expected_ncol is not None and int(flat.shape[0]) != int(expected_ncol):
        raise ValueError(
            f"parameter field length {flat.shape[0]} != expected_ncol "
            f"{expected_ncol}."
        )
    _check_field_promoted(config_obj, field_name, promoted_fields)
    return apply_param_overrides(config_obj, {field_name: flat})
