"""Collect spec-declared tunable physics parameters into a trainable module.

This is the machine-readable tunable/fixed split (user requirement) made usable
for the ML training loop. Each physics scheme ``*Config`` declares a
``__param_spec__`` (validated by ``tests/test_param_specs.py``) recording, per
field: bounds, a ``tunable_tier`` (the conservative<->broad continuum), a
constraint transform, and an optional array ``shape`` dimension key. This module
walks the registered spec modules and builds a :class:`TrainablePhysicsParams`
containing exactly the parameters selected by a tier threshold (and/or explicit
include/exclude), with **per-parameterization sizes** resolved from the active
config (e.g. a per-PFT field becomes an ``(n_pft,)`` array). Trained values are
spliced back into the owning ``*Config`` NamedTuples via
:func:`legoesm.core.param_overrides.apply_param_overrides` inside the loss, so
the leaves are *traced* (the SegmentForcing doctrine) while production keeps
static Python-float leaves.

Design notes:
  * The registry is an explicit list of spec-carrying modules (``SPEC_MODULES``),
    mirroring ``driver/kernel_registry``. A drift test
    (``tests/unit/test_param_collector.py``) asserts it matches the set the
    ``__param_spec__`` gate sees, so a spec added without registration (or vice
    versa) goes red.
  * Default *values* come from the live NamedTuple ``_field_defaults`` — the
    single source of truth (the spec carries no defaults).
  * Unknown scheme_key / dim / field raise ``ValueError`` (never a silent skip).
"""

from __future__ import annotations

import importlib
from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.training.trainable_params import (
    ParamConstraint,
    TrainablePhysicsParams,
    range_to_sigmoid_array,
)

# Dotted module paths that carry a validated ``__param_spec__``. Append as
# scheme configs are migrated; the drift test keeps this in sync with the gate.
SPEC_MODULES: tuple[str, ...] = (
    "legoesm.land.snow_bands",
    "legoesm.land.soil_thermal",
    "legoesm.ocean.physics.shortwave_penetration",
    "legoesm.ocean.physics.bottom_drag.config",
    "legoesm.ice.config",
    "legoesm.coupler.config",
    "legoesm.coupler.coupled_latlon_band",
    "legoesm.coupler.lake.config",
    "legoesm.land.canopy.config",
    "legoesm.land.canopy.interception",
    "legoesm.land.canopy.sif",
    "legoesm.land.carbon.config",
    "legoesm.land.stomata",
    "legoesm.land.config",
    "legoesm.land.global_surface_data",
    "legoesm.land.land_use_change",
    "legoesm.land.pedotransfer",
    "legoesm.land.richards",
    "legoesm.land.soil_albedo",
    "legoesm.land.soil_grid",
    "legoesm.land.soil_hydraulics",
    "legoesm.land.surface_scheme.patch_mosaic",
    "legoesm.land.topmodel_runoff",
    "legoesm.ocean.physics.convection.config",
    "legoesm.ocean.physics.ice_shelf",
    "legoesm.ocean.physics.lateral_mixing.backscatter",
    "legoesm.ocean.physics.lateral_mixing.config",
    "legoesm.ocean.physics.lateral_mixing.eke",
    "legoesm.ocean.physics.lateral_mixing.mle",
    "legoesm.ocean.physics.surface_forcing.config",
    "legoesm.ocean.physics.tidal_forcing",
    "legoesm.ocean.physics.vertical_mixing.config",
    "legoesm.ocean.physics.vertical_mixing.double_diffusion",
    "legoesm.ocean.physics.vertical_mixing.internal_wave_mixing",
    "legoesm.ocean.physics.vertical_mixing.tidal",
    # --- atmosphere physics (Phase 4 spec migration) ---
    "legoesm.atmosphere.physics.gravity_wave_drag.config",
    "legoesm.atmosphere.physics.clouds.config",
    "legoesm.atmosphere.physics.convection.config",
    "legoesm.atmosphere.physics.microphysics.config",
    "legoesm.atmosphere.physics.microphysics.sdm.config",
    "legoesm.atmosphere.physics.microphysics.fast_sbm.config",
    "legoesm.atmosphere.physics.microphysics.aerosol_activation",
    "legoesm.atmosphere.physics.microphysics.arg_activation",
    "legoesm.atmosphere.physics.microphysics.prognostic_aerosol",
    "legoesm.atmosphere.physics.turbulence.config",
    "legoesm.atmosphere.physics.turbulence.clubb",
    "legoesm.atmosphere.physics.turbulence.pbl_height",
    "legoesm.atmosphere.physics.radiation.config",
    "legoesm.atmosphere.physics.ml_parameterization",
)

# Tier thresholds for the conservative<->broad continuum. ``build_trainable_params``
# includes every parameter with ``1 <= tunable_tier <= level``.
_TIER_LEVELS: dict[str, int] = {"core": 1, "extended": 2, "aggressive": 3}


class ParamMeta(NamedTuple):
    """Resolved metadata for one spec-declared parameter."""
    scheme_key: str
    qualified_name: str          # f"{scheme_key}.{field}"
    field: str                   # NamedTuple field name
    module: str                  # dotted module path
    config_class: str
    default: float
    bounds: tuple                # (lo, hi) scalar, or (lo_tuple, hi_tuple)
    tunable_tier: int
    transform: str
    units: str
    category: str
    reference: str
    shape_key: str | None        # dimension key for array params (else None)
    legacy_name: str | None


def _load_spec(module_path: str):
    mod = importlib.import_module(module_path)
    spec = getattr(mod, "__param_spec__", None)
    if spec is None:
        raise ValueError(f"{module_path} has no __param_spec__ (stale SPEC_MODULES?)")
    return mod, spec


def build_registry(skipped: list[str] | None = None) -> list[ParamMeta]:
    """Read every importable registered spec module into a flat ``ParamMeta`` list.

    Defaults are pulled from the live NamedTuple ``_field_defaults`` so they can
    never drift from the class. A spec module whose **component is not installed**
    (federation members are separately installable) raises ``ModuleNotFoundError``
    on import; such a module is SKIPPED (its path appended to ``skipped`` if
    provided) so an e.g. ocean-only install can still collect ocean parameters
    without ``legoesm-land`` present. The drift test
    (``test_spec_modules_matches_gate_annotated_set``) runs under a full install,
    so a genuinely wrong ``SPEC_MODULES`` path is still caught in CI. Raises
    ``ValueError`` on a malformed registration (missing class, field not on the
    NamedTuple)."""
    out: list[ParamMeta] = []
    seen_keys: dict[str, str] = {}
    for module_path in SPEC_MODULES:
        try:
            mod, spec = _load_spec(module_path)
        except ModuleNotFoundError as exc:
            # Only treat this as a not-installed federation component when the
            # MISSING module is on the path TO the spec module (its own dotted
            # path or an ancestor package). A ModuleNotFoundError naming some
            # OTHER module is a broken transitive import inside an installed spec
            # module (renamed internal dep, packaging error) — that must fail
            # loudly, not silently drop the scheme's parameters.
            missing = exc.name or ""
            on_path = missing and (
                module_path == missing or module_path.startswith(missing + ".")
            )
            if not on_path:
                raise
            if skipped is not None:
                skipped.append(module_path)
            continue
        for cls_name, entry in spec.items():
            cls = getattr(mod, cls_name, None)
            if cls is None:
                raise ValueError(f"{module_path}.{cls_name} not found for its __param_spec__")
            defaults = getattr(cls, "_field_defaults", {})
            scheme_key = entry["scheme_key"]
            if scheme_key in seen_keys:
                raise ValueError(
                    f"duplicate scheme_key {scheme_key!r} in {module_path}:{cls_name} "
                    f"and {seen_keys[scheme_key]}"
                )
            seen_keys[scheme_key] = f"{module_path}:{cls_name}"
            for field, p in entry.get("params", {}).items():
                if field not in defaults:
                    raise ValueError(
                        f"{module_path}.{cls_name}.{field} is in __param_spec__ but is "
                        f"not a defaulted NamedTuple field"
                    )
                out.append(
                    ParamMeta(
                        scheme_key=scheme_key,
                        qualified_name=f"{scheme_key}.{field}",
                        field=field,
                        module=module_path,
                        config_class=cls_name,
                        default=float(defaults[field]),
                        bounds=tuple(p["bounds"]),
                        tunable_tier=int(p["tunable_tier"]),
                        transform=p["transform"],
                        units=p["units"],
                        category=p["category"],
                        reference=p["reference"],
                        shape_key=p.get("shape"),
                        legacy_name=p.get("legacy_name"),
                    )
                )
    return out


def _resolve_shape(meta: ParamMeta, dims: dict[str, int] | None) -> tuple[int, ...]:
    if meta.shape_key is None:
        return ()
    if not dims or meta.shape_key not in dims:
        raise ValueError(
            f"parameter {meta.qualified_name!r} needs dimension {meta.shape_key!r}; "
            f"pass dims={{'{meta.shape_key}': N}} to build_trainable_params"
        )
    n = int(dims[meta.shape_key])
    if n <= 0:
        raise ValueError(f"dimension {meta.shape_key!r} must be > 0, got {n}")
    return (n,)


def _seed_raw(meta: ParamMeta, shape: tuple[int, ...], dtype) -> jax.Array:
    base = jnp.full(shape, meta.default, dtype=dtype) if shape else jnp.asarray(
        meta.default, dtype=dtype
    )
    if meta.transform == "sigmoid":
        lo, hi = meta.bounds
        return range_to_sigmoid_array(base, lo, hi).astype(dtype)
    if meta.transform == "softplus":
        # Exact inverse of the forward softplus (``as_dict`` uses
        # ``jax.nn.softplus``): softplus_inv(y) = y + log1p(-e^{-y}).  The naive
        # ``log(expm1(y))`` overflows in float32 for y >~ 88 (expm1 -> inf ->
        # log -> inf), which corrupts the seed of every large-magnitude positive
        # scale (averaging lengths AL ~ 1e5 m, number concentrations ~ 1e8 m^-3,
        # relaxation timescales tau ~ 1e3-1e5 s).  Seeds are deliberately pinned
        # to float32 (test_seed_raw_dtype_pinned) to dodge the float64->float32
        # scan-carry hazard, so the inverse MUST be float32-overflow-safe.  The
        # log1p(-e^{-y}) form is overflow-free (-> y for large y) and round-trips
        # through ``jax.nn.softplus`` to float32 precision for all y.
        y = jnp.maximum(base, 1e-6)
        return (y + jnp.log1p(-jnp.exp(-y))).astype(dtype)
    return base


def build_trainable_params(
    *,
    active_scheme_keys: set[str] | None = None,
    tier: str | int = "core",
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    dims: dict[str, int] | None = None,
    dtype=None,
) -> TrainablePhysicsParams:
    """Build a :class:`TrainablePhysicsParams` from the registered specs.

    Parameters
    ----------
    active_scheme_keys
        If given, keep only parameters whose ``scheme_key`` is in this set
        (modularity: the trainable set shrinks/grows with the active schemes).
    tier
        ``"core"`` (tier 1), ``"extended"`` (<=2), ``"aggressive"`` (<=3), or an
        int level — the conservative<->broad continuum. A parameter is included
        iff ``1 <= tunable_tier <= level`` (or it is named in ``include``).
    include / exclude
        Qualified names (``"scheme_key.field"``) to force-include (any tier,
        except tier 0 which is fixed) / force-exclude.
    dims
        Map of array dimension keys -> sizes (e.g. ``{"n_pft": 14}``) for
        variable-size parameters.
    dtype
        Raw-value dtype; defaults to the precision policy's compute dtype
        (guards the float64-into-float32-scan-carry hazard).
    """
    level = _TIER_LEVELS.get(tier, tier) if isinstance(tier, str) else int(tier)
    if not isinstance(level, int):
        raise ValueError(f"unknown tier {tier!r}; use 'core'/'extended'/'aggressive' or an int")
    if dtype is None:
        try:
            from legoesm.core.precision import get_policy
            dtype = get_policy().compute
        except Exception:
            dtype = jnp.float32

    skipped: list[str] = []
    registry = build_registry(skipped=skipped)
    by_name = {m.qualified_name: m for m in registry}
    unknown_inc = [n for n in include if n not in by_name]
    unknown_exc = [n for n in exclude if n not in by_name]
    if unknown_inc or unknown_exc:
        raise ValueError(
            f"include/exclude name unknown parameters: include={unknown_inc}, "
            f"exclude={unknown_exc}; known={sorted(by_name)}"
            + (f" (uninstalled spec modules skipped: {skipped})" if skipped else "")
        )
    if active_scheme_keys is not None:
        known_schemes = {m.scheme_key for m in registry}
        unknown_schemes = sorted(set(active_scheme_keys) - known_schemes)
        if unknown_schemes:
            hint = (
                f" — some spec modules were not importable ({skipped}); the "
                f"requested scheme may belong to a component that is not installed"
                if skipped
                else ". A typo/stale key would otherwise yield an empty set"
            )
            raise ValueError(
                f"active_scheme_keys names unknown scheme(s): {unknown_schemes}; "
                f"known={sorted(known_schemes)}{hint}."
            )
    include_set, exclude_set = set(include), set(exclude)

    raw: dict[str, jax.Array] = {}
    constraints: list[ParamConstraint] = []
    for meta in registry:
        if active_scheme_keys is not None and meta.scheme_key not in active_scheme_keys:
            continue
        if meta.qualified_name in exclude_set:
            continue
        selected = (1 <= meta.tunable_tier <= level) or (meta.qualified_name in include_set)
        if not selected or meta.tunable_tier == 0:
            # tier 0 is fixed and never trainable, even via include
            if meta.qualified_name in include_set and meta.tunable_tier == 0:
                raise ValueError(
                    f"{meta.qualified_name!r} is tier 0 (fixed) and cannot be included"
                )
            continue
        shape = _resolve_shape(meta, dims)
        raw[meta.qualified_name] = _seed_raw(meta, shape, dtype)
        lo, hi = meta.bounds
        constraints.append(
            ParamConstraint(
                name=meta.qualified_name,
                min_val=lo,
                max_val=hi,
                transform=meta.transform,
                shape=shape,
                scheme_key=meta.scheme_key,
                field=meta.field,
            )
        )
    return TrainablePhysicsParams(raw_values=raw, constraints=constraints)
