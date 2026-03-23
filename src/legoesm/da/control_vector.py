"""Control variable transforms for 4D-Var data assimilation.

Maps bijectively between model state NamedTuples (with Field objects) and
flat 1D JAX arrays that the minimizer operates on. All transforms are
differentiable.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.field import Field


# Fields that are never included in the control vector (boundary conditions).
_STATIC_FIELDS = frozenset({
    "phis", "h_s", "H_bathy", "land_mask",
    "H_bathy_hat", "land_mask_grid", "phis_hat",
})


class ControlEntry(NamedTuple):
    """One entry mapping a state field to a control vector slice."""
    field_name: str
    offset: int
    size: int
    shape: tuple[int, ...]
    transform: str  # "identity", "log", "softplus"


class ControlVectorSpec(NamedTuple):
    """Specification mapping state fields to control vector slices."""
    entries: tuple[ControlEntry, ...]
    total_size: int


def _get_field_data(state, name: str) -> jax.Array:
    """Extract raw array from a state field (handles Field and plain arrays)."""
    val = getattr(state, name)
    if isinstance(val, Field):
        return val.data
    return val


def _field_names(state) -> tuple[str, ...]:
    """Get field names from a NamedTuple state."""
    return state._fields


def _is_prognostic(name: str) -> bool:
    """Check if a field name is prognostic (not static)."""
    return name not in _STATIC_FIELDS


def build_control_spec(
    state,
    grid=None,
    fields: tuple[str, ...] | None = None,
    transforms: dict[str, str] | None = None,
) -> ControlVectorSpec:
    """Build control vector specification from a template state.

    Parameters
    ----------
    state : NamedTuple
        Template state (any legoESM state type).
    grid : GridProtocol, optional
        Grid (unused, reserved for future use).
    fields : tuple of str, optional
        Which fields to include. Default: all prognostic non-static fields.
    transforms : dict, optional
        Per-field transforms. Keys are field names, values are
        "identity", "log", or "softplus".

    Returns
    -------
    ControlVectorSpec
    """
    if transforms is None:
        transforms = {}

    all_names = _field_names(state)
    if fields is None:
        fields = tuple(n for n in all_names if _is_prognostic(n))

    entries = []
    offset = 0

    for name in fields:
        if name not in all_names:
            raise ValueError(f"Field '{name}' not in state type {type(state).__name__}")
        val = getattr(state, name)

        # Handle dict fields (tracers)
        if isinstance(val, dict):
            for tname in sorted(val.keys()):
                tval = val[tname]
                arr = tval.data if isinstance(tval, Field) else tval
                shape = arr.shape
                size = int(jnp.prod(jnp.array(shape)))
                tfm = transforms.get(f"{name}.{tname}", "identity")
                entries.append(ControlEntry(
                    field_name=f"{name}.{tname}",
                    offset=offset,
                    size=size,
                    shape=shape,
                    transform=tfm,
                ))
                offset += size
        elif val is None:
            continue
        else:
            arr = val.data if isinstance(val, Field) else val
            shape = arr.shape
            size = int(jnp.prod(jnp.array(shape)))
            tfm = transforms.get(name, "identity")
            entries.append(ControlEntry(
                field_name=name,
                offset=offset,
                size=size,
                shape=shape,
                transform=tfm,
            ))
            offset += size

    return ControlVectorSpec(entries=tuple(entries), total_size=offset)


def _forward_transform(x: jax.Array, transform: str) -> jax.Array:
    """Apply forward transform (state space -> control space)."""
    if transform == "identity":
        return x
    elif transform == "log":
        return jnp.log(x)
    elif transform == "softplus":
        # softplus inverse: log(exp(x) - 1), numerically stable
        return x + jnp.log1p(-jnp.exp(-x))
    else:
        raise ValueError(f"Unknown transform: {transform}")


def _inverse_transform(x: jax.Array, transform: str) -> jax.Array:
    """Apply inverse transform (control space -> state space)."""
    if transform == "identity":
        return x
    elif transform == "log":
        return jnp.exp(x)
    elif transform == "softplus":
        return jnp.logaddexp(x, 0.0)  # log(1 + exp(x)), numerically stable
    else:
        raise ValueError(f"Unknown transform: {transform}")


def state_to_control(state, spec: ControlVectorSpec) -> jax.Array:
    """Flatten state to 1D control vector. Differentiable."""
    parts = []
    for entry in spec.entries:
        name = entry.field_name
        if "." in name:
            # Tracer: "tracers.q_v"
            parent, child = name.split(".", 1)
            container = getattr(state, parent)
            val = container[child]
        else:
            val = getattr(state, name)
        arr = val.data if isinstance(val, Field) else val
        flat = _forward_transform(arr.ravel(), entry.transform)
        parts.append(flat)
    return jnp.concatenate(parts)


def control_to_state(x: jax.Array, spec: ControlVectorSpec, template_state):
    """Unflatten 1D control vector to state NamedTuple. Differentiable.

    Static fields (phis, h_s, H_bathy, land_mask) are copied from
    template_state unchanged.
    """
    replacements = {}
    tracer_replacements = {}

    for entry in spec.entries:
        raw = jax.lax.dynamic_slice(x, (entry.offset,), (entry.size,))
        arr = _inverse_transform(raw.reshape(entry.shape), entry.transform)

        name = entry.field_name
        if "." in name:
            parent, child = name.split(".", 1)
            if parent not in tracer_replacements:
                tracer_replacements[parent] = {}
            # Reconstruct Field with metadata from template
            template_container = getattr(template_state, parent)
            template_val = template_container[child]
            if isinstance(template_val, Field):
                tracer_replacements[parent][child] = template_val.replace(data=arr)
            else:
                tracer_replacements[parent][child] = arr
        else:
            template_val = getattr(template_state, name)
            if isinstance(template_val, Field):
                replacements[name] = template_val.replace(data=arr)
            else:
                replacements[name] = arr

    # Build tracer dicts
    for parent, children in tracer_replacements.items():
        template_container = getattr(template_state, parent)
        new_container = dict(template_container)
        new_container.update(children)
        replacements[parent] = new_container

    return template_state._replace(**replacements)


def control_to_increment(dx: jax.Array, spec: ControlVectorSpec, template_state):
    """Unflatten a control-space increment into a state-shaped increment.

    Returns a state where each field contains the increment values.
    Static fields are set to zero.
    """
    # Build a zero state for the increment
    zero_state = jax.tree.map(lambda leaf: jnp.zeros_like(leaf), template_state)

    replacements = {}
    tracer_replacements = {}

    for entry in spec.entries:
        raw = jax.lax.dynamic_slice(dx, (entry.offset,), (entry.size,))
        arr = raw.reshape(entry.shape)

        name = entry.field_name
        if "." in name:
            parent, child = name.split(".", 1)
            if parent not in tracer_replacements:
                tracer_replacements[parent] = {}
            template_container = getattr(zero_state, parent)
            template_val = template_container[child]
            if isinstance(template_val, Field):
                tracer_replacements[parent][child] = template_val.replace(data=arr)
            else:
                tracer_replacements[parent][child] = arr
        else:
            template_val = getattr(zero_state, name)
            if isinstance(template_val, Field):
                replacements[name] = template_val.replace(data=arr)
            else:
                replacements[name] = arr

    for parent, children in tracer_replacements.items():
        template_container = getattr(zero_state, parent)
        new_container = dict(template_container)
        new_container.update(children)
        replacements[parent] = new_container

    return zero_state._replace(**replacements)
