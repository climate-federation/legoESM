"""Field: A coordinate-aware array registered as a JAX pytree.

The Field is the fundamental data container in legoESM. It wraps a JAX array
with metadata (name, dimensions, units, staggering) and is registered as a JAX
pytree so that jit, grad, vmap, and scan all work transparently.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from legoesm.core.precision import get_policy


class Field:
    """A coordinate-aware JAX array.

    Fields carry metadata alongside their numerical data, enabling:
    - Automatic dimension tracking through operations
    - Unit-aware diagnostics and I/O
    - Staggering information for C-grid operators

    As a JAX pytree, Field works seamlessly with all JAX transformations:
    jit, grad, vmap, scan, checkpoint, etc.

    Parameters
    ----------
    data : jax.Array
        The numerical data.
    name : str
        Variable name (e.g., "potential_temperature").
    dims : tuple of str
        Dimension names (e.g., ("face", "x", "y")).
    units : str
        Physical units (e.g., "K", "m/s").
    long_name : str, optional
        Human-readable description.
    staggering : str, optional
        Grid staggering: "cell", "edge", or "vertex". Default "cell".
    """

    __slots__ = ("data", "name", "dims", "units", "long_name", "staggering")

    def __init__(
        self,
        data: jax.Array,
        name: str = "",
        dims: tuple[str, ...] = (),
        units: str = "",
        long_name: str = "",
        staggering: str = "cell",
    ):
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "long_name", long_name)
        object.__setattr__(self, "staggering", staggering)

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("Field is immutable. Use .replace() to create a new Field.")

    # ---- JAX pytree registration ----

    def tree_flatten(self):
        """Flatten for JAX: data is the dynamic leaf, metadata is static.

        All metadata is part of aux_data (static). For tree_map to work
        between two pytrees, their aux_data must match exactly. Ensure that
        tendencies returned by physics modules use state.field.replace(data=...)
        to preserve metadata compatibility.
        """
        children = (self.data,)
        aux_data = (self.name, self.dims, self.units, self.long_name, self.staggering)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct Field from flattened representation."""
        (data,) = children
        name, dims, units, long_name, staggering = aux_data
        return cls(data=data, name=name, dims=dims, units=units,
                   long_name=long_name, staggering=staggering)

    # ---- Convenience methods ----

    def replace(self, **kwargs) -> Field:
        """Create a new Field with some attributes replaced."""
        return Field(
            data=kwargs.get("data", self.data),
            name=kwargs.get("name", self.name),
            dims=kwargs.get("dims", self.dims),
            units=kwargs.get("units", self.units),
            long_name=kwargs.get("long_name", self.long_name),
            staggering=kwargs.get("staggering", self.staggering),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def astype(self, dtype) -> Field:
        """Cast data to a different dtype."""
        return self.replace(data=self.data.astype(dtype))

    # ---- Arithmetic (operate on data, preserve metadata) ----

    def __add__(self, other):
        if isinstance(other, Field):
            return self.replace(data=self.data + other.data)
        return self.replace(data=self.data + other)

    def __radd__(self, other):
        return self.replace(data=other + self.data)

    def __sub__(self, other):
        if isinstance(other, Field):
            return self.replace(data=self.data - other.data)
        return self.replace(data=self.data - other)

    def __rsub__(self, other):
        return self.replace(data=other - self.data)

    def __mul__(self, other):
        if isinstance(other, Field):
            return self.replace(data=self.data * other.data)
        return self.replace(data=self.data * other)

    def __rmul__(self, other):
        return self.replace(data=other * self.data)

    def __truediv__(self, other):
        if isinstance(other, Field):
            return self.replace(data=self.data / other.data)
        return self.replace(data=self.data / other)

    def __rtruediv__(self, other):
        return self.replace(data=other / self.data)

    def __neg__(self):
        return self.replace(data=-self.data)

    def __pow__(self, other):
        return self.replace(data=self.data ** other)

    def __repr__(self) -> str:
        return (
            f"Field(name='{self.name}', shape={self.shape}, "
            f"dims={self.dims}, units='{self.units}', "
            f"staggering='{self.staggering}')"
        )


# Register Field as a JAX pytree
jax.tree_util.register_pytree_node(
    Field,
    lambda f: f.tree_flatten(),
    lambda aux, children: Field.tree_unflatten(aux, children),
)


def zeros_field(
    shape: tuple[int, ...],
    name: str = "",
    dims: tuple[str, ...] = (),
    units: str = "",
    long_name: str = "",
    staggering: str = "cell",
    dtype=None,
) -> Field:
    """Create a Field filled with zeros.

    If *dtype* is ``None``, defaults to the active precision policy's
    storage dtype (``get_policy().storage``).
    """
    if dtype is None:
        dtype = get_policy().storage
    return Field(
        data=jnp.zeros(shape, dtype=dtype),
        name=name, dims=dims, units=units,
        long_name=long_name, staggering=staggering,
    )


def ones_field(
    shape: tuple[int, ...],
    name: str = "",
    dims: tuple[str, ...] = (),
    units: str = "",
    long_name: str = "",
    staggering: str = "cell",
    dtype=None,
) -> Field:
    """Create a Field filled with ones.

    If *dtype* is ``None``, defaults to the active precision policy's
    storage dtype (``get_policy().storage``).
    """
    if dtype is None:
        dtype = get_policy().storage
    return Field(
        data=jnp.ones(shape, dtype=dtype),
        name=name, dims=dims, units=units,
        long_name=long_name, staggering=staggering,
    )
