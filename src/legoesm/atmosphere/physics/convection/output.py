"""Convection output container.

ConvectionOutput is the common interface between SBM and DCA backends.
Both schemes produce the same NamedTuple so that downstream integration
code can be backend-agnostic.
"""

from __future__ import annotations

from typing import NamedTuple

import jax


class ConvectionOutput(NamedTuple):
    """Output from a convection scheme (backend-agnostic).

    All tendency fields are at full levels with shape (ncol, nlev).
    Surface fields have shape (ncol,).

    Fields
    ------
    dT_dt : jax.Array
        Temperature tendency [K/s], shape (ncol, nlev).
    dq_v_dt : jax.Array
        Water vapor specific humidity tendency [kg/kg/s], shape (ncol, nlev).
    precipitation : jax.Array
        Surface precipitation rate [kg/m^2/s], shape (ncol,).
    cape : jax.Array
        CAPE diagnostic [J/kg], shape (ncol,).
    convective_mask : jax.Array
        Smooth 0-1 convective indicator, shape (ncol,).
    """
    dT_dt: jax.Array
    dq_v_dt: jax.Array
    precipitation: jax.Array
    cape: jax.Array
    convective_mask: jax.Array
