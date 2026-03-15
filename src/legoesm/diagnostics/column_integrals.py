"""Column-integrated diagnostic quantities.

Provides utilities for computing vertically integrated atmospheric
fields such as column water vapor (precipitable water).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


def column_water_vapor(q_v, p_s, dsigma):
    """Column-integrated water vapor [kg/m^2].

    CWV = (1/g) * sum_k(q_v_k * p_s * dsigma_k)

    Parameters
    ----------
    q_v : jax.Array
        Specific humidity [..., nlev].
    p_s : jax.Array
        Surface pressure [...] (same leading dims as q_v, without level axis).
    dsigma : array-like
        Layer thickness in sigma coordinates (nlev,).

    Returns
    -------
    jax.Array
        Column water vapor [...], same leading shape as p_s.
    """
    return jnp.sum(q_v * p_s[..., None] * dsigma, axis=-1) / constants.g
