"""Column-integrated diagnostic quantities.

Provides utilities for computing vertically integrated atmospheric
fields such as column water vapor (precipitable water).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants


def column_mass_integral(field, p_s, dsigma, dp=None):
    """Column mass-weighted integral ``(1/g) * sum_k(field_k * dp_k)``.

    The generic ∫ field dp/g on the sigma column: for a mixing-ratio-like
    ``field`` [X/kg] this is the column burden [X/m²]; for a tendency
    [X/kg/s] it is the column rate [X/m²/s].  ``column_water_vapor`` is the
    ``field = q_v`` specialization — new column integrals MUST call this
    instead of re-deriving the sum (CLAUDE.md shared-utilities rule).

    Parameters
    ----------
    field : jax.Array
        Per-level field [..., nlev].
    p_s : jax.Array
        Surface pressure [...] (same leading dims, without level axis).
    dsigma : array-like
        Layer thickness in sigma coordinates (nlev,).  Used only when ``dp`` is
        None, where the layer mass is taken as ``p_s * dsigma`` — correct ONLY
        for a PURE-sigma column.
    dp : jax.Array, optional
        Layer pressure thickness [Pa], shape broadcastable to ``field``.  Pass
        this on a HYBRID grid, where ``dp = dA*p_ref + dB*p_s`` and the
        ``p_s * dsigma`` form is wrong by ``dA*(p_s - p_ref)`` — tens of hPa per
        layer over high terrain.  ``VerticalCoordProtocol.layer_thickness_dp``
        supplies it for either coordinate.
    """
    # iter-48: promote to fp64 budget accumulator before the
    # column product+sum.  Same fp32-field convention as iter-42..47:
    # the canonical column-integral helper is imported by the
    # driver / model_driver / plotters, so promoting here cleans
    # every downstream diagnostic in one place.  q·p_s·dσ is
    # ~10⁻²·10⁵·10⁻¹ = 10² per cell, summed over nlev (~32) → ~10³
    # column total; fp32 quantum at that magnitude is ~10⁻⁴.
    from legoesm.core.conservation import conservation_accumulator
    _acc = conservation_accumulator()
    if dp is None:
        _dp = p_s.astype(_acc)[..., None] * jnp.asarray(dsigma).astype(_acc)
    else:
        _dp = jnp.asarray(dp).astype(_acc)
    return jnp.sum(
        field.astype(_acc) * _dp, axis=-1,
    ) / jnp.asarray(constants.g, dtype=_acc)


def column_water_vapor(q_v, p_s, dsigma, dp=None):
    """Column-integrated water vapor [kg/m^2].

    CWV = (1/g) * sum_k(q_v_k * dp_k)

    Parameters
    ----------
    q_v : jax.Array
        Specific humidity [..., nlev].
    p_s : jax.Array
        Surface pressure [...] (same leading dims as q_v, without level axis).
    dsigma : array-like
        Layer thickness in sigma coordinates (nlev,); used only when ``dp`` is
        None (pure-sigma layer mass ``p_s * dsigma``).
    dp : jax.Array, optional
        Layer pressure thickness [Pa]; REQUIRED for a correct answer on a
        hybrid grid.  See :func:`column_mass_integral`.

    Returns
    -------
    jax.Array
        Column water vapor [...], same leading shape as p_s.
    """
    return column_mass_integral(q_v, p_s, dsigma, dp=dp)


def column_mass_weighted_mean(field, mass_per_cell, axis: int = -1):
    """FV3_3D iter 605: column mass-weighted mean of a 3D field.

    Faithful port of FV3 ``dyn_core.F90:1313-1326`` (the d_ext
    external-mode-damping column-mean computation):

        out(i,j) = Σ_k mass[k] · field[k] / Σ_k mass[k]

    Used by FV3's d_ext > 0 external (barotropic) mode damping
    branch where ``field = vt`` (divergence-flux) and ``mass =
    ptc[k]`` (column mass weight per level).  Also useful for
    other column diagnostics (e.g., mass-weighted-mean potential
    temperature, mass-weighted divergence).

    Parameters
    ----------
    field : jax.Array, shape (..., nlev)
        Field to average.
    mass_per_cell : jax.Array, shape (..., nlev)
        Mass weight per level (delp, rho·dz, etc.).
    axis : int, default -1
        Vertical axis.

    Returns
    -------
    jax.Array, same shape as field minus the vertical axis.
        Column mass-weighted mean.

    Notes
    -----
    Guards against zero total mass (returns 0 in that case).
    """
    weighted_sum = jnp.sum(field * mass_per_cell, axis=axis)
    total_mass = jnp.sum(mass_per_cell, axis=axis)
    return jnp.where(
        jnp.abs(total_mass) > 1e-30,
        weighted_sum / jnp.where(jnp.abs(total_mass) > 1e-30,
                                  total_mass, 1.0),
        0.0,
    )


def column_d_ext_field(vt, delp, d_ext: float, da_min_c: float):
    """FV3_3D iter 605: compute the d_ext external-mode damping field.

    Faithful port of FV3 ``dyn_core.F90:1310-1326``:

        if d_ext > 0:
            d2_divg = d_ext · da_min_c
            divg2(i,j) = d2_divg · Σ_k ptc·vt / Σ_k ptc

    Returns ``divg2``, a 2D field that gets passed to ``one_grad_p``
    in FV3 to add a column-mean divergence-damping term to the
    pressure-gradient force.

    legoESM doesn't yet wire d_ext into the NH dycore (the PGF
    machinery is complex); this utility makes the computation
    available for users / tests / future plumbing.

    Parameters
    ----------
    vt : jax.Array, shape (..., nlev)
        Divergence-like field (FV3 vt at line 1316).
    delp : jax.Array, shape (..., nlev)
        Pressure thickness per cell (FV3 ptc).
    d_ext : float
        FV3 namelist d_ext parameter (default 0.02).
    da_min_c : float
        Minimum cell area at coarsest grid (FV3 gridstruct%da_min_c).

    Returns
    -------
    divg2 : jax.Array, shape (..., )
        d_ext·da_min_c · column mass-weighted vt.  Returns zeros
        if ``d_ext <= 0``.
    """
    if d_ext <= 0.0:
        return jnp.zeros(delp.shape[:-1])
    d2_divg = d_ext * da_min_c
    column_mean_vt = column_mass_weighted_mean(vt, delp)
    return d2_divg * column_mean_vt
