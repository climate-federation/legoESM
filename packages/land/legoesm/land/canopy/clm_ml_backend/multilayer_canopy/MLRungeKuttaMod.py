"""
JAX translation of MLRungeKuttaMod Fortran module.

Runge-Kutta method for state updates in the multilayer canopy model.
Provides Butcher tableau initialisation and per-step state updates for
explicit Runge-Kutta time integration of canopy and soil variables.

Original Fortran module: MLRungeKuttaMod
Fortran lines 1-240
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from legoesm.land.canopy.clm_ml_backend.clm_src_main.abortutils import endrun  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import spval  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varctl import iulog  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLCanopyFluxesType import mlcanopy_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varctl import nrk, runge_kutta_type  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varpar import isha, isun  # noqa: F401

# ---------------------------------------------------------------------------
# Public: Runge-Kutta state update
# ---------------------------------------------------------------------------


def RungeKuttaUpdate(
    irk: int,
    a: Array,
    b: Array,
    c: Array,
    num_filter: int,
    filter: Array,
    mlcanopy_inst: mlcanopy_type,
    grid: "GridInfo | None" = None,  # noqa: F821
) -> mlcanopy_type:
    """
    Runge-Kutta state update: update states for the next Runge-Kutta step.

    Implements the general p-order explicit Runge-Kutta scheme:

    .. code-block:: none

        k(1) = h * f[t_n, y_n]
        k(j) = h * f[t_n + c(j)*h, y_n + sum_{i<j} a(j,i) * k(i)]
        y_{n+1} = y_n + sum_j b(j) * k(j)

    At each step ``irk < nrk`` the states are set to an intermediate
    value for the next stage evaluation. At ``irk == nrk`` the states
    are set to the final weighted-average update.

    Mirrors Fortran subroutine ``RungeKuttaUpdate`` (lines 27-140).

    Args:
        irk: Current Runge-Kutta step index (1-based, matching Fortran).
        a: Butcher tableau lower-triangular matrix, shape ``(nrk, nrk)``.
            Fortran: ``real(r8), intent(in) :: a(nrk,nrk)``.
        b: Butcher tableau weight vector, shape ``(nrk,)``.
            Fortran: ``real(r8), intent(in) :: b(nrk)``.
        c: Butcher tableau node vector, shape ``(nrk,)`` (unused in
            state update but kept for interface consistency).
            Fortran: ``real(r8), intent(in) :: c(nrk)``.
        num_filter: Number of patches in the filter.
        filter: 1-D array of patch indices to process.
        mlcanopy_inst: Multilayer canopy state container.

    Returns:
        Updated :class:`mlcanopy_type` with state variables and
        accumulated Runge-Kutta increments written back.
    """
    # irk is 1-based (matching Fortran); convert to 0-based where needed
    irk0 = irk - 1

    # Pre-extract Butcher tableau as numpy once (small arrays, nrk×nrk)
    _a = np.asarray(a)
    _b = np.asarray(b)

    # Working copies of the JAX arrays (we'll update via slice assignment)
    dtair_arr = mlcanopy_inst.dtair_profile
    deair_arr = mlcanopy_inst.deair_profile
    dh2ocan_arr = mlcanopy_inst.dh2ocan_profile
    dtleaf_arr = mlcanopy_inst.dtleaf_leaf
    dlwp_arr = mlcanopy_inst.dlwp_leaf
    dtg_arr = mlcanopy_inst.dtg_soil
    tair_arr = mlcanopy_inst.tair_profile
    eair_arr = mlcanopy_inst.eair_profile
    h2ocan_arr = mlcanopy_inst.h2ocan_profile
    tleaf_arr = mlcanopy_inst.tleaf_leaf
    lwp_arr = mlcanopy_inst.lwp_leaf
    tg_arr = mlcanopy_inst.tg_soil

    # ------------------------------------------------------------------
    # Per-patch loop — Fortran lines 95-136
    # ------------------------------------------------------------------
    _diff_mode = grid is not None

    for fp in range(num_filter):
        if _diff_mode:
            p = grid.p
            n = grid.ncan
        else:
            p = int(filter[fp])
            n = int(mlcanopy_inst.ncan_canopy[p])

        # Always use JAX slices — no np.asarray() D→H syncs
        _tair_cur = tair_arr[p, :n]
        _eair_cur = eair_arr[p, :n]
        _h2ocan_cur = h2ocan_arr[p, :n]
        _tleaf_cur = tleaf_arr[p, :n, :]
        _lwp_cur = lwp_arr[p, :n, :]

        _tair_bef = mlcanopy_inst.tair_bef_profile[p, :n]
        _eair_bef = mlcanopy_inst.eair_bef_profile[p, :n]
        _h2ocan_bef = mlcanopy_inst.h2ocan_bef_profile[p, :n]
        _tleaf_bef = mlcanopy_inst.tleaf_bef_leaf[p, :n, :]
        _lwp_bef = mlcanopy_inst.lwp_bef_leaf[p, :n, :]

        _dtair_prev = dtair_arr[p, :n, :]
        _deair_prev = deair_arr[p, :n, :]
        _dh2ocan_prev = dh2ocan_arr[p, :n, :]
        _dtleaf_prev = dtleaf_arr[p, :n, :, :]
        _dlwp_prev = dlwp_arr[p, :n, :, :]

        _tg_cur = tg_arr[p]
        _tg_bef_val = mlcanopy_inst.tg_bef_soil[p]
        _dtg_prev = dtg_arr[p]

        # ------------------------------------------------------------------
        # Current step's deltas — pure numpy
        # Fortran lines 99-105: dtX[irk0] = X_cur - X_bef
        # ------------------------------------------------------------------
        _dtair_new = _tair_cur - _tair_bef  # (n,)
        _deair_new = _eair_cur - _eair_bef
        _dh2ocan_new = _h2ocan_cur - _h2ocan_bef
        _dtleaf_new = _tleaf_cur - _tleaf_bef  # (n, nleaf+1)
        _dlwp_new = _lwp_cur - _lwp_bef
        _dtg_new = _tg_cur - _tg_bef_val

        # ------------------------------------------------------------------
        # Compute updated state — pure numpy
        # The j0 loop (range(irk)) uses dtX[j0] for j0=0..irk-1=irk0.
        # j0=0..irk0-1 come from previous calls (_dtX_prev[:, :irk0]).
        # j0=irk0 is the current step's delta (_dtX_new), just computed above.
        # ------------------------------------------------------------------
        if irk < nrk:
            # Coefficients: a[irk0+1, 0..irk0]
            _a_prev = _a[irk0 + 1, :irk0]  # (irk0,) — previous steps
            _a_curr = _a[irk0 + 1, irk0]  # scalar  — current step

            _tair_new = (
                _tair_bef + (_dtair_prev[:, :irk0] * _a_prev).sum(axis=1) + _a_curr * _dtair_new
            )
            _eair_new = (
                _eair_bef + (_deair_prev[:, :irk0] * _a_prev).sum(axis=1) + _a_curr * _deair_new
            )
            _h2ocan_new = (
                _h2ocan_bef
                + (_dh2ocan_prev[:, :irk0] * _a_prev).sum(axis=1)
                + _a_curr * _dh2ocan_new
            )
            # tleaf/lwp: shape (n, nleaf+1); dt arrays shape (n, nleaf+1, nrk)
            _tleaf_new = (
                _tleaf_bef
                + (_dtleaf_prev[:, :, :irk0] * _a_prev).sum(axis=2)
                + _a_curr * _dtleaf_new
            )
            _lwp_new = (
                _lwp_bef + (_dlwp_prev[:, :, :irk0] * _a_prev).sum(axis=2) + _a_curr * _dlwp_new
            )
            _tg_new = _tg_bef_val + (_dtg_prev[:irk0] * _a_prev).sum() + _a_curr * _dtg_new

        elif irk == nrk:
            # Final weighted update: j0 = 0..nrk-1
            _b_prev = _b[:irk0]  # (irk0,) — previous steps
            _b_curr = _b[irk0]  # scalar  — current step

            _tair_new = (
                _tair_bef + (_dtair_prev[:, :irk0] * _b_prev).sum(axis=1) + _b_curr * _dtair_new
            )
            _eair_new = (
                _eair_bef + (_deair_prev[:, :irk0] * _b_prev).sum(axis=1) + _b_curr * _deair_new
            )
            _h2ocan_new = (
                _h2ocan_bef
                + (_dh2ocan_prev[:, :irk0] * _b_prev).sum(axis=1)
                + _b_curr * _dh2ocan_new
            )
            _tleaf_new = (
                _tleaf_bef
                + (_dtleaf_prev[:, :, :irk0] * _b_prev).sum(axis=2)
                + _b_curr * _dtleaf_new
            )
            _lwp_new = (
                _lwp_bef + (_dlwp_prev[:, :, :irk0] * _b_prev).sum(axis=2) + _b_curr * _dlwp_new
            )
            _tg_new = _tg_bef_val + (_dtg_prev[:irk0] * _b_prev).sum() + _b_curr * _dtg_new
        else:
            # Should not occur; keep state unchanged
            _tair_new = _tair_cur
            _eair_new = _eair_cur
            _h2ocan_new = _h2ocan_cur
            _tleaf_new = _tleaf_cur
            _lwp_new = _lwp_cur
            _tg_new = _tg_cur

        # ------------------------------------------------------------------
        # Write back to JAX arrays — bulk slice updates (few JAX ops total)
        # Data is always JAX so no jnp.array() conversion needed
        # ------------------------------------------------------------------
        dtair_arr = dtair_arr.at[p, :n, irk0].set(_dtair_new)
        deair_arr = deair_arr.at[p, :n, irk0].set(_deair_new)
        dh2ocan_arr = dh2ocan_arr.at[p, :n, irk0].set(_dh2ocan_new)
        dtleaf_arr = dtleaf_arr.at[p, :n, :, irk0].set(_dtleaf_new)
        dlwp_arr = dlwp_arr.at[p, :n, :, irk0].set(_dlwp_new)
        dtg_arr = dtg_arr.at[p, irk0].set(_dtg_new)

        tair_arr = tair_arr.at[p, :n].set(_tair_new)
        eair_arr = eair_arr.at[p, :n].set(_eair_new)
        h2ocan_arr = h2ocan_arr.at[p, :n].set(_h2ocan_new)
        tleaf_arr = tleaf_arr.at[p, :n, :].set(_tleaf_new)
        lwp_arr = lwp_arr.at[p, :n, :].set(_lwp_new)
        tg_arr = tg_arr.at[p].set(_tg_new)

    # ------------------------------------------------------------------
    # Write all updated arrays back into the immutable state container
    # ------------------------------------------------------------------
    mlcanopy_inst = mlcanopy_inst._replace(
        dtair_profile=dtair_arr,
        deair_profile=deair_arr,
        dh2ocan_profile=dh2ocan_arr,
        dtleaf_leaf=dtleaf_arr,
        dlwp_leaf=dlwp_arr,
        dtg_soil=dtg_arr,
        tair_profile=tair_arr,
        eair_profile=eair_arr,
        h2ocan_profile=h2ocan_arr,
        tleaf_leaf=tleaf_arr,
        lwp_leaf=lwp_arr,
        tg_soil=tg_arr,
    )

    return mlcanopy_inst


# ---------------------------------------------------------------------------
# Public: Butcher tableau initialisation
# ---------------------------------------------------------------------------


def RungeKuttaIni() -> Tuple[Array, Array, Array]:
    """
    Initialise the Butcher tableau for the selected Runge-Kutta method.

    The method is chosen by the module-level constant ``runge_kutta_type``
    from ``MLclm_varctl``. Unused tableau entries (upper-triangular part
    of ``a``) are set to ``spval`` matching the Fortran convention.

    Mirrors Fortran subroutine ``RungeKuttaIni`` (lines 142-240).

    References:
        Kutta, W. (1901). Beitrag zur näherungsweisen Integration totaler
        Differentialgleichungen. Z. Math. Phys., 46, 435-453.

        Butcher, J. C. (1996). A history of Runge-Kutta methods. Applied
        Numerical Mathematics, 20, 247-260.

        Ralston, A. (1962). Runge-Kutta methods with minimum error bounds.
        Mathematics of Computation, 16, 431-437.

    Returns:
        Tuple ``(a, b, c)`` where:

        - ``a``: Butcher tableau lower-triangular matrix,
          shape ``(nrk, nrk)``, dtype ``float64``.
          Fortran: ``real(r8), intent(out) :: a(nrk,nrk)``.
        - ``b``: Weight vector, shape ``(nrk,)``, dtype ``float64``.
          Fortran: ``real(r8), intent(out) :: b(nrk)``.
        - ``c``: Node vector, shape ``(nrk,)``, dtype ``float64``.
          Fortran: ``real(r8), intent(out) :: c(nrk)``.
    """
    # Initialise with spval; specific entries overwritten below
    a = jnp.full((nrk, nrk), spval, dtype=jnp.float64)
    b = jnp.zeros(nrk, dtype=jnp.float64)
    c = jnp.zeros(nrk, dtype=jnp.float64)

    if runge_kutta_type == 21:
        # 2nd-order trapezoidal rule (Butcher 1996) — Fortran lines 176-184
        a = a.at[1, 0].set(1.0)

        b = b.at[0].set(1.0 / 2.0)
        b = b.at[1].set(1.0 / 2.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0)

    elif runge_kutta_type == 22:
        # 2nd-order midpoint rule (Butcher 1996) — Fortran lines 185-193
        a = a.at[1, 0].set(1.0 / 2.0)

        b = b.at[0].set(0.0)
        b = b.at[1].set(1.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 2.0)

    elif runge_kutta_type == 23:
        # 2nd-order Ralston (1962) — Fortran lines 194-202
        a = a.at[1, 0].set(2.0 / 3.0)

        b = b.at[0].set(1.0 / 4.0)
        b = b.at[1].set(3.0 / 4.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(2.0 / 3.0)

    elif runge_kutta_type == 31:
        # 3rd-order Heun's method (Butcher 1996) — Fortran lines 203-215
        a = a.at[1, 0].set(1.0 / 3.0)
        a = a.at[2, 0].set(0.0)
        a = a.at[2, 1].set(2.0 / 3.0)

        b = b.at[0].set(1.0 / 4.0)
        b = b.at[1].set(0.0)
        b = b.at[2].set(3.0 / 4.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 3.0)
        c = c.at[2].set(2.0 / 3.0)

    elif runge_kutta_type == 32:
        # 3rd-order Ralston (1962) — Fortran lines 216-228
        a = a.at[1, 0].set(1.0 / 2.0)
        a = a.at[2, 0].set(0.0)
        a = a.at[2, 1].set(3.0 / 4.0)

        b = b.at[0].set(2.0 / 9.0)
        b = b.at[1].set(3.0 / 9.0)
        b = b.at[2].set(4.0 / 9.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 2.0)
        c = c.at[2].set(3.0 / 4.0)

    elif runge_kutta_type == 33:
        # 3rd-order Kutta's method (Kutta 1901) — Fortran lines 229-241
        a = a.at[1, 0].set(1.0 / 2.0)
        a = a.at[2, 0].set(-1.0)
        a = a.at[2, 1].set(2.0)

        b = b.at[0].set(1.0 / 6.0)
        b = b.at[1].set(4.0 / 6.0)
        b = b.at[2].set(1.0 / 6.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 2.0)
        c = c.at[2].set(1.0)

    elif runge_kutta_type == 41:
        # 4th-order classical Kutta's method (Butcher 1996, Kutta 1901)
        # Fortran lines 242-260
        a = a.at[1, 0].set(1.0 / 2.0)
        a = a.at[2, 0].set(0.0)
        a = a.at[2, 1].set(1.0 / 2.0)
        a = a.at[3, 0].set(0.0)
        a = a.at[3, 1].set(0.0)
        a = a.at[3, 2].set(1.0)

        b = b.at[0].set(1.0 / 6.0)
        b = b.at[1].set(2.0 / 6.0)
        b = b.at[2].set(2.0 / 6.0)
        b = b.at[3].set(1.0 / 6.0)

        c = c.at[0].set(0.0)
        c = c.at[1].set(1.0 / 2.0)
        c = c.at[2].set(1.0 / 2.0)
        c = c.at[3].set(1.0)

    return a, b, c
