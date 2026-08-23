"""Format-agnostic primitives for building SCM forcing from gridded files.

These helpers are the numerics shared by every host-side SCM-forcing loader
(DEPHY-SCM, NCAR-SCCM/ARM, ...): vertical interpolation of a source profile
onto the model pressure grid, the hydrostatic ``omega -> w`` conversion, and
the construction of the time-interpolating callables an :class:`SCMForcing`
carries.  They live here (rather than duplicated per format module) so the
conversions are defined ONCE — a format loader supplies only the
variable-name mapping.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics._shared import diagnose_grid_w_from_omega


def interp_profile_to_pressure(
    p_target: np.ndarray,
    p_source: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Interpolate ``values`` from ``p_source`` onto ``p_target`` [Pa].

    Non-finite samples are dropped, the source is sorted ascending in
    pressure and de-duplicated, then linearly interpolated (``np.interp``
    clamps to the source end-points outside its range).  At least two finite
    source samples are required.
    """
    p = np.asarray(p_source, dtype=np.float64).reshape(-1)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if p.size != v.size:
        raise ValueError(
            f"pressure/profile size mismatch: pressure={p.size}, values={v.size}"
        )
    mask = np.isfinite(p) & np.isfinite(v)
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two finite pressure/profile samples are required")
    p = p[mask]
    v = v[mask]
    order = np.argsort(p)
    p = p[order]
    v = v[order]
    p_unique, idx = np.unique(p, return_index=True)
    v_unique = v[idx]
    return np.interp(p_target, p_unique, v_unique)


def omega_to_w(
    omega: np.ndarray,
    T: np.ndarray,
    p: np.ndarray,
    q_v: np.ndarray | None = None,
) -> np.ndarray:
    """Convert pressure velocity ``omega`` [Pa/s] to upward ``w`` [m/s].

    Hydrostatic ``w = -omega / (rho g)`` (``omega`` positive DOWN, ``w`` positive
    UP — subsidence ``omega > 0`` gives ``w < 0``).  This is a thin host-side
    (numpy) wrapper around the canonical device-side reduction
    :func:`legoesm.atmosphere.physics._shared.diagnose_grid_w_from_omega` so the
    ``-omega/(rho g)`` numerics live in ONE place.

    ``q_v`` is water-vapour **specific** humidity [kg/kg] (NOT mixing ratio),
    matching ``diagnose_grid_w_from_omega``/``compute_rho``; convert a mixing
    ratio with ``thermo.mixing_ratio_to_specific_humidity`` before calling.
    ``omega``/``T``/``q_v`` are ``(nt, nz)`` (a ``(nz,)`` ``T``/``q_v`` is
    broadcast over time); ``p`` is the ``(nz,)`` shared full-level pressure.
    Returns ``(nt, nz)``.
    """
    omega2 = np.atleast_2d(np.asarray(omega, dtype=np.float64))
    nt, nz = omega2.shape
    p2 = np.broadcast_to(np.asarray(p, dtype=np.float64).reshape(1, nz), (nt, nz))
    T2 = np.broadcast_to(np.atleast_2d(np.asarray(T, dtype=np.float64)), (nt, nz))
    qv2 = None
    if q_v is not None:
        qv2 = jnp.asarray(
            np.broadcast_to(np.atleast_2d(np.asarray(q_v, dtype=np.float64)), (nt, nz))
        )
    w = diagnose_grid_w_from_omega(jnp.asarray(omega2), jnp.asarray(T2), jnp.asarray(p2), qv2)
    return np.asarray(w, dtype=np.float64)


def forcing_cadence(time: np.ndarray) -> float:
    """Smallest positive forcing time step [s] (fallback 300 s)."""
    time = np.asarray(time, dtype=np.float64)
    if time.size < 2:
        return 300.0
    diffs = np.diff(time)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 300.0
    return float(np.nanmin(diffs))


def profile_time_fn(time: np.ndarray, values: np.ndarray, dtype):
    """Callable ``t -> (nz,)`` linearly interpolating ``values`` (nt, nz) in time."""
    time_np = np.asarray(time, dtype=np.float64)
    values_np = np.asarray(values, dtype=np.float64)

    def fn(t_seconds: float):
        t = float(t_seconds)
        out = np.empty(values_np.shape[1], dtype=np.float64)
        for k in range(values_np.shape[1]):
            out[k] = np.interp(t, time_np, values_np[:, k])
        return jnp.asarray(out, dtype=dtype)

    return fn


def scalar_time_fn(time: np.ndarray, values: np.ndarray):
    """Callable ``t -> scalar`` linearly interpolating ``values`` (nt,) in time."""
    time_np = np.asarray(time, dtype=np.float64)
    values_np = np.asarray(values, dtype=np.float64)

    def fn(t_seconds: float):
        return jnp.asarray(np.interp(float(t_seconds), time_np, values_np))

    return fn


__all__ = [
    "interp_profile_to_pressure",
    "omega_to_w",
    "forcing_cadence",
    "profile_time_fn",
    "scalar_time_fn",
]
