"""Reference Potential Energy (RPE) diagnostic.

RPE is the potential energy that would remain if every water parcel
were adiabatically rearranged to minimise PE: the densest parcels
sit at the bottom, the lightest at the top. ``dRPE/dt`` then
quantifies *irreversible* diapycnal mixing -- in an adiabatic ocean
RPE is invariant, so any drift is spurious mixing from the dycore's
tracer advection (Griffies 2015; Petersen et al. 2015).

Algorithm

For every wet cell ``c`` and level ``k`` the diagnostic computes:

* ``rho[c, k]`` -- in-situ density from a configurable EOS.
* ``vol[c, k] = area_c * h[c, k] * mask[c, k]`` -- cell volume,
  zeroed on land.

The non-land parcels are flattened, sorted by density (densest
first), and stacked into a single notional column of horizontal
cross-section ``A_total = sum(area * mask_top)``. Cell ``i`` in the
sorted stack occupies depths ``[z_i^bot, z_i^top]`` with

.. math::
    z_i^\\mathrm{top} &= -\\frac{V^\\mathrm{below}_i}{A_\\mathrm{total}}
    \\\\
    z_i^\\mathrm{bot} &= z_i^\\mathrm{top} - \\frac{V_i}{A_\\mathrm{total}}

so the cell centre is at ``z_i = z_i^top - V_i / (2 A_total)``.
The RPE then evaluates as

.. math::
    \\mathrm{RPE} = g \\sum_i \\rho_i z_i V_i

(positive downward depth convention: ``z_i < 0`` in the column).
Petersen et al. 2015 Fig. 5 reports MPAS-Ocean spurious-mixing
``dRPE/dt`` of 0.1-0.3 mW/m^2 on the Petersen lock-exchange
benchmark; the acceptance bar in legoESM's long-term diagnostics is
``|dRPE/dt| < 0.5 mW/m^2``.
"""

from __future__ import annotations


import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.ocean.eos import make_eos_fn


def pack_sorted_rpe(rho: np.ndarray, vol: np.ndarray, total_area: float,
                    g_val: float | None = None) -> float:
    """Sorted-RPE packing kernel: densest parcel at the BOTTOM.

    Stack the parcels into one notional column of cross-section
    ``total_area``, densest first from the bottom
    (``z = -H``, ``H = sum(vol)/total_area``), and integrate
    ``g * rho * z * vol``.  Mixing (a mean-preserving contraction of the
    density multiset) RAISES this value; that sign is what makes it a
    spurious-mixing metric.

    ONE home for the packing convention (2026-08-10): shared by
    :func:`compute_rpe`, the ocean test matrix's ``_compute_sorted_rpe``
    and ``scripts/validate/lockex_rpe_trace.py``.  ``compute_rpe``'s
    previous inline packing stacked densest at the SURFACE — inverting
    the mixing sign against its own docstring (mixing lowered it).

    Tie order among equal densities is irrelevant to the sum: a block of
    equal-``rho`` parcels occupies a fixed depth range whose mass moment
    is permutation-invariant.
    """
    if g_val is None:
        g_val = float(constants.g)
    H = float(np.sum(vol) / total_area)
    order = np.argsort(-np.asarray(rho), kind="stable")   # densest first
    rho_s = np.asarray(rho)[order]
    vol_s = np.asarray(vol)[order]
    c_prev = np.concatenate([[0.0], np.cumsum(vol_s[:-1])])
    z_i = -H + (c_prev + 0.5 * vol_s) / total_area
    return float(g_val * np.sum(rho_s * z_i * vol_s))


def compute_rpe(state, z_coord, *, grid_type: str, grid,
                eos: str = "wright",
                eos_linear: object = None,
                p_ref: float | None = None,
                g_val: float | None = None) -> float:
    """Compute the Reference Potential Energy of the ocean state [J].

    Parameters
    ----------
    state : OceanState (cubed-sphere / lat-lon / MPAS / spectral)
        Must expose ``T.data``, ``S.data``, ``eta.data``,
        ``H_bathy.data``, ``land_mask.data``.
    z_coord : OceanZStarCoordinate
        Vertical coordinate used to recover per-cell layer thicknesses
        via ``compute_layer_thickness(eta, H_bathy, z_coord)``.
    grid_type : str
        ``"cubed_sphere"``, ``"latlon"``, ``"mpas"`` (or their
        ``_regional`` / ``_channel`` siblings). Determines how the cell
        areas are pulled off the grid object.
    grid : Grid object
        Provides ``grid_area`` (cube), ``area`` (lat-lon), or
        ``areaCell`` (MPAS).
    eos : str
        ``"wright"`` (default) or ``"linear"``.
    eos_linear : LinearEOSConfig or None
        Passed through to ``make_eos_fn``.
    p_ref : float or None
        Reference pressure for the EOS lookup. Wright EOS uses
        in-situ pressure; for the RPE diagnostic a constant
        ``p_ref = rho_0 * g * H_mean / 2`` is acceptable since RPE
        cares about relative density only (the absolute level cancels
        with the sorted-volume integration).
    g_val : float or None
        Gravity; defaults to ``constants.g``.

    Returns
    -------
    float : Reference Potential Energy in joules.
    """
    if g_val is None:
        g_val = float(constants.g)

    T = np.asarray(state.T.data, dtype=np.float64)
    S = np.asarray(state.S.data, dtype=np.float64)
    eta = np.asarray(state.eta.data, dtype=np.float64)
    H_bathy = np.asarray(state.H_bathy.data, dtype=np.float64)
    mask = np.asarray(state.land_mask.data, dtype=np.float64)

    # Cell areas: shape matches state.T spatial leading axes.
    if grid_type in ("mpas", "mpas_regional", "mpas_channel"):
        area = np.asarray(grid.areaCell, dtype=np.float64)
    elif grid_type == "cubed_sphere":
        # ``grid.grid_area`` is the conventional name on the cube;
        # fall back to ``grid.area`` if absent.
        area = np.asarray(
            getattr(grid, "grid_area", getattr(grid, "area", None)),
            dtype=np.float64,
        )
    else:
        # lat-lon C-grid / regional / channel / spectral all expose
        # ``grid.area`` of shape ``(n_lat, n_lon)``.
        area = np.asarray(grid.area, dtype=np.float64)

    # Layer thicknesses (rest dz_ref * jacobian via compute_layer_thickness).
    from legoesm.ocean.vertical import compute_layer_thickness
    h_k = np.asarray(
        compute_layer_thickness(jnp.asarray(eta), jnp.asarray(H_bathy), z_coord),
        dtype=np.float64,
    )

    # Reference pressure for the EOS lookup (column-mean depth proxy).
    if p_ref is None:
        # rho_0 * g * H_mean / 2 -- absolute level cancels in dRPE/dt.
        rho0 = float(constants.rho_ocean)
        H_mean = float(np.maximum(np.mean(H_bathy[mask > 0.5]), 1.0)) if (mask > 0.5).any() else 1.0
        p_ref_val = 0.5 * rho0 * g_val * H_mean
    else:
        p_ref_val = float(p_ref)

    eos_fn = make_eos_fn(eos=eos, eos_linear=eos_linear)
    rho = np.asarray(
        eos_fn(jnp.asarray(T), jnp.asarray(S),
               jnp.full_like(jnp.asarray(T), p_ref_val)),
        dtype=np.float64,
    )

    # Build per-cell volume = area * h * mask (mask broadcast over levels).
    horiz = (area * mask)
    vol = horiz[..., None] * h_k
    # Total horizontal area available for RPE (excludes dry columns).
    A_total = float(horiz.sum())
    if A_total <= 0.0:
        return 0.0

    rho_flat = rho.reshape(-1)
    vol_flat = vol.reshape(-1)
    # Drop dry parcels.
    wet = vol_flat > 1e-30
    rho_w = rho_flat[wet]
    vol_w = vol_flat[wet]

    # Densest-at-BOTTOM packing (shared kernel; the previous inline block
    # packed densest at the surface -- fixed 2026-08-10, see
    # pack_sorted_rpe's docstring).
    return pack_sorted_rpe(rho_w, vol_w, A_total, g_val=g_val)


def rpe_drift_rate_per_m2(rpe_t0: float, rpe_t1: float,
                          delta_t_s: float, area_total_m2: float) -> float:
    """Spurious mixing flux ``dRPE/dt / A_total`` [W/m^2].

    Petersen et al. 2015 quotes MPAS-Ocean values of
    0.1-0.3 mW/m^2 on the Petersen lock-exchange benchmark; the
    long-term-diagnostics acceptance bar in legoESM is
    ``|dRPE/dt| / A_total < 0.5 mW/m^2``.
    """
    if delta_t_s <= 0.0 or area_total_m2 <= 0.0:
        return float("nan")
    return (rpe_t1 - rpe_t0) / (delta_t_s * area_total_m2)


__all__ = ["compute_rpe", "pack_sorted_rpe", "rpe_drift_rate_per_m2"]
