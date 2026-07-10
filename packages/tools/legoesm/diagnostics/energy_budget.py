"""Energy budget closure diagnostics.

Provides functions to compute column-integrated moist static energy,
TOA and surface radiative fluxes, and track the energy budget residual
for validating conservation in climate simulations.

The key diagnostic is the energy budget residual:
    R = R_TOA - dE/dt
where R_TOA is the net TOA radiation (positive downward) and dE/dt
is the column energy tendency. For a well-conserving model, the
global-mean |R| should be < 0.1 W/m².

References
----------
- Lucarini & Ragone (2011), Energetics of climate models
- Trenberth et al. (2009), Earth's global energy budget
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.diagnostics.column_integrals import column_water_vapor


# ======================================================================
# Area-weighted global means
# ======================================================================

def area_weighted_mean(field: jax.Array, area: jax.Array | None) -> jax.Array:
    """Area-weighted global mean of a horizontal (optionally layered) field.

    ``jnp.mean`` weights every grid cell equally.  On a lat-lon grid the
    polar rows then count as much as the equatorial rows despite each cell
    spanning ``~cos(lat)`` less area, biasing every global-mean diagnostic
    toward the cold, ice-covered high latitudes.  On a 24-band coupled run
    this pulled ``<rsdt>`` to ~281 W/m² (the true area-weighted value is the
    energy-conserving ``S_0/4`` = 340 W/m²) and manufactured a ``<R_TOA>`` ≈
    −36 W/m² "cold drift" where the area-weighted TOA budget is near balance.
    Weight each cell by its true area instead.

    Parameters
    ----------
    field : array
        Field whose leading axes are the horizontal grid (shape matching
        ``area``).  Any trailing axes (e.g. vertical levels) are averaged
        uniformly first, reproducing ``jnp.mean`` semantics in those axes.
    area : array or None
        Per-cell area with the field's horizontal shape (``grid.grid_area``;
        lat-lon ``(n_lat, n_lon)``, cube ``(6, n, n)``).  Need not be
        normalized.  ``None`` falls back to an unweighted ``jnp.mean`` — cube
        cells are ~equal area, and legacy callers / tests stay byte-identical.
    """
    if area is None:
        return jnp.mean(field)
    nh = area.ndim
    # Defensive: a field whose horizontal axes don't match ``area`` (e.g. an
    # ocean field on a different grid than the atmosphere's area) falls back
    # to an unweighted mean rather than broadcasting silently.
    if field.ndim < nh or tuple(field.shape[:nh]) != tuple(area.shape):
        return jnp.mean(field)
    if field.ndim > nh:
        field = jnp.mean(field, axis=tuple(range(nh, field.ndim)))
    return jnp.sum(field * area) / jnp.sum(area)


def area_weighted_profile(field: jax.Array, area: jax.Array | None) -> jax.Array:
    """Area-weighted horizontal mean retaining the trailing (vertical) axis.

    Companion to :func:`area_weighted_mean` for level-resolved profiles.
    ``field`` has horizontal leading axes matching ``area`` plus one trailing
    level axis; returns the area-weighted mean over the horizontal axes only,
    shape ``(nlev,)``.  ``area=None`` falls back to an unweighted horizontal
    mean (``jnp.mean`` over all-but-last axis).
    """
    spatial_axes = tuple(range(field.ndim - 1))
    if area is None:
        return jnp.mean(field, axis=spatial_axes)
    nh = area.ndim
    # Same defensive guard as ``area_weighted_mean``: a field whose horizontal
    # axes don't match ``area`` (wrong grid, or no trailing level axis) falls
    # back to a plain horizontal mean rather than broadcasting silently.
    if field.ndim <= nh or tuple(field.shape[:nh]) != tuple(area.shape):
        return jnp.mean(field, axis=spatial_axes)
    w = area[..., None]
    return jnp.sum(field * w, axis=spatial_axes) / jnp.sum(area)


# ======================================================================
# Column Energy
# ======================================================================

def column_moist_static_energy(
    T: jax.Array,
    q_v: jax.Array,
    u: jax.Array,
    v: jax.Array,
    phis: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    sigma_full: jax.Array,
) -> jax.Array:
    """Compute column-integrated moist static energy.

    E = ∫ (c_p·T + L_v·q + g·z + ½(u²+v²)) dp/g

    For a hydrostatic sigma-coordinate model, dp = p_s · dσ, so:
    E = Σ_k (c_p·T_k + L_v·q_k + Φ_k + ½(u_k²+v_k²)) · p_s · dσ_k / g

    where Φ_k is the geopotential at level k computed hydrostatically.

    Parameters
    ----------
    T : array, shape (..., nlev)
        Temperature [K].
    q_v : array, shape (..., nlev)
        Specific humidity [kg/kg].
    u, v : array, shape (..., nlev)
        Wind components [m/s].
    phis : array, shape (...)
        Surface geopotential [m²/s²].
    p_s : array, shape (...)
        Surface pressure [Pa].
    dsigma : array, shape (nlev,)
        Sigma layer thicknesses.
    sigma_full : array, shape (nlev,)
        Full-level sigma values (for geopotential computation).

    Returns
    -------
    E : array, shape (...)
        Column-integrated energy [J/m²].
    """
    # iter-47: promote to fp64 budget accumulator (see iter-42..46
    # fp64-field convention).  Column MSE involves c_p·T (~10^5),
    # L_v·q (~10^4), Φ (~10^4), 0.5(u²+v²) (~10²) summed over nlev
    # levels — the fp32 product+sum in the pre-iter-47 path leaked
    # ~7 bits of relative precision.
    from legoesm.core.conservation import conservation_accumulator
    _acc_e = conservation_accumulator()
    T = T.astype(_acc_e)
    q_v = q_v.astype(_acc_e)
    u = u.astype(_acc_e)
    v = v.astype(_acc_e)
    phis = phis.astype(_acc_e)
    p_s = p_s.astype(_acc_e)
    dsigma = dsigma.astype(_acc_e)
    sigma_full = sigma_full.astype(_acc_e)

    g = jnp.asarray(constants.g, dtype=_acc_e)
    c_p = jnp.asarray(constants.c_pd, dtype=_acc_e)
    L_v = jnp.asarray(constants.L_v, dtype=_acc_e)
    R_d = jnp.asarray(constants.R_d, dtype=_acc_e)

    # Compute geopotential at full levels (hydrostatic, bottom-up)
    # Φ_k = phis + R_d * Σ_{j>k} T_j * dln(p)_j + R_d * T_k * 0.5 * dln(p)_k
    # For sigma coords: dln(p) = dσ/σ at each level
    # Simpler approach: integrate from bottom
    nlev = T.shape[-1]
    dp = p_s[..., None] * dsigma  # (..., nlev)
    p_full = p_s[..., None] * sigma_full  # (..., nlev)

    # Geopotential via hydrostatic integration (bottom to top)
    # Φ(k) = phis + R_d * sum_{j=nlev-1..k+1} T_j * dp_j / p_j + R_d * T_k * dp_k / (2*p_k)
    # We use the standard: Φ_{k-1} = Φ_k + R_d * T_k * ln(p_{k}/p_{k-1})
    # But with sigma: ln(p_k/p_{k-1}) = ln(σ_k/σ_{k-1})

    # Bottom level first
    dlnp = jnp.log(sigma_full[..., 1:] / sigma_full[..., :-1])  # (nlev-1,)

    # Build geopotential at full levels, bottom to top
    # Start with bottom level: Φ_bottom = phis + R_d * T_bottom * ln(σ_sfc / σ_bottom)
    # Approximate: phis + R_d * T_bottom * dσ_bottom / (2 * σ_bottom)
    Phi_bottom = phis + R_d * T[..., -1] * dsigma[-1] / (2.0 * sigma_full[-1])

    def _scan_fn(Phi_below, k_from_bot):
        # k_from_bot: 0 = second-from-bottom, 1 = third-from-bottom, etc.
        j = nlev - 1 - k_from_bot  # index into full arrays (going up from nlev-2)
        j_below = j + 1  # the level below (already computed)
        # Phi_j = Phi_{j+1} + R_d * (T_j + T_{j+1}) / 2 * ln(σ_{j+1}/σ_j)
        T_mean = 0.5 * (T[..., j] + T[..., j_below])
        Phi_here = Phi_below + R_d * T_mean * jnp.log(sigma_full[j_below] / sigma_full[j])
        return Phi_here, Phi_here

    # Scan over levels from bottom-1 upward
    if nlev > 1:
        _, Phi_upper = jax.lax.scan(
            _scan_fn, Phi_bottom, jnp.arange(nlev - 1)
        )
        # Phi_upper shape: (nlev-1, ...) — need to transpose and combine
        # Reverse to get top-to-bottom ordering, then flip back to level order
        # Phi_upper[0] = level nlev-2, Phi_upper[1] = nlev-3, ...
        # We want Phi[k] for k = 0..nlev-1
        # Rearrange: Phi = [Phi_upper[-1], ..., Phi_upper[0], Phi_bottom]
        Phi_upper_reversed = jnp.flip(Phi_upper, axis=0)  # now [top, ..., bottom-1]
        # Move level axis: Phi_upper is (nlev-1, ..spatial..)
        # We need (...spatial.., nlev)
        n_spatial = len(T.shape) - 1
        axes_perm = list(range(1, n_spatial + 1)) + [0]
        Phi_upper_t = jnp.transpose(Phi_upper_reversed, axes_perm)  # (..., nlev-1)
        Phi = jnp.concatenate([Phi_upper_t, Phi_bottom[..., None]], axis=-1)
    else:
        Phi = Phi_bottom[..., None]

    # Energy integrand per level: (c_p * T + L_v * q + Phi + KE) * dp / g
    KE = 0.5 * (u ** 2 + v ** 2)
    integrand = (c_p * T + L_v * q_v + Phi + KE) * dp / g

    # Sum over vertical
    E = jnp.sum(integrand, axis=-1)
    return E


def column_dry_static_energy(
    T: jax.Array,
    phis: jax.Array,
    p_s: jax.Array,
    dsigma: jax.Array,
    sigma_full: jax.Array,
) -> jax.Array:
    """Compute column-integrated dry static energy (c_p·T + Φ) dp/g.

    Useful for checking the dry energy budget separately.
    """
    # iter-47: fp64 budget accumulator (mirrors moist twin above).
    from legoesm.core.conservation import conservation_accumulator
    _acc_d = conservation_accumulator()
    T = T.astype(_acc_d)
    phis = phis.astype(_acc_d)
    p_s = p_s.astype(_acc_d)
    dsigma = dsigma.astype(_acc_d)
    sigma_full = sigma_full.astype(_acc_d)
    g = jnp.asarray(constants.g, dtype=_acc_d)
    c_p = jnp.asarray(constants.c_pd, dtype=_acc_d)
    R_d = jnp.asarray(constants.R_d, dtype=_acc_d)

    dp = p_s[..., None] * dsigma
    nlev = T.shape[-1]

    Phi_bottom = phis + R_d * T[..., -1] * dsigma[-1] / (2.0 * sigma_full[-1])

    def _scan_fn(Phi_below, k_from_bot):
        j = nlev - 1 - k_from_bot
        j_below = j + 1
        T_mean = 0.5 * (T[..., j] + T[..., j_below])
        Phi_here = Phi_below + R_d * T_mean * jnp.log(sigma_full[j_below] / sigma_full[j])
        return Phi_here, Phi_here

    if nlev > 1:
        _, Phi_upper = jax.lax.scan(_scan_fn, Phi_bottom, jnp.arange(nlev - 1))
        Phi_upper_reversed = jnp.flip(Phi_upper, axis=0)
        n_spatial = len(T.shape) - 1
        axes_perm = list(range(1, n_spatial + 1)) + [0]
        Phi_upper_t = jnp.transpose(Phi_upper_reversed, axes_perm)
        Phi = jnp.concatenate([Phi_upper_t, Phi_bottom[..., None]], axis=-1)
    else:
        Phi = Phi_bottom[..., None]

    E_dry = jnp.sum((c_p * T + Phi) * dp / g, axis=-1)
    return E_dry


# ======================================================================
# Radiative Fluxes
# ======================================================================

def toa_net_radiation(
    sw_flux_down_toa: jax.Array,
    sw_flux_up_toa: jax.Array,
    lw_flux_up_toa: jax.Array,
) -> jax.Array:
    """Net TOA radiation (positive downward).

    R_TOA = SW_down_TOA - SW_up_TOA - LW_up_TOA

    (LW_down_TOA ≈ 0 since there is no incoming LW from space.)
    """
    return sw_flux_down_toa - sw_flux_up_toa - lw_flux_up_toa


def surface_net_radiation(
    sw_net_sfc: jax.Array,
    lw_net_sfc: jax.Array,
) -> jax.Array:
    """Net surface radiation (positive into surface).

    R_sfc = SW_net_sfc + LW_net_sfc
    """
    return sw_net_sfc + lw_net_sfc


def surface_energy_flux(
    sw_net_sfc: jax.Array,
    lw_net_sfc: jax.Array,
    shflx: jax.Array,
    lhflx: jax.Array,
) -> jax.Array:
    """Net surface energy flux into the atmosphere (positive upward).

    F_sfc = -(R_sfc) + SH + LH
    (Radiation into surface is positive down, fluxes into atm are positive up.)

    Actually: from the atmosphere's perspective, the net energy entering
    the column from the surface is R_sfc - SH - LH (radiation heats the
    surface → atmosphere, while SH/LH go from surface to atmosphere but
    are already included in physics tendencies).

    For the energy budget: d/dt(E_atm) = R_TOA - F_sfc_net
    where F_sfc_net = net flux from atmosphere to surface.
    """
    return sw_net_sfc + lw_net_sfc - shflx - lhflx


# ======================================================================
# Energy Budget Tracker
# ======================================================================

class EnergyBudget(NamedTuple):
    """Snapshot of column energy budget diagnostics (global means, W/m²)."""
    toa_net: float          # R_TOA: net TOA radiation (positive down)
    toa_sw_down: float      # SW_down at TOA
    toa_sw_up: float        # SW_up at TOA
    toa_lw_up: float        # LW_up at TOA
    sfc_sw_net: float       # SW net at surface
    sfc_lw_net: float       # LW net at surface
    sfc_net: float          # total surface net radiation
    column_energy: float    # column-integrated moist static energy [J/m²]
    dE_dt: float            # energy tendency [W/m²]
    residual: float         # R_TOA - dE/dt [W/m²]


class EnergyBudgetTracker:
    """Track energy budget evolution over a simulation.

    Accumulates global-mean diagnostics at each call to `update()`.
    The residual R_TOA - dE/dt is computed from successive energy snapshots.

    Usage
    -----
    tracker = EnergyBudgetTracker()
    # At each diagnostic step:
    budget = tracker.update(
        T, q_v, u, v, phis, p_s, dsigma, sigma_full,
        sw_up_toa, lw_up_toa, sw_net_sfc, lw_net_sfc,
        dt_since_last,
    )
    # At the end:
    tracker.summary()
    """

    def __init__(self):
        self.times: list[float] = []
        self.toa_net: list[float] = []
        self.toa_sw_down: list[float] = []
        self.toa_sw_up: list[float] = []
        self.toa_lw_up: list[float] = []
        self.sfc_sw_net: list[float] = []
        self.sfc_lw_net: list[float] = []
        self.sfc_net: list[float] = []
        self.column_energy: list[float] = []
        self.dE_dt: list[float] = []
        self.residual: list[float] = []
        self._prev_energy: float | None = None
        self._prev_time: float | None = None

    def update(
        self,
        T: jax.Array,
        q_v: jax.Array,
        u: jax.Array,
        v: jax.Array,
        phis: jax.Array,
        p_s: jax.Array,
        dsigma: jax.Array,
        sigma_full: jax.Array,
        sw_down_toa: jax.Array,
        sw_up_toa: jax.Array,
        lw_up_toa: jax.Array,
        sw_net_sfc: jax.Array,
        lw_net_sfc: jax.Array,
        elapsed_seconds: float,
        area_weights: jax.Array | None = None,
    ) -> EnergyBudget:
        """Compute and record energy budget at current time.

        Parameters
        ----------
        T, q_v, u, v : arrays, shape (6, n, n, nlev)
            Atmospheric state.
        phis : array, shape (6, n, n)
            Surface geopotential.
        p_s : array, shape (6, n, n)
            Surface pressure.
        dsigma : array, shape (nlev,)
            Sigma layer thicknesses.
        sigma_full : array, shape (nlev,)
            Full-level sigma values.
        sw_down_toa : array, shape (6, n, n)
            Downward SW at TOA [W/m²].
        sw_up_toa : array, shape (6, n, n)
            Upward SW at TOA [W/m²].
        lw_up_toa : array, shape (6, n, n)
            Upward LW at TOA [W/m²].
        sw_net_sfc, lw_net_sfc : array, shape (6, n, n)
            Net surface SW and LW [W/m²].
        elapsed_seconds : float
            Time since simulation start [s].

        Returns
        -------
        EnergyBudget
            Snapshot of global-mean energy budget diagnostics.
        """
        # Column energy + TOA + surface fluxes — fuse all six means
        # into one ``jnp.stack`` + ``np.asarray`` host transfer.  The
        # previous chain serialised six GPU stalls per energy-budget
        # check.
        E = column_moist_static_energy(
            T, q_v, u, v, phis, p_s, dsigma, sigma_full,
        )
        _h = np.asarray(jnp.stack([
            area_weighted_mean(E, area_weights),
            area_weighted_mean(sw_down_toa, area_weights),
            area_weighted_mean(sw_up_toa, area_weights),
            area_weighted_mean(lw_up_toa, area_weights),
            area_weighted_mean(sw_net_sfc, area_weights),
            area_weighted_mean(lw_net_sfc, area_weights),
        ]))
        mean_E = float(_h[0])
        mean_sw_down_toa = float(_h[1])
        mean_sw_up_toa = float(_h[2])
        mean_lw_up_toa = float(_h[3])
        mean_sw_sfc = float(_h[4])
        mean_lw_sfc = float(_h[5])
        mean_toa_net = mean_sw_down_toa - mean_sw_up_toa - mean_lw_up_toa
        mean_sfc_net = mean_sw_sfc + mean_lw_sfc

        # Energy tendency and residual
        if self._prev_energy is not None and self._prev_time is not None:
            dt = elapsed_seconds - self._prev_time
            if dt > 0:
                dE_dt = (mean_E - self._prev_energy) / dt
            else:
                dE_dt = 0.0
            residual = mean_toa_net - dE_dt
        else:
            dE_dt = 0.0
            residual = 0.0

        self._prev_energy = mean_E
        self._prev_time = elapsed_seconds

        budget = EnergyBudget(
            toa_net=mean_toa_net,
            toa_sw_down=mean_sw_down_toa,
            toa_sw_up=mean_sw_up_toa,
            toa_lw_up=mean_lw_up_toa,
            sfc_sw_net=mean_sw_sfc,
            sfc_lw_net=mean_lw_sfc,
            sfc_net=mean_sfc_net,
            column_energy=mean_E,
            dE_dt=dE_dt,
            residual=residual,
        )

        self.times.append(elapsed_seconds)
        self.toa_net.append(mean_toa_net)
        self.toa_sw_down.append(mean_sw_down_toa)
        self.toa_sw_up.append(mean_sw_up_toa)
        self.toa_lw_up.append(mean_lw_up_toa)
        self.sfc_sw_net.append(mean_sw_sfc)
        self.sfc_lw_net.append(mean_lw_sfc)
        self.sfc_net.append(mean_sfc_net)
        self.column_energy.append(mean_E)
        self.dE_dt.append(dE_dt)
        self.residual.append(residual)

        return budget

    def flush_to_lists(self) -> dict[str, list]:
        """Return all accumulated lists and clear internal storage.

        Used by DiagnosticCollector.flush_to_disk() to periodically
        persist energy budget data without unbounded memory growth.
        """
        data = {
            "times": self.times,
            "toa_net": self.toa_net,
            "toa_sw_down": self.toa_sw_down,
            "toa_sw_up": self.toa_sw_up,
            "toa_lw_up": self.toa_lw_up,
            "sfc_sw_net": self.sfc_sw_net,
            "sfc_lw_net": self.sfc_lw_net,
            "sfc_net": self.sfc_net,
            "column_energy": self.column_energy,
            "dE_dt": self.dE_dt,
            "residual": self.residual,
        }
        # Clear lists but preserve _prev state for continuity
        self.times = []
        self.toa_net = []
        self.toa_sw_down = []
        self.toa_sw_up = []
        self.toa_lw_up = []
        self.sfc_sw_net = []
        self.sfc_lw_net = []
        self.sfc_net = []
        self.column_energy = []
        self.dE_dt = []
        self.residual = []
        return data

    def summary(self) -> str:
        """Return a formatted summary of the energy budget."""
        if len(self.times) < 2:
            return "Energy budget: not enough data points for summary."

        import numpy as np
        res = np.array(self.residual[1:])  # skip first (no dE/dt)
        toa = np.array(self.toa_net[1:])
        dEdt = np.array(self.dE_dt[1:])

        lines = [
            "Energy Budget Summary",
            "=" * 40,
            f"  Samples:           {len(res)}",
            f"  <R_TOA>:           {np.mean(toa):+.2f} W/m²",
            f"  <dE/dt>:           {np.mean(dEdt):+.2f} W/m²",
            f"  <Residual>:        {np.mean(res):+.4f} W/m²",
            f"  |Residual| max:    {np.max(np.abs(res)):.4f} W/m²",
            f"  |Residual| std:    {np.std(res):.4f} W/m²",
        ]
        status = "PASS" if np.max(np.abs(res)) < 1.0 else "CHECK"
        lines.append(f"  Status:            {status}")
        return "\n".join(lines)


# ======================================================================
# Moisture Budget Tracker
# ======================================================================

class MoistureBudget(NamedTuple):
    """Snapshot of moisture budget diagnostics (global means)."""
    column_water: float      # column-integrated water vapor [kg/m²]
    dW_dt: float             # water vapor tendency [kg/m²/s]
    precip_rate: float       # precipitation rate [mm/day]
    residual: float          # dW/dt + P [mm/day]


class MoistureBudgetTracker:
    """Track moisture budget evolution over a simulation.

    Tracks column-integrated water vapor and its tendency, precipitation,
    and the moisture budget residual.  For a well-conserving model,
    the annual-mean residual should be < 0.01 mm/day.

    Usage
    -----
    tracker = MoistureBudgetTracker()
    tracker.update(q_v, p_s, dsigma, precip, elapsed_seconds)
    print(tracker.summary())
    """

    def __init__(self):
        self.times: list[float] = []
        self.column_water: list[float] = []
        self.precip_rate: list[float] = []
        self.dW_dt: list[float] = []
        self.residual: list[float] = []
        self._prev_water: float | None = None
        self._prev_time: float | None = None

    def update(
        self,
        q_v: jax.Array,
        p_s: jax.Array,
        dsigma: jax.Array,
        precip: jax.Array,
        elapsed_seconds: float,
        area_weights: jax.Array | None = None,
    ) -> MoistureBudget:
        """Compute and record moisture budget at current time.

        Parameters
        ----------
        q_v : array, shape (..., nlev)
            Specific humidity [kg/kg].
        p_s : array, shape (...)
            Surface pressure [Pa].
        dsigma : array, shape (nlev,)
            Sigma layer thicknesses.
        precip : array, shape (...)
            Precipitation rate [kg/m²/s].
        elapsed_seconds : float
            Time since simulation start [s].

        Returns
        -------
        MoistureBudget
        """
        W = column_water_vapor(q_v, p_s, dsigma)
        # Fuse the column-water-vapor + precipitation means into one
        # host transfer.
        _h = np.asarray(jnp.stack([
            area_weighted_mean(W, area_weights),
            area_weighted_mean(precip, area_weights),
        ]))
        mean_W = float(_h[0])
        mean_P = float(_h[1]) * 86400.0  # kg/m²/s → mm/day

        # Tendency
        if self._prev_water is not None and self._prev_time is not None:
            dt = elapsed_seconds - self._prev_time
            if dt > 0:
                dW_dt = (mean_W - self._prev_water) / dt
            else:
                dW_dt = 0.0
            # Residual: dW/dt + P ≈ E (evaporation)
            # Residual = dW/dt + P - E; without explicit E, residual = dW/dt + P
            # For a closed system: dW/dt = E - P → residual = dW/dt + P - E = 0
            # We compute dW/dt + P as the "net source" — should be ~0 if E ≈ P over time
            residual_mm_day = dW_dt * 86400.0 + mean_P
        else:
            dW_dt = 0.0
            residual_mm_day = 0.0

        self._prev_water = mean_W
        self._prev_time = elapsed_seconds

        budget = MoistureBudget(
            column_water=mean_W,
            dW_dt=dW_dt,
            precip_rate=mean_P,
            residual=residual_mm_day,
        )

        self.times.append(elapsed_seconds)
        self.column_water.append(mean_W)
        self.precip_rate.append(mean_P)
        self.dW_dt.append(dW_dt)
        self.residual.append(residual_mm_day)

        return budget

    def flush_to_lists(self) -> dict[str, list]:
        """Return and clear accumulated lists (for periodic flush)."""
        data = {
            "times": self.times,
            "column_water": self.column_water,
            "precip_rate": self.precip_rate,
            "dW_dt": self.dW_dt,
            "residual": self.residual,
        }
        self.times = []
        self.column_water = []
        self.precip_rate = []
        self.dW_dt = []
        self.residual = []
        return data

    def summary(self) -> str:
        """Return a formatted summary of the moisture budget."""
        if len(self.times) < 2:
            return "Moisture budget: not enough data points for summary."

        import numpy as np
        res = np.array(self.residual[1:])
        W = np.array(self.column_water[1:])
        P = np.array(self.precip_rate[1:])

        lines = [
            "Moisture Budget Summary",
            "=" * 40,
            f"  Samples:           {len(res)}",
            f"  <CWV>:             {np.mean(W):.2f} kg/m²",
            f"  <Precip>:          {np.mean(P):.2f} mm/day",
            f"  <dW/dt + P>:       {np.mean(res):+.4f} mm/day",
            f"  |Residual| max:    {np.max(np.abs(res)):.4f} mm/day",
        ]
        status = "PASS" if np.max(np.abs(res)) < 0.1 else "CHECK"
        lines.append(f"  Status:            {status}")
        return "\n".join(lines)
