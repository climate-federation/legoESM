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

from legoesm import constants


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
    g = constants.g
    c_p = constants.c_pd
    L_v = constants.L_v
    R_d = constants.R_d

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
    g = constants.g
    c_p = constants.c_pd
    R_d = constants.R_d

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


def toa_net_radiation_from_output(rad_out, shape_2d: tuple[int, ...]) -> jax.Array:
    """Extract TOA net radiation from RadiationOutput.

    Parameters
    ----------
    rad_out : RadiationOutput
        Radiation output with fluxes at interfaces (ncol, nlev+1).
        Index 0 = TOA.
    shape_2d : tuple
        Spatial shape to reshape to (e.g., (6, n, n)).
    """
    sw_down_toa = rad_out.sw_flux_down[:, 0].reshape(shape_2d)
    sw_up_toa = rad_out.sw_flux_up[:, 0].reshape(shape_2d)
    lw_up_toa = rad_out.lw_flux_up[:, 0].reshape(shape_2d)
    return toa_net_radiation(sw_down_toa, sw_up_toa, lw_up_toa)


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
        # Column energy (global mean)
        E = column_moist_static_energy(
            T, q_v, u, v, phis, p_s, dsigma, sigma_full,
        )
        mean_E = float(jnp.mean(E))

        # TOA fluxes (global mean)
        mean_sw_down_toa = float(jnp.mean(sw_down_toa))
        mean_sw_up_toa = float(jnp.mean(sw_up_toa))
        mean_lw_up_toa = float(jnp.mean(lw_up_toa))
        mean_toa_net = mean_sw_down_toa - mean_sw_up_toa - mean_lw_up_toa

        # Surface fluxes (global mean)
        mean_sw_sfc = float(jnp.mean(sw_net_sfc))
        mean_lw_sfc = float(jnp.mean(lw_net_sfc))
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
