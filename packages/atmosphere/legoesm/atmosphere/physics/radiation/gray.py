"""Two-stream gray radiation (Frierson-2006 STRUCTURE, re-tuned constants).

Implements a simplified but physically consistent radiation scheme:

**Longwave (LW):**
  Hemispheric-mean two-stream with latitude- and moisture-dependent
  optical depth in the Frierson et al. (2006) functional form. The LW
  sweeps use `jax.lax.scan` for JIT-friendliness and differentiability.

**Shortwave (SW):**
  Frierson/Isca-style Beer-Lambert absorption of the downward stream
  only. Reflected upward SW escapes directly to TOA (no atmospheric
  absorption of the upward beam). The SW optical depth profile is
  tau_sw(sigma) = sw_tau_0 * sigma^sw_exponent.

**Heating rate:**
  dT/dt = (g / c_p) * dF_net / dp  at each layer.

Faithfulness to Frierson (2006) / Isca ``two_stream_gray_rad.F90`` (B_FRIERSON)
-----------------------------------------------------------------------------
The oracle is Isca's ``two_stream_gray_rad`` (ExeClim/Isca) ``B_FRIERSON``
branch, which implements Frierson, Held & Zurita-Gotor (2006). This module
reproduces the oracle's *functional forms* — matching it to round-off on the
restricted subset noted under FAITHFUL — but ships *re-tuned constants* and one
added term, so it is Frierson-STRUCTURED, not a drop-in Frierson/Isca replica.
``tests/atmosphere/hydrostatic/unit/test_gray_radiation_faithful.py`` pins the
FAITHFUL forms (the full recurrence against a direct NumPy port of the oracle
loops; the LW latitude/vertical forms, SW Beer-Lambert, and black-surface BC by
targeted checks) and the TUNABLE departures (D, tau, moisture, sw_exponent). The
STRUCTURAL departures below — no ``sw_diff``/``diabatic_acce`` knob, the 1-Pa
``dp`` floor, and the ``p/p_s`` coordinate — are documented, not separately tested.

FAITHFUL (forms reproduce the oracle to round-off on the matched subset:
``lw_diff_factor=1``, Isca tau constants, dry LW, ``sfc_emissivity=1``,
``sw_exponent=4``, Isca ``sw_diff=0``, ``diabatic_acce=1``, ``p_s=p_ref``,
ordinary layer thickness):
  - LW optical-depth latitude profile ``tau_ref = tau_e + (tau_p-tau_e)*sin^2(lat)``
    == Isca ``lw_tau_0 = ir_tau_eq + (ir_tau_pole-ir_tau_eq)*sin(lat)^2``.
  - LW vertical profile ``[f_l*sigma + (1-f_l)*sigma^4]`` == Isca
    ``linear_tau*sigma + (1-linear_tau)*sigma^wv_exponent`` with wv_exponent=4.
  - LW two-stream recurrence: down ``F(k+1)=t_k*F(k)+(1-t_k)*B_k`` and up
    ``F(k)=t_k*F(k+1)+(1-t_k)*B_k`` with ``B=sigma_sb*T^4``, layer transmittance
    ``t_k=exp(-D*dtau)``, and the surface source ``sigma_sb*T_sfc^4`` (== Isca's
    black ``b_surf`` when ``sfc_emissivity=1``).
  - SW: downward Beer-Lambert ``insolation*exp(-tau_sw)`` and upward flux held
    constant at ``sfc_albedo*sw_down(sfc)`` == Isca ``sw_up=albedo*sw_down(n+1)``.
  - Heating ``dT/dt = (g/c_p)*d(F_up-F_down)/dp`` == Isca ``tdt_rad`` (diabatic_acce=1).

DEPARTURES (documented; NOT the Frierson/Isca defaults — see ``GrayRadiationConfig``):
  - LW diffusivity factor ``lw_diff_factor=D=1.66`` multiplies dtau in the
    transmittance ``exp(-D*dtau)``. Isca/FHZ06 apply NO separate D (their
    prescribed tau is already the hemispheric-mean/diffusive optical depth), so
    the effective optical depth is multiplied by 1.66 for a given tau (this is an
    opacity, not a 1.66x flux/transmission change). To reproduce the oracle, D=1.
  - LW tau constants ``tau_equator=7.2 / tau_pole=1.8 / linear_frac=0.2`` vs
    Isca ``ir_tau_eq=6.0 / ir_tau_pole=1.5 / linear_tau=0.1``. Combined with the
    D=1.66 above, the effective equatorial diffusive tau ~ 7.2*1.66 ~ 12 vs 6.0.
  - Moisture LW term ``tau_moist_coeff*q_v*dp/g`` has NO counterpart in Isca's
    B_FRIERSON (whose LW tau is a purely prescribed dry function of (lat, sigma)).
    It is a Byrne & O'Gorman (2013)-style interactive-vapor add-on with a custom
    coefficient (Isca's B_BYRNE uses ``bog_b=1997.9 * q * dp/p_std``, ~17x larger
    per unit q*dp than the 0.0115 here). Set ``q_v=None`` to drop this term and
    recover the prescribed-dry LW optical depth (NOT full Frierson — the re-tuned
    tau, D, SW changes, and p/p_s all remain).
  - SW ``sw_exponent=2.0`` vs Isca ``solar_exponent=4.0``; ``sw_tau_0=0.22`` vs
    Isca ``atm_abs=0.0`` (Isca's *default* SW atmosphere is transparent). Set
    ``sw_tau_0=0`` for the Isca default; ``sw_exponent`` matches Isca only at 4.
    The SW also has NO ``sw_diff`` latitude-diffusivity knob (Isca:
    ``sw_tau_0=(1-sw_diff*sin^2 lat)*atm_abs``); we match Isca only at its default
    ``sw_diff=0``.
  - No ``diabatic_acce`` heating-acceleration multiplier (Isca default 1.0), and
    the heating divide floors ``dp`` at 1 Pa (a safety floor that differs from
    Isca only for pathologically thin layers).
  - Vertical coordinate ``sigma = p/p_s`` (the FHZ06 *paper* form) vs Isca code's
    ``p/p_std`` (fixed 1000 hPa reference); identical when ``p_s = p_ref``.

References
----------
- Frierson, D. M. W., Held, I. M., & Zurita-Gotor, P. (2006).
  A Gray-Radiation Aquaplanet Moist GCM. Part I: Static Stability
  and Eddy Scale. J. Atmos. Sci., 63, 2548-2566.
- O'Gorman, P. A. & Schneider, T. (2008). The Hydrological Cycle
  over a Wide Range of Climates Simulated with an Idealized GCM.
  J. Climate, 21, 3815-3832.
- Byrne, M. P. & O'Gorman, P. A. (2013). Land-ocean warming contrast over a
  wide range of climates. J. Climate, 26, 4000-4106 (interactive-vapor gray LW).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.radiation.config import GrayRadiationConfig
from legoesm.atmosphere.physics.radiation.output import RadiationOutput

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Two-stream gray-radiation in the Frierson (2006) FORM with re-tuned "
        "constants (NOT a drop-in Frierson/Isca replica): hemispheric-mean "
        "longwave with latitude/moisture-dependent optical depth plus "
        "Beer-Lambert shortwave absorption; returns LW+SW fluxes and radiative "
        "heating rates. Departures from Isca B_FRIERSON (D=1.66, larger tau, "
        "added moisture term, sw_exponent=2, sw_tau_0=0.22) documented in the "
        "module docstring's Faithfulness section."
    ),
    "inputs": {
        "T": "K", "p_full": "Pa", "p_half": "Pa", "sfc_temperature": "K",
        "lat": "rad", "q_v": "kg/kg", "insolation": "W/m^2",
        "sfc_albedo": "1 (surface shortwave albedo)",
    },
    "outputs": {
        "lw_flux_up": "W/m^2", "lw_flux_down": "W/m^2",
        "sw_flux_up": "W/m^2", "sw_flux_down": "W/m^2",
        "heating_rate": "K/s", "lw_heating_rate": "K/s",
        "sw_heating_rate": "K/s", "toa_insolation": "W/m^2",
    },
    "sign_convention": (
        "heating_rate dT/dt>0 warms the layer; fluxes are positive in their "
        "named direction (flux_up>=0 upward, flux_down>=0 downward); optical "
        "depth tau>=0; the downward SW beam is attenuated Beer-Lambert and the "
        "surface reflects sfc_albedo*F_down back up; insolation>=0. Radiation "
        "computes heating/fluxes only: photons enter at TOA and leave at TOA and "
        "the surface, so the column energy budget is OPEN (accounted, not "
        "conserved)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Frierson, Held & Zurita-Gotor (2006), J. Atmos. Sci. 63, 2548-2566 "
        "(functional form; oracle = Isca two_stream_gray_rad.F90 B_FRIERSON); "
        "O'Gorman & Schneider (2008), J. Climate 21, 3815-3832; moisture term "
        "after Byrne & O'Gorman (2013), J. Climate 26, 4000-4106."
    ),
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_gray_radiation_faithful.py "
        "(forms pinned to an Isca B_FRIERSON port) + "
        "tests/unit/test_physics_radiation.py + "
        "tests/atmosphere/hydrostatic/unit/test_radiation.py: isothermal/grey "
        "column gives radiative-equilibrium heating profiles; net-flux "
        "divergence sets the sign of the heating rate."
    ),
}


def _compute_lw_optical_depth(
    p_half: jnp.ndarray,
    p_s: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    config: GrayRadiationConfig,
) -> jnp.ndarray:
    """Compute LW optical depth per layer.

    dtau_dry_k = tau_dry(sigma_{k+1/2}) - tau_dry(sigma_{k-1/2})
      where tau_dry(sigma) = [f_l*sigma + (1-f_l)*sigma^4] * [tau_e + (tau_p - tau_e)*sin^2(lat)]
    dtau_moist_k = tau_moist_coeff * q_v_k * dp_k / g   [m^2/kg * kg/kg * Pa / (m/s^2) = dimensionless]
    dtau_k = dtau_dry_k + dtau_moist_k

    Parameters
    ----------
    p_half : jnp.ndarray
        Pressure at interfaces (ncol, nlev+1) [Pa].
    p_s : jnp.ndarray
        Surface pressure (ncol,) [Pa].
    lat : jnp.ndarray
        Latitude (ncol,) [radians].
    q_v : jnp.ndarray or None
        Water vapor mixing ratio (ncol, nlev) [kg/kg].
    config : GrayRadiationConfig

    Returns
    -------
    jnp.ndarray
        Optical depth per layer (ncol, nlev).
    """
    f_l = config.linear_frac
    tau_e = config.tau_equator
    tau_p = config.tau_pole

    # sigma at half levels: (ncol, nlev+1)
    sigma_half = p_half / p_s[:, None]

    # Latitude-dependent reference optical depth
    sin2_lat = jnp.sin(lat) ** 2
    tau_ref = tau_e + (tau_p - tau_e) * sin2_lat  # (ncol,)

    # Cumulative optical depth at each interface
    tau_at_half = (
        f_l * sigma_half + (1.0 - f_l) * sigma_half ** 4
    ) * tau_ref[:, None]  # (ncol, nlev+1)

    # Optical depth per layer (dry)
    dtau_dry = tau_at_half[:, 1:] - tau_at_half[:, :-1]  # (ncol, nlev)
    dtau_dry = jnp.maximum(dtau_dry, 0.0)

    # Moisture feedback
    if q_v is not None:
        # Moisture optical depth: proportional to column water per layer
        # dp per layer in Pa
        dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)
        # Column water [kg/m^2] per layer ≈ q_v * dp / g
        col_water = q_v * dp / constants.g
        # NOTE: Prior code divided by p_s here, which made dtau_moist ~1e-5×
        # too small (dimensionally incorrect: col_water is already in kg/m^2).
        # tau_moist_coeff has units [m^2/kg] and directly scales column water.
        dtau_moist = config.tau_moist_coeff * col_water
        dtau = dtau_dry + jnp.maximum(dtau_moist, 0.0)
    else:
        dtau = dtau_dry

    return dtau


def _lw_two_stream(
    T: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    dtau: jnp.ndarray,
    config: GrayRadiationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute LW fluxes using hemispheric-mean two-stream.

    Uses jax.lax.scan for the upward and downward sweeps.

    Parameters
    ----------
    T : jnp.ndarray
        Temperature at full levels (ncol, nlev) [K].
    sfc_temperature : jnp.ndarray
        Surface temperature (ncol,) [K].
    dtau : jnp.ndarray
        LW optical depth per layer (ncol, nlev).
    config : GrayRadiationConfig

    Returns
    -------
    lw_up : jnp.ndarray
        Upward LW flux at interfaces (ncol, nlev+1) [W/m^2].
    lw_down : jnp.ndarray
        Downward LW flux at interfaces (ncol, nlev+1) [W/m^2].
    """
    D = config.lw_diff_factor
    eps_sfc = config.sfc_emissivity
    ncol, nlev = T.shape

    # Layer transmittance and emissivity
    transmittance = jnp.exp(-D * dtau)  # (ncol, nlev)
    emissivity = 1.0 - transmittance     # (ncol, nlev)

    # Planck emission at each layer center
    B = constants.sigma_sb * T ** 4  # (ncol, nlev)

    # --- Downward sweep (top to bottom) using scan ---
    # BC: F_down(TOA) = 0.  Pin to the result-type that the scan body
    # actually computes (transmittance * F_above + emissivity*B), so
    # the carry dtype matches under any storage/compute precision combo.
    F_down_toa = jnp.zeros(ncol, dtype=jnp.result_type(transmittance, B))
    t_T = jnp.moveaxis(transmittance, 1, 0)    # (nlev, ncol)
    eB_T = jnp.moveaxis(emissivity * B, 1, 0)  # (nlev, ncol)

    def downward_step(F_above, layer_data):
        t_k, eB_k = layer_data  # each (ncol,)
        F_below = t_k * F_above + eB_k
        return F_below, F_below

    _, F_down_interfaces = jax.lax.scan(
        downward_step,
        F_down_toa,
        (t_T, eB_T),  # (nlev, ncol) — top to bottom
    )
    # F_down_interfaces: (nlev, ncol) — interfaces 1..nlev (below each layer)
    F_down_interfaces = jnp.moveaxis(F_down_interfaces, 0, 1)  # (ncol, nlev)
    lw_down = jnp.concatenate([
        F_down_toa[:, None],
        F_down_interfaces,
    ], axis=1)  # (ncol, nlev+1)

    # --- Upward sweep (bottom to top) using scan ---
    # Surface LW boundary: emitted + reflected downward LW for non-black surface.
    F_down_sfc = lw_down[:, -1]
    F_up_sfc = (
        eps_sfc * constants.sigma_sb * sfc_temperature ** 4
        + (1.0 - eps_sfc) * F_down_sfc
    )  # (ncol,)

    # Scan from bottom (level nlev-1) to top (level 0)
    # At each level k: F_up(k) = t_k * F_up(k+1) + eps_k * B_k
    # lax.scan iterates over the FIRST axis, so transpose to (nlev, ncol)
    t_rev = t_T[::-1]    # (nlev, ncol)
    eB_rev = eB_T[::-1]  # (nlev, ncol)

    def upward_step(F_below, layer_data):
        t_k, eB_k = layer_data  # each (ncol,)
        F_above = t_k * F_below + eB_k
        return F_above, F_above

    _, F_up_interfaces_rev = jax.lax.scan(
        upward_step,
        F_up_sfc,
        (t_rev, eB_rev),
    )
    # F_up_interfaces_rev: (nlev, ncol) — from bottom+1 upward to TOA
    # Reverse back to top-to-bottom order and transpose to (ncol, nlev)
    F_up_inner = jnp.moveaxis(F_up_interfaces_rev[::-1], 0, 1)  # (ncol, nlev)
    lw_up = jnp.concatenate([
        F_up_inner,
        F_up_sfc[:, None],
    ], axis=1)  # (ncol, nlev+1): interfaces 0..nlev

    return lw_up, lw_down


def _sw_beer_lambert(
    p_half: jnp.ndarray,
    p_s: jnp.ndarray,
    insolation: jnp.ndarray,
    config: GrayRadiationConfig,
    sfc_albedo: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute SW fluxes using Beer-Lambert absorption (Frierson/Isca style).

    Only the downward SW beam is absorbed by the atmosphere. Reflected
    upward SW escapes directly to TOA without further atmospheric
    absorption. The SW optical depth profile is:

        tau_sw(sigma) = sw_tau_0 * sigma^sw_exponent

    Parameters
    ----------
    p_half : jnp.ndarray
        Pressure at interfaces (ncol, nlev+1) [Pa].
    p_s : jnp.ndarray
        Surface pressure (ncol,) [Pa].
    insolation : jnp.ndarray
        TOA insolation (ncol,) [W/m^2].
    config : GrayRadiationConfig
    sfc_albedo : jnp.ndarray or None
        Per-column surface albedo (ncol,).  ``None`` falls back to the
        scalar ``config.sfc_albedo`` (the historical behavior).  The
        driver pipeline passes its blended ice/ocean/land albedo here so
        gray SW sees the same surface the surface energy budget uses.

    Returns
    -------
    sw_up : jnp.ndarray
        Upward SW flux at interfaces (ncol, nlev+1) [W/m^2].
    sw_down : jnp.ndarray
        Downward SW flux at interfaces (ncol, nlev+1) [W/m^2].
    """
    tau_sw_0 = config.sw_tau_0
    alpha = config.sfc_albedo if sfc_albedo is None else sfc_albedo

    # sigma at interfaces
    sigma_half = p_half / p_s[:, None]  # (ncol, nlev+1)

    # Cumulative SW optical depth from TOA: tau_sw(sigma) = tau_sw_0 * sigma^exponent
    tau_sw = tau_sw_0 * sigma_half ** config.sw_exponent  # (ncol, nlev+1)

    # Downward SW flux attenuated by Beer-Lambert
    sw_down = insolation[:, None] * jnp.exp(-tau_sw)  # (ncol, nlev+1)

    # Surface reflection: F_up(sfc) = alpha * F_down(sfc)
    F_up_sfc = alpha * sw_down[:, -1]  # (ncol,)

    # Frierson/Isca convention: reflected upward SW escapes directly to TOA
    # without further atmospheric absorption. F_up is constant at all levels.
    sw_up = jnp.broadcast_to(F_up_sfc[:, None], p_half.shape)  # (ncol, nlev+1)

    return sw_up, sw_down


def _compute_heating_rate(
    flux_up: jnp.ndarray,
    flux_down: jnp.ndarray,
    p_half: jnp.ndarray,
) -> jnp.ndarray:
    """Compute heating rate from net flux divergence.

    The standard radiative heating rate in pressure coordinates is:

        dT/dt = (g / c_p) * d(F_up - F_down) / dp

    Using the net upward flux F_net↑ = F_up - F_down:

        dT/dt_k = (g / c_p) * [F_net↑(k+1) - F_net↑(k)] / [p(k+1) - p(k)]

    where interface k is at the top (low p) and k+1 at the bottom (high p)
    of layer k. When more net upward flux exits the bottom than the top,
    the layer has gained energy (positive heating).

    Parameters
    ----------
    flux_up : jnp.ndarray
        Upward flux at interfaces (ncol, nlev+1) [W/m^2].
    flux_down : jnp.ndarray
        Downward flux at interfaces (ncol, nlev+1) [W/m^2].
    p_half : jnp.ndarray
        Pressure at interfaces (ncol, nlev+1) [Pa].

    Returns
    -------
    jnp.ndarray
        Heating rate (ncol, nlev) [K/s].
    """
    # Net upward flux at each interface
    F_net_up = flux_up - flux_down  # (ncol, nlev+1)

    # Flux divergence across each layer: F_net_up(bottom) - F_net_up(top)
    # Interface k is the top of layer k, interface k+1 is the bottom.
    dF = F_net_up[:, 1:] - F_net_up[:, :-1]  # (ncol, nlev)
    dp = p_half[:, 1:] - p_half[:, :-1]       # (ncol, nlev)
    dp = jnp.clip(dp, 1.0, None)

    return (constants.g / constants.c_pd) * dF / dp


def gray_radiation(
    T: jnp.ndarray,
    p_full: jnp.ndarray,
    p_half: jnp.ndarray,
    sfc_temperature: jnp.ndarray,
    lat: jnp.ndarray,
    q_v: jnp.ndarray | None,
    insolation: jnp.ndarray,
    config: GrayRadiationConfig,
    sfc_albedo: jnp.ndarray | None = None,
) -> RadiationOutput:
    """Compute two-stream gray radiation (LW + SW).

    Parameters
    ----------
    T : jnp.ndarray
        Temperature at full levels (ncol, nlev) [K].
    p_full : jnp.ndarray
        Pressure at full levels (ncol, nlev) [Pa].
    p_half : jnp.ndarray
        Pressure at interface levels (ncol, nlev+1) [Pa].
    sfc_temperature : jnp.ndarray
        Surface temperature (ncol,) [K].
    lat : jnp.ndarray
        Latitude (ncol,) [radians].
    q_v : jnp.ndarray or None
        Water vapor mixing ratio (ncol, nlev) [kg/kg]. None for dry.
    insolation : jnp.ndarray
        TOA insolation (ncol,) [W/m^2].
    config : GrayRadiationConfig
    sfc_albedo : jnp.ndarray or None
        Per-column surface albedo (ncol,) for the SW reflection.
        ``None`` (default) uses the scalar ``config.sfc_albedo`` —
        byte-identical to the historical behavior.  The driver pipeline
        passes its blended ice/ocean/land albedo so gray SW is
        consistent with the surface energy budget (and the albedo
        parameters become trainable under gray).

    Returns
    -------
    RadiationOutput
        Fluxes and heating rates.
    """
    p_s = p_half[:, -1]  # surface pressure

    # LW optical depth per layer
    dtau_lw = _compute_lw_optical_depth(p_half, p_s, lat, q_v, config)

    # LW fluxes
    lw_up, lw_down = _lw_two_stream(T, sfc_temperature, dtau_lw, config)

    # SW fluxes
    sw_up, sw_down = _sw_beer_lambert(
        p_half, p_s, insolation, config, sfc_albedo=sfc_albedo,
    )

    # Heating rates
    lw_hr = _compute_heating_rate(lw_up, lw_down, p_half)
    sw_hr = _compute_heating_rate(sw_up, sw_down, p_half)
    total_hr = lw_hr + sw_hr

    return RadiationOutput(
        lw_flux_up=lw_up,
        lw_flux_down=lw_down,
        sw_flux_up=sw_up,
        sw_flux_down=sw_down,
        heating_rate=total_hr,
        lw_heating_rate=lw_hr,
        sw_heating_rate=sw_hr,
        # Prescribed TOA incident SW for the CMOR rsdt diagnostic (#620).
        toa_insolation=insolation,
    )
