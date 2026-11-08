"""``turbulence="tke"``: the WRF v4.8.0 MYJ PBL (Mellor-Yamada-Janjic level 2.5).

The kernel :func:`tke_turbulence` runs the oracle-tested column port
:func:`legoesm.atmosphere.physics.turbulence.myj_wrf.myjpbl_column`
(``tests/atmosphere/hydrostatic/unit/test_myj_oracle.py``) on every column.
It replaces the earlier MY2.5-INSPIRED k-l closure (constant Ck and Pr_t),
which the F4 audit scored C-structural.

Host glue, not part of the oracle (declared departures):

* **Surface exchange from the lane's own surface layer, not MYJSFC.**  The
  lane's bulk fluxes (``surface_fluxes_at_lowest_level``, the same call the
  previous closure made) are handed to MYJPBL as MYJSFC-shaped inputs, built
  so that MYJ's implicit surface term reproduces them at the input state:
  ``AKMS = AKHS = ustar^2/|V_1|``, ``THZ0 = TH_1 + F_theta/(rho_1 AKHS)``,
  ``QZ0 = q_1 + F_q/(rho_1 AKHS)`` on the water path (``XLAND = 2``, so MYJ
  uses THZ0/QZ0 directly), ``UZ0 = U_1 + tau_x/(rho_1 AKMS)`` (likewise V),
  ``CT = 0``; where ``|V_1|`` or ``ustar`` is exactly zero the exchange
  coefficient is ``_AK_CALM_M_S`` so prescribed fluxes still enter.
  ``rho_1`` is MYJ's own density.
  The heat flux carries the latent-enthalpy correction, as before.
* **Vapour only.**  The kernel receives no condensate, so qc = qi = 0 and THE
  = theta (MYJ would otherwise mix cloud water/ice and use THE).
* ``Km``/``Kh`` diagnostics: MYJ's interface coefficients times its own
  layer depth (``EXCH_H``), averaged to full levels; ``Kh_iface`` exports
  the interface values.  ``h_pbl`` is MYJ's PBLH.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import half_to_full, half_to_iface_zero_ends
from legoesm.atmosphere.physics.turbulence.config import TKEConfig
from legoesm.atmosphere.physics.turbulence.myj_wrf import myjpbl_column
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_fluxes_at_lowest_level,
    surface_moisture_flux,
)


# Exchange coefficient [m/s] used where the lowest-level wind or u* is exactly zero, so
# prescribed surface fluxes still enter MYJ (vote Z12).
_AK_CALM_M_S = 1e-3

__physics_contract__ = {
    "summary": (
        "turbulence='tke': WRF v4.8.0 MYJ PBL column core (myj_wrf) driven by "
        "the lane's bulk surface fluxes; prognostic TKE carry."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg (specific)",
        "tke": "m^2/s^2", "p_full": "Pa", "p_half": "Pa",
        "z_full": "m", "z_half": "m", "T_sfc": "K", "q_sfc": "kg/kg",
        "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m", "tke_new": "m^2/s^2",
    },
    "sign_convention": (
        "z up; level index -1 is the surface (arrays top-down, as every "
        "lane); shflx, lhflx positive UPWARD, entering as MYJ's implicit "
        "surface exchange (a source), so the column budget is NOT closed."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Janjic (2002) NCEP Office Note 437; WRF v4.8.0 phys/module_bl_myjpbl.F",
    "idealized_test": (
        "tests/atmosphere/hydrostatic/unit/test_myj_oracle.py (oracle match); "
        "tests/atmosphere/hydrostatic/unit/test_turbulence.py (lane adapter)"
    ),
}


def tke_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    tke: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: TKEConfig,
) -> tuple[TurbulenceOutput, jax.Array]:
    """MYJ turbulence tendencies; level arrays ``(ncol, nlev)`` top-down
    (index -1 at the surface), ``p_half``/``z_half`` ``(ncol, nlev+1)``.
    Returns the output and the new TKE [m^2/s^2]."""
    ncol, nlev = T.shape
    if nlev < 4:
        raise ValueError(f"turbulence='tke' (MYJ) needs at least 4 levels, got {nlev}")
    f = lambda x: x[..., ::-1]                              # top-down <-> MYJ bottom-up
    tau_x, tau_y, shflx, lhflx, ustar = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])
    sflx_q = surface_moisture_flux(config.surface, lhflx, T_sfc)
    sflx_T = (shflx + latent_enthalpy_correction(lhflx, sflx_q)) / constants.c_pd
    exner = (p_full / constants.p_ref) ** (constants.R_d / constants.c_pd)
    th = T / exner
    q1 = q_v[:, -1]
    p608 = constants.R_v / constants.R_d - 1.0                # MYJ's P608
    rho1 = p_full[:, -1] / (constants.R_d * T[:, -1] * (1. + p608 * q1))
    w2 = u[:, -1] ** 2 + v[:, -1] ** 2
    calm = (w2 <= 0.0) | (ustar <= 0.0)                     # exact zero only
    wspd = jnp.sqrt(jnp.where(calm, 1.0, w2))               # no sqrt'(0) in the VJP
    ak = jnp.where(calm, _AK_CALM_M_S, ustar * ustar / wspd)
    rak = 1.0 / (rho1 * ak)
    thz0 = th[:, -1] + sflx_T / exner[:, -1] * rak
    qz0 = q1 + sflx_q * rak
    # stress ON the atmosphere = -rho1 AKMS (V1 - VZ0), so this VZ0 reproduces the lane's vector
    uz0 = u[:, -1] + tau_x * rak
    vz0 = v[:, -1] + tau_y * rak
    psfc = p_half[:, -1]
    zero = jnp.zeros_like(thz0)
    cols = dict(
        dt=jnp.full((ncol,), dt, T.dtype), ht=z_half[:, -1],
        dz=f(z_half[:, :-1] - z_half[:, 1:]), pmid=f(p_full), pint=f(p_half),
        th=f(th), t=f(T), exner=f(exner), qv=f(q_v / (1. - q_v)),
        qcw=jnp.zeros_like(T), qci=jnp.zeros_like(T), u=f(u), v=f(v), tke=f(tke),
        ustar=ustar, akhs=ak, akms=ak, thz0=thz0, qz0=qz0, uz0=uz0, vz0=vz0, ct=zero,
        chklowq=jnp.ones_like(thz0), elflx=lhflx,
        tsk=thz0 / (constants.p_ref / psfc) ** (constants.R_d / constants.c_pd),
        qsfc=qz0, xland=jnp.full_like(thz0, 2.0), snow=zero, sice=zero)
    out = jax.vmap(lambda c: myjpbl_column(**c, cfg=config.myj))(cols)
    kh_iface = f(out.exch_h)                                # top-down interior interfaces
    km_iface = f(out.exch_m)
    output = TurbulenceOutput(
        du_dt=f(out.rublten),
        dv_dt=f(out.rvblten),
        dT_dt=f(out.rthblten) * exner,
        dq_v_dt=f(out.dqdt),                                # MYJ's own specific-humidity DQDT
        Km=half_to_full(km_iface),
        Kh=half_to_full(kh_iface),
        shflx=shflx,
        lhflx=lhflx, evap_sfc=sflx_q,
        ustar=ustar,
        h_pbl=out.pblh,
        Kh_iface=half_to_iface_zero_ends(kh_iface),
    )
    return output, f(out.tke)
