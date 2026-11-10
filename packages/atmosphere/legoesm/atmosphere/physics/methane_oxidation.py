"""IFS 49r1 methane oxidation and upper-stratospheric water photolysis.

ORACLE.  OpenIFS 49r1 (8f6f722) ``arpifs/phys_ec/methox.F90`` with the
constants of ``phys_ec/sumethox.F90`` and the time-varying relaxation limit
of ``climate/updrgas.F90:140-156`` (``LMETHOX_TIMEVAR = .TRUE.``).  Water
vapour is relaxed towards ``q_lim`` (``2 CH4 + H2O`` is roughly constant in
the stratosphere) below 100 hPa with a time scale growing from 100 days at
0.5 hPa, and destroyed by photolysis above 0.2 hPa (3 days above 0.001 hPa).
:func:`methox_tendency` ports METHOX in the oracle's operation order and is
checked against the running oracle in ``tests/atmosphere/test_ifs_wms_ozone_oracle.py``.

RULE 5 (vote V7, oracle_audit/ifs_port/wms_ozone/VOTES.md): the oxidation
gate ``q < q_lim`` is a smooth_max of ``q_lim - q`` with width
``qlim_gate_rel_width * q_lim`` (0 = METHOX's hard gate).  The pressure gates
stay exact: the rate is continuous across each (1/tau -> 0 at 10000 Pa and
20 Pa; the piecewise values agree at 50 Pa and 0.1 Pa).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

from legoesm import constants
from legoesm.core.smooth import smooth_max

__physics_contract__ = {
    "summary": (
        "IFS 49r1 METHOX: stratospheric water-vapour source from methane "
        "oxidation (relaxation to q_lim) and mesospheric photolysis sink."
    ),
    "inputs": {"q": "kg/kg (specific humidity)", "p_full": "Pa (full-level pressure)"},
    "outputs": {"dq_dt": "kg/kg/s (specific-humidity tendency)"},
    "sign_convention": (
        "dq_dt > 0 where q < q_lim below 100 hPa (oxidation source); "
        "dq_dt < 0 above 20 Pa (photolysis sink)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "ECMWF IFS Documentation CY49R1 Part IV sec. 'Methane oxidation'; "
        "OpenIFS 49r1 phys_ec/methox.F90, sumethox.F90, climate/updrgas.F90."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ifs_wms_ozone_oracle.py: matches the running "
        "49r1 METHOX at rel 1e-12 (fp64) on the tier-2 audit columns."
    ),
}

__param_spec__ = {
    "MethoxConfig": {
        "scheme_key": "atm.chem.MethoxConfig",
        "excluded": {
            "q_lim_kg_kg": "forcing-derived: set from surface CH4 by methox_q_lim (updrgas), not tuned",
            "ch4_vmr_constant": "forcing: the constant-GHG run's CH4 mole fraction (radiation's), not tuned",
            "qlim_gate_rel_width": "numerics: RULE 5 smoothing width (vote V7), 0 = oracle hard form",
        },
        "params": {},
    },
}

# --- SUMETHOX constants (sumethox.F90:77-87) ---
_P_BOT_OX = 10000.0        # [Pa] RPBOTOX
_P_BOT_PH = 20.0           # [Pa] RPBOTPH
_P_TOP_OX = 50.0           # [Pa] RPTOPOX
_P_TOP_PH = 0.1            # [Pa] RPTOPPH
_TAU_TOP_OX_DAYS = 100.0   # methox.F90:113
_TAU_TOP_PH_DAYS = 3.0     # methox.F90:127
_PHOTO_OFFSET = 0.01       # methox.F90:130 (EXP(ZARG)-0.01)
_ALPHA1 = (19.0 * math.log(10.0)) / (math.log(20.0) ** 4)
_ALPHA2 = math.log(1.0 / 3.0 + 0.01)
_ALPHA3 = 0.5 * (math.log(100.0) + _ALPHA2)
_LOG_PPH = math.log(_P_TOP_PH / _P_BOT_PH)
_RQLIM_CONST = 4.81e-6     # [kg/kg] RQLIM (LMETHOX_TIMEVAR = F)
_RQLIM_REFYR = 4.60e-6     # [kg/kg] RQLIM_REFYR (year 2010)
_RCH4_REFYR = 1.00e-6      # [kg/kg] RCH4_REFYR (global mean surface CH4, 2010)
_SECONDS_PER_DAY = 86400.0  # exact conversion (YOMCST RDAYI)


class MethoxConfig(NamedTuple):
    # --- relaxation limit (sumethox / updrgas) ---
    q_lim_kg_kg: float = _RQLIM_CONST
    # --- numerics (RULE 5) ---
    qlim_gate_rel_width: float = 0.01
    # --- lane (make_methox_physics) ---
    enabled: bool = False
    # Constant-GHG run (ghg_forcing='constant'): the CH4 mole fraction
    # radiation uses (ch4_ppbv).  None = external GHG: CH4 must come from the
    # per-step forcing, and its absence is an error (retro-vote R5a).
    ch4_vmr_constant: float | None = None


def methox_q_lim(ch4_vmr):
    """49r1 UPDRGAS time-varying ``RQLIM`` [kg/kg] from the current global-mean
    surface CH4 volume mixing ratio (updrgas.F90:147-153): two water molecules
    per methane molecule above the 2010 reference."""
    ch4_rmwg = constants.M_ch4 / constants.M_dry
    qlim_vmr_ref = _RQLIM_REFYR * constants.M_dry / constants.M_h2o
    ch4_vmr_ref = _RCH4_REFYR / ch4_rmwg
    return (qlim_vmr_ref + (ch4_vmr - ch4_vmr_ref) * 2.0) * constants.M_h2o / constants.M_dry


def methox_tendency(q, p_full, cfg: MethoxConfig = MethoxConfig()):
    """METHOX ``PTENQ`` [kg/kg/s] (methox.F90:93-136); any column/level shape."""
    q = jnp.asarray(q)
    p = jnp.asarray(p_full)
    qlim = cfg.q_lim_kg_kg
    hard = cfg.qlim_gate_rel_width == 0.0       # static; q_lim may be traced
    deficit = (jnp.maximum(qlim - q, 0.0) if hard
               else smooth_max(qlim - q, jnp.zeros_like(q), 1.0 / (cfg.qlim_gate_rel_width * qlim)))
    ox = p < _P_BOT_OX
    # safe operands: the branch value is only used where its gate holds
    p_ox = jnp.where(ox & (p > _P_TOP_OX), p, 0.5 * (_P_TOP_OX + _P_BOT_OX))
    pratio = jnp.log(p_ox / _P_TOP_OX) ** 4.0 / jnp.log(_P_BOT_OX / p_ox)
    tdays_ox = jnp.where(p <= _P_TOP_OX, _TAU_TOP_OX_DAYS,
                         _TAU_TOP_OX_DAYS * (1.0 + _ALPHA1 * pratio))
    if hard:
        ox = ox & (q < qlim)
    ten = jnp.where(ox, deficit / (_SECONDS_PER_DAY * tdays_ox), 0.0)
    ph = p < _P_BOT_PH
    p_ph = jnp.where(ph & (p > _P_TOP_PH), p, math.sqrt(_P_TOP_PH * _P_BOT_PH))
    zarg = _ALPHA2 - _ALPHA3 * (1.0 + jnp.cos((math.pi * jnp.log(p_ph / _P_BOT_PH)) / _LOG_PPH))
    tdays_ph = jnp.where(p <= _P_TOP_PH, _TAU_TOP_PH_DAYS,
                         1.0 / (jnp.exp(zarg) - _PHOTO_OFFSET))
    return ten - jnp.where(ph, q / (_SECONDS_PER_DAY * tdays_ph), 0.0)


def make_methox_physics(cfg: MethoxConfig):
    """MPAS-lane physics module: METHOX water-vapour tendency on the
    pre-physics vapour (parallel split, DEPARTURE M1a from callpar's
    sequential update) with RQLIM from the CH4 radiation reads this step
    (``forcing["ghg_vmr"]["ch4"]``, UPDRGAS LMETHOX_TIMEVAR, vote M2a)."""
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticTendencies

    def physics_fn(state, mesh, sigma_coord, forcing=None):
        ghg = None if forcing is None else forcing.get("ghg_vmr")
        if cfg.ch4_vmr_constant is not None:
            ch4 = cfg.ch4_vmr_constant
        elif ghg is not None and "ch4" in ghg:
            ch4 = ghg["ch4"]
        else:
            raise ValueError("ifs_methox: external GHG forcing carries no CH4 "
                             "this step")
        T = state.T.data
        q_v = state.tracers["q_v"].data
        p_full = sigma_coord.pressure_at_full(state.p_s.data).reshape(T.shape)
        dq = methox_tendency(q_v, p_full, cfg._replace(
            q_lim_kg_kg=methox_q_lim(jnp.asarray(ch4, q_v.dtype))))
        zero = jnp.zeros_like
        return HydrostaticTendencies(
            du_dt=state.u.replace(data=zero(state.u.data), name="du_dt_methox"),
            dv_dt=(None if state.v is None
                   else state.v.replace(data=zero(state.v.data), name="dv_dt_methox")),
            dT_dt=state.T.replace(data=zero(T), name="dT_dt_methox"),
            dp_s_dt=state.p_s.replace(data=zero(state.p_s.data), name="dp_s_dt_methox"),
            dphis_dt=state.phis.replace(data=zero(state.phis.data), name="dphis_dt_methox"),
            tracer_tendencies={"q_v": Field(data=dq, name="dq_v_dt_methox",
                                            dims=state.T.dims, units="kg/kg/s")},
        )

    physics_fn._wants_forcing = True
    physics_fn.reset_state = lambda: None
    return physics_fn
