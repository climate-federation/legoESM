"""Probe: how often a standalone lake sits inside the smooth freeze band, and
how far the default band moves ice-on / ice-off against the hard switch.

Idealised seasonal forcing (no diurnal cycle), dt = 1800 s, 2 years; year 2
is reported.  Prints per width: fraction of steps with T_epi strictly inside
(T_freeze, T_freeze + width), ice days (ice reservoir > 0), ice-off day
(spring) and ice-on day (autumn), annual-mean absorbed shortwave.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.lake.two_layer_lake import step_lake
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.field import Field

jax.config.update("jax_enable_x64", True)

DT = 1800.0
N_DAY = int(86400 / DT)
N_YEAR = 365 * N_DAY


def forcing(step):
    phase = 2.0 * jnp.pi * (step * DT / 86400.0 - 15.0) / 365.0  # coldest mid-Jan
    T_air = 278.0 - 15.0 * jnp.cos(phase)
    sw = 170.0 - 130.0 * jnp.cos(phase)
    lw = 0.8 * constants.sigma_sb * T_air ** 4
    one = jnp.ones((1,), jnp.float64)
    return AtmToSurface(
        sw_down=sw * one, lw_down=lw * one, precip_total=0.0 * one,
        precip_snow=0.0 * one, T_lowest=T_air * one, q_lowest=0.002 * one,
        u_lowest=5.0 * one, v_lowest=2.0 * one, p_lowest=95000.0 * one,
        p_surface=1e5 * one, rho_lowest=1.2 * one, cos_zenith=0.5 * one,
        co2_ppmv=415.0 * one, has_radiation=one, has_precipitation=one,
    )


def run(cfg):
    f = lambda v, n, u: Field(data=jnp.full((1,), v, jnp.float64), name=n,
                              dims=("ncol",), units=u)
    s0 = LakeState(T_epi=f(285.0, "T_epi", "K"), T_hypo=f(280.0, "T_hypo", "K"),
                   Q_freeze=f(0.0, "lake_ice_energy", "J/m2"))

    def body(s, k):
        fo = forcing(k)
        s1, r = step_lake(s, fo, cfg, 1.0, DT)
        return s1, (s.T_epi.data[0], s1.Q_freeze.data[0],
                    fo.sw_down[0] * (1.0 - r.albedo[0]))

    _, (T, E, sw_abs) = jax.lax.scan(body, s0, jnp.arange(2 * N_YEAR))
    return T[N_YEAR:], E[N_YEAR:], sw_abs[N_YEAR:]


def main():
    base = LakeConfig()
    w0 = base.freeze_switch_width_K
    for w in (0.0, w0):
        T, E, sw_abs = run(base._replace(freeze_switch_width_K=w))
        dT = T - base.T_freeze
        in_band = float(jnp.mean((dT > 0.0) & (dT < w0)))
        ice = E > 0.0
        days = jnp.nonzero(ice, size=ice.shape[0], fill_value=-1)[0]
        days = days[days >= 0] / N_DAY
        ice_off = float(days[days < 200].max())  # winter straddles New Year
        ice_on = float(days[days >= 200].min())
        print(f"width {w:g} K: in-band step fraction {in_band:.4%}  "
              f"ice days {float(jnp.sum(ice)) / N_DAY:.3f}  "
              f"ice-off day {ice_off:.3f}  ice-on day {ice_on:.3f}  "
              f"mean absorbed SW {float(jnp.mean(sw_abs)):.4f} W/m2")


if __name__ == "__main__":
    main()
