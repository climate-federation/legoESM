"""RULE 5 tests for the lake's frozen/open-water switch (two_layer_lake)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.coupler.lake.config import LakeConfig
from legoesm.coupler.lake.state import LakeState
from legoesm.coupler.lake.two_layer_lake import step_lake
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.core.field import Field
from legoesm.thermo import saturation_mixing_ratio, saturation_mixing_ratio_ice

jax.config.update("jax_enable_x64", True)

CFG = LakeConfig()
# Kelvin above T_freeze per fixture column: pinned ice lake, three points
# inside the default 0.01 K band, just outside it, warm open water.
OFFSETS = (0.0, 0.0025, 0.005, 0.0075, 0.02, 5.0)
FIELDS = ("albedo", "q_surface", "lhflx", "T_sfc")
DT = 1.0  # short step keeps T_epi_new inside the band it started in


def _forcing(n):
    c = lambda v: jnp.full((n,), v, jnp.float64)
    return AtmToSurface(
        sw_down=c(200.0), lw_down=c(300.0), precip_total=c(0.0),
        precip_snow=c(0.0), T_lowest=c(270.0), q_lowest=c(0.003),
        u_lowest=c(5.0), v_lowest=c(2.0), p_lowest=c(95000.0),
        p_surface=c(1e5), rho_lowest=c(1.2), cos_zenith=c(0.5),
        co2_ppmv=c(415.0), has_radiation=c(1.0), has_precipitation=c(1.0),
    )


def _state(T_epi):
    f = lambda d, name, u: Field(data=d, name=name, dims=("ncol",), units=u)
    # Ice debt only where the lake sits at T_freeze: that column stays pinned
    # (frozen) through the step; open water in the band carries none, as the
    # reservoir guarantees at the end of every step.
    E_ice = jnp.where(T_epi <= CFG.T_freeze, 1.0e7, 0.0)
    return LakeState(T_epi=f(T_epi, "T_epi", "K"),
                     T_hypo=f(jnp.full_like(T_epi, CFG.T_freeze + 2.0), "T_hypo", "K"),
                     Q_freeze=f(E_ice, "lake_ice_energy", "J/m2"))


def run_cases(cfg, step=step_lake, T_epi=None):
    if T_epi is None:
        T_epi = cfg.T_freeze + jnp.asarray(OFFSETS, jnp.float64)
    _, resp = step(_state(T_epi), _forcing(T_epi.shape[0]), cfg, 1.0, DT)
    return {k: getattr(resp, k) for k in FIELDS}


# Response of the scheme BEFORE the RULE 5 change (origin/main 7e2b7b3f49) on
# the fixture columns; the hard limit must reproduce it.
_OLD_SCHEME = {
    "albedo": [0.6, 0.08, 0.08, 0.08, 0.08, 0.08],
    "q_surface": [0.003824919693680484, 0.00382575764087556,
                  0.003826455832932174, 0.003827154138850153,
                  0.003830647376929619, 0.004615778949416893],
    "lhflx": [23.054298289570934, 20.357521709834675, 20.374692662046186,
              20.39186634141828, 20.47777565909715, 60.67143140689696],
    "T_sfc": [273.15, 273.1530002101403, 273.1554995885315, 273.1579989669225,
              273.1704958588754, 275.750000041485],
}


def test_hard_limit_is_the_boolean_switch():
    cfg = CFG._replace(freeze_switch_width_K=0.0)
    out = run_cases(cfg)
    for k in FIELDS:
        np.testing.assert_allclose(out[k], _OLD_SCHEME[k], rtol=1e-12, atol=0.0)
    frozen = out["T_sfc"] <= cfg.T_freeze
    np.testing.assert_array_equal(
        out["albedo"], jnp.where(frozen, cfg.albedo_lake_ice, cfg.albedo_lake))
    p = jnp.full_like(out["T_sfc"], 1e5)
    np.testing.assert_array_equal(
        out["q_surface"],
        jnp.where(frozen, saturation_mixing_ratio_ice(out["T_sfc"], p),
                  saturation_mixing_ratio(out["T_sfc"], p)))
    # pinned ice lake is frozen, every in-band column is open water
    assert bool(frozen[0]) and not bool(jnp.any(frozen[1:]))


# Default (0.01 K) minus hard limit, per fixture column (OFFSETS order), on the
# post-step response fields.  One-signed by construction: the band sits wholly
# above T_freeze, so open water in it is charged part ice albedo / L_s.
# Measured 2026-11-08 (x64, CPU); the hard limit matched the pre-change
# scheme bitwise on these columns.
_GAP = {
    "albedo": [0.0, 0.40794586723171306, 0.22135716862166938,
               0.054171075037139596, 0.0, 0.0],
    "q_surface": [0.0, 8.707934705688009e-08, 8.719449542341634e-08,
                  3.10841420866903e-08, 0.0, 0.0],
    "lhflx": [0.0, 2.2940386270199085, 1.3617827126335165,
              0.4262592440486763, 0.0, 0.0],
    "T_sfc": [0.0, -4.2695134538917046e-06, -2.53019362617124e-06,
              -7.907187864475418e-07, 0.0, 0.0],
}


@pytest.mark.parametrize("field", FIELDS)
def test_default_gap_per_case(field):
    soft = run_cases(CFG)[field]
    hard = run_cases(CFG._replace(freeze_switch_width_K=0.0))[field]
    np.testing.assert_allclose(soft - hard, np.asarray(_GAP[field]),
                               rtol=1e-6, atol=1e-12)


def test_gradient_through_the_switch():
    T0 = CFG.T_freeze + jnp.asarray([0.005], jnp.float64)

    def albedo_sum(T):
        return jnp.sum(run_cases(CFG, T_epi=T)["albedo"])

    # Mid-band point: the FD stencil stays 0.005 K from both band edges, so it
    # never meets the cubic's C2 kink.  eps 1e-5 fails order 2 on truncation
    # error (curvature ~1/width^2); 3e-6, 1e-6 and 3e-7 all pass.
    check_grads(albedo_sum, (T0,), order=2, modes=["rev"], eps=1e-6)
    g = jax.grad(albedo_sum)(T0)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.abs(g[0])) > 0.0


def test_occupancy_probe_forcing_is_seasonal():
    import importlib.util
    from pathlib import Path
    path = (Path(__file__).resolve().parents[2]
            / "scripts/validate/lake_freeze_band_occupancy.py")
    spec = importlib.util.spec_from_file_location("_lake_occ", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    jan = mod.forcing(15 * mod.N_DAY)
    jul = mod.forcing(197 * mod.N_DAY)
    assert float(jan.T_lowest[0]) < CFG.T_freeze < float(jul.T_lowest[0])
    assert float(jan.sw_down[0]) < float(jul.sw_down[0])
