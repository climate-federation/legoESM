"""clubb_step hands the core the mixing length exactly as CLUBB computes it.

CLUBB's only lower bound on Lscale is the lminh ramp inside the mixing-length
routine (mixing_length.F90:1505-1526), which vanishes 500 m up; there is no
``Lscale >= lmin`` floor (user decision 2026-10-19 removed the bridge's).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence import clubb as clubb_mod  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBConfig, clubb_step, derive_lmin, init_clubb_moments)

from tests.unit.test_clubb_liquid_partition import _NCOL, _NLEV, _column  # noqa: E402


def test_clubb_step_passes_the_unfloored_mixing_length(monkeypatch):
    cfg = CLUBBConfig(trop_cloud_top_press=0.0, prognostic=True)
    got = {}
    real_ml, real_core = clubb_mod.compute_mixing_length, clubb_mod.advance_clubb_core

    def ml(*a, **k):
        out = real_ml(*a, **k)
        got["computed"] = out[0]
        return out

    def core(*a, **k):
        got["passed"] = k["Lscale"]
        return real_core(*a, **k)

    monkeypatch.setattr(clubb_mod, "compute_mixing_length", ml)
    monkeypatch.setattr(clubb_mod, "advance_clubb_core", core)
    col = _column()
    clubb_step(col["u"], col["v"], col["T"], col["q_v"],
               init_clubb_moments(_NCOL, _NLEV, cfg, dtype=jnp.float64),
               col["p_full"], col["p_half"], col["z_full"], col["z_half"],
               col["T_sfc"], col["q_sfc"], col["rho"], 300.0, cfg)
    computed = np.asarray(got["computed"])
    np.testing.assert_array_equal(np.asarray(got["passed"]), computed)
    # Non-vacuity: a floor at lmin would have changed these levels.
    assert (computed < derive_lmin(cfg.params.lmin_coef)).any()
