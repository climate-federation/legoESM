"""``MicrophysicsOutput.nc_budget`` (#2216): the MG2 prognostic droplet-number
budget by process, published under ``MorrisonConfig.publish_qc_budget``.

Closure: the published terms sum to ``dN_c_dt`` cell by cell, on a warm case
(activation, warm rain, sedimentation, ice melt, orphan number cleared) and a
cold case (snow and graupel riming, homogeneous freezing).  Every term is non-zero somewhere, so a term booked as
zero fails coverage and a term left out fails closure.  Publishing must not
change any output.  x64.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402

from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc import (  # noqa: E402
    _cfg, _fx, _inputs,
)

TERMS = ("activation", "autoconversion", "accretion", "riming_snow", "riming_graupel",
         "sedimentation", "ice_melt_homog_freeze", "size_readjust", "no_cloud_clear")


def _call(cold: bool, publish: bool = True):
    fx = _fx()
    g, dz = _inputs(fx)
    z = jnp.zeros_like(g["T"])
    one = jnp.ones_like(g["T"])
    # cold: 40 K colder (homogeneous freezing below 233 K) with snow and graupel to rime
    T = g["T"] - (40.0 if cold else 0.0)
    hyd = HydrometeorState(
        q_c=g["q_c"], q_r=g["q_r"], q_i=1e-6 * one, q_s=(1e-4 if cold else 0.0) * one,
        # cloud-free cells carry an orphan number (as transport leaves) to clear
        q_g=(1e-4 if cold else 0.0) * one, N_c=jnp.where(g["q_c"] > 0, g["nc_kg"], 1e6) * g["rho"], N_r=g["N_r"],
        N_i=1e4 * one,
        N_s=(1e4 if cold else 0.0) * one)
    npccn = jnp.where(g["q_c"] > 0, 1e3, 0.0) * g["rho"]       # 1e3 /kg/s, per volume
    return morrison_microphysics(
        T, g["q_v"], hyd, g["p_full"], g["p_half"], g["rho"], dz, float(fx["dt"]),
        _cfg()._replace(publish_qc_budget=publish),
        cloud_fraction=g["cf"], relvar=g["relvar"], npccn=npccn)


def test_nc_budget_closes_and_covers_every_term():
    seen = dict.fromkeys(TERMS, 0)
    for cold in (False, True):
        out = _call(cold)
        nb = {k: np.asarray(v) for k, v in out.nc_budget.items()}
        assert set(nb) == set(TERMS), sorted(nb)
        total = np.asarray(out.dN_c_dt)
        gross = sum(np.abs(v) for v in nb.values())
        resid = np.abs(sum(nb.values()) - total)
        assert (resid <= 1e-12 * gross + 1e-300).all(), float((resid / (gross + 1e-300)).max())
        for k, v in nb.items():
            seen[k] += int(np.count_nonzero(v))
    missing = [k for k, n in seen.items() if n == 0]
    assert not missing, (missing, seen)


def test_publishing_changes_no_output():
    for cold in (False, True):
        on, off = _call(cold, True), _call(cold, False)
        assert off.nc_budget is None and on.nc_budget is not None
        for k in off._fields:
            a, b = getattr(off, k), getattr(on, k)
            if a is None or isinstance(a, dict):
                continue
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=k)
