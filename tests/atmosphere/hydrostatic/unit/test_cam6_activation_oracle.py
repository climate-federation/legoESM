"""CAM6 ndrop activation (maxsat + activate_modal single-updraft branch) vs gfortran.

The oracle ``scripts/validate/cam6_ndrop_activate_oracle.F90`` compiles the
VERBATIM CAM6 ``activate_modal`` and ``maxsat`` with the user-decided B7 shim
(legoesm constants + legoesm.thermo saturation).  Its outputs for the cases in
``cam6_ndrop_activate_oracle_cases.txt`` are stored in
``cam6_ndrop_activate_oracle_expected.txt`` (gfortran 8.5, -O0, %.17e).
Tolerance: fp64 rtol 1e-12 / atol 1e-30 (user decision v8 20a).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.microphysics.arg_activation import (  # noqa: E402
    cam6_activate_single_updraft, cam6_maxsat, cam6_mode_sigma_factors,
)

_DIR = Path(__file__).resolve().parents[4] / "scripts" / "validate"


def _cases():
    toks = (_DIR / "cam6_ndrop_activate_oracle_cases.txt").read_text().split("\n")
    consts = [float(x) for x in toks[0].split()]
    n = int(toks[1])
    rows, i = [], 2
    for _ in range(n):
        nmode = int(toks[i])
        w, _sigw, _wdiab, _wminf, wmaxf, T, rho = (float(x) for x in toks[i + 1].split())
        arrs = [np.array([float(x) for x in toks[i + 2 + j].split()]) for j in range(4)]
        rows.append((nmode, w, wmaxf, T, rho, *arrs))
        i += 6
    return consts, rows


def _expected():
    return [np.array([float(x) for x in line.split()])
            for line in (_DIR / "cam6_ndrop_activate_oracle_expected.txt").read_text().splitlines()
            if line.strip()]


def test_oracle_constants_are_ours():
    """The oracle was fed exactly the legoesm constants this port uses (B7)."""
    consts, _ = _cases()
    ours = [np.pi, constants.rho_water, constants.M_water, constants.R_universal,
            constants.R_v, constants.R_d, constants.c_pd, constants.L_v, constants.g,
            constants.T_freeze, constants.epsilon, constants.T_ref_cam6_activation,
            constants.surften_cam6_activation]
    assert consts == [float(c) for c in ours]


@pytest.mark.parametrize("k", range(10))
def test_matches_gfortran_oracle(k):
    _, rows = _cases()
    nmode, w, wmaxf, T, rho, na, vol, hyg, sig = rows[k]
    fn, fm, fluxn, fluxm, ffa = cam6_activate_single_updraft(
        jnp.asarray(w), jnp.asarray(T), jnp.asarray(rho), jnp.asarray(na),
        jnp.asarray(vol), jnp.asarray(hyg), jnp.asarray(sig), wmaxf)
    got = np.concatenate([np.asarray(fn), np.asarray(fm), np.asarray(fluxn),
                          np.asarray(fluxm), np.atleast_1d(np.asarray(ffa))])
    np.testing.assert_allclose(got, _expected()[k], rtol=1e-12, atol=1e-30)


def test_maxsat_branches():
    """maxsat: smax = 1e-20 only if EVERY mode is tiny; eta <= 1e-20 SETS sum = 1e20."""
    _, _, f1, f2 = cam6_mode_sigma_factors(jnp.array([1.8, 1.6]))
    zeta = jnp.array(1.0)
    smc = jnp.array([1.0e-3, 1.0e-3])
    # both modes tiny (zeta > 1e5*eta) -> 1e-20
    assert float(cam6_maxsat(zeta, jnp.array([1e-6, 1e-6]), smc, f1, f2)) == 1e-20
    # only one tiny -> normal formula, not 1e-20
    s = float(cam6_maxsat(zeta, jnp.array([1e-6, 1.0e6]), smc, f1, f2))
    assert s != 1e-20 and np.isfinite(s)
    # mode 1 contributes ~3e19 (smc 1e-12, eta 1e-18); mode 2 has eta <= 1e-20, which
    # SETS sum = 1e20 (discarding mode 1) -> smax = 1e-10 exactly
    smc_big = jnp.array([1.0e-12, 1.0e-12])
    s2 = float(cam6_maxsat(jnp.array(1e-30), jnp.array([1e-18, 1e-21]), smc_big, f1, f2))
    assert s2 == 1.0 / np.sqrt(1e20)
    # reversed order: the reset comes first and mode 2's ~3e19 is ADDED after it
    s3 = float(cam6_maxsat(jnp.array(1e-30), jnp.array([1e-21, 1e-18]), smc_big, f1, f2))
    assert s3 < 1.0 / np.sqrt(1e20) * (1.0 - 1e-3)


def test_grad_finite_and_jit_parity():
    _, rows = _cases()
    nmode, w, wmaxf, T, rho, na, vol, hyg, sig = rows[0]
    args = (jnp.asarray(T), jnp.asarray(rho), jnp.asarray(na), jnp.asarray(vol),
            jnp.asarray(hyg), jnp.asarray(sig))

    def f(w_):
        return jnp.sum(cam6_activate_single_updraft(w_, *args, wmaxf)[0])

    g = jax.grad(f)(jnp.asarray(w))
    assert np.isfinite(float(g)) and float(g) > 0.0   # more updraft activates more
    eager = f(jnp.asarray(w))
    np.testing.assert_allclose(float(jax.jit(f)(jnp.asarray(w))), float(eager), rtol=1e-14)
    # gradient through the early-return (w <= 0) and empty-mode guards stays finite
    assert np.isfinite(float(jax.grad(f)(jnp.asarray(0.0))))
    _, _, _, _, _, na4, vol4, hyg4, sig4 = rows[3]
    g4 = jax.grad(lambda v: jnp.sum(cam6_activate_single_updraft(
        jnp.asarray(1.3), jnp.asarray(278.0), jnp.asarray(1.0), jnp.asarray(na4), v,
        jnp.asarray(hyg4), jnp.asarray(sig4), 10.0)[0]))(jnp.asarray(vol4))
    assert np.all(np.isfinite(np.asarray(g4)))


def test_maxsat_gradient_finite_at_zero_zeta_and_reset():
    """Codex S1 finding: zeta = 0 with a reset mode gave a NaN gradient."""
    _, _, f1, f2 = cam6_mode_sigma_factors(jnp.array([1.8]))
    for eta in (jnp.array([1e-21]), jnp.array([1.0])):
        g = jax.grad(lambda z: cam6_maxsat(z, eta, jnp.array([1e-10]), f1, f2))(jnp.asarray(0.0))
        assert np.isfinite(float(g))
