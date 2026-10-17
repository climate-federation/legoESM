"""tke_tke's zdiag and RHS in NEMO's -O0 arithmetic (V1-selected).

zdftke.f90:247 zfact2 = 1.5*rn_Dt*rn_ediss is 11340 on V1 (rn_Dt 10800,
rn_ediss 0.7); under jit XLA reassociates the unrounded form to
dt*(1.5*0.7) = 11339.999999999998, one ulp off on every wet diagonal.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.vertical_mixing import tke as tke_mod

pytestmark = pytest.mark.skipif(not jax.config.x64_enabled,
                                reason="binary64 receipt")

SHAPE = (3, 4, 6)
DT = 10800.0


def _inputs():
    rng = np.random.default_rng(919)

    def rand(scale, s=SHAPE):
        return jnp.asarray(rng.uniform(0.2, 1.0, s) * scale)

    return dict(
        e_old=rand(1e-3), K_M_old=rand(1e-3), K_H_old=rand(1e-4),
        P_s=rand(1e-6), N2=rand(1e-4), l_eps=rand(10.0), dz_half=rand(10.0),
        dz_face_surface=jnp.asarray(rng.uniform(5.0, 10.0, SHAPE[:-1])),
        surface_dirichlet=jnp.asarray(rng.uniform(1e-4, 1e-3, SHAPE[:-1])),
        K_M_surface=jnp.asarray(rng.uniform(1e-4, 1e-3, SHAPE[:-1])),
        nemo_e3t=rand(10.0, SHAPE[:-1] + (SHAPE[-1] + 1,)),
        dissl_old=rand(1e-2))


def _trace(flag, dt, x):
    cfg = tke_mod.TKEConfig(
        tke_matrix_evaluation="nemo_literal",
        tke_solver_evaluation="nemo_literal",
        dissipation_discretization="nemo_1p5_split",
        tke_buoyancy_sink="nemo_explicit",
        nemo_tke_matrix_rhs_source_rounded=flag)
    return tke_mod._solve_tke_backward_euler(
        surface_flux=jnp.zeros(SHAPE[:-1]), dt=dt, cfg=cfg,
        surface_bc_level="nemo_z0", w_active=jnp.ones(SHAPE),
        return_statement_trace=True, **x)


def test_selected_diag_and_rhs_are_nemo_written_order_under_jit():
    x = _inputs()
    out = jax.jit(lambda dt, x: _trace(True, dt, x))(jnp.float64(DT), x)
    eager = _trace(True, jnp.float64(DT), x)
    for a, b in zip(out, eager):
        if a is not None:
            assert np.array_equal(np.asarray(a), np.asarray(b)), "eager != jit"
    _e, _rb, rhs_ext, up, lw, diag, _ = (np.asarray(v) if v is not None else v
                                         for v in out)
    d, en = np.asarray(x["dissl_old"]), np.asarray(x["e_old"])
    k_h, n2, p_s = (np.asarray(x[k]) for k in ("K_H_old", "N2", "P_s"))
    zfact2 = (1.5 * DT) * 0.7                       # zdftke.f90:247
    assert zfact2 == 11340.0
    nemo_diag = ((1.0 - lw) - up) + (zfact2 * d) * 1.0          # :420
    nemo_rhs = en + DT * ((p_s - k_h * n2) + ((0.5 * 0.7) * d) * en) * 1.0
    assert np.array_equal(diag, nemo_diag)
    assert np.array_equal(rhs_ext[..., 1:], nemo_rhs)


def test_flag_defaults_off():
    assert tke_mod.TKEConfig().nemo_tke_matrix_rhs_source_rounded is False


def test_flag_without_the_literal_matrix_raises():
    # On the factored matrix the flag would otherwise run and change nothing.
    x = _inputs()
    for k in ("nemo_e3t", "dissl_old", "dz_face_surface", "K_M_surface"):
        x.pop(k)
    cfg = tke_mod.TKEConfig(nemo_tke_matrix_rhs_source_rounded=True)
    with pytest.raises(ValueError, match="requires tke_matrix_evaluation"):
        tke_mod._solve_tke_backward_euler(
            surface_flux=jnp.zeros(SHAPE[:-1]), dt=DT, cfg=cfg,
            w_active=jnp.ones(SHAPE), **x)
