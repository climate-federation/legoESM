"""mg2_cloud_fallout donor bound: when n_substeps_max clips MG2's sub-step
count, a sub-step must not export more than a level holds (negative q_i
otherwise: CFL 2 with cap 1 took q_i = 1e-4 to -1e-4), and the bound is
inert (to round-off of the loop's own arithmetic order) when the cap does
not bind."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.microphysics._warm_rain import mg2_cloud_fallout  # noqa: E402

DT = 600.0
P_HALF = jnp.array([[50000.0, 51000.0, 52000.0, 53000.0]])
PDEL = P_HALF[:, 1:] - P_HALF[:, :-1]
C0 = jnp.array([[1.0e-4, 5.0e-5, 0.0]])
N0 = jnp.array([[1.0e5, 5.0e4, 0.0]])
CLD = jnp.ones_like(C0)


def _run(courant, cap):
    f = courant * PDEL / DT
    return mg2_cloud_fallout(C0, N0, f, f, CLD, P_HALF, DT, cap)


def test_capped_substeps_never_drive_the_column_negative():
    dq, dn, dqv, sfc, req = _run(2.5, 1)
    assert int(req[0]) == 3
    c1, n1 = C0 + dq * DT, N0 + dn * DT
    assert float(jnp.min(c1)) >= 0.0 and float(jnp.min(n1)) >= 0.0, (c1, n1)
    # mass: column change + surface flux = 0 (full cloud, no sublimation)
    col = float(jnp.sum(dq * PDEL / constants.g))
    np.testing.assert_allclose(col + float(sfc[0]), 0.0, atol=1.0e-18)


def test_bound_is_inert_when_the_cap_does_not_bind():
    f = 2.5 * PDEL / DT
    dq, dn, _, sfc, _ = mg2_cloud_fallout(C0, N0, f, f, CLD, P_HALF, DT, 256)
    # reference: the unbounded explicit loop at MG2's own nstep = 3
    c, n, acc, accn, s = C0, N0, 0.0 * C0, 0.0 * N0, 0.0
    for _ in range(3):
        F, Fn = f * c, f * n
        sh = lambda x: jnp.concatenate([jnp.zeros_like(x[:, :1]), x[:, :-1]], axis=1)  # noqa: E731
        tc, tn = (F - sh(F)) / PDEL, (Fn - sh(Fn)) / PDEL
        c, n = c - tc * DT / 3.0, n - tn * DT / 3.0
        acc, accn, s = acc - tc / 3.0, accn - tn / 3.0, s + F[:, -1] / constants.g / 3.0
    np.testing.assert_allclose(np.asarray(dq), np.asarray(acc), rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(dn), np.asarray(accn), rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(sfc), np.asarray(s), rtol=1e-13, atol=0)
