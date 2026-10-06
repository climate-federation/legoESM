"""Direct tests for scripts/validate/ocean_fidelity/onestep_momentum_twin.py.

The dynzdf transcription must equal an independently assembled dense solve of
the dynzdf.F90:182-195 matrix; our production solver fed the same operands must
match it to roundoff, and a planted 10% viscosity error must be visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("netCDF4")
_DIR = Path(__file__).resolve().parents[2] / "scripts" / "validate" / "ocean_fidelity"
sys.path.insert(0, str(_DIR))
import onestep_momentum_twin as mt  # noqa: E402


def _column(n=20, seed=0):
    rng = np.random.default_rng(seed)
    e3u = np.linspace(1.0, 6.0, n) * (1.0 + 0.01 * rng.standard_normal(n))
    e3uw = np.concatenate([[0.5], 0.5 * (e3u[:-1] + e3u[1:]) * (1.0 - 0.003)])
    avm = np.concatenate([[0.0], 10 ** rng.uniform(-5, -1, n - 1)])
    x = 0.3 * rng.standard_normal(n)
    return x, avm, e3u, e3uw


def test_transcription_equals_dense_dynzdf_matrix():
    x, avm, e3u, e3uw = _column()
    dt, n = 150.0, x.size
    M = np.zeros((n, n))
    for k in range(n):
        lo = -dt * avm[k] / (e3u[k] * e3uw[k]) if k > 0 else 0.0
        up = -dt * avm[k + 1] / (e3u[k] * e3uw[k + 1]) if k + 1 < n else 0.0
        M[k, k] = 1.0 - lo - up
        if k > 0:
            M[k, k - 1] = lo
        if k + 1 < n:
            M[k, k + 1] = up
    np.testing.assert_allclose(mt.nemo_solve(x, avm, e3u, e3uw, dt), np.linalg.solve(M, x),
                               rtol=0, atol=1e-13)


def test_production_solver_matches_transcription_and_sees_plant():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_ocean)
    x, avm, e3u, e3uw = _column(seed=1)
    ref = mt.nemo_solve(x, avm, e3u, e3uw, 150.0)
    ours = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(x), jnp.asarray(avm[1:]), jnp.asarray(e3u), jnp.asarray(e3uw[1:]), 150.0))
    assert np.max(np.abs(ours - ref)) < 1e-12
    avp = avm.copy(); avp[5] *= 1.1
    planted = np.asarray(implicit_vertical_diffusion_ocean(
        jnp.asarray(x), jnp.asarray(avp[1:]), jnp.asarray(e3u), jnp.asarray(e3uw[1:]), 150.0))
    assert np.max(np.abs(planted - ref)) > 1e-6
