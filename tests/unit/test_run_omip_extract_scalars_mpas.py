"""The MPAS branch of run_omip._extract_scalars: masked jnp reductions on the
(possibly process-sharded) state -- a host gather via np.asarray fails under
the route-B multicontroller -- returning the same scalars as the old numpy
path in serial."""

import inspect
import types

import numpy as np

import scripts.run.run_omip as run_omip


def test_mpas_branch_source_uses_jnp_reductions():
    src = inspect.getsource(run_omip._extract_scalars)
    start = src.index('if grid_type == "mpas":')
    end = src.index("    else:", start)
    branch = src[start:end]
    assert "np.asarray(state" not in branch
    assert "jnp.sum(jnp.where(wet" in branch
    assert "jnp.max(jnp.abs(state.u.data))" in branch


def test_mpas_branch_matches_old_numpy_path():
    rng = np.random.default_rng(0)
    n_cells, n_lev, n_edges = 50, 4, 120
    T = rng.normal(size=(n_cells, n_lev))
    S = rng.normal(size=(n_cells, n_lev))
    eta = rng.normal(size=n_cells)
    u = rng.normal(size=(n_edges, n_lev))
    land_mask = np.ones(n_cells)
    land_mask[rng.choice(n_cells, size=10, replace=False)] = 0.0   # 1 = wet
    state = types.SimpleNamespace(
        T=types.SimpleNamespace(data=T), S=types.SimpleNamespace(data=S),
        eta=types.SimpleNamespace(data=eta),
        land_mask=types.SimpleNamespace(data=land_mask),
        u=types.SimpleNamespace(data=u),
    )
    wet = land_mask > 0.5
    ref = {"SST": float(np.mean(T[wet, 0])), "SSS": float(np.mean(S[wet, 0])),
           "SSH": float(np.mean(eta[wet])), "max_speed": float(np.max(np.abs(u)))}
    out = run_omip._extract_scalars(state, "mpas", None, None)
    for k, v in ref.items():
        np.testing.assert_allclose(out[k], v, rtol=1e-12, atol=1e-12)
    assert out["P_bt"] == 0.0 and out["j_maxu"] == -1 and out["i_maxu"] == -1
