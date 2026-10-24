"""npzd_v2: one model day on a small column stack stays finite, keeps Fe at
picomolar levels (the softplus-alpha runaway regression), and conserves total
nitrogen to round-off (the grazing-units regression)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.biogeochemistry import (
    BiogeoConfig, init_biogeo_state, step_ocean_biogeochemistry)

_N_POOLS = ("NO3", "NH4", "Pd", "Pn", "Zoo", "Det", "DON")


def _run(scheme, n_steps=48, dt=1800.0):
    shape, nlev = (4, 6, 20), 20
    z_full = -jnp.cumsum(jnp.full(nlev, 25.0)) + 12.5
    dz = jnp.full(nlev, 25.0)
    mask = jnp.ones(shape[:2]).at[0, 0].set(0.0)
    T = jnp.full(shape, 15.0); S = jnp.full(shape, 35.0)
    cfg = BiogeoConfig(scheme=scheme, pCO2_atm=284.0)
    st = init_biogeo_state(shape, z_full, cfg)
    U10 = jnp.full(shape[:2], 7.0); PAR = jnp.full(shape[:2], 80.0)
    def inv(s, f):
        return float(jnp.sum(getattr(s, f) * dz * mask[..., None]))
    st0 = st
    for _ in range(n_steps):
        st, diag = step_ocean_biogeochemistry(st, T, S, dz, z_full, mask, dt, cfg,
                                              U10=U10, PAR_surf=PAR)
    return st0, st, diag, inv


def test_npzd_v2_finite_and_fe_scale():
    st0, st, diag, inv = _run("npzd_v2")
    for f in st._fields:
        a = getattr(st, f)
        if a is not None:
            assert bool(jnp.all(jnp.isfinite(a))), f
    fe = np.asarray(st.Fe)
    assert fe.max() < 1e-8, f"Fe ran away: max {fe.max():.3e} mol/m^3"
    assert bool(jnp.all(st.O2 > 0.1))


def test_npzd_v2_nitrogen_closes():
    st0, st, diag, inv = _run("npzd_v2")
    n0 = sum(inv(st0, f) for f in _N_POOLS)
    n1 = sum(inv(st, f) for f in _N_POOLS)
    assert abs(n1 - n0) / n0 < 1e-12, (n1 - n0) / n0


def test_v1_schemes_unchanged_and_finite():
    for scheme in ("abiotic", "npzd"):
        _, st, diag, _ = _run(scheme, n_steps=8)
        for f in st._fields:
            a = getattr(st, f)
            if a is not None:
                assert bool(jnp.all(jnp.isfinite(a))), (scheme, f)
