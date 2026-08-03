"""Unit tests for the per-archetype simulated-SIF forward
(``legoesm.land.carbon.sif_forward``).

Compute-node scale (JIT-compiles the coupled-Farquhar solve + its reverse-mode); run via
the sbatch/srun wrapper, NOT the login node.  ``JAX_ENABLE_X64=1``; forced CPU here.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import numpy.testing as npt


def _table(pft_id, mat_k, sw, *, t_amp=None, map_yr=None):
    from legoesm.land.carbon.global_init import ArchetypeTable
    pft_id = np.asarray(pft_id, dtype=int)
    n = pft_id.shape[0]
    return ArchetypeTable(
        pft_id=pft_id,
        mat_k=np.asarray(mat_k, dtype=float),
        map_yr=np.asarray(map_yr if map_yr is not None else [1200.0] * n, dtype=float),
        t_seasonal_amp_k=np.asarray(t_amp if t_amp is not None else [6.0] * n, dtype=float),
        aridity=np.asarray([1.0] * n, dtype=float),
        sw_mean_w=np.asarray(sw, dtype=float),
        soil_class=np.asarray(["loam"] * n, dtype=object),
    )


def _tiny():
    return _table([1, 4, 7], [283.0, 295.0, 300.0], [170.0, 220.0, 250.0],
                  t_amp=[14.0, 7.0, 2.0], map_yr=[700.0, 1400.0, 2600.0])


def test_simulated_sif_nonnegative_finite_shape():
    """SIF is a non-negative emission [umol/m2/s], one value per archetype."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import simulate_archetype_sif

    sif = np.asarray(simulate_archetype_sif(_tiny(), SIFConfig()))
    assert sif.shape == (3,)
    assert np.all(np.isfinite(sif))
    assert np.all(sif >= 0.0)   # SIF >= 0 (emission; sign convention)


def test_simulated_sif_increases_with_shortwave():
    """More growing-season shortwave -> more absorbed PAR -> more SIF (sign check)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import simulate_archetype_sif

    def sif_at(sw):
        return float(np.asarray(simulate_archetype_sif(
            _table([4], [295.0], [sw]), SIFConfig()))[0])

    assert sif_at(260.0) > sif_at(120.0) > 0.0


def test_simulated_sif_zero_when_dark():
    """sw_mean = 0 -> APAR = 0 -> leaf_sif = 0 exactly (SIF = fs * APAR)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import simulate_archetype_sif

    sif = float(np.asarray(simulate_archetype_sif(
        _table([4], [295.0], [0.0]), SIFConfig()))[0])
    npt.assert_allclose(sif, 0.0, atol=1e-10)


def test_simulated_sif_scales_with_escape_probability():
    """fesc is a linear [0,1] multiplier on the observed TOC SIF (the aggregation wiring)."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import simulate_archetype_sif

    table = _tiny()
    full = np.asarray(simulate_archetype_sif(table, SIFConfig(escape_probability=1.0)))
    half = np.asarray(simulate_archetype_sif(table, SIFConfig(escape_probability=0.5)))
    npt.assert_allclose(half, 0.5 * full, rtol=1e-10)


def test_build_sif_forward_matches_simulate():
    """The trainer's precompute path (leaf_sif-only graph) equals the all-in-one forward."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import build_sif_forward, simulate_archetype_sif

    table = _tiny()
    sif_fn = build_sif_forward(table)
    for cfg in (SIFConfig(), SIFConfig(kn0=3.1, max_electron_yield=0.08, kf=0.06)):
        npt.assert_allclose(
            np.asarray(sif_fn(cfg)), np.asarray(simulate_archetype_sif(table, cfg)),
            rtol=1e-10)


def test_sif_forward_differentiable_in_sif_params():
    """AD de-risk (unit-test form): grad of a SIF MSE loss w.r.t. the SIFConfig params is
    FINITE and NON-ZERO -- the single-step SIF forward compiles under reverse mode (no
    spin-up scan, so no coupled-lax.scan XLA crash)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from legoesm.land.canopy.sif import SIFConfig
    from legoesm.land.carbon.sif_forward import simulate_archetype_sif
    from legoesm.core.param_overrides import apply_param_overrides
    from legoesm.training.param_collector import (
        build_trainable_params,
    )

    table = _tiny()
    params = build_trainable_params(
        active_scheme_keys={"land.canopy.sif"}, tier="extended", dtype=jnp.float64)
    target = jnp.asarray([0.4, 0.9, 1.3], dtype=jnp.float64)

    def loss(p):
        ov = p.to_overrides().get("land.canopy.sif", {})
        cfg = apply_param_overrides(SIFConfig(), ov)   # traced; prod defaults untouched
        return jnp.mean((simulate_archetype_sif(table, cfg) - target) ** 2)

    val, grads = eqx.filter_value_and_grad(loss)(params)
    assert np.isfinite(float(val))
    gmax = {c.field: float(jnp.max(jnp.abs(grads.raw_values[c.name])))
            for c in params.constraints}
    assert all(np.isfinite(v) for v in gmax.values()), gmax
    assert any(v > 1e-12 for v in gmax.values()), gmax   # at least one param moves the loss
