"""PR D: held_suarez MPAS dtype threading + gradient_checkpoint auto-enable."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_resolve_checkpoint_auto_enable_policy():
    """gradient_checkpoint=None must auto-enable jax.checkpoint for segments
    longer than the threshold (the documented contract that was missing — None
    silently disabled it, risking OOM on long reverse-mode-AD segments)."""
    from legoesm.driver.compiled_segments import (
        _CKPT_AUTO_STEPS,
        _resolve_checkpoint,
    )
    assert _resolve_checkpoint(True, 1) is True            # explicit wins
    assert _resolve_checkpoint(False, 10_000) is False     # explicit wins
    assert _resolve_checkpoint(None, _CKPT_AUTO_STEPS + 1) is True   # long -> on
    assert _resolve_checkpoint(None, _CKPT_AUTO_STEPS) is False      # short -> off


def test_held_suarez_init_mpas_threads_storage_dtype():
    """Every MPAS Held-Suarez field must take the precision-policy storage dtype
    (not just the perturbation) — else a mixed-precision policy leaves the
    state's dtypes disagreeing (the float64-config-through-float32-state
    scan-carry mismatch)."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.core.precision import get_policy
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(subdivision_level=2)
    sigma = create_sigma_coordinate(8)
    state = held_suarez_init_mpas(mesh, sigma)

    expected = get_policy().storage
    for name in ("u", "T", "p_s", "phis"):
        dt = getattr(state, name).data.dtype
        assert dt == expected, (
            f"MPAS HS field {name!r} dtype {dt} != policy storage {expected}"
        )


def test_regrid_scalar_non_floating_field_not_truncated():
    """PR D (codex): regridding an INTEGER / BOOL field must promote the
    fractional IDW interpolation to float — NOT downcast the weights to the
    field dtype (which truncates 0.25/0.75 to 0 and corrupts masks /
    categorical fields)."""
    from legoesm.grids.regridding import RegridWeights, regrid_scalar

    rw = RegridWeights(
        src_indices=jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
        weights=jnp.array([[0.25, 0.75], [0.5, 0.5]], dtype=jnp.float64),
        target_shape=(2,),
        src_flat_size=4,
    )
    out_i = regrid_scalar(jnp.array([0, 1, 2, 3], dtype=jnp.int32), rw)
    assert jnp.issubdtype(out_i.dtype, jnp.floating)
    np.testing.assert_allclose(np.asarray(out_i), [0.75, 2.5], rtol=1e-6)

    out_b = regrid_scalar(jnp.array([True, False, True, True]), rw)
    assert jnp.issubdtype(out_b.dtype, jnp.floating)
    np.testing.assert_allclose(np.asarray(out_b), [0.25, 1.0], rtol=1e-6)


def test_regrid_scalar_fp64_field_keeps_fp64():
    """A floating field keeps its own dtype (fp64 -> full-precision gradient)."""
    from legoesm.grids.regridding import RegridWeights, regrid_scalar

    rw = RegridWeights(
        src_indices=jnp.array([[0, 1]], dtype=jnp.int32),
        weights=jnp.array([[0.3, 0.7]], dtype=jnp.float64),
        target_shape=(1,),
        src_flat_size=2,
    )
    out = regrid_scalar(jnp.array([1.0, 2.0], dtype=jnp.float64), rw)
    assert out.dtype == jnp.float64
    np.testing.assert_allclose(np.asarray(out), [1.7], rtol=1e-9)
