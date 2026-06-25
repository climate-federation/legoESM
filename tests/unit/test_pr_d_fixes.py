"""PR D: held_suarez MPAS dtype threading + gradient_checkpoint auto-enable."""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def test_resolve_checkpoint_auto_enable_policy():
    """gradient_checkpoint=None must auto-enable jax.checkpoint for segments
    longer than the threshold (the documented contract that was missing — None
    silently disabled it, risking OOM on long reverse-mode-AD segments)."""
    from legoesm.driver.compiled_segments import (
        _resolve_checkpoint, _CKPT_AUTO_STEPS,
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
    from legoesm.atmosphere.held_suarez import held_suarez_init_mpas
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.core.precision import get_policy

    mesh = create_voronoi_mesh(subdivision_level=2)
    sigma = create_sigma_coordinate(8)
    state = held_suarez_init_mpas(mesh, sigma)

    expected = get_policy().storage
    for name in ("u", "T", "p_s", "phis"):
        dt = getattr(state, name).data.dtype
        assert dt == expected, (
            f"MPAS HS field {name!r} dtype {dt} != policy storage {expected}"
        )
