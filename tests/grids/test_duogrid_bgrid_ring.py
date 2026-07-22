"""Gates for the faithful B-grid ring-1 halo map (duogrid_bgrid_ring).

v2 (codex bgring-r1 closures): the map is CREATE-layout native — the
probe conjugates through the canonical ED face perm/rot.  The P0
detector here cross-checks the module's layout transform against the
CANONICAL jax implementation (cubed_sphere.gnomonic_ed_remap_to_create)
on an asymmetric field, so a wrong perm/rot/inverse cannot cancel out.
Application is tested under a REAL ``jax.jit`` at fp64.

C12 keeps the probe build fast in CI.
"""
from __future__ import annotations

import numpy as np
import pytest

N, NG = 16, 3   # n must be >= 2*_BAND (build guard)


@pytest.fixture(scope="module", autouse=True)
def _x64():
    import jax

    jax.config.update("jax_enable_x64", True)
    yield


@pytest.fixture(scope="module")
def ring_map():
    from legoesm.grids.duogrid_bgrid_ring import build_bgrid_ring1_map

    return build_bgrid_ring1_map(N, NG, use_disk_cache=False)


def test_layout_transform_matches_canonical():
    """Module-local create<->ref transform == the canonical jax
    gnomonic_ed_remap_to_create on an asymmetric field (P0 detector:
    a flipped perm/rot/inverse cannot pass this)."""
    import jax.numpy as jnp
    from legoesm.grids.cubed_sphere import gnomonic_ed_remap_to_create
    from legoesm.grids.duogrid_bgrid_ring import _layout_transforms

    create_to_ref, ref_window_to_create = _layout_transforms()
    rng = np.random.default_rng(5)
    ref = rng.standard_normal((6, N + 1, N + 1))
    # canonical: create = remap(ref)
    lo, _ = gnomonic_ed_remap_to_create(jnp.asarray(ref), jnp.asarray(ref))
    create_canon = np.asarray(lo)
    # module inverse must take it back
    assert np.array_equal(create_to_ref(create_canon), ref)
    # module forward (window path) must equal canonical forward
    assert np.array_equal(ref_window_to_create(ref), create_canon)


def test_map_matches_certified_exchange_jitted(ring_map):
    """Random CREATE-layout field: jitted apply == the conjugated
    certified ext_scalar B exchange (fp64, real jax.jit)."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.grids.duogrid_bgrid_ring import (
        _run_certified_exchange,
        apply_bgrid_ring1,
    )

    rng = np.random.default_rng(7)
    field = rng.standard_normal((6, N + 1, N + 1))
    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    want = _run_certified_exchange(field, N, NG, ctx["ectx"])

    apply_jit = jax.jit(
        lambda f: apply_bgrid_ring1(f, ring_map, N))
    padded = np.asarray(apply_jit(jnp.asarray(field)))
    npx = N + 1
    got = np.stack([padded[:, 0, 1:npx + 1], padded[:, npx + 1, 1:npx + 1],
                    padded[:, 1:npx + 1, 0], padded[:, 1:npx + 1, npx + 1]],
                   axis=1)
    assert np.max(np.abs(got - want)) < 1e-12          # fp64
    assert np.array_equal(padded[:, 1:npx + 1, 1:npx + 1], field)


def test_map_nonvacuous_vs_edge_pad(ring_map):
    import jax.numpy as jnp
    from legoesm.grids.duogrid_bgrid_ring import apply_bgrid_ring1

    rng = np.random.default_rng(11)
    field = rng.standard_normal((6, N + 1, N + 1))
    padded = np.asarray(apply_bgrid_ring1(jnp.asarray(field), ring_map, N))
    edgepad = np.pad(field, [(0, 0), (1, 1), (1, 1)], mode="edge")
    ring_slots = np.zeros_like(padded, dtype=bool)
    ring_slots[:, 0, 1:-1] = ring_slots[:, -1, 1:-1] = True
    ring_slots[:, 1:-1, 0] = ring_slots[:, 1:-1, -1] = True
    assert np.max(np.abs(padded - edgepad)[ring_slots]) > 0.1


def test_dispatch_rejects_unknown_mode_and_equiangular():
    from legoesm.core.fv3_sw_core import d_sw5_corner_divergence
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid,
    )

    grid = create_cubed_sphere(N, omega=0.0, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    z = np.zeros
    args = (z((6, N, N + 1)), z((6, N + 1, N)), z((6, N, N)),
            z((6, N, N)), cdgrid, 300.0)
    with pytest.raises(ValueError, match="cross_face_halo"):
        d_sw5_corner_divergence(*args, cross_face_halo="bogus")
    # equiangular duogrid must be REJECTED in faithful mode (codex
    # bgring-r1 P0-2: the map's k2e tables are ED-specific)
    with pytest.raises(ValueError, match="ED gnomonic"):
        d_sw5_corner_divergence(*args, cross_face_halo="faithful")


def test_env_parse_rejects_unknown(monkeypatch):
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        _parse_cross_face_halo_env,
    )

    monkeypatch.setenv("LEGOESM_SW_FB_CROSS_FACE_HALO", "3")
    with pytest.raises(ValueError, match="0/1/2"):
        _parse_cross_face_halo_env()
    monkeypatch.setenv("LEGOESM_SW_FB_CROSS_FACE_HALO", "2")
    assert _parse_cross_face_halo_env() == "faithful"


def test_faithful_mode_runs_finite_on_ed(ring_map):
    """One d_sw5 call in faithful mode on the ED duogrid: finite."""
    import jax.numpy as jnp
    from legoesm.core.fv3_sw_core import d_sw5_corner_divergence
    from legoesm.grids.cubed_sphere import create_fv3_native_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid,
    )

    grid = create_fv3_native_cubed_sphere(N, omega=0.0, use_duogrid=True,
                                          k2e_nord=4)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(3)
    u_d = jnp.asarray(rng.standard_normal((6, N, N + 1)))
    v_d = jnp.asarray(rng.standard_normal((6, N + 1, N)))
    ua = jnp.asarray(rng.standard_normal((6, N, N)))
    va = jnp.asarray(rng.standard_normal((6, N, N)))
    ke = d_sw5_corner_divergence(u_d, v_d, ua, va, cdgrid, 300.0,
                                 cross_face_halo="faithful")
    assert np.all(np.isfinite(np.asarray(ke)))
