"""COMPOSED 3D-PE surface-pressure-tendency stage (dp_s/dt) sub-face-tiled.

Composes the FLUX-FORM continuity into ONE shard_map: dgrid_to_cgrid (D->C
within-face) -> dp at the C-grid faces (in-stage halo-1 scalar pad + shared
2-point average) -> cgrid_divergence of dp·v (the staggered tile u_c/v_c
carry the tile boundary faces) -> column-integrated mass divergence
(per-column cumsum) -> dp_s/dt pointwise.  The in-stage scalar pad is
bit-identical to the global ``pad_halo_auto``, so dp_s/dt is bit-identical
(EXACT) to the global composition sliced.  Covers BOTH sigma and hybrid
coordinates, host-composition + np24/np54.

This is the continuity prognostic (the next after the D-grid momentum stage)
toward the full tiled fv3_hydrostatic_tendencies capstone.  Excludes the
downstream GLOBAL zero_mean_tendency reduction (applied after the stage).

24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.vertical import (
    create_sigma_coordinate, make_hybrid_levels,
    dp_from_hybrid,
    HybridSigmaPressureCoordinate,
)
from legoesm.core.operators_cdgrid import (
    dgrid_to_cgrid, cgrid_divergence, cgrid_divergence_local,
    cgrid_interp_cc_to_faces_local, pad_halo_auto,
)
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_dp_s_dt_stage_2d, dgrid_to_cgrid_tile_2d,
)

N = 24  # divisible by kt=2 (nl=12) and kt=3 (nl=8)
NLEV = 6


def _inputs(n, nlev, seed):
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    return u_d, v_d, p_s


def _dp_of(p_s, coord, hybrid):
    if hybrid:
        return dp_from_hybrid(coord, p_s)
    return p_s[..., None] * coord.dsigma.astype(p_s.dtype)


def _global_dps_dt(u_d, v_d, p_s, cdgrid, coord, hybrid):
    """Hand-composed global FLUX-FORM dp_s/dt = primitive_eq_cdgrid sec 10b
    base case (pre-zero-mean), single-device 'local' path."""
    u_c, v_c = dgrid_to_cgrid(u_d, v_d, cdgrid)
    dp = _dp_of(p_s, coord, hybrid)
    dp_u, dp_v = cgrid_interp_cc_to_faces_local(pad_halo_auto(dp, cdgrid))
    div_dp = cgrid_divergence(dp_u * u_c, dp_v * v_c, cdgrid)
    D_total_p = jnp.cumsum(div_dp, axis=-1)[..., -1:]
    if hybrid:
        return np.asarray(-D_total_p[..., 0] / coord.B_range)
    sigma_range = 1.0 - float(coord.sigma_half[0])
    return np.asarray(-D_total_p[..., 0] / sigma_range)


def _reassemble_cc(get_tile, kt):
    return np.concatenate(
        [np.concatenate([np.asarray(get_tile(ti, tj)) for tj in range(kt)], axis=2)
         for ti in range(kt)], axis=1)


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert g.base.duogrid is None, "base cut is non-duogrid"
    return g


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("KT", [2, 3])
def test_dp_s_dt_stage_matches_global(cdg, KT, hybrid):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    nl = N // KT
    coord = make_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    u_d, v_d, p_s = _inputs(N, NLEV, (30 if hybrid else 40) + KT)
    g = _global_dps_dt(u_d, v_d, p_s, cdg, coord, hybrid)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_dp_s_dt_stage_2d(mesh, cdg, coord, N, KT, NLEV)

    fw = NamedSharding(mesh, P("face", None, None, None))
    fo = NamedSharding(mesh, P("face", None, None))
    out = stage(jax.device_put(u_d, fw), jax.device_put(v_d, fw),
                jax.device_put(p_s, fo))
    a = np.asarray(out).reshape(6, KT, nl, KT, nl)
    t = _reassemble_cc(lambda ti, tj: a[:, ti, :, tj, :], KT)
    np.testing.assert_array_equal(
        t, g, err_msg=f"dp_s_dt != global (kt={KT}, hybrid={hybrid})")


@pytest.mark.parametrize("hybrid", [False, True])
def test_dp_s_dt_compose_host_body(cdg, hybrid):
    """Composition exactness WITHOUT 24 devices: the same per-tile chain on host
    (dgrid_to_cgrid_tile_2d -> dp face-average on the sliced halo-1 pad window
    -> cgrid_divergence_local of dp·v -> continuity) reassembled == global."""
    kt, nl = 3, N // 3
    coord = make_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    u_d, v_d, p_s = _inputs(N, NLEV, 55 if hybrid else 66)
    g = _global_dps_dt(u_d, v_d, p_s, cdg, coord, hybrid)
    cu, dye, dxe, ar = (cdg.cosa_u, cdg.dy_edge_x, cdg.dx_edge_y, cdg.base.area)
    sigma_range = 1.0 - float(coord.sigma_half[0])
    # Global halo-1 pad of dp; each tile reads its (nl+2, nl+2) window — the
    # values the in-stage scalar pad body delivers on-device.
    dp_pad_g = pad_halo_auto(_dp_of(p_s, coord, hybrid), cdg)

    def _chain(ti, tj):
        a_i, a_j = ti * nl, tj * nl
        u_c, v_c = dgrid_to_cgrid_tile_2d(u_d, v_d, cu, a_i, a_j, nl)
        dp_w = dp_pad_g[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2]
        dp_u, dp_v = cgrid_interp_cc_to_faces_local(dp_w)
        div_dp = cgrid_divergence_local(
            dp_u * u_c, dp_v * v_c, dye[:, a_i:a_i + nl + 1, a_j:a_j + nl],
            dxe[:, a_i:a_i + nl, a_j:a_j + nl + 1],
            ar[:, a_i:a_i + nl, a_j:a_j + nl])
        D_total_p = jnp.cumsum(div_dp, axis=-1)[..., -1:]
        if hybrid:
            return -D_total_p[..., 0] / coord.B_range
        return -D_total_p[..., 0] / sigma_range

    t = _reassemble_cc(_chain, kt)
    np.testing.assert_array_equal(
        t, g, err_msg=f"dp_s_dt host-body != global (hybrid={hybrid})")


def test_duogrid_refused():
    """codex 2026-07-12: the standalone dp_s/dt stage has no duogrid
    kinked->extended remap and no seam-flux sync — it must refuse a
    duogrid cdgrid loudly (like the full tiled stage), not silently
    produce seam-inconsistent fluxes.  The refusal fires before mesh
    validation, so no device mesh is needed."""
    from legoesm.parallel.tiled_production_cdgrid import (
        make_tiled_dp_s_dt_stage_2d,
    )
    from legoesm.grids.vertical import create_sigma_coordinate

    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N, use_duogrid=True))
    assert g.base.duogrid is not None
    with pytest.raises(NotImplementedError, match="non-duogrid"):
        make_tiled_dp_s_dt_stage_2d(
            None, g, create_sigma_coordinate(3), n=N, kt=2, nlev=3,
        )
