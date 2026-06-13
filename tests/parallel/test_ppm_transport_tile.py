"""Cube tiled np>6 stage (task #3) — PPM transport tiling tests.

U1: ``_ppm_transport_1d(..., rd_prepadded=True)`` is BIT-IDENTICAL to the
default internal edge-pad when fed the same padded rdelta — the hook that
lets a sub-face TILE supply a REAL depth-1 neighbour-tile rdelta halo at
interior cuts (the internal edge-pad is wrong there: the upwind CFL cell
lives in the neighbour tile).  See docs/scaling/cube_transport_tiling_design.md.

(U2 will add the tile-vs-global PPM flux parity here.)
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_sw_core import _ppm_transport_1d


def _bit_identical_for_axis(axis: int) -> None:
    rng = np.random.default_rng(0)
    eh = 2          # external (cross-face) halo, as d_sw3 supplies
    n, m = 8, 4     # interior cells along sweep axis, trailing dim
    if axis == 1:
        field = jnp.asarray(rng.standard_normal((6, n + 2 * eh, m)))
        courant = jnp.asarray(rng.standard_normal((6, n + 1, m)))
        rdelta = jnp.asarray(np.abs(rng.standard_normal((6, n, m))) + 0.1)
    else:
        field = jnp.asarray(rng.standard_normal((6, m, n + 2 * eh)))
        courant = jnp.asarray(rng.standard_normal((6, m, n + 1)))
        rdelta = jnp.asarray(np.abs(rng.standard_normal((6, m, n))) + 0.1)

    flux_def = _ppm_transport_1d(field, courant, rdelta, axis, external_halo=eh)

    # Replicate the function's internal rd edge-pad (done in its axis-1
    # working frame), then map back to the caller's axis so rd_prepadded
    # receives the identical rd_pad.
    rd_w = rdelta if axis == 1 else jnp.swapaxes(rdelta, 1, 2)
    rd_pad_w = jnp.pad(rd_w, [(0, 0), (1, 1), (0, 0)], mode="edge")
    rd_pad_caller = rd_pad_w if axis == 1 else jnp.swapaxes(rd_pad_w, 1, 2)

    flux_pp = _ppm_transport_1d(
        field, courant, rd_pad_caller, axis, external_halo=eh,
        rd_prepadded=True)

    np.testing.assert_array_equal(
        np.asarray(flux_def), np.asarray(flux_pp),
        err_msg=f"rd_prepadded flux != default (axis={axis})")


def test_rd_prepadded_bit_identical_axis1():
    _bit_identical_for_axis(1)


def test_rd_prepadded_bit_identical_axis2():
    _bit_identical_for_axis(2)


def test_rd_prepadded_wrong_length_raises():
    """codex U1 MEDIUM: a mis-sized prepadded rdelta must fail loudly, not
    silently truncate.  nn=8 ⇒ rd must be length nn+2=10; pass 9."""
    import pytest

    eh, n, m = 2, 8, 4
    field = jnp.zeros((6, n + 2 * eh, m))
    courant = jnp.zeros((6, n + 1, m))
    bad_rd = jnp.ones((6, n + 1, m))   # length nn+1=9, not nn+2=10
    with pytest.raises(ValueError, match="rd_prepadded expects"):
        _ppm_transport_1d(field, courant, bad_rd, 1, external_halo=eh,
                          rd_prepadded=True)


def _global_vs_tiled_ppm(kt: int, nl: int) -> None:
    """U2 approach-C: the global PPM sweep == per-tile (global pre-pad to
    h3=4, strided tile slice, rd_prepadded) reassembled.  Single face
    axis, axis=1, external_halo=0 (the global boundary edge-pads; INTERIOR
    tile cuts pick up REAL contiguous neighbour cells from the pre-padded
    array — the whole point).  Exact parity: each interface flux uses the
    same cells + same arithmetic whether computed globally or per-tile.

    SCOPE (codex U2 LOW): validates the INTERIOR same-face approach-C
    tiling MECHANISM only.  The cross-face D-grid halo
    (``_pad_halo_dgrid_for_ppm``, external_halo=2) + the staggered
    ``u_d``/``v_d`` production path of ``_bgrid_ke_transport`` are a
    SEPARATE sub-build (U2b/U3) — not proven here."""
    h3 = 4
    n, m = kt * nl, 4
    rng = np.random.default_rng(2)
    field = jnp.asarray(rng.standard_normal((6, n, m)))
    courant = jnp.asarray(rng.standard_normal((6, n + 1, m)))
    rd = jnp.asarray(np.abs(rng.standard_normal((6, n, m))) + 0.1)

    global_flux = np.asarray(
        _ppm_transport_1d(field, courant, rd, 1, external_halo=0))

    # Global pre-pad EXACTLY as _ppm_transport_1d(external_halo=0) does
    # internally (field→h3 edge-pad; rd→depth-1 edge-pad).
    vp_g = jnp.pad(field, [(0, 0), (h3, h3), (0, 0)], mode="edge")
    rd_g = jnp.pad(rd, [(0, 0), (1, 1), (0, 0)], mode="edge")

    tiles = []
    for t in range(kt):
        vp_t = vp_g[:, t * nl: t * nl + nl + 2 * h3, :]   # (6, nl+2h3, m)
        rd_t = rd_g[:, t * nl: t * nl + nl + 2, :]         # (6, nl+2, m)
        c_t = courant[:, t * nl: t * nl + nl + 1, :]       # (6, nl+1, m)
        f_t = _ppm_transport_1d(vp_t, c_t, rd_t, 1, external_halo=h3,
                                rd_prepadded=True)
        tiles.append(np.asarray(f_t))                      # (6, nl+1, m)

    # Shared-interface consistency (codex U2 MEDIUM): tile t's last
    # interface [nl] must equal tile t+1's first [0] — else the reassembly
    # (which takes t+1's [0]) could pass while tile t's boundary flux is
    # wrong.  Both compute the SAME global interface from their h3 halos.
    for t in range(kt - 1):
        np.testing.assert_allclose(
            tiles[t][:, nl, :], tiles[t + 1][:, 0, :],
            atol=1e-12, rtol=1e-12,
            err_msg=f"shared PPM tile interface mismatch "
                    f"(kt={kt} nl={nl} t={t})")

    # Tile t produces interfaces [t*nl : t*nl+nl] (inclusive both ends);
    # adjacent tiles SHARE the boundary interface, so take [0:nl] from each
    # + the last tile's final interface → n+1 global interfaces.
    reassembled = np.concatenate(
        [tiles[t][:, :nl, :] for t in range(kt)]
        + [tiles[-1][:, nl:nl + 1, :]], axis=1)            # (6, n+1, m)
    np.testing.assert_allclose(
        reassembled, global_flux, atol=1e-12, rtol=1e-12,
        err_msg=f"approach-C tiled PPM sweep != global (kt={kt} nl={nl})")


def test_ppm_sweep_tiling_approachC_kt2():
    _global_vs_tiled_ppm(kt=2, nl=6)


def test_ppm_sweep_tiling_approachC_kt3():
    _global_vs_tiled_ppm(kt=3, nl=6)


def test_ppm_sweep_tiling_real_crossface_kt3():
    """U2b: approach-C sweep tiling on the REAL cross-face-halo'd D-grid
    field (``_pad_halo_dgrid_for_ppm``, external_halo=2) + staggered cross
    axis (n+1) — exact parity vs the global PPM sweep.

    SCOPE (codex U2b MEDIUM): validates the production-shaped i-sweep
    (``xtp_u``) PPM tiling ONLY — real cross-face halo + staggered M +
    external_halo=2 pre-pad.  The j-sweep (``ytp_v`` axis=2), the REAL
    Courant (ub/vb built from uc/vc), and the B-grid corner sync of the
    full ``_bgrid_ke_transport`` remain U3."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.core.fv3_sw_core import _pad_halo_dgrid_for_ppm
    from tests.test_cases.cosine_bell import cosine_bell_cubesphere

    kt, nl, h3, eh = 3, 6, 4, 2
    n = kt * nl
    grid = create_cubed_sphere(n=n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    state = cosine_bell_cubesphere(grid, cdgrid)

    # Real cross-face D-grid halo for the i-sweep field (xtp_u path).
    u_d_ihalo, _ = _pad_halo_dgrid_for_ppm(
        state.u_d, state.v_d, cdgrid, halo=eh)        # (6, n+2eh, n+1)
    M = u_d_ihalo.shape[2]
    rdx = 1.0 / jnp.maximum(cdgrid.dx_edge_y, 1.0e-30)  # (6, n, n+1)=(6,N,M)
    # Synthetic but sign-varying courant (parity is structural; both flux
    # branches c>0 / c<=0 must be exercised).
    ub = jnp.asarray(
        np.random.default_rng(3).standard_normal((6, n + 1, M)) * 0.1)

    # Non-vacuity guards (codex U2b LOW): the cross-face halo must DIFFER
    # from a naive edge-replication of the first interior cell (else the
    # test couldn't catch a wrong/missing cross-face halo), and the courant
    # must exercise BOTH flux branches (c>0 and c<=0).
    assert float(jnp.max(jnp.abs(
        u_d_ihalo[:, :eh, :] - u_d_ihalo[:, eh:eh + 1, :]))) > 1e-9, \
        "cross-face i-halo == edge-replication (test would be vacuous)"
    assert bool(jnp.any(ub > 0)) and bool(jnp.any(ub <= 0)), \
        "courant must exercise both PPM flux branches"

    global_x = np.asarray(
        _ppm_transport_1d(u_d_ihalo, ub, rdx, 1, external_halo=eh))

    # Pre-pad to h3=4 EXACTLY as _ppm_transport_1d(external_halo=eh) does
    # internally (gap = h3-eh each side), then slice the sweep axis.
    vp_g = jnp.pad(u_d_ihalo, [(0, 0), (h3 - eh, h3 - eh), (0, 0)],
                   mode="edge")                         # (6, n+2h3, M)
    rd_g = jnp.pad(rdx, [(0, 0), (1, 1), (0, 0)], mode="edge")  # (6, n+2, M)
    tiles = []
    for t in range(kt):
        vp_t = vp_g[:, t * nl: t * nl + nl + 2 * h3, :]
        rd_t = rd_g[:, t * nl: t * nl + nl + 2, :]
        c_t = ub[:, t * nl: t * nl + nl + 1, :]
        f_t = _ppm_transport_1d(vp_t, c_t, rd_t, 1, external_halo=h3,
                                rd_prepadded=True)
        tiles.append(np.asarray(f_t))

    for t in range(kt - 1):
        np.testing.assert_allclose(
            tiles[t][:, nl, :], tiles[t + 1][:, 0, :], atol=1e-12,
            rtol=1e-12, err_msg=f"U2b shared interface mismatch t={t}")
    reassembled = np.concatenate(
        [tiles[t][:, :nl, :] for t in range(kt)]
        + [tiles[-1][:, nl:nl + 1, :]], axis=1)
    np.testing.assert_allclose(
        reassembled, global_x, atol=1e-12, rtol=1e-12,
        err_msg="U2b cross-face tiled PPM sweep != global")
