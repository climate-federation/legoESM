"""np24 ASSEMBLY gate: the tiled ``fv3_hydrostatic_tendencies`` D-grid MOMENTUM
stage — the 3D-PE analogue of the SW momentum capstone
(``test_tiled_fv3_sw_momentum.py``) and the genuine 3D >6-device unlock.

The assembly chains the production base momentum path
  dgrid_to_center_vector -> Phi -> B=KE+Phi ; dgrid_vorticity(u_d,v_d) -> zeta
  -> [PACKED SCALAR in-stage halo {zeta, B, 1/T, ln_ps(, hf)}]
  -> interp_center_to_corner(zeta)+f_corner ; arakawa_lamb_gradient(B)
  -> PGF: arakawa_lamb_gradient(ln_ps_hi) + T_corner + (hybrid factor)
  -> du_d_dt = zeta_corner*v_d - dB_dx - pg_corr_x   (dv symmetric)
on a ``(6, kt, kt)`` device mesh.  Unlike the SW momentum (cc-located, vector
halo), the 3D-PE momentum lives at the D-grid CORNERS ``(n+1, n+1)`` and needs
ONLY SCALAR halos on the cc intermediates {zeta, B, 1/T, ln_ps, hf} — exchanged
IN-STAGE via the ndim=4 ``make_tiled_pad_body`` (no global pre-pad; intermediates
have none).

Reference = a HAND-COMPOSED global momentum that mirrors
``primitive_eq_cdgrid.py:307-479`` VERBATIM (base case: div_damp=0, hyperdiff=0,
corner_div_damp=0, KE-heat off, non-duogrid, use_fv3_a2b_zeta_corner=False) —
the same methodology as ``test_tiled_bernoulli_stage.py``.  Each per-op kernel is
already validated against the production op individually; this gate proves the
TILING (slice + in-stage halo) of the composed momentum is bit-identical to the
global composition of the SAME ops.  Covers BOTH sigma and hybrid coords.

Gated by BIT-IDENTITY: each tile's ``(du_d_dt, dv_d_dt)`` corner block vs the
overlapping global staggered slice (the duplicated shared corner face validates
lower-owns-shared), to an FMA-robust relative tolerance.  NOT a wall-clock
measurement (np24 on Ginsburg = CPU shard_map / cross-node ppermute, both
anti-scale here); the future-HW win is the capability, the gate is correctness.

24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  kt=3 (nl=8 != 12) pins
the strip-ppermute DIRECTION (at kt=2 ``perm_hi == perm_lo``) and catches
nl-dependent slice bugs.
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
    compute_geopotential, compute_geopotential_hybrid,
    HybridSigmaPressureCoordinate, pressure_from_hybrid,
)
from legoesm.core.operators_cdgrid import (
    dgrid_to_center_vector, dgrid_vorticity, arakawa_lamb_gradient,
    interp_center_to_corner,
)
from legoesm.core.precision import resolve_dtype
from legoesm import constants
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_fv3_hydrostatic_momentum_stage_2d,
)

N = 24  # divisible by kt=2 (nl=12) and kt=3 (nl=8)
NLEV = 6


def _inputs(n, nlev, seed):
    rng = np.random.default_rng(seed)
    # Random but finite state — bit-identity tests numerical equivalence, not
    # physics.  T well above any T_min floor, p_s in the production clip range,
    # so the production's positivity clamps are no-ops (the stage takes valid
    # state, matching the Bernoulli/vertical stage gates).
    u_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    T = jnp.asarray(250.0 + 20.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    phis = jnp.asarray(1.0e3 * rng.standard_normal((6, n, n)))
    return u_d, v_d, T, p_s, phis


def _global_momentum(u_d, v_d, T, p_s, phis, cdgrid, coord, hybrid):
    """Hand-composed global base momentum = primitive_eq_cdgrid.py:307-479 (base
    case, single-device 'local' halo path)."""
    R_d = constants.R_d
    geo = compute_geopotential_hybrid if hybrid else compute_geopotential
    # 1/3/5: Bernoulli
    u_cell, v_cell = dgrid_to_center_vector(u_d, v_d)
    Phi = geo(T, p_s, coord, phis)
    KE = 0.5 * (u_cell ** 2 + v_cell ** 2)
    B = KE + Phi
    # 6: vorticity + corner
    zeta = dgrid_vorticity(u_d, v_d, cdgrid)
    zeta_corner = interp_center_to_corner(zeta, cdgrid) + cdgrid.f_corner[..., None]
    # 7: Bernoulli gradient
    dB_dx, dB_dy_perp = arakawa_lamb_gradient(B, cdgrid)
    # 8: PGF
    ln_ps = jnp.log(p_s)
    inv_T = 1.0 / T
    _pg_dt = jnp.result_type(
        ln_ps.dtype, resolve_dtype("atm_pressure_gradient", "compute"))
    ln_ps_hi = ln_ps.astype(_pg_dt)
    dln_dx_hi, dln_dy_perp_hi = arakawa_lamb_gradient(ln_ps_hi, cdgrid)
    T_corner = 1.0 / interp_center_to_corner(inv_T, cdgrid)
    T_corner_hi = T_corner.astype(_pg_dt)
    pg_corr_x = (R_d * T_corner_hi * dln_dx_hi[..., None]).astype(u_d.dtype)
    pg_corr_y_perp = (R_d * T_corner_hi * dln_dy_perp_hi[..., None]).astype(v_d.dtype)
    if hybrid:
        p_full = pressure_from_hybrid(coord, p_s)
        hf = coord.B_full * p_s[..., None] / p_full
        hf_corner = interp_center_to_corner(hf, cdgrid)
        pg_corr_x = pg_corr_x * hf_corner
        pg_corr_y_perp = pg_corr_y_perp * hf_corner
    # 9: D-grid momentum tendencies
    du_d_dt = zeta_corner * v_d - dB_dx - pg_corr_x
    dv_d_dt = -zeta_corner * u_d - dB_dy_perp - pg_corr_y_perp
    return np.asarray(du_d_dt), np.asarray(dv_d_dt)


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert g.base.duogrid is None, "base cut is non-duogrid"
    assert g.base.halo_interp_offsets is not None, "needs interp offsets"
    return g


def _compare_corner_staggered(du_t, dv_t, du_g, dv_g, kt, nl):
    """Per-tile overlap vs the global corner-staggered slice (both axes
    staggered (nl+1, nl+1)); the duplicated shared face validates
    lower-owns-shared.  Returns (rel_du, rel_dv)."""
    blk = nl + 1
    worst_du = worst_dv = 0.0
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                g_du = du_g[f, ti * nl: ti * nl + blk, tj * nl: tj * nl + blk]
                t_du = du_t[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                worst_du = max(worst_du, float(np.max(np.abs(t_du - g_du))))
                g_dv = dv_g[f, ti * nl: ti * nl + blk, tj * nl: tj * nl + blk]
                t_dv = dv_t[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                worst_dv = max(worst_dv, float(np.max(np.abs(t_dv - g_dv))))
    scale_du = float(np.max(np.abs(du_g))) + 1e-300
    scale_dv = float(np.max(np.abs(dv_g))) + 1e-300
    return worst_du / scale_du, worst_dv / scale_dv


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_momentum_matches_global(cdg, KT, hybrid):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    nl = N // KT
    coord = make_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    u_d, v_d, T, p_s, phis = _inputs(N, NLEV, (10 if hybrid else 20) + KT)
    du_g, dv_g = _global_momentum(u_d, v_d, T, p_s, phis, cdg, coord, hybrid)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fv3_hydrostatic_momentum_stage_2d(
        mesh, cdg, coord, N, KT, NLEV)

    fw = NamedSharding(mesh, P("face", None, None, None))
    fo = NamedSharding(mesh, P("face", None, None))
    du_t, dv_t = stage(
        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
        jax.device_put(T, fw), jax.device_put(p_s, fo),
        jax.device_put(phis, fo))

    def _re(arr):
        # gathered (6, KT*(nl+1), KT*(nl+1), nlev): keep per-tile blocks intact
        # (corner-staggered -> overlap comparison, NOT a cc reshape).
        return np.asarray(arr)

    rel_du, rel_dv = _compare_corner_staggered(
        _re(du_t), _re(dv_t), du_g, dv_g, KT, nl)
    # FMA-robust relative tolerance: the in-stage ppermute receives reorder the
    # global pad's contiguous arithmetic -> O(1e-13) ULP drift, not algorithmic.
    assert rel_du < 1e-10, f"du_d_dt rel {rel_du:.3e} (kt={KT}, hybrid={hybrid})"
    assert rel_dv < 1e-10, f"dv_d_dt rel {rel_dv:.3e} (kt={KT}, hybrid={hybrid})"


@pytest.mark.parametrize("hybrid", [False, True])
def test_momentum_compose_host_body(cdg, hybrid):
    """Composition exactness WITHOUT 24 devices: the same slice+per-op chain on
    host (interp_center_to_corner / arakawa_lamb_gradient on the GLOBAL pre-pad
    sliced per tile) reassembled == global.  Isolates the slice arithmetic from
    the in-stage ppermute halo (which the np24 lane exercises)."""
    from legoesm.core.operators_cdgrid import pad_halo_auto
    kt, nl = 3, N // 3
    coord = make_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    u_d, v_d, T, p_s, phis = _inputs(N, NLEV, 77 if hybrid else 88)
    du_g, dv_g = _global_momentum(u_d, v_d, T, p_s, phis, cdg, coord, hybrid)
    R_d = constants.R_d
    geo = compute_geopotential_hybrid if hybrid else compute_geopotential

    # GLOBAL pre-pads (halo=1) of the cc intermediates — the host-body analogue
    # of the in-stage scalar halo (proves the SLICE windows; the np24 lane
    # proves the ppermute halo equals these pads).
    u_cell, v_cell = dgrid_to_center_vector(u_d, v_d)
    Phi = geo(T, p_s, coord, phis)
    B = 0.5 * (u_cell ** 2 + v_cell ** 2) + Phi
    zeta = dgrid_vorticity(u_d, v_d, cdg)
    ln_ps = jnp.log(p_s)
    inv_T = 1.0 / T
    _pg_dt = jnp.result_type(
        ln_ps.dtype, resolve_dtype("atm_pressure_gradient", "compute"))
    ln_ps_3d = ln_ps.astype(_pg_dt)[..., None]

    def _pad(field):  # (6, n, n[, C]) -> (6, n+2, n+2[, C]) face-replicated
        # pad_halo_auto dispatches by ndim (4D -> pad_halo_4d) — the SAME pad the
        # production ops use internally, so the host-body matches the global ref.
        return np.asarray(pad_halo_auto(jnp.asarray(field), cdg))

    B_pad = _pad(B)
    zeta_pad = _pad(zeta)
    invT_pad = _pad(inv_T)
    lnps_pad = _pad(ln_ps_3d)
    hf_pad = _pad(coord.B_full * p_s[..., None] / pressure_from_hybrid(coord, p_s)) \
        if hybrid else None

    def _chain(ti, tj, which):
        a_i, a_j = ti * nl, tj * nl
        # corner outputs (nl+1): from the (nl+2) padded window
        Bp = jnp.asarray(B_pad[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2])
        zp = jnp.asarray(zeta_pad[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2])
        ip = jnp.asarray(invT_pad[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2])
        lp = jnp.asarray(lnps_pad[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2])
        ud = u_d[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1]
        vd = v_d[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1]
        zc = (interp_center_to_corner(zp, cdg, padded=zp)
              + cdg.f_corner[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1][..., None])
        from legoesm.core.operators_cdgrid import arakawa_lamb_gradient_core
        gc = (cdg.grad_c00[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1],
              cdg.grad_c01[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1],
              cdg.grad_c10[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1],
              cdg.grad_c11[:, a_i:a_i + nl + 1, a_j:a_j + nl + 1])
        dB_dx, dB_dy_perp = arakawa_lamb_gradient_core(Bp, *gc)
        dln_dx, dln_dy = arakawa_lamb_gradient_core(lp, *gc)
        T_corner = 1.0 / interp_center_to_corner(ip, cdg, padded=ip)
        T_corner_hi = T_corner.astype(_pg_dt)
        pgx = (R_d * T_corner_hi * dln_dx).astype(u_d.dtype)
        pgy = (R_d * T_corner_hi * dln_dy).astype(v_d.dtype)
        if hybrid:
            hp = jnp.asarray(hf_pad[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2])
            hfc = interp_center_to_corner(hp, cdg, padded=hp)
            pgx = pgx * hfc
            pgy = pgy * hfc
        if which == "du":
            return zc * vd - dB_dx - pgx
        return -zc * ud - dB_dy_perp - pgy

    blk = nl + 1
    for which, g in (("du", du_g), ("dv", dv_g)):
        worst = 0.0
        for f in range(6):
            for ti in range(kt):
                for tj in range(kt):
                    t = np.asarray(_chain(ti, tj, which))
                    gg = g[f, ti * nl: ti * nl + blk, tj * nl: tj * nl + blk]
                    worst = max(worst, float(np.max(np.abs(t[f] - gg))))
        rel = worst / (float(np.max(np.abs(g))) + 1e-300)
        assert rel < 1e-10, f"host-body {which} rel {rel:.3e} (hybrid={hybrid})"
