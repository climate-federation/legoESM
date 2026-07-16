"""np24 ASSEMBLY gate: the tiled ``fv3_hydrostatic_tendencies`` THERMODYNAMIC
stage — the temperature tendency dT/dt (step 11 of ``primitive_eq_cdgrid.py``),
the 3D-PE analogue of the SW/continuity capstones and part of the >6-device
hydrostatic unlock.

The stage chains the production base thermodynamic path
  dgrid_to_center_vector -> u_cell/v_cell ; p_full=pressure_from_*(coord,p_s)
  -> p_adiab=max(p_full,p_floor)
  -> [in-stage SCALAR halo {T, ln_ps_3d}]
  -> dT/dx,dT/dy (gradient_{x,y}_3d_core) -> horiz_adv_T = -(u.grad T)
  -> dln_ps/dx,dln_ps/dy -> adiabatic = kappa*T*omega/p_adiab
                                       + kappa*T*(v.grad ln p_s)[*hybrid factor]
  -> dT/dt = horiz_adv_T + vert_adv_T + adiabatic
on a ``(6, kt, kt)`` device mesh.  ``omega`` and ``vert_adv_T`` are UPSTREAM
stage outputs (omega carries the GLOBAL zero_mean_tendency baked into dp_s/dt;
both per-column-local) so they are random finite inputs here — the gate tests
numerical equivalence, not physics.

Reference = a HAND-COMPOSED global dT/dt mirroring ``primitive_eq_cdgrid.py:
856-871`` VERBATIM (base case: A_h=0, hyperdiff_coeff=0, T_diss_coeff=0,
non-duogrid, no physics tendency) on the single-device 'local' halo path.  The
output is cc, so the gathered tiled result is the EXACT cc partition (no shared
face) — compared elementwise to the global dT/dt.  Covers BOTH sigma and hybrid
coords.

Gated by BIT-IDENTITY (FMA-robust relative tolerance; the in-stage ppermute
reorders the global pad's contiguous arithmetic -> O(1e-13) ULP drift, not
algorithmic).  NOT a wall-clock measurement (np24 on Ginsburg = CPU shard_map /
cross-node ppermute, both anti-scale here); the future-HW win is the capability.

24 host CPU devices (kt=2) / 54 (kt=3):
``XLA_FLAGS=--xla_force_host_platform_device_count=54``.  kt=3 (nl=8 != 12) pins
the strip-ppermute DIRECTION (at kt=2 ``perm_hi == perm_lo``) and nl-dependent
slice bugs.
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
    pressure_from_hybrid, pressure_from_sigma,
)
from legoesm.core.operators_cdgrid import dgrid_to_center_vector
from legoesm.core.operators_3d import (
    gradient_x_3d, gradient_y_3d, gradient_x_3d_core, gradient_y_3d_core,
)
from legoesm import constants
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_fv3_hydrostatic_thermo_stage_2d,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationConfig,
)

N = 24                                              # nl=12 (kt=2), nl=8 (kt=3)
NLEV = 6
P_FLOOR = CDGridPrimitiveEquationConfig().p_floor   # 100.0 Pa — config, no literal


def _inputs(n, nlev, seed):
    """Random but finite state — the gate tests numerical equivalence, not
    physics (T well above any floor, p_s in the production clip range, so the
    production clamps are no-ops).  omega/vert_adv_T are random finite cc fields
    (in the model they are upstream stage outputs)."""
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    T = jnp.asarray(250.0 + 20.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    omega = jnp.asarray(rng.standard_normal((6, n, n, nlev)))        # Pa/s
    vert_adv_T = jnp.asarray(rng.standard_normal((6, n, n, nlev)))   # K/s
    return u_d, v_d, T, p_s, omega, vert_adv_T


def _global_thermo(u_d, v_d, T, p_s, omega, vert_adv_T, cdgrid, coord, hybrid):
    """Hand-composed global base dT/dt = primitive_eq_cdgrid.py:856-871 (base
    case, single-device 'local' halo path)."""
    kappa = constants.kappa
    grid = cdgrid.base
    u_cell, v_cell = dgrid_to_center_vector(u_d, v_d)
    if hybrid:
        p_full = pressure_from_hybrid(coord, p_s)
    else:
        p_full = pressure_from_sigma(coord.sigma_full, p_s)
    p_adiab = jnp.maximum(p_full, P_FLOOR)
    ln_ps_3d = jnp.log(p_s)[..., None]
    dT_dx = gradient_x_3d(T, grid)
    dT_dy = gradient_y_3d(T, grid)
    horiz_adv_T = -(u_cell * dT_dx + v_cell * dT_dy)
    dln_ps_dx = gradient_x_3d(ln_ps_3d, grid)[..., 0]
    dln_ps_dy = gradient_y_3d(ln_ps_3d, grid)[..., 0]
    adiabatic = kappa * T * omega / p_adiab
    v_dot_grad_lnps = (u_cell * dln_ps_dx[..., None]
                       + v_cell * dln_ps_dy[..., None])
    if hybrid:
        v_dot_grad_lnps = v_dot_grad_lnps * (
            coord.B_full * p_s[..., None] / p_adiab)
    adiabatic = adiabatic + kappa * T * v_dot_grad_lnps
    return np.asarray(horiz_adv_T + vert_adv_T + adiabatic)


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert g.base.duogrid is None, "base cut is non-duogrid"
    assert g.base.halo_interp_offsets is not None, "needs interp offsets"
    return g


def _rel_cc(dT_t, dT_g):
    """Exact cc partition: gathered (6, n, n, nlev) elementwise vs global."""
    diff = float(np.max(np.abs(np.asarray(dT_t) - dT_g)))
    return diff / (float(np.max(np.abs(dT_g))) + 1e-300)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_thermo_matches_global(cdg, KT, hybrid):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    coord = make_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    u_d, v_d, T, p_s, omega, vert_adv_T = _inputs(
        N, NLEV, (30 if hybrid else 40) + KT)
    dT_g = _global_thermo(u_d, v_d, T, p_s, omega, vert_adv_T, cdg, coord, hybrid)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fv3_hydrostatic_thermo_stage_2d(
        mesh, cdg, coord, N, KT, NLEV, p_floor=P_FLOOR)

    fw = NamedSharding(mesh, P("face", None, None, None))
    fo = NamedSharding(mesh, P("face", None, None))
    dT_t = stage(
        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
        jax.device_put(T, fw), jax.device_put(p_s, fo),
        jax.device_put(omega, fw), jax.device_put(vert_adv_T, fw))

    rel = _rel_cc(dT_t, dT_g)
    assert rel < 1e-10, f"dT_dt rel {rel:.3e} (kt={KT}, hybrid={hybrid})"


@pytest.mark.parametrize("hybrid", [False, True])
def test_thermo_compose_host_body(cdg, hybrid):
    """Composition exactness WITHOUT 24 devices: the same slice + centred-gradient
    chain on host (gradient_{x,y}_3d_core on the GLOBAL pre-pad sliced per tile)
    reassembled == global.  Isolates the slice arithmetic from the in-stage
    ppermute halo (which the np24 lane exercises)."""
    from legoesm.core.operators_cdgrid import pad_halo_auto
    kt, nl = 3, N // 3
    coord = make_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    u_d, v_d, T, p_s, omega, vert_adv_T = _inputs(N, NLEV, 71 if hybrid else 82)
    dT_g = _global_thermo(u_d, v_d, T, p_s, omega, vert_adv_T, cdg, coord, hybrid)
    kappa = constants.kappa

    u_cell_g, v_cell_g = dgrid_to_center_vector(u_d, v_d)
    if hybrid:
        p_full = pressure_from_hybrid(coord, p_s)
    else:
        p_full = pressure_from_sigma(coord.sigma_full, p_s)
    p_adiab_g = jnp.maximum(p_full, P_FLOOR)
    ln_ps_3d = jnp.log(p_s)[..., None]
    # GLOBAL pre-pads (halo=1) — the host-body analogue of the in-stage scalar
    # halo.  pad_halo_auto dispatches by ndim (4D -> pad_halo_4d) — the SAME pad
    # gradient_{x,y}_3d use internally, so the host-body matches the global ref.
    T_pad_g = np.asarray(pad_halo_auto(T, cdg))
    lnps_pad_g = np.asarray(pad_halo_auto(ln_ps_3d, cdg))
    dx_g, dy_g = cdg.base.dx, cdg.base.dy

    worst = 0.0
    for ti in range(kt):                       # all 6 faces carried in axis 0
        for tj in range(kt):
            a_i, a_j = ti * nl, tj * nl
            Tp = jnp.asarray(T_pad_g[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2])
            lp = jnp.asarray(lnps_pad_g[:, a_i:a_i + nl + 2, a_j:a_j + nl + 2])
            dxt = dx_g[:, a_i:a_i + nl, a_j:a_j + nl]
            dyt = dy_g[:, a_i:a_i + nl, a_j:a_j + nl]
            uc = u_cell_g[:, a_i:a_i + nl, a_j:a_j + nl]
            vc = v_cell_g[:, a_i:a_i + nl, a_j:a_j + nl]
            Tt = T[:, a_i:a_i + nl, a_j:a_j + nl]
            pa = p_adiab_g[:, a_i:a_i + nl, a_j:a_j + nl]
            om = omega[:, a_i:a_i + nl, a_j:a_j + nl]
            vadv = vert_adv_T[:, a_i:a_i + nl, a_j:a_j + nl]
            dT_dx = gradient_x_3d_core(Tp, dxt)
            dT_dy = gradient_y_3d_core(Tp, dyt)
            horiz = -(uc * dT_dx + vc * dT_dy)
            dlnx = gradient_x_3d_core(lp, dxt)[..., 0]
            dlny = gradient_y_3d_core(lp, dyt)[..., 0]
            adi = kappa * Tt * om / pa
            vdot = uc * dlnx[..., None] + vc * dlny[..., None]
            if hybrid:
                ps_t = p_s[:, a_i:a_i + nl, a_j:a_j + nl][..., None]
                vdot = vdot * (coord.B_full * ps_t / pa)
            adi = adi + kappa * Tt * vdot
            t = np.asarray(horiz + vadv + adi)
            gg = dT_g[:, a_i:a_i + nl, a_j:a_j + nl]
            worst = max(worst, float(np.max(np.abs(t - gg))))
    rel = worst / (float(np.max(np.abs(dT_g))) + 1e-300)
    assert rel < 1e-10, f"host-body dT_dt rel {rel:.3e} (hybrid={hybrid})"


def test_thermo_stage_rejects_bad_coord_and_p_floor(cdg):
    """Fail-loud guards: a coord whose n_levels != nlev (silent broadcast) and a
    non-positive p_floor must raise at factory time."""
    if len(jax.devices()) < 6:
        pytest.skip("guard test builds a (6,1,1) mesh -> needs 6 host devices")
    from jax.sharding import Mesh
    dev = np.array(jax.devices()[:6]).reshape(6, 1, 1)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    coord = create_sigma_coordinate(NLEV)
    with pytest.raises(ValueError, match="n_levels"):
        make_tiled_fv3_hydrostatic_thermo_stage_2d(
            mesh, cdg, create_sigma_coordinate(NLEV + 1), N, 1, NLEV,
            p_floor=P_FLOOR)
    with pytest.raises(ValueError, match="p_floor"):
        make_tiled_fv3_hydrostatic_thermo_stage_2d(
            mesh, cdg, coord, N, 1, NLEV, p_floor=0.0)
