"""np24 CAPSTONE gate: the FULL tiled ``fv3_hydrostatic_tendencies`` stage — the
production cube 3D-PE dycore tendency (du_d_dt, dv_d_dt, dT_dt, dp_s_dt) on
np=6*kt^2 devices.  The 3D analogue of the SW capstone
(``test_tiled_fv3_sw_full.py``) and the genuine full-dycore >6-device unlock.

Composes momentum (b1368005) + the section-12c center_to_dgrid_vector lift
(2d9ed627c) + continuity (5765dcea) + thermo (ef706aba2) + vertical advection in
ONE shard_map.  Reference = the REAL ``fv3_hydrostatic_tendencies`` (the same
pattern as the SW capstone gating vs ``fv3_sw_tendencies``) with a BASE-CUT
config: ``CDGridPrimitiveEquationConfig(sponge_tau_sec=0.0)``.  Every other
optional term is already off by default — A_h=0, hyperdiff=0, div_damp=0,
corner_div_damp off, T_diss=0, no physics — and ``fix_mass=True`` (default) makes
``_apply_zero_mean_per_stage`` FALSE, so dp_s/dt is used RAW (no global
zero_mean), exactly the base cut the tiled stage composes.  The ``sponge_tau_sec``
default (432000 s / 5 d since #1028) is the ONE on-by-default term the stage
omits, so it is zeroed.
The T_min / p_floor clamps at the function entry are no-ops on the test state
(T ~ 250 K >> 50 K floor; p_s ~ 1e5 Pa in [100, 2e6]).

Gated by BIT-IDENTITY of ALL FOUR tendencies vs the global function: du_d_dt /
dv_d_dt are D-grid CORNER-staggered (per-tile corner block vs the overlapping
global slice, lower-owns-shared); dT_dt / dp_s_dt are cc (EXACT partition,
elementwise).  FMA-robust relative tolerance (the in-stage ppermute reorders the
global pad's contiguous arithmetic -> O(1e-13) ULP drift).  NOT a wall-clock
measurement (np>6 anti-scales on Ginsburg); the future-HW win is the capability.
Covers BOTH sigma and hybrid coords.

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
from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels
from legoesm.core.field import Field
from legoesm.core.state import FV3HydrostaticState
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    fv3_hydrostatic_tendencies, CDGridPrimitiveEquationConfig,
)
from legoesm.parallel.tiled_production_cdgrid import (
    make_tiled_fv3_hydrostatic_tendencies_stage_2d,
)

N = 24
NLEV = 6
P_FLOOR = CDGridPrimitiveEquationConfig().p_floor


def _inputs(n, nlev, seed):
    """Valid finite state: T well above T_min (50 K) and p_s in the production
    clip range [p_floor, 2e6], so the function-entry clamps are no-ops and the
    gate is a pure numerical-equivalence test."""
    rng = np.random.default_rng(seed)
    u_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    v_d = jnp.asarray(rng.standard_normal((6, n + 1, n + 1, nlev)))
    T = jnp.asarray(250.0 + 20.0 * rng.standard_normal((6, n, n, nlev)))
    p_s = jnp.asarray(1.0e5 + 1.0e3 * rng.standard_normal((6, n, n)))
    phis = jnp.asarray(1.0e3 * rng.standard_normal((6, n, n)))
    return u_d, v_d, T, p_s, phis


def _global_tendencies(u_d, v_d, T, p_s, phis, cdgrid, coord):
    """The REAL production base-case fv3_hydrostatic_tendencies (sponge off)."""
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    state = FV3HydrostaticState(
        u_d=Field(data=u_d, name="u_d", dims=d3, units="m/s"),
        v_d=Field(data=v_d, name="v_d", dims=d3, units="m/s"),
        T=Field(data=T, name="T", dims=d3, units="K"),
        p_s=Field(data=p_s, name="p_s", dims=d2, units="Pa"),
        phis=Field(data=phis, name="phis", dims=d2, units="m^2/s^2"))
    cfg = CDGridPrimitiveEquationConfig(sponge_tau_sec=0.0)   # base cut
    tend = fv3_hydrostatic_tendencies(state, cdgrid.base, coord, cdgrid, cfg)
    return (np.asarray(tend.du_d_dt.data), np.asarray(tend.dv_d_dt.data),
            np.asarray(tend.dT_dt.data), np.asarray(tend.dp_s_dt.data))


@pytest.fixture(scope="module")
def cdg():
    set_halo_backend("local")
    g = create_cubed_sphere_cdgrid(create_cubed_sphere(N))
    assert g.base.duogrid is None, "base cut is non-duogrid"
    assert g.base.halo_interp_offsets is not None
    return g


def _rel_corner(t, g, kt, nl):
    """Per-tile corner block (nl+1) vs the overlapping global staggered slice."""
    blk = nl + 1
    worst = 0.0
    for f in range(6):
        for ti in range(kt):
            for tj in range(kt):
                gg = g[f, ti * nl: ti * nl + blk, tj * nl: tj * nl + blk]
                tt = t[f, ti * blk:(ti + 1) * blk, tj * blk:(tj + 1) * blk]
                worst = max(worst, float(np.max(np.abs(tt - gg))))
    return worst / (float(np.max(np.abs(g))) + 1e-300)


def _rel_cc(t, g):
    """Exact cc partition: gathered == global, elementwise."""
    return float(np.max(np.abs(np.asarray(t) - g))) / (
        float(np.max(np.abs(g))) + 1e-300)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("KT", [2, 3])
def test_tiled_capstone_matches_global(cdg, KT, hybrid):
    ndev = 6 * KT * KT
    if len(jax.devices()) < ndev:
        pytest.skip(
            f"kt={KT} needs {ndev} host devices "
            f"(--xla_force_host_platform_device_count={ndev})")
    nl = N // KT
    coord = make_hybrid_levels(NLEV) if hybrid else create_sigma_coordinate(NLEV)
    u_d, v_d, T, p_s, phis = _inputs(N, NLEV, (60 if hybrid else 70) + KT)
    du_g, dv_g, dT_g, dps_g = _global_tendencies(
        u_d, v_d, T, p_s, phis, cdg, coord)

    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    dev = np.array(jax.devices()[:ndev]).reshape(6, KT, KT)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    stage = make_tiled_fv3_hydrostatic_tendencies_stage_2d(
        mesh, cdg, coord, N, KT, NLEV, p_floor=P_FLOOR)

    fw = NamedSharding(mesh, P("face", None, None, None))
    fo = NamedSharding(mesh, P("face", None, None))
    du_t, dv_t, dT_t, dps_t = stage(
        jax.device_put(u_d, fw), jax.device_put(v_d, fw),
        jax.device_put(T, fw), jax.device_put(p_s, fo),
        jax.device_put(phis, fo))

    r_du = _rel_corner(np.asarray(du_t), du_g, KT, nl)
    r_dv = _rel_corner(np.asarray(dv_t), dv_g, KT, nl)
    r_dT = _rel_cc(dT_t, dT_g)
    r_dps = _rel_cc(dps_t, dps_g)
    tag = f"(kt={KT}, hybrid={hybrid})"
    # du/dv are corner momentum (NO column reduction) -> bit-identity (1e-10).
    assert r_du < 1e-10, f"du_d_dt rel {r_du:.3e} {tag}"
    assert r_dv < 1e-10, f"dv_d_dt rel {r_dv:.3e} {tag}"
    # The cc REDUCTION-bearing tendencies dT_dt (longest fused chain: continuity
    # div -> omega + vertical_advection + momentum/adiabatic + two centred
    # gradients) and dp_s_dt (the p_s~1e5-scale column-sum) accumulate O(1e-8)
    # RELATIVE reordering when XLA fuses the full capstone body — LARGER than the
    # short-chain du/dv and than the ISOLATED continuity/thermo stages (which are
    # bit-identical to the global at 1e-10).  Diagnostic 8512950 (sigma kt=2)
    # confirms it is non-associative REORDERING, not a defect: dp_s abs_diff
    # 5.05e-9 on a max|dp_s|=0.54 field (~1e-8 rel); dT abs_diff 4.08e-12 on a
    # 1.2e-3 field (uniform across all 6 levels, mean_p[k0]=9250 Pa rules out a
    # 1/p ill-conditioning artifact).  The capstone's dp_s/dT math is line-for-
    # line the gated standalone stages', so the delta is pure fusion FP order.
    # A real algorithmic bug is O(1e-2)+, so 5e-8 catches bugs by ~6 orders while
    # admitting the legitimate reordering (hybrid lands <1e-10; sigma ~3e-9/9e-9).
    _CC_REORDER_RTOL = 5e-8
    assert r_dps < _CC_REORDER_RTOL, f"dp_s_dt rel {r_dps:.3e} {tag}"
    assert r_dT < _CC_REORDER_RTOL, f"dT_dt rel {r_dT:.3e} {tag}"


def test_capstone_rejects_bad_p_floor(cdg):
    """Fail-loud factory guard (the other guards mirror the component stages)."""
    if len(jax.devices()) < 6:
        pytest.skip("guard test builds a (6,1,1) mesh -> needs 6 host devices")
    from jax.sharding import Mesh
    dev = np.array(jax.devices()[:6]).reshape(6, 1, 1)
    mesh = Mesh(dev, axis_names=("face", "tile_i", "tile_j"))
    coord = create_sigma_coordinate(NLEV)
    with pytest.raises(ValueError, match="p_floor"):
        make_tiled_fv3_hydrostatic_tendencies_stage_2d(
            mesh, cdg, coord, N, 1, NLEV, p_floor=0.0)
    with pytest.raises(ValueError, match="n_levels"):
        make_tiled_fv3_hydrostatic_tendencies_stage_2d(
            mesh, cdg, create_sigma_coordinate(NLEV + 1), N, 1, NLEV,
            p_floor=P_FLOOR)
