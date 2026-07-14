"""FV3_3D iter 703: verify iter-698 θ′ edge-ratio reduction at C24.

iter-698 (C8) and iter-699 (C16) measured −25.8 % and −21.9 %
respectively from ``use_fv3_a2b_ord4_vector_uv=True``.  iter-703
extends to C24, the lower end of the documented 5-6 mK floor
regime (C24-C32).  If the reduction holds at C24, the iter-698
factory promotion is unambiguously validated for the floor
regime.

Methodology: 2 seeds × 3 dycore steps (smaller seed count than
C8/C16 to bound wall time at C24 — ~12 min budget vs ~2 min
for C8).

Tests
-----

1. ``test_a2b_ord4_vector_uv_edge_ratio_c24``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerModel,
    make_fv3_faithful_nh_config,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate,
    compute_terrain_metric,
)


def _edge_interior_ratio(field_data: jnp.ndarray) -> float:
    n_face, n_x, n_y, n_lev = field_data.shape
    edge_mask = np.zeros((n_x, n_y), dtype=bool)
    edge_mask[0, :] = True
    edge_mask[-1, :] = True
    edge_mask[:, 0] = True
    edge_mask[:, -1] = True
    edge_mask_b = np.broadcast_to(
        edge_mask[None, :, :, None], field_data.shape,
    )
    interior_mask = ~edge_mask_b
    arr = np.asarray(field_data)
    if arr[interior_mask].std() == 0.0:
        return 0.0
    return float(arr[edge_mask_b].std() / arr[interior_mask].std())


def _build_nh_state(n, seed):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=seed)
    u_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-3.0, 3.0, size=(6, n, n, nlev))
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, hc, tm, state


def test_a2b_ord4_vector_uv_edge_ratio_c24(capsys):
    """θ′ edge ratio @ C24 with iter-696 flag ON vs OFF (2 seeds × 3 steps).

    C24 is the lower end of the documented 5-6 mK θ′ floor regime
    (C24-C32).  This iter checks whether the iter-698 reduction
    holds at the resolution that actually exhibits the floor.
    """
    seeds = [703, 704]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    variants = {
        "OFF": dict(use_fv3_a2b_ord4_vector_uv=False),
        "ON ": dict(use_fv3_a2b_ord4_vector_uv=True),
    }
    results: dict[str, list[float]] = {k: [] for k in variants}
    for seed in seeds:
        for name, ov in variants.items():
            grid, hc, tm, state = _build_nh_state(24, seed=seed)
            cfg = make_fv3_faithful_nh_config(**{**base_kw, **ov})
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            s = state
            for _ in range(3):
                s = m.step(s, dt=5.0)  # smaller dt at C24 for CFL
            results[name].append(
                _edge_interior_ratio(s.theta_prime.data)
            )
    off_m = float(np.mean(results["OFF"]))
    on_m = float(np.mean(results["ON "]))
    delta = on_m - off_m
    pct = 100.0 * delta / off_m if off_m != 0.0 else 0.0
    with capsys.disabled():
        print(
            f"\n[iter-703 a2b_ord4_vector_uv θ′ edge-ratio @ C24, "
            f"{len(seeds)} seeds × 3 steps]"
        )
        print(f"  OFF: mean = {off_m:.4f}  per-seed = "
              f"{[f'{r:.4f}' for r in results['OFF']]}")
        print(f"  ON : mean = {on_m:.4f}  per-seed = "
              f"{[f'{r:.4f}' for r in results['ON ']]}")
        sign = "+" if delta >= 0 else ""
        print(f"  delta (ON − OFF) = {sign}{delta:.4f}  ({sign}{pct:.1f}%)")
        print(
            "  Scaling so far: C8 −25.8% (iter-698), C16 −21.9% "
            "(iter-699), C24 = this iter."
        )
    assert np.all(np.isfinite(results["OFF"]))
    assert np.all(np.isfinite(results["ON "]))
