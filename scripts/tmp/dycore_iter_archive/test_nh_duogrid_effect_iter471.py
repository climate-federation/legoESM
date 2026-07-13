"""FV3_3D iter 471: isolate the duogrid effect on NH θ′ edge
ratio — duogrid on vs off, same factory config.

Background:
* iter-464 (no duogrid): heat_source_del2 REDUCED ratio
  (1.10 → 0.96).
* iter-465 (with duogrid): heat_source_del2 INCREASED ratio
  (factory 4.37; iters=0 drops to 2.95).

Hypothesis: duogrid itself is the culprit for the 4× higher
baseline ratio, not any factory flag.  This test isolates
the duogrid effect by running the SAME factory at C8 with
duogrid=True vs duogrid=False.

Tests
-----

1. ``test_nh_duogrid_on_vs_off_edge_ratio``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _build_nh_state(n, seed, use_duogrid):
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=use_duogrid)
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


def test_nh_duogrid_on_vs_off_edge_ratio(capsys):
    """Run SAME factory at C8 with use_duogrid=True vs =False.
    iter-465 baseline (duogrid=True) ratio ≈ 4.37; if turning
    duogrid OFF drops it to ~1, duogrid IS the culprit."""
    seeds = [471, 472, 473]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    ratios_dg_on = []
    ratios_dg_off = []
    # IMPORTANT: duogrid=False forces use_fv3_cross_face_du_proj
    # no-op (per iter-384 finding).  Use same factory but the
    # FLAG `use_fv3_cross_face_du_proj` is still True — has no
    # effect without grid duogrid.
    for seed in seeds:
        grid_on, hc, tm, state_on = _build_nh_state(8, seed, True)
        grid_off, _, _, state_off = _build_nh_state(8, seed, False)
        cfg = make_fv3_faithful_nh_config(**base_kw)
        m_on = CDGridCompressibleEulerModel(grid_on, hc, tm, cfg)
        m_off = CDGridCompressibleEulerModel(grid_off, hc, tm, cfg)
        s_on, s_off = state_on, state_off
        for _ in range(3):
            s_on = m_on.step(s_on, dt=10.0)
            s_off = m_off.step(s_off, dt=10.0)
        ratios_dg_on.append(
            _edge_interior_ratio(s_on.theta_prime.data)
        )
        ratios_dg_off.append(
            _edge_interior_ratio(s_off.theta_prime.data)
        )
    mean_on = float(np.mean(ratios_dg_on))
    mean_off = float(np.mean(ratios_dg_off))
    with capsys.disabled():
        print(
            f"\n[iter-471 NH duogrid effect, 3 seeds @ C8, 3 steps]"
        )
        print(f"  factory + duogrid=ON  θ′ ratio: {mean_on:.4f}")
        print(f"  factory + duogrid=OFF θ′ ratio: {mean_off:.4f}")
        print(f"  ratio: ON/OFF = {mean_on / mean_off:.4f}")
        if mean_on > 1.5 * mean_off:
            print(
                f"  Conclusion: duogrid INCREASES the metric "
                f"≥1.5× → duogrid IS the regime-changing factor."
            )
        elif mean_on < 0.67 * mean_off:
            print(f"  Conclusion: duogrid DECREASES the metric.")
        else:
            print(f"  Conclusion: duogrid effect is modest.")
    assert np.isfinite(mean_on) and np.isfinite(mean_off)
