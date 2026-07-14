"""FV3_3D iter 465: per-flag NH edge-ratio sweep to identify
which FV3-fidelity flags actively affect cube-edge spatial
structure.

Per iter-464 finding, ``heat_source_del2_iters`` (iter-457)
reduces NH θ′ edge ratio.  Per iter-463 insight, per-level
scalar scaling (RF, sponge_damp_*) cannot move the metric.
This test isolates which OTHER spatial FV3 flags affect the
metric by toggling each one individually OFF from factory
defaults.

Methodology:
* Build NH state at C8.
* Baseline: factory defaults (all FV3 flags ON).
* Per-flag sweep: factory with that ONE flag turned OFF.
* Report θ′ edge ratio difference for each flag.

Tests
-----

1. ``test_nh_per_flag_edge_ratio_sweep``.
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


def test_nh_per_flag_edge_ratio_sweep(capsys):
    """Toggle each FV3-fidelity flag OFF individually and
    measure NH θ′ edge ratio.  Reports baseline (factory) +
    delta for each flag."""
    seeds = [465, 466, 467]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    overrides_to_test = {
        "baseline (factory)": {},
        "no metric_aware_d_con": dict(use_fv3_metric_aware_d_con=False),
        "no cross_face_du_proj": dict(use_fv3_cross_face_du_proj=False),
        "no vector_halo_uv": dict(use_fv3_vector_halo_uv=False),
        "no d_con_cv (use cp)": dict(use_fv3_d_con_cv=False),
        "no dynamic_exner": dict(use_fv3_dynamic_exner=False),
        "no heat_source_del2": dict(heat_source_del2_iters=0),
        "no d_con_top_zero": dict(d_con_top_zero_levels=0),
    }
    results = {}
    for name, overrides in overrides_to_test.items():
        ratios = []
        for seed in seeds:
            grid, hc, tm, state = _build_nh_state(8, seed=seed)
            cfg_kw = {**base_kw, **overrides}
            cfg = make_fv3_faithful_nh_config(**cfg_kw)
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            s = state
            for _ in range(3):
                s = m.step(s, dt=10.0)
            ratios.append(
                _edge_interior_ratio(s.theta_prime.data)
            )
        results[name] = float(np.mean(ratios))

    baseline = results["baseline (factory)"]
    with capsys.disabled():
        print(
            f"\n[iter-465 NH per-flag θ′ edge-ratio sweep, "
            f"3 seeds @ C8, 3 steps]"
        )
        print(f"  baseline (factory): {baseline:.4f}")
        print(f"  Per-flag deltas (flag OFF − baseline):")
        for name, r in results.items():
            if name == "baseline (factory)":
                continue
            delta = r - baseline
            sign = "+" if delta >= 0 else ""
            print(f"    {name:30s}: {r:.4f} ({sign}{delta:.4f})")
        print(
            "  POSITIVE delta = flag was REDUCING the ratio "
            "(useful for edge-artifact reduction)."
        )
    assert all(np.isfinite(r) for r in results.values())
