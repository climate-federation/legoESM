"""FV3_3D iter 702: empirical impact of ``use_fv3_a2b_zeta_corner``
on NH θ′ cube-edge ratio at C8.

iter-170 ported the 4th-order ζ_corner cascade to the NH path but
left ``make_fv3_faithful_nh_config`` factory default OFF.  iter-702
measures impact under the same iter-698/699 methodology to decide
whether to promote.

ζ_corner directly feeds the Coriolis term ``abs_vor_corner * v_d``
on the du/dt momentum eqn — primary differentiated quantity, NOT
a coefficient.  Unlike iter-701 θ_corner null result, ζ is
expected to be sensitive.

Tests
-----

1. ``test_a2b_zeta_corner_nh_edge_ratio_impact``.
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


def test_a2b_zeta_corner_nh_edge_ratio_impact(capsys):
    """θ′ edge ratio @ C8 with iter-170 ζ-corner flag ON vs OFF."""
    seeds = [702, 703, 704]
    base_kw = dict(
        n_acoustic_substeps=4,
        damp_v=0.030, damp_v_d_con=1.0,
        corner_div_damp_d2_bg=5e-4,
        corner_div_damp_d_con=1.0,
        div_damp_coeff=1e6, div_damp_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
    )
    variants = {
        "OFF (default)": dict(use_fv3_a2b_zeta_corner=False),
        "ON (iter-170)": dict(use_fv3_a2b_zeta_corner=True),
    }
    results: dict[str, list[float]] = {k: [] for k in variants}
    for seed in seeds:
        for name, ov in variants.items():
            grid, hc, tm, state = _build_nh_state(8, seed=seed)
            cfg = make_fv3_faithful_nh_config(**{**base_kw, **ov})
            m = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
            s = state
            for _ in range(3):
                s = m.step(s, dt=10.0)
            results[name].append(
                _edge_interior_ratio(s.theta_prime.data)
            )

    off_mean = float(np.mean(results["OFF (default)"]))
    on_mean = float(np.mean(results["ON (iter-170)"]))
    delta = on_mean - off_mean
    pct = 100.0 * delta / off_mean if off_mean != 0.0 else 0.0
    with capsys.disabled():
        print(
            f"\n[iter-702 a2b_zeta_corner NH θ′ edge-ratio impact, "
            f"3 seeds @ C8, 3 steps]"
        )
        print(f"  OFF: mean = {off_mean:.4f}  "
              f"per-seed = {[f'{r:.4f}' for r in results['OFF (default)']]}")
        print(f"  ON : mean = {on_mean:.4f}  "
              f"per-seed = {[f'{r:.4f}' for r in results['ON (iter-170)']]}")
        sign = "+" if delta >= 0 else ""
        print(f"  delta (ON − OFF) = {sign}{delta:.4f}  ({sign}{pct:.1f}%)")
        print(
            "  NEGATIVE delta = ζ_corner ord4 reduces cube imprint "
            "(candidate for NH factory ON, parallels PE iter-170 +"
            " iter-698 vector promotion)."
        )

    assert np.all(np.isfinite(results["OFF (default)"]))
    assert np.all(np.isfinite(results["ON (iter-170)"]))
