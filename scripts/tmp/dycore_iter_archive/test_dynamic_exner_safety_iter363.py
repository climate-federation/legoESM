"""FV3_3D iter 363: safety test for iter-336/337 dynamic Exner —
guard against Π_total = Π_ref + π' going non-positive under
strong perturbations.

If π' < -Π_ref anywhere, Π_total ≤ 0 → division by ≤ 0 in d_con
denominator → NaN/Inf or wrong-sign heat.  iter-336/337 uses
``Π_total = exner_ref + pi_prime`` directly without clamp.  This
test verifies behavior under strong perturbation:

* Π_total stays positive everywhere for physically-realistic
  state (small θ', ρ' perturbations).
* No NaN/Inf in dθ_p tendency under typical NH state with
  ±15 K θ' + ±0.2 kg/m³ ρ' perturbations.

Tests
-----

1. ``test_dynamic_exner_finite_under_strong_pert`` — strong
   θ' / ρ' perturbations don't produce NaN/Inf in NH d_con
   output under ``use_fv3_dynamic_exner=True``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


def test_dynamic_exner_finite_under_strong_pert():
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)
    rng = np.random.default_rng(seed=363)
    u_p = rng.uniform(-10.0, 10.0, size=(6, n, n, nlev))
    v_p = rng.uniform(-10.0, 10.0, size=(6, n, n, nlev))
    # Strong but physical θ' / ρ' perturbation.
    theta_p = rng.uniform(-15.0, 15.0, size=(6, n, n, nlev))
    rho_p = rng.uniform(-0.2, 0.2, size=(6, n, n, nlev))

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.asarray(u_p), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.asarray(v_p), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.asarray(theta_p),
                          name="theta_prime",
                          dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.asarray(rho_p),
                        name="rho_prime",
                        dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    cfg = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=1e14, n_acoustic_substeps=4,
        corner_div_damp_d2_bg=0.0005, corner_div_damp_dddmp=0.20,
        corner_div_damp_d_con=1.0,
        damp_v=0.030, nord_v=1, damp_v_d_con=1.0,
        div_damp_coeff=1e6, div_damp_dddmp=0.20,
        div_damp_d_con=1.0,
        A_h=1e6, ah_d_con=1.0,
        damp_w=0.030, damp_w_d_con=1.0,
        delt_max=1.0,
        use_fv3_dynamic_exner=True,
    )
    m = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg,
    )
    s = m.step(state, 5.0)
    for fld_name, fld in [
        ("u", s.u.data), ("v", s.v.data), ("w", s.w.data),
        ("theta_prime", s.theta_prime.data),
        ("rho_prime", s.rho_prime.data),
    ]:
        assert np.all(np.isfinite(np.asarray(fld))), (
            f"Dynamic Exner produced non-finite {fld_name} under "
            f"strong θ'/ρ' perturbation."
        )
