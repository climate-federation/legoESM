"""FV3_3D iter 700: 4th-order theta_corner cc → B-grid corner.

Adds the ``use_fv3_a2b_ord4_theta_corner`` flag to
``CDGridCompressibleEulerConfig`` and wires the
``theta_corner = interp_center_to_corner(theta_total, cdgrid)``
site (~line 383) to switch to ``a2b_ord4`` when the flag is on.

Default OFF; impact measurement pending.  iter-696/697/698/699
already promoted the analogous VECTOR (u, v) lift; this is the
SCALAR θ_total path.

Tests
-----

1. ``test_flag_exists_default_off``.
2. ``test_factory_default_off``.
3. ``test_flag_override_takes_effect``.
4. ``test_flag_changes_dycore_one_step``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
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


def test_flag_exists_default_off():
    cfg = CDGridCompressibleEulerConfig()
    assert hasattr(cfg, "use_fv3_a2b_ord4_theta_corner")
    assert cfg.use_fv3_a2b_ord4_theta_corner is False


def test_factory_default_off():
    """make_fv3_faithful_nh_config leaves iter-700 flag OFF (default)."""
    cfg = make_fv3_faithful_nh_config()
    assert cfg.use_fv3_a2b_ord4_theta_corner is False


def test_flag_override_takes_effect():
    """Setting the flag flips the config attribute."""
    cfg = make_fv3_faithful_nh_config(use_fv3_a2b_ord4_theta_corner=True)
    assert cfg.use_fv3_a2b_ord4_theta_corner is True


def test_flag_changes_dycore_one_step(capsys):
    """Single dycore step at C8: enabling the flag changes
    theta_prime[...] output (proves the wiring is live, not a
    no-op).  Tolerance: any non-zero L∞ diff is enough.
    """
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(nlev, z_top)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    rng = np.random.default_rng(seed=700)
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

    s_off = state
    s_on = state
    m_off = CDGridCompressibleEulerModel(
        grid, hc, tm,
        make_fv3_faithful_nh_config(use_fv3_a2b_ord4_theta_corner=False),
    )
    m_on = CDGridCompressibleEulerModel(
        grid, hc, tm,
        make_fv3_faithful_nh_config(use_fv3_a2b_ord4_theta_corner=True),
    )
    s_off = m_off.step(s_off, dt=10.0)
    s_on = m_on.step(s_on, dt=10.0)
    delta = float(jnp.max(jnp.abs(
        s_on.theta_prime.data - s_off.theta_prime.data
    )))
    with capsys.disabled():
        print(
            f"\n[iter-700 theta_corner ord4 wiring "
            f"max|Δθ′| after 1 step @ C8] = {delta:.3e}"
        )
    assert jnp.all(jnp.isfinite(s_on.theta_prime.data))
    assert jnp.all(jnp.isfinite(s_off.theta_prime.data))
    assert delta > 0.0   # wiring is live (not a no-op)
