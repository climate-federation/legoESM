"""Iter-1002/1009 sentinel: W2 v_ll_Linf ≤ 0.119 m/s AND W5 day-5 artifact-free.

Iter-1002 discovered W2 v_ll_Linf=0.1154 m/s ≤ 0.119 met with
(div=12*cube, damp_v=0.04).  Iter-1009 found that (div=10*cube,
damp_v=0.04) is STRICTLY BETTER: v_ll_Linf=0.1147 AND extends W5
artifact-free window from day 4 → day 5 (typical W5 reference).

Final dual-target calibration (iter-1009):
  - div_damp = 10 * _div_damp_cube(N)
  - damp_v = 0.04
  - nord_v = 2
  - apply_fortran_xppm_boundary = True (iter-888 Fortran fidelity)
  - boundary_fix = True
  - hyperdiff_coeff = 0.0

W2 C36 1-day → v_ll_Linf = 0.1147 m/s ≤ 0.119 m/s ✓
W5 C36 day-5 → h_min=3885 m, speed_max=68.6 m/s ✓ (artifact-free)

This is a CALIBRATION refinement of the iter-893 production
baseline (which gave 0.132 m/s).  The PRODUCTION DEFAULT in
CDGridShallowWaterConfig stays at iter-893 baseline values to
preserve every other downstream sentinel; this test pins the
iter-1009 calibration as the demonstrated dual-target-met config.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import warnings

import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
    CDGridShallowWaterConfig,
    FV3EdgeShallowWaterModel,
    FV3EdgeShallowWaterState,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.regridding import (
    apply_cubedsphere_to_latlon,
    get_cubedsphere_to_latlon_weights,
)
from tests.atmosphere.shallow_water.test_cases.williamson import (
    williamson_test2,
    williamson_test5,
)
from tests.test_iter921_w2_v_vs_h_pareto_sentinel import (
    cell_centre_angles_from_4edge,
    _div_damp_cube,
)


def _make_iter1009_config(N):
    """Iter-1009/1021/1030 calibration that meets BOTH W2 ≤ 0.119 AND W5 day-5.

    Calibration evolution:
    - iter-1009: (10, 0.04) → v_ll=0.1147, W5 spd=68.6
    - iter-1021: (9, 0.035) → v_ll=0.1137, W5 spd=53.3
    - iter-1030: (8, 0.030) → v_ll=0.1138, W5 spd=45.1 (best W5)
    """
    return CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(N),
        boundary_fix=True,
        damp_v=0.030, nord_v=2,
        apply_fortran_xppm_boundary=True,
    )


def test_iter1020_dddmp_silent_noop_warning():
    """`fv3_sw_tendencies(dddmp>0, div_damp=0)` should warn loudly.

    Iter-872c-take5 added a UserWarning for the case where dddmp is
    set non-zero but div_damp is zero — the narrow gate inside
    `cdgrid_momentum_tendencies` silently no-ops dddmp in that
    regime.  Iter-1020 hardens that warning by pinning a sentinel.
    """
    import warnings as _warnings

    import jax.numpy as jnp_local

    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig as _Cfg,
        FV3EdgeShallowWaterModel as _Model,
        FV3EdgeShallowWaterState as _State,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere as _csp
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid as _cdgrid,
    )
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test2 as _w2,
    )

    grid = _csp(36)
    cdgrid = _cdgrid(grid)
    sw = _w2(grid)
    u_d = cdgrid.cos_angle_edge_x * jnp_local.cos(cdgrid.lat_edge_x)
    v_d = -cdgrid.sin_angle_edge_y * jnp_local.cos(cdgrid.lat_edge_y)
    state = _State(h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)

    # dddmp_prod > 0 with div_damp = 0 → warn
    cfg = _Cfg(div_damp=0.0, dddmp_prod=0.2,
                apply_fortran_xppm_boundary=True)
    model = _Model(grid, cfg)
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        _ = model.step(state, 100.0)
        silent_warns = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "silently no-ops" in str(x.message)
        ]
        assert len(silent_warns) >= 1, (
            f"Expected at least 1 silently-no-ops warning when "
            f"dddmp > 0 and div_damp = 0; got {len(silent_warns)}")


def test_iter1019_hyperdiff_silent_noop_warning():
    """`fv3_sw_tendencies(hyperdiff_coeff>0)` should warn loudly.

    Iter-1019 Codex audit found that `hyperdiff_coeff` appears in
    `fv3_sw_tendencies` signature but is never applied in the body.
    Callers passing `hyperdiff_coeff > 0` would silently see no
    biharmonic damping.  iter-1019 added a UserWarning to make this
    explicit.
    """
    import warnings as _warnings

    import jax.numpy as jnp_local

    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig as _Cfg,
        FV3EdgeShallowWaterModel as _Model,
        FV3EdgeShallowWaterState as _State,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere as _csp
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid as _cdgrid,
    )
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test2 as _w2,
    )

    grid = _csp(36)
    cdgrid = _cdgrid(grid)
    sw = _w2(grid)
    u_d = cdgrid.cos_angle_edge_x * jnp_local.cos(cdgrid.lat_edge_x)
    v_d = -cdgrid.sin_angle_edge_y * jnp_local.cos(cdgrid.lat_edge_y)
    state = _State(h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)

    # hyperdiff_coeff > 0 → MUST warn
    cfg = _Cfg(hyperdiff_coeff=0.005, apply_fortran_xppm_boundary=True)
    model = _Model(grid, cfg)
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        _ = model.step(state, 100.0)
        silent_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and "silently ignored" in str(x.message)
        ]
        assert len(silent_warnings) >= 1, (
            f"Expected at least 1 silent-noop warning when "
            f"hyperdiff_coeff > 0; got {len(silent_warnings)}")


def test_iter1017_preset_warns_on_non_c36():
    """`iter1009_dual_target_config(n != 36)` should emit UserWarning."""
    import warnings
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        iter1009_dual_target_config,
    )
    # C36 must not warn
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = iter1009_dual_target_config(36)
        c36_warnings = [
            x for x in w if issubclass(x.category, UserWarning)
        ]
        assert len(c36_warnings) == 0, (
            f"C36 should not emit a warning; got {len(c36_warnings)}")

    # C48 must warn (calibration not validated)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = iter1009_dual_target_config(48)
        c48_warnings = [
            x for x in w if issubclass(x.category, UserWarning)
        ]
        assert len(c48_warnings) == 1, (
            f"C48 should emit exactly 1 UserWarning; "
            f"got {len(c48_warnings)}")
        assert "validated only at N=36" in str(c48_warnings[0].message)


def test_iter1013_preset_helper_matches_explicit_config():
    """`iter1009_dual_target_config(N)` matches the explicit config."""
    from legoesm.atmosphere.dynamics.shallow_water_fv3_cdgrid import (
        iter1009_dual_target_config,
    )
    N = 36
    explicit = _make_iter1009_config(N)
    preset = iter1009_dual_target_config(N)
    # Compare key fields
    assert abs(preset.div_damp - explicit.div_damp) < 1e-6, (
        f"div_damp mismatch: preset={preset.div_damp}, "
        f"explicit={explicit.div_damp}")
    assert preset.damp_v == explicit.damp_v
    assert preset.nord_v == explicit.nord_v
    assert (preset.apply_fortran_xppm_boundary
             == explicit.apply_fortran_xppm_boundary)
    assert preset.boundary_fix == explicit.boundary_fix
    assert preset.hyperdiff_coeff == explicit.hyperdiff_coeff


def test_iter1002_w2_v_ll_linf_meets_target():
    """W2 C36 1-day v_ll_Linf ≤ 0.119 m/s with iter-1009 calibration."""
    N = 36
    DT = 300.0
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test2(grid)
    u0 = 2.0 * jnp.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = _make_iter1009_config(N)
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    n_steps = int(86400 / DT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_steps):
            state = model.step(state, DT)

    ca, sa = cell_centre_angles_from_4edge(cdgrid)
    u_arr = np.asarray(state.u_d)
    v_arr = np.asarray(state.v_d)
    u_cc = 0.5 * (u_arr[:, :, :-1] + u_arr[:, :, 1:])
    v_cc = 0.5 * (v_arr[:, :-1, :] + v_arr[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    weights = get_cubedsphere_to_latlon_weights(N, n_lon=360, n_lat=181)
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    v_ll_Linf = float(np.abs(v_ll).max())
    h_err_max = float(np.max(np.abs(np.asarray(state.h) - sw.h.data)))

    assert v_ll_Linf <= 0.119, (
        f"W2 v_ll_Linf = {v_ll_Linf:.4f} m/s > 0.119 m/s target.  "
        f"Iter-1009 measured 0.1147; allowing a small upward drift "
        f"to 0.119 catches obvious regressions while permitting tiny "
        f"numerical noise.")
    assert h_err_max < 20.0, (
        f"W2 h_err_max = {h_err_max:.4f} m > 20.0 m soft bound.  "
        f"Iter-1009 measured ~9.00 m.")


def test_iter1009_w5_day5_artifact_free():
    """W5 C36 day-5 artifact-free with iter-1009 calibration.

    Pins:
      h_min > 0  (no negative heights)
      speed_max < 80 m/s  (physically reasonable for W5 zonal flow
                            over mountain — analytical max ~30 m/s,
                            with allowance for Rossby wave dev)

    Iter-1009 measured at day 5: h_min=3885 m, speed_max=68.6 m/s.
    """
    N = 36
    DT = 300.0
    DAYS = 5
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw = williamson_test5(grid)
    u0 = 20.0
    u_d = cdgrid.cos_angle_edge_x * (u0 * jnp.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (u0 * jnp.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    cfg = _make_iter1009_config(N)
    model = FV3EdgeShallowWaterModel(grid, cfg)
    model.set_initial_mass(state)
    n_steps = int(DAYS * 86400 / DT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_steps):
            state = model.step(state, DT)

    h = np.asarray(state.h)
    u = np.asarray(state.u_d)
    v = np.asarray(state.v_d)
    assert np.isfinite(h).all() and np.isfinite(u).all() and np.isfinite(v).all(), (
        "W5 produced NaN at day 5 — instability.")

    h_min = float(h.min())
    assert h_min > 0.0, (
        f"W5 day-5 h_min = {h_min:.2f} m ≤ 0.  "
        f"Iter-1009 measured ~3885 m.")

    u_cc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
    v_cc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    speed_max = float(np.sqrt(u_cc**2 + v_cc**2).max())
    assert speed_max < 80.0, (
        f"W5 day-5 speed_max = {speed_max:.2f} m/s ≥ 80 m/s.  "
        f"Iter-1009 measured ~68.6 m/s.")
