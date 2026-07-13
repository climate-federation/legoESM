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

import jax
import jax.numpy as jnp
import numpy as np

# new_test_dycores iter-99: hard-fail at module import if JAX is not in
# fp64 mode.  The `os.environ.setdefault` above only helps if this is
# the FIRST file in pytest's session to touch the JAX env (otherwise
# JAX is already configured fp32).  Without fp64, the iter-3/iter-6
# anchored-fixer thresholds (mass_drift < 1e-7) reduce to fp32 round-
# off noise (~1e-7 already) and tests trip on environmental issues
# rather than real regressions.  Fail loudly so the user sees the
# fp64-not-enabled cause directly instead of a confusing mass-drift
# assertion failure.
assert jax.config.read("jax_enable_x64"), (
    "tests/test_iter1002_w2_target_met.py requires JAX_ENABLE_X64=1.  "
    "JAX is currently in fp32 mode (likely because an earlier test "
    "imported jax without setting the env var).  Run with "
    "`JAX_ENABLE_X64=1 pytest ...` from the shell."
)

from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
    """new_test_dycores iter-60: positive guard that the SW model
    step does NOT spuriously emit the ``silently no-ops`` warning.

    The original iter-1020 sentinel (iter-872c-take5) asserted that
    ``cdgrid_momentum_tendencies`` warns when ``dddmp>0`` and
    ``div_damp==0``.  That warning lives in
    ``cdgrid_momentum_tendencies`` (operators_cdgrid.py:1141), but
    ``FV3EdgeShallowWaterModel.step`` dispatches through
    ``fv3_sw_tendencies`` (a different code path that doesn't read
    ``dddmp_prod``), so the warning never fires via ``model.step``.
    The original sentinel asserted a behaviour that never actually
    held for SW model callers — a stale test against the wrong
    entry point.

    iter-60 inverts it: assert that ``model.step`` with
    ``dddmp_prod>0`` runs WITHOUT spurious silent-noop warnings.
    A future regression that wires ``dddmp_prod`` through
    ``fv3_sw_tendencies`` and forgets to add a matching warning
    would NOT trip this test (that would be caught by the
    iter-49/57 numerical sentinels via behaviour change instead);
    a regression that spuriously emits the warning even though
    nothing is silently dropped WOULD trip this test.
    """
    import warnings as _warnings

    import jax.numpy as jnp_local

    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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

    # dddmp_prod>0 on the SW model: NO spurious silent-noop warning
    # — the SW model uses fv3_sw_tendencies, not the
    # cdgrid_momentum_tendencies path that emits the warning.
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
        assert len(silent_warns) == 0, (
            f"iter-60 regression: SW model.step spuriously emits "
            f"``silently no-ops`` warning ({len(silent_warns)} "
            "instances).  This warning is only meaningful inside "
            "``cdgrid_momentum_tendencies``, which the SW model "
            "doesn't traverse."
        )


def test_iter1019_hyperdiff_silent_noop_warning():
    """new_test_dycores iter-60: positive guard that the SW model
    step does NOT emit the iter-1019 ``signature-only NO-OP``
    warning for ``hyperdiff_coeff``.

    The original iter-1019 sentinel pinned a warning saying
    ``hyperdiff_coeff`` was signature-only NO-OP in
    ``fv3_sw_tendencies``.  iter-41 (new_test_dycores) verified
    that the cell-centre biharmonic IS applied at
    operators_cdgrid.py L1444-1456 and removed the stale warning.
    The original sentinel went out of date with the implementation.

    iter-60 inverts: positive guard that ``model.step`` with
    ``hyperdiff_coeff>0`` runs cleanly (no spurious warnings)
    because the feature is now actually implemented.  This is the
    matching test for the iter-39/48 short-run numerical sentinels
    which verify the *numerical* effect; this one verifies the
    *quiet path*.
    """
    import warnings as _warnings

    import jax.numpy as jnp_local

    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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

    # hyperdiff_coeff>0 should NOT trigger the iter-1019
    # "signature-only NO-OP" warning — the feature is now
    # implemented in fv3_sw_tendencies (iter-41).
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
        assert len(silent_warnings) == 0, (
            f"iter-60 regression: SW model.step spuriously emits "
            f"``silently ignored`` warning for hyperdiff_coeff "
            f"({len(silent_warnings)} instances).  The feature is "
            "implemented at operators_cdgrid.py L1444-1456."
        )


def test_iter1017_preset_warns_on_non_c36():
    """`iter1009_dual_target_config(n != 36)` should emit UserWarning."""
    import warnings
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
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


def test_iter39_cube_w6_short_run_stable_with_hyperdiff():
    """new_test_dycores iter-39: numerical regression test for the
    iter-31 cube W6 stability fix.

    The iter-32 AST sentinel pins the FLAG (matrix runner sets
    ``hyperdiff_coeff=_hyperdiff_cube(n)``) but doesn't validate
    that the flag has the intended numerical effect.  A regression
    where the flag is correctly set but the hyperdiff implementation
    is broken (e.g., del-4 stencil bug) would silently re-introduce
    cube W6 instability past day 9.

    This test runs cube W6 for 576 steps (2 days at dt=300 s,
    matching the matrix-runner SW cube ``dt = 300.0``; ~10-15 s
    wall) with the iter-31 hyperdiff override and asserts finite
    state + small mass drift.  Catches a numerical break that the
    AST sentinel cannot.
    """
    import jax
    import jax.numpy as jnp_local
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState,
        iter1009_dual_target_config,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid,
    )
    from tests.test_cases.williamson_extended import (
        williamson_test6, _w6_winds_geo,
    )
    import warnings

    n = 36
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    def _div_damp_cube_local(nn, ref_n=48, ref_coeff=1.5e7):
        return ref_coeff * (ref_n / nn) ** 2

    def _hyperdiff_cube_local(nn, ref_n=48, ref_coeff=1e16):
        return ref_coeff * (ref_n / nn) ** 4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = iter1009_dual_target_config(
            n, hyperdiff_coeff=_hyperdiff_cube_local(n),
        )
    model = FV3EdgeShallowWaterModel(grid, cfg)

    sw = williamson_test6(grid)
    R = grid.radius
    u_east_x, v_north_x = _w6_winds_geo(
        cdgrid.lon_edge_x, cdgrid.lat_edge_x, R,
    )
    u_d = (cdgrid.cos_angle_edge_x * u_east_x
           + cdgrid.sin_angle_edge_x * v_north_x)
    u_east_y, v_north_y = _w6_winds_geo(
        cdgrid.lon_edge_y, cdgrid.lat_edge_y, R,
    )
    v_d = (-cdgrid.sin_angle_edge_y * u_east_y
           + cdgrid.cos_angle_edge_y * v_north_y)
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    model.set_initial_mass(state)

    dt = 300.0
    area64 = grid.area.astype(jnp_local.float64)
    mass_init = float(
        jnp_local.sum(state.h.astype(jnp_local.float64) * area64),
    )

    @jax.jit
    def step(s):
        return model.step(s, dt)

    s = state
    for _ in range(576):  # 2 days at dt=300 s
        s = step(s)

    # Finiteness check.
    assert jnp_local.all(jnp_local.isfinite(s.h)), (
        "iter-39 regression: cube W6 day-2 ``h`` field non-finite"
    )
    assert jnp_local.all(jnp_local.isfinite(s.u_d)), (
        "iter-39 regression: cube W6 day-2 ``u_d`` field non-finite"
    )
    # Mass conservation (anchored fixer should give bit-clean).
    mass_final = float(
        jnp_local.sum(s.h.astype(jnp_local.float64) * area64),
    )
    drift = abs(mass_final - mass_init) / abs(mass_init)
    assert drift < 1e-7, (
        f"iter-39 regression: cube W6 day-2 mass drift {drift:.2e} "
        "exceeds 1e-7"
    )
    # Match the matrix-runner's BLOWUP criterion: max|u_d| > 1000
    # m/s flags a blowup.  W6 initial winds peak at ~50 m/s, so a
    # 2-day max|u_d| under 100 m/s indicates stable propagation.
    u_d_max = float(jnp_local.max(jnp_local.abs(s.u_d)))
    assert u_d_max < 100.0, (
        f"iter-39 regression: cube W6 day-2 ``max|u_d|`` = "
        f"{u_d_max:.1f} m/s exceeds 100 m/s — wind blowup "
        "(matrix BLOWUP threshold is 1000 m/s)."
    )


def test_iter57_cube_w2_matrix_config_5day():
    """new_test_dycores iter-57: numerical sentinel at the matrix
    runner's test-of-record duration (5 days, matching the
    matrix-runner Williamson 2 test definition at line 198 of
    ``scripts/run_atmosphere_test_matrix.py``).

    Pins iter-44's measured matrix W2 5-day numerics
    (L2=4.58e-4, v_ll-equivalent peak ~0.51 m/s) so a regression
    that drops the 2x hyperdiff or shifts the iter-1030
    calibration would trip the test without re-running the matrix.

    Asserts:
      - finite h, u_d
      - mass_drift < 1e-7  (anchored fixer)
      - h-field L2 < 1.5e-3 (iter-44 measured 4.58e-4; ceiling has
                              3x headroom)
      - max|v_d - v_d_init| < 5.0 m/s (iter-44 measured ~3 m/s;
                                        ceiling has ~1.5x headroom)
    """
    import jax
    import jax.numpy as jnp_local
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState,
        iter1009_dual_target_config,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid,
    )
    from tests.test_cases.williamson import williamson_test2
    import warnings

    n = 36
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    def _hyperdiff_cube_local(nn, ref_n=48, ref_coeff=1e16):
        return ref_coeff * (ref_n / nn) ** 4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = iter1009_dual_target_config(
            n, hyperdiff_coeff=2.0 * _hyperdiff_cube_local(n),
        )
    model = FV3EdgeShallowWaterModel(grid, cfg)

    sw = williamson_test2(grid)
    u0 = 2.0 * jnp_local.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (
        u0 * jnp_local.cos(cdgrid.lat_edge_x))
    v_d_init = -cdgrid.sin_angle_edge_y * (
        u0 * jnp_local.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d_init, h_s=sw.h_s.data,
    )
    model.set_initial_mass(state)

    dt = 300.0
    area64 = grid.area.astype(jnp_local.float64)
    mass_init = float(
        jnp_local.sum(state.h.astype(jnp_local.float64) * area64),
    )

    @jax.jit
    def step(s):
        return model.step(s, dt)

    s = state
    for _ in range(1440):  # 5 days at dt=300 s
        s = step(s)

    assert jnp_local.all(jnp_local.isfinite(s.h)), (
        "iter-57 regression: cube W2 day-5 ``h`` non-finite"
    )
    assert jnp_local.all(jnp_local.isfinite(s.u_d)), (
        "iter-57 regression: cube W2 day-5 ``u_d`` non-finite"
    )
    mass_final = float(
        jnp_local.sum(s.h.astype(jnp_local.float64) * area64),
    )
    drift = abs(mass_final - mass_init) / abs(mass_init)
    assert drift < 1e-7, (
        f"iter-57 regression: cube W2 day-5 mass drift {drift:.2e} "
        "exceeds 1e-7"
    )
    h_init = sw.h.data
    h_err_l2 = float(
        jnp_local.sqrt(
            jnp_local.sum(
                ((s.h - h_init).astype(jnp_local.float64) ** 2)
                * area64,
            )
            / jnp_local.sum(h_init.astype(jnp_local.float64) ** 2 * area64)
        ),
    )
    assert h_err_l2 < 1.5e-3, (
        f"iter-57 regression: cube W2 day-5 h-field L2 = "
        f"{h_err_l2:.4e} exceeds 1.5e-3 ceiling — iter-44 measured "
        "L2=4.58e-4."
    )
    v_d_err = float(
        jnp_local.max(jnp_local.abs(s.v_d - v_d_init)),
    )
    assert v_d_err < 5.0, (
        f"iter-57 regression: cube W2 day-5 ``max|v_d - v_d_init|`` "
        f"= {v_d_err:.4f} m/s exceeds 5.0 m/s ceiling."
    )


def test_iter59_cube_cb_12day_matrix_config():
    """new_test_dycores iter-59: numerical sentinel for cube
    cosine_bell 12-day at the matrix-runner configuration (iter-58
    hord=10 + apply_fortran_xppm_boundary + iter-59 N=6 temporal
    substepping).

    Reproduces the matrix-runner CB cube branch
    (``scripts/run_atmosphere_test_matrix.py`` line ~2589) and pins
    the iter-59 measured 12-day L2=0.865 / Linf=0.866 / mass_drift
    ~1e-9.  Asserts:

      - finite h
      - mass_drift < 1e-7   (anchored fixer; matrix measured 1.3e-9)
      - L2  < 0.90          (iter-59 matrix 0.865; cushion ~4 %)
      - Linf < 0.90          (iter-59 matrix 0.866; cushion ~4 %)

    A refactor that drops the iter-58 hord=10/xppm_bdy combo or the
    iter-59 substep scan would trip the L2 ceiling without
    re-running the matrix.  Run-time on a developer workstation is
    ~10 s post-warm.
    """
    import jax
    import jax.numpy as jnp_local
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel,
        iter1009_dual_target_config,
    )
    from legoesm.core.fv3_sw_core import d2a2c_vect
    from legoesm.core.fv_tp_2d import transport_step
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from tests.test_cases.cosine_bell import (
        cosine_bell_cubesphere,
        cosine_bell_error_norms,
        cosine_bell_exact,
    )

    n = 36
    beta = float(jnp_local.pi / 4.0)
    grid = create_cubed_sphere(n)
    cfg = iter1009_dual_target_config(n)
    model = FV3EdgeShallowWaterModel(grid, cfg)
    cdgrid = model.cdgrid
    state = cosine_bell_cubesphere(grid, cdgrid, beta)
    _ua, _va, _uc, _vc, ut, vt = d2a2c_vect(state.u_d, state.v_d, cdgrid)
    area64 = grid.area.astype(jnp_local.float64)
    mass_target = float(
        jnp_local.sum(state.h.astype(jnp_local.float64) * area64),
    )

    dt_outer = 1800.0
    n_sub = 6
    dt_sub = dt_outer / n_sub

    @jax.jit
    def outer(h, _):
        def body(h, _):
            return transport_step(
                h, ut, vt, dt_sub, cdgrid,
                mass_target=mass_target,
                hord=10,
                apply_fortran_xppm_boundary=True,
            ), None
        h_new, _ = jax.lax.scan(body, h, None, length=n_sub)
        return h_new, None

    nsteps = int(12.0 * 86400.0 / dt_outer)
    h, _ = jax.lax.scan(outer, state.h, None, length=nsteps)

    assert jnp_local.all(jnp_local.isfinite(h)), (
        "iter-59 regression: cube CB 12-day ``h`` non-finite"
    )
    mass_final = float(jnp_local.sum(h.astype(jnp_local.float64) * area64))
    drift = abs(mass_final - mass_target) / abs(mass_target)
    assert drift < 1e-7, (
        f"iter-59 regression: cube CB 12-day mass drift {drift:.2e} "
        "exceeds 1e-7 (matrix measured 1.3e-9)"
    )

    t_final = nsteps * dt_outer
    h_exact = cosine_bell_exact(grid.lon, grid.lat, grid.radius, t_final, beta)
    norms = cosine_bell_error_norms(h, h_exact, grid.area)
    assert norms["l2"] < 0.90, (
        f"iter-59 regression: cube CB 12-day L2 = {norms['l2']:.4f} "
        "exceeds 0.90 ceiling — iter-59 matrix measured 0.865 with "
        "hord=10 + xppm_bdy + n_sub=6 substepping."
    )
    assert norms["linf"] < 0.90, (
        f"iter-59 regression: cube CB 12-day Linf = "
        f"{norms['linf']:.4f} exceeds 0.90 ceiling — iter-59 matrix "
        "measured 0.866."
    )


def test_iter61_latlon_cb_12day_mass_fixer():
    """new_test_dycores iter-61: numerical sentinel for the
    latlon CB anchored mass fixer applied to the matrix runner's
    CB latlon step_fn.

    Pre-iter-61 the latlon CB matrix branch was an intentional raw-FV
    benchmark (no mass correction) leaving the 12-day mass drift at
    5.35e-4 — above the iter-29 matrix tolerance (1e-4) → silent FAIL.
    iter-61 imported the cube CB anchored fixer (clip negatives +
    rescale positives to ``mass_target``) bringing the drift to
    2.07e-8 (26000x tighter) so all 4 grids now PASS at 12-day
    apples-to-apples.

    This sentinel reproduces the matrix's latlon CB step_fn with
    the iter-61 fixer applied and asserts:

      - finite h
      - mass_drift < 1e-7   (matrix measured 2.07e-8; the same 1e-7
                              ceiling the iter-59 cube sentinel uses)

    Together with `test_iter59_cube_cb_12day_matrix_config` and the
    AST guard `test_iter61_cb_latlon_has_anchored_mass_fixer` these
    cover the iter-61 cross-grid CB conservation parity result.
    Run-time ~5 s post-warm (latlon CB is fast — 72x144 cells at
    dt~126 s for 12 days).
    """
    import math
    import jax
    import jax.numpy as jnp_local
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterState,
        cell_to_cgrid_winds,
    )
    from legoesm.core.operators_fv_latlon import (
        cgrid_fv_flux_divergence_latlon,
    )
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.timestepping.dispatch import dispatch_integrator
    from legoesm.core.conservation import conservation_accumulator
    from tests.test_cases.cosine_bell import cosine_bell_latlon

    n_lat, n_lon = 72, 144
    beta = float(jnp_local.pi / 4.0)
    grid = create_latlon_grid(n_lat, n_lon)
    _u0 = 2.0 * math.pi * float(grid.radius) / (12.0 * 86400.0)
    _dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    dt = min(1800.0, 0.8 * _dx_pole / _u0)

    cb = cosine_bell_latlon(grid, beta)
    u_face, v_face = cell_to_cgrid_winds(cb.u.data, cb.v.data)
    state = CGridLatLonShallowWaterState(
        h=cb.h.data, u=u_face, v=v_face,
        h_s=jnp_local.zeros_like(cb.h.data),
    )
    _acc = conservation_accumulator()
    area64 = grid.area.astype(_acc)
    mass_target = jnp_local.sum(state.h.astype(_acc) * area64)

    @jax.jit
    def step(s):
        def tendency_fn(st):
            dh = cgrid_fv_flux_divergence_latlon(
                st.h, u_face, v_face, grid)
            return st._replace(
                h=dh,
                u=jnp_local.zeros_like(st.u),
                v=jnp_local.zeros_like(st.v),
                h_s=jnp_local.zeros_like(st.h_s))
        s_new = dispatch_integrator(s, tendency_fn, dt, "ssp_rk3")
        h_pos = jnp_local.maximum(s_new.h, 0.0)
        mass_pos = jnp_local.sum(h_pos.astype(_acc) * area64)
        scale = mass_target / jnp_local.maximum(mass_pos, 1.0)
        return s_new._replace(h=h_pos * scale.astype(h_pos.dtype))

    nsteps = int(12.0 * 86400.0 / dt)
    s = state
    for _ in range(nsteps):
        s = step(s)

    assert jnp_local.all(jnp_local.isfinite(s.h)), (
        "iter-61 regression: latlon CB 12-day ``h`` non-finite"
    )
    mass_final = jnp_local.sum(s.h.astype(_acc) * area64)
    drift = float(abs(mass_final - mass_target)
                  / abs(mass_target))
    assert drift < 1e-7, (
        f"iter-61 regression: latlon CB 12-day mass drift {drift:.2e} "
        "exceeds 1e-7 ceiling — matrix measured 2.07e-8 with the "
        "iter-61 anchored fixer."
    )


def test_iter49_cube_w2_matrix_config_short_run():
    """new_test_dycores iter-49: numerical sentinel for cube W2 at
    the matrix-runner configuration (iter-44 2× hyperdiff).

    The iter-1002 sentinel
    (``test_iter1002_w2_v_ll_linf_meets_target``) tests cube W2
    1-day at the iter-1030 dual-target calibration with
    ``hyperdiff_coeff=0`` — pinning the codex-iter-985..1030
    calibration narrative.  iter-49 adds a complementary sentinel
    that tests cube W2 2-day at the matrix-runner-actual config
    (iter-44 hyperdiff_coeff=2 × _hyperdiff_cube(n)) so a
    regression to the matrix-runner SW cube path also has a
    runtime check (the AST gate sentinel pins the SOURCE; this
    pins the OUTPUT).

    Asserts:
      - finite h, u_d
      - mass_drift < 1e-7
      - max|v_d - v_d_init| < 3.0 m/s  (iter-44 measured 0.96 m/s
        at day 5; at day 2 < 1 m/s; 3.0 is conservative ceiling)
    """
    import jax
    import jax.numpy as jnp_local
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState,
        iter1009_dual_target_config,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid,
    )
    from tests.test_cases.williamson import williamson_test2
    import warnings

    n = 36
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    def _hyperdiff_cube_local(nn, ref_n=48, ref_coeff=1e16):
        return ref_coeff * (ref_n / nn) ** 4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = iter1009_dual_target_config(
            n, hyperdiff_coeff=2.0 * _hyperdiff_cube_local(n),
        )
    model = FV3EdgeShallowWaterModel(grid, cfg)

    sw = williamson_test2(grid)
    u0 = 2.0 * jnp_local.pi * grid.radius / (12.0 * 86400.0)
    u_d = cdgrid.cos_angle_edge_x * (
        u0 * jnp_local.cos(cdgrid.lat_edge_x))
    v_d_init = -cdgrid.sin_angle_edge_y * (
        u0 * jnp_local.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d_init, h_s=sw.h_s.data,
    )
    model.set_initial_mass(state)

    dt = 300.0
    area64 = grid.area.astype(jnp_local.float64)
    mass_init = float(
        jnp_local.sum(state.h.astype(jnp_local.float64) * area64),
    )

    @jax.jit
    def step(s):
        return model.step(s, dt)

    s = state
    for _ in range(576):  # 2 days
        s = step(s)

    assert jnp_local.all(jnp_local.isfinite(s.h)), (
        "iter-49 regression: cube W2 day-2 ``h`` non-finite"
    )
    assert jnp_local.all(jnp_local.isfinite(s.u_d)), (
        "iter-49 regression: cube W2 day-2 ``u_d`` non-finite"
    )
    mass_final = float(
        jnp_local.sum(s.h.astype(jnp_local.float64) * area64),
    )
    drift = abs(mass_final - mass_init) / abs(mass_init)
    assert drift < 1e-7, (
        f"iter-49 regression: cube W2 day-2 mass drift {drift:.2e} "
        "exceeds 1e-7"
    )
    # W2 is steady state — v_d error indicates calibration drift.
    v_d_err = float(
        jnp_local.max(jnp_local.abs(s.v_d - v_d_init)),
    )
    assert v_d_err < 3.0, (
        f"iter-49 regression: cube W2 day-2 ``max|v_d - v_d_init|`` "
        f"= {v_d_err:.4f} m/s exceeds 3.0 m/s ceiling — likely "
        "hyperdiff coefficient regression (iter-44's 2x dropped "
        "back toward 0x or 0)."
    )
    # iter-53: also bound the height-field error.  W2 is steady-
    # state so h should stay close to ``sw.h.data``.  iter-44 matrix
    # W2 5-day shows L2=4.58e-4 (cube/latlon ratio 1.7x).  Day-2
    # subset should sit well under L2=3e-3 — would only exceed if
    # hyperdiff or calibration regressed.
    h_init = sw.h.data
    area = grid.area.astype(jnp_local.float64)
    h_err_l2 = float(
        jnp_local.sqrt(
            jnp_local.sum(
                ((s.h - h_init).astype(jnp_local.float64) ** 2)
                * area,
            )
            / jnp_local.sum(h_init.astype(jnp_local.float64) ** 2 * area)
        ),
    )
    assert h_err_l2 < 3e-3, (
        f"iter-53 regression: cube W2 day-2 h-field L2 = "
        f"{h_err_l2:.4e} exceeds 3e-3 ceiling — iter-44 measured "
        "5-day L2=4.58e-4."
    )


def test_iter48_cube_w5_short_run_stable_with_hyperdiff():
    """new_test_dycores iter-48: numerical regression test for the
    iter-33 cube W5 stability fix (extending iter-39's W6 sentinel).

    Cube W5 (mountain) 15-day BLEW UP at day 14.58 pre-iter-33.
    iter-33 added ``hyperdiff_coeff=_hyperdiff_cube(n)``; iter-44
    bumped to 2x.  This test runs cube W5 for 576 steps (2 days at
    dt=300 s) with the 2x hyperdiff override and asserts finite +
    mass-clean + max|u_d| < 200 m/s (matrix BLOWUP threshold is
    1000 m/s).
    """
    import jax
    import jax.numpy as jnp_local
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState,
        iter1009_dual_target_config,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        create_cubed_sphere_cdgrid,
    )
    from tests.test_cases.williamson import williamson_test5
    import warnings

    n = 36
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)

    def _hyperdiff_cube_local(nn, ref_n=48, ref_coeff=1e16):
        return ref_coeff * (ref_n / nn) ** 4

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = iter1009_dual_target_config(
            n, hyperdiff_coeff=2.0 * _hyperdiff_cube_local(n),
        )
    model = FV3EdgeShallowWaterModel(grid, cfg)

    sw = williamson_test5(grid)
    u0 = 20.0  # W5 standard
    u_d = cdgrid.cos_angle_edge_x * (
        u0 * jnp_local.cos(cdgrid.lat_edge_x))
    v_d = -cdgrid.sin_angle_edge_y * (
        u0 * jnp_local.cos(cdgrid.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    model.set_initial_mass(state)

    dt = 300.0
    area64 = grid.area.astype(jnp_local.float64)
    mass_init = float(
        jnp_local.sum(state.h.astype(jnp_local.float64) * area64),
    )

    @jax.jit
    def step(s):
        return model.step(s, dt)

    s = state
    for _ in range(576):  # 2 days at dt=300 s
        s = step(s)

    assert jnp_local.all(jnp_local.isfinite(s.h)), (
        "iter-48 regression: cube W5 day-2 ``h`` non-finite"
    )
    assert jnp_local.all(jnp_local.isfinite(s.u_d)), (
        "iter-48 regression: cube W5 day-2 ``u_d`` non-finite"
    )
    mass_final = float(
        jnp_local.sum(s.h.astype(jnp_local.float64) * area64),
    )
    drift = abs(mass_final - mass_init) / abs(mass_init)
    assert drift < 1e-7, (
        f"iter-48 regression: cube W5 day-2 mass drift {drift:.2e} "
        "exceeds 1e-7"
    )
    u_d_max = float(jnp_local.max(jnp_local.abs(s.u_d)))
    assert u_d_max < 200.0, (
        f"iter-48 regression: cube W5 day-2 ``max|u_d|`` = "
        f"{u_d_max:.1f} m/s exceeds 200 m/s — wind blowup "
        "(matrix BLOWUP threshold is 1000 m/s)."
    )


def test_iter35_hyperdiff_coeff_kwarg_threads_through():
    """new_test_dycores iter-35: ``iter1009_dual_target_config``
    accepts ``hyperdiff_coeff`` as an optional kwarg that threads
    through to the returned ``CDGridShallowWaterConfig.hyperdiff_coeff``
    field unchanged.

    The matrix-runner SW W5/W6 cube paths rely on this kwarg to add
    biharmonic hyperdiffusion that prevents long-run BLOWUP at day
    14.58 (W5) / day 9.03 (W6).

    iter-37 (review fix): also pin the default to exactly 0.0 so a
    future change that flips the default to nonzero would trip this
    test without needing the matrix runner to flag the W2 v_ll
    sentinel regression.
    """
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        iter1009_dual_target_config,
    )
    N = 36
    # Default: 0.0 (preserves iter-1030 backward compat).  W2 cube
    # path relies on this default being 0.0 — see W2 sentinel
    # ``test_iter1002_w2_v_ll_linf_meets_target``.
    default_cfg = iter1009_dual_target_config(N)
    assert default_cfg.hyperdiff_coeff == 0.0, (
        f"iter-35 regression: helper default hyperdiff_coeff != 0.0 "
        f"(got {default_cfg.hyperdiff_coeff!r}).  This would silently "
        "change the W2 cube path behaviour."
    )
    # Kwarg threads through.
    custom = iter1009_dual_target_config(N, hyperdiff_coeff=1.234e16)
    assert custom.hyperdiff_coeff == 1.234e16, (
        f"iter-35 regression: hyperdiff_coeff kwarg did not thread "
        f"through (got {custom.hyperdiff_coeff!r}, want 1.234e16)"
    )
    # Other fields stay at iter-1030 defaults.
    assert custom.damp_v == 0.030
    assert custom.div_damp == default_cfg.div_damp


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
