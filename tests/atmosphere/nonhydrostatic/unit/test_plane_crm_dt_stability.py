"""dt-stability regression test for the plane CRM dycore.

Pins the bare-dycore stability boundary documented in
``CRM_implementation.md`` (finding F1):

    dt = 0.5 s -> stable  (max|w| < 0.05 m/s through 100 steps)
    dt = 1.0 s -> stable  (max|w| < 0.05 m/s through 100 steps)
    dt = 1.5 s -> growing instability
    dt = 2.0 s -> blows up by step 70 (max|w| > 50 m/s)

Configuration: 48x48 nlev=30 dx=2 km H=33 km, Wing 2018 theta_ref,
warm bubble (0.5 K, z<1 km, cosine taper), SI acoustic with beta=0.1,
hyperdiff=5e6, Smag c_s=0.2, sponge_width=10 km / sponge_coeff=0.05.

48x48 matches the F1 measurement mesh. Smaller meshes (24x24) do
not contain enough horizontal modes to support the documented
buoyancy/w mode and underestimate growth rates.

This test guards against regressions in:

* Acoustic substep semi-implicit tridiagonal coefficients
* Outer SSP-RK3 stage weights
* Buoyancy / pressure-gradient sign conventions
* Hyperdiff stencil for u, v, theta', rho', w
* Sponge profile evaluation
* Smagorinsky strain tensor
* Mass fixer's anchored-mass branch
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_theta_ref_fn,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate

jax.config.update("jax_enable_x64", True)


def _build_model(dt: float, n_acoustic: int = None):
    if n_acoustic is None:
        # Scale acoustic substeps with dt to keep dt_substep ~ 0.04 s.
        n_acoustic = max(12, int(round(dt * 24)))
    nx = ny = 48
    nlev = 30
    H = 33_000.0
    dx = 2_000.0
    grid = create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=dx, dy=dx, dtype=jnp.float64,
    )
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=300.0, q_sfc=0.0224, z_t=15_000.0, Gamma=6.7e-3,
    )
    hc = create_height_coordinate(
        n_levels=nlev, H=H, theta_ref_fn=theta_fn,
    )
    terrain = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05,
        sponge_width=10_000.0,
        hyperdiff_coeff=5.0e6,
        hyperdiff_rho_coeff=5.0e6,
        hyperdiff_w_coeff=5.0e6,
        semi_implicit_acoustic=True,
        acoustic_off_centering=0.1,
        use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
        n_acoustic_substeps=n_acoustic,
    )
    return PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg), grid, hc


def _seed_warm_bubble(state, hc, grid, amp_k: float = 0.5):
    ny, nx = grid.ny, grid.nx
    jj = jnp.arange(ny)
    ii = jnp.arange(nx)
    yy, xx = jnp.meshgrid(jj, ii, indexing="ij")
    yc, xc = (ny - 1) / 2.0, (nx - 1) / 2.0
    r_cells = jnp.sqrt((yy - yc) ** 2 + (xx - xc) ** 2)
    r0 = min(10.0, 0.5 * min(ny, nx))
    horiz = jnp.where(
        r_cells < r0,
        0.5 * (1.0 + jnp.cos(jnp.pi * r_cells / r0)),
        0.0,
    )
    z = hc.z_full
    z_top = 1_000.0
    vert = jnp.where(z < z_top,
                     0.5 * (1.0 + jnp.cos(jnp.pi * z / z_top)),
                     0.0)
    bubble_theta = amp_k * horiz[:, :, None] * vert[None, None, :]
    bubble_rho = -hc.rho_ref * bubble_theta / hc.theta_ref
    return state._replace(
        theta_prime=state.theta_prime.replace(data=bubble_theta),
        rho_prime=state.rho_prime.replace(data=bubble_rho),
    )


def _run_steps(dt: float, n_steps: int = 100):
    model, grid, hc = _build_model(dt)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    state = _seed_warm_bubble(state, hc, grid, amp_k=0.5)
    for _ in range(n_steps):
        state = model.step(state, dt=dt, physics_fn=None)
    return float(jnp.max(jnp.abs(state.w.data)))


# Caps from F1 measurements at 48x48 nlev=30 dx=2 km H=33 km.
# F1 results at step 100:
#   dt=0.5 -> max|w| ~ 4e-3 m/s
#   dt=1.0 -> max|w| ~ 8e-3 m/s
#   dt=1.5 -> max|w| ~ 4.3 m/s
#   dt=2.0 -> max|w| ~ 180 m/s
STABLE_MAX_W = 0.1      # m/s — bare-dycore stable cap, well above dt<=1 measured
GROWING_MAX_W = 1.0     # m/s — minimum to confirm growing-instability regime
BLOWUP_MAX_W = 50.0     # m/s — minimum to confirm full blow-up at dt=2


@pytest.mark.parametrize("dt", [0.5, 1.0])
def test_bare_dycore_stable_at_or_below_1s(dt):
    """Bare dycore + warm bubble IC must keep max|w| < 0.1 m/s
    through 100 steps for dt <= 1.0 s."""
    max_w = _run_steps(dt=dt, n_steps=100)
    assert max_w < STABLE_MAX_W, (
        f"dt={dt}s: max|w|={max_w:.3e} m/s exceeds stable cap "
        f"{STABLE_MAX_W} m/s. Regression in dycore stability — see "
        "CRM_implementation.md F1 + recent commits touching "
        "compressible_euler_plane.py / compressible_euler.py."
    )


def test_bare_dycore_growing_at_1p5s():
    """At dt=1.5 s the dycore must show the documented mode growth.
    A *too-stable* result here means damping has been over-applied
    (e.g. an unintended hyperdiff boost) — also a regression we
    want to catch."""
    max_w = _run_steps(dt=1.5, n_steps=100)
    assert max_w > GROWING_MAX_W, (
        f"dt=1.5s: max|w|={max_w:.3e} m/s, below growing-instability "
        f"threshold {GROWING_MAX_W} m/s. Documentation in F1 says "
        "this regime should be visibly unstable. If damping has been "
        "deliberately increased to fix the production blow-up, "
        "update F1 + the test thresholds."
    )


def test_bare_dycore_blows_up_at_2s():
    """At dt=2 s the documented blow-up must be reproduced.
    Regression here typically means the dt limit has shifted —
    update F1 + relax this test only if the new limit is verified."""
    max_w = _run_steps(dt=2.0, n_steps=100)
    assert max_w > BLOWUP_MAX_W or not jnp.isfinite(max_w), (
        f"dt=2.0s: max|w|={max_w:.3e} m/s, below blow-up threshold "
        f"{BLOWUP_MAX_W} m/s. F1 says this regime should blow up by "
        "step 70."
    )


def _run_steps_no_bubble(dt: float, n_steps: int = 100):
    """F10 path: same setup as ``_run_steps`` but with the CLEAN
    Wing IC (no warm bubble, no qv noise) — matches the production
    `--bubble-theta-pert 0` default."""
    model, grid, hc = _build_model(dt)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    for _ in range(n_steps):
        state = model.step(state, dt=dt, physics_fn=None)
    return float(jnp.max(jnp.abs(state.w.data)))


# F10 finding (iter-9, 2026-05): without the warm bubble IC the bare
# dycore is BIT-stable at dt much larger than the bubble-driven F1
# ladder. Measured max|w| at step 100 stays below ~1e-12 (round-off
# only) for dt up to 10 s. Production default lifted 1 -> 5 s.
F10_CLEAN_MAX_W = 1e-10  # m/s — round-off-only cap


@pytest.mark.parametrize("dt", [2.0, 5.0, 10.0])
def test_bare_dycore_clean_ic_bit_stable_up_to_10s(dt):
    """F10: clean Wing IC (no bubble) — bare dycore must stay at
    round-off (max|w| < 1e-10 m/s) for dt up to 10 s. Regression
    here means a stability-relevant numerical change has crept in
    that breaks the production dt=5 s default; update F10 + this
    test if the regression is intentional."""
    max_w = _run_steps_no_bubble(dt=dt, n_steps=100)
    assert max_w < F10_CLEAN_MAX_W, (
        f"dt={dt}s with clean Wing IC: max|w|={max_w:.3e} m/s "
        f"exceeds F10 round-off cap {F10_CLEAN_MAX_W} m/s. F10 says "
        "bare-dycore stays at round-off for dt<=10 s with this IC. "
        "If a stability-relevant change is intentional, update F10 "
        "and the test threshold; otherwise this is a regression."
    )


# --------------------------------------------------------------------------
# #82 (iter-61): substep_horizontal_acoustic=True LIFTS the dt-stability limit.
# The F1 "dt=2 s blows up" boundary above is ENTIRELY the
# substep_horizontal_acoustic=False default (the horizontal acoustic mode is
# integrated at the OUTER dt). With the full Skamarock-Klemp split, dt=2 AND
# dt=4 are STABLE (measured max|w| 0.015 / 0.025 m/s vs 182 m/s at dt=2 without
# it). GATE/LBA drivers + the RCE driver (iter-61) default this ON.
# --------------------------------------------------------------------------

def _build_model_substep(dt: float):
    """As ``_build_model`` but with the full horizontal-acoustic substepping."""
    n_acoustic = max(12, int(round(dt * 24)))
    nx = ny = 48
    nlev = 30
    grid = create_plane_grid(nx=nx, ny=ny, nlev=nlev, dx=2_000.0, dy=2_000.0,
                             dtype=jnp.float64)
    theta_fn = make_wing2018_theta_ref_fn(
        T_v0=300.0, q_sfc=0.0224, z_t=15_000.0, Gamma=6.7e-3)
    hc = create_height_coordinate(n_levels=nlev, H=33_000.0,
                                  theta_ref_fn=theta_fn)
    terrain = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=10_000.0,
        hyperdiff_coeff=5.0e6, hyperdiff_rho_coeff=5.0e6, hyperdiff_w_coeff=5.0e6,
        semi_implicit_acoustic=True, substep_horizontal_acoustic=True,
        acoustic_off_centering=0.1, use_coriolis=False,
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=0.2, smagorinsky_prandtl=1.0,
        n_acoustic_substeps=n_acoustic)
    return PlaneCompressibleEulerModel(grid, hc, terrain, config=cfg), grid, hc


@pytest.mark.parametrize("dt", [2.0, 4.0])
def test_substep_horizontal_acoustic_keeps_dt2_dt4_stable(dt):
    """#82 regression: with substep_horizontal_acoustic=True a 0.5 K warm bubble
    stays STABLE at dt=2 s AND dt=4 s — the dt limit the F1 path (substep OFF)
    blew up at. If this regresses, the iter-61 horizontal-acoustic substep fix
    is broken (the RCE driver would blow up at finite amplitude again)."""
    model, grid, hc = _build_model_substep(dt)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    state = _seed_warm_bubble(state, hc, grid, amp_k=0.5)
    for _ in range(100):
        state = model.step(state, dt=dt, physics_fn=None)
    max_w = float(jnp.max(jnp.abs(state.w.data)))
    assert jnp.isfinite(max_w) and max_w < STABLE_MAX_W, (
        f"dt={dt}s with substep_horizontal_acoustic=True: max|w|={max_w:.3e} "
        f"m/s exceeds {STABLE_MAX_W} — the #82 horizontal-acoustic substep fix "
        "regressed (without it dt=2 blows up to ~182 m/s).")
