"""Metal (Apple Silicon GPU) backend validation for the RCE stack.

Per CLAUDE.md:
- "Apple Silicon: keep spectral work on CPU."
- "Cross-backend discrepancies: inspect dtype assumptions,
  x64 requirements, unsupported kernels, communication semantics."

Item 11/13 of the CRM RCE capability series. Validates that the
plane-CRM RCE helpers added in PRs #298–#307 run end-to-end on
the Metal backend and produce results numerically consistent with
the CPU reference within float32 tolerance.

Metal-specific caveats checked here:
* Float64 (x64) is NOT supported on Metal. The RCE helpers must
  work at float32. We run the test ONLY in float32 to keep the
  comparison honest.
* No spectral routine is exercised (Apple Silicon spectral routing
  is forced to CPU by `legoesm.parallel.metal.ensure_spectral_on_cpu`
  — those tests live elsewhere).
* JIT must compile without host-callback fallback (Metal's XLA path
  rejects host callbacks). We rely on each helper's documented
  pure-JAX implementation.

Skipped on non-Metal hardware. CPU CI still runs the same tests
under the `JAX_PLATFORMS=cpu` config (the skip guard fires there).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    make_flat_plane_terrain_metric, make_rest_state,
)
from legoesm.atmosphere.dynamics.shared.mean_wind_filter import (
    remove_horizontal_mean_wind,
)
from legoesm.atmosphere.dynamics.crm.moist_mass_fixer import (
    compute_total_water_mass_plane, fix_moist_mass_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
    cloud_fraction_profile_plane, column_moist_static_energy_plane,
    column_water_vapor_plane, precipitation_rate_proxy_plane,
)
from legoesm.atmosphere.dynamics.crm.rce_surface_flux import (
    apply_rce_surface_fluxes, wind_speed_at_lowest_level_plane,
)
from legoesm.atmosphere.dynamics.shared.tracer_positivity import (
    apply_positive_filter_state, clip_positive,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.metal import is_metal_backend, to_cpu, to_metal


@pytest.fixture(autouse=True)
def _pin_float32_precision():
    """Pin x64-OFF / fp32 policy for every test in this module.

    Each test here is float32 / Metal-oriented and asserts float32 dtype
    preservation.  When another module that flips ``jax_enable_x64`` ON at
    import shares the same xdist worker, the leaked global x64 flag would
    silently promote these helpers' outputs to float64 (e.g. a Python-float
    constant inside ``column_water_vapor_plane`` becomes float64), false-
    failing the dtype assertions.  Snapshot the global precision state,
    force the float32 policy with x64 off for the test body, and restore the
    snapshot at teardown.  In a fresh isolated process this is a no-op (the
    default is already x64-off / fp32).
    """
    from legoesm.core.precision import (
        PrecisionPolicy, get_policy, set_policy,
    )

    x64_before = jax.config.jax_enable_x64
    policy_before = get_policy()
    jax.config.update("jax_enable_x64", False)
    set_policy(PrecisionPolicy.fp32())
    try:
        yield
    finally:
        set_policy(policy_before)
        jax.config.update("jax_enable_x64", x64_before)


_ACCELERATOR_PLATFORMS = ("gpu", "cuda", "rocm", "mps")


def _accelerator_device():
    """Return the first functional accelerator device, or None."""
    if is_metal_backend():
        # On the Apple GPU (mps) backend, devices() returns the
        # MpsDevice.
        try:
            for d in jax.devices():
                if d.platform == "mps":
                    return d
        except Exception:
            return None
    try:
        for d in jax.devices():
            if d.platform in ("gpu", "cuda", "rocm"):
                return d
    except Exception:
        return None
    return None


def _has_functional_accelerator() -> bool:
    """True iff an accelerator device is actually usable (Metal that
    didn't fall back, or any GPU vendor)."""
    return _accelerator_device() is not None


accelerator_required = pytest.mark.skipif(
    not _has_functional_accelerator(),
    reason="No functional accelerator (Metal/GPU) — CPU-only CI.",
)
# Backwards-compat alias for the original Metal-only test names.
metal_required = accelerator_required


def _assert_no_host_callbacks(lowered) -> None:
    """Codex iter-2: scan lowered XLA HLO for host-callback ops.
    Metal rejects host callbacks; falling silently to CPU defeats
    the GPU-validation purpose."""
    ir = str(lowered.compiler_ir(dialect="stablehlo"))
    for needle in (
        "xla_python_cpu_callback", "io_callback", "host_callback",
        "PyCpuCallback",
    ):
        assert needle not in ir, (
            f"host-callback op {needle!r} found in compiled IR — "
            f"would fall back to CPU on Metal/GPU and silently "
            f"miss the validation."
        )


def _put_on_accelerator(pytree, dev):
    """device_put every leaf of pytree onto dev. Mirrors to_metal /
    to_cpu pattern from legoesm.parallel.metal."""
    return jax.tree_util.tree_map(
        lambda x: (
            jax.device_put(x, dev) if hasattr(x, "dtype") else x
        ),
        pytree,
    )


def _assert_on_accelerator(arr, dev) -> None:
    """Assert leaf array is placed on the accelerator device."""
    if not hasattr(arr, "devices"):
        return  # Python scalar — skip.
    arr_devs = arr.devices()
    assert dev in arr_devs, (
        f"expected output on accelerator device {dev}, got {arr_devs}"
    )

NLEV, NY, NX = 6, 4, 4


def _setup_f32():
    """Construct a float32 plane state — Metal-compatible dtype."""
    grid = create_plane_grid(
        nx=NX, ny=NY, nlev=NLEV, dx=2_000.0, dy=2_000.0,
        dtype=jnp.float32,
    )
    hc = create_height_coordinate(NLEV, H=6_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    state = make_rest_state(grid, hc, dtype=jnp.float32)
    tracers = jnp.zeros((NY, NX, NLEV, 3), dtype=jnp.float32)
    tracers = tracers.at[..., 0].set(jnp.float32(0.01))
    tracers = tracers.at[..., 1].set(jnp.float32(0.001))
    state = state._replace(
        tracers=state.tracers.replace(data=tracers),
    )
    return state, grid, hc, tm


# ----------------------------------------------------------------------
# Smoke tests: each helper runs on Metal without host-callback / dtype
# rejection.
# ----------------------------------------------------------------------

@metal_required
def test_cwv_runs_on_metal():
    """Codex iter-2: explicit device_put + output-device assertion."""
    dev = _accelerator_device()
    state, grid, hc, _ = _setup_f32()
    acc_state = _put_on_accelerator(state, dev)
    acc_hc = _put_on_accelerator(hc, dev)
    out = column_water_vapor_plane(acc_state, acc_hc)
    _assert_on_accelerator(out, dev)
    assert out.shape == (NY, NX)
    assert out.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(out)))


@metal_required
def test_mse_runs_on_metal():
    dev = _accelerator_device()
    state, grid, hc, _ = _setup_f32()
    acc_state = _put_on_accelerator(state, dev)
    acc_hc = _put_on_accelerator(hc, dev)
    out = column_moist_static_energy_plane(acc_state, acc_hc)
    _assert_on_accelerator(out, dev)
    assert out.shape == (NY, NX)
    assert out.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(out)))


@metal_required
def test_cloud_fraction_runs_on_metal():
    dev = _accelerator_device()
    state, grid, hc, _ = _setup_f32()
    acc_state = _put_on_accelerator(state, dev)
    acc_hc = _put_on_accelerator(hc, dev)
    cf = cloud_fraction_profile_plane(acc_state, acc_hc)
    _assert_on_accelerator(cf, dev)
    assert cf.shape == (NLEV,)
    assert bool(jnp.all(jnp.isfinite(cf)))


@metal_required
def test_precip_proxy_runs_on_metal():
    dev = _accelerator_device()
    state, grid, hc, _ = _setup_f32()
    acc_state = _put_on_accelerator(state, dev)
    acc_hc = _put_on_accelerator(hc, dev)
    p = precipitation_rate_proxy_plane(acc_state, acc_hc)
    _assert_on_accelerator(p, dev)
    assert p.shape == (NY, NX)
    assert bool(jnp.all(jnp.isfinite(p)))


@metal_required
def test_surface_flux_runs_on_metal():
    dev = _accelerator_device()
    state, grid, hc, _ = _setup_f32()
    wspd = wind_speed_at_lowest_level_plane(state)
    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float32)
    q_sfc = jnp.full((NY, NX), 0.022, dtype=jnp.float32)
    acc_state = _put_on_accelerator(state, dev)
    acc_hc = _put_on_accelerator(hc, dev)
    new_state = apply_rce_surface_fluxes(
        acc_state, acc_hc, dt=1.0,
        T_sfc=jax.device_put(T_sfc, dev),
        q_sfc=jax.device_put(q_sfc, dev),
        wind_speed=jax.device_put(wspd, dev),
    )
    _assert_on_accelerator(new_state.theta_prime.data, dev)
    assert bool(jnp.all(jnp.isfinite(new_state.theta_prime.data)))
    assert bool(jnp.all(jnp.isfinite(new_state.tracers.data)))


@metal_required
def test_moist_mass_fixer_runs_on_metal():
    dev = _accelerator_device()
    state, grid, hc, _ = _setup_f32()
    acc_state = _put_on_accelerator(state, dev)
    acc_hc = _put_on_accelerator(hc, dev)
    acc_grid = _put_on_accelerator(grid, dev)
    target = compute_total_water_mass_plane(acc_state, acc_hc, acc_grid)
    new_state = fix_moist_mass_plane(acc_state, acc_hc, acc_grid, target)
    _assert_on_accelerator(new_state.tracers.data, dev)
    np.testing.assert_allclose(
        float(compute_total_water_mass_plane(new_state, acc_hc, acc_grid)),
        float(target),
        rtol=1.0e-5,
    )


@metal_required
def test_mean_wind_runs_on_metal():
    dev = _accelerator_device()
    state, grid, hc, _ = _setup_f32()
    rng = np.random.default_rng(0)
    state = state._replace(
        u=state.u.replace(
            data=jnp.asarray(
                rng.standard_normal(state.u.data.shape), dtype=jnp.float32,
            ),
        ),
    )
    acc_state = _put_on_accelerator(state, dev)
    new_state = remove_horizontal_mean_wind(acc_state)
    _assert_on_accelerator(new_state.u.data, dev)
    np.testing.assert_allclose(
        np.asarray(jnp.mean(new_state.u.data, axis=(0, 1))),
        0.0, atol=1.0e-6,
    )


@metal_required
def test_tracer_positivity_runs_on_metal():
    dev = _accelerator_device()
    q_neg = jax.device_put(
        jnp.array([-1.0, 0.0, 2.0], dtype=jnp.float32), dev,
    )
    out = clip_positive(q_neg)
    _assert_on_accelerator(out, dev)
    np.testing.assert_array_equal(
        np.asarray(out), np.array([0.0, 0.0, 2.0], dtype=np.float32),
    )


# ----------------------------------------------------------------------
# CPU vs Metal numerical agreement at float32.
# ----------------------------------------------------------------------

@metal_required
def test_cwv_cpu_vs_metal_agreement():
    """Cross-backend agreement within float32 round-off (Codex iter-2:
    explicit device_put + output-device assertion to prevent
    false-pass on a CPU-default-with-visible-accelerator runner)."""
    dev = _accelerator_device()
    state, _, hc, _ = _setup_f32()
    acc_state = _put_on_accelerator(state, dev)
    acc_hc = _put_on_accelerator(hc, dev)
    out_acc = column_water_vapor_plane(acc_state, acc_hc)
    _assert_on_accelerator(out_acc, dev)
    cpu_state = jax.tree_util.tree_map(to_cpu, state)
    cpu_hc = jax.tree_util.tree_map(to_cpu, hc)
    out_cpu = column_water_vapor_plane(cpu_state, cpu_hc)
    np.testing.assert_allclose(
        np.asarray(to_cpu(out_acc)),
        np.asarray(out_cpu),
        rtol=1.0e-5, atol=1.0e-6,
    )


@metal_required
def test_surface_flux_cpu_vs_metal_agreement():
    """Surface flux update must agree across backends within float32.
    Codex iter-2: explicit device_put + output-device assertion."""
    dev = _accelerator_device()
    state, _, hc, _ = _setup_f32()
    wspd = wind_speed_at_lowest_level_plane(state)
    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float32)
    q_sfc = jnp.full((NY, NX), 0.022, dtype=jnp.float32)
    acc_state = _put_on_accelerator(state, dev)
    acc_hc = _put_on_accelerator(hc, dev)
    acc_wspd = jax.device_put(wspd, dev)
    acc_T = jax.device_put(T_sfc, dev)
    acc_q = jax.device_put(q_sfc, dev)
    acc_out = apply_rce_surface_fluxes(
        acc_state, acc_hc, dt=1.0,
        T_sfc=acc_T, q_sfc=acc_q, wind_speed=acc_wspd,
    )
    _assert_on_accelerator(acc_out.theta_prime.data, dev)
    cpu_state = jax.tree_util.tree_map(to_cpu, state)
    cpu_hc = jax.tree_util.tree_map(to_cpu, hc)
    cpu_out = apply_rce_surface_fluxes(
        cpu_state, cpu_hc, dt=1.0,
        T_sfc=to_cpu(T_sfc), q_sfc=to_cpu(q_sfc),
        wind_speed=to_cpu(wspd),
    )
    np.testing.assert_allclose(
        np.asarray(to_cpu(acc_out.theta_prime.data)),
        np.asarray(cpu_out.theta_prime.data),
        rtol=1.0e-5, atol=1.0e-6,
    )


# ----------------------------------------------------------------------
# JIT smoke under Metal — confirms no host-callback fallback.
# ----------------------------------------------------------------------

@metal_required
def test_apply_surface_fluxes_jit_on_metal():
    """Codex iter-2: inspect compiled IR for host-callback ops BEFORE
    running. Metal/GPU silently lower host callbacks via CPU fallback
    if XLA permits — IR inspection catches the regression."""
    dev = _accelerator_device()
    state, _, hc, _ = _setup_f32()
    wspd = wind_speed_at_lowest_level_plane(state)
    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float32)
    q_sfc = jnp.full((NY, NX), 0.022, dtype=jnp.float32)
    acc_state = _put_on_accelerator(state, dev)
    acc_wspd = jax.device_put(wspd, dev)
    acc_T = jax.device_put(T_sfc, dev)
    acc_q = jax.device_put(q_sfc, dev)
    fn = jax.jit(lambda s, T, q, w: apply_rce_surface_fluxes(
        s, hc, dt=1.0, T_sfc=T, q_sfc=q, wind_speed=w,
    ))
    _assert_no_host_callbacks(fn.lower(acc_state, acc_T, acc_q, acc_wspd))
    out = fn(acc_state, acc_T, acc_q, acc_wspd)
    _assert_on_accelerator(out.theta_prime.data, dev)
    assert bool(jnp.all(jnp.isfinite(out.theta_prime.data)))


@metal_required
def test_apply_positive_filter_state_jit_on_metal():
    """Codex iter-2: IR inspection for host callbacks."""
    dev = _accelerator_device()
    state, _, hc, _ = _setup_f32()
    state = state._replace(
        tracers=state.tracers.replace(
            data=state.tracers.data.at[0, 0, 0, 0].set(jnp.float32(-1.0)),
        ),
    )
    acc_state = _put_on_accelerator(state, dev)
    fn = jax.jit(lambda s: apply_positive_filter_state(
        s, tracer_slots_to_filter=(0,), mode="clip",
    ))
    _assert_no_host_callbacks(fn.lower(acc_state))
    out = fn(acc_state)
    assert float(jnp.min(out.tracers.data[..., 0])) >= 0.0


# ----------------------------------------------------------------------
# GPU-READINESS tests — active on CPU + Metal alike. Verify the RCE
# helpers all work at float32 (the GPU-compatible dtype), even when no
# functional GPU is present on the runner. Catches accidental float64-
# only code paths before GPU CI sees them.
# ----------------------------------------------------------------------

def test_cwv_float32_preserves_dtype():
    """float32 input → float32 output (no silent x64 promotion)."""
    state, _, hc, _ = _setup_f32()
    out = column_water_vapor_plane(state, hc)
    assert out.dtype == jnp.float32


def test_mse_float32_preserves_dtype():
    state, _, hc, _ = _setup_f32()
    out = column_moist_static_energy_plane(state, hc)
    assert out.dtype == jnp.float32


def test_surface_flux_float32_preserves_dtype():
    state, _, hc, _ = _setup_f32()
    wspd = wind_speed_at_lowest_level_plane(state)
    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float32)
    q_sfc = jnp.full((NY, NX), 0.022, dtype=jnp.float32)
    new_state = apply_rce_surface_fluxes(
        state, hc, dt=1.0, T_sfc=T_sfc, q_sfc=q_sfc,
        wind_speed=wspd,
    )
    assert new_state.theta_prime.data.dtype == jnp.float32
    assert new_state.tracers.data.dtype == jnp.float32


def test_moist_mass_fixer_float32_preserves_dtype():
    state, grid, hc, _ = _setup_f32()
    target = compute_total_water_mass_plane(state, hc, grid)
    new_state = fix_moist_mass_plane(state, hc, grid, target)
    assert new_state.tracers.data.dtype == jnp.float32


def test_mean_wind_filter_float32_preserves_dtype():
    state, _, hc, _ = _setup_f32()
    new_state = remove_horizontal_mean_wind(state)
    assert new_state.u.data.dtype == jnp.float32
    assert new_state.v.data.dtype == jnp.float32


def test_full_rce_step_jit_float32_compiles_no_host_callbacks():
    """Composite RCE step JIT-compiles at float32 AND emits no
    host-callback ops (Codex iter-2). Accelerator backends would
    fall back to CPU silently on host callbacks; checking the
    lowered IR on CPU catches the regression early."""
    state, grid, hc, _ = _setup_f32()
    target = compute_total_water_mass_plane(state, hc, grid)
    T_sfc = jnp.full((NY, NX), 300.0, dtype=jnp.float32)
    q_sfc = jnp.full((NY, NX), 0.022, dtype=jnp.float32)

    @jax.jit
    def step(s):
        wspd = wind_speed_at_lowest_level_plane(s)
        s = apply_rce_surface_fluxes(
            s, hc, dt=1.0, T_sfc=T_sfc, q_sfc=q_sfc, wind_speed=wspd,
        )
        s = remove_horizontal_mean_wind(s)
        s = apply_positive_filter_state(
            s, tracer_slots_to_filter=(0, 1, 2), mode="clip",
        )
        s = fix_moist_mass_plane(s, hc, grid, target)
        return s

    _assert_no_host_callbacks(step.lower(state))
    out = step(state)
    assert out.theta_prime.data.dtype == jnp.float32
    assert out.tracers.data.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(out.theta_prime.data)))
