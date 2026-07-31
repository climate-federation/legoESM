"""Differentiability tests for JAX transforms (jit / vmap / scan / checkpoint).

CATEGORY 10 of the differentiability test plan
(``.claude/agents/test-differentiability.md``).

Verifies that ``jax.grad`` interacts correctly with the standard JAX
transformations across *several* representative dynamical cores (not just
one), so a transform regression on any single grid family is caught.
Components exercised:

* CD-grid (cubed-sphere FV3) shallow water  — raw-array state
* lat-lon Arakawa C-grid shallow water      — raw-array state
* spectral (Gaussian/T5) shallow water      — Field-wrapped complex state
* MPAS (Voronoi) shallow water              — Field-wrapped unstructured state
* slab land surface (MOST bulk flux)        — NON-dycore component, so the
  transform battery also covers the surface stack and its ``fori_loop``
  flux iteration rather than dynamical cores alone

Catches:
* JIT tracing errors (Python side-effects leaking through ``jax.jit``)
* vmap shape errors / collapsed batching
* scan-carry dtype mismatches over long horizons
* checkpointing-induced gradient drift
* mixed-precision regressions in ``lax.cond`` / ``lax.scan`` branches

Sub-categories:
  10a) JIT compilation of gradients              (per component)
  10b) vmap over ensemble members                (per component)
  10c) lax.scan gradient accumulation N=1,5,20,50 (per component)
  10d) Gradient checkpointing                     (per component)
  10e) Mixed precision (fp32 vs fp64)            (CD-grid SW)

Each component is wrapped behind a small adapter that exposes a uniform
``(loss_fn, x0)`` contract so the same transform assertions run against
every grid family.  Field-wrapped states differentiate w.r.t. the
``.data`` array per the .md "Key pattern".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

# Enable x64 unconditionally for this module — the JIT/grad equivalence
# and checkpoint-vs-plain comparisons here require fp64 to keep the
# kernel-level numerical noise below the comparison tolerances we use.
# When the test runner does *not* set ``JAX_ENABLE_X64=1``, JAX silently
# truncates fp64 requests in source code, which leaks fp32 results into
# fp64 assertions.
jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.field import Field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assert_gradient_ok(grad_array, name="", min_nonzero_frac=0.1):
    """Finite + has spatial structure (>= min_nonzero_frac entries non-zero)."""
    grad_array = jnp.asarray(grad_array)
    assert jnp.all(jnp.isfinite(grad_array)), f"{name}: gradient has NaN/Inf"
    nonzero_frac = jnp.mean(jnp.abs(grad_array) > 0).item()
    assert nonzero_frac >= min_nonzero_frac, (
        f"{name}: only {nonzero_frac * 100:.1f}% non-zero "
        f"(need {min_nonzero_frac * 100:.0f}%)"
    )


# ---------------------------------------------------------------------------
# Component adapters
# ---------------------------------------------------------------------------
#
# Each adapter is a callable ``() -> dict`` returning:
#   step      : (state, dt) -> state        the model step
#   state     : a small initial state
#   dt        : float timestep
#   x0        : the leaf array we differentiate w.r.t.
#   loss      : x -> scalar                  loss(x) wraps x back into state
#   loss_n    : (x, n_steps) -> scalar       n-step scan version
#   batch     : (key, n_ens) -> array        a batch of perturbed x0's
#   readout   : state -> array               scalar-reduction target field
#
# All adapters share the *same* transform assertions below.

def _make_cdgrid_sw():
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterModel,
        CDGridShallowWaterConfig,
        CDGridShallowWaterState,
    )

    n = 4
    base = create_cubed_sphere(n)
    model = CDGridShallowWaterModel(base, CDGridShallowWaterConfig())
    h0 = 1000.0 * jnp.ones((6, n, n))
    h0 = h0 + 10.0 * jax.random.normal(jax.random.PRNGKey(0), (6, n, n))
    state = CDGridShallowWaterState(
        h=h0,
        u_d=jnp.zeros((6, n + 1, n + 1)),
        v_d=jnp.zeros((6, n + 1, n + 1)),
        h_s=jnp.zeros((6, n, n)),
    )
    dt = 60.0

    def step(s, _dt):
        return model.step(s, _dt)

    def wrap(x):
        return state._replace(h=x)

    def readout(s):
        return s.h

    return _build_adapter(state, state.h, dt, step, wrap, readout)


def _make_latlon_cgrid_sw():
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterModel,
        CGridLatLonShallowWaterConfig,
        CGridLatLonShallowWaterState,
    )

    n_lat, n_lon = 8, 16
    grid = create_latlon_grid(n_lat, n_lon)
    dt = 120.0
    model = CGridLatLonShallowWaterModel(grid, CGridLatLonShallowWaterConfig(), dt=dt)
    h0 = 1000.0 * jnp.ones((n_lat, n_lon))
    h0 = h0 + 10.0 * jax.random.normal(jax.random.PRNGKey(10), (n_lat, n_lon))
    state = CGridLatLonShallowWaterState(
        h=h0,
        u=jnp.zeros((n_lat, n_lon + 1)),
        v=jnp.zeros((n_lat + 1, n_lon)),
        h_s=jnp.zeros((n_lat, n_lon)),
    )

    def step(s, _dt):
        return model.step(s, _dt)

    def wrap(x):
        return state._replace(h=x)

    def readout(s):
        return s.h

    return _build_adapter(state, state.h, dt, step, wrap, readout)


def _make_spectral_sw():
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
        SpectralShallowWaterModel,
        SpectralSWConfig,
        SpectralSWState,
    )

    grid = create_gaussian_grid(5, allow_unsupported_backend=True)
    model = SpectralShallowWaterModel(
        grid, SpectralSWConfig(), allow_unsupported_backend=True
    )
    dt = 120.0
    n_sh = grid.n_sh
    H0 = 5960.0
    phi_hat = jnp.zeros(n_sh, dtype=jnp.complex128)
    phi_hat = phi_hat.at[0].set(constants.g * H0 * jnp.sqrt(4 * jnp.pi))
    phi_hat = phi_hat + 1.0 * jax.random.normal(jax.random.PRNGKey(2), (n_sh,))
    state = SpectralSWState(
        vor_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="vor_hat"),
        div_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="div_hat"),
        phi_hat=Field(phi_hat, name="phi_hat"),
        phis_hat=Field(jnp.zeros(n_sh, dtype=jnp.complex128), name="phis_hat"),
    )
    # We differentiate w.r.t. the *real* part of phi_hat; the imag part is
    # carried as a constant so the loss stays a real-input/real-output map.
    x0 = state.phi_hat.data.real
    phi_imag = state.phi_hat.data.imag

    def step(s, _dt):
        return model.step(s, _dt)

    def wrap(phi_real):
        phi_new = phi_real + 1j * phi_imag
        return state._replace(phi_hat=state.phi_hat.replace(data=phi_new))

    def readout(s):
        # complex magnitude -> a real, finite-magnitude reduction target
        return jnp.abs(s.phi_hat.data)

    return _build_adapter(state, x0, dt, step, wrap, readout)


def _make_mpas_sw():
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
        MPASShallowWaterModel,
        MPASShallowWaterConfig,
    )
    from legoesm.core.state import MPASShallowWaterState

    mesh = create_voronoi_mesh(2)
    model = MPASShallowWaterModel(mesh, MPASShallowWaterConfig())
    dt = 120.0
    h_data = 1000.0 * jnp.ones(mesh.nCells)
    h_data = h_data + 10.0 * jax.random.normal(jax.random.PRNGKey(3), (mesh.nCells,))
    state = MPASShallowWaterState(
        h=Field(h_data, name="h"),
        u=Field(jnp.zeros(mesh.nEdges), name="u"),
        h_s=Field(jnp.zeros(mesh.nCells), name="h_s"),
    )

    def step(s, _dt):
        return model.step(s, _dt)

    def wrap(x):
        return state._replace(h=state.h.replace(data=x))

    def readout(s):
        return s.h.data

    return _build_adapter(state, state.h.data, dt, step, wrap, readout)


def _make_slab_land():
    """Non-dycore component: the slab land surface.

    Every other adapter is an atmospheric dynamical core, so the whole
    transform battery (jit / vmap / scan / checkpoint) was previously
    blind to the surface stack — which is where the ``lax.fori_loop``
    MOST bulk-flux iteration and the snow/bucket branches live.  Its
    per-step arithmetic is cheap, so adding it costs almost nothing.
    """
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import LandConfig
    from legoesm.land.slab_land import step_land
    from legoesm.land.state import LandState

    ncol = 16
    # Spatially varying forcing so the gradient carries structure.
    ramp = jnp.linspace(0.9, 1.1, ncol)
    forcing = AtmToSurface(
        sw_down=200.0 * ramp, lw_down=300.0 * ramp,
        precip_total=1e-5 * ramp, precip_snow=0.0 * ramp,
        T_lowest=280.0 * ramp, q_lowest=5e-3 * ramp,
        u_lowest=5.0 * ramp, v_lowest=2.0 * ramp,
        p_lowest=1e5 * ramp, p_surface=1.013e5 * ramp,
        rho_lowest=1.2 * ramp, cos_zenith=0.7 * ramp,
        co2_ppmv=400.0 * ramp, has_radiation=jnp.ones(ncol),
        has_precipitation=jnp.ones(ncol),
    )
    # MOST bulk scheme -> the flux solve runs its fori_loop iteration.
    config = LandConfig(bulk_scheme="most")
    T0 = 280.0 + 2.0 * jax.random.normal(jax.random.PRNGKey(21), (ncol,))
    state = LandState(
        T_soil=Field(T0, name="T_soil"),
        W_bucket=Field(jnp.full(ncol, 50.0), name="W_bucket"),
        snow_depth=Field(jnp.zeros(ncol), name="snow_depth"),
        snow_age=Field(jnp.zeros(ncol), name="snow_age"),
    )
    dt = 600.0

    def step(s, _dt):
        out, _resp, *_rest = step_land(s, forcing, config, U_min=1.0, dt=_dt)
        return out

    # ``LandState.runoff`` / ``.TgC`` are OPTIONAL fields that default to
    # ``None`` but are POPULATED by ``step_land``.  A hand-built initial
    # state therefore has a different pytree structure from the step
    # output, and ``lax.scan`` rejects the carry ("carry input and carry
    # output must have the same pytree structure").  Priming the state
    # with one step gives a fixed point of the structure, so the same
    # adapter works for the 1-step and N-step (scan) losses alike.
    state = step(state, dt)

    def wrap(x):
        return state._replace(T_soil=state.T_soil.replace(data=x))

    def readout(s):
        return s.T_soil.data

    return _build_adapter(state, state.T_soil.data, dt, step, wrap, readout)


def _build_adapter(state, x0, dt, step, wrap, readout):
    """Assemble the uniform adapter contract from component primitives."""

    def loss(x):
        return jnp.sum(readout(step(wrap(x), dt)) ** 2)

    def loss_n(x, n_steps):
        s = wrap(x)

        def body(carry, _):
            return step(carry, dt), None

        s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
        return jnp.sum(readout(s_final) ** 2)

    def batch(key, n_ens):
        keys = jax.random.split(key, n_ens)
        scale = 5.0
        return jnp.stack([
            x0 + scale * jax.random.normal(k, x0.shape).astype(x0.dtype)
            for k in keys
        ])

    return {
        "state": state,
        "x0": x0,
        "dt": dt,
        "step": step,
        "wrap": wrap,
        "readout": readout,
        "loss": loss,
        "loss_n": loss_n,
        "batch": batch,
    }


# Registry of component adapters.  Built lazily (module-scoped cache) so
# grid construction (~hundreds of ms each) is amortised across tests.
_ADAPTER_FACTORIES = {
    "cdgrid_sw": _make_cdgrid_sw,
    "latlon_cgrid_sw": _make_latlon_cgrid_sw,
    "spectral_sw": _make_spectral_sw,
    "mpas_sw": _make_mpas_sw,
    # Non-dycore component (surface stack + MOST fori_loop bulk flux).
    "slab_land": _make_slab_land,
}

_ADAPTER_CACHE: dict[str, dict] = {}


@pytest.fixture(scope="module")
def adapter(request):
    name = request.param
    if name not in _ADAPTER_CACHE:
        _ADAPTER_CACHE[name] = _ADAPTER_FACTORIES[name]()
    return _ADAPTER_CACHE[name]


_ALL_COMPONENTS = list(_ADAPTER_FACTORIES.keys())


def _param_all():
    return pytest.mark.parametrize("adapter", _ALL_COMPONENTS, indirect=True)


# ===========================================================================
# 10a  JIT compilation of gradients  (per component)
# ===========================================================================

@_param_all()
class TestJitGrad:
    """``jax.jit(jax.grad(loss))`` must produce the same gradient as
    ``jax.grad(loss)`` — catches tracing errors and Python side-effects
    that leak through the JIT boundary."""

    def test_jit_grad_matches_eager(self, adapter):
        loss, x0 = adapter["loss"], adapter["x0"]
        grad_eager = jax.grad(loss)(x0)
        grad_jit = jax.jit(jax.grad(loss))(x0)
        assert_gradient_ok(grad_eager, "eager grad")
        assert_gradient_ok(grad_jit, "jit grad")
        assert jnp.allclose(grad_jit, grad_eager, atol=1e-10, rtol=1e-10), (
            "jit(grad) and grad disagree — JIT tracing is altering the "
            "gradient computation"
        )

    def test_value_and_grad_under_jit(self, adapter):
        loss, x0 = adapter["loss"], adapter["x0"]
        val_e, grad_e = jax.value_and_grad(loss)(x0)
        val_j, grad_j = jax.jit(jax.value_and_grad(loss))(x0)
        assert jnp.isfinite(val_e) and jnp.isfinite(val_j)
        assert jnp.allclose(val_j, val_e, atol=1e-8, rtol=1e-10)
        assert jnp.allclose(grad_j, grad_e, atol=1e-10, rtol=1e-10)


# ===========================================================================
# 10b  vmap over ensemble members  (per component)
# ===========================================================================

@_param_all()
class TestVmapEnsemble:
    """``jax.vmap(jax.grad(loss))`` over a batch of states must give a
    finite gradient for every member with the right shape."""

    def test_vmap_over_ensemble(self, adapter):
        loss, x0, batch = adapter["loss"], adapter["x0"], adapter["batch"]
        n_ens = 4
        batch_x = batch(jax.random.PRNGKey(1), n_ens)
        batched_grad = jax.vmap(jax.grad(loss))(batch_x)
        assert batched_grad.shape == (n_ens,) + x0.shape
        for i in range(n_ens):
            assert_gradient_ok(batched_grad[i], f"ensemble member {i}")

    def test_vmap_grads_distinct(self, adapter):
        """Distinct ICs must yield distinct gradients — a sanity check
        that vmap is batching, not broadcasting one computation."""
        loss, batch = adapter["loss"], adapter["batch"]
        b = batch(jax.random.PRNGKey(2), 2)
        gs = jax.vmap(jax.grad(loss))(b)
        assert not jnp.allclose(gs[0], gs[1]), (
            "vmap collapsed distinct ensemble members to identical gradients"
        )


# ===========================================================================
# 10c  lax.scan gradient accumulation N=1, 5, 20, 50  (per component)
# ===========================================================================

@_param_all()
class TestScanAccumulation:
    """Gradient through ``jax.lax.scan`` must remain finite over long
    horizons.  Catches gradient explosion / vanishing in dycore
    composition + scan-carry dtype mismatches."""

    @pytest.mark.parametrize("n_steps", [1, 5, 20, 50])
    def test_grad_finite_for_n_steps(self, adapter, n_steps):
        loss_n, x0 = adapter["loss_n"], adapter["x0"]
        grad = jax.grad(lambda x: loss_n(x, n_steps))(x0)
        assert_gradient_ok(grad, f"scan N={n_steps}")

    def test_grad_changes_with_horizon(self, adapter):
        """Gradients at N=1 and N=20 must differ — catches the silent
        ``lax.cond`` / ``lax.stop_gradient`` failure where the gradient
        gets truncated past step 1.

        Magnitude alone is a bad signal: every rest-state SW gradient is
        dominated by a near-constant *background* term that scan barely
        moves.  On grid-point cores that background is the uniform
        ``2·h_mean`` (removed by subtracting the array mean); on the
        spectral core it is the single mean-depth coefficient
        (``g·H0·√(4π)`` at SH index 0, magnitude ~4·10^5) which a mean
        subtraction does NOT remove.  We therefore strip BOTH: subtract
        the mean *and* mask the single dominant entry, then check the
        residual perturbation pattern genuinely evolves over the 20-step
        window.  ``rel_diff = 0`` would mean scan collapsed to step 1.
        """
        loss_n, x0 = adapter["loss_n"], adapter["x0"]
        g1 = jax.grad(lambda x: loss_n(x, 1))(x0)
        g20 = jax.grad(lambda x: loss_n(x, 20))(x0)

        # Mask the single dominant background coefficient (spectral
        # mean-depth mode) FIRST — subtracting the array mean while a
        # ~4·10^5 entry is still present would smear a large constant
        # offset across every small mode and inflate the denominator.
        # After masking, remove any residual uniform offset (the
        # grid-point cores' uniform ``2·h_mean`` background).
        dominant = jnp.argmax(jnp.abs(g1))
        keep = jnp.ones_like(g1).at[dominant].set(0.0)
        g1_k = g1 * keep
        g20_k = g20 * keep
        g1_bg = g1_k - jnp.mean(g1_k)
        g20_bg = g20_k - jnp.mean(g20_k)
        rel_diff = (
            jnp.linalg.norm(g20_bg - g1_bg)
            / (jnp.linalg.norm(g1_bg) + 1e-30)
        )
        assert rel_diff > 1e-3, (
            f"scan-N=20 perturbation gradient identical to N=1 "
            f"(rel_diff={float(rel_diff):.3e}) — scan may be silently "
            f"truncating the gradient chain"
        )


# ===========================================================================
# 10d  Gradient checkpointing  (per component)
# ===========================================================================

@_param_all()
class TestCheckpointing:
    """``jax.checkpoint`` must not change the *value* of the gradient,
    only the memory footprint."""

    def test_checkpoint_grad_matches_unckpt(self, adapter):
        x0, dt = adapter["x0"], adapter["dt"]
        step, wrap, readout = adapter["step"], adapter["wrap"], adapter["readout"]
        n_steps = 8

        def loss_plain(x):
            s = wrap(x)

            def body(carry, _):
                return step(carry, dt), None

            s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
            return jnp.sum(readout(s_final) ** 2)

        def loss_ckpt(x):
            s = wrap(x)
            ckpt_step = jax.checkpoint(lambda c: step(c, dt))

            def body(carry, _):
                return ckpt_step(carry), None

            s_final, _ = jax.lax.scan(body, s, None, length=n_steps)
            return jnp.sum(readout(s_final) ** 2)

        g_plain = jax.grad(loss_plain)(x0)
        g_ckpt = jax.grad(loss_ckpt)(x0)
        assert_gradient_ok(g_plain, "plain")
        assert_gradient_ok(g_ckpt, "checkpointed")
        assert jnp.allclose(g_plain, g_ckpt, atol=1e-8, rtol=1e-8), (
            "checkpointing changed the gradient values — should only affect "
            "memory, not numerical result"
        )


# ===========================================================================
# 10e  Mixed precision (fp32 vs fp64)
# ===========================================================================
#
# Restricted to the CD-grid SW core: the spectral path is inherently
# complex128 (x64-only by construction) and the mass-fixer accumulator in
# the lat-lon / cube cores promotes to fp64 internally, so a clean fp32
# vs fp64 contrast lives on the CD-grid float-state core.

class TestMixedPrecision:
    """fp32 and fp64 gradients must both be finite.  Catches dtype
    promotion bugs in ``lax.cond`` branches (e.g., one branch returns
    fp32 while the other returns fp64, breaking the cond's same-shape
    invariant under AD)."""

    @pytest.fixture(scope="class")
    def cdgrid(self):
        return _make_cdgrid_sw()

    def _grad_at_dtype(self, cdgrid, dtype):
        state, dt = cdgrid["state"], cdgrid["dt"]
        step, readout = cdgrid["step"], cdgrid["readout"]
        s_dtype = state._replace(
            h=state.h.astype(dtype),
            u_d=state.u_d.astype(dtype),
            v_d=state.v_d.astype(dtype),
            h_s=state.h_s.astype(dtype),
        )

        def loss(h_init):
            s = s_dtype._replace(h=h_init)
            return jnp.sum(readout(step(s, dt)) ** 2)

        return jax.grad(loss)(s_dtype.h)

    def test_grad_finite_in_fp64(self, cdgrid):
        g64 = self._grad_at_dtype(cdgrid, jnp.float64)
        assert g64.dtype == jnp.float64
        assert_gradient_ok(g64, "fp64")

    def test_grad_finite_in_fp32(self, cdgrid):
        g32 = self._grad_at_dtype(cdgrid, jnp.float32)
        assert g32.dtype == jnp.float32
        assert_gradient_ok(g32, "fp32")

    def test_fp32_fp64_signs_agree(self, cdgrid):
        """Different precisions may differ in magnitude, but the sign of
        each dominant gradient entry should agree with the fp64 ref."""
        g64 = self._grad_at_dtype(cdgrid, jnp.float64)
        g32 = self._grad_at_dtype(cdgrid, jnp.float32).astype(jnp.float64)
        thresh = jnp.quantile(jnp.abs(g64), 0.75)
        mask = jnp.abs(g64) > thresh
        sign_match = jnp.mean(jnp.sign(g32[mask]) == jnp.sign(g64[mask])).item()
        assert sign_match > 0.95, (
            f"fp32 vs fp64 gradient signs disagree on {(1 - sign_match) * 100:.1f}% "
            f"of dominant entries — possible dtype promotion bug in cond/scan"
        )
