"""Bit-faithfulness + conservation + AD tests for the FIELD-BATCHED ocean
backward-Euler vertical-mixing solve.

``LatLonCGridOceanModel._apply_implicit_vertical_mixing`` issues FOUR separate
``implicit_vertical_diffusion_ocean`` (Thomas) solves per step — one each for
T, S, u, v — eight serialised ``fori_loop`` while-loops that XLA cannot fuse
across.  An OPT-IN ``implicit_vertical_diffusion_ocean_batched`` path
(``LEGOESM_VMIX_BATCHED=1``) concatenates the four independent systems (they
share ``nlev`` on the last axis) along the column axis into a single
``thomas_solve_batched``.  It is the DEFAULT-OFF: A/B measurement (2026-06-10)
showed it helps small grids (LL128) but regresses at production resolution
(LL192) on both CPU and GPU — the vmix is memory-bandwidth-bound, so batching's
extra intermediate-array traffic dominates its dispatch saving at scale.  These
tests pin both paths: the model-level tests parametrize / force
``LEGOESM_VMIX_BATCHED`` so the BATCHED path is exercised (not just the scalar
default), and prove it is bit-faithful + conservation-correct + AD-safe vs the
4-separate solve.

These tests pin the #1 acceptance gate: the batched solve must equal the
four-separate solve to f64 roundoff (the per-field coefficients — K_v vs A_v,
dt vs dt_mom, the no-flux BCs, the face interpolations — must be carried
EXACTLY).  ``LEGOESM_TRIDIAG=legacy`` is forced so ``thomas_solve_batched`` is
``vmap(thomas_solve)`` on every backend → bit-identical to the four standalone
``thomas_solve`` calls (the CUDA PCR/cuSPARSE paths are equally exact direct
solvers but only machine-eps-equal, which would mask a real coefficient bug).

Run with::

    JAX_ENABLE_X64=1 LEGOESM_TRIDIAG=legacy .venv/bin/python -m pytest \
        tests/ocean/unit/test_implicit_vmix_batched.py -v
"""
from __future__ import annotations

import os

# thomas_solve_batched reads LEGOESM_TRIDIAG at JIT trace time and bakes the
# backend into the compiled graph.  HARD-force the legacy vmap(thomas_solve)
# path so the batched result is BIT-IDENTICAL to the four standalone
# thomas_solve calls on any backend (PCR/cuSPARSE are only machine-eps-equal,
# which would mask a real coefficient bug).  Use a hard assignment, NOT
# setdefault — the latter would silently leave an inherited LEGOESM_TRIDIAG=pcr
# / cusparse in place and exercise the wrong backend under assert_array_equal.
# Must be set before any legoesm import that may trace the solver.
os.environ["LEGOESM_TRIDIAG"] = "legacy"

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.core.precision import (  # noqa: E402
    PrecisionPolicy,
    get_policy,
    set_policy,
)
from legoesm.grids.latlon import create_latlon_grid  # noqa: E402
from legoesm.ocean.vertical import create_ocean_z_star  # noqa: E402
from legoesm.ocean.init_latlon_cgrid import (  # noqa: E402
    rest_state_latlon_cgrid_ocean,
)
from legoesm.ocean.state import LatLonCGridOceanConfig  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.physics.vertical_mixing import (  # noqa: E402
    implicit_vertical_diffusion_ocean,
    implicit_vertical_diffusion_ocean_batched,
    build_dz_half,
)


@pytest.fixture(autouse=True)
def _x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_model(n_lat=12, n_lon=24, n_lev=8, dt_mom_ratio=1.0):
    """A small lat-lon C-grid ocean model exercising the implicit vmix path."""
    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=n_lev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat(
        A_v=1e-3, K_v=1e-4,
        implicit_vertical_mixing=True,
        physics=None,                 # bench-default fallback K path
        dt_mom_ratio=dt_mom_ratio,
        # Async dt_mom!=dt_tracer requires the rigid lid (model
        # validate-strict guard: a moving free surface would leak
        # O((dt_tracer-dt_mom)*dh/dt) tracer mass). dt_mom_ratio=1.0 works
        # with any barotropic solver.
        barotropic_solver=("rigid_lid" if dt_mom_ratio != 1.0
                           else "explicit_substep"),
    )
    return LatLonCGridOceanModel(grid, z_coord, cfg), grid, z_coord


def _perturbed_rest_state(grid, z_coord, n_lev, seed=0):
    """Rest state with shear + stratification perturbations so the vertical
    diffusion has real work to do (otherwise every field is flat in z and the
    solve is a no-op that hides coefficient bugs)."""
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=2.0,
        S_uniform=35.0, H_max=4000.0,
    )
    rng = np.random.RandomState(seed)
    # Preserve each field's storage dtype (the default fp32 policy) so the
    # state stays dtype-uniform exactly as production — the batched path's
    # single work-dtype then equals every field's per-field result_type and
    # the bit-faithfulness comparison is meaningful.
    def _pert(field, scale):
        d = np.asarray(field.data) + rng.normal(0, scale, field.data.shape)
        return field.replace(data=jnp.asarray(d, dtype=field.data.dtype))
    return state._replace(
        T=_pert(state.T, 0.3),
        S=_pert(state.S, 0.05),
        u=_pert(state.u, 0.1),
        v=_pert(state.v, 0.1),
    )


# ---------------------------------------------------------------------------
# 1. Kernel-level bit-faithfulness: batched list == per-field separate
# ---------------------------------------------------------------------------
class TestKernelBitFaithful:
    def test_uniform_shape_fields(self):
        """Four same-shape fields, different K/dt → batched == 4 separate."""
        nlat, nlon, nlev = 5, 7, 9
        rng = np.random.RandomState(1)
        dz = jnp.asarray(np.linspace(50.0, 400.0, nlev))
        dzh = build_dz_half(dz)
        fields = [jnp.asarray(rng.normal(0, 1, (nlat, nlon, nlev)))
                  for _ in range(4)]
        Ks = [jnp.asarray(np.abs(rng.normal(1e-3, 5e-4, (nlat, nlon, nlev - 1))))
              for _ in range(4)]
        dts = [120.0, 120.0, 90.0, 90.0]

        sep = [implicit_vertical_diffusion_ocean(f, K, dz, dzh, dt)
               for f, K, dt in zip(fields, Ks, dts)]
        batched = implicit_vertical_diffusion_ocean_batched(
            [(f, K, dz, dzh, dt) for f, K, dt in zip(fields, Ks, dts)]
        )
        for a, b in zip(sep, batched):
            # f64 roundoff: legacy thomas_solve_batched == vmap(thomas_solve),
            # so this must be EXACT (no roundoff at all), assert array_equal.
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_heterogeneous_column_shapes(self):
        """Different LEADING shapes (mimics T/S vs u-face vs v-face) but the
        same nlev → still batches (concat along the flattened column axis)."""
        nlev = 6
        dz = jnp.asarray(np.linspace(30.0, 200.0, nlev))
        dzh = build_dz_half(dz)
        rng = np.random.RandomState(2)
        shapes = [(4, 4, nlev), (4, 5, nlev), (5, 4, nlev)]
        fields = [jnp.asarray(rng.normal(0, 1, s)) for s in shapes]
        Ks = [jnp.asarray(np.abs(rng.normal(2e-3, 1e-3, (*s[:-1], nlev - 1))))
              for s in shapes]
        dts = [100.0, 75.0, 75.0]
        sep = [implicit_vertical_diffusion_ocean(f, K, dz, dzh, dt)
               for f, K, dt in zip(fields, Ks, dts)]
        batched = implicit_vertical_diffusion_ocean_batched(
            [(f, K, dz, dzh, dt) for f, K, dt in zip(fields, Ks, dts)]
        )
        for s, a, b in zip(shapes, sep, batched):
            assert b.shape == s
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_mixed_f32_f64_per_field_dtype_faithful(self):
        """A f32-only system batched with a f64 system must stay bit-faithful
        to its OWN scalar solve — the helper groups by per-field work dtype so
        the f32 system is NEVER promoted to f64 (which would change its f32
        result bits).  This is the public-API contract the production path
        (dtype-uniform fields) does not exercise."""
        nlev = 7
        rng = np.random.RandomState(7)
        # f32 system: every input f32 -> scalar work dtype f32, output f32.
        f32 = jnp.asarray(rng.normal(0, 1, (2, 3, nlev)), dtype=jnp.float32)
        dz32 = jnp.asarray(np.linspace(20, 200, nlev), dtype=jnp.float32)
        dzh32 = build_dz_half(dz32)
        K32 = jnp.asarray(
            np.abs(rng.normal(1e-3, 5e-4, (2, 3, nlev - 1))), dtype=jnp.float32)
        # f64 system.
        f64 = jnp.asarray(rng.normal(0, 1, (2, 4, nlev)), dtype=jnp.float64)
        dz64 = jnp.asarray(np.linspace(20, 200, nlev), dtype=jnp.float64)
        dzh64 = build_dz_half(dz64)
        K64 = jnp.asarray(
            np.abs(rng.normal(1e-3, 5e-4, (2, 4, nlev - 1))), dtype=jnp.float64)

        sep32 = implicit_vertical_diffusion_ocean(f32, K32, dz32, dzh32, 900.0)
        sep64 = implicit_vertical_diffusion_ocean(f64, K64, dz64, dzh64, 900.0)
        bat32, bat64 = implicit_vertical_diffusion_ocean_batched([
            (f32, K32, dz32, dzh32, 900.0),
            (f64, K64, dz64, dzh64, 900.0),
        ])
        assert bat32.dtype == jnp.float32 and bat64.dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(sep32), np.asarray(bat32))
        np.testing.assert_array_equal(np.asarray(sep64), np.asarray(bat64))

    def test_scalar_K_and_1d_dz_broadcast(self):
        """Scalar K + 1-D dz must broadcast identically in the batched path."""
        nlev = 5
        dz = jnp.full((nlev,), 100.0)
        dzh = build_dz_half(dz)
        f = jnp.asarray(np.linspace(20.0, 5.0, nlev))
        sep = implicit_vertical_diffusion_ocean(f, 0.01, dz, dzh, 3600.0)
        (batched,) = implicit_vertical_diffusion_ocean_batched(
            [(f, 0.01, dz, dzh, 3600.0)]
        )
        np.testing.assert_array_equal(np.asarray(sep), np.asarray(batched))

    def test_empty_and_one_level_noops(self):
        assert implicit_vertical_diffusion_ocean_batched([]) == []
        f = jnp.array([[42.0]])  # nlev=1
        (out,) = implicit_vertical_diffusion_ocean_batched(
            [(f, jnp.zeros((1, 0)), jnp.array([10.0]), jnp.zeros((1, 0)), 600.0)]
        )
        np.testing.assert_array_equal(np.asarray(out), np.asarray(f))

    def test_rejects_mismatched_nlev(self):
        f1 = jnp.zeros((2, 5))
        f2 = jnp.zeros((2, 6))
        dz5 = jnp.full((5,), 10.0)
        dz6 = jnp.full((6,), 10.0)
        with pytest.raises(ValueError):
            implicit_vertical_diffusion_ocean_batched([
                (f1, 0.0, dz5, build_dz_half(dz5), 60.0),
                (f2, 0.0, dz6, build_dz_half(dz6), 60.0),
            ])

    def test_rejects_nonpositive_dt(self):
        f = jnp.zeros((2, 4))
        dz = jnp.full((4,), 10.0)
        with pytest.raises(ValueError):
            implicit_vertical_diffusion_ocean_batched(
                [(f, 0.0, dz, build_dz_half(dz), 0.0)]
            )


# ---------------------------------------------------------------------------
# 2. Model-level bit-faithfulness: the batched _apply_implicit_vertical_mixing
#    must equal a faithful 4-separate reconstruction of the SAME coefficients.
# ---------------------------------------------------------------------------
def _reference_four_separate(model, state, dt, dt_mom, surface_tracer_forcing=None):
    """Reconstruct the PRE-batching 4-separate solve from the model's own
    coefficient build, so we compare the new batched model path against the
    exact prior numerics (not against the batched kernel under test)."""
    from legoesm.ocean.physics.vertical_mixing import compute_vertical_K_profiles
    from legoesm.ocean.vertical import compute_ocean_jacobian
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        interp_cell_to_uface,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.eos import make_eos_fn

    cfg = model.config
    z = model.z_coord
    physics_config = OceanPhysicsConfig(
        vertical_mixing=VerticalMixingConfig(scheme="none"),
        convection=OceanConvectionConfig(scheme="none"),
    )
    u_cell = 0.5 * (state.u.data[:, :-1, :] + state.u.data[:, 1:, :])
    v_cell = 0.5 * (state.v.data[:-1, :, :] + state.v.data[1:, :, :])
    cc_state = state._replace(
        u=state.u.replace(data=u_cell), v=state.v.replace(data=v_cell),
    )
    eos_fn = make_eos_fn(eos=cfg.eos, eos_linear=cfg.eos_linear)
    K_v_cell, A_v_cell = compute_vertical_K_profiles(
        cc_state, z, None, physics_config,
        A_v_background=float(cfg.A_v), K_v_background=float(cfg.K_v),
        eos_fn=eos_fn,
    )
    J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z)
    dz_cell = z.dz_ref * J[..., jnp.newaxis]
    dz_half_cell = build_dz_half(dz_cell)
    mask_3d = state.land_mask.data[..., jnp.newaxis]

    K_v_cell = K_v_cell.astype(state.T.data.dtype)
    T_in, S_in = state.T.data, state.S.data
    if surface_tracer_forcing is not None:
        dT = surface_tracer_forcing.dT_dt.data.astype(state.T.data.dtype)
        dS = surface_tracer_forcing.dS_dt.data.astype(state.S.data.dtype)
        T_in = state.T.data + dt * dT * mask_3d
        S_in = state.S.data + dt * dS * mask_3d

    A_v_cell = A_v_cell.astype(state.u.data.dtype)
    A_v_u = interp_cell_to_uface(A_v_cell)
    A_v_v = interp_to_v_points(A_v_cell)
    dz_u = interp_cell_to_uface(dz_cell)
    dz_v = interp_to_v_points(dz_cell)

    T_new = implicit_vertical_diffusion_ocean(
        T_in, K_v_cell, dz_cell, dz_half_cell, dt)
    S_new = implicit_vertical_diffusion_ocean(
        S_in, K_v_cell, dz_cell, dz_half_cell, dt)
    u_new = implicit_vertical_diffusion_ocean(
        state.u.data, A_v_u, dz_u, build_dz_half(dz_u), dt_mom)
    v_new = implicit_vertical_diffusion_ocean(
        state.v.data, A_v_v, dz_v, build_dz_half(dz_v), dt_mom)
    u_mask = state.u_mask.data[..., jnp.newaxis]
    v_mask = state.v_mask.data[..., jnp.newaxis]
    T_new = jnp.where(mask_3d > 0.5, T_new, state.T.data)
    S_new = jnp.where(mask_3d > 0.5, S_new, state.S.data)
    u_new = jnp.where(u_mask > 0.5, u_new, state.u.data)
    v_new = jnp.where(v_mask > 0.5, v_new, state.v.data)
    return T_new, S_new, u_new, v_new


class TestModelBitFaithful:
    # Parametrize over the LEGOESM_VMIX_BATCHED gate so the model path is
    # exercised under BOTH the production default ("0" -> 4-separate) AND the
    # opt-in ("1" -> batched).  The default-off flip (2026-06-10) otherwise
    # leaves these model-level assertions comparing the scalar path against a
    # scalar reference (vacuous for the batched path) — codex final-review
    # finding.  ``_apply_implicit_vertical_mixing`` reads the env eagerly at
    # call time, so setting it before the call (fresh model per param) selects
    # the branch.  The reference is ALWAYS the explicit 4-separate
    # reconstruction, so "1" proves batched==4-separate (the real
    # bit-faithfulness) and "0" proves default==4-separate.
    @pytest.mark.parametrize("vmix_batched", ["0", "1"])
    @pytest.mark.parametrize("dt_mom_ratio", [1.0, 2.0])
    def test_apply_implicit_vmix_matches_four_separate(
        self, dt_mom_ratio, vmix_batched, monkeypatch,
    ):
        monkeypatch.setenv("LEGOESM_VMIX_BATCHED", vmix_batched)
        model, grid, z = _build_model(dt_mom_ratio=dt_mom_ratio)
        state = _perturbed_rest_state(grid, z, n_lev=8, seed=3)
        dt = 600.0
        dt_mom = dt / dt_mom_ratio

        out = model._apply_implicit_vertical_mixing(
            state, dt, None, K_v_phys=None, A_v_phys=None, K33_iso=None,
            dt_mom=dt_mom, surface_tracer_forcing=None,
        )
        T_ref, S_ref, u_ref, v_ref = _reference_four_separate(
            model, state, dt, dt_mom,
        )
        # Bit-identical (legacy batched solver == vmap(thomas_solve)).
        np.testing.assert_array_equal(np.asarray(out.T.data), np.asarray(T_ref))
        np.testing.assert_array_equal(np.asarray(out.S.data), np.asarray(S_ref))
        np.testing.assert_array_equal(np.asarray(out.u.data), np.asarray(u_ref))
        np.testing.assert_array_equal(np.asarray(out.v.data), np.asarray(v_ref))


# ---------------------------------------------------------------------------
# 3. Conservation: zero-flux BCs ⇒ column heat / salt integral unchanged.
# ---------------------------------------------------------------------------
class TestConservation:
    def test_column_tracer_integral_conserved(self):
        """Each batched tracer solve must conserve ∫ φ·dz per column (no-flux
        top + bottom) to f64 roundoff — vertical mixing redistributes, never
        creates/destroys, column heat & salt."""
        nlev = 10
        dz = jnp.asarray(np.linspace(20.0, 300.0, nlev))
        dzh = build_dz_half(dz)
        rng = np.random.RandomState(4)
        T = jnp.asarray(rng.normal(10.0, 5.0, (3, 4, nlev)))
        S = jnp.asarray(rng.normal(35.0, 1.0, (3, 4, nlev)))
        K = jnp.asarray(np.abs(rng.normal(0.05, 0.02, (3, 4, nlev - 1))))
        T_new, S_new = implicit_vertical_diffusion_ocean_batched([
            (T, K, dz, dzh, 3600.0),
            (S, K, dz, dzh, 3600.0),
        ])
        for before, after in ((T, T_new), (S, S_new)):
            col_before = jnp.sum(before * dz, axis=-1)
            col_after = jnp.sum(after * dz, axis=-1)
            np.testing.assert_allclose(
                np.asarray(col_after), np.asarray(col_before),
                rtol=1e-12, atol=1e-9,
            )

    @staticmethod
    def _model_step_col_integral_drift(policy: PrecisionPolicy, *,
                                       batched: bool = True):
        """Run the REAL model vmix under *policy* and return, per tracer,
        ``(max|Δ|, max|integral|, eps_work, per_column_drift)`` for the masked
        column integral ∫ φ·dz_cell·mask — the scalar max + magnitude for a
        relative roundoff bound, and the FULL per-column drift field so callers
        can compare the whole pattern (not just its max) between paths.

        The model (and every field it builds) follows the ACTIVE precision
        policy: ``rest_state_latlon_cgrid_ocean`` casts T/S/eta/H_bathy to
        ``policy.storage`` and ``z_coord.dz_ref`` is built in ``policy.control``
        (see ``ocean/vertical.create_ocean_z_star`` / ``init_latlon_cgrid``).
        So under ``fp32()`` the whole tridiagonal solve runs in float32 and the
        integral can only be conserved to the float32 Thomas-solve roundoff
        floor; under ``fp64()`` it runs in float64 and the discretisation's
        EXACT (telescoping-flux) conservation is recovered to f64 roundoff.

        ``batched=False`` sets ``LEGOESM_VMIX_BATCHED=0`` so
        ``_apply_implicit_vertical_mixing`` issues the PRE-batching 4-separate
        ``implicit_vertical_diffusion_ocean`` solves instead of the single
        batched solve.  Because the conservation property is a property of the
        SHARED ``_build_implicit_tridiag`` flux form + float32 Thomas sweep (not
        of the batching), the unbatched path has the IDENTICAL drift — proof
        the original failure was never a batching regression.  The env var is
        read at trace time inside the (here un-JITted, directly-called) function
        body, so toggling it per call works.

        ``dz_cell`` here is byte-identical to the dz the solver builds
        internally — ``z.dz_ref * J[..., None]`` with
        ``J = compute_ocean_jacobian(eta, H_bathy, z)`` — exactly
        ``ocean_model_latlon_cgrid._apply_implicit_vertical_mixing`` line
        ``dz_cell = self.z_coord.dz_ref * J_cell[..., jnp.newaxis]``.  That match
        is WHY this measures the integral the discretisation actually conserves
        (a mismatched dz would show an O(field·dz) drift, far above any roundoff
        floor, so this check is non-vacuous).
        """
        from legoesm.ocean.vertical import compute_ocean_jacobian

        orig = get_policy()
        orig_env = os.environ.get("LEGOESM_VMIX_BATCHED")
        set_policy(policy)
        os.environ["LEGOESM_VMIX_BATCHED"] = "1" if batched else "0"
        try:
            # Build model + state UNDER the policy so every field's dtype
            # follows it (storage for T/S/eta/H_bathy, control for dz_ref).
            model, grid, z = _build_model(n_lev=10)
            state = _perturbed_rest_state(grid, z, n_lev=10, seed=5)
            J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, z)
            # dz_cell == the solver's own dz (same formula, same line in
            # _apply_implicit_vertical_mixing) so we weight by exactly the dz
            # the flux-form discretisation conserves.
            dz_cell = z.dz_ref * J[..., jnp.newaxis]
            mask = state.land_mask.data[..., jnp.newaxis]
            out = model._apply_implicit_vertical_mixing(
                state, 600.0, None, K_v_phys=None, A_v_phys=None, K33_iso=None,
                dt_mom=600.0, surface_tracer_forcing=None,
            )
            drifts = {}
            for name, before, after in (
                ("T", state.T.data, out.T.data),
                ("S", state.S.data, out.S.data),
            ):
                col_b = jnp.sum(before * dz_cell * mask, axis=-1)
                col_a = jnp.sum(after * dz_cell * mask, axis=-1)
                per_col_drift = np.asarray(col_a - col_b)   # (n_lat, n_lon)
                max_abs_diff = float(np.max(np.abs(per_col_drift)))
                max_col_mag = float(jnp.max(jnp.abs(col_b)))
                work_eps = float(jnp.finfo(before.dtype).eps)
                # Full per-column drift field returned too, so callers can
                # compare the WHOLE pattern (not just its max) — two different
                # drift maps with the same max must NOT look identical.
                drifts[name] = (max_abs_diff, max_col_mag, work_eps,
                                per_col_drift)
            return drifts
        finally:
            set_policy(orig)
            if orig_env is None:
                os.environ.pop("LEGOESM_VMIX_BATCHED", None)
            else:
                os.environ["LEGOESM_VMIX_BATCHED"] = orig_env

    def test_model_step_tracer_integral_conserved_fp64(self):
        """Through the real model call UNDER THE fp64 PRECISION POLICY: the
        masked column heat/salt integral ∫ φ·dz_cell·mask is preserved by the
        batched implicit vmix to f64 roundoff.

        This is the load-bearing conservation gate.  The backward-Euler
        flux-form discretisation conserves Σ_k φ_k·dz_k EXACTLY in exact
        arithmetic: multiplying tridiag row k by dz_k turns α_k/β_k into the
        dz_k-free interface fluxes dt·K_{k∓1/2}/dz_half_{k∓1/2}
        (implicit_solver.py:171-173 — ``inv_dz = 1/dz_arr`` cancels the dz_k
        weight), so the column sum telescopes to the top+bottom interface
        fluxes, which the no-flux BCs (K_{-1/2}=K_{N-1/2}=0, encoded by the
        zero pads at implicit_solver.py:168-169) set to zero.  The land_mask is
        2-D (uniform per column), so masking a fully-wet column is the identity
        and introduces no wet/dry-interface flux leak.  Hence under f64 the
        integral is conserved to ~1e-11, the tight tolerance this asserts.
        """
        drifts = self._model_step_col_integral_drift(PrecisionPolicy.fp64())
        for name, (max_abs_diff, max_col_mag, _eps, _drift) in drifts.items():
            # rtol·|integral| + atol, matching np.testing.assert_allclose's
            # mixed bound — exact-arithmetic conservation ⇒ only f64 roundoff.
            tol = 1e-11 * max_col_mag + 1e-8
            assert max_abs_diff <= tol, (
                f"{name}: f64 column-integral drift {max_abs_diff:.3e} exceeds "
                f"{tol:.3e} (|integral|={max_col_mag:.3e}); the flux-form vmix "
                f"should conserve ∫φ·dz to f64 roundoff"
            )

    def test_model_step_tracer_integral_conserved_fp32_roundoff_floor(self):
        """Through the real model call UNDER THE DEFAULT fp32 PRECISION POLICY:
        the masked column integral is conserved to the float32 Thomas-solve
        ROUNDOFF FLOOR, NOT to 1e-8.

        The default policy is all-float32 (PrecisionPolicy.fp32 / get_policy
        default), so ``thomas_solve`` runs its forward+backward sweeps in
        float32 (work dtype = result_type(a,b,c,d), every input f32).  The
        discretisation is still EXACTLY conservative in exact arithmetic
        (see the fp64 test), but the float32 Thomas sweep injects O(eps_f32)
        relative roundoff into the redistributed column, so Δ(∫φ·dz) is bounded
        by ~eps_f32·|∫φ·dz|, NOT the f64 1e-8 the original test demanded — the
        original failure (max |Δ|≈3.9e-3 ≈ 0.4·eps_f32·|∫T·dz|, |∫T·dz|≈8e4)
        was this float32 floor, not a discretisation/dz-weighting bug.

        Asserting the DERIVED machine-precision floor (a small multiple of
        eps_f32·|integral|) keeps this non-vacuous: a genuine flux-form or
        dz-mismatch bug would drift by O(field·dz) ≫ eps_f32·|integral| and
        still fail here.
        """
        drifts = self._model_step_col_integral_drift(PrecisionPolicy.fp32())
        for name, (max_abs_diff, max_col_mag, eps32, _drift) in drifts.items():
            # Machine-precision floor for a single float32 Thomas solve:
            # the redistribution touches the whole column, so the conserved
            # sum drifts by at most a few ulp of the column-integral magnitude.
            # Factor 8 covers the per-level fma/division roundoff accumulation
            # across the forward+backward sweep (nlev=10) with margin; it is
            # still ~10⁶× below the O(field·dz)~1e4 a real non-conservation
            # bug would produce.
            tol = 8.0 * eps32 * max_col_mag
            assert max_abs_diff <= tol, (
                f"{name}: f32 column-integral drift {max_abs_diff:.3e} exceeds "
                f"the float32 Thomas roundoff floor {tol:.3e} "
                f"(eps_f32={eps32:.2e}, |integral|={max_col_mag:.3e}) — drift "
                f"this large is a real non-conservation bug, not f32 roundoff"
            )

    def test_unbatched_path_has_identical_conservation(self):
        """The pre-batching 4-separate solve (LEGOESM_VMIX_BATCHED=0) has the
        SAME column-integral conservation behaviour as the batched solve — at
        BOTH precisions — proving the original 1e-8 conservation failure was a
        property of the (shared) flux-form discretisation + float32 Thomas
        sweep, NOT a regression introduced by the field-batching.

        Under f64 the unbatched path conserves to f64 roundoff (telescoping
        flux form, exact); under f32 it lands on the SAME machine-precision
        floor.  Equality of the drifts (to f64 roundoff under fp64, exactly
        under the legacy backend forced at module top) confirms batching is
        conservation-neutral, as the bit-faithfulness tests already pin for the
        field values themselves.
        """
        bat64 = self._model_step_col_integral_drift(
            PrecisionPolicy.fp64(), batched=True)
        unb64 = self._model_step_col_integral_drift(
            PrecisionPolicy.fp64(), batched=False)
        for name in ("T", "S"):
            u_diff, u_mag, _eps, u_drift = unb64[name]
            _b_diff, _b_mag, _b_eps, b_drift = bat64[name]
            # (a) the unbatched path conserves to f64 roundoff too;
            assert u_diff <= 1e-11 * u_mag + 1e-8, (
                f"{name}: unbatched f64 drift {u_diff:.3e} not at f64 roundoff "
                f"(|integral|={u_mag:.3e})"
            )
            # (b) batched and unbatched conservation drifts agree across the
            # FULL per-column field (not just its max — two different drift
            # maps with the same max must not pass), confirming batching is
            # conservation-neutral pointwise.  LEGOESM_TRIDIAG is forced
            # 'legacy' at module top, so batched == vmap(thomas_solve) == the 4
            # separate thomas_solve calls bit-for-bit; the field values are
            # identical, so the only residual is summation-order roundoff in the
            # integral reduction itself (tiny, f64).
            np.testing.assert_allclose(
                u_drift, b_drift, rtol=1e-9, atol=1e-9,
            )
        # f32 floor: the unbatched path must also satisfy the SAME derived
        # roundoff floor the batched fp32 test asserts.
        unb32 = self._model_step_col_integral_drift(
            PrecisionPolicy.fp32(), batched=False)
        for name in ("T", "S"):
            u_diff, u_mag, eps32, _drift = unb32[name]
            assert u_diff <= 8.0 * eps32 * u_mag, (
                f"{name}: unbatched f32 drift {u_diff:.3e} exceeds the float32 "
                f"Thomas roundoff floor {8.0 * eps32 * u_mag:.3e}"
            )


# ---------------------------------------------------------------------------
# 4. Differentiability: the batched solve must be AD-safe (the ocean step is
#    differentiated in training) — gradients finite, and match a finite diff.
# ---------------------------------------------------------------------------
class TestDifferentiable:
    def test_grad_finite_through_batched_kernel(self):
        nlev = 6
        dz = jnp.asarray(np.linspace(40.0, 200.0, nlev))
        dzh = build_dz_half(dz)
        K = jnp.asarray(np.abs(np.linspace(1e-3, 5e-3, nlev - 1)))

        def loss(field):
            (out,) = implicit_vertical_diffusion_ocean_batched(
                [(field, K, dz, dzh, 600.0)]
            )
            return jnp.sum(out ** 2)

        f0 = jnp.asarray(np.linspace(20.0, 5.0, nlev).reshape(1, nlev))
        g = jax.grad(loss)(f0)
        assert np.all(np.isfinite(np.asarray(g)))
        # Finite-difference check on one component.
        eps = 1e-4
        i = (0, 2)
        fp = f0.at[i].add(eps)
        fm = f0.at[i].add(-eps)
        fd = float((loss(fp) - loss(fm)) / (2 * eps))
        np.testing.assert_allclose(float(g[i]), fd, rtol=1e-5, atol=1e-6)

    def test_grad_through_model_apply(self, monkeypatch):
        """jax.grad through the real model batched vmix is finite (the ocean
        step is differentiated in training).  Force the opt-in batched path
        (default is now off) so this actually differentiates the batched
        solve, not the scalar default."""
        monkeypatch.setenv("LEGOESM_VMIX_BATCHED", "1")
        model, grid, z = _build_model(n_lev=6)
        state = _perturbed_rest_state(grid, z, n_lev=6, seed=6)

        def loss(T):
            st = state._replace(T=state.T.replace(data=T))
            out = model._apply_implicit_vertical_mixing(
                st, 600.0, None, K_v_phys=None, A_v_phys=None, K33_iso=None,
                dt_mom=600.0, surface_tracer_forcing=None,
            )
            return jnp.sum(out.T.data ** 2) + jnp.sum(out.u.data ** 2)

        g = jax.grad(loss)(state.T.data)
        assert np.all(np.isfinite(np.asarray(g)))
        assert float(jnp.sum(jnp.abs(g))) > 0.0


# ===========================================================================
# T+S shared-factor pair solve (one matrix, two RHS) — bit-faithful.
# ===========================================================================


class TestPairSolveBitFaithful:
    """``implicit_vertical_diffusion_ocean_pair`` == two separate
    ``thomas_solve`` calls on the same prebuilt bands, bitwise (shared forward
    factors are arithmetic-identical per RHS).  The single-field solve builds
    its bands inside the sweep instead, so it agrees with the pair to
    rounding, not bitwise."""

    @staticmethod
    def _prebuilt_single(f, K, dz, dz_half, dt):
        from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
            _build_implicit_tridiag,
        )
        from legoesm.timestepping.tridiagonal import thomas_solve
        return thomas_solve(*_build_implicit_tridiag(f, K, dz, dz_half, dt))

    def _problem(self, dtype, seed=0, shape=(6, 8), nlev=12):
        rng = np.random.default_rng(seed)
        f1 = jnp.asarray(
            rng.standard_normal(shape + (nlev,)), dtype=dtype)
        f2 = jnp.asarray(
            rng.standard_normal(shape + (nlev,)), dtype=dtype)
        K = jnp.asarray(
            np.abs(rng.standard_normal(shape + (nlev - 1,))) * 1e-3,
            dtype=dtype)
        dz = jnp.asarray(
            1.0 + np.abs(rng.standard_normal((nlev,))), dtype=dtype)
        dz_half = 0.5 * (dz[1:] + dz[:-1])
        return f1, f2, K, dz, dz_half

    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    def test_pair_matches_two_singles_bitwise(self, dtype):
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean,
            implicit_vertical_diffusion_ocean_pair,
        )
        f1, f2, K, dz, dz_half = self._problem(dtype)
        dt = 900.0
        x1, x2 = implicit_vertical_diffusion_ocean_pair(
            f1, f2, K, dz, dz_half, dt)
        np.testing.assert_array_equal(
            np.asarray(x1), np.asarray(self._prebuilt_single(f1, K, dz, dz_half, dt)))
        np.testing.assert_array_equal(
            np.asarray(x2), np.asarray(self._prebuilt_single(f2, K, dz, dz_half, dt)))
        tol = 1e-13 if dtype == jnp.float64 else 1e-5
        y1 = implicit_vertical_diffusion_ocean(f1, K, dz, dz_half, dt)
        y2 = implicit_vertical_diffusion_ocean(f2, K, dz, dz_half, dt)
        np.testing.assert_allclose(np.asarray(x1), np.asarray(y1), rtol=tol, atol=tol)
        np.testing.assert_allclose(np.asarray(x2), np.asarray(y2), rtol=tol, atol=tol)
        assert x1.dtype == dtype and x2.dtype == dtype

    def test_mixed_dtype_falls_back_faithfully(self):
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean,
        )
        from legoesm.timestepping.tridiagonal import thomas_solve_shared
        f1, f2, K, dz, dz_half = self._problem(jnp.float64, seed=3)
        f2_32 = f2.astype(jnp.float32)
        from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
            _build_implicit_tridiag,
        )
        a, b, c, d1 = _build_implicit_tridiag(f1, K, dz, dz_half, 900.0)
        x1, x2 = thomas_solve_shared(a, b, c, (d1, f2_32))
        np.testing.assert_array_equal(
            np.asarray(x1), np.asarray(self._prebuilt_single(f1, K, dz, dz_half, 900.0)))
        np.testing.assert_array_equal(
            np.asarray(x2), np.asarray(self._prebuilt_single(f2_32, K, dz, dz_half, 900.0)))
        y1 = implicit_vertical_diffusion_ocean(f1, K, dz, dz_half, 900.0)
        y2 = implicit_vertical_diffusion_ocean(
            f2_32, K, dz, dz_half, 900.0)
        np.testing.assert_allclose(np.asarray(x1), np.asarray(y1), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(np.asarray(x2), np.asarray(y2), rtol=1e-6, atol=1e-6)
        assert x2.dtype == jnp.float32

    def test_shape_mismatch_raises(self):
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean_pair,
        )
        f1, f2, K, dz, dz_half = self._problem(jnp.float64, seed=4)
        with pytest.raises(ValueError, match="share a"):
            implicit_vertical_diffusion_ocean_pair(
                f1, f2[..., :-1], K, dz, dz_half, 900.0)

    def test_one_level_noop(self):
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean_pair,
        )
        f = jnp.ones((4, 4, 1))
        x1, x2 = implicit_vertical_diffusion_ocean_pair(
            f, 2.0 * f, jnp.zeros((4, 4, 0)), jnp.ones((1,)),
            jnp.ones((0,)), 900.0)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(f))
        np.testing.assert_array_equal(np.asarray(x2), 2.0)

    def test_grad_flows_through_pair(self):
        from legoesm.ocean.physics.vertical_mixing import (
            implicit_vertical_diffusion_ocean_pair,
        )
        f1, f2, K, dz, dz_half = self._problem(jnp.float64, seed=5)

        def loss(a_, b_):
            x1, x2 = implicit_vertical_diffusion_ocean_pair(
                a_, b_, K, dz, dz_half, 900.0)
            return jnp.sum(x1**2) + jnp.sum(x2**2)

        g1, g2 = jax.grad(loss, argnums=(0, 1))(f1, f2)
        assert bool(jnp.all(jnp.isfinite(g1)))
        assert bool(jnp.all(jnp.isfinite(g2)))
        assert float(jnp.max(jnp.abs(g1))) > 0.0
