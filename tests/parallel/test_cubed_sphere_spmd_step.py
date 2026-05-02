"""End-to-end cubed-sphere SPMD step equivalence vs single-device.

Verifies that ``CDGridPrimitiveEquationModel.step`` under 6-device
face-sharded SPMD execution produces the same result as the single-
device path.  The 6 SPMD halo bit-equivalence tests in
``test_cubesphere_exchange.py::TestSPMDWithOffsets`` exercise each
halo kernel individually; this test composes everything (halo
exchange + Lagrange offsets + tendency operators + RK3 integration)
and asserts the dycore stays equivalent to single-device end-to-end.

Iter-7 wired ``interp_offsets`` through the SPMD halo path, which
restored 10–200× lower drift than the iter-5 baseline.  Iter-8
packed the halo=1 cell-field exchange.  Iter-11 / iter-12 packed PPM
transport's q_i / q_j halo.

Test gated on ≥6 CPU devices; skip otherwise.  Run with
``XLA_FLAGS=--xla_force_host_platform_device_count=6``.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _need_devices(n: int):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices "
            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
        )


class TestCubedSphereSPMDStep:
    """End-to-end cubed-sphere PE step on 6 emulated CPU devices
    matches single-device to floating-point precision after several
    SSP-RK3 steps.
    """

    def _run(self, *, devices: int, n_steps: int = 3, n_grid: int = 24):
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.cfl import (
            adaptive_hyperdiff_coeff, estimate_min_dx_cubed_sphere,
        )
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
            hydrostatic_to_fv3, fv3_to_hydrostatic,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        # dt scales with resolution to stay CFL-stable
        dt = 450.0 if n_grid <= 24 else 225.0
        n_lev = 8
        grid = create_cubed_sphere(n_grid)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sigma = create_sigma_coordinate(n_lev)
        dx_min = estimate_min_dx_cubed_sphere(n_grid)
        nu4 = adaptive_hyperdiff_coeff(dx_min, dt, order=4, safety=0.5)
        state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
        cfg = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=nu4, hyperdiff_ps_coeff=nu4,
            use_conservation_fixer=True, fix_mass=True,
            anchor_mass_to_initial=True, zero_mean_ps_tendency=False,
            time_integrator='ssp_rk3',
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
        s = hydrostatic_to_fv3(state_cc, cdgrid)

        if devices > 1:
            from legoesm.parallel.mesh import (
                create_device_mesh, shard_pytree,
            )
            from legoesm.parallel.cubesphere_exchange import (
                activate_spmd_halo_backend, deactivate_spmd_halo_backend,
            )
            dev_config = create_device_mesh(n_devices=devices)
            activate_spmd_halo_backend(dev_config.mesh, n=n_grid, nlev=n_lev)
            try:
                s = shard_pytree(s, dev_config)
                for _ in range(n_steps):
                    s = model.step(s, dt)
            finally:
                deactivate_spmd_halo_backend()
        else:
            for _ in range(n_steps):
                s = model.step(s, dt)
        return fv3_to_hydrostatic(s, cdgrid)

    def _run_with_physics(self, *, devices: int, n_steps: int = 1):
        """Variant of ``_run`` that wires Held-Suarez physics into the
        step.  Mirrors what ``run_levante_gpu_scaling.py`` does for
        ``--physics held_suarez``.
        """
        from legoesm.grids.vertical import create_sigma_coordinate
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.cfl import (
            adaptive_hyperdiff_coeff, estimate_min_dx_cubed_sphere,
        )
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
            hydrostatic_to_fv3, fv3_to_hydrostatic,
        )
        from legoesm.atmosphere.held_suarez import held_suarez_forcing
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        n_grid, n_lev, dt = 24, 8, 450.0
        grid = create_cubed_sphere(n_grid)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sigma = create_sigma_coordinate(n_lev)
        dx_min = estimate_min_dx_cubed_sphere(n_grid)
        nu4 = adaptive_hyperdiff_coeff(dx_min, dt, order=4, safety=0.5)
        state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
        cfg = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=nu4, hyperdiff_ps_coeff=nu4,
            use_conservation_fixer=True, fix_mass=True,
            anchor_mass_to_initial=True, zero_mean_ps_tendency=False,
            time_integrator='ssp_rk3',
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
        s = hydrostatic_to_fv3(state_cc, cdgrid)
        if devices > 1:
            from legoesm.parallel.mesh import (
                create_device_mesh, shard_pytree,
            )
            from legoesm.parallel.cubesphere_exchange import (
                activate_spmd_halo_backend, deactivate_spmd_halo_backend,
            )
            dev_config = create_device_mesh(n_devices=devices)
            activate_spmd_halo_backend(dev_config.mesh, n=n_grid, nlev=n_lev)
            try:
                s = shard_pytree(s, dev_config)
                for _ in range(n_steps):
                    s = model.step(s, dt, physics_fn=held_suarez_forcing)
            finally:
                deactivate_spmd_halo_backend()
        else:
            for _ in range(n_steps):
                s = model.step(s, dt, physics_fn=held_suarez_forcing)
        return fv3_to_hydrostatic(s, cdgrid)

    def test_6device_with_physics_matches_1device(self):
        """6-device SPMD with Held-Suarez physics matches single-device.

        Iter-44: physics_fn closures are captured in the JIT closure
        (not passed as traced args), so the with-physics SPMD path
        could in principle differ from the no-physics path.  Verify
        the iter-31 vector-halo offsets fix carries through to the
        with-physics path too.
        """
        _need_devices(6)
        ref = self._run_with_physics(devices=1, n_steps=1)
        out = self._run_with_physics(devices=6, n_steps=1)
        for name, atol, rtol in (
            ("u", 1e-12, 1e-13),
            ("v", 1e-12, 1e-13),
            ("T", 1e-11, 1e-13),
            ("p_s", 1.0, 1e-5),
        ):
            r = np.asarray(getattr(ref, name).data)
            o = np.asarray(getattr(out, name).data)
            np.testing.assert_allclose(
                o, r, atol=atol, rtol=rtol,
                err_msg=f"{name}: 6-dev cubed-sphere with-physics drift "
                        f"exceeds float-pt envelope",
            )

    def test_C48_6device_matches_1device(self):
        """Verify cubed-sphere SPMD bit-equivalence at production-ish
        resolution C48.  Iter-45 confirms the iter-31 fix carries
        through to higher horizontal resolution.

        Same envelope as C24: u/v/T at FMA precision, p_s ~1e-7 rel
        (allreduce float-pt).
        """
        _need_devices(6)
        ref = self._run(devices=1, n_steps=1, n_grid=48)
        out = self._run(devices=6, n_steps=1, n_grid=48)
        for name, atol, rtol in (
            ("u", 1e-12, 1e-13),
            ("v", 1e-12, 1e-13),
            ("T", 1e-11, 1e-13),
            ("p_s", 1.0, 1e-5),
        ):
            r = np.asarray(getattr(ref, name).data)
            o = np.asarray(getattr(out, name).data)
            np.testing.assert_allclose(
                o, r, atol=atol, rtol=rtol,
                err_msg=f"{name}: 6-dev cubed-sphere C48 SPMD drift "
                        f"exceeds float-pt envelope",
            )

    def test_hybrid_coord_6device_matches_1device(self):
        """6-device hybrid σ-pressure SPMD step matches single-device.

        Iter-60 added ``_hybrid_factor = B_full * p_s / p_full`` to the
        merged stage halo exchange so the PGF correction's corner
        interpolation reuses the pre-padded buffer.  This is only
        exercised on the hybrid coordinate path; the existing tests
        all use ``create_sigma_coordinate``, leaving iter-60's new
        SPMD branch untested.

        Tolerance envelope same as the σ-coord 1-step test:
        u/v at FMA precision (1e-12), T at 1e-11, p_s at 1.0 abs.
        """
        _need_devices(6)
        from legoesm.grids.vertical import make_hybrid_levels
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.core.cfl import (
            adaptive_hyperdiff_coeff, estimate_min_dx_cubed_sphere,
        )
        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
            hydrostatic_to_fv3, fv3_to_hydrostatic,
        )
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init
        from legoesm.parallel.mesh import (
            create_device_mesh, shard_pytree,
        )
        from legoesm.parallel.cubesphere_exchange import (
            activate_spmd_halo_backend, deactivate_spmd_halo_backend,
        )
        n_grid, n_lev, dt = 24, 8, 450.0
        grid = create_cubed_sphere(n_grid)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sigma = make_hybrid_levels(n_lev)  # ← hybrid coord
        dx_min = estimate_min_dx_cubed_sphere(n_grid)
        nu4 = adaptive_hyperdiff_coeff(dx_min, dt, order=4, safety=0.5)
        state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
        cfg = CDGridPrimitiveEquationConfig(
            hyperdiff_coeff=nu4, hyperdiff_ps_coeff=nu4,
            use_conservation_fixer=True, fix_mass=True,
            anchor_mass_to_initial=True, zero_mean_ps_tendency=False,
            time_integrator='ssp_rk3',
        )
        model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
        s0 = hydrostatic_to_fv3(state_cc, cdgrid)
        # 1-device reference
        s_ref = model.step(s0, dt)
        ref = fv3_to_hydrostatic(s_ref, cdgrid)
        # 6-device SPMD
        dev_config = create_device_mesh(n_devices=6)
        activate_spmd_halo_backend(dev_config.mesh, n=n_grid, nlev=n_lev)
        try:
            s_spmd = shard_pytree(s0, dev_config)
            s_spmd = model.step(s_spmd, dt)
            out = fv3_to_hydrostatic(s_spmd, cdgrid)
        finally:
            deactivate_spmd_halo_backend()
        for name, atol, rtol in (
            ("u", 1e-12, 1e-13),
            ("v", 1e-12, 1e-13),
            ("T", 1e-11, 1e-13),
            ("p_s", 1.0, 1e-5),
        ):
            r = np.asarray(getattr(ref, name).data)
            o = np.asarray(getattr(out, name).data)
            np.testing.assert_allclose(
                o, r, atol=atol, rtol=rtol,
                err_msg=f"{name}: 6-dev cubed-sphere hybrid-coord SPMD "
                        f"drift exceeds float-pt envelope",
            )

    @pytest.mark.parametrize("devices", [2, 3])
    def test_multiface_shard_matches_1device(self, devices):
        """Iter-49 generalises the SPMD halo kernel to multi-face shards.

        On 2 devices each shard owns 3 faces (n_faces_per_shard=3); on
        3 devices each shard owns 2 faces.  The iter-1 activation guard
        previously restricted the SPMD halo to exactly 6 devices and
        fell back to the auto-gather backend for 2 / 3 — iter-49 lifted
        that.  This test composes everything end-to-end and asserts
        the multi-face SPMD path matches the single-device dycore.
        """
        _need_devices(devices)
        ref = self._run(devices=1, n_steps=1)
        out = self._run(devices=devices, n_steps=1)
        for name, atol, rtol in (
            ("u", 1e-12, 1e-13),
            ("v", 1e-12, 1e-13),
            ("T", 1e-11, 1e-13),
            ("p_s", 1.0, 1e-5),
        ):
            r = np.asarray(getattr(ref, name).data)
            o = np.asarray(getattr(out, name).data)
            np.testing.assert_allclose(
                o, r, atol=atol, rtol=rtol,
                err_msg=f"{name}: {devices}-dev cubed-sphere multi-face "
                        f"SPMD drift exceeds float-pt envelope",
            )

    @pytest.mark.parametrize("n_steps", [1, 10])
    def test_6device_matches_1device(self, n_steps):
        """6-device face-sharded cubed-sphere matches single-device to
        floating-point precision on dynamical fields after SSP-RK3
        steps.

        Iter-30 surfaced ~6e-4 relative drift on u; iter-31 root-caused
        it to ``pad_halo_vector_4d`` silently dropping
        ``interp_offsets`` under SPMD when called by ``divergence_3d``
        inside the hyperdiffusion path.  Forwarding offsets through
        ``explicit_pad_halo_vector_4d`` restored bit-equivalence on
        dynamical fields.

        Iter-32 verifies the bit-equivalence holds over multiple steps.
        After 1 step u/v/T are at FMA precision; after 10 steps they
        bound to ~1e-5 absolute as the post-step
        ``fix_mass_hydrostatic_target`` allreduce float-pt drift on ps
        (5e-7 rel per step) feeds through the pressure gradient back
        into u/v.
        """
        _need_devices(6)
        ref = self._run(devices=1, n_steps=n_steps)
        out = self._run(devices=6, n_steps=n_steps)
        # Tolerance scales with n_steps to account for accumulated
        # float-pt drift from the sharded mass-fixer allreduce.
        u_atol = 1e-12 if n_steps == 1 else 1e-4
        T_atol = 1e-11 if n_steps == 1 else 1e-4
        ps_atol = 1.0 if n_steps == 1 else 5.0
        for name, atol, rtol in (
            ("u", u_atol, 1e-9),
            ("v", u_atol, 1e-9),
            ("T", T_atol, 1e-9),
            ("p_s", ps_atol, 1e-5),
        ):
            r = np.asarray(getattr(ref, name).data)
            o = np.asarray(getattr(out, name).data)
            np.testing.assert_allclose(
                o, r, atol=atol, rtol=rtol,
                err_msg=f"{name}: 6-device cubed-sphere SPMD drift "
                        f"exceeds float-pt envelope at n_steps={n_steps}",
            )
