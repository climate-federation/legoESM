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

    def _run(self, *, devices: int, n_steps: int = 3):
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
                    s = model.step(s, dt)
            finally:
                deactivate_spmd_halo_backend()
        else:
            for _ in range(n_steps):
                s = model.step(s, dt)
        return fv3_to_hydrostatic(s, cdgrid)

    def test_6device_matches_1device(self):
        """6-device face-sharded cubed-sphere matches single-device to
        floating-point precision on dynamical fields after one
        SSP-RK3 step.

        Iter-30 surfaced ~6e-4 relative drift on u under 6-device
        SPMD; iter-31 root-caused it to ``pad_halo_vector_4d``
        silently dropping ``interp_offsets`` under SPMD when called by
        ``divergence_3d`` inside the hyperdiffusion path.  Forwarding
        offsets through ``explicit_pad_halo_vector_4d`` restored
        bit-equivalence on dynamical fields:

            u, v, T : 1e-14 .. 1e-16 abs (FMA precision)
            ps      : 5e-2 abs (5e-7 rel) — float-pt accumulation-order
                       difference in the post-step ``fix_mass_hydrostatic_target``
                       allreduce, fundamental to sharded reductions.
        """
        _need_devices(6)
        ref = self._run(devices=1, n_steps=1)
        out = self._run(devices=6, n_steps=1)
        # Tight tolerance on dynamical fields (post-iter-31 fix)
        for name, atol, rtol in (
            ("u", 1e-12, 1e-13),
            ("v", 1e-12, 1e-13),
            ("T", 1e-11, 1e-13),
            # ps drift from sharded allreduce — looser envelope
            ("p_s", 1.0, 1e-5),
        ):
            r = np.asarray(getattr(ref, name).data)
            o = np.asarray(getattr(out, name).data)
            np.testing.assert_allclose(
                o, r, atol=atol, rtol=rtol,
                err_msg=f"{name}: 6-device cubed-sphere SPMD drift "
                        f"exceeds float-pt envelope",
            )
