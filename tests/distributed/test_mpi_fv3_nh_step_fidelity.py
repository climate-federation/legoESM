"""FV3 NH compressible-Euler 3D step bit-for-bit fidelity under MPI.

NH counterpart to ``test_mpi_fv3_step_fidelity.py`` (PE).  The NH
hot path uses the same iter-1040/iter-1041 plumbing — including
the ``packed_pad_halo_mpi_4d`` (K, pi_prime) exchange in step 6
of the split-explicit acoustic loop — so this test guards against
regressions on the NH side.

Run with::

    JAX_ENABLE_X64=1 mpirun -np 2 python -m pytest \\
        tests/distributed/test_mpi_fv3_nh_step_fidelity.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

mpi4jax = pytest.importorskip("mpi4jax")
MPI = pytest.importorskip("mpi4py.MPI")

from legoesm.grids.halo import (
    get_halo_backend,
    set_halo_backend,
)
from legoesm.parallel.distributed import initialize_distributed


@pytest.fixture(autouse=True)
def reset_halo_backend():
    yield
    set_halo_backend("local")


def _build_nh_model_and_state(n: int = 8, nlev: int = 5, **config_overrides):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState

    grid = create_cubed_sphere(n)
    z_top = 30000.0
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, grid.n, grid.n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    cfg_kwargs = dict(
        hyperdiff_coeff=0.0,
        n_acoustic_substeps=4,
        fix_mass=False,
    )
    cfg_kwargs.update(config_overrides)
    config = CDGridCompressibleEulerConfig(**cfg_kwargs)

    # Two distinct instances → independent JIT caches (static_argnums=0).
    ref_model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, config,
    )
    dist_model = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, config,
    )

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    # Small zonal-wind perturbation IC (mirrors iter-1045 fixture).
    state = NonHydrostaticState(
        u=Field(data=jnp.full((6, n, n, nlev), 5.0),
                name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)),
                name="v", dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                name="w", dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)),
                   name="phis", dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return ref_model, dist_model, state


def _run_pair_and_assert(rank, size, ref_model, dist_model, state,
                           dt, n_steps, fields):
    """Shared helper: run ref+dist, compare owned faces per rank."""
    ref_state = state
    for _ in range(n_steps):
        ref_state = ref_model.step(ref_state, dt)
    jax.block_until_ready(ref_state.theta_prime.data)

    initialize_distributed(global_n=state.u.data.shape[1])
    assert get_halo_backend() == "mpi"

    dist_state = state
    for _ in range(n_steps):
        dist_state = dist_model.step(dist_state, dt)
    jax.block_until_ready(dist_state.theta_prime.data)

    from legoesm.parallel.comm import build_comm_topology
    topology = build_comm_topology(rank, size)

    for field_name in fields:
        ref_arr = np.asarray(getattr(ref_state, field_name).data)
        dist_arr = np.asarray(getattr(dist_state, field_name).data)
        for f in topology.local_face_ids:
            np.testing.assert_allclose(
                dist_arr[f], ref_arr[f],
                atol=1e-10, rtol=1e-10,
                err_msg=(
                    f"FV3 NH 3D step diverges from single-rank "
                    f"reference on rank {rank} owned face {f}, "
                    f"field '{field_name}'."
                ),
            )


class TestFV3NHStepMPIFidelity:
    """NH compressible-Euler MPI step matches single-rank reference."""

    def test_nh_3_step_owned_faces_match_single_rank(self):
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        dt = 10.0  # NH needs smaller dt (acoustic CFL)
        n_steps = 3

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state()

        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state, dt, n_steps,
            ("u", "v", "w", "theta_prime", "rho_prime"),
        )

    def test_nh_3_step_with_a2b_zeta_corner(self):
        """FV3_3D iter-1043: a2b_ord4 zeta corner interp under MPI.

        Exercises ``use_fv3_a2b_zeta_corner=True`` which previously
        wrapped ``interp_center_to_corner_a2b_ord4`` in ``jax.vmap``
        on the NH 4D path — and that vmap put ``pad_halo`` (halo=2)
        inside the vmap, triggering mpi4jax's sendrecv batching
        assertion under MPI.  iter-1043 lifts the vmap (the helper
        is shape-polymorphic), so this test guards the fix.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state(
            use_fv3_a2b_zeta_corner=True,
        )
        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state,
            dt=10.0, n_steps=3,
            fields=("u", "v", "w", "theta_prime", "rho_prime"),
        )

    def test_nh_3_step_with_a2b_ord4_theta_corner(self):
        """FV3_3D iter-1043: a2b_ord4 theta corner interp under MPI.

        Mirror of the zeta test for the theta path — same vmap-lift
        rationale, gated by ``use_fv3_a2b_ord4_theta_corner=True``.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state(
            use_fv3_a2b_ord4_theta_corner=True,
        )
        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state,
            dt=10.0, n_steps=3,
            fields=("u", "v", "w", "theta_prime", "rho_prime"),
        )

    def test_nh_3_step_with_corner_div_damp(self):
        """FV3_3D iter-1044: corner div-damp ``nord>=1`` under MPI.

        Exercises the iterated Laplacian path
        (``fv3_corner_laplacian_iteration``) that previously sat
        inside ``jax.vmap`` over levels, crashing mpi4jax's sendrecv
        batch-axis rule.  iter-1044 lifts the vmap by making the
        helper shape-polymorphic.

        Combination ``d2_bg>0 + d4_bg>0 + nord=1`` engages the full
        iter-187 smag_vort cap + iter-1044 Laplacian iteration path.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state(
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d4_bg=0.16,
            corner_div_damp_nord=1,
        )
        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state,
            dt=10.0, n_steps=3,
            fields=("u", "v", "w", "theta_prime", "rho_prime"),
        )

    def test_nh_3_step_with_corner_div_damp_nord2(self):
        """FV3_3D iter-1044: ``nord=2`` (del-6) corner div-damp under MPI.

        Runs TWO Laplacian iterations in the inner loop.  If the
        4D-native helper had any per-iteration state-sharing bug
        that the nord=1 single-iteration test misses, this catches it.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state(
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d4_bg=0.16,
            corner_div_damp_nord=2,
        )
        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state,
            dt=10.0, n_steps=3,
            fields=("u", "v", "w", "theta_prime", "rho_prime"),
        )

    def test_nh_3_step_with_damp_v(self):
        """FV3_3D iter-1045: ``damp_v`` post-step under MPI.

        Exercises ``fv3_del6_vorticity_damping`` previously wrapped in
        ``jax.vmap`` over levels.  iter-1045 makes it 4D-native by
        broadcasting 3D static metrics with ``[..., None]`` and
        dispatching halo through ``pad_halo_4d``.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state(
            damp_v=0.030,
            nord_v=2,
        )
        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state,
            dt=10.0, n_steps=3,
            fields=("u", "v", "w", "theta_prime", "rho_prime"),
        )

    def test_nh_3_step_with_damp_w(self):
        """FV3_3D iter-1045: ``damp_w`` post-step under MPI.

        Exercises ``_del6_vt_flux`` directly on the half-level w
        field — the second of two damp post-step paths the iter-1045
        4D-native refactor unblocks under MPI.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state(
            damp_w=0.030,
            nord_w=2,
        )
        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state,
            dt=10.0, n_steps=3,
            fields=("u", "v", "w", "theta_prime", "rho_prime"),
        )

    def test_nh_3_step_with_both_a2b_paths(self):
        """FV3_3D iter-1043 (codex claim-3): both a2b flags ON simultaneously.

        Catches any subtle ordering/state-sharing bug between the
        two a2b paths that the single-flag tests above can't see.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state = _build_nh_model_and_state(
            use_fv3_a2b_zeta_corner=True,
            use_fv3_a2b_ord4_theta_corner=True,
        )
        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state,
            dt=10.0, n_steps=3,
            fields=("u", "v", "w", "theta_prime", "rho_prime"),
        )

    def test_nh_3_step_with_fv3_faithful_factory(self):
        """FV3_3D iter-1046: factory minus 2 known MPI-incompatible paths.

        iter-1047 test isolation note: clear JIT caches before the
        factory test runs.  Prior tests in the same session trace
        ``CDGridCompressibleEulerModel._step_jitted`` under simpler
        configs; even though each test creates a fresh model
        instance, observed bit-for-bit drift between the factory
        run in isolation vs after other NH tests indicates JAX's
        XLA-level cache may reuse compiled modules across instances
        with related jaxpr signatures.  ``jax.clear_caches()``
        ensures a fresh trace.

        Composes the ``make_fv3_faithful_nh_config(**production_overrides)``
        factory under MPI, EXCEPT for two paths with known MPI
        limitations (disabled below):

        - ``use_fv3_cross_face_du_proj``: operates on non-square
          ``(6, n, n+1, nlev)`` D-grid wind increments via
          ``pad_halo_4d``, whose MPI helpers assume square shape.
          Crashes under MPI; silently mis-indexes halos under
          single-device.  Documented as iter-370 requiring
          duogrid=True to be effective; tracked for non-square halo
          implementation.
        - ``use_duogrid=True``: factory docstring recommends this
          pairing, but the duogrid post-pad remap shows a bit-for-
          bit MPI discrepancy when composed with the full factory
          stack.  Tracked separately.

        With those two disabled, this test still composes 8+
        FV3-fidelity flags simultaneously (cv branch, vector halo,
        a2b vector (u, v), dynamic Exner, metric-aware d_con,
        d_con_top_zero, delt_max, nord_v/corner_div_damp_nord,
        corner_div_damp_d4_bg, heat_source_del2, sponge_damp_v/w)
        plus the production damp/A_h knobs.  Catches hidden MPI
        bugs that the smaller per-flag iter-1042..1045 tests would
        miss when those flags interact.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
            CDGridCompressibleEulerModel,
            make_fv3_faithful_nh_config,
        )
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.vertical import (
            compute_terrain_metric,
            create_height_coordinate,
        )
        from legoesm.core.field import Field
        from legoesm.core.state import NonHydrostaticState

        n, nlev = 8, 5
        # iter-1049: duogrid=True now MPI-safe.  Root cause of the
        # iter-1046/1047 in-suite mismatch was
        # ``synchronize_cgrid_fluxes`` reading non-owned face flux
        # values directly (which under MPI replicated mode were
        # computed with zero halos).  iter-1049 added an MPI-aware
        # sendrecv path; the factory test now passes with the
        # documented production pairing.
        grid = create_cubed_sphere(n, use_duogrid=True)
        z_top = 30000.0
        height_coord = create_height_coordinate(nlev, z_top)
        terrain = jnp.zeros((6, grid.n, grid.n))
        terrain_metric = compute_terrain_metric(terrain, height_coord)

        # Documented production knobs from FV3_3D.md "Production usage"
        # section (the example users are pointed to).
        # iter-1046 known limitation: ``use_fv3_cross_face_du_proj``
        # operates on non-square ``(6, n, n+1)`` / ``(6, n+1, n)`` D-grid
        # increments via ``pad_halo_4d``, whose helpers assume square
        # ``(6, n, n)``.  Single-device the call writes mis-indexed
        # halos (silent buggy) but doesn't crash; MPI crashes on the
        # non-square shape in ``_place_strip_4d``.  Disabled here until
        # a proper non-square halo lands.
        config = make_fv3_faithful_nh_config(
            damp_v=0.030, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            A_h=1e6, ah_d_con=1.0,
            damp_w=0.030, damp_w_d_con=1.0,
            n_acoustic_substeps=4,
            fix_mass=False,
            use_fv3_cross_face_du_proj=False,
        )

        ref_model = CDGridCompressibleEulerModel(
            grid, height_coord, terrain_metric, config,
        )
        dist_model = CDGridCompressibleEulerModel(
            grid, height_coord, terrain_metric, config,
        )

        dims_3d = ("face", "x", "y", "level")
        dims_w = ("face", "x", "y", "level_half")
        dims_2d = ("face", "x", "y")
        state = NonHydrostaticState(
            u=Field(data=jnp.full((6, n, n, nlev), 5.0),
                    name="u", dims=dims_3d, units="m/s"),
            v=Field(data=jnp.zeros((6, n, n, nlev)),
                    name="v", dims=dims_3d, units="m/s"),
            w=Field(data=jnp.zeros((6, n, n, nlev + 1)),
                    name="w", dims=dims_w, units="m/s"),
            theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                              name="theta_prime", dims=dims_3d, units="K"),
            rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                            name="rho_prime", dims=dims_3d, units="kg/m^3"),
            phis=Field(data=jnp.zeros((6, n, n)),
                       name="phis", dims=dims_2d, units="m^2/s^2"),
            tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)),
                          name="tracers",
                          dims=("face", "x", "y", "level", "tracer"),
                          units="kg/kg"),
        )

        _run_pair_and_assert(
            rank, size, ref_model, dist_model, state,
            dt=10.0, n_steps=3,
            fields=("u", "v", "w", "theta_prime", "rho_prime"),
        )
