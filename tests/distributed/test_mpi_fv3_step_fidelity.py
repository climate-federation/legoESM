"""FV3 3D step bit-for-bit fidelity: MPI vs single-rank reference.

Iter-1040 enabled ``interp_offsets`` on ``pad_halo_mpi`` /
``pad_halo_mpi_4d``.  Iter-1041 threads them through the packed
exchange ``packed_pad_halo_mpi_4d`` used by ``fv3_hydrostatic_tendencies``
(PE) and ``_step_jitted`` (NH) on the MPI hot path.

The pre-existing ``test_mpi_driver.py::test_distributed_3_steps_
matches_single_rank`` runs both the local-reference step and the
"MPI" step on the same JIT-compiled ``model.step`` function — JAX
caches the trace from the first call (under ``set_halo_backend
("local")``) and re-uses it for the second call, so the MPI branch
inside ``fv3_hydrostatic_tendencies`` is never executed.  The test
"passes" trivially.

This file forces a fresh JIT trace under MPI by clearing the JAX
cache between the reference and distributed runs, so the
``_halo_backend == "mpi"`` Python branch actually fires.

Run with::

    JAX_ENABLE_X64=1 mpirun -np 2 python -m pytest \\
        tests/distributed/test_mpi_fv3_step_fidelity.py -v
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


def _build_pe_model_and_state(n: int = 8, nlev: int = 5, **config_overrides):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
        hydrostatic_to_fv3,
    )
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init

    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(nlev)

    cfg_kwargs = dict(
        hyperdiff_coeff=0.0,
        hyperdiff_ps_coeff=0.0,
        use_conservation_fixer=False,
        fix_mass=False,
        zero_mean_ps_tendency=False,
    )
    cfg_kwargs.update(config_overrides)
    config = CDGridPrimitiveEquationConfig(**cfg_kwargs)
    state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
    state_global = hydrostatic_to_fv3(state_cc, cdgrid)

    # Two distinct model instances guarantee independent JIT caches
    # (static_argnums=0 keys the cache on the model identity).
    ref_model = CDGridPrimitiveEquationModel(grid, sigma, config)
    dist_model = CDGridPrimitiveEquationModel(grid, sigma, config)

    return ref_model, dist_model, state_global


def _run_pe_pair_and_assert(rank, size, ref_model, dist_model,
                              state_global, dt, n_steps):
    ref_state = state_global
    for _ in range(n_steps):
        ref_state = ref_model.step(ref_state, dt)
    jax.block_until_ready(ref_state.T.data)

    initialize_distributed(global_n=state_global.T.data.shape[1])
    assert get_halo_backend() == "mpi"

    dist_state = state_global
    for _ in range(n_steps):
        dist_state = dist_model.step(dist_state, dt)
    jax.block_until_ready(dist_state.T.data)

    from legoesm.parallel.comm import build_comm_topology
    topology = build_comm_topology(rank, size)
    for field_name in ("T", "u_d", "v_d", "p_s"):
        ref_arr = np.asarray(getattr(ref_state, field_name).data)
        dist_arr = np.asarray(getattr(dist_state, field_name).data)
        for f in topology.local_face_ids:
            np.testing.assert_allclose(
                dist_arr[f], ref_arr[f],
                atol=1e-10, rtol=1e-10,
                err_msg=(
                    f"FV3 PE 3D step diverges from single-rank "
                    f"reference on rank {rank} owned face {f}, "
                    f"field '{field_name}'."
                ),
            )


class TestFV3PEStepMPIFidelity:
    """Force fresh JIT traces under each backend; compare owned faces."""

    def test_pe_3_step_owned_faces_match_single_rank(self):
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        dt = 300.0
        n_steps = 3

        # --- Pre-build reference state on local backend ---
        set_halo_backend("local")
        ref_model, dist_model, state_global = _build_pe_model_and_state()

        # Reference: local-backend trace of ref_model
        ref_state = state_global
        for _ in range(n_steps):
            ref_state = ref_model.step(ref_state, dt)
        # Block until done so the next set_halo_backend is observed
        # after the local trace materialises (JIT trace closes over
        # the module-level _halo_backend value at trace time).
        jax.block_until_ready(ref_state.T.data)

        # --- Switch to MPI backend and trace dist_model fresh ---
        initialize_distributed(global_n=state_global.T.data.shape[1])
        assert get_halo_backend() == "mpi", (
            "initialize_distributed should leave backend in mpi mode"
        )

        dist_state = state_global
        for _ in range(n_steps):
            dist_state = dist_model.step(dist_state, dt)
        jax.block_until_ready(dist_state.T.data)

        # --- Compare owned faces only (MPI replicated mode contract) ---
        from legoesm.parallel.comm import build_comm_topology
        topology = build_comm_topology(rank, size)

        # FV3_3D iter-1042 (codex review claim-6): assert on every rank's
        # owned faces, not just rank 0.
        for field_name in ("T", "u_d", "v_d", "p_s"):
            ref_arr = np.asarray(getattr(ref_state, field_name).data)
            dist_arr = np.asarray(getattr(dist_state, field_name).data)
            for f in topology.local_face_ids:
                np.testing.assert_allclose(
                    dist_arr[f], ref_arr[f],
                    atol=1e-10, rtol=1e-10,
                    err_msg=(
                        f"FV3 PE 3D step diverges from single-rank "
                        f"reference on rank {rank} owned face {f}, "
                        f"field '{field_name}'.  This means the MPI "
                        f"halo path (packed_pad_halo_mpi_4d with "
                        f"interp_offsets) is not producing the "
                        f"same numerics as the local interp_offsets "
                        f"path."
                    ),
                )

    def test_pe_3_step_with_a2b_zeta_corner(self):
        """FV3_3D iter-1043 (codex claim-6): PE MPI with a2b_zeta_corner.

        The PE path had the same ``jax.vmap`` around
        ``_interp_center_to_corner_a2b_ord4`` that crashed NH MPI.
        iter-1043 lifts it; this test guards.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state_global = _build_pe_model_and_state(
            use_fv3_a2b_zeta_corner=True,
        )
        _run_pe_pair_and_assert(
            rank, size, ref_model, dist_model, state_global,
            dt=300.0, n_steps=3,
        )

    def test_pe_3_step_with_damp_v(self):
        """FV3_3D iter-1045: PE ``damp_v`` post-step under MPI.

        PE counterpart to NH ``test_nh_3_step_with_damp_v``.  PE has
        no ``damp_w`` (no prognostic w in hydrostatic).  Same
        ``fv3_del6_vorticity_damping`` 4D-native helper used.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state_global = _build_pe_model_and_state(
            damp_v=0.030,
            nord_v=2,
        )
        _run_pe_pair_and_assert(
            rank, size, ref_model, dist_model, state_global,
            dt=300.0, n_steps=3,
        )

    def test_pe_3_step_with_fv3_faithful_factory(self):
        """FV3_3D iter-1046: factory minus 2 known MPI-incompatible paths.

        Mirror of NH ``test_nh_3_step_with_fv3_faithful_factory``.
        Composes ``make_fv3_faithful_pe_config(**production_overrides)``
        under MPI except for ``use_fv3_cross_face_du_proj`` (non-
        square ``(6, n, n+1, nlev)`` `pad_halo_4d` MPI crash) and
        ``use_duogrid=True`` (full-factory + duogrid bit-for-bit
        mismatch).  Both disabled here; see NH factory docstring
        for the architectural notes.

        Remaining factory flags exercised: use_fv3_a2b_zeta_corner,
        use_fv3_metric_aware_d_con, d_con_top_zero_levels, delt_max,
        nord_v, corner_div_damp_nord, corner_div_damp_d4_bg,
        heat_source_del2_iters, use_fv3_sponge_damp_v + the production
        damp/A_h overrides.
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        from legoesm.atmosphere.dynamics.primitive_eq_cdgrid import (
            CDGridPrimitiveEquationModel,
            hydrostatic_to_fv3,
            make_fv3_faithful_pe_config,
        )
        from legoesm.grids.cubed_sphere import create_cubed_sphere
        from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
        from legoesm.grids.vertical import create_sigma_coordinate
        from tests.test_cases.baroclinic_wave import baroclinic_wave_init

        n, nlev = 8, 5
        # iter-1049: duogrid=True now MPI-safe (see NH factory docstring).
        grid = create_cubed_sphere(n, use_duogrid=True)
        cdgrid = create_cubed_sphere_cdgrid(grid)
        sigma = create_sigma_coordinate(nlev)

        # Documented PE production knobs from FV3_3D.md.
        # iter-1046: ``use_fv3_cross_face_du_proj`` disabled — see NH
        # factory test for the same known limitation (non-square
        # ``(6, n, n+1)`` D-grid increment through ``pad_halo_4d``).
        config = make_fv3_faithful_pe_config(
            damp_v=0.030, damp_v_d_con=1.0,
            corner_div_damp_d2_bg=0.0005, corner_div_damp_d_con=1.0,
            div_damp_coeff=1e6, div_damp_d_con=1.0,
            A_h=1e6, ah_d_con=1.0,
            use_conservation_fixer=False,
            fix_mass=False,
            zero_mean_ps_tendency=False,
            use_fv3_cross_face_du_proj=False,
        )
        state_cc = baroclinic_wave_init(grid, sigma, perturbed=True)
        state_global = hydrostatic_to_fv3(state_cc, cdgrid)

        ref_model = CDGridPrimitiveEquationModel(grid, sigma, config)
        dist_model = CDGridPrimitiveEquationModel(grid, sigma, config)

        set_halo_backend("local")
        _run_pe_pair_and_assert(
            rank, size, ref_model, dist_model, state_global,
            dt=300.0, n_steps=3,
        )

    def test_pe_3_step_with_corner_div_damp(self):
        """FV3_3D iter-1044 (codex claim-3): PE MPI with corner-div-damp nord=1.

        PE counterpart to NH ``test_nh_3_step_with_corner_div_damp``.
        Exercises the iter-1044 4D-native ``fv3_corner_laplacian_iteration``
        + the iter-1044 4D-native ``fv3_divergence_corner_3d`` (no vmap
        around any pad_halo / sendrecv).
        """
        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        if size > 6 or 6 % size != 0:
            pytest.skip("Face-only mode only (1/2/3/6 ranks).")

        set_halo_backend("local")
        ref_model, dist_model, state_global = _build_pe_model_and_state(
            corner_div_damp_d2_bg=5e-4,
            corner_div_damp_d4_bg=0.02,
            corner_div_damp_nord=1,
        )
        _run_pe_pair_and_assert(
            rank, size, ref_model, dist_model, state_global,
            dt=300.0, n_steps=3,
        )
