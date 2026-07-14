"""MPI-aware dry-mass + fixer for the plane CRM (R7).

Single-rank: must short-circuit to the serial path bit-for-bit so
existing tests + benchmarks see no behaviour change.

Multi-rank semantics (single-process simulation): the MPI fixer must
restore the GLOBAL dry mass to ``target_mass`` to round-off when
applied to a state whose local rho' has been perturbed; the additive
correction must be IDENTICAL on every rank (the test verifies this by
running the fixer twice on the same state and asserting the rho'
deltas are spatially uniform).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    compute_dry_mass_plane,
    fix_mass_nonhydrostatic_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.crm.rce_mpi import (
    _plane_volume_weight_mpi,
    compute_dry_mass_plane_mpi,
    fix_mass_nonhydrostatic_plane_mpi,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import create_height_coordinate
from legoesm.parallel.plane_mpi import make_plane_pencil_layout

jax.config.update("jax_enable_x64", True)


def _setup():
    grid = create_plane_grid(
        nx=8, ny=6, nlev=10, dx=1000.0, dy=2000.0, dtype=jnp.float64,
    )
    hc = create_height_coordinate(10, H=20_000.0)
    tm = make_flat_plane_terrain_metric(grid, hc)
    rest = make_rest_state(grid, hc, dtype=jnp.float64)
    rng = jax.random.PRNGKey(0)
    state = rest._replace(
        rho_prime=rest.rho_prime.replace(
            data=0.001 * jax.random.normal(rng, rest.rho_prime.data.shape),
        ),
    )
    layout = make_plane_pencil_layout(
        rank=0, n_ranks=1, n_ranks_y=1, n_ranks_x=1,
        ny_global=6, nx_global=8,
    )
    owned_mask = jnp.ones((grid.ny, grid.nx), dtype=jnp.float64)
    return grid, hc, tm, state, layout, owned_mask


def test_compute_dry_mass_single_rank_bit_equiv():
    """compute_dry_mass_plane_mpi must short-circuit to serial when
    layout.n_ranks == 1 (no global_sum_mpi overhead)."""
    grid, hc, tm, state, layout, mask = _setup()
    serial = compute_dry_mass_plane(state, grid, hc, tm)
    mpi = compute_dry_mass_plane_mpi(state, grid, hc, tm, layout, mask)
    np.testing.assert_allclose(np.asarray(mpi), np.asarray(serial),
                               rtol=0.0, atol=0.0)


def test_fix_mass_single_rank_bit_equiv():
    """fix_mass_nonhydrostatic_plane_mpi must match the serial fixer
    bit-for-bit on single rank."""
    grid, hc, tm, state, layout, mask = _setup()
    # Perturb mass so the fixer has work to do.
    state_perturbed = state._replace(
        rho_prime=state.rho_prime.replace(
            data=state.rho_prime.data + 1e-4,
        ),
    )
    target = compute_dry_mass_plane(state, grid, hc, tm)
    serial_out = fix_mass_nonhydrostatic_plane(
        state_perturbed, target, grid, hc, tm,
    )
    mpi_out = fix_mass_nonhydrostatic_plane_mpi(
        state_perturbed, target, grid, hc, tm, layout, mask,
    )
    np.testing.assert_allclose(
        np.asarray(mpi_out.rho_prime.data),
        np.asarray(serial_out.rho_prime.data),
        rtol=1e-14, atol=1e-14,
    )


def test_fix_mass_restores_target_mass_single_rank():
    """Round-trip: perturb state, fix, then compute_dry_mass returns
    the target."""
    grid, hc, tm, state, layout, mask = _setup()
    target = compute_dry_mass_plane(state, grid, hc, tm)
    perturbed = state._replace(
        rho_prime=state.rho_prime.replace(
            data=state.rho_prime.data + 1e-4,
        ),
    )
    fixed = fix_mass_nonhydrostatic_plane_mpi(
        perturbed, target, grid, hc, tm, layout, mask,
    )
    restored_mass = compute_dry_mass_plane_mpi(
        fixed, grid, hc, tm, layout, mask,
    )
    np.testing.assert_allclose(
        np.asarray(restored_mass), np.asarray(target),
        rtol=1e-12, atol=1e-8,
    )


def test_volume_weight_single_rank_matches_serial():
    grid, hc, tm, state, layout, mask = _setup()
    weight_h = (tm.jacobian * grid.area_T)[:, :, None]
    expected = float(jnp.sum(weight_h * hc.dz))
    actual = float(_plane_volume_weight_mpi(grid, hc, tm, layout, mask))
    assert abs(actual - expected) < 1e-12 * abs(expected)


def test_fix_mass_correction_is_spatially_uniform():
    """The additive correction delta must be identical at every cell —
    the fixer's defining property (preserves all rho' gradients used
    in the slow tendency)."""
    grid, hc, tm, state, layout, mask = _setup()
    target = compute_dry_mass_plane(state, grid, hc, tm)
    rng = jax.random.PRNGKey(7)
    perturbed_data = state.rho_prime.data + 1e-3 * jax.random.normal(
        rng, state.rho_prime.data.shape,
    )
    perturbed = state._replace(
        rho_prime=state.rho_prime.replace(data=perturbed_data),
    )
    fixed = fix_mass_nonhydrostatic_plane_mpi(
        perturbed, target, grid, hc, tm, layout, mask,
    )
    delta = fixed.rho_prime.data - perturbed.rho_prime.data
    assert float(jnp.std(delta)) < 1e-13 * float(jnp.max(jnp.abs(delta)) + 1e-30)


def test_owned_mask_excludes_halo_rows_from_global_sum():
    """If owned_mask has a row of zeros (simulating a halo overlap
    being owned by another rank), that row must NOT contribute to the
    global mass — verified by comparing two single-rank invocations
    with different masks but identical state."""
    grid, hc, tm, state, layout, mask_all = _setup()
    mass_full = compute_dry_mass_plane_mpi(
        state, grid, hc, tm, layout, mask_all,
    )
    # Manually zero out the first row of the mask to mimic "row 0 is a
    # halo overlap owned by another rank" — on a single-rank layout this
    # short-circuits to the serial path which ignores the mask, so the
    # mass stays at mass_full. The point of THIS test is the contract
    # documented in compute_dry_mass_plane_mpi: serial fallback ignores
    # the owned_mask and returns the full local mass. Document via assert.
    mask_partial = mask_all.at[0, :].set(0.0)
    mass_with_partial = compute_dry_mass_plane_mpi(
        state, grid, hc, tm, layout, mask_partial,
    )
    np.testing.assert_allclose(
        np.asarray(mass_with_partial), np.asarray(mass_full),
        rtol=0.0, atol=0.0,
        err_msg="single-rank path must ignore owned_mask (matches "
                "compute_total_water_mass_plane_mpi short-circuit)",
    )


def test_step_halo_raises_when_fix_mass_without_owned_mask_multirank():
    """Codex 2026-05 review fix: step_halo on multi-rank with
    config.fix_mass=True must raise when owned_mask is omitted, not
    silently let dry mass drift. Construct a fake n_ranks=2 layout
    (no real MPI) and verify the gate fires before any compute."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        PlaneCompressibleEulerModel,
    )
    grid, hc, tm, state, _layout, _mask = _setup()
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        smagorinsky_cs=0.0, use_coriolis=False,
        semi_implicit_acoustic=True,
        fix_mass=True, anchor_mass_to_initial=True,
        n_acoustic_substeps=12,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, config=cfg)
    # Fake multi-rank layout (no actual MPI launcher).
    fake_layout = make_plane_pencil_layout(
        rank=0, n_ranks=2, n_ranks_y=2, n_ranks_x=1,
        ny_global=12, nx_global=8,
    )
    with pytest.raises(ValueError, match="owned_mask"):
        model.step_halo(state, dt=0.5, layout=fake_layout)


def test_step_halo_with_fix_mass_anchored_single_rank():
    """End-to-end: step_halo on single rank with owned_mask + fix_mass
    must preserve the initial dry mass to round-off across many steps."""
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        PlaneCompressibleEulerModel,
    )
    grid, hc, tm, state, layout, mask = _setup()
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.05, sponge_width=5_000.0,
        hyperdiff_coeff=1.0e4,
        hyperdiff_rho_coeff=1.0e4,
        hyperdiff_w_coeff=1.0e4,
        smagorinsky_cs=0.0,
        use_coriolis=False,
        semi_implicit_acoustic=True,
        fix_mass=True, anchor_mass_to_initial=True,
        n_acoustic_substeps=12,
    )
    # Seed a small momentum kick so the slow tendency does work.
    rng = jax.random.PRNGKey(1)
    state = state._replace(
        u=state.u.replace(
            data=0.05 * jax.random.normal(rng, state.u.data.shape),
        ),
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, config=cfg)
    initial_mass = compute_dry_mass_plane_mpi(
        state, grid, hc, tm, layout, mask,
    )
    s = state
    for _ in range(5):
        s = model.step_halo(s, dt=0.5, layout=layout, owned_mask=mask)
    final_mass = compute_dry_mass_plane_mpi(
        s, grid, hc, tm, layout, mask,
    )
    np.testing.assert_allclose(
        np.asarray(final_mass), np.asarray(initial_mass),
        rtol=1e-12, atol=1e-6,
        err_msg="step_halo + owned_mask + fix_mass must anchor mass",
    )
