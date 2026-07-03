"""Unit tests for the rigid-lid barotropic streamfunction solver (lat-lon C-grid).

Covers the operator (nullspace, harmonic basis, manufactured-solution Poisson
round-trip), the island machinery (flood-fill count, symmetric line_psin,
masks), velocity recovery (non-divergent transport), face depths, the dense
island-constant solve, dispatch validation, and the full coupling (a forced
channel develops a sensible barotropic transport; eta unchanged; differentiable).

Source files exercised:
  - legoesm/ocean/dynamics/rigid_lid_latlon_cgrid.py
  - legoesm/ocean/dynamics/rigid_lid_islands.py
  - the new operators in latlon_cgrid_operators.py (recover_velocity_from_
    streamfunction, streamfunction_vorticity_operator, vertex_area_cgrid).
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import create_ocean_z_star, compute_layer_thickness
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    streamfunction_vorticity_operator, recover_velocity_from_streamfunction,
    vertex_area_cgrid, divergence_cgrid, min_cell_to_uface, min_cell_to_vface,
)
from legoesm.ocean.dynamics.rigid_lid_latlon_cgrid import (
    barotropic_face_depths, solve_streamfunction_interior, rigid_lid_step,
    _solve_island_constants, RigidLidStaticData,
)
from legoesm.ocean.dynamics.rigid_lid_islands import build_rigid_lid_data


N_LAT, N_LON = 18, 36


def _channel():
    """A periodic channel: south + north 2-row land walls, flat 4000 m interior."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    lm = np.ones((N_LAT, N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((N_LAT, N_LON), 4000.0) * lm
    mE = lm; mW = np.roll(lm, 1, axis=1)
    u_int = (mE * mW) > 0.5
    u_mask = np.concatenate([u_int, u_int[:, :1]], axis=1).astype(float)
    v_int = (lm[:-1] * lm[1:]) > 0.5
    v_mask = np.concatenate([np.zeros((1, N_LON)), v_int, np.zeros((1, N_LON))],
                            axis=0).astype(float)
    cfg = LatLonCGridOceanConfig.from_flat()
    rl = build_rigid_lid_data(Hb, lm, u_mask, v_mask, cfg, grid, periodic_x=True)
    return grid, lm, Hb, u_mask, v_mask, cfg, rl


# ---------------------------------------------------------------- operator ----
def test_operator_constant_nullspace():
    """L(const) = 0 on the wet interior (the Laplacian annihilates constants)."""
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    const = jnp.ones((N_LAT + 1, N_LON + 1))
    L = streamfunction_vorticity_operator(const, rl.inv_H_u, rl.inv_H_v, grid,
                                          u_mask=rl.u_mask, v_mask=rl.v_mask)
    assert float(jnp.max(jnp.abs(L * rl.solve_mask))) < 1e-18


def test_psin_basis_is_harmonic():
    """Each island basis function satisfies L(psin_k) ≈ 0 on the interior."""
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    for k in range(rl.nisle):
        L = streamfunction_vorticity_operator(rl.psin[..., k], rl.inv_H_u,
                                              rl.inv_H_v, grid,
                                              u_mask=rl.u_mask, v_mask=rl.v_mask)
        assert float(jnp.max(jnp.abs(L * rl.solve_mask))) < 1e-12


def test_manufactured_poisson_roundtrip():
    """Solve L(ψ)=rhs for a manufactured ψ; recover it on the interior."""
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    latv = np.linspace(-1, 1, N_LAT + 1)[:, None]
    lonv = np.linspace(0, 2 * np.pi, N_LON + 1)[None, :]
    psi_true = jnp.asarray(np.cos(np.pi / 2 * latv) * np.sin(lonv)) * rl.solve_mask
    rhs = streamfunction_vorticity_operator(psi_true, rl.inv_H_u, rl.inv_H_v, grid,
                                            u_mask=rl.u_mask, v_mask=rl.v_mask)
    psi_solved = solve_streamfunction_interior(
        rhs, rl, grid, jnp.zeros_like(psi_true), tol=1e-12, maxiter=2000)
    rel = float(jnp.max(jnp.abs((psi_solved - psi_true) * rl.solve_mask))
                / jnp.max(jnp.abs(psi_true)))
    assert rel < 1e-6


def test_recovered_transport_is_nondivergent():
    """H·u_bt = ∇⊥ψ is discretely non-divergent (div ≈ 0)."""
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    latv = np.linspace(-1, 1, N_LAT + 1)[:, None]
    lonv = np.linspace(0, 2 * np.pi, N_LON + 1)[None, :]
    psi = jnp.asarray(np.cos(np.pi / 2 * latv) * np.cos(lonv)) * rl.solve_mask
    u_bt, v_bt = recover_velocity_from_streamfunction(
        psi, rl.inv_H_u, rl.inv_H_v, grid, u_mask=rl.u_mask, v_mask=rl.v_mask)
    # transport = H·u_bt = -∂ψ/∂y, H·v_bt = +∂ψ/∂x  (the H cancels inv_H).
    H_u = 1.0 / jnp.where(rl.inv_H_u > 0, rl.inv_H_u, jnp.inf)
    H_v = 1.0 / jnp.where(rl.inv_H_v > 0, rl.inv_H_v, jnp.inf)
    div = divergence_cgrid(H_u * u_bt, H_v * v_bt, grid,
                           u_mask=rl.u_mask, v_mask=rl.v_mask)
    assert float(jnp.max(jnp.abs(div * lm))) < 1e-9


def test_vertex_area_positive_interior_zero_poles():
    grid, *_ = _channel()
    A = vertex_area_cgrid(grid)
    assert A.shape == (N_LAT + 1, N_LON + 1)
    assert float(jnp.min(A[1:-1])) > 0.0
    assert float(jnp.max(jnp.abs(A[0]))) == 0.0
    assert float(jnp.max(jnp.abs(A[-1]))) == 0.0


# ------------------------------------------------------------------ islands ----
def test_channel_has_two_islands():
    """Periodic channel with S + N walls => 2 disconnected land masses."""
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    assert rl.nisle == 2
    assert rl.psin.shape == (N_LAT + 1, N_LON + 1, 2)
    assert rl.line_psin.shape == (2, 2)


def test_line_psin_symmetric_and_free_block_nonsingular():
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    lp = np.asarray(rl.line_psin)
    assert np.allclose(lp, lp.T, atol=1e-12 * max(1.0, np.abs(lp).max()))
    # The free-island block [1:,1:] (here 1x1) must be invertible (transport mode).
    assert abs(np.linalg.det(lp[1:, 1:])) > 0.0


def test_island_vertex_masks_disjoint():
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    # Each island's vertices are nonempty and the two walls don't share vertices.
    s0 = jnp.sum(rl.island_vertex_masks[0])
    s1 = jnp.sum(rl.island_vertex_masks[1])
    assert float(s0) > 0 and float(s1) > 0
    overlap = jnp.sum(rl.island_vertex_masks[0] * rl.island_vertex_masks[1])
    assert float(overlap) == 0.0


def test_face_depths_min_rule_and_guards():
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    H_u, H_v, inv_H_u, inv_H_v = barotropic_face_depths(
        jnp.asarray(Hb), jnp.asarray(lm), grid)
    # Wet interior faces: depth 4000, reciprocal 1/4000.
    assert float(jnp.max(H_u)) == pytest.approx(4000.0)
    # inv guards: 0 where dry (no inf/nan).
    assert bool(jnp.all(jnp.isfinite(inv_H_u))) and bool(jnp.all(jnp.isfinite(inv_H_v)))
    assert float(jnp.max(inv_H_v[0])) == 0.0  # south wall row dry


def test_solve_island_constants_no_free_islands():
    """nisle<=1 -> no transport mode -> zeros."""
    fake = RigidLidStaticData(
        inv_H_u=None, inv_H_v=None, u_mask=None, v_mask=None, solve_mask=None,
        A_vertex=None, inv_diag=None, psin=None, line_psin=jnp.zeros((1, 1)),
        island_vertex_masks=None, nisle=1)
    out = _solve_island_constants(jnp.zeros((1,)), fake)
    assert float(jnp.max(jnp.abs(out))) == 0.0


# ------------------------------------------------------------ step + grad ----
def test_rigid_lid_step_forced_response_and_grad():
    """A forced step produces finite ψ, a nonzero transport mode, and a grad."""
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    z = (N_LAT + 1, N_LON + 1)
    psi0 = jnp.zeros(z); dpsin0 = jnp.zeros((rl.nisle,))
    F_u = jnp.full((N_LAT, N_LON + 1), 1.0e-6) * rl.u_mask
    F_v = jnp.zeros((N_LAT + 1, N_LON))
    out = rigid_lid_step(psi0, psi0, psi0, dpsin0, dpsin0, F_u, F_v,
                         rl, 4800.0, cfg, grid)
    psi_new, dpsi_new, _, dpsin_new, _, u_bt, v_bt = out
    assert bool(jnp.all(jnp.isfinite(psi_new)))
    assert float(jnp.max(jnp.abs(psi_new))) > 0.0           # responds to forcing
    assert float(jnp.abs(dpsin_new[1])) > 0.0               # transport mode active

    # Gradient through the recovered VELOCITY (u_bt,v_bt) — the cotangent that
    # exposed the ill-conditioned-VJP NaN bug before Jacobi row-scaling.  Check
    # it matches a finite difference (guards the BUG-1 regression).
    def loss(amp):
        Fu = jnp.full((N_LAT, N_LON + 1), amp) * rl.u_mask
        o = rigid_lid_step(psi0, psi0, psi0, dpsin0, dpsin0, Fu, F_v,
                           rl, 4800.0, cfg, grid)
        return jnp.sum(o[5] ** 2) + jnp.sum(o[6] ** 2)   # u_bt, v_bt
    a0 = 1.0e-6
    g = jax.grad(loss)(a0)
    assert bool(jnp.isfinite(g)) and g != 0.0
    fd = (loss(a0 + 1e-9) - loss(a0 - 1e-9)) / 2e-9
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-30      # AD == finite-difference


# --------------------------------------------------------------- coupling ----
def test_dispatch_accepts_rigid_lid_rejects_bad():
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z = create_ocean_z_star(n_levels=4)
    LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(barotropic_solver="rigid_lid"))
    with pytest.raises(ValueError, match="barotropic_solver"):
        LatLonCGridOceanModel(grid, z, LatLonCGridOceanConfig.from_flat(barotropic_solver="bogus"))


def test_coupled_rigid_lid_runs_eta_unchanged_and_differentiable():
    """Full coupled step: eta stays 0 (rigid lid), runs finite, differentiable."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z = create_ocean_z_star(n_levels=4)
    lm = np.ones((N_LAT, N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((N_LAT, N_LON), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=jnp.asarray(lm), H_bathy_override=jnp.asarray(Hb))
    cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="rigid_lid")
    model = LatLonCGridOceanModel(grid, z, cfg)
    final, _ = model.integrate_scan(state, n_steps=5, dt=3600.0)
    assert bool(jnp.all(jnp.isfinite(final.psi)))
    assert bool(jnp.all(jnp.isfinite(final.u.data)))
    assert float(jnp.max(jnp.abs(final.eta.data))) == 0.0   # rigid lid: eta unchanged

    def loss(scale):
        st = state._replace(T=state.T.replace(data=state.T.data * scale))
        f, _ = model.integrate_scan(st, n_steps=3, dt=3600.0)
        return jnp.sum(f.u.data ** 2)
    g = jax.grad(loss)(1.0)
    assert bool(jnp.isfinite(g))


# ----------------------------------------------- MPI/SPMD dispatch guard ----
# The rigid-lid streamfunction solve is single-rank only (global elliptic CG
# with rank-local dots + domain-wide island line integrals). The guard must
# FAIL LOUDLY on a decomposed domain instead of returning a per-rank-wrong
# answer. These tests exercise the predicate's serial / MPI / SPMD branches
# and confirm the guard is wired at the solver entry.
import types  # noqa: E402

from legoesm.ocean.dynamics import rigid_lid_latlon_cgrid as _rll  # noqa: E402
import legoesm.grids.halo as _halo  # noqa: E402


def test_guard_predicate_serial_is_false():
    """Serial (single rank, no SPMD backend) is NOT decomposed."""
    assert _rll.rigid_lid_is_decomposed() is False
    _rll.assert_rigid_lid_single_rank()  # must not raise


def test_guard_predicate_true_under_multiprocess(monkeypatch):
    """is_multi_process() True (mpi4jax / multi-host / Voronoi) ⇒ decomposed."""
    monkeypatch.setattr(_rll, "is_multi_process", lambda: True)
    assert _rll.rigid_lid_is_decomposed() is True
    with pytest.raises(NotImplementedError, match="single-rank only"):
        _rll.assert_rigid_lid_single_rank()


def test_guard_predicate_spmd_latband(monkeypatch):
    """SPMD lat-band mesh with >1 device ⇒ decomposed (is_multi_process False)."""
    monkeypatch.setattr(_rll, "is_multi_process", lambda: False)
    monkeypatch.setattr(_halo, "get_halo_backend", lambda: "spmd")
    # >1 device on "lat" is a real decomposition.
    monkeypatch.setattr(_halo, "get_spmd_mesh",
                        lambda: types.SimpleNamespace(shape={"lat": 2}))
    assert _rll.rigid_lid_is_decomposed() is True
    # A single-device "lat" axis is NOT a decomposition (falls through to
    # is_multi_process, here False).
    monkeypatch.setattr(_halo, "get_spmd_mesh",
                        lambda: types.SimpleNamespace(shape={"lat": 1}))
    assert _rll.rigid_lid_is_decomposed() is False
    # A cube-atm SPMD mesh (no "lat" axis) also falls through.
    monkeypatch.setattr(_halo, "get_spmd_mesh",
                        lambda: types.SimpleNamespace(shape={"face": 6}))
    assert _rll.rigid_lid_is_decomposed() is False


def test_guard_spmd_backend_without_mesh_raises(monkeypatch):
    """Backend armed 'spmd' but no mesh set is an invalid state ⇒ fail fast."""
    monkeypatch.setattr(_halo, "get_halo_backend", lambda: "spmd")
    monkeypatch.setattr(_halo, "get_spmd_mesh", lambda: None)
    with pytest.raises(RuntimeError, match="no SPMD mesh"):
        _rll.rigid_lid_is_decomposed()


def test_guard_wired_at_solver_entry_integration(monkeypatch):
    """Full coupled rigid-lid step raises (not silently wrong) when decomposed."""
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z = create_ocean_z_star(n_levels=4)
    lm = np.ones((N_LAT, N_LON)); lm[:2] = 0.0; lm[-2:] = 0.0
    Hb = np.full((N_LAT, N_LON), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=jnp.asarray(lm), H_bathy_override=jnp.asarray(Hb))
    cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="rigid_lid")
    model = LatLonCGridOceanModel(grid, z, cfg)
    monkeypatch.setattr(_rll, "is_multi_process", lambda: True)
    with pytest.raises(NotImplementedError, match="single-rank only"):
        model.integrate_scan(state, n_steps=1, dt=3600.0)


def _rigid_lid_model():
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z = create_ocean_z_star(n_levels=4)
    lm = np.ones((N_LAT, N_LON))
    lm[:2] = 0.0
    lm[-2:] = 0.0
    Hb = np.full((N_LAT, N_LON), 4000.0)
    state = rest_state_latlon_cgrid_ocean(
        grid, z, land_mask_override=jnp.asarray(lm), H_bathy_override=jnp.asarray(Hb))
    cfg = LatLonCGridOceanConfig.from_flat(barotropic_solver="rigid_lid")
    return LatLonCGridOceanModel(grid, z, cfg), state


def test_guard_fires_before_rigid_lid_data_build(monkeypatch):
    """Decomposition is caught BEFORE _ensure_rigid_lid_data builds the ψ-basis.

    build_rigid_lid_data itself runs the global streamfunction-basis solves +
    island line integrals, so the guard must fire before construction, not only
    inside the solver body (codex finding 1).
    """
    model, state = _rigid_lid_model()
    assert model.rigid_lid_data is None
    monkeypatch.setattr(_rll, "is_multi_process", lambda: True)
    with pytest.raises(NotImplementedError, match="single-rank only"):
        model._ensure_rigid_lid_data(state)
    # No per-rank-wrong basis was constructed.
    assert model.rigid_lid_data is None


def test_guard_eager_survives_jit_cache_reuse(monkeypatch):
    """A serially-compiled step still refuses to run once decomposed.

    The in-body guard is host-Python at trace time, so a step traced+compiled
    serially would be reused verbatim if the process later goes distributed
    (codex finding 2). The eager guard in step() fires on every call, before
    the cached _step_jitted, so the reuse is refused.
    """
    model, state = _rigid_lid_model()
    # Build the rigid-lid data eagerly from concrete masks (documented contract:
    # the in-jit build cannot np.asarray traced masks), then warm the compiled
    # step on a single rank.
    model._ensure_rigid_lid_data(state)
    state1 = model.step(state, 3600.0)
    assert bool(jnp.all(jnp.isfinite(state1.psi)))
    assert model.rigid_lid_data is not None            # cache is warm
    # Now the process "becomes distributed": the compiled step is cached, but
    # the eager guard must still fire.
    monkeypatch.setattr(_rll, "is_multi_process", lambda: True)
    with pytest.raises(NotImplementedError, match="single-rank only"):
        model.step(state1, 3600.0)


def test_guard_leaf_solve_streamfunction_interior(monkeypatch):
    """Direct solve_streamfunction_interior refuses a decomposed domain.

    Covers a direct/library call (and rigid_lid_step, which reaches it) that
    bypasses the model wrapper (codex round-2 finding).
    """
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    rhs = jnp.zeros((N_LAT + 1, N_LON + 1))
    monkeypatch.setattr(_rll, "is_multi_process", lambda: True)
    with pytest.raises(NotImplementedError, match="single-rank only"):
        solve_streamfunction_interior(
            rhs, rl, grid, jnp.zeros_like(rhs), tol=1e-10, maxiter=100)


def test_guard_leaf_build_rigid_lid_data(monkeypatch):
    """Direct build_rigid_lid_data refuses a decomposed domain BEFORE building.

    The basis build runs the global CG + domain-wide line integrals, so the
    guard fires at the TOP of build_rigid_lid_data (codex round-2 finding).
    Non-vacuous for the TOP guard specifically: _label_islands is booby-trapped
    so that if the top guard were removed (and the raise instead came from the
    downstream solve_streamfunction_interior guard), the build would first reach
    _label_islands and surface a different error — failing this test.
    """
    import legoesm.ocean.dynamics.rigid_lid_islands as _isl
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()   # serial build OK

    def _boom(*a, **k):
        raise AssertionError("build reached _label_islands — top-of-build guard missing")
    monkeypatch.setattr(_isl, "_label_islands", _boom)
    monkeypatch.setattr(_rll, "is_multi_process", lambda: True)
    with pytest.raises(NotImplementedError, match="single-rank only"):
        build_rigid_lid_data(Hb, lm, u_mask, v_mask, cfg, grid, periodic_x=True)


def test_guard_leaf_island_line_integrals(monkeypatch):
    """Direct island_line_integrals refuses a decomposed domain.

    The domain-wide island jnp.sum is rank-local under decomposition; guard
    covers direct callers such as build_line_psin (codex round-3 finding).
    """
    grid, lm, Hb, u_mask, v_mask, cfg, rl = _channel()
    u_f = jnp.zeros((N_LAT, N_LON + 1))
    v_f = jnp.zeros((N_LAT + 1, N_LON))
    monkeypatch.setattr(_rll, "is_multi_process", lambda: True)
    with pytest.raises(NotImplementedError, match="single-rank only"):
        _rll.island_line_integrals(u_f, v_f, rl, grid)
