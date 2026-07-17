"""Stage 5 — the serial-vs-SPMD EQUIVALENCE GATE for the atm latlon step.

THE make-or-break validation of make_sharded_atm_latlon_step: N-band lat-band
SPMD integration must reproduce the single-device C-grid hydrostatic step. A
subtly-wrong band decomposition (wrong Coriolis / v-face interp at interior
cuts, band-local mass denominator, mis-reconstructed v-face row, wrong band
geometry) gives SILENT wrong answers and is caught here.

Two complementary gates:

1. ``test_atm_latlon_spmd_tendency_matches_serial`` — the RIGOROUS,
   RK-stage-independent decomposition check at the TENDENCY level.  The
   vector-invariant momentum (vorticity + Bernoulli gradient, incl. the
   absolute-vorticity Coriolis) and the centered flux-form continuity
   reconstruct the band cut EXACTLY from halo'd neighbour rows, so du/dv/dp_s
   and cor_u/cor_v are BIT-EXACT (fp64) on every band.  The ONLY non-bit-exact
   term is the LIMITED FV PPM scalar (T/tracer) advection: ``ppm_edge_values``
   uses a 2nd-order edge at the outermost ``halo=2`` padded row, where serial
   computes a 4th-order edge, and the CW84 limiter leaks a ~5e-12 difference
   into the two cut-ADJACENT T rows (verified: scalar-advection diff is
   confined to those rows; the scalar halo ghost rows are byte-identical).
   This is a PRE-EXISTING boundary-order property of band-decomposed limited
   FV PPM — the production MPI lat-band path uses the SAME backend-dispatched
   operator — NOT an SPMD bug, so it is asserted as a BOUNDED residual.

2. ``test_atm_latlon_spmd_step_matches_serial`` — the integrated full-step
   gate through the PUBLIC make_sharded interface.  Over a few RK3 steps the
   tiny cut-row T truncation advects into the small-magnitude u field; bound it
   FAR below any real decomposition error (~1e-3) while above the PPM
   truncation floor.

Runs on host CPU devices (``XLA_FLAGS=--xla_force_host_platform_device_count=4``);
the production target is multi-GPU/TPU but the shard_map/ppermute/psum logic is
device-agnostic.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
    make_sharded_atm_latlon_step,
    shard_state_atm_latlon,
    gather_state_atm_latlon,
    run_atm_latlon_spmd_segment,
    build_band_grids_atm,
    atm_grid_array_field_names,
    lat_spec,
)
from legoesm.parallel.latlon_spmd import replicate_leaf
from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
    cgrid_to_hydrostatic,
)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

N_DEV = 4
N_LAT = 16        # divisible by N_DEV
N_LON = 16
NLEV = 4


@pytest.fixture(autouse=True)
def _restore_halo_backend():
    yield
    from legoesm.grids.halo import set_halo_backend
    set_halo_backend("local")


def _mesh():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")
    return jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))


def _model_and_state(use_polar_filter: bool):
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True,                 # exercises batch_global_area_sums psum
        use_polar_filter=use_polar_filter,
        use_ppm_transport=True,
        time_integrator="ssp_rk3",
    )
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    rng = np.random.default_rng(31337)
    eps = 1.0e-3
    state = CGridLatLonHydrostaticState(
        u=jnp.asarray(eps * rng.standard_normal((N_LAT, N_LON + 1, NLEV))),
        v=jnp.asarray(eps * rng.standard_normal((N_LAT + 1, N_LON, NLEV))),
        T=jnp.asarray(300.0 + eps * rng.standard_normal((N_LAT, N_LON, NLEV))),
        p_s=jnp.asarray(1.0e5 + 10.0 * rng.standard_normal((N_LAT, N_LON))),
        phis=jnp.zeros((N_LAT, N_LON)),
    )
    # Pole-wall v (the state the model maintains: v == 0 at both poles), so the
    # v_lower drop + gather re-append round-trips. np.array makes a WRITABLE copy
    # (np.asarray of a jax array is read-only).
    v0 = np.array(state.v)
    v0[0] = 0.0
    v0[-1] = 0.0
    state = state._replace(v=jnp.asarray(v0))
    return model, state


def _cut_adjacent_lat_rows():
    """The two cell rows flanking every interior band cut (the only rows the
    limited FV PPM scalar advection truncates under band decomposition)."""
    nl = N_LAT // N_DEV
    rows = set()
    for r in range(1, N_DEV):
        rows.add(r * nl - 1)   # last cell of the south band
        rows.add(r * nl)       # first cell of the north band
    return rows


def _serial_and_band_tendency(use_polar_filter):
    """Serial ``cgrid_latlon_hydrostatic_tendencies`` and the N-band shard_map
    body's tendency, gathered to global. Returns ((du,dv,dT,dps)_serial,
    (du,dv,dT,dps)_band) with dv/band-dv as full-length v-face arrays."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        cgrid_latlon_hydrostatic_tendencies)
    from legoesm.parallel.latlon_spmd import (
        latlon_band_perms, reconstruct_vface_lower, to_vface_lower,
        activate_latlon_spmd_halo, deactivate_latlon_spmd_halo)
    from legoesm.parallel.shard_map_compat import shard_map

    mesh = _mesh()
    model, state = _model_and_state(use_polar_filter)
    grid, sigma, cfg = model.grid, model.sigma_coord, model.config

    du_s, dv_s, dT_s, dps_s, _ = cgrid_latlon_hydrostatic_tendencies(
        state, grid, sigma, cfg)
    serial = tuple(np.asarray(x) for x in (du_s, dv_s, dT_s, dps_s))

    band_grids = build_band_grids_atm(grid, N_DEV)
    template = band_grids[0]
    afn = atm_grid_array_field_names(template)
    rep = NamedSharding(mesh, P())
    stacks = {n: jax.device_put(jnp.stack([jnp.asarray(getattr(g, n))
              for g in band_grids], 0), rep) for n in afn}
    perm_north, _ = latlon_band_perms(N_DEV)

    def _body(sl, st):
        r = jax.lax.axis_index("lat")
        bg = template._replace(**{n: st[n][r] for n in afn})
        vf = reconstruct_vface_lower(sl.v, "lat", perm_north)
        du, dv, dT, dps, _ = cgrid_latlon_hydrostatic_tendencies(
            sl._replace(v=vf), bg, sigma, cfg)
        return du, to_vface_lower(dv), dT, dps

    sc = shard_state_atm_latlon(state, mesh)
    in_spec = jax.tree.map(lat_spec, sc)
    sp3 = P("lat", None, None)
    sp2 = P("lat", None)
    fn = shard_map(
        _body, mesh=mesh,
        in_specs=(in_spec, jax.tree.map(lambda _x: P(), stacks)),
        out_specs=(sp3, sp3, sp3, sp2), check_vma=False)
    activate_latlon_spmd_halo(mesh)
    try:
        out = fn(sc, stacks)
    finally:
        deactivate_latlon_spmd_halo()
    g = lambda x: np.asarray(jax.device_put(x, NamedSharding(mesh, P())))
    du_b, dvl_b, dT_b, dps_b = (g(x) for x in out)
    # band returns v_lower (N_LAT rows); re-cap the north pole row (=0) so the
    # shape matches the serial (N_LAT+1)-row v-face array for comparison.
    dv_b = np.concatenate([dvl_b, np.zeros_like(dvl_b[:1])], axis=0)
    return serial, (du_b, dv_b, dT_b, dps_b)


@pytest.mark.parametrize("use_polar_filter", [False, True])
def test_atm_latlon_spmd_tendency_matches_serial(use_polar_filter):
    """RIGOROUS decomposition gate: the single-tendency band reconstruction is
    BIT-EXACT for momentum + continuity; only the limited FV PPM T advection
    carries a bounded ~5e-12 truncation at the cut-adjacent rows (see module
    docstring). RK-stage-independent, so it isolates the decomposition from the
    downstream propagation tested below."""
    (du_s, dv_s, dT_s, dps_s), (du_b, dv_b, dT_b, dps_b) = \
        _serial_and_band_tendency(use_polar_filter)

    # The zero north-pole cap re-appended to dv_b (the band drops the shared
    # north v-face row) is only valid because BOTH serial and band zero the
    # physical poles by the same wall BC (apply_pole_end_masks / _zero_v_at_pole).
    # Assert that invariant explicitly so the cap can't mask a pole-BC bug; the
    # band's own pole rows are additionally exercised by the production gather in
    # test_atm_latlon_spmd_step_matches_serial over a full step.
    np.testing.assert_array_equal(
        dv_s[0], np.zeros_like(dv_s[0]),
        err_msg="serial south-pole dv_dt is not zero — wall BC invariant broke.")
    np.testing.assert_array_equal(
        dv_s[-1], np.zeros_like(dv_s[-1]),
        err_msg="serial north-pole dv_dt is not zero — wall BC invariant broke.")

    # Momentum (incl. absolute-vorticity Coriolis) + continuity: BIT-EXACT.
    for name, a, b in (("du_dt", du_b, du_s), ("dv_dt", dv_b, dv_s),
                       ("dp_s_dt", dps_b, dps_s)):
        np.testing.assert_allclose(
            a, b, rtol=1e-10, atol=1e-12,
            err_msg=(
                f"atm lat-band SPMD tendency '{name}' diverged from serial "
                f"beyond fp64 machine precision — a real band-decomposition "
                f"bug (Coriolis/v-face interp at cuts, band geometry, mass "
                f"denominator, or v-face reconstruction)."))

    # Temperature: bit-exact at every INTERIOR row; cut-adjacent rows carry the
    # bounded FV-PPM halo-2 boundary-order truncation only.
    cut_rows = _cut_adjacent_lat_rows()
    interior = [j for j in range(N_LAT) if j not in cut_rows]
    np.testing.assert_allclose(
        dT_b[interior], dT_s[interior], rtol=1e-10, atol=1e-12,
        err_msg="dT_dt diverged at an INTERIOR (non-cut) row — real bug.")
    cut_resid = float(np.max(np.abs(dT_b[sorted(cut_rows)] - dT_s[sorted(cut_rows)])))
    assert cut_resid < 1e-9, (
        f"cut-row dT_dt residual {cut_resid:.2e} exceeds the FV-PPM halo-2 "
        f"truncation floor (~5e-12) by too much — investigate (expected the "
        f"limited PPM boundary-order reduction only, not a gross cut bug).")


@pytest.mark.parametrize("use_polar_filter", [False, True])
def test_atm_latlon_spmd_step_matches_serial(use_polar_filter):
    """Integrated full-step gate through the PUBLIC make_sharded interface. Over
    a few RK3 steps the bounded cut-row T truncation (see the tendency gate)
    advects into the small-magnitude u field; this asserts the integrated
    difference stays FAR below any real decomposition error (~1e-3) while above
    the PPM truncation floor."""
    mesh = _mesh()
    model, state = _model_and_state(use_polar_filter)
    dt = 100.0
    n_steps = 3

    # Serial reference: the single-device C-grid step (no physics, no anchor).
    s = state
    for _ in range(n_steps):
        s, _ = model._step_cgrid(s, dt, target_mass=None, physics_fn=None)
    serial_out = s

    # SPMD: lay out -> step N times on N bands -> gather.
    sharded_step = make_sharded_atm_latlon_step(model, mesh)
    sc = shard_state_atm_latlon(state, mesh)
    for _ in range(n_steps):
        sc = sharded_step(sc, dt)
    spmd_out = gather_state_atm_latlon(sc, mesh)

    # atol=1e-9 bounds the integrated FV-PPM cut truncation (measured ~8e-11 in
    # u after 3 steps); a real band-decomposition bug is O(1e-3), 6+ orders up.
    for field in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(spmd_out, field))
        b = np.asarray(getattr(serial_out, field))
        assert a.shape == b.shape, f"{field} shape {a.shape} vs {b.shape}"
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-9,
            err_msg=(
                f"atm lat-band SPMD ({N_DEV} bands, polar_filter="
                f"{use_polar_filter}) diverged from serial in '{field}' beyond "
                f"the FV-PPM cut-truncation bound — a real band-decomposition "
                f"bug (Coriolis/v-face interp at cuts, band geometry, mass "
                f"denominator, or v-face reconstruction)."))


def test_replicate_leaf_multiprocess_branch_matches_device_put():
    """The multi-controller gather branch (jit-compiled identity with
    replicated out_shardings) must produce the SAME replicated array as the
    single-process device_put branch — on values, sharding, and for both a
    lat-sharded and an already-replicated input. This exercises the
    ``multiprocess=True`` code path for real on a single process (where both
    mechanisms are legal), so the route-B gather cannot silently diverge.
    Covers the primitive SHARED by the atm and ocean gathers
    (``legoesm.parallel.latlon_spmd.replicate_leaf``)."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = _mesh()
    rep = NamedSharding(mesh, P())
    rng = np.random.default_rng(7)
    full = jnp.asarray(rng.standard_normal((N_LAT, N_LON, NLEV)))
    sharded = jax.device_put(full, NamedSharding(mesh, P("lat", None, None)))

    for arr in (sharded, jax.device_put(full, rep)):
        a = replicate_leaf(arr, rep, multiprocess=False)
        b = replicate_leaf(arr, rep, multiprocess=True)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_array_equal(np.asarray(a), np.asarray(full))
        assert b.sharding.is_fully_replicated, (
            "multiprocess replicate branch did not produce a fully "
            "replicated sharding")


def test_make_sharded_atm_step_rejects_anchor_mass():
    """Dispatch-hardening: anchor_mass_to_initial uses a band-local sum target
    not yet SPMD-routed -> must raise loudly, not silently mis-anchor."""
    mesh = _mesh()
    grid = create_latlon_grid(
        n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
        omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(anchor_mass_to_initial=True)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    with pytest.raises(NotImplementedError, match="anchor_mass_to_initial"):
        make_sharded_atm_latlon_step(model, mesh)


# ==============================================================================
# Stage 7 — the cell-centered HydrostaticState <-> sharded C-grid DRIVER bridge.
# run_atm_latlon_spmd_segment converts+shards a HydrostaticState ONCE, runs N
# sharded C-grid steps, gathers+converts back ONCE. It must match the serial
# model.step(HydrostaticState) loop, which stays C-grid across the loop via its
# _cgrid_cache — so the lossy cell<->face conversion happens once on BOTH paths.
# ==============================================================================

def _hs_from_cgrid_state(model, c_state):
    """Build a cell-centered HydrostaticState IC from the C-grid test state."""
    return cgrid_to_hydrostatic(c_state, model.grid)


def test_run_atm_latlon_spmd_segment_single_device_bit_exact():
    """mesh=None: the segment driver reduces to the serial cgrid step with the
    SAME boundary conversions, so it is BIT-EXACT to the serial model.step loop.
    Needs no extra devices -> runs anywhere."""
    model, c_state = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(model, c_state)
    dt, n_steps = 100.0, 3

    serial = _model_and_state(use_polar_filter=False)[0]  # fresh model, no cache
    hs_s = _hs_from_cgrid_state(serial, c_state)
    for _ in range(n_steps):
        hs_s = serial.step(hs_s, dt)

    hs_b = run_atm_latlon_spmd_segment(model, None, hs0, dt, n_steps)
    for field in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(hs_b, field).data),
            np.asarray(getattr(hs_s, field).data),
            rtol=1e-12, atol=1e-13,
            err_msg=f"mesh=None segment diverged from serial in '{field}'.")


def test_run_atm_latlon_spmd_segment_rejects_bad_nsteps():
    """Dispatch-hardening: n_steps < 1 must raise, never silently no-op."""
    model, _ = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(model, _model_and_state(False)[1])
    with pytest.raises(ValueError, match="n_steps"):
        run_atm_latlon_spmd_segment(model, None, hs0, 100.0, 0)


@pytest.mark.parametrize("use_polar_filter", [False, True])
def test_run_atm_latlon_spmd_segment_matches_serial(use_polar_filter):
    """The Stage-7 end-to-end gate: the N-band SPMD segment driven from a
    cell-centered HydrostaticState matches the serial model.step loop at the
    Stage-5 integrated bound (the bridge composes the validated step + the lossy
    boundary conversions identically on both paths)."""
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        shard_hydrostatic_to_atm_latlon)
    mesh = _mesh()
    # Two independent model instances so serial-loop cache mutation cannot leak
    # into the SPMD path (SPMD uses _step_cgrid_impl, cache-free, but be explicit).
    serial, c_state = _model_and_state(use_polar_filter)
    spmd_model, _ = _model_and_state(use_polar_filter)
    hs0 = _hs_from_cgrid_state(serial, c_state)
    dt, n_steps = 100.0, 3

    # Non-vacuity guard A — the bridge ACTUALLY shards across the mesh (else the
    # segment trivially == serial and the comparison below proves nothing).
    sharded_c = shard_hydrostatic_to_atm_latlon(hs0, spmd_model.grid, mesh)
    assert sharded_c.T.sharding.num_devices == N_DEV, (
        f"bridge did not shard across the mesh: "
        f"{sharded_c.T.sharding.num_devices} devices != {N_DEV}")

    hs_s = hs0
    for _ in range(n_steps):
        hs_s = serial.step(hs_s, dt)

    hs_b = run_atm_latlon_spmd_segment(spmd_model, mesh, hs0, dt, n_steps)

    for field in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(hs_b, field).data)
        b = np.asarray(getattr(hs_s, field).data)
        assert a.shape == b.shape, f"{field} shape {a.shape} vs {b.shape}"
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-9,
            err_msg=(
                f"Stage-7 SPMD segment (HydrostaticState, polar_filter="
                f"{use_polar_filter}) diverged from the serial model.step loop "
                f"in '{field}' beyond the FV-PPM cut-truncation bound."))

    # Non-vacuity guard B — the band decomposition is GENUINELY exercised: the
    # limited-FV-PPM scalar advection truncates at cut rows (~8e-11 in u after 3
    # steps, Stage 5), so the SPMD trajectory must differ from serial by MORE
    # than the fp64 roundoff floor. A zero diff would mean sharding silently
    # collapsed to single-device (the one way this gate could pass vacuously).
    u_diff = float(np.max(np.abs(
        np.asarray(hs_b.u.data) - np.asarray(hs_s.u.data))))
    assert u_diff > 1e-13, (
        f"SPMD u is bit-identical to serial ({u_diff:.2e}) — the FV-PPM cut "
        f"truncation is absent, so the band decomposition did not actually run "
        f"(sharding collapsed to single-device); this gate would be vacuous.")


# ==============================================================================
# physics_fn threading — a STATELESS, COLUMN-LOCAL physics (Held-Suarez) must
# thread through the SPMD step and match the serial model.step(physics_fn) loop.
# The band grid is routed into _call_physics so the lat-dependent forcing sees
# each band's latitudes; column-local physics adds no cross-band coupling.
# ==============================================================================

def _mk_phys_state(ncol, nlev):
    """Minimal all-zeros PhysicsState with a nonzero deterministic tke seed."""
    import jax as _jax
    from legoesm.atmosphere.physics.physics_state import (
        NO_SFC_T_OVERRIDE, PhysicsState,
    )
    return PhysicsState(
        tke=jnp.full((ncol, nlev), 0.01),
        conv_prog_profile=jnp.zeros((ncol, nlev)),
        conv_stoch_state=jnp.zeros((ncol,)),
        gwd_spectrum=jnp.zeros((ncol, 1, 1)),
        prng_key=_jax.random.PRNGKey(0),
        surface_T_sfc_override=jnp.full((ncol,), NO_SFC_T_OVERRIDE),  # #911 finite sentinel
        qke=jnp.zeros((ncol, nlev)),
        clubb_moments=jnp.zeros((ncol, 15, nlev + 1)),
        rad_heating=jnp.zeros((ncol, nlev)),
        col_index=jnp.arange(ncol, dtype=jnp.int32),
    )


def test_stochastic_draw_is_decomposition_invariant():
    """A1 increment 2: a stochastic per-column draw (fold_in with the
    GLOBAL col_index carried in PhysicsState) must produce the SAME
    trajectory and carry-out under 4-band SPMD as the serial twin —
    stochastic physics is no longer refused, it is invariant."""
    import jax as _jax
    mesh = _mesh()
    serial, c_state = _model_and_state(use_polar_filter=False)
    spmd_model, _ = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(serial, c_state)
    dt, n_steps = 100.0, 3
    nlat, nlon, nlev = np.asarray(hs0.T.data).shape
    ps0 = _mk_phys_state(nlat * nlon, nlev)

    def _stoch_phys(hs, grid, sigma, ps):
        # Mirrors the Bechtold invariant-draw pattern: per-GLOBAL-column
        # fold of a per-step sub-key, AR1 carry in conv_stoch_state, and
        # a T tendency scaled by the noise — trajectory-coupled.
        from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
        tend = held_suarez_forcing_latlon(hs, grid, sigma)
        nlat_loc, nlon_loc, _ = hs.T.data.shape
        sub, master_new = _jax.random.split(ps.prng_key, 2)
        eps = _jax.vmap(
            lambda i: _jax.random.normal(_jax.random.fold_in(sub, i))
        )(ps.col_index)
        stoch_new = 0.8 * ps.conv_stoch_state + 0.2 * eps
        factor = (1.0 + 1e-3 * stoch_new).reshape(nlat_loc, nlon_loc, 1)
        tend = tend._replace(
            dT_dt=tend.dT_dt.replace(data=tend.dT_dt.data * factor))
        return tend, ps._replace(conv_stoch_state=stoch_new,
                                 prng_key=master_new)

    _stoch_phys._requires_phys_state = True

    hs_s, ps_s = run_atm_latlon_spmd_segment(
        serial, None, hs0, dt, n_steps,
        physics_fn=_stoch_phys, phys_state=ps0)
    hs_b, ps_b = run_atm_latlon_spmd_segment(
        spmd_model, mesh, hs0, dt, n_steps,
        physics_fn=_stoch_phys, phys_state=ps0)

    # Non-vacuity: the noise actually perturbed the AR1 carry.
    assert float(np.max(np.abs(np.asarray(ps_b.conv_stoch_state)))) > 1e-6

    for field in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(hs_b, field).data),
            np.asarray(getattr(hs_s, field).data),
            rtol=1e-6, atol=1e-9,
            err_msg=f"stochastic SPMD diverged from serial in '{field}' — "
                    "the per-global-column draw is decomposition-variant")
    np.testing.assert_allclose(
        np.asarray(ps_b.conv_stoch_state), np.asarray(ps_s.conv_stoch_state),
        rtol=1e-6, atol=1e-12,
        err_msg="AR1 stochastic carry diverged under band decomposition")


def _stateful_hs_physics():
    """A DETERMINISTIC, COLUMN-LOCAL stateful physics: Held-Suarez tendencies
    scaled by a column tke factor, with tke relaxed toward the column
    temperature — the carry both INFLUENCES the trajectory and EVOLVES, so
    equivalence is non-vacuous in both directions.  Shapes derive from the
    state itself (band or global), never from the grid statics."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon

    def _phys(hs, grid, sigma, ps):
        tend = held_suarez_forcing_latlon(hs, grid, sigma)
        nlat_loc, nlon, nlev = hs.T.data.shape
        T_col = hs.T.data.reshape(nlat_loc * nlon, nlev)
        tke_new = 0.9 * ps.tke + 0.1 * (T_col / 300.0)
        factor = (1.0 + 1e-2 * jnp.mean(ps.tke, axis=-1)).reshape(
            nlat_loc, nlon, 1)
        tend = tend._replace(
            dT_dt=tend.dT_dt.replace(data=tend.dT_dt.data * factor))
        return tend, ps._replace(tke=tke_new)

    _phys._requires_phys_state = True
    return _phys


def test_run_atm_latlon_spmd_segment_stateful_carry_matches_serial():
    """Increment-1 gate: a DETERMINISTIC PhysicsState carry threads through
    the SPMD segment — every (ncol, ...) leaf band-splits on dim 0 (lat-major
    flatten) — and both the trajectory AND the carry-out match the
    single-device twin through the same wrapper."""
    mesh = _mesh()
    serial, c_state = _model_and_state(use_polar_filter=False)
    spmd_model, _ = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(serial, c_state)
    dt, n_steps = 100.0, 3
    phys = _stateful_hs_physics()
    nlat, nlon, nlev = np.asarray(hs0.T.data).shape
    ps0 = _mk_phys_state(nlat * nlon, nlev)

    hs_s, ps_s = run_atm_latlon_spmd_segment(
        serial, None, hs0, dt, n_steps, physics_fn=phys, phys_state=ps0)
    hs_b, ps_b = run_atm_latlon_spmd_segment(
        spmd_model, mesh, hs0, dt, n_steps, physics_fn=phys, phys_state=ps0)

    # Non-vacuity 1: the carry actually changed (tke evolved from the seed).
    assert float(np.max(np.abs(
        np.asarray(ps_b.tke) - np.asarray(ps0.tke)))) > 1e-6
    # Non-vacuity 2: the carry influenced the trajectory (vs stateless HS).
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
    hs_nostate = run_atm_latlon_spmd_segment(
        spmd_model, mesh, hs0, dt, n_steps,
        physics_fn=held_suarez_forcing_latlon)
    assert float(np.max(np.abs(
        np.asarray(hs_b.T.data) - np.asarray(hs_nostate.T.data)))) > 1e-9

    for field in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(hs_b, field).data),
            np.asarray(getattr(hs_s, field).data),
            rtol=1e-6, atol=1e-9,
            err_msg=f"stateful-carry SPMD diverged from serial in '{field}'")
    np.testing.assert_allclose(
        np.asarray(ps_b.tke), np.asarray(ps_s.tke), rtol=1e-6, atol=1e-12,
        err_msg="PhysicsState.tke carry-out diverged under band decomposition")


def test_run_atm_latlon_spmd_segment_rejects_tagged_stateful_physics_fn():
    """Dispatch-hardening (codex): a STATEFUL (``_requires_phys_state``-tagged)
    physics_fn with NO carry must raise — the SPMD body calls _step_cgrid(_impl)
    directly, bypassing model.step()'s refuse_unthreaded_stateful_physics guard,
    so without this it would silently reseed the carry every step (#405/#413).
    Covers the mesh=None path; make_sharded_atm_latlon_step guards both."""
    model, c_state = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(model, c_state)

    def _fake_stateful_physics(hs, grid, sigma):  # never actually called
        raise AssertionError("guard should fire before evaluation")
    _fake_stateful_physics._requires_phys_state = True

    with pytest.raises(NotImplementedError, match="stateful physics_fn"):
        run_atm_latlon_spmd_segment(
            model, None, hs0, 100.0, 1, physics_fn=_fake_stateful_physics)


def test_run_atm_latlon_spmd_segment_physics_matches_serial():
    """physics_fn gate: Held-Suarez (stateless, column-local) threads through the
    SPMD segment and matches the serial model.step(hs, physics_fn=...) loop at
    the Stage-5 integrated bound. Non-vacuity: physics must actually change the
    trajectory vs dynamics-only."""
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
    mesh = _mesh()
    serial, c_state = _model_and_state(use_polar_filter=False)
    spmd_model, _ = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(serial, c_state)
    dt, n_steps = 100.0, 3
    phys = held_suarez_forcing_latlon

    hs_s = hs0
    for _ in range(n_steps):
        hs_s = serial.step(hs_s, dt, physics_fn=phys)

    hs_b = run_atm_latlon_spmd_segment(
        spmd_model, mesh, hs0, dt, n_steps, physics_fn=phys)

    # Non-vacuity: the physics genuinely altered the trajectory (vs dynamics-only)
    # — guards against physics_fn being silently dropped under shard_map.
    hs_dyn = run_atm_latlon_spmd_segment(spmd_model, mesh, hs0, dt, n_steps)
    phys_effect = float(np.max(np.abs(
        np.asarray(hs_b.T.data) - np.asarray(hs_dyn.T.data))))
    assert phys_effect > 1e-6, (
        f"Held-Suarez physics did not change the SPMD trajectory "
        f"({phys_effect:.2e}) — physics_fn was not applied under shard_map.")

    for field in ("u", "v", "T", "p_s"):
        a = np.asarray(getattr(hs_b, field).data)
        b = np.asarray(getattr(hs_s, field).data)
        np.testing.assert_allclose(
            a, b, rtol=1e-6, atol=1e-9,
            err_msg=(
                f"SPMD Held-Suarez segment diverged from the serial "
                f"model.step(physics_fn) loop in '{field}' beyond the FV-PPM "
                f"cut-truncation bound — band physics coupling bug."))


# ==============================================================================
# run_atm_latlon_spmd — the production multi-segment run driver. Stays in the
# sharded C-grid layout for the whole run (gather is an output-only side copy),
# so it is segmentation-invariant and matches the serial model.step loop.
# ==============================================================================

def test_run_atm_latlon_spmd_rejects_bad_segment_steps():
    """Dispatch-hardening: segment_steps < 1 must raise, never silently no-op."""
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        run_atm_latlon_spmd)
    model, c_state = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(model, c_state)
    with pytest.raises(ValueError, match="segment_steps"):
        run_atm_latlon_spmd(model, None, hs0, 100.0, 4, segment_steps=0)


def test_run_atm_latlon_spmd_blowup_detection():
    """A NaN-injecting (stateless) physics_fn must end the run as 'BLOWUP at
    step N', not integrate garbage. mesh=None keeps it cheap."""
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        run_atm_latlon_spmd)
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_forcing_latlon
    model, c_state = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(model, c_state)

    def _nan_physics(hs, grid, sigma):
        phys = held_suarez_forcing_latlon(hs, grid, sigma)
        nan = jnp.full_like(phys.dT_dt.data, jnp.nan)
        return phys._replace(dT_dt=phys.dT_dt.replace(data=nan))

    hs_out, status = run_atm_latlon_spmd(
        model, None, hs0, 100.0, 3, segment_steps=1, physics_fn=_nan_physics)
    assert status.startswith("BLOWUP at step"), f"expected blowup, got {status!r}"


def test_run_atm_latlon_spmd_segmentation_invariant_and_matches_serial():
    """The production runner stays C-grid across the whole run, so segment_steps
    is OUTPUT cadence only: a 2-step-segment run is BIT-IDENTICAL to a 4-step
    single segment, and both match the serial model.step loop at the Stage-5
    bound. The on_segment callback fires once per segment with the GLOBAL state."""
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        run_atm_latlon_spmd)
    mesh = _mesh()
    serial, c_state = _model_and_state(use_polar_filter=False)
    spmd_model, _ = _model_and_state(use_polar_filter=False)
    hs0 = _hs_from_cgrid_state(serial, c_state)
    dt, n_steps = 100.0, 4

    seg_calls = []
    hs_seg2, st2 = run_atm_latlon_spmd(
        spmd_model, mesh, hs0, dt, n_steps, segment_steps=2,
        on_segment=lambda hs, k: seg_calls.append((k, np.asarray(hs.T.data).shape)))
    hs_seg4, st4 = run_atm_latlon_spmd(
        spmd_model, mesh, hs0, dt, n_steps, segment_steps=4)
    assert st2 == "COMPLETED" and st4 == "COMPLETED"

    # Callback fired once per 2-step segment (steps 2 and 4) with GLOBAL shape.
    assert [k for k, _ in seg_calls] == [2, 4], f"callback steps {seg_calls}"
    assert all(shape == (N_LAT, N_LON, NLEV) for _, shape in seg_calls)

    # Segmentation-invariant: gather is output-only, not fed back -> bit-identical.
    for field in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(hs_seg2, field).data),
            np.asarray(getattr(hs_seg4, field).data),
            rtol=1e-12, atol=1e-13,
            err_msg=f"segment_steps changed the '{field}' trajectory — the "
                    f"output gather is being fed back into the dynamics.")

    # Matches the serial model.step loop at the Stage-5 integrated bound.
    hs_s = hs0
    for _ in range(n_steps):
        hs_s = serial.step(hs_s, dt)
    for field in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(hs_seg4, field).data),
            np.asarray(getattr(hs_s, field).data),
            rtol=1e-6, atol=1e-9,
            err_msg=f"production runner diverged from serial in '{field}'.")
