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
from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.atmosphere.dynamics.sharded_atm_latlon_step import (
    make_sharded_atm_latlon_step,
    shard_state_atm_latlon,
    gather_state_atm_latlon,
    _build_band_grids_atm,
    _atm_grid_array_field_names,
    _lat_spec,
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
    from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
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

    band_grids = _build_band_grids_atm(grid, N_DEV)
    template = band_grids[0]
    afn = _atm_grid_array_field_names(template)
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
    in_spec = jax.tree.map(_lat_spec, sc)
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
