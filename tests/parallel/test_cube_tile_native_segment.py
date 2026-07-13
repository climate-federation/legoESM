"""M3b increment-1 gates: the PERSISTENT tile-native cube SEGMENT.

``scan_tiled_cc_steps`` / ``make_tiled_cc_segment``: ``n_steps`` of the
blocked tiled step in ONE compiled executable (``jit(lax.scan(step))``) —
state ENTERS tile layout once, scans with only in-stage tile halos (+ the
mass-fixer psum), EXITS once.  Gates:

* scanned segment == per-step jit loop (the no-new-numerics-from-the-scan
  property; same ``step`` object both sides),
* scanned segment == SERIAL ``model.step`` trajectory (the shipped
  adapter-gate tolerance class),
* scanned segment == the EXISTING 6-device face-SPMD production step
  (cross-lane; ``kt=1`` tiled meshes are refused BY DESIGN —
  ``cubesphere_exchange._build_tiled_tables``: kt=1 IS the face exchange —
  so at 6 devices the production path is the face-SPMD lane and the
  comparison runs across lanes),
* compiled-HLO receipts: ZERO ``all-gather`` anywhere in the segment
  executable; ``collective-permute`` (tile halos) present; collective
  counts of the 5-step segment EQUAL the 4-step segment's (the loop body
  is SHARED; the fixed extra copies are the ONE dtype-unrolled leading
  step + XLA's boundary peel — 2 total copies @n=2 vs 3 @n=5, job
  8970742 — so 4-vs-5 compares past them, where a real per-step
  unroll/regather would still scale counts),
* per-device memory ~constant in ``n_steps`` (no leak of full-face temps
  out of the scan),
* the M2b dtype fixed-point unroll genuinely FIRES on the production
  carry (f32 compute; the fixer's f64 accumulator promotes ``p_s`` on the
  first application and the promotion may cascade to ``T`` on the second
  — a 1-2 step unroll per fresh segment), plus a direct unit for the
  promoted public helper.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.halo import set_halo_backend
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
)
from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import (
    make_tiled_cc_loop, make_tiled_cc_segment, scan_tiled_cc_steps,
)
from legoesm.atmosphere.held_suarez import held_suarez_init

# Shared tiny-cube constants + helpers (single source — the shipped blocked
# loop gate; N/NLEV/KT identical so the mesh/skip logic composes).
from tests.parallel.test_tiled_blocked_loop import (  # noqa: E402
    N, NLEV, KT, NL, DT, _fv3_state, _mesh,
)

N_STEPS = 5


def _production_model():
    grid = create_cubed_sphere(N)
    coord = create_sigma_coordinate(NLEV)
    cfg = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=True, fix_mass=True,
        anchor_mass_to_initial=False, zero_mean_ps_tendency=True,
    )
    return CDGridPrimitiveEquationModel(grid, coord, cfg)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b))) / (float(np.max(np.abs(b))) + 1e-300)


def _blocked_rel(a, b):
    """Worst per-leaf rel diff of two blocked pytrees (same layout)."""
    worst = {}
    for k in b:
        worst[k] = _rel(a[k], b[k])
    return worst


#: Scan-vs-per-step parity bounds — identical numerics; the only deltas are
#: XLA compilation-order roundoff between the standalone step executable and
#: the scan's (unrolled + boundary-peeled + loop) body copies, and that
#: roundoff is quantized by the COMPUTE dtype.  The production cube carry is
#: FLOAT32 (``enter`` mirrors the serial ``_step_fv3`` compute cast — the
#: f32-honest bound rationale of the shipped adapter gate,
#: test_tiled_cc_step_adapter) except ``p_s``, which the in-stage
#: ``fix_ps_mass`` f64 accumulator promotes.  Measured (jobs
#: 8970742/8970908, two node ISAs): f32 leaves bit-identical on one node,
#: f32-ulp class (~1e-7 rel) on another; ``p_s`` (f64) 4.5e-12 @5 steps.
#: Bounds = ulp class x margin; a real halo/layout/tile bug is O(1e-3+)
#: rel — 2+ orders above.
_SCAN_PARITY_RTOL = {"p_s": 1e-9}
_SCAN_PARITY_RTOL_DEFAULT = 1e-5


def _assert_scan_parity(got, ref, label):
    for k, r in _blocked_rel(got, ref).items():
        tol = _SCAN_PARITY_RTOL.get(k, _SCAN_PARITY_RTOL_DEFAULT)
        assert r < tol, f"{k} rel {r:.3e} >= {tol:g} ({label})"


def _worst_corner_abs(blocked_field, global_field):
    """Worst per-tile ABS diff of a BLOCKED corner field vs global corners
    (the blocked layout duplicates shared tile faces — compare per tile)."""
    t = np.asarray(blocked_field)
    g = np.asarray(global_field)
    b = NL + 1
    worst = 0.0
    for f in range(6):
        for ti in range(KT):
            for tj in range(KT):
                gg = g[f, ti * NL: ti * NL + b, tj * NL: tj * NL + b]
                tt = t[f, ti * b:(ti + 1) * b, tj * b:(tj + 1) * b]
                worst = max(worst, float(np.max(np.abs(tt - gg))))
    return worst


# ---------------------------------------------------------------------------
# Numerics parity
# ---------------------------------------------------------------------------

def test_segment_matches_per_step_loop():
    """Scanned segment == per-step jit loop of the SAME step object.

    THE no-new-numerics gate for the scan wrapper: the scan body is the
    identical blocked step, so any delta is XLA compilation-order roundoff
    (``_SCAN_PARITY_RTOL`` — the p_s fixer-psum class)."""
    mesh = _mesh()
    model = _production_model()
    hs = held_suarez_init(model.grid, model.sigma_coord)

    enter, step, _exit = make_tiled_cc_loop(model, mesh, kt=KT, dt=DT)
    blk0 = enter(hs)

    step_jit = jax.jit(step)
    ref = blk0
    for _ in range(N_STEPS):
        ref = step_jit(ref)

    seg = scan_tiled_cc_steps(step, N_STEPS, donate=False)
    got = seg(blk0)

    _assert_scan_parity(got, ref, "scan vs per-step")


def test_segment_matches_serial_trajectory():
    """make_tiled_cc_segment(N_STEPS) vs the serial production ``model.step``
    trajectory from the SAME entry conversion — the shipped adapter-gate
    ABS tolerance class (pre-existing O(1e-6) face-corner wind term;
    measured @3 steps u 1.8e-6 / T 2.9e-8 / p_s 1.2e-5 abs — see
    test_tiled_blocked_loop.test_adapter_loop_matches_serial_from_same_entry).
    The bounds deliberately stay in that ESTABLISHED family rather than
    hugging the measured values (codex M3b review asked for tighter T/p_s;
    tightening is a cross-node f32-ISA calibration exercise tracked as a
    follow-up — the scan-specific regression surface is pinned far tighter
    by the scan-vs-per-step gates above)."""
    from legoesm.core.operators_cdgrid import center_to_dgrid_vector

    mesh = _mesh()
    model = _production_model()
    hs = held_suarez_init(model.grid, model.sigma_coord)

    u_d0, v_d0 = center_to_dgrid_vector(hs.u.data, hs.v.data, model.cdgrid)
    ref = _fv3_state(u_d0, v_d0, hs.T.data, hs.p_s.data, hs.phis.data)
    for _ in range(N_STEPS):
        ref = model.step(ref, DT)

    enter, segment, exit_ = make_tiled_cc_segment(
        model, mesh, kt=KT, dt=DT, n_steps=N_STEPS)
    blk = segment(enter(hs))

    def _abs(t, g):
        return float(np.max(np.abs(np.asarray(t) - np.asarray(g))))

    d_u = _worst_corner_abs(blk["u_d"], ref.u_d.data)
    d_v = _worst_corner_abs(blk["v_d"], ref.v_d.data)   # codex: BOTH winds
    d_T = _abs(blk["T"], ref.T.data)
    d_ps = _abs(blk["p_s"], ref.p_s.data)
    assert d_u < 2e-5, f"u_d abs {d_u:.3e}"
    assert d_v < 2e-5, f"v_d abs {d_v:.3e}"
    assert d_T < 1e-4, f"T abs {d_T:.3e}"
    assert d_ps < 0.06, f"p_s abs {d_ps:.3e}"

    # exit_ reassembles a cc HydrostaticState (segment-boundary I/O path).
    out = exit_(blk, hs)
    assert out.u.data.shape == hs.u.data.shape
    assert bool(jnp.all(jnp.isfinite(out.u.data)))


def test_segment_matches_face_spmd_step():
    """Cross-LANE gate: the tiled segment (6,2,2 over 24 devices) == the
    EXISTING face-SPMD production step (6-device face mesh +
    activate_spmd_halo_backend — the <=6-device production lane).

    The task-level '(6,1,1)' tiled mesh does not exist BY DESIGN: kt=1 is
    refused (``cubesphere_exchange._build_tiled_tables`` — at one tile per
    face the tiled exchange IS the face exchange), and the driver
    dispatches 6-device runs to the face-SPMD lane.  Both trajectories
    start from the SAME entry conversion; bounds are the adapter-gate ABS
    class (the face-SPMD lane is bit-tight to serial — its own gate,
    test_cubed_sphere_spmd_step — so the tiled-vs-serial term dominates)."""
    from legoesm.core.operators_cdgrid import center_to_dgrid_vector
    from legoesm.parallel.mesh import create_device_mesh, shard_pytree
    from legoesm.parallel.cubesphere_exchange import (
        activate_spmd_halo_backend, deactivate_spmd_halo_backend,
    )

    mesh = _mesh()                      # 24-dev tiled mesh (skip guard)
    model = _production_model()
    hs = held_suarez_init(model.grid, model.sigma_coord)
    u_d0, v_d0 = center_to_dgrid_vector(hs.u.data, hs.v.data, model.cdgrid)
    fv3_0 = _fv3_state(u_d0, v_d0, hs.T.data, hs.p_s.data, hs.phis.data)

    # Face-SPMD reference: the production <=6-device lane (global SPMD halo
    # backend + face-sharded state through the serial model.step).
    dc6 = create_device_mesh(n_devices=6, devices=jax.devices()[:6])
    activate_spmd_halo_backend(dc6.mesh, n=N, nlev=NLEV)
    try:
        ref = shard_pytree(fv3_0, dc6)
        for _ in range(N_STEPS):
            ref = model.step(ref, DT)
        ref = jax.device_get(ref)
    finally:
        deactivate_spmd_halo_backend()   # restores the local backend

    set_halo_backend("local")
    enter, segment, _exit = make_tiled_cc_segment(
        model, mesh, kt=KT, dt=DT, n_steps=N_STEPS)
    blk = segment(enter(hs))

    d_u = _worst_corner_abs(blk["u_d"], ref.u_d.data)
    d_v = _worst_corner_abs(blk["v_d"], ref.v_d.data)   # codex: BOTH winds
    d_T = float(np.max(np.abs(np.asarray(blk["T"]) - np.asarray(ref.T.data))))
    d_ps = float(np.max(np.abs(np.asarray(blk["p_s"])
                               - np.asarray(ref.p_s.data))))
    assert d_u < 2e-5, f"u_d abs {d_u:.3e} (tiled segment vs face-SPMD)"
    assert d_v < 2e-5, f"v_d abs {d_v:.3e} (tiled segment vs face-SPMD)"
    assert d_T < 1e-4, f"T abs {d_T:.3e} (tiled segment vs face-SPMD)"
    assert d_ps < 0.06, f"p_s abs {d_ps:.3e} (tiled segment vs face-SPMD)"


def test_moist_segment_matches_per_step_loop():
    """The Kessler moist carry (q_pack) rides the SAME scan — segment ==
    per-step loop with the production Kessler bridge injected."""
    from legoesm.core.field import Field
    from legoesm.atmosphere.kessler_forcing import (
        make_kessler_column_physics_fn,
    )
    from legoesm.atmosphere.dynamics.gcm.tiled_step_adapter import _TILED_TRACERS

    mesh = _mesh()
    model = _production_model()
    hs = held_suarez_init(model.grid, model.sigma_coord)
    rng = np.random.default_rng(31)
    d3 = ("face", "x", "y", "level")
    tracers = {
        nm: Field(data=jnp.asarray(np.abs(
                s + 1e-4 * rng.standard_normal((6, N, N, NLEV)))),
                  name=nm, dims=d3, units="kg/kg")
        for nm, s in zip(_TILED_TRACERS, (5e-3, 5e-4, 5e-5))
    }
    hs = hs._replace(tracers=tracers)

    col_fn = make_kessler_column_physics_fn(model.sigma_coord, DT)
    enter, step, _exit = make_tiled_cc_loop(
        model, mesh, kt=KT, dt=DT, column_physics_fn=col_fn)
    blk0 = enter(hs)

    step_jit = jax.jit(step)
    ref = blk0
    for _ in range(3):
        ref = step_jit(ref)

    got = scan_tiled_cc_steps(step, 3, donate=False)(blk0)
    _assert_scan_parity(got, ref, "moist scan vs per-step")
    assert float(jnp.min(got["q_pack"])) >= 0.0   # floor rode the scan


# ---------------------------------------------------------------------------
# Compiled receipts: HLO collective counts + memory
# ---------------------------------------------------------------------------

#: compiled-segment cache shared by the receipt tests (one lower+compile per
#: distinct length; the setup is rebuilt per test but the executables are the
#: expensive part).  Keyed by n_steps — valid because every receipt test uses
#: the SAME dry production setup below.
_COMPILED: dict[int, object] = {}


def _dry_compiled(n_steps):
    c = _COMPILED.get(n_steps)
    if c is None:
        mesh = _mesh()
        model = _production_model()
        hs = held_suarez_init(model.grid, model.sigma_coord)
        enter, step, _exit = make_tiled_cc_loop(model, mesh, kt=KT, dt=DT)
        blk = enter(hs)
        seg = scan_tiled_cc_steps(step, n_steps, donate=False)
        c = _COMPILED[n_steps] = seg.lower(blk).compile()
    return c


def test_segment_hlo_no_allgather_collectives_once():
    """HLO receipts for the WHOLE segment executable:

    * ZERO ``all-gather`` (no full-face re-gather anywhere — the M3b
      property, asserted on the compiled program, not inferred),
    * ``collective-permute`` present (the in-stage tile halos really run),
    * ``all-reduce`` present (the in-stage telescoping mass fixer's psum —
      the production config's ONE algorithmic global reduction),
    * collective COUNTS of the 5-step segment == the 4-step segment's:
      the scan body is SHARED between iterations, so the counts are
      n-independent past the FIXED extra body copies — the ONE
      dtype-unrolled leading step (the f32 carry's p_s promotion,
      test_production_carry_dtype_unroll_fires) plus XLA's boundary
      iteration peeling.  (Measured, job 8970742: 2 total copies @n=2 vs
      3 @n=5; both fixed in n, so 4-vs-5 is the invariant comparison — a
      real per-step unroll or per-step re-layout would still scale
      counts with n and trip this.)

    KNOWN LIMIT (codex M3b MINOR): textual counts cannot see a
    DEVICE-LOCAL copy that sits once in the while body yet executes per
    iteration (e.g. re-slicing the face-replicated metrics).  Cross-
    device movement IS fully covered — any per-iteration reshard would
    emit collectives in the body text (all-gather == 0 module-wide, and
    permute/reduce counts are pinned); local per-iteration copies are a
    perf-tuning follow-up, not a layout-correctness hazard.
    """
    hlo4 = _dry_compiled(N_STEPS - 1).as_text()
    hlo5 = _dry_compiled(N_STEPS).as_text()

    for tag, hlo in (("4-step", hlo4), ("5-step", hlo5)):
        assert hlo.count("all-gather") == 0, (
            f"full-face all-gather in the {tag} segment")
    assert "collective-permute" in hlo5, "tile halos missing from segment"
    assert "all-reduce" in hlo5, "in-stage mass-fixer psum missing"
    assert " while(" in hlo5, "5-step segment did not compile as a scan loop"
    for op in ("collective-permute", "all-reduce", "all-gather",
               "all-to-all"):
        c4, c5 = hlo4.count(op), hlo5.count(op)
        assert c5 == c4, (
            f"{op}: {c5} in 5-step segment vs {c4} in 4-step segment — "
            f"the scan body must be shared (counts n-independent past "
            f"the fixed boundary peel)")


def test_segment_memory_constant_in_n_steps():
    """Per-device memory of the segment executable is ~constant in
    ``n_steps`` (the scan carries ONE blocked state; a leak of full-face
    temporaries out of the scan body would scale with the step count).

    n=4 and n=5 compile to the SAME structure (fixed unroll + peel + a
    while loop — the HLO gate above), so their temp allocations should be
    near-identical; the tolerance is ONE blocked-carry's bytes (codex:
    a fixed 1 MiB dwarfed the N=8 state), i.e. leaking even one carry-
    sized buffer per additional step trips this."""
    try:
        ma4 = _dry_compiled(N_STEPS - 1).memory_analysis()
        ma5 = _dry_compiled(N_STEPS).memory_analysis()
        t4 = int(ma4.temp_size_in_bytes)
        t5 = int(ma5.temp_size_in_bytes)
        o4 = int(ma4.output_size_in_bytes)
        o5 = int(ma5.output_size_in_bytes)
    except (AttributeError, NotImplementedError, TypeError) as e:  # pragma: no cover
        pytest.skip(f"memory_analysis unavailable on this backend: {e}")

    assert o5 == o4, f"output size changed with n_steps: {o4} -> {o5}"
    # One blocked carry (GLOBAL bytes — an overestimate of any per-device
    # leak, conservative in the right direction for the bound below).
    mesh = _mesh()
    model = _production_model()
    hs = held_suarez_init(model.grid, model.sigma_coord)
    enter, _step, _exit = make_tiled_cc_loop(model, mesh, kt=KT, dt=DT)
    carry_bytes = sum(int(v.nbytes) for v in enter(hs).values())
    assert t5 - t4 < carry_bytes, (
        f"segment temp memory scales with n_steps: {t4} -> {t5} bytes "
        f"(> one blocked carry = {carry_bytes} bytes)")


# ---------------------------------------------------------------------------
# Dtype fixed-point unroll (the M2b helper, promoted public)
# ---------------------------------------------------------------------------

def test_unroll_to_dtype_fixed_point_public():
    """Direct unit for the PROMOTED helper: a mixed-dtype carry unrolls
    exactly until the step's output dtypes are a fixed point; a stable
    carry unrolls nothing."""
    from legoesm.atmosphere.dynamics.gcm.sharded_atm_latlon_step import (
        unroll_to_dtype_fixed_point,
    )

    def step1(s):
        # First application promotes 'a' f32->f64 (mixing the f64 'b');
        # then stable.
        return {"a": s["a"] + 0.0 * s["b"], "b": s["b"]}

    mixed = {"a": jnp.zeros(3, jnp.float32), "b": jnp.zeros(3, jnp.float64)}
    out, n_left = unroll_to_dtype_fixed_point(step1, mixed, 5)
    assert n_left == 4
    assert out["a"].dtype == jnp.float64

    stable = {"a": jnp.zeros(3, jnp.float64), "b": jnp.zeros(3, jnp.float64)}
    out, n_left = unroll_to_dtype_fixed_point(step1, stable, 5)
    assert n_left == 5
    assert out is stable


def test_production_carry_dtype_unroll_fires():
    """The PRODUCTION enter carry is NOT a dtype fixed point of the step:
    the cube compute policy is f32 (``enter`` mirrors the serial
    ``_step_fv3`` compute cast) while the in-stage ``fix_ps_mass`` f64
    accumulator promotes ``p_s`` on the first application, and the
    promotion may CASCADE (the promoted ``p_s`` promotes ``T`` on the
    second — the exact multi-application class the M2b helper exists for,
    observed on the lat-lon bench IC jobs 8916406/8916740 and here on the
    cube: diag job 8970924 measured final-state T diffs in the f64 class).
    Pins that the unroll genuinely fires (a 1-2 application cascade, then
    fixed) on every production segment — the scan-carry dtype contract is
    live, not vacuous — and that the scanned segment still matches the
    per-step loop across the promotion boundary."""
    mesh = _mesh()
    model = _production_model()
    hs = held_suarez_init(model.grid, model.sigma_coord)
    enter, step, _exit = make_tiled_cc_loop(model, mesh, kt=KT, dt=DT)
    blk0 = enter(hs)

    def _sig(t):
        return {k: str(v.dtype) for k, v in t.items()}

    assert _sig(blk0)["p_s"] == "float32", (
        "test premise: the production enter carry is f32 compute; if the "
        "cube compute policy ever changes, update this gate")
    sigs = [_sig(blk0)]
    probe = blk0
    for _ in range(4):
        probe = jax.eval_shape(step, probe)
        sigs.append(_sig(probe))
        if sigs[-1] == sigs[-2]:
            break
    assert sigs[-1] == sigs[-2], (
        f"no dtype fixed point within 4 applications: {sigs}")
    k = len(sigs) - 2          # applications that still changed the sig
    assert 1 <= k <= 2, (
        f"unroll depth {k} outside the observed promotion cascade "
        f"(p_s at application 1, T possibly at 2): {sigs}")
    assert sigs[1]["p_s"] == "float64", (
        "the fixer's f64 accumulator must promote p_s on the first "
        f"application: {sigs}")

    step_jit = jax.jit(step)
    ref = blk0
    for _ in range(3):
        ref = step_jit(ref)

    got = scan_tiled_cc_steps(step, 3, donate=False)(blk0)
    _assert_scan_parity(got, ref, "production-carry scan vs per-step")
    assert got["p_s"].dtype == jnp.float64


def test_exit_state_survives_carry_donation():
    """The gathered ``exit_`` state must be DONATION-INDEPENDENT: the
    production lane donates the carry on every segment call, so an
    ``exit_`` output aliasing carry buffers (T/p_s are passthrough in
    ``fv3_to_hydrostatic``) would be invalidated one segment later —
    poisoning any retained ``driver.state`` (deferred/async writer
    callbacks; codex M3b MAJOR).  Red-green: without the explicit copies
    in ``exit_``, reading ``out.T`` below raises the deleted/donated-
    buffer error."""
    mesh = _mesh()
    model = _production_model()
    hs = held_suarez_init(model.grid, model.sigma_coord)
    enter, segment, exit_ = make_tiled_cc_segment(
        model, mesh, kt=KT, dt=DT, n_steps=2)   # donate=True default
    blk1 = segment(enter(hs))
    out = exit_(blk1, hs)                       # gathered mid-run state
    blk2 = segment(blk1)                        # donates blk1
    jax.block_until_ready(blk2["T"])
    if not blk1["T"].is_deleted():
        pytest.skip(
            "buffer donation is a no-op on this backend — the aliasing "
            "hazard cannot be exercised here (the gate is live where "
            "donation works, e.g. GPU)")
    # The retained gathered state must still be fully readable even though
    # the carry it was gathered from is now donated.
    assert bool(np.all(np.isfinite(np.asarray(out.T.data))))
    assert bool(np.all(np.isfinite(np.asarray(out.p_s.data))))
    assert bool(np.all(np.isfinite(np.asarray(out.u.data))))


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_scan_rejects_bad_n_steps():
    with pytest.raises(ValueError, match="n_steps must be >= 1"):
        scan_tiled_cc_steps(lambda c: c, 0)
