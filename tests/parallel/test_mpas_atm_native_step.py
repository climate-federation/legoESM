"""Production-parity gates for the NATIVE (ppermute SPMD, non-mpi4jax)
MPAS-atmosphere step — scaling-M3c increment 1.

``make_voronoi_sharded_step`` historically covered only the dry dynamics
state (u, T, p_s, phis) with stateless closure physics.  Increment 1
extends it to full production support: tracers through the packed
ppermute halo exchange + RK advection, traced per-step ``forcing``,
the prognostic ``phys_state`` carry (``return_phys_state=True``),
``config.time_integrator`` dispatch, tracer floors, compute/storage
precision casts, and LOCAL-ONLY metadata (P("device")-sharded local
meshes + halo schedules threaded as jit arguments).

Gates here:

* full-production parity vs the serial ``MPASPrimitiveEquationModel``
  step (moist Kessler tracers / traced forcing / stateful carry /
  non-default integrator), 2 virtual devices vs single device;
* packed-exchange five-field sentinel ROUTING test (the M3d
  ``exchange_state_mpas_ocean`` pattern) — an omitted field or a swapped
  pack slot cannot pass;
* 10-step ``jax.lax.scan`` stability with dtype fixed point;
* the schema tripwire + carry refusal contracts (loud, not silent);
* route-A (``make_voronoi_mpi_step``) np=1 cross-check when mpi4jax is
  available.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=2`` (or
more).  Multi-process (np=2) coverage of the native path lives in
``test_mpas_spmd_multicontroller_selfspawn.py``; the route-A exchange
layer's np=2 srun gate is ``tests/distributed/test_voronoi_mpi.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Parity tolerances vs the serial reference: the FLOATING-POINT
# RE-ASSOCIATION floor of the sharded step (halo-cell operator coverage +
# mass-fix allreduce ordering), NOT a bug margin — same envelope as
# tests/parallel/test_voronoi_sharded_equivalence.py.  Measured (x64,
# subdiv 3, 2 steps): u/T ~1e-9..1e-8, p_s ~1e-4, q ~1e-12.
_TOLS = {
    "u": dict(atol=1e-6, rtol=1e-6),
    "T": dict(atol=1e-6, rtol=1e-7),
    "p_s": dict(atol=1e-1, rtol=1e-6),
}
_TOL_Q = dict(atol=1e-9, rtol=1e-6)

_SUBDIV = 3
_NLEV = 4
_DT = 200.0  # CFL-stable at subdivision 3 (test_voronoi_sharded_equivalence)


def _need_multi_device(n: int):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices "
            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
        )


def _build(*, moist: bool, reorder_for: int = 2,
           time_integrator: str = "ssp_rk3"):
    """Reordered mesh + config + model + (moist) BCW initial state.

    The mesh is reordered for ``reorder_for``-way sharding on BOTH the
    serial and sharded paths so cell/edge indices match for direct
    array comparison (the test_voronoi_sharded_equivalence pattern).
    """
    from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )

    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    mesh = create_voronoi_mesh(subdivision_level=_SUBDIV)
    mesh = reorder_voronoi_for_sharding(mesh, reorder_for)
    sigma = create_sigma_coordinate(_NLEV)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16,
        fix_mass=True, pv_scheme="energy",
        time_integrator=time_integrator,
    )
    model = MPASPrimitiveEquationModel(mesh, sigma, cfg)
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True,
                                      moist=moist)
    return mesh, sigma, cfg, model, state


def _make_sharded_step(mesh, model, n_dev: int, **kw):
    from legoesm.parallel.mesh import (
        create_voronoi_device_mesh,
        replicate_pytree,
    )
    from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step

    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges,
        nVertices=mesh.nVertices, n_devices=n_dev,
    )
    # The model handed to the sharder only supplies mesh/sigma/config;
    # replicate its mesh like the equivalence test / levante bench do.
    model_rep = type(model)(
        replicate_pytree(mesh, dev_config), model.sigma_coord, model.config)
    return make_voronoi_sharded_step(model_rep, dev_config, **kw), dev_config


# ---------------------------------------------------------------------------
# Toy physics (module scope: stable identities for the executable cache)
# ---------------------------------------------------------------------------

# Trace counter: the Python body runs at TRACE time only under jit, so
# its length counts (re)traces — the SegmentForcing no-retrace gate.
_FORCING_TRACE_MARKS: list = []


def _toy_forcing_relax_physics(state, mesh, sigma_coord, *,
                               phys_state=None, forcing=None):
    """Stateless physics consuming the TRACED per-step forcing.

    Relaxes the lowest-level temperature toward ``forcing["T_sfc"]``
    over 6 h — the operator-split analogue of a prescribed-SST AMIP
    forcing, with no dependence on mesh metadata.
    """
    from legoesm.core.state import MPASHydrostaticTendencies

    _FORCING_TRACE_MARKS.append(1)
    tau_s = 6.0 * 3600.0
    T_sfc = forcing["T_sfc"]  # (nCells,)
    dT = jnp.zeros_like(state.T.data)
    dT = dT.at[:, -1].set((T_sfc - state.T.data[:, -1]) / tau_s)
    return MPASHydrostaticTendencies(
        du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
        dT_dt=state.T.replace(data=dT),
        dp_s_dt=state.p_s.replace(data=jnp.zeros_like(state.p_s.data)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        tracer_tendencies=None,
    )


def _toy_stateful_physics(state, mesh, sigma_coord, *,
                          phys_state=None, forcing=None):
    """Stateful physics with a prognostic carry (issue #405/#413 shape).

    The carry accumulates the column-mean temperature; the tendency
    depends on the ACCUMULATED carry, so dropping/reseeding the carry
    changes the trajectory (non-vacuous threading test).
    Returns ``(tendencies, phys_state_out)``.
    """
    from legoesm.core.state import MPASHydrostaticTendencies

    accum = phys_state["accum"] + jnp.mean(state.T.data, axis=-1)
    dT = jnp.broadcast_to((1.0e-9 * accum)[:, None], state.T.data.shape)
    tend = MPASHydrostaticTendencies(
        du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
        dT_dt=state.T.replace(data=dT),
        dp_s_dt=state.p_s.replace(data=jnp.zeros_like(state.p_s.data)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        tracer_tendencies=None,
    )
    return tend, {"accum": accum}


_toy_stateful_physics._requires_phys_state = True


def _toy_one_tracer_physics(state, mesh, sigma_coord, *,
                            phys_state=None, forcing=None):
    """Stateless physics with a tendency for EXACTLY ONE tracer key.

    Pins the partial-key tracer application (``if k in
    tracer_tendencies``): only ``q_c`` may receive the physics
    increment; ``q_v``/``q_r`` must follow the dynamics-only
    trajectory.  Rate 1e-8 kg/kg/s over dt=200 s gives a 2e-6 signal,
    three orders above the tracer parity tolerance."""
    from legoesm.core.state import MPASHydrostaticTendencies

    q_c = state.tracers["q_c"]
    return MPASHydrostaticTendencies(
        du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
        dT_dt=state.T.replace(data=jnp.zeros_like(state.T.data)),
        dp_s_dt=state.p_s.replace(data=jnp.zeros_like(state.p_s.data)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        tracer_tendencies={
            "q_c": q_c.replace(data=jnp.full_like(q_c.data, 1.0e-8)),
        },
    )


_ONE_TRACER_RATE = 1.0e-8  # [kg/kg/s] must match _toy_one_tracer_physics


def _assert_state_close(got, want, *, moist: bool, label: str):
    for name, tol in _TOLS.items():
        np.testing.assert_allclose(
            np.asarray(getattr(got, name).data),
            np.asarray(getattr(want, name).data),
            err_msg=f"{label}: field {name} diverges from serial", **tol,
        )
    if moist:
        assert got.tracers is not None and want.tracers is not None
        assert sorted(got.tracers) == sorted(want.tracers)
        for k in want.tracers:
            np.testing.assert_allclose(
                np.asarray(got.tracers[k].data),
                np.asarray(want.tracers[k].data),
                err_msg=f"{label}: tracer {k} diverges from serial",
                **_TOL_Q,
            )


# ===========================================================================
# Full-production parity vs serial
# ===========================================================================

class TestNativeFullProductionParity:

    @pytest.mark.parametrize("halo_strategy", ["ppermute", "allgather"])
    def test_moist_kessler_matches_serial(self, halo_strategy):
        """Moist BCW + Kessler microphysics: tracers ride the packed
        exchange, the RK advection, the physics tendencies and the
        non-negativity floor — and match the serial step.  Both kernel
        strategies are forced (auto would pick allgather at this size,
        leaving the production ppermute rounds untested)."""
        _need_multi_device(2)
        from legoesm.atmosphere.kessler_forcing import (
            make_kessler_forcing_mpas,
        )

        mesh, sigma, cfg, model, state0 = _build(moist=True)
        kessler = make_kessler_forcing_mpas(_DT)

        ref = state0
        for _ in range(2):
            ref = model.step(ref, _DT, physics_fn=kessler)

        step, _dc = _make_sharded_step(
            mesh, model, 2, halo_strategy=halo_strategy)
        out = state0
        for _ in range(2):
            out = step(out, _DT, physics_fn=kessler)

        # Non-vacuity: the tracers must actually have been transported
        # (a step that silently dropped tracer advection would keep q_v
        # frozen at the initial field and could still "match" a broken
        # reference).
        # Threshold ABOVE the tracer comparison atol (1e-9): a frozen
        # q_v could otherwise hide inside the parity envelope.
        dq = np.max(np.abs(np.asarray(ref.tracers["q_v"].data)
                           - np.asarray(state0.tracers["q_v"].data)))
        assert dq > 1e-8, f"q_v unchanged after 2 steps (max dq={dq:.3e})"

        _assert_state_close(
            out, ref, moist=True, label=f"moist kessler [{halo_strategy}]")

    def test_physics_tracer_tendencies_partial_key(self):
        """Physics tracer tendencies are applied, and ONLY to the keys
        the scheme returns (the ``if k in tracer_tendencies`` guard):
        a one-tracer toy physics must move q_c by dt*rate while
        q_v/q_r follow the dynamics-only trajectory — on the native
        step AND in parity with serial (codex M3c-1 MINOR: the Kessler
        gate alone could pass with the application deleted if the
        scheme is inactive over a short window)."""
        _need_multi_device(2)
        mesh, sigma, cfg, model, state0 = _build(moist=True)

        ref_dyn = model.step(state0, _DT)  # no physics
        ref_phys = model.step(state0, _DT,
                              physics_fn=_toy_one_tracer_physics)
        step, _dc = _make_sharded_step(
            mesh, model, 2, halo_strategy="ppermute")
        out = step(state0, _DT, physics_fn=_toy_one_tracer_physics)

        _assert_state_close(out, ref_phys, moist=True,
                            label="one-tracer physics")
        # Exactly q_c received the increment (q_c starts at 0 and is
        # not produced by dynamics, so the signal is clean dt*rate).
        dqc = (np.asarray(out.tracers["q_c"].data)
               - np.asarray(ref_dyn.tracers["q_c"].data))
        np.testing.assert_allclose(
            dqc, _DT * _ONE_TRACER_RATE, rtol=1e-6,
            err_msg="q_c did not receive the physics tendency")
        for k in ("q_v", "q_r"):
            np.testing.assert_allclose(
                np.asarray(out.tracers[k].data),
                np.asarray(ref_dyn.tracers[k].data),
                atol=1e-9, rtol=1e-6,
                err_msg=(f"{k} moved: physics increment leaked to a "
                         f"key the scheme did not return"),
            )

    def test_fp32_state_dtype_fixed_point_under_x64(self):
        """codex M3c-1 MAJOR: the fp64 mass-fix accumulator must not
        promote an fp32 carry under x64 (the downcast-skipping storage
        cast cannot undo it; a promoted p_s breaks the lax.scan
        carry-dtype contract).  All-fp32 state AND mesh under the
        default fp32 policy (compute/storage casts are no-ops, sigma
        is policy-fp32 already): every float leaf out must stay
        float32 across two fix_mass steps."""
        if not jax.config.jax_enable_x64:
            pytest.skip("needs x64 so the fp64 accumulator is real")
        _need_multi_device(2)
        from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
            MPASPrimitiveEquationModel,
        )

        mesh, sigma, cfg, model, state0 = _build(moist=True)

        def _to_f32(tree):
            return jax.tree.map(
                lambda x: (x.astype(jnp.float32)
                           if (hasattr(x, "dtype")
                               and jnp.issubdtype(x.dtype, jnp.floating))
                           else x),
                tree,
            )

        mesh32 = _to_f32(mesh)
        state32 = _to_f32(state0)
        model32 = MPASPrimitiveEquationModel(mesh32, sigma, cfg)
        step, _dc = _make_sharded_step(
            mesh32, model32, 2, halo_strategy="ppermute")
        out = state32
        for _ in range(2):
            out = step(out, _DT)

        bad = [
            f"{path}: {leaf.dtype}"
            for path, leaf in jax.tree_util.tree_leaves_with_path(out)
            if jnp.issubdtype(leaf.dtype, jnp.floating)
            and leaf.dtype != jnp.float32
        ]
        assert not bad, (
            "fp32 carry promoted (scan carry-dtype contract broken): "
            + "; ".join(str(b) for b in bad)
        )
        assert bool(jnp.all(jnp.isfinite(out.p_s.data)))

    def test_traced_forcing_threaded_no_retrace(self):
        """Per-step forcing values thread through as TRACED args: the
        native step matches serial under the same forcing sequence,
        different forcing values change the answer (non-vacuity), and
        changing values do NOT retrace (SegmentForcing doctrine)."""
        _need_multi_device(2)
        mesh, sigma, cfg, model, state0 = _build(moist=False)
        lat = np.asarray(mesh.latCell)
        f_seq = [
            {"T_sfc": jnp.asarray(288.0 + 5.0 * np.sin(lat) + 2.0 * k)}
            for k in range(3)
        ]
        f_warm = [{"T_sfc": f["T_sfc"] + 20.0} for f in f_seq]

        ref = state0
        for f in f_seq:
            ref = model.step(
                ref, _DT, physics_fn=_toy_forcing_relax_physics, forcing=f)

        step, dc = _make_sharded_step(
            mesh, model, 2, halo_strategy="ppermute")

        # SegmentForcing no-retrace gate.  The first TWO calls each
        # compile once (call 1: unsharded initial-state layout; call 2:
        # the step's own sharded output layout — a one-time layout
        # promotion, observed identically with a pre-sharded input).
        # From then on, steps with CHANGING forcing VALUES must add
        # ZERO traces — forcing is a traced jit arg, never a closure
        # constant.
        out = step(state0, _DT, physics_fn=_toy_forcing_relax_physics,
                   forcing=f_seq[0])
        out = step(out, _DT, physics_fn=_toy_forcing_relax_physics,
                   forcing=f_seq[1])
        _FORCING_TRACE_MARKS.clear()
        out = step(out, _DT, physics_fn=_toy_forcing_relax_physics,
                   forcing=f_seq[2])
        # One more steady-state call with a FRESH forcing value (result
        # unused — parity below is over the 3-step f_seq trajectory).
        step(out, _DT, physics_fn=_toy_forcing_relax_physics,
             forcing={"T_sfc": f_seq[2]["T_sfc"] + 1.0})
        n_traces = len(_FORCING_TRACE_MARKS)

        # Counter non-vacuity: a changed forcing STRUCTURE (new key) is
        # a legitimate retrace, and the counter must see it.
        _FORCING_TRACE_MARKS.clear()
        step(out, _DT, physics_fn=_toy_forcing_relax_physics,
             forcing={**f_seq[0], "extra": jnp.zeros(())})
        n_structural = len(_FORCING_TRACE_MARKS)

        out_warm = state0
        for f in f_warm:
            out_warm = step(out_warm, _DT,
                            physics_fn=_toy_forcing_relax_physics, forcing=f)

        _assert_state_close(out, ref, moist=False, label="traced forcing")

        # Non-vacuity: a 20 K warmer forcing must move surface T beyond
        # the parity tolerance (relaxation over 3x200 s / 6 h * 20 K
        # ~ 0.55 K >> 1e-6).
        dT = np.max(np.abs(np.asarray(out_warm.T.data)
                           - np.asarray(out.T.data)))
        assert dT > 1e-3, f"forcing signal too small (max dT={dT:.3e} K)"

        # No-retrace: post-warmup steps with changing VALUES (stable
        # structure) must add zero traces.
        assert n_traces == 0, (
            f"native step retraced {n_traces}x on steady-state steps with "
            f"changing forcing values — forcing must be a traced arg, not "
            f"a closure constant"
        )
        assert n_structural >= 1, (
            "trace counter saw no retrace on a forcing STRUCTURE change — "
            "the no-retrace assertion above is vacuous"
        )

    def test_stateful_carry_threaded(self):
        """``return_phys_state=True`` threads the prognostic physics
        carry exactly like the serial step (which stashes it eagerly on
        ``model._phys_state``); dropping the carry changes the answer."""
        _need_multi_device(2)
        mesh, sigma, cfg, model, state0 = _build(moist=False)
        nCells = mesh.nCells
        carry0 = {"accum": jnp.zeros((nCells,))}

        # Serial reference: model.step threads phys_state and stashes
        # the OUT carry on the model (documented eager side-channel).
        ref, ref_carry = state0, carry0
        for _ in range(3):
            ref = model.step(ref, _DT, physics_fn=_toy_stateful_physics,
                             phys_state=ref_carry)
            ref_carry = model._phys_state

        step, _dc = _make_sharded_step(
            mesh, model, 2, return_phys_state=True,
            halo_strategy="ppermute")
        out, carry = state0, carry0
        for _ in range(3):
            out, carry = step(out, _DT, physics_fn=_toy_stateful_physics,
                              phys_state=carry)

        _assert_state_close(out, ref, moist=False, label="stateful carry")
        np.testing.assert_allclose(
            np.asarray(carry["accum"]), np.asarray(ref_carry["accum"]),
            atol=1e-6, rtol=1e-9,
            err_msg="phys_state carry diverges from serial",
        )

        # Non-vacuity: reseeding the carry every step (the #405 failure
        # mode) must change the trajectory beyond the parity tolerance.
        out_reseed = state0
        for _ in range(3):
            out_reseed, _ = step(
                out_reseed, _DT, physics_fn=_toy_stateful_physics,
                phys_state=carry0)
        dT = np.max(np.abs(np.asarray(out_reseed.T.data)
                           - np.asarray(out.T.data)))
        assert dT > 1e-5, (
            f"carry-dependence too weak to certify threading "
            f"(max dT={dT:.3e} K)"
        )

    def test_time_integrator_honored(self):
        """config.time_integrator now dispatches (the old path silently
        hard-coded SSP-RK3): rk4 matches serial rk4 and differs from
        the rk3 trajectory."""
        _need_multi_device(2)
        mesh, sigma, cfg4, model4, state0 = _build(
            moist=False, time_integrator="rk4")

        ref4 = model4.step(state0, _DT)
        step4, _dc = _make_sharded_step(mesh, model4, 2)
        out4 = step4(state0, _DT)
        _assert_state_close(out4, ref4, moist=False, label="rk4")

        _, _, _, model3, _ = _build(moist=False, time_integrator="ssp_rk3")
        ref3 = model3.step(state0, _DT)
        du = np.max(np.abs(np.asarray(ref4.u.data)
                           - np.asarray(ref3.u.data)))
        assert du > 1e-10, (
            "rk4 and ssp_rk3 references are identical — the integrator "
            "gate is vacuous"
        )


# ===========================================================================
# Packed-exchange sentinel routing (the M3d five-field pattern)
# ===========================================================================

class TestPackedExchangeSentinelRouting:

    def test_pack_unpack_roundtrip_dry_and_moist(self):
        """The production pack/unpack pair round-trips named fields,
        including the dry zero-width tracer block."""
        from legoesm.parallel.sharded_dynamics import (
            _pack_cell_state,
            _unpack_cell_state,
        )

        n, nlev = 7, 3
        rng = np.random.default_rng(0)
        T = jnp.asarray(rng.normal(size=(n, nlev)))
        ps = jnp.asarray(rng.normal(size=(n,)))
        phis = jnp.asarray(rng.normal(size=(n,)))
        for n_q in (0, 2):
            q = jnp.asarray(rng.normal(size=(n, nlev * n_q)))
            buf = _pack_cell_state(T, ps, phis, q)
            assert buf.shape == (n, nlev + 2 + nlev * n_q)
            T2, ps2, phis2, q2 = _unpack_cell_state(buf, nlev)
            np.testing.assert_array_equal(np.asarray(T2), np.asarray(T))
            np.testing.assert_array_equal(np.asarray(ps2), np.asarray(ps))
            np.testing.assert_array_equal(
                np.asarray(phis2), np.asarray(phis))
            np.testing.assert_array_equal(np.asarray(q2), np.asarray(q))

    @pytest.mark.parametrize("n_dev", [2, 3])
    def test_five_field_sentinel_routing(self, n_dev):
        """Five NAMED cell fields (T | p_s | phis | q_v | q_c) + the
        edge field carry PER-SLOT sentinels ``(slot+1)*1e4 +
        global_id`` through the PRODUCTION pack helper, the ppermute
        fill, and the production unpack helper: every halo cell/edge
        must receive its owner's value in every slot of every field.
        An omitted field (stale zero), a swapped pack slot (wrong
        base), or a tracer-order divergence (q built from a dict
        inserted in REVERSED order but packed in sorted wire order)
        fails loudly.  ``n_dev=3`` exercises a multi-round schedule
        (edge-colored comm graph); bases stay below 2^24 so the test
        is exact in float32 too."""
        _need_multi_device(n_dev)
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P
        from legoesm.grids.voronoi import create_voronoi_mesh
        from legoesm.parallel.shard_map_compat import shard_map
        from legoesm.parallel.sharded_dynamics import (
            _build_ppermute_schedule,
            _build_voronoi_partition_infra,
            _pack_cell_state,
            _ppermute_halo_fill,
            _unpack_cell_state,
        )
        from legoesm.parallel.voronoi_partition import (
            reorder_voronoi_for_sharding,
        )

        nlev = 3
        mesh = create_voronoi_mesh(subdivision_level=2)
        mesh = reorder_voronoi_for_sharding(mesh, n_dev)
        nCells, nEdges = mesh.nCells, mesh.nEdges
        cells_per, edges_per = nCells // n_dev, nEdges // n_dev

        (_sm, _gc, _ge, _noc, _noe, max_lc, max_le, partitions,
         cell_owner) = _build_voronoi_partition_infra(
            mesh, n_dev, halo_depth=3)
        sched = _build_ppermute_schedule(
            partitions, cell_owner, n_dev, cells_per, edges_per,
            max_lc, max_le)
        assert sched["n_rounds"] >= 1, "no comm rounds — test is vacuous"
        if n_dev >= 3:
            # 3 mutually-adjacent partitions edge-color to >= 2 rounds:
            # the multi-round scatter path is genuinely exercised.
            assert sched["n_rounds"] >= 2, (
                "expected a multi-round schedule at 3 devices")

        # Named per-field sentinels: value[g, lev] = base + lev*1e4 + g.
        def _cellf(base):
            g = np.arange(nCells, dtype=np.float64)
            return np.stack([base + lev * 1.0e4 + g
                             for lev in range(nlev)], axis=-1)

        fields = {
            "T": _cellf(1.0e5),
            "q_v": _cellf(3.0e5),   # dict INSERTED q_v before q_c ...
            "q_c": _cellf(4.0e5),
            "p_s": 2.0e5 + np.arange(nCells, dtype=np.float64),
            "phis": 2.5e5 + np.arange(nCells, dtype=np.float64),
        }
        tkeys = tuple(sorted(("q_v", "q_c")))  # ... wire order is sorted
        q_flat = np.concatenate([fields[k] for k in tkeys], axis=-1)
        gedge = np.arange(nEdges, dtype=np.float64)
        u_vals = np.stack(
            [6.0e5 + lev * 1.0e4 + 0.5 + gedge for lev in range(nlev)],
            axis=-1)

        # PRODUCTION wire layout (same helper the kernel uses).
        cell_vals = np.asarray(_pack_cell_state(
            jnp.asarray(fields["T"]), jnp.asarray(fields["p_s"]),
            jnp.asarray(fields["phis"]), jnp.asarray(q_flat)))

        devices = jax.devices("cpu")[:n_dev]
        jmesh = Mesh(np.array(devices), axis_names=("device",))
        dev_sharding = NamedSharding(jmesh, P("device"))
        halo_args = tuple(
            tuple(jax.device_put(a, dev_sharding) for a in (
                sched["send_cell_idx"][r], sched["recv_cell_pos"][r],
                sched["send_edge_idx"][r], sched["recv_edge_pos"][r]))
            for r in range(sched["n_rounds"])
        )
        halo_specs = jax.tree.map(lambda _: P("device"), halo_args)
        perms = sched["ppermute_perms"]

        def _kernel(cell_shard, u_shard, halo_sl):
            return _ppermute_halo_fill(
                cell_shard, u_shard, halo_sl, perms, max_lc, max_le)

        fill = shard_map(
            _kernel, mesh=jmesh,
            in_specs=(P("device"), P("device"), halo_specs),
            out_specs=(P("device"), P("device")),
            check_vma=False,
        )
        W = cell_vals.shape[-1]
        cell_out, u_out = fill(
            jnp.asarray(cell_vals), jnp.asarray(u_vals), halo_args)
        cell_out = np.asarray(cell_out).reshape(n_dev, max_lc, W)
        u_out = np.asarray(u_out).reshape(n_dev, max_le, nlev)

        n_halo_checked = 0
        for d, part in enumerate(partitions):
            # Owned passthrough + every HALO entity == owner's sentinel,
            # checked FIELD BY FIELD through the production unpack.
            lc = np.asarray(part.local_cells)
            nl = part.n_local_cells
            T_l, ps_l, phis_l, q_l = _unpack_cell_state(
                cell_out[d], nlev)
            got = {
                "T": np.asarray(T_l)[:nl],
                "p_s": np.asarray(ps_l)[:nl],
                "phis": np.asarray(phis_l)[:nl],
            }
            for i, k in enumerate(tkeys):
                got[k] = np.asarray(q_l)[:nl, i * nlev:(i + 1) * nlev]
            for k, arr in got.items():
                np.testing.assert_array_equal(
                    arr, fields[k][lc],
                    err_msg=(f"device {d}: field {k} slot routing wrong "
                             f"(n_dev={n_dev})"),
                )
            le = np.asarray(part.local_edges)
            np.testing.assert_array_equal(
                u_out[d, : part.n_local_edges], u_vals[le],
                err_msg=f"device {d}: edge routing wrong",
            )
            n_halo_checked += (nl - part.n_owned_cells)
        assert n_halo_checked > 0, "no halo cells checked — vacuous"


# ===========================================================================
# Scan stability (10 steps) + dtype fixed point
# ===========================================================================

class TestScanStability:

    def test_scan_10_steps_moist_with_carry(self):
        """The full-production native step runs inside jit + lax.scan
        for 10 steps (state AND phys_state as the scan carry, traced
        forcing broadcast) with finite output and a dtype fixed point."""
        _need_multi_device(2)
        from legoesm.atmosphere.kessler_forcing import (
            make_kessler_forcing_mpas,
        )

        mesh, sigma, cfg, model, state0 = _build(moist=True)
        kessler = make_kessler_forcing_mpas(_DT)
        step, _dc = _make_sharded_step(
            mesh, model, 2, halo_strategy="ppermute")
        step_c, _dc2 = _make_sharded_step(
            mesh, model, 2, return_phys_state=True,
            halo_strategy="ppermute")
        carry0 = {"accum": jnp.zeros((mesh.nCells,))}

        @jax.jit
        def run_moist(s0):
            def body(s, _):
                return step(s, _DT, physics_fn=kessler), None
            return jax.lax.scan(body, s0, None, length=10)[0]

        out = run_moist(state0)
        for leaf in jax.tree.leaves(out):
            assert bool(jnp.all(jnp.isfinite(leaf))), \
                "non-finite state after 10-step scan"

        # dtype fixed point: the scan carry structure/dtypes must be
        # preserved (compute->storage cast closes the loop).
        in_dt = [str(x.dtype) for x in jax.tree.leaves(state0)]
        out_dt = [str(x.dtype) for x in jax.tree.leaves(out)]
        assert in_dt == out_dt, f"dtype drift over scan: {in_dt} -> {out_dt}"

        @jax.jit
        def run_carry(s0, c0):
            def body(carry, _):
                s, c = carry
                s2, c2 = step_c(s, _DT, physics_fn=_toy_stateful_physics,
                                phys_state=c)
                return (s2, c2), None
            return jax.lax.scan(body, (s0, c0), None, length=10)[0]

        s_dry = state0._replace(tracers=None)
        out2, carry2 = run_carry(s_dry, carry0)
        for leaf in jax.tree.leaves((out2, carry2)):
            assert bool(jnp.all(jnp.isfinite(leaf))), \
                "non-finite state/carry after 10-step carry scan"
        assert str(carry2["accum"].dtype) == str(carry0["accum"].dtype)


# ===========================================================================
# Refusal + schema tripwire contracts (loud, never silent)
# ===========================================================================

class TestContracts:

    def test_stateful_refused_without_carry_channel(self):
        """return_phys_state=False + tagged stateful physics_fn →
        NotImplementedError (issue #405/#413), at CALL time."""
        _need_multi_device(2)
        mesh, sigma, cfg, model, state0 = _build(moist=False)
        step, _dc = _make_sharded_step(mesh, model, 2)
        with pytest.raises(NotImplementedError, match="carry"):
            step(state0, _DT, physics_fn=_toy_stateful_physics)

    def test_stateful_refused_without_carry_value(self):
        """return_phys_state=True + phys_state=None + stateful physics →
        NotImplementedError (would silently reseed)."""
        _need_multi_device(2)
        mesh, sigma, cfg, model, state0 = _build(moist=False)
        step, _dc = _make_sharded_step(
            mesh, model, 2, return_phys_state=True)
        with pytest.raises(NotImplementedError, match="phys_state"):
            step(state0, _DT, physics_fn=_toy_stateful_physics,
                 phys_state=None)

    def test_single_device_carry_contract_refused(self):
        """A single-device config cannot honor the (state, carry) return
        contract (model.step stashes eagerly) — refuse loudly."""
        from legoesm.parallel.mesh import DeviceConfig
        from legoesm.parallel.sharded_dynamics import (
            make_voronoi_sharded_step,
        )

        config = DeviceConfig(
            mesh=None, face_sharding=None, replicated_sharding=None,
            n_devices=1, backend="CPU", is_distributed=False,
            grid_type="voronoi", voronoi_dims=(100, 200, 50),
        )
        with pytest.raises(ValueError, match="return_phys_state"):
            make_voronoi_sharded_step(
                object(), config, return_phys_state=True)

    def test_schema_tripwire_extra_field(self):
        """A NEW state field must fail loudly (it would silently ride
        UNEXCHANGED through the packed halo otherwise) — synthetic
        violation proves the tripwire is non-vacuous."""
        from legoesm.parallel.sharded_dynamics import (
            check_voronoi_spmd_state_schema,
        )

        class _GrownState(NamedTuple):
            u: object
            T: object
            p_s: object
            phis: object
            v: object
            tracers: object
            brand_new_field: object

        bad = _GrownState(*([None] * 7))
        with pytest.raises(ValueError, match="brand_new_field"):
            check_voronoi_spmd_state_schema(bad)

    def test_schema_tripwire_v_not_none(self):
        from legoesm.parallel.sharded_dynamics import (
            check_voronoi_spmd_state_schema,
        )

        _, _, _, _, state0 = _build(moist=True)
        # Canonical wire order comes back sorted.
        assert check_voronoi_spmd_state_schema(state0) == (
            "q_c", "q_r", "q_v")
        assert check_voronoi_spmd_state_schema(
            state0._replace(tracers=None)) == ()
        with pytest.raises(ValueError, match="v must be None"):
            check_voronoi_spmd_state_schema(state0._replace(v=state0.T))


# ===========================================================================
# Route-A (mpi4jax) np=1 cross-check
# ===========================================================================

class TestRouteACrossCheck:

    def test_np1_route_a_matches_native_and_serial(self):
        """Three-way full-production gate: serial model.step == route-A
        ``make_voronoi_mpi_step`` on an np=1 layout == native 2-device
        SPMD step, for the moist Kessler configuration with the carry
        contract armed.  Skips without an mpi4jax stack (route-A's mass
        fixer allreduces even at np=1)."""
        pytest.importorskip("mpi4jax")
        _need_multi_device(2)
        from legoesm.atmosphere.kessler_forcing import (
            make_kessler_forcing_mpas,
        )
        from legoesm.parallel.voronoi_mpi import (
            make_voronoi_mpi_step,
            make_voronoi_partition_layout,
            scatter_state_voronoi,
        )

        mesh, sigma, cfg, model, state0 = _build(moist=True)
        kessler = make_kessler_forcing_mpas(_DT)

        ref = state0
        for _ in range(2):
            ref = model.step(ref, _DT, physics_fn=kessler)

        layout = make_voronoi_partition_layout(mesh, 0, 1)
        mpi_step = make_voronoi_mpi_step(
            model, layout, sigma, config=cfg, physics_fn=kessler,
            return_phys_state=True,
        )
        local = scatter_state_voronoi(state0, layout.partition)
        for _ in range(2):
            local, _carry = mpi_step(local, _DT)
        # np=1: local ordering == global ordering (identity partition).
        _assert_state_close(local, ref, moist=True, label="route-A np1")

        step, _dc = _make_sharded_step(mesh, model, 2)
        out = state0
        for _ in range(2):
            out = step(out, _DT, physics_fn=kessler)
        _assert_state_close(out, ref, moist=True, label="native vs serial")
        _assert_state_close(out, local, moist=True,
                            label="native vs route-A")
