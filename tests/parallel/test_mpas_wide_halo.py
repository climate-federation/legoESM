"""Wide-halo (communication-avoiding) MPAS sharded step.

``LEGOESM_MPAS_WIDE_HALO=1`` makes ``make_voronoi_sharded_step`` build
its partition infra at depth ``evals x SPMD_HALO_DEPTH`` and run the
WHOLE RK body inside ``shard_map`` after ONE packed halo fill, instead
of one depth-3 fill per tendency evaluation.  Motivation: the GPU lane
is bound by sequential-collective count x per-collective latency floor
(campaign doc 2026-08-10); one wide fill cuts 3 x n_rounds -> n_rounds
sequential collectives per step.

Correctness rests on the Shu-Osher shrinking-region argument: each
tendency evaluation reads at most ``SPMD_HALO_DEPTH`` rings, so after
the k-th of N evaluations the stage state is valid on depth
``(N-k) * SPMD_HALO_DEPTH`` and the final state is valid exactly on the
owned cells the kernel returns.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _need_multi_device(n: int):
    devices = jax.devices("cpu")
    if len(devices) < n:
        pytest.skip(
            f"Need at least {n} CPU devices "
            f"(set XLA_FLAGS=--xla_force_host_platform_device_count={n})"
        )


def _build(devices: int, *, wide: bool, monkeypatch, subdivision_level=4,
           halo_strategy="ppermute"):
    """Sharded model + step + initial state on a mesh reordered for
    ``devices`` (both arms share the reorder so arrays compare
    directly). Returns (step_fn, state, dt, mesh)."""
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.parallel.voronoi_partition import (
        reorder_voronoi_for_sharding,
    )
    from legoesm.parallel.mesh import (
        create_voronoi_device_mesh, replicate_pytree,
    )
    from legoesm.parallel.sharded_dynamics import make_voronoi_sharded_step
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_mpas

    monkeypatch.setenv("LEGOESM_MPAS_WIDE_HALO", "1" if wide else "0")

    dt = 600.0
    mesh = create_voronoi_mesh(subdivision_level=subdivision_level)
    mesh = reorder_voronoi_for_sharding(mesh, devices)
    sigma = create_sigma_coordinate(8)
    cfg = MPASPrimitiveEquationConfig(
        nu_del4=1e16, nu_del4_ps=1e16,
        fix_mass=True, pv_scheme="energy",
        time_integrator="ssp_rk3",
    )
    dev_config = create_voronoi_device_mesh(
        nCells=mesh.nCells, nEdges=mesh.nEdges,
        nVertices=mesh.nVertices, n_devices=devices,
    )
    model = MPASPrimitiveEquationModel(
        replicate_pytree(mesh, dev_config), sigma, cfg)
    state = baroclinic_wave_init_mpas(mesh, sigma, perturbed=True)
    step_fn = make_voronoi_sharded_step(
        model, dev_config, halo_strategy=halo_strategy)
    return step_fn, state, dt, mesh


class TestWideHaloEquivalence:

    def test_wide_matches_default_owned_state(self, monkeypatch):
        """5 steps wide vs per-fill: owned prognostics agree.

        Not asserted bit-identical: the wide path RECOMPUTES stage
        values on halo rings from the local mesh, whose outermost-ring
        connectivity is AND-filtered (see
        test_voronoi_sharded_equivalence docstring); that ring sits
        ``(evals-1) * SPMD_HALO_DEPTH`` hops from any owned cell, so
        the owned drift stays within the same physical-equivalence
        band the default path itself holds against single-device.
        """
        _need_multi_device(4)
        step_w, state, dt, _ = _build(4, wide=True, monkeypatch=monkeypatch)
        assert step_w._wide_halo_effective is True
        assert step_w._halo_depth_effective == 9
        sw = state
        for _ in range(5):
            sw = step_w(sw, dt)

        step_d, state, dt, _ = _build(4, wide=False, monkeypatch=monkeypatch)
        assert step_d._wide_halo_effective is False
        assert step_d._halo_depth_effective == 3
        sd = state
        for _ in range(5):
            sd = step_d(sd, dt)

        for name in ("u", "T", "p_s"):
            got = np.asarray(getattr(sw, name).data)
            want = np.asarray(getattr(sd, name).data)
            assert np.all(np.isfinite(got)), f"{name}: non-finite (wide)"
            scale = max(float(np.max(np.abs(want))), 1e-30)
            rel = float(np.max(np.abs(got - want))) / scale
            assert rel < 1e-2, (
                f"{name}: wide vs default max rel diff {rel:.3e} "
                f"exceeds the physical-equivalence band")

    def test_wide_collective_count_drops(self, monkeypatch):
        """The compiled wide step contains exactly ONE fill's worth of
        collective-permutes (= the estimator's depth-9 round count) vs
        3 fills' worth on the default path.  CPU census: XLA:CPU does
        not combine collective-permutes, so the HLO count equals the
        schedule round count (the GPU count can only be lower)."""
        _need_multi_device(4)
        from legoesm.parallel.sharded_dynamics import spmd_schedule_cost

        step_w, state, dt, mesh = _build(
            4, wide=True, monkeypatch=monkeypatch)
        step_d, _, _, _ = _build(4, wide=False, monkeypatch=monkeypatch)

        def _cp_count(step_fn):
            txt = jax.jit(
                lambda s: step_fn(s, dt)).lower(state).compile().as_text()
            # Async form counts the start only; the sync spelling
            # " collective-permute(" (leading space + paren) matches
            # neither -start nor -done lines.
            return (txt.count("collective-permute-start(")
                    + txt.count(" collective-permute("))

        n_wide = _cp_count(step_w)
        n_default = _cp_count(step_d)
        r9 = spmd_schedule_cost(mesh, 4, already_reordered=True,
                                halo_depth=9)["n_rounds"]
        r3 = spmd_schedule_cost(mesh, 4, already_reordered=True,
                                halo_depth=3)["n_rounds"]
        assert n_wide == r9, (n_wide, r9)
        assert n_default == 3 * r3, (n_default, r3)
        assert n_wide < n_default

    def test_wide_gradients_match_default(self, monkeypatch):
        """d(sum T after 1 step)/d(T0) agrees wide vs default — the
        deep fill + in-shard RK body is AD-transparent (ppermute and
        gather/scatter transposes accumulate ghost cotangents to the
        owner by construction)."""
        _need_multi_device(2)
        step_w, state, dt, _ = _build(2, wide=True, monkeypatch=monkeypatch)
        step_d, _, _, _ = _build(2, wide=False, monkeypatch=monkeypatch)

        def _loss(step_fn):
            def f(T0):
                s = state._replace(T=state.T.replace(data=T0))
                return jnp.sum(step_fn(s, dt).T.data ** 2)
            return jax.grad(f)(state.T.data)

        gw = np.asarray(_loss(step_w))
        gd = np.asarray(_loss(step_d))
        # KNOWN PRE-EXISTING: the DEFAULT sharded path's gradient
        # already contains non-finite entries (96 at subdiv-4/2dev,
        # identical rows both paths — measured 2026-08-10; 0*inf in the
        # tendency backward at rows whose primal is a halo-boundary /
        # padding artefact). Wide halo must not ADD any: the
        # non-finite masks must be IDENTICAL, and every finite entry
        # must agree.
        assert np.array_equal(np.isfinite(gw), np.isfinite(gd)), (
            "wide halo changed WHICH gradient entries are non-finite "
            f"(wide {int((~np.isfinite(gw)).sum())} vs default "
            f"{int((~np.isfinite(gd)).sum())})")
        fin = np.isfinite(gd)
        scale = max(float(np.max(np.abs(gd[fin]))), 1e-30)
        rel = float(np.max(np.abs(gw[fin] - gd[fin]))) / scale
        assert rel < 1e-2, f"grad wide vs default max rel diff {rel:.3e}"


class TestWideHaloDispatch:

    def test_env_validation_raises(self, monkeypatch):
        _need_multi_device(2)
        from legoesm.parallel.sharded_dynamics import _resolve_wide_halo
        with pytest.raises(ValueError, match="LEGOESM_MPAS_WIDE_HALO"):
            _resolve_wide_halo("yes")
        assert _resolve_wide_halo("") is False
        assert _resolve_wide_halo("0") is False
        assert _resolve_wide_halo("1") is True

    def test_unknown_integrator_refused(self, monkeypatch):
        """Wide mode must refuse an integrator with no evals entry
        rather than guess a halo depth."""
        from legoesm.parallel.sharded_dynamics import (
            _INTEGRATOR_TENDENCY_EVALS,
        )
        from legoesm.timestepping.dispatch import available_integrators
        missing = [n for n in available_integrators()
                   if n not in _INTEGRATOR_TENDENCY_EVALS]
        assert missing == [], (
            f"integrators without a wide-halo evals entry: {missing} — "
            "add them to _INTEGRATOR_TENDENCY_EVALS (grow-only) or wide "
            "mode silently cannot size their halo")
