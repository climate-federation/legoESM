#!/usr/bin/env python
"""Blocker-1 decisive experiment: does the GSPMD ``shard_map`` cube-panel halo
hold strong-scaling efficiency past the ``mpi4jax`` np6 anti-scale cliff?

Context (``docs/performance/scream_parity_scope.md`` §5). The single question that
gates SCREAM-parity feasibility is whether the native-NCCL GSPMD cube halo
(``jax.lax.ppermute`` inside ``shard_map``) scales, unlike the ``mpi4jax``
point-to-point path which is blocking, host-staged, and measured to *anti-scale*
across cube faces on a TCP fabric (107→206 ms/step np1→6,
``coupler/.../compiled_segments.py:484``).

This harness drives the **production** SPMD halo path — ``activate_spmd_halo_backend``
+ ``model.step`` (whose FV3 operators call ``pad_halo`` → ``explicit_pad_halo``) —
so the result speaks for the model, not a bespoke copy. It runs three gates:

* **Correctness** — the face-sharded SPMD dycore step matches the single-device
  step (the halo must move exactly the right data). The pytree structure, leaf
  count, shape and dtype are all asserted equal first, so the check cannot pass
  vacuously.
* **AD** — a nonuniform-cotangent VJP through the SPMD ``ppermute`` halo matches
  the single-device reference VJP, AND the lowered SPMD function is asserted to
  contain a ``collective_permute`` — the ppermute kernel's signature (the
  all_gather diagnostic kernel must NOT satisfy this) — proving the ppermute path
  ran, not a silent local fallback. This is the ``collective_permute`` VJP
  correctness that Blocker 1 depends on (CLAUDE.md MPI-AD doctrine).
* **Strong scaling** — steady-state ms/step at fixed global cube size, sweeping
  device count (1 → 2 → 3 → 6, and ``6·kt²`` tiled if enough devices), reported as
  speed-up and parallel efficiency vs the single-device baseline.

Pass condition (Blocker 1 clears its gate): the GSPMD cube halo holds
**≥ ``--pass-efficiency`` strong-scaling efficiency at the largest device count**.
On a local macOS/CPU box (Metal is unusable — see memory) this validates
correctness + AD + the ``shard_map`` plumbing on emulated devices and yields a CPU
scaling proxy; the real GPU numbers come from the Derecho 2-node NCCL job. The
``--gate-efficiency`` flag makes this a *decisive* run: the efficiency threshold
becomes a hard exit gate AND at least one device count > 1 is required (a
single-device run can never be decisive). Left off, the harness is a
correctness/AD smoke.

Local (emulated 6 devices)::

    XLA_FLAGS=--xla_force_host_platform_device_count=6 \
      .venv/bin/python scripts/bench/bench_cube_shardmap_halo.py \
      --device-counts 1,2,3,6 --n-grid 24 --output-dir results/blocker1

Real hardware (2-node NCCL, via scripts/cluster/scaling_derecho/): add
``--gate-efficiency --pass-efficiency 0.6`` and launch under ``jax.distributed``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path

logger = logging.getLogger("bench_cube_shardmap_halo")

# Valid single-mesh face-shard counts (must divide 6); >6 uses 6*kt^2 tiling.
_FACE_DIVISORS = (1, 2, 3, 6)


# ---------------------------------------------------------------------------
# Global halo-backend snapshot/restore (never leak SPMD state to a later call)
# ---------------------------------------------------------------------------
def _snapshot_halo() -> tuple:
    """Snapshot the public halo-backend state (backend, SPMD mesh, MPI topology)."""
    from legoesm.grids import halo as halo_mod
    return (halo_mod.get_halo_backend(), halo_mod.get_spmd_mesh(),
            halo_mod.get_mpi_topology())


def _restore_halo(snap: tuple) -> None:
    """Restore a snapshot taken by :func:`_snapshot_halo`, as fully as the public
    API allows.

    ``deactivate_spmd_halo_backend`` clears BOTH ``grids.halo`` and
    ``cubesphere_exchange`` module meshes, so restoring a prior *SPMD* caller
    requires re-activating it (which repopulates ``cubesphere_exchange._spmd_mesh``);
    local/MPI priors are restored via the public setters.

    Precondition: this harness is expected to be entered from a ``local`` (or MPI)
    backend — the standalone decisive experiment never runs inside another SPMD
    context. Restoring a *foreign* SPMD prior cannot recover the private
    ppermute/all_gather kernel-selection flag through the public API (importing
    the private ``_use_ppermute`` across modules is disallowed by repo policy), so
    reactivation resets it to the default ppermute selection and warns.
    """
    backend, mesh, topo = snap
    from legoesm.grids import halo as halo_mod
    from legoesm.parallel.cubesphere_exchange import (
        activate_spmd_halo_backend,
        deactivate_spmd_halo_backend,
    )
    deactivate_spmd_halo_backend()  # clean baseline: local, both module meshes None
    if backend == "spmd" and mesh is not None:
        logger.warning(
            "restoring a prior SPMD halo backend: the private ppermute/all_gather "
            "selection is reset to the default ppermute (not snapshottable via the "
            "public API). Enter the harness from a local/MPI backend to avoid this.")
        activate_spmd_halo_backend(mesh)  # restores halo + cubesphere_exchange globals
    else:
        halo_mod.set_halo_backend(backend, topology=topo)
        halo_mod.set_spmd_mesh(mesh)


@contextlib.contextmanager
def _spmd_backend(mesh, n: int, nlev: int):
    """Activate the SPMD halo backend, then fully restore the caller's prior state.

    Left bare, ``deactivate_spmd_halo_backend`` would drop the backend to
    ``"local"`` and clear both module meshes, corrupting a later in-process
    caller. This snapshots and restores it (see :func:`_restore_halo`).
    """
    from legoesm.parallel.cubesphere_exchange import activate_spmd_halo_backend
    snap = _snapshot_halo()
    activate_spmd_halo_backend(mesh, n=n, nlev=nlev)
    try:
        yield
    finally:
        _restore_halo(snap)


_PPERMUTE_RE = re.compile(r"collective[_-]permute")
_ALLGATHER_RE = re.compile(r"all[_-]gather")


def _hlo_has_ppermute(fn, *args) -> bool | None:
    """Best-effort: does the lowered HLO of ``fn(*args)`` contain a
    ``collective_permute`` — the signature of the *ppermute* cube-halo kernel?

    Deliberately matches ONLY ``collective[_-]permute`` (StableHLO underscore +
    optimized-XLA hyphen), NOT ``all_gather``: this experiment is about the
    bandwidth-optimal ppermute path; the all_gather diagnostic kernel replicates
    all compute and must never satisfy the proof. If ``all_gather`` is found
    instead, that is logged as a warning. Returns ``None`` if lowering is
    unsupported (never raises), so a best-effort timing probe can record "unknown".
    """
    import jax
    try:
        text = jax.jit(fn).lower(*args).as_text()
    except Exception as exc:  # lowering a bound step method can be finicky
        logger.debug("HLO probe skipped: %r", exc)
        return None
    if _PPERMUTE_RE.search(text):
        return True
    if _ALLGATHER_RE.search(text):
        logger.warning("HLO contains all_gather (diagnostic replicated kernel), "
                       "NOT the ppermute collective this experiment measures.")
    return False


# ---------------------------------------------------------------------------
# Model / state construction (reuses the exact cube IC the scaling bench uses)
# ---------------------------------------------------------------------------
def _build_cube_model(n_grid: int, n_lev: int, dt: float):
    """Build the cube PE dycore + a baroclinic-wave FV3 initial state.

    Mirrors ``run_levante_gpu_scaling.py`` and
    ``tests/parallel/test_cubed_sphere_spmd_step.py`` so the timed integrand is
    the production step, not a reduced proxy.
    """
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationConfig,
        CDGridPrimitiveEquationModel,
        fv3_to_hydrostatic,
        hydrostatic_to_fv3,
    )
    from legoesm.core.cfl import (
        adaptive_hyperdiff_coeff,
        estimate_min_dx_cubed_sphere,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.vertical import create_sigma_coordinate

    from tests.test_cases.baroclinic_wave import baroclinic_wave_init

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
        time_integrator="ssp_rk3",
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, cfg)
    state_fv3 = hydrostatic_to_fv3(state_cc, cdgrid)
    return model, state_fv3, cdgrid, fv3_to_hydrostatic


def _available_devices() -> int:
    import jax
    return len(jax.devices())


def _valid_counts(requested: list[int], available: int,
                  n_grid: int) -> tuple[list[int], list[int]]:
    """Split requested device counts into (runnable, skipped).

    Runnable = available AND either a face divisor (1,2,3,6) or a genuine
    ``6·kt²`` sub-face tile whose tile factor ``kt`` divides ``n_grid`` (a face
    of ``n_grid`` cells must split evenly into ``kt`` tiles, else sharding would
    fail or silently fall back). Skipped counts are returned so the caller can
    LOG them (no silent coverage drop).
    """
    runnable, skipped = [], []
    for c in requested:
        if c > available or c < 1:
            skipped.append(c)
            continue
        if c in _FACE_DIVISORS:
            runnable.append(c)
            continue
        ok_tile = False
        if c > 6 and c % 6 == 0:
            kt = math.isqrt(c // 6)
            ok_tile = (kt * kt * 6 == c and kt >= 2 and n_grid % kt == 0)
        (runnable if ok_tile else skipped).append(c)
    return runnable, skipped


# ---------------------------------------------------------------------------
# Timing + correctness sweep
# ---------------------------------------------------------------------------
def _run_at_count(model, state_fv3, dt, n_devices, n_warmup, n_timing, n_steps_final,
                  gather_final: bool = True, devices=None):
    """Time steady-state ms/step and return the final state at one device count.

    ``n_devices == 1`` uses the serial local halo backend; ``>1`` activates the
    SPMD (``ppermute``/``shard_map``) backend over a face-sharded mesh. For SPMD
    counts the timed step's HLO is probed for a cross-device collective so we can
    confirm the measured executable really is the SPMD one.

    ``gather_final=False`` skips the host gather of the final state — used by the
    timing-only point mode, where a cross-process ``device_get`` collective on an
    unused state would be wasted (and needlessly synchronising under
    multi-controller).

    ``devices`` (explicit list) is REQUIRED under multi-controller: without it
    ``create_device_mesh`` falls back to ``jax.local_devices()`` (one GPU per rank
    on a 1-GPU/rank launch), which cannot build the intended global face mesh.
    Point mode passes ``jax.devices()[:n_devices]`` (the global slice).
    """
    import jax

    if n_timing < 1 or n_warmup < 0:
        raise ValueError(f"n_timing must be >=1 and n_warmup >=0, "
                         f"got n_timing={n_timing}, n_warmup={n_warmup}")

    def _timed(step_state):
        s = step_state
        for _ in range(n_warmup):
            s = model.step(s, dt)
        jax.block_until_ready(s)
        t0 = time.perf_counter()
        for _ in range(n_timing):
            s = model.step(s, dt)
        jax.block_until_ready(s)
        elapsed = time.perf_counter() - t0
        return elapsed / n_timing * 1e3  # ms/step

    def _final(step_state):
        s = step_state
        for _ in range(n_steps_final):
            s = model.step(s, dt)
        jax.block_until_ready(s)
        return s

    if n_devices == 1:
        ms = _timed(state_fv3)
        return ms, _final(state_fv3), None

    from legoesm.parallel.mesh import create_device_mesh, shard_pytree

    # Under multi-controller an explicit `devices` is mandatory: with devices=None
    # create_device_mesh uses jax.local_devices() (1 GPU/rank), which — if a rank
    # happens to hold >= n_devices locally — could build a WRONG local mesh that
    # the (mesh, n_devices) check below would not catch. Refuse it up front.
    if devices is None and jax.process_count() > 1:
        raise RuntimeError(
            "_run_at_count under multi-controller (process_count>1) requires an "
            "explicit `devices` (e.g. jax.devices()[:n]); the local_devices "
            "fallback cannot form the global face mesh.")
    dev_config = create_device_mesh(n_devices=n_devices, devices=devices)
    if dev_config.mesh is None or dev_config.n_devices != n_devices:
        raise RuntimeError(
            f"SPMD mesh build failed: requested {n_devices} devices but got "
            f"n_devices={dev_config.n_devices}, mesh={dev_config.mesh}. Under "
            f"multi-controller pass devices=jax.devices()[:n] (local_devices "
            f"fallback cannot form the global face mesh).")
    with _spmd_backend(dev_config.mesh, n=0, nlev=0):
        s0 = shard_pytree(state_fv3, dev_config)
        hlo_collective = _hlo_has_ppermute(lambda st: model.step(st, dt), s0)
        ms = _timed(s0)
        s = _final(s0)
        if gather_final:
            s = jax.tree_util.tree_map(lambda x: jax.device_get(x), s)
    return ms, s, hlo_collective


def _max_state_diff(fv3_to_hydrostatic, cdgrid, state_a, state_b) -> float:
    """Max abs difference between two FV3 states, in hydrostatic space.

    Fails loudly on any pytree-structure / leaf-count / shape / dtype mismatch
    and refuses a comparison with zero floating leaves, so the correctness gate
    can never pass vacuously.
    """
    import jax
    import numpy as np

    a = fv3_to_hydrostatic(state_a, cdgrid)
    b = fv3_to_hydrostatic(state_b, cdgrid)
    sa = jax.tree_util.tree_structure(a)
    sb = jax.tree_util.tree_structure(b)
    if sa != sb:
        raise ValueError(f"correctness: pytree structure mismatch {sa} != {sb}")
    la = [np.asarray(x) for x in jax.tree_util.tree_leaves(a)]
    lb = [np.asarray(x) for x in jax.tree_util.tree_leaves(b)]
    if len(la) != len(lb):
        raise ValueError(f"correctness: leaf count mismatch {len(la)} != {len(lb)}")
    worst, compared = 0.0, 0
    for i, (xa, xb) in enumerate(zip(la, lb)):
        if xa.shape != xb.shape:
            raise ValueError(f"correctness: leaf {i} shape {xa.shape} != {xb.shape}")
        if xa.dtype != xb.dtype:
            raise ValueError(f"correctness: leaf {i} dtype {xa.dtype} != {xb.dtype}")
        if np.issubdtype(xa.dtype, np.floating):
            worst = max(worst, float(np.max(np.abs(xa - xb))))
            compared += 1
    if compared == 0:
        raise ValueError("correctness: no floating leaves compared (vacuous)")
    return worst


# ---------------------------------------------------------------------------
# AD gate: VJP through the SPMD ppermute halo vs single-device reference
# ---------------------------------------------------------------------------
def _check_ad(n_grid: int, n_devices: int, dtype_name: str) -> dict:
    """VJP of the cube halo with a NONUNIFORM cotangent; SPMD must match local.

    A uniform loss like ``sum(pad(x)**2)`` is too symmetric — a wrong permutation
    with equal copy-multiplicity yields the same gradient. Here the cotangent
    ``w`` is unique per output cell, so any mis-routed strip changes the adjoint.
    The lowered SPMD function is also asserted to contain a ``collective_permute``
    (the ppermute kernel's signature), proving the ppermute path ran rather than a
    silent local fallback that would trivially match. The whole routine snapshots
    and restores the caller's prior halo backend so neither the forced-local
    reference nor the SPMD activation leaks.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from legoesm.grids.halo import pad_halo
    from legoesm.parallel.cubesphere_exchange import deactivate_spmd_halo_backend, explicit_pad_halo
    from legoesm.parallel.mesh import create_device_mesh, shard_pytree

    dtype = jnp.float64 if dtype_name == "float64" else jnp.float32
    # Deterministic input (index-varying; no host RNG, no Math.random).
    base = np.arange(6 * n_grid * n_grid, dtype=np.float64).reshape(6, n_grid, n_grid)
    x = jnp.asarray((np.sin(base) + 0.5 * np.cos(0.3 * base)).astype(np.float64)).astype(dtype)

    snap = _snapshot_halo()
    try:
        # --- Local reference VJP (forced serial backend) ---
        deactivate_spmd_halo_backend()
        y_local, vjp_local = jax.vjp(pad_halo, x)
        pad_n = y_local.shape[1]
        # Nonuniform, unique-per-cell cotangent.
        widx = np.arange(6 * pad_n * pad_n, dtype=np.float64).reshape(6, pad_n, pad_n)
        w = jnp.asarray((1.0 + np.sin(1.7 * widx) + 0.11 * widx).astype(np.float64)).astype(dtype)
        g_local = np.asarray(vjp_local(w)[0])

        # --- SPMD VJP (ppermute) ---
        dev_config = create_device_mesh(n_devices=n_devices)
        with _spmd_backend(dev_config.mesh, n=n_grid, nlev=1):
            x_sh = shard_pytree(x, dev_config)

            def _f(z):
                return explicit_pad_halo(z, dev_config.mesh)

            has_collective = _hlo_has_ppermute(_f, x_sh)
            y_sh, vjp_sh = jax.vjp(_f, x_sh)
            w_sh = jax.device_put(w, dev_config.face_sharding)
            g_spmd = np.asarray(jax.device_get(vjp_sh(w_sh)[0]))
    finally:
        _restore_halo(snap)

    max_abs = float(np.max(np.abs(g_local - g_spmd)))
    scale = float(np.max(np.abs(g_local))) or 1.0
    return {
        "n_devices": n_devices,
        "grad_max_abs_diff": max_abs,
        "grad_rel_diff": max_abs / scale,
        "collective_in_hlo": has_collective,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(args) -> dict:
    import jax

    available = _available_devices()
    backend = jax.default_backend()
    requested = [int(c) for c in args.device_counts.split(",") if c.strip()]
    runnable, skipped = _valid_counts(requested, available, args.n_grid)
    if 1 not in runnable:
        runnable = [1] + runnable  # single-device baseline is mandatory
    runnable = sorted(set(runnable))
    if skipped:
        logger.warning(
            "SKIPPING device counts %s (exceed %d available, not a face divisor, "
            "or a 6*kt^2 tile with kt | n_grid) — coverage NOT silently dropped.",
            skipped, available,
        )

    dt = float(args.dt)
    model, state_fv3, cdgrid, fv3_to_hydro = _build_cube_model(
        args.n_grid, args.n_lev, dt)

    # --- Strong-scaling + correctness sweep -------------------------------
    per_count, final_states = [], {}
    for c in runnable:
        ms, final, hlo_collective = _run_at_count(
            model, state_fv3, dt, c,
            args.n_warmup, args.n_timing, args.n_correctness_steps)
        final_states[c] = final
        per_count.append({"n_devices": c, "ms_per_step": ms,
                          "spmd_hlo_collective": hlo_collective})
        logger.info("n_devices=%d  ms/step=%.3f  hlo_collective=%s",
                    c, ms, hlo_collective)

    base_ms = next(r["ms_per_step"] for r in per_count if r["n_devices"] == 1)
    ref_state = final_states[1]
    for r in per_count:
        c = r["n_devices"]
        r["speedup"] = base_ms / r["ms_per_step"] if r["ms_per_step"] > 0 else 0.0
        r["efficiency"] = r["speedup"] / c
        r["correctness_max_abs_diff"] = (
            0.0 if c == 1
            else _max_state_diff(fv3_to_hydro, cdgrid, ref_state, final_states[c]))

    largest = max(r["n_devices"] for r in per_count)
    largest_row = next(r for r in per_count if r["n_devices"] == largest)
    multi_device_ran = largest > 1
    decisive = bool(args.gate_efficiency)

    # --- Correctness gate --------------------------------------------------
    worst_corr = max(r["correctness_max_abs_diff"] for r in per_count)
    correctness_pass = worst_corr <= args.correctness_tol

    # --- AD gate (only meaningful with >1 device) -------------------------
    ad_result, ad_pass, ad_ran = None, True, False
    if multi_device_ran:
        try:
            ad_result = _check_ad(args.n_grid, largest, args.precision)
            ad_ran = True
            # Pass requires BOTH the gradient match AND proof the collective ran
            # (a silent local fallback would match trivially). ``None`` (lowering
            # unsupported) does not fail the gate but is surfaced.
            grad_ok = ad_result["grad_rel_diff"] <= args.ad_tol
            coll = ad_result["collective_in_hlo"]
            # Require the ppermute collective to be PROVEN present (True). None
            # (lowering unsupported) or False both fail — a passing AD gate must
            # have exercised the cross-device ppermute VJP, not a local fallback.
            ad_pass = grad_ok and (coll is True)
        except Exception as exc:  # AD failure IS a finding, not a crash
            ad_result = {"n_devices": largest, "error": repr(exc)}
            ad_pass = False
            logger.error("AD gate raised (Blocker-1 AD risk): %r", exc)
    else:
        logger.warning(
            "AD gate SKIPPED — only 1 device available; run with "
            "XLA_FLAGS=--xla_force_host_platform_device_count>=2 to exercise it.")

    # --- Efficiency gate (hard only with --gate-efficiency) ---------------
    eff_at_largest = largest_row["efficiency"]
    efficiency_pass = (not decisive) or (
        multi_device_ran and eff_at_largest >= args.pass_efficiency)

    # --- Timed-path ppermute gate -----------------------------------------
    # The AD gate proves ``explicit_pad_halo`` lowered ppermute, but the
    # efficiency number is timed on ``model.step``; gate that the TIMED executable
    # at every >1-device count actually used the ppermute collective (else we may
    # be measuring the replicated all_gather path). None (probe unsupported) fails.
    timed_multi = [r for r in per_count if r["n_devices"] > 1]
    timed_ppermute_pass = (
        all(r["spmd_hlo_collective"] is True for r in timed_multi)
        if timed_multi else True)

    # --- Decisive-mode preconditions: a single-device run is never decisive -
    reasons = []
    if decisive and not multi_device_ran:
        reasons.append(
            "decisive run (--gate-efficiency) requires >=1 device count > 1; "
            f"only single-device ran (available={available}). Set "
            "XLA_FLAGS=--xla_force_host_platform_device_count / launch multi-GPU.")
    if decisive and multi_device_ran and not ad_ran:
        reasons.append("decisive run requires the AD gate to execute")

    gates = {"correctness": correctness_pass, "ad": ad_pass,
             "efficiency": efficiency_pass, "timed_ppermute": timed_ppermute_pass}
    enforced = ["correctness"]
    if multi_device_ran:
        enforced.append("ad")
    if decisive:
        enforced += ["efficiency", "timed_ppermute"]
    overall_pass = all(gates[g] for g in enforced) and not reasons

    result = {
        "config": {
            "n_grid": args.n_grid, "n_lev": args.n_lev, "dt": dt,
            "precision": args.precision, "backend": backend,
            "devices_available": available,
            "device_counts_run": runnable, "device_counts_skipped": skipped,
            "n_warmup": args.n_warmup, "n_timing": args.n_timing,
            "n_correctness_steps": args.n_correctness_steps,
            "pass_efficiency": args.pass_efficiency,
            "correctness_tol": args.correctness_tol, "ad_tol": args.ad_tol,
            "gate_efficiency": decisive,
        },
        "scaling": per_count,
        "correctness": {"worst_max_abs_diff": worst_corr,
                        "tol": args.correctness_tol, "pass": correctness_pass},
        "ad": {"result": ad_result, "tol": args.ad_tol, "ran": ad_ran,
               "pass": ad_pass},
        "efficiency_gate": {"efficiency_at_largest": eff_at_largest,
                            "n_devices": largest,
                            "threshold": args.pass_efficiency,
                            "enforced": decisive, "pass": efficiency_pass},
        "gates": gates,
        "enforced_gates": enforced,
        "decisive_reasons": reasons,
        "overall_pass": overall_pass,
    }
    return result


# Launcher world-size env vars (MPI first). A degrade is detected if ANY present
# numeric value is >1, so take the MAX — precedence ordering could read a stale
# SLURM_NTASKS=1 and miss an OMPI_COMM_WORLD_SIZE>1 fallback.
_WORLD_SIZE_VARS = ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
                    "SLURM_STEP_NUM_TASKS", "SLURM_NTASKS")


def _launcher_world_size() -> int:
    """Largest world size any launcher env var reports (1 if none present)."""
    sizes = [int(v) for v in (os.environ.get(k) for k in _WORLD_SIZE_VARS)
             if v is not None and v.isdigit()]
    return max(sizes) if sizes else 1


def _multihost_barrier(tag: str) -> None:
    """Synchronise all processes (no-op for a single process)."""
    import jax
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(tag)


def _reduce_ms_max_across_ranks(ms: float) -> float:
    """Return the max ms/step over all processes (identity for one process).

    A collective step finishes when the slowest rank does, so the worst-rank wall
    time is the honest per-step cost; the all-gather doubles as a final barrier.
    """
    import jax
    if jax.process_count() <= 1:
        return ms
    import numpy as np
    from jax.experimental import multihost_utils
    gathered = multihost_utils.process_allgather(np.asarray(ms, dtype=np.float64))
    return float(np.max(np.asarray(gathered)))


def run_point(args) -> dict:
    """Multi-controller TIMING-POINT mode: measure ms/step at the GLOBAL device
    count of THIS launch — one point of the cross-node scaling curve.

    Under ``mpiexec -n N`` (one process per GPU) the processes federate via
    ``initialize_jax_distributed_multiprocess`` (called in :func:`main` before any
    device query), so ``jax.device_count()`` is the GLOBAL count N; a single
    process (no launcher) degrades to single-controller at the local device count.
    Correctness + AD are single-controller concerns validated by :func:`run` and
    the emulated-device tests — NOT re-checked here (a cross-process single-device
    reference would need a separate launch). The per-N points are combined + gated
    by ``scripts/bench/aggregate_cube_shardmap_scaling.py``.
    """
    import jax

    n_devices = jax.device_count()          # GLOBAL count under multi-controller
    proc_count = jax.process_count()
    proc_index = jax.process_index()

    # Detect a FAILED federation loudly: if the launcher reports >1 MPI rank but
    # jax.distributed did not federate (proc_count==1), every rank would run
    # single-controller and silently record n_devices=1 — a wasted sweep the
    # aggregator only catches later as a duplicate-count error.
    expected = _launcher_world_size()
    if expected > 1 and proc_count == 1:
        raise RuntimeError(
            f"launched under {expected} MPI ranks but jax.process_count()==1 — "
            "distributed init did not federate (mpi4py missing, or the launcher "
            f"exports none of {_WORLD_SIZE_VARS}). Fix the launcher/env before "
            "the sweep.")

    dt = float(args.dt)
    model, state_fv3, _cdgrid, _ = _build_cube_model(args.n_grid, args.n_lev, dt)

    # Align all ranks before the measurement so no rank races ahead into timing
    # (a no-op for a single process).
    _multihost_barrier("blk1_point_start")
    ms, _final, hlo = _run_at_count(
        model, state_fv3, dt, n_devices,
        args.n_warmup, args.n_timing, n_steps_final=0, gather_final=False,
        devices=jax.devices()[:n_devices])
    # A collective step is only finished when the SLOWEST rank finishes, so the
    # representative wall time is the max over ranks (the all-gather also acts as
    # the final barrier before rank 0 writes/exits). No-op for one process.
    ms = _reduce_ms_max_across_ranks(ms)
    logger.info("POINT n_devices=%d (procs=%d) ms/step=%.3f hlo_ppermute=%s",
                n_devices, proc_count, ms, hlo)
    return {
        "mode": "point",
        "n_devices": n_devices,
        "process_count": proc_count,
        "process_index": proc_index,
        "ms_per_step": ms,
        "spmd_hlo_collective": hlo,
        "config": {
            "n_grid": args.n_grid, "n_lev": args.n_lev, "dt": dt,
            "precision": args.precision, "backend": jax.default_backend(),
            "n_warmup": args.n_warmup, "n_timing": args.n_timing,
        },
    }


def _write_point_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))


def _write_outputs(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))

    cfg = result["config"]
    lines = ["# Blocker-1 cube shard_map halo — decisive experiment", "",
             f"backend={cfg['backend']} devices_available={cfg['devices_available']} "
             f"precision={cfg['precision']} n_grid={cfg['n_grid']} n_lev={cfg['n_lev']} "
             f"decisive={cfg['gate_efficiency']}"]
    if cfg["device_counts_skipped"]:
        lines.append(f"SKIPPED counts (logged, not dropped): {cfg['device_counts_skipped']}")
    lines += ["", "| n_devices | ms/step | speedup | efficiency | correctness Δ | hlo_coll |",
              "|---|---|---|---|---|---|"]
    for r in result["scaling"]:
        lines.append(f"| {r['n_devices']} | {r['ms_per_step']:.3f} | "
                     f"{r['speedup']:.2f} | {r['efficiency']:.2f} | "
                     f"{r['correctness_max_abs_diff']:.2e} | "
                     f"{r['spmd_hlo_collective']} |")
    ad = result["ad"]["result"]
    if ad and "grad_rel_diff" in ad:
        ad_str = f"rel_diff={ad['grad_rel_diff']:.2e}, collective_in_hlo={ad['collective_in_hlo']}"
    elif ad:
        ad_str = f"ERROR {ad.get('error')}"
    else:
        ad_str = "skipped (single device)"
    lines += ["",
              f"correctness gate: {'PASS' if result['correctness']['pass'] else 'FAIL'} "
              f"(worst Δ={result['correctness']['worst_max_abs_diff']:.2e}, "
              f"tol={result['correctness']['tol']:.0e})",
              f"AD gate: {'PASS' if result['ad']['pass'] else 'FAIL'} ({ad_str})",
              f"efficiency gate: {'PASS' if result['efficiency_gate']['pass'] else 'FAIL'} "
              f"(eff={result['efficiency_gate']['efficiency_at_largest']:.2f} @ "
              f"{result['efficiency_gate']['n_devices']} dev, "
              f"threshold={result['efficiency_gate']['threshold']:.2f}, "
              f"enforced={result['efficiency_gate']['enforced']})"]
    for reason in result["decisive_reasons"]:
        lines.append(f"DECISIVE PRECONDITION FAILED: {reason}")
    lines += ["",
              f"OVERALL: {'PASS' if result['overall_pass'] else 'FAIL'} "
              f"(enforced: {', '.join(result['enforced_gates'])})"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device-counts", default="1,2,3,6",
                   help="Comma list of device counts to sweep (face divisors "
                        "1,2,3,6 or 6*kt^2 tiles with kt | n_grid). "
                        "Unavailable/invalid are logged+skipped.")
    p.add_argument("--n-grid", type=int, default=24, help="Per-face resolution.")
    p.add_argument("--n-lev", type=int, default=8, help="Vertical levels.")
    p.add_argument("--dt", type=float, default=450.0, help="Timestep [s].")
    p.add_argument("--n-warmup", type=int, default=2, help="Warmup steps before timing.")
    p.add_argument("--n-timing", type=int, default=20, help="Timed steps.")
    p.add_argument("--n-correctness-steps", type=int, default=3,
                   help="Steps run to compare SPMD vs single-device final state.")
    p.add_argument("--precision", choices=("float32", "float64"), default="float32")
    p.add_argument("--correctness-tol", dest="correctness_tol", type=float, default=1e-4,
                   help="Max abs SPMD-vs-single-device state diff allowed.")
    p.add_argument("--ad-tol", dest="ad_tol", type=float, default=1e-5,
                   help="Max relative SPMD-vs-single-device gradient diff allowed.")
    p.add_argument("--pass-efficiency", dest="pass_efficiency", type=float, default=0.6,
                   help="Strong-scaling efficiency threshold at the largest count.")
    p.add_argument("--gate-efficiency", dest="gate_efficiency", action="store_true",
                   help="Decisive run: efficiency threshold becomes a hard exit "
                        "gate AND a device count >1 is required.")
    p.add_argument("--output-dir", type=Path, default=Path("results/blocker1_cube_shardmap"))
    p.add_argument("--point", action="store_true",
                   help="Multi-controller TIMING-POINT mode: measure one point at "
                        "this launch's global device count (for the cross-node "
                        "per-N lane). Federates via jax.distributed under mpiexec; "
                        "combine per-N points with aggregate_cube_shardmap_scaling.py.")
    return p


def _init_distributed_if_multiprocess() -> None:
    """Federate ``jax.distributed`` when launched multi-process (no-op otherwise).

    Reuses the repo's ``initialize_jax_distributed_multiprocess`` (the mpi4py-based,
    mpi4jax-free bootstrap used by the cs_spmd SPMD driver): a genuine single
    process returns without importing mpi4py, so the default path is unchanged.
    Must be called BEFORE any device query (a device query initialises the XLA
    backend, after which ``jax.distributed.initialize`` raises).
    """
    from legoesm.parallel.distributed import initialize_jax_distributed_multiprocess
    initialize_jax_distributed_multiprocess()


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    import jax

    # x64 is a config flag (no backend init); safe to set before distributed init.
    if args.precision == "float64":
        jax.config.update("jax_enable_x64", True)

    if args.point:
        _init_distributed_if_multiprocess()  # before any device query
        result = run_point(args)
        # Only the coordinator process writes, so N ranks don't clobber one file.
        if jax.process_index() == 0:
            _write_point_outputs(result, args.output_dir)
            print(json.dumps({"mode": "point", "n_devices": result["n_devices"],
                              "process_count": result["process_count"],
                              "ms_per_step": result["ms_per_step"],
                              "spmd_hlo_collective": result["spmd_hlo_collective"],
                              "output_dir": str(args.output_dir)}, indent=2))
        return 0  # a point never gates; aggregate_cube_shardmap_scaling.py does

    result = run(args)
    _write_outputs(result, args.output_dir)
    print(json.dumps({"overall_pass": result["overall_pass"],
                      "gates": result["gates"],
                      "enforced": result["enforced_gates"],
                      "decisive_reasons": result["decisive_reasons"],
                      "output_dir": str(args.output_dir)}, indent=2))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
