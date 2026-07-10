"""Parity gate for the PERSISTENT lat-band-sharded OMIP ocean loop
(scaling-M2 increment 1, ``run_omip_core2 --spmd-persistent-state``).

The production multi-GPU OMIP host loop used to call
``make_sharded_ocean_step_global`` EVERY step — a full-state scatter
(``shard_state_latlon``) + gather (``gather_state_latlon``) per step, i.e.
2 full-state transfers/step.  The persistent lane instead keeps the state
lat-band SHARDED across steps via ``make_sharded_ocean_step`` and gathers
ONLY at real output boundaries.  This gate asserts, over a 10-step loop with
production-shaped surface + freshwater forcing (``normalize_freshwater=True``
so the in-step psum reduction is load-bearing):

* EQUIVALENCE — the persistent loop's final state equals the per-step
  global-wrapper loop's BIT-FOR-BIT (atol=rtol=0; the wrapper is literally
  scatter->inner->gather and every band stays on its device, so the per-step
  round trip is value-preserving).  Far inside the 1e-12 f64 merge bar.
* the driver's HOST-BC pattern mid-loop — a leaf-wise host update (the SSS-
  restore shape: ``np.asarray`` a cell-centred leaf off the SHARDED state,
  modify, write back an UNSHARDED leaf) — is exact: the addressable sharded
  array assembles to identical host values and the next sharded step
  reshards the mixed-sharding input per its in_specs.
* one mid-loop GATHER-FOR-OUTPUT round-trip (the snapshot boundary: gather,
  host-read the full staggered ``(n_lat+1, ...)`` v, re-shard) is lossless.
* GATHER COUNTS — old wrapper lane: N_STEPS shards + N_STEPS gathers
  (2 full-state transfers per step); persistent lane: 2 + 2 TOTAL
  (initial shard + mid re-shard; mid output gather + final gather),
  independent of step count ⇒ 0 per-step transfers.

Run: ``XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_ENABLE_X64=1
pytest tests/parallel/test_persistent_sharded_ocean_loop.py``
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_THIS = Path(__file__).resolve()


def _load_sibling():
    """Load the calibrated SPMD-step gate module to REUSE its fixture builders
    (``_perturbed_state`` / ``_omip_like_forcing``) — no duplicated fixture
    numerics (same pattern as tests/unit/test_run_omip_latlon_spmd.py)."""
    sib = _THIS.parent / "test_latlon_ocean_spmd_step.py"
    spec = importlib.util.spec_from_file_location("_latlon_spmd_gate_sibling",
                                                  sib)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_latlon_spmd_gate_sibling"] = mod
    spec.loader.exec_module(mod)
    return mod


def _have_sharded_step():
    try:
        from legoesm.ocean.dynamics.sharded_ocean_step import (  # noqa: F401
            make_sharded_ocean_step,
        )
        from legoesm.parallel.mesh import create_latlon_mesh  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(jax.device_count() < 2,
                    reason="needs >=2 devices (XLA_FLAGS host device count)")
@pytest.mark.skipif(not _have_sharded_step(),
                    reason="sharded_ocean_step module not present")
def test_persistent_loop_matches_wrapper_loop_and_gather_counts(monkeypatch):
    sib = _load_sibling()
    import legoesm.ocean.dynamics.sharded_ocean_step as sos
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.parallel.mesh import create_latlon_mesh

    n_lat, n_lon, nlev = 48, 96, 10          # the calibrated sibling-gate size
    dt, n_steps = 600.0, 10
    # DISTINCT steps (codex r1): the host-BC leaf write must land on a step
    # WITHOUT the gather round-trip, so the UNSHARDED written-back leaf feeds
    # the NEXT sharded step directly (the production SSS-restore condition —
    # an immediate re-shard would mask it); the snapshot boundary follows two
    # steps later.
    bc_step = 3                              # leaf-wise host BC write-back
    snap_step = 5                            # gather-for-output round-trip

    grid = create_latlon_grid(n_lat=n_lat, n_lon=n_lon)
    z_coord = create_ocean_z_star(n_levels=nlev, H_max=4000.0)
    cfg = LatLonCGridOceanConfig.from_flat()._replace(
        normalize_freshwater=True)           # in-step psum is load-bearing
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state0 = sib._perturbed_state(grid, z_coord)
    fw, sf, _sponge = sib._omip_like_forcing(grid, z_coord)
    model._ensure_vertex_mask(state0)
    mesh = create_latlon_mesh(n_devices=2).mesh

    # ---- count every full-state transfer through the module entry points ----
    calls = {"shard": 0, "gather": 0}
    _shard0, _gather0 = sos.shard_state_latlon, sos.gather_state_latlon

    def _counting_shard(st, m):
        calls["shard"] += 1
        return _shard0(st, m)

    def _counting_gather(st, m):
        calls["gather"] += 1
        return _gather0(st, m)

    monkeypatch.setattr(sos, "shard_state_latlon", _counting_shard)
    monkeypatch.setattr(sos, "gather_state_latlon", _counting_gather)

    def _bc_touch_S(S_host):
        """The leaf-wise host post-step BC pattern (SSS-restore shape):
        read a cell-centred leaf to host, update the top layer, return the
        new full-leaf host array."""
        S_new = np.asarray(S_host, dtype=np.float64).copy()
        S_new[..., 0] = S_new[..., 0] + 0.01
        return S_new

    # ---------------- OLD lane: per-step global-in/global-out wrapper --------
    glob = sos.make_sharded_ocean_step_global(model, mesh)
    sg = state0
    for k in range(1, n_steps + 1):
        sg = glob(sg, dt, surface_forcing=sf, freshwater=fw)
        if k == bc_step:
            # host BC on the (gathered) global state
            S_new = _bc_touch_S(np.asarray(sg.S.data))
            sg = sg._replace(S=sg.S.replace(data=jnp.asarray(S_new)))
        if k == snap_step:
            # the snapshot host read
            v_out = np.asarray(sg.v.data)
            assert v_out.shape[0] == n_lat + 1   # full staggered v for output
    sg = jax.block_until_ready(sg)
    old_calls = dict(calls)
    # the wrapper lane pays 2 full-state transfers PER STEP
    assert old_calls == {"shard": n_steps, "gather": n_steps}, old_calls

    # ---------------- NEW lane: persistent sharded loop ----------------------
    calls["shard"] = calls["gather"] = 0
    inner = sos.make_sharded_ocean_step(model, mesh)
    ss = sos.shard_state_latlon(state0, mesh)          # ONE initial shard
    for k in range(1, n_steps + 1):
        ss = inner(ss, dt, surface_forcing=sf, freshwater=fw)
        if k == bc_step:
            # host BC leaf-wise ON THE SHARDED STATE (driver semantics:
            # np.asarray assembles the addressable sharded leaf to the
            # identical host values; the write-back leaf is UNSHARDED and the
            # NEXT inner step — with NO intervening re-shard — consumes the
            # mixed-sharding state and reshards it per its in_specs, exactly
            # the per-step production SSS-restore condition).
            S_new = _bc_touch_S(np.asarray(ss.S.data))
            ss = ss._replace(S=ss.S.replace(data=jnp.asarray(S_new)))
        if k == snap_step:
            # the snapshot output boundary: ONE gather-for-output round-trip;
            # the gathered view carries the full staggered v.
            ss = sos.gather_state_latlon(ss, mesh)
            v_out = np.asarray(ss.v.data)
            assert v_out.shape[0] == n_lat + 1
            ss = sos.shard_state_latlon(ss, mesh)      # lazy re-shard
    ss = sos.gather_state_latlon(ss, mesh)             # final output gather
    ss = jax.block_until_ready(ss)
    new_calls = dict(calls)
    # step-count-INDEPENDENT totals: initial shard + mid re-shard, mid output
    # gather + final gather => 0 per-step full-state transfers.
    assert new_calls == {"shard": 2, "gather": 2}, new_calls

    # ---------------- equivalence: bitwise, far inside the 1e-12 bar ---------
    for nm in ("u", "v", "eta", "T", "S"):
        a = np.asarray(getattr(sg, nm).data)
        b = np.asarray(getattr(ss, nm).data)
        print(f"[parity] {nm}: max|persistent - wrapper| = "
              f"{float(np.max(np.abs(b - a))):.3e}")
        np.testing.assert_allclose(
            b, a, atol=0.0, rtol=0.0,
            err_msg=(f"persistent-sharded loop diverged from the per-step "
                     f"global-wrapper loop on {nm}"))
