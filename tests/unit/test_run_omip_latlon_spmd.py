"""Gate the ``run_omip --enable-latlon-spmd`` restoring lane (part 2a of the
ocean-SPMD promotion, after #758's forcing channels).

* ``_run_omip_loop`` with ``spmd_step``/``spmd_gather`` (the lat-band SPMD
  dynamics step over 2 virtual CPU devices) matches the plain
  ``model.step`` loop — including the post-step Haney restoring and the
  restart checkpoint, whose file must carry the FULL ``(n_lat+1, ...)``
  staggered ``v`` (the sharded loop carries ``v_lower``; a save without
  the gather hook writes the wrong shape and poisons every resume).
  The parity case runs in a SUBPROCESS so the 2-virtual-device XLA flag
  always takes effect — an in-process variant silently skips whenever a
  previously-collected test already initialized the single-device JAX
  backend (codex r1 #4).
* The ValueError refusal (JRA lane) fires without any device work.

CLI round-trip for the new flags lives in tests/unit/test_run_omip_cli.py;
the args-level SystemExit refusals (forcing mode, negative device count)
are validated there too via run_omip_single's early guard.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()


def _load_harness():
    day23 = _THIS.parent / "test_run_omip_jra55_dispatch.py"
    spec = importlib.util.spec_from_file_location("_day23_spmd", day23)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_day23_spmd"] = mod
    spec.loader.exec_module(mod)
    return mod


def _parity_worker(tmp_dir: str) -> None:
    """Runs in the SUBPROCESS: SPMD-vs-serial restoring-loop parity."""
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    assert jax.device_count() >= 2, (
        f"worker expected 2 virtual devices, got {jax.device_count()}")

    _day23 = _load_harness()
    run_omip = _day23.run_omip

    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon,
        make_sharded_ocean_step,
        shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    tmp_path = Path(tmp_dir)
    n_steps, dt = 6, 600.0
    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
        n_lat=8, n_lon=16)
    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    T_woa, S_woa = _day23._make_woa_like_targets(grid)
    targets = (T_woa[..., 0], S_woa[..., 0])

    common = dict(
        grid_type="latlon", grid=grid, z_coord=z_coord,
        dt=dt, n_steps=n_steps, diag_every=3,
        restoring_targets=targets, restoring_tau_s=5.0 * 86400.0,
        checkpoint_days=dt * n_steps / 86400.0,   # one save at step n_steps
    )

    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
        model, state0, checkpoint_dir=tmp_path / "serial", **common)
    assert ok_serial

    model.prime_step_caches(state0)
    dev = create_latlon_mesh(n_devices=2)
    spmd_step = make_sharded_ocean_step(model, dev.mesh)
    ss0 = shard_state_latlon(state0, dev.mesh)
    ckpt_dir = tmp_path / "spmd"
    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
        model, ss0, checkpoint_dir=ckpt_dir,
        spmd_step=spmd_step,
        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
        **common)
    assert ok_spmd
    spmd_final = gather_state_latlon(spmd_final, dev.mesh)

    # Same re-association floor as the step-level SPMD gate.
    for nm in ("u", "v", "eta", "T", "S"):
        np.testing.assert_allclose(
            np.asarray(getattr(spmd_final, nm).data),
            np.asarray(getattr(serial_final, nm).data),
            atol=2e-4, rtol=1e-3, err_msg=f"SPMD loop {nm} mismatch")

    # The checkpoint written FROM the sharded loop must carry the full
    # (n_lat+1, n_lon, nlev) staggered v — the save choke point gathers.
    saved = sorted(ckpt_dir.glob("*.npz"))
    assert saved, f"no restart saved under {ckpt_dir}"
    with np.load(saved[-1]) as data:
        v_key = [k for k in data.files if k in ("v", "v_data")]
        assert v_key, f"no v field in restart: {data.files}"
        v_saved = data[v_key[0]]
    assert v_saved.shape[0] == grid.n_lat + 1, (
        f"restart v leading dim {v_saved.shape[0]} != n_lat+1 "
        f"({grid.n_lat + 1}) — spmd save missed the gather hook")
    print("PARITY-OK")


@pytest.mark.timeout(900)
def test_restoring_loop_spmd_matches_serial(tmp_path):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    # A bare `python tests/unit/<file>.py` puts tests/unit on sys.path, not
    # the repo root pytest provides — prepend it so the harness's
    # repo-relative imports resolve without an editable install (codex r2).
    _root = str(_THIS.parents[2])
    env["PYTHONPATH"] = _root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(_THIS), "--parity-worker", str(tmp_path)],
        env=env, capture_output=True, text=True, timeout=870,
    )
    assert proc.returncode == 0 and "PARITY-OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")


def test_loop_refuses_spmd_with_jra55_single_step():
    """The single-step JRA55 fallback (``model.step``) cannot run sharded —
    only the block-scan lanes thread ``spmd_step`` (part 2b)."""
    _day23 = _load_harness()
    run_omip = _day23.run_omip
    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
        n_lat=8, n_lon=16)
    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    with pytest.raises(ValueError, match="single-step fallback is unsupported"):
        run_omip._run_omip_loop(
            model, state0, grid_type="latlon", grid=grid, z_coord=z_coord,
            dt=600.0, n_steps=1, diag_every=1,
            jra55_state={"_use_single_step": True},
            spmd_step=lambda s, d: s)


# ---------------------------------------------------------------------------
# Part 2b: the JRA55 block-scan lanes thread spmd_step through the scan body
# and shard the per-block forcing stacks.  Both the CPU-interp
# (_build_jra55_block_fn) and the GPU-interp (_build_jra55_block_fn_interp,
# the run_omip default) block builders must match the serial block loop to
# the same re-association floor as the restoring lane.  Subprocess again so
# the 2-virtual-device XLA flag always takes effect.
# ---------------------------------------------------------------------------

def _jra55_parity_worker(tmp_dir: str, interp_mode: str) -> None:
    """Runs in the SUBPROCESS: JRA55 block-scan SPMD-vs-serial parity.

    ``interp_mode`` is ``"cpu"`` (host-interp block) or ``"gpu"`` (in-scan
    GPU-interp block).  Uses the SAME shard_forcing_stack_latlon layout the
    driver uses (imported, not re-derived) so the test can't drift from it.
    """
    import jax
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    assert jax.device_count() >= 2, (
        f"worker expected 2 virtual devices, got {jax.device_count()}")

    _day23 = _load_harness()
    run_omip = _day23.run_omip

    from functools import partial

    from legoesm.ocean.dynamics.sharded_ocean_step import (
        gather_state_latlon,
        make_sharded_ocean_step,
        shard_forcing_stack_latlon,
        shard_state_latlon,
    )
    from legoesm.parallel.mesh import create_latlon_mesh

    tmp_path = Path(tmp_dir)
    n_lat, n_lon = 8, 16
    dt, n_steps = 600.0, 6

    cache = _day23._make_synthetic_cache(
        tmp_path, n_lat=n_lat, n_lon=n_lon, n_records=64)
    grid, z_coord, _config, model, _gt = _day23._make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon)
    T_woa, S_woa = _day23._make_woa_like_targets(grid, nlev=4)
    args = _day23._argparse_namespace(jra55_cache=str(cache))

    def _fresh_js():
        # _setup_jra55_forcing_state does NOT set _gpu_interp (the driver
        # does, from --gpu-interp); set it here to pick the lane.  A fresh
        # dict per loop avoids sharing the sponge/target arrays across the
        # serial and sharded runs.
        js = run_omip._setup_jra55_forcing_state(
            args, grid, "latlon", z_coord=z_coord, T_woa=T_woa, S_woa=S_woa)
        js["_gpu_interp"] = (interp_mode == "gpu")
        return js

    common = dict(
        grid_type="latlon", grid=grid, z_coord=z_coord,
        dt=dt, n_steps=n_steps, diag_every=3,
        checkpoint_days=None, checkpoint_dir=None,
    )

    state0 = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    serial_final, _d, _w, ok_serial, _b = run_omip._run_omip_loop(
        model, state0, jra55_state=_fresh_js(), **common)
    assert ok_serial, "serial JRA55 loop reported not-ok"

    dev = create_latlon_mesh(n_devices=2)
    spmd_step = make_sharded_ocean_step(model, dev.mesh)
    model.prime_step_caches(state0)
    ss0 = shard_state_latlon(state0, dev.mesh)
    spmd_final, _d2, _w2, ok_spmd, _b2 = run_omip._run_omip_loop(
        model, ss0, jra55_state=_fresh_js(),
        spmd_step=spmd_step,
        spmd_gather=lambda st: gather_state_latlon(st, dev.mesh),
        spmd_shard_stack=partial(shard_forcing_stack_latlon, mesh=dev.mesh),
        **common)
    assert ok_spmd, "SPMD JRA55 loop reported not-ok"
    spmd_final = gather_state_latlon(spmd_final, dev.mesh)

    # Same re-association floor as the restoring-lane and step-level gates.
    for nm in ("u", "v", "eta", "T", "S"):
        np.testing.assert_allclose(
            np.asarray(getattr(spmd_final, nm).data),
            np.asarray(getattr(serial_final, nm).data),
            atol=2e-4, rtol=1e-3,
            err_msg=f"JRA55 {interp_mode}-interp SPMD {nm} mismatch")
    print("JRA55-PARITY-OK")


def _run_jra55_parity_subprocess(tmp_path, interp_mode: str):
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    _root = str(_THIS.parents[2])
    env["PYTHONPATH"] = _root + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, str(_THIS), "--jra55-parity-worker",
         str(tmp_path), interp_mode],
        env=env, capture_output=True, text=True, timeout=870,
    )
    assert proc.returncode == 0 and "JRA55-PARITY-OK" in proc.stdout, (
        f"rc={proc.returncode}\nstdout:\n{proc.stdout[-3000:]}\n"
        f"stderr:\n{proc.stderr[-3000:]}")


@pytest.mark.timeout(900)
def test_jra55_block_loop_spmd_matches_serial_cpu_interp(tmp_path):
    _run_jra55_parity_subprocess(tmp_path, "cpu")


@pytest.mark.timeout(900)
def test_jra55_block_loop_spmd_matches_serial_gpu_interp(tmp_path):
    _run_jra55_parity_subprocess(tmp_path, "gpu")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--parity-worker":
        _parity_worker(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--jra55-parity-worker":
        _jra55_parity_worker(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit("usage: test_run_omip_latlon_spmd.py "
                         "--parity-worker <tmp_dir> | "
                         "--jra55-parity-worker <tmp_dir> <cpu|gpu>")
