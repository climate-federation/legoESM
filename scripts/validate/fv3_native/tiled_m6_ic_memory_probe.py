"""Peak-RSS after each sub-stage of the window model's initial state, to
name the stage whose memory grows with the global grid on EVERY rank
(216-rank C192 OOM at 24.6 GB/rank, job 9648703; the IC stage was
+0.4 GB at C48 kt=2 and +1.9 GB at C96 kt=3).

    XLA_FLAGS=--xla_force_host_platform_device_count=6*kt*kt \
    python tiled_m6_ic_memory_probe.py N KT
"""
import resource
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

jax.config.update("jax_enable_x64", True)


def rss(stage, t0):
    mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[icmem] {stage:34s} peak {mb:7.0f} MB  t={time.perf_counter() - t0:6.1f}s",
          flush=True)


def main(n, kt):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.core.fv3_cgrid_phase_3d import state_3d_to_jax
    from legoesm.core.fv3_dynamics import p_var_hydrostatic
    from legoesm.core.fv3_native_dcmip16_ic import dcmip16_bc_six_face_state
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA
    from legoesm.grids.factory import create_fv3_duo_grid
    t0 = time.perf_counter()
    rss("start", t0)
    grid = create_fv3_duo_grid(n)
    rss("grid", t0)
    mesh = Mesh(np.array(jax.devices()[:6 * kt * kt]).reshape(6, kt, kt),
                ("face", "tile_i", "tile_j"))
    cfg = FV3DuoConfig(km=10, hydrostatic=True, n_split=3)
    m = FV3DuoDynamicsModel(grid, cfg, step_spmd_mesh=mesh,
                            step_windows=(kt, 11))
    rss("window model built", t0)
    st6, sphum6 = dcmip16_bc_six_face_state(
        grid.ctx_np, m._ak, m._bk, cfg.km, hydrostatic=True, do_pert=True)
    rss("dcmip16 numpy state", t0)
    jstate = state_3d_to_jax(st6)
    jax.block_until_ready(jstate)
    rss("state_3d_to_jax", t0)
    press = p_var_hydrostatic(jstate["delp"], ptop=m._ptop, akap=FV3_KAPPA,
                              n=grid.n, ng=grid.ng, km=cfg.km)
    jax.block_until_ready(press)
    rss("p_var_hydrostatic", t0)
    q = [jnp.asarray(np.stack(sphum6))]
    omga = jnp.zeros_like(jstate["delp"])
    bundle = {"state": jstate, "press": press, "q": q, "omga": omga,
              "nh": None}
    from legoesm.grids.fv3_duo_windows import gather_windows, horizontal_axes
    lay = m.window_layout
    hosted = jax.tree_util.tree_map(
        lambda a: gather_windows(lay, np.asarray(a), np)
        if hasattr(a, "ndim") and horizontal_axes(lay, a.shape, 6) is not None
        else a, bundle)
    rss("host gather_windows", t0)
    win = jax.tree_util.tree_map(
        lambda a: jax.device_put(a, m._window_sharding)
        if isinstance(a, np.ndarray) else a, hosted)
    jax.block_until_ready(win)
    rss("device_put windows", t0)
    win2 = m.dcmip16_initial_state(do_pert=True)
    jax.block_until_ready(win2)
    rss("model.dcmip16_initial_state", t0)
    print("DONE icmem")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
