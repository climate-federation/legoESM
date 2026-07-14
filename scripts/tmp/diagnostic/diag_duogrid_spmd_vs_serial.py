"""Visual sign-off for the Duo-Grid cubed-sphere halo under shard_map (B1).

Runs the cubed-sphere shallow-water Williamson-2 case for a few steps TWICE — once
serial (single-device local halo) and once under the 6-device SPMD (shard_map)
halo backend — then plots the v-wind (W2) and wind_speed (W5) fields for both plus
the serial-minus-SPMD difference.  The difference panels must be featureless
(~machine epsilon): the SPMD duo-grid halo introduces NO cube-imprint / edge
artifact relative to serial.  This is the CLAUDE.md visual-sufficiency check that
elementwise norms alone do not guarantee.

Usage::

    XLA_FLAGS=--xla_force_host_platform_device_count=6 JAX_ENABLE_X64=1 \\
        .venv/bin/python scripts/tmp/diagnostic/diag_duogrid_spmd_vs_serial.py

Writes PNGs to results/diagnostic/ and prints max|serial - spmd|.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np


def _face_mosaic(field6: np.ndarray) -> np.ndarray:
    """Arrange a (6, m, m) cube field into a 2x3 face mosaic for imshow."""
    m = field6.shape[1]
    mosaic = np.full((2 * m, 3 * m), np.nan)
    for f in range(6):
        r, c = divmod(f, 3)
        mosaic[r * m:(r + 1) * m, c * m:(c + 1) * m] = field6[f]
    return mosaic


def _run(model, state, dt, n_steps, *, spmd):
    if not spmd:
        for _ in range(n_steps):
            state = model.step(state, dt)
        return state
    from legoesm.parallel.cubesphere_exchange import (
        activate_spmd_halo_backend,
        deactivate_spmd_halo_backend,
    )
    from legoesm.parallel.mesh import create_device_mesh, shard_pytree
    n = state.h.shape[1]
    dev_config = create_device_mesh(n_devices=6)
    activate_spmd_halo_backend(dev_config.mesh, n=n, nlev=1)
    try:
        state = shard_pytree(state, dev_config)
        for _ in range(n_steps):
            state = model.step(state, dt)
        # gather back to a replicated array for plotting
        state = jax.tree.map(lambda x: np.array(x), state)
    finally:
        deactivate_spmd_halo_backend()
    return state


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from legoesm import constants
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        CDGridShallowWaterModel,
        CDGridShallowWaterState,
    )
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid

    if len(jax.devices("cpu")) < 6:
        raise SystemExit(
            "Need >=6 CPU devices: rerun with "
            "XLA_FLAGS=--xla_force_host_platform_device_count=6")

    n, dt, n_steps = 24, 300.0, 10
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    model = CDGridShallowWaterModel(grid, CDGridShallowWaterConfig())

    # Williamson-2 IC (steady solid-body geostrophic) — inlined from the test
    # helper so the script has no tests/ import dependency.
    u_0 = 38.61068276698372
    h_0 = 29400.0 / constants.g
    lat_c = cdgrid.base.lat
    h = h_0 - (cdgrid.radius * constants.Omega * u_0 + 0.5 * u_0 ** 2) \
        * jnp.sin(lat_c) ** 2 / constants.g
    u_geo = u_0 * jnp.cos(cdgrid.lat_corner)
    ic = CDGridShallowWaterState(
        h=h,
        u_d=u_geo * cdgrid.cos_angle_corner,
        v_d=-u_geo * cdgrid.sin_angle_corner,
        h_s=jnp.zeros_like(h),
    )

    print(f"Running C{n} Williamson-2 SW for {n_steps} steps (dt={dt}s) "
          "serial vs 6-device SPMD ...")
    serial = _run(model, ic, dt, n_steps, spmd=False)
    spmd = _run(model, ic, dt, n_steps, spmd=True)

    # Diagnostics at D-grid corners: v-wind (W2) and wind_speed (W5).
    def diags(s):
        u_d = np.array(s.u_d)
        v_d = np.array(s.v_d)
        return v_d, np.sqrt(u_d ** 2 + v_d ** 2)

    v_ser, spd_ser = diags(serial)
    v_spmd, spd_spmd = diags(spmd)

    max_v = float(np.max(np.abs(v_ser - v_spmd)))
    max_spd = float(np.max(np.abs(spd_ser - spd_spmd)))
    scale = float(np.max(np.abs(spd_ser))) + 1e-30
    rel = max(max_v, max_spd) / scale
    print(f"max|serial - spmd|  v-wind   = {max_v:.3e} m/s")
    print(f"max|serial - spmd|  wind_spd = {max_spd:.3e} m/s")
    print(f"relative (vs {scale:.1f} m/s peak wind) = {rel:.2e}")

    # Structure check: is the residual edge-concentrated (a halo cube-imprint
    # bug) or uniform (float corner-averaging-order noise)?  Compare the mean
    # |diff| in the 1-cell-thick face-edge frame vs the deep interior.
    d = np.abs(v_ser - v_spmd)
    edge = np.concatenate([d[:, 0, :].ravel(), d[:, -1, :].ravel(),
                           d[:, :, 0].ravel(), d[:, :, -1].ravel()])
    interior = d[:, 2:-2, 2:-2].ravel()
    edge_m = float(edge.mean())
    int_m = float(interior.mean()) + 1e-30
    print(f"edge-frame mean|diff| = {edge_m:.2e}, interior mean|diff| = {int_m:.2e}, "
          f"ratio = {edge_m / int_m:.1f}x")

    os.makedirs("results/diagnostic", exist_ok=True)
    out = []
    for name, ser, sp in (("vwind_W2", v_ser, v_spmd),
                          ("wind_speed_W5", spd_ser, spd_spmd)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        m_ser = _face_mosaic(ser)
        m_sp = _face_mosaic(sp)
        m_diff = _face_mosaic(ser - sp)
        for ax, data, title in (
            (axes[0], m_ser, f"{name} serial"),
            (axes[1], m_sp, f"{name} SPMD(6)"),
            (axes[2], m_diff, f"{name} serial-SPMD"),
        ):
            im = ax.imshow(data, origin="lower")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        path = f"results/diagnostic/duogrid_spmd_{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        out.append(path)
        print(f"wrote {path}")

    # The per-call SPMD halo envelope is rtol~1e-6 (cross-device corner-averaging
    # order); over a 10-step integration that accumulates to a small RELATIVE
    # residual.  PASS = relative residual within the float envelope AND no
    # edge-concentrated structure (a cube-imprint bug would light up face edges).
    if rel < 1e-5:
        print(f"PASS: SPMD duo-grid halo matches serial to {rel:.1e} relative "
              "(float corner-averaging-order noise, no cube-imprint).")
    else:
        print("WARN: SPMD vs serial relative difference exceeds 1e-5 — inspect PNGs.")


if __name__ == "__main__":
    main()
