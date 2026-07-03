"""Compare legoESM dst3_multidim tracer advection to MITgcm tutorial_advection_in_gyre.

Isolates the PASSIVE-TRACER ADVECTION SCHEME: prescribes MITgcm's equilibrated
(steady) gyre velocity into a legoESM beta-plane C-grid state, advects the dye
blob ``N_STEPS`` steps with the CANONICAL legoESM advection operator
(``_compute_advection_flux_div`` dispatched on ``tracer_advection`` from the
recipe), and compares the resulting dye field to MITgcm's ``PTRACER01`` dumps.

This deliberately removes any legoESM-from-rest vs MITgcm-from-pickup dynamical
difference: BOTH models see the identical velocity field, so a difference in the
advected dye is a difference in the ADVECTION SCHEME (MITgcm advScheme=80 vs
legoESM dst3_multidim) only.

MITgcm reference is the tiled-or-global single-CPU binary dumps in the run dir
(``PTRACER01.*.data``, ``U.*.data``, ``V.*.data``), read as big-endian float32.

Run (CPU, fp64):
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/validate/ocean_fidelity/compare_mitgcm_advection_gyre.py \
        --run-dir /tmp/mitgcm_advgyre/run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _rd_global(path: Path, ny: int, nx: int) -> np.ndarray:
    """Read a MITgcm single-CPU global binary dump as (ny, nx) big-endian f32."""
    a = np.fromfile(path, dtype=">f4")
    if a.size != ny * nx:
        raise ValueError(f"{path}: expected {ny*nx} values, got {a.size}")
    # MITgcm global file: x (i) fastest -> reshape (ny, nx), row index = y.
    return a.reshape(ny, nx).astype(np.float64)


def _pattern_correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    av = a[mask > 0.5].ravel()
    bv = b[mask > 0.5].ravel()
    av = av - av.mean()
    bv = bv - bv.mean()
    denom = np.sqrt((av * av).sum() * (bv * bv).sum())
    if denom == 0.0:
        return float("nan")
    return float((av * bv).sum() / denom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iter0", type=int, default=259200)
    args = ap.parse_args()

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface,
        min_cell_to_vface,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _compute_advection_flux_div,
    )
    from legoesm.ocean.fidelity.mitgcm_advection_gyre_recipe import (
        DT_S,
        H_DEPTH_M,
        N_STEPS,
        NX,
        NY,
        build_advgyre_config,
        build_advgyre_geometry,
        build_advgyre_land_mask,
        build_dye_field,
    )

    run = Path(args.run_dir)
    it0 = args.iter0

    geom = build_advgyre_geometry()
    grid = geom.grid if hasattr(geom, "grid") else geom
    mask2d = np.asarray(build_advgyre_land_mask())            # (NY, NX) wet=1
    cfg = build_advgyre_config()
    scheme = cfg.tracer_advection

    # --- MITgcm equilibrated velocity (steady over the 4 dye steps) ---
    # MITgcm U[y,x] is the WEST face of cell (x); legoESM u-face[y,x] (x in
    # 0..NX) is also the west face of cell x -> direct column map; the extra
    # legoESM u-face column x=NX is the periodic wrap (masked: wall at x=NX-1).
    u_mit = _rd_global(run / f"U.{it0:010d}.data", NY, NX)   # (NY, NX)
    v_mit = _rd_global(run / f"V.{it0:010d}.data", NY, NX)   # (NY, NX)

    u_face = np.zeros((NY, NX + 1), dtype=np.float64)
    u_face[:, :NX] = u_mit
    u_face[:, NX] = u_mit[:, 0]                               # periodic wrap
    v_face = np.zeros((NY + 1, NX), dtype=np.float64)
    v_face[:NX, :] = v_mit[:NY, :]                            # south faces of cells
    # v-face row NY (north of last cell) is a wall -> 0.

    # --- Face masks (zero flux through walls) ---
    # u-face j is the west face of cell j: open iff both cell j-1 and cell j wet.
    u_mask = np.zeros((NY, NX + 1), dtype=np.float64)
    for j in range(1, NX):
        u_mask[:, j] = mask2d[:, j - 1] * mask2d[:, j]
    # face 0 and face NX touch the domain edge (open cell j=0 has no west
    # neighbour; MITgcm walls the west boundary via the no-normal-flow BC) -> 0.
    v_mask = np.zeros((NY + 1, NX), dtype=np.float64)
    for i in range(1, NY):
        v_mask[i, :] = mask2d[i - 1, :] * mask2d[i, :]

    # --- Thickness-weighted mass fluxes (single layer, H = const) ---
    h_k = jnp.full((NY, NX, 1), H_DEPTH_M)
    h_u_old = min_cell_to_uface(h_k)
    h_v_old = min_cell_to_vface(h_k, grid)
    mass_flux_u = jnp.asarray(u_face[:, :, None]) * h_u_old * jnp.asarray(u_mask[:, :, None])
    mass_flux_v = jnp.asarray(v_face[:, :, None]) * h_v_old * jnp.asarray(v_mask[:, :, None])
    w_baro = jnp.zeros((NY, NX, 2))                          # nlev+1; no vertical flow
    active = jnp.asarray(mask2d[:, :, None])

    # --- Advect the dye N_STEPS forward-Euler flux-form steps ---
    dye = build_dye_field()                                  # (NY, NX, 1)
    dt = DT_S
    dye_steps = [np.asarray(dye)[:, :, 0]]
    for _ in range(N_STEPS):
        div_h, vert = _compute_advection_flux_div(
            dye, scheme, mass_flux_u, mass_flux_v, w_baro,
            h_k, h_u_old, h_v_old, grid, dt,
        )
        hT_new = h_k * dye - dt * (div_h + vert)  # noqa: N806 (h*T thickness-weighted tracer, model convention)
        dye_new = hT_new / jnp.maximum(h_k, 1e-10)
        dye = jnp.where(active > 0.5, dye_new, dye)
        dye_steps.append(np.asarray(dye)[:, :, 0])

    # --- Compare to MITgcm PTRACER dumps ---
    print(f"scheme={scheme}  grid={NY}x{NX}  H={H_DEPTH_M}  dt={dt}  steps={N_STEPS}")
    print(f"MITgcm |u|max={np.abs(u_mit).max():.5f}  |v|max={np.abs(v_mit).max():.5f}")
    print()
    header = f"{'step':>4} {'corr':>9} {'lego_sum':>10} {'mit_sum':>10} "
    header += f"{'lego_max':>9} {'mit_max':>9} {'max_ratio':>9} {'L2_rel':>9}"
    print(header)
    final = {}
    for k in range(N_STEPS + 1):
        it = it0 + k
        mit = _rd_global(run / f"PTRACER01.{it:010d}.data", NY, NX)
        lego = dye_steps[k]
        corr = _pattern_correlation(lego, mit, mask2d)
        ls, ms = lego.sum(), mit.sum()
        lmax, mmax = lego.max(), mit.max()
        ratio = lmax / mmax if mmax else float("nan")
        diff = lego[mask2d > 0.5] - mit[mask2d > 0.5]
        l2 = np.sqrt((diff * diff).sum()) / np.sqrt((mit[mask2d > 0.5] ** 2).sum())
        print(f"{k:>4} {corr:>9.5f} {ls:>10.6f} {ms:>10.6f} "
              f"{lmax:>9.5f} {mmax:>9.5f} {ratio:>9.4f} {l2:>9.5f}")
        final = dict(step=k, corr=corr, lego_sum=ls, mit_sum=ms,
                     max_ratio=ratio, l2_rel=l2)

    print()
    print("FINAL STEP:", {k: (round(v, 5) if isinstance(v, float) else v)
                          for k, v in final.items()})


if __name__ == "__main__":
    main()
