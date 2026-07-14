"""Long-run cosine-bell faithfulness diagnostic (issue #521 / FV3 cube).

Reuses the PRODUCTION cube transport path (streamfunction divergence-free mass
flux, fv_tp_2d.streamfunction_transport_step, hord=10, n_sub=6) exactly as
``run_cosine_bell`` does, but integrates MANY revolutions to expose long-term
distortion (amplitude decay, phase drift) and any slow edge-artifact growth.

Two probes:
  (A) bell  : cosine bell at alpha=pi/4 (hardest, corner-crossing). Per-rev L2/L1/
              Linf vs IC (exact returns to IC each 12-day rev), peak height (decay),
              min height (undershoot), and great-circle phase drift of the mass
              centroid vs the exact bell centre.
  (B) free  : constant h=1 advected by the SAME winds. max|h-1| over time = the
              GCL / edge-artifact probe (must stay ~machine-zero for ALL time;
              the streamfunction flux is divergence-free by construction).

Throwaway (scripts/tmp/). Run:
  JAX_ENABLE_X64=1 .venv/bin/python scripts/tmp/diag_cosine_bell_longrun.py --revs 10 --n 48
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _centroid_lonlat(h, lon, lat, area):
    """Mass-weighted centroid (lon,lat) of a positive field on the sphere."""
    w = jnp.maximum(h, 0.0) * area
    x = jnp.sum(w * jnp.cos(lat) * jnp.cos(lon))
    y = jnp.sum(w * jnp.cos(lat) * jnp.sin(lon))
    z = jnp.sum(w * jnp.sin(lat))
    norm = jnp.sqrt(x * x + y * y + z * z)
    return jnp.arctan2(y, x), jnp.arcsin(z / norm)


def _gc_dist_deg(lon1, lat1, lon2, lat2):
    d = jnp.arccos(jnp.clip(
        jnp.sin(lat1) * jnp.sin(lat2)
        + jnp.cos(lat1) * jnp.cos(lat2) * jnp.cos(lon1 - lon2), -1.0, 1.0))
    return float(jnp.degrees(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=48)        # C<n>
    p.add_argument("--revs", type=int, default=10)     # number of 12-day revolutions
    p.add_argument("--alpha", type=float, default=float(jnp.pi / 4))
    args = p.parse_args()

    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel, iter1009_dual_target_config)
    from legoesm.core.fv_tp_2d import (
        streamfunction_mass_fluxes, streamfunction_transport_step)
    from tests.test_cases.cosine_bell import (
        cosine_bell_cubesphere, cosine_bell_error_norms, rotation_streamfunction)

    n = args.n
    beta = args.alpha
    grid = create_cubed_sphere(n, use_duogrid=False)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    config = iter1009_dual_target_config(n)
    model = FV3EdgeShallowWaterModel(grid, config)
    cdgrid = model.cdgrid

    dt = 1800.0
    N_SUB = 6
    steps_per_day = int(86400 / dt)            # 48
    steps_per_rev = steps_per_day * 12         # 576

    psi = rotation_streamfunction(cdgrid.lon_corner, cdgrid.lat_corner,
                                  grid.radius, beta)
    lon_c, lat_c = grid.lon, grid.lat
    area = grid.area

    def make_step(mass_target):
        @jax.jit
        def step_fn(h):
            dt_sub = dt / N_SUB
            fluxes = streamfunction_mass_fluxes(cdgrid, psi, dt_sub)

            def body(hh, _):
                return streamfunction_transport_step(
                    hh, fluxes, cdgrid, mass_target=mass_target,
                    hord=10, apply_fortran_xppm_boundary=True), None
            h_new, _ = jax.lax.scan(body, h, None, length=N_SUB)
            return h_new
        return step_fn

    # ---- (A) bell ----
    state = cosine_bell_cubesphere(grid, cdgrid, beta)
    h0 = state.h
    mass0 = float(jnp.sum(h0 * area))
    step_bell = make_step(mass0)

    # ---- (B) free-stream h=1 ----
    h1 = jnp.ones_like(h0)
    mass1 = float(jnp.sum(h1 * area))
    step_free = make_step(mass1)

    lon0_c, lat0_c = _centroid_lonlat(h0, lon_c, lat_c, area)
    print(f"# C{n}  alpha={beta:.4f}  revs={args.revs}  "
          f"dt={dt}  n_sub={N_SUB}  steps/rev={steps_per_rev}")
    print(f"# IC: peak={float(jnp.max(h0)):.3f}  mass={mass0:.6e}  "
          f"centroid=({float(jnp.degrees(lon0_c))%360:.2f}E,{float(jnp.degrees(lat0_c)):.2f}N)")
    print(f"{'rev':>4} {'day':>5} {'L2':>9} {'L1':>9} {'Linf':>9} "
          f"{'peak':>8} {'min':>9} {'drift_deg':>9} {'mass_err':>10} "
          f"{'free|h-1|':>10}")

    hb = h0
    hf = h1
    for rev in range(1, args.revs + 1):
        for _ in range(steps_per_rev):
            hb = step_bell(hb)
            hf = step_free(hf)
        hb.block_until_ready()
        norms = cosine_bell_error_norms(hb, h0, area)
        lonc, latc = _centroid_lonlat(hb, lon_c, lat_c, area)
        drift = _gc_dist_deg(lonc, latc, lon0_c, lat0_c)
        peak = float(jnp.max(hb))
        hmin = float(jnp.min(hb))
        mass_err = float(jnp.sum(hb * area) / mass0 - 1.0)
        free_err = float(jnp.max(jnp.abs(hf - 1.0)))
        print(f"{rev:>4} {rev*12:>5} {norms['l2']:>9.4f} {norms['l1']:>9.4f} "
              f"{norms['linf']:>9.4f} {peak:>8.3f} {hmin:>9.2e} {drift:>9.3f} "
              f"{mass_err:>10.2e} {free_err:>10.2e}")

    # visual dump: final bell + error vs IC
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scripts.matrix.run_atmosphere_test_matrix import _get_cs_weights  # noqa
    except Exception:
        pass
    # localization of the free-stream residual: is the 1.5e-6 at seams/corners
    # (an edge artifact) or uniform roundoff? compare boundary-ring vs interior.
    res = np.abs(np.asarray(hf) - 1.0)               # (6,n,n)
    ring = np.zeros((n, n), dtype=bool)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = True
    edge_max = float(res[:, ring].max())
    int_max = float(res[:, ~ring].max())
    corner = float(max(res[:, 0, 0].max(), res[:, 0, -1].max(),
                       res[:, -1, 0].max(), res[:, -1, -1].max()))
    print(f"# free-stream residual loc: edge-ring max={edge_max:.2e}  "
          f"interior max={int_max:.2e}  face-corner max={corner:.2e}  "
          f"(ratio edge/int={edge_max/max(int_max,1e-30):.2f})")
    np.save("/tmp/cb_final.npy", np.asarray(hb))
    np.save("/tmp/cb_init.npy", np.asarray(h0))
    print("# saved /tmp/cb_{final,init}.npy")


if __name__ == "__main__":
    main()
