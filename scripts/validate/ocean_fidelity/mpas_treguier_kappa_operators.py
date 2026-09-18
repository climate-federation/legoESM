"""How much does the slope OPERATOR change the MPAS Treguier coefficient?

The MPAS lane can now run the Treguier adaptive GM coefficient, but there is a
real choice in how the isopycnal slope reaches it, and the choice is not
cosmetic.  An edge on an unstructured mesh knows only its own normal
component, so the existing slope routine clips and tapers EACH COMPONENT
separately.  The structured lane instead limits the slope MAGNITUDE.  Those
are different operators: per-component is not rotationally invariant, and a
vector at 45 degrees to the edge set survives at sqrt(2) times the limit.

Two reviewers reached that independently, and the shipped branch therefore
reconstructs the cell-centred slope VECTOR (wet-edge Perot) and limits its
magnitude.  This probe measures what that decision is worth on a REAL ocean
state rather than on an idealised one — the taper only bites where slopes are
steep, which is exactly where an idealised test has nothing to say.

It compares, per column, the Treguier kappa built from:

  (a) PER-COMPONENT  — clip+taper each edge normal, then reconstruct.  The
      operator the lane would have used for free.
  (b) VECTOR         — reconstruct the raw slope on the wet edges, then clip
      and taper the magnitude.  What ships.
  (c) EDGE-WISE |S|^2 — the Perot INNER PRODUCT of the per-component tapered
      slopes, i.e. <|S|^2> rather than |<S>|^2.  This is what the Redi
      vertical flux in the same module already uses, and it is the other end
      of the Jensen inequality: |<S>|^2 <= <|S|^2>, so (b) is biased LOW in
      fronts relative to (c).

and reports the ratios binned by how much of the column is tapered and by the
cell's Perot anisotropy.  No coupled run: a free-running comparison of two
lanes is confounded by trajectory divergence within days, which is precisely
why this is done offline from one saved state.

Usage (needs a compute node — it builds a level-7 mesh):
    python scripts/validate/ocean_fidelity/mpas_treguier_kappa_operators.py \
        --restart <path/to/restart.npz> --level 7 --lloyd 20
"""
from __future__ import annotations

import argparse


def main() -> int:
    import numpy as np

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--restart", required=True)
    p.add_argument("--level", type=int, default=7)
    p.add_argument("--lloyd", type=int, default=20)
    p.add_argument("--s-max", type=float, default=1.0e-2)
    p.add_argument("--aei0", type=float, default=900.0)
    p.add_argument("--map", default=None,
                   help="write a PNG of the Jensen gap to this path")
    args = p.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm import constants
    from legoesm.grids.voronoi import (
        create_voronoi_mesh,
        reconstruct_cell_velocity,
        reconstruct_cell_velocity_wet,
    )
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    from legoesm.ocean.eos import make_eos_fn, rho_0 as RHO_0
    from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
        compute_treguier_kappa_gm,
        dm95_taper,
        dm95_taper_scalar,
    )
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig, TreguierConfig,
    )
    from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
        _perot_inner_product_cell,
        compute_isopycnal_slopes_mpas,
        voronoi_neumann_fill,
    )
    from legoesm.ocean.vertical import compute_ocean_jacobian, create_ocean_z_star

    z = np.load(args.restart, allow_pickle=True)
    T, S = jnp.asarray(z["T"]), jnp.asarray(z["S"])
    eta, H = jnp.asarray(z["eta"]), jnp.asarray(z["H_bathy"])
    mask = jnp.asarray(np.asarray(z["land_mask"], dtype=np.float64))
    nlev = int(T.shape[1])
    print(f"[state] {args.restart}")
    print(f"[state] step={int(z['_step'])} nCells={T.shape[0]} nlev={nlev}")

    mesh = create_voronoi_mesh(subdivision_level=args.level,
                               lloyd_iterations=args.lloyd)
    assert int(mesh.nCells) == int(T.shape[0]), (
        f"mesh nCells {mesh.nCells} != restart {T.shape[0]}; wrong "
        "--level/--lloyd for this state")

    z_coord = create_ocean_z_star(n_levels=nlev, H_max=float(np.max(np.asarray(H))))
    jac = compute_ocean_jacobian(eta, H, z_coord)
    cfg = GMRediConfig(
        kappa_GM=1e3, kappa_Redi=1e3, S_max=args.s_max,
        slope_scheme="centered",
        treguier=TreguierConfig(enabled=True, aei0=args.aei0),
    )

    rho, _, _ = iterate_eos_and_pressure_anomaly(
        T, S, mask, lambda f: voronoi_neumann_fill(f, mask, mesh),
        make_eos_fn("wright", None), z_coord.dz_ref, RHO_0, constants.g,
        n_iter=2)

    S_n, taper_c, S_n_raw = compute_isopycnal_slopes_mpas(
        rho, mask, z_coord, jac, mesh, cfg, return_raw=True)

    c1, c2 = mesh.cellsOnEdge[0], mesh.cellsOnEdge[1]
    edge_mask = mask[c1] * mask[c2]
    em = edge_mask[:, None]

    f = 2.0 * constants.Omega * jnp.sin(mesh.latCell)
    ia = None
    if hasattr(z_coord, "is_active"):
        act = z_coord.is_active
        ia = (act[:, :-1] & act[:, 1:]).astype(rho.dtype)

    def kappa(sx, sy):
        return np.asarray(compute_treguier_kappa_gm(
            rho, sx, sy, z_coord, jac, f, cfg.treguier,
            rho_ref=RHO_0, g=constants.g, omega=constants.Omega,
            interface_active=ia))

    # (a) per-component: taper each normal, then reconstruct
    ax, ay = reconstruct_cell_velocity(S_n * em, mesh)
    k_component = kappa(ax, ay)

    # (b) vector: wet-edge reconstruction of the RAW slope, magnitude-limited
    bx, by = reconstruct_cell_velocity_wet(S_n_raw, mesh, em)
    mag = jnp.sqrt(bx ** 2 + by ** 2 + 1e-30)
    lim = jnp.minimum(1.0, cfg.S_max / mag)
    bx, by = bx * lim, by * lim
    bx, by, _ = dm95_taper(bx, by, cfg.S_max,
                           transition_width_frac=cfg.taper_width_frac)
    k_vector = kappa(bx, by)

    # (c) edge-wise <|S|^2>: the Perot inner product the Redi flux uses.
    #     Fed to the same formula by handing it (sqrt(<|S|^2>), 0) — the
    #     routine consumes the slopes only through S_x^2 + S_y^2.
    s_sq = _perot_inner_product_cell(S_n * em, S_n * em, mesh)
    k_edgewise = kappa(jnp.sqrt(jnp.maximum(s_sq, 0.0)), jnp.zeros_like(s_sq))

    wet = np.asarray(mask) > 0.5
    lat = np.degrees(np.asarray(mesh.latCell))
    # How much of each COLUMN the taper actually bit.  ``taper_c`` is an EDGE
    # field (nEdges, nlev-1), so it cannot be indexed by cell; measure the
    # steepness at the cell instead, from the reconstructed magnitude that the
    # shipped branch limits.  A cell counts as tapered at a level when |S|
    # reaches half S_max, which is where the DM95 tanh starts to bite.
    mag_cell = np.asarray(jnp.sqrt(bx ** 2 + by ** 2))
    tapered_frac = np.mean(mag_cell > 0.5 * cfg.S_max, axis=1)

    def rel(a, b):
        d = np.maximum(np.abs(b), 1e-30)
        return (a - b) / d

    print("\n[kappa] wet-cell medians [m^2/s]")
    for name, k in (("per-component", k_component), ("vector (shipped)",
                                                     k_vector),
                    ("edge-wise <|S|^2>", k_edgewise)):
        kk = k[wet]
        print(f"  {name:<20s} median {np.median(kk):9.2f}  p90 "
              f"{np.percentile(kk, 90):9.2f}  max {kk.max():9.2f}  "
              f"frac at cap {np.mean(kk >= 0.999 * args.aei0):6.3f}")

    r_cv = rel(k_component, k_vector)[wet]
    r_ev = rel(k_edgewise, k_vector)[wet]
    print("\n[diff] (per-component - vector)/vector   — the operator choice")
    print(f"  median {np.median(r_cv):+.4f}  p10 {np.percentile(r_cv, 10):+.4f}"
          f"  p90 {np.percentile(r_cv, 90):+.4f}")
    print("[diff] (edge-wise - vector)/vector       — the Jensen gap")
    print(f"  median {np.median(r_ev):+.4f}  p10 {np.percentile(r_ev, 10):+.4f}"
          f"  p90 {np.percentile(r_ev, 90):+.4f}")

    print("\n[bin] by tapered fraction of the column (wet cells)")
    print("   band          n    med (comp-vec)/vec   med (edge-vec)/vec")
    for lo, hi in ((0.0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 1.01)):
        m = wet & (tapered_frac >= lo) & (tapered_frac < hi)
        if m.sum() == 0:
            continue
        print(f"  {lo:4.2f}-{hi:4.2f} {int(m.sum()):8d}      "
              f"{np.median(rel(k_component, k_vector)[m]):+10.4f}      "
              f"{np.median(rel(k_edgewise, k_vector)[m]):+10.4f}")

    print("\n[bin] by latitude band (wet cells)")
    print("   band          n    med (comp-vec)/vec   med (edge-vec)/vec")
    for lo, hi in ((-90, -60), (-60, -30), (-30, 30), (30, 60), (60, 90)):
        m = wet & (lat >= lo) & (lat < hi)
        if m.sum() == 0:
            continue
        print(f"  {lo:+4d}..{hi:+4d} {int(m.sum()):8d}      "
              f"{np.median(rel(k_component, k_vector)[m]):+10.4f}      "
              f"{np.median(rel(k_edgewise, k_vector)[m]):+10.4f}")
    if args.map:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lon = np.degrees(np.asarray(mesh.lonCell))
        lon = np.where(lon > 180.0, lon - 360.0, lon)
        gap = rel(k_edgewise, k_vector)
        fig, ax = plt.subplots(figsize=(12, 6))
        sc = ax.scatter(lon[wet], lat[wet], c=100.0 * gap[wet], s=1.5,
                        cmap="magma", vmin=0.0, vmax=20.0, linewidths=0)
        ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title(
            "Treguier kappa: how much LOWER the shipped cell-vector form is\n"
            "than the edge-wise mean-square form  [% of the shipped value]")
        fig.colorbar(sc, ax=ax, shrink=0.85, label="percent")
        fig.tight_layout()
        fig.savefig(args.map, dpi=130)
        print(f"\n[map] wrote {args.map}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
