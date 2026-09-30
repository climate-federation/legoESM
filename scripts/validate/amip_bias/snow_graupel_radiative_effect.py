#!/usr/bin/env python3
"""What the radiation would see if snow and graupel were not invisible to it.

Four fifths of this model's frozen water sits in snow and graupel, and the
radiation solver is handed only cloud ice, so that mass is radiatively inert.
This measures the instantaneous sensitivity to it: the SAME columns from a
saved production checkpoint, the SAME cloud properties, the SAME overlap and
radii, with only the ICE PATH handed to the solver changed.

Scoped by codex as the smallest honest experiment, and its caveats are kept
here rather than in a commit nobody re-reads:

  * the extra mass is given CLOUD-ICE optics and CLOUD-ICE coverage, because
    the solver has no third condensate and the subcolumn generator distributes
    only liquid and ice.  Snow is not cloud ice -- it is larger, falls, and
    sits partly below cloud base -- so this is an upper-ish bound under stated
    assumptions, not a forcing estimate.
  * cloud fraction, effective radii and the diagnostic floors are computed
    ONCE from the checkpoint and held byte-identical across arms.  Feeding the
    extra mass in upstream instead would change diagnosed cover, the floors
    and the particle-size distribution radius, and the result would no longer
    be attributable to the omitted mass.
  * nothing here says how much of the model's radiation bias this explains.

Usage: snow_graupel_radiative_effect.py <checkpoint.npz> [--mesh-level 6]
"""
from __future__ import annotations

import argparse

import numpy as np


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("--mesh-level", type=int, default=6)
    p.add_argument("--n-times", type=int, default=8,
                   help="local-time quadrature points over the diurnal cycle")
    args = p.parse_args(argv)

    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties)
    from legoesm.atmosphere.physics.clouds.config import CloudConfig
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics._shared import compute_rho
    from legoesm.grids.voronoi import create_voronoi_mesh

    d = np.load(args.checkpoint, allow_pickle=True)
    T = jnp.asarray(d["T"])
    p_s = jnp.asarray(d["p_s"])
    q_v, q_c, q_i = (jnp.asarray(d[k]) for k in ("trc_q_v", "trc_q_c", "trc_q_i"))
    q_s, q_g = jnp.asarray(d["trc_q_s"]), jnp.asarray(d["trc_q_g"])
    n_c, n_i = jnp.asarray(d["trc_N_c"]), jnp.asarray(d["trc_N_i"])

    _vg = np.asarray(d["meta_vgrid"], dtype=np.float64)
    assert _vg.shape[0] == 2 and float(np.abs(_vg[0]).max()) == 0.0, (
        "meta_vgrid row 0 is not all-zero: hybrid checkpoint, this probe "
        "only handles the sigma ladder")
    sig_half_np = _vg[1]
    sig_full = jnp.asarray(0.5 * (sig_half_np[:-1] + sig_half_np[1:]))
    ncol, nlev = T.shape
    day = float(d["day"]) if "day" in d else 12.0

    mesh = create_voronoi_mesh(args.mesh_level)
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    assert lat.shape[0] == ncol, (lat.shape, ncol)

    p_full = sig_full[None, :] * p_s[:, None]
    p_half = jnp.asarray(np.maximum(sig_half_np, 1e-5))[None, :] * p_s[:, None]
    dp = p_half[:, 1:] - p_half[:, :-1]
    T_sfc = T[:, -1]

    conv = str(d["number_convention"]) if "number_convention" in d else "per_mass"
    rho = compute_rho(T, p_full, q_v)
    n_c_vol = n_c * rho if conv == "per_mass" else n_c
    n_i_vol = n_i * rho if conv == "per_mass" else n_i

    ccfg = CloudConfig(scheme="sundqvist", rh_crit=0.85, q_c_diagnostic=5.0e-6)
    cp = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp, config=ccfg,
        q_cloud=q_c, q_ice=q_i, n_cloud=n_c_vol, n_ice=n_i_vol)
    kw = cp.to_rrtmg_kwargs()

    # The omitted mass as a path, on the same footing as the ice path the
    # solver already receives.
    iwp = np.asarray(kw["cloud_path_ice"])
    swp = np.asarray(jnp.sum((q_s + q_g) * dp, axis=1) / constants.g
                     ) if iwp.ndim == 1 else np.asarray(
        (q_s + q_g) * dp / constants.g)
    seen = float(np.sum(iwp))
    omitted = float(np.sum(swp))
    print(f"{args.checkpoint}: {ncol} columns, {nlev} levels, day {day:g}")
    print(f"ice the solver sees   : {seen / ncol * 1e3:8.3f} g/m2 per column")
    print(f"snow+graupel omitted  : {omitted / ncol * 1e3:8.3f} g/m2 per column")
    if seen > 0:
        print(f"omitted / seen        : {omitted / seen:8.2f}x")

    doy = 1.0 + day
    decl = -23.44 * np.cos(2 * np.pi * (doy + 10.0) / 365.0) * np.pi / 180.0
    hours = np.arange(args.n_times) * 24.0 / args.n_times + 12.0 / args.n_times
    cosz_t = []
    for h in hours:
        H = (h - 12.0) * np.pi / 12.0
        mu = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(H)
        cosz_t.append(np.clip(mu, 0.0, 1.0))

    solver = RRTMGP.from_legoesm_config(RRTMGPConfig(
        gpoint_batch_size=16, gpoint_checkpoint=False, include_clouds=True))
    area_w = jnp.ones(ncol) / ncol          # SCVT cells are ~equal area

    def toa(path_ice):
        """Diurnally averaged outgoing shortwave and longwave [W/m2]."""
        sw = 0.0
        lw = 0.0
        for mu in cosz_t:
            mu_j = jnp.asarray(np.maximum(mu, 0.0))
            out = solver.solve_columns(
                T=T, p_full=p_full, p_half=p_half, sfc_temperature=T_sfc,
                q_v=q_v, cos_zenith=jnp.maximum(mu_j, 1e-4),
                sfc_albedo=0.06, sfc_emissivity=0.97,
                cloud_path_liq=kw["cloud_path_liq"],
                cloud_path_ice=path_ice,
                cloud_r_eff_liq=kw["cloud_r_eff_liq"],
                cloud_r_eff_ice=kw.get("cloud_r_eff_ice"))
            day_mask = jnp.asarray(mu > 0.0)
            sw = sw + jnp.sum(area_w * jnp.where(
                day_mask, out.sw_flux_up[:, 0], 0.0))
            lw = lw + jnp.sum(area_w * out.lw_flux_up[:, 0])
        n = len(cosz_t)
        return float(sw / n), float(lw / n)

    base_sw, base_lw = toa(kw["cloud_path_ice"])
    with_sg = jnp.asarray(kw["cloud_path_ice"]) + jnp.asarray(swp)
    sg_sw, sg_lw = toa(with_sg)

    print(f"\n{'arm':<26}{'rsut':>10}{'rlut':>10}{'net':>10}")
    print(f"{'cloud ice only (as run)':<26}{base_sw:10.2f}{base_lw:10.2f}"
          f"{-(base_sw + base_lw):10.2f}")
    print(f"{'+ snow and graupel':<26}{sg_sw:10.2f}{sg_lw:10.2f}"
          f"{-(sg_sw + sg_lw):10.2f}")
    print(f"{'difference':<26}{sg_sw - base_sw:10.2f}{sg_lw - base_lw:10.2f}"
          f"{-(sg_sw + sg_lw) + (base_sw + base_lw):10.2f}")
    print("\nOutgoing shortwave and longwave at the top of the atmosphere, "
          "diurnally averaged, area-weighted.\nThe extra mass is given cloud-"
          "ice optics and cloud-ice coverage because the solver has no third\n"
          "condensate: this is the sensitivity to the omitted mass under that "
          "assumption, not a forcing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
