"""#1226 heat discriminator: redistribution vs net-loss.

Compares a lego vs NEMO 90-day twin (both starting from the same NEMO
developed restart, see ``kamm_twin_90d.py``):

  1. Horizontal wet-mean T profile(z, day) - profile(z, day0), by level, for
     lego and NEMO, plus their difference, at day 90 (and 30/60 for the
     printed table). The lego-minus-NEMO drift-difference profile crosses
     zero ("X-crossing") at a depth that flags where the two models start
     disagreeing about whether heat is being *redistributed* (profile
     reshapes, integral roughly conserved) vs *lost/gained* (integral shifts
     without an offsetting sign flip elsewhere). Finding, 2026-07-24: the
     X-crossing sits at ~88 m for the plain nemo_dino_kamm_mlf twin, i.e.
     right at the base of the mixed layer -- consistent with a TKE vertical-
     mixing/entrainment scheme difference as the suspect, not a basin-scale
     heat leak.
  2. Volume-integrated heat content H(day) = rho0*cp*sum(T*vol_wet) for each
     model, using each model's own (rho0, cp) convention (legoESM:
     constants.rho_ocean=1025, constants.c_sw=3994; NEMO: rho0=1026,
     rcp=3991.86795711963 [eosbn2.F90:1899]) -- reported both in native
     convention and cross-checked with legoESM's constants to isolate the
     convention-difference contribution from a genuine T-field difference.
  3. Restoring-flux difference: mean over the twin window of
     -A_theta*(SST_lego - SST_nemo) integrated over wet area, A_theta=40
     W/m^2/K (dino.py DINOConfig.A_theta). Positive => restoring pumps MORE
     heat into lego (because lego is colder) -- if lego nonetheless stays
     colder, the sink must be internal (redistribution or a compensating
     loss elsewhere).
  4. SST difference map at the final day.

Usage
-----
    python heat_discriminator.py <lego_3dT.npz> <nemo_3dT.npz> \\
        <lego_daily.npz> <nemo_daily.npz> --nemo-run-traj <RUN_TRAJ dir> \\
        [--out heat_discriminator.png]

Inputs are pre-computed npz files (see ``kamm_twin_90d.py --save-3d`` for the
lego side; NEMO side stitched from NEMO's own grid_T output the same way).
``--nemo-run-traj`` points at the NEMO run directory holding ``mesh_mask.nc``
(for e3t_0/e1t/e2t/tmask geometry) -- geometry only, no live JAX run needed.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from legoesm import constants
from legoesm.ocean.vertical import compute_layer_thickness

DAYS_3D = (0, 30, 60, 90)
A_THETA = 40.0  # W/m^2/K, DINOConfig.A_theta (dino.py)

RHO0_LEGO = float(constants.rho_ocean)   # 1025.0
CP_LEGO = float(constants.c_sw)          # 3994.0
RHO0_NEMO = 1026.0                       # DINO/GYRE/ORCA rau0 (phycst.F90 convention)
CP_NEMO = 3991.86795711963               # eosbn2.F90:1899 (rcp)


def wet_mean_profile(t3d: np.ndarray, wet_mask) -> np.ndarray:
    """Horizontal wet-mean profile by level: (y,x,z) T field -> (z,) profile.

    ``wet_mask`` is a (y,x) boolean or a (y,x,z) boolean (per-level mask, e.g.
    NEMO ``tmask``); broadcasts either way.
    """
    t3d = np.asarray(t3d, dtype=np.float64)
    wet_mask = np.asarray(wet_mask, dtype=bool)
    m = np.broadcast_to(wet_mask[..., None] if wet_mask.ndim == 2 else wet_mask, t3d.shape)
    num = (t3d * m).sum(axis=(0, 1))
    den = m.sum(axis=(0, 1)).astype(np.float64)
    return num / np.maximum(den, 1)


def drift_crossing_depth(drift_diff: np.ndarray, depth: np.ndarray):
    """First zero-crossing depth of a (lego - NEMO) drift-difference profile.

    ``drift_diff[k]`` is the lego-minus-NEMO drift difference at level k;
    ``depth[k]`` the corresponding (positive-down) depth. Returns the
    linearly-interpolated depth of the first sign change (shallow to deep),
    or None if the profile never changes sign.
    """
    drift_diff = np.asarray(drift_diff, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    sign = np.sign(drift_diff)
    for k in range(len(sign) - 1):
        if sign[k] == 0:
            return float(depth[k])
        if sign[k] != sign[k + 1] and sign[k + 1] != 0:
            # linear interpolation between level k and k+1
            d0, d1 = drift_diff[k], drift_diff[k + 1]
            z0, z1 = depth[k], depth[k + 1]
            frac = d0 / (d0 - d1)
            return float(z0 + frac * (z1 - z0))
    return None


def _nemo_h_k(e3t_0, tmask3d, eta):
    """NEMO layer thickness incl. vvl free-surface stretch: h_k = e3t_0[k]*(1+sshn/H)."""
    h_col = e3t_0.sum(axis=-1)
    h_col_safe = np.where(h_col > 0, h_col, 1.0)
    stretch = 1.0 + eta / h_col_safe
    return e3t_0 * stretch[..., None] * tmask3d


def _lego_h_k(eta, h_bathy, z_coord, land_mask):
    import jax.numpy as jnp
    h_k = compute_layer_thickness(jnp.asarray(eta), jnp.asarray(h_bathy), z_coord)
    return np.asarray(h_k) * land_mask[..., None]


def run_discriminator(lego_path, nemo_path, lego_daily_path, nemo_daily_path,
                       nemo_run_traj: str, out_png: str):
    import netCDF4  # noqa: N813 (module alias kept literal)
    from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
    from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo

    lego = np.load(lego_path)
    nemo = np.load(nemo_path)
    lego_daily = np.load(lego_daily_path)
    nemo_daily = np.load(nemo_daily_path)

    g = read_nemo_mesh_mask(f"{nemo_run_traj}/mesh_mask.nc", nn_hls=0)
    # A restart is only needed for the bridge's geometry construction; any
    # restart in the run dir works since we only read geometry (H_bathy,
    # z_coord, land_mask), not the state.
    import glob
    restart_candidates = sorted(glob.glob(f"{nemo_run_traj}/*restart.nc"))
    if not restart_candidates:
        raise SystemExit(f"no *restart.nc found under {nemo_run_traj} for geometry bridge")
    s = read_nemo_restart(restart_candidates[0], nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)

    land_mask = np.asarray(br.land_mask) > 0.5
    area = np.asarray(br.geometry.area, dtype=np.float64)
    h_bathy = np.asarray(br.state.H_bathy.data, dtype=np.float64)
    z_coord = br.z_coord
    n_lev = lego["T3d_day0"].shape[-1]

    print(f"rho0/cp convention: lego=({RHO0_LEGO},{CP_LEGO})  nemo=({RHO0_NEMO},{CP_NEMO})  "
          f"rho0*cp ratio (nemo/lego) = {(RHO0_NEMO*CP_NEMO)/(RHO0_LEGO*CP_LEGO):.4f}")

    mm = netCDF4.Dataset(f"{nemo_run_traj}/mesh_mask.nc")
    e3t_0 = np.moveaxis(np.array(mm.variables["e3t_0"][0]), 0, -1)
    e1t = np.array(mm.variables["e1t"][0])
    e2t = np.array(mm.variables["e2t"][0])
    tmask3d = np.moveaxis(np.array(mm.variables["tmask"][0]), 0, -1) > 0.5
    mm.close()
    area_nemo = e1t * e2t

    t_depth_ref = getattr(z_coord, "t_depth_ref", None)
    z_ref_lego = (np.asarray(t_depth_ref).ravel()[:n_lev] if t_depth_ref is not None
                  else np.abs(np.asarray(z_coord.z_full_ref)).ravel()[:n_lev])

    profiles_lego = {d: wet_mean_profile(lego[f"T3d_day{d}"], land_mask) for d in DAYS_3D}
    profiles_nemo = {d: wet_mean_profile(nemo[f"T3d_day{d}"], tmask3d) for d in DAYS_3D}
    drift_lego = {d: profiles_lego[d] - profiles_lego[0] for d in DAYS_3D}
    drift_nemo = {d: profiles_nemo[d] - profiles_nemo[0] for d in DAYS_3D}
    drift_diff = {d: drift_lego[d] - drift_nemo[d] for d in DAYS_3D}

    x_depth = drift_crossing_depth(drift_diff[90], z_ref_lego)

    def heat_content_lego(day):
        t_field = lego[f"T3d_day{day}"].astype(np.float64)
        h_k = _lego_h_k(lego[f"eta3d_day{day}"], h_bathy, z_coord, land_mask)
        vol = area[..., None] * h_k
        return RHO0_LEGO * CP_LEGO * (t_field * vol).sum(), vol.sum()

    def heat_content_nemo(day, use_lego_convention=False):
        t_field = nemo[f"T3d_day{day}"].astype(np.float64)
        h_k = _nemo_h_k(e3t_0, tmask3d, nemo[f"eta3d_day{day}"])
        vol = area_nemo[..., None] * h_k
        rho0, cp = (RHO0_LEGO, CP_LEGO) if use_lego_convention else (RHO0_NEMO, CP_NEMO)
        return rho0 * cp * (t_field * vol).sum(), vol.sum()

    h_lego, h_nemo, h_nemo_legoconv, v_lego, v_nemo = {}, {}, {}, {}, {}
    for d in DAYS_3D:
        h_lego[d], v_lego[d] = heat_content_lego(d)
        h_nemo[d], v_nemo[d] = heat_content_nemo(d)
        h_nemo_legoconv[d], _ = heat_content_nemo(d, use_lego_convention=True)

    a_theta_local = A_THETA
    sst_lego_daily = lego_daily["sst"].astype(np.float64)
    sst_nemo_daily = nemo_daily["sst"].astype(np.float64)
    wet_b = land_mask[None, :, :]
    sst_diff_daily = np.where(wet_b, sst_lego_daily - sst_nemo_daily, np.nan)
    restoring_diff_flux = -a_theta_local * sst_diff_daily
    restoring_diff_power = np.nansum(restoring_diff_flux * area[None, :, :], axis=(1, 2))
    restoring_diff_power_mean = np.nanmean(restoring_diff_power)
    n_days = sst_lego_daily.shape[0]
    restoring_diff_energy = restoring_diff_power_mean * n_days * 86400.0
    mean_sst_diff = np.nanmean(sst_diff_daily)

    print("\n" + "=" * 78)
    print("DRIFT PROFILE TABLE (day90 vs day0), top 10 levels + deep levels")
    print("=" * 78)
    print(f"{'k':>3} {'depth_m':>9} {'lego_drift_K':>13} {'nemo_drift_K':>13} {'diff(l-n)_K':>13}")
    top_levels = list(range(10))
    deep_levels = list(range(n_lev - 6, n_lev))
    for k in top_levels + ["..."] + deep_levels:
        if k == "...":
            print("  ...")
            continue
        print(f"{k:>3} {z_ref_lego[k]:>9.1f} {drift_lego[90][k]:>13.4f} "
              f"{drift_nemo[90][k]:>13.4f} {drift_diff[90][k]:>13.4f}")
    if x_depth is not None:
        print(f"\nX-crossing depth (lego-NEMO drift diff, day 90) = {x_depth:.1f} m")
    else:
        print("\nX-crossing depth: profile does not change sign -- no crossing found")

    print("\n" + "=" * 78)
    print("HEAT CONTENT (volume-integrated, native rho0*cp convention)")
    print("=" * 78)
    for d in DAYS_3D:
        print(f"day {d:3d}: h_lego={h_lego[d]:.6e} J   h_nemo={h_nemo[d]:.6e} J   "
              f"h_nemo(lego rho0*cp)={h_nemo_legoconv[d]:.6e} J   "
              f"v_lego={v_lego[d]:.6e} m^3  v_nemo={v_nemo[d]:.6e} m^3")

    dh_lego = h_lego[90] - h_lego[0]
    dh_nemo = h_nemo[90] - h_nemo[0]
    dh_nemo_legoconv = h_nemo_legoconv[90] - h_nemo_legoconv[0]
    print(f"\nDelta-H (day90-day0), native convention: lego={dh_lego:.6e} J   nemo={dh_nemo:.6e} J")
    print(f"Delta-H (day90-day0), NEMO T-field with LEGO rho0*cp: {dh_nemo_legoconv:.6e} J "
          f"(convention artifact vs native-nemo: {dh_nemo - dh_nemo_legoconv:.6e} J)")
    print(f"Delta-H difference (lego - nemo_legoconv), same convention: "
          f"{dh_lego - dh_nemo_legoconv:.6e} J")

    print("\n" + "=" * 78)
    print(f"RESTORING-FLUX DIFFERENCE (surface, daily series, {n_days} days)")
    print("=" * 78)
    print(f"A_theta = {a_theta_local} W/m^2/K (DINOConfig.A_theta, dino.py)")
    print(f"mean(SST_lego - SST_nemo), wet cells = {mean_sst_diff:.4f} K")
    print(f"mean implied EXTRA restoring power into lego = {restoring_diff_power_mean:.6e} W")
    print(f"  integrated over {n_days} days = {restoring_diff_energy:.6e} J")
    print(f"  compare to Delta-H(lego) - Delta-H(nemo, lego-convention) = "
          f"{dh_lego - dh_nemo_legoconv:.6e} J")

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(drift_lego[90], -z_ref_lego, "o-", label="lego drift (d90-d0)", ms=3)
    ax1.plot(drift_nemo[90], -z_ref_lego, "s-", label="NEMO drift (d90-d0)", ms=3)
    ax1.plot(drift_diff[90], -z_ref_lego, "^-", label="lego-NEMO drift diff", ms=3, color="k")
    ax1.axvline(0, color="gray", lw=0.5)
    if x_depth is not None:
        ax1.axhline(-x_depth, color="r", lw=0.8, ls="--", label=f"X-crossing {x_depth:.0f}m")
    ax1.set_xlabel("Temperature drift [K]")
    ax1.set_ylabel("Depth [m]")
    ax1.set_title("(a) Wet-mean T drift by level, day 90")
    ax1.legend(fontsize=8)
    ax1.set_ylim(-1500, 0)

    ax2 = fig.add_subplot(1, 3, 2)
    days_arr = np.array(DAYS_3D)
    h_lego_arr = np.array([h_lego[d] for d in DAYS_3D])
    h_nemo_arr = np.array([h_nemo_legoconv[d] for d in DAYS_3D])
    ax2.plot(days_arr, (h_lego_arr - h_lego_arr[0]) / 1e18, "o-", label="lego (native conv.)")
    ax2.plot(days_arr, (h_nemo_arr - h_nemo_arr[0]) / 1e18, "s-", label="NEMO (lego rho0*cp)")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Delta heat content [EJ]")
    ax2.set_title("(b) Volume-integrated heat content drift")
    ax2.legend(fontsize=8)
    ax2.axhline(0, color="gray", lw=0.5)

    ax3 = fig.add_subplot(1, 3, 3)
    sst_diff_final = np.where(land_mask, sst_lego_daily[-1] - sst_nemo_daily[-1], np.nan)
    vmax = np.nanmax(np.abs(sst_diff_final))
    im = ax3.imshow(sst_diff_final, origin="lower", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, aspect="auto")
    ax3.set_title(f"(c) SST diff (lego-NEMO), day {n_days} [K]")
    plt.colorbar(im, ax=ax3, label="K")

    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"\nSAVED {out_png}")
    return x_depth


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("lego_3dT", help="lego 3-D T/S npz (kamm_twin_90d.py --save-3d output)")
    p.add_argument("nemo_3dT", help="NEMO stitched 3-D T/S npz (same snapshot days)")
    p.add_argument("lego_daily", help="lego daily surface-series npz")
    p.add_argument("nemo_daily", help="NEMO daily surface-series npz")
    p.add_argument("--nemo-run-traj", required=True,
                    help="NEMO run dir with mesh_mask.nc + a restart (geometry only)")
    p.add_argument("--out", default="heat_discriminator.png", help="output PNG path")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run_discriminator(args.lego_3dT, args.nemo_3dT, args.lego_daily, args.nemo_daily,
                       args.nemo_run_traj, args.out)


if __name__ == "__main__":
    main()
