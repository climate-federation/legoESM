#!/usr/bin/env python3
"""The gustiness sweep on INSTANTANEOUS columns, where the knob can act.

The monthly-mean version of this sweep found the COARE convective-gustiness
depth nearly inert (0.1% of the tropical ocean latent heat between the
production 300 m and the scheme-native 600 m).  That result is
instrument-limited and was reported as such: gustiness exists to represent
calm convective columns, and monthly averaging is precisely what removes the
calm tail.  A mean wind of 8 m/s cannot show an effect that lives at 1 m/s.

So the same sweep runs here on a saved checkpoint, which is an instantaneous
state with its calm columns intact, with the corrected surface law (true
level height, sea-water humidity).  Cell winds come from the model's own
Perot reconstruction rather than a re-derived average.

Reports the area mean AND the calm-column subset, because an area mean over
mostly-windy ocean can hide a large effect confined to the tail -- which is
the same averaging mistake one level down.

Usage: gustiness_instantaneous.py <checkpoint.npz> --run <run-name>
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import regional_bias as rb  # noqa: E402
from ledger_regional import _sftlf_on_mesh  # noqa: E402


def _sst_on_mesh(run, months, lat, lon):
    """The run's own prescribed SST [K] on the SCVT cells.

    The lat-lon binning used by the published-field probes assumes a regular
    target grid and silently returns nothing useful for scattered mesh cells
    (it reported 0% ocean here).  Nearest neighbour, the same indexing the
    land-mask helper next door uses, is the right tool on a mesh and is
    adequate for a sensitivity sweep -- the SST field is smooth on the scale
    of a cell.  Read from the run's OWN forcing file, never a default path.
    """
    import json
    import xarray as xr
    man = json.load(open(f"{rb.ROOT}/{run}/run_manifest.json"))
    cmd = man["run"]["command_line"].split()
    if "--forcing-path" not in cmd:
        raise SystemExit(f"FATAL: {run} manifest names no --forcing-path")
    path = cmd[cmd.index("--forcing-path") + 1]
    off = float(cmd[cmd.index("--sst-offset") + 1]) if "--sst-offset" in cmd else 0.0
    d = xr.open_dataset(path, decode_times=True)
    var = "tosbcs" if "tosbcs" in d else "tos"
    clim = (d[var].groupby("time.month").mean("time").sel(month=months)
            .mean("month").load())
    glat = np.asarray(clim["lat"], dtype=np.float64)
    glon = np.asarray(clim["lon"], dtype=np.float64) % 360.0
    arr = np.asarray(clim, dtype=np.float64)
    if arr.shape != (glat.size, glon.size):
        arr = arr.T
    i = np.abs(glat[None, :] - lat[:, None]).argmin(axis=1)
    j = np.abs(((glon[None, :] - lon[:, None] + 180.0) % 360.0) - 180.0
               ).argmin(axis=1)
    return arr[i, j] + off

SALINE = 0.98


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--run", required=True, help="run whose SST forcing to use")
    ap.add_argument("--mesh-level", type=int, default=6)
    ap.add_argument("--sftlf-from", default=None,
                    help="run whose land mask to use when this one has "
                         "published no monthly stream yet")
    ap.add_argument("--gustiness", type=float, nargs="+",
                    default=[0.0, 300.0, 600.0, 900.0])
    ap.add_argument("--z0h-ratio", type=float, nargs="*", default=None,
                    help="also sweep the thermal/momentum roughness ratio, "
                         "which scales heat and moisture transfer directly")
    ap.add_argument("--unstable-gamma", type=float, nargs="*", default=None,
                    help="sweep the MOST unstable-branch stability coefficient")
    ap.add_argument("--calm-below", type=float, default=4.0,
                    help="wind speed [m/s] defining the calm subset")
    a = ap.parse_args(argv)

    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.core.bulk_flux import compute_most_fluxes
    from legoesm.thermo import saturation_specific_humidity
    from legoesm.grids.voronoi import create_voronoi_mesh, reconstruct_cell_velocity
    from legoesm.atmosphere.physics._shared import compute_rho

    d = np.load(a.checkpoint, allow_pickle=True)
    T = jnp.asarray(d["T"])
    p_s = jnp.asarray(d["p_s"])
    q_v = jnp.asarray(d["trc_q_v"])
    _vg = np.asarray(d["meta_vgrid"], dtype=np.float64)
    assert _vg.shape[0] == 2 and float(np.abs(_vg[0]).max()) == 0.0, (
        "hybrid checkpoint: this probe handles the sigma ladder only")
    sig_half = _vg[1]
    sig_full = jnp.asarray(0.5 * (sig_half[:-1] + sig_half[1:]))
    ncol, nlev = T.shape

    mesh = create_voronoi_mesh(a.mesh_level)
    lat = np.rad2deg(np.asarray(mesh.latCell))
    lon = np.rad2deg(np.asarray(mesh.lonCell)) % 360.0
    assert lat.shape[0] == ncol, (lat.shape, ncol)

    # The model's OWN edge-to-cell reconstruction, not a re-derived average.
    u_cell, v_cell = reconstruct_cell_velocity(jnp.asarray(d["u"]), mesh)
    u_low = np.asarray(u_cell[:, -1])
    v_low = np.asarray(v_cell[:, -1])
    wind = np.hypot(u_low, v_low)

    p_full = sig_full[None, :] * p_s[:, None]
    p_half = jnp.asarray(np.maximum(sig_half, 1e-5))[None, :] * p_s[:, None]
    T_a = np.asarray(T[:, -1])
    q_a = np.asarray(q_v[:, -1])
    rho = np.asarray(compute_rho(T, p_full, q_v)[:, -1])
    z_low = np.asarray((constants.R_d * T_a / constants.g)
                       * np.log(1.0 / float(0.5 * (sig_half[-1] + sig_half[-2]))))

    months = rb._month_labels(rb._load_model(a.run, "ps"))
    sst = _sst_on_mesh(a.run, months, lat, lon)
    # The mesh-native land fraction, via the helper the ledger probes use --
    # nearest-neighbour onto the SCVT cells, and it raises rather than borrow
    # another run's mask.
    f_land = _sftlf_on_mesh(a.sftlf_from or a.run, lat, lon)
    ocean = (np.asarray(f_land) < 0.5) & np.isfinite(sst)
    calm = ocean & (wind < a.calm_below)
    q_sfc = np.asarray(saturation_specific_humidity(
        jnp.asarray(sst), p_s)) * SALINE

    print(f"{a.checkpoint}: {ncol} columns, instantaneous")
    print(f"ocean columns {100 * ocean.mean():.1f}%, of which calm "
          f"(<{a.calm_below:g} m/s) {100 * calm.sum() / max(ocean.sum(), 1):.1f}%")
    print(f"mean ocean wind {wind[ocean].mean():.2f} m/s, "
          f"calm-subset mean {wind[calm].mean() if calm.any() else float('nan'):.2f}")

    def lh(zi, z0h=None, gamma=None):
        _tx, _ty, _sh, lhf, _us = compute_most_fluxes(
            jnp.asarray(wind), jnp.zeros_like(jnp.asarray(wind)),
            jnp.asarray(T_a + (constants.g / constants.c_pd) * z_low),
            jnp.asarray(q_a), jnp.asarray(sst), jnp.asarray(q_sfc),
            jnp.asarray(rho), z_ref=jnp.asarray(z_low), scheme="coare3",
            stability_scheme="dyer1974", gustiness_w_zi=float(zi),
            **({} if z0h is None else {"z0h_z0_ratio": float(z0h)}),
            **({} if gamma is None else {"unstable_gamma": float(gamma)}))
        return np.asarray(lhf)

    w = np.cos(np.deg2rad(lat))
    print(f"\n{'subset':<22}" + "".join(f"{'zi=' + str(int(z)):>10}"
                                        for z in a.gustiness))
    for name, m in (("all ocean", ocean), (f"calm only", calm)):
        if not m.any():
            continue
        vals = [float((lh(z)[m] * w[m]).sum() / w[m].sum()) for z in a.gustiness]
        print(f"{name:<22}" + "".join(f"{v:10.1f}" for v in vals))
    if a.z0h_ratio:
        # The thermal roughness ratio multiplies the heat and moisture
        # transfer coefficient, so unlike gustiness it acts on EVERY column
        # rather than the calm tail.  If anything in the surface exchange can
        # return the evaporation the height correction removed, it is this.
        print(f"\n{'subset':<22}" + "".join(f"{'z0h/z0=' + f'{r:g}':>12}"
                                            for r in a.z0h_ratio))
        for name, m in (("all ocean", ocean), ("calm only", calm)):
            if not m.any():
                continue
            vals = [float((lh(300.0, r)[m] * w[m]).sum() / w[m].sum())
                    for r in a.z0h_ratio]
            print(f"{name:<22}" + "".join(f"{v:12.1f}" for v in vals))

    if a.unstable_gamma:
        # The last surface-exchange knob the ocean scheme actually reads.
        # COARE evolves its own roughness, so z0 and the thermal-roughness
        # ratio are inert for it by construction (bulk_flux states this), and
        # gustiness is measured above.  If this one is flat too, the ocean
        # exchange has no tuning freedom left and the retune must happen
        # somewhere other than the surface.
        print(f"\n{'subset':<22}" + "".join(f"{'gamma=' + f'{g:g}':>12}"
                                            for g in a.unstable_gamma))
        for name, m in (("all ocean", ocean), ("calm only", calm)):
            if not m.any():
                continue
            vals = [float((lh(300.0, None, g)[m] * w[m]).sum() / w[m].sum())
                    for g in a.unstable_gamma]
            print(f"{name:<22}" + "".join(f"{v:12.1f}" for v in vals))

    print("\nlatent heat flux [W/m2], corrected surface law, area-weighted.")
    print("The calm row is the test: if gustiness matters anywhere it is "
          "there, and an all-ocean mean would dilute it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
