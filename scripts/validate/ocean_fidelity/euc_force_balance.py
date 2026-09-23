#!/usr/bin/env python
"""What holds the Equatorial Undercurrent up, ours against the ORCA1 oracle.

WHY THIS EXISTS.  Our Pacific undercurrent runs at roughly 0.55 of NEMO's core
speed at every mark from day 10 to day 90.  Twelve marks spanning days 35-90
show that ratio flat, but flatness cannot decide whether the deficit is
structural or still spinning up: the undercurrent's transport is a basin-scale
mass balance whose adjustment needs western-boundary reflection and Rossby
propagation, order six to twelve months, so ninety days is INSIDE the window
and two runs with different effective mixing look exactly like this mid
transient.

A force balance does not have that problem.  It asks which term is wrong
rather than how far the jet has got, so it is readable at day 35 as well as at
day 400.  At the equator the Coriolis term vanishes and the jet is held up by
an eastward pressure gradient working against the downward transfer of
westward surface momentum by vertical friction.  This probe measures both,
from files we already hold, for both models, through one code path.

WHAT IT COMPUTES, per longitude, as a profile in depth, in m/s^2:

  PGF_baro(z) = -g d(eta)/dx                      (depth independent)
  PGF_bc(z)   = -(g/rho0) d/dx [ integral of rho' from z to the surface ]
  FRIC(z)     = d/dz ( Kv du/dz )
  RESID       = PGF_baro + PGF_bc + FRIC

RESID holds the zonal advection, the lateral friction and the local
acceleration, which this probe does NOT separate.  The local acceleration IS
separable and must be removed before RESID is interpreted: run the probe at
every five-day mark with ``--dump``, then difference the dumped core velocity
in time on BOTH sides.  ``scripts/cluster/omip_nemo/_euc_force_balance.sbatch``
does exactly that and is the only sanctioned reader of these dumps.  Reading a
difference in RESID as
"advection, therefore spin-up, therefore no verdict" without removing du/dt
first is a false exoneration: advection is part of the STATIONARY undercurrent
balance too, and it can differ structurally through numerical diffusion along
the equatorial waveguide or through the vertical advection of the thermocline.

EVERYTHING IS ON T POINTS, AND THAT IS DELIBERATE.  The two models index the
C grid differently -- NEMO puts u(i) between T(i) and T(i+1), while ours puts
u(i) between T(i-1) and T(i) (``barotropic_latlon_cgrid.py``'s NEMO-literal
surface PGF pairs ``roll(eta,-1)-eta`` with ``dx_u[:, 1:]``).  Comparing face
quantities across that shift is a one-cell error that no norm would catch, so
every term here is carried to cell centres first, with the per-side face
pairing named in ``_dphi_dx`` and nowhere else.  This campaign has already
shipped and retracted one claim built on comparing two different staggerings.

EQUATIONS OF STATE.  Our run uses Wright with rho0 = 1025; ORCA1 runs TEOS-10
(``namelist_cfg``: ``ln_teos10 = .true.``), whose Roquet polynomial this repo
already implements as ``nemo_teos10``.  Both are reachable from
``make_eos_fn``, so the default here gives each model the EOS it actually
integrated with -- the pressure gradient each one FELT.  ``--eos-both`` instead
applies one EOS to both sides' T and S, which asks the different question of
whether the density FIELDS differ.  Run the default first; the difference
between the two answers is the part of any pressure-gradient gap that is EOS
formulation rather than ocean state.  Do not mix them in one sentence.

SAMPLING, STATED BECAUSE IT CANNOT BE FIXED HERE.  Our snapshots are
INSTANTANEOUS; the oracle publishes FIVE-DAY MEANS.  Vertical viscosity is
intermittent, so FRIC is the term most exposed, and the mean of a product is
not the product of the means -- NEMO's own residual therefore will NOT close
even on a perfectly healthy oracle.  Two consequences are built into the
gates: the closure check below is applied only to the terms LINEAR in the
means, and no conclusion from a single mark is reportable.  Run all twelve.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from global_tracer_content import (  # noqa: E402
    load_mesh_latitude,
    load_mesh_longitude,
    load_mesh_metrics,
)

# eORCA1 native frame inside a snapshot array: rows 0..330 (row 331 is the
# reversed north-fold ghost) and columns 1..360 (cyclic overlap both sides).
# Identical to global_tracer_content's slices; imported names are private
# there, so they are restated rather than reached into.
_NATIVE_J = slice(0, 331)
_NATIVE_I = slice(1, 361)
_STEPS_PER_DAY = 576


def _native(a):
    """Trim a snapshot array's halo, whatever its trailing axes."""
    return a[_NATIVE_J, _NATIVE_I]


def _dphi_dx(phi, dx_w, dx_e):
    """Zonal derivative of a T-point field, returned on T points.

    ``phi`` is (nj, ni[, nk]).  ``dx_w`` and ``dx_e`` are the widths of the
    two faces bounding each T cell, already sliced by the CALLER to that
    side's own convention -- this function never guesses which face is which.
    The stencil is centred over both faces, so it is the average of the two
    one-sided face gradients weighted by their own widths:

        dphi/dx |_i = (phi[i+1] - phi[i-1]) / (dx_w[i] + dx_e[i])

    Periodic in x, which eORCA1 is away from the fold.
    """
    east = np.roll(phi, -1, axis=1)
    west = np.roll(phi, 1, axis=1)
    span = dx_w + dx_e
    if phi.ndim == 3:
        span = span[..., None]
    return (east - west) / span


def _ddz_flux(u_c, kv_if, z_c, z_if):
    """d/dz ( Kv du/dz ) on the CELL CENTRES of ``z_c``.

    ``u_c`` is (nj, ni, nk) on centres, ``kv_if`` is (nj, ni, nk-1) on the
    interfaces between them, ``z_c`` and ``z_if`` are POSITIVE-DOWN depths in
    metres.  The stress Kv du/dz lives on interfaces with the viscosity; its
    divergence returns to centres.

    Sign convention, stated at the term: z is measured POSITIVE DOWNWARD, so
    d/dz here is d/d(depth).  Applying the same operator to both sides twice
    makes the pair comparable regardless, but the sign matters for reading
    FRIC against PGF, so the caller converts once: with depth increasing
    downward, du/d(depth) and the true du/dz differ by a sign, and the double
    application in the divergence cancels it.  Hence no explicit flip here.
    """
    dz_if = np.diff(z_c)                      # centre-to-centre, (nk-1,)
    dudz = np.diff(u_c, axis=2) / dz_if       # on interfaces
    stress = kv_if * dudz
    dz_c = np.diff(z_if)                      # interface-to-interface
    out = np.full_like(u_c, np.nan)
    out[:, :, 1:-1] = np.diff(stress, axis=2) / dz_c[None, None, :]
    return out


def _hydrostatic_p_prime(rho, dz, rho0):
    """Baroclinic pressure anomaly / rho0 at cell centres, units m^2/s^2 per g.

    Returns ``integral from z to the surface of (rho - rho0) dz' / rho0`` with
    the cell's own half-thickness included, so the value sits at the centre
    rather than at the interface above it.  ``dz`` is the LIVE thickness of
    each cell on that side -- NEMO's time-varying ``e3t`` where available and
    our z-star thickness otherwise.  Interpolating either side onto the
    other's reference levels would bias the vertical structure of exactly the
    integral this probe reads.
    """
    d = (rho - rho0) / rho0
    above = np.cumsum(np.nan_to_num(d) * dz, axis=2) - 0.5 * np.nan_to_num(d) * dz
    return above


def _box_profile(field, lat, lon, lon0, band):
    """Mean over |lat| <= band at longitude ``lon0``, as a depth profile."""
    dlon = np.abs(((lon - lon0 + 180.0) % 360.0) - 180.0)
    sel = (np.abs(lat) <= band) & (dlon <= 0.5)
    if not sel.any():
        raise SystemExit(f"VACUOUS: no cells within {band} deg of the equator "
                         f"at {lon0}E")
    if field.ndim == 2:
        vals = field[sel]
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
    vals = field[sel, :]
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(vals).any(axis=0), np.nanmean(vals, axis=0),
                        np.nan)


def _eos_fn(name, rho0):
    from legoesm.ocean.eos import make_eos_fn
    import jax.numpy as jnp

    base = make_eos_fn(eos=name)

    def fn(T, S, depth_m, g):
        p = rho0 * g * np.asarray(depth_m)
        out = base(jnp.asarray(T), jnp.asarray(S), jnp.asarray(p))
        return np.asarray(out, dtype=np.float64)

    return fn


def _load_ours(path, expect_day, eos_name, rho0, g):
    d = np.load(path, allow_pickle=True)
    if "time_days" not in d.files:
        raise SystemExit(f"snapshot has no time_days stamp: {sorted(d.files)[:8]}")
    day = float(np.asarray(d["time_days"]).ravel()[0])
    if expect_day is not None and abs(day - expect_day) > 1e-6:
        raise SystemExit(f"snapshot is day {day:g}, expected {expect_day:g}")

    T = _native(np.asarray(d["T"], dtype=np.float64))
    S = _native(np.asarray(d["S"], dtype=np.float64))
    eta = _native(np.asarray(d["eta"], dtype=np.float64))
    H = _native(np.asarray(d["H_bathy"], dtype=np.float64))
    lat = _native(np.asarray(d["lat_T"], dtype=np.float64))
    lon = ((_native(np.asarray(d["lon_T"], dtype=np.float64))) + 360.0) % 360.0
    wet = _native(np.asarray(d["land_mask"], dtype=np.float64)) > 0.5
    z_c = np.asarray(d["z_center_ref"], dtype=np.float64)
    z_if = np.asarray(d["z_interface_ref"], dtype=np.float64)

    # u is one column WIDER than T: u[j, i] is the face between T[j, i-1] and
    # T[j, i], so the two faces bounding T[j, i] are u[:, i] and u[:, i+1].
    u_face = np.asarray(d["u"], dtype=np.float64)[_NATIVE_J, :, :]
    dx_face = np.asarray(d["dx_u"], dtype=np.float64)[_NATIVE_J, :]
    i0 = _NATIVE_I.start
    u_c = 0.5 * (u_face[:, i0:i0 + 360, :] + u_face[:, i0 + 1:i0 + 361, :])
    dx_w = dx_face[:, i0:i0 + 360]
    dx_e = dx_face[:, i0 + 1:i0 + 361]

    kv = _native(np.asarray(d["K_M_diag"], dtype=np.float64))

    # z-star: every cell's thickness scales with the free surface.
    dz_ref = np.diff(np.concatenate([[0.0], z_if, [z_c[-1] + (z_c[-1] - z_if[-1])]]))
    dz_ref = dz_ref[:len(z_c)]
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = np.where(H > 0, (eta + H) / H, 1.0)
    dz = dz_ref[None, None, :] * jac[:, :, None]

    T = np.where(wet[:, :, None], T, np.nan)
    S = np.where(wet[:, :, None], S, np.nan)
    rho = _eos_fn(eos_name, rho0)(np.nan_to_num(T, nan=0.0),
                                  np.nan_to_num(S, nan=35.0),
                                  z_c[None, None, :], g)
    rho = np.where(wet[:, :, None], rho, np.nan)
    return dict(day=day, T=T, S=S, rho=rho, eta=np.where(wet, eta, np.nan),
                u=np.where(wet[:, :, None], u_c, np.nan), kv=kv, dz=dz,
                z_c=z_c, z_if=z_if, lat=lat, lon=lon, wet=wet,
                dx_w=dx_w, dx_e=dx_e, eos=eos_name)


def _nemo_var(ds, name, rec):
    a = np.asarray(ds.variables[name][rec], dtype=np.float64)
    a = np.squeeze(a)
    if a.ndim == 3:                      # (k, j, i) -> (j, i, k) native frame
        a = np.transpose(a, (1, 2, 0))
    return a


def _load_nemo(gridt, gridu, gridw, mesh, rec, eos_name, rho0, g):
    import netCDF4 as nc

    e1t, _e2t, e3t0, tmask = load_mesh_metrics(mesh)
    lat = load_mesh_latitude(mesh)
    lon = (load_mesh_longitude(mesh) + 360.0) % 360.0
    wet = np.transpose(tmask, (1, 2, 0))[:, :, 0] > 0.5

    dst = nc.Dataset(gridt)
    try:
        T = _nemo_var(dst, "to", rec)
        S = _nemo_var(dst, "so", rec)
        eta = _nemo_var(dst, "zos", rec)
        dz = _nemo_var(dst, "e3t", rec) if "e3t" in dst.variables else None
        depth = np.asarray(dst.variables["deptht"][:], dtype=np.float64)
    finally:
        dst.close()
    if dz is None:
        dz = np.transpose(e3t0, (1, 2, 0))

    dsu = nc.Dataset(gridu)
    try:
        u_face = _nemo_var(dsu, "uo", rec)
    finally:
        dsu.close()
    dsw = nc.Dataset(gridw)
    try:
        kv_w = _nemo_var(dsw, "avm", rec)
    finally:
        dsw.close()

    # NEMO: u(i) is the face between T(i) and T(i+1), so the two faces
    # bounding T(i) are u(i-1) and u(i) -- the MIRROR of our convention.
    u_c = 0.5 * (u_face + np.roll(u_face, 1, axis=1))
    dx_e = e1t                            # e1u(i) to the east of T(i)
    dx_w = np.roll(e1t, 1, axis=1)

    # avm sits on w levels, index k being the interface ABOVE cell k, so the
    # nk-1 interfaces between centres are levels 1..nk-1.
    kv_if = kv_w[:, :, 1:]

    wet3 = np.transpose(tmask, (1, 2, 0)) > 0.5
    T = np.where(wet3, T, np.nan)
    S = np.where(wet3, S, np.nan)
    rho = _eos_fn(eos_name, rho0)(np.nan_to_num(T, nan=0.0),
                                  np.nan_to_num(S, nan=35.0),
                                  depth[None, None, :], g)
    rho = np.where(wet3, rho, np.nan)
    z_if = 0.5 * (depth[:-1] + depth[1:])
    return dict(day=None, T=T, S=S, rho=rho, eta=np.where(wet, eta, np.nan),
                u=np.where(wet3, u_c, np.nan), kv=kv_if, dz=dz,
                z_c=depth, z_if=z_if, lat=lat, lon=lon, wet=wet,
                dx_w=dx_w, dx_e=dx_e, eos=eos_name)


def terms(m, g, rho0):
    """The three force terms on T points, each (nj, ni, nk) in m/s^2."""
    pgf_baro2d = -g * _dphi_dx(np.nan_to_num(m["eta"]), m["dx_w"], m["dx_e"])
    pgf_baro = np.broadcast_to(pgf_baro2d[:, :, None], m["rho"].shape).copy()
    pprime = _hydrostatic_p_prime(m["rho"], m["dz"], rho0)
    pgf_bc = -g * _dphi_dx(pprime, m["dx_w"], m["dx_e"])
    fric = _ddz_flux(np.nan_to_num(m["u"]), m["kv"], m["z_c"], m["z_if"])
    dry = ~m["wet"]
    for a in (pgf_baro, pgf_bc, fric):
        a[dry, :] = np.nan
    return pgf_baro, pgf_bc, fric


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--legoesm-snapshot", required=True)
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--nemo-gridu", required=True)
    p.add_argument("--nemo-gridw", required=True)
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--rec", type=int, required=True,
                   help="oracle 5-day record; REC = DAY/5 - 1")
    p.add_argument("--expect-day", type=float, default=None,
                   help="fail unless the snapshot stamps this day")
    p.add_argument("--lons", default="200,220,240")
    p.add_argument("--lat-band", type=float, default=1.0)
    p.add_argument("--eos-ours", default="wright")
    p.add_argument("--eos-nemo", default="nemo_teos10")
    p.add_argument("--eos-both", default=None,
                   help="override BOTH sides with one EOS (the density-field "
                        "question, not the force-felt question)")
    p.add_argument("--rho0", type=float, default=1025.0)
    p.add_argument("--dump", default=None, help="write the profiles to .npz")
    a = p.parse_args()

    from legoesm import constants
    g = float(constants.g)
    eos_o = a.eos_both or a.eos_ours
    eos_n = a.eos_both or a.eos_nemo

    ours = _load_ours(a.legoesm_snapshot, a.expect_day, eos_o, a.rho0, g)
    nemo = _load_nemo(a.nemo_gridt, a.nemo_gridu, a.nemo_gridw, a.mesh_mask,
                      a.rec, eos_n, a.rho0, g)

    print(f"day {ours['day']:g} (snapshot stamp) against oracle record {a.rec} "
          f"= the 5-day mean ENDING on day {(a.rec + 1) * 5}")
    print(f"EOS: ours {eos_o}, NEMO {eos_n}"
          + ("  [--eos-both: this is the DENSITY-FIELD question]"
             if a.eos_both else "  [each model's own: the FORCE-FELT question]"))
    print("all terms on T points, m/s^2, + = eastward; depth positive down")
    if ours["rho"].shape[:2] != nemo["rho"].shape[:2]:
        raise SystemExit(f"frame mismatch: ours {ours['rho'].shape[:2]} vs "
                         f"NEMO {nemo['rho'].shape[:2]}")

    to = terms(ours, g, a.rho0)
    tn = terms(nemo, g, a.rho0)

    # GATE, on the terms LINEAR in a time mean only.  A five-day-mean
    # residual cannot be expected to close -- mean(Kv du/dz) is not
    # mean(Kv) d(mean u)/dz -- so the oracle's FRIC is never gated.  What must
    # hold is that the oracle's barotropic pressure gradient, which is linear
    # in its own mean sea level, is finite and of ocean magnitude.  A probe
    # that returns 10^-3 m/s^2 for it has a metric or unit error, not a
    # finding: 10^-7 is the equatorial scale.
    pb_n = _box_profile(tn[0][:, :, 0], nemo["lat"], nemo["lon"], 220.0,
                        a.lat_band)
    if not np.isfinite(pb_n) or abs(pb_n) > 1e-4:
        raise SystemExit(f"GATE FAILED: oracle barotropic PGF at 220E is "
                         f"{pb_n:.3e} m/s^2, outside any ocean scale")

    dump = {}
    for lon0 in [float(x) for x in a.lons.split(",")]:
        print(f"\n=== {lon0:g}E, |lat| <= {a.lat_band:g} deg ===")
        print(f"{'depth':>7} {'u_ours':>8} {'u_NEMO':>8} | "
              f"{'PGFbc_o':>10} {'PGFbc_n':>10} | {'FRIC_o':>10} {'FRIC_n':>10} "
              f"| {'RESID_o':>10} {'RESID_n':>10}")
        rows = {}
        for tag, m, t in (("ours", ours, to), ("nemo", nemo, tn)):
            prof = {k: _box_profile(v, m["lat"], m["lon"], lon0, a.lat_band)
                    for k, v in (("u", m["u"]), ("pgf_baro", t[0]),
                                 ("pgf_bc", t[1]), ("fric", t[2]))}
            prof["z"] = m["z_c"]
            prof["resid"] = prof["pgf_baro"] + prof["pgf_bc"] + prof["fric"]
            rows[tag] = prof
        zo, zn = rows["ours"]["z"], rows["nemo"]["z"]
        for zi in (5, 25, 50, 75, 100, 150, 200, 300, 400):
            io = int(np.argmin(np.abs(zo - zi)))
            inn = int(np.argmin(np.abs(zn - zi)))
            o, n = rows["ours"], rows["nemo"]
            print(f"{zi:7d} {o['u'][io]:8.3f} {n['u'][inn]:8.3f} | "
                  f"{o['pgf_bc'][io]:10.3e} {n['pgf_bc'][inn]:10.3e} | "
                  f"{o['fric'][io]:10.3e} {n['fric'][inn]:10.3e} | "
                  f"{o['resid'][io]:10.3e} {n['resid'][inn]:10.3e}")
        # At the CORE, which is where the jet is actually held up.
        for tag in ("ours", "nemo"):
            r = rows[tag]
            k = int(np.nanargmax(np.where(r["z"] <= 400.0, r["u"], -np.inf)))
            print(f"  core {tag:4s}: u {r['u'][k]:+.3f} m/s at {r['z'][k]:.0f} m"
                  f"   PGF_baro {r['pgf_baro'][k]:+.3e}"
                  f"  PGF_bc {r['pgf_bc'][k]:+.3e}"
                  f"  FRIC {r['fric'][k]:+.3e}"
                  f"  RESID {r['resid'][k]:+.3e}")
            dump[f"{lon0:g}_{tag}_core"] = np.array(
                [r["z"][k], r["u"][k], r["pgf_baro"][k], r["pgf_bc"][k],
                 r["fric"][k], r["resid"][k]])
        for tag in ("ours", "nemo"):
            for k, v in rows[tag].items():
                dump[f"{lon0:g}_{tag}_{k}"] = np.asarray(v)

    print("\nRESID holds advection, lateral friction AND the local "
          "acceleration; du/dt is separable across the five-day marks and "
          "MUST be removed before RESID is read. Not done here.")
    if a.dump:
        dump["day"] = np.array([ours["day"]])
        dump["rec"] = np.array([a.rec])
        np.savez(a.dump, **dump)
        print(f"wrote {a.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
