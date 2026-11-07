#!/usr/bin/env python
"""Remap an MPAS sigma-lane restart checkpoint onto a tropopause-refined sigma
grid with the same level count, so the driver can restart on the new vertical
grid (its ``meta_vgrid`` guard rejects a mismatched grid).  Interfaces are
sigma_half * p_s, so the old->new remap weights are column-independent and
mass-conserving per column (p_s unchanged).  Mass-like fields use the FV3 PPM
remap (``core.fv3_mapz``); per-level profiles are interpolated in sigma.
Provenance goes into ``remap_meta``; conservative fields are drift-checked.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import argparse
import hashlib
import json
import sys

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from legoesm.core.fv3_mapz import map1_ppm, map_scalar, pad1, unpad1
from legoesm.grids.vertical import create_sigma_coordinate

CHUNK = 8192
T_FLOOR_K = 150.0          # scalar_profile's extremum-flattening threshold for T
# (an FV3 T_min-style value the atmosphere never reaches, NOT a guaranteed floor)
# (field, iv, q_min): q_min None -> map1_ppm (cs_profile, signed fields, no
# qmin clauses); q_min set -> map_scalar (scalar_profile).  FV3 remaps T with
# map_scalar(iv=1, T_min), winds with map1_ppm(iv=-1), tracers with iv=0.
SIGNED = (("T", 1, T_FLOOR_K), ("u", -1, None))
POS = ("trc_q_v", "trc_q_c", "trc_q_i", "trc_q_r", "trc_q_s", "trc_q_g",
       "trc_N_c", "trc_N_i", "trc_N_r", "physstate_aerosol_number",
       "physstate_tke", "physstate_qke", "physstate_cloud_fraction")
PROF = ("physstate_conv_prog_profile", "physstate_rad_heating")


def err(msg):
    print(f"error: {msg}", file=sys.stderr)
    return 1


def remap_cons(field, s_old, s_new, iv, kord, q_min):
    """Conservative monotone remap of an (im, km) field between sigma interfaces."""
    km, kn = s_old.size - 1, s_new.size - 1
    out = np.empty((field.shape[0], kn), dtype=np.float64)
    for a in range(0, field.shape[0], CHUNK):
        f = jnp.asarray(np.ascontiguousarray(field[a:a + CHUNK]), dtype=jnp.float64)
        im = f.shape[0]
        pe1 = pad1(jnp.asarray(np.repeat((s_old * 1.0e5)[None, :], im, 0)))
        pe2 = pad1(jnp.asarray(np.repeat((s_new * 1.0e5)[None, :], im, 0)))
        if q_min is None:
            q2 = map1_ppm(pe1, pad1(f), pe2, km, kn, iv, kord)
        else:
            q2 = map_scalar(pe1, pad1(f), pe2, km, kn, iv, kord, q_min)
        out[a:a + im] = np.asarray(unpad1(q2))
    return out


def interp_prof(field, sf_old, sf_new):
    """Linear interpolation of (im, km) profiles onto the new full sigma levels."""
    out = np.empty((field.shape[0], sf_new.size), dtype=np.float64)
    for k, s in enumerate(sf_new):
        j = int(np.clip(np.searchsorted(sf_old, s), 1, sf_old.size - 1))
        w = float(np.clip((s - sf_old[j - 1]) / (sf_old[j] - sf_old[j - 1]), 0.0, 1.0))
        out[:, k] = (1.0 - w) * field[:, j - 1] + w * field[:, j]
    return out


def report(name, fb, fa, s_old, s_new):
    """Per-column inventory check (no cross-column cancellation); returns the
    worst relative change, scaled by the mean absolute column inventory so
    signed winds and empty columns are handled; NaN anywhere -> inf."""
    ib = np.sum(fb * np.diff(s_old)[None, :], axis=1, dtype=np.float64)
    ia = np.sum(fa * np.diff(s_new)[None, :], axis=1, dtype=np.float64)
    scale = max(float(np.mean(np.abs(ib))), 1e-300)
    rel = float(np.max(np.abs(ia - ib))) / scale
    if not (np.isfinite(fa).all() and np.isfinite(rel)):
        rel = float("inf")
    print(f"{name}: inv {ib.sum():.12e} -> {ia.sum():.12e} worst-column rel {rel:.3e} | "
          f"min/max {fb.min():.6e}/{fb.max():.6e} -> {fa.min():.6e}/{fa.max():.6e}")
    return rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--tropopause-refine", required=True, type=float)
    ap.add_argument("--sigma-refine", type=float, default=0.12,
                    help="centre (sigma) of the density bump; 0.95 for a boundary-layer refinement")
    ap.add_argument("--sigma-refine-width", type=float, default=0.45,
                    help="log-sigma half-width of the bump")
    ap.add_argument("--kord", type=int, default=9)
    args = ap.parse_args(argv)
    if not args.dst.endswith(".npz"):
        return err("--dst must end with .npz")
    if os.path.exists(args.dst):
        return err(f"refusing to overwrite existing dst {args.dst}")
    with np.load(args.src, allow_pickle=True) as z:
        d = {k: z[k] for k in z.files}
    if "meta_vgrid" not in d:
        return err("meta_vgrid absent: not a sigma-lane checkpoint")
    mv = np.asarray(d["meta_vgrid"], dtype=np.float64)
    if mv.ndim != 2 or mv.shape[0] != 2 or not np.all(mv[0] == 0.0):
        return err("A_half is not all zero: not a sigma-lane checkpoint")
    nlev = mv.shape[1] - 1
    s_old = np.ascontiguousarray(mv[1])
    if not (np.isfinite(s_old).all() and np.all(np.diff(s_old) > 0)
            and abs(s_old[-1] - 1.0) < 1e-12):
        return err("meta_vgrid B_half is not a finite, increasing sigma ending at 1")
    names = [n for n, _, _ in SIGNED] + list(POS) + list(PROF)
    for name in names:
        if name not in d:
            return err(f"missing field {name}")
        if np.asarray(d[name]).shape[-1] != nlev:
            return err(f"{name} width != nlev {nlev}")
        if not np.isfinite(np.asarray(d[name], dtype=np.float64)).all():
            return err(f"{name} contains non-finite values; refusing to remap or copy")
    for name in ("physstate_clubb_moments", "physstate_gwd_spectrum"):
        if name in d and np.asarray(d[name]).shape[-1] in (nlev, nlev + 1):
            return err(f"{name} carries a level-resolved state this tool does not remap")
    coord = create_sigma_coordinate(nlev, tropopause_refine=args.tropopause_refine,
                                    sigma_refine=args.sigma_refine,
                                    refine_width=args.sigma_refine_width,
                                    dtype=jnp.float64)
    s_new = np.asarray(coord.sigma_half, dtype=np.float64)
    sf_old = 0.5 * (s_old[:-1] + s_old[1:])
    sf_new = np.asarray(coord.sigma_full, dtype=np.float64)
    meta = np.array(json.dumps({
        "src": str(args.src),
        "sha256": hashlib.sha256(open(args.src, "rb").read()).hexdigest(),
        "tropopause_refine": args.tropopause_refine, "kord": args.kord,
        "sigma_refine": args.sigma_refine, "sigma_refine_width": args.sigma_refine_width,
        "sigma_half_old": s_old.tolist(), "sigma_half_new": s_new.tolist()}))
    if np.all(np.abs(s_new - s_old) <= 1e-12):
        print("new sigma_half equals old within 1e-12: copying every field unchanged")
        d["remap_meta"] = meta
        np.savez(args.dst, **d)
        print(f"wrote {args.dst}")
        return 0
    worst = 0.0
    for name, iv, q_min in list(SIGNED) + [(n, 0, 0.0) for n in POS]:
        f0 = np.asarray(d[name])
        fb = f0.astype(np.float64)
        fa = remap_cons(fb, s_old, s_new, iv, args.kord, q_min)
        if name == "physstate_cloud_fraction":
            if not np.isfinite(report(name, fb, fa, s_old, s_new)):
                worst = float("inf")
            fa = np.clip(fa, 0.0, 1.0)       # bounded field: the clip may cost ~1e-4
        else:
            fa = fa.astype(f0.dtype)         # check what will be STORED
            worst = max(worst, report(name, fb, fa.astype(np.float64), s_old, s_new))
        d[name] = fa.astype(f0.dtype, copy=False)
    for name in PROF:
        f0 = np.asarray(d[name])
        fa = interp_prof(f0.astype(np.float64), sf_old, sf_new).astype(f0.dtype, copy=False)
        report(name, f0.astype(np.float64), fa.astype(np.float64), s_old, s_new)
        if not np.isfinite(fa).all():
            worst = float("inf")
        d[name] = fa
    d["meta_vgrid"] = np.stack([np.zeros(nlev + 1), s_new])
    if not worst <= 1e-10:
        return err(f"worst column inventory change {worst:.3e} > 1e-10 (or NaN); nothing written")
    d["remap_meta"] = meta                   # overwrite a prior remap's provenance
    np.savez(args.dst, **d)
    print(f"wrote {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
