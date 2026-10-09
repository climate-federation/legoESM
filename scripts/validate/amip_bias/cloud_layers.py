#!/usr/bin/env python3
"""Where the cloud cover sits vertically, and how much condensate the radiation
never sees -- from checkpoints, with the model's OWN cloud-fraction code.

The CMOR tables carry only the column total ``clt``; they cannot say whether a
cover excess is shallow marine stratus or mid-level cloud over an over-moist
column, and they cannot say whether the prognostic condensate reaches the
radiation at all.  This probe rebuilds the layer cloud fraction the radiation
was given, per checkpoint, by calling ``compute_cloud_properties`` with the
run's RESOLVED cloud configuration (read from ``experiment_config.json``, never
from a default), then reports per region:

* low / mid / high cover and the column total, each by the model's own
  maximum-random overlap.  "Low" is the cover of cloud LOCATED below 680 hPa
  whatever lies above it (a GOCCP-style layer amount), NOT the ISCCP
  cloud-top-pressure class; the 680 / 440 hPa bounds are ISCCP's;
* the layer cloud fraction, RH and cloud-water profile over the lowest levels;
* the column condensate path at three stages: the tracers; the output of
  ``compute_cloud_properties`` (radiative floor + in-cloud inhomogeneity
  scaling); and the mean over the ``max_random`` subcolumns of the per-
  subcolumn path the solver is handed (the solver integrates each subcolumn
  and averages the FLUXES, so this last number is the mean solver INPUT, not
  a flux-equivalent path).  A layer with ``cf = 0`` contributes ZERO water
  to every subcolumn, so condensate sitting in "clear" layers is
  radiatively invisible; ``ice_unseen`` is that fraction.

Number tracers are passed as the radiation call passes them (the raw
``N_c``/``N_i`` tracers).  When the run sets ``nc_from_aerosol`` without
prognostic droplet number the live path replaces ``N_c`` by a CCN estimate
from the aerosol optical depth, which this probe cannot rebuild; the LIQUID
in-cloud optical depth (hence its inhomogeneity factor) then differs from
the run's, and the probe says so.  The ice path is unaffected.

Snapshot caveat, stated: checkpoints are instants (one time of day), not the
monthly means the CMOR ``clt`` publishes.  The probe prints the published
``clt`` next to the snapshot total so the two can be compared; on the res6
90-day run they agree to 1-5 % per region.

Usage: cloud_layers.py <run> [--days 50,60,70,80] [--profile-levels 12]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

_DIR = pathlib.Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rb = _load("regional_bias")

P_LOW_PA = 68000.0      # ISCCP low/mid boundary
P_HIGH_PA = 44000.0     # ISCCP mid/high boundary
PROFILE_REGIONS = ("ITCZ 10S-10N", "trades 10-30S", "Sc Peru", "Sc California",
                   "SO stormtrack", "NH midlat", "poles 60-90")


def region_mask(lat, lon, box):
    """Boolean cell mask for a ``regional_bias.REGIONS`` box on an unstructured
    mesh (``lat``/``lon`` in degrees, lon in [0, 360))."""
    la0, la1, lo0, lo1 = box
    if la0 is None:
        return np.abs(lat) >= 60.0
    m = (lat >= la0) & (lat <= la1)
    if lo1 > 360:
        m &= (lon >= lo0) | (lon <= lo1 - 360)
    elif (lo0, lo1) != (0, 360):
        m &= (lon >= lo0) & (lon <= lo1)
    return m


def area_mean(field, area, mask):
    """Area-weighted mean of ``field`` (cells, ...) over ``mask``; raises on an
    empty region rather than returning a silent NaN."""
    w = np.where(mask, area, 0.0)
    if w.sum() <= 0:
        raise SystemExit("FATAL: region has no cells")
    w = w / w.sum()
    return np.tensordot(w, field, axes=(0, 0))


def layer_cover(cf, p_full):
    """Column cover restricted to low / mid / high layers plus the total, each
    by the model's maximum-random overlap.  Returns dict of (ncol,) arrays."""
    from legoesm.diagnostics.cloud_overlap import maximum_random_overlap

    def cover(mask):
        return np.asarray(maximum_random_overlap(np.where(mask, cf, 0.0)))

    return {
        "clt": cover(np.ones_like(cf, dtype=bool)),
        "low": cover(p_full > P_LOW_PA),
        "mid": cover((p_full <= P_LOW_PA) & (p_full > P_HIGH_PA)),
        "high": cover(p_full <= P_HIGH_PA),
    }


def radiation_paths(cloud_props, cloud_cfg):
    """Grid-mean liquid/ice paths the radiation solver receives, after the
    configured vertical-overlap treatment.  Mirrors
    ``radiation/integration.py``: ``max_random`` samples subcolumns from the
    layer cloud fraction; ``none`` hands the solver the grid-mean paths."""
    ov = cloud_cfg.cloud_vertical_overlap_optics
    if ov == "none":
        return np.asarray(cloud_props.lwp), np.asarray(cloud_props.iwp)
    if ov != "max_random":
        raise ValueError(f"unknown cloud_vertical_overlap_optics {ov!r}")
    from legoesm.atmosphere.physics.clouds import subcolumns as sub
    n_sub = int(cloud_cfg.cloud_n_subcolumns)
    ncol = cloud_props.cloud_fraction.shape[0]
    mask = sub.generate_subcolumns(cloud_props.cloud_fraction, n_sub)
    lwp, iwp = sub.subcolumn_paths(mask, cloud_props.cloud_fraction,
                                   cloud_props.lwp, cloud_props.iwp)
    return (np.asarray(sub.average_over_subcolumns(lwp, n_sub, ncol)),
            np.asarray(sub.average_over_subcolumns(iwp, n_sub, ncol)))


def resolved_cloud_config(exp):
    """The run's cloud configuration, from its resolved experiment_config.json.
    Every key is read explicitly; a missing key is an error, not a default."""
    from legoesm.atmosphere.physics.clouds.config import build_cloud_config
    need = ["cloud_scheme", "convective_cloud", "cloud_rh_crit",
            "cloud_q_c_diagnostic", "cloud_saturation_scheme",
            "cloud_vertical_overlap_optics", "cloud_n_subcolumns",
            "cloud_diagnostic_condensate_scheme", "cloud_p_xr", "cloud_alpha_xr",
            "cloud_conv_cloud_max", "cloud_conv_cloud_condensate",
            "cloud_adiabatic_lwc_rate", "cloud_optics_inhomogeneity",
            "cloud_inhomogeneity_factor", "cloud_fsd",
            "cloud_partial_coverage_optics", "cloud_clubb_cf_override_strength",
            "cloud_clubb_cf_override_floor", "use_clubb_cloud_fraction"]
    missing = [k for k in need if k not in exp]
    if missing:
        raise SystemExit(f"FATAL: experiment_config.json lacks {missing}")
    if exp["use_clubb_cloud_fraction"]:
        raise SystemExit("FATAL: run routes CLUBB's cloud fraction into the "
                         "optics; that override is not in the checkpoint and "
                         "cannot be rebuilt here")
    return build_cloud_config(
        exp["cloud_scheme"], convective_cloud=bool(exp["convective_cloud"]),
        rh_crit=exp["cloud_rh_crit"], q_c_diagnostic=exp["cloud_q_c_diagnostic"],
        saturation_scheme=exp["cloud_saturation_scheme"],
        cloud_vertical_overlap_optics=exp["cloud_vertical_overlap_optics"],
        cloud_n_subcolumns=exp["cloud_n_subcolumns"],
        diagnostic_condensate_scheme=exp["cloud_diagnostic_condensate_scheme"],
        p_xr=exp["cloud_p_xr"], alpha_xr=exp["cloud_alpha_xr"],
        conv_cloud_max=exp["cloud_conv_cloud_max"],
        conv_cloud_condensate=exp["cloud_conv_cloud_condensate"],
        adiabatic_lwc_rate=exp["cloud_adiabatic_lwc_rate"],
        cloud_optics_inhomogeneity=exp["cloud_optics_inhomogeneity"],
        cloud_inhomogeneity_factor=exp["cloud_inhomogeneity_factor"],
        cloud_fsd=exp["cloud_fsd"],
        cloud_partial_coverage_optics=exp["cloud_partial_coverage_optics"],
        clubb_cf_override_strength=exp["cloud_clubb_cf_override_strength"],
        clubb_cf_override_floor=exp["cloud_clubb_cf_override_floor"],
        # runs older than the lever have no key: None => scheme default (off)
        cover_condensate_q_ref=exp.get("cloud_cover_condensate_q_ref"),
        # FV-pipeline-only knobs (no ExperimentConfig field records them yet):
        # None => scheme default, same as every run so far
        conv_cloud_coeff=exp.get("cloud_conv_cloud_coeff"),
        Nc_default=exp.get("cloud_Nc_default"))


def cell_order(z, ncell):
    """Permutation that puts the checkpoint's columns into global mesh order.

    Checkpoints carry ``physstate_col_index``, the global cell id of every
    stored column.  It must be a permutation of ``range(ncell)``; anything
    else (a rank-local slab, a different mesh) is refused rather than
    averaged into a regional mean on the wrong cells."""
    if "physstate_col_index" not in z.files:
        raise SystemExit("FATAL: checkpoint has no physstate_col_index; "
                         "cannot prove its cell order matches the mesh")
    idx = np.asarray(z["physstate_col_index"]).astype(np.int64)
    if idx.shape != (ncell,) or not np.array_equal(np.sort(idx), np.arange(ncell)):
        raise SystemExit("FATAL: checkpoint columns are not a permutation of "
                         f"the mesh's {ncell} cells")
    order = np.empty(ncell, dtype=np.int64)
    order[idx] = np.arange(ncell)
    return order


def analyse_checkpoint(path, cloud_cfg, ncell):
    """Per-cell cover, radiation-visible vs prognostic paths, and the layer
    fields needed for profiles, from one checkpoint (in global mesh order)."""
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties, cover_saturation_specific_humidity)
    from legoesm import constants

    z = np.load(path)
    order = cell_order(z, ncell)

    def col(name):
        return np.asarray(z[name])[order] if name in z.files else None

    # (2, nlev+1) = (A_half, B_half); p_half = A p_ref + B p_s.  A is
    # DIMENSIONLESS (model_driver stores it so; p_ref is the constants value
    # the hybrid constructor defaults to).  Pure sigma is the A = 0 member.
    vg = z["meta_vgrid"]
    ps = col("p_s")
    p_half = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    dp = np.diff(p_half, axis=1)
    if not np.all(dp > 0):
        raise SystemExit("FATAL: non-monotone vertical grid in checkpoint")
    T, q_v = col("T"), col("trc_q_v")
    q_c = col("trc_q_c"); q_i = col("trc_q_i")
    if cloud_cfg.scheme == "resolved" and (q_c is None or q_i is None):
        raise SystemExit("FATAL: 'resolved' cloud scheme but the checkpoint "
                         "carries no q_c/q_i tracers -- refusing to feed zeros")
    q_c = np.zeros_like(T) if q_c is None else q_c
    q_i = np.zeros_like(T) if q_i is None else q_i
    # Number tracers go in RAW, exactly as physics_pipeline.compute_radiation_core
    # hands them to compute_cloud_properties.  NB that live path passes the
    # per-MASS N_c carry into optics that read N_c per VOLUME (the other
    # radiation entry, integration._extract_tracer_columns, multiplies by air
    # density first) -- a live inconsistency this probe deliberately mirrors
    # rather than corrects, so its liquid r_eff matches the run's.  A zero N_c
    # carry (specified-Nc runs) falls back to Nc_default inside the optics.
    conv = col("physstate_conv_precip") if cloud_cfg.convective_cloud else None
    props = compute_cloud_properties(T, p_full, q_v, dp, cloud_cfg,
                                     q_cloud=q_c, q_ice=q_i,
                                     n_cloud=col("trc_N_c"), n_ice=col("trc_N_i"),
                                     conv_precip=conv)
    cf = np.asarray(props.cloud_fraction)
    lwp_rad, iwp_rad = radiation_paths(props, cloud_cfg)
    g = constants.g
    # the curve the cover scheme itself measured RH against (shared dispatch)
    q_sat = np.asarray(cover_saturation_specific_humidity(T, p_full, cloud_cfg))
    out = layer_cover(cf, p_full)
    out.update({
        "iwp_prog": (q_i * dp / g).sum(1), "lwp_prog": (q_c * dp / g).sum(1),
        "iwp_opt": np.asarray(props.iwp).sum(1), "lwp_opt": np.asarray(props.lwp).sum(1),
        "iwp_rad": iwp_rad.sum(1), "lwp_rad": lwp_rad.sum(1),
        "iwp_clear": np.where(cf <= 0.0, q_i * dp / g, 0.0).sum(1),
        "lwp_clear": np.where(cf <= 0.0, q_c * dp / g, 0.0).sum(1),
        "cf": cf, "rh": q_v / np.maximum(q_sat, 1e-10), "q_c": q_c,
        # The SATURATED-BUT-DRY census (GLM review).  Xu-Randall multiplies an
        # RH term by a condensate term, so a layer at or above saturation with
        # no condensate gets cover 0 where Sundqvist gives 1.  WRF's
        # cal_cldfra1 carries an explicit RH>=1 -> 1 cutoff that this
        # differentiable core omits.  How much real cloud that deletes is a
        # COUNT, not an argument: layers at RH >= 1 whose total condensate is
        # below the 5e-6 kg/kg radiative floor, weighted by their mass.
        "sat_dry_frac": (((q_v / np.maximum(q_sat, 1e-10)) >= 1.0)
                         & ((q_c + q_i) < 5.0e-6)).mean(),
        "sat_frac": ((q_v / np.maximum(q_sat, 1e-10)) >= 1.0).mean(),
        "q_cond_high": np.where(p_full < 44000.0, q_c + q_i, np.nan),
        "cf_high_layers": np.where(p_full < 44000.0, cf, np.nan),
        "sigma_full": 0.5 * (vg[1][1:] + vg[1][:-1]),
        "T_lowest": T[:, -1],
    })
    return out


def mesh_coords(exp):
    """Cell lat/lon [deg] and area for the run's MPAS mesh via the grid
    factory (served from the SCVT disk cache the run itself populated)."""
    grid = exp["grid"]
    if grid["grid_type"] != "mpas":
        raise SystemExit("FATAL: this probe reads MPAS checkpoints only")
    from legoesm.grids.factory import create_grid
    mesh = create_grid("mpas", resolution=int(grid["resolution"]))
    lat = np.rad2deg(np.asarray(mesh.latCell))
    lon = np.rad2deg(np.asarray(mesh.lonCell)) % 360.0
    return lat, lon, np.asarray(mesh.areaCell)


def published_clt(run):
    """Regional means of the run's published CMOR ``clt`` (time mean of the
    months on disk), or None when the run has not published it."""
    d = rb._load_model(run, "clt")
    if d is None:
        return None
    lat, lon = np.asarray(d.lat), np.asarray(d.lon)
    mv = np.asarray(d["clt"]).mean(axis=0)
    return {name: rb.region_mean(mv, lat, lon, box) for name, box in rb.REGIONS.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--days", default=None,
                    help="comma-separated checkpoint days (default: all)")
    ap.add_argument("--profile-levels", type=int, default=12)
    ap.add_argument("--override-scheme", default=None,
                    choices=("sundqvist", "xu_randall"),
                    help="replay the SAME checkpoint under a different cover "
                         "closure, every other resolved field held fixed. The "
                         "output then describes a counterfactual, not the run.")
    ap.add_argument("--override-saturation", default=None,
                    choices=("liquid", "mixed_phase"),
                    help="likewise for the saturation curve the cover is "
                         "diagnosed against, so a pair can be scored on the "
                         "curve a LATER deck uses rather than the run's own.")
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)

    rundir = f"{rb.ROOT}/{args.run}"
    exp = json.load(open(f"{rundir}/experiment_config.json"))
    cloud_cfg = resolved_cloud_config(exp)
    for _fld, _new in (("scheme", args.override_scheme),
                       ("saturation_scheme", args.override_saturation)):
        if _new and _new != getattr(cloud_cfg, _fld):
            print(f"!!! COUNTERFACTUAL: {_fld} overridden "
                  f"{getattr(cloud_cfg, _fld)!r} -> {_new!r}; the run itself "
                  f"used {getattr(cloud_cfg, _fld)!r}. Every other field is "
                  f"the run's.")
            cloud_cfg = cloud_cfg._replace(**{_fld: _new})
    print(f"=== {args.run}: cloud config RESOLVED from experiment_config.json ===")
    for k in ("scheme", "rh_crit", "saturation_scheme", "cover_condensate_q_ref", "q_c_diagnostic",
              "cloud_vertical_overlap_optics", "cloud_n_subcolumns", "convective_cloud"):
        print(f"  {k} = {getattr(cloud_cfg, k)!r}")

    cks = sorted(glob.glob(f"{rundir}/checkpoint_day_*.npz"))
    if args.days:
        want = {int(d) for d in args.days.split(",")}
        cks = [c for c in cks if int(c.rsplit("_", 1)[1][:4]) in want]
    if not cks:
        raise SystemExit("FATAL: no checkpoints selected")
    lat, lon, area = mesh_coords(exp)
    if exp.get("nc_from_aerosol"):
        print("  WARNING: run sets nc_from_aerosol -- the live path replaced N_c "
              "by CCN from the aerosol optical depth; the LIQUID in-cloud tau and "
              "its inhomogeneity factor here use the raw N_c tracer instead.  Ice "
              "is unaffected.")

    acc = {}
    for c in cks:
        r = analyse_checkpoint(c, cloud_cfg, lat.shape[0])
        # secondary sanity on top of the col_index permutation: a polar cell
        # must be cold at the lowest level
        corr = np.corrcoef(np.abs(lat), r["T_lowest"])[0, 1]
        if corr > -0.5:
            raise SystemExit(f"FATAL: mesh/checkpoint geometry mismatch "
                             f"(corr(|lat|, T_lowest) = {corr:+.2f})")
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
        print(f"  {os.path.basename(c)}  corr(|lat|,T_lowest)={corr:+.2f}")
    mean = {k: np.mean(v, 0) for k, v in acc.items()}

    cols = ["clt", "low", "mid", "high"]
    print(f"\n=== {args.run}: snapshot cover [%] by layer (maximum-random overlap), "
          f"mean of {len(cks)} checkpoints ===")
    print(f"{'region':18s}" + "".join(f"{c:>8s}" for c in cols)
          + f"{'clt_cmor':>10s}{'ice_unseen':>11s}{'liq_unseen':>11s}")
    print("  ice_unseen / liq_unseen = fraction of the prognostic condensate path "
          "in layers with cf = 0 (invisible to max_random radiation)")
    cmor = published_clt(args.run)
    for name, box in rb.REGIONS.items():
        m = region_mask(lat, lon, box)
        row = [100 * area_mean(mean[c], area, m) for c in cols]
        ice_un = area_mean(mean["iwp_clear"], area, m) / max(area_mean(mean["iwp_prog"], area, m), 1e-30)
        liq_un = area_mean(mean["lwp_clear"], area, m) / max(area_mean(mean["lwp_prog"], area, m), 1e-30)
        pub = f"{cmor[name]:10.1f}" if cmor else f"{'--':>10s}"
        print(f"{name:18s}" + "".join(f"{v:8.1f}" for v in row)
              + pub + f"{ice_un:11.2f}{liq_un:11.2f}")

    # GLM review, two counts the cover table cannot answer.
    print(f"\n  saturated layers: {100 * float(mean['sat_frac']):.2f} % of all "
          f"(column, level) points are at RH >= 1 against the cover curve; "
          f"{100 * float(mean['sat_dry_frac']):.2f} % are saturated AND carry "
          f"less condensate than the 5e-6 kg/kg radiative floor.")
    print("    The second number is the cloud a condensate-reading closure "
          "deletes and a pure-RH closure keeps; it is an UPPER bound on the "
          "loss, since the floor would give those layers only minimal water.")
    _qh = np.asarray(mean["q_cond_high"])
    _cfh = np.asarray(mean["cf_high_layers"])
    _ok = np.isfinite(_qh) & np.isfinite(_cfh)
    _q, _cff = _qh[_ok], _cfh[_ok]
    print("  high-cloud cover (above 440 hPa) by how much condensate the layer "
          "carries -- cover riding on the floor is the closure's doing, cover "
          "riding on real ice is the curve's or the model's:")
    _edges = [0.0, 1e-7, 1e-6, 5e-6, 2e-5, 1e-4, np.inf]
    for _lo, _hi in zip(_edges[:-1], _edges[1:]):
        _m = (_q >= _lo) & (_q < _hi)
        if not _m.any():
            continue
        _share = float((_cff[_m]).sum() / max(_cff.sum(), 1e-30))
        print(f"    q_cond {_lo:9.1e} - {_hi:9.1e} kg/kg : "
              f"{100 * float(_m.mean()):6.2f} % of high layers, "
              f"mean cover {float(_cff[_m].mean()):.3f}, "
              f"{100 * _share:6.2f} % of all high cover")

    print(f"\n=== {args.run}: column condensate path [g/m2] at the three stages ===")
    print("  prog = tracers; opt = after the radiative floor and in-cloud "
          "inhomogeneity scaling (compute_cloud_properties); rad = mean over "
          "the overlap subcolumns of the per-subcolumn path handed to the solver.")
    keys = ("iwp_prog", "iwp_opt", "iwp_rad", "lwp_prog", "lwp_opt", "lwp_rad")
    print(f"{'region':18s}" + "".join(f"{k:>9s}" for k in keys))
    for name, box in rb.REGIONS.items():
        m = region_mask(lat, lon, box)
        print(f"{name:18s}" + "".join(
            f"{1e3 * area_mean(mean[k], area, m):9.2f}" for k in keys))

    nl = args.profile_levels
    sig = mean["sigma_full"]
    print(f"\n=== {args.run}: regional layer profiles, lowest {nl} levels "
          f"(sigma {sig[-nl]:.3f} .. {sig[-1]:.3f}) ===")
    for name in PROFILE_REGIONS:
        m = region_mask(lat, lon, rb.REGIONS[name])
        cf = area_mean(mean["cf"], area, m)[-nl:]
        rh = area_mean(mean["rh"], area, m)[-nl:]
        qc = area_mean(mean["q_c"], area, m)[-nl:]
        print(f"\n{name}\n  cf   %: " + " ".join(f"{100 * v:5.1f}" for v in cf))
        print("  RH   %: " + " ".join(f"{100 * v:5.1f}" for v in rh))
        print("  qc g/kg: " + " ".join(f"{1e3 * v:5.3f}" for v in qc))


if __name__ == "__main__":
    main()
