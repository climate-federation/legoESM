#!/usr/bin/env python
"""Canonical committed probe for the ``ssh_nxt / div_hor`` gate row
(``fidelity_bar_gate.py``).

#1492 evidence audit: this row's cited script ``probe_1226_r2_item3_
sshnxt.py`` was NEVER committed (git log --all --diff-filter=A returns zero
commits) -- the row's own PROVENANCE_SCRIPT entry says so explicitly
("cited, never committed"). This is the highest-priority gap in that audit:
the row's classify()-time value (ratio=1.000004, |ratio-1|=4e-6 > BAR_RATIO_
EPS=1e-6) keeps it at DEBT under classify(), but its note's "conditioning,
not a defect" verdict is exactly the kind of "climate-exonerated by
construction" argument that must rest on a real, re-runnable instrument, not
a phantom filename.

This probe rebuilds BOTH halves of that verdict against the CONFIRMED-PRESENT
dump ``sshnxt_dump_hdiv.bin`` on RUN_GDB (the column-integrated ``ssh(Kaa) =
ssh(Kbb) - dt*SUM_k e3t(Kmm)*hdiv(k)`` identity against NEMO's OWN
``sshnxt_dump_ssh_after.bin`` is already proven, committed, and passing in
``ww_inheritance_walk.py``'s self-check 2 -- not re-derived a second time
here):

  (1) the row's own aggregate tuple: legoESM's own ``divergence_cgrid``
      output vs NEMO's dumped 3-D ``hdiv`` (``sshnxt_dump_hdiv.bin``), pooled
      FLAT across all wet T-columns x levels (n=9920*35=347200, matching the
      row's own recorded n exactly) -- NOT depth-summed; the depth-summed
      aggregate is a completely different, ill-conditioned number (Part 2b).
  (2) the amplification statistic that justifies treating this as
      CONDITIONING rather than a transcription defect: per-level hdiv
      (pre-depth-sum) vs NEMO's own dumped hdiv3d at every sampled level, and
      the per-column max-partial-cumsum / final-cumsum ratio across wet
      T-columns (median/p90/max), reusing the exact 32-level DINO water
      column and the same-e3t resum construction as
      ``scripts/tmp/_probe_sshnxt_divhor_localize.py`` (never committed;
      this script supersedes it as the row's canonical, committed probe).

Ported from the reference bridge machinery (``coverage_rows_measure.py``):
kept the guards this row's investigation actually used --
``require_fp64``/``require_explicit_e3t_mode`` (LEGOESM_NEMO_E3T=both is
REQUIRED, not optional, per the row's own "[e3t=both ...]" citation),
``metric_convention`` passed EXPLICITLY (the bridge's own default is
'exact', a harness gap the row's note calls out at HEAD c8e5d305b -- this
script does not silently fall back to it), and the SAME dump byte layout
(``_load_haloed``, little-endian (nlev,jpj,jpi) stream, halo-stripped to the
bridged interior shape). Dropped nothing else framework-wise; this row's
own investigation used exactly this machinery, not a bespoke solver.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both LEGOESM_METRIC_CONVENTION=nemo_isotropic \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/sshnxt_divhor_canonical.py

Pass ``LEGOESM_METRIC_CONVENTION=exact`` to reproduce the pre-#1226-option
baseline tuple (corr=1.000000, |x|ratio=0.999991) instead.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sibling(name: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, os.path.join(_HERE, name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_cov = _load_sibling("coverage_rows_measure.py", "_coverage_rows_measure")
_read_dims = _cov._read_dims
_load_haloed = _cov._load_haloed
RUN_DIR = _cov.RUN_DIR
RESTART = _cov.RESTART
# coverage_rows_measure.py is already dump_lane-wired (#1455); reuse its
# already-executed dump_lane module rather than re-importing it (RUN_DIR/
# RESTART above already come from it, so this probe is already lane-
# switchable through that sibling -- verified, not duplicated).
dump_lane = _cov._dump_lane

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid  # noqa: E402
from legoesm.ocean.experiments.dino import dino_config_for_recipe  # noqa: E402
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart  # noqa: E402
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import (  # noqa: E402
    require_explicit_e3t_mode, require_fp64,
)
from legoesm.ocean.vertical import compute_layer_thickness  # noqa: E402


def build_state():
    """Bridge the dump_lane-selected run's nit000 restart, fp64, explicit e3t + metric."""
    e3t_mode = require_explicit_e3t_mode(context="sshnxt_divhor_canonical")
    set_policy(PrecisionPolicy.fp64())

    metric = os.environ.get("LEGOESM_METRIC_CONVENTION", "auto")
    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    print(f"{dump_lane.LANE} dims: jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls}  "
          f"LEGOESM_NEMO_E3T={e3t_mode}  LEGOESM_METRIC_CONVENTION={metric}")

    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    print(f"  grid.gphit.dtype={np.asarray(grid.gphit).dtype}  "
          f"restart T.dtype={np.asarray(now.T).dtype}  (want float64 both)")

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                      metric_convention=metric, omega=cfg.omega)

    require_fp64(br.z_coord, br.state.T.data, br.state.S.data, br.geometry.dx_u,
                 context="sshnxt_divhor_canonical.build_state")
    print(f"  br.state.T.data.dtype={br.state.T.data.dtype}  "
          f"br.geometry.dx_u.dtype={br.geometry.dx_u.dtype}")

    tmask2d = np.asarray(grid.tmask[..., 0]) > 0.5
    return dict(jpi=jpi, jpj=jpj, hls=hls, grid=grid, br=br, cfg=cfg, tmask2d=tmask2d)


def measure_row_tuple(st) -> dict:
    """Part (1): the row's own aggregate corr/|x|ratio tuple.

    METHODOLOGY (recovered empirically, since the citing script was never
    committed): the row's recorded n=347200 = 9920 wet T-columns x 35 levels
    -- i.e. the FLAT 3-D per-level ``hdiv`` field (legoESM's own
    ``divergence_cgrid`` output vs NEMO's dumped ``sshnxt_dump_hdiv.bin``,
    each level's cells pooled together), NOT the depth-summed ``zhdiv``. The
    depth-summed comparison (this script's Part 2b) gives a WILDLY different,
    ill-conditioned aggregate (corr~0.25, |x|ratio~2.5) -- that is the
    conditioning-amplification story documented in the row's own note, and
    is reported separately below, never as this row's headline tuple.
    """
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    br, tmask2d = st["br"], st["tmask2d"]

    hdiv3d = _load_haloed(os.path.join(RUN_DIR, "sshnxt_dump_hdiv.bin"), jpi, jpj, hls)
    print(f"hdiv3d shape={hdiv3d.shape} dtype={hdiv3d.dtype}")

    u = np.asarray(br.state.u.data)
    v = np.asarray(br.state.v.data)
    div_lego_3d = np.asarray(
        divergence_cgrid(u, v, br.geometry,
                          u_mask=br.state.u_mask.data, v_mask=br.state.v_mask.data))

    nlev = min(div_lego_3d.shape[-1], hdiv3d.shape[-1])
    tmask3 = np.repeat(tmask2d[:, :, None], nlev, axis=2)

    print("\n--- Part 1: row's own tuple (FLAT 3-D per-level hdiv, n=cols*levels) ---")
    lo = div_lego_3d[:, :, :nlev][tmask3]
    ne = hdiv3d[:, :, :nlev][tmask3]
    corr = float(np.corrcoef(lo, ne)[0, 1])
    ratio_abs = float(np.abs(lo).sum() / np.abs(ne).sum())
    print(f"  ssh_nxt / div_hor (flat 3-D hdiv)  n={lo.size}  corr={corr:.8f}  "
          f"|x|ratio={ratio_abs:.8f}")

    min_water_col = getattr(getattr(st["cfg"], "barotropic", None),
                             "min_water_column_m", None)
    e3t_now = np.asarray(compute_layer_thickness(
        br.state.eta.data, br.state.H_bathy.data, br.z_coord,
        min_water_column_m=min_water_col))
    zhdiv_nemo = (e3t_now[:, :, :nlev] * hdiv3d[:, :, :nlev]).sum(axis=-1)
    zhdiv_lego = (e3t_now * div_lego_3d).sum(axis=-1)

    return dict(corr=corr, ratio_abs=ratio_abs, n=int(lo.size),
                zhdiv_lego=zhdiv_lego, zhdiv_nemo=zhdiv_nemo,
                hdiv3d=hdiv3d, e3t_now=e3t_now, div_lego_3d=div_lego_3d)


def measure_conditioning(st, part1: dict) -> dict:
    """Part (2): per-level hdiv exactness + depth-sum amplification statistic."""
    tmask2d = st["tmask2d"]
    hdiv3d, e3t_now, div_lego_3d = part1["hdiv3d"], part1["e3t_now"], part1["div_lego_3d"]
    nlev = min(e3t_now.shape[-1], hdiv3d.shape[-1])

    print("\n--- Part 2a: per-level hdiv (pre-depth-sum) exactness ---")
    levels = sorted(set(np.linspace(0, nlev - 1, 8).round().astype(int).tolist()))
    per_level_ratios = []
    for k in levels:
        wet_k = tmask2d & np.isfinite(div_lego_3d[:, :, k]) & np.isfinite(hdiv3d[:, :, k])
        if wet_k.sum() == 0:
            continue
        lo = div_lego_3d[:, :, k][wet_k]
        ne = hdiv3d[:, :, k][wet_k]
        corr = float(np.corrcoef(lo, ne)[0, 1]) if lo.std() > 0 and ne.std() > 0 else float("nan")
        abs_ne = float(np.abs(ne).sum())
        ratio = float(np.abs(lo).sum() / abs_ne) if abs_ne > 0 else float("nan")
        per_level_ratios.append((k, corr, ratio, int(wet_k.sum())))
        print(f"  level k={k:2d}  n={int(wet_k.sum()):5d}  corr={corr:.6f}  |x|ratio={ratio:.6f}")

    print("\n--- Part 2b: same-e3t resum -- depth-sum conditioning amplification ---")
    # Same-e3t resum: legoESM's OWN e3t times BOTH legoESM's and NEMO's
    # per-level hdiv, isolating the sum operator from any e3t/metric diff.
    zhdiv_lego_own_e3t = (e3t_now[:, :, :nlev] * div_lego_3d[:, :, :nlev]).sum(axis=-1)
    zhdiv_nemo_same_e3t = (e3t_now[:, :, :nlev] * hdiv3d[:, :, :nlev]).sum(axis=-1)
    wet = tmask2d & np.isfinite(zhdiv_lego_own_e3t) & np.isfinite(zhdiv_nemo_same_e3t)
    corr_resum = float(np.corrcoef(zhdiv_lego_own_e3t[wet], zhdiv_nemo_same_e3t[wet])[0, 1])
    abs_ne = float(np.abs(zhdiv_nemo_same_e3t[wet]).sum())
    ratio_resum = float(np.abs(zhdiv_lego_own_e3t[wet]).sum() / abs_ne)
    print(f"  same-e3t resum: corr={corr_resum:.6f}  |x|ratio={ratio_resum:.6f}  "
          f"n_wet={int(wet.sum())}")

    print("\n--- Part 2c: per-column cumulative-sum cancellation amplification ---")
    partial = np.cumsum(e3t_now[:, :, :nlev] * div_lego_3d[:, :, :nlev], axis=-1)
    max_partial = np.max(np.abs(partial), axis=-1)
    final = np.abs(partial[:, :, -1])
    wet_cols = tmask2d & np.isfinite(final) & (final > 0)
    amp = max_partial[wet_cols] / np.maximum(final[wet_cols], 1e-300)
    n_cols = int(wet_cols.sum())
    med_amp = float(np.median(amp))
    p90_amp = float(np.percentile(amp, 90))
    max_amp = float(np.max(amp))
    print(f"  wet T-columns with final!=0: n={n_cols}")
    print(f"  amplification (max|partial_sum| / |final_sum|): "
          f"median={med_amp:.1f}x  p90={p90_amp:.1f}x  max={max_amp:.3e}x")

    return dict(per_level_ratios=per_level_ratios, corr_resum=corr_resum,
                ratio_resum=ratio_resum, n_cols=n_cols, med_amp=med_amp,
                p90_amp=p90_amp, max_amp=max_amp)


def main() -> int:
    print(dump_lane.banner())
    st = build_state()
    part1 = measure_row_tuple(st)
    part2 = measure_conditioning(st, part1)

    print("\n=== SUMMARY ===")
    print(f"Row tuple (flat 3-D hdiv, n={part1['n']}): "
          f"corr={part1['corr']:.6f}  |x|ratio={part1['ratio_abs']:.6f}")
    print("Recorded in fidelity_bar_gate.py (2026-07-28, metric_convention="
          "'nemo_isotropic'): corr=1.000000, |x|ratio=1.000004, n=347200.")
    print(f"Per-level hdiv: {len(part2['per_level_ratios'])} levels sampled, "
          f"all corr/ratio ~1.0000 expected if div_hor transcription is exact")
    print(f"Depth-sum amplification: n_cols={part2['n_cols']}  "
          f"median={part2['med_amp']:.1f}x  p90={part2['p90_amp']:.1f}x  "
          f"max={part2['max_amp']:.3e}x")
    print("Recorded (2026-08-03, scripts/tmp/_probe_sshnxt_divhor_localize.py, "
          "never committed): median=4107x  max=2.08e7x  n=9913 wet columns.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
