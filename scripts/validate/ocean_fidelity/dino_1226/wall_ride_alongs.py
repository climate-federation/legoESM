#!/usr/bin/env python
"""#1455 -- the two cheap wall-row residuals nobody had measured.

Both are offline reads of the same one-step state the EEN work uses, and both
come off the ranked residual list in
``docs/ocean/fidelity/dino_wall_ldf_alignment.md``.

PART 1 -- THE WALL-ROW WIND STRESS, IN ABSOLUTE PASCALS.
The campaign's "surface forcing matches at 1.0" was earned on an RMS-normalised
score, and the wind stress at the southern wall is 0.23 mPa at row 1 against
174 mPa at row 40 -- three orders below the basin maximum, so an RMS-normalised
score is structurally blind there.  The acceleration the wall-row stress
supplies is only 2-11x the campaign's magnitude bar, so a 9-58% RELATIVE error
at those rows would carry the deficit.  Registered bar: **1.4e-4 Pa** absolute,
rows 1-4.  Both sides evaluate the same analytic profile; what is being tested
is whether they evaluate it at the same LATITUDE and with the same segment
selection (NEMO reads ``gphiu``, usrdef_sbc.F90:221; legoESM reads the cell
latitude, dino.py:3746).

PART 2 -- THE CLOSED SUB-BASIN CONTROL VOLUME.
Rows 1-13 are NOT a zonally periodic band -- they carry land at columns 0 and
51, and the re-entrant channel is rows 14-48.  That makes rows 1-13 a closed
sub-basin, and nobody has drawn its control volume.  This reports its geometry
and its barotropic volume budget from the one-step state: the net volume flux
through every open face, against the rate of change of the volume the free
surface stores.  A closed sub-basin whose budget closes says the wall error is
momentum; one that does not says everything upstream is downstream of a mass
defect.

This probe prints numbers and never prints a verdict.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      .venv/bin/python -m \
      scripts.validate.ocean_fidelity.dino_1226.wall_ride_alongs
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

_DIR = Path(__file__).resolve().parent
REPO_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from wall_term_discriminators import (  # noqa: E402
    HLS, JPI, JPJ, MESH, RUN, WALL_ROWS, build_lego)

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

BAR_TAU_PA = 1.4e-4          # registered absolute bar, rows 1-4
SUB_BASIN_ROWS = list(range(1, 14))   # the closed sub-basin, rows 1-13


def stamp() -> None:
    sha = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"PROVENANCE  HEAD={sha}  run={RUN}")
    print(f"PROVENANCE  JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')}")
    print(f"PROVENANCE  bars: wind stress {BAR_TAU_PA:.1e} Pa absolute on "
          f"rows {WALL_ROWS}")


def _finite(name, a):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).all():
        raise SystemExit(f"{name} carries non-finite values -- NaN is fatal")
    return a


def part1(g, br, cfg):
    """The wall-row wind stress, in absolute Pa."""
    import jax.numpy as jnp
    from legoesm.ocean.experiments.dino import dino_wind_stress
    print("\n=== PART 1 -- THE WALL-ROW WIND STRESS, ABSOLUTE Pa ===")
    utau = _finite("utau dump", np.fromfile(RUN / "sbc_dump_utau.bin",
                                            dtype="<f8").reshape(JPJ, JPI))
    utau = utau[HLS:-HLS, HLS:-HLS]
    m = xr.open_dataset(str(MESH), decode_times=False)
    gphiu = np.asarray(m["gphiu"].values).squeeze()
    gphit = np.asarray(m["gphit"].values).squeeze()
    lat_lego = np.degrees(np.asarray(br.geometry.lat, dtype=np.float64))
    print(f"  the latitude each side evaluates the profile AT:")
    print(f"    max|gphiu - gphit| over the domain = "
          f"{float(np.abs(gphiu - gphit).max()):.3e} deg  "
          f"(NEMO reads gphiu, usrdef_sbc.F90:221)")
    print(f"    max|legoESM cell lat - gphit(col 0)| = "
          f"{float(np.abs(lat_lego - gphit[:, 0]).max()):.3e} deg")
    tau_lego = np.asarray(dino_wind_stress(jnp.asarray(lat_lego), cfg),
                          dtype=np.float64)
    print(f"\n  {'row':>5}{'lat[deg]':>11}{'NEMO tau[Pa]':>16}"
          f"{'legoESM[Pa]':>15}{'diff[Pa]':>13}{'relative':>11}")
    worst = 0.0
    for j in WALL_ROWS + [8, 20, 40]:
        n = float(utau[j].mean())
        l = float(tau_lego[j])
        d = abs(n - l)
        worst = max(worst, d) if j in WALL_ROWS else worst
        tag = "  WALL" if j in WALL_ROWS else ""
        print(f"  {j:>5}{lat_lego[j]:>11.4f}{n:>16.6e}{l:>15.6e}{d:>13.3e}"
              f"{(d / abs(n) if n else np.nan):>11.2e}{tag}")
    print(f"\n  worst wall-row |difference| = {worst:.3e} Pa against the "
          f"{BAR_TAU_PA:.1e} Pa bar -> "
          f"{'ABOVE the bar' if worst >= BAR_TAU_PA else 'below the bar'}")
    # Zonal spread: the analytic profile is zonally uniform on both sides, so
    # a nonzero spread would mean one side is not evaluating what it claims.
    print(f"  NEMO zonal spread on the wall rows: "
          f"{float(np.ptp(utau[WALL_ROWS[0]:WALL_ROWS[-1] + 1], axis=1).max()):.3e} Pa")
    return worst


def part2(g, br, cfg, mc):
    """The closed sub-basin: geometry, then the barotropic volume budget."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        min_cell_to_uface, min_cell_to_vface)
    from legoesm.ocean.vertical import compute_layer_thickness
    print("\n=== PART 2 -- THE CLOSED SUB-BASIN, ROWS 1-13 ===")
    st = br.state
    tmask_s = np.asarray(g.tmask)[..., 0] > 0.5
    ny, nx = tmask_s.shape
    print(f"  domain {ny} x {nx};  wet columns per row:")
    for j in [0] + SUB_BASIN_ROWS[:4] + [13, 14, 20, 40, 48, 49]:
        w = np.nonzero(tmask_s[j])[0]
        span = f"{w.min()}..{w.max()}" if w.size else "none"
        print(f"    row {j:>3}: {w.size:>3} wet, columns {span}"
              + ("   <- sub-basin" if j in SUB_BASIN_ROWS else "")
              + ("   <- channel" if 14 <= j <= 48 else ""))

    h_k = _finite("h_k", compute_layer_thickness(
        st.eta.data, st.H_bathy.data, br.z_coord,
        min_water_column_m=mc.min_water_column_m))
    h_u = np.asarray(min_cell_to_uface(jnp.asarray(h_k)), dtype=np.float64)
    h_v = np.asarray(min_cell_to_vface(jnp.asarray(h_k), br.geometry),
                     dtype=np.float64)
    u = np.asarray(st.u.data, dtype=np.float64)
    v = np.asarray(st.v.data, dtype=np.float64)
    um = np.asarray(st.u_mask.data)[..., None] > 0
    vm = np.asarray(st.v_mask.data)[..., None] > 0
    dy_u = np.asarray(br.geometry.dy_u, dtype=np.float64)[..., None]
    dx_v = np.asarray(br.geometry.dx_v, dtype=np.float64)[..., None]
    area = np.asarray(br.geometry.area_T, dtype=np.float64)

    # Volume flux [m^3/s] through the faces bounding rows 1..13.
    U = (h_u * u * um * dy_u).sum(axis=-1)        # (n_lat, n_lon+1)
    V = (h_v * v * vm * dx_v).sum(axis=-1)        # (n_lat+1, n_lon)
    j0, j1 = SUB_BASIN_ROWS[0], SUB_BASIN_ROWS[-1]
    south = float(V[j0][tmask_s[j0]].sum())       # in through the south face
    north = float(V[j1 + 1][tmask_s[j1]].sum())   # out through the north face
    # Zonal faces: the sub-basin is blocked at columns 0 and 51, so every u-face
    # on its perimeter should be closed.  Report the leak rather than assume it.
    _um2 = np.asarray(st.u_mask.data)[j0:j1 + 1] > 0.5
    zonal = float(np.abs(U[j0:j1 + 1][~_um2]).sum())
    net = south - north
    a_tot = float(area[j0:j1 + 1][tmask_s[j0:j1 + 1]].sum())
    print(f"\n  volume flux IN through the south face  {south:>14.5e} m3/s")
    print(f"  volume flux OUT through the north face {north:>14.5e} m3/s")
    print(f"  net convergence                        {net:>14.5e} m3/s")
    print(f"  transport through CLOSED zonal faces   {zonal:>14.5e} m3/s "
          f"(must be 0)")
    print(f"  sub-basin wet area                     {a_tot:>14.5e} m2")
    print(f"  implied mean d(eta)/dt = net/area      {net / a_tot:>14.5e} m/s"
          f"  = {net / a_tot * 86400.0 * 365.0 * 1e3:>10.4f} mm/yr")
    print(f"  mean eta over the sub-basin            "
          f"{float((np.asarray(st.eta.data) * area)[j0:j1 + 1][tmask_s[j0:j1+1]].sum() / a_tot):>14.5e} m")

    # The consistency question the control volume was drawn to answer.  Rows
    # 1-13 are CLOSED except for their northern face, so over a year the only
    # thing a net transport imbalance can do is raise the sub-basin's sea
    # surface.  Both inputs are the parent document's published measurements,
    # named here so the arithmetic is checkable:
    #   - the southern-basin transport deficit at one year, and
    #   - the sea-surface excess the ablation table measured over these rows.
    d_transport_sv = 0.95          # Sv, docs/.../dino_basin_budget_result.md
    d_eta_mm = 4.4                 # mm, the measured sea-surface excess
    year_s = 365.0 * 86400.0
    eta_if_all_converged = d_transport_sv * 1e6 * year_s / a_tot
    flux_implied_by_eta = d_eta_mm * 1e-3 * a_tot / year_s
    print(f"\n  IF the {d_transport_sv} Sv transport error were a net "
          f"convergence into this closed sub-basin for a year, its sea surface "
          f"would rise {eta_if_all_converged:.2f} m")
    print(f"  the measured excess is {d_eta_mm} mm, i.e. "
          f"{d_eta_mm * 1e-3 / eta_if_all_converged:.3e} of that")
    print(f"  the sea-surface excess accounts for a mean flux imbalance of "
          f"{flux_implied_by_eta:.4e} m3/s = "
          f"{flux_implied_by_eta / 1e6:.3e} Sv")
    print(f"  so {100.0 * (1.0 - flux_implied_by_eta / (d_transport_sv * 1e6)):.4f}% "
          f"of the transport error RECIRCULATES inside the sub-basin rather "
          f"than accumulating as mass")
    return net, a_tot


def self_test(g, br, cfg, mc):
    """Non-vacuity: perturb each scored quantity and require it to move."""
    print("\n=== SELF-TEST ===")
    import copy
    base = part1(g, br, cfg)
    # DINOConfig is a dataclass, not a NamedTuple -- copy and mutate rather
    # than reach for a ``_replace`` that does not exist.  An earlier version
    # of this check tested for ``_replace`` and SKIPPED, i.e. it silently did
    # not run at all, which is the same as not having the check.
    bad = copy.deepcopy(cfg)
    bad.wind_tau_values = tuple(x + 1e-3 for x in cfg.wind_tau_values)
    moved = part1(g, br, bad)
    assert abs(moved - base) > 1e-6, (
        "shifting every wind knot by 1 mPa did not move the scored "
        "wall-row difference -- part 1 cannot fail")
    print(f"(i) a 1 mPa shift of every wind knot moved the wall score "
          f"{base:.3e} -> {moved:.3e} Pa")
    net0, _a = part2(g, br, cfg, mc)
    st0 = br.state
    # The SOUTH face is the wall: its v-mask is zero, so a plant there is
    # correctly annihilated -- which is exactly what makes it a useless
    # non-vacuity probe.  The first version of this check used it and the
    # assertion fired.  The OPEN face is the northern one, v-face row 14.
    vv = np.asarray(st0.v.data, dtype=np.float64).copy()
    vv[SUB_BASIN_ROWS[-1] + 1, :, :] += 1e-3
    br2 = br._replace(state=st0._replace(v=st0.v.replace(data=vv)))
    net1, _a = part2(g, br2, cfg, mc)
    assert net1 != net0, ("planting 1 mm/s on the OPEN northern face did not "
                          "move the net convergence -- part 2 cannot fail")
    print(f"(ii) a 1 mm/s plant on the open northern face moved the net "
          f"convergence {net0:.5e} -> {net1:.5e} m3/s")
    # And the closed south face MUST annihilate its plant -- the other half of
    # the same statement, asserted rather than assumed.
    vs = np.asarray(st0.v.data, dtype=np.float64).copy()
    vs[SUB_BASIN_ROWS[0], :, :] += 1e-3
    br3 = br._replace(state=st0._replace(v=st0.v.replace(data=vs)))
    net2, _a = part2(g, br3, cfg, mc)
    assert net2 == net0, (
        f"a plant on the CLOSED southern wall changed the budget "
        f"{net0:.6e} -> {net2:.6e} -- the sub-basin is not closed where this "
        "probe says it is")
    print(f"(iii) the same plant on the closed southern wall changed nothing "
          f"({net0:.5e} m3/s), so the wall is closed as claimed")
    print("SELF-TEST PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    set_policy(PrecisionPolicy.fp64())
    stamp()
    g, br, cfg, mc = build_lego()
    if args.self_test:
        return self_test(g, br, cfg, mc)
    part1(g, br, cfg)
    part2(g, br, cfg, mc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
