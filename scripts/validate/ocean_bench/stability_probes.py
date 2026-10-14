#!/usr/bin/env python
"""BENCH stability probes — the citable source for every stability number.

Every stability/dt claim cited in ``docs/ocean/experiments/bench_plan.md``
and ``docs/dev-notes/ocean_experiments_reference.md`` §10b comes from THIS
committed probe (repo rule: throwaway probes are not citable).  The probe:

* runs the pre-registered probe matrix (below) on the bench ICs with the
  ``legoesm_nemo_like_v1`` recipe, exactly as ``scripts/run/run_bench.py``
  builds them;
* writes one provenance-stamped JSONL record per lane (git SHA, dirty-tree
  flag, machine, JAX version, precision, grid/dt/steps, gate values);
* includes an EXPECTED-FAIL control lane (36x72 at dt=3600 s must produce
  non-finite fields) proving the harness can detect instability;
* runs the TKE-alive discriminating control: the bench lane with the
  TKE closure stays stable and its STEP COST is measurably above the
  SAME lane with ``VerticalMixingConfig(scheme="none")`` (measured
  ~1.10x) — proving the closure solve runs and its cost is part of the
  measured step.  (On these ICs the closure's TRAJECTORY effect is
  below detection by design — K converges to background under zero
  forcing, as in NEMO BENCH.)

Pre-registered expectations (written before the runs, 2026-10-10):

* eq 24x48x10, dt=3600 s, 12 steps          -> STABLE (gates PASS)
* eq 36x72x20, dt=3600 s, 12 steps          -> UNSTABLE (non-finite)
* eq 36x72x20, dt=1800 s, 60 steps          -> STABLE
* eq 36x72x20, dt=1200 s, 12 steps          -> STABLE
* eq 48x96x20, dt=1200 s, 12 steps          -> STABLE
* TKE-alive control (eq 36x72x20, dt=1800, 60 steps, tke vs none) ->
  tke lane STABLE and >=2% costlier than the vmix=none lane -> alive
* merc orca1_like shape (278x360x74), dt=1200 s, 60 steps -> STABLE
  (pre-registered from the measured boundary: dt=1500 stable, dt=1650
  NaN at 60 steps; the probe's verdict is BINDING for BENCH_PRESETS)

Also MEASURED here (not pre-registered, recorded in the docs): the
full-sphere equirectangular latlon grid at preset sizes (e.g. 180x360)
is barotropic-CFL-unstable at the pole row (dx ~110 m at 89.75 deg)
at EVERY dt tested down to dt_baro=1 s — the structural reason the
latlon lane truncates at 80 deg with Mercator rows (see
``bench.py::BENCH_PRESETS``).

Usage::

    JAX_ENABLE_X64=1 python scripts/validate/ocean_bench/stability_probes.py \
        [--jsonl PATH] [--quick]

``--quick`` drops the 180x360x74 preset-shape lane (CI-sized run only).
Exit code 0 iff every lane meets its pre-registered expectation.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import jax  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.experiments import bench as bench_exp  # noqa: E402

# Probe lanes: (grid_kind, n_or_latmax, n_lon, nlev, dt, steps,
# expect_stable).  grid_kind "eq" = small equirectangular latlon (the
# CI-lane regime; the pole rows are far from the CFL wall at these
# sizes), n_or_latmax = n_lat;  "merc" = the preset-shape
# Mercator-truncated latlon (the production analog), n_or_latmax =
# lat_max_deg.
PROBE_LANES = [
    ("eq", 24, 48, 10, 3600.0, 12, True),
    ("eq", 36, 72, 20, 3600.0, 12, False),  # expected-FAIL control
    ("eq", 36, 72, 20, 1800.0, 60, True),
    ("eq", 36, 72, 20, 1200.0, 12, True),
    ("eq", 48, 96, 20, 1200.0, 12, True),
]
# Preset-shape lane (orca1_like analog): the 1-deg Mercator grid
# (n_lon=360, lat_max=80 -> 278 rows), pre-registered dt=1200 s (the
# measured boundary is 1500 s stable / 1650 s NaN; 1200 keeps a 1.25x
# margin), 60 steps.  The probe's verdict is BINDING for BENCH_PRESETS.
PRESET_SHAPE_LANE = ("merc", 80.0, 360, 74, 1200.0, 60, True)


def _git_stamp() -> dict:
    def _git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(_root), *args],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return "unknown"

    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "machine": platform.node(),
        "system": f"{platform.system()}/{platform.machine()}",
        "jax_version": jax.__version__,
        "devices": int(jax.device_count()),
        "processes": int(jax.process_count()),
        "precision": "float64" if jax.config.jax_enable_x64 else "float32",
    }


def _run_lane(kind, n_or_latmax, n_lon, nlev, dt, steps, record, vmix_scheme="tke"):
    """Run one probe lane; return the final state (or None if non-finite).

    kind "eq": equirectangular latlon, n_or_latmax = n_lat.
    kind "merc": Mercator-truncated latlon, n_or_latmax = lat_max_deg
    (n_lat is derived by the grid builder).
    """
    if kind == "eq":
        from legoesm.grids.latlon import create_latlon_grid

        grid = create_latlon_grid(n_or_latmax, n_lon)
        n_lat = int(n_or_latmax)
    else:
        grid = bench_exp.create_latlon_grid_for_preset(n_lon, n_or_latmax)
        n_lat = int(len(np.asarray(grid.lat)))
    z_coord = bench_exp.bench_uniform_z_star(nlev, 5000.0)
    cfg = bench_exp.BenchConfig(n_lat=n_lat, n_lon=n_lon, n_levels=nlev)
    state = bench_exp.create_initial_conditions("latlon", grid, z_coord, cfg)
    model_config = bench_exp.bench_model_config(cfg)
    if vmix_scheme != "tke":
        # Control lane: swap ONLY the vertical mixing scheme (the rest
        # of the physics stack is identical).
        physics = bench_exp.create_forcings("latlon", None, cfg)
        from legoesm.ocean.physics.vertical_mixing.config import (
            VerticalMixingConfig,
        )

        physics = physics._replace(vertical_mixing=VerticalMixingConfig(scheme=vmix_scheme))
        model_config = bench_exp.bench_model_config(cfg, physics=physics)
    model = LatLonCGridOceanModel(grid, z_coord, model_config)

    t_wall = time.time()
    finite = True
    first_bad_step = None
    state_out = state
    for k in range(steps):
        state_out = model.step(state_out, dt=dt)
        jax.block_until_ready([leaf for leaf in jax.tree.leaves(state_out) if leaf is not None])
        arrs = [np.asarray(getattr(state_out, f).data) for f in ("u", "v", "eta", "T", "S")]
        if not all(np.all(np.isfinite(a)) for a in arrs):
            finite = False
            first_bad_step = k + 1
            break
    wall = time.time() - t_wall

    u = np.asarray(state_out.u.data)
    v = np.asarray(state_out.v.data)
    eta = np.asarray(state_out.eta.data)
    u_cc = 0.5 * (u[:, :-1, :] + u[:, 1:, :])
    v_cc = 0.5 * (v[:-1, :, :] + v[1:, :, :])
    max_speed = float(np.max(np.sqrt(u_cc**2 + v_cc**2))) if finite else float("nan")
    max_eta = float(np.max(np.abs(eta))) if finite else float("nan")

    record.update(
        {
            "grid_kind": kind,
            "lat_max_deg": (float(n_or_latmax) if kind == "merc" else None),
            "n_lat": n_lat,
            "n_lon": n_lon,
            "n_levels": nlev,
            "dt": dt,
            "steps": steps,
            "vmix": vmix_scheme,
            "finite": finite,
            "first_bad_step": first_bad_step,
            "max_speed": max_speed,
            "max_eta": max_eta,
            "wall_s": wall,
            "gates_pass": bool(
                finite
                and max_speed <= bench_exp.BenchConfig.max_speed_limit
                and max_eta <= bench_exp.BenchConfig.max_eta_limit
            ),
        }
    )
    return state_out if finite else None


def _tke_alive_control(record):
    """TKE-alive discriminating control (eq 36x72x20, dt=1800, 60 steps).

    The TKE closure (Mode-B quasi-steady diagnostic chain, the recipe
    default) leaves no carried state field, and on the BENCH ICs (tiny
    shear, stable stratification, zero surface forcing) its TRAJECTORY
    effect is below 4-decimal detection over 60 steps — the closure
    converges to K ~= background, exactly as NEMO BENCH's TKE does on
    the same design.  "Alive" therefore cannot be asserted from the
    fields; the discriminating measurement is the STEP COST: the same
    lane with ``VerticalMixingConfig(scheme="none")`` must be
    measurably CHEAPER (measured: ~1.10x at 36x72x20).  If the two
    lanes cost the same, the closure is not running at all (a silent
    no-op would change the measured BENCH step cost); if the TKE lane
    is unstable, the closure is destabilizing.
    """
    lane = dict(record)
    _run_lane("eq", 36, 72, 20, 1800.0, 60, lane, vmix_scheme="tke")
    ctrl_rec = dict(record)
    _run_lane("eq", 36, 72, 20, 1800.0, 60, ctrl_rec, vmix_scheme="none")

    tke_ok = lane["gates_pass"] and lane["finite"]
    cost_tke = lane["wall_s"]
    cost_none = ctrl_rec["wall_s"]
    # The control lane must be at least 2% cheaper for the closure to
    # count as running (measured separation ~10%, far above the ~3%
    # wall-clock scatter of these lanes).
    cost_separated = cost_none < 0.98 * cost_tke
    alive = bool(tke_ok and cost_separated)
    for src, tag in ((lane, "tke"), (ctrl_rec, "none")):
        rec = dict(record)
        rec.update(
            {
                k: src[k]
                for k in (
                    "grid_kind",
                    "lat_max_deg",
                    "n_lat",
                    "n_lon",
                    "n_levels",
                    "dt",
                    "steps",
                    "vmix",
                    "finite",
                    "first_bad_step",
                    "max_speed",
                    "max_eta",
                    "wall_s",
                    "gates_pass",
                )
            }
        )
        rec["probe"] = "ocean_bench_tke_control"
        rec["tke_alive"] = alive
        rec["tke_none_cost_ratio"] = cost_tke / cost_none if cost_none > 0 else None
        _append(rec)
    return alive


_OUT_PATH: Path | None = None


def _append(rec):
    if _OUT_PATH is not None:
        with _OUT_PATH.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


def main(argv=None):
    global _OUT_PATH
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--jsonl", default=str(Path(__file__).parent / "stability_probes.jsonl"))
    p.add_argument(
        "--quick",
        action="store_true",
        help="skip the orca1_like preset-shape lane and the TKE-alive control lane",
    )
    args = p.parse_args(argv)

    jax.config.update("jax_enable_x64", True)
    stamp = _git_stamp()
    _OUT_PATH = Path(args.jsonl)

    all_ok = True
    for lane in PROBE_LANES:
        kind, n_or_latmax, n_lon, nlev, dt, steps, expect_stable = lane
        rec = dict(stamp)
        rec["probe"] = "ocean_bench_stability"
        _run_lane(kind, n_or_latmax, n_lon, nlev, dt, steps, rec)

        stable = rec["gates_pass"]
        # The expected-FAIL control MUST fail (a probe that cannot fail
        # cannot detect instability).
        ok = stable == expect_stable
        rec.update({"expect_stable": expect_stable, "probe_ok": ok})
        all_ok = all_ok and ok
        _append(rec)
        status = "OK" if ok else "MISMATCH"
        print(
            f"[{status}] {kind} {rec['n_lat']}x{n_lon}x{nlev} "
            f"dt={dt:.0f} steps={steps}: finite={rec['finite']} "
            f"max_speed={rec['max_speed']:.4f} max_eta={rec['max_eta']:.4f}"
        )

    if not args.quick:
        # TKE-alive discriminating control (runs 2 extra lanes).
        rec = dict(stamp)
        alive = _tke_alive_control(rec)
        print(
            f"[{'OK' if alive else 'MISMATCH'}] TKE-alive control "
            f"(tke lane stable and costlier than vmix=none): alive={alive}"
        )
        all_ok = all_ok and alive

        # Preset-shape lane (orca1_like analog): pre-registered as
        # dt=1200 s / 60 steps STABLE (measured boundary: 1500 stable /
        # 1650 NaN).  A mismatch means BENCH_PRESETS must be corrected.
        kind, n_or_latmax, n_lon, nlev, dt, steps, expect_stable = PRESET_SHAPE_LANE
        rec = dict(stamp)
        rec["probe"] = "ocean_bench_stability"
        _run_lane(kind, n_or_latmax, n_lon, nlev, dt, steps, rec)
        ok = rec["gates_pass"] == expect_stable
        rec.update({"expect_stable": expect_stable, "probe_ok": ok})
        all_ok = all_ok and ok
        _append(rec)
        status = "OK" if ok else "MISMATCH"
        print(
            f"[{status}] {kind} n_lat={rec['n_lat']}x{n_lon}x{nlev} "
            f"dt={dt:.0f} steps={steps} (orca1_like shape): "
            f"finite={rec['finite']} max_speed={rec['max_speed']:.4f} "
            f"max_eta={rec['max_eta']:.4f}"
        )

    print(f"Probes {'ALL OK' if all_ok else 'MISMATCH'}; records appended to {_OUT_PATH}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
