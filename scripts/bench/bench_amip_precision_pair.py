#!/usr/bin/env python
"""Time the production AMIP deck in fp64 against the fp32 deck with 64-bit
land and microphysics, back to back on ONE GPU, from the same inputs.

Arms (run in this order, each a fresh ``scripts/run/run_amip.py`` process):

* ``fp64``  -- ``config/amip/amip_production.yaml`` with ``JAX_ENABLE_X64=1``;
* ``fp32``  -- ``config/amip/amip_production_fp32_land64_micro64.yaml``
  (fp32 run, fp64 land step and microphysics), ``JAX_ENABLE_X64`` unset.

Per arm it reports seconds per simulated day (wall time from the end of day 1
to the end of the last day, so compilation is excluded; needs ``--days >= 2``),
peak device memory of the GPU it runs on (``nvidia-smi`` ``memory.used``
sampled every ``--mem-poll-s``; ``null`` if no sample succeeded), the run's
status line and exit code (an arm passes only with ``COMPLETED``, exit 0 and
a measured rate) and the land-drag fallback count (the run's total of land
column-steps that reused the last valid land drag because the land solve was
held or non-finite).  A JSON summary with the git SHA, GPU name
and jax / jaxlib / CUDA-plugin versions is written to ``<output>/summary.json``.

Input files: the SAME environment variables as ``config/amip/amip_production.sh``
(that file is sourced to build the path flags, so the two cannot drift).
Unset variables fall back to its Levante defaults; on another machine set
every one.  Missing files are listed by variable name before anything runs.

  ERA5_IC         ERA5 initial state, 1979-01-01 (zarr directory)
  SST, SIC        PCMDI CMIP7 AMIP tosbcs / siconcbcs 1870-2022 (degC; leave
                  AMIP_SST_OFFSET at its amip_production.sh default)
  SOLAR           swflux_14band_cmip6_1850-2299-v3.2.nc
  GHG             greenhouse_historical_plus.nc
  OZONE           vmro3 UReading-CCMI 1979-2014
  AEROSOL         aeropt_macv2sp_sw_b14_tot_1850-2014_MACv1bg.nc
  AEROSOL_CCN     (same file as AEROSOL)
  VOLCANIC        bc_aeropt_cmip6_volc_lw_b16_sw_b14_1850-2014_IACETH-v4.nc
  ETOPO           etopo2022_60s_0p25deg.nc
  CLM_SURFDATA    surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc
  AMIP_SURFDATA   legoesm_surfdata_c260716.nc (default: <repo>/data/)
  AMIP_LAND_MASK  sftlf_clm_1.9x2.5.nc
  AMIP_SSO        sso_stdh_2deg_from1deg_seamfix.nc

Plus the deck's land initial state, read relative to the repo:
<repo>/data/lmip_soil_ic/soil_ic_mpas6_era5T_19790101_v2.npz (or pass it with
--land-ic).

Usage::

    export ERA5_IC=/data/era5_ic_1979-01-01_gproxy_L21.zarr SST=... (all above)
    python scripts/bench/bench_amip_precision_pair.py --days 3 --output pair_out
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ARMS = (("fp64", "amip_production.yaml", True),
        ("fp32", "amip_production_fp32_land64_micro64.yaml", False))
# run_amip flags whose value is an input file/directory that must exist
_PATH_FLAGS = {"--surfdata": "AMIP_SURFDATA", "--ic-path": "ERA5_IC",
               "--forcing-path": "SST", "--sic-path": "SIC",
               "--solar-file": "SOLAR", "--ozone-file": "OZONE",
               "--ghg-file": "GHG", "--aerosol-file": "AEROSOL",
               "--aerosol-ccn-file": "AEROSOL_CCN",
               "--volcanic-aerosol-file": "VOLCANIC", "--topography": "ETOPO",
               "--clm-surfdata-path": "CLM_SURFDATA",
               "--land-mask-file": "AMIP_LAND_MASK",
               "--subgrid-orography-file": "AMIP_SSO"}
_DAY = re.compile(r"^\s*Day\s+([\d.]+):.*\(\S+ sim-days/s, (\d+)s\)")
_HELD = re.compile(r"run totals \d+ / (\d+)")


def _days(v):
    if int(v) < 2:
        raise argparse.ArgumentTypeError("--days must be >= 2")
    return int(v)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                epilog=__doc__.split("\n\n", 1)[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=_days, default=3,
                   help="simulated days per arm (>= 2: day 1 holds compilation)")
    p.add_argument("--output", type=Path, required=True,
                   help="new directory; one sub-directory per arm + summary.json")
    p.add_argument("--land-ic", default=None,
                   help="optional soil initial state (.npz) passed to both arms")
    p.add_argument("--params", default=None,
                   help="optional --params YAML passed to both arms")
    p.add_argument("--gpu", default="0", help="GPU index to sample memory on")
    p.add_argument("--mem-poll-s", type=float, default=2.0)
    p.add_argument("--arms", nargs="+", default=[a[0] for a in ARMS],
                   choices=[a[0] for a in ARMS])
    return p


def path_flags(repo=REPO):
    """The run_amip path flags of config/amip/amip_production.sh, built from
    the caller's environment by sourcing that file."""
    env = dict(os.environ, REPO=str(repo))
    out = subprocess.run(
        ["bash", "-c", 'source config/amip/amip_production.sh && '
         'printf "%s\\0" "${AMIP_PATH_FLAGS[@]}"'],
        cwd=repo, env=env, capture_output=True, text=True)
    if out.returncode:
        sys.exit(f"sourcing config/amip/amip_production.sh failed:\n{out.stderr}")
    out = out.stdout
    flags = [a for a in out.split("\0") if a]
    # the model runs from the repo: make relative input paths absolute here
    return [os.path.abspath(v) if f in _PATH_FLAGS else v
            for f, v in zip([None] + flags, flags)]


def missing_inputs(flags):
    """[(env var, path)] for every input flag whose path does not exist."""
    return [(_PATH_FLAGS[f], v) for f, v in zip(flags, flags[1:])
            if f in _PATH_FLAGS and not Path(v).exists()]


def parse_log(text):
    """Day-end wall times {day: s} and the run's total land-drag fallback
    count (last "run totals" line; None if never printed)."""
    days = {float(m.group(1)): int(m.group(2))
            for m in map(_DAY.match, text.splitlines()) if m}
    held = _HELD.findall(text)
    return days, (int(held[-1]) if held else None)


def s_per_day(days):
    """Wall seconds per simulated day after day 1 (None if < 2 days)."""
    if len(days) < 2 or 1.0 not in days:
        return None
    last = max(days)
    return (days[last] - days[1.0]) / (last - 1.0)


def run_status(results_txt):
    m = re.search(r"^Status: (.*)$", Path(results_txt).read_text(), re.M) \
        if Path(results_txt).exists() else None
    return m.group(1).strip() if m else "NO STATUS (run died)"


class _MemPoll(threading.Thread):
    def __init__(self, gpu, period):
        super().__init__(daemon=True)
        self.gpu, self.period, self.peak, self._halt = gpu, period, None, threading.Event()

    def run(self):
        while not self._halt.is_set():
            r = subprocess.run(["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                self.peak = max(self.peak or 0, int(r.stdout.split()[0]))
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join()
        return self.peak


def _driver_cuda(smi):
    m = re.search(r"CUDA Version:\s*([\d.]+)", smi or "")
    return m.group(1) if m else None


def environment(gpu="0"):
    from importlib import metadata

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    def cmd(*a):
        try:
            r = subprocess.run(a, capture_output=True, text=True, cwd=REPO)
        except FileNotFoundError:
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    return {"git_sha": cmd("git", "rev-parse", "HEAD"),
            "git_dirty": bool(cmd("git", "status", "--porcelain", "--untracked-files=no")),
            "gpu": cmd("nvidia-smi", "-i", gpu, "--query-gpu=name,driver_version",
                       "--format=csv,noheader"),
            "driver_cuda": _driver_cuda(cmd("nvidia-smi")),
            "jax": ver("jax"), "jaxlib": ver("jaxlib"),
            "cuda_plugin": ver("jax-cuda12-plugin") or ver("jax-cuda13-plugin"),
            "cuda_runtime": ver("nvidia-cuda-runtime-cu12") or ver("nvidia-cuda-runtime"),
            "python": sys.version.split()[0]}


def write_summary(path, args, env, arms):
    summary = {"days": args.days, "land_ic": args.land_ic, "params": args.params,
               **env, "arms": arms}
    Path(path).write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def arm_ok(a):
    return a["status"] == "COMPLETED" and a["exit_code"] == 0 \
        and a["s_per_sim_day"] is not None


def arm_command(name, deck, x64, flags, args):
    """(argv, environment) of one arm's run_amip process."""
    out = args.output / name
    cmd = [sys.executable, "-u", "scripts/run/run_amip.py", "--config",
           f"config/amip/{deck}", *flags, "--days", str(args.days),
           "--checkpoint-days", str(args.days), "--output", str(out)]
    if args.land_ic:
        cmd += ["--land-ic", args.land_ic]
    if args.params:
        cmd += ["--params", args.params]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    env.pop("JAX_ENABLE_X64", None)
    if x64:
        env["JAX_ENABLE_X64"] = "1"
    return cmd, env


def run_arm(name, deck, x64, flags, args):
    out = args.output / name
    cmd, env = arm_command(name, deck, x64, flags, args)
    out.mkdir(parents=True)
    poll = _MemPoll(args.gpu, args.mem_poll_s)
    poll.start()
    t0 = time.time()
    with open(out / "run.log", "w") as log:
        rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    peak = poll.stop()
    days, held = parse_log((out / "run.log").read_text())
    return {"deck": deck, "x64": x64, "exit_code": rc, "wall_s": round(wall, 1),
            "status": run_status(out / "results.txt"), "day_end_wall_s": days,
            "s_per_sim_day": s_per_day(days), "peak_device_mem_mib": peak,
            "land_drag_fallback_column_steps": held}


def main(argv=None):
    args = build_parser().parse_args(argv)
    plat = os.environ.get("JAX_PLATFORMS", "")
    if plat and "cuda" not in plat:
        sys.exit(f"JAX_PLATFORMS={plat!r} would time a non-GPU run: unset it")
    args.output = args.output.resolve()
    args.land_ic = args.land_ic and os.path.abspath(args.land_ic)
    args.params = args.params and os.path.abspath(args.params)
    flags = path_flags()
    miss = missing_inputs(flags)
    if miss:
        sys.exit("missing input files (set these environment variables):\n"
                 + "\n".join(f"  {k}={v}" for k, v in miss))
    if args.output.exists():
        sys.exit(f"{args.output} exists: give a new directory")
    args.output.mkdir(parents=True)
    env, arms = environment(args.gpu), {}
    for name, deck, x64 in ARMS:
        if name in args.arms:
            arms[name] = a = run_arm(name, deck, x64, flags, args)
            print(f"[{name}] {a['status']}  s/sim-day={a['s_per_sim_day']}  "
                  f"exit={a['exit_code']}  peak_device_mem={a['peak_device_mem_mib']} MiB  "
                  f"land_fallback={a['land_drag_fallback_column_steps']}", flush=True)
            write_summary(args.output / "summary.json", args, env, arms)
    print(f"summary: {args.output / 'summary.json'}")
    return 0 if all(map(arm_ok, arms.values())) else 1


if __name__ == "__main__":
    sys.exit(main())
