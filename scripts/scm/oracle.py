"""Generate jax_scm reference (oracle) NetCDF outputs for the SCM benchmarks.

Drives the upstream ``jax_scm`` package (Pierzyna 2026, arXiv:2605.24544)
in its isolated ``.venv-jax-scm`` virtual environment and saves the
resulting trajectories under ``tests/validation/scm_oracle/<case>.nc``
together with a sha256 manifest pinned to the upstream commit hash.

Usage
-----
After ``scripts/setup_jax_scm_oracle.sh`` populates ``.venv-jax-scm/``
with the upstream package::

    .venv-jax-scm/bin/python scripts/run_scm_test_matrix.py oracle \\
        [--cases gabls1 andren1994 wangara]

Each case writes a single NetCDF to ``tests/validation/scm_oracle/``.
The legoESM Phase E1/E2/E3 benchmark tests read these files and compare
their own SCM output against the saved oracle trajectories.

Cases
-----
gabls1:
    Stable boundary layer (Cuxart et al. 2006).  9-hr run with surface
    cooling 0.25 K/hr; 400 m domain; Nz=64.
andren1994:
    Neutral / weakly-stable Ekman-like spin-up (Andren et al. 1994).
    Used as the closest jax_scm equivalent to a pure-neutral Ekman
    case (jax_scm does not ship a standalone Ekman setup).
wangara:
    Convective dry boundary layer (Wangara Day 33 radiosonde IC).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys


def _jax_scm_commit_hash() -> str:
    """Best-effort SHA of the installed jax_scm package."""
    try:
        import scm  # type: ignore[import-not-found]
        repo_path = pathlib.Path(scm.__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _file_sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_case(name: str, Nz: int):
    """Construct a jax_scm Simulation for the requested case."""
    if name == "gabls1":
        from scm.examples.gabls1 import get_gabls1  # type: ignore[import-not-found]
        return get_gabls1(Nz=Nz)
    if name == "andren1994":
        from scm.examples.andren1994 import get_andren1994  # type: ignore[import-not-found]
        return get_andren1994(Nz=Nz)
    if name == "wangara":
        from scm.examples.wangara.wangara import get_wangara_day33  # type: ignore[import-not-found]
        return get_wangara_day33(Nz=Nz)
    raise ValueError(f"Unknown case: {name!r}")


def _run_case(name: str, out_dir: pathlib.Path, Nz: int, dt_s: float, dt_out_s: float) -> pathlib.Path:
    """Run one jax_scm case end-to-end and save its NetCDF trajectory."""
    from scm.config import Namelist, LogConfig  # type: ignore[import-not-found]
    from scm.mynn.model import init_model  # type: ignore[import-not-found]
    from scm.time_stepping.base import simulate  # type: ignore[import-not-found]
    from scm.io.local import out_to_ds  # type: ignore[import-not-found]

    sim = _build_case(name, Nz=Nz)
    cfg = Namelist(
        time_int="implicit",
        dt_s=dt_s,
        dt_s_out=dt_out_s,
        logging=LogConfig(log_every_n=20),
    )
    model = init_model(sim, cfg)
    out = simulate(model=model, sim=sim, cfg=cfg)
    ds = out_to_ds(out, sim)
    ds.attrs["jax_scm_commit"] = _jax_scm_commit_hash()
    ds.attrs["case"] = name
    ds.attrs["Nz"] = Nz
    ds.attrs["dt_s"] = dt_s
    ds.attrs["dt_out_s"] = dt_out_s

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_Nz{Nz}.nc"
    ds.to_netcdf(out_path)
    print(f"[oracle] {name}: {out_path}  ({out_path.stat().st_size/1e3:.1f} kB)")
    return out_path


def _write_manifest(out_dir: pathlib.Path, paths: list[pathlib.Path]) -> None:
    manifest = {
        "jax_scm_commit": _jax_scm_commit_hash(),
        "files": {
            p.name: {"sha256": _file_sha256(p), "bytes": p.stat().st_size}
            for p in paths
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[oracle] wrote {out_dir / 'manifest.json'}")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cases",
        nargs="+",
        default=["gabls1", "andren1994", "wangara"],
        choices=["gabls1", "andren1994", "wangara"],
        help="Subset of jax_scm benchmark cases to regenerate.",
    )
    p.add_argument("--Nz", type=int, default=64,
                   help="Vertical grid count for the jax_scm runs.")
    p.add_argument("--dt-s", type=float, default=1.0,
                   help="Inner integration time step [s].")
    p.add_argument("--dt-out-s", type=float, default=300.0,
                   help="Output cadence [s].")
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=pathlib.Path("tests/validation/scm_oracle"),
                   help="Directory for the resulting NetCDF files.")


def run(args: argparse.Namespace) -> int:
    out_paths: list[pathlib.Path] = []
    for case in args.cases:
        out_paths.append(_run_case(
            case, args.out_dir, Nz=args.Nz,
            dt_s=args.dt_s, dt_out_s=args.dt_out_s,
        ))
    _write_manifest(args.out_dir, out_paths)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
