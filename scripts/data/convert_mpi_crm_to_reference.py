#!/usr/bin/env python
"""Convert an MPI plane-CRM RCE run into an SCM-RCE campaign reference.

Reads the volumes + surface snapshots written by
``scripts/run/run_rce_mpi_long.py`` and re-emits them in the exact layout
``scripts/run/run_scm_rce_campaign.py`` globs for a resolved-CRM reference:

    <reference-dir>/snapshots3d/vol_NNNN.npz   (z, T, mse[kJ/kg], cond, [w])
    <reference-dir>/snapshots/sfc_NNNN.npz     (day, precip, + surface fields)

This makes the model's *own* radiative-convective equilibrium (full
nonhydrostatic dycore + Kessler microphysics + radiation + surface flux)
usable as the reference the single-column convection-scheme intercomparison
scores against, in place of the Wing-2018 analytic stand-in built by
``scripts/data/build_rcemip1_small_reference.py``.

Bridging transforms (the ONLY things this script changes)
--------------------------------------------------------
* ``mse``: the driver writes moist static energy in **J/kg**; the campaign
  reader multiplies ``vol['mse']`` by ``MSE_KJ_TO_J`` (=1000) to recover
  J/kg before inverting q_v, i.e. it expects **kJ/kg** on disk.  We divide
  by that same constant (imported, single source of truth).
* ``z``: the driver already writes the column **top->surface** (descending:
  ``z[0]`` = model top, ``z[-1]`` = lowest full level), which is exactly what
  the reader's ``z_half`` construction assumes.  We *validate* the ordering
  and only flip (z and the matching level axis of every field) if a future
  driver emits it ascending — never silently assume.
* file rename: ``snap_hr_*.npz -> vol_*.npz`` and
  ``snap_day_*.npz -> sfc_*.npz``, renumbered sequentially so lexical sort
  matches chronological order.

No physics is recomputed.  ``T``/``mse``/``cond`` are copied straight from the
CRM state, so the reader's inversion
``q_v = (mse*1000 - c_pd*T - g*z) / L_v`` reproduces the CRM q_v to the float32
precision of the source snapshot (``moist_static_energy_3d_plane`` builds mse
from the same ``c_pd*T + L_v*q_v + g*z``; the driver stores T, mse and q_v as
independent float32 arrays, so the recovered q_v matches to ~1e-6 kg/kg, the
snapshot quantization floor — the mse-scale division is kept in float64 so it
adds nothing on top).

Run::

    .venv/bin/python scripts/data/convert_mpi_crm_to_reference.py \
        --mpi-output results/rce_30day \
        --reference-dir results/rcemip1_small_crm_ocean \
        --spinup-days 15.0
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Single source of truth for the mse J<->kJ scale the reader applies.
from scripts.run.run_scm_rce_campaign import MSE_KJ_TO_J

# 3D volume fields carried level-by-level; flipped together with ``z`` if the
# column ever arrives surface->top.  ``mse`` is handled separately because it
# also gets the unit division.
_VOLUME_LEVEL_FIELDS = ("T", "qv", "cond", "w")


def _day_of(path: Path) -> float:
    with np.load(path) as ds:
        if "day" in ds.files:
            return float(np.asarray(ds["day"]))
        if "t_sim" in ds.files:
            return float(np.asarray(ds["t_sim"])) / 86_400.0
    return float("nan")


def _by_sim_day(paths: list[Path]) -> list[tuple[float, Path]]:
    """Return ``(day, path)`` sorted chronologically by sim-day.

    The driver names snapshots ``snap_hr_{idx:04d}``/``snap_day_{idx:04d}``, so a
    lexical filename sort silently mis-orders once an index needs 5 digits (a
    long enough run). We renumber the reference into ``last_n``-chronological
    order, so sort on the embedded ``day``/``t_sim`` — never the filename.
    Files with no time stamp (nan) sort last, in stable filename order.
    """
    stamped = [(_day_of(p), p) for p in paths]
    return sorted(
        stamped,
        key=lambda dp: (not math.isfinite(dp[0]), dp[0], dp[1].name),
    )


def _orient_top_to_surface(
    z: np.ndarray, fields: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return ``z`` (and every level field) ordered model-top -> surface.

    The reader builds half levels assuming ``z[0]`` is the highest altitude
    and ``z[-1]`` the lowest, closing ``z_half[-1] = 0`` at the ground.  A
    strictly-descending column passes through untouched; a strictly-ascending
    one is flipped on the level axis (last axis of each field).  Anything
    non-monotone is a corrupt volume -> raise.
    """
    dz = np.diff(z)
    if np.all(dz < 0.0):
        return z, fields
    if np.all(dz > 0.0):
        return z[::-1].copy(), {
            k: np.flip(v, axis=-1).copy() for k, v in fields.items()
        }
    raise ValueError(
        "Reference z column is not monotone; cannot orient top->surface. "
        f"z[:4]={z[:4]} z[-4:]={z[-4:]}"
    )


def convert(
    mpi_output: Path,
    reference_dir: Path,
    *,
    spinup_days: float = 0.0,
    clobber: bool = False,
) -> dict[str, Any]:
    """Convert one MPI-CRM run directory into a campaign reference.

    Returns a diagnostics dict (counts, day range, mean surface precip proxy).
    """
    vol_src = _by_sim_day(list((mpi_output / "snapshots_3d").glob("snap_hr_*.npz")))
    sfc_src = _by_sim_day(list((mpi_output / "snapshots").glob("snap_day_*.npz")))
    if not vol_src:
        raise FileNotFoundError(
            f"No 3D volumes under {mpi_output / 'snapshots_3d'} "
            "(run run_rce_mpi_long.py with --snapshot-3d-hours > 0)."
        )

    vol_dir = reference_dir / "snapshots3d"
    sfc_dir = reference_dir / "snapshots"
    for d in (vol_dir, sfc_dir):
        d.mkdir(parents=True, exist_ok=True)
        existing = list(d.glob("vol_*.npz")) + list(d.glob("sfc_*.npz"))
        if existing and not clobber:
            raise FileExistsError(
                f"{d} already holds reference files; pass --clobber to "
                "overwrite (prevents mixing two runs' snapshots)."
            )
        for stale in d.glob("vol_*.npz"):
            stale.unlink()
        for stale in d.glob("sfc_*.npz"):
            stale.unlink()

    z_ref: np.ndarray | None = None
    vol_days: list[float] = []
    vol_written = 0
    for day, src in vol_src:
        if math.isfinite(day) and day < spinup_days:
            continue
        with np.load(src) as ds:
            if "cond" not in ds.files:
                raise KeyError(
                    f"{src} lacks a 'cond' field. Re-run run_rce_mpi_long.py "
                    "with the condensate-emitting save_snapshot_3d."
                )
            z = np.asarray(ds["z"], dtype=np.float32)
            level_fields = {
                k: np.asarray(ds[k]) for k in _VOLUME_LEVEL_FIELDS
                if k in ds.files
            }
            # Keep mse in float64: the reader multiplies by MSE_KJ_TO_J, so a
            # float32 re-cast here would perturb the recovered J/kg (and thus
            # q_v) by ~1e-8 kg/kg. float64 keeps the round-trip exact to the
            # float32 quantization already baked into the source snapshot.
            level_fields["mse"] = (
                np.asarray(ds["mse"], dtype=np.float64) / MSE_KJ_TO_J
            )
        z, level_fields = _orient_top_to_surface(z, level_fields)
        if z_ref is None:
            z_ref = z
        elif not np.allclose(z_ref, z):
            raise ValueError(
                f"z grid changed at {src}; the campaign reader requires a "
                "static reference column."
            )
        out = vol_dir / f"vol_{vol_written:04d}.npz"
        np.savez_compressed(out, z=z, day=np.float32(day), **level_fields)
        vol_written += 1
        vol_days.append(day)

    if vol_written == 0:
        latest = max((d for d, _ in vol_src), default=float("nan"))
        raise ValueError(
            f"No 3D volumes survived the spinup filter (spinup_days="
            f"{spinup_days}); latest volume day was {latest:.3f}."
        )

    sfc_written = 0
    precip_means: list[float] = []
    for day, src in sfc_src:
        if math.isfinite(day) and day < spinup_days:
            continue
        with np.load(src) as ds:
            payload = {k: np.asarray(ds[k]) for k in ds.files}
        if "precip" in payload:
            precip_means.append(float(np.nanmean(payload["precip"])))
        out = sfc_dir / f"sfc_{sfc_written:04d}.npz"
        np.savez_compressed(out, **payload)
        sfc_written += 1

    return {
        "reference_dir": str(reference_dir),
        "n_volumes": vol_written,
        "n_surface": sfc_written,
        "vol_day_first": float(min(vol_days)) if vol_days else float("nan"),
        "vol_day_last": float(max(vol_days)) if vol_days else float("nan"),
        "n_levels": int(z_ref.size) if z_ref is not None else 0,
        "mean_surface_precip_proxy": (
            float(np.mean(precip_means)) if precip_means else float("nan")
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mpi-output", type=Path, required=True,
        help="run_rce_mpi_long.py output dir (holds snapshots_3d/ + snapshots/).",
    )
    parser.add_argument(
        "--reference-dir", type=Path, required=True,
        help="Campaign reference dir to write (snapshots3d/ + snapshots/).",
    )
    parser.add_argument(
        "--spinup-days", type=float, default=0.0,
        help="Discard snapshots before this sim-day (equilibrium window).",
    )
    parser.add_argument(
        "--clobber", action="store_true",
        help="Overwrite an existing reference dir's vol_*/sfc_* files.",
    )
    args = parser.parse_args(argv)
    diag = convert(
        args.mpi_output, args.reference_dir,
        spinup_days=args.spinup_days, clobber=args.clobber,
    )
    print("[convert-mpi-crm] wrote campaign reference:")
    for k, v in diag.items():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
