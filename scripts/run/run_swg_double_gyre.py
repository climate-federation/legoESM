#!/usr/bin/env python3
"""Run the parameterised SWG double gyre from rest with daily NPZ output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SECONDS_PER_DAY = 86_400


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_hashes_atomic(path: Path, hashes: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    temporary.replace(path)


def resolved_daily_namelist(days: int):
    """Return the user-selected Maddison deck resolved for a number of days."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import SWG_MADDISON_NAMELIST

    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    dt_s = SWG_MADDISON_NAMELIST.rn_Dt_s
    steps_per_day = SECONDS_PER_DAY / dt_s
    if not float(steps_per_day).is_integer():
        raise ValueError(f"rn_Dt={dt_s} does not divide one model day")
    cadence = int(steps_per_day)
    return SWG_MADDISON_NAMELIST._replace(
        nn_itend=days * cadence, nn_write=cadence, nn_stock=cadence)


def run_long(
        namelist, output: Path, *, days: int, mode: str, commit: str,
) -> dict:
    """Integrate one resolved SWG deck and write one native state per day."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_swg_card,
        swg_surface_wind,
        validate_swg_namelist,
    )
    from legoesm.ocean.state import OceanSurfaceForcing

    validate_swg_namelist(namelist)
    if mode not in ("gpu-native", "cpu-certified"):
        raise ValueError("mode must be 'gpu-native' or 'cpu-certified'")
    if type(days) is not int or days <= 0:
        raise ValueError("days must be a positive integer")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("commit must be a forty-character lowercase Git SHA")
    steps_per_day_f = SECONDS_PER_DAY / namelist.rn_Dt_s
    if not float(steps_per_day_f).is_integer():
        raise ValueError("resolved rn_Dt must divide one model day")
    steps_per_day = int(steps_per_day_f)
    if namelist.nn_itend != days * steps_per_day:
        raise ValueError("resolved nn_itend does not equal days times steps/day")

    if mode == "gpu-native":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("gpu-native mode requires CUDA_VISIBLE_DEVICES=1")
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        certified_policy = False
        expected_backend = "gpu"
    else:
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        certified_policy = True
        expected_backend = "cpu"
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("JAX x64 is disabled")
    if jax.default_backend() != expected_backend:
        raise RuntimeError(
            f"{mode} requires JAX backend {expected_backend}, got "
            f"{jax.default_backend()}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory {output}")

    card = build_swg_card(namelist)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    tau_u, tau_v = swg_surface_wind(namelist)
    forcing = OceanSurfaceForcing(
        tau_i_native=jax.numpy.asarray(tau_u, dtype=jax.numpy.float64),
        tau_j_native=jax.numpy.asarray(tau_v, dtype=jax.numpy.float64),
    )
    output.mkdir(parents=True)
    manifest = {
        "schema": "legoesm-swg-double-gyre-daily-v1",
        "legoesm_git_sha": commit,
        "mode": mode,
        "certified_policy": certified_policy,
        "deck_certification": "pending NEMO ladder acquisition",
        "backend": jax.default_backend(),
        "precision_policy": "fp64",
        "transcendentals": "native" if mode == "gpu-native" else "libm",
        "days": days,
        "steps_per_day": steps_per_day,
        "namelist": namelist._asdict(),
        "fields": {
            "ssh_m": "T points",
            "u_m_s": "native NEMO U record (west redundant face removed)",
            "v_m_s": "native NEMO V record (south redundant face removed)",
        },
    }
    pending_path = output / "manifest.pending.json"
    day_hashes: dict[str, str] = {}
    _write_json_atomic(
        pending_path,
        manifest | {"status": "pending", "completed_days": 0,
                    "daily_sha256": day_hashes})

    state = card.recipe.initial_state
    for day in range(1, days + 1):
        for _ in range(steps_per_day):
            state = model.step(state, card.dt_s, surface_forcing=forcing)
        arrays = {
            "schema": np.str_(manifest["schema"]),
            "legoesm_git_sha": np.str_(commit),
            "mode": np.str_(mode),
            "certified_policy": np.bool_(certified_policy),
            "day": np.int64(day),
            "step": np.int64(day * steps_per_day),
            "dt_s": np.float64(card.dt_s),
            "ssh_m": np.asarray(state.eta.data, dtype=np.float64),
            "u_m_s": np.asarray(state.u.data[:, 1:, 0], dtype=np.float64),
            "v_m_s": np.asarray(state.v.data[1:, :, 0], dtype=np.float64),
        }
        name = f"day{day:04d}.npz"
        temporary = output / f".{name}.tmp"
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
        temporary.replace(output / name)
        day_hashes[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
        _write_hashes_atomic(output / "daily.sha256", day_hashes)
        _write_json_atomic(
            pending_path,
            manifest | {"status": "pending", "completed_days": day,
                        "daily_sha256": day_hashes})
    completed = manifest | {
        "status": "complete", "completed_days": days,
        "daily_sha256": day_hashes,
    }
    _write_json_atomic(pending_path, completed)
    pending_path.replace(output / "manifest.json")
    return completed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("gpu-native", "cpu-certified"),
        default="gpu-native")
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import git_sha

        namelist = resolved_daily_namelist(args.days)
        manifest = run_long(
            namelist, args.output, days=args.days, mode=args.mode,
            commit=git_sha())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(
        f"COMPLETED days={manifest['days']} backend={manifest['backend']} "
        f"certified_policy={manifest['certified_policy']} "
        f"deck_certification=pending output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
