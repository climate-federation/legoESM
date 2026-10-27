#!/usr/bin/env python3
"""Run the existing ORCA2 comparison card from per-step NEMO replay records.

The driver advances in one-day (eight-step) ``integrate_scan`` batches and
writes instantaneous daily candidate states.  It deliberately does not assign
a long-run verdict or bar; those await the decisions in the design document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import jax
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2x_round1_longrun_acquisition as acquisition,
)

STEPS_PER_DAY = 8


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _forcing_card(card):
    """A host-only copy that lets the reviewed builder defer hm_i injection."""
    cfg = card.recipe.model_config
    vm = cfg.physics.vertical_mixing
    physics = cfg.physics._replace(vertical_mixing=vm._replace(
        tke=vm.tke._replace(nemo_mxlice=0)))
    return card._replace(recipe=card.recipe._replace(
        model_config=cfg._replace(physics=physics)))


def build_inputs(card, *, surface_root: Path, hmi_root: Path, deck_root: Path,
                 start_kt: int, steps: int,
                 admitted_round326_hmi: bool = False) -> list[dict]:
    require(start_kt >= 1 and steps >= 1, "invalid replay interval")
    host_card = _forcing_card(card)
    inputs = []
    for kt in range(start_kt, start_kt + steps):
        fields = ladder.assemble_surface_fields(surface_root, kt)
        freshwater, surface = ladder._surface_forcings(host_card, deck_root, fields, kt)
        if admitted_round326_hmi:
            require(hmi_root == ladder.ROUND326_HMI_RUN,
                    "legacy hm_i mode is restricted to the admitted round-326 root")
            hmi_fields = {"hm_i": ladder.recorded_hm_i(kt, fields["fr_i"]),
                          "fr_i": fields["fr_i"]}
        else:
            hmi_fields = acquisition.read_hmi_pair(hmi_root, kt)
        require(np.array_equal(hmi_fields["fr_i"].view(np.uint64),
                               fields["fr_i"].view(np.uint64)),
                f"kt {kt}: hm_i record is not paired with its surface record")
        surface = surface._replace(
            ice_mean_thickness_hm_i=jax.numpy.asarray(hmi_fields["hm_i"]))
        item = {"freshwater": freshwater, "surface_forcing": surface}
        item.update(ladder.damping_step_kwargs(card, deck_root, kt))
        inputs.append(item)
    return inputs


def build_comparison(record_root: Path):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "fp64/libm policy is not active")
    require(jax.config.jax_enable_x64 and jax.default_backend() == "cpu",
            "long-run comparison requires CPU fp64")
    _, shipped = ladder.card_fields(record_root)
    # The comparison helper currently reads the admitted ten-step record's
    # resolved initial-ice echo.  A long record must resolve identically before
    # it may reuse that state; no silent initial-condition substitution.
    require(ladder.record_initial_ice(record_root) ==
            ladder.record_initial_ice(ladder.RUNG10_RECORD_RUN),
            "long-run record resolved different initial-ice settings")
    card = ladder.comparison_card(shipped)
    recipe = card.recipe
    model = LatLonCGridOceanModel(recipe.grid, recipe.z_coord, recipe.model_config,
                                  iwm_forcing=recipe.iwm_forcing)
    return card, model, recipe.initial_state


def snapshot_fields(state) -> dict[str, np.ndarray]:
    return {name: np.asarray(values, dtype=np.float64)
            for name, values in ladder._candidate_fields(state).items()}


def dom_uniq_mask(grid: str, shape: tuple[int, int] = (148, 180)) -> np.ndarray:
    """NEMO ``dom_uniq`` unique-point mask on the owned T-pivot domain.

    Implements ``dom_uniq`` (``domutl.f90:106-136``) and ``lbc_nfd``
    (``lbcnfd.f90:619-748``) for ``c_NFtype == 'T'`` on the de-haloed
    ``(148, 180)`` domain:
    - T/W: zeros ``j == 147, i >= 91`` (``lbcnfd.f90:624-628``);
    - U: zeros ``j == 147, i >= 90`` (``lbcnfd.f90:669-673``);
    - V/F: zeros all of ``j == 147`` (``lbcnfd.f90:688-747``).
    """
    require(len(shape) == 2 and shape[0] >= 2 and shape[1] % 2 == 0,
            f"invalid ORCA2 shape {shape}")
    mask = np.ones(shape, dtype=np.float64)
    half = shape[1] // 2
    if grid in ("T", "W"):
        mask[-1, half + 1 :] = 0.0
    elif grid == "U":
        mask[-1, half:] = 0.0
    elif grid in ("V", "F"):
        mask[-1, :] = 0.0
    else:
        raise RuntimeError(f"unsupported grid {grid!r}")
    return mask


def reduction_weights(
    deck_root: Path, r3t: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Build NEMO-interior area and volume weights from ``ORCA_R2_zps_domcfg.nc``."""
    from netCDF4 import Dataset

    domcfg = deck_root / "ORCA_R2_zps_domcfg.nc"
    require(domcfg.is_file(), f"missing domain_cfg: {domcfg}")
    with Dataset(domcfg) as dataset:
        e1t = np.asarray(dataset.variables["e1t"][:, :], dtype=np.float64)
        e2t = np.asarray(dataset.variables["e2t"][:, :], dtype=np.float64)
        e1u = np.asarray(dataset.variables["e1u"][:, :], dtype=np.float64)
        e2u = np.asarray(dataset.variables["e2u"][:, :], dtype=np.float64)
        e1v = np.asarray(dataset.variables["e1v"][:, :], dtype=np.float64)
        e2v = np.asarray(dataset.variables["e2v"][:, :], dtype=np.float64)
        e3t_0 = np.moveaxis(
            np.asarray(dataset.variables["e3t_0"][:30, :, :], dtype=np.float64),
            0, -1,
        )
        e3u_0 = np.moveaxis(
            np.asarray(dataset.variables["e3u_0"][:30, :, :], dtype=np.float64),
            0, -1,
        )
        e3v_0 = np.moveaxis(
            np.asarray(dataset.variables["e3v_0"][:30, :, :], dtype=np.float64),
            0, -1,
        )
        top = np.asarray(dataset.variables["top_level"][:, :], dtype=np.int64)
        bottom = np.asarray(dataset.variables["bottom_level"][:, :], dtype=np.int64)
    levels = np.arange(1, 31, dtype=np.int64)[None, None, :]
    tmask = ((top[:, :, None] != 0)
             & (levels >= top[:, :, None])
             & (levels <= bottom[:, :, None])).astype(np.float64)
    umask = tmask * np.roll(tmask, -1, axis=1)
    vmask = np.zeros_like(tmask)
    vmask[:-1, :, :] = tmask[:-1, :, :] * tmask[1:, :, :]
    ssmask = np.max(tmask, axis=2)
    ssumask = np.max(umask, axis=2)
    ssvmask = np.max(vmask, axis=2)
    puniq_t = dom_uniq_mask("T", ssmask.shape)
    puniq_u = dom_uniq_mask("U", ssumask.shape)
    puniq_v = dom_uniq_mask("V", ssvmask.shape)
    # dommsk.f90:263-264: tmask_i(:,:) = ssmask(:,:) * tmask_i(:,:)
    tmask_i = ssmask * puniq_t
    umask_i = ssumask * puniq_u
    vmask_i = ssvmask * puniq_v
    scale = (1.0 if r3t is None else
             1.0 + np.asarray(r3t, dtype=np.float64)[:, :, None] * tmask)
    e3t = e3t_0 * scale
    area_t = e1t * e2t * tmask_i
    area_u = e1u * e2u * umask_i
    area_v = e1v * e2v * vmask_i
    volume_t = area_t[:, :, None] * e3t * tmask
    return {
        "tmask": tmask,
        "umask": umask,
        "vmask": vmask,
        "puniq_t": puniq_t,
        "puniq_u": puniq_u,
        "puniq_v": puniq_v,
        "tmask_i": tmask_i,
        "umask_i": umask_i,
        "vmask_i": vmask_i,
        "area_t": area_t,
        "area_u": area_u,
        "area_v": area_v,
        "e2u": e2u,
        "e3u_0": e3u_0,
        "e3v_0": e3v_0,
        "volume_t": volume_t,
    }


def nemo_total_domain_area_km2(log_path: Path) -> float:
    """Parse ``Total Domain area (km**2):`` printed by ``sbcfwb.f90:211-212``."""
    require(log_path.is_file(), f"missing NEMO log: {log_path}")
    match = re.search(
        r"Total Domain area \(km\*\*2\):\s*([0-9]+(?:\.[0-9]+)?)",
        log_path.read_text(),
    )
    require(match is not None, f"no Total Domain area line in {log_path}")
    return float(match.group(1))


DRAKE_I = 106
DRAKE_J_SOUTH_BOUND = 20
DRAKE_J_NORTH_BOUND = 37
DRAKE_J_SLICE = slice(DRAKE_J_SOUTH_BOUND, DRAKE_J_NORTH_BOUND + 1)


def drake_section(deck_root: Path) -> dict[str, object]:
    """Closed ORCA2 Drake Passage U section at ``i=106, j=20..37``."""
    weights = reduction_weights(deck_root)
    umask_col = weights["umask"][:, DRAKE_I, :]
    require(bool(np.all(umask_col[DRAKE_J_SOUTH_BOUND] == 0.0)),
            "Drake southern boundary j=20 is not dry")
    require(bool(np.any(umask_col[DRAKE_J_SOUTH_BOUND + 1] > 0.0)),
            "Drake first interior row j=21 is not wet")
    require(bool(np.all(umask_col[DRAKE_J_NORTH_BOUND] == 0.0)),
            "Drake northern boundary j=37 is not dry")
    return {
        "i": DRAKE_I,
        "j_south_bound": DRAKE_J_SOUTH_BOUND,
        "j_north_bound": DRAKE_J_NORTH_BOUND,
        "j_wet": [int(j) for j in range(DRAKE_J_SOUTH_BOUND, DRAKE_J_NORTH_BOUND + 1)
                  if np.any(umask_col[j] > 0.0)],
        "umask": umask_col[DRAKE_J_SLICE],
        "e2u": weights["e2u"][DRAKE_J_SLICE, DRAKE_I],
        "e3u_0": weights["e3u_0"][DRAKE_J_SLICE, DRAKE_I, :],
    }


def _require_admitted_records(record_root: Path, surface_root: Path,
                              hmi_root: Path, steps: int,
                              admitted_round326_hmi: bool,
                              *, fixture_steps: int | None = None) -> None:
    if admitted_round326_hmi:
        require(steps <= 10 and surface_root == ladder.RUNG10_RECORD_RUN
                and hmi_root == ladder.ROUND326_HMI_RUN,
                "ten-step compatibility mode needs the two admitted roots")
        ladder.validate_round5_admission(surface_root)
        # This is the same fail-closed admission used by recorded_hm_i.
        ladder._admitted_hm_i_records()
        return
    if fixture_steps is None:
        require(steps in (acquisition.PILOT_STEPS, acquisition.YEAR_STEPS),
                "long-run driver accepts only the undecided 30-day or one-year window")
        days = steps // STEPS_PER_DAY
    else:
        require(steps == fixture_steps == 2,
                "fixture admission check requires two steps")
        days = acquisition.PILOT_DAYS
    require(surface_root == record_root and hmi_root == record_root,
            "long-run driver requires surface_root and hmi_root to equal record_root")
    stamp = record_root / "producer_commit.txt"
    require(stamp.is_file(), "long-run record lacks producer_commit.txt")
    expected_commit = stamp.read_text().strip()
    admitted = acquisition.verify_admitted_outputs(
        record_root,
        expected_commit,
        days=days,
        fixture_steps=fixture_steps,
    )
    result = acquisition.run_gate(
        record_root,
        expected_commit,
        days=days,
        fixture_steps=fixture_steps,
    )
    require(result == admitted, "long-run record differs from its admission")


def run(record_root: Path, output_root: Path, *, steps: int,
        surface_root: Path | None = None, hmi_root: Path | None = None,
        admitted_round326_hmi: bool = False) -> dict:
    require(steps >= 1, "steps must be positive")
    surface_root = record_root if surface_root is None else surface_root
    hmi_root = record_root if hmi_root is None else hmi_root
    _require_admitted_records(record_root, surface_root, hmi_root, steps,
                              admitted_round326_hmi)
    card, model, state = build_comparison(record_root)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "resolved_model_config.txt").write_text(
        repr(card.recipe.model_config) + "\n")
    completed = 0
    snapshots = []
    wall_s = []
    start = time.monotonic()
    while completed < steps:
        count = min(STEPS_PER_DAY, steps - completed)
        inputs = build_inputs(card, surface_root=surface_root, hmi_root=hmi_root,
                              deck_root=record_root, start_kt=completed + 1,
                              steps=count,
                              admitted_round326_hmi=admitted_round326_hmi)
        state, _ = model.integrate_scan(state, count, card.dt_s,
                                        step_inputs=inputs,
                                        nemo_kt0=completed + 1)
        completed += count
        path = output_root / f"candidate_step{completed:08d}.npz"
        np.savez(path, **snapshot_fields(jax.device_get(state)))
        snapshots.append(path.name)
        wall_s.append(round(time.monotonic() - start, 3))
        # Each batch bakes its start step into a new program; drop the old
        # executables or LLVM exhausts vm.max_map_count after ~9 days.
        jax.clear_caches()
        print(f"day-batch {len(snapshots)} steps {completed} wall_s {wall_s[-1]}",
              flush=True)
    result = {
        "format": "orca2x-longrun-candidate-v1",
        "steps": steps,
        "dt_s": float(card.dt_s),
        "sampling": "instantaneous after each <=8-step integrate_scan batch",
        "comparison_card": {"implicit_ice_ocean_top_drag": True,
                            "nemo_mxlice": 2,
                            "record_initial_ice": True},
        "snapshots": snapshots,
        "cumulative_wall_s": wall_s,
        "final_candidate": {
            name: {"shape": list(values.shape),
                   "sha256": hashlib.sha256(
                       np.ascontiguousarray(values).view(np.uint8)).hexdigest()}
            for name, values in snapshot_fields(jax.device_get(state)).items()
        },
    }
    (output_root / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--surface-root", type=Path)
    parser.add_argument("--hmi-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=acquisition.PILOT_STEPS)
    args = parser.parse_args()
    result = run(args.record_root, args.output_root, steps=args.steps,
                 surface_root=args.surface_root, hmi_root=args.hmi_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
