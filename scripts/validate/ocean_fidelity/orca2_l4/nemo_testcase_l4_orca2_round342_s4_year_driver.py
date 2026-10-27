#!/usr/bin/env python3
"""ORCA2 Round 342: standalone K0 composed-card legoESM driver (2,920 steps / 1 year + smoke).

Runs the landed standalone ORCA2 K0 composed card (`build_nemo_orca2_k0_ocean_card`,
`nemo_orca2_k0_ocean_step`, explicit `NemoStandaloneK0Config(nn_ice=1,
ln_drgice_imp=False, ln_mxl0=True, nn_mxlice=0, ln_icebergs=False,
initial_state='nemo_k0_istate')`) on the production path across steps `1..steps`
(up to `2,920` steps = 1 no-leap year at `dt_s = 10800` s), writing daily
snapshots `candidate_step{kt:08d}.npz` at `kt = 8, 16, ..., steps` (and terminal
`candidate_step{steps:08d}.npz` when `steps % 8 != 0`) for the Decision-160
ORCA2X long-run metric card.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _pkg in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean import bulk_flux_omip as bulk  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_damping_for_step,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from legoesm.ocean.forcing.nemo_surface import NemoSurfaceMeans  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_o1_numerical_gate as o1,
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    nemo_testcase_l4_orca2_round339_common as common,
    nemo_testcase_l4_orca2_round340_standalone_gate as r340_s4,
)

require = common.require
sha256_file = common.sha256

DEFAULT_DECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
    "acquisition/orca1ice_surface_entry_every_step_a_np2"
)
DEFAULT_CORE_MANIFEST = DEFAULT_DECK_ROOT / "input_files.sha256"
DEFAULT_ICE_COVER_PATH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/"
    "ice_cover_producer/orca2_full_si3_year1_np2/ice_cover_clim.nc"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/legoesm_smoke_20step"
)

YEAR_STEPS = 2920
STEPS_PER_DAY = 8
MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)
MONTH_END_SEC = (np.cumsum(MONTH_DAYS) * 86400).astype(np.int64)
MONTH_CENTRES_SEC = ((np.cumsum(MONTH_DAYS) - MONTH_DAYS / 2) * 86400).astype(np.int64)
EXTENDED_MONTH_CENTRES_SEC = np.concatenate((
    [MONTH_CENTRES_SEC[-1] - 365 * 86400],
    MONTH_CENTRES_SEC,
    [MONTH_CENTRES_SEC[0] + 365 * 86400],
))

EXPLICIT_K0_CONFIG = bulk.NemoStandaloneK0Config(
    nn_ice=1,
    ln_drgice_imp=False,
    ln_mxl0=True,
    nn_mxlice=0,
    ln_icebergs=False,
    initial_state="nemo_k0_istate",
)


def load_core_window_for_steps(
    deck_root: Path,
    manifest: Path,
    max_steps: int,
) -> dict[str, np.ndarray]:
    """Load and interpolate all CORE records reachable across steps 1..max_steps.

    Extends `r340_s4.load_core_normal_year` from 10 steps (`count = {1460: 5, 365: 2, 12: 1}`)
    to any `1 <= max_steps <= 2920` (`n_6h = (max_steps - 1) // 2 + 1`,
    `n_daily = (max_steps - 1) // 8 + 1`, `n_monthly = 12`).
    """
    require(1 <= max_steps <= YEAR_STEPS, f"max_steps must be in 1..{YEAR_STEPS}, got {max_steps}")
    declared: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            declared[parts[1]] = parts[0]
    required = {filename for _, filename, _, _ in o1.FIELD_SPECS}
    required |= {
        "weights_core2_orca2_bicub.nc",
        "weights_core2_orca2_bilin.nc",
        "ORCA_R2_zps_domcfg.nc",
    }
    for name in required:
        require(name in declared, f"CORE input manifest omits {name}")
        require(sha256_file(deck_root / name) == declared[name],
                f"CORE input digest moved: {name}")

    n_6h = min(1460, max(5, (max_steps - 1) // 2 + 1))
    n_daily = min(365, max(2, (max_steps - 1) // 8 + 1))
    n_monthly = 1 if max_steps <= 10 else 12

    mapped: dict[str, np.ndarray] = {}
    weights = {bicubic: o1.load_weights(deck_root, bicubic) for bicubic in (False, True)}
    for field, filename, variable, bicubic in o1.FIELD_SPECS:
        with netCDF4.Dataset(deck_root / filename, "r") as ds:
            src_var = ds[variable]
            total_recs = int(src_var.shape[0])
            count = {1460: n_6h, 365: n_daily, 12: n_monthly}.get(total_recs)
            require(count is not None, f"CORE cadence moved for {variable}: {src_var.shape}")
            raw_slice = np.asarray(src_var[:count].data, dtype=np.float64)
        indices, coefficients = weights[bicubic]
        mapped[field] = np.stack([
            np.asarray(o1.nemo_fld_interp(raw_slice[idx], indices, coefficients, bicubic=bicubic))
            for idx in range(count)
        ])

    with netCDF4.Dataset(deck_root / "ORCA_R2_zps_domcfg.nc", "r") as ds:
        cosine, sine = o1.nemo_t_rotation_from_domain(*(
            np.asarray(ds[name][:].data, dtype=np.float64)
            for name in ("glamt", "gphit", "glamv", "gphiv")
        ))
    wind_shape = mapped["wndi"].shape
    mapped["wndi"], mapped["wndj"] = map(
        np.asarray,
        o1.rotate_en_to_ij(
            mapped["wndi"],
            mapped["wndj"],
            np.broadcast_to(cosine, wind_shape),
            np.broadcast_to(sine, wind_shape),
        ),
    )
    require(all(np.isfinite(val).all() for val in mapped.values()),
            "mapped CORE normal-year input contains non-finite values")
    return mapped


def _load_monthly_12(path: Path, variable: str) -> np.ndarray:
    """Load all 12 monthly climatology records for `sorunoff` or `sss`."""
    require(path.is_file(), f"missing K0 monthly file: {path}")
    with netCDF4.Dataset(path, "r") as ds:
        require(variable in ds.variables, f"K0 variable {variable!r} absent from {path.name}")
        src = ds.variables[variable]
        src.set_auto_maskandscale(False)
        val = np.asarray(src[:], dtype=np.float64)
    if val.ndim == 4:
        val = val[:, 0]
    require(val.shape == (12, 148, 180), f"unexpected K0 monthly shape in {path.name}: {val.shape}")
    require(bool(np.isfinite(val).all()), f"non-finite monthly values in {path.name}")
    return val


def _monthly_pair_for_clock(
    records_12: np.ndarray,
    isecsbc: int,
) -> bulk.NemoStandaloneFieldPair:
    """Select the bracketing cyclic monthly records and timestamps (`fldread.f90:243-246,885-937`)."""
    after_ext = int(np.searchsorted(EXTENDED_MONTH_CENTRES_SEC, isecsbc, side="right"))
    before_ext = after_ext - 1
    before_s = int(EXTENDED_MONTH_CENTRES_SEC[before_ext])
    after_s = int(EXTENDED_MONTH_CENTRES_SEC[after_ext])
    before_idx = (before_ext - 1) % 12
    after_idx = (after_ext - 1) % 12
    return bulk.NemoStandaloneFieldPair(
        jnp.asarray(records_12[before_idx], dtype=jnp.float64),
        jnp.asarray(records_12[after_idx], dtype=jnp.float64),
        isecsbc,
        before_s,
        after_s,
    )


def _save_snapshot(
    output_dir: Path,
    kt: int,
    state: Any,
    surface_result: Any,
) -> dict[str, Any]:
    fields = ladder._candidate_fields(state)
    fr_i = np.asarray(surface_result.ice_if.fr_i, dtype=np.float64)[2:-2, 2:-2]
    ice_cover = np.asarray(surface_result.ice_if.ice_cover, dtype=np.float64)
    payload = {
        **fields,
        "fr_i": fr_i,
        "ice_cover": ice_cover,
        "kt": np.asarray(kt, dtype=np.int64),
        "day": np.asarray(kt // STEPS_PER_DAY, dtype=np.int64),
    }
    for name, arr in payload.items():
        require(bool(np.isfinite(arr).all()), f"step {kt}: non-finite {name} in candidate snapshot")
    out_path = output_dir / f"candidate_step{kt:08d}.npz"
    np.savez(out_path, **payload)
    return {
        "kt": kt,
        "path": out_path.name,
        "sha256": sha256_file(out_path),
        "field_sha256": {
            k: hashlib.sha256(np.ascontiguousarray(fields[k]).view(np.uint8)).hexdigest()
            for k in ("T", "S", "u", "v", "ssh")
        },
    }


def run_standalone_k0_driver(
    *,
    steps: int,
    output_dir: Path,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    save_stride: int = STEPS_PER_DAY,
    save_steps: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Run the production standalone ORCA2 K0 composed card for `steps` steps."""
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        jax.config.jax_enable_x64 and jax.default_backend() == "cpu",
        f"standalone K0 card requires CPU fp64/libm backend, got {jax.default_backend()!r}",
    )
    require(1 <= steps <= YEAR_STEPS, f"steps must be in 1..{YEAR_STEPS}, got {steps}")
    require(save_stride > 0, f"save_stride must be positive, got {save_stride}")
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    card = bulk.build_nemo_orca2_k0_ocean_card(deck_root)
    require(card.k0 == EXPLICIT_K0_CONFIG,
            f"card.k0 {card.k0!r} != explicit {EXPLICIT_K0_CONFIG!r}")
    bulk.validate_nemo_standalone_k0(card.k0)

    recipe = card.ocean.recipe
    model = LatLonCGridOceanModel(
        recipe.grid,
        recipe.z_coord,
        recipe.model_config,
        iwm_forcing=recipe.iwm_forcing,
    )

    mapped_core = load_core_window_for_steps(deck_root, core_manifest, steps)
    cover = r340_s4.read_decision171_cover(ice_cover_path)
    runoff_12 = _load_monthly_12(deck_root / "runoff_core_monthly.nc", "sorunoff")
    sss_12 = _load_monthly_12(deck_root / "sss_data.nc", "sss")
    geothermal = jnp.asarray(
        ladder.read_geothermal_heatflow(deck_root, (148, 180)),
        dtype=jnp.float64,
    )
    with netCDF4.Dataset(deck_root / "ORCA_R2_zps_domcfg.nc", "r") as ds:
        ff_f = jnp.asarray(np.asarray(ds.variables["ff_f"][:], dtype=np.float64))
    with netCDF4.Dataset(deck_root / "runoff_core_monthly.nc", "r") as ds:
        mouth = jnp.asarray(np.asarray(ds.variables["socoefr0"][:], dtype=np.float64))

    state = recipe.initial_state
    mask_np = np.asarray(state.land_mask.data, dtype=np.float64)
    ny, nx = mask_np.shape
    require((ny, nx) == (148, 180), f"unexpected ORCA2 grid shape: {(ny, nx)}")
    active_np = np.asarray(recipe.z_coord.is_active) > 0.5
    wet_2d = mask_np > 0.5

    stress_lbc, tmask_east, tmask_north, umask, vmask = r340_s4._build_stress_lbc_and_masks(
        mask_np,
        np.asarray(state.u_mask.data, dtype=np.float64),
        np.asarray(state.v_mask.data, dtype=np.float64),
        recipe.grid.fold,
    )
    zero = jnp.zeros((ny, nx), dtype=jnp.float64)
    one = jnp.ones((ny, nx), dtype=jnp.float64)
    mask_jnp = jnp.asarray(mask_np, dtype=jnp.float64)
    cover_jnp = jnp.asarray(cover, dtype=jnp.float64)

    means_init = NemoSurfaceMeans(*(zero for _ in range(7)))
    s0_base = bulk.NemoS0Inputs(
        prior=means_init,
        kt=1,
        nit000=1,
        forcing_stride=2,
        restart_means_present=False,
        use_conservative_temperature=False,
        remove_inverse_barometer=False,
        u_before=zero,
        v_before=zero,
        temperature_now=zero,
        salinity_now=zero,
        ssh_now=zero,
        ssh_inverse_barometer=zero,
        e3t_reference=one,
        r3t_now=zero,
        surface_mask=mask_jnp,
        shortwave_fraction_top=one,
    )
    s2_base = bulk.NemoS2Inputs(
        stress_lbc=stress_lbc,
        post_ice_sfx=zero,
        runoff_now=zero,
        prior_runoff_mass=zero,
        prior_runoff_temperature_content=zero,
        prior_runoff_salinity_content=zero,
        prior_restoring_volume_flux=zero,
        prior_restoring_heat_flux=zero,
        prior_restoring_coefficient=one,
        sss_target=zero,
        runoff_mouth_mask=mouth,
        surface_mask=mask_jnp,
        tmask_east=jnp.asarray(tmask_east, dtype=jnp.float64),
        tmask_north=jnp.asarray(tmask_north, dtype=jnp.float64),
        umask=jnp.asarray(umask, dtype=jnp.float64),
        vmask=jnp.asarray(vmask, dtype=jnp.float64),
        freshwater_budget_correction=jnp.asarray(0.0, dtype=jnp.float64),
        runoff_factor=jnp.asarray(1.0, dtype=jnp.float64),
        reference_density=jnp.asarray(recipe.model_config.physics.constants.rho_0, dtype=jnp.float64),
        restoring_mm_per_day=jnp.asarray(-166.67, dtype=jnp.float64),
        restoring_bound_mm_per_day=jnp.asarray(4.0, dtype=jnp.float64),
        seconds_per_day=jnp.asarray(86400.0, dtype=jnp.float64),
        salinity_floor=jnp.asarray(1.0e-20, dtype=jnp.float64),
        heat_capacity=jnp.asarray(recipe.model_config.physics.constants.c_sw, dtype=jnp.float64),
    )

    dt_s = int(card.ocean.dt_s)
    carry_s0 = None
    carry_s2 = None
    prior_surface = None
    saved_snapshots: list[dict[str, Any]] = []
    extra_save_set = set(save_steps) | {steps}

    for kt in range(1, steps + 1):
        isecsbc = kt * dt_s
        retained_kt = kt if kt % 2 else kt - 1
        retained_clock = retained_kt * dt_s

        idx_6h = retained_clock // 21600
        idx_daily = retained_clock // 86400
        idx_monthly = int(np.searchsorted(MONTH_END_SEC, retained_clock, side="right"))
        if mapped_core["precip_raw"].shape[0] == 1:
            idx_monthly = 0

        atmosphere = bulk.NemoStandaloneAtmosphereFields(
            jnp.asarray(mapped_core["wndi"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["wndj"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["tair"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["humi"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["slp"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["qsr_down"][idx_daily], dtype=jnp.float64),
            jnp.asarray(mapped_core["qlw_down"][idx_daily], dtype=jnp.float64),
            jnp.asarray(mapped_core["precip_raw"][idx_monthly], dtype=jnp.float64),
            jnp.asarray(mapped_core["snow_raw"][idx_monthly], dtype=jnp.float64),
        )
        pair_runoff = _monthly_pair_for_clock(runoff_12, isecsbc)
        pair_sss = _monthly_pair_for_clock(sss_12, isecsbc)
        s0f = bulk.NemoStandaloneS0FInputs(
            atmosphere,
            atmosphere,
            isecsbc,
            isecsbc - 1,
            isecsbc + 1,
            pair_runoff,
            pair_sss,
        )
        ice_before_s, ice_after_s = (
            (-15768000, 15768000)
            if isecsbc <= 15768000
            else (15768000, 47304000)
        )
        ice_if = bulk.NemoSbcIceIfInputs(
            cover_jnp,
            cover_jnp,
            isecsbc,
            ice_before_s,
            ice_after_s,
            zero,
            zero,
            zero,
            ff_f,
            mask_jnp,
            mask_jnp,
        )
        s0 = (
            s0_base
            if carry_s0 is None
            else s0_base._replace(
                prior=carry_s0.prior,
                shortwave_fraction_top=carry_s0.shortwave_fraction_top,
            )
        )
        s2 = (
            s2_base
            if carry_s2 is None
            else s2_base._replace(
                prior_runoff_mass=carry_s2.prior_runoff_mass,
                prior_runoff_temperature_content=carry_s2.prior_runoff_temperature_content,
                prior_runoff_salinity_content=carry_s2.prior_runoff_salinity_content,
                prior_restoring_volume_flux=carry_s2.prior_restoring_volume_flux,
                prior_restoring_heat_flux=carry_s2.prior_restoring_heat_flux,
                prior_restoring_coefficient=carry_s2.prior_restoring_coefficient,
            )
        )
        chl = jnp.asarray(
            ladder.chlorophyll_at_step(deck_root, kt, wet_2d, dt_s=card.ocean.dt_s),
            dtype=jnp.float64,
        )
        sponge = build_orca2_damping_for_step(
            deck_root,
            active_np,
            kt=kt,
            dt_s=card.ocean.dt_s,
            mask_evaluation="nemo_tmask_multiply",
        )

        result = bulk.nemo_orca2_k0_ocean_step(
            card,
            state,
            model=model,
            nemo_kt=kt,
            s0f=s0f,
            ice_if=ice_if,
            s0=s0,
            s2=s2,
            chl=chl,
            geothermal_heat_flux_mwm2=geothermal,
            prior_surface=prior_surface,
            sponge=sponge,
        )
        carry_s0, carry_s2 = result.next_s0, result.next_s2
        prior_surface = result.surface
        state = result.ocean

        if kt % save_stride == 0 or kt in extra_save_set:
            snap_info = _save_snapshot(output_dir, kt, state, result.surface)
            saved_snapshots.append(snap_info)
            print(
                f"[orca2_r342_k0_driver] saved kt={kt:04d}/{steps:04d} "
                f"({snap_info['path']}) elapsed={time.monotonic() - t0:.1f}s",
                flush=True,
            )

    wall_seconds = time.monotonic() - t0
    devices = [
        {"id": int(d.id), "platform": d.platform, "device_kind": d.device_kind}
        for d in jax.devices()
    ]
    manifest = {
        "status": "PASS_R342_S4_K0_LEGOESM_RUN",
        "round": "orca2_r342",
        "steps": steps,
        "dt_s": dt_s,
        "save_stride": save_stride,
        "k0_config": card.k0._asdict(),
        "deck_root": str(deck_root),
        "ice_cover_path": str(ice_cover_path),
        "ice_cover_sha256": sha256_file(ice_cover_path),
        "output_dir": str(output_dir),
        "wall_seconds": wall_seconds,
        "jax_default_backend": jax.default_backend(),
        "jax_devices": devices,
        "snapshots": saved_snapshots,
        "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "worktree": worktree_stamp(allow_dirty=False),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20,
                        help="Number of K0 steps to run (20 for smoke, 2920 for 1-year)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    parser.add_argument("--core-manifest", type=Path, default=DEFAULT_CORE_MANIFEST)
    parser.add_argument("--ice-cover", type=Path, default=DEFAULT_ICE_COVER_PATH)
    parser.add_argument("--save-stride", type=int, default=STEPS_PER_DAY)
    parser.add_argument("--also-save-step", type=int, action="append", default=[10],
                        help="Additional step(s) to save (default includes kt=10 for R340 cross-check)")
    args = parser.parse_args(argv)

    manifest = run_standalone_k0_driver(
        steps=args.steps,
        output_dir=args.output_dir,
        deck_root=args.deck_root,
        core_manifest=args.core_manifest,
        ice_cover_path=args.ice_cover,
        save_stride=args.save_stride,
        save_steps=tuple(args.also_save_step),
    )
    print(json.dumps({
        "status": manifest["status"],
        "steps": manifest["steps"],
        "wall_seconds": round(manifest["wall_seconds"], 2),
        "jax_default_backend": manifest["jax_default_backend"],
        "snapshot_count": len(manifest["snapshots"]),
        "output_dir": manifest["output_dir"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
