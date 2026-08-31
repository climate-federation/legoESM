#!/usr/bin/env python
"""Score the preregistered 20-year DINO/NEMO climate-equivalence ensemble.

The statistic set and verdict arithmetic are frozen in
``docs/ocean/fidelity/PREREG_multi_year_climate_equivalence.md``.  This is an
offline CPU/fp64 scorer.  It never advances either model.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]
PARENT = HERE.parent
for path in (HERE, PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import acc_driver_decomp as D  # noqa: E402
import acc_thermal_wind as A  # noqa: E402
import basin_seasonal_decomp as B  # noqa: E402
import floor90_ensemble as F  # noqa: E402
from rebuild_nemo_restart import rebuild  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402
from legoesm.ocean.eos import make_eos_fn  # noqa: E402
from legoesm.ocean.experiments.dino import (  # noqa: E402
    dino_config_for_recipe,
    dino_lat_lon_model_config,
)
from legoesm.ocean.fidelity.nemo_io import (  # noqa: E402
    read_nemo_mesh_mask,
    read_nemo_restart,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (  # noqa: E402
    bridge_nemo_to_legoesm_topo,
)
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (  # noqa: E402
    _nemo_mld_from_n2_integral,
    _nemo_native_active_3d,
)
from legoesm.ocean.vertical import compute_ocean_jacobian  # noqa: E402

SESSION = "01a04e34-d1fb-73e0-b25a-177641f0a246"
PRODUCER = "ddd3a8476afd87da4afa5747eb7e5057490b893c"
NEMO_DINO = Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO")
MESH = NEMO_DINO / "RUN_TRAJ/mesh_mask.nc"
START = NEMO_DINO / "RUN_90D_TWIN/DINO_00005760_restart.nc"
PREREG = ROOT / "docs/ocean/fidelity/PREREG_multi_year_climate_equivalence.md"
EXPECTED_DAYS = tuple(range(360, 5401, 360)) + tuple(range(5430, 7201, 30))
MONTH_DAYS = tuple(range(5430, 7201, 30))
ANNUAL_DAYS = tuple(range(360, 5401, 360)) + tuple(range(5760, 7201, 360))
SLOPE_DAYS = tuple(range(3960, 7201, 360))
FIELDS = ("T3d", "S3d", "eta3d", "u3d", "v3d")
SNAPSHOT_RE = re.compile(r"^(T3d|S3d|eta3d|u3d|v3d)_day([0-9]+)$")
REGIONS = tuple((name, rows) for name, rows in D.LAT_GROUPS)
DEPTHS = (
    ("upper_lt200m", np.asarray(A.gdept1d, dtype=np.float64) < 200.0),
    ("interior_200_1400m", (np.asarray(A.gdept1d, dtype=np.float64) >= 200.0)
     & (np.asarray(A.gdept1d, dtype=np.float64) < 1400.0)),
    ("abyss_ge1400m", np.asarray(A.gdept1d, dtype=np.float64) >= 1400.0),
)
SIGMA_THRESHOLDS = (1.20, 1.40, 1.50, 1.60)
N_BOOT = 20_000
BOOT_SEED = 1455
EQUIV_R = 2.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_days_from_keys(keys: list[str] | tuple[str, ...]) -> dict[str, tuple[int, ...]]:
    """Parse the writer's actual ``FIELD_dayN`` schema, fail on asymmetry."""
    by_field: dict[str, list[int]] = {field: [] for field in FIELDS}
    for key in keys:
        match = SNAPSHOT_RE.fullmatch(key)
        if match:
            by_field[match.group(1)].append(int(match.group(2)))
    out = {field: tuple(sorted(values)) for field, values in by_field.items()}
    require(len(set(out.values())) == 1,
            f"per-field snapshot day sets disagree: {out}")
    require(out["T3d"] == EXPECTED_DAYS,
            f"snapshot days differ from preregistration: got {out['T3d']}")
    return out


def scalar(npz: np.lib.npyio.NpzFile, key: str):
    require(key in npz.files, f"missing NPZ key {key}")
    value = npz[key]
    require(value.shape == (), f"{key} is not scalar: {value.shape}")
    return value.item()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    require(values.ndim == weights.ndim == 1 and values.size == weights.size,
            "weighted quantile inputs must be aligned 1-D arrays")
    require(values.size > 0 and np.isfinite(values).all(),
            "weighted quantile values are empty/nonfinite")
    require(np.isfinite(weights).all() and np.all(weights > 0.0),
            "weighted quantile weights must be finite and positive")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    target = q * float(cumulative[-1])
    index = min(int(np.searchsorted(cumulative, target, side="left")), values.size - 1)
    return float(values[order[index]])


def monthly_summary(values: np.ndarray) -> dict[str, float | np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (60,), f"monthly series must have 60 values, got {values.shape}")
    matrix = values.reshape(5, 12)
    climatology = matrix.mean(axis=0, dtype=np.float64)
    residual = matrix - climatology[None, :]
    annual = matrix.mean(axis=1, dtype=np.float64)
    return {
        "mean": float(values.mean(dtype=np.float64)),
        "climatology": climatology,
        "seasonal_amplitude": float(climatology.max() - climatology.min()),
        "deseasonalized_std": float(np.std(residual, ddof=1)),
        "interannual_std": float(np.std(annual, ddof=1)),
    }


def slope_annual(values_by_day: dict[int, float]) -> float:
    y = np.asarray([values_by_day[d] for d in SLOPE_DAYS], dtype=np.float64)
    x = np.arange(11.0, 21.0, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def census(temp: np.ndarray, salt: np.ndarray, rho: np.ndarray,
           wet: np.ndarray, cell_volume: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    sigma = rho - A.RHO0
    for region, rows in REGIONS:
        row_mask = np.zeros((A.NY, 1, 1), dtype=bool)
        row_mask[rows, :, :] = True
        for depth, levels in DEPTHS:
            sel = wet & row_mask & levels[None, None, :]
            require(bool(sel.any()), f"empty census cell {region}/{depth}")
            weight = np.asarray(cell_volume[sel], dtype=np.float64)
            t = np.asarray(temp[sel], dtype=np.float64)
            s = np.asarray(salt[sel], dtype=np.float64)
            sig = np.asarray(sigma[sel], dtype=np.float64)
            require(np.isfinite(t).all() and np.isfinite(s).all() and np.isfinite(sig).all(),
                    f"nonfinite census input {region}/{depth}")
            wsum = float(weight.sum())
            tmean = float(np.sum(weight * t) / wsum)
            smean = float(np.sum(weight * s) / wsum)
            prefix = f"{region}.{depth}"
            out[f"{prefix}.T_mean"] = tmean
            out[f"{prefix}.S_mean"] = smean
            for q in (0.10, 0.50, 0.90):
                tag = f"q{int(q * 100):02d}"
                out[f"{prefix}.T_{tag}"] = weighted_quantile(t, weight, q)
                out[f"{prefix}.S_{tag}"] = weighted_quantile(s, weight, q)
            out[f"{prefix}.TS_covariance"] = float(
                np.sum(weight * (t - tmean) * (s - smean)) / wsum)
            for threshold in SIGMA_THRESHOLDS:
                out[f"{prefix}.sigma_gt_{threshold:.2f}"] = float(
                    np.sum(weight[sig > threshold]) / wsum)
    return out


def region_area_means(field: np.ndarray, wet2: np.ndarray,
                      area: np.ndarray) -> dict[str, float]:
    out = {}
    for region, rows in REGIONS:
        select = np.zeros_like(wet2)
        select[rows, :] = True
        mask = wet2 & select
        require(bool(mask.any()) and np.isfinite(field[mask]).all(),
                f"invalid regional field {region}")
        w = area[mask]
        out[region] = float(np.sum(w * field[mask]) / np.sum(w))
    return out


def mld_function():
    grid = read_nemo_mesh_mask(str(MESH), nn_hls=0)
    initial = read_nemo_restart(str(START), nn_hls=0)
    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    bridge = bridge_nemo_to_legoesm_topo(
        grid, initial, periodic_i=True, full_step=True, omega=cfg.omega,
        e3t_mode="both",
    )
    mc, _ = dino_lat_lon_model_config(bridge.geometry, cfg)
    z_coord = bridge.z_coord
    mask = jnp.asarray(bridge.state.land_mask.data, dtype=jnp.float64)
    bathy = jnp.asarray(bridge.state.H_bathy.data, dtype=jnp.float64)
    eos_fn = make_eos_fn(mc.eos, mc.eos_linear)
    active = _nemo_native_active_3d(mask, z_coord, bathy, jnp.float64)

    @jax.jit
    def evaluate(temp, salt, eta):
        jacobian = compute_ocean_jacobian(eta, bathy, z_coord)
        return _nemo_mld_from_n2_integral(
            temp, salt, mask, z_coord, eos_fn, mc.gm_redi.mld_rho_c,
            mc.g, mc.rho_0, active_3d=active, jacobian=jacobian,
        )[0]

    return evaluate, np.asarray(grid.e1t * grid.e2t, dtype=np.float64), \
        np.asarray(grid.tmask[..., 0]) > 0.5


def nemo_source(nemo_root: Path, member: int, day: int) -> Path:
    if member <= 2 and day == 360:
        return NEMO_DINO / f"RUN_VERDICT360_M{member}"
    phase = "phase_a" if day <= 5400 else "phase_b"
    return nemo_root / f"m{member}" / phase


def load_nemo(nemo_root: Path, member: int, day: int) -> dict[str, np.ndarray]:
    kt = 5760 + 32 * day
    source = nemo_source(nemo_root, member, day)
    pattern = str(source / f"DINO_{kt:08d}_restart_*.nc")
    raw = rebuild(pattern, ["tn", "sn", "un", "sshn"])
    require(set(raw) == {"tn", "sn", "un", "sshn"},
            f"incomplete NEMO state m{member} day{day}: {sorted(raw)}")
    yxz = lambda value: np.moveaxis(value, 0, -1)  # noqa: E731
    return {"T": yxz(raw["tn"]), "S": yxz(raw["sn"]),
            "u": yxz(raw["un"]), "eta": np.asarray(raw["sshn"], dtype=np.float64)}


def validate_lego(npz: np.lib.npyio.NpzFile, member: int,
                  schema0: dict[str, tuple[int, ...]] | None) -> dict[str, tuple[int, ...]]:
    schema = snapshot_days_from_keys(npz.files)
    if schema0 is not None:
        require(schema == schema0, f"m{member}: snapshot schema differs from m0")
    require(tuple(int(x) for x in npz["reduced_days"]) == EXPECTED_DAYS,
            f"m{member}: reduced days differ from preregistration")
    cfg = json.loads(str(scalar(npz, "run_config")))
    require(cfg["recipe"] == "nemo_dino_kamm_mlf" and cfg["n_days"] == 7200,
            f"m{member}: wrong recipe/horizon")
    require(cfg["snap_days"] == list(EXPECTED_DAYS), f"m{member}: config cadence mismatch")
    require(str(scalar(npz, "producer_git_sha")) == PRODUCER,
            f"m{member}: wrong producer")
    require(bool(scalar(npz, "stable")) and int(scalar(npz, "blew_up_at_step")) == -1,
            f"m{member}: unstable")
    require(str(scalar(npz, "control_dtype")) == "float64", f"m{member}: not fp64")
    require(str(scalar(npz, "nemo_ladder_mode")) == "both", f"m{member}: wrong ladder")
    require(str(scalar(npz, "twin_start_mode")) == "bridged", f"m{member}: wrong start")
    require(str(scalar(npz, "bridge_before_stress_stagger")) == "T",
            f"m{member}: wrong stress stagger")
    require(int(scalar(npz, "producer_dirty_tracked_files")) == 0,
            f"m{member}: dirty producer")
    storage = json.loads(str(scalar(npz, "storage_dtypes")))
    require(all(storage[field] == "float64" for field in (*FIELDS, "reduced")),
            f"m{member}: non-fp64 storage {storage}")
    return schema


def empty_base() -> dict[str, list[float] | list[np.ndarray]]:
    return {
        "acc.full": [], "acc.channel": [],
        **{f"basin.{name}": [] for name, _ in REGIONS},
        **{f"row.{j:03d}": [] for j in range(A.NY)},
        "density.upper": [], "density.deep": [],
        **{f"mld.{name}": [] for name, _ in REGIONS},
    }


def reduce_legoesm(path: Path, member: int, mld_eval, area, wet2,
                   cell_volume) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    base = empty_base()
    census_series: dict[str, list[float]] = {}
    with np.load(path, allow_pickle=False) as npz:
        schema = validate_lego(npz, member, None)
        reduced_days = np.asarray(npz["reduced_days"], dtype=np.int64)
        require(np.array_equal(reduced_days, np.asarray(EXPECTED_DAYS)), "reduced day mismatch")
        transport = {
            "acc.full": np.asarray(npz["reduced_acc_mean"], dtype=np.float64),
            "acc.channel": np.asarray(npz["reduced_band_c"], dtype=np.float64),
            **{f"basin.{name}": np.asarray(npz[f"reduced_{key}"], dtype=np.float64)
               for (name, _), key in zip(REGIONS, ("g_south", "g_band", "g_north"))},
        }
        rows = np.asarray(npz["reduced_row_sv"], dtype=np.float64)
        require(rows.shape == (75, A.NY), f"m{member}: wrong row series {rows.shape}")
        land = np.asarray(npz["land_mask"], dtype=np.float64)
        wet = A.tmask & (land[:, :, None] > 0.5)
        require(np.array_equal(wet[..., 0], wet2), f"m{member}: wet mask differs")
        for index, day in enumerate(EXPECTED_DAYS):
            for name, values in transport.items():
                base[name].append(float(values[index]))
            for j in range(A.NY):
                base[f"row.{j:03d}"].append(float(rows[index, j]))
            closure = sum(base[f"basin.{name}"][-1] for name, _ in REGIONS) - base["acc.full"][-1]
            require(abs(closure) <= 1.0e-9, f"m{member} day{day}: transport closure {closure}")
            base["density.upper"].append(float(npz["reduced_up"][index]))
            base["density.deep"].append(float(npz["reduced_deep"][index]))
            if day not in MONTH_DAYS:
                continue
            temp = np.asarray(npz[f"T3d_day{day}"], dtype=np.float64)
            salt = np.asarray(npz[f"S3d_day{day}"], dtype=np.float64)
            eta = np.asarray(npz[f"eta3d_day{day}"], dtype=np.float64)
            rho = A.rho_of({"T": temp, "S": salt}, wet)
            for key, value in census(temp, salt, rho, wet, cell_volume).items():
                census_series.setdefault(key, []).append(value)
            hml = np.asarray(mld_eval(jnp.asarray(temp), jnp.asarray(salt), jnp.asarray(eta)),
                             dtype=np.float64)
            for name, value in region_area_means(hml, wet2, area).items():
                base[f"mld.{name}"].append(value)
    return ({key: np.asarray(value, dtype=np.float64) for key, value in base.items()},
            {key: np.asarray(value, dtype=np.float64) for key, value in census_series.items()},
            schema)


def reduce_nemo(nemo_root: Path, member: int, mld_eval, area, wet2,
                cell_volume) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = empty_base()
    census_series: dict[str, list[float]] = {}
    wet = A.tmask
    for day in EXPECTED_DAYS:
        state = load_nemo(nemo_root, member, day)
        temp, salt, u = state["T"], state["S"], state["u"]
        require(np.isfinite(temp).all() and np.isfinite(salt).all()
                and np.isfinite(u).all() and np.isfinite(state["eta"]).all(),
                f"NEMO m{member} day{day}: nonfinite state")
        base["acc.full"].append(float(D._avg(D.section_total(u, A.umask))))
        base["acc.channel"].append(float(F.band_transport_campaign(u, A.umask)))
        for name, rows_sel in REGIONS:
            base[f"basin.{name}"].append(float(D._avg(D.group_transport(u, A.umask, rows_sel))))
        row = B.row_transport(u, A.umask, slice(0, A.NY))
        for j in range(A.NY):
            base[f"row.{j:03d}"].append(float(row[j]))
        closure = sum(base[f"basin.{name}"][-1] for name, _ in REGIONS) - base["acc.full"][-1]
        require(abs(closure) <= 1.0e-9,
                f"NEMO m{member} day{day}: transport closure {closure}")
        rho = A.rho_of(state, wet)
        upper, deep = A.depth_split(A.contrast_profile(rho, wet), wet)
        base["density.upper"].append(upper)
        base["density.deep"].append(deep)
        if day not in MONTH_DAYS:
            continue
        for key, value in census(temp, salt, rho, wet, cell_volume).items():
            census_series.setdefault(key, []).append(value)
        hml = np.asarray(mld_eval(jnp.asarray(temp), jnp.asarray(salt),
                                  jnp.asarray(state["eta"])), dtype=np.float64)
        for name, value in region_area_means(hml, wet2, area).items():
            base[f"mld.{name}"].append(value)
    return ({key: np.asarray(value, dtype=np.float64) for key, value in base.items()},
            {key: np.asarray(value, dtype=np.float64) for key, value in census_series.items()})


def scalar_statistics(base: dict[str, np.ndarray], census_series: dict[str, np.ndarray]) \
        -> dict[str, dict[str, float]]:
    index = {day: i for i, day in enumerate(EXPECTED_DAYS)}
    monthly_index = np.asarray([index[day] for day in MONTH_DAYS])
    out: dict[str, dict[str, float]] = {name: {} for name in
        ("acc_series", "basin_row_transports", "mld_seasonal_cycle",
         "ts_water_mass_census", "density_contrasts", "variability")}

    def add_climate(family: str, name: str, values: np.ndarray, slope: bool) -> None:
        monthly = values[monthly_index]
        summary = monthly_summary(monthly)
        out[family][f"{name}.mean"] = float(summary["mean"])
        if slope:
            out[family][f"{name}.slope_y11_y20"] = slope_annual(
                {day: float(values[index[day]]) for day in SLOPE_DAYS})
        for month, value in enumerate(summary["climatology"], 1):
            out[family][f"{name}.month{month:02d}"] = float(value)
        for metric in ("seasonal_amplitude", "deseasonalized_std", "interannual_std"):
            out[family][f"{name}.{metric}"] = float(summary[metric])
            out["variability"][f"{name}.{metric}"] = float(summary[metric])

    for name in ("acc.full", "acc.channel"):
        add_climate("acc_series", name, base[name], True)
    for name, _ in REGIONS:
        add_climate("basin_row_transports", f"basin.{name}", base[f"basin.{name}"], True)
    for j in range(A.NY):
        values = base[f"row.{j:03d}"]
        out["basin_row_transports"][f"row.{j:03d}.mean"] = float(values[monthly_index].mean())
        out["basin_row_transports"][f"row.{j:03d}.slope_y11_y20"] = slope_annual(
            {day: float(values[index[day]]) for day in SLOPE_DAYS})
    for name in ("density.upper", "density.deep"):
        add_climate("density_contrasts", name, base[name], True)
    for name, _ in REGIONS:
        values = base[f"mld.{name}"]
        summary = monthly_summary(values)
        out["mld_seasonal_cycle"][f"mld.{name}.mean"] = float(summary["mean"])
        for month, value in enumerate(summary["climatology"], 1):
            out["mld_seasonal_cycle"][f"mld.{name}.month{month:02d}"] = float(value)
        for metric in ("seasonal_amplitude", "deseasonalized_std", "interannual_std"):
            out["mld_seasonal_cycle"][f"mld.{name}.{metric}"] = float(summary[metric])
            out["variability"][f"mld.{name}.{metric}"] = float(summary[metric])
    for name, values in census_series.items():
        require(values.shape == (60,), f"census {name} has {values.shape}")
        out["ts_water_mass_census"][f"{name}.mean_y16_y20"] = float(values.mean())
        if name.endswith((".T_mean", ".S_mean")):
            summary = monthly_summary(values)
            for metric in ("seasonal_amplitude", "deseasonalized_std", "interannual_std"):
                out["variability"][f"census.{name}.{metric}"] = float(summary[metric])
    return out


def bootstrap_stat(lego: np.ndarray, nemo: np.ndarray, li: np.ndarray,
                   ni: np.ndarray) -> tuple[dict, np.ndarray | None]:
    lego = np.asarray(lego, dtype=np.float64)
    nemo = np.asarray(nemo, dtype=np.float64)
    lstd, nstd = float(np.std(lego, ddof=1)), float(np.std(nemo, ddof=1))
    floor = math.hypot(lstd, nstd)
    gap = float(np.mean(lego) - np.mean(nemo))
    scale = max(float(np.max(np.abs(lego))), float(np.max(np.abs(nemo))), 1.0)
    quantum = float(np.spacing(np.float64(scale)))
    unique_l, unique_n = int(np.unique(lego).size), int(np.unique(nemo).size)
    quantized = unique_l < 4 or unique_n < 4 or floor < 10.0 * quantum
    lb = lego[li]
    nb = nemo[ni]
    bf = np.hypot(np.std(lb, axis=1, ddof=1), np.std(nb, axis=1, ddof=1))
    bg = np.mean(lb, axis=1) - np.mean(nb, axis=1)
    valid = np.isfinite(bf) & (bf > 0.0)
    invalid_fraction = float(1.0 - np.mean(valid))
    quantized = quantized or invalid_fraction > 0.01
    ratio_samples = None
    if valid.any():
        ratio_samples = np.full(bf.shape, np.nan, dtype=np.float64)
        ratio_samples[valid] = np.abs(bg[valid]) / bf[valid]
        lo, hi = (float(x) for x in np.nanquantile(ratio_samples, (0.025, 0.975)))
    else:
        lo = hi = float("inf")
    ratio = abs(gap) / floor if floor > 0.0 else float("inf")
    if quantized:
        verdict = "UNRESOLVED_QUANTIZED"
    elif hi <= EQUIV_R:
        verdict = "CONFIRM"
    elif lo > EQUIV_R:
        verdict = "REFUTE"
    else:
        verdict = "UNRESOLVED"
    return ({"lego_mean": float(np.mean(lego)), "nemo_mean": float(np.mean(nemo)),
             "gap": gap, "lego_std": lstd, "nemo_std": nstd, "floor": floor,
             "R": ratio, "R_lo": lo, "R_hi": hi, "verdict": verdict,
             "unique_lego": unique_l, "unique_nemo": unique_n,
             "quantum": quantum, "invalid_bootstrap_fraction": invalid_fraction,
             "one_sided": bool(min(lstd, nstd) == 0.0 or
                               max(lstd, nstd) / max(min(lstd, nstd), quantum) > 10.0)},
            ratio_samples)


def score_families(lego: list[dict[str, dict[str, float]]],
                   nemo: list[dict[str, dict[str, float]]]) -> tuple[dict, str]:
    rng = np.random.default_rng(BOOT_SEED)
    li = rng.integers(0, 6, size=(N_BOOT, 6))
    ni = rng.integers(0, 6, size=(N_BOOT, 6))
    families = {}
    for family in lego[0]:
        names = tuple(sorted(lego[0][family]))
        require(all(tuple(sorted(member[family])) == names for member in lego + nemo),
                f"{family}: member statistic schemas differ")
        stats = {}
        family_max = np.zeros(N_BOOT, dtype=np.float64)
        family_valid = np.ones(N_BOOT, dtype=bool)
        for name in names:
            lv = np.asarray([member[family][name] for member in lego])
            nv = np.asarray([member[family][name] for member in nemo])
            row, samples = bootstrap_stat(lv, nv, li, ni)
            stats[name] = row
            if samples is None:
                family_valid[:] = False
            else:
                family_valid &= np.isfinite(samples)
                family_max = np.maximum(family_max, np.nan_to_num(samples, nan=0.0))
        simultaneous_hi = (float(np.quantile(family_max[family_valid], 0.975))
                           if family_valid.any() else float("inf"))
        verdicts = {row["verdict"] for row in stats.values()}
        if "REFUTE" in verdicts:
            verdict = "REFUTE"
        elif verdicts == {"CONFIRM"} and simultaneous_hi <= EQUIV_R:
            verdict = "CONFIRM"
        else:
            verdict = "UNRESOLVED"
        worst_name = max(stats, key=lambda name: stats[name]["R"])
        refuted = [name for name in names if stats[name]["verdict"] == "REFUTE"]
        unresolved = [name for name in names
                      if stats[name]["verdict"] == "UNRESOLVED"]
        candidates = refuted or unresolved or list(names)
        decisive_name = max(candidates, key=lambda name: stats[name]["R"])
        families[family] = {"verdict": verdict, "simultaneous_R_hi": simultaneous_hi,
                            "worst_statistic": worst_name,
                            "worst": stats[worst_name],
                            "decisive_statistic": decisive_name,
                            "decisive": stats[decisive_name], "statistics": stats}
    verdicts = {value["verdict"] for value in families.values()}
    if "REFUTE" in verdicts:
        overall = "DISTINGUISHABLE_AT_20Y"
    elif verdicts == {"CONFIRM"}:
        overall = "STATISTICALLY_INDISTINGUISHABLE_AT_20Y"
    else:
        overall = "UNRESOLVED_AT_20Y"
    return families, overall


def endpoint_floor_diagnostics(lego: list[dict[str, np.ndarray]],
                               nemo: list[dict[str, np.ndarray]]) -> dict:
    """Report horizon-matched spread growth without changing any verdict input."""
    endpoint_days = (360, 1800, 3600, 5400, 7200)
    day_index = {day: index for index, day in enumerate(EXPECTED_DAYS)}
    names = tuple(sorted(name for name, values in lego[0].items()
                         if np.asarray(values).shape == (len(EXPECTED_DAYS),)))
    require(all(tuple(sorted(name for name, values in member.items()
                             if np.asarray(values).shape == (len(EXPECTED_DAYS),))) == names
                for member in lego + nemo), "endpoint diagnostic schemas differ")
    out = {}
    for name in names:
        by_day = {}
        for day in endpoint_days:
            index = day_index[day]
            lv = np.asarray([member[name][index] for member in lego], dtype=np.float64)
            nv = np.asarray([member[name][index] for member in nemo], dtype=np.float64)
            lstd = float(np.std(lv, ddof=1))
            nstd = float(np.std(nv, ddof=1))
            floor = math.hypot(lstd, nstd)
            gap = float(np.mean(lv) - np.mean(nv))
            by_day[str(day)] = {
                "lego_mean": float(np.mean(lv)), "nemo_mean": float(np.mean(nv)),
                "gap": gap, "lego_std": lstd, "nemo_std": nstd, "floor": floor,
                "R": abs(gap) / floor if floor > 0.0 else float("inf"),
            }
        first, last = by_day["360"]["floor"], by_day["7200"]["floor"]
        out[name] = {"by_day": by_day,
                     "floor_growth_y20_over_y1": (last / first if first > 0.0
                                                   else float("inf"))}
    return {"days": list(endpoint_days), "statistics": out,
            "verdict_input": False}


def controls() -> dict:
    # Classifier branches and quantization plant.
    rng = np.random.default_rng(11)
    base = rng.normal(0.0, 1.0, 6)
    li = np.random.default_rng(BOOT_SEED).integers(0, 6, size=(N_BOOT, 6))
    ni = np.random.default_rng(BOOT_SEED + 1).integers(0, 6, size=(N_BOOT, 6))
    confirm, _ = bootstrap_stat(base, base + 0.01, li, ni)
    refute, _ = bootstrap_stat(base, base + 100.0, li, ni)
    unresolved, _ = bootstrap_stat(base, base + 2.0, li, ni)
    collapsed, _ = bootstrap_stat(np.ones(6), np.ones(6), li, ni)
    require(confirm["verdict"] == "CONFIRM", "CONFIRM classifier plant failed")
    require(refute["verdict"] == "REFUTE", "REFUTE classifier plant failed")
    require(unresolved["verdict"] == "UNRESOLVED", "UNRESOLVED classifier plant failed")
    require(collapsed["verdict"] == "UNRESOLVED_QUANTIZED",
            "quantization plant failed")
    family_lego = [{"plant": {"a": float(base[i]), "b": float(2.0 * base[i])}}
                   for i in range(6)]
    family_nemo = [{"plant": {"a": float(base[i] + 0.01),
                               "b": float(2.0 * base[i] + 0.01)}}
                   for i in range(6)]
    planted_family, planted_overall = score_families(family_lego, family_nemo)
    require(planted_family["plant"]["verdict"] == "CONFIRM"
            and planted_family["plant"]["simultaneous_R_hi"] <= EQUIV_R
            and planted_overall == "STATISTICALLY_INDISTINGUISHABLE_AT_20Y",
            "simultaneous family-maximum plant failed")

    # Reduction plants: each directly owns the named reduction behavior.
    wet = A.tmask.copy()
    volume = (np.asarray(A.mm["e1t"][0]).squeeze()
              * np.asarray(A.mm["e2t"][0]).squeeze())[:, :, None] * A.e3t0
    temp = np.zeros(wet.shape, dtype=np.float64)
    salt = np.full(wet.shape, 35.0, dtype=np.float64)
    rho = np.where(wet, A.RHO0 + 1.0, np.nan)
    base_census = census(temp, salt, rho, wet, volume)
    wet_cell = tuple(int(x) for x in np.argwhere(wet)[0])
    temp[wet_cell] = 0.25
    salt[wet_cell] += 0.10
    moved_census = census(temp, salt, rho, wet, volume)
    require(any(moved_census[k] != base_census[k] for k in base_census),
            "wet T census plant did not move")

    area = np.asarray(A.mm["e1t"][0]).squeeze() * np.asarray(A.mm["e2t"][0]).squeeze()
    mld0 = np.zeros(wet.shape[:2], dtype=np.float64)
    mld1 = mld0.copy()
    j, i = tuple(int(x) for x in np.argwhere(wet[..., 0])[0])
    mld1[j, i] = 100.0
    require(region_area_means(mld0, wet[..., 0], area)
            != region_area_means(mld1, wet[..., 0], area), "MLD plant did not move")
    u0 = np.zeros(A.umask.shape, dtype=np.float64)
    u1 = u0.copy()
    uj, ui0, uk = tuple(int(x) for x in np.argwhere(A.umask[:, 2:-2, :])[0])
    ui = ui0 + 2  # the campaign's longitude reducer excludes both halo rings
    u1[uj, ui, uk] = 1.0
    row0 = B.row_transport(u0, A.umask, slice(0, A.NY))
    row1 = B.row_transport(u1, A.umask, slice(0, A.NY))
    require(row1[uj] != row0[uj]
            and D._avg(D.section_total(u1, A.umask))
            != D._avg(D.section_total(u0, A.umask)),
            "wet U-row transport plant did not move")
    cycle = np.arange(60, dtype=np.float64)
    require(not np.array_equal(monthly_summary(cycle)["climatology"],
                               monthly_summary(np.roll(cycle, 1))["climatology"]),
            "monthly rotation plant did not move climatology")
    return {"classifier_confirm": True, "classifier_refute": True,
            "classifier_unresolved": True, "collapsed_quantized": True,
            "family_maximum_confirms": True, "wet_u_row_moves": True,
            "wet_census_cell_moves": True, "wet_mld_column_moves": True,
            "monthly_rotation_moves": True}


def archive_controls() -> dict:
    cases = {
        "y20": (NEMO_DINO / "RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_T.nc",
                 NEMO_DINO / "RUN_20Y_REBUILD/DINO_1y_00060101_00201230_grid_U.nc",
                 142.80981676941545),
        "y40": (NEMO_DINO / "RUN_40Y_REBUILD/DINO_1y_00210101_00401230_grid_T.nc",
                 NEMO_DINO / "RUN_40Y_REBUILD/DINO_1y_00210101_00401230_grid_U.nc",
                 176.12412451228047),
    }
    out = {}
    for name, (tpath, upath, expected) in cases.items():
        value = float(A.acc_full(A.load_nemo(str(tpath), -1, str(upath))["u"], A.umask))
        require(abs(value - expected) <= 1.0e-10, f"archive {name} ACC control failed")
        out[name] = {"measured_Sv": value, "expected_Sv": expected,
                     "absolute_error_Sv": abs(value - expected), "verdict_input": False}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reduced-output", type=Path, required=True)
    args = parser.parse_args(argv)

    require(jax.default_backend() == "cpu", "scorer must run on CPU")
    require(bool(jax.config.x64_enabled), "scorer requires JAX fp64")
    set_policy(PrecisionPolicy.fp64())
    require(PREREG.is_file(), f"missing preregistration {PREREG}")
    require(args.raw_manifest.is_file(), f"missing raw manifest {args.raw_manifest}")
    raw = json.loads(args.raw_manifest.read_text())
    require(raw["schema"] == "dino-climate-equivalence-20y-raw-v1"
            and raw["producer"] == PRODUCER and raw["session_id"] == SESSION,
            "raw manifest epoch mismatch")
    require(tuple(raw["days"]) == EXPECTED_DAYS and raw["verdict"] == "NOT_SCORED",
            "raw manifest cadence/status mismatch")

    plant_receipts = controls()
    archived = archive_controls()
    mld_eval, area, wet2 = mld_function()
    cell_volume = (np.asarray(A.mm["e1t"][0]).squeeze()
                   * np.asarray(A.mm["e2t"][0]).squeeze())[:, :, None] * A.e3t0
    require(np.array_equal(wet2, A.tmask[..., 0]), "MLD/census wet masks disagree")

    lego_stats, nemo_stats = [], []
    lego_bases, nemo_bases = [], []
    reduced_arrays = {}
    schema0 = None
    config0 = None
    for member in range(6):
        path = args.run_root / "arms" / f"m{member}.npz"
        require(path.is_file(), f"missing legoESM member {path}")
        with np.load(path, allow_pickle=False) as npz:
            schema = validate_lego(npz, member, schema0)
            cfg = json.loads(str(scalar(npz, "run_config")))
            cfg_compare = dict(cfg)
            cfg_compare.pop("perturb_seed", None)
            if schema0 is None:
                schema0, config0 = schema, cfg_compare
            else:
                require(cfg_compare == config0,
                        f"m{member}: resolved configuration differs beyond seed")
        print(f"REDUCE legoESM m{member}", flush=True)
        lbase, lcensus, _ = reduce_legoesm(path, member, mld_eval, area, wet2, cell_volume)
        lego_bases.append(lbase)
        lstats = scalar_statistics(lbase, lcensus)
        lego_stats.append(lstats)
        print(f"REDUCE NEMO m{member}", flush=True)
        nbase, ncensus = reduce_nemo(args.nemo_root, member, mld_eval, area, wet2, cell_volume)
        nemo_bases.append(nbase)
        nstats = scalar_statistics(nbase, ncensus)
        nemo_stats.append(nstats)
        for model, stats in (("lego", lstats), ("nemo", nstats)):
            for family, rows in stats.items():
                for name, value in rows.items():
                    reduced_arrays.setdefault(f"{model}.{family}.{name}", np.full(6, np.nan))[member] = value

    args.reduced_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.reduced_output, **reduced_arrays)
    families, overall = score_families(lego_stats, nemo_stats)
    endpoint_floors = endpoint_floor_diagnostics(lego_bases, nemo_bases)
    result = {
        "schema": "dino-climate-equivalence-20y-score-v1",
        "disposition": overall,
        "producer": PRODUCER,
        "session_id": SESSION,
        "horizon_model_years": 20,
        "model_days": 7200,
        "ensemble_members_per_side": 6,
        "bootstrap": {"draws": N_BOOT, "seed": BOOT_SEED,
                      "equivalence_R": EQUIV_R,
                      "bands": {"CONFIRM": "R_hi <= 2",
                                "REFUTE": "R_lo > 2",
                                "UNRESOLVED": "otherwise"}},
        "raw_manifest": {"path": str(args.raw_manifest),
                         "sha256": sha256(args.raw_manifest)},
        "reduced_artifact": {"path": str(args.reduced_output),
                             "sha256": sha256(args.reduced_output)},
        "snapshot_schema": {field: list(days) for field, days in schema0.items()},
        "endpoint_floor_growth": endpoint_floors,
        "controls": {"plants": plant_receipts, "historical_archives": archived},
        "families": families,
        "scope": {"recipe": "nemo_dino_kamm_mlf",
                  "claim": "20-year horizon climate distributions, not state tracking",
                  "out_of_scope": ["standalone nemo_dino_kamm", "cross-recipe transfer",
                                   "equilibrium claim"]},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"CAPSTONE disposition={overall}")
    for family, row in families.items():
        decisive = row["decisive"]
        print(f"FAMILY {family} verdict={row['verdict']} "
              f"decisive={row['decisive_statistic']} "
              f"gap={decisive['gap']:.12e} floor={decisive['floor']:.12e} "
              f"R={decisive['R']:.6g} "
              f"CI=[{decisive['R_lo']:.6g},{decisive['R_hi']:.6g}] "
              f"simultaneous_R_hi={row['simultaneous_R_hi']:.6g}")
    print(f"SCORE {args.output} sha256={sha256(args.output)}")
    print(f"REDUCED {args.reduced_output} sha256={sha256(args.reduced_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
