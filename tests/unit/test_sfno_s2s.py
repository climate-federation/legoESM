"""Unit tests for the local SFNO S2S package.

Requires the ``[ml]`` extras (equinox, optax).
"""

from __future__ import annotations

from pathlib import Path

import pytest

eqx = pytest.importorskip("equinox", reason="requires legoesm[ml] extras")

import jax.numpy as jnp
import numpy as np
import xarray as xr

from legoesm.ml.loss import almost_fair_crps
from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.s2s.sfno_slab import coupling as s2s_coupling
from legoesm.ml.s2s.sfno_slab import training as s2s_training
from legoesm.ml.s2s.sfno_slab import (
    ArcoSurfaceForcingConfig,
    ChaosBenchS2SConfig,
    LAND_SEA_MASK_VAR,
    S2SStochasticConfig,
    S2STrainingConfig,
    available_s2s_dates,
    build_target_grid,
    compute_ensemble_daily_metrics,
    compute_daily_metrics,
    count_trainable_parameters,
    coupled_rollout_to_dataset,
    create_s2s_dataloader,
    create_s2s_training_iterator,
    denormalize_atmospheric_channels,
    estimate_sfno_parameter_count,
    extra_input_channels,
    load_s2s_sample,
    parse_window_specs,
    prepare_arco_surface_forcing,
    regrid_channels_to_gaussian,
    resolve_s2s_sample_dates,
    rollout_with_forcing,
    summarize_window_metrics,
    train_sfno_s2s,
)


def _write_minimal_s2s_store(tmp_path: Path) -> ChaosBenchS2SConfig:
    target = build_target_grid(3)
    source_lat = np.linspace(90.0, -90.0, 7)
    source_lon = target.longitude_deg.copy()
    levels = [500, 850]
    dates = ["20000101", "20000102", "20000103", "20000104"]

    for subdir in ("era5", "lra5", "oras5", "climatology"):
        (tmp_path / subdir).mkdir()

    for day_index, date in enumerate(dates):
        era5 = xr.Dataset(
            {
                "t": xr.DataArray(
                    np.stack(
                        [
                            np.full((source_lat.size, source_lon.size), 270.0 + day_index, dtype=np.float32),
                            np.full((source_lat.size, source_lon.size), 280.0 + day_index, dtype=np.float32),
                        ]
                    ),
                    coords={"level": levels, "latitude": source_lat, "longitude": source_lon},
                    dims=("level", "latitude", "longitude"),
                ),
                "q": xr.DataArray(
                    np.stack(
                        [
                            np.full((source_lat.size, source_lon.size), 1.0e-3 + 1.0e-4 * day_index, dtype=np.float32),
                            np.full((source_lat.size, source_lon.size), 2.0e-3 + 1.0e-4 * day_index, dtype=np.float32),
                        ]
                    ),
                    coords={"level": levels, "latitude": source_lat, "longitude": source_lon},
                    dims=("level", "latitude", "longitude"),
                ),
            },
            coords={"time": np.datetime64(f"2000-01-{day_index + 1:02d}")},
        )
        era5.to_zarr(tmp_path / "era5" / f"era5_full_1.5deg_{date}.zarr")

        oras5 = xr.Dataset(
            {
                "sosstsst": xr.DataArray(
                    np.full((source_lat.size, source_lon.size), 290.0 + day_index, dtype=np.float32),
                    coords={"latitude": source_lat, "longitude": source_lon},
                    dims=("latitude", "longitude"),
                ),
                "ileadfra": xr.DataArray(
                    np.zeros((source_lat.size, source_lon.size), dtype=np.float32),
                    coords={"latitude": source_lat, "longitude": source_lon},
                    dims=("latitude", "longitude"),
                ),
                "iicethic": xr.DataArray(
                    np.zeros((source_lat.size, source_lon.size), dtype=np.float32),
                    coords={"latitude": source_lat, "longitude": source_lon},
                    dims=("latitude", "longitude"),
                ),
            }
        )
        oras5.to_zarr(tmp_path / "oras5" / f"oras5_full_1.5deg_{date}.zarr")

        lra5 = xr.Dataset(
            {
                "sp": xr.DataArray(
                    np.full((source_lat.size, source_lon.size), 101325.0, dtype=np.float32),
                    coords={"latitude": source_lat, "longitude": source_lon},
                    dims=("latitude", "longitude"),
                ),
                "ssrd": xr.DataArray(
                    np.full((source_lat.size, source_lon.size), 100.0, dtype=np.float32),
                    coords={"latitude": source_lat, "longitude": source_lon},
                    dims=("latitude", "longitude"),
                ),
                "strd": xr.DataArray(
                    np.full((source_lat.size, source_lon.size), 50.0, dtype=np.float32),
                    coords={"latitude": source_lat, "longitude": source_lon},
                    dims=("latitude", "longitude"),
                ),
            }
        )
        lra5.to_zarr(tmp_path / "lra5" / f"lra5_full_1.5deg_{date}.zarr")

    xr.Dataset(
        {
            "mean": xr.DataArray(
                np.zeros(4, dtype=np.float32),
                coords={"param": ["t-500", "t-850", "q-500", "q-850"]},
                dims=("param",),
            ),
            "sigma": xr.DataArray(
                np.ones(4, dtype=np.float32),
                coords={"param": ["t-500", "t-850", "q-500", "q-850"]},
                dims=("param",),
            ),
        }
    ).to_zarr(tmp_path / "climatology" / "climatology_era5.zarr")

    xr.Dataset(
        {
            "mean": xr.DataArray(
                np.zeros(1, dtype=np.float32),
                coords={"param": ["sosstsst"]},
                dims=("param",),
            ),
            "sigma": xr.DataArray(
                np.ones(1, dtype=np.float32),
                coords={"param": ["sosstsst"]},
                dims=("param",),
            ),
        }
    ).to_zarr(tmp_path / "climatology" / "climatology_oras5.zarr")

    return ChaosBenchS2SConfig(
        years=(2000,),
        data_dir=str(tmp_path),
        atmosphere_vars=("t", "q"),
        pressure_levels=(500, 850),
        ocean_vars=("sosstsst",),
        n_steps=2,
        lead_time=1,
        gaussian_n_max=3,
    )


def _write_minimal_arco_sst_cache(tmp_path: Path) -> tuple[Path, Path]:
    cache_dir = tmp_path / "arco_sst_cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "arco_sst_daily_20000101_20000104.zarr"
    stats_path = cache_dir / "arco_sst_daily_20000101_20000104_stats.zarr"

    latitude = np.linspace(90.0, -90.0, 7, dtype=np.float32)
    longitude = build_target_grid(3).longitude_deg.astype(np.float32)
    times = np.array(
        [
            np.datetime64("2000-01-01T00:00:00"),
            np.datetime64("2000-01-02T00:00:00"),
            np.datetime64("2000-01-03T00:00:00"),
            np.datetime64("2000-01-04T00:00:00"),
        ]
    )
    sst = np.stack(
        [
            np.full((latitude.size, longitude.size), 10.0 + day, dtype=np.float32)
            for day in range(times.size)
        ],
        axis=0,
    )
    ocean_mask = np.ones_like(sst, dtype=np.int8)
    ocean_mask[:, :2, :] = 0
    sst = np.where(ocean_mask.astype(bool), sst, np.nan).astype(np.float32)

    xr.Dataset(
        data_vars={
            "sosstsst": (("time", "latitude", "longitude"), sst),
            "ocean_mask": (("time", "latitude", "longitude"), ocean_mask),
        },
        coords={
            "time": times,
            "date": ("time", np.asarray([str(t)[:10] for t in times], dtype="U10")),
            "latitude": latitude,
            "longitude": longitude,
        },
    ).to_zarr(cache_path)

    xr.Dataset(
        {
            "mean": xr.DataArray(
                np.asarray([11.5], dtype=np.float32),
                coords={"param": ["sosstsst"]},
                dims=("param",),
            ),
            "sigma": xr.DataArray(
                np.asarray([1.0], dtype=np.float32),
                coords={"param": ["sosstsst"]},
                dims=("param",),
            ),
        }
    ).to_zarr(stats_path)

    return cache_path, stats_path


def test_regrid_channels_to_gaussian_handles_descending_latitude():
    target = build_target_grid(3)
    source_lat = np.linspace(90.0, -90.0, 7)
    source_lon = target.longitude_deg.copy()
    base = np.broadcast_to(source_lat[:, None], (source_lat.size, source_lon.size))
    channels = np.stack([base.astype(np.float32)], axis=0)

    out = regrid_channels_to_gaussian(
        channels,
        source_lat=source_lat,
        source_lon=source_lon,
        target=target,
    )

    assert out.shape == (1, target.grid.n_lat, target.grid.n_lon)
    assert np.allclose(out[0, :, 0], target.latitude_deg, atol=2.0)


def test_regrid_channels_to_gaussian_handles_reduced_longitude_count():
    target = build_target_grid(3)
    source_lat = np.linspace(-90.0, 90.0, 7)
    source_lon = np.linspace(0.0, 315.0, 8, dtype=np.float64)
    lon_field = np.broadcast_to(source_lon[None, :], (source_lat.size, source_lon.size))
    channels = np.stack([lon_field.astype(np.float32)], axis=0)

    out = regrid_channels_to_gaussian(
        channels,
        source_lat=source_lat,
        source_lon=source_lon,
        target=target,
    )

    assert out.shape == (1, target.grid.n_lat, target.grid.n_lon)
    assert np.isfinite(out).all()


def test_load_s2s_sample_packs_atmosphere_and_forcing_channels(tmp_path: Path):
    target = build_target_grid(3)
    config = _write_minimal_s2s_store(tmp_path)
    dates = ["20000101", "20000102", "20000103", "20000104"]

    x, y, forcing = load_s2s_sample(config, sample_index=0)

    assert x.shape == (target.grid.n_lat, target.grid.n_lon, 5)
    assert y.shape == (2, target.grid.n_lat, target.grid.n_lon, 4)
    assert forcing.shape == (2, target.grid.n_lat, target.grid.n_lon, 1)
    assert np.allclose(x[..., -1], 290.0, atol=1.0e-4)
    assert np.allclose(forcing[0, ..., 0], 291.0, atol=1.0e-4)
    assert np.allclose(y[0, ..., 0], 271.0, atol=1.0e-4)
    assert available_s2s_dates(config) == dates
    assert resolve_s2s_sample_dates(config, sample_index=0) == ("20000101", ["20000102", "20000103"])
    denorm_y = denormalize_atmospheric_channels(y, config)
    assert np.allclose(denorm_y[0, ..., 0], 271.0, atol=1.0e-4)


def test_load_s2s_sample_supports_arco_sst_cache_and_land_sea_mask(tmp_path: Path):
    target = build_target_grid(3)
    _write_minimal_s2s_store(tmp_path)
    cache_path, stats_path = _write_minimal_arco_sst_cache(tmp_path)

    config = ChaosBenchS2SConfig(
        years=(2000,),
        data_dir=str(tmp_path),
        atmosphere_vars=("t", "q"),
        pressure_levels=(500, 850),
        ocean_vars=("sosstsst", LAND_SEA_MASK_VAR),
        ocean_source="arco_sst",
        arco_sst_cache_path=str(cache_path),
        arco_sst_stats_path=str(stats_path),
        n_steps=2,
        lead_time=1,
        gaussian_n_max=3,
    )

    x, y, forcing = load_s2s_sample(config, sample_index=0)
    mask = x[..., 5]

    assert x.shape == (target.grid.n_lat, target.grid.n_lon, 6)
    assert y.shape == (2, target.grid.n_lat, target.grid.n_lon, 4)
    assert forcing.shape == (2, target.grid.n_lat, target.grid.n_lon, 2)
    assert np.nanmin(x[..., 4]) < -1.0
    assert np.nanmax(x[..., 4]) <= 0.1
    assert np.any(mask < 0.2)
    assert np.any(mask > 0.8)
    assert np.nanmin(mask) >= 0.0
    assert np.nanmax(mask) <= 1.0


def test_torch_dataloader_and_iterator_preserve_sfno_s2s_batch_contract(tmp_path: Path):
    pytest.importorskip("torch", reason="Torch required for DataLoader test")
    target = build_target_grid(3)
    config = _write_minimal_s2s_store(tmp_path)

    loader = create_s2s_dataloader(
        config,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    batch_x, batch_y, batch_forcing = next(iter(loader))
    assert batch_x.shape == (1, target.grid.n_lat, target.grid.n_lon, 5)
    assert batch_y.shape == (1, 2, target.grid.n_lat, target.grid.n_lon, 4)
    assert batch_forcing.shape == (1, 2, target.grid.n_lat, target.grid.n_lon, 1)

    iterator = create_s2s_training_iterator(
        config,
        batch_size=1,
        seed=0,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    x, y, forcing = next(iterator)
    assert x.shape == (1, target.grid.n_lat, target.grid.n_lon, 5)
    assert y.shape == (1, 2, target.grid.n_lat, target.grid.n_lon, 4)
    assert forcing.shape == (1, 2, target.grid.n_lat, target.grid.n_lon, 1)


class _DummyAtmosModel(eqx.Module):
    def __call__(self, x, grid):
        del grid
        return x[..., :2] + x[..., -1:]


class _ConstantTendencyModel(eqx.Module):
    tendency: float = 1.0

    def __call__(self, x, grid):
        del grid
        return jnp.full(x[..., :2].shape, self.tendency, dtype=jnp.float32)


def test_rollout_with_forcing_injects_input_only_forcing_channels():
    target = build_target_grid(3)
    initial = jnp.zeros((target.grid.n_lat, target.grid.n_lon, 3), dtype=jnp.float32)
    initial = initial.at[..., -1].set(1.0)
    forcing = jnp.stack(
        [
            jnp.full((target.grid.n_lat, target.grid.n_lon, 1), 2.0, dtype=jnp.float32),
            jnp.full((target.grid.n_lat, target.grid.n_lon, 1), 3.0, dtype=jnp.float32),
        ],
        axis=0,
    )

    predictions = rollout_with_forcing(_DummyAtmosModel(), initial, forcing, target.grid)

    assert predictions.shape == (2, target.grid.n_lat, target.grid.n_lon, 2)
    assert jnp.allclose(predictions[0], 1.0)
    assert jnp.allclose(predictions[1], 3.0)


def test_rollout_with_forcing_is_seeded_when_noise_channels_are_enabled():
    target = build_target_grid(3)
    initial = jnp.zeros((target.grid.n_lat, target.grid.n_lon, 3), dtype=jnp.float32)
    initial = initial.at[..., :2].set(1.0)
    forcing = jnp.zeros((2, target.grid.n_lat, target.grid.n_lon, 1), dtype=jnp.float32)
    stochastic = S2SStochasticConfig(
        ensemble_members=2,
        noise_channels=1,
        noise_lat=2,
        noise_lon=4,
        use_time_signal=False,
        afcrps_alpha=0.95,
    )

    pred_a = rollout_with_forcing(
        _DummyAtmosModel(),
        initial,
        forcing,
        target.grid,
        rng_key=jnp.array([0, 1], dtype=jnp.uint32),
        stochastic_config=stochastic,
    )
    pred_b = rollout_with_forcing(
        _DummyAtmosModel(),
        initial,
        forcing,
        target.grid,
        rng_key=jnp.array([0, 1], dtype=jnp.uint32),
        stochastic_config=stochastic,
    )
    pred_c = rollout_with_forcing(
        _DummyAtmosModel(),
        initial,
        forcing,
        target.grid,
        rng_key=jnp.array([2, 3], dtype=jnp.uint32),
        stochastic_config=stochastic,
    )

    assert jnp.allclose(pred_a, pred_b)
    assert not jnp.allclose(pred_a, pred_c)


def test_rollout_with_forcing_tendency_prediction_accumulates_state():
    target = build_target_grid(3)
    initial = jnp.zeros((target.grid.n_lat, target.grid.n_lon, 3), dtype=jnp.float32)
    forcing = jnp.zeros((2, target.grid.n_lat, target.grid.n_lon, 1), dtype=jnp.float32)

    predictions = rollout_with_forcing(
        _ConstantTendencyModel(),
        initial,
        forcing,
        target.grid,
        tendency_prediction=True,
    )

    assert predictions.shape == (2, target.grid.n_lat, target.grid.n_lon, 2)
    assert jnp.allclose(predictions[0], 1.0)
    assert jnp.allclose(predictions[1], 2.0)


def test_almost_fair_crps_matches_paper_coefficient():
    ensemble = jnp.asarray([[0.0], [2.0], [3.0]], dtype=jnp.float32)
    target = jnp.asarray([1.5], dtype=jnp.float32)
    alpha = 0.95
    n_members = ensemble.shape[0]

    obs_term = jnp.mean(jnp.abs(ensemble - target[None, ...]), axis=0)
    pairwise_sum = jnp.sum(
        jnp.abs(ensemble[:, None, ...] - ensemble[None, :, ...]),
        axis=(0, 1),
    )
    expected = obs_term - (
        (n_members - 1 + alpha)
        / (2.0 * n_members * n_members * (n_members - 1))
    ) * pairwise_sum

    actual = almost_fair_crps(ensemble, target, alpha=alpha)
    assert jnp.allclose(actual, expected)


def test_periodic_noise_resize_keeps_wrap_gap_comparable_to_interior():
    coarse = jnp.asarray([[[0.0, 1.0]]], dtype=jnp.float32)
    resized = s2s_training._resize_noise_periodic_longitude(coarse, n_lat=1, n_lon=8)
    lon_values = np.asarray(resized[0, 0], dtype=np.float32)
    wrap_gap = float(abs(lon_values[-1] - lon_values[0]))
    interior_gaps = np.abs(np.diff(lon_values))

    assert interior_gaps.size > 0
    assert wrap_gap <= float(np.max(interior_gaps)) + 1.0e-6


def test_extra_input_channels_counts_noise_and_time_signal():
    assert extra_input_channels(S2SStochasticConfig()) == 0
    assert extra_input_channels(S2SStochasticConfig(noise_channels=2, use_time_signal=False)) == 2
    assert extra_input_channels(S2SStochasticConfig(noise_channels=2, use_time_signal=True)) == 3


def test_sfno_parameter_estimate_matches_actual_count():
    target = build_target_grid(3)
    model = SFNO(
        SFNOConfig(
            in_channels=5,
            out_channels=4,
            embed_dim=8,
            n_blocks=2,
            mlp_expansion=2,
        ),
        grid=target.grid,
        key=jnp.array([0, 1], dtype=jnp.uint32),
    )

    estimated = estimate_sfno_parameter_count(
        n_sh=target.grid.n_sh,
        in_channels=5,
        out_channels=4,
        embed_dim=8,
        n_blocks=2,
        mlp_expansion=2,
    )
    actual = count_trainable_parameters(model)

    assert estimated == actual


def test_compute_daily_and_window_metrics():
    target = build_target_grid(3)
    lead_day = np.array([1, 2], dtype=np.int32)
    lat = target.latitude_deg.astype(np.float32)
    lon = target.longitude_deg.astype(np.float32)
    truth = np.zeros((2, target.grid.n_lat, target.grid.n_lon, 1), dtype=np.float32)
    pred = np.ones_like(truth)
    ds = xr.Dataset(
        data_vars={
            "prediction": (("lead_day", "latitude", "longitude", "channel"), pred),
            "target": (("lead_day", "latitude", "longitude", "channel"), truth),
        },
        coords={
            "lead_day": lead_day,
            "latitude": lat,
            "longitude": lon,
            "channel": np.array(["t-850"], dtype="U8"),
        },
        attrs={"gaussian_n_max": 3},
    )

    daily = compute_daily_metrics(ds, fields=["t-850"])
    windows = summarize_window_metrics(daily, windows=parse_window_specs("wk1=1:2"))

    assert len(daily) == 2
    assert all(row["field"] == "t-850" for row in daily)
    assert np.allclose([row["rmse"] for row in daily], 1.0)
    assert np.allclose([row["mae"] for row in daily], 1.0)
    assert len(windows) == 1
    assert windows[0]["window"] == "wk1"
    assert np.isclose(windows[0]["rmse"], 1.0)
    assert np.isclose(windows[0]["mae"], 1.0)


def test_compute_daily_metrics_for_forcing_channel():
    target = build_target_grid(3)
    forcing = np.ones((2, target.grid.n_lat, target.grid.n_lon, 1), dtype=np.float32)
    target_forcing = np.zeros_like(forcing)
    ds = xr.Dataset(
        data_vars={
            "prediction": (
                ("lead_day", "latitude", "longitude", "channel"),
                np.zeros((2, target.grid.n_lat, target.grid.n_lon, 1), dtype=np.float32),
            ),
            "target": (
                ("lead_day", "latitude", "longitude", "channel"),
                np.zeros((2, target.grid.n_lat, target.grid.n_lon, 1), dtype=np.float32),
            ),
            "forcing": (("lead_day", "latitude", "longitude", "forcing_channel"), forcing),
            "target_forcing": (
                ("lead_day", "latitude", "longitude", "forcing_channel"),
                target_forcing,
            ),
        },
        coords={
            "lead_day": np.array([1, 2], dtype=np.int32),
            "latitude": target.latitude_deg.astype(np.float32),
            "longitude": target.longitude_deg.astype(np.float32),
            "channel": np.array(["t-850"], dtype="U8"),
            "forcing_channel": np.array(["sosstsst"], dtype="U8"),
        },
        attrs={"gaussian_n_max": 3},
    )

    rows = compute_daily_metrics(ds, fields=["sosstsst"])
    assert len(rows) == 2
    assert all(row["field"] == "sosstsst" for row in rows)
    assert np.allclose([row["rmse"] for row in rows], 1.0)


def test_compute_ensemble_daily_metrics_uses_ensemble_mean_and_crps():
    target = build_target_grid(3)
    truth = np.full((2, target.grid.n_lat, target.grid.n_lon, 1), 2.0, dtype=np.float32)
    member_a = np.full_like(truth, 1.0)
    member_b = np.full_like(truth, 3.0)

    def _dataset(prediction: np.ndarray) -> xr.Dataset:
        return xr.Dataset(
            data_vars={
                "prediction": (("lead_day", "latitude", "longitude", "channel"), prediction),
                "target": (("lead_day", "latitude", "longitude", "channel"), truth),
            },
            coords={
                "lead_day": np.array([1, 2], dtype=np.int32),
                "latitude": target.latitude_deg.astype(np.float32),
                "longitude": target.longitude_deg.astype(np.float32),
                "channel": np.array(["t-850"], dtype="U8"),
            },
            attrs={"gaussian_n_max": 3},
        )

    rows = compute_ensemble_daily_metrics([_dataset(member_a), _dataset(member_b)], fields=["t-850"])
    assert len(rows) == 2
    assert all(row["field"] == "t-850" for row in rows)
    assert all(row["n_members"] == 2 for row in rows)
    assert np.allclose([row["rmse"] for row in rows], 0.0)
    assert np.allclose([row["mae"] for row in rows], 0.0)
    assert np.allclose([row["crps"] for row in rows], 0.5)


def test_coupling_helpers_fill_nans_and_use_sst_unit_roundtrip():
    field = np.array([[1.0, np.nan], [3.0, np.nan]], dtype=np.float32)
    filled = s2s_coupling._fill_nan_with_mean(field, default=0.0)
    assert np.isfinite(filled).all()
    assert np.allclose(filled, np.array([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32))

    ocean_mask = np.array([[True, False], [True, True]])
    sst_celsius = np.array([[10.0, np.nan], [15.0, 20.0]], dtype=np.float32)
    assert s2s_coupling._sst_uses_celsius_units(sst_celsius, ocean_mask)
    sst_kelvin = s2s_coupling._sst_to_ocean_units(sst_celsius, uses_celsius=True)
    restored = s2s_coupling._sst_from_ocean_units(sst_kelvin, uses_celsius=True)
    assert np.allclose(restored[ocean_mask], sst_celsius[ocean_mask])


def test_sea_ice_cover_uses_thickness_and_handles_missing_values():
    leadfra = np.array([[0.0, 1.0], [0.2, 0.8]], dtype=np.float32)
    thickness = np.array([[0.0, 0.1], [np.nan, 0.06]], dtype=np.float32)

    cover = s2s_coupling._sea_ice_cover_from_oras(leadfra, thickness)

    assert np.array_equal(
        cover,
        np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )


def test_prepare_arco_surface_forcing_regrids_and_preserves_sst_mask(tmp_path: Path):
    start = np.datetime64("2000-01-01T00:00:00")
    times = np.arange(start, start + np.timedelta64(49, "h"), np.timedelta64(1, "h"))
    lat = np.linspace(-90.0, 90.0, 5, dtype=np.float32)
    lon = np.linspace(0.0, 288.0, 5, dtype=np.float32)

    sst = np.full((times.size, lat.size, lon.size), 12.0, dtype=np.float32)
    sst[..., 0] = np.nan
    sea_ice = np.zeros_like(sst)
    sp = np.full_like(sst, 101325.0)
    sw = np.full_like(sst, 100.0)
    lw = np.full_like(sst, 50.0)

    xr.Dataset(
        {
            "surface_pressure": (("time", "latitude", "longitude"), sp),
            "sea_surface_temperature": (("time", "latitude", "longitude"), sst),
            "sea_ice_cover": (("time", "latitude", "longitude"), sea_ice),
            "surface_solar_radiation_downwards": (("time", "latitude", "longitude"), sw),
            "surface_thermal_radiation_downwards": (("time", "latitude", "longitude"), lw),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    ).to_zarr(tmp_path / "arco.zarr")

    ds = prepare_arco_surface_forcing(
        start_time=str(start),
        forecast_days=2,
        gaussian_n_max=3,
        config=ArcoSurfaceForcingConfig(era5_store=str(tmp_path / "arco.zarr")),
    )

    assert ds.sizes["lead_day"] == 2
    assert "sea_surface_temperature" in ds
    assert "sw_down" in ds
    assert np.isnan(ds["initial_sea_surface_temperature"].values).any()
    assert np.isfinite(ds["surface_pressure"].values).all()


class _PassThroughCoupledAtmosModel(eqx.Module):
    def __call__(self, x, grid):
        del grid
        return x[..., :4]


def test_coupled_rollout_targets_future_sst_without_external_surface_forcing(tmp_path: Path):
    config = _write_minimal_s2s_store(tmp_path)._replace(normalize=False)

    ds = coupled_rollout_to_dataset(
        _PassThroughCoupledAtmosModel(),
        config,
        sample_index=0,
        coupled=True,
        surface_forcing=None,
        stochastic_config=S2SStochasticConfig(
            ensemble_members=1,
            noise_channels=0,
            use_time_signal=False,
        ),
    )

    assert ds.sizes["lead_day"] == 2
    assert np.isclose(
        float(ds["target_sea_surface_temperature"].isel(lead_day=0, latitude=0, longitude=0)),
        291.0,
    )
    assert np.isclose(
        float(ds["target_sea_surface_temperature"].isel(lead_day=1, latitude=0, longitude=0)),
        292.0,
    )


def test_uncoupled_rollout_keeps_fixed_sst_finite_in_forcing_state(tmp_path: Path):
    config = _write_minimal_s2s_store(tmp_path)._replace(normalize=False)
    target = build_target_grid(config.gaussian_n_max)
    lat = target.latitude_deg.astype(np.float32)
    lon = target.longitude_deg.astype(np.float32)

    initial_sst = np.full((lat.size, lon.size), 12.0, dtype=np.float32)
    initial_sst[0, :] = np.nan
    future_sst = np.stack(
        [np.full((lat.size, lon.size), 13.0 + day, dtype=np.float32) for day in range(config.n_steps)],
        axis=0,
    )
    future_sst[:, 0, :] = np.nan

    surface_forcing = xr.Dataset(
        data_vars={
            "surface_pressure": (
                ("lead_day", "latitude", "longitude"),
                np.full((config.n_steps, lat.size, lon.size), 101325.0, dtype=np.float32),
            ),
            "sw_down": (
                ("lead_day", "latitude", "longitude"),
                np.full((config.n_steps, lat.size, lon.size), 100.0, dtype=np.float32),
            ),
            "lw_down": (
                ("lead_day", "latitude", "longitude"),
                np.full((config.n_steps, lat.size, lon.size), 50.0, dtype=np.float32),
            ),
            "sea_surface_temperature": (("lead_day", "latitude", "longitude"), future_sst),
            "sea_ice_cover": (
                ("lead_day", "latitude", "longitude"),
                np.zeros((config.n_steps, lat.size, lon.size), dtype=np.float32),
            ),
            "initial_sea_surface_temperature": (("latitude", "longitude"), initial_sst),
        },
        coords={
            "lead_day": np.arange(config.n_steps, dtype=np.int32),
            "latitude": lat,
            "longitude": lon,
        },
    )

    ds = coupled_rollout_to_dataset(
        _PassThroughCoupledAtmosModel(),
        config,
        sample_index=0,
        coupled=False,
        surface_forcing=surface_forcing,
        stochastic_config=S2SStochasticConfig(
            ensemble_members=1,
            noise_channels=0,
            use_time_signal=False,
        ),
    )

    assert np.isfinite(ds["forcing"].values).all()
    assert np.isfinite(ds["prediction"].values).all()
    assert np.isnan(ds["sea_surface_temperature"].isel(latitude=0).values).all()
    assert np.isfinite(ds["sea_surface_temperature"].isel(latitude=1).values).all()


def test_train_sfno_s2s_saves_best_checkpoint_only(tmp_path: Path):
    target = build_target_grid(3)
    model = SFNO(
        SFNOConfig(
            in_channels=3,
            out_channels=2,
            embed_dim=4,
            n_blocks=1,
            mlp_expansion=2,
        ),
        grid=target.grid,
        key=jnp.array([0, 1], dtype=jnp.uint32),
    )

    batch_input = jnp.zeros((1, target.grid.n_lat, target.grid.n_lon, 3), dtype=jnp.float32)
    batch_target = jnp.zeros((1, 2, target.grid.n_lat, target.grid.n_lon, 2), dtype=jnp.float32)
    batch_forcing = jnp.zeros((1, 2, target.grid.n_lat, target.grid.n_lon, 1), dtype=jnp.float32)

    def iterator():
        while True:
            yield batch_input, batch_target, batch_forcing

    config = S2STrainingConfig(
        warmup_steps=1,
        total_steps=4,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=1,
        validation_every=1,
        validation_batches=1,
        save_best_checkpoint=True,
        save_final_checkpoint=False,
        save_step_checkpoints=False,
        stochastic_seed=0,
        stochastic_config=S2SStochasticConfig(
            ensemble_members=1,
            noise_channels=0,
            use_time_signal=False,
        ),
    )
    train_sfno_s2s(
        model,
        target.grid,
        iterator(),
        config,
        val_iterator=iterator(),
        log_every=1,
    )

    assert (tmp_path / "best.eqx").exists()
    assert not (tmp_path / "final.eqx").exists()
    assert not (tmp_path / "step_000001.eqx").exists()


def test_summarize_window_metrics_preserves_crps_columns():
    rows = [
        {"field": "t-850", "lead_day": 1, "rmse": 1.0, "mae": 0.5, "crps": 0.25, "n_members": 5},
        {"field": "t-850", "lead_day": 2, "rmse": 3.0, "mae": 1.5, "crps": 0.75, "n_members": 5},
    ]
    windows = summarize_window_metrics(rows, windows=parse_window_specs("wk1=1:2"))
    assert len(windows) == 1
    assert windows[0]["window"] == "wk1"
    assert np.isclose(windows[0]["rmse"], 2.0)
    assert np.isclose(windows[0]["mae"], 1.0)
    assert np.isclose(windows[0]["crps"], 0.5)
    assert np.isclose(windows[0]["n_members"], 5.0)


def _sfno_lowest_level_inputs(missing: str | None = None):
    labels = ["t-1000", "q-1000", "u-1000", "v-1000", "t-850"]
    if missing is not None:
        labels = [lab for lab in labels if lab != missing] + ["z-500"]
    rng = np.random.default_rng(3)
    pred = np.stack(
        [rng.uniform(280.0, 300.0, (3, 4)), rng.uniform(0.005, 0.02, (3, 4)),
         rng.normal(0.0, 5.0, (3, 4)), rng.normal(0.0, 5.0, (3, 4)),
         rng.uniform(270.0, 290.0, (3, 4))], axis=-1).astype(np.float32)
    aux = {"sp": np.full((3, 4), 101000.0, np.float32),
           "sw_down": np.full((3, 4), 200.0, np.float32),
           "lw_down": np.full((3, 4), 350.0, np.float32)}
    return pred, aux, labels


def test_sfno_slab_forcing_uses_shared_builder_virtual_density():
    pred, aux, labels = _sfno_lowest_level_inputs()
    forcing = s2s_coupling._build_atm_to_surface(
        pred, aux, channel_labels=labels, pressure_levels=(1000, 850),
        config=s2s_coupling.S2SSlabCouplingConfig())
    from legoesm import constants
    t, q = pred[..., 0].astype(float), pred[..., 1].astype(float)
    rho = 100000.0 / (constants.R_d * t * (1.0 + (1.0 / constants.epsilon - 1.0) * q))
    np.testing.assert_allclose(np.asarray(forcing.rho_lowest), rho, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(forcing.p_lowest), 100000.0)
    np.testing.assert_allclose(np.asarray(forcing.p_surface), 101000.0)
    np.testing.assert_allclose(np.asarray(forcing.q_lowest), pred[..., 1])


@pytest.mark.parametrize("missing", ["q-1000", "u-1000", "v-1000"])
def test_sfno_slab_forcing_missing_channel_raises(missing):
    pred, aux, labels = _sfno_lowest_level_inputs(missing)
    with pytest.raises(KeyError, match=missing):
        s2s_coupling._build_atm_to_surface(
            pred, aux, channel_labels=labels, pressure_levels=(1000, 850),
            config=s2s_coupling.S2SSlabCouplingConfig())


def test_sfno_slab_forcing_passthrough_and_nonfinite_raises():
    pred, aux, labels = _sfno_lowest_level_inputs()
    kw = dict(channel_labels=labels, pressure_levels=(1000, 850),
              config=s2s_coupling.S2SSlabCouplingConfig())
    forcing = s2s_coupling._build_atm_to_surface(pred, aux, **kw)
    np.testing.assert_allclose(np.asarray(forcing.T_lowest), pred[..., 0])
    np.testing.assert_allclose(np.asarray(forcing.u_lowest), pred[..., 2])
    np.testing.assert_allclose(np.asarray(forcing.v_lowest), pred[..., 3])
    np.testing.assert_allclose(np.asarray(forcing.sw_down), 200.0)
    np.testing.assert_allclose(np.asarray(forcing.lw_down), 350.0)
    np.testing.assert_allclose(np.asarray(forcing.cos_zenith), 1.0)
    bad_pred = pred.copy()
    bad_pred[1, 2, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite lowest-level"):
        s2s_coupling._build_atm_to_surface(bad_pred, aux, **kw)
    bad_aux = dict(aux, lw_down=aux["lw_down"].copy())
    bad_aux["lw_down"][0, 0] = np.nan
    with pytest.raises(ValueError, match="lw_down"):
        s2s_coupling._build_atm_to_surface(pred, bad_aux, **kw)
