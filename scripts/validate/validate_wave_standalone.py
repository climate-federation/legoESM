#!/usr/bin/env python
"""Standalone skill check of the ported wave component on the ERA5 demo.

Drives the model with the ERA5 10 m wind of the Fujian demo domain,
initialises from the analysed spectrum, and scores significant wave
height against ERA5 SWH at each lead time.  Persistence (the initial SWH
held constant) is the baseline: a one-step-correct port that does not
beat persistence over 24 h would mean the coupling is not adding value.

This is the component's own validation; it does not exercise the coupler.
Run it on CPU::

    JAX_PLATFORMS=cpu python scripts/validate/validate_wave_standalone.py \
        --neuralswm-root ~/NeuralSWM \
        --weights ~/NeuralSWM/examples/well_trained_checkpoints/case_c9_dist_260515.npz \
        --hours 24

Results are printed and written to ``results/wave_standalone/`` (ignored
by git).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_demo(root: Path) -> dict[str, np.ndarray]:
    import xarray as xr

    wave = xr.open_dataset(root / "examples/data/environ_forcings/ERA5_2022_wave_fujian_demo.nc")
    wind = xr.open_dataset(root / "examples/data/environ_forcings/ERA5_2022_wind_fujian_demo.nc")

    files = sorted((root / "examples/data/wave_spectrum_nc_2022_demo").glob("*/*.nc"))
    spectra = []
    for path in files:
        with xr.open_dataset(path) as ds:
            raw = np.asarray(ds.d2fd.values, dtype=np.float32)
        spectra.append(10.0 ** raw)  # (direction, frequency, lat, lon)
    spectra = np.stack(spectra, axis=0)          # (time, D, F, lat, lon)
    spectra = np.transpose(spectra, (0, 4, 3, 1, 2))  # (time, lon, lat, D, F)

    lat = np.asarray(wave.latitude.values, dtype=np.float32)
    lon = np.asarray(wave.longitude.values, dtype=np.float32)
    depth = np.asarray(wave.wmb.isel(time=0).values, dtype=np.float32).T
    mask = np.isfinite(depth)
    depth = np.nan_to_num(depth)  # masked columns carry zero depth, never NaN
    u10 = np.asarray(wind.u10.values, dtype=np.float32)[:, ::2, ::2].transpose(0, 2, 1)
    v10 = np.asarray(wind.v10.values, dtype=np.float32)[:, ::2, ::2].transpose(0, 2, 1)
    swh = np.asarray(wave.swh.values, dtype=np.float32).transpose(0, 2, 1)
    mask = mask & np.isfinite(u10[0]) & np.isfinite(v10[0])
    return {
        "lat": lat, "lon": lon, "depth": depth,
        "u10": np.nan_to_num(u10), "v10": np.nan_to_num(v10),
        "spectra": np.nan_to_num(spectra), "swh": swh,
        "mask": mask,
    }


def _score(predicted, observed, initial, valid):
    """RMSE, bias, persistence RMSE and sample count on the valid mask."""
    err = predicted[valid] - observed[valid]
    persist_err = initial[valid] - observed[valid]
    return (float(np.sqrt(np.mean(err**2))), float(np.mean(err)),
            float(np.sqrt(np.mean(persist_err**2))), int(valid.sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuralswm-root", default="~/NeuralSWM")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--out", default="results/wave_standalone")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from legoesm.wave import (
        WaveConfig,
        WaveGridConfig,
        WaveModel,
        init_wave_state,
        load_wave_weights,
        prepare_env,
        wave_parameters,
    )

    data = _load_demo(Path(args.neuralswm_root).expanduser())
    n_steps = min(args.hours, data["spectra"].shape[0] - 1, data["u10"].shape[0] - 1)
    if n_steps <= 0:
        raise SystemExit("not enough demo time steps for the requested rollout")

    theta_deg = np.arange(7.5, 352.5 + 15.0, 15.0, dtype=np.float32)
    freq = (np.full(30, 0.03453, dtype=np.float32) * (1.1 ** np.arange(30))).astype(np.float32)
    config = WaveConfig(grid=WaveGridConfig(
        lon_deg=tuple(float(x) for x in data["lon"]),
        lat_deg=tuple(float(x) for x in data["lat"]),
        freq_hz=tuple(float(x) for x in freq),
        theta_rad=tuple(float(x) for x in ((180.0 - theta_deg) % 360.0) * np.pi / 180.0),
        boundary_mode="absorbing",
    ))
    config.validate()

    model = WaveModel(config, key=jax.random.key(0))
    model = load_wave_weights(model, args.weights)

    depth = jnp.asarray(data["depth"])[None, ...]
    mask = jnp.asarray(data["mask"])[None, ...]
    state = init_wave_state(depth, n_dir=len(theta_deg), n_freq=len(freq))
    state = state._replace(mask=mask)
    env = prepare_env(config, state.depth)

    spectra = jnp.asarray(data["spectra"][0])[None, ...]
    state = state._replace(spectra=spectra)
    initial_swh = np.asarray(wave_parameters(spectra, env)["swh"])[0]
    mask_np = data["mask"]

    # The absorbing edge zeroes the outer ring, so report the interior
    # (open-ocean) skill separately from the full-mask skill.  The source's
    # validated pipeline feeds the analysis edge ring back in, which is data
    # assimilation, not forecast skill.
    interior = np.zeros_like(mask_np)
    interior[1:-1, 1:-1] = True
    interior = interior & mask_np

    rows = []
    for step in range(n_steps):
        state = model.advance(
            state, env,
            jnp.asarray(data["u10"][step])[None, ...],
            jnp.asarray(data["v10"][step])[None, ...],
        )
        spectra = state.spectra
        predicted = np.asarray(wave_parameters(spectra, env)["swh"])[0]
        observed = data["swh"][step + 1]
        valid = interior & np.isfinite(observed)
        rmse, bias, persist_rmse, count = _score(
            predicted, observed, initial_swh, valid)
        rows.append((step + 1, rmse, bias, persist_rmse, count))

    print(f"{'lead_h':>6} {'rmse_m':>9} {'bias_m':>9} {'persist_rmse_m':>15} {'n':>7}")
    for lead, rmse, bias, persist, n in rows:
        print(f"{lead:6d} {rmse:9.4f} {bias:9.4f} {persist:15.4f} {n:7d}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "standalone_skill.npz",
             lead_hours=np.asarray([r[0] for r in rows]),
             rmse=np.asarray([r[1] for r in rows]),
             bias=np.asarray([r[2] for r in rows]),
             persistence_rmse=np.asarray([r[3] for r in rows]))
    final = rows[-1]
    if final[1] >= final[3]:
        raise SystemExit(
            f"standalone skill FAILED: at {final[0]} h the model RMSE "
            f"{final[1]:.4f} m does not beat persistence {final[3]:.4f} m")


if __name__ == "__main__":
    main()
