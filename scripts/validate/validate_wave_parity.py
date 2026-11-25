#!/usr/bin/env python
"""Historical NeuralSWM parity probe for the pre-WADEN operator.

The corrected WADEN dynamics intentionally change this operator. Old reported
parity numbers are withdrawn as current claims. This retained comparison keeps
its original strict threshold; new physics validation uses validate_wave_ww3.py.

The two models live in different environments (NeuralSWM needs Flax NNX
and Orbax; legoESM must not depend on either), so the probe runs in two
steps:

1. ``--emit`` runs the *source* NeuralSWM model on the ERA5 demo fields
   and writes the inputs and the one-step output to an ``.npz``.  Run it
   from an environment with ``flax``, ``gin`` and ``orbax-checkpoint``
   and with the NeuralSWM repository on ``PYTHONPATH``::

       PYTHONPATH=~/NeuralSWM python scripts/validate/validate_wave_parity.py \
           --emit --neuralswm-root ~/NeuralSWM \
           --checkpoint-dir ~/NeuralSWM/examples/well_trained_checkpoints/case_c9_dist_260515 \
           --out /tmp/wave_parity.npz

2. ``--compare`` runs the ported model on the same arrays and reports the
   max absolute and relative differences::

       python scripts/validate/validate_wave_parity.py --compare \
           --weights ~/NeuralSWM/examples/well_trained_checkpoints/case_c9_dist_260515.npz \
           --reference /tmp/wave_parity.npz

The probe is committed so the parity claim can be re-run against a changed
model, not just asserted in a PR body.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _demo_inputs(neuralswm_root: Path) -> dict[str, np.ndarray]:
    import xarray as xr

    wave = xr.open_dataset(
        neuralswm_root / "examples/data/environ_forcings/ERA5_2022_wave_fujian_demo.nc")
    wind = xr.open_dataset(
        neuralswm_root / "examples/data/environ_forcings/ERA5_2022_wind_fujian_demo.nc")
    spectrum = xr.open_dataset(
        neuralswm_root / "examples/data/wave_spectrum_nc_2022_demo/20220101/"
        "wave_spectrum_2022_01_01_00.nc")

    lon = np.asarray(wave.longitude.values, dtype=np.float32)
    lat = np.asarray(wave.latitude.values, dtype=np.float32)
    # The NetCDF fields are (lat, lon); the model's spatial order is (lon, lat).
    depth = np.asarray(wave.wmb.isel(time=0).values, dtype=np.float32).T
    wind_u = np.asarray(wind.u10.isel(time=0).values, dtype=np.float32)[::2, ::2].T
    wind_v = np.asarray(wind.v10.isel(time=0).values, dtype=np.float32)[::2, ::2].T
    # Each spectrum file holds one time: ``time`` is a scalar coordinate,
    # so there is no time dimension to select.
    raw = np.asarray(spectrum.d2fd.values, dtype=np.float32)
    # (direction, frequency, lat, lon) -> (lon, lat, direction, frequency)
    spectra = np.transpose(10.0 ** raw, (3, 2, 0, 1))[None, ...]
    mask = ~(np.isnan(depth) | np.isnan(wind_u) | np.isnan(wind_v))
    # Match the source loader: zero bathy and wind together on missing-bathy
    # cells, so the network sees the same forcing.
    invalid = ~mask
    return {
        "lon": lon,
        "lat": lat,
        "depth": np.where(invalid, 0.0, np.nan_to_num(depth)),
        "wind_u": np.where(invalid, 0.0, np.nan_to_num(wind_u)),
        "wind_v": np.where(invalid, 0.0, np.nan_to_num(wind_v)),
        "spectra": np.nan_to_num(spectra),
        "mask": mask.astype(np.float32),
        "freq": (np.full(30, 0.03453, dtype=np.float32) * (1.1 ** np.arange(30))).astype(np.float32),
        "theta_deg": np.arange(7.5, 352.5 + 15.0, 15.0, dtype=np.float32),
    }


def emit(args) -> None:
    root = Path(args.neuralswm_root).expanduser().resolve()
    sys.path.insert(0, str(root))

    import jax.numpy as jnp
    from inference.jax_core import _restore_inference_checkpoint
    from neuralwave_jax.architecture import create_jax_model_from_gin

    data = _demo_inputs(root)
    theta_rad = ((180.0 - data["theta_deg"]) % 360.0) * np.pi / 180.0

    model = create_jax_model_from_gin(root / "configs/model_mlpParam_c5_sdnEmb2_NNTend.gin")
    model.set_wave_environment(
        lon=data["lon"], lat=data["lat"], theta=theta_rad, freq=data["freq"])
    model = _restore_inference_checkpoint(model, Path(args.checkpoint_dir).expanduser())

    spectra = jnp.asarray(data["spectra"])
    environments = {
        "depth": jnp.asarray(data["depth"])[None, ...],
        "bathymetry": jnp.asarray(data["depth"])[None, ...],
        "wind_u": jnp.asarray(data["wind_u"])[None, ...],
        "wind_v": jnp.asarray(data["wind_v"])[None, ...],
    }
    mask = jnp.asarray(data["mask"])[None, ...]

    # Run under nnx.jit, NOT eagerly.  The source's ``feature_generators``
    # dict is rebuilt in sorted key order inside a jit, and BOTH training and
    # the validated inference run under that jit; an eager call keeps
    # insertion order and is a different (never-trained) function.  The
    # intermediates are returned from the same jitted graph so they share the
    # trained feature order.
    import flax.nnx as nnx

    def _forward(model, spectra, environments, mask):
        physics_info = model.get_physics_info(environments)
        physics_metadata = {
            **model.metadata,
            "wavenum": physics_info["wavenum"],
            "Cg": physics_info["group_velocity"],
        }
        enhanced = model.encoder(spectra=spectra, environdata=environments, mask=mask)
        stot = model.parameterization(
            spectra=enhanced, environdata=environments,
            metadata=physics_metadata, mask=mask)
        output = model(
            spectra=spectra, environments=environments, mask=mask,
            boundary_condition=jnp.zeros_like(spectra))
        return physics_info["wavenum"], physics_info["group_velocity"], stot, output

    wavenum, cg, stot, output = nnx.jit(_forward)(model, spectra, environments, mask)

    np.savez_compressed(
        args.out, output=np.asarray(output),
        wavenum=np.asarray(wavenum),
        cg=np.asarray(cg),
        stot=np.asarray(stot),
        **data)
    print(f"wrote {args.out}: output shape {np.asarray(output).shape}, "
          f"sum {float(np.asarray(output).sum()):.6e}")


def compare(args) -> None:
    import jax
    import jax.numpy as jnp

    from legoesm.wave import (
        WaveConfig,
        WaveGSEConfig,
        WaveGridConfig,
        WaveModel,
        init_wave_state,
        load_wave_weights,
        prepare_env,
        wave_parameters,
    )

    print("Historical NeuralSWM comparison: corrected WADEN dynamics require separate validation.")
    ref = np.load(args.reference)
    theta_rad = tuple(float(x) for x in ((180.0 - ref["theta_deg"]) % 360.0) * np.pi / 180.0)
    config = WaveConfig(grid=WaveGridConfig(
        lon_deg=tuple(float(x) for x in ref["lon"]),
        lat_deg=tuple(float(x) for x in ref["lat"]),
        freq_hz=tuple(float(x) for x in ref["freq"]),
        theta_rad=theta_rad,
    ), gse=WaveGSEConfig(enabled=False))  # the NeuralSWM reference has no GSE
    config.validate()

    model = WaveModel(config, key=jax.random.key(0))
    model = load_wave_weights(model, args.weights)

    depth = jnp.asarray(ref["depth"])[None, ...]
    state = init_wave_state(depth, n_dir=len(theta_rad), n_freq=len(ref["freq"]))
    env = prepare_env(config, state.depth)
    output = model.step(
        jnp.asarray(ref["spectra"]),
        env,
        state.depth,
        state.mask,
        jnp.asarray(ref["wind_u"])[None, ...],
        jnp.asarray(ref["wind_v"])[None, ...],
    )

    expected = jnp.asarray(ref["output"])
    diff = np.asarray(jnp.abs(output - expected))
    scale = float(np.asarray(jnp.abs(expected)).max())
    rel = float(diff.max()) / max(scale, 1e-30)
    print(f"max |diff| = {diff.max():.3e}   max |ref| = {scale:.3e}   relative = {rel:.3e}")

    ref_params = np.asarray(jnp.asarray(
        wave_parameters(expected, env)["swh"]))
    out_params = np.asarray(jnp.asarray(
        wave_parameters(output, env)["swh"]))
    swh_diff = np.abs(out_params - ref_params)
    print(f"swh max |diff| = {swh_diff.max():.3e}  (max swh {ref_params.max():.3f} m)")

    if rel > 1e-5:
        raise SystemExit(f"parity FAILED: relative difference {rel:.3e} > 1e-5")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--emit", action="store_true")
    mode.add_argument("--compare", action="store_true")
    parser.add_argument("--neuralswm-root", default="~/NeuralSWM")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.emit:
        if args.checkpoint_dir is None or args.out is None:
            parser.error("--emit needs --checkpoint-dir and --out")
        emit(args)
    else:
        if args.weights is None or args.reference is None:
            parser.error("--compare needs --weights and --reference")
        compare(args)


if __name__ == "__main__":
    main()
