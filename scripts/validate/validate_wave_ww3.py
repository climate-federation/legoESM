"""Run archived WW3 cases through the legoESM wave public dynamics path.

Input construction and scores reuse WADEN 46aadc7's MIT-licensed example.
No neural checkpoint, training, downloads or future interior forcing is used.
"""

import argparse
import dataclasses
from functools import partial
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.wave.config import (
    WaveConfig, WaveGSEConfig, WaveGridConfig, WaveNetworkConfig,
    WavePhysicsConfig, WaveTimeConfig,
)
from legoesm.wave.grid import spectral_integral
from legoesm.wave.model import WaveModel, prepare_env
from legoesm.wave.state import WaveState
from scripts.validate.wave_fidelity.case_io import load_case, moments, read_config
from scripts.validate.wave_fidelity.compare import errors, circular_errors

WADEN_REVISION = "46aadc7834ad850442a535a08f94ca173e4b22e8"
# Inherited NeuralSWM scientific regression bars
# (test/direct/test_real.py; the pinned WADEN checkout has no test tree).
SWH_RMSE_BARS = {"tp1.1": 1e-4, "tp1.2": .01, "tp1.4": .008, "tp2.1": 2e-5}
TP2_1_MAX_ABS_M = .002


def _plain(value):
    """Recursively turn NamedTuples into NAMED dicts for the manifest.

    ``json.dumps(NamedTuple)`` serialises nested configs positionally, so a
    field reorder would silently change the meaning of an old manifest.
    """
    if hasattr(value, "_asdict"):
        return {key: _plain(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def free_mask(case):
    """Scoring domain: wet cells that are not prescribed boundaries."""
    return np.asarray(case["wet"], dtype=bool) & ~np.asarray(
        case["boundary_mask"], dtype=bool)


def case_passed(case_name, bulk):
    """Apply the inherited case gate (plus tp2.1's max-absolute bound)."""
    passed = bulk["rmse"] < SWH_RMSE_BARS[case_name]
    if case_name == "tp2.1":
        passed = passed and bulk["max_absolute"] < TP2_1_MAX_ABS_M
    return bool(passed)


def compare_waden(case, config, state, prediction, final_phase, root, output):
    """Replay WADEN's existing public WW3 rollout on identical case arrays."""
    root = root.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if revision != WADEN_REVISION or dirty:
        raise ValueError("Direct comparison requires the pinned clean WADEN source")
    sys.path.insert(0, str(root/"src"))
    try:
        waden = importlib.import_module("waden")
    finally:
        sys.path.pop(0)
    if Path(waden.__file__).resolve() != root/"src/waden/__init__.py":
        raise ValueError("Imported WADEN differs from the declared source")
    grid, spectral = waden.prepare_environment(
        case["x"], case["y"], case["theta"], case["freq"],
        coordinate_type=config["coordinate_type"], dtype=state.spectra.dtype)
    oracle_state = waden.State(state.spectra)
    fixed = jnp.asarray(case["boundary_mask"])[None]
    environment = waden.Environment(
        depth=state.depth[..., None, None], mask=state.mask,
        boundary=jnp.where(fixed[..., None, None], state.spectra, 0),
        wind_u=jnp.zeros_like(state.depth), wind_v=jnp.zeros_like(state.depth),
        boundary_mask=fixed)
    steps = len(case["times"])-1
    environments = jax.tree.map(lambda value: jnp.broadcast_to(value, (steps,)+value.shape), environment)
    environments = dataclasses.replace(environments, depth=environment.depth)
    step_config = waden.StepConfig(**config["step_config"])

    @jax.jit
    def run(start, forcing):
        result = waden.rollout(start, forcing, grid, spectral, step_config,
            fixed_depth=True, audit_clipping=True,
            boundary_update_func=waden.identity_boundary_update,
            x_tendency_fn=partial(waden.x_tendency, stencil_mode=config["x_stencil"]),
            y_tendency_fn=partial(waden.y_tendency, stencil_mode="replicate"))
        return result.spectra, result.final_state.spatial_phase

    reference, phase = run(oracle_state, environments)
    reference = np.concatenate((np.asarray(state.spectra), np.asarray(reference)[:,0]))
    if not np.isfinite(reference).all() or np.any(reference < 0):
        raise ValueError("Invalid direct WADEN trajectory")
    if int(phase) != int(final_phase):
        raise ValueError("Spatial phase differs between legoESM and WADEN")
    wet = free_mask(case)
    reference_bulk = moments(reference, case["theta"], case["freq"])
    np.savez_compressed(output/"waden_prediction.npz", spectra=reference,
                        time=case["times"].astype("datetime64[s]").astype(np.int64))
    return dict(revision=revision, dirty=dirty, module=waden.__file__,
        phase=int(phase), spectrum=errors(prediction, reference, wet[None,...,None,None]),
        swh=errors(moments(prediction,case["theta"],case["freq"])["swh"],reference_bulk["swh"],wet[None]),
        verdict="descriptive full-trajectory comparison; no new acceptance bar")


def configure(case, config, dtype):
    """Map the oracle's explicit axis, grid, time and no-source choices."""
    sx = 1 if case["x"][1] > case["x"][0] else -1
    sy = 1 if case["y"][1] > case["y"][0] else -1
    # Static grid angles: transform in float64 so float32 rounding does not
    # make a uniform direction grid fail the config's uniformity check.  The
    # model casts the grid to the run dtype when it builds the environment.
    data_theta = np.asarray(case["theta"], dtype=np.float64)
    angle = np.deg2rad(data_theta)
    if sx == 1 and sy == -1:
        theta = ((180.-data_theta) % 360.)*np.pi/180.
    else:
        theta = np.arctan2(sx*np.sin(angle), sy*np.cos(angle)) % (2*np.pi)
    grid = WaveGridConfig(tuple(map(float,case["x"])), tuple(map(float,case["y"])), tuple(map(float,case["freq"])), tuple(map(float,theta)),
                          coordinate_type=config["coordinate_type"], boundary_mode="physical",
                          x_stencil=config["x_stencil"], y_stencil="replicate")
    timing = config["step_config"]
    wave_config = WaveConfig(grid, time=WaveTimeConfig(**{
        name+"_s": timing[name] for name in ("dt_global", "dt_x", "dt_y", "dt_th", "dt_k", "dt_stot", "dt_gse")}),
        physics=WavePhysicsConfig(yfirst=timing["yfirst"]),
        # These are pure propagation cases: GSE is off, so the config must say
        # so explicitly (the released default is enabled at 3600 s).
        gse=WaveGSEConfig(enabled=False),
        network=WaveNetworkConfig(parameterization="none", use_transport_networks=False))
    depth = jnp.asarray(case["depth"][None], dtype=dtype)
    state = WaveState(jnp.asarray(case["initial"][None], dtype=dtype), depth, jnp.asarray(case["wet"][None]))
    boundary_mask = jnp.asarray(case["boundary_mask"][None])
    boundary = jnp.where(boundary_mask[..., None, None], state.spectra, 0)
    return wave_config, state, boundary, boundary_mask


def run_case(config_path, data_root, output, dtype_name, waden_root=None):
    config = read_config(config_path)
    case = load_case(config, data_root)
    if dtype_name == "float64":
        jax.config.update("jax_enable_x64", True)
    dtype = getattr(jnp, dtype_name)
    wave_config, state, boundary, boundary_mask = configure(case, config, dtype)
    wave_config.validate()
    intervals = np.diff(case["times"]).astype("timedelta64[s]").astype(int)
    if not np.all(intervals == wave_config.time.dt_global_s):
        raise ValueError("Wave step must match every actual WW3 output interval")
    model = WaveModel(wave_config, key=jax.random.key(0))
    env = prepare_env(wave_config, state.depth)
    wind = jnp.zeros(state.mask.shape, dtype=dtype)

    @jax.jit
    def advance(start):
        def body(current, _):
            updated, audit = model.advance(current, env, wind, wind, boundary, boundary_mask=boundary_mask, audit_clipping=True)
            return updated, (updated.spectra, audit["clipping_mass"])
        return jax.lax.scan(body, start, None, length=len(intervals))

    final, (trajectory, clipping) = advance(state)
    spectra = np.concatenate((np.asarray(state.spectra), np.asarray(trajectory)[:, 0]))
    if not np.isfinite(spectra).all() or np.any(spectra < 0):
        raise ValueError("Nonfinite or negative predicted spectrum")
    diagnostics = moments(spectra, case["theta"], case["freq"])
    integral = np.asarray(spectral_integral(jnp.asarray(spectra), env,
                         jnp.broadcast_to(state.mask, spectra.shape[:3])))
    free = free_mask(case)
    bulk = errors(diagnostics["swh"], case["fields"]["swh"], free[None])
    common, pi, ri = np.intersect1d(case["times"], case["spectrum_times"], return_indices=True)
    station = spectra[pi][:, case["station_ix"], case["station_iy"]]
    station_free = free[case["station_ix"], case["station_iy"]]
    spectral_score = errors(station, case["station_spectra"][ri], station_free[None, :, None, None])
    valid_direction = free[None] & (case["fields"]["swh"] >= config["direction_min_swh_m"])
    direction_score = circular_errors(diagnostics["mwd"], (case["fields"]["mwd_from"]+180)%360, valid_direction)
    period_score = errors(diagnostics["mwp"], case["fields"]["mwp"], valid_direction) if "mwp" in case["fields"] else None
    passed = case_passed(config["case"], bulk)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    inputs = sorted((Path(data_root)/config["data_directory"]/"input").glob("*"))
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs if p.is_file()}
    manifest = dict(case=config["case"], config=config, resolved_config=_plain(wave_config),
                    legoesm_revision=revision, tracked_or_untracked_changes=dirty,
                    waden_revision=WADEN_REVISION, reference=case["provenance"], input_sha256=hashes,
                    dtype=dtype_name, python=platform.python_version(), jax=jax.__version__,
                    jax_enable_x64=bool(jax.config.read("jax_enable_x64")),
                    jax_platforms=os.environ.get("JAX_PLATFORMS"),
                    jax_default_backend=jax.default_backend(),
                    devices=[str(d) for d in jax.devices()], command=sys.argv,
                    wave_module=str(Path(sys.modules["legoesm.wave"].__file__).resolve()),
                    actual_dtypes=dict(spectra=str(state.spectra.dtype), frequency=str(env.freq.dtype), geometry=str(env.lon.dtype)),
                    source="disabled", learned_transport="disabled", gse="disabled",
                    frames=len(spectra), station_common_frames=len(common), spatial_phase=int(final.spatial_phase),
                    swh=bulk, station_spectrum=spectral_score, direction=direction_score, period=period_score,
                    cumulative_clipping_mass=float(np.sum(np.asarray(clipping), dtype=np.float64)),
                    final_to_initial_integral=float(integral[-1]/integral[0]),
                    rmse_bar=SWH_RMSE_BARS[config["case"]], passed=bool(passed))
    output.mkdir(parents=True, exist_ok=True)
    if waden_root is not None:
        manifest["direct_waden"] = compare_waden(case, config, state, spectra, final.spatial_phase, waden_root, output)
    np.savez_compressed(output/"prediction.npz", spectra=spectra, integral=integral,
                        time=case["times"].astype("datetime64[s]").astype(np.int64),
                        final_spatial_phase=np.asarray(final.spatial_phase), clipping_mass=np.asarray(clipping)[:,0],
                        x=case["x"], y=case["y"], theta_deg=case["theta"], freq=case["freq"],
                        wet=case["wet"], depth=case["depth"], boundary_mask=case["boundary_mask"], **diagnostics)
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, default=lambda value: value.item())+"\n")
    print(json.dumps(dict(case=config["case"], passed=bool(passed), swh=bulk, station_spectrum=spectral_score)))
    if not passed:
        raise SystemExit("WW3 inherited case-specific SWH bar failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--waden-root", type=Path, help="Pinned clean WADEN checkout for direct public-rollout comparison")
    args = parser.parse_args()
    run_case(args.config, args.data_root, args.output, args.dtype, args.waden_root)


if __name__ == "__main__":
    main()
