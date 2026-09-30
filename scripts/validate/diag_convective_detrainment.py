"""How much condensate does each arm's convection detrain into the resolved cloud?

    diag_convective_detrainment.py --config <deck> --restart <ckpt.npz> [--label X]

The CAM6 AMIP arm carries 46% less cloud water than the paired production arm.
Vertical resolution, the diagnostic condensate floor, the cloud-fraction path and
the microphysical sink have each been measured and excluded, leaving convective
DETRAINMENT of condensate as the remaining candidate.  The carries cannot settle
it: the production arm carries exactly zero plume condensate and zero updraught
mass flux, so the two arms are not comparable on those fields.

This probe calls the deck's OWN convection scheme on the restored state through
the production factory and integrates the condensate source it hands to the
resolved cloud (the ``q_c`` tracer tendency the convection bridge publishes),
per column and area-weighted, in kg/m2/day.  It also reports the vapour sink and
the surface convective rain so the column water budget of the call is visible.

Caveat, stated because it bounds the claim: the convection is called with a
FRESH physics-state carry, so any prognostic or stochastic closure state the run
had accumulated is reset.  Schemes whose trigger depends on that carry will be
reported off their run trajectory.  NUMBERS ONLY -- no verdict.
"""
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_DAY = 86400.0


def _load_run_amip():
    path = _ROOT / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_detrprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_detrprobe"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def column_integral(rate, dp, area_w, g):
    """Area-weighted column integral of a mixing-ratio rate [kg/m^2/s]."""
    return float(np.sum(area_w * np.sum(rate * dp, axis=1) / g))


def main():
    a = build_arg_parser().parse_args()
    import jax.numpy as jnp
    from legoesm import constants

    ra = _load_run_amip()
    argv = ["--config", a.config] + list(a.extra)
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config(a.config, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    config = ra.build_config_from_args(args)

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    step0, day0 = driver.load_checkpoint(a.restart)
    print(f"[{a.label}] restored step={step0} day={day0}", flush=True)

    state = driver.state
    model = driver.model
    mesh = model.mesh
    sc = model.sigma_coord
    ncol, nlev = np.asarray(state.T.data).shape
    area = np.asarray(mesh.areaCell).reshape(ncol)
    A = area / area.sum()
    p_half = np.asarray(sc.pressure_at_half(jnp.asarray(state.p_s.data)))
    dp = p_half[:, 1:] - p_half[:, :-1]

    from legoesm.driver.physics_pipeline import convection_config_for
    from legoesm.atmosphere.physics.convection.integration import (
        make_convection_physics,
    )
    conv_cfg = convection_config_for(config)
    dt_phys = float(config.dycore.dt) * int(getattr(config, "physics_update_steps", 1) or 1)
    print(f"[{a.label}] convection scheme = {conv_cfg.scheme}   dt = {dt_phys:.1f} s")

    fn = make_convection_physics(conv_cfg, model_type="mpas", dt=dt_phys)
    out = fn(state, mesh, sc, None, None)
    tends = out[0] if isinstance(out, tuple) else out
    tr = getattr(tends, "tracer_tendencies", None)
    if tr is None:
        tr = getattr(tends, "tracers", None)
    if tr is None and isinstance(tends, dict):
        tr = tends.get("tracer_tendencies") or tends.get("tracers")
    if tr is None:
        raise SystemExit(f"convection returned no tracer tendencies: {type(tends)}")

    def grab(name):
        v = tr.get(name)
        if v is None:
            return None
        d = v.data if hasattr(v, "data") else v
        return np.asarray(d).reshape(ncol, nlev)

    dqc = grab("q_c")
    dqv = grab("q_v")
    dqr = grab("q_r")
    g = constants.g
    if dqc is not None:
        det = column_integral(np.maximum(dqc, 0.0), dp, A, g) * _DAY
        net = column_integral(dqc, dp, A, g) * _DAY
        print(f"[{a.label}] condensate detrained into the resolved cloud: "
              f"{det:.6e} kg/m2/day (positive part), net {net:.6e}")
    else:
        print(f"[{a.label}] scheme publishes NO condensate tendency")
    if dqv is not None:
        print(f"[{a.label}] vapour tendency from convection: "
              f"{column_integral(dqv, dp, A, g) * _DAY:.6e} kg/m2/day")
    if dqr is not None:
        print(f"[{a.label}] rain tendency published as a tracer: "
              f"{column_integral(dqr, dp, A, g) * _DAY:.6e} kg/m2/day")
    else:
        print(f"[{a.label}] no q_r tracer tendency (rain takes the surface route)")

    if dqc is not None:
        edges = np.array([0., 200., 400., 500., 600., 700., 800., 900., 1100.]) * 100.0
        pf = np.asarray(sc.pressure_at_full(jnp.asarray(state.p_s.data)))
        print(f"[{a.label}] pressure band | detrained condensate kg/m2/day")
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (pf >= lo) & (pf < hi)
            v = float(np.sum(A[:, None] * np.where(m, np.maximum(dqc, 0.0) * dp, 0.0)) / g) * _DAY
            print(f"[{a.label}]   {lo/100:6.0f}-{hi/100:6.0f} hPa | {v:.6e}")


if __name__ == "__main__":
    main()
