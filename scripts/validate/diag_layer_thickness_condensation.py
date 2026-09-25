"""Does a THICK grid-mean layer condense less than a thin one, same profile?

    diag_layer_thickness_condensation.py --config <deck> --restart <ckpt.npz> \
        [--coarse-config <deck>] [--iters 3]

The CAM6 AMIP arm carries 46% less cloud water than the paired production arm,
and the deficit is confined to 400-900 hPa, where the CAM 32-level table has one
or two layers of 66-85 hPa against the production grid's three of 33 hPa.  That
co-location is measured; the CAUSAL step is not.  This probe tests it with ONE
variable.

Method.  Restore the FINE-grid state (the production 36-level arm).  Compute the
condensate a grid-mean moist adjustment produces on its own layers.  Then
coarse-grain the SAME columns onto the COARSE grid's layer edges by mass-weighted
averaging of T and q_v (conserving dry mass and total water per column, each
column on its own surface pressure), and recompute the SAME adjustment on the
SAME columns with the SAME saturation curve.  Everything but the layer thickness
is held fixed.

The adjustment is the grid-mean fixed point of q_c = max(q_v - q_sat(T), 0) with
the latent heating fed back, using ``legoesm.thermo.saturation_mixing_ratio`` --
the model's own curve, never a re-derived one.

This measures a LOWER BOUND on the thickness effect: structure finer than the
fine grid's own 33 hPa layers is already absent from the input, so any real
sub-33 hPa structure would make the coarse-graining loss larger, not smaller.
NUMBERS ONLY -- no verdict.
"""
from __future__ import annotations
import argparse, importlib.util, os, sys
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]


def _load_run_amip():
    path = _ROOT / "scripts" / "run" / "run_amip.py"
    spec = importlib.util.spec_from_file_location("run_amip_thickprobe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_thickprobe"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="deck of the FINE state to restore")
    ap.add_argument("--restart", required=True)
    ap.add_argument("--coarse-config", required=True,
                    help="deck whose vertical table supplies the COARSE layer edges")
    ap.add_argument("--iters", type=int, default=3,
                    help="fixed-point iterations of the moist adjustment")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def moist_adjust(T, q_v, p_full, iters):
    """Grid-mean moist adjustment: condensate with the latent heating fed back.

    Uses the model's saturation curve.  Returns (q_c, T_adj).
    """
    from legoesm.thermo import saturation_mixing_ratio
    from legoesm import constants
    T_a = np.array(T, dtype=np.float64)
    q_a = np.array(q_v, dtype=np.float64)
    q_c = np.zeros_like(q_a)
    for _ in range(max(1, iters)):
        q_sat = np.asarray(saturation_mixing_ratio(T_a, p_full))
        d = np.maximum(q_a - q_sat, 0.0)
        q_c = q_c + d
        q_a = q_a - d
        T_a = T_a + constants.L_v * d / constants.c_pd
    return q_c, T_a


def coarse_grain(x, p_half_fine, p_half_coarse):
    """Mass-weighted layer means of ``x`` from the fine layers onto the coarse ones.

    Conserves the column integral of ``x * dp`` exactly wherever the two column
    tops and bottoms agree, by accumulating the overlap of every fine layer with
    every coarse layer.  ``x`` is (ncol, nfine); the half-level arrays are
    (ncol, n+1) and increase downward.
    """
    ncol, nf = x.shape
    nc = p_half_coarse.shape[1] - 1
    out = np.zeros((ncol, nc), dtype=np.float64)
    wsum = np.zeros((ncol, nc), dtype=np.float64)
    for k in range(nc):
        lo = p_half_coarse[:, k][:, None]
        hi = p_half_coarse[:, k + 1][:, None]
        top = np.maximum(p_half_fine[:, :-1], lo)
        bot = np.minimum(p_half_fine[:, 1:], hi)
        w = np.maximum(bot - top, 0.0)
        out[:, k] = np.sum(w * x, axis=1)
        wsum[:, k] = np.sum(w, axis=1)
    return out / np.maximum(wsum, 1e-30)


def _build(cfg_path, extra):
    ra = _load_run_amip()
    argv = ["--config", cfg_path] + list(extra)
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config(cfg_path, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    return ra.build_config_from_args(args)


def main():
    a = build_arg_parser().parse_args()
    import jax.numpy as jnp
    from legoesm import constants

    config = _build(a.config, a.extra)
    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    step0, day0 = driver.load_checkpoint(a.restart)
    print(f"restored step={step0} day={day0}", flush=True)

    state = driver.state
    mesh = driver.model.mesh
    sc_fine = driver.model.sigma_coord
    T = np.asarray(state.T.data, dtype=np.float64)
    p_s = np.asarray(state.p_s.data, dtype=np.float64)
    ncol, nf = T.shape
    area = np.asarray(mesh.areaCell).reshape(ncol)
    A = area / area.sum()
    qv = np.asarray(state.tracers["q_v"].data, dtype=np.float64).reshape(ncol, nf)

    pf_fine = np.asarray(sc_fine.pressure_at_full(jnp.asarray(p_s)), dtype=np.float64)
    ph_fine = np.asarray(sc_fine.pressure_at_half(jnp.asarray(p_s)), dtype=np.float64)

    cfg_c = _build(a.coarse_config, a.extra)
    drv_c = ModelDriver(cfg_c)
    drv_c.setup()
    sc_coarse = drv_c.model.sigma_coord
    pf_c = np.asarray(sc_coarse.pressure_at_full(jnp.asarray(p_s)), dtype=np.float64)
    ph_c = np.asarray(sc_coarse.pressure_at_half(jnp.asarray(p_s)), dtype=np.float64)
    nc = pf_c.shape[1]
    print(f"fine nlev={nf}  coarse nlev={nc}  ncol={ncol}")

    dpf = ph_fine[:, 1:] - ph_fine[:, :-1]
    dpc = ph_c[:, 1:] - ph_c[:, :-1]

    T_c = coarse_grain(T, ph_fine, ph_c)
    qv_c = coarse_grain(qv, ph_fine, ph_c)

    wat_f = float(np.sum(A * np.sum(qv * dpf, axis=1) / constants.g))
    wat_c = float(np.sum(A * np.sum(qv_c * dpc, axis=1) / constants.g))
    print(f"water conservation check: fine {wat_f:.6e}  coarse {wat_c:.6e} kg/m2  "
          f"(rel diff {abs(wat_c-wat_f)/wat_f:.3e})")

    qc_f, _ = moist_adjust(T, qv, pf_fine, a.iters)
    qc_c, _ = moist_adjust(T_c, qv_c, pf_c, a.iters)
    C_f = float(np.sum(A * np.sum(qc_f * dpf, axis=1) / constants.g))
    C_c = float(np.sum(A * np.sum(qc_c * dpc, axis=1) / constants.g))
    print(f"condensate on the FINE layers   = {C_f:.6e} kg/m2")
    print(f"condensate on the COARSE layers = {C_c:.6e} kg/m2")
    print(f"coarse / fine = {C_c/max(C_f,1e-30):.4f}")

    edges = np.array([0., 100., 200., 300., 400., 500., 600., 700., 800., 900., 1100.]) * 100.0
    print("pressure band | fine condensate | coarse condensate | ratio")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mf = (pf_fine >= lo) & (pf_fine < hi)
        mc = (pf_c >= lo) & (pf_c < hi)
        cf = float(np.sum(A[:, None] * np.where(mf, qc_f * dpf, 0.0)) / constants.g)
        cc = float(np.sum(A[:, None] * np.where(mc, qc_c * dpc, 0.0)) / constants.g)
        if cf <= 0.0 and cc <= 0.0:
            continue
        print(f"  {lo/100:6.0f}-{hi/100:6.0f} hPa | {cf:.4e} | {cc:.4e} | "
              f"{cc/max(cf,1e-30):.4f}")

    # Saturated-fraction check: how often does each grid's layer mean reach saturation?
    from legoesm.thermo import saturation_mixing_ratio
    rh_f = qv / np.maximum(np.asarray(saturation_mixing_ratio(T, pf_fine)), 1e-30)
    rh_c = qv_c / np.maximum(np.asarray(saturation_mixing_ratio(T_c, pf_c)), 1e-30)
    for lo, hi in [(40000., 90000.)]:
        mf = (pf_fine >= lo) & (pf_fine < hi)
        mc = (pf_c >= lo) & (pf_c < hi)
        ff = float(np.sum(A[:, None] * (mf & (rh_f >= 1.0))) / max(np.sum(A[:, None] * mf), 1e-30))
        fc = float(np.sum(A[:, None] * (mc & (rh_c >= 1.0))) / max(np.sum(A[:, None] * mc), 1e-30))
        mrf = float(np.sum(A[:, None] * np.where(mf, rh_f, 0.0)) / max(np.sum(A[:, None] * mf), 1e-30))
        mrc = float(np.sum(A[:, None] * np.where(mc, rh_c, 0.0)) / max(np.sum(A[:, None] * mc), 1e-30))
        print(f"400-900 hPa saturated-layer share: fine {100*ff:.3f}%  coarse {100*fc:.3f}%; "
              f"mean relative humidity fine {mrf:.4f} coarse {mrc:.4f}")


if __name__ == "__main__":
    main()
