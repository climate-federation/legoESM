"""Golden Morrison outputs for the prognostic_droplet_number switch-OFF bit-identity test.

Runs ``morrison_microphysics`` at the CURRENT commit on the MG2 oracle columns
(``tests/atmosphere/microphysics/oracle_fixtures/mg2_warm_rain.npz``) and on a cold
copy of them (T - 25 K, cloud ice and snow where there is cloud water) and a frozen
copy (T - 60 K, below the -40 C homogeneous-freezing threshold, same ice/snow) for four
switch-off configurations, and stores every output leaf.  The test
``tests/atmosphere/microphysics/unit/test_morrison_switch_off_golden.py`` compares
them bitwise.  Regenerating this file is itself a reviewed change: run it at the
commit recorded in the fixture meta and report the diff of the arrays.

Usage::

    JAX_ENABLE_X64=1 python scripts/validate/oracle/morrison_mg2_warm_rain/make_switch_off_golden.py \\
        --out tests/atmosphere/microphysics/oracle_fixtures/morrison_switch_off_golden.npz
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_warm_rain.npz"
CONFIGS = {
    "kk2000": dict(warm_rain_scheme="kk2000"),
    "kk2000_cam6_incloud": dict(warm_rain_scheme="kk2000_cam6", warm_rain_incloud=True,
                                liquid_from_closure=True),
    "kk2000_cam6_incloud_predict": dict(warm_rain_scheme="kk2000_cam6", warm_rain_incloud=True,
                                        liquid_from_closure=True, predict_Nc=True),
    "seifert_beheng": dict(warm_rain_scheme="seifert_beheng"),
}


def column_sets():
    """``{name: inputs}``: the oracle columns and their cold and frozen copies."""
    fx = dict(np.load(FIXTURE))
    g = {k: fx[f"col_{k}"] for k in
         ("T", "q_v", "q_c", "cf", "q_r", "N_r", "N_c", "p_full", "p_half", "rho")}
    z = np.zeros_like(g["T"])
    warm = dict(g, q_i=z, q_s=z, N_i=z)
    cold = dict(g, T=g["T"] - 25.0, q_i=np.where(g["q_c"] > 0, 1.0e-5, 0.0),
                q_s=np.where(g["q_c"] > 0, 1.0e-5, 0.0),
                N_i=np.where(g["q_c"] > 0, 1.0e5, 0.0))
    frozen = dict(cold, T=g["T"] - 60.0)
    return {"warm": warm, "cold": cold, "frozen": frozen}, float(fx["dt"])


def run(cfg_kw, g, dt):
    import jax.numpy as jnp

    from legoesm import constants as C
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    a = {k: jnp.asarray(v) for k, v in g.items()}
    dz = (a["p_half"][:, 1:] - a["p_half"][:, :-1]) / (C.g * a["rho"])
    hyd = HydrometeorState(q_c=a["q_c"], q_r=a["q_r"], q_i=a["q_i"], q_s=a["q_s"],
                           q_g=jnp.zeros_like(a["T"]), N_c=a["N_c"], N_r=a["N_r"],
                           N_i=a["N_i"])
    cfg = MorrisonConfig(morrison_flavor="mg", publish_qc_budget=True, **cfg_kw)
    kw = {"cloud_fraction": a["cf"]} if cfg_kw.get("warm_rain_incloud") else {}
    return morrison_microphysics(a["T"], a["q_v"], hyd, a["p_full"], a["p_half"], a["rho"],
                                 dz, dt, cfg, **kw)


def leaves(out):
    import jax
    flat, _ = jax.tree_util.tree_flatten_with_path(out._asdict())
    return {jax.tree_util.keystr(p): np.asarray(v) for p, v in flat}


def environment():
    import jax
    import jaxlib
    cpu = platform.processor() or platform.machine()
    try:
        cpu = next(line.split(":", 1)[1].strip() for line in
                   Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    except (OSError, StopIteration):
        pass
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    return dict(legoesm_sha=sha, jax=jax.__version__, jaxlib=jaxlib.__version__, cpu=cpu)


def main(argv=None):
    import jax
    jax.config.update("jax_enable_x64", True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    sets, dt = column_sets()
    fx = {}
    for sname, g in sets.items():
        for cname, kw in CONFIGS.items():
            for k, v in leaves(run(kw, g, dt)).items():
                fx[f"{sname}/{cname}/{k}"] = v
    fx["meta"] = np.array(json.dumps(environment()))
    np.savez_compressed(a.out, **fx)
    print("GOLDEN_OK", a.out, len(fx) - 1, "arrays")


if __name__ == "__main__":
    main()
