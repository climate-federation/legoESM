"""Attribute the IFS lane's clear-sky gap to the RRTMGP solve.

``main``: ecRad solvers against our RTE solver on the SAME gas optics, which isolates
the solver.  ``ablations``: removes one input difference at a time, cumulatively.
Run one per process: the ablations rebind module constants.
"""
import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).parents[4]
sys.path.insert(0, str(_ROOT))
from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ecrad  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import gas_optics as go  # noqa: E402
from tests.atmosphere.test_ecrad_ifs_rrtmgp_gas import clear_sky_inputs  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("audit.py"))
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _sfc_first(a):
    """(ncol, n, ng) top-down -> (ng, ncol, n) surface-first."""
    return np.moveaxis(np.asarray(a), -1, 0)[..., ::-1]


def main():
    args, ref = clear_sky_inputs()
    arm_ecrad = ecrad.radiation(**args)
    albedo = ecrad.lw_albedo_from_emissivity(args["lw_emissivity"])
    gas = go.gas_optics(args["pressure_hl"], args["temperature_hl"], args["gas_mmr"],
                        args["skin_temperature"], albedo, args["solar_irradiance"],
                        constants.g_ecrad, 1e-15, args["sharpness"])
    tau_lw, tau_sw, ssa_sw = _sfc_first(gas.od_lw), _sfc_first(gas.od_sw), _sfc_first(gas.ssa_sw)
    zero_lw = np.zeros_like(tau_lw)
    sfc = np.moveaxis(np.asarray(gas.lw_emission / (1.0 - albedo)), -1, 0) / np.pi
    emis = np.asarray(args["lw_emissivity"][:, 0])
    alb = np.asarray(args["sw_albedo"][:, 0])
    arms = audit.solve_with_oracle_optics(
        tau_lw, zero_lw, zero_lw, _sfc_first(gas.planck_hl) / np.pi, sfc, tau_sw, ssa_sw,
        np.zeros_like(tau_sw), np.asarray(gas.incoming_sw[0]), emis, alb, alb,
        np.asarray(args["cos_sza"]), jnp.float64)
    names = (("flux_up_lw", "lw_flux_up"), ("flux_dn_lw", "lw_flux_down"),
             ("flux_up_sw", "sw_flux_up"), ("flux_dn_sw", "sw_flux_down"))
    for (name, ref_key), arm in zip(names, arms):
        ours = np.asarray(arm)[:, ::-1]  # surface-first -> top-down
        rrtmgp = np.asarray(getattr(ref, ref_key))
        ifs = np.asarray(arm_ecrad[name])
        for label, d in (("total", ifs - rrtmgp), ("solver", ifs - ours), ("preparation", ours - rrtmgp)):
            k = int(np.unravel_index(np.abs(d).argmax(), d.shape)[1])
            print(f"{name:11s} {label:12s} max {np.abs(d).max():.3e} at level {k}/{d.shape[1]-1} "
                  f"toa {np.abs(d[:, 0]).max():.3e} sfc {np.abs(d[:, -1]).max():.3e} "
                  f"rms {np.sqrt((d**2).mean()):.3e}", flush=True)


def _gap(args, ref, tag):
    out = ecrad.radiation(**args)
    print(f"ABLATION {tag:22s}", " ".join(
        f"{k} {np.abs(np.asarray(out[k]) - np.asarray(getattr(ref, r))).max():.3e}"
        for k, r in (("flux_up_lw", "lw_flux_up"), ("flux_dn_lw", "lw_flux_down"),
                     ("flux_up_sw", "sw_flux_up"), ("flux_dn_sw", "sw_flux_down"))), flush=True)


def ablations():
    """Remove one preparation difference at a time; cumulative."""
    args, ref = clear_sky_inputs()
    _gap(args, ref, "as shipped")
    means = go._optics().vmr_lib.global_means
    go._O2_VMR = float(means["o2"])
    _gap(args, ref, "o2 library")
    go._ABSENT_FROM_RRTM = ()
    mmr = args["gas_mmr"]
    for name, index in (("cfc11", go.ICFC11), ("cfc12", go.ICFC12), ("cfc22", go.IHCFC22), ("ccl4", go.ICCL4)):
        mmr = mmr.at[..., index].set(float(means[name]) * go._MOLAR_MASS[name][1] / go._AIR_MOLAR_MASS)
    _gap(dict(args, gas_mmr=mmr), ref, "co, no2, halocarbons")
    from legoesm.atmosphere.physics.radiation.rrtmgp import constants as rc
    ecrad._GRAVITY = rc.G
    _gap(dict(args, gas_mmr=mmr), ref, "gravity")
    from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as rr
    rr.cam_interface_temperature = lambda *_: args["temperature_hl"]
    _, same_levels = clear_sky_inputs()
    _gap(dict(args, gas_mmr=mmr), same_levels, "level temperature")


if __name__ == "__main__":
    main()
    ablations()
