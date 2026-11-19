"""Validation gate of the interactive CloudBench radiation against the LES's own.

Per case, the LES day-5 mean state (theta_l, q_t, q_c, q_r from the artifact, pressure
from the GCM sounding's ln p in height, as the LES reference state) is given to
:func:`cloudbench_radiation.column_radiation` on two grids:

* ``les``: the LES heights (interfaces halfway between nodes, top at the last node
  plus half a cell, surface at 0 m);
* ``scm``: the 32-layer column the SCM runs (``cloudbench_grid``), with the day-5
  state averaged over each layer's pressure range (q_c too, as the SCM carries it).

Each is compared with the LES's saved day-5 mean radiative heating on the same
layers: the column-integrated 0-6 km heating difference [W/m2] and the
mass-weighted RMS profile difference [K/day]. Variants on the ``scm`` grid: clear
sky (no cloud optics), standard instead of MLS ozone, and the cloud water in the
LES's own day-5 cloud cover (McICA, in-cloud water and radius), and a Xu-Randall
cover diagnosed from the column's own state (McICA). A check that the solver's incident
SW at the extension top equals irrad * cos(zenith) is recorded per case. Numbers
only; the gate thresholds are applied by the caller.

LIMITATION: the LES heating is compared after interpolating and averaging its stored
heating profile, which does not keep the LES's boundary-flux budget; the
flux-differenced comparison of ``cloudbench_rad_resolution.py`` supersedes these
numbers as the gate.

usage: cloudbench_rad_validate.py OUT_JSON CASE [CASE ...]   (CASE = sSSS_mMM)
Run with JAX_ENABLE_X64=1 on a compute node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.les_suite.bridge import load_artifact
from legoesm.atmosphere.les_suite.cloudbench import read_sounding
from legoesm.atmosphere.les_suite.cloudbench_radiation import (
    case_radiation,
    column_radiation,
    rrtmgp_config,
)
from legoesm.atmosphere.les_suite.cloudbench_scm import cloudbench_grid
from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    cover_saturation_specific_humidity,
    xu_randall_cloud_fraction,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import standard_o3_profile

ART = Path("/glade/derecho/scratch/pg2328/les_suite/artifacts/cloudbench")
FORCING = Path("/glade/derecho/scratch/pg2328/les_suite/external/cloudbench/forcing")
NLEV = 32


def _p_of_z(sounding, z):
    return np.exp(np.interp(z, sounding["z"], np.log(sounding["p"])))


def day5_state(art):
    """Day-5 mean LES state on its heights (surface-first)."""
    th, qt = np.asarray(art.theta)[1:].mean(0), np.asarray(art.qt)[1:].mean(0)
    qc, qr = np.asarray(art.qc)[1:].mean(0), np.asarray(art.qr)[1:].mean(0)
    cf = np.asarray(art.cloud_frac)[1:].mean(0)
    rad = np.asarray(art.rad_T_tendency)[1:].mean(0)
    return th, qt, qc, qr, cf, rad


def thermo(th, qt, qc, qr, p):
    exner = (p / constants.p_ref) ** constants.kappa
    T = exner * th + constants.L_v * qc / constants.c_pd
    qv = qt - qc                     # swirl-lm's vapour (its q_t carries no rain)
    rho = p / (constants.R_d * T * (1.0 + (constants.R_v / constants.R_d - 1.0) * qv - qc - qr))
    return T, qv, rho


def solve(case, T, qv, qc, pf, ph, rho, **kw):
    """Top-first inputs; returns heating [K/s] top-first and the record (compiled,
    as the column runs it)."""
    if abs(float(case.ext_p_half[-1]) - float(ph[0])) > 1e-9 * float(ph[0]):
        raise ValueError("the extension does not end at the column's top interface")
    fn = jax.jit(lambda *a: column_radiation(case, *a, **kw))
    rec = fn(*(jnp.asarray(a) for a in (T, qv, qc, pf, ph, rho)))
    return np.asarray(rec.dT_dt), rec


def layer_mean(x, ph, pf, p_half):
    """Pressure-weighted mean of the LES profile ``x`` (surface-first, on levels
    ``pf`` between interfaces ``ph``) over each SCM layer (``p_half`` top-first)."""
    xi = np.concatenate([[x[0]], x, [x[-1]]])
    pi = np.concatenate([[ph[0]], pf, [ph[-1]]])   # surface-first, decreasing p
    fine = np.linspace(0.0, 1.0, 64)
    res = []
    for k in range(len(p_half) - 1):
        pp = p_half[k] + fine * (p_half[k + 1] - p_half[k])
        res.append(np.interp(-pp, -pi, xi).mean())
    return np.asarray(res)


def metrics(ours, les, dp):
    """``ours``/``les`` heating [K/s] on layers of pressure thickness ``dp``."""
    w = dp / dp.sum()
    col = constants.c_pd / constants.g * (dp * (ours - les)).sum()
    return {"col_diff_W_m2": float(col),
            "les_col_W_m2": float(constants.c_pd / constants.g * (dp * les).sum()),
            "rms_K_day": float(86400 * np.sqrt((w * (ours - les) ** 2).sum()))}


def case_report(c):
    art = load_artifact(ART / f"cloudbench_{c}_amip.npz")
    par = art.les_setup["parameters"]
    snd = read_sounding(FORCING / str(int(c[1:4])) / str(int(c[6:8])) / par["experiment"]
                        / "sounding.csv")
    z = np.asarray(art.heights_m)
    th, qt, qc, qr, cf, rad = day5_state(art)
    out = {"regime_max_qc_g_kg": float(1e3 * qc.max()),
           "clear_sky_like": bool(qc.max() < 1e-5)}

    # --- LES grid ---
    zh = np.concatenate([[0.0], 0.5 * (z[1:] + z[:-1]), [z[-1] + 0.5 * (z[-1] - z[-2])]])
    ph, pf = _p_of_z(snd, zh), _p_of_z(snd, z)
    T, qv, rho = thermo(th, qt, qc, qr, pf)
    case = case_radiation(par, snd, float(ph[-1]))
    h, rec = solve(case, T[::-1], qv[::-1], qc[::-1], pf[::-1], ph[::-1], rho[::-1])
    dp = np.diff(ph[::-1])
    out["les_grid"] = metrics(h, rad[::-1], dp)

    # --- SCM grid: layer means over each SCM layer's pressure range ---
    grid, _ = cloudbench_grid(snd, float(par["p_sfc"]), NLEV)
    p_half = np.asarray(grid.p_half_pa)          # top-first
    pf_s = np.asarray(grid.p_full_pa)

    ths, qts, qcs, qrs, cfs, rads = (layer_mean(a, ph, pf, p_half)
                                     for a in (th, qt, qc, qr, cf, rad))
    Ts, qvs, rhos = thermo(ths, qts, qcs, qrs, pf_s)
    case_s = case_radiation(par, snd, float(p_half[0]))
    dps = np.diff(p_half)
    hs, rec_s = solve(case_s, Ts, qvs, qcs, pf_s, p_half, rhos)
    out["scm_grid"] = metrics(hs, rads, dps)
    out["scm_grid"]["net_down_top_W_m2"] = float(rec_s.net_down_top)
    out["scm_grid"]["net_down_sfc_W_m2"] = float(rec_s.net_down_sfc)
    out["scm_clear"] = metrics(solve(case_s, Ts, qvs, qcs, pf_s, p_half, rhos, clouds=False)[0],
                               rads, dps)
    out["scm_std_ozone"] = metrics(solve(case_s, Ts, qvs, qcs, pf_s, p_half, rhos,
                                         ozone=standard_o3_profile)[0], rads, dps)
    # the LES's own day-5 cloud cover, McICA-sampled with in-cloud water: what a
    # column that knew the cover would get (a reference, not a column input)
    out["scm_mcica_les_cover"] = metrics(
        solve(case_s, Ts, qvs, qcs, pf_s, p_half, rhos, cloud_fraction=cfs)[0], rads, dps)
    out["cover_max"] = float(cfs.max())
    # a cover the column can diagnose from its own state: Xu-Randall (1996) at the
    # repo's CloudConfig defaults, McICA-sampled
    ccfg = CloudConfig()
    qs = np.asarray(cover_saturation_specific_humidity(jnp.asarray(Ts), jnp.asarray(pf_s), ccfg))
    cf_xr = np.asarray(xu_randall_cloud_fraction(jnp.asarray(qvs / qs), jnp.asarray(qcs),
                                                 jnp.asarray(qs), ccfg))
    out["scm_mcica_xu_randall"] = metrics(
        solve(case_s, Ts, qvs, qcs, pf_s, p_half, rhos, cloud_fraction=cf_xr)[0], rads, dps)
    out["xu_randall_cover_max"] = float(cf_xr.max())

    # --- incident SW at the top of the extension ---
    from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import rrtmgp_radiation
    o = rrtmgp_radiation(jnp.asarray(case_s.ext_T)[None], jnp.asarray(case_s.ext_p_full)[None],
                         jnp.asarray(case_s.ext_p_half)[None], jnp.asarray([case_s.sst]),
                         jnp.asarray(case_s.ext_q)[None], jnp.asarray([case_s.cos_zenith]),
                         rrtmgp_config(case_s.irrad))
    out["sw_down_top_W_m2"] = float(o.sw_flux_down[0, 0])
    out["irrad_cos_zenith_W_m2"] = float(case_s.irrad * case_s.cos_zenith)
    return out


def main() -> int:
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1")
    res = {}
    for c in sys.argv[2:]:
        res[c] = case_report(c)
        print(c, json.dumps({k: res[c][k] for k in ("les_grid", "scm_grid")}), flush=True)
        Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
