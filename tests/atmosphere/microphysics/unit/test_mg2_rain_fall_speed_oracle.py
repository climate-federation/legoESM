"""CAM6 MG2 rain on the 12 warm-cloud columns dumped from a C1 run, against
the RUNNING MG2 (unmodified micro_mg2_0.F90 of cam_cesm2_1_rel_60, nccons).

Fixture: ``oracle_fixtures/mg2_specified_nc_sed.npz`` (rain outputs qrtend,
nrtend and the surface rain flux rflx(nlev+1) of the same driver run; see
scripts/validate/oracle/morrison_mg2_specified_nc_sed).  The process-by-process
probe that located the difference is
scripts/validate/oracle/morrison_mg2_rain_procs.

Known fault (strict xfail until fixed): legoESM sizes the MG2 rain fall speed
of the post-process state with
  (a) a number set to 0 where the GRID-mean post-process rain is below qsmall
      (morrison.py, N_r_sed), while MG2 sizes it from the in-precip
      dumnr = max((nr + nrtend dt)/precip_frac, 0) (micro_mg2_0.F90:2068,
      :2160-2172) and zeroes only the number it transports (:2209, :2215);
  (b) rhof = (rhosu/max(rho, 0.1))^0.54, while MG2's rhof is unfloored
      (micro_mg2_0.F90:895).
Both change the trace-rain fall speeds aloft, hence MG2's one-per-column rain
sub-step count (:2386), hence the rain that reaches the surface (1 to 12.5
percent here).  Remove the xfail with the fix.  Bound 1e-10 relative, x64.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import (  # noqa: E402
    HARD_LIMIT, TOL, _production_morrison, _rel)

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_specified_nc_sed.npz"
FIELDS = ("T", "q_v", "q_c", "cf", "q_r", "N_r", "N_c", "p_full", "p_half", "rho", "relvar")


def _run(a, dt, cfg, snow):
    dz = (a["p_half"][:, 1:] - a["p_half"][:, :-1]) / (C.g * a["rho"])
    z = jnp.zeros_like(a["T"])
    hyd = HydrometeorState(q_c=a["q_c"], q_r=a["q_r"], q_i=z, q_s=z, q_g=z, N_c=a["N_c"],
                           N_r=a["N_r"], N_i=z, N_s=z if snow else None)
    return morrison_microphysics(a["T"], a["q_v"], hyd, a["p_full"], a["p_half"], a["rho"],
                                 dz, dt, cfg, cloud_fraction=a["cf"], relvar=a["relvar"])


def _cases(fx):
    base = _production_morrison()._replace(**HARD_LIMIT, warm_rain_scheme="kk2000_cam6")
    n = fx["dump_col_T"].shape[0]
    out = [("field", np.arange(n), base, False)]
    for nc in np.unique(fx["dump_col_N_c"][:, 0]):
        rows = np.flatnonzero(fx["dump_col_N_c"][:, 0] == nc)
        out.append((f"Nc_0={nc:g}", rows, base._replace(nc_from_aerosol=False, Nc_0=float(nc)),
                    True))
    return out


@pytest.mark.xfail(strict=True, reason="MG2 rain fall-speed port fault (a) and (b) in the "
                   "module docstring; results/legoesm/mg2_rain/mg2_rain_vs_cam.md")
def test_mg2_rain_matches_cam_on_dumped_c1_columns():
    fx = dict(np.load(FIXTURE))
    dt = float(fx["dt"])
    worst = {}
    for label, rows, cfg, snow in _cases(fx):
        a = {k: jnp.asarray(fx[f"dump_col_{k}"][rows]) for k in FIELDS}
        # droplet sedimentation off: rain does not see it (asserted in
        # test_morrison_specified_nc_sed) and the precipitation is then rain only
        out = _run(a, dt, cfg, snow)
        o = {k: fx[f"dump_ora_{k}"][rows] for k in ("qrtend", "nrtend", "rflx_bot")}
        worst[label] = {
            "qrtend": float(_rel(out.dq_r_dt, o["qrtend"]).max()),
            "nrtend": float(_rel(np.asarray(out.dN_r_dt) / fx["dump_col_rho"][rows],
                                 o["nrtend"]).max()),
            "surface rain": float(_rel(out.precipitation, o["rflx_bot"][:, -1]).max()),
        }
    print("MG2 rain on dumped C1 columns, max rel err", worst)
    assert max(max(w.values()) for w in worst.values()) <= TOL, worst
