"""MorrisonConfig.prognostic_droplet_number, stage S10.1: the warm rain on the
prognostic droplet number and CLUBB's per-cell relvar, against the RUNNING MG2
oracle with ``nccons_in=.false.``.

Fixture ``oracle_fixtures/mg2_prognostic_nc.npz``: ``make_fixture.py --prognostic-nc``
(unmodified micro_mg2_0.F90 / micro_mg_utils.F90 of cam_cesm2_1_rel_60, gfortran
13.4 on Levante; meta in the npz).  Columns = the six warm-rain oracle columns
with per-level grid-mean nc (a fully cloudy cell with nc = 0 included), per-level
relvar in [0.001, 10], npccn = 0; points = the autoconversion kernel at rho 1 and
0.005 kg/m^3.  Bound: main's MG2 tolerance, max relative error <= 1e-10, hard-switch
limit.  The droplet-number budget (nctend) and droplet sedimentation are later
stages, so only rain-side outputs and the warm-rain mass rates are compared.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants as C  # noqa: E402
from legoesm.atmosphere.physics.microphysics import _warm_rain as W  # noqa: E402
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics  # noqa: E402
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState  # noqa: E402

from tests.atmosphere.microphysics.unit.test_mg2_warm_rain_oracle import (  # noqa: E402
    HARD_LIMIT, TOL, _production_morrison, _rel,
)

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_prognostic_nc.npz"


def _fx():
    return dict(np.load(FIXTURE))


def _cfg(**kw):
    return _production_morrison()._replace(
        **HARD_LIMIT, warm_rain_scheme="kk2000_cam6", warm_rain_incloud=True,
        prognostic_droplet_number=True, publish_qc_budget=True,
        nc_from_aerosol=False,   # the production deck's aerosol-CCN mode; refused with the switch (18a)
        **kw)


def _inputs(fx):
    g = {k: jnp.asarray(fx[f"col_{k}"]) for k in
         ("T", "q_v", "q_c", "cf", "q_r", "N_r", "p_full", "p_half", "rho", "nc_kg", "relvar")}
    dz = (g["p_half"][:, 1:] - g["p_half"][:, :-1]) / (C.g * g["rho"])
    return g, dz


def _run(fx, cfg, N_c=None, q_c=None, relvar=None):
    g, dz = _inputs(fx)
    z = jnp.zeros_like(g["T"])
    hyd = HydrometeorState(q_c=g["q_c"] if q_c is None else q_c, q_r=g["q_r"], q_i=z, q_s=z,
                           q_g=z, N_c=g["nc_kg"] * g["rho"] if N_c is None else N_c,
                           N_r=g["N_r"], N_i=z, N_s=z)   # the switch needs a snow number (1a)
    return morrison_microphysics(
        g["T"], g["q_v"], hyd, g["p_full"], g["p_half"], g["rho"], dz, float(fx["dt"]), cfg,
        cloud_fraction=g["cf"], relvar=g["relvar"] if relvar is None else relvar,
        npccn=jnp.zeros_like(g["T"]))   # S10.1 fixture: npccn = 0


def test_whole_call_rain_matches_mg2_prognostic_nc():
    fx = _fx()
    out = _run(fx, _cfg())
    qb = out.qc_budget
    rho = fx["col_rho"]
    errs = {
        "prc": _rel(-np.asarray(qb["autoconversion"]), fx["ora_prctot"]),
        "pra": _rel(-np.asarray(qb["accretion"]), fx["ora_pratot"]),
        # vapour: rain evaporation + evaporation of droplets sedimenting into
        # clear air (stage S10.3); no condensation under liquid_from_closure
        "qvlat": _rel(out.dq_v_dt, fx["ora_qvlat"]),
        "dq_r": _rel(out.dq_r_dt, fx["ora_qrtend"]),
        "dN_r": _rel(np.asarray(out.dN_r_dt) / rho, fx["ora_nrtend"]),
        "precip": _rel(out.precipitation, fx["ora_rain_sfc_flux"] + fx["ora_lflx_bot"][:, -1]),
    }
    worst = {k: float(v.max()) for k, v in errs.items()}
    print("prognostic-nc whole-call max rel err", worst)
    assert max(worst.values()) <= TOL, worst
    # coverage: the zero-number cloudy cell autoconverts in MG2 (size reset), and
    # relvar varies by level across its clip
    zero = (fx["col_nc_kg"] == 0) & (fx["col_q_c"] >= 1e-8) & (fx["col_cf"] > 0)
    assert zero.any() and (fx["ora_prctot"][zero] > 0).all()
    assert {0.001, 10.0} <= set(np.unique(fx["col_relvar"]))
    assert np.count_nonzero(fx["ora_prctot"]) > 10 and np.count_nonzero(fx["ora_pratot"]) > 10


def test_autoconversion_kernel_without_floor_matches_mg2():
    fx = _fx()
    i = {k[6:]: jnp.asarray(v) for k, v in fx.items() if k.startswith("pt_in_")}
    o = {k[7:]: v for k, v in fx.items() if k.startswith("pt_out_")}
    rho = i["rho"]
    prc, nr_au, x_c = W.autoconversion_kk2000_cam6(i["qcic"], i["ncic"] * rho, rho, i["relvar"],
                                                   nc_floor=False)
    errs = {"prc": _rel(prc, o["prc"]), "nprc": _rel(nr_au / rho, o["nprc"]),
            "nprc1": _rel(jnp.where(prc > 0, prc / x_c, 0.0), o["nprc1"])}
    worst = {k: float(v.max()) for k, v in errs.items()}
    assert max(worst.values()) <= TOL, worst
    # where the 1 m^-3 floor would bind: MG2's reset number is below it and the
    # cell autoconverts (so the floor would change the answer)
    low = (o["ncic_adj"] * np.asarray(rho) < 1.0) & (o["prc"] > 0)
    assert low.any()
    # every gated cell has a positive reset number (no floor needed for finiteness)
    gated = np.asarray(i["qcic"]) >= 1e-8
    assert (o["ncic_adj"][gated] > 0).all()


def test_guards():
    fx = _fx()
    base = _cfg()
    for kw, msg in ((dict(warm_rain_incloud=False), "warm_rain_incloud=True"),
                    (dict(warm_rain_scheme="kk2000"), 'warm_rain_scheme="kk2000_cam6"'),
                    (dict(predict_Nc=True), "predict_Nc=False"),
                    (dict(nc_from_aerosol=True), "nc_from_aerosol=False"),
                    (dict(liquid_from_closure=False), "liquid_from_closure=True")):
        with pytest.raises(ValueError, match=msg):
            _run(fx, base._replace(**kw))
    g, dz = _inputs(fx)
    z = jnp.zeros_like(g["T"])
    hyd = HydrometeorState(q_c=g["q_c"], q_r=g["q_r"], q_i=z, q_s=z, q_g=z,
                           N_c=g["nc_kg"] * g["rho"], N_r=g["N_r"], N_i=z, N_s=z)
    args = (g["T"], g["q_v"], hyd, g["p_full"], g["p_half"], g["rho"], dz, float(fx["dt"]))
    with pytest.raises(ValueError, match="relvar passed as relvar="):
        morrison_microphysics(*args, base, cloud_fraction=g["cf"])
    with pytest.raises(ValueError, match="per-cell relvar"):
        morrison_microphysics(*args, base, cloud_fraction=g["cf"], relvar=jnp.asarray(1.0))
    # switch off: CLUBB's relvar is read by every CAM6 warm-rain run (main), npccn is refused
    off = base._replace(prognostic_droplet_number=False)
    morrison_microphysics(*args, off, cloud_fraction=g["cf"], relvar=g["relvar"])
    with pytest.raises(ValueError, match="read only under"):
        morrison_microphysics(*args, off, cloud_fraction=g["cf"], relvar=g["relvar"], npccn=z)


def test_gradients_finite_through_zero_number_cell():
    fx = _fx()
    g, _ = _inputs(fx)
    cfg = _cfg()

    def f(nc, qc):
        out = _run(fx, cfg, N_c=nc, q_c=qc)
        return jnp.sum(out.dq_r_dt) + jnp.sum(out.dN_r_dt) * 1e-12

    gn, gq = jax.grad(f, argnums=(0, 1))(g["nc_kg"] * g["rho"], g["q_c"])
    assert np.isfinite(np.asarray(gn)).all() and np.isfinite(np.asarray(gq)).all()
    assert np.abs(np.asarray(gn)).max() > 0.0


def test_switch_on_call_drops_the_floor(monkeypatch):
    """The Morrison call site passes nc_floor=False under the switch (the
    whole-call fixture has no cell of low enough density for the floor to bind)."""
    import legoesm.atmosphere.physics.microphysics.morrison as M
    seen = []
    real = M.autoconversion_kk2000_cam6

    def spy(*a, **kw):
        seen.append(kw.get("nc_floor", True))
        return real(*a, **kw)

    monkeypatch.setattr(M, "autoconversion_kk2000_cam6", spy)
    _run(_fx(), _cfg())
    assert seen == [False]
