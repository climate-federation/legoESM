"""MorrisonConfig.specified_nc_droplet_sedimentation: CAM6 MG2 cloud-droplet
sedimentation with a SPECIFIED droplet number (MG2 nccons=.true.), against the
RUNNING MG2.

Fixture ``oracle_fixtures/mg2_specified_nc_sed.npz``:
``scripts/validate/oracle/morrison_mg2_specified_nc_sed/make_fixture.py``, the
unmodified micro_mg2_0.F90 of cam_cesm2_1_rel_60 with nccons_in=.true. and the
column's ncnst, on seven synthetic warm columns and twelve ice-free warm-cloud
columns dumped from a legoESM C1 run.  Bound 1e-10 relative, x64.
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
    HARD_LIMIT, TOL, _production_morrison, _rel)
from tests.atmosphere.microphysics.unit.test_morrison_prognostic_nc_sed import _rel_flux  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
FIXTURE = ROOT / "tests/atmosphere/microphysics/oracle_fixtures/mg2_specified_nc_sed.npz"
GROUPS = ("syn", "dump")
FIELDS = ("T", "q_v", "q_c", "cf", "q_r", "N_r", "N_c", "p_full", "p_half", "rho", "relvar")


def _fx():
    return dict(np.load(FIXTURE))


def _base():
    return _production_morrison()._replace(**HARD_LIMIT, warm_rain_scheme="kk2000_cam6",
                                           publish_qc_budget=True)


def _cols(fx, g, rows):
    return {k: jnp.asarray(fx[f"{g}_col_{k}"][rows]) for k in FIELDS}


def _run(a, dt, cfg, snow=False, q_c=None):
    z = jnp.zeros_like(a["T"])
    dz = (a["p_half"][:, 1:] - a["p_half"][:, :-1]) / (C.g * a["rho"])
    hyd = HydrometeorState(q_c=a["q_c"] if q_c is None else q_c, q_r=a["q_r"], q_i=z,
                           q_s=z, q_g=z, N_c=a["N_c"], N_r=a["N_r"], N_i=z,
                           N_s=z if snow else None)
    return morrison_microphysics(a["T"], a["q_v"], hyd, a["p_full"], a["p_half"], a["rho"],
                                 dz, dt, cfg, cloud_fraction=a["cf"], relvar=a["relvar"])


def _cases(fx):
    """(label, group, rows, config, snow): the specified number as the
    caller-filled field (nc_from_aerosol) on every column, and as the constant
    Nc_0 with an MG2 snow number present (the C1 deck's path) per ncnst."""
    on = _base()._replace(specified_nc_droplet_sedimentation=True)
    out = []
    for g in GROUPS:
        n = fx[f"{g}_col_T"].shape[0]
        out.append((f"{g} field", g, np.arange(n), on, False))
        for nc in np.unique(fx[f"{g}_col_N_c"][:, 0]):
            rows = np.flatnonzero(fx[f"{g}_col_N_c"][:, 0] == nc)
            out.append((f"{g} Nc_0={nc:g}", g, rows,
                        on._replace(nc_from_aerosol=False, Nc_0=float(nc)), True))
    return out


def _sed_scale(a, out, dt):
    """Largest droplet flux term entering or leaving each cell, |fc q|/pdel on
    the post-process state: the scale of the round-off of MG2's flux
    differences (the evaporated inflow share is F(k) - F(k-1) minus the net)."""
    q = a["q_c"] + (out.dq_c_dt - out.qc_budget["sedimentation"]) * dt
    lcl = jnp.maximum(a["cf"], W.MG2_MINCLD)
    fc, _ = W.mg2_droplet_fall_coefficients(q, a["N_c"] / a["rho"] * lcl, lcl, a["T"],
                                            a["rho"], presence_top=W.MG2_QSMALL)
    F = np.abs(np.asarray(fc * q))
    up = np.concatenate([np.zeros_like(F[:, :1]), F[:, :-1]], axis=1)
    return np.maximum(F, up) / np.diff(np.asarray(a["p_half"]), axis=1)


def test_fixture_is_nccons_and_sediments():
    import json
    fx = _fx()
    meta = json.loads(str(fx["meta"]))
    assert meta["tag"] == "cam_cesm2_1_rel_60" and meta["nccons"] is True
    for g in GROUPS:
        assert np.abs(fx[f"{g}_ora_qcsedten"]).max() > 0
        assert np.abs(fx[f"{g}_ora_lflx_bot"][:, -1]).max() > 0
        wet = fx[f"{g}_col_q_c"] > 0
        assert (fx[f"{g}_col_T"][wet] > C.T_freeze).all()


def test_specified_nc_sedimentation_matches_mg2_nccons():
    fx = _fx()
    dt = float(fx["dt"])
    worst_all = {}
    for label, g, rows, cfg, snow in _cases(fx):
        a = _cols(fx, g, rows)
        o = {k: fx[f"{g}_ora_{k}"][rows] for k in
             ("qctend", "qcsedten", "qvlat", "tlat", "lflx_bot", "qcrat")}
        on = _run(a, dt, cfg, snow)
        off = _run(a, dt, cfg._replace(specified_nc_droplet_sedimentation=False), snow)
        scale = _sed_scale(a, on, dt)
        errs = {
            "qctend": _rel(on.dq_c_dt, o["qctend"]),
            "qcsedten": _rel_flux(on.qc_budget["sedimentation"], o["qcsedten"],
                                  np.where(o["qcrat"] < 1.0, scale, 0.0)),
            # the evaporated inflow share is a difference of fluxes, so its
            # round-off is eps x the flux scale, not eps x the share
            "qvlat": _rel_flux(on.dq_v_dt, o["qvlat"], scale),
            "tlat": _rel_flux(np.asarray(on.dT_dt) * C.c_pd, o["tlat"], C.L_v * scale),
            # the droplet share of the surface precipitation (rain is unchanged)
            "sfc_droplets": _rel(np.asarray(on.precipitation) - np.asarray(off.precipitation),
                                 o["lflx_bot"][:, -1]),
        }
        worst = {k: float(v.max()) for k, v in errs.items()}
        worst_all[label] = worst
        # the rain and the (not evolved) droplet number do not see the flag
        assert np.array_equal(np.asarray(on.dq_r_dt), np.asarray(off.dq_r_dt))
        assert np.array_equal(np.asarray(on.dN_c_dt), np.asarray(off.dN_c_dt))
    print("specified-Nc sedimentation max rel err", worst_all)
    assert max(max(w.values()) for w in worst_all.values()) <= TOL, worst_all


def test_flag_off_does_not_sediment_and_misses_mg2():
    """Non-vacuity: without the flag cloud water does not sediment and the
    whole-call cloud-water tendency is far from MG2 nccons."""
    fx = _fx()
    for g in GROUPS:
        a = _cols(fx, g, slice(None))
        off = _run(a, float(fx["dt"]), _base())
        assert "sedimentation" not in off.qc_budget
        assert float(_rel(off.dq_c_dt, fx[f"{g}_ora_qctend"]).max()) > 1e-3


@pytest.mark.parametrize("change, msg", [
    (dict(warm_rain_incloud=False), "warm_rain_incloud=True"),
    (dict(predict_Nc=True), "predict_Nc=False"),
    (dict(prognostic_droplet_number=True), "prognostic_droplet_number=False"),
])
def test_needs_the_mg2_specified_droplet_mode(change, msg):
    fx = _fx()
    a = _cols(fx, "syn", slice(0, 1))
    cfg = _base()._replace(specified_nc_droplet_sedimentation=True, **change)
    with pytest.raises(ValueError, match=msg):
        _run(a, float(fx["dt"]), cfg)


def test_gradients_through_specified_nc_sedimentation():
    fx = _fx()
    a = _cols(fx, "dump", slice(0, 2))
    cfg = _base()._replace(specified_nc_droplet_sedimentation=True)

    def f(qc):
        out = _run(a, float(fx["dt"]), cfg, q_c=qc)
        return jnp.sum(out.qc_budget["sedimentation"]) * 1e6 + jnp.sum(out.precipitation) * 1e3

    g = np.asarray(jax.grad(f)(a["q_c"]))
    assert np.isfinite(g).all() and np.abs(g).max() > 0


def test_deck_key_and_cli_flag_reach_the_scheme(tmp_path):
    """Default off at every level; --morrison-specified-nc-droplet-sedimentation
    and the deck key morrison_specified_nc_droplet_sedimentation both reach
    MorrisonConfig through the lanes' threading."""
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.driver.config import ExperimentConfig
    from legoesm.driver.physics_pipeline import thread_morrison_scalars
    from legoesm.driver.run_config_yaml import load_yaml_config
    from scripts.run.run_amip import _postprocess_args, build_arg_parser, build_config_from_args
    key = "morrison_specified_nc_droplet_sedimentation"
    assert ExperimentConfig._field_defaults[key] is False
    assert MorrisonConfig().specified_nc_droplet_sedimentation is False
    leaf = MorrisonConfig()
    assert thread_morrison_scalars(ExperimentConfig(microphysics="morrison"), "morrison",
                                   leaf) is leaf
    on = ExperimentConfig(microphysics="morrison", **{key: True})
    assert thread_morrison_scalars(on, "morrison", leaf).specified_nc_droplet_sedimentation
    parser = build_arg_parser()
    argv = ["--dataset", "analytical", "--microphysics", "morrison"]
    c = build_config_from_args(_postprocess_args(parser.parse_args(argv), parser))
    assert getattr(c, key) is False
    c = build_config_from_args(_postprocess_args(parser.parse_args(
        argv + ["--morrison-specified-nc-droplet-sedimentation"]), parser))
    assert getattr(c, key) is True
    deck = tmp_path / "deck.yaml"
    deck.write_text(f"microphysics: morrison\n{key}: true\n")
    parser = build_arg_parser()
    keys = load_yaml_config(deck, parser, example_keys="'microphysics'")
    parser.set_defaults(**keys)
    c = build_config_from_args(_postprocess_args(parser.parse_args(
        ["--config", str(deck), "--dataset", "analytical"]), parser))
    assert getattr(c, key) is True


def test_sed_reset_width_is_accepted_with_specified_sedimentation():
    """The specified-number sedimentation reads mg2_drop_sed_reset_log_width,
    so a non-default width is not refused as inert there."""
    from tests.unit.test_cam6_cloud_fraction import _exp_cfg
    cfg = _exp_cfg(microphysics="morrison", morrison_warm_rain_incloud=True,
                   morrison_specified_nc_droplet_sedimentation=True,
                   morrison_mg2_drop_sed_reset_log_width=0.0)
    try:
        cfg.validate_strict()
    except ValueError as e:
        assert "mg2_drop_sed_reset_log_width" not in str(e), str(e)
    off = _exp_cfg(microphysics="morrison", morrison_mg2_drop_sed_reset_log_width=0.0)
    with pytest.raises(ValueError, match="mg2_drop_sed_reset_log_width"):
        off.validate_strict()
