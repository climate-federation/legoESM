"""CAM RRTMG-style transported layer above the model top
(``RRTMGPConfig.overhead_layer`` / ``ExperimentConfig.rrtmgp_overhead_layer``).

One layer from p_top to a 1 Pa lid (top-layer T/q_v, no cloud/aerosol,
overhead column-mean ozone ``o3_top_vmr``) is solved with the column; its
heating is discarded and face 0 of the returned fluxes is the lid.  Off is
byte-identical.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

NCOL, NLEV = 2, 8
_REPO = Path(__file__).resolve().parents[2]


def _inputs():
    p_half = jnp.broadcast_to(jnp.array(
        [225.5, 503.0, 1.5e3, 5e3, 1.5e4, 3.5e4, 6e4, 8.5e4, 1e5]),
        (NCOL, NLEV + 1))
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = jnp.broadcast_to(
        jnp.array([255., 240., 225., 215., 220., 250., 275., 290.]),
        (NCOL, NLEV))
    o3 = jnp.broadcast_to(
        jnp.array([6e-6, 8e-6, 6e-6, 2e-6, 3e-7, 6e-8, 4e-8, 3e-8]),
        (NCOL, NLEV))
    return dict(T=T, p_full=p_full, p_half=p_half,
                sfc_temperature=jnp.full((NCOL,), 290.0),
                q_v=jnp.full((NCOL, NLEV), 3e-6), cos_zenith=jnp.full((NCOL,), 0.6),
                o3_vmr=o3)


@pytest.fixture(scope="module")
def solvers():
    return (RRTMGP.from_legoesm_config(RRTMGPConfig()),
            RRTMGP.from_legoesm_config(RRTMGPConfig(overhead_layer=True)))


def test_layer_absorbs_and_its_heating_is_not_applied(solvers):
    off, on = solvers
    inp = _inputs()
    o3_top = jnp.full((NCOL,), 3e-6)
    r0 = off.solve_columns(**inp)
    r1 = on.solve_columns(**inp, o3_top_vmr=o3_top)
    # Model-grid shapes; TOA incident (face 0 = lid) unchanged.
    assert r1.sw_heating_rate.shape == (NCOL, NLEV)
    assert r1.sw_flux_down.shape == (NCOL, NLEV + 1)
    np.testing.assert_allclose(r1.sw_flux_down[:, 0], r0.sw_flux_down[:, 0],
                               rtol=1e-12)
    # Overhead ozone shields k0: its SW heating drops.
    assert np.all(np.asarray(r1.sw_heating_rate[:, 0])
                  < np.asarray(r0.sw_heating_rate[:, 0]))
    # Heating discarded: model-layer SW heating integrates to LESS than
    # (TOA net - surface net) by the overhead absorption; off closes exactly.
    dp = np.asarray(inp["p_half"][:, 1:] - inp["p_half"][:, :-1])

    def _gap(r):
        net = np.asarray(r.sw_flux_down - r.sw_flux_up)
        from legoesm.atmosphere.physics.radiation.rrtmgp import constants as C
        col = np.sum(np.asarray(r.sw_heating_rate) * dp, axis=1) * C.CP_D / C.G
        return net[:, 0] - net[:, -1] - col

    gap0, gap1 = _gap(r0), _gap(r1)
    assert np.all(np.abs(gap0) < 0.05 * np.abs(gap1))
    assert np.all(gap1 > 0.0)


def test_switch_off_is_byte_identical(solvers):
    off, _ = solvers
    inp = _inputs()
    a = off.solve_columns(**inp)
    b = off.solve_columns(**inp, o3_top_vmr=jnp.full((NCOL,), 3e-6))
    for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    assert not RRTMGPConfig().overhead_layer


def test_flag_reaches_every_lane(monkeypatch):
    """Spy: every driver-built RRTMGPConfig carries the experiment flag, and
    the threaded overhead ozone reaches the solver call."""
    from legoesm.driver import model_driver, physics_pipeline
    from legoesm.driver.config import ExperimentConfig
    import legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp as rrtmgp_mod

    # Compiled cube / lat-lon (+ coupled) lane: functional spy.
    seen = {}

    def _spy(cls, cfg):
        seen["cfg"] = cfg
        raise RuntimeError("spy")

    monkeypatch.setattr(rrtmgp_mod.RRTMGP, "from_legoesm_config",
                        classmethod(_spy))
    with pytest.raises(RuntimeError, match="spy"):
        physics_pipeline._build_rrtmgp_radiation_fn(ExperimentConfig(
            radiation="rrtmgp", rrtmgp_overhead_layer=True))
    assert seen["cfg"].overhead_layer is True
    # MPAS / duo-column and spectral lanes build RRTMGPConfig inline in the
    # driver: every construction must forward the flag.
    for mod in (model_driver, physics_pipeline):
        src = inspect.getsource(mod)
        n_cfg = len(re.findall(r"\bRRTMGPConfig\(\n", src))
        n_flag = src.count('rrtmgp_overhead_layer", False)') + src.count(
            "rrtmgp_overhead_layer', False)")
        assert n_cfg >= 1 and n_flag == n_cfg, mod.__name__

    # integration (MPAS / spectral) forwards o3_top_vmr with the override.
    from legoesm.atmosphere.physics.radiation import integration
    src = inspect.getsource(integration)
    assert src.count("o3_top_vmr=_o3_top_ext,") == 3
    assert '_rad_kwargs["o3_top_vmr"] = o3_top_vmr' in src


def test_cli_and_cam6_decks():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args)
    from scripts.run.run_coupled import build_parser as coupled_parser
    p = build_arg_parser()
    off = build_config_from_args(_postprocess_args(
        p.parse_args(["--dataset", "analytical"]), p))
    on = build_config_from_args(_postprocess_args(p.parse_args(
        ["--dataset", "analytical", "--rrtmgp-overhead-layer"]), p))
    assert (off.rrtmgp_overhead_layer, on.rrtmgp_overhead_layer) == (False, True)
    assert coupled_parser().parse_args([]).rrtmgp_overhead_layer is False
    assert coupled_parser().parse_args(
        ["--rrtmgp-overhead-layer"]).rrtmgp_overhead_layer is True
    from scripts.run import run_coupled
    assert ("rrtmgp_overhead_layer=args.rrtmgp_overhead_layer,"
            in inspect.getsource(run_coupled.main))
    for deck in ("amip_production.yaml", "amip_production_fv3duo_c24.yaml"):
        d = yaml.safe_load((_REPO / "config" / "amip" / deck).read_text())
        assert d["rrtmgp_overhead_layer"] is True, deck
    sh = (_REPO / "config" / "amip" / "amip_production.ginsburg.sh").read_text()
    assert "CMIP6_FORCING_FLAGS+=( --rrtmgp-overhead-layer )" in sh


def test_edge_cases_cloudy_no_ozone_zero_top():
    """Liquid-only cloud (no r_eff: the solver's own fallbacks), no o3_vmr
    (standard profile; o3_top_vmr still used) and a 0 Pa top all solve and
    stay finite with the layer on."""
    on = RRTMGP.from_legoesm_config(
        RRTMGPConfig(overhead_layer=True, include_clouds=True))
    inp = _inputs()
    inp.pop("o3_vmr")
    lwp = jnp.zeros((NCOL, NLEV)).at[:, 5].set(0.05)
    a = on.solve_columns(**inp, cloud_path_liq=lwp)
    b = on.solve_columns(**inp, cloud_path_liq=lwp,
                         o3_top_vmr=jnp.full((NCOL,), 3e-5))
    assert np.all(np.asarray(b.sw_heating_rate[:, 0])
                  < np.asarray(a.sw_heating_rate[:, 0]))  # o3_top honoured
    z = dict(inp, p_half=inp["p_half"].at[:, 0].set(0.0))
    z["p_full"] = 0.5 * (z["p_half"][:, 1:] + z["p_half"][:, :-1])
    for r in (a, b, on.solve_columns(**z)):
        for leaf in jax.tree_util.tree_leaves(r):
            assert np.all(np.isfinite(np.asarray(leaf)))
