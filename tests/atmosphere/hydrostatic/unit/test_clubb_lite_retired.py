"""clubb_lite is the LES-correction campaign's stand-in only: every other
selection path refuses it and points to the faithful ``clubb``."""
import json
import logging
from pathlib import Path

import pytest

from legoesm.atmosphere.physics.turbulence.config import (
    CLUBB_LITE_STANDIN_NOTE,
    refuse_retired_turbulence,
)

REPO = Path(__file__).resolve().parents[4]


def test_helper_refuses_only_clubb_lite():
    with pytest.raises(ValueError, match="'clubb'"):
        refuse_retired_turbulence("clubb_lite", "here")
    for s in ("clubb", "louis", "tke", "none"):
        refuse_retired_turbulence(s, "here")


def test_aimip_scheme_set_refuses():
    from legoesm.training.aimip_params import validate_classical_scheme_set
    kw = dict(convection="tiedtke", cloud="xu_randall", microphysics="sundqvist",
              radiation="rrtmgp", gwd="mcfarlane")
    with pytest.raises(ValueError, match="clubb_lite"):
        validate_classical_scheme_set(turbulence="clubb_lite", **kw)
    validate_classical_scheme_set(turbulence="clubb", **kw)
    from legoesm.training.aimip_params import aimip_scheme_keys_for
    with pytest.raises(ValueError, match="clubb_lite"):
        aimip_scheme_keys_for(turbulence="clubb_lite")


def test_scm_refuses():
    from legoesm.atmosphere.physics.turbulence.les_tuned import (
        scm_turbulence_config,
    )
    with pytest.raises(ValueError, match="'clubb'"):
        scm_turbulence_config("clubb_lite")
    from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    cfg = PhysicsConfig(turbulence=TurbulenceConfig(scheme="clubb_lite"))
    with pytest.raises(ValueError, match="SingleColumnModel"):
        SingleColumnModel(physics_fn=None, state=None, phys_state=None,
                          grid=None, sigma_coord=None, dt=1.0,
                          physics_config=cfg)
    with pytest.raises(ValueError, match="physics_config"):
        SingleColumnModel(physics_fn=None, state=None, phys_state=None,
                          grid=None, sigma_coord=None, dt=1.0)


def test_sweeps_and_scm_tuning_lists_drop_it():
    import importlib.util
    paths = ["scripts/run/run_aimip_classical_sweep_stage1.py",
             "scripts/run/run_aimip_latlon_sweep.py",
             "scripts/run/run_scm_les_turbulence_tuning.py",
             "scripts/run/run_scm_rce_campaign.py",
             "scripts/run/tune_scm_to_les.py",
             "scripts/validate/rce_scm_scheme_sweep.py"]
    for p in paths:
        assert "clubb_lite" not in (REPO / p).read_text(), p
    for p in ["config/aimip/sweep/stage1/manifest.json",
              "config/aimip/sweep/stage2/stage1_winners.json"]:
        assert "clubb_lite" not in json.dumps(json.loads((REPO / p).read_text())), p
    assert not (REPO / "config/aimip/sweep/stage1/combo_turb_clubb_lite").exists()
    assert importlib.util.find_spec(
        "legoesm.atmosphere.physics.turbulence.clubb_lite") is not None


def test_campaign_path_still_accepts_and_warns_once(caplog, monkeypatch):
    import legoesm.driver.config as dc
    from scripts.experiment.write_amip_clubb_lite_config import (
        build_amip_clubb_lite_config,
    )
    monkeypatch.setattr(dc, "_CLUBB_LITE_WARNED", False)
    with caplog.at_level(logging.WARNING, logger=dc.__name__):
        cfg = build_amip_clubb_lite_config()
        cfg.validate_strict()
        cfg.validate_strict()
    assert sum(CLUBB_LITE_STANDIN_NOTE in r.getMessage()
               for r in caplog.records) == 1
