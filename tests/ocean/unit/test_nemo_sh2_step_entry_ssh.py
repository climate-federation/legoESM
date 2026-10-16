"""zdf_sh2's sea level under NEMO RK3: the whole-step entry level (V1-selected).

stprk3.f90:185 calls zdf_phy(kstp, Nbb, Nbb, Nrhs), so zdfsh2.f90:98-108's
e3w_1d*(1+r3u(Kmm)) uses the step-entry ssh, not the stage Kmm ssh that
dyn_zdf/tra_zdf read.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _nemo_sh2_eta
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig

STAGE, ENTRY = jnp.asarray([0.3]), jnp.asarray([0.1])


def test_default_keeps_the_stage_eta_on_every_other_card():
    assert _nemo_sh2_eta(TKEConfig(), prognostic=True, eta_now=STAGE,
                         eta_step_entry=ENTRY) is STAGE


def test_selected_flag_reads_the_whole_step_entry_eta():
    cfg = TKEConfig(nemo_sh2_step_entry_ssh=True)
    assert _nemo_sh2_eta(cfg, prognostic=True, eta_now=STAGE,
                         eta_step_entry=ENTRY) is ENTRY
    # no stage eta at all: the caller's own state.eta is the entry level
    assert _nemo_sh2_eta(cfg, prognostic=True, eta_now=None,
                         eta_step_entry=None) is None


def test_selected_flag_refuses_paths_that_would_ignore_it():
    cfg = TKEConfig(nemo_sh2_step_entry_ssh=True)
    with pytest.raises(ValueError, match="needs the whole-step entry eta"):
        _nemo_sh2_eta(cfg, prognostic=True, eta_now=STAGE, eta_step_entry=None)
    with pytest.raises(ValueError, match="requires prognostic TKE"):
        _nemo_sh2_eta(cfg, prognostic=False, eta_now=STAGE, eta_step_entry=ENTRY)
