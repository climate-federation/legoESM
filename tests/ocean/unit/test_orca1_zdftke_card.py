"""The ORCA1 zdftke card lives in the ocean package; the OMIP driver re-exports
the same object (one builder, two consumers)."""
from __future__ import annotations

import pytest

from legoesm.ocean.fidelity import orca1_zdftke_card as card


def test_card_is_prognostic_dirichlet_nemo_mxl3_when_asked():
    cfg = card.orca1_zdftke_config(surface_bc="nemo_dirichlet", mxl_choice=3,
                                   prognostic=True, kappa_convention="veros_sqrte")
    assert cfg.prognostic and cfg.surface_bc == "nemo_dirichlet"
    assert cfg.tke_mxl_choice == 3 and cfg.kappa_convention == "veros_sqrte"
    assert cfg.eice == 3                      # NEMO ORCA1 nn_eice=3


def test_card_refuses_an_unlisted_mixing_length_choice():
    with pytest.raises(ValueError):
        card.orca1_zdftke_config(mxl_choice=max(card.TKE_MXL_CHOICES) + 1)


def test_omip_driver_uses_the_package_builder():
    from scripts.run import run_omip_core2 as r
    assert r.orca1_zdftke_config is card.orca1_zdftke_config
    assert r.TKE_MXL_CHOICES is card.TKE_MXL_CHOICES
