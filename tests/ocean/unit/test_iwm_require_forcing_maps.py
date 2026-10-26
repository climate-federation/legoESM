"""The wave-mixing arm refuses the uniform fallback when a card reads real maps.

A configuration that loads de Lavergne power maps and then silently falls back
to the constant-power defaults would be running different physics with nothing
going red.  ``IWMConfig.require_forcing_maps`` turns that into a refusal; this
proves the refusal fires, that it is opt-in, and that the ORCA2 card sets it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (
    IWMConfig,
    uniform_iwm_forcing,
)

DECK = Path("/data/abyssal/dbalwada/nemo-testcases-l4/runs/"
            "variant_orca1ice_phase2x_a_10step_np2")


def test_the_flag_is_off_by_default() -> None:
    assert IWMConfig().require_forcing_maps is False


def test_the_uniform_fallback_still_builds_when_the_flag_is_off() -> None:
    forcing = uniform_iwm_forcing(IWMConfig(), (2, 3))
    assert forcing.ebot.shape == (2, 3)
    # zdfiwm_init stores the RECIPROCAL of the critical-slope decay scale.
    assert float(forcing.hcri_inv[0, 0]) == pytest.approx(1.0 / 100.0)


def test_a_card_that_requires_maps_refuses_the_fallback() -> None:
    import jax.numpy as jnp

    from legoesm.ocean.physics.vertical_mixing import k_profiles

    cfg = IWMConfig(enabled=True, require_forcing_maps=True)

    class _Stub:
        pass

    # The refusal must happen before any operand is touched, so a stub state
    # is enough to reach it; if the guard were removed this call would get a
    # long way further and then fail differently.
    with pytest.raises(ValueError, match="require_forcing_maps"):
        k_profiles.iwm_K_profile(
            _Stub(), _Stub(), _Stub(), cfg, eos_fn=None, iwm_fields=None)
    assert jnp is not None  # the module imported under the x64 policy


@pytest.mark.skipif(not DECK.is_dir(), reason="ORCA2 deck not on this machine")
def test_the_orca2_card_reads_real_maps_and_says_so() -> None:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card

    card = build_orca2_zps_card(DECK)
    iwm = card.recipe.model_config.physics.vertical_mixing.iwm
    assert iwm.enabled is True
    assert iwm.require_forcing_maps is True
    # namelist_cfg's namzdf_iwm: ln_mevar = .false.
    assert iwm.mevar is False
    forcing = card.recipe.iwm_forcing
    assert forcing is not None
    assert forcing.ebot.shape == (148, 180)
    # NEMO masks the four power maps with the surface tracer mask and leaves
    # the decay scales alone.
    assert float(forcing.ebot.min()) == 0.0
    assert float(forcing.hcri_inv.min()) > 0.0
    # The deck's salt/heat differentials are declared, not silently dropped.
    assert "internal_wave_salt_heat_differential" in card.unmeasured_features
    assert "double_diffusive_salt_heat_split" in card.unmeasured_features
    assert "internal_wave_mixing" not in card.unmeasured_features
