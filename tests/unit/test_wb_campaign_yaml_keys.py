"""Campaign-YAML unknown-key rejection (scale_build.validate_wb_campaign_yaml).

Inert deck keys are how the wb_classical_v2 campaign trained at the wrong
learning rate (its ``lr: 1.5e-3`` was never read); a ``si_substep`` typo would
silently revert a deck to the unstable full timestep.  Every committed deck
must validate clean, and typos must be refused with a nearest-match hint.
"""
from pathlib import Path

import pytest
import yaml
from legoesm.training.scale_build import validate_wb_campaign_yaml

_WB_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "wb"
# campaign decks AND the trainer's default deck live under config/wb/
_DECKS = sorted((_WB_CONFIG_DIR / "campaign").glob("*.yaml")) + \
    sorted((_WB_CONFIG_DIR / "scale").glob("*.yaml"))


@pytest.mark.parametrize("deck", _DECKS, ids=lambda p: p.stem)
def test_committed_decks_validate_clean(deck):
    validate_wb_campaign_yaml(yaml.safe_load(deck.read_text()))


def test_decks_found():
    assert len(_DECKS) >= 3


def test_top_level_typo_rejected_with_hint():
    with pytest.raises(SystemExit, match=r"learning_rate.*did you mean 'lr'"):
        validate_wb_campaign_yaml({"learning_rate": 1e-3})


def test_spectral_block_typo_rejected_with_hint():
    with pytest.raises(SystemExit,
                       match=r"si_substep.*did you mean 'si_substeps'"):
        validate_wb_campaign_yaml({"spectral": {"si_substep": 3}})


def test_metadata_keys_tolerated():
    validate_wb_campaign_yaml({"grid": "gaussian", "cache_dir": "x"})


def test_loss_block_typo_rejected():
    with pytest.raises(SystemExit, match=r"w_t.*did you mean 'w_T'"):
        validate_wb_campaign_yaml({"loss": {"w_t": 1.0}})


def test_runtime_cfl_guard_is_wired_into_integrator_factory():
    """codex P2: prove _make_spectral_integrator CALLS the guard — a direct
    check_advective_cfl test would survive the call being deleted."""
    from types import SimpleNamespace

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.neural_gcm_spectral import _make_spectral_integrator

    bad = SpectralPEConfig(semi_implicit=True, si_substeps=1)
    with pytest.raises(ValueError, match="advective Courant"):
        _make_spectral_integrator(bad, SimpleNamespace(n_max=63), None,
                                  1800.0, "ssp_rk3")


def test_spectral_allowlist_matches_config_fields():
    """Drift gate (GLM review): every spectral-block allowlist entry must be a
    real SpectralPEConfig field (or the block-level n_max/dt), so a typo in
    the allowlist itself cannot silently brick or admit a deck key."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    from legoesm.training.scale_build import _WB_SPECTRAL_KEYS

    extra = _WB_SPECTRAL_KEYS - set(SpectralPEConfig._fields) - {"n_max", "dt"}
    assert not extra, f"allowlist keys that are not config fields: {extra}"
