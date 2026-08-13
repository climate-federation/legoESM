"""Direct tests for the prescribed-SST AMIP inference driver's pure helpers
and for the public ``spectral_amip_rollout`` entry point's importability.

The full rollout is a GPU integration run (validated by the sbatch); here we
pin the pure config/calendar helpers and that the public rollout symbol exists
with the documented keyword contract, so a signature drift fails fast.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DRIVER = _REPO / "scripts" / "run" / "run_aimip_amip_inference.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("_amip_infer", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_date_parses_iso_and_protocol_anchors():
    # Gregorian protocol anchors: start 1978-10-01, end 2025-01-01.
    mod = _load_driver()
    import datetime as dt
    assert mod._date("1978-10-01") == dt.date(1978, 10, 1)
    assert mod._date("2025-01-01") == dt.date(2025, 1, 1)
    assert (mod._date("1979-01-01") - mod._date("1978-10-01")).days == 92  # 3-mo spin-up


def test_merged_cfg_applies_classical_overlay(tmp_path):
    mod = _load_driver()
    base = tmp_path / "base.yaml"
    base.write_text(
        "n_max: 63\nn_levels: 8\ndt: 600.0\n"
        "aimip_convection: tiedtke\naimip_radiation: rrtmgp\n"
    )
    overlay = tmp_path / "variant_classical.yaml"
    overlay.write_text("aimip_convection: edmf\naimip_microphysics: sundqvist\n")
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"base: {base}\ncfg_overrides:\n  aimip_lr: 0.001\n")
    cfg = mod._merged_cfg(suite)
    assert cfg["aimip_convection"] == "edmf"  # overlay wins over base
    assert cfg["aimip_microphysics"] == "sundqvist"
    assert cfg["aimip_lr"] == 0.001  # cfg_overrides applied
    assert cfg["nlev"] == 8  # n_levels mapped to nlev


def test_build_spec_cfg_forwards_conservation_knobs():
    """The AMIP inference lane must honour the same four conservation keys
    ``run_aimip`` honours; before this the two energy-numerics keys were
    silently ignored here (a suite pinning legacy 'upwind' still ran
    sb_centered)."""
    if importlib.util.find_spec("legoesm") is None:  # pragma: no cover
        pytest.skip("legoesm not importable in this environment")
    mod = _load_driver()
    base = dict(n_max=63, nlev=8, dt=600.0)

    pe = mod._build_spec_cfg(base).pe_config
    assert pe.fix_mass is False and pe.anchor_mass_to_initial is False
    assert pe.vertical_advection_scheme == "sb_centered"
    assert pe.frictional_heating is True

    pe_on = mod._build_spec_cfg(dict(
        base, fix_mass=True, anchor_mass_to_initial=True,
        vertical_advection_scheme="upwind", frictional_heating=False,
    )).pe_config
    assert pe_on.fix_mass is True and pe_on.anchor_mass_to_initial is True
    assert pe_on.vertical_advection_scheme == "upwind"
    assert pe_on.frictional_heating is False


def test_spectral_amip_rollout_signature():
    spec = importlib.util.find_spec(
        "legoesm.training.neural_gcm_spectral"
    )
    if spec is None:
        pytest.skip("legoesm not importable in this environment")
    mod = importlib.import_module("legoesm.training.neural_gcm_spectral")
    fn = getattr(mod, "spectral_amip_rollout", None)
    assert fn is not None, "spectral_amip_rollout must be public"
    params = inspect.signature(fn).parameters
    # keyword-only contract the driver relies on
    for kw in ("sst_col", "sizing_phys_state", "day_of_year_base",
               "seconds_offset", "rad_update_interval"):
        assert kw in params, f"missing kwarg {kw}"


def test_merged_cfg_selects_variant_overlay(tmp_path):
    """--variant column_nn/sfno_physics must merge THAT variant's overlay
    (and an unknown variant is a hard error, not a silent classical run)."""
    mod = _load_driver()
    base = tmp_path / "base.yaml"
    base.write_text("n_max: 63\nn_levels: 8\ndt: 600.0\n")
    (tmp_path / "variant_classical.yaml").write_text(
        "aimip_convection: edmf\n")
    (tmp_path / "variant_column_nn.yaml").write_text(
        "nn_hidden_dim: 123\n")
    (tmp_path / "variant_sfno_physics.yaml").write_text(
        "sfno_embed_dim: 48\n")
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"base: {base}\n")
    assert mod._merged_cfg(suite, "column_nn")["nn_hidden_dim"] == 123
    assert mod._merged_cfg(suite, "sfno_physics")["sfno_embed_dim"] == 48
    assert mod._merged_cfg(suite, "classical")["aimip_convection"] == "edmf"
    with pytest.raises(ValueError, match="variant"):
        mod._merged_cfg(suite, "bogus")


def test_nn_rollout_signature_contract():
    """The NN-variant AMIP path drives spectral_rollout(forcing_base=...) and
    physics factories that accept forcing — pin the keyword contracts."""
    spec = importlib.util.find_spec("legoesm.training.neural_gcm_spectral")
    if spec is None:
        pytest.skip("legoesm not importable in this environment")
    mod = importlib.import_module("legoesm.training.neural_gcm_spectral")
    params = inspect.signature(mod.spectral_rollout).parameters
    assert "forcing_base" in params
    assert mod.N_SFNO_FORCING_CHANNELS == 3


def test_reinit_yearly_flag_and_out_suffix():
    """--reinit-yearly (hindcast-IAV mode) must route outputs to _reinit
    files so free-run protocol CSVs are never overwritten."""
    mod = _load_driver()
    import argparse
    # the parser is built inside main(); assert the flag exists by source
    # contract: the module must reference reinit_yearly and the _reinit
    # suffix (cheap AST-free check that the wiring survives refactors).
    src = _DRIVER.read_text()
    assert "--reinit-yearly" in src
    assert 'legoesm_{args.variant}_amip{_suffix}' in src
    assert "REINIT from ERA5" in src


def test_reinit_every_months_wiring():
    """--reinit-every-months (sub-annual hindcast) + suffix wiring survives."""
    src = _DRIVER.read_text()
    assert "--reinit-every-months" in src
    assert "(day.month - 1) % _reinit_months == 0" in src
    assert '_reinit{_reinit_months}mo' in src
    # --reinit-yearly maps to the 12-month interval
    assert "_reinit_months = 12" in src
