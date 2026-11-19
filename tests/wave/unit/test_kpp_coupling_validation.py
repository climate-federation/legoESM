"""Independent monochromatic oracle for the committed coupling instrument."""

import json

import jax
import numpy as np
import pytest

from scripts.validate.wave_kpp_coupling import INPUTS, prescribed_stokes


def test_prescribed_s_l2_spectrum_matches_analytic_stokes():
    jax.config.update("jax_enable_x64", True)
    cfg = json.loads((INPUTS / "config.json").read_text())
    values = []
    for amplitude in cfg["wave_amplitudes_m"]:
        field, reference = prescribed_stokes(amplitude, cfg)
        np.testing.assert_allclose(field, reference["analytic_surface_mps"], rtol=1e-12, atol=0)
        assert reference["paper_first_cell_mps"] < reference["deep_water_mps"]
        values.append(float(np.asarray(field)[0, 0, 0]))
    np.testing.assert_allclose(
        np.array(values) / np.array(cfg["wave_amplitudes_m"]) ** 2,
        values[0] / cfg["wave_amplitudes_m"][0] ** 2,
        rtol=1e-12,
    )
    # Planted quadrature/amplitude mistake must fail this independent oracle.
    assert not np.isclose(2 * values[0], values[0], rtol=1e-12)
    zero, _ = prescribed_stokes(0.0, cfg)
    np.testing.assert_array_equal(zero, np.zeros((1, 8, 4)))


def test_source_manifest_rejects_mixed_or_modified_modules(tmp_path):
    from types import SimpleNamespace

    from scripts.validate import wave_kpp_coupling as instrument

    relative = "packages/ocean/legoesm/ocean/physics/vertical_mixing/kpp.py"
    source_file = tmp_path / relative
    source_file.parent.mkdir(parents=True)
    source_file.write_text(instrument._git("show", f"HEAD:{relative}") + "\n")
    modules = {
        "legoesm.ocean.physics.vertical_mixing.kpp": SimpleNamespace(__file__=str(source_file))
    }
    manifest = instrument._checked_module_manifest(tmp_path, "HEAD", modules=modules)
    assert relative in manifest
    source_file.write_text(source_file.read_text() + "# planted source modification\n")
    with pytest.raises(RuntimeError, match="pinned git tree"):
        instrument._checked_module_manifest(tmp_path, "HEAD", modules=modules)
    with pytest.raises(RuntimeError, match="mixed source import"):
        instrument._checked_module_manifest(instrument.ROOT, "HEAD", modules=modules)


def test_instrument_gate_rejects_untracked_and_uncommitted_files(monkeypatch):
    import subprocess

    from scripts.validate import wave_kpp_coupling as instrument

    def untracked(*args):
        if args[0] == "ls-files":
            raise subprocess.CalledProcessError(1, args)
        return ""

    monkeypatch.setattr(instrument, "_git", untracked)
    with pytest.raises(subprocess.CalledProcessError):
        instrument._require_committed_instrument()
    monkeypatch.setattr(instrument, "_git", lambda *args: "0" * 40)
    with pytest.raises(RuntimeError, match="not committed at HEAD"):
        instrument._require_committed_instrument()


@pytest.mark.parametrize("first_status", [1, 2])
def test_failed_cases_are_retained_without_hiding_fatal_errors(tmp_path, monkeypatch, first_status):
    from types import SimpleNamespace

    from scripts.validate import wave_kpp_coupling as instrument

    calls, plots = [], []
    monkeypatch.setattr(instrument, "_git", lambda *args: "")
    monkeypatch.setattr(instrument, "_require_committed_instrument", lambda: None)

    def worker(command, **kwargs):
        out = tmp_path / "results"
        label = command[command.index("--label") + 1]
        amplitude = float(command[command.index("--amplitude") + 1])
        stem = f"{label}_{amplitude:g}"
        for suffix in ("npz", "json"):
            (out / f"{stem}.{suffix}").touch()
        calls.append(stem)
        return SimpleNamespace(returncode=first_status if label == "A" else 0)

    monkeypatch.setattr(instrument.subprocess, "run", worker)
    monkeypatch.setattr(instrument, "plot_results", lambda out: plots.append(out))
    message = "strict budget" if first_status == 1 else "fatal worker/source"
    with pytest.raises(RuntimeError, match=message):
        instrument.run_suite(tmp_path / "results", tmp_path / "upstream")
    assert len(calls) == 5  # collecting all outcomes never changes the gate
    assert len(plots) == (1 if first_status == 1 else 0)
