"""Guard the RCEMIP1 CRM wrapper against the silent laminar-run failure mode.

``run_rcemip_plane.py`` defaults to ``--theta-noise-amp 0.0 --seed-kind
smooth_k1``, i.e. no initial perturbation at all. A run launched that way
completes cleanly, conserves mass and exits 0 while producing a horizontally
uniform radiative-equilibrium column with no convection anywhere — a failure
with no error message (docs/dev-notes/CRM_faithful_SAM.md iter-175/176,
CONV-TRIGGER #83). These checks fail if the wrapper ever loses the seed.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

DRIVER = Path(__file__).resolve().parents[2] / "scripts" / "run" / "run_rcemip1_crm.sh"


@pytest.fixture(scope="module")
def source() -> str:
    return DRIVER.read_text()


def test_driver_exists_and_is_executable():
    assert DRIVER.is_file(), f"{DRIVER} missing"
    assert DRIVER.stat().st_mode & 0o111, f"{DRIVER} is not executable"


def test_spinup_seeds_a_nonzero_theta_perturbation(source):
    """--theta-noise-amp must be passed AND default to something > 0."""
    assert "--theta-noise-amp" in source
    m = re.search(r"SEED_AMP=\$\{SEED_AMP:-([0-9.eE+-]+)\}", source)
    assert m, "SEED_AMP default not found"
    assert float(m.group(1)) > 0.0, (
        "SEED_AMP defaults to zero — the run would stay laminar with no error")


def test_spinup_uses_band_noise_not_smooth_k1(source):
    """smooth_k1 seeds one domain-scale circulation, not a cell population."""
    assert "--seed-kind band_noise" in source
    assert "--seed-kind smooth_k1" not in source


def test_seed_kmax_is_below_nyquist_for_the_default_grid(source):
    """seed-kmax must stay well under nx/2 or the seed is grid-scale."""
    nx = int(re.search(r"NX=\$\{NX:-(\d+)\}", source).group(1))
    kmax = int(re.search(r"SEED_KMAX=\$\{SEED_KMAX:-(\d+)\}", source).group(1))
    assert kmax < nx / 2, f"seed-kmax {kmax} not below Nyquist {nx / 2}"


def test_spinup_defaults_to_rrtmgp_not_gray(source):
    """Gray radiation under-drives the circulation (w RMS ~0.07 m/s, inverted
    w'^2 — CRM_faithful_SAM.md #85), so a gray spin-up hands the production leg
    a state RRTMGP would never have equilibrated to."""
    m = re.search(r"SPINUP_RAD=\$\{SPINUP_RAD:-(\w+)\}", source)
    assert m, "SPINUP_RAD default not found"
    assert m.group(1) == "rrtmgp", f"spin-up radiation defaults to {m.group(1)}"


def test_clouds_only_paired_with_the_band_model(source):
    """--clouds is meaningless for the gray scheme, so the extra flag must be
    conditional on rrtmgp rather than passed unconditionally."""
    assert 'if [ "$SPINUP_RAD" = "rrtmgp" ]' in source
    assert "--radiation gray --clouds" not in source


def test_production_phase_emits_validator_input(source):
    """compare_rce_vs_rcemip_sam reads snapshots/profile_step_*.npz, which only
    --snapshot-every writes; the underlying driver defaults it off."""
    assert "--snapshot-every" in source
    assert "compare_rce_vs_rcemip_sam.py" in source


def test_two_moment_restart_resets_condensate(source):
    """Inheriting Kessler condensate mass with zero number concentration is
    inconsistent and blows up double-moment schemes."""
    assert "--restart-reset-condensate" in source


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_driver_is_valid_bash():
    subprocess.run(["bash", "-n", str(DRIVER)], check=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_missing_checkpoint_reports_instead_of_dying_in_the_lookup(tmp_path):
    """Regression: under ``set -e`` + ``pipefail`` an unmatched ckpt glob makes
    ls exit 1, which failed the CKPT assignment and killed the script (exit 2)
    before the explanatory message could print."""
    env = {**os.environ, "SPINUP_OUT": str(tmp_path / "nope")}
    r = subprocess.run(["bash", str(DRIVER), "production", "morrison"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1, f"expected the guard's exit 1, got {r.returncode}"
    assert "run the spinup phase first" in r.stderr


def test_grep_filters_cannot_abort_the_run(source):
    """Regression: ``... | grep -vE "$FILT" | tee`` aborts under pipefail when
    grep filters out every line (grep exits 1 on no match)."""
    for line in source.splitlines():
        if "grep -vE" in line:
            assert "|| true" in line, f"unguarded grep pipeline: {line.strip()}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
def test_unknown_phase_is_rejected():
    r = subprocess.run(["bash", str(DRIVER), "not-a-phase"],
                       capture_output=True, text=True)
    assert r.returncode == 2, r.stderr
    assert "usage" in r.stderr
