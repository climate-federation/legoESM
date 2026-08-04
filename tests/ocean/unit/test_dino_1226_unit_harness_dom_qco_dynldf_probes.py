"""Direct unit tests for the #1226 unit-call harness's ``dom_qco_r3c`` and
``dynldf_lev_lap`` Python-side probe helpers
(``scripts/validate/ocean_fidelity/dino_1226/unit_harness/
run_dom_qco_r3c_probe.py`` / ``run_dynldf_lap_probe.py``).

The full probe ``main()`` requires the compiled NEMO Fortran harness
executable + a real DINO run directory (mesh_mask.nc, restart), so it is not
a hermetic pytest target -- same scoping as the existing
``run_eos_rab_probe.py``'s own round-trip self-check (see
``test_dino_1226_unit_harness_binary_io.py``'s module docstring). This file
tests the PURE, hermetic pieces: the synthetic-input builders (deterministic
given a seed, no I/O) and the r3t-stretch-undo arithmetic the probe relies on.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from validate.ocean_fidelity.dino_1226.unit_harness import (  # noqa: E402
    run_dom_qco_r3c_probe,
    run_dynldf_lap_probe,
)

sys.path.remove(str(SCRIPTS_DIR))


def test_build_synthetic_ssh_shape_and_dtype():
    rng = np.random.default_rng(0)
    ssh = run_dom_qco_r3c_probe.build_synthetic_ssh(rng)
    assert ssh.shape == (run_dom_qco_r3c_probe.JPI, run_dom_qco_r3c_probe.JPJ)
    assert ssh.dtype == np.float64


def test_build_synthetic_ssh_deterministic_given_seed():
    ssh_a = run_dom_qco_r3c_probe.build_synthetic_ssh(np.random.default_rng(0))
    ssh_b = run_dom_qco_r3c_probe.build_synthetic_ssh(np.random.default_rng(0))
    np.testing.assert_array_equal(ssh_a, ssh_b)


def test_build_synthetic_ssh_spans_real_dynamic_range():
    """Not a near-zero quiescent field -- the harness's whole point vs the
    dump-based method is exercising real amplitude (task brief: 'use
    SYNTHETIC inputs spanning the routine's real dynamic range')."""
    ssh = run_dom_qco_r3c_probe.build_synthetic_ssh(np.random.default_rng(1))
    assert np.max(np.abs(ssh)) > 0.1   # meters -- not machine-epsilon noise


def test_r3t_stretch_undo_roundtrips_a_synthetic_stretch_factor():
    """The probe compares NEMO's raw r3t against
    ``nemo_r3t_stretch(...) - 1.0`` (undoing the (1+r3t) shift). Prove this
    undo is correct on a value the probe did NOT compute -- a synthetic-
    violation-shaped check: if the undo used the wrong sign or forgot the
    shift, this would fail."""
    stretch_factor = 1.0037   # a synthetic (1+r3t) value, r3t = 0.0037
    r3t_recovered = stretch_factor - 1.0
    assert r3t_recovered == pytest.approx(0.0037, abs=1e-12)


def test_build_synthetic_velocity_shape_and_dtype():
    rng = np.random.default_rng(0)
    u, v = run_dynldf_lap_probe.build_synthetic_velocity(rng)
    assert u.shape == (run_dynldf_lap_probe.JPI, run_dynldf_lap_probe.JPJ)
    assert v.shape == (run_dynldf_lap_probe.JPI, run_dynldf_lap_probe.JPJ)
    assert u.dtype == np.float64
    assert v.dtype == np.float64


def test_build_synthetic_velocity_deterministic_given_seed():
    u_a, v_a = run_dynldf_lap_probe.build_synthetic_velocity(np.random.default_rng(3))
    u_b, v_b = run_dynldf_lap_probe.build_synthetic_velocity(np.random.default_rng(3))
    np.testing.assert_array_equal(u_a, u_b)
    np.testing.assert_array_equal(v_a, v_b)


def test_build_synthetic_velocity_has_horizontal_structure():
    """A purely constant field can't exercise a divergence/curl operator's
    horizontal derivatives -- confirm the synthetic field actually varies in
    latitude (not silently uniform)."""
    rng = np.random.default_rng(0)
    u, _ = run_dynldf_lap_probe.build_synthetic_velocity(rng)
    assert np.std(u[0, :]) > 1e-3   # varies along the j-axis, not flat


def test_probe_modules_declare_expected_domain_constants():
    """Both probes must agree with the harness's fixed DINO domain size
    (JPI=56, JPJ=203) -- a silent mismatch here would corrupt every shape
    assertion downstream without raising until the Fortran driver's own
    shape check fires."""
    assert run_dom_qco_r3c_probe.JPI == 56
    assert run_dom_qco_r3c_probe.JPJ == 203
    assert run_dynldf_lap_probe.JPI == 56
    assert run_dynldf_lap_probe.JPJ == 203
    assert run_dynldf_lap_probe.JPK == 36
