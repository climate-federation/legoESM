"""Regression test for the iter-71 ``scripts/run/run_rce.py`` CLI input
validation (mirrors iter-67/70 plane-CRM driver validation).

iter-71 added guards rejecting bad numeric inputs with clean
SystemExit + exit 1, instead of:
* ``--days -1`` → silent "Complete: 0.0s wall time" zero-row run.
* ``--resolution 0`` → deep grid-creation crash.
* ``--sst-init nan`` → NaN propagation through all column ICs.

This test parametric-asserts each guard produces:
* non-zero exit code
* concise ``error: <flag> rejected: ...`` marker
* no raw Python traceback (catches a future regression that
  re-introduces the raw exception path)

Wall: ~10 s per case (lightweight hydrostatic driver, no JAX MPI
init). ~100 s total for ~10 cases.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
DRIVER = REPO_ROOT / "scripts" / "run" / "run_rce.py"


@pytest.mark.parametrize("flag,value,expected_err", [
    ("--days", "-1", "must be positive integer"),
    ("--days", "0", "must be positive integer"),
    ("--resolution", "0", "must be positive integer"),
    ("--nlev", "-1", "must be positive integer"),
    ("--diag-days", "-5", "must be positive integer"),
    ("--dt", "nan", "must be finite"),
    ("--dt", "0.0", "must be positive"),
    ("--dt", "-1.0", "must be positive"),
    ("--sst-init", "nan", "must be finite"),
    # iter-74 Codex HIGH coverage: --sst-init negative was missed.
    ("--sst-init", "-1.0", "must be positive Kelvin"),
    ("--sst-init", "0.0", "must be positive Kelvin"),
    ("--truncation", "0", "must be positive integer"),
    ("--rce-cos-zenith", "0", "must be in (0, 1]"),
    ("--rce-cos-zenith", "1.5", "must be in (0, 1]"),
    ("--convective-detrainment-frac", "1.5", "must be in [0, 1]"),
    ("--convective-detrainment-frac", "-0.1", "must be in [0, 1]"),
    # iter-75 post-derivation guard: huge dt / tiny days → n_steps=0.
    ("--dt", "1e10", "n_steps=0"),
])
def test_run_rce_rejects_bad_cli_args(tmp_path, flag, value, expected_err):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    cmd = [
        sys.executable, str(DRIVER),
        "--grid-type", "cubed_sphere",
        "--discretization", "cdgrid",
        "--resolution", "12",
        "--days", "1",
        "--nlev", "20",
        flag, value,
        "--output", str(tmp_path / f"rce_bad_{flag.strip('-')}"),
    ]
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0, (
        f"run_rce.py exited 0 with {flag}={value}; expected non-zero.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    combined = result.stdout + result.stderr
    assert "rejected" in combined and expected_err in combined, (
        f"run_rce.py exit message missing 'rejected' + "
        f"'{expected_err}'.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
    assert "Traceback" not in combined, (
        f"run_rce.py emitted Python traceback for {flag}={value}; "
        f"iter-71 contract: clean SystemExit.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )


def test_gaussian_grid_auto_enables_x64(tmp_path):
    """A spectral grid must NOT hard-crash when the caller forgot x64: the
    driver auto-enables float64 (the transforms require it) and runs. Env
    explicitly sets JAX_ENABLE_X64=0 to prove the driver overrides it."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"          # caller forgot / disabled x64
    cmd = [
        sys.executable, str(DRIVER),
        "--grid-type", "gaussian", "--truncation", "10",
        "--days", "1", "--diag-days", "1", "--nlev", "20",
        "--output", str(tmp_path / "rce_gauss_autox64"),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=180)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"gaussian run_rce.py should auto-enable x64 and succeed, got exit "
        f"{result.returncode}.\nstdout: {result.stdout[-800:]}\n"
        f"stderr: {result.stderr[-800:]}")
    assert "auto-enabled JAX_ENABLE_X64" in combined, \
        "missing the auto-enable notice (coercion must be non-silent)"
    assert "Traceback" not in combined and "require JAX_ENABLE_X64" not in combined, (
        "the 'requires JAX_ENABLE_X64=True' RuntimeError leaked — the "
        f"auto-enable did not fire in time.\nstderr: {result.stderr[-800:]}")


def test_nonspectral_grid_keeps_caller_precision(tmp_path):
    """A non-spectral grid must NOT trip the auto-enable (stays float32 when the
    caller runs float32) — the sniff is grid-specific, not a blanket override."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"
    cmd = [
        sys.executable, str(DRIVER),
        "--grid-type", "cubed_sphere", "--discretization", "cdgrid",
        "--resolution", "12", "--days", "1", "--diag-days", "1", "--nlev", "20",
        "--output", str(tmp_path / "rce_cs_f32"),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=180)
    assert result.returncode == 0, (
        f"cubed_sphere f32 run failed: exit {result.returncode}\n"
        f"stdout: {result.stdout[-800:]}\nstderr: {result.stderr[-800:]}")
    assert "auto-enabled JAX_ENABLE_X64" not in (result.stdout + result.stderr), \
        "non-spectral grid wrongly triggered the spectral x64 auto-enable"


def test_rrtmgp_morrison_runs(tmp_path):
    """RRTMGP radiation + Morrison microphysics wiring runs end-to-end (finite,
    non-zero exit only on real failure). Small/short — this is a smoke of the
    branch selection + the physics_step/micro_step signatures, not a science run."""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "0"
    cmd = [
        sys.executable, str(DRIVER),
        "--grid-type", "cubed_sphere", "--discretization", "cdgrid",
        "--resolution", "8", "--days", "1", "--diag-days", "1", "--nlev", "20",
        "--radiation", "rrtmgp", "--microphysics", "morrison",
        "--cloud-fraction-scheme", "sundqvist",
        "--output", str(tmp_path / "rce_rrtmgp_morrison"),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            timeout=600)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        f"rrtmgp+morrison run failed: exit {result.returncode}\n"
        f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}")
    assert "RRTMGP radiation" in combined and "morrison" in combined, \
        "run header did not report RRTMGP radiation + morrison microphysics"
    # The subgrid cloud-fraction scheme (Sundqvist / Xu-Randall) feeds RRTMGP so
    # a coarse subsaturated column still carries radiatively-active cloud; the
    # flag must round-trip and be honored (reported) in the run header.
    assert "cloud fraction: sundqvist" in combined, \
        "run header did not report the wired --cloud-fraction-scheme"
    assert "Traceback" not in combined and "nan" not in result.stdout.lower(), \
        f"rrtmgp+morrison produced a traceback / NaN.\nstdout: {result.stdout[-1000:]}"


def test_rrtmgp_morrison_clouds_are_radiatively_active():
    """The subgrid cloud fraction must actually REACH the RRTMGP optics.  The
    solver gates cloud optics on ``RRTMGPConfig.include_clouds`` (defaults
    ``False``), so run_rce sets ``include_clouds=USE_MORRISON``.  Without it the
    cloud path/r_eff kwargs are silently dropped and morrison RCE runs clear-sky
    regardless of the cloud fraction — a silent no-op the header/smoke test above
    cannot catch.  This reproduces the run_rce cloud path (Sundqvist cf +
    condensate floor -> to_rrtmg_kwargs) and asserts a real cloud radiative
    effect vs clear-sky."""
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.radiation.rrtmgp_radiation import (
        rrtmgp_radiation)
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    from legoesm.atmosphere.physics.clouds.cloud_fraction import (
        compute_cloud_properties, CloudConfig)
    from legoesm.thermo import saturation_mixing_ratio

    nlev = 20
    p_half = jnp.linspace(1.0e5, 1.0e3, nlev + 1)[None, :]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = jnp.linspace(300.0, 220.0, nlev)[None, :]
    q_v = 0.92 * saturation_mixing_ratio(T, p_full)   # high RH, subsaturated
    q_c = jnp.full_like(T, 1.0e-9)                     # resolved condensate ~ 0
    dp = p_half[:, :-1] - p_half[:, 1:]                # positive
    mu0 = jnp.full((1,), 0.42)
    cloud_props = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp,
        config=CloudConfig(scheme="sundqvist"), q_cloud=q_c)
    # Sundqvist diagnoses cloud from RH even though the grid-mean q_c evaporated.
    assert float(cloud_props.lwp.sum()) > 0.0, \
        "sundqvist produced no in-cloud condensate for the optics"
    kw = dict(T=T, p_full=p_full, p_half=p_half,
              sfc_temperature=jnp.full((1,), 302.0), q_v=q_v, cos_zenith=mu0)
    cloud_kw = cloud_props.to_rrtmg_kwargs()
    # include_clouds=True: clouds active.  include_clouds=False: the SAME cloud
    # kwargs are gated off -> a true clear-sky call.  The difference is the cloud
    # radiative effect (a saturated column should trap tens of W/m^2 of LW).
    rad_cloud = rrtmgp_radiation(
        config=RRTMGPConfig(sfc_albedo=0.07, S_0=975.0, include_clouds=True),
        **kw, **cloud_kw)
    rad_clear = rrtmgp_radiation(
        config=RRTMGPConfig(sfc_albedo=0.07, S_0=975.0, include_clouds=False),
        **kw, **cloud_kw)
    d_lw = float(jnp.abs(rad_cloud.lw_flux_up - rad_clear.lw_flux_up).max())
    assert d_lw > 1.0, (
        f"clouds had ~no LW radiative effect ({d_lw:.3f} W/m^2): the "
        f"include_clouds gate is off or the diagnosed cloud is empty")
