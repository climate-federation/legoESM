"""Unit test for ``scripts/data/build_rcemip1_small_reference.py``.

Exercises the leaf builder directly (not just via the CLI) and verifies the
bundle is readable by the campaign reference loader and round-trips the Wing
profiles exactly through the ``mse`` encoding.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from scripts.data import build_rcemip1_small_reference as builder


def test_stretched_heights_top_to_surface_monotone_and_bounded():
    z = builder.stretched_heights_top_to_surface(48, 33_000.0, 100.0)
    assert z.shape == (48,)
    assert np.all(np.diff(z) < 0.0)  # strictly decreasing (top -> surface)
    assert z[0] < 33_000.0 and z[-1] > 0.0
    # near-surface spacing is the finest
    dz = np.abs(np.diff(z))
    assert dz[-1] < dz[0]


def test_stretched_heights_rejects_bad_args():
    with pytest.raises(ValueError):
        builder.stretched_heights_top_to_surface(2, 33_000.0, 100.0)
    with pytest.raises(ValueError):
        builder.stretched_heights_top_to_surface(48, 100.0, 33_000.0)


def test_build_reference_roundtrips_wing_profiles_through_campaign_reader(tmp_path):
    import jax.numpy as jnp

    from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
        wing2018_qv_profile,
        wing2018_temperature_profile,
    )
    from scripts.run.run_scm_rce_campaign import build_reference_profiles

    outdir = tmp_path / "ref"
    diag = builder.build_reference(outdir, n_levels=48, n_volumes=3)

    # Wing ANALYTIC cold point is T_v0 - Gamma*z_t with T_v0 = T0*(1+0.608*q0)
    # = 303.4 K (Eq. 3), i.e. 202.9 K at 15 km. That is the IC, NOT the 100-day
    # equilibrium cold point (~194-198 K) the resolved runs settle to.
    assert 201.0 < diag["cold_point_T_K"] < 205.0
    assert 13.0 < diag["cold_point_z_km"] < 17.0
    # Lowest-level Wing air temperature = T_v0/(1+eps^-1 q_sfc) = T0 = 300 K.
    assert 298.0 < diag["sfc_T_K"] < 302.0
    assert diag["precip_mm_day"] == builder.RCEMIP1_SMALL_PRECIP_MM_DAY

    # Bundle is readable by the campaign loader and recovers the Wing profiles.
    ref = build_reference_profiles(outdir, last_n=3)
    z = np.asarray(ref.z_m)
    T_expected = np.asarray(wing2018_temperature_profile(jnp.asarray(z)))
    qv_expected = np.asarray(wing2018_qv_profile(jnp.asarray(z)))
    # float32 storage in the volumes sets the tolerance.
    assert np.allclose(ref.T_ref, T_expected, atol=1e-2)
    assert np.allclose(ref.qv_ref, qv_expected, atol=1e-5)
    assert np.all(ref.qcond_ref >= 0.0)
    assert float(np.max(ref.qcond_ref)) > 0.0
    assert abs(ref.precip_ref_mm_day - builder.RCEMIP1_SMALL_PRECIP_MM_DAY) < 1e-6
    # w is zero -> no CRM subsidence available.
    assert ref.crm_clear_sky_subsidence_m_s is None or np.allclose(
        ref.crm_clear_sky_subsidence_m_s, 0.0
    )

    assert (outdir / "README.md").exists()


def test_cli_writes_bundle(tmp_path):
    import subprocess

    outdir = tmp_path / "cli_ref"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "data" / "build_rcemip1_small_reference.py"),
            "--output",
            str(outdir),
            "--n-levels",
            "40",
            "--n-volumes",
            "2",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(list((outdir / "snapshots3d").glob("vol_*.npz"))) == 2
    assert len(list((outdir / "snapshots").glob("sfc_*.npz"))) == 2
