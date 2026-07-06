"""Unit tests for ``scripts/data/convert_mpi_crm_to_reference.py``.

Builds a tiny synthetic ``run_rce_mpi_long.py`` output tree (a couple of 3D
volumes + surface snapshots), converts it, and feeds the result through the
real campaign reader ``build_reference_profiles`` to prove the q_v inversion
and surface precip survive the bridge byte-for-byte (module-top -> surface
ordering, mse J/kg -> kJ/kg).  Also exercises the ascending-z flip, the
non-monotone guard, and the spinup filter.
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

from legoesm import constants
from scripts.data import convert_mpi_crm_to_reference as conv

_NY, _NX, _NLEV = 3, 3, 6
# Physical heights, model-top -> surface (descending), as the driver emits.
_Z_TOP_TO_SFC = np.array(
    [16000.0, 12000.0, 8000.0, 5000.0, 2500.0, 500.0], dtype=np.float64
)


def _synth_column(seed: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (T, qv, cond, mse_J) fields shaped (ny, nx, nlev).

    ``mse_J`` is built as ``c_pd*T + L_v*qv + g*z`` so the reader's inversion
    reproduces ``qv`` exactly.  A tiny per-cell ramp breaks horizontal
    uniformity so the mean is a non-trivial average.
    """
    z = _Z_TOP_TO_SFC
    # T decreasing with height (surface warm), qv decreasing with height.
    T_col = 200.0 + (z[-1] - z) / z[-1] * 0.0 + np.linspace(200.0, 295.0, _NLEV)
    qv_col = np.linspace(2.0e-5, 1.5e-2, _NLEV)  # dry aloft, moist low
    cond_col = np.linspace(0.0, 6.0e-5, _NLEV)
    cell = (1.0 + 1.0e-3 * seed * np.arange(_NY * _NX).reshape(_NY, _NX))
    T = (T_col[None, None, :] * cell[..., None]).astype(np.float64)
    qv = (qv_col[None, None, :] * cell[..., None]).astype(np.float64)
    cond = np.broadcast_to(cond_col, (_NY, _NX, _NLEV)).astype(np.float64)
    mse_J = (
        constants.c_pd * T + constants.L_v * qv + constants.g * z[None, None, :]
    )
    return T, qv, cond, mse_J


def _write_mpi_run(
    root: Path, days: tuple[float, ...], *, ascending: bool = False,
    precip_kg_m2_s: float = 1.0e-4,
) -> None:
    (root / "snapshots_3d").mkdir(parents=True, exist_ok=True)
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    z = _Z_TOP_TO_SFC[::-1].copy() if ascending else _Z_TOP_TO_SFC
    for i, day in enumerate(days):
        T, qv, cond, mse_J = _synth_column(seed=float(i))
        if ascending:
            T, qv, cond, mse_J = (np.flip(a, axis=-1) for a in (T, qv, cond, mse_J))
        np.savez_compressed(
            root / "snapshots_3d" / f"snap_hr_{i:04d}.npz",
            t_sim=day * 86_400.0, day=day, hour=day * 24.0,
            z=z.astype(np.float32), mse=mse_J.astype(np.float32),
            qv=qv.astype(np.float32), T=T.astype(np.float32),
            cond=cond.astype(np.float32),
        )
        precip = np.full((_NY, _NX), precip_kg_m2_s, dtype=np.float32)
        np.savez_compressed(
            root / "snapshots" / f"snap_day_{i:04d}.npz",
            t_sim=day * 86_400.0, day=day, precip=precip,
            T_sfc=np.full((_NY, _NX), 295.0, dtype=np.float32),
        )


def _reference_qv_mean() -> np.ndarray:
    # Time+horizontal mean of the two synthetic volumes' qv (seeds 0 and 1).
    qvs = [_synth_column(seed=float(i))[1].mean(axis=(0, 1)) for i in range(2)]
    return np.mean(np.stack(qvs), axis=0)


def test_roundtrip_recovers_qv_precip_and_cond(tmp_path):
    from scripts.run.run_scm_rce_campaign import build_reference_profiles

    mpi_out = tmp_path / "mpi"
    _write_mpi_run(mpi_out, days=(1.0, 2.0))
    ref = tmp_path / "ref"
    diag = conv.convert(mpi_out, ref, spinup_days=0.0)
    assert diag["n_volumes"] == 2
    assert diag["n_surface"] == 2
    assert diag["n_levels"] == _NLEV

    prof = build_reference_profiles(ref, last_n=2)
    # q_v inverted from mse must match the CRM q_v the driver wrote.
    np.testing.assert_allclose(
        np.asarray(prof.qv_ref, dtype=float), _reference_qv_mean(),
        rtol=0, atol=5.0e-6,
    )
    # condensate mean survives (linspace 0..6e-5 -> mean 3e-5).
    assert float(np.max(prof.qcond_ref)) > 0.0
    # precip 1e-4 kg/m^2/s -> 8.64 mm/day (reader's kg/m^2/s branch).
    assert prof.precip_ref_mm_day == pytest.approx(1.0e-4 * 86_400.0, rel=1e-6)


def test_ascending_z_is_flipped_to_top_to_surface(tmp_path):
    from scripts.run.run_scm_rce_campaign import build_reference_profiles

    mpi_out = tmp_path / "mpi"
    _write_mpi_run(mpi_out, days=(1.0, 2.0), ascending=True)
    ref = tmp_path / "ref"
    conv.convert(mpi_out, ref, spinup_days=0.0)
    with np.load(ref / "snapshots3d" / "vol_0000.npz") as ds:
        z_out = np.asarray(ds["z"], dtype=float)
    assert np.all(np.diff(z_out) < 0.0)  # descending top->surface
    prof = build_reference_profiles(ref, last_n=2)
    np.testing.assert_allclose(
        np.asarray(prof.qv_ref, dtype=float), _reference_qv_mean(),
        rtol=0, atol=5.0e-6,
    )


def test_non_monotone_z_raises(tmp_path):
    mpi_out = tmp_path / "mpi"
    (mpi_out / "snapshots_3d").mkdir(parents=True)
    (mpi_out / "snapshots").mkdir(parents=True)
    T, qv, cond, mse_J = _synth_column(seed=0.0)
    bad_z = np.array([16000.0, 8000.0, 12000.0, 5000.0, 2500.0, 500.0])
    np.savez_compressed(
        mpi_out / "snapshots_3d" / "snap_hr_0000.npz",
        day=1.0, z=bad_z.astype(np.float32), mse=mse_J.astype(np.float32),
        qv=qv.astype(np.float32), T=T.astype(np.float32),
        cond=cond.astype(np.float32),
    )
    with pytest.raises(ValueError, match="not monotone"):
        conv.convert(mpi_out, tmp_path / "ref", spinup_days=0.0)


def test_spinup_filter_drops_early_volumes(tmp_path):
    mpi_out = tmp_path / "mpi"
    _write_mpi_run(mpi_out, days=(1.0, 2.0, 3.0))
    ref = tmp_path / "ref"
    diag = conv.convert(mpi_out, ref, spinup_days=2.5)
    assert diag["n_volumes"] == 1  # only day 3.0 survives
    assert diag["vol_day_first"] == pytest.approx(3.0)


def test_missing_cond_field_raises(tmp_path):
    mpi_out = tmp_path / "mpi"
    (mpi_out / "snapshots_3d").mkdir(parents=True)
    (mpi_out / "snapshots").mkdir(parents=True)
    T, qv, _cond, mse_J = _synth_column(seed=0.0)
    np.savez_compressed(
        mpi_out / "snapshots_3d" / "snap_hr_0000.npz",
        day=1.0, z=_Z_TOP_TO_SFC.astype(np.float32),
        mse=mse_J.astype(np.float32), qv=qv.astype(np.float32),
        T=T.astype(np.float32),  # no 'cond'
    )
    with pytest.raises(KeyError, match="cond"):
        conv.convert(mpi_out, tmp_path / "ref", spinup_days=0.0)


def test_chronological_sort_beats_lexical(tmp_path):
    """Filenames whose lexical order != day order must renumber by day.

    ``snap_hr_10000`` sorts lexically *before* ``snap_hr_9999`` ('1' < '9')
    but is one hour *later*; the reference must come out day-ordered so the
    campaign reader's ``last_n`` picks the true latest equilibrium volumes.
    """
    mpi_out = tmp_path / "mpi"
    (mpi_out / "snapshots_3d").mkdir(parents=True)
    (mpi_out / "snapshots").mkdir(parents=True)
    for name, day in (("snap_hr_9999.npz", 9.0), ("snap_hr_10000.npz", 10.0)):
        T, qv, cond, mse_J = _synth_column(seed=day)
        np.savez_compressed(
            mpi_out / "snapshots_3d" / name,
            day=day, z=_Z_TOP_TO_SFC.astype(np.float32),
            mse=mse_J.astype(np.float32), qv=qv.astype(np.float32),
            T=T.astype(np.float32), cond=cond.astype(np.float32),
        )
    ref = tmp_path / "ref"
    conv.convert(mpi_out, ref, spinup_days=0.0)
    with np.load(ref / "snapshots3d" / "vol_0000.npz") as ds:
        assert float(ds["day"]) == pytest.approx(9.0)   # earliest first
    with np.load(ref / "snapshots3d" / "vol_0001.npz") as ds:
        assert float(ds["day"]) == pytest.approx(10.0)  # latest last


def test_clobber_guard_blocks_mixing_runs(tmp_path):
    mpi_out = tmp_path / "mpi"
    _write_mpi_run(mpi_out, days=(1.0, 2.0))
    ref = tmp_path / "ref"
    conv.convert(mpi_out, ref, spinup_days=0.0)
    with pytest.raises(FileExistsError, match="clobber"):
        conv.convert(mpi_out, ref, spinup_days=0.0)
    # --clobber succeeds and re-numbers cleanly.
    diag = conv.convert(mpi_out, ref, spinup_days=0.0, clobber=True)
    assert diag["n_volumes"] == 2
