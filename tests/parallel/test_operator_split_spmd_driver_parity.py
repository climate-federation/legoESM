"""End-to-end parity of the run_amip lat-band-SPMD OPERATOR-SPLIT lane
(``ModelDriver._run_operator_split_spmd``) against the serial compiled-segment
path (``ModelDriver._run_compiled``).

Both drivers integrate the SAME tiny gray-radiation + TKE-turbulence AMIP config
(analytical forcing, lat-lon C-grid) from an identical initial state; one runs
the serial fused operator-split scan, the other the lat-band-sharded operator
split over a 1-D ``"lat"`` mesh (2 devices).  Because the operator-split physics
is PURELY column-local (decomposition-invariant) and the ONLY collectives are
the mass/moisture fixers (lat-band-psum-aware) + the moisture halo, the two
trajectories agree to the limited-FV-PPM cut-boundary residual the sharded
dynamics carries at band interfaces — NOT an O(1) split-wiring bug.  This is the
driver-wiring gate: it exercises the real ``build_physics_pipeline`` step_unified
(built on a band grid), the carry seed/shard/thread, the per-segment forcing, and
the gather — the pieces the shape-agnostic ``test_atm_latlon_operator_split_spmd``
mock cannot.

4 host CPU devices via ``XLA_FLAGS=--xla_force_host_platform_device_count=4``;
``JAX_ENABLE_X64=1``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from legoesm.driver.config import (
    ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
)
from legoesm.driver.model_driver import ModelDriver

N_DEV = 2
DT = 100.0
N_STEPS = 3
RES = 8          # n_lat = 8 (divisible by N_DEV=2 -> 4 rows/band), n_lon = 16
NLEV = 4


def _make_cfg(out_dir, *, spmd):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=RES, nlev=NLEV),
        dycore=DycoreConfig(
            model_type="hydrostatic", discretization="latlon_cgrid", dt=DT,
            fix_mass=True),
        output=OutputConfig(output_dir=out_dir, diag_days=0, checkpoint_days=0),
        days=N_STEPS * DT / 86400.0,
        dataset="analytical",
        radiation="gray",
        turbulence="tke",
        convection="none",
        microphysics="none",
        gravity_wave_drag="none",
        cloud_scheme="none",
        fix_moisture=True,
        enable_latlon_spmd=spmd,
        distributed=False,
        n_devices=(N_DEV if spmd else 1),
    )


def _run(out_dir, *, spmd):
    driver = ModelDriver(_make_cfg(out_dir, spmd=spmd), output_dir=out_dir)
    driver.setup()
    status = driver.run(compiled=True)
    return driver, status


def test_operator_split_spmd_driver_matches_serial(tmp_path):
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    ser, ser_status = _run(str(tmp_path / "serial"), spmd=False)
    assert ser_status == "COMPLETED", f"serial: {ser_status}"

    spmd, spmd_status = _run(str(tmp_path / "spmd"), spmd=True)
    assert spmd_status == "COMPLETED", f"spmd: {spmd_status}"

    # Report per-field drift (dynamical winds carry the band-cut residual; the
    # thermodynamic column state is physics-dominated -> tighter).
    worst = {}
    for field in ("p_s", "T", "u", "v"):
        a = np.asarray(getattr(ser.state, field).data, dtype=np.float64)
        b = np.asarray(getattr(spmd.state, field).data, dtype=np.float64)
        assert a.shape == b.shape, f"{field} shape {a.shape} vs {b.shape}"
        denom = np.maximum(np.abs(a).max(), 1e-30)
        worst[field] = float(np.abs(a - b).max() / denom)
    qa = np.asarray(ser.q_v, dtype=np.float64)
    qb = np.asarray(spmd.q_v, dtype=np.float64)
    worst["q_v"] = float(np.abs(qa - qb).max() / max(np.abs(qa).max(), 1e-30))
    print("operator-split SPMD-vs-serial relative drift:", worst)

    # The COLUMN-LOCAL thermodynamic state — p_s (mass-fixer target computed on
    # the full grid; the in-step global_area_sum is lat-band-psum-aware), T and
    # q_v — is decomposition-INVARIANT, so it matches to reduction round-off
    # (bitwise 0 on this hardware).  Only the winds carry the limited-FV-PPM
    # cut residual at the two band interfaces.  Field-specific gates: a wiring
    # bug (wrong carry field / stale GHG / global-ncol step_unified) would push
    # even the thermodynamic fields O(1).
    for field in ("p_s", "T", "q_v"):
        assert worst[field] < 1e-9, (
            f"{field} relative drift {worst[field]:.3e} — column-local physics "
            f"must be decomposition-invariant; wiring regression? {worst}")
    for field in ("u", "v"):
        assert worst[field] < 1e-5, (
            f"{field} relative drift {worst[field]:.3e} exceeds the FV-PPM "
            f"band-cut bound (~1e-7); wiring regression? {worst}")

    # Non-vacuity: the run actually integrated (state != initial, finite).
    assert np.isfinite(np.asarray(spmd.state.T.data)).all()
    assert float(np.abs(np.asarray(spmd.state.u.data)).max()) > 0.0


def test_operator_split_spmd_segment_callback_fold_back(tmp_path):
    """A per-segment callback that MUTATES driver.state must (a) fire with the
    gathered cell-centered state and (b) have its mutation folded back into the
    threaded sharded carry — matching _run_compiled, which re-packs from self.*
    each segment.  Exercised via the driver's _segment_callback hook: a sharding
    mismatch or a dropped mutation in the fold-back re-shard would raise / diverge
    here (the run would not COMPLETE)."""
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")

    driver = ModelDriver(_make_cfg(str(tmp_path / "cb"), spmd=True),
                         output_dir=str(tmp_path / "cb"))
    driver.setup()

    seen = {"calls": 0, "shapes_ok": True}

    def _cb(drv, day, dt_seg):
        # (a) the callback sees a finite, cell-centered gathered state.
        seen["calls"] += 1
        T = np.asarray(drv.state.T.data)
        seen["shapes_ok"] &= bool(np.isfinite(T).all())
        # (b) mutate the prognostic state — the fold-back must re-shard this onto
        # the carry's per-leaf sharding (a mismatch would raise inside the loop).
        drv.state = drv.state._replace(
            T=drv.state.T.replace(data=drv.state.T.data + 0.01))

    # run() sets self._segment_callback from its argument, so pass it here.
    status = driver.run(compiled=True, segment_callback=_cb)
    assert status == "COMPLETED", status
    assert seen["calls"] >= 1 and seen["shapes_ok"]
    assert np.isfinite(np.asarray(driver.state.T.data)).all()
