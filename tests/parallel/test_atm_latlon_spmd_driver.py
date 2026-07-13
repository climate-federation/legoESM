"""End-to-end of the ModelDriver lat-band SPMD run method
(``_run_compiled_latlon_spmd``), exercised through a lightweight driver stub so
no data-loading ModelDriver.setup() is needed: the method reads only
config / model / grid / state / _segment_callback. Validates the production glue
(mesh build + run_atm_latlon_spmd + on_segment + self.state update + status)
against a direct run_atm_latlon_spmd reference. 4 host CPU devices via
``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from legoesm import constants
from legoesm.driver.config import ExperimentConfig
from legoesm.driver.model_driver import ModelDriver
from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
    CGridLatLonPrimitiveEquationConfig,
    CGridLatLonPrimitiveEquationModel,
)
from legoesm.atmosphere.dynamics.sharded_atm_latlon_step import (
    run_atm_latlon_spmd,
)
from legoesm.atmosphere.held_suarez import (
    held_suarez_forcing_latlon, held_suarez_init_latlon)
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

N_DEV, N_LAT, N_LON, NLEV = 4, 16, 16, 4
DT = 100.0


class _DriverStub:
    """Minimal stand-in exposing exactly what _run_compiled_latlon_spmd reads,
    with the three real ModelDriver methods bound onto it (so ``self.*`` resolves
    without a full data-loading ModelDriver construction)."""
    _latlon_spmd_mesh = ModelDriver._latlon_spmd_mesh
    _latlon_spmd_physics_fn = ModelDriver._latlon_spmd_physics_fn
    _operator_split_spmd_active = ModelDriver._operator_split_spmd_active
    _run_compiled_latlon_spmd = ModelDriver._run_compiled_latlon_spmd

    def __init__(self, model, state, cfg):
        self.config = cfg
        self.model = model
        self.grid = model.grid
        self.state = state
        self._segment_callback = None
        self._current_day = 0.0


def _model_and_hs():
    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON, radius=constants.R_earth,
                              omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=NLEV)
    cfg = CGridLatLonPrimitiveEquationConfig(
        fix_mass=True, use_polar_filter=False, use_ppm_transport=True,
        time_integrator="ssp_rk3")
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, cfg)
    hs0 = held_suarez_init_latlon(grid, sigma)
    return model, hs0


def _exp_cfg(n_steps, **over):
    cfg = ExperimentConfig()
    fields = dict(
        enable_latlon_spmd=True, held_suarez_forcing=True, distributed=False,
        n_devices=N_DEV, days=n_steps * DT / 86400.0,
        radiation="none", convection="none", turbulence="none",
        microphysics="none", gravity_wave_drag="none", cloud_scheme="none",
        grid=cfg.grid._replace(grid_type="latlon"),
        dycore=cfg.dycore._replace(dt=DT),
        output=cfg.output._replace(diag_days=0, checkpoint_days=0),
    )
    fields.update(over)
    return cfg._replace(**fields)


def _require_devices():
    if len(jax.devices()) < N_DEV:
        pytest.skip(f"needs --xla_force_host_platform_device_count={N_DEV}")


def test_run_compiled_latlon_spmd_completes_and_matches_reference():
    """The driver method runs a Held-Suarez lat-lon SPMD integration to
    COMPLETED and yields the SAME final state as a direct run_atm_latlon_spmd
    (same model/IC/physics) — the driver glue adds no numerical drift."""
    _require_devices()
    model, hs0 = _model_and_hs()
    # days rounds to exactly 4 steps; the method recomputes n_steps from days.
    cfg = _exp_cfg(n_steps=4)
    n_steps = int(cfg.days * 86400.0 / DT)
    assert n_steps == 4

    stub = _DriverStub(model, hs0, cfg)
    status = stub._run_compiled_latlon_spmd()
    assert status == "COMPLETED", status
    assert bool(np.isfinite(np.asarray(stub.state.T.data)).all())

    # Reference: the validated runner directly, fresh model (cache-independent).
    ref_model, _ = _model_and_hs()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]), axis_names=("lat",))
    hs_ref, st_ref = run_atm_latlon_spmd(
        ref_model, mesh, hs0, DT, n_steps,
        segment_steps=max(1, int(86400.0 / DT)),
        physics_fn=held_suarez_forcing_latlon)
    assert st_ref == "COMPLETED"
    for field in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(stub.state, field).data),
            np.asarray(getattr(hs_ref, field).data),
            rtol=1e-10, atol=1e-12,
            err_msg=f"driver method diverged from run_atm_latlon_spmd in {field}")


def test_run_compiled_latlon_spmd_rejects_diag_checkpoint():
    """Dispatch-hardening: the diagnostics/checkpoint writers are a follow-up;
    diag_days/checkpoint_days > 0 must raise, not silently skip output."""
    model, hs0 = _model_and_hs()
    cfg = _exp_cfg(n_steps=1)
    cfg = cfg._replace(output=cfg.output._replace(diag_days=1))
    stub = _DriverStub(model, hs0, cfg)
    with pytest.raises(NotImplementedError, match="diagnostics"):
        stub._run_compiled_latlon_spmd()


def test_run_compiled_latlon_spmd_operator_split_refuses_compiled_segments():
    """Dispatch-hardening (M2b CLI): the operator-split unified-physics SPMD
    lane has no compiled-scan segments — a set ``latlon_spmd_compiled_segments``
    must raise loudly there, never silently step per-step (the guard sits in
    ``_run_compiled_latlon_spmd`` BEFORE the operator-split dispatch)."""
    _require_devices()
    model, hs0 = _model_and_hs()
    # held_suarez_forcing=False + one non-'none' scheme => operator-split lane.
    cfg = _exp_cfg(n_steps=1, held_suarez_forcing=False, turbulence="louis",
                   latlon_spmd_compiled_segments=True)
    stub = _DriverStub(model, hs0, cfg)
    with pytest.raises(NotImplementedError, match="operator-split"):
        stub._run_compiled_latlon_spmd()


def test_run_compiled_latlon_spmd_passes_compiled_segments_flag(monkeypatch):
    """M2b CLI wiring: ``config.latlon_spmd_compiled_segments=True`` reaches
    ``run_atm_latlon_spmd`` as ``compiled_segments=True`` (spied kwarg), the
    run COMPLETEs, and the final state matches a DIRECT
    ``run_atm_latlon_spmd(compiled_segments=True)`` reference (same-lane
    comparison, the sibling of test_run_compiled_latlon_spmd_completes_and_
    matches_reference).  Deliberately NOT compared against the flag-OFF lane
    here: scan-vs-loop parity is owned by
    tests/parallel/test_atm_latlon_segment.py at its calibrated near-rest
    config — on this Held-Suarez JET IC the PPM limiters amplify the
    compiled scan's last-bit compilation-order roundoff to ~1e-6, which is a
    property of the lanes, not of the driver wiring under test."""
    _require_devices()
    import legoesm.atmosphere.dynamics.sharded_atm_latlon_step as sas

    n_steps = 4
    # driver run with the flag ON + a spy asserting the kwarg flows through
    seen = {}
    real = sas.run_atm_latlon_spmd

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(sas, "run_atm_latlon_spmd", _spy)
    model_on, hs0 = _model_and_hs()
    stub_on = _DriverStub(
        model_on, hs0,
        _exp_cfg(n_steps=n_steps, latlon_spmd_compiled_segments=True))
    assert stub_on._run_compiled_latlon_spmd() == "COMPLETED"
    assert seen.get("compiled_segments") is True
    assert bool(np.isfinite(np.asarray(stub_on.state.T.data)).all())

    # Reference: the validated runner directly on the SAME lane (fresh model,
    # cache-independent), same segment cadence as the driver method.
    ref_model, _ = _model_and_hs()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:N_DEV]),
                             axis_names=("lat",))
    hs_ref, st_ref = run_atm_latlon_spmd(
        ref_model, mesh, hs0, DT, n_steps,
        segment_steps=max(1, int(86400.0 / DT)),
        physics_fn=held_suarez_forcing_latlon,
        compiled_segments=True)
    assert st_ref == "COMPLETED"
    for field in ("u", "v", "T", "p_s"):
        np.testing.assert_allclose(
            np.asarray(getattr(stub_on.state, field).data),
            np.asarray(getattr(hs_ref, field).data),
            rtol=1e-10, atol=1e-12,
            err_msg=(f"compiled-segments driver run diverged from the "
                     f"direct compiled-segments reference in {field}"))
