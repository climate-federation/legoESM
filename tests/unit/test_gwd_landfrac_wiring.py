"""E3SM GWD landfrac wiring: the coupled pipeline threads its ``f_land``
(set by the model driver next to ``subgrid_topo_stddev``) into the GWD call
for exactly the ``e3sm_cam`` scheme (the only kernel accepting
``land_frac_col``; E3SM gw_drag.F90:904-906 oro landfrac scaling).

Guards the codex-flagged phantom: an extractor reading an attribute nothing
sets.  The recorder tests below EXECUTE ``physics_step_no_rad`` with the
kernel replaced by a capturing stub, so deleting the forwarding at the GWD
call site goes red (codex round-2 MEDIUM: flag-only assertions were
vacuous).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.gravity_wave_drag import GWDOutput
from legoesm.driver.config import (
    DycoreConfig,
    ExperimentConfig,
    GridConfig,
)
from legoesm.driver.physics_pipeline import build_physics_pipeline
from legoesm.grids.latlon import create_latlon_grid
from legoesm.grids.vertical import create_sigma_coordinate

NLEV = 5
NLAT, NLON = 8, 16


def _config(gwd):
    return ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=NLAT, nlev=NLEV),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray",
        gravity_wave_drag=gwd,
    )


def _pipe(gwd):
    grid = create_latlon_grid(NLAT, NLON, dtype=jnp.float64)
    sigma = create_sigma_coordinate(NLEV)
    pipe = build_physics_pipeline(grid, sigma, _config(gwd))
    return grid, pipe


def _run_step(grid, pipe):
    """Minimal physics_step_no_rad invocation (radiation held at zero)."""
    s2 = grid.grid_lat.shape
    s3 = (*s2, NLEV)
    Tprof = 290.0 - 50.0 * np.linspace(0, 1, NLEV)[::-1]
    T = jnp.asarray(np.broadcast_to(Tprof, s3).copy())
    q_v = jnp.full(s3, 5e-3)
    u = jnp.full(s3, 12.0)
    v = jnp.zeros(s3)
    z2 = jnp.zeros(s2)
    z3 = jnp.zeros(s3)
    return pipe.physics_step_no_rad(
        T, jnp.full(s2, 1.0e5), q_v, jnp.zeros(s3), jnp.zeros(s3), None,
        u, v, jnp.full(s2, 300.0), jnp.zeros(s2),
        jnp.asarray(grid.grid_lat), 600.0,
        z3, z2, z2, z2, z2, z2,
    )


class _Recorder:
    """Stands in for the GWD kernel; captures the call kwargs."""

    def __init__(self):
        self.kwargs = None

    def __call__(self, **kw):
        self.kwargs = kw
        shape = kw["u"].shape
        zero = jnp.zeros(shape)
        return GWDOutput(du_dt=zero, dv_dt=zero, dT_dt=zero,
                         eps_gwd=jnp.zeros(shape[0]))


def test_e3sm_cam_pipeline_threads_f_land_as_land_frac_col():
    """END-TO-END: with f_land set, physics_step_no_rad delivers it to the
    e3sm_cam kernel as (ncol,) land_frac_col with the right values."""
    grid, pipe = _pipe("e3sm_cam")
    assert pipe._gwd_takes_land_frac is True
    f_land = jnp.asarray(
        np.linspace(0.0, 1.0, NLAT * NLON).reshape(grid.grid_lat.shape)
    )
    pipe.f_land = f_land
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None, "GWD kernel was never called"
    assert "land_frac_col" in rec.kwargs, (
        "pipeline did not forward f_land as land_frac_col")
    got = rec.kwargs["land_frac_col"]
    assert got.shape == (NLAT * NLON,)
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(f_land).reshape(-1))


def test_e3sm_cam_f_land_none_sends_no_kwarg():
    """f_land=None (driver never populated it) -> no land_frac_col kwarg ->
    the kernel default (no scaling) applies — legacy bit-identity."""
    grid, pipe = _pipe("e3sm_cam")
    assert pipe.f_land is None
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None
    assert "land_frac_col" not in rec.kwargs


@pytest.mark.parametrize("scheme", ["mcfarlane", "lindzen", "hines"])
def test_other_gwd_schemes_never_send_land_frac(scheme):
    """mcfarlane/lindzen/hines kernels do NOT accept ``land_frac_col`` —
    sending it would TypeError in production.  Even with f_land set, the
    pipeline must not forward it (builder flag False + recorder proof)."""
    grid, pipe = _pipe(scheme)
    assert pipe._gwd_takes_land_frac is False
    pipe.f_land = jnp.full(grid.grid_lat.shape, 0.5)
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None
    assert "land_frac_col" not in rec.kwargs


def test_none_scheme_flag_false():
    """gravity_wave_drag='none' -> no GWD at all; the flag stays False."""
    _, pipe = _pipe("none")
    assert pipe._gwd_takes_land_frac is False
    assert pipe.gwd_fn is None


# ---------------------------------------------------------------------------
# Beres convective-source threading (netdt_col; E3SM TTEND_DP analogue,
# gw_drag.F90:766-778): the pipeline passes THIS STEP's convection heating
# to the e3sm_cam kernel when its resolved source is "convective".
# ---------------------------------------------------------------------------

def _pipe_over(gwd_over, convection="sbm"):
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    grid = create_latlon_grid(NLAT, NLON, dtype=jnp.float64)
    sigma = create_sigma_coordinate(NLEV)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=NLAT, nlev=NLEV),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        radiation="gray",
        convection=convection,
        gravity_wave_drag="e3sm_cam",
        gravity_wave_drag_override=gwd_over,
    )
    cfg.validate_strict()
    return grid, build_physics_pipeline(grid, sigma, cfg)


def test_e3sm_convective_source_receives_netdt():
    """END-TO-END: with the override selecting the Beres source, the kernel
    receives netdt_col in column layout.  The convection kernel is replaced
    by a stub emitting a LEVEL- and COLUMN-DISTINCT sentinel heating, and
    the recorder asserts netdt_col equals that exact (ncol, nlev) array —
    catching a transpose, vertical reversal, or wrong-source array (codex
    netdt r1)."""
    from types import SimpleNamespace

    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        E3SMCAMConfig,
        GravityWaveDragConfig,
    )
    over = GravityWaveDragConfig(
        scheme="e3sm_cam", e3sm_cam=E3SMCAMConfig(source="convective", pgwv=8))
    grid, pipe = _pipe_over(over, convection="none")
    assert pipe._gwd_takes_netdt is True

    ncol = NLAT * NLON
    sentinel = (jnp.arange(ncol)[:, None] * 100.0
                + jnp.arange(NLEV)[None, :]) * 1e-9   # distinct per (col, lev)
    zeros3 = jnp.zeros((ncol, NLEV))

    def conv_stub(*a, **k):
        out = SimpleNamespace(
            dT_dt=sentinel, dq_v_dt=zeros3, dq_c_conv_dt=zeros3,
            dq_r_conv_dt=None, du_dt=zeros3, dv_dt=zeros3,
            precip=jnp.zeros((ncol,)), M_c=zeros3,
        )
        return out   # the convection='none' branch takes a bare return

    pipe.convection_fn = conv_stub
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None
    assert "netdt_col" in rec.kwargs, "netdt_col not threaded to the kernel"
    nd = rec.kwargs["netdt_col"]
    assert nd.shape == (ncol, NLEV)
    assert jnp.array_equal(nd, sentinel), (
        "netdt_col is not the convection heating array (transpose/reversal/"
        "wrong source)")


def test_e3sm_orographic_source_gets_no_netdt():
    """Default oro source: the flag stays False and no netdt kwarg is sent
    (the kernel accepts it but only Beres consumes it — no useless
    plumbing)."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )
    grid, pipe = _pipe_over(GravityWaveDragConfig(scheme="e3sm_cam"))
    assert pipe._gwd_takes_netdt is False
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None
    assert "netdt_col" not in rec.kwargs


def test_e3sm_frontal_source_threads_frontgf_on_latlon(monkeypatch):
    """END-TO-END: with the frontal source on a SUPPORTED grid family
    (lat-lon), the builder no longer rejects — it sets ``_gwd_takes_frontgf``
    and the kernel receives ``frontgf_col`` from the frontogenesis producer.
    The producer is replaced by a sentinel emitting a LEVEL- and
    COLUMN-DISTINCT array so the assert catches a transpose / wrong-source
    array (mirrors the netdt sentinel test)."""
    from legoesm.atmosphere.physics.gravity_wave_drag import frontogenesis
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        E3SMCAMConfig,
        GravityWaveDragConfig,
    )
    over = GravityWaveDragConfig(
        scheme="e3sm_cam", e3sm_cam=E3SMCAMConfig(source="frontal", pgwv=8))
    grid, pipe = _pipe_over(over, convection="none")
    assert pipe._gwd_takes_frontgf is True

    ncol = NLAT * NLON
    sentinel = (jnp.arange(ncol)[:, None] * 10.0
                + jnp.arange(NLEV)[None, :]) * 1e-6   # distinct per (col, lev)
    seen = {}

    def fgf_stub(u, v, T, p_full, grid_arg):
        seen["grid"] = grid_arg
        return sentinel, jnp.zeros_like(sentinel)

    monkeypatch.setattr(frontogenesis, "compute_frontogenesis", fgf_stub)
    rec = _Recorder()
    pipe.gwd_fn = rec
    _run_step(grid, pipe)
    assert rec.kwargs is not None
    assert "frontgf_col" in rec.kwargs, "frontgf_col not threaded to the kernel"
    fg = rec.kwargs["frontgf_col"]
    assert fg.shape == (ncol, NLEV)
    assert jnp.array_equal(fg, sentinel), (
        "frontgf_col is not the producer's array (transpose/wrong source)")
    assert seen["grid"] is pipe._grid, "producer must receive the pipeline grid"


def test_e3sm_frontal_source_rejected_on_unsupported_grid(monkeypatch):
    """Dispatch hardening RETAINED where no producer exists: when the grid
    family is unsupported (cubed-sphere / MPAS today) the builder still
    rejects the frontal selection loudly — the kernel's frontgf_col=None ->
    zeros path would otherwise be a silent no-op."""
    from legoesm.atmosphere.physics.gravity_wave_drag import frontogenesis
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        E3SMCAMConfig,
        GravityWaveDragConfig,
    )
    monkeypatch.setattr(
        frontogenesis, "frontogenesis_supported", lambda g: False)
    over = GravityWaveDragConfig(
        scheme="e3sm_cam", e3sm_cam=E3SMCAMConfig(source="frontal", pgwv=8))
    with pytest.raises(ValueError, match="frontal.*not.*wired|FRONTGF"):
        _pipe_over(over, convection="none")


def test_e3sm_nonfrontal_sources_never_send_frontgf():
    """Orographic / convective sources: the flag stays False and no
    frontgf kwarg is sent (no useless per-step producer calls)."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        E3SMCAMConfig,
        GravityWaveDragConfig,
    )
    for src in ("orographic", "convective"):
        grid, pipe = _pipe_over(GravityWaveDragConfig(
            scheme="e3sm_cam", e3sm_cam=E3SMCAMConfig(source=src, pgwv=8)))
        assert pipe._gwd_takes_frontgf is False
        rec = _Recorder()
        pipe.gwd_fn = rec
        _run_step(grid, pipe)
        assert rec.kwargs is not None
        assert "frontgf_col" not in rec.kwargs
