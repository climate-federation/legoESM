"""Oracle match: longwave level temperatures and level sources.

Level temperatures (tlev): the radiation host builds them as CAM6 does
(CAM cam_cesm2_1_rel_60 ``rrtmg_state.F90`` rrtmg_state_create): log-p
interpolation between layer midpoints in the interior, the top-layer
temperature at the model top and on both interfaces of the extra layer above
it, the surface temperature at the surface (CAM6: (lwup/stebol)**0.25, equal
to T_sfc for a surface emitting stebol*T_sfc**4).

Level sources: RRTMGP's ``compute_Planck_source`` (rte-rrtmgp v1.8
mo_gas_optics_rrtmgp_kernels.F90): the surface level is
``pfrac(lowest layer) * B(tlev(sfc))``, without the halo layer's Planck
fraction.

Fixture ``oracle_fixtures/rrtmgp_lw_sfc_face.npz``: 8 RFMIP clear-sky columns
(model columns and the arrays the production entry ``RRTMGP.solve_columns``
hands to the LW solver), with tlev from RUNNING the CAM file and level sources
from RUNNING RTE+RRTMGP v1.8 (rte-rrtmgp 41c5fcd950fed09b8afe186dede266824eca7fd3,
rrtmgp-data v1.8.2 34967d68703e682305ec4da72ee22bbe1f11b198, CAM
a03b84b7c4e34f965b115686f22a043b85739e56, cime5.6.47
d8e9b894d7eb96fe96f8f8a6943b421147f94606) via
scripts/validate/oracle/rrtmgp_lw_sfc_face/{build.sh,build_cam_state.sh,make_fixture.py}.

Interior levels are compared with the solver's existing sqrt floor applied
to the oracle (max(lev_src, sqrt(floor)), exactly what the floor does to an
unfloored source).  Not covered: the model-top (lid) level, whose halo Planck
fraction is a separate audit finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import (
    monochromatic_two_stream, two_stream)

_FIX = Path(__file__).parent / "oracle_fixtures" / "rrtmgp_lw_sfc_face.npz"
_TOL = 1e-10  # user-approved oracle-match tolerance (max relative error, fp64)


class _Captured(Exception):
    pass


def _max_rel(ours, ref):
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    assert ours.shape == ref.shape, (ours.shape, ref.shape)
    return float(np.max(np.abs(ours - ref) / np.abs(ref)))


@pytest.fixture(scope="module")
def case():
    z = np.load(_FIX)
    meta = json.loads(str(z["meta"]))
    opt = RRTMGP.from_legoesm_config(RRTMGPConfig(**meta["cfg"])).optics_lib
    idx = opt.gas_optics_lw.idx_gases
    vmr = {idx[str(k)]: jnp.asarray(v) for k, v in zip(z["vmr_names"], z["vmr"])}
    return z, meta, opt, vmr


def _solver_level_temperature(z, meta, **cfg_over):
    """The level temperatures the production entry hands to the LW solver."""
    cap = {}

    def _stop(*a, **k):
        cap["tlev"] = k["level_temperature"]
        raise _Captured

    solver = RRTMGP.from_legoesm_config(RRTMGPConfig(**{**meta["cfg"], **cfg_over}))
    mp = pytest.MonkeyPatch()
    mp.setattr(two_stream, "solve_lw", _stop)
    try:
        with pytest.raises(_Captured):
            solver.solve_columns(
                T=jnp.asarray(z["T_model"]), p_full=jnp.asarray(z["p_full_model"]),
                p_half=jnp.asarray(z["p_half_model"]),
                sfc_temperature=jnp.asarray(z["tsfc_model"]),
                q_v=jnp.asarray(z["q_model"]), cos_zenith=jnp.asarray(z["mu0_model"]),
                sfc_albedo=jnp.asarray(z["alb_model"]),
                sfc_emissivity=jnp.asarray(z["emis_model"]),
                o3_vmr=jnp.asarray(z["o3_model"]),
                o3_top_vmr=jnp.asarray(z["o3_top_model"]),
                ghg_vmr_override=meta["ghg"])
    finally:
        mp.undo()
    return np.asarray(cap["tlev"])[:, 0, :]  # (ncol, nlev+1[+1]), surface first


def test_level_temperature_matches_cam_rrtmg_state(case):
    """Production config (overhead layer on, as CAM6's extra layer)."""
    z, meta, _, _ = case
    assert meta["cfg"]["overhead_layer"]
    ours = _solver_level_temperature(z, meta)
    rel = _max_rel(ours, z["tlev_cam_oracle"][:, ::-1])
    assert rel <= _TOL, f"max rel error {rel:.3e} > {_TOL}"


def test_level_temperature_without_overhead_layer_is_cam_tint(case):
    """Overhead layer off: CAM's model-interface tint (tlev minus the extra
    layer's top interface)."""
    z, meta, _, _ = case
    ours = _solver_level_temperature(z, meta, overhead_layer=False)
    rel = _max_rel(ours, z["tlev_cam_oracle"][:, 1:][:, ::-1])
    assert rel <= _TOL, f"max rel error {rel:.3e} > {_TOL}"


def _sources(opt, vmr, p, T, sfcT, tlev, ig):
    return opt.compute_planck_sources(p, T, ig, vmr, sfc_temperature=sfcT,
                                      level_temperature=tlev)


def _by_gpoint(fn, opt):
    one = jax.jit(fn)
    return jnp.stack([one(jnp.asarray(i)) for i in range(opt.n_gpt_lw)], axis=-1)


def test_surface_level_source_matches_rrtmgp_v18(case):
    """Uncombined surface-face source pfrac(1) * B(tlev(sfc))."""
    z, meta, opt, vmr = case
    tlev = jnp.asarray(_solver_level_temperature(z, meta))[:, None, :]
    p, T, sfcT = (jnp.asarray(z[k]) for k in ("p", "T", "sfcT"))
    ours = np.asarray(_by_gpoint(
        lambda ig: _sources(opt, vmr, p, T, sfcT, tlev, ig)["planck_src_bottom"][:, 0, 1],
        opt))
    ref = z["lev_src_sfc_oracle"]
    assert np.all(ref > 0)
    rel = _max_rel(ours, ref)
    assert rel <= _TOL, f"max rel error {rel:.3e} > {_TOL}"


def test_combined_surface_level_source_matches_rrtmgp_v18(case):
    """The surface level source the two-stream solver reads (combined)."""
    z, meta, opt, vmr = case
    tlev = jnp.asarray(_solver_level_temperature(z, meta))[:, None, :]
    p, T, sfcT = (jnp.asarray(z[k]) for k in ("p", "T", "sfcT"))

    def fn(ig):
        comb = monochromatic_two_stream.lw_combine_sources(
            _sources(opt, vmr, p, T, sfcT, tlev, ig))
        return comb["planck_src_bottom"][:, 0, 1]  # node 1's bottom = surface level

    ours = np.asarray(_by_gpoint(fn, opt))
    rel = _max_rel(ours, z["lev_src_oracle"][:, 0, :])
    assert rel <= _TOL, f"max rel error {rel:.3e} > {_TOL}"


def test_interior_level_sources_match_rrtmgp_v18(case):
    """Every interior level source the two-stream solver reads, i.e. the face
    temperatures as actually mapped inside compute_planck_sources."""
    z, meta, opt, vmr = case
    tlev = jnp.asarray(_solver_level_temperature(z, meta))[:, None, :]
    p, T, sfcT = (jnp.asarray(z[k]) for k in ("p", "T", "sfcT"))

    def fn(ig):
        comb = monochromatic_two_stream.lw_combine_sources(
            _sources(opt, vmr, p, T, sfcT, tlev, ig))
        return comb["planck_src_top"][:, 0, 1:-2]  # levels 2..nlay (surface first)

    ours = np.asarray(_by_gpoint(fn, opt))
    floor = np.sqrt(monochromatic_two_stream._EPSILON)
    ref = np.maximum(z["lev_src_oracle"][:, 1:-1, :], floor)
    rel = _max_rel(ours, ref)
    assert rel <= _TOL, f"max rel error {rel:.3e} > {_TOL}"


@pytest.mark.parametrize("fp32", [False, True])
def test_level_source_gradients_finite(case, fp32):
    """d(sum of combined level sources)/d(T, p_half, T_sfc) finite, fp64 and
    the production fp32 optics."""
    from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import (
        cam_interface_temperature)
    z, meta, opt, vmr = case
    dt = jnp.float32 if fp32 else jnp.float64
    if fp32:
        opt = RRTMGP.from_legoesm_config(
            RRTMGPConfig(**{**meta["cfg"], "compute_fp32": True})).optics_lib
        vmr = {k: v.astype(dt) for k, v in vmr.items()}
    p, Tc = jnp.asarray(z["p"], dt), jnp.asarray(z["T"], dt)
    pf = jnp.asarray(z["p_full_model"], dt)

    def loss(T, ph, ts):
        # production layout: overhead layer on (its two interfaces = tint(1))
        tl = cam_interface_temperature(T, pf, ph, ts)
        tlev = jnp.concatenate([tl[:, :1], tl], axis=1)[:, None, ::-1]
        comb = monochromatic_two_stream.lw_combine_sources(
            _sources(opt, vmr, p, Tc, ts[:, None], tlev, jnp.asarray(0)))
        return jnp.sum(comb["planck_src_bottom"][:, 0, 1:-1])

    g = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(z["T_model"], dt), jnp.asarray(z["p_half_model"], dt),
        jnp.asarray(z["tsfc_model"], dt))
    for gi in g:
        assert gi.dtype == dt and np.all(np.isfinite(np.asarray(gi)))
        assert np.any(np.asarray(gi) != 0)  # no silently cut AD path
    assert np.all(np.asarray(g[2]) > 0)  # dB/dT > 0 at the surface level


_LANE_SCRIPT = r'''
import json, sys, jax
jax.config.update("jax_enable_x64", True)
from legoesm.atmosphere.physics.radiation.rrtmgp import rrtmgp as R
from legoesm.atmosphere.physics.radiation.rrtmgp.rte import monochromatic_two_stream as M
calls = {"tlev": 0, "combine": 0}
_t, _c = R.cam_interface_temperature, M.lw_combine_sources
def t_spy(*a, **k):
    calls["tlev"] += 1
    return _t(*a, **k)
def c_spy(*a, **k):
    calls["combine"] += 1
    return _c(*a, **k)
R.cam_interface_temperature, M.lw_combine_sources = t_spy, c_spy
from legoesm.driver.model_driver import ModelDriver
from legoesm.driver.config import ExperimentConfig, GridConfig, DycoreConfig, OutputConfig
out, lane = sys.argv[1], sys.argv[2]
common = dict(output=OutputConfig(output_dir=out, diag_days=0, checkpoint_days=0),
              days=2 * 600.0 / 86400.0, convection="none", microphysics="none",
              turbulence="none", gravity_wave_drag="none", precision="fp64",
              distributed=False)
if lane == "mpas":
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=8, vertical_coord="hybrid"),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="mpas", dt=600.0),
        dataset="analytical", radiation="rrtmg", **common)
else:
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=12, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="fv3_duo",
                            dt=600.0, column_lane=True, energy_fixer=False),
        radiation="rrtmgp", rad_update_steps=1, **common)
d = ModelDriver(cfg, output_dir=out)
d.setup()
status = d.run()
print("RESULT " + json.dumps({"status": status, **calls}))
'''


@pytest.mark.slow
@pytest.mark.parametrize("lane", ["mpas", "fv3_duo_column"])
def test_production_column_lanes_run_the_cam_levels_and_surface_source(tmp_path, lane):
    """End to end on ``ModelDriver._run_column`` (MPAS) and the FV3-duo
    column lane (which runs ``_run_column``): the LW solve is traced through
    the CAM level temperatures and the surface-level source combine.  PATH
    proof only; the values of those two functions are pinned by the oracle
    tests above (the lanes run fp32 radiation, where a recomputed source
    differs by rounding, so no value is asserted here)."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "PMI", "PMIX", "OMPI_"))}
    env.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, "-c", _LANE_SCRIPT, str(tmp_path), lane],
                          env=env, capture_output=True, text=True, timeout=3600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-3000:]
    rec = json.loads(lines[-1][len("RESULT "):])
    assert rec["status"] == "COMPLETED"
    assert rec["tlev"] >= 1 and rec["combine"] >= 1, rec
