"""Oracle match: CAM6 physics sequencing (tphysbc/tphysac order) vs CAM, RUN as Fortran.

Oracle: ``scripts/validate/oracle/cam6_sequencing/oracle_seq.F90`` -- verbatim
extracts of CAM cam_cesm2_1_rel_60 physics_update / ptend_scale arithmetic and the
physpkg control flow (deep convection, macmic loop, radiation, then gw_tend), with
analytic stub schemes identical to ``stubs.py``.  Fixture
``oracle_fixtures/cam6_sequencing.npz`` from make_fixture.py in that directory.

Ours = the production combined-physics glue (MPAS lane, physics every 16 x 112.5 s,
radiation every 32 steps, cld_macmic_num_steps = 3, the driver's own step selection)
with the scheme factories replaced by the same stubs; 32 steps of
``state += dt * tendency`` (no dynamics), i.e. two physics windows.

Compared: the 1-hour increments of T, u, q_v, q_c, q_i and both windows' precipitation,
normalised by max|oracle increment| per field.  The CAM features not ported here
(detrainment inside CLUBB, per-process qneg3) are off in the "1100" oracle mode.

Run with ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import legoesm.atmosphere.physics.combined as cmb
from legoesm.atmosphere.physics.convection.config import ConvectionConfig
from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.physics_state import init_physics_state
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
from legoesm.core.field import Field
from legoesm.core.state import HydrostaticState, HydrostaticTendencies
from legoesm.driver.mpas_diff_step import mpas_cadence, pick_physics_fn

FIXTURE = Path(__file__).parents[2] / "oracle_fixtures" / "cam6_sequencing.npz"
_STUBS = (Path(__file__).parents[4] / "scripts" / "validate" / "oracle" / "cam6_sequencing"
          / "stubs.py")
_TOL = 1e-10  # user-approved oracle-match tolerance (max relative error, fp64)
_DIMS = ("nCells", "nlev")
DT, PHYS, RAD, NMAC = 112.5, 16, 32, 3


def _load_stubs():
    spec = importlib.util.spec_from_file_location("cam6_seq_stubs", _STUBS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _F(x, name=""):
    return Field(data=x, name=name, dims=_DIMS, units="")


def _run_ours(z, monkeypatch):
    stubs = _load_stubs()
    cp, Lv, g = float(z["cp"]), float(z["Lv"]), float(z["g"])
    pm, dp, q0ref = (jnp.asarray(z[k]) for k in ("pm", "dp", "q0ref"))

    def tend(state, du=None, dT=None, tr=None, precip=None):
        zz, z2 = jnp.zeros_like(state.T.data), jnp.zeros_like(state.p_s.data)
        kw = {}
        if precip is not None:
            kw = dict(precip=Field(precip, "precip", ("nCells",)),
                      precip_frozen=Field(jnp.zeros_like(precip), "pf", ("nCells",)))
        return HydrostaticTendencies(
            du_dt=_F(zz if du is None else du), dT_dt=_F(zz if dT is None else dT),
            dp_s_dt=Field(z2, "dps", ("nCells",)), dphis_dt=Field(z2, "dphis", ("nCells",)),
            dv_dt=None,
            tracer_tendencies=None if tr is None else {k: _F(v, k) for k, v in tr.items()},
            **kw)

    def st(s):
        return (s.T.data, s.tracers["q_v"].data, s.tracers["q_c"].data,
                s.tracers["q_i"].data, s.u.data)

    def f_rad(*a, **k):
        def fn(state, grid, sigma):
            T, q, qc, qi, u = st(state)
            return tend(state, dT=stubs.rad(T, qc, qi))
        return fn

    def f_conv(cfg, model_type, dt, *a, **k):
        def fn(state, grid, sigma, phys_state=None):
            T, q, qc, qi, u = st(state)
            dT, dq, dlf, du = stubs.conv(T, q, u, pm, cp, Lv)
            return tend(state, du=du, dT=dT, tr={"q_v": dq, "q_c": dlf}), None
        return fn

    def f_turb(cfg, model_type, dt, *a, **k):
        def fn(state, grid, sigma, phys_state=None):
            T, q, qc, qi, u = st(state)
            dT, dq, dqc, du = stubs.turb(T, q, qc, u, dt, dp, cp, Lv, g)
            return tend(state, du=du, dT=dT, tr={"q_v": dq, "q_c": dqc}), None
        return fn

    def f_micro(cfg, model_type, dt, *a, **k):
        def fn(state, grid, sigma):
            T, q, qc, qi, u = st(state)
            dT, dq, dqc, dqi, pr = stubs.micro(T, q, qc, qi, dt, q0ref, dp, cp, Lv, g)
            return tend(state, dT=dT, tr={"q_v": dq, "q_c": dqc, "q_i": dqi}, precip=pr)
        return fn

    def f_gwd(cfg, model_type, dt, *a, **k):
        def fn(state, grid, sigma, phys_state=None):
            T, q, qc, qi, u = st(state)
            du, dT = stubs.gw(T, u, cp)
            return tend(state, du=du, dT=dT), None
        return fn

    for name, f in (("make_radiation_physics", f_rad), ("make_convection_physics", f_conv),
                    ("make_turbulence_physics", f_turb), ("make_microphysics_physics", f_micro),
                    ("make_gwd_physics", f_gwd)):
        monkeypatch.setattr(cmb, name, f)

    cfg = cmb.PhysicsConfig(
        radiation=RadiationConfig(scheme="gray"),
        convection=ConvectionConfig(scheme="zhang_mcfarlane"),
        turbulence=TurbulenceConfig(scheme="tke"),
        microphysics=MicrophysicsConfig(scheme="kessler"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="mcfarlane"),
    )
    kw = dict(model_type="mpas", dt=DT * PHYS, physics_cadence="write",
              physics_cadence_steps=PHYS, cld_macmic_num_steps=NMAC)
    full = cmb.make_physics(cfg, **kw)
    norad = cmb.make_physics(cfg, need_rad=False, **kw)
    held = cmb.held_physics_variant(full)

    ncol, nlev = z["T"].shape
    state = HydrostaticState(
        u=_F(jnp.asarray(z["u"]), "u"), T=_F(jnp.asarray(z["T"]), "T"),
        p_s=Field(jnp.full((ncol,), 1.0e5), "p_s", ("nCells",)),
        phis=Field(jnp.zeros((ncol,)), "phis", ("nCells",)), v=None,
        tracers={"q_v": _F(jnp.asarray(z["q"]), "q_v"), "q_c": _F(jnp.asarray(z["qc"]), "q_c"),
                 "q_i": _F(jnp.asarray(z["qi"]), "q_i")})
    ps = init_physics_state(ncol, nlev, cfg, dtype=jnp.float64)
    ps = ps._replace(held_physics=cmb.physics_cache_seed(full, state, None, None, ps))
    precips, picks = [], []
    for step in range(2 * PHYS):
        use_rad, is_phys = mpas_cadence(step, subcycle_rad=True, rad_update_steps=RAD,
                                        hold_phys=True, phys_update_steps=PHYS)
        pfn = pick_physics_fn(use_rad, is_phys, full, norad, held)
        picks.append("F" if pfn is full else "N" if pfn is norad else "H")
        tnd, ps = pfn(state, None, None, phys_state=ps)
        state = state._replace(
            u=state.u.replace(data=state.u.data + DT * tnd.du_dt.data),
            T=state.T.replace(data=state.T.data + DT * tnd.dT_dt.data),
            p_s=state.p_s.replace(data=state.p_s.data + DT * tnd.dp_s_dt.data),
            tracers={k: f.replace(data=f.data + DT * tnd.tracer_tendencies[k].data)
                     for k, f in state.tracers.items()})
        if is_phys:
            precips.append(np.asarray(tnd.precip.data))
    assert "".join(picks) == "F" + "H" * 15 + "N" + "H" * 15
    return dict(T=np.asarray(state.T.data), U=np.asarray(state.u.data),
                Q=np.asarray(state.tracers["q_v"].data),
                QC=np.asarray(state.tracers["q_c"].data),
                QI=np.asarray(state.tracers["q_i"].data), PREC=np.stack(precips))


def _worst(ours, z, mode):
    init = dict(T=z["T"], U=z["u"], Q=z["q"], QC=z["qc"], QI=z["qi"])
    errs = {}
    for k in ("T", "U", "Q", "QC", "QI", "PREC"):
        a = ours[k] if k == "PREC" else ours[k] - init[k]
        b = z[f"{k}_{mode}"] if k == "PREC" else z[f"{k}_{mode}"] - init[k]
        scale = np.max(np.abs(b))
        errs[k] = float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(a)))
    return errs


@pytest.fixture(scope="module")
def fix():
    return np.load(FIXTURE)


def test_cam6_sequencing_matches_cam(fix, monkeypatch):
    errs = _worst(_run_ours(fix, monkeypatch), fix, "1100")
    assert max(errs.values()) <= _TOL, errs


def test_cam6_sequencing_old_order_is_rejected(fix, monkeypatch):
    """Control: the oracle run in the OLD legoESM order (radiation and GWD on the
    pre-physics state) must not match ours."""
    errs = _worst(_run_ours(fix, monkeypatch), fix, "0000")
    assert max(errs.values()) > 1e3 * _TOL, errs
