"""CAM6 detrainment of ZM condensate (oracle rows 51, 53) and ZM convective
transport on the spectral lane (row 52).

CAM6 does not book ZM's detrained condensate ``dlf`` at convection time: the
CLUBB host books it in every macmic sub-step on its post-CLUBB state
(clubb_intr.F90:2345-2386, cam_cesm2_1_rel_60,
a03b84b7c4e34f965b115686f22a043b85739e56): ice fraction ``dum1`` ramping from
0 above 268.15 K to 1 below 238.15 K, fusion heating ``dlf*dum1*latice``, and
droplet / crystal number sources for 8 um / 25 um particles.  Ported as
``zhang_mcfarlane.cam6_detrain_condensate``, applied by the MPAS/column CLUBB
bridge on the ``dlf`` the combined macmic loop hands it.

Fixture ``oracle_fixtures/zm_detrain_cam.npz``: the CAM loop, #included
verbatim, RUN by ``scripts/validate/oracle/zm_detrain/{build.sh,
make_fixture.py}`` on the 15 columns of ``zm_pblh_cam.npz`` (their T and the
ZM oracle's own dlf).

Metric (the merged ZM oracle test's ``_rel``, reused unchanged):
``max |ours - oracle| / max |oracle|`` per field; bound 1e-10 in fp64 at the
hard limit (both ZhangMcFarlaneConfig detrain widths 0).  RULE 5: the default
smooths the liquid/ice ramp and the number source's ``max(0, dlf)``; its gap
to CAM is stored per fixture column in ``GAPS``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    cam6_detrain_condensate,
)
from legoesm.core.field import Field
from tests.atmosphere.test_zm_pblh_oracle import TRACERS, _fp64_policy, _rel  # noqa: F401

jax.config.update("jax_enable_x64", True)

TOL = 1.0e-10
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "atmosphere" / "oracle_fixtures" / "zm_detrain_cam.npz"
GEN = ROOT / "scripts" / "validate" / "oracle" / "zm_detrain" / "make_fixture.py"


@pytest.fixture(scope="module")
def fx():
    return dict(np.load(FIXTURE))


def test_fixture_provenance_and_coverage(fx):
    meta = json.loads(str(fx["meta"]))
    assert meta["oracle"]["sha"] == "a03b84b7c4e34f965b115686f22a043b85739e56"
    c = meta["constants"]
    assert (c["latice"], c["gravit"], c["cpair"]) == (
        constants.L_f, constants.g, constants.c_pd)
    assert FIXTURE.stat().st_size < 1_000_000
    T, dlf = fx["in_T"], fx["in_dlf"]
    on = dlf > 0.0
    # detrainment above, inside and below the 268.15 -> 238.15 K ramp
    assert (on & (T > 268.15)).sum() >= 2
    assert (on & (T <= 268.15) & (T >= 238.15)).sum() >= 2
    assert (on & (T < 238.15)).sum() >= 2


def test_generator_rebuilds_the_committed_inputs(fx):
    """Direct test of scripts/validate/oracle/zm_detrain/make_fixture.py."""
    spec = importlib.util.spec_from_file_location("zm_detrain_mf", GEN)
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    src = np.load(mf.SRC)
    np.testing.assert_array_equal(src["T"], fx["in_T"])
    np.testing.assert_array_equal(src["oracle_dlf"], fx["in_dlf"])
    np.testing.assert_array_equal(np.diff(src["p_half"], axis=1), fx["in_pdel"])


HARD = ZhangMcFarlaneConfig(detrain_phase_width_K=0.0, detrain_number_width_kgkg_s=0.0)
FIELDS = ("dT", "dq_c", "dq_i", "dN_c", "dN_i")

# Measured gap of the smooth default to CAM, per fixture column:
# max_lev |ours - oracle| / max |oracle| (whole fixture) per field.  Re-measure
# and update in the same PR on any change to the smooth form or its widths.
GAPS = {
    "dT": (0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
           0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
           0.000000e+00, 1.568669e-08, 0.000000e+00, 0.000000e+00, 0.000000e+00),
    "dq_c": (0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
             0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
             0.000000e+00, 1.361092e-08, 0.000000e+00, 0.000000e+00, 0.000000e+00),
    "dq_i": (0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
             0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
             0.000000e+00, 1.568669e-08, 0.000000e+00, 0.000000e+00, 0.000000e+00),
    "dN_c": (1.143260e-04, 1.143260e-04, 1.143260e-04, 1.143260e-04, 1.143260e-04,
             1.143260e-04, 8.061375e-05, 1.143260e-04, 1.143260e-04, 1.143260e-04,
             1.143260e-04, 1.143260e-04, 1.143260e-04, 1.143260e-04, 1.143260e-04),
    "dN_i": (1.317616e-04, 1.317616e-04, 1.317616e-04, 1.317616e-04, 1.317616e-04,
             1.317616e-04, 1.317616e-04, 1.317616e-04, 1.317616e-04, 1.317616e-04,
             1.317616e-04, 1.317616e-04, 1.317616e-04, 1.317616e-04, 1.317616e-04),
}


def _oracle(fx):
    # CAM physics_update: T += s*dt/cpair (physics_types.F90:411-417)
    return (fx["ora_s"] / constants.c_pd, fx["ora_dq_c"], fx["ora_dq_i"],
            fx["ora_dN_c"], fx["ora_dN_i"])


def test_detrainment_matches_cam_oracle_at_the_hard_limit(fx):
    out = cam6_detrain_condensate(jnp.asarray(fx["in_T"]), jnp.asarray(fx["in_dlf"]), HARD)
    errs = {nm: _rel(o, r) for nm, o, r in zip(FIELDS, out, _oracle(fx))}
    assert max(errs.values()) <= TOL, errs
    # liquid + ice = dlf: the partition conserves the detrained water
    np.testing.assert_allclose(np.asarray(out[1] + out[2]), fx["in_dlf"], rtol=1e-15, atol=0)


def test_smooth_default_gap_to_cam_per_column(fx):
    """RULE 5 KNOWN DIFFERENCE: the default's gap, column by column."""
    out = cam6_detrain_condensate(jnp.asarray(fx["in_T"]), jnp.asarray(fx["in_dlf"]),
                                  ZhangMcFarlaneConfig())
    for nm, o, r in zip(FIELDS, out, _oracle(fx)):
        gap = np.max(np.abs(np.asarray(o) - r), axis=1) / np.max(np.abs(r))
        np.testing.assert_allclose(gap, GAPS[nm], rtol=1e-6, atol=1e-12, err_msg=nm)
    np.testing.assert_allclose(np.asarray(out[1] + out[2]), fx["in_dlf"], rtol=1e-15, atol=0)


def test_negative_width_is_refused(fx):
    with pytest.raises(ValueError, match="detrain_phase_width_K"):
        cam6_detrain_condensate(jnp.asarray(fx["in_T"]), jnp.asarray(fx["in_dlf"]),
                                ZhangMcFarlaneConfig(detrain_phase_width_K=-0.1))


def test_check_grads_through_the_switches_at_the_default(fx):
    """Second-order AD vs finite differences at the smooth default, on the
    fixture columns plus points ON the ramp corners and at dlf = 0."""
    from jax.test_util import check_grads
    scale = float(np.max(np.abs(fx["in_dlf"])))
    T = jnp.concatenate([jnp.asarray(fx["in_T"]).ravel(),
                         jnp.asarray([268.15, 238.15, 268.2, 238.1, 250.0])])
    d = jnp.concatenate([jnp.asarray(fx["in_dlf"]).ravel() / scale,
                         jnp.asarray([0.5, 0.5, 0.0, 0.3, 0.0])])
    cfg = ZhangMcFarlaneConfig()

    norm = [float(jnp.max(jnp.abs(x))) for x in cam6_detrain_condensate(T, d * scale, cfg)]

    def f(T, d):
        return tuple(x / n for x, n in zip(cam6_detrain_condensate(T, d * scale, cfg), norm))
    check_grads(f, (T, d), order=2, modes=("fwd", "rev"), eps=1e-6)
    gT, gd = jax.jacfwd(lambda T: f(T, d)[2])(T), jax.jacfwd(lambda d: f(T, d)[3])(d)
    # non-zero through the phase ramp at its corner and through max(0, dlf) at 0
    assert float(gT[-5, -5]) != 0.0 and float(gd[-1, -1]) != 0.0


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_gradient_is_finite(fx, dtype):
    T = jnp.asarray(fx["in_T"], dtype)
    dlf = jnp.asarray(fx["in_dlf"], dtype)

    def loss(T, dlf):
        dT, dqc, dqi, dnc, dni = cam6_detrain_condensate(T, dlf, ZhangMcFarlaneConfig())
        return jnp.sum(dT * 1e3 + dqi * 1e8 + dqc * 1e8 + dnc * 1e-3 + dni * 1e-1)
    gT, gd = jax.grad(loss, argnums=(0, 1))(T, dlf)
    assert np.all(np.isfinite(np.asarray(gT))) and np.all(np.isfinite(np.asarray(gd)))
    assert float(jnp.max(jnp.abs(gT))) > 0.0 and float(jnp.max(jnp.abs(gd))) > 0.0


# ---------------------------------------------------------------------------
# Wiring: the MPAS/column CLUBB bridge books dlf on its post-CLUBB state
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mpas_case():
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(subdivision_level=1, lloyd_iterations=2)
    sigma = create_sigma_coordinate(8, sigma_top=0.05)
    state = held_suarez_init_mpas(mesh, sigma, T_init=290.0, perturbation_amplitude=0.0)
    ncell, nlev = state.T.data.shape
    dtype = state.T.data.dtype
    # 300 K at the surface to 215 K aloft: levels on both sides of the ramp
    T = jnp.broadcast_to(jnp.linspace(215.0, 300.0, nlev, dtype=dtype), (ncell, nlev))
    u_edge = (jnp.cos(jnp.asarray(mesh.angleEdge))[:, None]
              * jnp.linspace(20.0, 2.0, nlev)[None, :]).astype(dtype)
    tr = {k: state.T.replace(data=jnp.full((ncell, nlev), v, dtype), name=k)
          for k, v in (("q_v", 0.008), ("q_c", 1e-5), ("q_i", 1e-6),
                       ("N_c", 1e7), ("N_i", 1e4))}
    return mesh, sigma, state._replace(T=state.T.replace(data=T),
                                       u=state.u.replace(data=u_edge), tracers=tr)


def _land_forcing(ncell):
    return {"T_sfc": jnp.full((ncell,), 300.0),
            "shflx_land": jnp.full((ncell,), 150.0),
            "lhflx_land": jnp.full((ncell,), 80.0),
            "evap_land": jnp.full((ncell,), 80.0 / constants.L_v)}


def test_mpas_clubb_bridge_books_dlf_on_the_post_clubb_state(mpas_case, monkeypatch):
    """Fails if the bridge ignores the handed dlf, ramps on the pre-CLUBB
    temperature, or runs the PBL height on the pre-detrainment state."""
    from legoesm.atmosphere.physics.turbulence import integration as TI
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig

    mesh, sigma, state = mpas_case
    ncell, nlev = state.T.data.shape
    cap = {}
    real = TI.cam_pblintd_height

    def spy(*a, **kw):
        cap["T"] = a[0]
        return real(*a, **kw)
    monkeypatch.setattr(TI, "cam_pblintd_height", spy)
    dt = 600.0
    zm = ZhangMcFarlaneConfig()
    fn = TI._make_mpas_turbulence(
        TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True)), dt,
        f_land=jnp.ones((ncell,)), zm_config=zm)
    frc = _land_forcing(ncell)
    t0, _ = fn(state, mesh, sigma, forcing=frc)
    dlf = jnp.broadcast_to(jnp.linspace(1e-8, 8e-8, nlev), (ncell, nlev))
    t1, _ = fn(state, mesh, sigma, forcing={**frc, "conv_dlf": dlf})
    ref = cam6_detrain_condensate(state.T.data + dt * t0.dT_dt.data, dlf, zm)
    np.testing.assert_allclose(np.asarray(t1.dT_dt.data - t0.dT_dt.data),
                               np.asarray(ref[0]), rtol=1e-12, atol=1e-20)
    for nm, r in zip(("q_c", "q_i", "N_c", "N_i"), ref[1:]):
        base = (t0.tracer_tendencies[nm].data if nm in t0.tracer_tendencies else 0.0)
        np.testing.assert_allclose(np.asarray(t1.tracer_tendencies[nm].data - base),
                                   np.asarray(r), rtol=1e-12, atol=1e-30, err_msg=nm)
    dum1 = np.asarray(ref[2] / dlf)
    assert abs(dum1.min()) < 1e-12 and abs(dum1.max() - 1.0) < 1e-12, "ramp sides not exercised"
    # the PBL height sees state1 after the detrainment update (clubb_intr :2397, :2616)
    np.testing.assert_allclose(np.asarray(cap["T"]),
                               np.asarray(state.T.data + dt * t1.dT_dt.data), rtol=1e-14)
    # the rest of CLUBB is untouched by the hand-off
    np.testing.assert_array_equal(np.asarray(t1.tracer_tendencies["q_v"].data),
                                  np.asarray(t0.tracer_tendencies["q_v"].data))


def test_dlf_handoff_without_zm_config_is_refused(mpas_case):
    from legoesm.atmosphere.physics.turbulence import integration as TI
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    mesh, sigma, state = mpas_case
    ncell, nlev = state.T.data.shape
    cfg = TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True))
    fn = TI._make_mpas_turbulence(cfg, 600.0, f_land=jnp.ones((ncell,)))
    with pytest.raises(ValueError, match="zm_config"):
        fn(state, mesh, sigma, forcing={**_land_forcing(ncell),
                                        "conv_dlf": jnp.ones((ncell, nlev)) * 1e-8})
    with pytest.raises(ValueError, match="zm_config"):
        TI.make_turbulence_physics(cfg, "spectral_pe", 600.0,
                                   zm_config=ZhangMcFarlaneConfig())
    fn = TI._make_mpas_turbulence(cfg, 600.0, f_land=jnp.ones((ncell,)),
                                  zm_config=ZhangMcFarlaneConfig())
    no_ni = state._replace(tracers={k: v for k, v in state.tracers.items() if k != "N_i"})
    with pytest.raises(ValueError, match=r"lacks \['N_i'\]"):
        fn(no_ni, mesh, sigma, forcing={**_land_forcing(ncell),
                                        "conv_dlf": jnp.ones((ncell, nlev)) * 1e-8})


# ---------------------------------------------------------------------------
# Glue: ZM publishes dlf, the macmic loop hands it to every CLUBB sub-step
# ---------------------------------------------------------------------------

def test_macmic_hands_dlf_to_every_clubb_substep(monkeypatch):
    import legoesm.atmosphere.physics.combined as cmb
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    from legoesm.core.state import HydrostaticState, HydrostaticTendencies

    ncol, nlev, N, dt = 3, 5, 3, 1800.0
    DIMS = ("nCells", "nlev")
    F = lambda x: Field(data=x, name="", dims=DIMS, units="")  # noqa: E731
    dlf = jnp.linspace(1e-8, 5e-8, ncol * nlev).reshape(ncol, nlev)
    seen = {"defer": None, "turb": [], "zm": "unset"}

    def tend(state, dT=None, tr=None):
        z, z2 = jnp.zeros_like(state.T.data), jnp.zeros_like(state.p_s.data)
        return HydrostaticTendencies(
            du_dt=F(z), dT_dt=F(z if dT is None else dT),
            dp_s_dt=Field(z2, "dps", ("nCells",)), dphis_dt=Field(z2, "dphis", ("nCells",)),
            dv_dt=None, tracer_tendencies={k: F(v) for k, v in (tr or {}).items()})

    def f_conv(cfg, model_type, dt_, *, defer_detrainment=False):
        seen["defer"] = defer_detrainment

        def fn(state, grid, sigma, phys_state=None):
            return tend(state, dT=jnp.full_like(state.T.data, 1e-5),
                        tr={"q_c": jnp.zeros_like(dlf)}), {"conv_dlf": dlf}
        return fn

    def f_turb(cfg, model_type, dt_, *a, **k):
        seen["zm"] = k.get("zm_config")

        def fn(state, grid, sigma, phys_state=None, forcing=None):
            seen["turb"].append((np.asarray(state.T.data), forcing.get("conv_dlf")))
            return tend(state, dT=jnp.full_like(state.T.data, 2e-5)), None
        fn._wants_forcing = True
        return fn

    def f_micro(cfg, model_type, dt_, *a, **k):
        return lambda state, grid, sigma: tend(state, dT=jnp.full_like(state.T.data, -1e-5))

    monkeypatch.setattr(cmb, "make_convection_physics", f_conv)
    monkeypatch.setattr(cmb, "make_turbulence_physics", f_turb)
    monkeypatch.setattr(cmb, "make_microphysics_physics", f_micro)
    cfg = cmb.PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="zhang_mcfarlane"),
        turbulence=TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True)),
        microphysics=MicrophysicsConfig(scheme="kessler"))
    T0 = jnp.full((ncol, nlev), 280.0)
    state = HydrostaticState(
        u=F(jnp.zeros((ncol, nlev))), T=F(T0),
        p_s=Field(jnp.full((ncol,), 1.0e5), "p_s", ("nCells",)),
        phis=Field(jnp.zeros((ncol,)), "phis", ("nCells",)), v=None,
        tracers={k: F(jnp.full((ncol, nlev), 1e-4)) for k in ("q_v", "q_c", "q_i")})
    ps = init_physics_state(ncol, nlev, cfg, dtype=jnp.float64)
    fn = cmb.make_physics(cfg, model_type="mpas", dt=dt, cld_macmic_num_steps=N)
    out, ps_out = fn(state, None, None, phys_state=ps)
    assert seen["defer"] is True
    assert seen["zm"] is cfg.convection.zhang_mcfarlane
    assert len(seen["turb"]) == N
    dts = dt / N
    for i, (T_seen, d) in enumerate(seen["turb"]):
        np.testing.assert_array_equal(np.asarray(d), np.asarray(dlf))
        # sub-step i sees the convective increment and i (turb + micro) increments
        np.testing.assert_allclose(T_seen, np.asarray(T0) + 1e-5 * dt + i * (2e-5 - 1e-5) * dts,
                                   rtol=1e-14)
    # ZM's own q_c tendency no longer carries dlf (it is booked by the CLUBB host)
    np.testing.assert_array_equal(np.asarray(out.tracer_tendencies["q_c"].data), 0.0)


def _zm_bridge(fx, defer, names):
    """The real hydrostatic/MPAS ZM bridge on the ZM oracle fixture columns,
    carrying the convtran1 tracers in ``names``."""
    import types

    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.convection.integration import make_convection_physics
    from legoesm.atmosphere.physics.physics_state import (
        init_physics_state,
        update_physics_state,
    )
    from legoesm.core.state import HydrostaticState
    from legoesm.grids.vertical import make_cam6_l32_levels
    ncol, nlev = fx["T"].shape
    tracers = {"q_v": Field(jnp.asarray(fx["q"]))}
    tracers.update({nm: Field(jnp.asarray(fx[key])) for key, nm, _ in TRACERS if nm in names})
    state = HydrostaticState(
        u=Field(jnp.asarray(fx["u"])), v=Field(jnp.asarray(fx["v"])),
        T=Field(jnp.asarray(fx["T"])), p_s=Field(jnp.asarray(fx["p_s"])),
        phis=Field(jnp.zeros((ncol,))), tracers=tracers)
    ps = update_physics_state(init_physics_state(ncol, nlev, PhysicsConfig()),
                              {"cloud_fraction": jnp.asarray(fx["cld"]),
                               "pblh": jnp.asarray(fx["pblh"])})
    fn = make_convection_physics(ConvectionConfig(scheme="zhang_mcfarlane"), model_type="mpas",
                                 dt=float(fx["dt"]), defer_detrainment=defer)
    return fn(state, types.SimpleNamespace(land_frac=jnp.asarray(fx["land"])),
              make_cam6_l32_levels(), phys_state=ps)


def test_real_zm_bridge_hands_dlf_over_instead_of_booking_it():
    """With deferral the bridge books only the convective transport into q_c
    and hands CAM's dlf over (once, not twice); without deferral it keeps the
    all-liquid booking and hands nothing over; deferral on a state lacking a
    species the CLUBB booking writes is refused."""
    fx = dict(np.load(ROOT / "tests" / "atmosphere" / "oracle_fixtures" / "zm_pblh_cam.npz"))
    all_names = tuple(nm for _, nm, _ in TRACERS)
    tend, carry = _zm_bridge(fx, True, all_names)
    assert _rel(carry["conv_dlf"], fx["oracle_dlf"]) <= TOL
    assert _rel(tend.tracer_tendencies["q_c"].data, fx["oracle_dqc_convtran"]) <= TOL
    off, c_off = _zm_bridge(fx, False, all_names)
    assert "conv_dlf" not in c_off
    assert _rel(off.tracer_tendencies["q_c"].data,
                fx["oracle_dlf"] + fx["oracle_dqc_convtran"]) <= TOL
    for drop in ("q_i", "N_c", "N_i"):
        with pytest.raises(ValueError, match=rf"lacks \['{drop}'\]"):
            _zm_bridge(fx, True, tuple(n for n in all_names if n != drop))


def test_no_deferral_off_the_production_combination():
    """Only ZM + CLUBB on the MPAS/column lane with a macmic loop defers dlf;
    the bridge refuses the flag on the lanes that do not implement it."""
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.convection.integration import make_convection_physics
    with pytest.raises(ValueError, match="defer_detrainment"):
        make_convection_physics(ConvectionConfig(scheme="zhang_mcfarlane"),
                                "spectral_pe", 600.0, defer_detrainment=True)


# ---------------------------------------------------------------------------
# Row 52: the spectral PE bridge transports the carried convtran1 tracers
# ---------------------------------------------------------------------------

def test_spectral_bridge_adds_the_convtran_of_the_carried_tracers(monkeypatch):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import isothermal_rest_state_spectral
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.convection import integration as CI
    from legoesm.atmosphere.physics.convection.config import (
        ConvectionConfig,
        ZhangMcFarlaneConfig,
    )
    from legoesm.atmosphere.physics.physics_state import (
        init_physics_state,
        update_physics_state,
    )
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate

    grid = create_gaussian_grid(n_max=21)
    sigma = create_sigma_coordinate(6)
    nlat, nlon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
    ncol = nlat * nlon
    state = isothermal_rest_state_spectral(grid, sigma, perturbation_amplitude=0.0)
    dims = ("lat", "lon", "level")
    names = ("q_c", "q_i", "N_c", "N_i", "q_r", "q_s", "N_r")
    base = jnp.arange(nlat * nlon * nlev, dtype=jnp.float64).reshape(nlat, nlon, nlev)
    tracers = {"q_v": Field(base * 0 + 1e-3, dims=dims)}
    tracers.update({nm: Field(base * (j + 1) * 1e-9, dims=dims) for j, nm in enumerate(names)})
    state = state._replace(tracers=tracers)
    seen = {}
    real = CI.zhang_mcfarlane_convection

    def spy(*a, **k):
        seen["tracers"] = k.get("tracers")
        out, prog = real(*a, **k)
        inj = {nm: jnp.full((ncol, nlev), (j + 1) * 1e-7) for j, nm in enumerate(names)}
        seen["inj"] = inj
        return out._replace(tracer_transport=inj), prog
    monkeypatch.setattr(CI, "zhang_mcfarlane_convection", spy)
    ps = update_physics_state(init_physics_state(ncol, nlev, PhysicsConfig()),
                              {"pblh": jnp.full((ncol,), 500.0)})
    cc = ConvectionConfig(scheme="zhang_mcfarlane",
                          zhang_mcfarlane=ZhangMcFarlaneConfig(land_fraction="none"))
    fn = CI.make_convection_physics(cc, model_type="spectral_pe", dt=600.0)
    tend, _ = fn(state, grid, sigma, phys_state=ps)
    assert seen["tracers"] is not None, "the spectral bridge passed no tracers to ZM"
    assert set(seen["tracers"]) == set(names)
    for nm in names:
        np.testing.assert_array_equal(np.asarray(seen["tracers"][nm]),
                                      np.asarray(tracers[nm].data).reshape(ncol, nlev))
    out_qc = np.asarray(tend.tracers["q_c"].data)
    for nm in names[1:]:
        np.testing.assert_allclose(np.asarray(tend.tracers[nm].data),
                                   np.asarray(seen["inj"][nm]).reshape(nlat, nlon, nlev),
                                   rtol=0, atol=0, err_msg=nm)
    # q_c: detrained condensate (zero at rest) + transport
    np.testing.assert_allclose(out_qc, np.asarray(seen["inj"]["q_c"]).reshape(nlat, nlon, nlev),
                               rtol=0, atol=1e-30)
