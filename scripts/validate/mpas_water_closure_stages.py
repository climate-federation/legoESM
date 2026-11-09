"""Stage-split global water / dry-mass budget for the small MPAS test
atmosphere (level-2 mesh, the P5b closure-probe config), standalone (no
coupler).  Numbers only, no verdicts.

Per model.step: global water W [kg] and dry mass Md [kg] on the input and
output state (hybrid layer masses, all water species the state carries,
area-weighted).  Between steps: W(k+1 in) - W(k out) = anything the driver
does outside model.step.  Inside the step, the three shared conservation
helpers are wrapped (trace-time monkeypatch; values reported through
jax.debug.callback) to give the water change of: physics application
(apply_physics_water_mass), dry-mass fixer (shift_ps_keep_tracer_mass) and
positivity (apply_water_positivity).  Dycore + rest = step total minus those.
Physics water change is compared with sum(E - P) dt from the step's own
surface diagnostics.

When the Kessler scheme runs, its own column residual (column water change +
surface rain, in rho*dz, area-weighted) is reported per call.

argv: DAYS [default 1], ARM in {full, dyn} (dyn = radiation/turbulence/
microphysics none), MICRO [kessler], TURB [louis], then key=value
ExperimentConfig overrides (vertical_coord=... picks the coordinate).
Usage: JAX_ENABLE_X64=1 python scripts/validate/mpas_water_closure_stages.py 2 full"""
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
from legoesm import constants
import legoesm.core.conservation as C
from legoesm.core.state import MPAS_SFC_DIAG_BASE_KEYS, MPAS_SFC_DIAG_EXTRA_KEYS
from legoesm.driver.config import (
    DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
from legoesm.driver.model_driver import ModelDriver

DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
ARM = sys.argv[2] if len(sys.argv) > 2 else "full"
MICRO = sys.argv[3] if len(sys.argv) > 3 else "kessler"
TURB = sys.argv[4] if len(sys.argv) > 4 else "louis"
EXTRA = dict(kv.split("=", 1) for kv in sys.argv[5:])
VCOORD = EXTRA.pop("vertical_coord", "hybrid")
DT = 600.0
phys = dict(radiation="gray", rad_update_steps=1, convection="none",
            turbulence=TURB, microphysics=MICRO)
if ARM == "dyn":
    phys = dict(radiation="none", convection="none", turbulence="none",
                microphysics="none")
cfg = ExperimentConfig(
    grid=GridConfig(grid_type="mpas", resolution=2, nlev=8,
                    vertical_coord=VCOORD),
    dycore=DycoreConfig(discretization="mpas", dt=DT),
    output=OutputConfig(output_dir="", diag_days=0, checkpoint_days=0),
    days=DAYS, dataset="analytical", precision="fp64", **phys, **EXTRA)
print("ARM", ARM, "MICRO", MICRO, "TURB", TURB, "VCOORD", VCOORD, "EXTRA", EXTRA, "DAYS", DAYS, "cfg", {k: getattr(cfg, k) for k in phys})
atm = ModelDriver(cfg, output_dir=tempfile.mkdtemp())
atm.setup()
model = atm.model
sig = model.sigma_coord
area = jnp.asarray(model.mesh.areaCell).reshape(-1)
area_np = np.asarray(area)
print(f"INSTR area_sum/4piR2-1 = {area_np.sum() / (4 * np.pi * constants.R_earth ** 2) - 1:.3e}"
      f"  nCells {area_np.size}")
WATER = C.WATER_MASS_SPECIES


def _water(tracers, dp):
    Q = 0.0
    for n in WATER:
        if tracers is not None and n in tracers:
            Q = Q + C._d(tracers[n])
    return jnp.sum(jnp.sum(Q * dp, axis=-1) * area) / constants.g


def W_of(state):
    return float(_water(state.tracers, sig.layer_thickness_dp(state.p_s.data)))


def Md_of(state):
    ps_dry = C.dry_surface_pressure(state.p_s.data, state.tracers, sig)
    return float(jnp.sum(ps_dry * area) / constants.g)


def Mtot_of(state):
    return float(jnp.sum(state.p_s.data * area) / constants.g)


STAGE = {"phys": [], "fix": [], "pos": [], "psphys": [], "dycore": [], "pd_q_v": [], "pd_q_c": [], "pd_q_r": []}


def _cb(key):
    def f(v):
        STAGE[key].append(float(v))
    return f


_orig_pwm = C.apply_physics_water_mass
_orig_shift = C.shift_ps_keep_tracer_mass
_orig_pos = C.apply_water_positivity


def pwm(tracers, tend, p_s, sigma_coord, dt, **kw):
    out, p_s_new = _orig_pwm(tracers, tend, p_s, sigma_coord, dt, **kw)
    dW = (_water(out, sigma_coord.layer_thickness_dp(p_s_new))
          - _water(tracers, sigma_coord.layer_thickness_dp(p_s)))
    jax.debug.callback(_cb("phys"), dW)
    return out, p_s_new


def shift(p_s, tracers, sigma_coord, correction):
    p_s_new, out = _orig_shift(p_s, tracers, sigma_coord, correction)
    dW = (_water(out, sigma_coord.layer_thickness_dp(p_s_new))
          - _water(tracers, sigma_coord.layer_thickness_dp(p_s)))
    jax.debug.callback(_cb("fix"), dW)
    return p_s_new, out


def pos(tracers, T, dp, **kw):
    tr_out, T_out = _orig_pos(tracers, T, dp, **kw)
    dW = _water(tr_out, dp) - _water(tracers, dp)
    jax.debug.callback(_cb("pos"), dW)
    return tr_out, T_out


import legoesm.atmosphere.physics.microphysics.kessler as _K
_orig_kss = _K.kessler_step_states
KRES = []


def _kss(T, q_v, q_c0, q_r0, p_full, rho, dz, dt, *a, **kw):
    out = _orig_kss(T, q_v, q_c0, q_r0, p_full, rho, dz, dt, *a, **kw)
    T1, qv1, qc1, qr1, pr = out[:5]
    res = jnp.sum(((qv1 + qc1 + qr1) - (q_v + q_c0 + q_r0)) * rho * dz, -1) + pr * dt
    jax.debug.callback(lambda v: KRES.append(float(np.sum(np.asarray(v).reshape(-1) * area_np))), res)
    return out


_K.kessler_step_states = _kss
C.apply_physics_water_mass = pwm
C.shift_ps_keep_tracer_mass = shift
C.apply_water_positivity = pos

n0 = len(MPAS_SFC_DIAG_BASE_KEYS)
SL = {k: i for i, k in enumerate(MPAS_SFC_DIAG_BASE_KEYS)}
SL.update({k: n0 + i for i, k in enumerate(MPAS_SFC_DIAG_EXTRA_KEYS)})
rows = []
MINDP = []
orig_step = model.step


_PFW = {}


def _wrap_pf(pf):
    """Water change of the physics' own p_s tendency (applied BEFORE
    apply_physics_water_mass, so the stage split books it under dyn+rest),
    and the dycore's own water change (input state -> post-integrator)."""
    if pf is None:
        return None
    if id(pf) not in _PFW:
        def w(s, mesh, sc, *aa, **kk):
            pr = pf(s, mesh, sc, *aa, **kk)
            pt = pr[0] if type(pr) is tuple else pr
            ps = s.p_s.data
            dW = (_water(s.tracers, sc.layer_thickness_dp(ps + DT * pt.dp_s_dt.data))
                  - _water(s.tracers, sc.layer_thickness_dp(ps)))
            jax.debug.callback(_cb("psphys"), dW)
            dpp = sc.layer_thickness_dp(ps)
            for nm in ("q_v", "q_c", "q_r"):
                if nm in s.tracers:
                    jax.debug.callback(_cb("pd_" + nm), _water({nm: s.tracers[nm]}, dpp))
            return pr
        _PFW[id(pf)] = w
    return _PFW[id(pf)]


def step(state, dt, *a, **k):
    if "physics_fn" in k:
        k["physics_fn"] = _wrap_pf(k["physics_fn"])
    n = {key: len(v) for key, v in STAGE.items()}
    _dpi = sig.layer_thickness_dp(state.p_s.data)
    w_in_sp = {nm: float(_water({nm: state.tracers[nm]}, _dpi)) for nm in ("q_v", "q_c", "q_r") if nm in state.tracers}
    w_in, md_in, mt_in = W_of(state), Md_of(state), Mtot_of(state)
    out = orig_step(state, dt, *a, **k)
    jax.block_until_ready(out.p_s.data)
    jax.effects_barrier()
    MINDP.append(float(jnp.min(sig.layer_thickness_dp(out.p_s.data))))
    sd = getattr(model, "_sfc_diag", None) or ()

    def flux(slot):
        i = SL[slot]
        v = sd[i] if len(sd) > i else None
        return 0.0 if v is None else float(jnp.sum(jnp.asarray(C._d(v)).reshape(-1) * area)) * dt
    st = {key: sum(STAGE[key][n[key]:]) for key in STAGE}
    for nm, v in w_in_sp.items():
        st["pd_" + nm] = st["pd_" + nm] - v
    rows.append(dict(w_in=w_in, w_out=W_of(out), md_in=md_in, md_out=Md_of(out),
                     mt_in=mt_in, mt_out=Mtot_of(out), P=flux("precip"),
                     E=flux("evap_sfc"), **st))
    return out


model.step = step
s0 = atm.state if hasattr(atm, "state") else None
status = atm.run()
print("status", status, "steps", len(rows))
R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
step_dW = R["w_out"] - R["w_in"]
gap_dW = R["w_in"][1:] - R["w_out"][:-1]
gap_dMd = R["md_in"][1:] - R["md_out"][:-1]
inside = R["phys"] + R["fix"] + R["pos"]
dyn = step_dW - inside
W0 = R["w_in"][0]
print(f"W0={W0:.6e} kg  W0/area={W0 / area_np.sum():.3f} kg/m2  Md0={R['md_in'][0]:.6e} kg")
print(f"W_end-W0={R['w_out'][-1] - W0:.6e}  Md_end-Md0={R['md_out'][-1] - R['md_in'][0]:.6e}"
      f"  Mtot_end-Mtot0={R['mt_out'][-1] - R['mt_in'][0]:.6e}")
print(f"SUM step_dW={step_dW.sum():.6e}  gap_dW(driver,between steps)={gap_dW.sum():.6e}"
      f"  gap_dMd={gap_dMd.sum():.6e}")
print(f"SUM psphys(physics dp_s_dt water)={R['psphys'].sum():.6e}  dyn_minus_psphys={(dyn - R['psphys']).sum():.6e}")
print("SUM dycore per species", {nm: f"{R['pd_' + nm].sum():.6e}" for nm in ("q_v", "q_c", "q_r")})
print("k<5 dycore q_c", R["pd_q_c"][:5], "q_v", R["pd_q_v"][:5], "q_r", R["pd_q_r"][:5])
print(f"SUM kessler own column residual (in rho*dz, area-weighted) = {sum(KRES):.6e} kg over {len(KRES)} calls")
print(f"SUM phys={R['phys'].sum():.6e}  fix={R['fix'].sum():.6e}  pos={R['pos'].sum():.6e}"
      f"  dyn+rest={dyn.sum():.6e}")
print(f"SUM E={R['E'].sum():.6e}  P={R['P'].sum():.6e}  E-P={R['E'].sum() - R['P'].sum():.6e}"
      f"  phys-(E-P)={R['phys'].sum() - (R['E'].sum() - R['P'].sum()):.6e}")
print(f"SUM step_dMd={(R['md_out'] - R['md_in']).sum():.6e}")
print(f"min layer dp over run = {min(MINDP):.3f} Pa  min p_s-proxy ok")
print("stage call counts", {k: len(v) for k, v in STAGE.items()})
for i in list(range(min(5, len(rows)))) + [len(rows) - 1]:
    r = rows[i]
    print(f"k={i:4d} dW={r['w_out'] - r['w_in']: .4e} phys={r['phys']: .4e} fix={r['fix']: .4e}"
          f" pos={r['pos']: .4e} dyn={dyn[i]: .4e} E={r['E']: .3e} P={r['P']: .3e}"
          f" dMd={r['md_out'] - r['md_in']: .3e}"
          + (f" gapW={gap_dW[i - 1]: .3e} gapMd={gap_dMd[i - 1]: .3e}" if i > 0 else ""))
print("driver q_v attr:", type(getattr(atm, "q_v", None)).__name__,
      "state tracers:", sorted(getattr(atm.state, "tracers", None) or {}))
