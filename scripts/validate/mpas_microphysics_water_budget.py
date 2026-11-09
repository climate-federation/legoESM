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

argv: DAYS [default 1], ARM in {full, dyn} (dyn = radiation/turbulence/
microphysics none)."""
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
from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig

DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
ARM = sys.argv[2] if len(sys.argv) > 2 else "full"
MICRO = sys.argv[3] if len(sys.argv) > 3 else "kessler"
TURB = sys.argv[4] if len(sys.argv) > 4 else "louis"
EXTRA = dict(kv.split("=", 1) for kv in sys.argv[5:])
VCOORD = EXTRA.pop("vertical_coord", "hybrid")
# Attribution arms only: patch a module constant of the WRF Thompson port,
# e.g. patch_thompson=_W_CLOUD_SED:-1e30 (read at trace time).
for _pv in filter(None, EXTRA.pop("patch_thompson", "").split(",")):
    import legoesm.atmosphere.physics.microphysics.thompson_wrf as _TW
    _n, _v = _pv.split(":")
    print("PATCH thompson_wrf", _n, getattr(_TW, _n), "->", float(_v))
    setattr(_TW, _n, float(_v))
# thompson_wrf_exact=1: run Thompson with WRF's cloud-sedimentation loss.
WRF_EXACT = EXTRA.pop("thompson_wrf_exact", "0") == "1"
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
import hashlib, subprocess  # noqa: E401,E402
_git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout  # noqa: E731
print("PROVENANCE sha", _git("rev-parse", "HEAD").strip(),
      "diff_sha256", hashlib.sha256(_git("diff", "HEAD").encode()).hexdigest()[:16],
      "argv", sys.argv[1:], "thompson_wrf_exact", WRF_EXACT,
      "cloud_sedimentation_to_precip",
      (not WRF_EXACT) and ThompsonConfig().cloud_sedimentation_to_precip)
print("RESOLVED_CONFIG", cfg)
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


STAGE = {"phys": [], "fix": [], "pos": []}


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


C.apply_physics_water_mass = pwm
C.shift_ps_keep_tracer_mass = shift
C.apply_water_positivity = pos

# Scheme level: the microphysics call's own water change on the host layer
# mass (p_half) vs its own surface precip.  resid = dW + P dt (0 = closed).
import legoesm.atmosphere.physics.microphysics.integration as MI  # noqa: E402

for k in ("micro_dW", "micro_P", "micro_Pfz", "micro_rc0A") + tuple(f"micro_d{s}" for s in WATER):
    STAGE[k] = []
_MICRO_FN = {"thompson": "thompson_microphysics", "morrison": "morrison_microphysics",
             "kessler": "kessler_microphysics"}[MICRO]
_orig_micro = getattr(MI, _MICRO_FN)


def micro(T, q_v, h, p_full, p_half, rho, dz, dt, *a, **kw):
    if WRF_EXACT:
        if a:
            a = (a[0]._replace(cloud_sedimentation_to_precip=False),) + a[1:]
        else:
            kw["config"] = kw.get("config", ThompsonConfig())._replace(
                cloud_sedimentation_to_precip=False)
    o = _orig_micro(T, q_v, h, p_full, p_half, rho, dz, dt, *a, **kw)
    mass = (p_half[:, 1:] - p_half[:, :-1]) / constants.g
    a_c = area.reshape(-1)
    if rho is not None and h.q_c is not None:
        # bottom-level cloud water [kg/m^3] x area: flux / this = mean settling speed
        jax.debug.callback(_cb("micro_rc0A"), jnp.sum(h.q_c[:, -1] * rho[:, -1] * a_c))
    tend = {"q_v": o.dq_v_dt, "q_c": o.dq_c_dt, "q_r": o.dq_r_dt, "q_i": o.dq_i_dt,
            "q_s": o.dq_s_dt, "q_g": o.dq_g_dt}
    tot = 0.0
    for s in WATER:
        if s in tend and tend[s] is not None:
            d = jnp.sum(jnp.sum(tend[s] * dt * mass, axis=-1) * a_c)
            tot = tot + d
            jax.debug.callback(_cb(f"micro_d{s}"), d)
    jax.debug.callback(_cb("micro_dW"), tot)
    jax.debug.callback(_cb("micro_P"), jnp.sum(o.precipitation.reshape(-1) * a_c) * dt)
    pf = o.precip_frozen if getattr(o, "precip_frozen", None) is not None else 0.0 * o.precipitation
    jax.debug.callback(_cb("micro_Pfz"), jnp.sum(pf.reshape(-1) * a_c) * dt)
    return o


setattr(MI, _MICRO_FN, micro)

n0 = len(MPAS_SFC_DIAG_BASE_KEYS)
SL = {k: i for i, k in enumerate(MPAS_SFC_DIAG_BASE_KEYS)}
SL.update({k: n0 + i for i, k in enumerate(MPAS_SFC_DIAG_EXTRA_KEYS)})
rows = []
MINDP = []
orig_step = model.step


def step(state, dt, *a, **k):
    n = {key: len(v) for key, v in STAGE.items()}
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
print(f"SUM phys={R['phys'].sum():.6e}  fix={R['fix'].sum():.6e}  pos={R['pos'].sum():.6e}"
      f"  dyn+rest={dyn.sum():.6e}")
print(f"SUM E={R['E'].sum():.6e}  P={R['P'].sum():.6e}  E-P={R['E'].sum() - R['P'].sum():.6e}"
      f"  phys-(E-P)={R['phys'].sum() - (R['E'].sum() - R['P'].sum()):.6e}")
print(f"SUM step_dMd={(R['md_out'] - R['md_in']).sum():.6e}")
print(f"min layer dp over run = {min(MINDP):.3f} Pa  min p_s-proxy ok")
print("stage call counts", {k: len(v) for k, v in STAGE.items()})
mdw, mp = np.sum(STAGE["micro_dW"]), np.sum(STAGE["micro_P"])
print(f"MICRO {MICRO}: SUM dW={mdw:.6e}  P*dt={mp:.6e}  frozen*dt={np.sum(STAGE['micro_Pfz']):.6e}"
      f"  resid(dW+P)={mdw + mp:.6e}")
if STAGE["micro_rc0A"]:
    rc0 = np.sum(STAGE["micro_rc0A"]) * DT
    print(f"MICRO bottom-level cloud water x area x dt = {rc0:.6e} kg s/m;"
          f"  -resid/that = {-(mdw + mp) / rc0:.4e} m/s (mean settling speed if"
          f" the residual is the bottom cloud flux; WRF-exact arm {WRF_EXACT})")
print("MICRO per-species dW:", {s: f"{np.sum(STAGE[f'micro_d{s}']):.4e}" for s in WATER})
print(f"PHYS minus MICRO (other physics water) = {R['phys'].sum() - mdw:.6e}"
      f"  E*dt = {R['E'].sum():.6e}  sfc-diag P vs micro P: {R['P'].sum():.6e} vs {mp:.6e}")
for i in list(range(min(5, len(rows)))) + [len(rows) - 1]:
    r = rows[i]
    print(f"k={i:4d} dW={r['w_out'] - r['w_in']: .4e} phys={r['phys']: .4e} fix={r['fix']: .4e}"
          f" pos={r['pos']: .4e} dyn={dyn[i]: .4e} E={r['E']: .3e} P={r['P']: .3e}"
          f" dMd={r['md_out'] - r['md_in']: .3e}"
          + (f" gapW={gap_dW[i - 1]: .3e} gapMd={gap_dMd[i - 1]: .3e}" if i > 0 else ""))
print("driver q_v attr:", type(getattr(atm, "q_v", None)).__name__,
      "state tracers:", sorted(getattr(atm.state, "tracers", None) or {}))
