"""Generate the SI3 ORCA2 lane bitwise regression-pin fixture.

SELF-REFERENCED REGRESSION PIN, NOT AN ORACLE: the stored outputs are what
``main``'s ORCA2 lane functions computed on deterministic synthetic
ORCA2-shaped inputs.  They certify nothing about NEMO; they only make any
later change to the lane's numbers visible bit for bit.

Run (compute node, never under pytest)::

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python tests/ice/fixtures/si3_orca2_regression/make_fixture.py

Writes one compressed ``<rung>.npz`` per rung (keys ``in/<leaf>`` and
``out/<leaf>``) and ``provenance.json``.  The pin test imports this module
for ``RUNGS`` (the compute functions) and ``set_pin_policy``; importing it
runs nothing.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ice import sea_ice as sea_ice_module  # noqa: E402
from legoesm.ice.config import build_si3_orca2_o2_config  # noqa: E402
from legoesm.ice.dynamics import (  # noqa: E402
    SI3CGridAEVPConfig,
    SI3CGridAEVPForcing,
    SI3CGridAEVPState,
    SI3CGridMetrics,
    SI3ORCA2FoldProgram,
    SI3ORCA2StrengthConfig,
    si3_orca2_aevp_solver,
    si3_orca2_aevp_trace,
    si3_orca2_ice_strength,
)
from legoesm.ice.si3_finalize import (  # noqa: E402
    SI3FinalizeForcing,
    SI3Fluxes,
    SI3StressInputs,
    SI3UpdateFluxForcing,
    si3_o15_step,
    si3_o16_step,
)
from legoesm.ice.si3_orca2_state import (  # noqa: E402
    SI3ORCA2AdvectionGrid,
    SI3ORCA2StateConfig,
    SI3State,
    si3_apply_state_halos_orca2,
    si3_dom_uniq_t_orca2,
    si3_lbc_lnk_orca2,
    si3_o3_istate_fill_fields,
    si3_o4_step,
    si3_var_agg,
    si3_var_glo2eqv,
)
from legoesm.ice.si3_surface import SI3SurfaceConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
NAMELISTS = REPO / "tests" / "ice" / "unit" / "data" / "orca2_sas_ice_expref"

# Voted detail choices (see the PR body): V1 grid 16x12 shrunk to 12x10 by V19
# (fixture budget), V2 seed.
SEED = 20261108
JPI, JPJ = 12, 10
NN_HLS = 2
NATURES = ("T", "U", "V", "F")

# ORCA2 namelist values (tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref).
RN_PSTAR = 2.0e4  # :77
RN_CRHG = 20.0  # :78
RN_ECC = 2.0  # :112
RN_CREEPL = 2.0e-9  # :111
NN_NEVP = 100  # :113
RN_ISHLAT = 2.0  # :57
RN_CD_IO = 5.0e-3  # :136
RN_HIMIN = 0.1  # :45
RN_AMAX_N = 0.997  # :31
LN_DRGICE_IMP = True  # namelist_ref:818
TRACE_SUBCYCLES = 4  # V7: the trace keeps every jter frame


@contextlib.contextmanager
def set_pin_policy():
    """fp64 with libm transcendentals (vote X1), restoring the previous policy."""
    previous = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    try:
        yield
    finally:
        set_policy(previous)


def state_config() -> SI3ORCA2StateConfig:
    return SI3ORCA2StateConfig()


def thd_config() -> SI3ORCA2StateConfig:
    # O16 refuses ln_icethd=False (si3_finalize.si3_ice_update_flx); V12.
    return SI3ORCA2StateConfig()._replace(ln_icethd=True)


def surface_config() -> SI3SurfaceConfig:
    return SI3SurfaceConfig.from_input_namelists(
        NAMELISTS / "namelist_ref",
        NAMELISTS / "namelist_cfg",
        NAMELISTS / "namelist_ice_ref",
        NAMELISTS / "namelist_ice_cfg",
    )


def aevp_config(n_subcycles: int) -> SI3CGridAEVPConfig:
    ice = state_config().ice_constants
    return SI3CGridAEVPConfig(
        scheme="si3_aevp",
        staggering="si3_c_grid",
        dt_s=state_config().rDt_ice,
        n_subcycles=n_subcycles,
        eccentricity=RN_ECC,
        creep_limit_s_inv=RN_CREEPL,
        strength_parameter_pa=RN_PSTAR,
        strength_decay=RN_CRHG,
        rho_snow=ice.rho_snow,
        rho_ice=ice.rho_ice,
        rho_water=constants.rho_water,
        rho_ocean=ice.rho_ocean,
        gravity=constants.g_nemo,
        rn_ishlat=RN_ISHLAT,
        halo_width=NN_HLS,
        category_count=5,
        landfast=False,
        landfast_depth_fraction=0.0,
        landfast_basal_friction_n_m3=0.0,
        landfast_relaxation_s_inv=0.0,
        landfast_tensile_fraction=0.0,
        convergence_check=0,
    )


# ---------------------------------------------------------------------------
# pytree <-> flat dict
# ---------------------------------------------------------------------------


def pack(prefix: str, tree) -> dict[str, np.ndarray]:
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {prefix + jax.tree_util.keystr(p): np.asarray(v) for p, v in leaves}


def unpack(prefix: str, template, d: dict[str, np.ndarray]):
    paths, treedef = jax.tree_util.tree_flatten_with_path(template)
    leaves = [jnp.asarray(d[prefix + jax.tree_util.keystr(p)]) for p, _ in paths]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def state_template(cfg: SI3ORCA2StateConfig, delayed: bool) -> SI3State:
    s = SI3State.zeros(JPI, JPJ, cfg)
    return s._replace(cflice_delayed=jnp.zeros(())) if delayed else s


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint64) if a.dtype == np.float64 else a


def _owned(x):
    h = NN_HLS
    return x[h:-h, h:-h]


# ---------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------


def _lbc(x, nat, sgn=1.0):
    return np.array(si3_lbc_lnk_orca2(jnp.asarray(x), nat, sgn, nn_hls=NN_HLS, ldfull=True))


def _metric(rng, nat):
    raw = rng.uniform(1.0e5, 2.2e5, (JPI, JPJ))
    out = _lbc(raw, nat)
    out[:, :NN_HLS] = raw[:, :NN_HLS]  # vote W1: keep the closed-south halo finite
    return out


def build_inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """All synthetic base fields (vote V3/V4/V11/V12/W1/W4)."""
    d: dict[str, np.ndarray] = {}
    tmask = _lbc((rng.random((JPI, JPJ)) > 0.2).astype(np.float64), "T")
    umask = _lbc(tmask * np.roll(tmask, -1, axis=0), "U")
    vmask = _lbc(tmask * np.roll(tmask, -1, axis=1), "V")
    d.update(tmask=tmask, umask=umask, vmask=vmask)
    for name, nat in (
        ("e1t", "T"), ("e2t", "T"), ("e1u", "U"), ("e2u", "U"),
        ("e1v", "V"), ("e2v", "V"), ("e1f", "F"), ("e2f", "F"),
    ):
        d[name] = _metric(rng, nat)

    # ice_istate 1-category inputs: ~20% ice-free, ~20% thin, rest 0.1-5 m.
    cls = rng.random((JPI, JPJ))
    free = cls < 0.2
    thin = (cls >= 0.2) & (cls < 0.4)
    h_i = np.where(thin, rng.uniform(0.01, 0.1, (JPI, JPJ)), rng.uniform(0.1, 5.0, (JPI, JPJ)))
    d["zht_i_ini"] = np.where(free, 0.0, h_i) * tmask
    d["zat_i_ini"] = np.where(free, 0.0, rng.uniform(0.05, 1.0, (JPI, JPJ))) * tmask
    d["zht_s_ini"] = np.where(free, 0.0, rng.uniform(0.0, 0.5, (JPI, JPJ))) * tmask
    t0 = constants.T_freeze
    d["ztm_i_ini"] = t0 - rng.uniform(1.0, 20.0, (JPI, JPJ))
    d["ztm_s_ini"] = t0 - rng.uniform(1.0, 25.0, (JPI, JPJ))
    d["zt_su_ini"] = t0 - rng.uniform(0.5, 30.0, (JPI, JPJ))
    d["zsm_i_ini"] = rng.uniform(1.0, 10.0, (JPI, JPJ))
    d["zapnd_ini"] = rng.uniform(0.0, 0.3, (JPI, JPJ))
    d["zhpnd_ini"] = rng.uniform(0.0, 0.5, (JPI, JPJ))
    d["zhlid_ini"] = rng.uniform(0.0, 0.05, (JPI, JPJ))
    d["sss_m"] = rng.uniform(30.0, 35.0, (JPI, JPJ))
    d["sst_m"] = rng.uniform(-1.9, 2.0, (JPI, JPJ))
    d["rn_amax_2d"] = np.full((JPI, JPJ), RN_AMAX_N)
    d["age_s"] = rng.uniform(0.0, 3.0e7, (JPI, JPJ, 5))

    # lbc rung fields (vote V14).
    d["lbc_2d"] = rng.uniform(-1.0, 1.0, (JPI, JPJ))
    d["lbc_4d"] = rng.uniform(-1.0, 1.0, (JPI, JPJ, 3, 2))

    # aEVP forcing (vote V11).
    lat = np.deg2rad(np.linspace(40.0, 89.0, JPJ))[None, :] * np.ones((JPI, 1))
    d["coriolis_t"] = _lbc(2.0 * constants.Omega * np.sin(lat), "T")
    d["air_u_t"] = _lbc(rng.uniform(-0.2, 0.2, (JPI, JPJ)) * tmask, "T", -1.0)
    d["air_v_t"] = _lbc(rng.uniform(-0.2, 0.2, (JPI, JPJ)) * tmask, "T", -1.0)
    d["ocean_u_u"] = _lbc(rng.uniform(-0.2, 0.2, (JPI, JPJ)) * umask, "U", -1.0)
    d["ocean_v_v"] = _lbc(rng.uniform(-0.2, 0.2, (JPI, JPJ)) * vmask, "V", -1.0)
    d["ssh_m"] = _lbc(rng.uniform(-0.5, 0.5, (JPI, JPJ)) * tmask, "T")
    d["u_ice_b"] = _lbc(rng.uniform(-0.1, 0.1, (JPI, JPJ)) * umask, "U", -1.0)
    d["v_ice_b"] = _lbc(rng.uniform(-0.1, 0.1, (JPI, JPJ)) * vmask, "V", -1.0)

    # finalize forcing (vote V12): heat +-200 W/m2, water/salt +-1e-4.
    for name in SI3Fluxes._fields:
        scale = 200.0 if name.startswith(("hfx", "qlead", "qsb", "fhld")) else 1.0e-4
        d["flx_" + name] = rng.uniform(-scale, scale, (JPI, JPJ))
    o2, o3 = (JPI - 2 * NN_HLS, JPJ - 2 * NN_HLS), (JPI - 2 * NN_HLS, JPJ - 2 * NN_HLS, 5)
    ranges = {
        "qsr_oce": (0.0, 300.0), "qns_oce": (-200.0, 200.0), "qemp_oce": (-50.0, 50.0),
        "qemp_ice": (-50.0, 50.0), "emp_oce": (-1e-4, 1e-4), "emp_ice": (-1e-4, 1e-4),
        "qns_tot": (-200.0, 200.0), "qsr_tot": (0.0, 300.0), "frq_m": (0.0, 1.0),
        "qsr_ice": (0.0, 300.0), "qns_ice": (-200.0, 200.0), "qtr_ice_bot": (0.0, 20.0),
        "qevap_ice": (-50.0, 50.0), "qcn_ice": (-100.0, 100.0), "qcn_ice_top": (-100.0, 100.0),
        "qcn_ice_bot": (-100.0, 100.0), "qml_ice": (0.0, 100.0), "qtr_ice_top": (0.0, 30.0),
        "alb_ice": (0.1, 0.85), "wfx_snw_dyn": (-1e-4, 1e-4), "cloud_fra": (0.0, 1.0),
        "qsr": (0.0, 300.0), "sfx": (-1e-4, 1e-4),
    }
    three_d = {"a_i_b", "qsr_ice", "qns_ice", "qtr_ice_bot", "qevap_ice", "qcn_ice",
               "qcn_ice_top", "qcn_ice_bot", "qml_ice", "qtr_ice_top", "alb_ice"}
    for name, (lo, hi) in ranges.items():
        d["upd_" + name] = rng.uniform(lo, hi, o3 if name in three_d else o2)
    d["upd_a_i_b"] = rng.uniform(0.0, 0.2, o3)
    d["upd_at_i_b"] = rng.uniform(0.0, 1.0, o2)
    for name in ("uu_kbb", "vv_kbb", "u_ice", "v_ice", "u_oce", "v_oce", "utau", "vtau"):
        d["str_" + name] = rng.uniform(-0.2, 0.2, (JPI, JPJ))
    d["str_tmod_io"] = rng.uniform(0.0, 0.3, (JPI, JPJ))
    d["str_taum"] = rng.uniform(0.0, 0.3, o2)
    d["str_utau_oce"] = rng.uniform(-0.2, 0.2, o2)
    d["str_vtau_oce"] = rng.uniform(-0.2, 0.2, o2)
    return d


def base_state(d: dict[str, np.ndarray], cfg: SI3ORCA2StateConfig) -> SI3State:
    """ice_istate-consistent SI3State (vote V3/W4); Prather moments zero."""
    j = {k: jnp.asarray(v) for k, v in d.items()}
    f = si3_o3_istate_fill_fields(
        zht_i_ini=j["zht_i_ini"], zht_s_ini=j["zht_s_ini"], zat_i_ini=j["zat_i_ini"],
        ztm_i_ini=j["ztm_i_ini"], ztm_s_ini=j["ztm_s_ini"], zt_su_ini=j["zt_su_ini"],
        zsm_i_ini=j["zsm_i_ini"], zapnd_ini=j["zapnd_ini"], zhpnd_ini=j["zhpnd_ini"],
        zhlid_ini=j["zhlid_ini"], tmask=j["tmask"], sss_m=j["sss_m"],
        rn_amax_2d=j["rn_amax_2d"], config=cfg,
    )
    s = SI3State.zeros(JPI, JPJ, cfg)
    s = s._replace(
        a_i=f.a_i, v_i=f.v_i, v_s=f.v_s, sv_i=f.sv_i, t_su=f.t_su, oa_i=f.a_i * j["age_s"],
        a_ip=f.a_ip, v_ip=f.v_ip, v_il=f.v_il, e_s=f.e_s, e_i=f.e_i, szv_i=f.szv_i,
        t1_ice=f.t1_ice, t_si=f.t_su,
    )
    eqv = si3_var_glo2eqv(
        v_i=s.v_i, v_s=s.v_s, sv_i=s.sv_i, a_i=s.a_i, oa_i=s.oa_i, a_ip=s.a_ip,
        v_ip=s.v_ip, v_il=s.v_il, e_s=s.e_s, e_i=s.e_i, szv_i=s.szv_i, kn=2,
        hi_max_top=s.bounds.hi_max[-1], sss_m=j["sss_m"], config=cfg,
    )
    agg = si3_var_agg(
        v_i=s.v_i, v_s=eqv.v_s, sv_i=s.sv_i, a_i=eqv.a_i, oa_i=s.oa_i, a_ip=s.a_ip,
        v_ip=s.v_ip, v_il=s.v_il, t_su=s.t_su, t_si=s.t_si, e_s=s.e_s, e_i=s.e_i,
        szv_i=s.szv_i, a_ip_eff=eqv.a_ip_eff, t_i=eqv.t_i, t_s=eqv.t_s, kn=2, config=cfg,
    )
    return s._replace(a_i=eqv.a_i, v_s=eqv.v_s, equivalent=eqv, aggregated=agg)


# ---------------------------------------------------------------------------
# rungs: each maps a flat input dict to a flat output dict
# ---------------------------------------------------------------------------


def _grid(d) -> SI3ORCA2AdvectionGrid:
    j = {k: jnp.asarray(d[k]) for k in ("e1u", "e2v", "e2u", "e1v", "e1t", "e2t", "tmask", "umask", "vmask")}
    return SI3ORCA2AdvectionGrid.from_metrics(
        e1u=j["e1u"], e2v=j["e2v"], e2u=j["e2u"], e1v=j["e1v"], e1e2t=j["e1t"] * j["e2t"],
        tmask=j["tmask"], umask=j["umask"], vmask=j["vmask"],
    )


def rung_lbc(d):
    out = {}
    for field in ("lbc_2d", "lbc_4d"):
        for nat in NATURES:
            for sgn in (1.0, -1.0):
                for full in (True, False):
                    r = si3_lbc_lnk_orca2(jnp.asarray(d[field]), nat, sgn, nn_hls=NN_HLS, ldfull=full)
                    out[f"{field}/{nat}/{sgn:+.0f}/{full}"] = np.asarray(r)
    return out


def rung_halos(d):
    cfg = state_config()
    s = unpack("state", state_template(cfg, False), d)
    out = {}
    for full in (True, False):
        out.update(pack(f"ldfull_{full}", si3_apply_state_halos_orca2(s, config=cfg, ldfull=full)))
    return out


O4_CASES = (
    # (name, input state key, rn_uice, rn_vice, apply_halos, cflice_delayed, icycle)
    ("c1", "state", 0.1, 0.05, True, None, 1),
    ("c2", "c1", 8.0, -4.0, False, None, 2),
    ("cdelay", "c1", 8.0, -4.0, False, 2.0, 3),
)


def rung_o4(d):
    cfg = state_config()
    grid = _grid(d)
    sss, sst = jnp.asarray(d["sss_m"]), jnp.asarray(d["sst_m"])
    out, states = {}, {}
    for name, src, u, v, halos, delayed, _ in O4_CASES:
        if src == "state":
            s = unpack("state", state_template(cfg, False), d)
        else:
            s = states[src]
        if delayed is not None:
            s = s._replace(cflice_delayed=jnp.asarray(delayed, dtype=jnp.float64))
        s_out, diag = si3_o4_step(
            s, grid, kt=1, sss_m=sss, sst_m=sst, rn_uice=u, rn_vice=v,
            config=cfg, apply_halos=halos,
        )
        states[name] = s_out._replace(cflice_delayed=None)
        out.update(pack(f"{name}/state", s_out))
        out.update(pack(f"{name}/diag", diag))
    return out


def _strength_cfg(smooth: bool) -> SI3ORCA2StrengthConfig:
    return SI3ORCA2StrengthConfig(ln_str_h79=True, rn_pstar=RN_PSTAR, rn_crhg=RN_CRHG, ln_str_smooth=smooth)


def rung_rheo(d):
    cfg = state_config()
    s = unpack("state", state_template(cfg, False), d)
    j = {k: jnp.asarray(v) for k, v in d.items() if not k.startswith("state")}
    z = jnp.zeros((JPI, JPJ))
    metrics = SI3CGridMetrics(
        e1t=j["e1t"], e2t=j["e2t"], e1u=j["e1u"], e2u=j["e2u"], e1v=j["e1v"], e2v=j["e2v"],
        e1f=j["e1f"], e2f=j["e2f"], area_t=j["e1t"] * j["e2t"], area_u=j["e1u"] * j["e2u"],
        area_v=j["e1v"] * j["e2v"], area_f=j["e1f"] * j["e2f"],
    )
    out = {}
    for full in (True, False):
        fold = SI3ORCA2FoldProgram(nn_hls=NN_HLS, nfd_full_when_absent=full)
        for smooth in (True, False):
            st = si3_orca2_ice_strength(
                a_i=s.a_i, v_i=s.v_i, tmask=j["tmask"], config=_strength_cfg(smooth), fold=fold
            )
            out.update(pack(f"strength/{full}/{smooth}", st))
        st = si3_orca2_ice_strength(
            a_i=s.a_i, v_i=s.v_i, tmask=j["tmask"], config=_strength_cfg(True), fold=fold
        )
        forcing = SI3CGridAEVPForcing(
            concentration_t=st.at_i,
            ice_volume_t=jnp.sum(s.v_i, axis=-1),
            snow_volume_t=jnp.sum(s.v_s, axis=-1),
            pond_volume_t=jnp.sum(s.v_ip, axis=-1),
            lid_volume_t=jnp.sum(s.v_il, axis=-1),
            air_stress_u_t=j["air_u_t"], air_stress_v_t=j["air_v_t"],
            drag_io_t=jnp.full((JPI, JPJ), RN_CD_IO),
            ocean_u_u=j["ocean_u_u"], ocean_v_v=j["ocean_v_v"], ssh_t=None,
            coriolis_t=j["coriolis_t"], tmask_t=j["tmask"], umask_u=j["umask"], vmask_v=j["vmask"],
            fast_tmask=z, depth_t=z, depth_u=z, depth_v=z,
            iceberg_tmask=z, iceberg_umask=z, iceberg_vmask=z,
        )
        state0 = SI3CGridAEVPState(u_ice_u=j["u_ice_b"], v_ice_v=j["v_ice_b"], stress1_t=z, stress2_t=z, stress12_f=z)
        kw = dict(
            strength_t=st.strength, u_ice_b=j["u_ice_b"], v_ice_b=j["v_ice_b"], ssh_m=j["ssh_m"],
            snwice_mass=s.snwice_mass, snwice_mass_b=s.snwice_mass_b, ln_ice_embd=False, fold=fold,
        )
        for diff in (False, True):
            r = si3_orca2_aevp_solver(state0, forcing, metrics, aevp_config(NN_NEVP), differentiable=diff, **kw)
            out.update(pack(f"solver/{full}/{diff}", r))
            t = si3_orca2_aevp_trace(state0, forcing, metrics, aevp_config(TRACE_SUBCYCLES), differentiable=diff, **kw)
            out.update(pack(f"trace/{full}/{diff}", t))
    return out


def rung_uniq(d):
    return {
        "fixture": np.asarray(si3_dom_uniq_t_orca2(JPI, JPJ, NN_HLS)),
        "orca2_default": np.asarray(si3_dom_uniq_t_orca2()),
    }


def _finalize_inputs(d):
    cfg = thd_config()
    s = unpack("state", state_template(cfg, False), d)
    j = {k: jnp.asarray(v) for k, v in d.items() if not k.startswith("state")}
    fluxes = SI3Fluxes(**{n: j["flx_" + n] for n in SI3Fluxes._fields})
    upd = {n: j["upd_" + n] for n in SI3UpdateFluxForcing._fields if "upd_" + n in j}
    upd.update(
        tmask=j["tmask"], smask0=_owned(j["tmask"]),
        snwice_mass=s.snwice_mass, snwice_mass_b=s.snwice_mass_b,
    )
    update = SI3UpdateFluxForcing(**upd)
    stress = SI3StressInputs(
        uu_kbb=j["str_uu_kbb"], vv_kbb=j["str_vv_kbb"], umask_k1=j["umask"], vmask_k1=j["vmask"],
        u_ice=j["str_u_ice"], v_ice=j["str_v_ice"], u_oce=j["str_u_oce"], v_oce=j["str_v_oce"],
        drag_io=jnp.full((JPI, JPJ), RN_CD_IO), at_i=s.aggregated.at_i, utau=j["str_utau"],
        vtau=j["str_vtau"], tmod_io=j["str_tmod_io"], taum=j["str_taum"],
        utau_oce=j["str_utau_oce"], vtau_oce=j["str_vtau_oce"],
    )
    return cfg, s, j, fluxes, update, stress


def rung_finalize(d):
    cfg, s, j, fluxes, update, stress = _finalize_inputs(d)
    surf = surface_config()
    out = {}
    cor = si3_o15_step(
        s, fluxes, sss_m=j["sss_m"], sst_m=j["sst_m"], rn_amax_2d=j["rn_amax_2d"],
        rn_himin=RN_HIMIN, config=cfg,
    )
    out.update(pack("o15", cor))
    owned_fluxes = SI3Fluxes(*(_owned(f) for f in cor.fluxes))
    fin = si3_o16_step(cor.state, owned_fluxes, update, sss_m=j["sss_m"], config=cfg, surface_config=surf)
    out.update(pack("o16", fin))
    forcing = SI3FinalizeForcing(
        fluxes=fluxes, update=update, stress=stress, sss_m=j["sss_m"], sst_m=j["sst_m"],
        rn_amax_2d=j["rn_amax_2d"], rn_himin=RN_HIMIN, surface_config=surf,
        ln_drgice_imp=LN_DRGICE_IMP,
    )
    returned, resp = sea_ice_module._si3_orca2_o2_step_with_trace(
        s, forcing, build_si3_orca2_o2_config(cfg), cfg.rDt_ice
    )
    # The dispatcher's O15/O16 stages are pinned through the direct calls
    # above (stored once): they must agree bit for bit, then only the stress
    # stage adds new leaves.
    same = (
        (pack("o15", resp.correction), pack("o15", cor)),
        (pack("o16", resp.final), pack("o16", fin)),
        (pack("o16.state", returned), pack("o16.state", fin.state)),
    )
    for got, want in same:
        if sorted(got) != sorted(want) or any(
            not np.array_equal(_bits(got[k]), _bits(want[k])) for k in want
        ):
            raise AssertionError("dispatcher O15/O16 differ from the direct O15/O16 calls")
    out.update(pack("dispatch/stress", resp.stress))
    return out


RUNGS = {
    "lbc": rung_lbc,
    "halos": rung_halos,
    "o4": rung_o4,
    "rheo": rung_rheo,
    "uniq": rung_uniq,
    "finalize": rung_finalize,
}


def _inputs_for(rung: str, base: dict, state: SI3State, halo_state: SI3State) -> dict:
    if rung == "lbc":
        return {k: base[k] for k in ("lbc_2d", "lbc_4d")}
    if rung == "uniq":
        return {}
    if rung == "halos":
        return pack("state", halo_state)
    masks = ("tmask", "umask", "vmask")
    keys = {
        "o4": masks + ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "sss_m", "sst_m"),
        "rheo": masks + ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f", "coriolis_t",
                         "air_u_t", "air_v_t", "ocean_u_u", "ocean_v_v", "ssh_m", "u_ice_b", "v_ice_b"),
        "finalize": masks + ("sss_m", "sst_m", "rn_amax_2d")
        + tuple(k for k in base if k.startswith(("flx_", "upd_", "str_"))),
    }[rung]
    d = {k: base[k] for k in keys}
    d.update(pack("state", state))
    return d


def main() -> None:
    rng = np.random.default_rng(SEED)
    with set_pin_policy():
        base = build_inputs(rng)
        state = base_state(base, state_config())
        # Halo rung input: base state with nonzero Prather moments everywhere so
        # the -1 sign on sx/sy is observable (fold/seam act on halos only).
        leaves, treedef = jax.tree_util.tree_flatten(state.prather)
        # Vote H1: random in category 1 (and layer 1 of the 4-D families) only.
        def _cat1(x):
            out = np.zeros(x.shape)
            idx = (slice(None), slice(None)) + (0,) * (x.ndim - 2)
            out[idx] = rng.uniform(-1.0, 1.0, (JPI, JPJ))
            return jnp.asarray(out)

        prather = jax.tree_util.tree_unflatten(treedef, [_cat1(x) for x in leaves])
        halo_state = state._replace(
            prather=prather,
            u_ice=jnp.asarray(rng.uniform(-0.2, 0.2, (JPI, JPJ))),
            v_ice=jnp.asarray(rng.uniform(-0.2, 0.2, (JPI, JPJ))),
        )
        total = 0
        for rung, fn in RUNGS.items():
            inputs = _inputs_for(rung, base, state, halo_state)
            outputs = fn(inputs)
            for k, v in outputs.items():
                if v.dtype.kind == "f" and not np.all(np.isfinite(v)):
                    raise FloatingPointError(f"{rung}:{k} is not finite")
            if rung == "o4":
                for name, *_, icycle in O4_CASES:
                    got = int(outputs[f"{name}/diag.icycle"])
                    if got != icycle:
                        raise AssertionError(f"o4 case {name}: icycle {got} != {icycle}")
            path = HERE / f"{rung}.npz"
            np.savez_compressed(
                path,
                **{"in/" + k: v for k, v in inputs.items()},
                **{"out/" + k: v for k, v in outputs.items()},
            )
            total += path.stat().st_size
            print(f"{rung}: {len(inputs)} inputs, {len(outputs)} outputs, {path.stat().st_size} B")
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    # The PR changes no model code, so the lane that produced the outputs is
    # origin/main at the merge base; the generator blob pins this script.
    main_sha = git("merge-base", "HEAD", "origin/main")
    blob = git("hash-object", "tests/ice/fixtures/si3_orca2_regression/make_fixture.py")
    prov = {
        "kind": "self-referenced regression pin, NOT an oracle",
        "generator": "tests/ice/fixtures/si3_orca2_regression/make_fixture.py",
        "origin_main_sha": main_sha,
        "generator_blob_sha": blob,
        "seed": SEED,
        "grid": [JPI, JPJ],
        "nn_hls": NN_HLS,
        "jax": jax.__version__,
        "jaxlib": __import__("jaxlib").__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "JAX_ENABLE_X64": "1",
        "platform": jax.default_backend(),
        "precision_policy": "PrecisionPolicy.fp64(transcendentals='libm')",
        "aevp_subcycles": {"solver": NN_NEVP, "trace": TRACE_SUBCYCLES},
        "o4_cases": [list(c) for c in O4_CASES],
        "total_npz_bytes": total,
    }
    (HERE / "provenance.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(f"total {total} B")


if __name__ == "__main__":
    main()
