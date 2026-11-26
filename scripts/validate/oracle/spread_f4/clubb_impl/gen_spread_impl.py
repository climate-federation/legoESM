"""Generate the CLUBB-core oracle fixture by RUNNING the Fortran CLUBB core.

Stage 2 of the fixture (run with the clubb_release uv venv python, after
``build.sh`` and ``dump_cam_settings.py``).  Oracle: larson-group/clubb_release
@ 54403a04390fc4a0d4865357556e77c1ed6ad513, ``advance_clubb_core`` through its
f2py API, constants patched to legoESM's (constants_legoesm.patch).

For each benchmark case (namelists from CLUBB's own create_case_namelist.py):
  1. init with CLUBB's Python driver; parameters = legoESM ``CLUBBParams``;
     flags = legoESM's CAM flag table (cam_flags.json); ``calc_derived_params``;
  2. spin up with the Fortran core at the case dt (l_do_expldiff_rtm_thlm off:
     the eddy-scalar slots it needs are set up only for the recorded steps);
  3. prescribe the next forcings, set the surface fluxes to the state's
     level-0 values (so the carried closure is consistent with the boundary
     condition), switch on l_do_expldiff_rtm_thlm (edsclr_dim = 2, zero
     forcing and surface flux on the copies, as CAM), and record N_STEPS
     consecutive 300 s Fortran steps: states s0..sN (prognostic moments plus
     the carried post-advance closure outputs and Lscale/Kh_zm).
  ``<case>_dryut``: s0 of the base case with rtm x1e-3 in the top 40% of the
  column, one warm-up Fortran step, then N_STEPS recorded steps (exercises the
  large-rtp2 clip).

With ``CLUBB_FIXTURE_IMPLEMENTED=1`` (``run.sh implemented``) every warm-up and
recorded step calls the core with ``l_implemented = .true.``, as CAM does
(clubb_intr.F90:113 cam_cesm2_1_rel_60), on IMPLEMENTED_CASES (three with
subsidence wm, one without), plus ``<case>_bigup2``: s0 with wp2/up2/vp2
scaled by BIGUP2_VAR and a linear shear BIGUP2_SHEAR [1/s] added to um and vm
(production strong enough that the up2/vp2 <= 1000 cap binds), one warm-up
step (closure consistent with the new state), then N_STEPS recorded steps.
Spin-up stays standalone.

The forcings/environment are held fixed over the recorded steps (asserted).
Output: one compressed npz, keys ``<case>/<field>`` and ``<case>/s<n>/<field>``,
plus ``meta/*`` provenance.

Usage (from clubb_release/run_scripts):
  CLUBB_ROOT=<clubb_release> python gen_clubb_core_fixture.py SETTINGS_DIR OUT.npz
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ORACLE_URL = "https://github.com/larson-group/clubb_release"
ORACLE_SHA = "54403a04390fc4a0d4865357556e77c1ed6ad513"
DT = 300.0          # CAM clubb_timestep [s]
N_STEPS = 3
# case -> (spin-up steps at the case dt, namelist overrides, base namelist)
CASES = {
    "bomex": (180, [], "bomex"),
    "dycoms2_rf01": (120, [], "dycoms2_rf01"),
    "arm": (240, [], "arm"),
    "wangara": (180, [], "wangara"),
    "mpace_b": (120, ["microphys_scheme='none'"], "mpace_b"),
    "bomex_dz100": (90, ["deltaz_nl=100.0", "nzmax=30"], "bomex"),
    "arm_dz160": (120, ["deltaz_nl=160.0", "nzmax=28"], "arm"),
}
DRYUT_CASES = ("bomex", "arm")
IMPLEMENTED = os.environ.get("CLUBB_FIXTURE_IMPLEMENTED") == "1"
IMPLEMENTED_CASES = ("bomex", "dycoms2_rf01", "mpace_b", "arm")
BIGUP2_CASES = ("bomex",)
# Test-column design (not physics): smaller values tried first did not reach
# the cap (factor 1000 / shear 0.3-0.5 1/s bind only on later steps; wind
# zigzags or a warm-up with scaled variances alone never bind).
BIGUP2_VAR = 1000.0      # wp2/up2/vp2 multiplier
BIGUP2_SHEAR = 1.0       # added d(um)/dz = d(vm)/dz [1/s]
STATE = ("rtm", "thlm", "um", "vm", "wp2", "wp3", "up2", "vp2", "wprtp",
         "wpthlp", "upwp", "vpwp", "rtp2", "thlp2", "rtpthlp")
CARRIED = ("rcm", "cloud_frac", "wpthvp", "wp2thvp", "rtpthvp", "thlpthvp",
           "Lscale", "Kh_zm")
# Post-advance PDF-closure components (Fortran pdf_params on zt) and CAM's
# qclvar_out, recorded only with CLUBB_FIXTURE_PDF=1 on a build patched by
# clubb_mg2_relvar/build_oracle.sh (the clubb_mg2_relvar oracle reads them).
PDF = ("mixt_frac", "chi_1", "chi_2", "rc_1", "rc_2", "cloud_frac_1",
       "cloud_frac_2", "stdev_chi_1", "stdev_chi_2")
ENV = ("rtm_forcing", "thlm_forcing", "um_forcing", "vm_forcing",
       "wprtp_forcing", "wpthlp_forcing", "rtp2_forcing", "thlp2_forcing",
       "rtpthlp_forcing", "exner", "p_in_Pa", "p_sfc", "thv_ds_zt",
       "thv_ds_zm", "rho_ds_zm", "rho_ds_zt", "invrs_rho_ds_zm",
       "invrs_rho_ds_zt", "wm_zt", "wm_zm", "sfc_elevation", "fcor", "ug",
       "vg", "wprtp_sfc", "wpthlp_sfc", "upwp_sfc", "vpwp_sfc", "host_dx",
       "host_dy")


# ---- SPREAD adaptation (oracle_audit/spread): perturbed copies of each
# recorded step call; SPREAD_MODE=pert enables it, unset = the generator as committed.
SPREAD_MODE = os.environ.get("SPREAD_MODE", "")
sys.path.insert(0, "/burg/glab/users/pg2328/oracle_audit/spread")
# USER ANSWER 2026-10-28: perturb EVERY float the generator passes to the oracle
# call (the state-dict entries read by _advance_clubb_core_api's call, including the
# float fields of the derived-type NamedTuples), independently, no recomputation.
F2PY_NAME = {"dt_main": "dt", "T0": "t0_val", "p_in_Pa": "p_in_pa"}   # state key -> f2py kwarg
SPY = {}


def call_keys(A):
    """State keys the committed call passes INTO clubb_api.advance_clubb_core."""
    import inspect
    import re
    src = inspect.getsource(A._advance_clubb_core_api)
    return sorted(set(re.findall(r"state\['(\w+)'\]", src.split("clubb_api.advance_clubb_core(", 1)[1])))


def _is_float(v):
    return v is not None and not isinstance(v, (str, bytes)) and \
        np.asarray(v).dtype.kind == "f"


def flatten(d, keys):
    """Flat {name: float value} of every float input; NamedTuple fields as key.field."""
    flat = {}
    for k in keys:
        v = d[k]
        if hasattr(v, "_fields"):
            for f in v._fields:
                if _is_float(getattr(v, f)):
                    flat[f"{k}.{f}"] = getattr(v, f)
        elif _is_float(v):
            flat[k] = v
    return flat


def apply_flat(d, pert):
    tops = {k: v for k, v in pert.items() if "." not in k}
    d.update(tops)
    nested = {}
    for k, v in pert.items():
        if "." in k:
            t, f = k.split(".", 1)
            nested.setdefault(t, {})[f] = v
    for t, fs in nested.items():
        d[t] = d[t]._replace(**fs)


def _cmp(got, want, b):
    got, want, b = np.asarray(got), np.asarray(want), np.asarray(b)
    moved = want != b
    tgt = np.where(want[moved] > b[moved], np.inf, -np.inf)
    ok = (got.dtype == want.dtype and got.shape == want.shape
          and got.tobytes() == np.ascontiguousarray(want).tobytes()
          and bool(np.all(np.nextafter(b[moved], tgt) == want[moved]))
          and np.array_equal(got[~moved], b[~moved]))
    return [bool(ok), int(moved.sum()), int(want.size)]


def c3_check(base, pert, kw, fortran_readback):
    """Top-level inputs: as handed to the compiled extension (spy).  Grid and
    nu_vert_res_dep fields: read back from Fortran module storage after the call."""
    out = {}
    for k in pert:
        if "." in k:
            if k in fortran_readback:
                out[k + "(fortran)"] = _cmp(fortran_readback[k], pert[k], base[k])
            continue
        nm = F2PY_NAME.get(k, k)
        out[k] = _cmp(kw[nm], pert[k], base[k]) if nm in kw else ["absent", 0, 0]
    return out


def main(settings: Path, out: Path):
    """Each case runs in its own process (the f2py module keeps global flag and
    scalar-index state between cases), then the parts are merged."""
    res = {}
    cases = IMPLEMENTED_CASES if IMPLEMENTED else tuple(CASES)
    variants = ([f"{c}_bigup2" for c in BIGUP2_CASES] if IMPLEMENTED
                else [f"{c}_dryut" for c in DRYUT_CASES])
    with tempfile.TemporaryDirectory() as tmp:
        for case in cases:
            part = Path(tmp) / f"{case}.npz"
            subprocess.run([sys.executable, __file__, str(settings), str(part),
                            case], check=True)
            res.update(np.load(part))
    res["meta/oracle"] = np.array(f"{ORACLE_URL}@{ORACLE_SHA} + constants_legoesm.patch")
    res["meta/generator"] = np.array(
        "scripts/validate/oracle/clubb_core/run.sh (build.sh; "
        "dump_cam_settings.py; gen_clubb_core_fixture.py)")
    res["meta/cam_flags"] = np.array(json.dumps(
        json.loads((settings / "cam_flags.json").read_text()), sort_keys=True))
    res["meta/cam_params"] = np.array(json.dumps(
        json.loads((settings / "cam_params.json").read_text()), sort_keys=True))
    res["meta/dt"] = np.array(DT)
    res["meta/n_steps"] = np.array(N_STEPS)
    res["meta/cases"] = np.array(list(cases) + variants)
    res["meta/l_implemented"] = np.array(IMPLEMENTED)
    if IMPLEMENTED:
        res["meta/bigup2"] = np.array([BIGUP2_VAR, BIGUP2_SHEAR])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **res)
    print("wrote", out, os.path.getsize(out), "bytes")


def run_case(settings: Path, out: Path, case: str):
    root = Path(os.environ["CLUBB_ROOT"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "install/latest/python"))
    from clubb_python import clubb_api
    from clubb_python.derived_types.config_flags import ConfigFlags
    from clubb_python.derived_types.nu_vert_res_dep import NuVertResDep
    from clubb_python.derived_types.sclr_idx import SclrIdx
    import clubb_python_driver.advance_clubb_to_end as A
    from clubb_python_driver.clubb_case_initalization import (
        init_clubb_case, read_namelist)

    import clubb_f2py
    _real_adv = clubb_f2py.f2py_advance_clubb_core

    def _spy(*a, **kw):      # records the float inputs handed to the extension
        SPY.clear()
        SPY.update({k: np.array(v, copy=True) for k, v in kw.items() if _is_float(v)})
        SPY["_all"] = {k: [str(np.asarray(v).dtype), list(np.shape(v)),
                           int(np.count_nonzero(np.asarray(v)))]
                       for k, v in kw.items() if isinstance(v, (np.ndarray, float, int))}
        return _real_adv(*a, **kw)
    clubb_f2py.f2py_advance_clubb_core = _spy

    cam_params = json.loads((settings / "cam_params.json").read_text())
    cam_flags = json.loads((settings / "cam_flags.json").read_text())
    res = {}

    def set_flags(st, expldiff):
        flags = st["flags"]._asdict()
        for k, v in cam_flags.items():
            if k in flags:
                flags[k] = type(flags[k])(v)
        flags["l_do_expldiff_rtm_thlm"] = bool(expldiff)
        flags = ConfigFlags(**flags)
        clubb_api.init_config_flags(flags)
        st["flags"] = flags
        return [k for k in cam_flags if k not in st["flags"]._asdict()]

    def set_params(st, grid_type, deltaz):
        names = [n.strip() for n in clubb_api.get_param_names()]
        lower = {k.lower(): v for k, v in cam_params.items()}
        missing = [n for n in names if n.lower() not in lower]
        extra = sorted(set(lower) - {n.lower() for n in names})
        if missing or extra:
            raise SystemExit(f"param mismatch: missing {missing} extra {extra}")
        cp = np.array(st["clubb_params"], dtype=np.float64, order="F")
        for j, n in enumerate(names):
            cp[:, j] = lower[n.lower()]
        st["clubb_params"] = cp
        nu, lmin, mfm = clubb_api.calc_derived_params(
            gr=st["gr"], ngrdcol=st["ngrdcol"], grid_type=grid_type,
            deltaz=np.full(st["ngrdcol"], deltaz), clubb_params=cp,
            nu_vert_res_dep=NuVertResDep.zeros(st["ngrdcol"]),
            l_prescribed_avg_deltaz=False)
        st["nu_vert_res_dep"], st["lmin"], st["mixt_frac_max_mag"] = nu, lmin, mfm
        st["l_calc_thlp2_rad"] = False
        return nu

    def expldiff_slots(st):
        ng, nzt = st["thlm"].shape
        st["edsclr_dim"] = 2
        st["sclr_idx"] = SclrIdx(iisclr_rt=-1, iisclr_thl=-1, iisclr_CO2=-1,
                                 iiedsclr_rt=2, iiedsclr_thl=1, iiedsclr_CO2=-1)
        clubb_api.set_sclr_idx(-1, -1, -1, 2, 1, -1)
        st["edsclrm"] = np.zeros((ng, nzt, 2), order="F")
        st["edsclrm_forcing"] = np.zeros((ng, nzt, 2), order="F")
        st["wpedsclrp_sfc"] = np.zeros((ng, 2), order="F")

    def _advance(st):   # the committed step()'s core call, shared with spread_step
        if IMPLEMENTED:
            # _advance_clubb_core_api hard-codes l_implemented=False (the SCM
            # driver); CAM passes .true.
            orig = clubb_api.advance_clubb_core
            clubb_api.advance_clubb_core = (
                lambda *a, **k: orig(*a, **dict(k, l_implemented=True)))
            try:
                A._advance_clubb_core_api(st)
            finally:
                clubb_api.advance_clubb_core = orig
        else:
            A._advance_clubb_core_api(st)

    def step(st):
        st["dt_main"] = DT
        st["l_sample"] = False
        _advance(st)
        err = np.array(st["err_info"].err_code)
        if np.any(err != 0):
            raise SystemExit(f"oracle err_code {err}")

    def record(name, st, n):
        for k in STATE + CARRIED:
            res[f"{name}/s{n}/{k}"] = np.array(st[k], dtype=np.float64)
        if os.environ.get("CLUBB_FIXTURE_PDF") == "1":
            for k in PDF:
                res[f"{name}/s{n}/pdf/{k}"] = np.array(
                    getattr(st["pdf_params"], k), dtype=np.float64)
            if n > 0:   # CAM's qclvar_out (clubb_mg2_relvar/patch_qclvar_out.py)
                res[f"{name}/s{n}/qclvar_out"] = np.array(st["qclvar"], dtype=np.float64)

    def record_env(name, st):
        gr = st["gr"]
        res[f"{name}/zm"] = np.array(gr.zm, dtype=np.float64)
        res[f"{name}/zt"] = np.array(gr.zt, dtype=np.float64)
        for k in ENV:
            res[f"{name}/{k}"] = np.array(st[k], dtype=np.float64)

    def spread_step(name, st, n):
        """SPREAD: 20 perturbed copies (plus a seed-0 rerun) of step n's call."""
        import perturb
        from clubb_python.derived_types.grid_class_converter import get_fortran_grid
        from clubb_python.derived_types.nu_vert_res_dep_converter import (
            get_fortran_nu_vert_res_dep)
        keys = call_keys(A)
        for sd in tuple(perturb.SEEDS) + ("rerun0",):
            d = copy.deepcopy(st)
            d["dt_main"] = DT            # as step() sets it, then perturbed
            d["l_sample"] = False
            base_vals = flatten(d, keys)
            pert_vals = perturb.perturb_inputs(base_vals, 0 if sd == "rerun0" else sd)
            # intent(inout) arrays of shape (1, n) are overwritten in place by the
            # call, so C3 compares against copies taken before it (codex r2 #1)
            intended = {k: np.array(v, copy=True) for k, v in pert_vals.items()}
            apply_flat(d, pert_vals)
            _advance(d)
            # the getter's "nzm" argument is the length of the nu arrays (ngrdcol)
            g = get_fortran_grid()
            nu = get_fortran_nu_vert_res_dep(int(np.size(d["nu_vert_res_dep"].nu1)))
            rb = {f"gr.{f}": getattr(g, f) for f in g._fields if f"gr.{f}" in intended}
            rb.update({f"nu_vert_res_dep.{f}": getattr(nu, f) for f in nu._fields
                       if f"nu_vert_res_dep.{f}" in intended})
            err = np.array(d["err_info"].err_code)
            if np.any(err != 0):
                raise SystemExit(f"oracle err_code {err} ({name} seed {sd} step {n})")
            if sd == 0 and n == 1:
                res[f"{name}/inventory"] = np.array(json.dumps(
                    {"call_keys": keys, "perturbed": sorted(pert_vals),
                     "f2py": SPY["_all"]}))
            res[f"{name}/c3/s{n}/seed{sd}"] = np.array(json.dumps(
                c3_check(base_vals, intended, SPY, rb)))
            for k in STATE + CARRIED:
                res[f"{name}/seed{sd}/s{n}/{k}"] = np.array(d[k], dtype=np.float64)

    def trajectory(name, st):
        record_env(name, st)
        record(name, st, 0)
        for n in range(1, N_STEPS + 1):
            if SPREAD_MODE == "pert":
                spread_step(name, st, n)
            step(st)
            record(name, st, n)
        for k in ENV:   # forcing / environment held fixed over the steps
            if not np.array_equal(np.array(st[k]), res[f"{name}/{k}"]):
                raise SystemExit(f"{name}: {k} changed during the recorded steps")

    run_dir = root / "run_scripts"
    n_spin, overrides, base = CASES[case]
    nldir = Path(tempfile.mkdtemp(prefix=f"clubb_nl_{case}_"))
    nldir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(root / "utilities/create_case_namelist.py"),
           "-output_dir", str(nldir), "-override", "l_stats=.false."]
    for o in overrides:
        cmd += ["-override", o]
    subprocess.run(cmd + [base], check=True, cwd=run_dir)
    nl = str(nldir / f"{base}.in")
    st = init_clubb_case(nl)
    st["l_stats"] = False
    cfg = read_namelist(nl)
    unknown = set_flags(st, expldiff=False)
    nu = set_params(st, int(cfg["grid_type"]), float(cfg["deltaz_nl"]))
    dt_case = float(st["dt_main"])
    for itime in range(1, n_spin + 1):
        A._calculate_thvm(st)
        A._prescribe_forcings(st, itime, l_sample=False)
        st["rtm_forcing"] = st["rtm_forcing"] + st["rcm_mc"]
        st["thlm_forcing"] = st["thlm_forcing"] + st["thlm_mc"] + st["radht"]
        st["l_sample"] = False
        A._advance_clubb_core_api(st)
        if (itime % int(st["dt_rad"] / dt_case) == 0) or itime == 1:
            A._advance_radiation(state=st, l_sample=False,
                                 time_current=st["time_initial"]
                                 + (itime - 1) * dt_case)
    A._calculate_thvm(st)
    A._prescribe_forcings(st, n_spin + 1, l_sample=False)
    st["rtm_forcing"] = st["rtm_forcing"] + st["rcm_mc"]
    st["thlm_forcing"] = st["thlm_forcing"] + st["thlm_mc"] + st["radht"]
    for v in ("wprtp", "wpthlp", "upwp", "vpwp"):
        st[f"{v}_sfc"] = np.asfortranarray(np.array(st[v])[:, 0])
    set_flags(st, expldiff=True)
    expldiff_slots(st)
    s0 = copy.deepcopy(st)
    trajectory(case, st)
    res[f"{case}/nu_vert_res_dep"] = np.array(
        [float(np.array(getattr(nu, f))[0])
         for f in ("nu1", "nu2", "nu6", "nu8", "nu9", "nu10")])
    res[f"meta/{case}/unknown_flags"] = np.array(unknown)
    if IMPLEMENTED and case in BIGUP2_CASES:
        d = copy.deepcopy(s0)
        zt = np.array(d["gr"].zt)
        for k in ("wp2", "up2", "vp2"):
            d[k] = np.asfortranarray(np.array(d[k]) * BIGUP2_VAR)
        for k in ("um", "vm"):
            d[k] = np.asfortranarray(np.array(d[k]) + BIGUP2_SHEAR * (zt - zt[:, :1]))
        step(d)                                   # warm-up: closure consistent
        trajectory(f"{case}_bigup2", d)
        for n in range(1, N_STEPS + 1):
            print(case, "bigup2 step", n, "levels at the 1000 cap: up2",
                  int(np.sum(res[f"{case}_bigup2/s{n}/up2"] == 1000.0)), "vp2",
                  int(np.sum(res[f"{case}_bigup2/s{n}/vp2"] == 1000.0)), flush=True)
    if not IMPLEMENTED and case in DRYUT_CASES:
        d = copy.deepcopy(s0)
        r = np.array(d["rtm"])
        r[:, int(0.6 * d["nzt"]):] *= 1.0e-3
        d["rtm"] = np.asfortranarray(r)
        step(d)                                   # warm-up: closure consistent
        trajectory(f"{case}_dryut", d)
    print(case, "nzt", int(st["nzt"]), "dt_case", dt_case, flush=True)

    np.savez(out, **res)


if __name__ == "__main__":
    if len(sys.argv) == 4:
        run_case(Path(sys.argv[1]), Path(sys.argv[2]).resolve(), sys.argv[3])
    else:
        main(Path(sys.argv[1]), Path(sys.argv[2]).resolve())
