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
       "vg", "wprtp_sfc", "wpthlp_sfc", "upwp_sfc", "vpwp_sfc")


def main(settings: Path, out: Path):
    """Each case runs in its own process (the f2py module keeps global flag and
    scalar-index state between cases), then the parts are merged."""
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
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
    res["meta/cases"] = np.array(
        list(CASES) + [f"{c}_dryut" for c in DRYUT_CASES])
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

    def step(st):
        st["dt_main"] = DT
        st["l_sample"] = False
        A._advance_clubb_core_api(st)
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

    def trajectory(name, st):
        record_env(name, st)
        record(name, st, 0)
        for n in range(1, N_STEPS + 1):
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
    if case in DRYUT_CASES:
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
