"""Fortran CLUBB host loop on BOMEX at CAM's 300 s step, Fortran core (mode
fortran, CLUBB venv) vs our JAX core on the same host inputs (mode jax).
fortran: SETTINGS OUT.npz  -> dumps step-0 state/env + per-step profiles.
jax:     IN.npz OUT.npz     -> steps our core from the same step-0 state."""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np

NSTEP = int(os.environ.get("NSTEP", 72))
DT = float(os.environ.get("DT", 300.0))
GEN = Path(os.environ["WT"]) / "scripts/validate/oracle/clubb_core/gen_clubb_core_fixture.py"
REC = ("rtm", "thlm", "wprtp", "wpthlp", "wp2", "rcm", "cloud_frac")


def fortran(settings, out):
    spec = importlib.util.spec_from_file_location("gen", GEN)
    gen = importlib.util.module_from_spec(spec); spec.loader.exec_module(gen)
    root = Path(os.environ["CLUBB_ROOT"])
    sys.path[:0] = [str(root), str(root / "install/latest/python")]
    import subprocess, tempfile
    from clubb_python import clubb_api
    from clubb_python.derived_types.config_flags import ConfigFlags
    from clubb_python.derived_types.nu_vert_res_dep import NuVertResDep
    from clubb_python.derived_types.sclr_idx import SclrIdx
    import clubb_python_driver.advance_clubb_to_end as A
    from clubb_python_driver.clubb_case_initalization import init_clubb_case, read_namelist
    cam_params = json.loads((settings / "cam_params.json").read_text())
    cam_flags = json.loads((settings / "cam_flags.json").read_text())
    nldir = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, str(root / "utilities/create_case_namelist.py"),
                    "-output_dir", str(nldir), "-override", "l_stats=.false.", "bomex"],
                   check=True, cwd=root / "run_scripts")
    nl = str(nldir / "bomex.in")
    st = init_clubb_case(nl); st["l_stats"] = False
    cfg = read_namelist(nl)
    flags = st["flags"]._asdict()
    for k, v in cam_flags.items():
        if k in flags:
            flags[k] = type(flags[k])(v)
    flags["l_do_expldiff_rtm_thlm"] = True
    flags = ConfigFlags(**flags); clubb_api.init_config_flags(flags); st["flags"] = flags
    names = [n.strip() for n in clubb_api.get_param_names()]
    lower = {k.lower(): v for k, v in cam_params.items()}
    cp = np.array(st["clubb_params"], dtype=np.float64, order="F")
    for j, n in enumerate(names):
        cp[:, j] = lower[n.lower()]
    st["clubb_params"] = cp
    nu, lmin, mfm = clubb_api.calc_derived_params(
        gr=st["gr"], ngrdcol=st["ngrdcol"], grid_type=int(cfg["grid_type"]),
        deltaz=np.full(st["ngrdcol"], float(cfg["deltaz_nl"])), clubb_params=cp,
        nu_vert_res_dep=NuVertResDep.zeros(st["ngrdcol"]), l_prescribed_avg_deltaz=False)
    st["nu_vert_res_dep"], st["lmin"], st["mixt_frac_max_mag"] = nu, lmin, mfm
    st["l_calc_thlp2_rad"] = False
    ng, nzt = st["thlm"].shape
    st["edsclr_dim"] = 2
    st["sclr_idx"] = SclrIdx(iisclr_rt=-1, iisclr_thl=-1, iisclr_CO2=-1,
                             iiedsclr_rt=2, iiedsclr_thl=1, iiedsclr_CO2=-1)
    clubb_api.set_sclr_idx(-1, -1, -1, 2, 1, -1)
    st["edsclrm"] = np.zeros((ng, nzt, 2), order="F")
    st["edsclrm_forcing"] = np.zeros((ng, nzt, 2), order="F")
    st["wpedsclrp_sfc"] = np.zeros((ng, 2), order="F")
    st["dt_main"] = DT
    res = {"zm": np.array(st["gr"].zm), "zt": np.array(st["gr"].zt),
           "case_dt": float(cfg.get("dt_main", np.nan)) if hasattr(cfg, "get") else np.nan}
    for it in range(1, NSTEP + 1):
        A._calculate_thvm(st)
        A._prescribe_forcings(st, it, l_sample=False)
        st["rtm_forcing"] = st["rtm_forcing"] + st["rcm_mc"]
        st["thlm_forcing"] = st["thlm_forcing"] + st["thlm_mc"] + st["radht"]
        st["dt_main"] = DT; st["l_sample"] = False
        st["edsclrm_forcing"] = np.zeros_like(st["edsclrm_forcing"], order="F")
        st["wpedsclrp_sfc"] = np.zeros_like(st["wpedsclrp_sfc"], order="F")
        if it == 1:
            for k in [k for k in gen.STATE + gen.CARRIED + gen.ENV if k in st]:
                res[f"s0/{k}"] = np.array(st[k], dtype=np.float64)
        for k in gen.ENV:
            res[f"env{it}/{k}"] = np.array(st[k], dtype=np.float64)
        A._advance_clubb_core_api(st)
        if np.any(np.array(st["err_info"].err_code) != 0):
            raise SystemExit(f"err at {it}")
        for k in REC:
            res[f"F{it}/{k}"] = np.array(st[k], dtype=np.float64)
    np.savez(out, **res)
    print("wrote", out, "nzt", nzt, "dz", float(cfg["deltaz_nl"]))


def jax_arm(inp, out):
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    spec = importlib.util.spec_from_file_location(
        "orc", Path(os.environ["WT"]) / "tests/atmosphere/test_clubb_core_oracle.py")
    T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
    C = T.C
    d = dict(np.load(inp))
    fx = {f"x/{k}": v for k, v in d.items() if not k.startswith(("s0/", "env", "F"))}
    fx.update({f"x/{k[3:]}": v for k, v in d.items() if k.startswith("s0/")})
    gr, forcing, env = T._inputs(fx, "x")
    st = C.CLUBBMomentState(**{k: jnp.asarray(fx[f"x/{k}"]) for k in C.CLUBBMomentState._fields})
    rcm = jnp.asarray(fx["x/rcm"])
    res = {}
    for it in range(1, NSTEP + 1):
        e = {k: d[f"env{it}/{k}"] for k in ("rtm_forcing", "thlm_forcing", "wprtp_sfc", "wpthlp_sfc", "upwp_sfc", "vpwp_sfc")}
        for k in ("rtm_forcing", "thlm_forcing"):
            dd = np.max(np.abs(e[k] - d[f"env1/{k}"])) / max(np.max(np.abs(d[f"env1/{k}"])), 1e-300)
            if it in (2, NSTEP):
                print("forcing drift", it, k, dd, flush=True)
        forcing = forcing._replace(rtm=jnp.asarray(e["rtm_forcing"]), thlm=jnp.asarray(e["thlm_forcing"]))
        sfc = {v: jnp.asarray(e[f"{v}_sfc"]) for v in ("wprtp", "wpthlp", "upwp", "vpwp")}
        st = st._replace(**{v: getattr(st, v).at[..., 0].set(sfc[v].reshape(getattr(st, v)[..., 0].shape)) for v in sfc})
        o = T._step(st, rcm, forcing, env, gr, T._CFG)
        st = C.CLUBBMomentState(**{k: o[k] for k in C.CLUBBMomentState._fields})
        rcm = o["rcm"]
        for k in REC:
            res[f"J{it}/{k}"] = np.asarray(o[k])
    np.savez(out, **res)
    print("wrote", out)


if __name__ == "__main__":
    (fortran if sys.argv[1] == "fortran" else jax_arm)(Path(sys.argv[2]), Path(sys.argv[3]))
