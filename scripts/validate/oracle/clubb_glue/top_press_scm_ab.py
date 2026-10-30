"""Offline CLUBB-top A/B: trop_cloud_top_press 15000 Pa (old production) vs 100 Pa (CAM6).

Usage: top_press_scm_ab.py CHECKPOINT DECK --steps N --stride S

Single-column, CPU, fp64.  Columns are every S-th cell of an MPAS production
checkpoint (state, cloud liquid and the carried CLUBB moments).  Each arm calls
the production entry ``clubb_turbulence_prognostic`` (glue + core + fixer) N
times with the production CLUBBConfig built from DECK, the macmic sub-step
length (physics dt / cld_macmic_num_steps), surface fluxes prescribed to zero
(same in both arms) and no other physics; the state is updated with the
returned tendencies and heights/density re-derived each call.  The ONLY
difference between the arms is trop_cloud_top_press.

Reports per arm: first non-finite call, max |dT/dt| [K/day] (all levels and
p < 150 hPa), cloud fraction at p < 150 hPa (max over the run, and mean at the
last call), and the largest |T - T0| at p < 150 hPa.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

_P_CUT = 15000.0     # the pre-2026-10-20 production top (A/B arm 1); diagnostics split at it [Pa]
_S_PER_DAY = 86400.0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("deck")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--stride", type=int, required=True)
    ap.add_argument("--mesh-level", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())   # production deck: precision fp64
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.atmosphere.physics._shared import compute_heights_from_sigma, compute_rho
    from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: N812
    from legoesm.atmosphere.physics.turbulence.integration import materialize_sub_config
    from legoesm.driver.physics_pipeline import turbulence_config_for
    from legoesm.grids.vertical import make_cam6_l32_levels
    from legoesm.grids.voronoi import create_voronoi_mesh, reconstruct_cell_velocity
    from scripts.run import run_amip

    from legoesm.driver.run_config_yaml import load_yaml_config
    # The --config path run_amip.main takes: YAML keys become parser defaults.
    # The checkpoint's own run: its recorded command line, with the deck given
    # here (normally the same file at this checkout).
    import shlex
    man = json.loads((Path(a.checkpoint).parent / "run_manifest.json").read_text())
    argv = shlex.split(man["run"]["command_line"])[1:]
    argv[argv.index("--config") + 1] = a.deck
    argv = [x for x in argv if x != "--distributed"]
    parser = run_amip.build_arg_parser()
    keys = load_yaml_config(a.deck, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    cfg = run_amip.build_config_from_args(run_amip._postprocess_args(
        parser.parse_args(argv), parser))
    # The rebuilt run must be the recorded one wherever CLUBB can see it.
    rc = man["config"]["resolved_config"]
    if man["config"]["params_applied"]:
        raise SystemExit("run applied --params overrides; not reconstructed here")
    rebuilt = cfg._asdict()
    keys = [k for k in rc if k.startswith(("clubb_", "surface_"))] + [
        "turbulence", "cld_macmic_num_steps", "physics_update_steps", "precision"]
    bad = {k: (rc[k], rebuilt.get(k)) for k in keys if rc[k] != rebuilt.get(k)}
    for k in ("dt", "discretization", "column_lane"):
        if getattr(cfg.dycore, k) != rc["dycore"].get(k, getattr(cfg.dycore, k)):
            bad[f"dycore.{k}"] = (rc["dycore"][k], getattr(cfg.dycore, k))
    if cfg.grid.grid_type != rc["grid"]["grid_type"]:
        bad["grid.grid_type"] = (rc["grid"]["grid_type"], cfg.grid.grid_type)
    # An override replaces the whole turbulence config and is not serialised.
    if cfg.turbulence_override is not None:
        bad["turbulence_override"] = ("unserialised", "set")
    # The probe always hands CLUBB the host liquid (production MPAS lane).
    if not cfg.clubb_liquid_partition:
        bad["clubb_liquid_partition"] = (rc["clubb_liquid_partition"], False)
    if bad:
        raise SystemExit(f"rebuilt config differs from the run's: {bad}")
    ccfg = materialize_sub_config(turbulence_config_for(cfg)).clubb
    if not ccfg.prognostic or float(ccfg.trop_cloud_top_press) not in (_P_CUT, 100.0):
        raise SystemExit(f"deck CLUBB is not the production one: {ccfg.prognostic=}, "
                         f"{ccfg.trop_cloud_top_press=}")
    phys_dt = cfg.dycore.dt * cfg.physics_update_steps
    dt = phys_dt / cfg.cld_macmic_num_steps

    d = np.load(a.checkpoint, allow_pickle=True)
    vg = np.asarray(d["meta_vgrid"], dtype=np.float64)
    T_all = np.asarray(d["T"], dtype=np.float64)
    ncell = T_all.shape[0]
    mesh = create_voronoi_mesh(a.mesh_level)
    if mesh.latCell.shape[0] != ncell:
        raise SystemExit(f"mesh level {a.mesh_level} has {mesh.latCell.shape[0]} "
                         f"cells, checkpoint {ncell}")
    u_all, v_all = (np.asarray(x) for x in
                    reconstruct_cell_velocity(jnp.asarray(d["u"]), mesh))
    pick = np.arange(0, ncell, a.stride)
    g = lambda k: jnp.asarray(np.asarray(d[k], dtype=np.float64)[pick])  # noqa: E731
    T0, qv0, qc0, ps = g("T"), g("trc_q_v"), g("trc_q_c"), g("p_s")
    u0, v0 = jnp.asarray(u_all[pick]), jnp.asarray(v_all[pick])
    mom0 = g("physstate_clubb_moments")
    lat = jnp.asarray(np.asarray(mesh.latCell)[pick])
    coord = make_cam6_l32_levels()
    p_half = coord.pressure_at_half(ps)
    p_full = coord.pressure_at_full(ps)
    ph_ck = vg[0][None, :] * constants.p_ref + vg[1][None, :] * np.asarray(ps)[:, None]
    if not np.allclose(np.asarray(p_half), ph_ck, rtol=1e-12, atol=0.0):
        raise SystemExit("checkpoint vertical grid is not CAM6 L32")
    for k, x in dict(T=T0, qv=qv0, qc=qc0, u=u0, v=v0, mom=mom0).items():
        if not np.all(np.isfinite(np.asarray(x))):
            raise SystemExit(f"non-finite checkpoint field {k}")
    ncol = T0.shape[0]
    zero = jnp.zeros((ncol,))
    upper = np.asarray(p_full) < _P_CUT

    def make_step(top):
        c = ccfg._replace(trop_cloud_top_press=top)

        @jax.jit
        def step(u, v, T, qv, qc, mom):
            z_full, z_half = compute_heights_from_sigma(T, p_half, q_v=qv)
            rho = compute_rho(T, p_full, q_v=qv)
            o, mom_n = C.clubb_turbulence_prognostic(
                u, v, T, qv, mom, p_full, p_half, z_full, z_half, T[:, -1],
                qv[:, -1], rho, dt, c, sfc_wpthlp=zero, sfc_wprtp=zero,
                sfc_upwp=zero, sfc_vpwp=zero, q_c=qc, lat=lat)
            return (u + dt * o.du_dt, v + dt * o.dv_dt, T + dt * o.dT_dt,
                    qv + dt * o.dq_v_dt, qc + dt * o.dq_c_dt, mom_n,
                    o.dT_dt, o.cloud_fraction)
        return step

    res = {}
    for top in (_P_CUT, 100.0):
        step = make_step(top)
        s = (u0, v0, T0, qv0, qc0, mom0)
        r = dict(first_nonfinite_call=None, max_abs_dTdt_K_day=0.0,
                 max_abs_dTdt_above150_K_day=0.0, max_cf_above150=0.0)
        for n in range(a.steps):
            *s, dT, cf = step(*s)
            dT, cf, T = np.asarray(dT), np.asarray(cf), np.asarray(s[2])
            if not all(np.all(np.isfinite(np.asarray(x))) for x in (*s, dT, cf)):
                r["first_nonfinite_call"] = n
                break
            r["max_abs_dTdt_K_day"] = max(r["max_abs_dTdt_K_day"],
                                          float(np.abs(dT).max()) * _S_PER_DAY)
            r["max_abs_dTdt_above150_K_day"] = max(
                r["max_abs_dTdt_above150_K_day"],
                float(np.abs(dT[upper]).max()) * _S_PER_DAY)
            r["max_cf_above150"] = max(r["max_cf_above150"], float(cf[upper].max()))
        if r["first_nonfinite_call"] is None:
            r["mean_cf_above150_last"] = float(cf[upper].mean())
            r["max_abs_T_change_above150_K"] = float(
                np.abs(T - np.asarray(T0))[upper].max())
            r["max_abs_T_change_K"] = float(np.abs(T - np.asarray(T0)).max())
        res[str(int(top))] = r
        print(top, json.dumps(r), flush=True)

    root = Path(__file__).resolve().parents[4]
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    meta = dict(git=sha, checkpoint=str(Path(a.checkpoint).resolve()),
                deck=a.deck, steps=a.steps, stride=a.stride, ncol=int(ncol),
                dt_s=dt, columns_with_p_below_cut=int(upper.any(axis=1).sum()))
    Path(a.out).write_text(json.dumps(dict(meta=meta, arms=res), indent=1))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
