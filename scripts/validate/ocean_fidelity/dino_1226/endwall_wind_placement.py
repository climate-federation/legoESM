#!/usr/bin/env python
"""Measure DINO/NEMO wind source operands at the two placement entry points.

Pre-registration: ``PREREG_endwall_wind_placement.md``.  The oracle term is
not ``utrd_tau`` (that slot is allocated and emitted but never populated).
It is wired here from NEMO's own dyn_zdf bracket:

    tau_increment = zdf_dump_*1_poststress - zdf_dump_*1_prestress

The legoESM vertical-route operand is reconstructed from the production
``surface_stress_faces`` output as ``rDt*tau/(rho0*dz0)``.  The same A/B
captures the exact ``F_slow`` argument entering the production barotropic
loop, closing the fourth structural premise without re-deriving its
face-depth convention.  The post-barotropic B1 state is diagnostic only.

This probe does not run the alternative implicit placement through the
tridiagonal solve.  It therefore classifies source-operand agreement only,
not the placement response or sea-surface ownership; the registered free-run
arm is required for those questions.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parents[3]
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DINO = os.environ.get(
    "DINO_ORACLE_ROOT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO",
)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")
os.environ.setdefault("DINO_1226_IC_STEP", "5760")
os.environ.setdefault("DINO_NEMO_RUN_TWIN_STEP1",
                      os.path.join(_DINO, "RUN_D180_STEP1"))

import dump_lane  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import post_tendency_stage_birth as ptsb  # noqa: E402

KT = 5761
DT = 2700.0
RDT = 2.0 * DT
SOUTH_ROW = 1
PREREG_COMMIT = "ae0a25e191e2c5f9415c5c2e6b949d53d20bcbf6"


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(_ROOT), *args], text=True).strip()


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> dict:
    if lego.shape != nemo.shape or mask.shape != lego.shape:
        raise SystemExit(
            f"shape mismatch lego={lego.shape} nemo={nemo.shape} mask={mask.shape}")
    if not (np.isfinite(lego).all() and np.isfinite(nemo).all()):
        raise SystemExit("non-finite operand -- fatal")
    x, y = lego[mask], nemo[mask]
    if x.size < 2:
        raise SystemExit("statistic mask contains fewer than two faces")
    rn = float(np.sqrt(np.mean(y * y)))
    rl = float(np.sqrt(np.mean(x * x)))
    rd = float(np.sqrt(np.mean((x - y) ** 2)))
    corr = (float(np.corrcoef(x, y)[0, 1])
            if x.std() > 0.0 and y.std() > 0.0 else float("nan"))
    return {
        "n": int(x.size),
        "nemo_rms": rn,
        "lego_rms": rl,
        "delta_rms": rd,
        "err_norm": rd / rn if rn > 0.0 else float("nan"),
        "corr": corr,
        "rms_ratio": rl / rn if rn > 0.0 else float("nan"),
        "max_abs_delta": float(np.max(np.abs(x - y))),
    }


def _scaled_surface_forcing(sf, scale: float):
    return sf._replace(
        tau_x=jnp.asarray(sf.tau_x) * scale,
        tau_y=jnp.asarray(sf.tau_y) * scale,
    )


def _entry_identity_control(state, sf) -> dict:
    """Prove the 1x arm is identical and 0x/2x change stress fields only."""
    def exact(a, b) -> bool:
        aa = jax.tree_util.tree_leaves(a)
        bb = jax.tree_util.tree_leaves(b)
        return len(aa) == len(bb) and all(
            np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(aa, bb)
        )

    one_state = state._replace(
        tau_x_prev=jnp.asarray(state.tau_x_prev) * 1.0,
        tau_y_prev=jnp.asarray(state.tau_y_prev) * 1.0,
    )
    one_sf = _scaled_surface_forcing(sf, 1.0)
    one_exact = exact(one_state, state) and exact(one_sf, sf)
    nonstress_state_exact = True
    for scale in (0.0, 2.0):
        arm_state = state._replace(
            tau_x_prev=jnp.asarray(state.tau_x_prev) * scale,
            tau_y_prev=jnp.asarray(state.tau_y_prev) * scale,
        )
        arm_sf = _scaled_surface_forcing(sf, scale)
        for name in state._fields:
            if name not in ("tau_x_prev", "tau_y_prev"):
                nonstress_state_exact &= exact(
                    getattr(arm_state, name), getattr(state, name))
        for name in sf._fields:
            if name not in ("tau_x", "tau_y"):
                nonstress_state_exact &= exact(getattr(arm_sf, name), getattr(sf, name))
    print("CONTROL entry identity: "
          f"one_x_exact={one_exact} "
          f"nonstress_fields_exact={nonstress_state_exact}")
    if not (one_exact and nonstress_state_exact):
        raise SystemExit("entry-state identity control failed")
    return {"one_x_exact": bool(one_exact),
            "nonstress_fields_exact": bool(nonstress_state_exact)}


def _run_arm(model, state, sf, *, external_rate, scale: float) -> dict:
    """Capture the actual pre-vmix state and F_slow once for one stress scale."""
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pemod
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    cap: dict[str, np.ndarray | int] = {
        "baro_calls": 0, "vmix_calls": 0, "stress_calls": 0,
    }
    real_baro = ocmod.barotropic_substeps_latlon_cgrid
    real_vmix = LatLonCGridOceanModel._apply_implicit_vertical_mixing
    real_stress = pemod.surface_stress_faces

    def spy_stress(*args, **kwargs):
        out = real_stress(*args, **kwargs)
        cap["stress_calls"] = int(cap["stress_calls"]) + 1
        if out is not None and "tau_i_u" not in cap:
            cap["tau_i_u"] = np.asarray(out[0], dtype=np.float64)
            cap["tau_j_v"] = np.asarray(out[1], dtype=np.float64)
            cap["dz0_u"] = np.asarray(out[2], dtype=np.float64)
            cap["dz0_v"] = np.asarray(out[3], dtype=np.float64)
        return out

    def spy_baro(*args, **kwargs):
        if kwargs.get("eta_init") is not None:
            cap["baro_calls"] = int(cap["baro_calls"]) + 1
            if "F_slow_u" not in cap:
                cap["F_slow_u"] = np.asarray(kwargs["F_slow_u"], dtype=np.float64)
                cap["F_slow_v"] = np.asarray(kwargs["F_slow_v"], dtype=np.float64)
        return real_baro(*args, **kwargs)

    def spy_vmix(self, state_in, *args, **kwargs):
        if kwargs.get("do_momentum", True):
            cap["vmix_calls"] = int(cap["vmix_calls"]) + 1
            if "B1_u" not in cap:
                cap["B1_u"] = np.asarray(state_in.u.data, dtype=np.float64)
                cap["B1_v"] = np.asarray(state_in.v.data, dtype=np.float64)
        return real_vmix(self, state_in, *args, **kwargs)

    ocmod.barotropic_substeps_latlon_cgrid = spy_baro
    LatLonCGridOceanModel._apply_implicit_vertical_mixing = spy_vmix
    pemod.surface_stress_faces = spy_stress
    if state.tau_x_prev is None or state.tau_y_prev is None:
        raise SystemExit("centred wind arm requires the bridged previous stress")
    state_scaled = state._replace(
        tau_x_prev=jnp.asarray(state.tau_x_prev) * scale,
        tau_y_prev=jnp.asarray(state.tau_y_prev) * scale,
    )
    try:
        with jax.disable_jit():
            model.step(
                state_scaled, DT,
                surface_forcing=_scaled_surface_forcing(sf, scale),
                external_tracer_rate=external_rate,
            )
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = real_baro
        LatLonCGridOceanModel._apply_implicit_vertical_mixing = real_vmix
        pemod.surface_stress_faces = real_stress
    required = {"F_slow_u", "F_slow_v", "B1_u", "B1_v",
                "tau_i_u", "tau_j_v", "dz0_u", "dz0_v"}
    if not required <= cap.keys():
        raise SystemExit(f"arm {scale} missed hooks: {required - cap.keys()}")
    if (cap["baro_calls"] != 1 or cap["vmix_calls"] != 1
            or cap["stress_calls"] != 1):
        raise SystemExit(
            f"arm {scale}: expected one baro/vmix/stress hook, got "
            f"{cap['baro_calls']}/{cap['vmix_calls']}/{cap['stress_calls']}")
    return cap


def _whole_step_linearity_diagnostic(
        name: str, d1: np.ndarray, d2: np.ndarray) -> float:
    err = float(np.max(np.abs(d2 - 2.0 * d1)))
    print(f"DIAGNOSTIC {name}: whole-step max|2x-2*1x|={err:.6e} "
          "(not gated; downstream recurrence is state-dependent)")
    return err


def _stress_helper_control(arms: dict) -> dict:
    """Exact 0x/1x/2x check on the genuinely linear single-owner helper."""
    out = {}
    for key in ("tau_i_u", "tau_j_v"):
        z = np.asarray(arms[0.0][key])
        one = np.asarray(arms[1.0][key])
        two = np.asarray(arms[2.0][key])
        zero_exact = bool(np.array_equal(z, np.zeros_like(z)))
        double_exact = bool(np.array_equal(two, 2.0 * one))
        # Prove the equality gate can fail by changing one finite value by one
        # representable step and sending it through the identical comparison.
        planted = two.copy()
        idx = np.unravel_index(int(np.argmax(np.abs(one))), one.shape)
        planted[idx] = np.nextafter(planted[idx], np.inf)
        plant_fires = not np.array_equal(planted, 2.0 * one)
        print(f"CONTROL {key}: zero_exact={zero_exact} "
              f"double_exact={double_exact} planted_nextafter_fires="
              f"{plant_fires}")
        if not (zero_exact and double_exact and plant_fires):
            raise SystemExit(f"surface-stress helper control failed for {key}")
        out[key] = {"zero_exact": zero_exact,
                    "double_exact": double_exact,
                    "planted_nextafter_fires": plant_fires}
    return out


def main() -> int:
    import multistep_replay as mr
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
        dino_step_surface_forcing,
    )
    from legoesm.ocean.fidelity.precision_gate import require_fp64

    set_policy(PrecisionPolicy.fp64())
    sha = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain"])
    expected_lane = os.path.join(_DINO, "RUN_SEQDUMP_D180_1R")
    expected_step1 = os.path.join(_DINO, "RUN_D180_STEP1")
    lane_inputs = {
        "DINO_1226_LANE": os.environ.get("DINO_1226_LANE"),
        "DINO_1226_IC_STEP": os.environ.get("DINO_1226_IC_STEP"),
        "DINO_NEMO_RUN_TWIN_STEP1": os.environ.get("DINO_NEMO_RUN_TWIN_STEP1"),
        "DINO_ORACLE_ROOT": _DINO,
    }
    if lane_inputs["DINO_1226_LANE"] != "d180":
        raise SystemExit(f"expected d180 lane, got {lane_inputs}")
    if lane_inputs["DINO_1226_IC_STEP"] != "5760":
        raise SystemExit(f"expected IC step 5760, got {lane_inputs}")
    step1_actual = os.path.realpath(
        str(lane_inputs["DINO_NEMO_RUN_TWIN_STEP1"]))
    if step1_actual != os.path.realpath(expected_step1):
        raise SystemExit(f"unexpected step-1 donor: {lane_inputs}")
    if os.path.realpath(dump_lane.RUN_DIR) != os.path.realpath(expected_lane):
        raise SystemExit(
            f"unexpected oracle lane {dump_lane.RUN_DIR}; expected {expected_lane}")
    nemo_root = Path(_DINO).parents[1]
    source_paths = [
        Path(_DINO) / "MY_SRC" / name for name in
        ("dynspg_ts.F90", "dynzdf.F90", "stpmlf.F90", "usrdef_sbc.F90",
         "trddump.F90")
    ] + [
        nemo_root / "src/OCE/SBC/sbcmod.F90",
        nemo_root / "src/OCE/TRD/trddyn.F90",
        nemo_root / "src/OCE/TRD/trd_oce.F90",
    ]
    dump_names = (
        "zdf_dump_u1_prestress.bin", "zdf_dump_u1_poststress.bin",
        "zdf_dump_v1_prestress.bin", "zdf_dump_v1_poststress.bin",
        "wnd_dump_zu_frc_inc.bin", "wnd_dump_zv_frc_inc.bin",
    )
    content_sha256 = {
        str(p): _sha256(p) for p in source_paths
    }
    content_sha256.update({
        dump_lane.dump_path(name): _sha256(dump_lane.dump_path(name))
        for name in dump_names
    })
    content_sha256[str(Path(__file__).resolve())] = _sha256(Path(__file__).resolve())
    content_sha256[str(_DIR / "PREREG_endwall_wind_placement.md")] = _sha256(
        _DIR / "PREREG_endwall_wind_placement.md")
    print(json.dumps({
        "provenance": {
            "git_sha": sha,
            "git_dirty": bool(dirty),
            "prereg_commit": PREREG_COMMIT,
            "oracle_root": _DINO,
            "oracle_lane": dump_lane.RUN_DIR,
            "kt": KT,
            "dt_s": DT,
            "lane_inputs": lane_inputs,
            "content_sha256": content_sha256,
            "flags": {
                "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
                "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
                "LEGOESM_NEMO_E3T": os.environ.get("LEGOESM_NEMO_E3T"),
            },
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    }, indent=2, sort_keys=True))
    ptsb.dump_lane.banner()

    g, br, cfg, state = mr.build_replay_ic()
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    if mc.outer_integrator != "leapfrog":
        raise SystemExit(f"active path changed: outer_integrator={mc.outer_integrator!r}")
    if mc.surface_stress_implicit:
        raise SystemExit("baseline no longer uses explicit surface stress")
    require_fp64(br.geometry, br.z_coord, state,
                 context="endwall_wind_placement bridged state")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)
    if sf is None or sf.tau_x is None or sf.tau_y is None:
        raise SystemExit("wind-through-step forcing absent")

    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    external_rate = None
    if placement == "leapfrog_rhs":
        state, external_rate = apply_dino_lat_lon_surface_forcing(
            state, forcing, br.z_coord, cfg, DT,
            t_seconds=KT * DT, return_rate=True)
    else:
        state = apply_dino_lat_lon_surface_forcing(
            state, forcing, br.z_coord, cfg, DT, t_seconds=KT * DT)
    print(f"ACTIVE card outer_integrator={mc.outer_integrator!r} "
          f"surface_stress_implicit={mc.surface_stress_implicit!r} "
          f"surface_tendency_placement={placement!r}")

    entry_control = _entry_identity_control(state, sf)
    arms = {s: _run_arm(model, state, sf, external_rate=external_rate, scale=s)
            for s in (0.0, 1.0, 2.0)}
    b1u = arms[1.0]["B1_u"] - arms[0.0]["B1_u"]
    b1v = arms[1.0]["B1_v"] - arms[0.0]["B1_v"]
    b2u = arms[2.0]["B1_u"] - arms[0.0]["B1_u"]
    b2v = arms[2.0]["B1_v"] - arms[0.0]["B1_v"]
    f1u = arms[1.0]["F_slow_u"] - arms[0.0]["F_slow_u"]
    f1v = arms[1.0]["F_slow_v"] - arms[0.0]["F_slow_v"]
    f2u = arms[2.0]["F_slow_u"] - arms[0.0]["F_slow_u"]
    f2v = arms[2.0]["F_slow_v"] - arms[0.0]["F_slow_v"]
    controls = _stress_helper_control(arms)
    whole_step_diagnostics = {
        "b1_u": _whole_step_linearity_diagnostic("B1_u", b1u, b2u),
        "b1_v": _whole_step_linearity_diagnostic("B1_v", b1v, b2v),
        "fslow_u": _whole_step_linearity_diagnostic("F_slow_u", f1u, f2u),
        "fslow_v": _whole_step_linearity_diagnostic("F_slow_v", f1v, f2v),
    }

    # Reuse the campaign's registered loader and known u-face offset.
    n_pre_u = ptsb._load("zdf_dump_u1_prestress.bin")
    n_post_u = ptsb._load("zdf_dump_u1_poststress.bin")
    n_pre_v = ptsb._load("zdf_dump_v1_prestress.bin")
    n_post_v = ptsb._load("zdf_dump_v1_poststress.bin")
    n_ws_u = n_post_u - n_pre_u
    n_ws_v = n_post_v - n_pre_v
    n_fu = ptsb._load("wnd_dump_zu_frc_inc.bin")
    n_fv = ptsb._load("wnd_dump_zv_frc_inc.bin")
    um = np.asarray(g.umask, dtype=bool)[..., 0]
    vm = np.asarray(g.vmask, dtype=bool)[..., 0]

    # Direct vertical-route deposit, matching _bc_external_surface_forcing's
    # tau/(rho0*dz0) source integrated over the leapfrog rDt.  Do not use B1:
    # that state already includes the independent rotating barotropic route.
    rho0 = float(mc.constants.rho_0)
    dz0u = np.asarray(arms[1.0]["dz0_u"])
    dz0v = np.asarray(arms[1.0]["dz0_v"])
    direct_u = np.zeros_like(dz0u)
    direct_v = np.zeros_like(dz0v)
    np.divide(RDT * np.asarray(arms[1.0]["tau_i_u"]), rho0 * dz0u,
              out=direct_u, where=dz0u > 0.0)
    np.divide(RDT * np.asarray(arms[1.0]["tau_j_v"]), rho0 * dz0v,
              out=direct_v, where=dz0v > 0.0)
    # NEMO u(i) maps to lego u(i+1); v(j) maps directly on this bridge.
    lu = direct_u[:, 1:1 + n_ws_u.shape[1]]
    lv = direct_v[:n_ws_v.shape[0], :n_ws_v.shape[1]]
    lfu = np.asarray(f1u)[:, 1:1 + n_fu.shape[1]]
    lfv = np.asarray(f1v)[:n_fv.shape[0], :n_fv.shape[1]]
    masks = {
        "domain_u": um,
        "south_j1_u": um & (np.arange(um.shape[0])[:, None] == SOUTH_ROW),
        "domain_v": vm,
        "south_j1_v": vm & (np.arange(vm.shape[0])[:, None] == SOUTH_ROW),
    }
    scores = {
        "domain_u_zdf": _stats(lu, n_ws_u, masks["domain_u"]),
        "south_j1_u_zdf": _stats(lu, n_ws_u, masks["south_j1_u"]),
        "domain_u_fslow": _stats(lfu, n_fu, masks["domain_u"]),
        "south_j1_u_fslow": _stats(lfu, n_fu, masks["south_j1_u"]),
    }

    # Structural-zero v control: use max norms, since correlation/normalised
    # error are undefined against an exact zero.
    v_zero = {
        "nemo_zdf_max": float(np.max(np.abs(n_ws_v))),
        "nemo_fslow_max": float(np.max(np.abs(n_fv))),
        "lego_zdf_max": float(np.max(np.abs(lv[vm]))),
        "lego_fslow_max": float(np.max(np.abs(lfv[vm]))),
    }
    vbar = 1.0e-12
    vpass = all(x <= vbar for x in v_zero.values())
    print(f"CONTROL meridional structural zero: {v_zero} bar={vbar:.1e} "
          f"{'PASS' if vpass else 'FAIL'}")
    if not vpass:
        raise SystemExit("meridional structural-zero control failed")

    term_confirmed = (
        all(scores[k]["err_norm"] <= 0.05 and scores[k]["corr"] >= 0.999
            for k in ("domain_u_zdf", "south_j1_u_zdf"))
        and all(scores[k]["err_norm"] <= 0.05
                for k in ("domain_u_fslow", "south_j1_u_fslow"))
    )
    material_diff = (
        scores["south_j1_u_zdf"]["err_norm"] >= 0.25
        or scores["south_j1_u_fslow"]["err_norm"] >= 0.25
    )
    label = ("CONFIRMED_ALGEBRAIC_EQUIVALENCE" if term_confirmed else
             "CONFIRMED_MATERIAL_DIFF" if material_diff else
             "PLAUSIBLE_UNRESOLVED")
    out = {
        "provenance": {"git_sha": sha, "prereg_commit": PREREG_COMMIT,
                       "oracle_lane": dump_lane.RUN_DIR, "kt": KT},
        "controls": {"entry_identity": entry_control,
                     "stress_helper": controls},
        "whole_step_linearity_diagnostics": whole_step_diagnostics,
        "b1_downstream_max": {
            "u": float(np.max(np.abs(b1u))),
            "v": float(np.max(np.abs(b1v))),
        },
        "v_structural_zero": v_zero,
        "scores": scores,
        "term_label": label,
        "ownership_label": "UNRESOLVED_REQUIRES_REGISTERED_FREE_RUN",
        "retractions": [
            "first run invalid: whole B1 is not an exactly linear stress control",
            "second run invalid: 0x/2x failed to scale the previous centred-stress carry",
            "third run invalid: an already halo-free oracle mask was stripped twice",
            "fourth run invalid: B1 includes the rotating barotropic response "
            "and is not the direct dyn-zdf deposit",
            "utrd_tau is not a physical zero: its dump slot is emitted but never populated",
            "NEMO wind is not implicit-only: dynspg_ts adds it independently to zu_frc",
            "the offline probe reconstructs source operands; it does not fill the oracle slot",
            "this source-operand score measures neither the implicit-solve "
            "response nor eta ownership",
        ],
    }
    print("RESULT " + json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
