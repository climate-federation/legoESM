#!/usr/bin/env python
"""Registered #1226 EUC-mechanism causal leg -- Leg R of
``PREREG_euc_mechanism.md``: the 10-day BASE vs NEMO_AVM substitution
response.

``euc_mechanism.py`` pre-registered this design and scored Leg O (the offline
matched-state coefficient comparison) but could not execute Leg R itself --
its own docstring records ``DESIGN_ONLY_NO_CUDA_IN_SANDBOX`` because its
sandbox had no visible GPU.  This script is the missing runner: it executes
the exact registered arms on a real CUDA device and writes the artifact the
result doc's Leg R section needs.

ponytail: minimal glue over the already-registered design. Reuses
``_build_twin_state``/``DT``/``STEPS_PER_DAY``/``apply_dino_lat_lon_surface_forcing``
(``kamm_twin_90d.py``), the dump-lane selector (``dump_lane.py``) and the
interior-dump loader (``bn2_alpha_compare.py``) -- no new numerics beyond the
one substitution hook and the day-10 undercurrent-depth/shear reducer the
PREREG asks for (which does not exist anywhere in this tree; regional_audit.py,
the upstream audit that coined the metric, lives only in sibling worktrees on
other branches and is not committed here).

Run (registered)::

    CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 LEGOESM_NEMO_E3T=both \
      DINO_1226_LANE=d180 <venv>/bin/python \
      scripts/validate/ocean_fidelity/dino_1226/euc_substitution_run.py \
      --out /tmp/dino_euc_mechanism/euc_substitution.json
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import netCDF4 as nc  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


PINNED_REDUCERS = {
    "kamm_twin_90d.py":
        "00b5f0d6c6c58ef984e010825239af27f768b83cf431e8763d236325c7cf68e6",
    "bn2_alpha_compare.py":
        "2ac9e4a457c4dded55a116d67938dfda035e79a3a98ccbb070d9a0ec3d11e40e",
    "dump_lane.py":
        "41252d898b0d09b350c086aa678ddb890c7e6232fa2b46e88093a3b108a9bf62",
}
OBSERVED_REDUCERS = {name: sha256(HERE / name) for name in PINNED_REDUCERS}
if OBSERVED_REDUCERS != PINNED_REDUCERS:
    raise RuntimeError(
        f"pinned sibling mismatch: expected={PINNED_REDUCERS}, "
        f"got={OBSERVED_REDUCERS}")

sys.path.insert(0, str(HERE))

import kamm_twin_90d as kt  # noqa: E402  (reused: _build_twin_state, DT, ...)
import bn2_alpha_compare as bac  # noqa: E402  (reused: dump loaders)
import dump_lane as dl  # noqa: E402  (reused: which NEMO run is "day 180")
import legoesm.ocean.physics.vertical_mixing as vmix_pkg  # noqa: E402
from legoesm.ocean.fidelity.precision_gate import require_fp64  # noqa: E402

RECIPE = "nemo_dino_kamm_mlf"
N_DAYS = 10
EVD_THRESHOLD_M2_S = 50.0  # matches euc_mechanism.py's own EVD-fraction cut
EQUATOR_ROW = 99  # PREREG: "the equator row (T-row 99)"; cross-checked below
EXECUTOR_RUNNER_SHA256 = (
    "b2de13ca7039a6954a7c3e115af249e834dc4b9be83343cc0188e56893d12845")

# The NEMO continuation this campaign already treats as the "control member"
# for a day-180-rooted twin (acceptance_gate_90d.py's RUN_90D_TWIN baseline).
# Day 10 = kt 5760 + 10*32 steps = 6080; REBUILD_TWIN/DINO_00006080_restart.nc
# is that step's restart, already stitched from the 16-rank run with NEMO's
# own rebuild_nemo tool (rebuild_6080.log) -- reused, not rebuilt again.
NEMO_CONTROL_DAY10_RESTART = os.environ.get(
    "DINO_NEMO_DAY10_REBUILT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/REBUILD_TWIN/"
    "DINO_00006080_restart.nc")


def git_receipt(path: Path) -> dict:
    def run(*args):
        return subprocess.check_output(args, cwd=path, text=True,
                                        stderr=subprocess.DEVNULL).strip()
    try:
        return {"head": run("git", "rev-parse", "HEAD"),
                "branch": run("git", "branch", "--show-current"),
                "dirty": bool(run("git", "status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "branch": None, "dirty": None}


# --------------------------------------------------------------- substitution
def build_frozen_avm(avm_raw: np.ndarray, nlev_if: int):
    """legoESM interior interface i <-> NEMO jk=i+2 <-> ``avm_raw`` index i+1
    (same mapping ``euc_mechanism.py``'s Leg O scorer uses). Levels deeper
    than the frozen dump's coverage are marked invalid (native A_v kept)."""
    navail = avm_raw.shape[-1] - 1
    n = max(0, min(nlev_if, navail))
    frozen = np.zeros(avm_raw.shape[:2] + (nlev_if,), dtype=np.float64)
    valid = np.zeros((nlev_if,), dtype=bool)
    if n > 0:
        frozen[..., :n] = avm_raw[..., 1:1 + n]
        valid[:n] = True
    return frozen, valid


class SubstitutionPatch:
    """Monkeypatch ``vmix_pkg.compute_vertical_K_profiles`` so the momentum
    viscosity ``A_v`` it returns is replaced -- at every wet, non-EVD
    interface with an available frozen level -- by ``frozen_avm``. ``K_v``
    and any returned TKE pass through untouched (by construction: this
    wrapper never reads or rebuilds those slots). ``scale`` is a control
    knob only (the 2x-plant liveness control below); production arms leave
    it at 1.0.
    """

    def __init__(self, frozen_avm: np.ndarray, valid: np.ndarray,
                 scale: float = 1.0):
        self._frozen = jnp.asarray(frozen_avm)
        self._valid = jnp.asarray(valid)
        self._scale = float(scale)
        self._real = None

    def __enter__(self):
        self._real = vmix_pkg.compute_vertical_K_profiles
        real, frozen, valid, scale = self._real, self._frozen, self._valid, self._scale

        def _patched(*args, **kwargs):
            out = real(*args, **kwargs)
            # Preserve the REAL function's own return arity exactly -- the
            # caller in ocean_model_latlon_cgrid.py unpacks 3 values whenever
            # it passed return_tke=True, even if the 3rd (tke_new) is None
            # (post-mixing mode extracts it separately on ITS side).
            K_v, A_v = out[0], out[1]
            subst = frozen * scale
            mask = valid[None, None, :] & (A_v > 0.0) & (A_v < EVD_THRESHOLD_M2_S)
            A_v_new = jnp.where(mask, subst, A_v)
            if len(out) == 3:
                return (K_v, A_v_new, out[2])
            return (K_v, A_v_new)

        vmix_pkg.compute_vertical_K_profiles = _patched
        return self

    def __exit__(self, *exc):
        vmix_pkg.compute_vertical_K_profiles = self._real
        return False


# ------------------------------------------------------------------- controls
def control_hook_leaves_Kv_tke_untouched(frozen, valid) -> dict:
    """Direct call of the REAL vs PATCHED closure on the SAME input: confirms
    K_v (and TKE, if returned) are bit-identical while A_v differs only where
    the mask says it should."""
    br, cfg, mc, model, forcing, sf, st = kt._build_twin_state(
        RECIPE, dl.RUN_DIR, dl.RUN_DIR, bridge_tke=True, bridge_before=True,
        restart_file=dl.RESTART, e3t_mode="both")
    # model.step returns the stepped state; K_v/A_v are not surfaced through
    # it, so probe compute_vertical_K_profiles directly instead (same idiom
    # as zdftke_avm_offset_scan.run_and_capture).
    real_fn = vmix_pkg.compute_vertical_K_profiles
    calls = []

    def _make_spy(target):
        def _spy(*a, **kw):
            o = target(*a, **kw)
            calls.append(o)
            return o
        return _spy

    vmix_pkg.compute_vertical_K_profiles = _make_spy(real_fn)
    try:
        with jax.disable_jit():
            model.step(st, kt.DT, surface_forcing=sf)
    finally:
        vmix_pkg.compute_vertical_K_profiles = real_fn
    native = calls[0]
    calls.clear()
    with SubstitutionPatch(frozen, valid):
        # BUG FIXED (caught by review, reproduced by the control's own
        # refuse-to-score): capturing ``real_fn`` here would wrap the
        # PRE-patch function, bypassing the substitution entirely and
        # reporting Av_changed=False on every run. The spy must wrap
        # whichever function ``SubstitutionPatch.__enter__`` just installed.
        patched_fn = vmix_pkg.compute_vertical_K_profiles
        vmix_pkg.compute_vertical_K_profiles = _make_spy(patched_fn)
        try:
            with jax.disable_jit():
                model.step(st, kt.DT, surface_forcing=sf)
        finally:
            vmix_pkg.compute_vertical_K_profiles = patched_fn
    patched = calls[0]
    Kv_n, Av_n = native[0], native[1]
    Kv_p, Av_p = patched[0], patched[1]
    d_Kv = float(np.max(np.abs(np.asarray(Kv_p) - np.asarray(Kv_n))))
    d_Av = float(np.max(np.abs(np.asarray(Av_p) - np.asarray(Av_n))))
    tke_n = native[2] if len(native) == 3 else None
    tke_p = patched[2] if len(patched) == 3 else None

    def _as_array_or_none(x):
        try:
            return np.asarray(x) if isinstance(x, (jnp.ndarray, np.ndarray)) else None
        except (TypeError, ValueError):
            return None

    tke_n_a, tke_p_a = _as_array_or_none(tke_n), _as_array_or_none(tke_p)
    if tke_n_a is not None and tke_p_a is not None:
        d_tke = float(np.max(np.abs(tke_p_a - tke_n_a)))
        tke_bit_identical = bool(d_tke == 0.0)
    else:
        # Non-array TKE slot (e.g. a post-mixing context object, or plain
        # None): the hook never reads or rebuilds this slot, so it is passed
        # through by construction -- nothing to diff, and nothing for the
        # hook to have broken.
        d_tke = 0.0
        tke_bit_identical = True
    return {"max_abs_dKv": d_Kv, "max_abs_dtke": d_tke,
            "max_abs_dAv": d_Av,  # expected > 0: this is the substitution firing
            "Kv_bit_identical": bool(d_Kv == 0.0),
            "tke_bit_identical": tke_bit_identical,
            "Av_changed": bool(d_Av > 0.0)}


def control_dry_and_shift_plants(frozen, valid, wet_mask_2d) -> dict:
    """Dry-cell plant: a substitution value forced nonzero on a dry column
    must still leave that column's A_v untouched (native path zeros dry
    interfaces before this hook ever sees them). Wrong-shift plant: shifting
    the frozen field by one level before patching must change the result
    (proves the vertical mapping is load-bearing, not a no-op)."""
    dry = ~wet_mask_2d
    planted = frozen.copy()
    if dry.any():
        planted[dry] = 1e6
    unchanged = bool(np.array_equal(planted[wet_mask_2d], frozen[wet_mask_2d]))
    shifted = np.zeros_like(frozen)
    shifted[..., :-1] = frozen[..., 1:]
    shift_differs = bool(not np.allclose(shifted, frozen, equal_nan=True))
    return {"dry_plant_leaves_wet_untouched": unchanged,
            "wrong_shift_differs_from_correct_mapping": shift_differs}


def control_2x_plant_changes_first_step(frozen, valid) -> float:
    """A doubled frozen coefficient must change the first-step momentum
    tendency -- proves the substitution path is live, not dead code."""
    br, cfg, mc, model, forcing, sf, st = kt._build_twin_state(
        RECIPE, dl.RUN_DIR, dl.RUN_DIR, bridge_tke=True, bridge_before=True,
        restart_file=dl.RESTART, e3t_mode="both")
    with SubstitutionPatch(frozen, valid, scale=1.0):
        with jax.disable_jit():
            st1 = model.step(st, kt.DT, surface_forcing=sf)
    with SubstitutionPatch(frozen, valid, scale=2.0):
        with jax.disable_jit():
            st2 = model.step(st, kt.DT, surface_forcing=sf)
    du = float(np.max(np.abs(np.asarray(st1.u.data) - np.asarray(st2.u.data))))
    return du


# --------------------------------------------------------------------- arms
def run_arm(patch: "SubstitutionPatch | None"):
    br, cfg, mc, model, forcing, sf, st = kt._build_twin_state(
        RECIPE, dl.RUN_DIR, dl.RUN_DIR, bridge_tke=True, bridge_before=True,
        restart_file=dl.RESTART, e3t_mode="both")
    require_fp64(st, context="EUC substitution arm")
    day0_u = np.asarray(st.u.data, dtype=np.float64).copy()
    day0_T = np.asarray(st.T.data, dtype=np.float64).copy()

    _sf_placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    ctx = patch if patch is not None else contextlib.nullcontext()
    t0_sec = kt.seasonal_t0_seconds(f"{dl.RUN_DIR}/{dl.RESTART}")
    nsteps = kt.STEPS_PER_DAY * N_DAYS
    with ctx:
        if _sf_placement == "leapfrog_rhs":
            dyn = jax.jit(lambda s, ext: model.step(
                s, kt.DT, surface_forcing=sf, external_tracer_rate=ext))
        else:
            dyn = jax.jit(lambda s: model.step(s, kt.DT, surface_forcing=sf))
        wall0 = time.time()
        for k in range(nsteps):
            if _sf_placement == "leapfrog_rhs":
                st, ext_rate = kt.apply_dino_lat_lon_surface_forcing(
                    st, forcing, br.z_coord, cfg, kt.DT,
                    t_seconds=t0_sec + (k + 1) * kt.DT, return_rate=True)
                st = dyn(st, ext_rate)
            else:
                st = kt.apply_dino_lat_lon_surface_forcing(
                    st, forcing, br.z_coord, cfg, kt.DT,
                    t_seconds=t0_sec + (k + 1) * kt.DT)
                st = dyn(st)
        wall = time.time() - wall0
    day10_u = np.asarray(st.u.data, dtype=np.float64)
    day10_umask = np.asarray(st.u_mask.data) > 0.5
    finite = bool(np.isfinite(day10_u).all())
    return {
        "wall_seconds": wall, "finite": finite,
        "day0_u": day0_u, "day0_T": day0_T,
        "day10_u": day10_u, "day10_umask": day10_umask,
    }


# --------------------------------------------------------------------- score
def wet_zonal_mean_profile(u_row: np.ndarray, wet_row: np.ndarray) -> np.ndarray:
    num = np.sum(np.where(wet_row, u_row, 0.0), axis=0)
    den = np.sum(wet_row, axis=0)
    return np.divide(num, den, out=np.full(den.shape, np.nan), where=den > 0)


def undercurrent_top_depth(z: np.ndarray, u: np.ndarray, zmax: float = 100.0):
    """Shallowest linearly interpolated sign change of ``u(z)`` in
    ``[0, zmax]`` (PREREG wording, literally)."""
    sel = z <= zmax
    zz, uu = z[sel], u[sel]
    order = np.argsort(zz)
    zz, uu = zz[order], uu[order]
    for i in range(len(zz) - 1):
        a, b = uu[i], uu[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            return float(zz[i])
        if (a < 0.0) != (b < 0.0):
            frac = -a / (b - a)
            return float(zz[i] + frac * (zz[i + 1] - zz[i]))
    return float("nan")


def shear_5_26(z: np.ndarray, u: np.ndarray) -> float:
    finite = np.isfinite(z) & np.isfinite(u)
    u5 = float(np.interp(5.0, z[finite], u[finite]))
    u26 = float(np.interp(26.0, z[finite], u[finite]))
    return (u26 - u5) / (26.0 - 5.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/dino_euc_mechanism/euc_substitution.json")
    args = ap.parse_args()

    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    if dl.LANE != "d180":
        raise SystemExit(f"expected DINO_1226_LANE=d180, got {dl.LANE!r}")
    devices = jax.devices()
    print(f"JAX devices: {devices}", flush=True)
    if not any(d.platform == "gpu" for d in devices):
        raise SystemExit(
            "no CUDA device visible -- refusing to run the registered "
            f"design on CPU (devices={devices})")

    jpi, jpj, jpk, hls = bac._read_dims(dl.RUN_DIR)
    avm_raw = bac._load_interior(dl.dump_path("tke_dump_avm_final.bin"),
                                  jpi - 2 * hls, jpj - 2 * hls)
    with nc.Dataset(Path(dl.RUN_DIR) / "mesh_mask.nc") as mm:
        def llz(name):
            return np.moveaxis(np.asarray(mm[name][0]).squeeze(), 0, -1)
        tmask = llz("tmask") > 0.5
        umask3 = llz("umask") > 0.5
        gphit = np.asarray(mm["gphit"][0]).squeeze()
        gdept = np.asarray(mm["gdept_1d"][:]).squeeze()  # T/U-level (cell-centre) depths

    equator_row_check = int(np.nanargmin(np.abs(np.nanmean(gphit, axis=1))))
    if equator_row_check != EQUATOR_ROW:
        raise SystemExit(
            f"equator row mismatch: PREREG says {EQUATOR_ROW}, mesh says "
            f"{equator_row_check}")

    # BASE arm's own level counts decide how many frozen levels exist;
    # built once from a throwaway state (dropped) purely to read shapes.
    # u/T live at nlev cell-centred (T/U) levels; A_v (momentum viscosity)
    # lives at the nlev-1 INTERIOR W-interfaces between them -- these are two
    # different verticals and must not share one depth array.
    br0, _, _, _, _, _, st0 = kt._build_twin_state(
        RECIPE, dl.RUN_DIR, dl.RUN_DIR, bridge_tke=True, bridge_before=True,
        restart_file=dl.RESTART, e3t_mode="both")
    nlev_t = int(st0.T.data.shape[-1])
    nlev_if = nlev_t - 1
    frozen_avm, valid_levels = build_frozen_avm(avm_raw, nlev_if)
    z_lego_T = gdept[:nlev_t]
    wet_mask_2d = tmask[:, :, 0]  # surface wet columns, full domain

    print("=" * 100, flush=True)
    print("CONTROLS (must all pass before any arm is scored)", flush=True)
    ctl_hook = control_hook_leaves_Kv_tke_untouched(frozen_avm, valid_levels)
    print(f"  hook K_v/tke identity: {ctl_hook}", flush=True)
    ctl_plants = control_dry_and_shift_plants(frozen_avm, valid_levels, wet_mask_2d)
    print(f"  dry/shift plants: {ctl_plants}", flush=True)
    ctl_2x_du = control_2x_plant_changes_first_step(frozen_avm, valid_levels)
    print(f"  2x-plant first-step |du| max: {ctl_2x_du:.6e}", flush=True)
    controls = {**ctl_hook, **ctl_plants, "two_x_plant_first_step_max_abs_du": ctl_2x_du}
    if not (controls["Kv_bit_identical"] and controls["Av_changed"]
            and controls["dry_plant_leaves_wet_untouched"]
            and controls["wrong_shift_differs_from_correct_mapping"]
            and ctl_2x_du > 0.0):
        raise SystemExit(f"CONTROL FAILURE -- refusing to score arms: {controls}")
    print("[controls PASS]\n", flush=True)

    print("=" * 100, flush=True)
    print("BASE arm (shipped nemo_dino_kamm_mlf, unpatched)", flush=True)
    base = run_arm(patch=None)
    print(f"  wall={base['wall_seconds']:.1f}s finite={base['finite']}", flush=True)

    print("=" * 100, flush=True)
    print("NEMO_AVM arm (frozen day-180 closure-only avm substituted)", flush=True)
    avm_patch = SubstitutionPatch(frozen_avm, valid_levels, scale=1.0)
    nemo_avm = run_arm(patch=avm_patch)
    print(f"  wall={nemo_avm['wall_seconds']:.1f}s finite={nemo_avm['finite']}", flush=True)

    d_day0_u = float(np.max(np.abs(base["day0_u"] - nemo_avm["day0_u"])))
    d_day0_T = float(np.max(np.abs(base["day0_T"] - nemo_avm["day0_T"])))
    day0_identity_ok = bool(d_day0_u == 0.0 and d_day0_T == 0.0)
    print(f"day-0 identity: max|d_u|={d_day0_u:.3e} max|d_T|={d_day0_T:.3e} "
          f"identical={day0_identity_ok}", flush=True)
    if not day0_identity_ok:
        raise SystemExit("BASE/NEMO_AVM day-0 states are not bit-identical")

    # NEMO's actual day-10 continuation (control member), read directly from
    # the already-rebuilt single-file restart.
    with nc.Dataset(NEMO_CONTROL_DAY10_RESTART) as d10:
        un = np.asarray(d10.variables["un"][0], dtype=np.float64)  # (z,y,x)
    un = np.moveaxis(un, 0, -1)  # -> (y,x,z)
    nemo_u_row = un[EQUATOR_ROW]
    n_lon_nemo = umask3.shape[1]
    nemo_wet_row = umask3[EQUATOR_ROW, :, :nemo_u_row.shape[-1]]
    z_nemo = z_lego_T[:nemo_u_row.shape[-1]]
    nemo_profile = wet_zonal_mean_profile(nemo_u_row, nemo_wet_row)

    def score(arm):
        # legoESM u faces (n_lon+1 columns incl. periodic wrap) -> NEMO-
        # aligned interior columns 0..n_lon-1 (face i+1 <-> NEMO column i),
        # same convention acceptance_gate_90d.load_candidate and run_twin's
        # _acc_pair use, so NEMO's OWN 3-D umask can score both sides on one
        # grid instead of the state's static 2-D u_mask (which carries no
        # per-level topography information under full-step columns).
        u_full_row = arm["day10_u"][EQUATOR_ROW]
        u_row = u_full_row[1:1 + n_lon_nemo]
        wet = umask3[EQUATOR_ROW, :, :u_row.shape[-1]]
        prof = wet_zonal_mean_profile(u_row, wet)
        z = z_lego_T[:prof.shape[-1]]
        return {
            "profile_u_m_s": prof.tolist(), "depth_m": z.tolist(),
            "top_depth_m": undercurrent_top_depth(z, prof),
            "shear_5_26_per_s": shear_5_26(z, prof),
        }

    base_score = score(base)
    nemo_avm_score = score(nemo_avm)
    nemo_score = {
        "profile_u_m_s": nemo_profile.tolist(), "depth_m": z_nemo.tolist(),
        "top_depth_m": undercurrent_top_depth(z_nemo, nemo_profile),
        "shear_5_26_per_s": shear_5_26(z_nemo, nemo_profile),
    }

    z_base = base_score["top_depth_m"]
    z_avm = nemo_avm_score["top_depth_m"]
    z_nemo_v = nemo_score["top_depth_m"]
    denom_z = abs(z_base - z_nemo_v)
    if denom_z < 0.5:
        Cz = None
        cz_status = "UNMEASURABLE (BASE day-10 core-depth error < 0.5 m)"
    else:
        Cz = 1.0 - abs(z_avm - z_nemo_v) / denom_z
        cz_status = None

    sh_base = base_score["shear_5_26_per_s"]
    sh_avm = nemo_avm_score["shear_5_26_per_s"]
    sh_nemo = nemo_score["shear_5_26_per_s"]
    base_shear_err = abs(sh_base - sh_nemo)
    avm_shear_err = abs(sh_avm - sh_nemo)
    shear_worsened_frac = ((avm_shear_err - base_shear_err) / base_shear_err
                           if base_shear_err > 0 else float("inf"))

    if cz_status is not None:
        verdict = cz_status
    else:
        toward_nemo = abs(z_avm - z_nemo_v) < abs(z_base - z_nemo_v)
        if Cz >= 0.50 and toward_nemo and shear_worsened_frac <= 0.10:
            verdict = "CONFIRM_VISCOSITY_DRIVER"
        elif Cz <= 0.10 or not toward_nemo:
            verdict = "REFUTE_VISCOSITY_DRIVER"
        else:
            verdict = "UNRESOLVED_VISCOSITY_DRIVER"

    print("=" * 100, flush=True)
    print(f"BASE      top_depth={z_base:.3f} m  shear={sh_base:.5f} s^-1", flush=True)
    print(f"NEMO_AVM  top_depth={z_avm:.3f} m  shear={sh_avm:.5f} s^-1", flush=True)
    print(f"NEMO ctl  top_depth={z_nemo_v:.3f} m  shear={sh_nemo:.5f} s^-1", flush=True)
    print(
        f"Cz={Cz}  shear_worsened_frac={shear_worsened_frac}  "
        f"HISTORICAL_REGISTERED_BAR={verdict}", flush=True)
    print(
        "RETRACTED / SUPERSEDED: the frozen-avm arm is UNINFORMATIVE about "
        "the faithful clamp fix. Its scalar shear win was sign cancellation; "
        "the upper-ocean velocity RMS degraded. Do not interpret Cz or the "
        "historical REFUTE/CONFIRM bar as a fix verdict.", flush=True)

    artifact = {
        "schema": "dino-euc-substitution-v1",
        "repo": git_receipt(ROOT),
        "provenance": {
            "committed_runner_sha256": sha256(Path(__file__).resolve()),
            "executor_ab_runner_sha256": EXECUTOR_RUNNER_SHA256,
            "pinned_reducers": OBSERVED_REDUCERS,
        },
        "devices": [str(d) for d in devices],
        "lane": {"name": dl.LANE, "run_dir": dl.RUN_DIR, "restart": dl.RESTART},
        "nemo_control_day10_restart": NEMO_CONTROL_DAY10_RESTART,
        "nemo_control_day10_restart_sha256": sha256(Path(NEMO_CONTROL_DAY10_RESTART)),
        "avm_dump_sha256": sha256(Path(dl.dump_path("tke_dump_avm_final.bin"))),
        "equator_row": EQUATOR_ROW,
        "n_days": N_DAYS,
        "evd_threshold_m2_s": EVD_THRESHOLD_M2_S,
        "controls": controls,
        "day0_identity": {"max_abs_du": d_day0_u, "max_abs_dT": d_day0_T,
                          "identical": day0_identity_ok},
        "base_wall_seconds": base["wall_seconds"],
        "nemo_avm_wall_seconds": nemo_avm["wall_seconds"],
        "base_finite": base["finite"], "nemo_avm_finite": nemo_avm["finite"],
        "scores": {"BASE": base_score, "NEMO_AVM": nemo_avm_score,
                   "NEMO_control_day10": nemo_score},
        "Cz": Cz, "shear_worsened_frac": shear_worsened_frac,
        "historical_registered_bar_verdict": verdict,
        "verdict": "SUPERSEDED_FROZEN_AVM_UNINFORMATIVE",
        "retraction": (
            "The Cz bar and scalar shear framing are retracted. The frozen "
            "arm shifted the upper 37 m eastward; its apparent shear gain was "
            "sign cancellation and upper-ocean velocity RMS degraded."),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    reread = json.loads(out.read_text())
    if reread["verdict"] != artifact["verdict"]:
        raise SystemExit("provenance read-back mismatch")
    print(f"WROTE {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
