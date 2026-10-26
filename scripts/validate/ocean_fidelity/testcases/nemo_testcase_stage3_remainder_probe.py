#!/usr/bin/env python3
"""OVERFLOW-zps kt=2 stage-3 ``u`` remainder: growth attribution + owner replays.

Two committed instruments (Rule: a throwaway probe's number is unmeasured):

``growth``
    (A) Is the kt>=3 OVERFLOW ``u`` walk the propagation of the kt=2 stage-3
    remainder, or a separately injected error?  Arms, all scored against
    NEMO's dumped step-entry states with the trajectory gate's own reducer:

    * ``free``            legoESM from the card initial state (the gate rows);
    * ``exact@k0``        NEMO's exact kt=k0 entry, ONE step -> the pure
                          per-step injection at k0 (k0 = 1 is the free run);
    * ``exact2_traj``     NEMO's exact kt=2 entry stepped to the end;
    * ``plus_du@2``       NEMO's exact kt=2 entry plus legoESM's MEASURED kt=2
                          ``u`` remainder field (u only), stepped to the end.

    ``plus_du@2`` reproducing ``free`` (amplitude within 2x, same face
    pattern) means the remainder owns the walk; ``exact@k0`` growing with
    k0 means a state-dependent per-step term.

``candidates``
    (B) Stage-3 operators replayed on NEMO's own stage-2 Kaa operands
    (``oracle_stage_kt00000001_s2.bin``, ``oracle_transport_..._s3.bin``)
    against the measured kt=2 remainder field ``R``: structure (depth
    profile, face pattern) and amplitude (corr / slope / max|R - E|), one
    candidate per row, no owner label printed by the tool.

Everything fp64 (``PrecisionPolicy.fp64()`` + ``JAX_ENABLE_X64=1``, dtypes
printed), CPU.  Oracle readers are the committed gates'; the NumPy
transcriptions of ``dynadv_up3`` / ``wzv`` / ``dynzdf`` are ported from the
hunt probes that owned the UP3 selector round, and are validated against
NEMO's own closure before any candidate number is quoted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import (
    allow_dirty_stamps,
    scoped_allow_dirty,
    worktree_stamp,
)

CASE = "OVERFLOW-zps"
ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10")
BTWALK_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/barotropic_walk/oracle_kt1_4")
DEFAULT_OUT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/stage3_remainder")
GAMMA1 = 1.0 / 3.0
FRONT_FACES = tuple(range(16, 25))


class ProbeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProbeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(name: str):
    script = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gates():
    """The three sibling gates: one reader / one reducer each."""
    return (_load("nemo_testcase_phase3_stage_sweep_gate"),
            _load("nemo_testcase_phase3_trajectory_gate"),
            _load("nemo_testcase_overflow_barotropic_gate"))


# ---------------------------------------------------------------------------
# Oracle readers not held by the gates (formats of the P3/BTWALK4 MY_SRC)
# ---------------------------------------------------------------------------
def _read_full(path: Path, magic_expect: str, nhead: int):
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack(f"={nhead}i", handle.read(4 * nhead))
        values = np.fromfile(handle, dtype=np.float64)
    require(magic == magic_expect, f"{path}: magic {magic!r} != {magic_expect!r}")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return header, values


def read_transport(path: Path, dims, expected_stage: int) -> dict:
    """``NEMO_L1_TRANSP_1``: (zFu, zFv, zFw) after stprk3_stg.F90:301."""
    nx, ny, nz = dims
    header, values = _read_full(path, "NEMO_L1_TRANSP_1", 8)
    version, kt, stage, kmm, hx, hy, hz, bits = header
    require((version, stage, hx, hy, hz, bits) == (1, expected_stage, nx, ny, nz, 64),
            f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 3 * count, f"{path}: bad payload")
    xyz = lambda v: v.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)
    return {"kt": kt, "stage": stage, "Kmm": kmm,
            "Fu": xyz(values[:count]), "Fv": xyz(values[count:2 * count]),
            "Fw": xyz(values[2 * count:])}


def read_bt_frames(path: Path, dims) -> dict:
    """``NEMO_L1_BTFRM_1``: (ua_b, va_b, un_adv, vn_adv) of one step's external solve."""
    nx, ny, _ = dims
    header, values = _read_full(path, "NEMO_L1_BTFRM_1", 6)
    version, kt, kaa, hx, hy, bits = header
    require((version, hx, hy, bits) == (1, nx, ny, 64), f"{path}: bad header {header}")
    count = nx * ny
    require(values.size == 4 * count, f"{path}: bad payload")
    xy = lambda v: v.reshape((nx, ny), order="F")[2:-2, 2:-2].T
    return {"kt": kt, "ua_b": xy(values[:count]), "va_b": xy(values[count:2 * count]),
            "un_adv": xy(values[2 * count:3 * count]), "vn_adv": xy(values[3 * count:])}


def read_rhs(path: Path, dims) -> dict:
    """``NEMO_L1_RHS___1``: stp2d's Kbb 3-D RHS (hpg + ldf + vor; no adv)."""
    nx, ny, nz = dims
    header, values = _read_full(path, "NEMO_L1_RHS___1", 7)
    version, kt, level, hx, hy, hz, bits = header
    require((version, hx, hy, hz, bits) == (1, nx, ny, nz, 64), f"{path}: bad header {header}")
    count = nx * ny * nz
    require(values.size == 2 * count, f"{path}: bad payload")
    xyz = lambda v: v.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)
    return {"kt": kt, "level": level, "u": xyz(values[:count]), "v": xyz(values[count:])}


def read_row_mesh_full(root: Path, gate) -> dict:
    """The gate's row mesh plus the horizontal metrics dom_qco_r3c needs."""
    import netCDF4

    mesh = gate.read_row_mesh(root)
    with netCDF4.Dataset(str(root / "mesh_mask.nc")) as data:
        def field(name):
            return np.asarray(data.variables[name][:]).squeeze()
        e1t, e2t = field("e1t")[1], field("e2t")[1]
        e1u, e2u = field("e1u")[1], field("e2u")[1]
        mesh["e1e2t"] = np.asarray(e1t * e2t, dtype=np.float64)
        mesh["e1e2u"] = np.asarray(e1u * e2u, dtype=np.float64)
        mesh["e2u"] = np.asarray(e2u, dtype=np.float64)
    hu_0 = mesh["hu_0"]
    mesh["r1_hu_0"] = np.where(hu_0 > 0, 1.0 / np.maximum(hu_0, 1e-300), 0.0)
    return mesh


# ---------------------------------------------------------------------------
# Row transcriptions (single wet row, arrays (i_face, k)); NEMO 5.0.2 sources
# quoted at each.  ``e3u_h`` / ``e3u_v`` are the divisors of the horizontal /
# vertical flux divergences so a candidate can swap ONE of them.
# ---------------------------------------------------------------------------
def r3t_row(ssh, mesh):
    return ssh * mesh["r1_ht_0"]                                # domqco.F90:209


def r3u_row(ssh, mesh):
    """domqco.F90:219-220: e1e2t-weighted ssh mean / hu_0 / e1e2u."""
    ni = ssh.shape[0]
    out = np.zeros(ni)
    out[:-1] = (0.5 * (mesh["e1e2t"][:-1] * ssh[:-1] + mesh["e1e2t"][1:] * ssh[1:])
                * mesh["r1_hu_0"][:-1] / mesh["e1e2u"][:-1])
    return out


def wumask_row(umask):
    out = np.zeros_like(umask)
    out[:, 0] = umask[:, 0]
    out[:, 1:] = umask[:, 1:] * umask[:, :-1]
    return out


def up3_row(u, Fu, Fw, e3u_h, e3u_v, mesh, *, selector="velocity"):
    """dynadv_up3.F90:137-214 (horizontal) + :239-358 (vertical) u-trend, v = 0.

    Returns ``(adv_h, adv_v)`` in m/s^2 on (i_face, k).  ``selector``
    'velocity' = NEMO :166-170 (zui sign); 'transport' = the legacy legoESM
    branch choice (flux-pair sign).
    """
    umask = mesh["umask"]
    ni, nz = u.shape
    e1e2u = mesh["e1e2u"]
    adv_h = np.zeros_like(u)
    for k in range(nz - 1):
        uk, fk, um = u[:, k], Fu[:, k], umask[:, k]
        zlu = np.zeros(ni)
        zlu[1:-1] = ((uk[2:] - uk[1:-1]) + (uk[:-2] - uk[1:-1])) * um[1:-1]      # :142-143
        zui = uk[:-1] + uk[1:]                                                   # :166
        crit = zui if selector == "velocity" else (fk[:-1] + fk[1:])
        zl = np.where(crit > 0, zlu[:-1], zlu[1:])                               # :169-170
        zfu_t = np.zeros(ni + 1)
        zfu_t[1:ni] = (fk[:-1] + fk[1:]) * (zui - GAMMA1 * zl)                   # :176
        with np.errstate(divide="ignore", invalid="ignore"):
            adv_h[:, k] = -np.where(
                um > 0, 0.25 * (zfu_t[1:ni + 1] - zfu_t[0:ni]) / e1e2u / e3u_h[:, k], 0.0)  # :205-207
    wum = wumask_row(umask)
    adv_v = np.zeros_like(u)
    zfwu = np.zeros(ni)                                                          # :264-267 (qco: 0)
    zlu_uw = np.zeros(ni)                                                        # :270-273
    for k in range(0, nz - 2):                                                   # :275 jk = 1..jpk-2
        fw_kp1 = Fw[:, k + 1]
        zfwi = np.zeros(ni)
        zfwi[:-1] = fw_kp1[:-1] + fw_kp1[1:]                                     # :322
        zlu_kp1 = ((u[:, k] - u[:, k + 1]) * wum[:, k + 1]
                   - (u[:, k + 1] - u[:, k + 2]) * wum[:, k + 2])                # :317-318
        zl = np.where(zfwi > 0, zlu_kp1, zlu_uw)                                 # :323-325
        zzfu_kp1 = 0.25 * zfwi * (u[:, k + 1] + u[:, k] - GAMMA1 * zl)           # :331
        with np.errstate(divide="ignore", invalid="ignore"):
            adv_v[:, k] = -np.where(umask[:, k] > 0, (zfwu - zzfu_kp1) / e1e2u / e3u_v[:, k], 0.0)  # :334
        zfwu, zlu_uw = zzfu_kp1, zlu_kp1                                         # :337-341
    k = nz - 2                                                                   # :346 jk = jpkm1
    with np.errstate(divide="ignore", invalid="ignore"):
        adv_v[:, k] = -np.where(umask[:, k] > 0, zfwu / e1e2u / e3u_v[:, k], 0.0)  # :355
    return adv_h * umask, adv_v * umask


def wzv_row(Fu, ssh_aa, ssh_bb, dt_stage, mesh):
    """sshwzv.F90:332-336 (qco) with divhor.F90:116-123,139-141 (np_transport).

    ``e1e2t*ww(k) = e1e2t*ww(k+1) - [ (Fu_i - Fu_{i-1}) + e1e2t*e3t_0*(r3t(Kaa)-r3t(Kbb))/rDt ] * tmask``.
    """
    tmask, e3t0 = mesh["tmask"], mesh["e3t_0"]
    ni, nz = Fu.shape
    div = np.zeros((ni, nz))
    div[1:, :] = Fu[1:, :] - Fu[:-1, :]
    qco = mesh["e1e2t"][:, None] * e3t0 * (
        (r3t_row(ssh_aa, mesh) - r3t_row(ssh_bb, mesh)) / dt_stage)[:, None]
    Fw = np.zeros((ni, nz))
    for k in range(nz - 2, -1, -1):
        Fw[:, k] = Fw[:, k + 1] - (div[:, k] + qco[:, k]) * tmask[:, k]
    return Fw


def zdf_apply_row(u, ssh_mm, ssh_aa, mesh, dt, avm):
    """dynzdf.F90:181-196 tridiagonal ``M`` applied to ``u`` (pre-solve field = M u_post).

    ``e3u(Kaa) = e3u_0*(1+r3u(Kaa))``, ``e3uw(Kmm) = e3uw_0*(1+r3u(Kmm))`` with
    ``e3uw_0 = e3w_1d`` (tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:167).
    """
    umask = mesh["umask"]
    e3u_aa = mesh["e3u_0"] * (1.0 + r3u_row(ssh_aa, mesh))[:, None]
    e3uw_mm = mesh["e3w_1d"][None, :] * (1.0 + r3u_row(ssh_mm, mesh))[:, None]
    wum = wumask_row(umask)
    zwi = np.zeros_like(u)
    zws = np.zeros_like(u)
    zwi[:, 1:] = -0.5 * dt * (2.0 * avm) / (e3u_aa[:, 1:] * e3uw_mm[:, 1:]) * wum[:, 1:]
    zws[:, :-1] = -0.5 * dt * (2.0 * avm) / (e3u_aa[:, :-1] * e3uw_mm[:, 1:]) * wum[:, 1:]
    zwd = 1.0 - zwi - zws
    out = zwd * u
    out[:, 1:] += zwi[:, 1:] * u[:, :-1]
    out[:, :-1] += zws[:, :-1] * u[:, 1:]
    return out * umask


def bc_row(field, act, mesh):
    """Deviation from the e3u_0-weighted column mean (stprk3_stg.F90:440)."""
    w = mesh["e3u_0"] * act
    mean = np.sum(field * w, axis=-1) * mesh["r1_hu_0"]
    return np.where(act, field - mean[:, None], 0.0), mean


def fit_linear_in_depth(profile, gdept, active):
    """Least-squares ``a + b*gdept`` on the active levels; returns (a, b, r2)."""
    z = gdept[active]
    y = profile[active]
    if y.size < 3 or float(np.max(np.abs(y))) == 0.0:
        return {"a": 0.0, "b": 0.0, "r2": float("nan"), "n": int(y.size)}
    A = np.stack([np.ones_like(z), z], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"a": float(coef[0]), "b": float(coef[1]),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), "n": int(y.size)}


def fit_pattern(target, pattern, act):
    x = pattern[act]
    y = target[act]
    xx = float(x @ x)
    if xx == 0.0:
        return {"corr": float("nan"), "slope": float("nan"),
                "max_abs_pattern": 0.0, "max_abs_residual": float(np.max(np.abs(y)))}
    slope = float(x @ y / xx)
    corr = float(np.corrcoef(x, y)[0, 1]) if float(y @ y) > 0 else float("nan")
    return {"corr": corr, "slope": slope,
            "max_abs_pattern": float(np.max(np.abs(x))),
            "max_abs_residual": float(np.max(np.abs(y - slope * x))),
            "max_abs_residual_unit_slope": float(np.max(np.abs(y - x)))}


def face_summary(field, act, faces=FRONT_FACES, nlev=None):
    nlev = field.shape[1] if nlev is None else nlev
    out = {}
    for i in faces:
        col = np.where(act[i, :nlev], field[i, :nlev], 0.0)
        kmax = int(np.argmax(np.abs(col)))
        out[int(i)] = {"max_abs": float(np.abs(col).max()),
                       "k_of_max": kmax,
                       "value_at_kmax": float(col[kmax]),
                       "profile": [float(v) for v in col]}
    return out


# ---------------------------------------------------------------------------
# legoESM helpers
# ---------------------------------------------------------------------------
def set_fp64():
    """Rule 1c: the policy must be set BEFORE the card is built (constructors
    cast to ``get_policy().control``, which defaults to float32)."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")


def fp64_model(card, hooks=None):
    from legoesm.core.precision import PrecisionPolicy, get_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)

    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks if hooks is not None else _NEMOWSRK3TestHooks())


def dtype_report(card, state) -> dict:
    z = card.recipe.z_coord
    report = {
        "T": str(np.asarray(state.T.data).dtype), "u": str(np.asarray(state.u.data).dtype),
        "eta": str(np.asarray(state.eta.data).dtype),
        "h_partial": str(np.asarray(z.h_partial).dtype),
        "dz_ref": str(np.asarray(z.dz_ref).dtype),
    }
    require(set(report.values()) == {"float64"}, f"non-fp64 arrays: {report}")
    return report


def git_sha(allow_dirty: bool) -> str:
    from legoesm.ocean.fidelity.provenance import git_sha as _stamp
    return _stamp(allow_dirty=allow_dirty)


def u_residual_row(state, entry_u, act_u, nlev):
    """legoESM minus NEMO ``u`` on the wet row, (i_face, k) on the active faces."""
    lego = np.asarray(state.u.data)[1, 1:, :nlev]
    nemo = np.asarray(entry_u)[1, :, :nlev]
    return np.where(act_u[1], lego - nemo, 0.0)


# ---------------------------------------------------------------------------
# (A) growth attribution
# ---------------------------------------------------------------------------
def growth(*, max_kt: int, out_dir: Path, allow_dirty: bool, frames: bool) -> dict:
    SWEEP, TRAJ, BARO = gates()
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    import jax

    allow_dirty_stamps(allow_dirty)

    legoesm_git_sha = git_sha(allow_dirty)
    set_fp64()
    card = build_nemo_testcase_card(CASE)
    model = fp64_model(card)
    dtypes = dtype_report(card, card.recipe.initial_state)
    masks = TRAJ.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    dims = SWEEP.DIMS[CASE]
    entries = {}
    artifacts = {}
    for kt in range(1, max_kt + 1):
        path = ROOT / f"oracle_step_entry_kt{kt:08d}.bin"
        entries[kt] = TRAJ.read_entry(path, CASE)
        require(int(entries[kt]["step"]) == kt, f"{path}: step mismatch")
        artifacts[path.name] = sha256(path)
    act_u = masks["u"]

    def rows_at(state, kt):
        fields = TRAJ.lego_fields(state)
        rows = {}
        for field in ("T", "S", "u", "v", "ssh"):
            ref = np.asarray(entries[kt][field])
            if field != "ssh":
                ref = ref[..., :nlev]
            row = TRAJ.score(f"{CASE}.kt{kt}.{field}", ref, fields[field], masks[field],
                             allow_empty_no_active_face=field == "v")
            rows[field] = {"normalized_max_abs": row["normalized_max_abs"],
                           "status": row["status"], "exact": row["exact"]}
        res = u_residual_row(state, entries[kt]["u"], act_u, nlev)
        a = np.abs(res)
        i, k = np.unravel_index(int(np.argmax(a)), a.shape)
        rows["u_structure"] = {
            "argmax_face": int(i), "argmax_k": int(k), "max_abs": float(a.max()),
            "per_face_max": {int(f): float(a[f].max()) for f in FRONT_FACES},
            "depth_mean_part_max": float(np.max(np.abs(bc_row(res, act_u[1], MESH_NLEV)[1]))),
        }
        return rows, res

    global MESH, MESH_NLEV
    MESH = read_row_mesh_full(ROOT, SWEEP)
    MESH_NLEV = {"e3u_0": MESH["e3u_0"][:, :nlev], "r1_hu_0": MESH["r1_hu_0"]}

    def step(state):
        return model.step(state, dt=card.dt_s)

    report = {"worktree": worktree_stamp(), "format": "nemo-testcase-l1-stage3-remainder-growth-v1", "case": CASE,
              "legoesm_git_sha": legoesm_git_sha, "backend": jax.default_backend(),
              "dtypes": dtypes, "dt_s": card.dt_s, "nlev": nlev, "max_kt": max_kt,
              "oracle_root": str(ROOT), "artifacts": artifacts, "arms": {}}

    # --- free run (the trajectory gate's own rows, re-measured here) ---
    state = card.recipe.initial_state
    free_rows, free_res = {}, {}
    r, res = rows_at(state, 1)
    free_rows[1], free_res[1] = r, res
    free_states = {1: state}
    for kt in range(2, max_kt + 1):
        state = step(state)
        free_states[kt] = state
        free_rows[kt], free_res[kt] = rows_at(state, kt)
        print(f"[free] kt={kt} T {free_rows[kt]['T']['normalized_max_abs']:.6e} "
              f"u {free_rows[kt]['u']['normalized_max_abs']:.6e} "
              f"ssh {free_rows[kt]['ssh']['normalized_max_abs']:.6e} "
              f"argmax face {free_rows[kt]['u_structure']['argmax_face']} k {free_rows[kt]['u_structure']['argmax_k']}",
              flush=True)
    report["arms"]["free"] = {kt: free_rows[kt] for kt in free_rows}

    # --- the measured kt=2 remainder field (u only), on NEMO's layout ---
    du_row = free_res[2]                                    # (i, k<nlev), active faces only
    du_nemo = np.zeros_like(np.asarray(entries[2]["u"]))
    du_nemo[1, :, :nlev] = du_row
    report["kt2_remainder"] = {
        "max_abs": float(np.abs(du_row).max()),
        "faces": face_summary(du_row, act_u[1], nlev=nlev),
    }

    def reseed(kt0, du=None):
        entry = dict(entries[kt0])
        if du is not None:
            entry = dict(entry)
            entry["u"] = np.asarray(entry["u"]) + du
        st = BARO.state_from_oracle_entry(card.recipe.initial_state, entry, masks)
        r0, res0 = rows_at(st, kt0)
        return st, r0, res0

    # --- exact@k0, one step: the pure per-step injection ---
    inj = {}
    for k0 in range(2, max_kt):
        st, r0, _ = reseed(k0)
        require(r0["u"]["normalized_max_abs"] <= 1e-15 and r0["T"]["normalized_max_abs"] <= 1e-15,
                f"exact reseed at kt={k0} is not exact: {r0['u']} {r0['T']}")
        st1 = step(st)
        r1, _ = rows_at(st1, k0 + 1)
        inj[k0 + 1] = {"from_kt": k0, "entry_rows": r0, "after_one_step": r1}
        print(f"[exact@{k0}] -> kt={k0 + 1}: u {r1['u']['normalized_max_abs']:.6e} "
              f"T {r1['T']['normalized_max_abs']:.6e} ssh {r1['ssh']['normalized_max_abs']:.6e} "
              f"argmax face {r1['u_structure']['argmax_face']} k {r1['u_structure']['argmax_k']}", flush=True)
    inj[2] = {"from_kt": 1, "entry_rows": free_rows[1], "after_one_step": free_rows[2]}
    report["arms"]["one_step_injection"] = {kt: inj[kt] for kt in sorted(inj)}

    # --- exact@2 trajectory and plus_du@2 trajectory ---
    # What a reseed keeps from the card initial state (inactive cells, every
    # history slot): the plus_du entry vs legoESM's own kt=2 state on ALL
    # stored cells, so a reseed-vs-free difference later is attributable.
    _pd_state = reseed(2, du_nemo)[0]
    report["reseed_vs_free_kt2_all_cells"] = {
        f: float(np.max(np.abs(np.asarray(getattr(_pd_state, f).data)
                               - np.asarray(getattr(free_states[2], f).data))))
        for f in ("T", "S", "u", "v", "eta")}
    print("[reseed vs free kt=2, all cells]", report["reseed_vs_free_kt2_all_cells"], flush=True)
    for name, du in (("exact2_traj", None), ("plus_du2_traj", du_nemo)):
        st, r0, _ = reseed(2, du)
        rows = {2: r0}
        res_by_kt = {}
        for kt in range(3, max_kt + 1):
            st = step(st)
            rows[kt], res_by_kt[kt] = rows_at(st, kt)
            comp = fit_pattern(free_res[kt], res_by_kt[kt], act_u[1])
            rows[kt]["vs_free"] = {
                "u_linf_ratio_free_over_arm": (
                    free_rows[kt]["u"]["normalized_max_abs"]
                    / max(rows[kt]["u"]["normalized_max_abs"], 1e-300)),
                "pattern_corr_with_free": comp["corr"],
                "pattern_slope_free_on_arm": comp["slope"],
                "max_abs_free_minus_arm": float(np.max(np.abs(free_res[kt] - res_by_kt[kt]))),
                "T_linf_ratio_free_over_arm": (
                    free_rows[kt]["T"]["normalized_max_abs"]
                    / max(rows[kt]["T"]["normalized_max_abs"], 1e-300)),
                "ssh_linf_ratio_free_over_arm": (
                    free_rows[kt]["ssh"]["normalized_max_abs"]
                    / max(rows[kt]["ssh"]["normalized_max_abs"], 1e-300)),
            }
            print(f"[{name}] kt={kt}: u {rows[kt]['u']['normalized_max_abs']:.6e} "
                  f"(free {free_rows[kt]['u']['normalized_max_abs']:.6e}, ratio "
                  f"{rows[kt]['vs_free']['u_linf_ratio_free_over_arm']:.3f}, corr "
                  f"{rows[kt]['vs_free']['pattern_corr_with_free']:+.4f}) T {rows[kt]['T']['normalized_max_abs']:.3e} "
                  f"ssh {rows[kt]['ssh']['normalized_max_abs']:.3e} argmax face "
                  f"{rows[kt]['u_structure']['argmax_face']}", flush=True)
        report["arms"][name] = rows
        if name == "plus_du2_traj":
            plus_du_states = {2: reseed(2, du)[0]}
            st = plus_du_states[2]
            for kt in range(3, 5):
                st = step(st)
                plus_du_states[kt] = st

    # --- slow_u frames of the plus_du trajectory's external solve (kt=2..4) ---
    if frames:
        frame_rows = {}
        for kt in (2, 3, 4):
            trace_path = BTWALK_ROOT / f"oracle_overflow_bt_substeps_kt{kt:08d}_call1.bin"
            oracle = BARO.read_oracle_trace(trace_path, expected_kt=kt)
            artifacts[trace_path.name] = sha256(trace_path)
            cand = BARO.capture_legoesm_trace(flux_form_override=None, kt=kt,
                                              start_state=plus_du_states[kt])
            substeps, first, masked = BARO._score_substeps(f"plus_du2.kt{kt}", oracle, cand)
            sel = {}
            for row in substeps[0]["rows"]:
                frame = row["name"].rsplit(".", 1)[-1]
                if frame in ("u_entry", "transport_u", "eta_continuity", "pgf_u",
                             "slow_u", "u_exit", "eta_exit"):
                    sel[frame] = {"normalized_max_abs": row["normalized_max_abs"],
                                  "absolute_max": row["absolute_max"], "status": row["status"]}
            frame_rows[kt] = {"substep1": sel, "first_over_bar": first}
            print(f"[plus_du2 frames] kt={kt} substep 1: " + ", ".join(
                f"{k} {v['normalized_max_abs']:.3e}" for k, v in sel.items()), flush=True)
        report["arms"]["plus_du2_frames"] = frame_rows

    report["artifacts"] = artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "growth.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    print("wrote", out)
    return report


# ---------------------------------------------------------------------------
# (B) candidate replays on NEMO's stage-2 Kaa operands
# ---------------------------------------------------------------------------
def candidates(*, out_dir: Path, allow_dirty: bool) -> dict:
    SWEEP, TRAJ, BARO = gates()
    import jax
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks, _nemo_ws_qco_stage_faces, _nemo_ws_stage_transport)
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _bc_horizontal_momentum_advection_flux_form)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d, min_cell_to_uface, min_cell_to_vface)
    from legoesm.grids.operators_latlon_cgrid import interp_cell_to_uface
    from legoesm.ocean.vertical import (
        compute_layer_thickness, nemo_up3_vertical_momentum_advection)
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_nemo_momentum)

    allow_dirty_stamps(allow_dirty)

    legoesm_git_sha = git_sha(allow_dirty)
    set_fp64()
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config
    z = card.recipe.z_coord
    grid = card.recipe.grid
    init = card.recipe.initial_state
    model = fp64_model(card)
    dtypes = dtype_report(card, init)
    masks = SWEEP.expected_masks(card)
    nlev = z.n_levels
    dims = SWEEP.DIMS[CASE]
    nx, ny, nz = dims
    dt = float(card.dt_s)
    mesh = read_row_mesh_full(ROOT, SWEEP)
    ni = mesh["tmask"].shape[0]
    act = mesh["umask"].astype(bool) & (np.arange(ni)[:, None] < ni - 1)
    act[-1] = False
    require(np.array_equal(act[:, :nlev], masks["u"][1]), "row mask disagrees with the gate's u mask")
    gdept = mesh["gdept_1d"]

    artifacts = {}

    def stamped(path):
        artifacts[path.name] = sha256(path)
        return path

    e1 = SWEEP.read_entry(stamped(ROOT / "oracle_step_entry_kt00000001.bin"), CASE, expected=(1, 1))
    e2 = SWEEP.read_entry(stamped(ROOT / "oracle_step_entry_kt00000002.bin"), CASE)
    s1 = SWEEP.read_stage(stamped(ROOT / "oracle_stage_kt00000001_s1.bin"), CASE, 1)
    s2 = SWEEP.read_stage(stamped(ROOT / "oracle_stage_kt00000001_s2.bin"), CASE, 2)
    s3 = SWEEP.read_stage(stamped(ROOT / "oracle_stage_kt00000001_s3.bin"), CASE, 3)
    t2 = read_transport(stamped(ROOT / "oracle_transport_kt00000001_s2.bin"), dims, 2)
    t3 = read_transport(stamped(ROOT / "oracle_transport_kt00000001_s3.bin"), dims, 3)
    fr1 = read_bt_frames(stamped(ROOT / "oracle_bt_frames_kt00000001.bin"), dims)
    rhs1 = read_rhs(stamped(ROOT / "oracle_rhs_kt00000001.bin"), dims)
    row = lambda a: np.asarray(a)[1]
    T0, S0, ssh0 = row(e1["T"]), row(e1["S"]), row(e1["ssh"])
    T2, S2, ssh2 = row(s2["T"]), row(s2["S"]), row(s2["ssh"])
    ssha = row(e2["ssh"])
    u1n, u2n, u3n = row(s1["u"]), row(s2["u"]), row(s3["u"])
    Fu3, Fw3 = row(t3["Fu"]), row(t3["Fw"])
    un_adv = row(fr1["un_adv"])
    ua_b = row(fr1["ua_b"])
    require(np.array_equal(u3n[:, :nlev], row(e2["u"])[:, :nlev]), "stage-3 dump != kt=2 entry")

    # --- legoESM: kt=1 step (stage 3 == kt=2 entry) and the exposed stage 2 ---
    st3 = model.step(init, dt=dt)
    st2 = fp64_model(card, _NEMOWSRK3TestHooks(expose_momentum_stage=2)).step(init, dt=dt)
    pad = lambda a: np.concatenate([a, np.zeros((a.shape[0], nz - a.shape[1]))], axis=1)
    u3l = pad(np.asarray(st3.u.data)[1, 1:, :])
    u2l = pad(np.asarray(st2.u.data)[1, 1:, :])
    eta3l = np.asarray(st3.eta.data)
    eta_half_l = 0.5 * (np.asarray(init.eta.data) + eta3l)
    print("dtypes", dtypes, "nlev", nlev, "dt", dt, "backend", jax.default_backend())
    print(f"[operands] max|eta3_lego - ssha| = {np.abs(eta3l[1] - ssha).max():.3e}  "
          f"max|u2_lego - u2_nemo| = {np.abs(np.where(act, u2l - u2n, 0)).max():.3e}")

    # --- the target: R = bc(u3_lego - u3_nemo) ---
    R_full = np.where(act, u3l - u3n, 0.0)
    R, R_mean = bc_row(R_full, act, mesh)
    a = np.abs(R)
    i_max, k_max = np.unravel_index(int(np.argmax(a)), a.shape)
    target = {
        "max_abs_full": float(np.abs(R_full).max()),
        "max_abs_baroclinic": float(a.max()), "argmax_face": int(i_max), "argmax_k": int(k_max),
        "max_abs_depth_mean": float(np.abs(R_mean).max()),
        "faces": face_summary(R, act, nlev=nlev),
        "linear_fit_in_gdept": {int(i): fit_linear_in_depth(R[i], gdept, act[i]) for i in (19, 20, 21)},
        "corr_face19_face21": float(np.corrcoef(R[19, act[19]], R[21, act[21]])[0, 1]),
        "corr_face19_face20": float(np.corrcoef(R[19, act[19]], R[20, act[20]])[0, 1]),
    }
    print(f"[target] R max|full| {target['max_abs_full']:.4e}  bc {target['max_abs_baroclinic']:.4e} at face {i_max} k {k_max}; "
          f"depth-mean {target['max_abs_depth_mean']:.2e}")
    for i in (18, 19, 20, 21, 22):
        print(f"   R face {i} k=0..{nlev - 1}: {np.array2string(R[i, :nlev], precision=2, max_line_width=300)}")
    print(f"   linear fits: {target['linear_fit_in_gdept']}  corr(19,21)={target['corr_face19_face21']:+.4f}")

    # --- transcription controls (must pass before any candidate number) ---
    eos_fn = make_eos_fn(cfg.eos, None, rho0=cfg.rho_0)
    ppm = cfg.rho_0 * constants.g
    H0 = SWEEP.hpg_sco_row(mesh, T0, S0, ssh0, eos_fn, cfg.g, cfg.rho_0, ppm)
    H2 = SWEEP.hpg_sco_row(mesh, T2, S2, ssh2, eos_fn, cfg.g, cfg.rho_0, ppm)
    r3u_m = r3u_row(ssh2, mesh)          # stage 3 Kmm = N+1/2 (stprk3_stg.F90:210-212)
    r3u_a = r3u_row(ssha, mesh)          # stage 3 Kaa = N+1   (:231-233)
    e3u_m = mesh["e3u_0"] * (1.0 + r3u_m)[:, None]
    qfac = ((1.0 + r3u_m) / (1.0 + r3u_a))[:, None]
    avm = float(cfg.A_v) if hasattr(cfg, "A_v") else 1.0e-4
    controls = {}
    d = np.abs(np.where(act, H0 - row(rhs1["u"]), 0.0))
    controls["hpg_sco_vs_oracle_rhs_kt1_rest"] = float(d.max())
    Fw3_rec = wzv_row(Fu3, ssha, ssh0, dt, mesh)
    controls["wzv_transport_vs_dumped_Fw3"] = float(np.abs(Fw3 - Fw3_rec)[mesh["tmask"] > 0].max())
    u3n_pre = zdf_apply_row(u3n, ssh2, ssha, mesh, dt, avm)
    u3l_pre = zdf_apply_row(u3l, ssh2, ssha, mesh, dt, avm)
    rhs_n = u3n_pre / qfac / dt                           # dynzdf.F90:127-129 inverted, u(Kbb)=0
    adv_h_n, adv_v_n = up3_row(u2n, Fu3, Fw3, e3u_m, e3u_m, mesh)
    closure_n = bc_row(rhs_n - H2 - adv_h_n - adv_v_n, act, mesh)[0]
    controls["nemo_stage3_rhs_closure_bc"] = float(np.abs(closure_n).max())
    zdf_inc = np.where(act, u3n - u3n_pre, 0.0)
    controls["nemo_stage3_zdf_increment_faces"] = {
        int(i): {"k0": float(zdf_inc[i, 0]), "k_bottom": float(zdf_inc[i, int(act[i].sum()) - 1]),
                 "interior_max": float(np.abs(zdf_inc[i, 1:int(act[i].sum()) - 1]).max())}
        for i in (19, 20, 21)}
    print("[controls]", json.dumps(controls, default=float))
    require(controls["hpg_sco_vs_oracle_rhs_kt1_rest"] < 1e-14, "hpg_sco transcription failed its rest control")
    require(controls["nemo_stage3_rhs_closure_bc"] < 1e-13, "NEMO stage-3 closure failed: transcription untrusted")

    # --- legoESM RHS remainder with the current code (the previous round's D_l) ---
    rhs_l = u3l_pre / qfac / dt
    adv_h_l, adv_v_l = up3_row(u2l, Fu3, Fw3, e3u_m, e3u_m, mesh)
    D_l = bc_row(rhs_l - H2 - adv_h_l - adv_v_l, act, mesh)[0]
    remainder = {"max_abs_m_s2": float(np.abs(D_l).max()),
                 "times_dt_m_s": float(np.abs(D_l).max() * dt),
                 "faces": face_summary(D_l, act, nlev=nlev),
                 "input_sensitivity_u2": float(np.abs(bc_row(adv_h_l + adv_v_l - adv_h_n - adv_v_n, act, mesh)[0]).max())}
    print(f"[remainder D_l] max {remainder['max_abs_m_s2']:.4e} m/s^2 (x dt {remainder['times_dt_m_s']:.4e} m/s); "
          f"input sensitivity {remainder['input_sensitivity_u2']:.2e}")
    # Predicted stage-3 velocity error from D_l alone (what R should be if D_l were the whole story)
    E_D = dt * qfac * D_l
    remainder["fit_R_on_dt_D_l"] = fit_pattern(R, bc_row(E_D, act, mesh)[0], act)

    rows = {}

    def record(name, E_rhs, source, note=""):
        """E_rhs: candidate's RHS difference (lego - nemo), m/s^2, full field.
        Prediction of the stage-3 velocity error: dt*(1+r3u_m)/(1+r3u_a)*bc(E)."""
        E = bc_row(dt * qfac * E_rhs, act, mesh)[0]
        entry = {"source": source, "note": note,
                 "max_abs_prediction_m_s": float(np.abs(E).max()),
                 "faces": {int(i): float(np.abs(E[i]).max()) for i in (18, 19, 20, 21, 22)},
                 "linear_fit_in_gdept": {int(i): fit_linear_in_depth(E[i], gdept, act[i]) for i in (19, 21)},
                 "fit_R_on_E": fit_pattern(R, E, act)}
        rows[name] = entry
        f = entry["fit_R_on_E"]
        print(f"[cand {name}] max|E| {entry['max_abs_prediction_m_s']:.3e}  faces19/20/21 "
              f"{entry['faces'][19]:.2e}/{entry['faces'][20]:.2e}/{entry['faces'][21]:.2e}  corr {f['corr']:+.5f} "
              f"slope {f['slope']:+.4f}  max|R-E| {f['max_abs_residual_unit_slope']:.3e}", flush=True)
        return E

    # ---- operands on legoESM's layout ----
    lat = np.asarray(init.eta.data).shape[0]
    to_lego_u = lambda r: np.pad(r[:, :nlev][None], ((1, lat - 2), (1, 0), (0, 0)))  # (3, 203, nlev), row 1
    u2n_lego = jnp.asarray(to_lego_u(u2n))
    v_zero = jnp.zeros_like(init.v.data)
    h_ref = compute_layer_thickness(jnp.zeros_like(init.eta.data), init.H_bathy.data, z,
                                    min_water_column_m=cfg.min_water_column_m)
    u_mask3, v_mask3 = compute_face_masks_3d(z.is_active, grid)
    u_mask3 = u_mask3.astype(h_ref.dtype)
    v_mask3 = v_mask3.astype(h_ref.dtype)
    eta2_lego = jnp.asarray(np.pad(ssh2[None], ((1, lat - 2), (0, 0))))
    etaa_lego = jnp.asarray(np.pad(ssha[None], ((1, lat - 2), (0, 0))))
    # thickness the stage program's tendencies() builds from the stage eta
    # (ocean_pe_latlon_cgrid.py:1221 compute_layer_thickness, :1364 min_cell_to_uface)
    h_k_stage = compute_layer_thickness(eta2_lego, init.H_bathy.data, z,
                                        min_water_column_m=cfg.min_water_column_m)
    h_u_min = min_cell_to_uface(h_k_stage)
    h_v_min = min_cell_to_vface(h_k_stage, grid)
    h_u_nemo, h_v_nemo, q_m_lego, _ = _nemo_ws_qco_stage_faces(eta2_lego, h_ref, u_mask3, v_mask3, grid)
    _, _, q_a_lego, _ = _nemo_ws_qco_stage_faces(etaa_lego, h_ref, u_mask3, v_mask3, grid)
    h_u_pre = min_cell_to_uface(h_ref)
    H_u_pre = jnp.sum(h_u_pre, axis=-1)

    # X4: qco stage ratios (rule + operand) vs the NEMO formula on NEMO's ssh
    qm_l = np.asarray(q_m_lego)[1, 1:]
    qa_l = np.asarray(q_a_lego)[1, 1:]
    face2d = act.any(axis=1)
    d_qm = np.where(face2d, qm_l - (1.0 + r3u_m), 0.0)
    d_qa = np.where(face2d, qa_l - (1.0 + r3u_a), 0.0)
    qfac_l = (qm_l / qa_l)[:, None]
    rows_qco = {"max_abs_delta_1_plus_r3u_kmm_rule": float(np.abs(d_qm).max()),
                "max_abs_delta_1_plus_r3u_kaa_rule": float(np.abs(d_qa).max()),
                "max_abs_r3u_kmm": float(np.abs(r3u_m[face2d]).max()),
                "max_abs_r3u_kaa": float(np.abs(r3u_a[face2d]).max())}
    qm_l_own = np.asarray(_nemo_ws_qco_stage_faces(jnp.asarray(eta_half_l), h_ref, u_mask3, v_mask3, grid)[2])[1, 1:]
    qa_l_own = np.asarray(_nemo_ws_qco_stage_faces(jnp.asarray(eta3l), h_ref, u_mask3, v_mask3, grid)[2])[1, 1:]
    rows_qco["max_abs_delta_kmm_own_operand"] = float(np.abs(np.where(face2d, qm_l_own - (1.0 + r3u_m), 0.0)).max())
    rows_qco["max_abs_delta_kaa_own_operand"] = float(np.abs(np.where(face2d, qa_l_own - (1.0 + r3u_a), 0.0)).max())
    E_q = (np.asarray((qm_l_own / qa_l_own)[:, None]) - qfac) / qfac * rhs_n   # -> dt*qfac*E = dt*(q_l - q_n)*rhs
    record("X4_qco_stage_ratio", np.where(act, E_q, 0.0),
           "stprk3_stg.F90:373-378 / dynzdf.F90:127-129 vs _nemo_ws_qco_stage_faces on legoESM's own eta")
    rows["X4_qco_stage_ratio"].update(rows_qco)

    # X2: HPG on NEMO's stage-2 operands through legoESM's tendencies() at rest
    T2_lego = jnp.asarray(np.pad(T2[None, :, :nlev], ((1, lat - 2), (0, 0), (0, 0))))
    S2_lego = jnp.asarray(np.pad(S2[None, :, :nlev], ((1, lat - 2), (0, 0), (0, 0))))
    st_hpg = init._replace(T=init.T.replace(data=T2_lego), S=init.S.replace(data=S2_lego),
                           eta=init.eta.replace(data=eta2_lego),
                           u=init.u.replace(data=jnp.zeros_like(init.u.data)),
                           v=init.v.replace(data=v_zero))
    td = model.tendencies(st_hpg, dt=dt, momentum_only=True)
    hpg_lego = pad(np.asarray(td.du_dt.data)[1, 1:, :])
    record("X2_hpg_operator", np.where(act, hpg_lego - H2, 0.0),
           "dynhpg.F90:340-390 hpg_sco vs legoESM tendencies(u=0) on NEMO's stage-2 (T,S,ssh)")

    # X1: the horizontal momentum advection's thickness inside tendencies():
    # min-rule of the STRETCHED stage T thicknesses (opl:1221,1364) vs NEMO's
    # e3u(Kmm) = e3u_0*(1+r3u(Kmm)) (domzgr_substitute.h90:127) -- legoESM's
    # OWN operator, one variable (h_u), on NEMO's u2 and NEMO's un_adv.
    H_u_safe = jnp.where(H_u_pre > 0.0, H_u_pre, 1.0)
    zub_l = jnp.where(
        H_u_pre > 0.0,
        jnp.asarray(np.pad(un_adv[None], ((1, lat - 2), (1, 0)))) / H_u_safe
        - jnp.sum(u2n_lego * h_u_pre, axis=-1) / H_u_safe,
        0.0) * init.u_mask.data
    transport_velocity = ((u2n_lego + zub_l[..., None]) * u_mask3, v_zero)
    zeros_u = jnp.zeros_like(init.u.data)
    zeros_v = jnp.zeros_like(init.v.data)

    def lego_adv(h_u, h_v, selector="velocity"):
        du, _, _, _ = _bc_horizontal_momentum_advection_flux_form(
            zeros_u, zeros_v, u2n_lego, v_zero, h_u, h_v, u_mask3, v_mask3,
            init.land_mask.data, grid, cfg, transport_velocity=transport_velocity,
            up3_upwind_selector=selector)
        return pad(np.asarray(du)[1, 1:, :])

    adv_min = lego_adv(h_u_min, h_v_min)
    adv_e3u = lego_adv(h_u_nemo, h_v_nemo)
    E1 = record("X1_hadv_thickness_min_rule_vs_e3u_kmm", np.where(act, adv_min - adv_e3u, 0.0),
                "ocean_pe_latlon_cgrid.py:1221,1364 (h_u = min_cell_to_uface(compute_layer_thickness(eta_stage))) "
                "vs domzgr_substitute.h90:127 e3u(Kmm)=e3u_0*(1+r3u(Kmm)); legoESM's own operator, one variable")
    _hmin = np.asarray(h_u_min)[1, 1:, :nlev]
    _hnem = np.asarray(h_u_nemo)[1, 1:, :nlev]
    _rel = np.where(act[:, :nlev], (_hmin - _hnem) / np.where(_hnem > 0, _hnem, 1.0), 0.0)
    rows["X1_hadv_thickness_min_rule_vs_e3u_kmm"]["max_rel_delta_h_u_faces_18_22"] = {
        int(i): float(np.max(np.abs(_rel[i]))) for i in (18, 19, 20, 21, 22)}
    rows["X1_hadv_thickness_min_rule_vs_e3u_kmm"]["signed_rel_delta_h_u_faces_18_22_k0"] = {
        int(i): float(_rel[i, 0]) for i in (18, 19, 20, 21, 22)}
    # legoESM operator on the NEMO thickness vs the NumPy transcription on NEMO's own transport:
    # the operator itself, with h_u = e3u(Kmm) and legoESM's zub, against dynadv_up3 on the dumped zFu.
    rows["X1_hadv_thickness_min_rule_vs_e3u_kmm"]["lego_operator_on_e3u_kmm_vs_transcription_on_dumped_zFu"] = float(
        np.abs(bc_row(np.where(act, adv_e3u - adv_h_n, 0.0), act, mesh)[0]).max())
    # transcription cross-check of the same candidate (NEMO formulas with h_min substituted)
    h_min_row = pad(np.asarray(h_u_min)[1, 1:, :])
    zub_l_row = np.asarray(zub_l)[1, 1:]
    Fu_l = mesh["e2u"][:, None] * h_min_row * (u2n + zub_l_row[:, None]) * mesh["umask"]
    adv_h_min_tr, _ = up3_row(u2n, Fu_l, Fw3, h_min_row, e3u_m, mesh)
    record("X1t_transcription_h_min", np.where(act, adv_h_min_tr - adv_h_n, 0.0),
           "dynadv_up3.F90:137-214 with zFu=e2u*h_min*(u+zub_lego) and /h_min (horizontal only)")

    # X1 at STAGE 2 (Kmm = stage-1 Kaa, ssh = ssha/3, dt/2): same one-variable
    # replay on NEMO's u1 against the measured stage-2 residual.
    ssh1 = row(s1["ssh"])
    r3u_13 = r3u_row(ssh1, mesh)
    qfac2 = ((1.0 + r3u_13) / (1.0 + r3u_m))[:, None]
    u1n_lego = jnp.asarray(to_lego_u(u1n))
    eta1_lego = jnp.asarray(np.pad(ssh1[None], ((1, lat - 2), (0, 0))))
    h_k_s1 = compute_layer_thickness(eta1_lego, init.H_bathy.data, z,
                                     min_water_column_m=cfg.min_water_column_m)
    h_u_min1, h_v_min1 = min_cell_to_uface(h_k_s1), min_cell_to_vface(h_k_s1, grid)
    h_u_nemo1, h_v_nemo1, _, _ = _nemo_ws_qco_stage_faces(eta1_lego, h_ref, u_mask3, v_mask3, grid)
    zub1 = jnp.where(
        H_u_pre > 0.0,
        jnp.asarray(np.pad(un_adv[None], ((1, lat - 2), (1, 0)))) / H_u_safe
        - jnp.sum(u1n_lego * h_u_pre, axis=-1) / H_u_safe, 0.0) * init.u_mask.data
    tv1 = ((u1n_lego + zub1[..., None]) * u_mask3, v_zero)

    def lego_adv1(h_u, h_v):
        du, _, _, _ = _bc_horizontal_momentum_advection_flux_form(
            zeros_u, zeros_v, u1n_lego, v_zero, h_u, h_v, u_mask3, v_mask3,
            init.land_mask.data, grid, cfg, transport_velocity=tv1, up3_upwind_selector="velocity")
        return pad(np.asarray(du)[1, 1:, :])

    R2 = bc_row(np.where(act, u2l - u2n, 0.0), act, mesh)[0]
    E1_s2 = bc_row(0.5 * dt * qfac2 * np.where(act, lego_adv1(h_u_min1, h_v_min1) - lego_adv1(h_u_nemo1, h_v_nemo1), 0.0),
                   act, mesh)[0]
    rows["X1_stage2_hadv_thickness"] = {
        "source": "same as X1 at stage 2 (Kmm ssh = stage-1 Kaa, rDt = dt/2) on NEMO's u1",
        "target_stage2_R_max_abs": float(np.abs(R2).max()),
        "max_abs_prediction_m_s": float(np.abs(E1_s2).max()),
        "faces": {int(i): float(np.abs(E1_s2[i]).max()) for i in (18, 19, 20, 21, 22)},
        "fit_R2_on_E": fit_pattern(R2, E1_s2, act),
        "max_abs_R2_minus_E": float(np.abs(R2 - E1_s2).max())}
    print(f"[cand X1_stage2] R2 max {rows['X1_stage2_hadv_thickness']['target_stage2_R_max_abs']:.3e}  max|E| "
          f"{rows['X1_stage2_hadv_thickness']['max_abs_prediction_m_s']:.3e}  corr "
          f"{rows['X1_stage2_hadv_thickness']['fit_R2_on_E']['corr']:+.5f} slope "
          f"{rows['X1_stage2_hadv_thickness']['fit_R2_on_E']['slope']:+.4f}  max|R2-E| "
          f"{rows['X1_stage2_hadv_thickness']['max_abs_R2_minus_E']:.3e}", flush=True)

    # X3: vertical UP3 implementation on NEMO's operands (u2, zFw) vs the transcription
    w_T = np.zeros((lat, ni, nlev + 1))
    w_T[1] = (Fw3[:, :nlev + 1] / mesh["e1e2t"][:, None])
    w_face = interp_cell_to_uface(jnp.asarray(w_T))
    vert_lego = pad(np.asarray(nemo_up3_vertical_momentum_advection(
        u2n_lego * u_mask3, w_face, h_u_nemo, face_active=u_mask3))[1, 1:, :])
    record("X3_vertical_up3_operator", np.where(act, vert_lego - adv_v_n, 0.0),
           "dynadv_up3.F90:239-358 transcription vs nemo_up3_vertical_momentum_advection on NEMO's (u2, zFw, e3u(Kmm))")

    # X5: the stage-3 transport triplet legoESM builds vs NEMO's dumped zFu / zFw
    Hu_avg_from_eta = np.zeros(ni)
    area_row = mesh["e1e2t"]
    deta = (eta3l[1] - np.asarray(init.eta.data)[1])
    # continuity on the row: (Hu_i - Hu_{i-1})*e2u = -area*deta/dt  ->  cumulative from the closed western wall
    Hu_avg_from_eta = np.cumsum(-area_row * deta / dt / mesh["e2u"])
    rows_tr = {"max_abs_Hu_from_lego_eta_minus_un_adv": float(np.abs(np.where(face2d, Hu_avg_from_eta - un_adv, 0.0)).max()),
               "max_abs_un_adv": float(np.abs(un_adv).max())}
    geom2 = _nemo_ws_stage_transport(
        (u2n_lego, v_zero), h_k_stage, 2, eta_stage=eta2_lego, h_ref=h_ref,
        Hu_avg=jnp.asarray(np.pad(un_adv[None], ((1, lat - 2), (1, 0)))), Hv_avg=jnp.zeros_like(init.v.data[..., 0]),
        u_mask_3d=u_mask3, v_mask_3d=v_mask3, grid=grid, z_coord=z,
        H_bathy=init.H_bathy.data, config=cfg, dt=dt,
        eta_before=init.eta.data, eta_after=etaa_lego)
    mf_u = pad(np.asarray(geom2[0])[1, 1:, :])
    w_stage = np.asarray(geom2[2])[1]
    Fu_l_stage = mf_u * mesh["e2u"][:, None]
    rows_tr["max_abs_stage_transport_minus_dumped_zFu_per_e2u"] = float(
        np.abs(np.where(act, Fu_l_stage - Fu3, 0.0)).max() / mesh["e2u"][20])
    rows_tr["max_abs_stage_w_minus_dumped_ww"] = float(np.abs(w_stage[:, :nlev + 1] - w_T[1][:, :nlev + 1])[mesh["tmask"][:, :nlev + 1] > 0].max())
    Fw_l = w_stage[:, :nlev + 1] * mesh["e1e2t"][:, None]
    Fw_l_full = np.zeros_like(Fw3)
    Fw_l_full[:, :nlev + 1] = Fw_l
    adv_h_tl, adv_v_tl = up3_row(u2n, Fu_l_stage, Fw_l_full, e3u_m, e3u_m, mesh)
    record("X5_stage_transport_triplet", np.where(act, adv_h_tl + adv_v_tl - adv_h_n - adv_v_n, 0.0),
           "stprk3_stg.F90:257-304 triplet from _nemo_ws_stage_transport (NEMO's u2, NEMO's un_adv) vs the dumped zFu/zFw, through dynadv_up3")
    rows["X5_stage_transport_triplet"].update(rows_tr)

    # X6: the implicit vertical solve on NEMO's pre-solve field and operands
    pre_lego = jnp.asarray(to_lego_u(u3n_pre))
    e3u_aa_l = jnp.asarray(to_lego_u(mesh["e3u_0"] * (1.0 + r3u_a)[:, None]))
    e3uw_m_row = mesh["e3w_1d"][None, 1:nlev] * (1.0 + r3u_m)[:, None]
    e3w_now = jnp.asarray(np.pad(e3uw_m_row[None], ((1, lat - 2), (1, 0), (0, 0))))
    avm_face = jnp.full(e3w_now.shape, avm, dtype=e3w_now.dtype)
    solved = pad(np.asarray(implicit_vertical_diffusion_nemo_momentum(
        pre_lego, avm_face, e3u_aa_l, e3w_now, dt, u_mask3))[1, 1:, :])
    E6 = bc_row(np.where(act, solved - u3n, 0.0), act, mesh)[0]
    rows["X6_implicit_zdf_solver"] = {
        "source": "dynzdf.F90:181-196,322-345 M vs implicit_vertical_diffusion_nemo_momentum on NEMO's pre-solve field",
        "max_abs_prediction_m_s": float(np.abs(E6).max()),
        "faces": {int(i): float(np.abs(E6[i]).max()) for i in (18, 19, 20, 21, 22)},
        "fit_R_on_E": fit_pattern(R, E6, act),
        "nemo_zdf_increment_structure": controls["nemo_stage3_zdf_increment_faces"]}
    print(f"[cand X6_implicit_zdf_solver] max|E| {rows['X6_implicit_zdf_solver']['max_abs_prediction_m_s']:.3e}")

    # X7: the depth-mean weights of the stage correction (h_u_pre vs e3u_0)
    hpre_row = pad(np.asarray(h_u_pre)[1, 1:, :])
    rows["X7_stage_mean_weights"] = {
        "source": "stprk3_stg.F90:440 e3u_0/r1_hu_0 vs _replace_stage_mean h_u_pre/H_u_pre",
        "max_abs_delta_weights_m": float(np.abs(np.where(act, hpre_row - mesh["e3u_0"], 0.0)).max())}
    print(f"[cand X7_stage_mean_weights] max|h_u_pre - e3u_0| {rows['X7_stage_mean_weights']['max_abs_delta_weights_m']:.3e}")

    # The selector pattern, as a reference row (already removed by 42ac525cc)
    adv_h_ts, _ = up3_row(u2n, Fu3, Fw3, e3u_m, e3u_m, mesh, selector="transport")
    record("S44_reference_up3_selector_already_fixed", np.where(act, adv_h_ts - adv_h_n, 0.0),
           "dynadv_up3.F90:166-170 transport-sign vs velocity-sign branch (reference only)")

    # Combined X1 + everything measured: what remains of R after removing E1
    resid_after_X1 = R - E1
    rows["R_minus_X1"] = {"max_abs": float(np.abs(resid_after_X1).max()),
                          "faces": {int(i): float(np.abs(resid_after_X1[i]).max()) for i in (18, 19, 20, 21, 22)}}
    print(f"[R - E(X1)] max {rows['R_minus_X1']['max_abs']:.3e}  faces {rows['R_minus_X1']['faces']}")

    report = {"worktree": worktree_stamp(), "format": "nemo-testcase-l1-stage3-remainder-candidates-v1", "case": CASE,
              "legoesm_git_sha": legoesm_git_sha, "backend": jax.default_backend(), "dtypes": dtypes,
              "dt_s": dt, "nlev": nlev, "oracle_root": str(ROOT), "artifacts": artifacts,
              "target_R": target, "controls": controls, "remainder_D_l": remainder, "candidates": rows,
              "prediction_convention": "E = dt*(1+r3u(Kmm))/(1+r3u(Kaa)) * bc(RHS_lego - RHS_nemo); fit_R_on_E "
                                       "regresses the measured R on E (slope 1, corr 1 = E is R)"}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "candidates.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    print("wrote", out)
    return report


# ---------------------------------------------------------------------------
# (B2) the barotropic slow forcing: the X1 mechanism at the step entry
# ---------------------------------------------------------------------------
def slow(*, out_dir: Path, allow_dirty: bool, kts=(2, 3, 4)) -> dict:
    """``slow_u`` (NEMO ``zu_frc`` = ``Ue_rhs``) residual at substep 1 from an
    EXACT NEMO kt-entry, against the X1 prediction: the h_u_pre-weighted
    depth mean of legoESM's flux-form advection with the min-rule Kbb
    thickness minus the same operator with NEMO's ``e3u(Kbb)``."""
    SWEEP, TRAJ, BARO = gates()
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _nemo_ws_qco_stage_faces
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        _bc_horizontal_momentum_advection_flux_form)
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d, min_cell_to_uface, min_cell_to_vface)
    from legoesm.ocean.vertical import compute_layer_thickness

    allow_dirty_stamps(allow_dirty)

    legoesm_git_sha = git_sha(allow_dirty)
    set_fp64()
    card = build_nemo_testcase_card(CASE)
    cfg, z, grid, init = card.recipe.model_config, card.recipe.z_coord, card.recipe.grid, card.recipe.initial_state
    dtypes = dtype_report(card, init)
    masks = TRAJ.expected_masks(card)
    nlev = z.n_levels
    mesh = read_row_mesh_full(ROOT, SWEEP)
    h_ref = compute_layer_thickness(jnp.zeros_like(init.eta.data), init.H_bathy.data, z,
                                    min_water_column_m=cfg.min_water_column_m)
    u_mask3, v_mask3 = compute_face_masks_3d(z.is_active, grid)
    u_mask3, v_mask3 = u_mask3.astype(h_ref.dtype), v_mask3.astype(h_ref.dtype)
    zeros_u, zeros_v = jnp.zeros_like(init.u.data), jnp.zeros_like(init.v.data)
    wet_u = np.asarray(init.u_mask.data)[1, 1:] > 0.5
    artifacts, rows = {}, {}
    for kt in kts:
        entry_path = ROOT / f"oracle_step_entry_kt{kt:08d}.bin"
        trace_path = BTWALK_ROOT / f"oracle_overflow_bt_substeps_kt{kt:08d}_call1.bin"
        artifacts[entry_path.name] = sha256(entry_path)
        artifacts[trace_path.name] = sha256(trace_path)
        entry = TRAJ.read_entry(entry_path, CASE)
        oracle = BARO.read_oracle_trace(trace_path, expected_kt=kt)
        cand = BARO.capture_legoesm_trace(flux_form_override=None, kt=kt, reseed_entry=entry)
        res = np.where(wet_u, cand["substeps"][0]["slow_u"][1] - oracle["substeps"][0]["slow_u"][1], 0.0)
        # X1 prediction on the exact entry (NEMO's u(kt), ssh(kt)); no separate transport at the step entry
        st = BARO.state_from_oracle_entry(init, entry, masks)
        u_l = jnp.asarray(st.u.data)
        h_k = compute_layer_thickness(st.eta.data, init.H_bathy.data, z, min_water_column_m=cfg.min_water_column_m)
        h_u_min, h_v_min = min_cell_to_uface(h_k), min_cell_to_vface(h_k, grid)
        h_u_nemo, h_v_nemo, _, _ = _nemo_ws_qco_stage_faces(st.eta.data, h_ref, u_mask3, v_mask3, grid)

        def adv(h_u, h_v):
            du, _, _, _ = _bc_horizontal_momentum_advection_flux_form(
                zeros_u, zeros_v, u_l, jnp.asarray(st.v.data), h_u, h_v, u_mask3, v_mask3,
                init.land_mask.data, grid, cfg, transport_velocity=None, up3_upwind_selector="velocity")
            return np.asarray(du)

        d_adv = adv(h_u_min, h_v_min) - adv(h_u_nemo, h_v_nemo)
        hpre = np.asarray(h_u_min)
        E = (np.sum(d_adv * hpre, axis=-1) / np.maximum(np.sum(hpre, axis=-1), 1e-10))[1, 1:]
        E = np.where(wet_u, E, 0.0)
        rel = np.where(wet_u, (np.asarray(h_u_min)[1, 1:, 0] - np.asarray(h_u_nemo)[1, 1:, 0])
                       / np.where(np.asarray(h_u_nemo)[1, 1:, 0] > 0, np.asarray(h_u_nemo)[1, 1:, 0], 1.0), 0.0)
        fit = fit_pattern(res[None, :], E[None, :], wet_u[None, :])
        rows[kt] = {
            "slow_u_residual_max_abs": float(np.abs(res).max()),
            "slow_u_residual_faces": {int(i): float(res[i]) for i in FRONT_FACES},
            "prediction_E_max_abs": float(np.abs(E).max()),
            "prediction_E_faces": {int(i): float(E[i]) for i in FRONT_FACES},
            "rel_delta_h_u_k0_faces": {int(i): float(rel[i]) for i in FRONT_FACES},
            "fit_residual_on_E": fit,
            "max_abs_residual_minus_E": float(np.abs(res - E).max()),
            "reseeded_from_oracle_entry": cand["reseeded_from_oracle_entry"],
        }
        print(f"[slow kt={kt}] slow_u residual {rows[kt]['slow_u_residual_max_abs']:.3e} "
              f"faces19/20/21 {res[19]:+.2e}/{res[20]:+.2e}/{res[21]:+.2e}; E {rows[kt]['prediction_E_max_abs']:.3e} "
              f"faces {E[19]:+.2e}/{E[20]:+.2e}/{E[21]:+.2e}; corr {fit['corr']:+.5f} slope {fit['slope']:+.4f} "
              f"max|res-E| {rows[kt]['max_abs_residual_minus_E']:.3e}", flush=True)
        jax.clear_caches()
    report = {"worktree": worktree_stamp(), "format": "nemo-testcase-l1-stage3-remainder-slow-v1", "case": CASE,
              "legoesm_git_sha": legoesm_git_sha, "backend": jax.default_backend(), "dtypes": dtypes,
              "artifacts": artifacts, "rows": rows,
              "prediction_convention": "E = h_u_pre-weighted depth mean of adv_lego(h_min(Kbb)) - adv_lego(e3u(Kbb)) on the exact NEMO entry"}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "slow.json"
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    print("wrote", out)
    return report


@scoped_allow_dirty
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("growth", "candidates", "slow"))
    parser.add_argument("--max-kt", type=int, default=10)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-frames", action="store_true",
                        help="growth: skip the plus_du external-solve frame captures")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "growth":
        growth(max_kt=args.max_kt, out_dir=args.out_dir, allow_dirty=args.allow_dirty,
               frames=not args.no_frames)
    elif args.mode == "candidates":
        candidates(out_dir=args.out_dir, allow_dirty=args.allow_dirty)
    else:
        slow(out_dir=args.out_dir, allow_dirty=args.allow_dirty)
    return 0


if __name__ == "__main__":
    sys.exit(main())
