#!/usr/bin/env python3
"""Scaling-first WS-RK3 stage owner controls for the NEMO testcase lane.

The public testcase cards are never changed here.  Each arm is a private,
one-variable test hook, and every stage row compares NEMO's instantaneous Kaa
velocity with legoESM's instantaneous prognostic U-face velocity at that same
stage.  Final rows use the next Nbb entry state.  Owner labels are derived only
after the arm movement is compared with the faithful residual scale.
"""

from __future__ import annotations

import argparse
import hashlib
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


BAR = 1.0e-15
ROOTS = {
    "LOCK_EXCHANGE-zco": Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/lock_kt1_10"),
    "OVERFLOW-zps": Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_10"),
}
DIMS = {
    "LOCK_EXCHANGE-zco": (134, 7, 21),
    "OVERFLOW-zps": (206, 7, 101),
}
EXPECTED_LEVELS = {
    1: {"Kaa": 3},
    2: {"Kaa": 2},
    3: {"Kaa": 3},
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def git_sha(*, allow_dirty: bool = False) -> str:
    """Exact legoESM producer revision (fails closed on tracked dirt)."""
    from legoesm.ocean.fidelity.provenance import git_sha as _stamp

    try:
        return _stamp(allow_dirty=allow_dirty)
    except RuntimeError as error:
        raise GateError(f"cannot stamp legoESM git SHA: {error}") from error


def require_planted(rows: list[dict], name: str) -> None:
    """A planted +1.0 must be visible in its row, else the gate is broken."""
    row = next((row for row in rows if row["name"] == name), None)
    require(row is not None, f"planted row {name} missing")
    require(row["status"] == "DEBT" and row["absolute_max"] >= 0.5,
            f"planted control {name} did not land: {row['absolute_max']:.3e}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def read_stage(path: Path, case: str, expected_stage: int) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kaa, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_STAGE_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *DIMS[case], 2, 64), f"{path}: bad header {header}"
    )
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    require(
        (kt, stage, kaa) == (1, expected_stage, EXPECTED_LEVELS[expected_stage]["Kaa"]),
        f"{path}: wrong time-level registry {header}",
    )
    return {
        "kt": kt,
        "stage": stage,
        "Kaa": kaa,
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count : 2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count : 3 * count], nx, ny, nz),
        # The record DOES carry v: stprk3.F90:326-327 writes ts, uu, vv, ssh
        # and the payload check above is 4*count + nx*ny for exactly that
        # reason.  This reader used to skip the third block, which made a
        # Rule-12 discharge on these cards report the v face UNMEASURED when
        # the oracle had provided it all along (Rule 1: coverage is driven by
        # what the oracle provides, not by what the reader asks for).
        "v": _xyz(values[3 * count : 4 * count], nx, ny, nz),
        "ssh": _xy(values[4 * count :], nx, ny),
    }


def read_entry(path: Path, case: str, expected=(2, 3)) -> dict:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, nbb, nx, ny, nz, ntr, bits = header
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    require(
        (version, nx, ny, nz, ntr, bits) == (1, *DIMS[case], 2, 64), f"{path}: bad header {header}"
    )
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, f"{path}: bad payload")
    require((kt, nbb) == tuple(expected), f"{path}: expected kt/Nbb {expected}")
    return {
        "T": _xyz(values[:count], nx, ny, nz),
        "S": _xyz(values[count : 2 * count], nx, ny, nz),
        "u": _xyz(values[2 * count : 3 * count], nx, ny, nz),
        "ssh": _xy(values[4 * count :], nx, ny),
    }


# ---------------------------------------------------------------------------
# hpg_sco replay (NEMO 5.0.2 dynhpg.F90:340-390 with the key_qco macros
# domzgr_substitute.h90:131,139,145 and r3t = ssh/ht_0, domqco.F90:160) on the
# single wet row of the OVERFLOW-zps card.  Ported from the independent hunt
# (docs/ocean/fidelity/testcases/overflow_kt2_independent_hunt.md) so its
# numbers are gate numbers.  Face index i is the U face east of T column i.
# ---------------------------------------------------------------------------
def read_row_mesh(root: Path) -> dict:
    import netCDF4

    with netCDF4.Dataset(str(root / "mesh_mask.nc")) as data:
        def field(name):
            return np.asarray(data.variables[name][:]).squeeze()

        tmask = field("tmask")[:, 1, :].T.astype(np.float64)
        umask = field("umask")[:, 1, :].T.astype(np.float64)
        e3t_0 = np.asarray(field("e3t_0")[:, 1, :].T, dtype=np.float64)
        e3u_0 = np.asarray(field("e3u_0")[:, 1, :].T, dtype=np.float64)
        e3w_1d = np.asarray(field("e3w_1d"), dtype=np.float64)
        gdept_1d = np.asarray(field("gdept_1d"), dtype=np.float64)
        e1u = np.asarray(field("e1u")[1], dtype=np.float64)
        # tests/OVERFLOW/MY_SRC/usrdef_zgr.F90:164-166 (key_vco_3d): the
        # partial cell reduces e3t_0/e3u_0 only; e3w/gdept stay 1-D ladders.
        require(
            np.array_equal(field("e3w_0")[:, 1, :].T, np.broadcast_to(e3w_1d, e3t_0.shape))
            and np.array_equal(field("gdept_0")[:, 1, :].T, np.broadcast_to(gdept_1d, e3t_0.shape)),
            "mesh_mask e3w_0/gdept_0 are not the 1-D ladders: hpg_sco replay not applicable",
        )
    ht_0 = np.sum(e3t_0 * tmask, axis=-1)
    hu_0 = np.sum(e3u_0 * umask, axis=-1)
    return {
        "tmask": tmask, "umask": umask, "e3t_0": e3t_0, "e3u_0": e3u_0,
        "e3w_1d": e3w_1d, "gdept_1d": gdept_1d, "e1u": e1u,
        "r1_ht_0": np.where(ht_0 > 0, 1.0 / np.maximum(ht_0, 1e-300), 0.0),
        "hu_0": hu_0,
    }


def hpg_sco_row(
    mesh: dict, T, S, ssh, eos_fn, g: float, rho0: float, eos_pressure_per_metre: float,
) -> np.ndarray:
    """dynhpg.F90:340-390 u-trend ``zhpi + zuap`` on one row, (i_face, k).

    ``g`` is the oracle's ``grav`` (the HPG coefficient); ``eos_pressure_per_
    metre`` is whatever the supplied ``eos_fn`` divides by to recover depth
    (legoESM ``nemo_roquet_eos``: ``zh = p/(rho0*constants.g)``, so pass
    ``rho0*constants.g`` to hand it the exact live ``gdept``).
    """
    T = np.asarray(T, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    ssh = np.asarray(ssh, dtype=np.float64)
    ni, nz = T.shape
    r3t = ssh * mesh["r1_ht_0"]
    gdept = mesh["gdept_1d"][None, :] * (1.0 + r3t)[:, None]     # gdept_0*(1+r3t)
    e3w = mesh["e3w_1d"][None, :] * (1.0 + r3t)[:, None]         # E3w_0*(1+r3t)
    gdept_z0 = gdept - ssh[:, None]                               # gdept - ssh
    rho = np.asarray(eos_fn(T, S, eos_pressure_per_metre * gdept), dtype=np.float64)
    rhd = (rho / rho0 - 1.0) * mesh["tmask"]
    zcoef0 = -g * 0.5
    trend = np.zeros((ni, nz))
    west = slice(0, ni - 1)
    east = slice(1, ni)
    r1_e1u = 1.0 / mesh["e1u"][west]
    zhpi = zcoef0 * r1_e1u * (e3w[east, 0] * rhd[east, 0] - e3w[west, 0] * rhd[west, 0])
    zuap = -zcoef0 * (rhd[east, 0] + rhd[west, 0]) * (gdept_z0[east, 0] - gdept_z0[west, 0]) * r1_e1u
    trend[west, 0] = zhpi + zuap
    for k in range(1, nz - 1):
        zhpi = zhpi + zcoef0 * r1_e1u * (
            e3w[east, k] * (rhd[east, k] + rhd[east, k - 1])
            - e3w[west, k] * (rhd[west, k] + rhd[west, k - 1]))
        zuap = -zcoef0 * (rhd[east, k] + rhd[west, k]) * (gdept_z0[east, k] - gdept_z0[west, k]) * r1_e1u
        trend[west, k] = zhpi + zuap
    return trend * mesh["umask"]


def face_thickness_and_qco_scaling(card, masks, nlev, oracle_stages, entry1, hu0):
    """Preregistered scaling for the two stage-geometry hypotheses.

    Computed from the ORACLE kt=1 dumps and the card's own geometry only --
    no legoESM arm enters -- so the frozen predictions in
    ``nemo_testcases_l1_overflow_face_thickness_preregister.md`` are
    reproduced on every run.

    H1: NEMO's stage face thickness is ``e3u_0*(1+r3u(Kmm))``
    (domqco.F90:219-222, domzgr_substitute.h90:127); legoESM's pre-fix rule
    was ``min`` of the two stretched T thicknesses.  The oracle's per-stage
    tracer increment is 100% advective on these cards (K_h = K_v = 0, no
    surface forcing), so the predicted T movement is the relative transport
    error times that increment.

    H2: NEMO weights every stage velocity update by ``(1+r3u(Kbb))`` /
    ``(1+r3u(Kmm))`` / ``(1+r3u(Kaa))`` (stprk3_stg.F90:373-378).  With
    ``uu(Kbb) = 0`` at kt=1 the omitted factor leaves
    ``u_lego - u_nemo = u_nemo*(r3u(Kaa)-r3u(Kmm))/(1+r3u(Kmm))``, and the
    stage barotropic correction removes its depth mean.
    """
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
        min_cell_to_uface,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    init = card.recipe.initial_state
    z_coord = card.recipe.z_coord
    grid = card.recipe.grid
    min_wc = card.recipe.model_config.min_water_column_m
    h_ref = compute_layer_thickness(
        jnp.zeros_like(init.eta.data), init.H_bathy.data, z_coord,
        min_water_column_m=min_wc)
    u_mask3, v_mask3 = compute_face_masks_3d(z_coord.is_active, grid)
    u_mask3 = u_mask3.astype(h_ref.dtype)
    v_mask3 = v_mask3.astype(h_ref.dtype)
    wet_u = np.asarray(masks["u"], dtype=bool)
    wet_t = np.asarray(masks["T"], dtype=bool)

    kmm_ssh = {1: entry1["ssh"], 2: oracle_stages[1]["ssh"],
               3: oracle_stages[2]["ssh"]}
    kmm_T = {1: entry1["T"], 2: oracle_stages[1]["T"],
             3: oracle_stages[2]["T"]}
    h1_rows, h2_rows = [], []
    predicted_T = 0.0
    for stage in (1, 2, 3):
        eta = jnp.asarray(kmm_ssh[stage], dtype=h_ref.dtype)
        h_stage = compute_layer_thickness(
            eta, init.H_bathy.data, z_coord, min_water_column_m=min_wc)
        nemo_u = np.asarray(_nemo_ws_qco_stage_faces(
            eta, h_ref, u_mask3, v_mask3, grid)[0])[:, 1:, :nlev]
        min_u = np.asarray(min_cell_to_uface(h_stage))[:, 1:, :nlev]
        delta = np.abs(min_u - nemo_u)[wet_u]
        relative = float(np.max(delta / nemo_u[wet_u]))
        d_tracer = float(np.max(np.abs(
            (oracle_stages[stage]["T"][..., :nlev]
             - kmm_T[stage][..., :nlev])[wet_t])))
        predicted_T += relative * d_tracer
        h1_rows.append({
            "stage": stage,
            "max_abs_delta_e3u_m": float(np.max(delta)),
            "max_relative_delta_e3u": relative,
            "oracle_stage_max_abs_delta_T_K": d_tracer,
            "predicted_T_movement_K": relative * d_tracer,
        })

        r3a = np.asarray(_nemo_ws_qco_stage_faces(
            jnp.asarray(oracle_stages[stage]["ssh"], dtype=h_ref.dtype),
            h_ref, u_mask3, v_mask3, grid)[2])[:, 1:] - 1.0
        r3m = np.asarray(_nemo_ws_qco_stage_faces(
            eta, h_ref, u_mask3, v_mask3, grid)[2])[:, 1:] - 1.0
        u_bc = remove_depth_mean(
            oracle_stages[stage]["u"][..., :nlev], hu0, masks["u"])
        h2_rows.append({
            "stage": stage,
            "max_abs_r3u_kaa": float(np.max(np.abs(r3a[wet_u.any(axis=-1)]))),
            "max_abs_r3u_kaa_minus_kmm": float(
                np.max(np.abs((r3a - r3m)[wet_u.any(axis=-1)]))),
            "predicted_baroclinic_u_movement_m_s": float(np.max(np.abs(
                ((r3a - r3m) / (1.0 + r3m))[..., None] * u_bc)[wet_u])),
        })
    return {
        "source": (
            "domqco.F90:219-222; domzgr_substitute.h90:127; "
            "stprk3_stg.F90:272-273,373-378; dynzdf.F90 key_qco branch"),
        "inputs": "oracle kt=1 stage dumps and the card geometry only",
        "h1_face_thickness": h1_rows,
        "h1_predicted_total_T_movement_K": predicted_T,
        "h2_qco_stage_factor": h2_rows,
    }


def score(name: str, oracle, candidate, mask, *, plant=False, quantity="u") -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate)
    active = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == active.shape, f"{name}: shape mismatch")
    require(candidate.dtype == np.float64, f"{name}: candidate {candidate.dtype}")
    require(bool(active.any()), f"{name}: empty mask")
    if plant:
        candidate = candidate.copy()
        candidate[tuple(np.argwhere(active)[0])] += 1.0
    require(np.all(np.isfinite(candidate[active])), f"{name}: non-finite")
    from legoesm.ocean.fidelity.ulp_move_gate import record_residual_field
    record_residual_field(name, oracle, candidate, active)
    absolute = float(np.max(np.abs(candidate[active] - oracle[active])))
    reference = float(np.max(np.abs(oracle[active])))
    normalized = absolute / max(reference, 1.0)
    # AT-BAR is a RELATIVE verdict, so it can be true of a row that is not bit
    # for bit.  Report the bit-unequal COUNT next to it, the way the lane-2
    # GYRE gate already does (nemo_testcase_l2_gyre_phase3_gate.py:607-616),
    # so a Rule-12 row can never quote AT-BAR without saying how far from
    # exact it is.
    n_unequal = int(np.count_nonzero(
        candidate[active].view(np.uint64) != oracle[active].view(np.uint64)))
    return {
        "name": name,
        "status": "AT-BAR" if normalized <= BAR else "DEBT",
        "exact": bool(np.array_equal(candidate[active], oracle[active])),
        "n_unequal": n_unequal,
        "normalized_max_abs": normalized,
        "absolute_max": absolute,
        "reference_max_abs": reference,
        "bar": BAR,
        "n": int(active.sum()),
        "frame": (
            "instantaneous_prognostic_Kaa" if quantity == "u" else "instantaneous_tracer_Nbb"
        ),
        "staggering_and_reduction": (
            "oracle uu(:,:,:,Kaa) and legoESM's exposed WS-RK3 stage u are "
            "both instantaneous 3-D C-grid U-face velocities; both are scored "
            "on the same wet U-face mask by elementwise L-infinity, with no "
            "vertical or substep-time reduction"
            if quantity == "u"
            else "oracle ts(:,:,:,temperature,Nbb) and legoESM T are both "
            "instantaneous 3-D T-point tracers; both are scored on the same "
            "wet T-cell mask by elementwise L-infinity, with no vertical or "
            "time reduction"
        ),
    }


def movement(faithful, control, mask) -> dict:
    faithful = np.asarray(faithful, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    value = float(np.max(np.abs(control[active] - faithful[active])))
    return {"absolute_max": value}


def remove_depth_mean(values, thickness, mask):
    values = np.asarray(values, dtype=np.float64)
    thickness = np.asarray(thickness, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    weights = thickness * active
    mean = np.sum(values * weights, axis=-1) / np.maximum(
        np.sum(weights, axis=-1), np.finfo(np.float64).tiny)
    return values - mean[..., None]


def expected_masks(card) -> dict:
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    u[:, -1] = False
    # S, v and ssh are built by the SAME rule the lane-2 GYRE gate uses
    # (nemo_testcase_l2_gyre_phase3_gate.py:573-581), so one convention serves
    # every card.  They were absent here, not because these cards lack those
    # faces, but because nothing had asked for them yet -- which is how a
    # Rule-12 discharge came to report the v face UNMEASURED on a record that
    # carries it.
    v = active & np.roll(active, -1, axis=0)
    v[-1] = False
    return {"T": active, "S": active, "u": u, "v": v, "ssh": wet}


def run_round49_pair_boundary(
    root: Path, *, plant: bool = False, allow_dirty: bool = False,
) -> tuple[dict, dict[str, np.ndarray]]:
    """One-compile tracer-to-HPG boundary for the round-49 pair walk.

    The two WRITE-only hooks share one ordinary compiled step: T/S/eta expose
    stage-1 Kaa, while u/v expose the stage-2 HPG operator that consumes that
    Kaa after the pointer swap.  No diagnostic array is read by production.
    """
    import jax
    from legoesm import constants as _constants
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    allow_dirty_stamps(allow_dirty)
    revision = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(
        get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
        "precision policy is not fp64/libm",
    )
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")

    case = "OVERFLOW-zps"
    card = build_nemo_testcase_card(case)
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    stage_path = root / "oracle_stage_kt00000001_s1.bin"
    mesh_path = root / "mesh_mask.nc"
    require(stage_path.is_file(), f"missing {stage_path}")
    require(mesh_path.is_file(), f"missing {mesh_path}")
    oracle = read_stage(stage_path, case, 1)

    hooks = _NEMOWSRK3TestHooks(
        expose_tracer_stage=1,
        expose_momentum_operator="hpg",
        expose_momentum_operator_stage=2,
    )
    exposed = LatLonCGridOceanModel(
        card.recipe.grid,
        card.recipe.z_coord,
        card.recipe.model_config,
        _nemo_ws_test_hooks=hooks,
    ).step(card.recipe.initial_state, dt=card.dt_s)
    arrays = {
        "stage1_T": np.asarray(exposed.T.data, dtype=np.float64),
        "stage1_S": np.asarray(exposed.S.data, dtype=np.float64),
        "stage1_ssh": np.asarray(exposed.eta.data, dtype=np.float64),
        "stage2_hpg_u": np.asarray(exposed.u.data, dtype=np.float64),
        "stage2_hpg_v": np.asarray(exposed.v.data, dtype=np.float64),
    }
    if plant:
        planted = arrays["stage1_T"].copy()
        index = tuple(np.argwhere(masks["T"])[0])
        planted[index] += 1.0
        arrays["stage1_T"] = planted

    rows = [
        score(
            f"{case}.kt1.stage1.pair_boundary.T",
            oracle["T"][..., :nlev], arrays["stage1_T"], masks["T"],
            quantity="T",
        ),
        score(
            f"{case}.kt1.stage1.pair_boundary.S",
            oracle["S"][..., :nlev], arrays["stage1_S"], masks["S"],
            quantity="T",
        ),
        score(
            f"{case}.kt1.stage1.pair_boundary.ssh",
            oracle["ssh"], arrays["stage1_ssh"], masks["ssh"],
            quantity="T",
        ),
    ]
    if plant:
        require(
            rows[0]["status"] == "DEBT" and rows[0]["absolute_max"] >= 0.5,
            "round-49 planted tracer change did not make the gate red",
        )

    # Independent source replay of the HPG expected from NEMO's stage-1
    # T/S/ssh.  This is the existing calibrated replay used by run(); the
    # narrow mode merely avoids compiling all historical causal arms.
    mesh = read_row_mesh(root)
    cfg = card.recipe.model_config
    require(cfg.eos == "nemo_teos10", f"expected nemo_teos10, got {cfg.eos}")
    eos_fn = make_eos_fn(cfg.eos, None, rho0=cfg.rho_0)
    expected_hpg_u = hpg_sco_row(
        mesh, oracle["T"][1], oracle["S"][1], oracle["ssh"][1], eos_fn,
        cfg.g, cfg.rho_0, cfg.rho_0 * _constants.g,
    )
    model_hpg_u = arrays["stage2_hpg_u"][1, 1:, :]
    model_hpg_u = np.pad(
        model_hpg_u,
        ((0, 0), (0, mesh["umask"].shape[-1] - model_hpg_u.shape[-1])),
    )
    active_hpg_u = mesh["umask"].astype(bool)
    active_hpg_u[-1] = False
    hpg_row = score(
        f"{case}.kt1.stage2.pair_boundary.hpg_u",
        expected_hpg_u, model_hpg_u, active_hpg_u,
    )
    hpg_row["frame"] = "stage2_Krhs_immediately_after_dyn_hpg"
    hpg_row["source"] = (
        "OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:323-327; "
        "dynhpg.f90:341-414")
    rows.append(hpg_row)

    return ({
        "format": "nemo-testcase-overflow-round49-pair-boundary-v1",
        "case": case,
        "status": "AT-BAR" if all(row["status"] == "AT-BAR" for row in rows)
        else "DEBT",
        "worktree": worktree_stamp(),
        "legoesm_git_sha": revision,
        "precision_policy": "fp64/libm",
        "jax_backend": jax.default_backend(),
        "source_order": ["stage1_T", "stage1_S", "stage1_ssh", "stage2_hpg_u"],
        "rows": rows,
        "controls": {"plant": plant},
        "record_artifacts": {
            stage_path.name: sha256(stage_path),
            mesh_path.name: sha256(mesh_path),
        },
    }, arrays)


def persist_round49_pair_arrays(
    output: Path, report: dict, arrays: dict[str, np.ndarray],
) -> None:
    sidecar = output.with_suffix(".pair_arrays.npz")
    np.savez_compressed(sidecar, **arrays)
    report["pair_arrays"] = {
        "path": str(sidecar),
        "sha256": sha256(sidecar),
        "fields": list(arrays),
    }


def compare_round49_pair_arrays(reference_report: Path, candidate: dict) -> dict:
    reference = json.loads(reference_report.read_text())
    require(
        reference.get("format") == candidate.get("format"),
        "round-49 pair reports have different formats",
    )

    def load(report: dict) -> dict[str, np.ndarray]:
        artifact = report.get("pair_arrays")
        require(isinstance(artifact, dict), "pair-array artifact is absent")
        path = Path(artifact.get("path", ""))
        require(path.is_file(), f"missing pair-array artifact {path}")
        require(sha256(path) == artifact.get("sha256"), f"hash drift in {path}")
        with np.load(path) as stored:
            require(stored.files == artifact.get("fields"), "pair-array field drift")
            return {name: np.asarray(stored[name]) for name in stored.files}

    before, after = load(reference), load(candidate)
    require(before.keys() == after.keys(), "pair-array key drift")
    rows = []
    for name in candidate["source_order"]:
        left, right = before[name], after[name]
        require(left.shape == right.shape, f"{name}: shape drift")
        unequal = left.view(np.uint64) != right.view(np.uint64)
        rows.append({
            "name": name,
            "exact": bool(np.array_equal(left, right)),
            "n": int(left.size),
            "n_unequal": int(np.count_nonzero(unequal)),
            "max_abs_move": float(np.max(np.abs(right - left))),
        })
    first = next((row["name"] for row in rows if not row["exact"]), None)
    return {
        "format": "nemo-testcase-overflow-round49-pair-comparison-v1",
        "reference": str(reference_report),
        "reference_commit": reference["legoesm_git_sha"],
        "candidate_commit": candidate["legoesm_git_sha"],
        "first_moved_boundary": first,
        "rows": rows,
    }


def classify_arm(
    faithful_row: dict, control_row: dict, arm_movement: dict, *, improving: bool
) -> dict:
    residual = faithful_row["absolute_max"]
    moved = arm_movement["absolute_max"]
    ratio = moved / residual if residual else float("inf")
    clears = control_row["normalized_max_abs"] <= BAR
    if clears:
        classification = "CONFIRMED"
    elif improving and ratio >= 0.1:
        classification = "PLAUSIBLE_CONTRIBUTOR_NOT_OWNER"
    else:
        classification = "REFUTED_AS_PRIMARY_OWNER"
    return {
        "classification": classification,
        "faithful_absolute_error": residual,
        "control_absolute_error": control_row["absolute_max"],
        "arm_movement_absolute": moved,
        "movement_over_faithful_residual": ratio,
        "improves_residual": improving,
        "clears_bar": clears,
    }


def preregistered_prediction_check(
    case, scaling, rows, baroclinic_rows, stage_states, masks, arms,
    *, plant_selector=False, live_u_face_mask=None,
) -> dict:
    """Score the frozen predictions of
    ``nemo_testcases_l1_overflow_face_thickness_preregister.md`` (P1-P4) and
    ``nemo_testcases_l1_stage3_baroclinic_preregister.md`` (S1/S2).

    Nothing here is inferred at run time: the windows are the committed
    constants, and each is compared with the arm's own measured movement.
    ``plant_selector`` inflates the faithful stage-3 residual 1000x so the
    S1 predicate must flip to NOT-MET (the ``--plant-prediction`` control).
    """
    def _row(collection, name):
        return next(row for row in collection if row["name"] == name)

    def _stage_move(arm, stage):
        return movement(
            np.asarray(stage_states["faithful"][stage].u.data)[:, 1:, :],
            np.asarray(stage_states[arm][stage].u.data)[:, 1:, :],
            masks["u"])["absolute_max"]

    check = {}
    if "legacy_stage_min_face_thickness" in arms:
        faithful_T = _row(rows, f"{case}.kt2.faithful.T")["absolute_max"]
        control_T = _row(
            rows, f"{case}.kt2.legacy_stage_min_face_thickness.T")["absolute_max"]
        moved_T = abs(control_T - faithful_T)
        predicted_T = scaling["h1_predicted_total_T_movement_K"]
        moved_u = _stage_move("legacy_stage_min_face_thickness", 3)
        if case == "OVERFLOW-zps":
            check["P1_h1_moves_kt2_T_within_2x_of_scaling"] = {
                "predicate": "2.03e-07 <= |movement| <= 8.12e-07 K and the residual improves",
                "predicted_from_oracle_dumps_K": predicted_T,
                "faithful_kt2_T_residual_K": faithful_T,
                "legacy_min_rule_kt2_T_residual_K": control_T,
                "measured_movement_K": moved_T,
                "status": ("MET" if (2.03e-7 <= moved_T <= 8.12e-7
                                     and faithful_T < control_T) else "NOT-MET"),
            }
            check["P2_h1_is_refuted_for_the_u_residual"] = {
                "predicate": "stage-3 u movement < 2.6e-08 m/s",
                "measured_stage3_u_movement_m_s": moved_u,
                "status": "MET" if moved_u < 2.6e-8 else "NOT-MET",
            }
        else:
            check["P4_h1_is_inert_on_LOCK"] = {
                "predicate": "every movement < 1e-15 (oracle ssh is identically zero)",
                "measured_stage3_u_movement_m_s": moved_u,
                "measured_kt2_T_movement_K": moved_T,
                "status": ("MET" if max(moved_u, moved_T) < 1.0e-15 else "NOT-MET"),
            }
    if "omit_stage_qco_factor" in arms:
        predicted = {row["stage"]: row["predicted_baroclinic_u_movement_m_s"]
                     for row in scaling["h2_qco_stage_factor"]}
        for stage, floor in ((1, 1.3e-13), (2, 1.4e-11)):
            faithful = _row(
                baroclinic_rows,
                f"{case}.kt1.stage{stage}.faithful.baroclinic_u")["absolute_max"]
            control = _row(
                baroclinic_rows,
                f"{case}.kt1.stage{stage}.omit_stage_qco_factor.baroclinic_u",
            )["absolute_max"]
            if case == "OVERFLOW-zps":
                check[f"P3_h2_owns_stage{stage}"] = {
                    "predicate": (
                        f"faithful stage-{stage} baroclinic u < {floor:.1e} m/s "
                        "and the omission reproduces the pre-fix residual"),
                    "predicted_movement_m_s": predicted[stage],
                    "faithful_baroclinic_u_m_s": faithful,
                    "omitted_baroclinic_u_m_s": control,
                    "status": ("MET" if (faithful < floor and control > faithful)
                               else "NOT-MET"),
                }
            else:
                check[f"P4_h2_is_inert_on_LOCK_stage{stage}"] = {
                    "predicate": "movement < 1e-15 (oracle ssh is identically zero)",
                    "predicted_movement_m_s": predicted[stage],
                    "measured_movement_m_s": _stage_move(
                        "omit_stage_qco_factor", stage),
                    "status": ("MET" if _stage_move("omit_stage_qco_factor", stage)
                               < 1.0e-15 else "NOT-MET"),
                }
        if case == "OVERFLOW-zps":
            faithful_u = _row(
                rows, f"{case}.kt2.faithful.instantaneous_u")["absolute_max"]
            check["P3_h2_is_refuted_as_the_stage3_owner"] = {
                "predicate": (
                    "predicted stage-3 movement is < 0.01x the faithful kt=2 u "
                    "residual, so the kt=2 u row does not clear"),
                "predicted_stage3_movement_m_s": predicted[3],
                "faithful_kt2_u_residual_m_s": faithful_u,
                "predicted_over_residual": (
                    predicted[3] / faithful_u if faithful_u else float("inf")),
                "status": ("MET" if predicted[3] < 0.01 * faithful_u else "NOT-MET"),
                "note": (
                    "frozen against the pre-selector-fix kt=2 u residual "
                    "(2.5988e-07); after the UP3 selector round (S1) that "
                    "residual is gone and this predicate reads NOT-MET by "
                    "construction -- STALE, kept for the record"),
            }
    if "legacy_up3_transport_sign_selector" in arms:
        # nemo_testcases_l1_stage3_baroclinic_preregister.md: the UP3 upwind
        # selector (dynadv_up3.F90:166-170) owns the stage-3 baroclinic u
        # debt; the legacy transport-sign arm must reproduce the pre-fix
        # residual exactly (it IS the old code).
        def _bc(stage, arm):
            return _row(baroclinic_rows,
                        f"{case}.kt1.stage{stage}.{arm}.baroclinic_u")["absolute_max"]
        legacy = "legacy_up3_transport_sign_selector"
        if case == "OVERFLOW-zps":
            faithful3 = _bc(3, "faithful") * (1000.0 if plant_selector else 1.0)
            legacy3 = _bc(3, legacy)
            check["S1_up3_selector_owns_the_stage3_baroclinic_u"] = {
                "predicate": (
                    "faithful stage-3 baroclinic u < 1.0e-09 m/s (frozen: "
                    "linearised replay 4.5e-10) and the legacy transport-sign "
                    "arm reproduces the pre-fix 2.598798e-07 within 1%"),
                "faithful_stage3_baroclinic_u_m_s": faithful3,
                "legacy_selector_stage3_baroclinic_u_m_s": legacy3,
                "status": ("MET" if (faithful3 < 1.0e-9
                                     and abs(legacy3 - 2.598798e-7) < 2.6e-9)
                           else "NOT-MET"),
            }
        else:
            faithful2, faithful3 = _bc(2, "faithful"), _bc(3, "faithful")
            legacy2, legacy3 = _bc(2, legacy), _bc(3, legacy)
            check["S2_up3_selector_owns_the_LOCK_stage_debt"] = {
                "predicate": (
                    "faithful LOCK stage-2 and stage-3 baroclinic u <= 1e-15 "
                    "and the legacy arm reproduces the pre-fix 9.7656e-11 / "
                    "2.1388e-10 within 1%"),
                "faithful_stage2_baroclinic_u_m_s": faithful2,
                "faithful_stage3_baroclinic_u_m_s": faithful3,
                "legacy_selector_stage2_baroclinic_u_m_s": legacy2,
                "legacy_selector_stage3_baroclinic_u_m_s": legacy3,
                "status": ("MET" if (max(faithful2, faithful3) <= 1.0e-15
                                     and abs(legacy2 - 9.765645e-11) < 9.8e-13
                                     and abs(legacy3 - 2.138804e-10) < 2.2e-12)
                           else "NOT-MET"),
            }
    if "legacy_hadv_min_face_thickness" in arms and case == "OVERFLOW-zps":
        # nemo_testcases_l1_stage3_remainder_preregister.md P1/P2/P4: the
        # stage momentum-advection thickness owns the stage-3 remainder
        # (frozen: linearised replay 7.1e-12) and the stage-2 residual; the
        # legacy min-rule arm IS the pre-fix code and must reproduce
        # 4.551736e-10 / 9.433404e-11 within 1%.
        def _bc3(stage, arm):
            return _row(baroclinic_rows,
                        f"{case}.kt1.stage{stage}.{arm}.baroclinic_u")["absolute_max"]
        legacy_h = "legacy_hadv_min_face_thickness"
        f2, f3 = _bc3(2, "faithful"), _bc3(3, "faithful")
        l2, l3 = _bc3(2, legacy_h), _bc3(3, legacy_h)
        check["S3_hadv_face_thickness_owns_the_stage3_remainder"] = {
            "predicate": (
                "faithful stage-3 baroclinic u <= 1.0e-11 and stage-2 <= 3.0e-12 "
                "m/s; the legacy min-rule arm reproduces the pre-fix "
                "4.551736e-10 / 9.433404e-11 within 1%"),
            "faithful_stage2_baroclinic_u_m_s": f2,
            "faithful_stage3_baroclinic_u_m_s": f3,
            "legacy_hadv_stage2_baroclinic_u_m_s": l2,
            "legacy_hadv_stage3_baroclinic_u_m_s": l3,
            "status": ("MET" if (f3 <= 1.0e-11 and f2 <= 3.0e-12
                                 and abs(l3 - 4.551736e-10) < 4.6e-12
                                 and abs(l2 - 9.433404e-11) < 9.5e-13)
                       else "NOT-MET"),
        }
    if "legacy_2d_stage_face_mask" in arms and case == "OVERFLOW-zps":
        # nemo_testcases_l1_phantom_velocity_preregister.md P3/P10: NEMO
        # masks every stage velocity with the 3-D umask
        # (stprk3_stg.F90:367,375,382) and the barotropic correction with the
        # same array (:444), so uu is EXACTLY zero below the seabed and
        # dyn_adv_up3's k-slab stencil reads that zero.  The 2-D arm IS the
        # pre-fix code and must reproduce 7.064252e-12 / 1.566344e-12
        # within 1%.
        def _bc4(stage, arm):
            return _row(baroclinic_rows,
                        f"{case}.kt1.stage{stage}.{arm}.baroclinic_u")["absolute_max"]
        legacy_m = "legacy_2d_stage_face_mask"
        m2, m3 = _bc4(2, "faithful"), _bc4(3, "faithful")
        n2, n3 = _bc4(2, legacy_m), _bc4(3, legacy_m)
        # The scored MAXIMUM sits at a different face from the one the mask
        # rank moves, so it is NOT the discriminator (measured: both arms
        # report the same stage-3 max).  What NEMO's rule fixes is the value
        # BELOW the seabed, which the gate's active mask never scores -- so
        # score it directly, and require the wet stage-3 field to move.
        require(live_u_face_mask is not None,
                "S4 needs the 3-D live u-face mask")
        _dry = np.asarray(live_u_face_mask)[:, 1:, :] == 0.0

        def _below_seabed_max(arm, stage):
            return float(np.max(np.abs(
                np.asarray(stage_states[arm][stage].u.data)[:, 1:, :][_dry])))

        check["S4_stage_velocity_carries_NEMOs_3d_umask"] = {
            "predicate": (
                "every stage velocity is EXACTLY zero below the live seabed "
                "(stprk3_stg.F90:367,375,382,444,273 carry umask(ji,jj,jk)); the 2-D "
                "arm leaves a finite velocity there; and the wet stage-3 "
                "field moves between the two arms"),
            "faithful_below_seabed_max_m_s": [
                _below_seabed_max("faithful", k) for k in (1, 2, 3)],
            "legacy_2d_mask_below_seabed_max_m_s": [
                _below_seabed_max(legacy_m, k) for k in (1, 2, 3)],
            "wet_stage3_movement_m_s": _stage_move(legacy_m, 3),
            "faithful_stage2_baroclinic_u_m_s": m2,
            "faithful_stage3_baroclinic_u_m_s": m3,
            "legacy_2d_mask_stage2_baroclinic_u_m_s": n2,
            "legacy_2d_mask_stage3_baroclinic_u_m_s": n3,
            "status": ("MET" if (
                all(_below_seabed_max("faithful", k) == 0.0 for k in (1, 2, 3))
                and _below_seabed_max(legacy_m, 1) > 1.0e-3
                and _stage_move(legacy_m, 3) > 0.0)
                else "NOT-MET"),
        }
    return check


def run(case: str, root: Path, *, plant_stage=False, plant_operand=False,
        plant_prediction=False, allow_dirty=False, faithful_only=False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    # Stamp FIRST so a dirty tree refuses before any compute (fail closed).
    allow_dirty_stamps(allow_dirty)
    legoesm_git_sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = build_nemo_testcase_card(case)
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    artifacts = {}
    oracle_stages = {}
    for stage in (1, 2, 3):
        path = root / f"oracle_stage_kt00000001_s{stage}.bin"
        require(path.is_file(), f"missing {path}")
        oracle_stages[stage] = read_stage(path, case, stage)
        artifacts[path.name] = sha256(path)
    entry_path = root / "oracle_step_entry_kt00000002.bin"
    require(entry_path.is_file(), f"missing {entry_path}")
    oracle_entry = read_entry(entry_path, case)
    artifacts[entry_path.name] = sha256(entry_path)
    entry1_path_scaling = root / "oracle_step_entry_kt00000001.bin"
    require(entry1_path_scaling.is_file(), f"missing {entry1_path_scaling}")
    entry1_for_scaling = read_entry(entry1_path_scaling, case, expected=(1, 1))
    artifacts[entry1_path_scaling.name] = sha256(entry1_path_scaling)

    if case == "OVERFLOW-zps":
        arms = {
            "faithful": _NEMOWSRK3TestHooks(),
            "freeze_stage_hpg_operands": _NEMOWSRK3TestHooks(
                freeze_stage_hpg_operands=True
            ),
            # Split of the freeze arm (review round): one operand class each.
            "freeze_stage_hpg_tracers": _NEMOWSRK3TestHooks(
                freeze_stage_hpg_tracers=True
            ),
            "freeze_stage_hpg_eta": _NEMOWSRK3TestHooks(
                freeze_stage_hpg_eta=True
            ),
            "omit_stage_vertical_up3": _NEMOWSRK3TestHooks(
                omit_stage_vertical_up3=True
            ),
            "legacy_velocity_primary_average": _NEMOWSRK3TestHooks(
                primary_transport_average=False
            ),
            "omit_stage_primary_velocity_correction": _NEMOWSRK3TestHooks(
                stage_barotropic_correction=False
            ),
            # Face-thickness / qco-factor round (preregistered in
            # nemo_testcases_l1_overflow_face_thickness_preregister.md).
            "legacy_stage_min_face_thickness": _NEMOWSRK3TestHooks(
                legacy_stage_min_face_thickness=True
            ),
            "omit_stage_qco_factor": _NEMOWSRK3TestHooks(
                omit_stage_qco_factor=True
            ),
            # Stage-3 baroclinic round (preregistered in
            # nemo_testcases_l1_stage3_baroclinic_preregister.md): ablate the
            # UP3 upwind-selector fix (dynadv_up3.F90:166-170).
            "legacy_up3_transport_sign_selector": _NEMOWSRK3TestHooks(
                legacy_up3_transport_sign_selector=True
            ),
            # Stage-3 remainder round (preregistered in
            # nemo_testcases_l1_stage3_remainder_preregister.md): ablate the
            # NEMO e3u(Kmm) in the stage flux-form momentum advection.
            "legacy_hadv_min_face_thickness": _NEMOWSRK3TestHooks(
                legacy_hadv_min_face_thickness=True
            ),
            # Phantom-velocity round (preregistered in
            # nemo_testcases_l1_phantom_velocity_preregister.md): ablate
            # NEMO's 3-D umask on the stage velocity update
            # (stprk3_stg.F90:367,375,382,444,273).
            "legacy_2d_stage_face_mask": _NEMOWSRK3TestHooks(
                legacy_2d_stage_face_mask=True
            ),
        }
    else:
        arms = {
            "faithful": _NEMOWSRK3TestHooks(),
            "omit_stage_primary_velocity_correction": _NEMOWSRK3TestHooks(
                stage_barotropic_correction=False
            ),
            "omit_momentum_transport_reconcile": _NEMOWSRK3TestHooks(
                momentum_transport_reconcile=False
            ),
            "legacy_stage_min_face_thickness": _NEMOWSRK3TestHooks(
                legacy_stage_min_face_thickness=True
            ),
            "omit_stage_qco_factor": _NEMOWSRK3TestHooks(
                omit_stage_qco_factor=True
            ),
            "legacy_up3_transport_sign_selector": _NEMOWSRK3TestHooks(
                legacy_up3_transport_sign_selector=True
            ),
            "legacy_hadv_min_face_thickness": _NEMOWSRK3TestHooks(
                legacy_hadv_min_face_thickness=True
            ),
            # Phantom-velocity round (preregistered in
            # nemo_testcases_l1_phantom_velocity_preregister.md): ablate
            # NEMO's 3-D umask on the stage velocity update
            # (stprk3_stg.F90:367,375,382,444,273).
            "legacy_2d_stage_face_mask": _NEMOWSRK3TestHooks(
                legacy_2d_stage_face_mask=True
            ),
        }
    if faithful_only:
        require(not (plant_operand or plant_prediction),
                "--faithful-only cannot run arm-dependent planted controls")
        arms = {"faithful": arms["faithful"]}

    states = {}
    stage_states = {}
    rows = []
    baroclinic_rows = []
    from legoesm.ocean.vertical import compute_layer_thickness
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    h0 = compute_layer_thickness(
        card.recipe.initial_state.eta.data,
        card.recipe.initial_state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    hu0 = np.asarray(min_cell_to_uface(h0))[:, 1:, :]
    for arm, hooks in arms.items():
        final_state = LatLonCGridOceanModel(
            card.recipe.grid,
            card.recipe.z_coord,
            card.recipe.model_config,
            _nemo_ws_test_hooks=hooks,
        ).step(card.recipe.initial_state, dt=card.dt_s)
        # Materialize before compiling either exposed-stage executable.  A
        # GYRE/OVERFLOW production step is small, but retaining three XLA
        # executables at once is not; the comparison harness must not turn a
        # 30x20-class card into a host-memory exhaustion.
        states[arm] = final_state._replace(
            u=final_state.u.replace(data=np.asarray(final_state.u.data)),
            T=final_state.T.replace(data=np.asarray(final_state.T.data)),
            eta=final_state.eta.replace(data=np.asarray(final_state.eta.data)),
        )
        jax.clear_caches()
        stage_states[arm] = {}
        for stage in (1, 2, 3):
            if stage == 3:
                stage_state = states[arm]
            else:
                stage_state = LatLonCGridOceanModel(
                    card.recipe.grid,
                    card.recipe.z_coord,
                    card.recipe.model_config,
                    _nemo_ws_test_hooks=hooks._replace(expose_momentum_stage=stage),
                ).step(card.recipe.initial_state, dt=card.dt_s)
            stage_states[arm][stage] = stage_state
            row = score(
                f"{case}.kt1.stage{stage}.{arm}.instantaneous_u",
                oracle_stages[stage]["u"][..., :nlev],
                np.asarray(stage_state.u.data)[:, 1:, :],
                masks["u"],
                plant=plant_stage and arm == "faithful" and stage == 1,
            )
            row["verdict"] = arm == "faithful"
            rows.append(row)
            baroclinic_rows.append(score(
                f"{case}.kt1.stage{stage}.{arm}.baroclinic_u",
                remove_depth_mean(
                    oracle_stages[stage]["u"][..., :nlev], hu0,
                    masks["u"]),
                remove_depth_mean(
                    np.asarray(stage_state.u.data)[:, 1:, :], hu0,
                    masks["u"]),
                masks["u"],
            ))
            if stage != 3:
                stage_states[arm][stage] = stage_state._replace(
                    u=stage_state.u.replace(data=np.asarray(stage_state.u.data)))
                jax.clear_caches()
        # Each arm compiles three fresh executables; with 9 arms the process
        # exhausts vm.max_map_count (LLVM "Unable to allocate section memory")
        # long before host RAM.  Materialize the arm's fields as numpy, then
        # drop the compilation cache.  Pure resource hygiene: no arithmetic.
        stage_states[arm][3] = states[arm]
        jax.clear_caches()

    faithful_u = score(
        f"{case}.kt2.faithful.instantaneous_u",
        oracle_entry["u"][..., :nlev],
        np.asarray(states["faithful"].u.data)[:, 1:, :],
        masks["u"],
    )
    faithful_u["frame"] = "instantaneous_prognostic_Nbb"
    rows.append(faithful_u)
    faithful_u_bc = score(
        f"{case}.kt2.faithful.baroclinic_u",
        remove_depth_mean(oracle_entry["u"][..., :nlev], hu0, masks["u"]),
        remove_depth_mean(
            np.asarray(states["faithful"].u.data)[:, 1:, :], hu0,
            masks["u"]),
        masks["u"],
    )
    baroclinic_rows.append(faithful_u_bc)
    faithful_T = score(
        f"{case}.kt2.faithful.T",
        oracle_entry["T"][..., :nlev],
        np.asarray(states["faithful"].T.data),
        masks["T"],
        quantity="T",
    )
    faithful_T["frame"] = "instantaneous_tracer_Nbb"
    rows.append(faithful_T)

    arm_results = {}
    for arm in arms:
        if arm == "faithful":
            continue
        control_u = score(
            f"{case}.kt2.{arm}.instantaneous_u",
            oracle_entry["u"][..., :nlev],
            np.asarray(states[arm].u.data)[:, 1:, :],
            masks["u"],
        )
        control_u.update({"verdict": False, "frame": "instantaneous_prognostic_Nbb"})
        rows.append(control_u)
        control_u_bc = score(
            f"{case}.kt2.{arm}.baroclinic_u",
            remove_depth_mean(oracle_entry["u"][..., :nlev], hu0, masks["u"]),
            remove_depth_mean(
                np.asarray(states[arm].u.data)[:, 1:, :], hu0, masks["u"]),
            masks["u"],
        )
        baroclinic_rows.append(control_u_bc)
        control_T = score(
            f"{case}.kt2.{arm}.T",
            oracle_entry["T"][..., :nlev],
            np.asarray(states[arm].T.data),
            masks["T"],
            quantity="T",
        )
        control_T.update({"verdict": False, "frame": "instantaneous_tracer_Nbb"})
        rows.append(control_T)
        target = (
            "u" if arm in ("freeze_stage_hpg_operands", "freeze_stage_hpg_tracers",
                           "freeze_stage_hpg_eta", "omit_stage_vertical_up3",
                           "omit_stage_qco_factor",
                           "legacy_up3_transport_sign_selector",
                           "legacy_hadv_min_face_thickness",
                           "legacy_2d_stage_face_mask")
            else "T" if case == "OVERFLOW-zps" else "u"
        )
        faithful_row = faithful_T if target == "T" else faithful_u
        control_row = control_T if target == "T" else control_u
        faithful_field = (
            np.asarray(states["faithful"].T.data)
            if target == "T"
            else np.asarray(states["faithful"].u.data)[:, 1:, :]
        )
        control_field = (
            np.asarray(states[arm].T.data)
            if target == "T"
            else np.asarray(states[arm].u.data)[:, 1:, :]
        )
        arm_move = movement(faithful_field, control_field, masks[target])
        improving = control_row["absolute_max"] < faithful_row["absolute_max"]
        arm_results[arm] = classify_arm(faithful_row, control_row, arm_move, improving=improving)
        arm_results[arm]["target"] = target
        if arm in ("legacy_stage_min_face_thickness", "omit_stage_qco_factor",
                   "legacy_up3_transport_sign_selector",
                   "legacy_hadv_min_face_thickness",
                   "legacy_2d_stage_face_mask"):
            # These arms ABLATE a landed fix, so classify_arm's "does the
            # arm improve the residual" question is inverted: the meaningful
            # statement is whether REMOVING the NEMO rule makes the target
            # worse.  Recorded explicitly rather than by reading the inverted
            # label.
            worse = control_row["absolute_max"] > faithful_row["absolute_max"]
            factor = (control_row["absolute_max"] / faithful_row["absolute_max"]
                      if faithful_row["absolute_max"] else float("inf"))
            arm_results[arm]["arm_kind"] = "ablation_of_landed_fix"
            arm_results[arm]["removing_the_nemo_rule_worsens_the_target"] = worse
            arm_results[arm]["control_over_faithful"] = factor
            arm_results[arm]["ablation_classification"] = (
                "CONFIRMED_REQUIRED" if worse and factor >= 10.0
                else "REQUIRED_SMALL_EFFECT" if worse
                else "NOT_REQUIRED_BY_THIS_TARGET")
        if arm == "freeze_stage_hpg_operands":
            # NOT one variable: the stage eta also sets the h_k bundle that
            # _bc_vertical_and_depthmean_velocity turns into the flux-form
            # advection face thickness/w (ocean_pe_latlon_cgrid:4384-4395).
            one_variable = (
                "stage-2/3 EOS+HPG T/S/ssh time level AND the stage-eta "
                "advection geometry (two variables; see the split arms)")
        elif arm == "freeze_stage_hpg_tracers":
            one_variable = "stage-2/3 EOS T/S time level (live stage eta)"
        elif arm == "freeze_stage_hpg_eta":
            one_variable = (
                "stage-2/3 eta time level: HPG r3t stretching AND the "
                "advection geometry it sets (live stage T/S)")
        elif arm == "omit_stage_vertical_up3":
            one_variable = "vertical UP3 in each momentum-stage RHS"
        elif arm == "legacy_stage_min_face_thickness":
            one_variable = (
                "stage transport face thickness: NEMO e3u_0*(1+r3u(Kmm)) "
                "(domqco.F90:219-222) vs min of the two stretched T cells")
        elif arm == "omit_stage_qco_factor":
            one_variable = (
                "qco weighting of the stage velocity update "
                "(stprk3_stg.F90:373-378)")
        elif arm == "legacy_up3_transport_sign_selector":
            one_variable = (
                "UP3 upwind selector of the stage horizontal momentum "
                "advection: NEMO's advected-velocity pair sign "
                "(dynadv_up3.F90:166-170) vs the stage-transport pair sign")
        elif arm == "legacy_hadv_min_face_thickness":
            one_variable = (
                "face thickness of the stage flux-form momentum advection: "
                "NEMO e3u(Kmm) = e3u_0*(1+r3u(Kmm)) (dynadv_up3.F90:160,"
                "205-207; domzgr_substitute.h90:127) vs tendencies()' min of "
                "the two stretched T cells")
        elif arm == "legacy_2d_stage_face_mask":
            one_variable = (
                "rank of the mask applied to the WS-RK3 stage velocity "
                "update and to the stage velocities the tracer transports "
                "consume: NEMO's 3-D umask(ji,jj,jk) "
                "(stprk3_stg.F90:367,375,382,444,273) vs the 2-D state.u_mask "
                "broadcast over every level")
        elif arm == "legacy_velocity_primary_average":
            one_variable = "flux_form_primary_transport_average"
        elif "primary" in arm:
            one_variable = "stage_barotropic_correction"
        else:
            one_variable = "momentum_transport_reconcile"
        arm_results[arm]["one_variable"] = one_variable

    if faithful_only:
        ownership = {
            "classification": "COMPATIBILITY_GUARD_ONLY",
            "reason": "private ablation arms intentionally not compiled",
            "arms": {},
        }
    elif case == "LOCK_EXCHANGE-zco" and not any(
        result["clears_bar"] for result in arm_results.values()
    ):
        ownership = {
            "classification": "UNMEASURED_AFTER_TWO_ARMS",
            "reason": (
                "the instantaneous-u tail resisted both registered "
                "one-variable arms; no further owner is assigned this round"
            ),
            "arms": arm_results,
        }
    else:
        faithful_stage1 = next(
            row for row in rows if row["name"] == f"{case}.kt1.stage1.faithful.instantaneous_u"
        )
        control_stage1 = next(
            row
            for row in rows
            if row["name"]
            == f"{case}.kt1.stage1.omit_stage_primary_velocity_correction.instantaneous_u"
        )
        ownership = {
            "classification": arm_results["omit_stage_primary_velocity_correction"][
                "classification"
            ],
            "scheme_component_requirement": "CONFIRMED_REQUIRED",
            "direct_stage1_faithful_absolute_error_m_s": faithful_stage1["absolute_max"],
            "direct_stage1_omission_absolute_error_m_s": control_stage1["absolute_max"],
            "cancellation_warning": (
                "omission improves the final T residual but makes the direct "
                "instantaneous stage-1 u comparison much worse; the T movement "
                "is scale-compatible evidence for a residual contribution, not "
                "evidence that NEMO's required correction should be removed"
            ),
            "arms": arm_results,
        }

    stage_operand_ownership = None
    operand_rows = []
    replay = None
    if case == "OVERFLOW-zps" and not faithful_only:
        def _bc_row(stage, arm):
            suffix = f".stage{stage}.{arm}.baroclinic_u"
            return next(row for row in baroclinic_rows if suffix in row["name"])

        def _within(value, target):
            return 0.5 * target <= value <= 2.0 * target

        # Arm A (live stage EOS/HPG operands) was preregistered WITHOUT Arm B:
        # its frozen prediction is read off the omit-vertical arm (A only).
        # Arm B (vertical UP3 in every stage RHS) is read off the faithful
        # arm (A + B).  The frozen-operand arm (B only) isolates the HPG
        # operand time level as the one remaining variable.
        a_only_s2 = _bc_row(2, "omit_stage_vertical_up3")
        a_only_k2 = next(row for row in baroclinic_rows if row["name"].endswith(
            "kt2.omit_stage_vertical_up3.baroclinic_u"))
        b_only_s2 = _bc_row(2, "freeze_stage_hpg_operands")
        b_only_k2 = next(row for row in baroclinic_rows if row["name"].endswith(
            "kt2.freeze_stage_hpg_operands.baroclinic_u"))
        live_s2 = _bc_row(2, "faithful")
        live_s3 = _bc_row(3, "faithful")
        live_k2 = faithful_u_bc
        # The preregistered kt=2 baseline (3.31e-6) was measured on code that
        # still applied the once-per-step post-hoc vertical UP3 after stage 3;
        # the A-only arm carries NO vertical term, so its kt=2 leg differs
        # from that baseline in two variables.  Arm A's label rests on the
        # stage-2 leg alone (the post-hoc term landed after stage 3, so that
        # leg IS one variable); the kt=2 leg is reported, not used.
        confirmed_a = _within(a_only_s2["absolute_max"], 1.65e-7)
        kt2_leg_a = a_only_k2["absolute_max"] <= 7.0e-7
        confirmed_b = (
            _within(live_s2["absolute_max"], 1.0e-10)
            and _within(live_s3["absolute_max"], 2.6e-7)
        )
        stage_operand_ownership = {
            "arm_a_live_stage_hpg_operands": {
                "label": "CONFIRMED_ON_STAGE2_LEG" if confirmed_a else "REFUTED",
                "kt2_leg": {
                    "status": "TWO_VARIABLE_NOT_USED_FOR_LABEL",
                    "predicate_met": kt2_leg_a,
                    "reason": (
                        "preregistered baseline 3.31e-6 carried the post-hoc "
                        "once-per-step vertical UP3 after stage 3; the A-only "
                        "arm carries no vertical term at all"),
                },
                "scaling_check_before_owner_label": True,
                "frozen_prediction": (
                    "A without B: kt2 <=7e-7; stage2 approximately 1.65e-7 "
                    "within factor 2 (nemo_testcases_l1_overflow_stage_"
                    "composition_preregister.md)"),
                "a_only_stage2_baroclinic_u_error_m_s": a_only_s2["absolute_max"],
                "a_only_kt2_baroclinic_u_error_m_s": a_only_k2["absolute_max"],
                "b_only_stage2_baroclinic_u_error_m_s": b_only_s2["absolute_max"],
                "b_only_kt2_baroclinic_u_error_m_s": b_only_k2["absolute_max"],
                "one_variable_vs_faithful": "stage-2/3 EOS+HPG T/S/ssh time level",
            },
            "arm_b_vertical_up3_every_stage": {
                "label": "CONFIRMED" if confirmed_b else "REFUTED",
                "scaling_check_before_owner_label": True,
                "frozen_prediction": (
                    "A + B: stage2 approximately 1e-10, stage3 approximately "
                    "2.6e-7, each within factor 2"),
                "live_stage2_baroclinic_u_error_m_s": live_s2["absolute_max"],
                "live_stage3_baroclinic_u_error_m_s": live_s3["absolute_max"],
                "live_kt2_baroclinic_u_error_m_s": live_k2["absolute_max"],
                "one_variable_vs_faithful": "vertical UP3 in each momentum-stage RHS",
            },
        }

        # ---- stage tracer/ssh operands as the faithful arm hands them to the
        # stage-2/3 eos+dyn_hpg calls, against NEMO's stage Kaa dumps ----
        wet2d = masks["T"].any(axis=-1)
        eta0 = np.asarray(card.recipe.initial_state.eta.data)
        operand_states = {}
        for stage in (1, 2):
            operand_states[stage] = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_tracer_stage=stage),
            ).step(card.recipe.initial_state, dt=card.dt_s)
        # HYB stage ssh (stprk3_stg.F90:146 N+1/3, :209 N+1/2) exactly as the
        # solver hands it to the stage eos+dyn_hpg (the hook exposes eta too).
        stage_ssh = {stage: np.asarray(operand_states[stage].eta.data) for stage in (1, 2)}
        s_uniform = np.unique(oracle_stages[1]["S"][..., :nlev][masks["T"]]).size == 1
        for stage in (1, 2):
            for name, oracle_field, candidate in (
                ("T", oracle_stages[stage]["T"][..., :nlev],
                 np.asarray(operand_states[stage].T.data)),
                ("S", oracle_stages[stage]["S"][..., :nlev],
                 np.asarray(operand_states[stage].S.data)),
            ):
                row = score(
                    f"{case}.kt1.stage{stage}.faithful.tracer_operand_{name}",
                    oracle_field, candidate, masks["T"], quantity="T",
                    plant=plant_operand and stage == 1 and name == "T",
                )
                row["frame"] = "instantaneous_tracer_Kaa"
                row["staggering_and_reduction"] = (
                    "oracle ts(:,:,:,jn,Kaa) after the stage tracer update and "
                    "legoESM's exposed WS-RK3 stage tracer are both instantaneous "
                    "3-D T-point fields; elementwise L-infinity on the wet T mask")
                if name == "S" and s_uniform and row["status"] == "AT-BAR":
                    row["status"] = "UNINFORMATIVE"
                    row["reason"] = (
                        "oracle stage salinity is spatially uniform (n_unique=1); "
                        "this row cannot detect a stage-transport error")
                operand_rows.append(row)
            row = score(
                f"{case}.kt1.stage{stage}.faithful.ssh_operand",
                oracle_stages[stage]["ssh"], stage_ssh[stage], wet2d, quantity="T",
            )
            row["frame"] = "instantaneous_ssh_Kaa"
            row["staggering_and_reduction"] = (
                "oracle ssh(:,:,Kaa) and legoESM's HYB stage eta (as handed to "
                "the stage eos+dyn_hpg, exposed by the solver hook) are both "
                "instantaneous 2-D T-point sea levels; elementwise L-infinity "
                "on the wet column mask; note max(|ssh|,1)=1 normalization")
            operand_rows.append(row)

        # ---- hpg_sco replay: instrument check + operand-difference prediction
        entry1_path = root / "oracle_step_entry_kt00000001.bin"
        require(entry1_path.is_file(), f"missing {entry1_path}")
        entry1 = read_entry(entry1_path, case, expected=(1, 1))
        artifacts[entry1_path.name] = sha256(entry1_path)
        mesh_path = root / "mesh_mask.nc"
        require(mesh_path.is_file(), f"missing {mesh_path}")
        artifacts[mesh_path.name] = sha256(mesh_path)
        mesh = read_row_mesh(root)
        require(int(wet2d.sum(axis=1).astype(bool).sum()) == 1, "replay expects one wet row")
        cfg = card.recipe.model_config
        from legoesm.ocean.eos import make_eos_fn
        require(cfg.eos == "nemo_teos10", f"replay expects nemo_teos10, got {cfg.eos}")
        eos_fn = make_eos_fn(cfg.eos, None, rho0=cfg.rho_0)
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
        act = masks["u"][1]
        pad = lambda a: np.concatenate(  # noqa: E731
            [a, np.zeros(a.shape[:-1] + (mesh["tmask"].shape[-1] - a.shape[-1],))], axis=-1)
        init = card.recipe.initial_state
        T0 = np.asarray(init.T.data)
        S0 = np.asarray(init.S.data)

        def lego_hpg(T_full, S_full, ssh_full):
            state = init._replace(
                T=init.T.replace(data=np.where(masks["T"], T_full[..., :nlev], T0)),
                S=init.S.replace(data=np.where(masks["T"], S_full[..., :nlev], S0)),
                eta=init.eta.replace(data=np.where(wet2d, ssh_full, eta0)),
            )
            du = model.tendencies(state, dt=card.dt_s, momentum_only=True).du_dt.data
            return pad(np.asarray(du)[1, 1:, :])

        from legoesm import constants as _constants

        def replay(T_full, S_full, ssh_full):
            return hpg_sco_row(
                mesh, T_full[1], S_full[1], ssh_full[1], eos_fn, cfg.g, cfg.rho_0,
                cfg.rho_0 * _constants.g)

        act_row = np.zeros_like(mesh["umask"], dtype=bool)
        act_row[:, :nlev] = act
        act_row[-1] = False
        s1 = oracle_stages[1]
        replay_rows = []
        for label, operands in (
            ("rest", (entry1["T"], entry1["S"], entry1["ssh"])),
            ("nemo_stage1_kaa", (s1["T"], s1["S"], s1["ssh"])),
        ):
            row = score(
                f"{case}.replay.instrument.legoesm_hpg_vs_hpg_sco.{label}",
                replay(*operands), lego_hpg(*operands), act_row,
            )
            row["frame"] = "instantaneous_u_rhs_from_given_T_S_ssh"
            row["staggering_and_reduction"] = (
                "legoESM momentum tendency at rest velocity (u=v=0: HPG only "
                "on this card) and the dynhpg.F90:340-390 replay are both "
                "U-face trends on the wet row; elementwise L-infinity")
            replay_rows.append(row)
        T_lego_s1 = pad(np.asarray(operand_states[1].T.data))
        S_lego_s1 = pad(np.asarray(operand_states[1].S.data))
        H_nemo_s1 = replay(s1["T"], s1["S"], s1["ssh"])
        H_frozen = replay(entry1["T"], entry1["S"], entry1["ssh"])
        H_lego_s1 = replay(T_lego_s1, S_lego_s1, stage_ssh[1])
        H_frozen_tracers = replay(entry1["T"], entry1["S"], stage_ssh[1])
        H_frozen_eta = replay(T_lego_s1, S_lego_s1, entry1["ssh"])
        e3u_row = mesh["e3u_0"]

        def stage2_prediction(H_candidate, H_reference=H_nemo_s1):
            diff = remove_depth_mean(
                (card.dt_s / 2.0) * (H_candidate - H_reference), e3u_row, act_row)
            return float(np.max(np.abs(diff[act_row])))

        def stage2_movement(arm):
            return movement(
                remove_depth_mean(
                    np.asarray(stage_states["faithful"][2].u.data)[:, 1:, :], hu0, masks["u"]),
                remove_depth_mean(
                    np.asarray(stage_states[arm][2].u.data)[:, 1:, :], hu0, masks["u"]),
                masks["u"])["absolute_max"]

        measured_a = stage2_movement("freeze_stage_hpg_operands")
        # Each split arm moves ONE operand class off legoESM's own live stage
        # operands, so its HPG-only prediction is E_hpg(frozen) - E_hpg(live
        # legoESM): whatever the measured movement carries beyond that is a
        # non-HPG consumer of the operand (the advection geometry for eta).
        split = {}
        for arm, H_arm in (("freeze_stage_hpg_tracers", H_frozen_tracers),
                           ("freeze_stage_hpg_eta", H_frozen_eta)):
            predicted = stage2_prediction(H_arm, H_lego_s1)
            measured = stage2_movement(arm)
            split[arm] = {
                "predicted_hpg_only_stage2_movement_m_s": predicted,
                "measured_stage2_movement_m_s": measured,
                "measured_minus_predicted_m_s": measured - predicted,
                "predicted_over_measured": (
                    predicted / measured if measured else float("inf")),
            }
        replay = {
            "instrument_rows": replay_rows,
            "predicted_stage2_baroclinic_u_error_from_frozen_kbb_operands_m_s": (
                stage2_prediction(H_frozen)),
            "measured_stage2_baroclinic_u_movement_frozen_vs_live_m_s": measured_a,
            "predicted_over_measured": (
                stage2_prediction(H_frozen) / measured_a if measured_a else float("inf")),
            "measured_minus_predicted_m_s": measured_a - stage2_prediction(H_frozen),
            "predicted_stage2_baroclinic_u_error_from_legoesm_stage1_operands_m_s": (
                stage2_prediction(H_lego_s1)),
            "split_arms": split,
            "source": (
                "dynhpg.F90:340-390; domzgr_substitute.h90:131,139,145; "
                "domqco.F90:160; eos at live gdept (eosbn2.F90:1166)"),
        }

    scaling = (None if faithful_only else face_thickness_and_qco_scaling(
        card, masks, nlev, oracle_stages, entry1_for_scaling, hu0))
    if plant_prediction:
        # Planted control for the prediction block: inflate the frozen H2
        # predictions by 1000x.  Every stage predicate must then read NOT-MET;
        # if any still reads MET the block is not looking at these numbers and
        # the gate exits 2.
        for row in scaling["h2_qco_stage_factor"]:
            row["predicted_baroclinic_u_movement_m_s"] *= 1000.0
    prediction_check = None
    if not faithful_only:
        from legoesm.ocean.dynamics.latlon_cgrid_operators import (
            compute_face_masks_3d as _face_masks_3d)
        prediction_check = preregistered_prediction_check(
            case, scaling, rows, baroclinic_rows, stage_states, masks, arms,
            plant_selector=plant_prediction,
            live_u_face_mask=np.asarray(_face_masks_3d(
                card.recipe.z_coord.is_active, card.recipe.grid)[0]).astype(float))
    if plant_prediction:
        # The gated predicate is the live round's owner claim (S1); the
        # older P3 reads NOT-MET by construction since the selector round
        # removed the residual it was scaled against, so it can no longer
        # discriminate a plant.
        gated = "S1_up3_selector_owns_the_stage3_baroclinic_u"
        require(gated in prediction_check,
                f"planted prediction control needs {gated} (OVERFLOW only)")
        require(prediction_check[gated]["status"] == "NOT-MET",
                "planted prediction control did not land: "
                f"{prediction_check[gated]}")

    failed = [row["name"] for row in rows if row["status"] == "DEBT" and row.get("verdict", True)]
    failed += [row["name"] for row in operand_rows if row["status"] == "DEBT"]
    # Planted controls: the planted row must carry the +1.0, else exit 2.
    if plant_stage:
        require_planted(rows, f"{case}.kt1.stage1.faithful.instantaneous_u")
    if plant_operand:
        require(case == "OVERFLOW-zps", "--plant-operand needs the OVERFLOW operand rows")
        require_planted(operand_rows, f"{case}.kt1.stage1.faithful.tracer_operand_T")
    if plant_prediction:
        require(case == "OVERFLOW-zps",
                "--plant-prediction needs the OVERFLOW prediction rows")
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l1-phase3-stage-sweep-v1",
        "case": case,
        "status": "AT-BAR" if not failed else "DEBT",
        "bar": BAR,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "candidate_dtypes": {
            "T": str(np.asarray(states["faithful"].T.data).dtype),
            "u": str(np.asarray(states["faithful"].u.data).dtype),
        },
        "oracle_root": str(root),
        "time_level_registry": {
            "stage1": "Kaa=3 after stprk3_stg.F90:433-446",
            "stage2": "Kaa=2 after stprk3_stg.F90:433-446",
            "stage3": "Kaa=3 after stprk3_stg.F90:433-446",
            "next_entry": "kt=2 Nbb=3 from stprk3.F90 step-entry instrument",
            "stage1_tracer_operand": (
                "ts(:,:,:,:,Kaa=3) after the stage-1 update stprk3_stg.F90:"
                "535-552; consumed as Kmm by stage-2 eos/dyn_hpg (:317-320) "
                "after the stprk3.F90:218 Nnn<->Naa swap"),
            "stage2_tracer_operand": (
                "ts(:,:,:,:,Kaa=2) after stage 2; consumed as Kmm by stage-3 "
                "eos/dyn_hpg after the stprk3.F90:224 swap"),
            "stage1_ssh_operand": "HYB ssh(Kaa)=2/3 ssh(Kbb)+1/3 ssha, stprk3_stg.F90:146",
            "stage2_ssh_operand": "HYB ssh(Kaa)=1/2(ssh(Kbb)+ssha), stprk3_stg.F90:209",
        },
        "rows": rows,
        "baroclinic_rows": baroclinic_rows,
        "stage_operand_rows": operand_rows,
        "hpg_sco_replay": replay,
        "face_thickness_and_qco_scaling": scaling,
        "preregistered_prediction_check": prediction_check,
        "failed_rows": failed,
        "ownership": ownership,
        "stage_operand_ownership": stage_operand_ownership,
        "controls": {"plant_stage": plant_stage, "plant_operand": plant_operand,
                     "plant_prediction": plant_prediction,
                     "faithful_only": faithful_only},
        "legoesm_git_sha": legoesm_git_sha,
        "artifacts": artifacts,
    }


@scoped_allow_dirty
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(ROOTS))
    parser.add_argument("--oracle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant-stage", action="store_true")
    parser.add_argument("--plant-operand", action="store_true")
    parser.add_argument("--plant-prediction", action="store_true")
    parser.add_argument(
        "--round49-pair-boundary", action="store_true",
        help=("compile only the stage-1 tracer / stage-2 HPG boundary used by "
              "the ORCA2 round-49 OVERFLOW cancelling-pair walk"),
    )
    parser.add_argument(
        "--round49-pair-reference", type=Path,
        help="base report produced by --round49-pair-boundary",
    )
    parser.add_argument(
        "--round49-pair-plant", action="store_true",
        help="plant +1 K in the exposed stage-1 tracer; the gate must be red",
    )
    parser.add_argument(
        "--faithful-only", action="store_true",
        help=("compile only the public faithful stage path for a compatibility "
              "comparison; private causal arms and their owner labels are omitted"))
    parser.add_argument("--allow-dirty", action="store_true",
                        help="stamp '<sha>-dirty' instead of refusing a dirty tree")
    from legoesm.ocean.fidelity.ulp_move_gate import (
        add_ulp_compare_arguments, capture_residual_fields,
        comparison_exit_code, persist_ulp_comparison, run_ulp_comparison,
        write_residual_artifact,
    )
    add_ulp_compare_arguments(parser)
    args = parser.parse_args(argv)
    if args.round49_pair_boundary:
        try:
            require(args.case == "OVERFLOW-zps",
                    "--round49-pair-boundary is OVERFLOW-only")
            require(args.output is not None,
                    "--round49-pair-boundary requires --output")
            require(not (args.plant_stage or args.plant_operand
                         or args.plant_prediction or args.faithful_only
                         or args.compare_to),
                    "round-49 pair mode cannot be combined with the full-arm options")
            report, pair_arrays = run_round49_pair_boundary(
                args.oracle_root or ROOTS[args.case],
                plant=args.round49_pair_plant,
                allow_dirty=args.allow_dirty,
            )
            persist_round49_pair_arrays(args.output, report, pair_arrays)
            if args.round49_pair_reference:
                report["pair_comparison"] = compare_round49_pair_arrays(
                    args.round49_pair_reference, report)
            encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
            args.output.write_text(encoded)
            print(encoded, end="")
            return 2 if args.round49_pair_plant else (
                0 if report["status"] == "AT-BAR" else 1)
        except (GateError, OSError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
    require(not args.round49_pair_reference,
            "--round49-pair-reference requires --round49-pair-boundary")
    require(not args.round49_pair_plant,
            "--round49-pair-plant requires --round49-pair-boundary")
    # Exit codes: 0 AT-BAR, 1 DEBT (measured), 2 gate failure (a planted
    # control that did not land, a dirty tree, a bad oracle record).
    try:
        with capture_residual_fields() as residuals:
            report = run(args.case, args.oracle_root or ROOTS[args.case],
                         plant_stage=args.plant_stage, plant_operand=args.plant_operand,
                         plant_prediction=args.plant_prediction,
                         allow_dirty=args.allow_dirty, faithful_only=args.faithful_only)
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    if args.output:
        write_residual_artifact(report, args.output, residuals)
    elif args.compare_to:
        print("FAIL: --compare-to requires --output for the residual sidecar", file=sys.stderr)
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    if args.compare_to:
        # Exit status then reports the oracle-relative cellwise comparison
        # against a committed before report. Every arm here
        # other than "faithful" is a private ablation whose meaning a refactor
        # may legitimately redefine, so a comparison of this gate normally
        # passes --compare-rows-matching .faithful. and says so in the receipt.
        comparison = run_ulp_comparison(args, report)
        print(json.dumps(comparison, indent=2, sort_keys=True))
        print(persist_ulp_comparison(args, comparison))
        code = comparison_exit_code(comparison)
        if code == 2:
            print("PLANTED CONTROL DID NOT PRODUCE ITS REQUIRED VERDICT: "
                  f"{comparison['plant']}",
                  file=sys.stderr)
        return code
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
