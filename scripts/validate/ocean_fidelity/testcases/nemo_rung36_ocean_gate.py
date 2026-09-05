#!/usr/bin/env python3
"""Ordered exact-entry gate for the rung-3.6 one-layer ocean boundaries."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.source_rounding import nemo_source_round as rnd
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_flux_form_barotropic_velocity_update,
)

BAR = 1.0e-15

STAGGER_FIELDS = (
    "utau", "utau_east", "vtau", "vtau_north", "umask", "vmask",
    "tmask", "tmask_east", "tmask_north", "utauU", "vtauV",
)
DRAG_FIELDS = (
    "rCdU_ice", "rCdU_top", "rCdU_top_east", "rCdU_top_north",
    "rCdU_bot", "rCdU_bot_east", "rCdU_bot_north", "top_rate",
    "bottom_rate", "ssmask", "tmask", "miku", "mikv", "mbku", "mbkv",
)
STP_FIELDS = (
    "emp", "sshe_rhs", "Ue_rhs", "Ve_rhs", "CdU_u", "CdU_v",
    "utauU", "vtauV", "rCdU_top", "rCdU_bot", "ssh_before",
    "u_before", "v_before", "ssh_after", "u_after", "v_after",
    "un_adv", "vn_adv", "is_pre", "is_post",
)
SPG_FIELDS = (
    "dt_e", "ssh_frc", "p1", "p2", "p3", "b0", "b1", "b2", "b3",
    "wgt1", "wgt2", "sshbb", "sshb", "sshn", "ssha", "sshmid",
    "hu", "hv", "div", "usp", "vsp", "utrd", "vtrd", "ufrc", "vfrc",
    "ua", "va", "hue", "hve", "un_adv", "vn_adv", "ub_sum", "vb_sum",
    "ssh_sum", "wgt1_sum",
)
SPG_STATEMENT_FIELDS = tuple(f"operand_{index}" for index in range(21))
DYNZDF_FIELDS = (
    "u_diag_pre", "u_diag", "u_rhs_pre", "u_rhs", "v_diag_pre", "v_diag",
    "v_rhs_pre", "v_rhs", "utauU", "vtauV", "top", "top_east",
    "top_north", "bottom", "bottom_east", "bottom_north", "e3u", "e3v",
    "umask", "vmask", "rho0", "dt",
)
_VALIDATED_SPG_HEADERS: set[Path] = set()


def _records(path: Path, magic: bytes, header: str, nvalue: int):
    result = []
    with path.open("rb") as stream:
        while marker := stream.read(16):
            if marker != magic:
                raise ValueError(f"bad magic in {path.name}: {marker!r}")
            head = struct.unpack("=" + header, stream.read(struct.calcsize(header)))
            if head[-2:] != (nvalue, 64):
                raise ValueError(f"untrue header in {path.name}: {head!r}")
            values = np.fromfile(stream, np.float64, nvalue)
            if values.size != nvalue:
                raise ValueError(f"truncated record in {path.name}")
            result.append((head, values))
    return result


def _row(name: str, got, oracle):
    got = np.asarray(got, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    diff = np.abs(got - oracle)
    scale = np.maximum(np.abs(oracle), 1.0)
    norm = diff / scale
    bit = got.view(np.uint64) == oracle.view(np.uint64)
    first = np.flatnonzero(norm > BAR)
    return {
        "name": name,
        "count": int(got.size),
        "first_bit_identical": bool(bit.reshape(-1)[0]),
        "bit_identical": int(np.count_nonzero(bit)),
        "non_bit": int(got.size - np.count_nonzero(bit)),
        "max_abs": float(np.max(diff, initial=0.0)),
        "max_normalized": float(np.max(norm, initial=0.0)),
        "first_over_bar_index": int(first[0]) if first.size else None,
        "status": "DEBT" if first.size else "AT_BAR",
    }


def _spg_field(path: Path, index: int, *, all_cycles: bool = False):
    """Zero-copy field view of the fixed, truthful R14D SPG schema."""
    header_nint = 9
    record_bytes = 16 + struct.calcsize(f"={header_nint}i") + 8 * len(SPG_FIELDS)
    cycles = 946
    raw = np.memmap(path, np.uint8, "r")
    if raw.size % (record_bytes * cycles):
        raise ValueError("SPG stream size is not an integer number of steps")
    if path not in _VALIDATED_SPG_HEADERS:
        nrecord = raw.size // record_bytes
        count = np.ndarray(
            (nrecord,), dtype="=i4", buffer=raw,
            offset=16 + 4 * (header_nint - 2), strides=(record_bytes,))
        bits = np.ndarray(
            (nrecord,), dtype="=i4", buffer=raw,
            offset=16 + 4 * (header_nint - 1), strides=(record_bytes,))
        if not np.all(count == len(SPG_FIELDS)) or not np.all(bits == 64):
            raise ValueError("untrue SPG nvalue/STORAGE_SIZE header")
        if (bytes(raw[:16]) != b"NEMO_L3SPG__001 "
                or bytes(raw[(nrecord - 1) * record_bytes:
                             (nrecord - 1) * record_bytes + 16])
                != b"NEMO_L3SPG__001 "):
            raise ValueError("bad first/last SPG magic")
        _VALIDATED_SPG_HEADERS.add(path)
    steps = raw.size // (record_bytes * cycles)
    view = np.ndarray(
        (steps, cycles), dtype="=f8", buffer=raw,
        offset=52 + 8 * index,
        strides=(record_bytes * cycles, record_bytes),
    )
    return np.asarray(view if all_cycles else view[:, -1])


def evaluate(
    root: Path,
    statement_root: Path,
    plant: str | None = None,
    top_drag_scale: float = 1.0,
):
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    rows = []

    stagger = np.asarray([z for _, z in _records(
        root / "oracle_rung36_stagger_frames.bin", b"NEMO_L3STG__001 ",
        "5i", len(STAGGER_FIELDS))])
    s = jnp.asarray(stagger)
    got_u = rnd(rnd(rnd(jnp.asarray(0.5) * rnd(s[:, 0] + s[:, 1]))
                    * rnd(jnp.asarray(2.0) - s[:, 4]))
                * jnp.maximum(s[:, 6], s[:, 7]))
    got_v = rnd(rnd(rnd(jnp.asarray(0.5) * rnd(s[:, 2] + s[:, 3]))
                    * rnd(jnp.asarray(2.0) - s[:, 5]))
                * jnp.maximum(s[:, 6], s[:, 8]))
    if plant == "stagger":
        got_u = got_u.at[0].add(jnp.asarray(1.0e-8))
    rows += [_row("POST_SBC_STAGGER.utauU", got_u, stagger[:, 9]),
             _row("POST_SBC_STAGGER.vtauV", got_v, stagger[:, 10])]

    drag = np.asarray([z for _, z in _records(
        root / "oracle_rung36_drag_frames.bin", b"NEMO_L3DRG__001 ",
        "6i", len(DRAG_FIELDS))])
    d = jnp.asarray(drag)
    drag_scale = jnp.asarray(top_drag_scale, dtype=d.dtype)
    raw_top = rnd(d[:, 1] * drag_scale)
    raw_top_east = rnd(d[:, 2] * drag_scale)
    raw_top_north = rnd(d[:, 3] * drag_scale)
    top_u = rnd(jnp.asarray(0.5) * rnd(raw_top + raw_top_east))
    top_v = rnd(jnp.asarray(0.5) * rnd(raw_top + raw_top_north))
    if plant == "drag":
        top_u = top_u.at[0].add(jnp.asarray(1.0e-8))
    rows += [_row("POST_ZDF_DRG_COEFF.rCdU_top", raw_top, drag[:, 1]),
             _row("POST_ZDF_DRG_COEFF.CdU_u_top", top_u,
                  0.5 * (drag[:, 1] + drag[:, 2])),
             _row("POST_ZDF_DRG_COEFF.CdU_v_top", top_v,
                  0.5 * (drag[:, 1] + drag[:, 3]))]

    stp = _records(root / "oracle_rung36_stp2d_frames.bin",
                   b"NEMO_L3STP__001 ", "8i", len(STP_FIELDS))
    pre = next(z for h, z in stp if h[0] == 1 and h[5] == 0)
    # e3u/e3v at kt=1 is the exact one-cell SSM entry thickness.  It is not
    # reconstructed as bathymetry+SSH: iceistate's load adjustment has a
    # different source association.
    ssm = _records(root / "oracle_rung36_ssm_frames.bin",
                   b"NEMO_L3SSM__001 ", "7i", 15)
    e3 = next(z[12] for h, z in ssm if h[1] == 1 and h[4] == 1)
    r1rho = rnd(jnp.asarray(1.0) / jnp.asarray(NEMO_CONSTANTS_CONFIG.rho_0))
    r1h = rnd(jnp.asarray(1.0) / jnp.asarray(e3))
    got_ssh = rnd(r1rho * jnp.asarray(pre[0]))
    got_ue = rnd(rnd(r1rho * jnp.asarray(pre[6])) * r1h)
    got_ve = rnd(rnd(r1rho * jnp.asarray(pre[7])) * r1h)
    if plant == "pre_spg":
        got_ue = got_ue + jnp.asarray(1.0e-8)
    rows += [_row("PRE_DYN_SPG_TS.sshe_rhs", got_ssh, pre[1]),
             _row("PRE_DYN_SPG_TS.Ue_rhs", got_ue, pre[2]),
             _row("PRE_DYN_SPG_TS.Ve_rhs", got_ve, pre[3])]

    stmt = _records(statement_root / "oracle_rung36_spg_statement_frames.bin",
                    b"NEMO_L3SPGS_001 ", "9i", len(SPG_STATEMENT_FIELDS))
    if len(stmt) != 946 or [h[5] for h, _ in stmt] != list(range(1, 947)):
        raise ValueError("statement stream is not the registered kt=1 946-cycle")
    q = jnp.asarray(np.asarray([z for _, z in stmt]))
    one = jnp.ones((q.shape[0],), dtype=jnp.float64)
    got_su = nemo_flux_form_barotropic_velocity_update(
        q[:, 1], q[:, 0], q[:, 3], q[:, 5], q[:, 7], q[:, 9],
        rnd(one / q[:, 17]), q[:, 11], q[:, 13], q[:, 15], one,
        inverse_depth_after=q[:, 17])
    got_sv = nemo_flux_form_barotropic_velocity_update(
        q[:, 2], q[:, 0], q[:, 4], q[:, 6], q[:, 8], q[:, 10],
        rnd(one / q[:, 18]), q[:, 12], q[:, 14], q[:, 16], one,
        inverse_depth_after=q[:, 18])
    if plant == "substep":
        got_su = got_su.at[0].add(jnp.asarray(1.0e-8))
    rows += [_row("SSH_SUBSTEP.ua_statement", got_su, np.asarray(q[:, 19])),
             _row("SSH_SUBSTEP.va_statement", got_sv, np.asarray(q[:, 20]))]

    # POST_STP2D: dynspg_ts.F90:867-903 first divides each primary sum by
    # r1_wgt1s, then converts the flux-form primary transport to velocity using
    # the full metric-weighted ssh-to-face statement.  The 100 m C1D metric is
    # uniform but must not be algebraically cancelled before rounding.
    spg_path = root / "oracle_rung36_spg_frames.bin"
    rows += [_row("SSH_SUBSTEP.ua_after_lbc", np.asarray(q[:, 19]),
                  _spg_field(spg_path, 25, all_cycles=True)[0]),
             _row("SSH_SUBSTEP.va_after_lbc", np.asarray(q[:, 20]),
                  _spg_field(spg_path, 26, all_cycles=True)[0])]
    sum_u, sum_v = _spg_field(spg_path, 31), _spg_field(spg_path, 32)
    sum_ssh, divisor = _spg_field(spg_path, 33), _spg_field(spg_path, 34)
    sum_adv_u = _spg_field(spg_path, 29)
    sum_adv_v = _spg_field(spg_path, 30)
    raw_w2 = _spg_field(spg_path, 10, all_cycles=True)
    transport_divisor = np.sum(raw_w2, axis=1, dtype=np.float64)
    eta_out = rnd(jnp.asarray(sum_ssh) / jnp.asarray(divisor))
    primary_u = rnd(jnp.asarray(sum_u) / jnp.asarray(divisor))
    primary_v = rnd(jnp.asarray(sum_v) / jnp.asarray(divisor))
    metric = jnp.asarray(10000.0)  # const-ok: 100 m C1D e1e2
    inv_metric = rnd(jnp.asarray(1.0) / metric)
    ssh_face = rnd(rnd(jnp.asarray(0.5) * inv_metric)
                   * rnd(rnd(metric * eta_out) + rnd(metric * eta_out)))
    depth_face = rnd(jnp.asarray(10.0) + ssh_face)  # Decision 6 bathymetry
    primary_u = rnd(primary_u / depth_face)
    primary_v = rnd(primary_v / depth_face)
    adv_u = rnd(jnp.asarray(sum_adv_u) / jnp.asarray(transport_divisor))
    adv_v = rnd(jnp.asarray(sum_adv_v) / jnp.asarray(transport_divisor))
    post_values = np.asarray([z for h, z in stp if h[5] == 1])
    if plant == "post_stp2d":
        primary_u = primary_u.at[0].add(jnp.asarray(1.0e-8))
    rows += [_row("POST_STP2D.ssh", eta_out, post_values[:, 13]),
             _row("POST_STP2D.uu_b", primary_u, post_values[:, 14]),
             _row("POST_STP2D.vv_b", primary_v, post_values[:, 15]),
             _row("POST_STP2D.un_adv", adv_u, post_values[:, 16]),
             _row("POST_STP2D.vn_adv", adv_v, post_values[:, 17])]

    dynzdf = np.asarray([z for _, z in _records(
        root / "oracle_rung36_dynzdf_frames.bin", b"NEMO_L3ZDF__001 ",
        "8i", len(DYNZDF_FIELDS))])
    z = jnp.asarray(dynzdf)
    half_dt = rnd(z[:, 21] * jnp.asarray(0.5))
    got_ud1 = rnd(rnd(z[:, 0] - rnd(half_dt * rnd(z[:, 14] + z[:, 13]) / z[:, 16]))
                  - rnd(half_dt * rnd(z[:, 11] + z[:, 10]) / z[:, 16]))
    got_vd1 = rnd(rnd(z[:, 4] - rnd(half_dt * rnd(z[:, 15] + z[:, 13]) / z[:, 17]))
                  - rnd(half_dt * rnd(z[:, 12] + z[:, 10]) / z[:, 17]))
    got_ur1 = rnd(z[:, 2] + rnd(
        rnd(z[:, 21] * z[:, 8]) / rnd(z[:, 16] * z[:, 20])) * z[:, 18])
    got_vr1 = rnd(z[:, 6] + rnd(
        rnd(z[:, 21] * z[:, 9]) / rnd(z[:, 17] * z[:, 20])) * z[:, 19])
    if plant == "dynzdf":
        got_ud1 = got_ud1.at[0].add(jnp.asarray(1.0e-8))
    rows += [_row("PRE_DYN_ZDF_SOLVE.u_diagonal", got_ud1, dynzdf[:, 1]),
             _row("PRE_DYN_ZDF_SOLVE.v_diagonal", got_vd1, dynzdf[:, 5]),
             _row("PRE_DYN_ZDF_SOLVE.u_rhs", got_ur1, dynzdf[:, 3]),
             _row("PRE_DYN_ZDF_SOLVE.v_rhs", got_vr1, dynzdf[:, 7])]

    first = next((row for row in rows if row["status"] == "DEBT"), None)
    target = {"stagger": "POST_SBC_STAGGER.utauU",
              "drag": "POST_ZDF_DRG_COEFF.CdU_u_top",
              "pre_spg": "PRE_DYN_SPG_TS.Ue_rhs",
              "substep": "SSH_SUBSTEP.ua_statement",
              "post_stp2d": "POST_STP2D.uu_b",
              "dynzdf": "PRE_DYN_ZDF_SOLVE.u_diagonal"}.get(plant)
    step1_all_bit = all(
        (row["non_bit"] == 0 if row["name"].startswith("SSH_SUBSTEP")
         else row["first_bit_identical"])
        for row in rows)
    return {
        "verdict": "AT_BAR" if first is None else "DEBT",
        "bar": BAR,
        "backend": jax.default_backend(),
        "dtype": str(q.dtype),
        "rows": rows,
        "step1_all_registered_boundaries_bit_identical": step1_all_bit,
        "first_over_bar": first,
        "time_level_registry": {
            "stagger_drag_dynzdf": "every kt; odd Kbb=1/even Kbb=3",
            "pre_dyn_spg_ts": "kt=1 Kbb=Kmm=1, Krhs=Kaa=3",
            "ssh_statement": "kt=1, jn=1..946, immediately after source statement before lbc_lnk",
        },
        "plant": plant,
        "top_drag_scale": top_drag_scale,
        "plant_binding": None if plant is None else {
            "target": target,
            "red": next(r for r in rows if r["name"] == target)["status"] == "DEBT",
        },
    }


def evaluate_continuous(root: Path, plant: str | None = None):
    """Advance the shared slab once and compare the next registered entry.

    The ordered exact-entry gate ends immediately before ``dyn_zdf`` solves.
    ``sbc_ssm`` at kt=2 is therefore the first already-registered observation
    of the fully advanced kt=1 state.  Stop on its first red row; do not make
    claims about later trajectory frames from a divergent carry.
    """
    from legoesm.coupler.ocean_forcing import (
        NemoSI3ExchangeConfig,
        nemo_si3_exchange_forcing,
    )
    from legoesm.grids.halo_latlon import set_meridionally_periodic
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_c1d_omip_l3_slab_ocean_card,
    )
    try:
        from scripts.validate.ocean_fidelity.testcases.nemo_rung36_exchange_gate import (
            _exchange, _fwb, _qsr, _ssm,
        )
    except ModuleNotFoundError:  # direct ``python path/to/gate.py`` invocation
        from nemo_rung36_exchange_gate import _exchange, _fwb, _qsr, _ssm

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    exchange = _exchange(root / "oracle_si3_exchange_frames.bin")
    fwb = _fwb(root / "oracle_rung36_fwb_frames.bin")
    qsr = _qsr(root / "oracle_rung36_qsr_frames.bin")
    instant, _before, _after, registry = _ssm(
        root / "oracle_rung36_ssm_frames.bin")
    if instant.shape[0] != 8760:
        raise ValueError("continuous SSM registry is not the full 8760 steps")

    def scalar(name, source):
        return jnp.asarray(source[name][0:1].reshape((1, 1)))

    freshwater, surface = nemo_si3_exchange_forcing(
        qsr=scalar("qsr", exchange),
        qns=scalar("qns", fwb),
        emp=scalar("emp", fwb),
        sfx=scalar("sfx", exchange),
        utau=scalar("utau", exchange),
        vtau=scalar("vtau", exchange),
        chl=scalar("chl", qsr),
        rCdU_ice=scalar("rCdU_ice", exchange),
        snwice_fmass=scalar("snwice_fmass", exchange),
        config=NemoSI3ExchangeConfig(),
    )
    card = build_c1d_omip_l3_slab_ocean_card()
    if card.precision_policy != PrecisionPolicy.fp64(transcendentals="libm"):
        raise ValueError("slab card did not bind fp64 scalar-libm")
    set_meridionally_periodic(card.meridionally_periodic)
    try:
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
        state = model.step(
            card.recipe.initial_state, card.dt_s,
            freshwater=freshwater, surface_forcing=surface)
    finally:
        set_meridionally_periodic(False)

    got = np.asarray([
        state.u.data[0, 0, 0], state.v.data[0, 0, 0],
        state.T.data[0, 0, 0], state.S.data[0, 0, 0],
        state.eta.data[0, 0],
        np.float64(10.0) + state.eta.data[0, 0],
    ], dtype=np.float64)
    if plant == "trajectory":
        got[0] += np.float64(1.0e-8)
    expected = np.asarray(instant[1, :6], dtype=np.float64)
    names = ("u", "v", "temperature", "salinity", "ssh", "e3t")
    rows = [_row(f"CONTINUOUS.kt2_PRE_SSM.{name}", got[i], expected[i])
            for i, name in enumerate(names)]
    first = next((row for row in rows if row["status"] == "DEBT"), None)
    return {
        "verdict": "AT_BAR" if first is None else "STOP_FIRST_OVER_BAR",
        "bar": BAR,
        "backend": jax.default_backend(),
        "dtype": str(state.T.data.dtype),
        "advanced_steps": 1,
        "next_registered_step": 2,
        "rows": rows,
        "first_over_bar": first,
        "owner_interval": (
            "after kt=1 PRE_DYN_ZDF_SOLVE and before kt=2 PRE_SSM; "
            "the shared implicit-ZDF solve / RK3 momentum reconciliation is "
            "the first unregistered owner interval"),
        "error_growth": "withheld after the ordered first-over-bar stop",
        "ssm_registry": registry,
        "plant": plant,
        "plant_binding": None if plant is None else {
            "target": "CONTINUOUS.kt2_PRE_SSM.u",
            "red": rows[0]["status"] == "DEBT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--statement-root", type=Path)
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--top-drag-scale", type=float, default=1.0)
    parser.add_argument("--plant", choices=("stagger", "drag", "pre_spg", "substep", "post_stp2d", "dynzdf", "trajectory"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trajectory:
        if args.plant not in (None, "trajectory"):
            parser.error("--trajectory only accepts --plant trajectory")
        report = evaluate_continuous(args.root, args.plant)
    else:
        if args.plant == "trajectory":
            parser.error("--plant trajectory requires --trajectory")
        report = evaluate(
            args.root, args.statement_root or args.root, args.plant,
            args.top_drag_scale)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.plant:
        return 1 if report["plant_binding"]["red"] else 2
    return 0 if report["verdict"] == "AT_BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
