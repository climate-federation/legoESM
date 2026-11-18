#!/usr/bin/env python3
"""Fail-closed C0/C1 scorer and D225/D226 compiled-C2 comparison."""

from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.nemo_namelist import parse_namelist
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.config import SI3_HARD_SWITCHES
from legoesm.ice.si3_finalize import (
    SI3StressInputs,
    SI3StressOutputs,
    si3_ice_update_flx,
    si3_ice_update_tau,
)
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig, SI3State
from legoesm.ice.si3_surface import SI3SurfaceConfig
from legoesm.ocean.forcing.nemo_surface import (
    NemoSurfaceMeans,
    nemo_sbc_ssm_step,
    nemo_t_stress_to_uv,
)

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    _fluxes_from_record,
    _late_config,
    _state_from_record,
    _update_forcing,
    read_record_verified,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o16_master_gate import (
    record_name,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_composition_scorer import (
    read_record as read_o17_record,
)

MAGIC = b"CLAD0001"
UNSCORABLE = "NOT_YET_SCORABLE_NO_COMPOSED_LEGOESM_COUPLED_STEP"
ADMITTED_ICE_DIMENSIONS = (5, 10, 5)
C2_VERDICT_KINDS = (3, 4, 5, 6, 15, 16, 19, 20, 34)
C2_REPORT_ONLY_KINDS = (1, 2)
_HEADER_BYTES = 8 + 4 + 11 * 4
_KIND1_FLAGS = (
    ("restart_means_present", "i4"),
    ("use_conservative_temperature", "i4"),
    ("remove_inverse_barometer", "i4"),
)
_KIND1_BEFORE_IB = (
    ("temperature_now", "F2"),
    ("salinity_now", "F2"),
    ("u_before", "F2"),
    ("v_before", "F2"),
    ("ssh_now", "F2"),
)
_KIND1_IB = (("ssh_inverse_barometer", "F2"),)
_KIND1_AFTER_IB = tuple(
    (name, "F2")
    for name in (
        "e3t_reference",
        "r3t_now",
        "surface_mask",
        "shortwave_fraction_top",
        "ssu_prior",
        "ssv_prior",
        "sst_prior",
        "sss_prior",
        "ssh_prior",
        "e3t_prior",
        "frq_prior",
    )
)
_FIXED_SPECS = {
    2: tuple((name, "F2") for name in ("ssu", "ssv", "sst", "sss", "ssh", "e3t", "frq")),
    3: tuple((name, "F2") for name in ("ssu", "ssv", "sst", "sss", "ssh", "e3t", "frq")),
    4: (
        ("utau", "F2"),
        ("vtau", "F2"),
        ("taum", "A2D0"),
        ("utauU", "F2"),
        ("vtauV", "F2"),
        ("qsr", "A2D0"),
        ("qns", "A2D0"),
        ("emp", "F2"),
        ("sfx", "A2D0"),
        ("fr_i", "F2"),
        ("fwfice", "A2D0"),
        ("tmask", "F2"),
        ("umask", "F2"),
        ("vmask", "F2"),
    ),
    5: (("ts", "F4"), ("uu", "F3"), ("vv", "F3"), ("ssh", "F2")),
    6: (("ts", "F4"), ("uu", "F3"), ("vv", "F3"), ("ssh", "F2")),
}


def _record_specs(kind: int, remove_inverse_barometer: int | None = None):
    if kind == 1:
        if remove_inverse_barometer not in (0, 1):
            raise ValueError(
                "invalid kind-1 remove_inverse_barometer flag: "
                f"expected 0 or 1, got {remove_inverse_barometer}"
            )
        optional = _KIND1_IB if remove_inverse_barometer == 1 else ()
        return _KIND1_FLAGS + _KIND1_BEFORE_IB + optional + _KIND1_AFTER_IB
    try:
        return _FIXED_SPECS[kind]
    except KeyError as exc:
        raise ValueError(f"unsupported C record kind: {kind}") from exc


def _field_shape(code: str, *, jpi: int, jpj: int, jpk: int, jpts: int) -> tuple[int, ...]:
    if code == "F2":
        return jpi, jpj
    if code == "A2D0":
        if jpi <= 4 or jpj <= 4:
            raise ValueError(f"invalid C record A2D(0) domain: jpi={jpi}, jpj={jpj}")
        return jpi - 4, jpj - 4
    if code == "F3":
        return jpi, jpj, jpk
    if code == "F4":
        return jpi, jpj, jpk, jpts
    raise ValueError(f"unsupported C record field shape: {code}")


def read_c_record(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < _HEADER_BYTES:
        raise ValueError(
            f"truncated C record header: expected {_HEADER_BYTES} bytes, found {len(raw)}"
        )
    if raw[:8] != MAGIC:
        raise ValueError(f"bad C record magic: expected {MAGIC!r}, found {raw[:8]!r}")
    version, *meta = struct.unpack_from("<12i", raw, 8)
    kind, kt, kbb, kmm, rank, jpi, jpj, jpk, jpts, bits, nfields = meta
    if version != 1:
        raise ValueError(f"unsupported C record version: expected 1, got {version}")
    if bits != 64:
        raise ValueError(f"unsupported C record real width: expected 64 bits, got {bits}")
    if min(jpi, jpj, jpk, jpts) <= 0:
        raise ValueError(
            f"invalid C record dimensions: jpi={jpi}, jpj={jpj}, jpk={jpk}, jpts={jpts}"
        )
    offset = _HEADER_BYTES
    fields = {}
    if kind == 1:
        for name, _ in _KIND1_FLAGS:
            stop = offset + 4
            if stop > len(raw):
                raise ValueError(f"truncated C record payload: kind=1 field={name}")
            fields[name] = struct.unpack_from("<i", raw, offset)[0]
            if fields[name] not in (0, 1):
                raise ValueError(
                    f"invalid kind-1 {name} flag: expected 0 or 1, got {fields[name]}"
                )
            offset += 4
        specs = _record_specs(kind, fields["remove_inverse_barometer"])
    else:
        specs = _record_specs(kind)
    if nfields != len(specs):
        if kind == 1:
            flag = fields["remove_inverse_barometer"]
            raise ValueError(
                "inconsistent kind-1 layout: "
                f"remove_inverse_barometer={flag} requires nfields={len(specs)}, got {nfields}"
            )
        raise ValueError(
            f"inconsistent kind-{kind} layout: expected nfields={len(specs)}, got {nfields}"
        )
    for name, code in specs[len(_KIND1_FLAGS) if kind == 1 else 0 :]:
        shape = _field_shape(code, jpi=jpi, jpj=jpj, jpk=jpk, jpts=jpts)
        count = int(np.prod(shape))
        stop = offset + count * 8
        if stop > len(raw):
            raise ValueError(
                f"truncated C record payload: kind={kind} field={name}; "
                f"need {count * 8} bytes, have {max(0, len(raw) - offset)}"
            )
        fields[name] = np.frombuffer(raw, "<f8", count=count, offset=offset).reshape(
            shape, order="F"
        )
        offset = stop
    if offset != len(raw):
        raise ValueError(
            f"trailing C record bytes: kind={kind}; parsed {offset}, file has {len(raw)}"
        )
    return {
        "kind": kind,
        "kt": kt,
        "kbb": kbb,
        "kmm": kmm,
        "rank": rank,
        "jpi": jpi,
        "jpj": jpj,
        "jpk": jpk,
        "jpts": jpts,
        "nfields": nfields,
        "fields": fields,
    }


def _write_stub_kind1_record(
    path: Path,
    *,
    remove_inverse_barometer: int,
    old_unconditional_layout: bool,
) -> None:
    """Write a flag-governed kind-1 record or the allocation-off old-layout plant."""
    jpi, jpj, jpk, jpts = 6, 7, 2, 2
    if old_unconditional_layout and remove_inverse_barometer != 0:
        raise ValueError("old unconditional kind-1 plant requires allocation-off flag")
    specs = _record_specs(1, remove_inverse_barometer)
    nfields = 20 if old_unconditional_layout else len(specs)
    header = MAGIC + struct.pack(
        "<12i",
        1,
        1,
        1,
        1,
        1,
        0,
        jpi,
        jpj,
        jpk,
        jpts,
        64,
        nfields,
    )
    payload = bytearray()
    flag_values = (0, 0, remove_inverse_barometer)
    for (_, code), value in zip(_KIND1_FLAGS, flag_values, strict=True):
        assert code == "i4"
        payload.extend(struct.pack("<i", value))
    for index, (_, code) in enumerate(specs[len(_KIND1_FLAGS) :], start=1):
        shape = _field_shape(code, jpi=jpi, jpj=jpj, jpk=jpk, jpts=jpts)
        payload.extend(np.full(shape, float(index), dtype="<f8").tobytes(order="F"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + payload)


def _equal(name: str, got, want) -> None:
    if not np.array_equal(np.asarray(got).view(np.uint64), np.asarray(want).view(np.uint64)):
        raise ValueError(f"STATUS PLANT-FIRED: {name} is not 0 ULP")


def _bit_words(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.dtype.kind != "f" or value.dtype.itemsize != 8:
        raise ValueError(f"D225 requires float64 observations, got {value.dtype}")
    return np.ascontiguousarray(value).view(np.uint64).reshape(value.shape)


def _d225_owned_arrays(got, want, mask: np.ndarray):
    """Remove MPI halos, retaining an offset for record-coordinate reporting."""
    got = np.asarray(got)
    want = np.asarray(want)
    mask = np.asarray(mask, dtype=bool)
    mask_rank = mask.ndim
    if got.shape[:mask_rank] == mask.shape:
        if min(mask.shape[:2]) <= 4:
            raise ValueError(f"D225 full shard is too small for two-cell halos: {mask.shape}")
        owned = (slice(2, -2), slice(2, -2)) + (slice(None),) * (got.ndim - 2)
        mask_owned = (slice(2, -2), slice(2, -2)) + (slice(None),) * (mask_rank - 2)
        return got[owned], want[owned], mask[mask_owned], (2, 2)
    if (
        got.shape[0] == mask.shape[0] + 4
        and got.shape[1] == mask.shape[1] + 4
        and got.shape[2:mask_rank] == mask.shape[2:]
    ):
        owned = (slice(2, -2), slice(2, -2)) + (slice(None),) * (got.ndim - 2)
        return got[owned], want[owned], mask, (2, 2)
    owned_mask_shape = (mask.shape[0] - 4, mask.shape[1] - 4, *mask.shape[2:])
    if min(mask.shape[:2]) >= 5 and got.shape[:mask_rank] == owned_mask_shape:
        mask_owned = (slice(2, -2), slice(2, -2)) + (slice(None),) * (mask_rank - 2)
        return got, want, mask[mask_owned], (0, 0)
    raise ValueError(f"D225 mask {mask.shape} cannot cover field {got.shape}")


def _d225_field(got, want, wet_mask: np.ndarray) -> dict:
    """Apply D225 exactly: wet raw words; dry signed-zero exception only."""
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        raise ValueError(f"C2 shape mismatch: candidate={got.shape}, NEMO={want.shape}")
    got, want, wet_base, offset = _d225_owned_arrays(got, want, wet_mask)
    got_bits, want_bits = _bit_words(got), _bit_words(want)
    wet = np.broadcast_to(
        wet_base.reshape(wet_base.shape + (1,) * (got.ndim - wet_base.ndim)),
        got.shape,
    )
    different = got_bits != want_bits
    sign_zero = different & (got == 0.0) & (want == 0.0)
    wet_bad = different & wet
    dry_sign_zero = sign_zero & ~wet
    dry_bad = different & ~wet & ~sign_zero
    bad = wet_bad | dry_bad
    first = None
    if np.any(bad):
        owned_index = tuple(int(i) for i in np.argwhere(bad)[0])
        index = (
            owned_index[0] + offset[0],
            owned_index[1] + offset[1],
            *owned_index[2:],
        )
        first = {
            "cell": index,
            "got_bits": f"0x{int(got_bits[owned_index]):016x}",
            "nemo_bits": f"0x{int(want_bits[owned_index]):016x}",
            "domain": "wet" if bool(wet[owned_index]) else "dry",
        }
    return {
        "values": int(got.size),
        "wet_values": int(np.count_nonzero(wet)),
        "wet_different": int(np.count_nonzero(wet_bad)),
        "dry_zero_sign_different": int(np.count_nonzero(dry_sign_zero)),
        "dry_other_different": int(np.count_nonzero(dry_bad)),
        "first": first,
    }


def _merge_d225_counts(total: dict, row: dict) -> None:
    for name in (
        "values", "wet_values", "wet_different",
        "dry_zero_sign_different", "dry_other_different",
    ):
        total[name] = total.get(name, 0) + row[name]


def score_c2_observations(
    candidate: dict[tuple[int, int, int], dict[str, object]],
    expected: dict[tuple[int, int, int], dict[str, object]],
    masks: dict[tuple[int, int], dict[str, np.ndarray]],
) -> dict:
    """Compare compiled recipe observations; perform no physical arithmetic.

    Keys are ``(kind, kt, rank)``.  The runner constructs ``candidate`` from
    the two jitted recipe traces; this function only compares and reports.
    """
    required = {
        (kind, kt, rank)
        for kind in (*C2_REPORT_ONLY_KINDS, *C2_VERDICT_KINDS)
        for kt in (1, 2)
        for rank in (0, 1)
        if not (kind in (15, 16) and kt % 2 == 0)
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"C2 candidate observations incomplete; missing={missing[:8]}")
    missing_expected = sorted(required - set(expected))
    if missing_expected:
        raise ValueError(f"C2 NEMO observations incomplete; missing={missing_expected[:8]}")

    counts: dict[int, dict] = {}
    report_only: dict[int, dict] = {}
    first_failure = None
    boundary_order = (1, 2, 3, 15, 16, 19, 20, 34, 4, 5, 6)
    order_index = {kind: index for index, kind in enumerate(boundary_order)}
    for kind, kt, rank in sorted(
        required, key=lambda key: (key[1], order_index[key[0]], key[2])
    ):
        got_fields = candidate[(kind, kt, rank)]
        want_fields = expected[(kind, kt, rank)]
        if set(got_fields) != set(want_fields):
            raise ValueError(
                f"C2 kind={kind} kt={kt} rank={rank} field set mismatch: "
                f"candidate_only={sorted(set(got_fields) - set(want_fields))}, "
                f"NEMO_only={sorted(set(want_fields) - set(got_fields))}"
            )
        kind_total = (report_only if kind in C2_REPORT_ONLY_KINDS else counts).setdefault(
            kind, {}
        )
        mask_names = masks[(kt, rank)]
        for field in sorted(want_fields):
            got = np.asarray(got_fields[field])
            if got.dtype.kind != "f":
                if not np.array_equal(got, np.asarray(want_fields[field])):
                    raise ValueError(
                        f"C2 integer metadata differs kind={kind} field={field} kt={kt} rank={rank}"
                    )
                continue
            if field in {"uu", "ssu", "utauU", "uu_kbb", "u_ice", "u_oce"}:
                staggering = "umask"
            elif field in {"vv", "ssv", "vtauV", "vv_kbb", "v_ice", "v_oce"}:
                staggering = "vmask"
            else:
                staggering = "tmask"
            mask_name = f"{staggering}_3d" if got.ndim >= 3 and field in {"ts", "uu", "vv"} else staggering
            row = _d225_field(got, want_fields[field], mask_names[mask_name] > 0.5)
            _merge_d225_counts(kind_total, row)
            if (
                kind in C2_VERDICT_KINDS
                and first_failure is None
                and row["first"] is not None
            ):
                first_failure = {
                    "kind": kind, "field": field, "kt": kt, "rank": rank,
                    **row["first"],
                }
    if first_failure is None:
        status = "PASS_C2_0_ULP_WET"
    else:
        f = first_failure
        status = (
            f"FAIL_C2 kind={f['kind']} field={f['field']} cell={f['cell']} "
            f"kt={f['kt']} rank={f['rank']} got={f['got_bits']} "
            f"nemo={f['nemo_bits']} domain={f['domain']}"
        )
    return {
        "status": status,
        "compiled": True,
        "verdict_kinds": list(C2_VERDICT_KINDS),
        "report_only_kinds": list(C2_REPORT_ONLY_KINDS),
        "counts": counts,
        "report_only_counts": report_only,
        "first_failure": first_failure,
    }


def load_c2_expected(evidence: Path) -> tuple[dict, dict]:
    """Load only the admitted NEMO observations and D225 physical masks."""
    run = evidence / "REC_C2"
    namelist = _coupled_resolved_namelist(run)
    dimensions = tuple(int(namelist[name]) for name in ("jpl", "nlay_i", "nlay_s"))
    if dimensions != ADMITTED_ICE_DIMENSIONS:
        raise ValueError(
            "STATUS PLANT-FIRED: resolved ice dimensions "
            f"{dimensions} are inadmissible; expected ICE5 {ADMITTED_ICE_DIMENSIONS}"
        )
    c_root = run / "c_ladder_records"
    master_root = run / "o8_o16_master_records"
    o17_root = run / "o17_master_records"
    expected: dict[tuple[int, int, int], dict[str, object]] = {}
    masks: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    for kt in (1, 2):
        for rank in (0, 1):
            for kind in (*C2_REPORT_ONLY_KINDS, 3, 4, 5, 6):
                record = read_c_record(
                    c_root / f"c_kind{kind}_step{kt:04d}_rank{rank:04d}.bin"
                )
                expected[(kind, kt, rank)] = record["fields"]
                if kind == 4:
                    import netCDF4

                    with netCDF4.Dataset(run / f"mesh_mask_{rank:04d}.nc") as dataset:
                        mask_3d = {
                            f"{name}_3d": np.asarray(dataset.variables[name][0], dtype=np.float64).transpose(2, 1, 0)
                            for name in ("tmask", "umask", "vmask")
                        }
                    masks[(kt, rank)] = {
                        name: record["fields"][name]
                        for name in ("tmask", "umask", "vmask")
                    }
                    masks[(kt, rank)].update(mask_3d)
            for kind in (15, 16, 19, 20):
                if kind in (15, 16) and kt % 2 == 0:
                    continue
                record = read_record_verified(
                    master_root / record_name(kind, kt, 0, rank),
                    expected_ice_dimensions=dimensions,
                )
                expected[(kind, kt, rank)] = record["fields"]
            private = read_o17_record(
                o17_root / f"o17_kind34_tau_private_step{kt:04d}_rank{rank:04d}.bin"
            )
            private_dimensions = tuple(
                int(private["meta"][name]) for name in ("jpl", "nlay_i", "nlay_s")
            )
            if private_dimensions != dimensions:
                raise ValueError(
                    f"O17 kind 34 dimensions {private_dimensions} != ICE5 {dimensions}"
                )
            expected[(34, kt, rank)] = private["fields"]
    return expected, masks


def score_c2(evidence: Path, candidate: dict[tuple[int, int, int], dict]) -> dict:
    """Score observations emitted by one externally jitted coupled-pair run."""
    expected, masks = load_c2_expected(evidence)
    return score_c2_observations(candidate, expected, masks)


def _c2_cold_start(run: Path):
    """Build REC_C2 from the deck's K0 ocean and production ice_istate."""
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    from legoesm.ice.dynamics import SI3CGridAEVPState
    from legoesm.ice.si3_finalize import SI3UpdateFluxOutputs
    from legoesm.ice.si3_o17 import SI3O17StepState
    from legoesm.ice.si3_orca2_state import (
        SI3O2InitConfig,
        si3_dom_uniq_t_orca2,
        si3_o2_scalar_initial_state,
        si3_update_snwice_mass,
        si3_var_agg,
        si3_var_glo2eqv,
    )
    from legoesm.ocean import bulk_flux_omip as bulk
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.forcing.nemo_surface import NemoSurfaceMeans
    from legoesm.ocean.freshwater import NemoRunoffSurface
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round340_score_composed_card as r340_score,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round340_standalone_gate as r340,
    )
    from scripts.validate.ocean_fidelity.testcases import (
        nemo_si3_o17_composition_scorer as o17_score,
    )
    from scripts.validate.ocean_fidelity.testcases.nemo_si3_o2_orca2_sas_gate import (
        _load_orca2_sas_ice_bundle_inputs,
    )

    k0 = bulk.build_nemo_orca2_k0_ocean_card(run)
    controls = r340.build_real_composed_inputs(
        k0, run, r340_score.DEFAULT_CORE_MANIFEST, r340_score.DEFAULT_ICE_COVER
    )
    receipt = json.loads((run / "C_DECK_RECEIPT.json").read_text())
    master = Path(receipt["sas_ice_reference"]).parents[1]
    master_run = master / "record_100step"
    ice_card = o17_score.card_for(master_run)._replace(ln_drgice_imp=True)
    ice_dt = 2.0 * float(k0.ocean.dt_s)
    ice_card = ice_card._replace(
        state=ice_card.state._replace(nn_fsbc=2, rDt_ice=ice_dt),
        aevp=ice_card.aevp._replace(dt_s=ice_dt),
        redistribution=ice_card.redistribution._replace(rdt_ice=ice_dt),
        correction=ice_card.correction._replace(rdt_ice=ice_dt),
    )
    index = o17_score.record_paths(master_run / "o17_master_records")
    records = o17_score.load_step(index[(1, 0)], 1, 0)
    o17 = o17_score.live_forcing(records, ice_card)
    o17 = o17._replace(
        stress=o17.stress._replace(tmask_k1=o17.aevp_forcing.tmask_t)
    )

    ocean_state = k0.ocean.recipe.initial_state
    h = int(ice_card.state.nn_hls)
    bundle = _load_orca2_sas_ice_bundle_inputs()
    if bundle is None:
        raise ValueError("C2 cold start requires the admitted ORCA2 input bundle")
    sst = bulk._orca2_si3_full_field(
        ocean_state.T.data[..., 0], ice_card.fold, "T", h
    )
    sss = bulk._orca2_si3_full_field(
        ocean_state.S.data[..., 0], ice_card.fold, "T", h
    )
    tmask = jnp.asarray(bundle["tmask"])
    smask0_i = si3_dom_uniq_t_orca2(nn_hls=h)
    ssh0 = jnp.zeros_like(tmask)
    o2 = si3_o2_scalar_initial_state(
        sst_m=sst,
        sss_m=sss,
        tmask=tmask,
        ff_t=jnp.asarray(bundle["ff_t"]),
        gphit=jnp.asarray(bundle["gphit"]),
        e1e2t=jnp.asarray(bundle["e1e2t"]),
        ssmask=tmask,
        smask0_i=smask0_i,
        ssh_before_kbb=ssh0,
        ssh_before_kmm=ssh0,
        config=SI3O2InitConfig.for_orca2_omip_l4(nn_hls=h),
        state_config=ice_card.state,
    )
    o3 = o2.o3
    zero3 = jnp.zeros_like(o3.a_i)
    eqv = si3_var_glo2eqv(
        v_i=o3.v_i, v_s=o3.v_s, sv_i=o3.sv_i, a_i=o3.a_i,
        oa_i=zero3, a_ip=o3.a_ip, v_ip=o3.v_ip, v_il=o3.v_il,
        e_s=o3.e_s, e_i=o3.e_i, szv_i=o3.szv_i, kn=1,
        hi_max_top=ice_card.state.rn_himax, sss_m=sss, config=ice_card.state,
    )
    agg = si3_var_agg(
        v_i=o3.v_i, v_s=eqv.v_s, sv_i=o3.sv_i, a_i=eqv.a_i,
        oa_i=zero3, a_ip=o3.a_ip, v_ip=o3.v_ip, v_il=o3.v_il,
        t_su=o3.t_su, t_si=zero3, e_s=o3.e_s, e_i=o3.e_i,
        szv_i=o3.szv_i, a_ip_eff=eqv.a_ip_eff, t_i=eqv.t_i,
        t_s=eqv.t_s, kn=1, config=ice_card.state,
    )
    mass, mass_b, _ = si3_update_snwice_mass(
        vt_i=agg.vt_i, vt_s=agg.vt_s, vt_ip=agg.vt_ip, vt_il=agg.vt_il,
        snwice_mass_prev=jnp.zeros_like(agg.vt_i), tmask=tmask,
        config=ice_card.state, is_init=True,
    )
    ice = SI3State.zeros(*tmask.shape, config=ice_card.state)._replace(
        a_i=eqv.a_i, v_i=o3.v_i, v_s=eqv.v_s, sv_i=o3.sv_i,
        t_su=o3.t_su, oa_i=zero3, a_ip=o3.a_ip, v_ip=o3.v_ip,
        v_il=o3.v_il, e_s=o3.e_s, e_i=o3.e_i, szv_i=o3.szv_i,
        t1_ice=o3.t1_ice, t_si=zero3, equivalent=eqv, aggregated=agg,
        snwice_mass=mass, snwice_mass_b=mass_b,
    )
    zero2 = jnp.zeros_like(tmask)
    ice_entry = SI3O17StepState(
        ice, SI3CGridAEVPState(ice.u_ice, ice.v_ice, zero2, zero2, zero2)
    )
    eta = o2.ssh_kmm[h:-h, h:-h].T
    ocean_state = ocean_state._replace(eta=ocean_state.eta.replace(data=eta))

    config = k0.ocean.recipe.model_config
    vmix = config.physics.vertical_mixing
    physics = config.physics._replace(
        vertical_mixing=vmix._replace(tke=vmix.tke._replace(nemo_mxlice=2))
    )
    config = config._replace(
        physics=physics, implicit_ice_ocean_top_drag=True
    )
    recipe = k0.ocean.recipe._replace(
        model_config=config, initial_state=ocean_state
    )
    ocean_card = k0.ocean._replace(recipe=recipe)
    model = LatLonCGridOceanModel(
        recipe.grid, recipe.z_coord, config, iwm_forcing=recipe.iwm_forcing
    )
    s2 = controls[0]["s2"]
    mask = ocean_state.land_mask.data
    unique = smask0_i[h:-h, h:-h].T
    card = bulk.NemoORCA2SI3Card(
        ocean_card, ice_card, s2.stress_lbc,
        recipe.z_coord.nemo_e1e2t, unique, s2.runoff_mouth_mask, mask,
        s2.tmask_east, s2.tmask_north, s2.umask, s2.vmask,
        s2.runoff_factor, s2.reference_density, s2.restoring_mm_per_day,
        s2.restoring_bound_mm_per_day, s2.seconds_per_day, jnp.asarray(365.0),
        s2.salinity_floor, s2.heat_capacity, 2920, 2,
        jnp.asarray(k0.albedo), jnp.asarray(k0.precip_factor),
        jnp.asarray(k0.evap_factor), k0.nb_iter,
    )
    forcing = tuple(
        bulk.NemoORCA2SI3Forcing(
            row["s0f"], o17, row["chl"], row["geothermal_heat_flux_mwm2"]
        )
        for row in controls[:2]
    )
    owned = ocean_state.eta.data.shape
    full = tmask.shape
    ice_owned = (full[0] - 2 * h, full[1] - 2 * h)
    fluxes = SI3UpdateFluxOutputs(*(
        jnp.zeros(
            full if name in ("fr_i", "snwice_mass", "snwice_mass_b", "snwice_fmass")
            else ice_owned,
            dtype=jnp.float64,
        )
        for name in SI3UpdateFluxOutputs._fields
    ))
    stress_inputs = o17.stress
    stress = SI3StressOutputs(
        jnp.zeros_like(stress_inputs.utau), jnp.zeros_like(stress_inputs.vtau),
        jnp.zeros((full[0] - 2 * h, full[1] - 2 * h)),
        jnp.zeros_like(stress_inputs.tmod_io),
        jnp.zeros((full[0] - 2 * h, full[1] - 2 * h)),
        jnp.zeros((full[0] - 2 * h, full[1] - 2 * h)), None,
    )
    zero_yx = jnp.zeros(owned, dtype=jnp.float64)
    carry = bulk.nemo_orca2_si3_cold_start(
        ocean=ocean_state, ice=ice_entry,
        means=NemoSurfaceMeans(
            zero_yx, zero_yx, zero_yx, zero_yx, zero_yx, zero_yx,
            jnp.ones_like(zero_yx),
        ),
        stress=stress, fluxes=fluxes,
        runoff=NemoRunoffSurface(zero_yx, zero_yx, zero_yx),
        restoring_volume_flux=zero_yx, restoring_heat_flux=zero_yx,
        restoring_coefficient=jnp.ones_like(zero_yx),
        rn_fwb0=jnp.asarray(0.0), kt=1,
    )
    sponge = (controls[0]["sponge"], controls[1]["sponge"])
    return carry, forcing, sponge, card, model


def _c2_ocean_fields(state) -> dict[str, np.ndarray]:
    ts = jnp.stack((state.T.data, state.S.data), axis=-1)
    return {
        "ts": np.asarray(ts.transpose(1, 0, 2, 3)),
        "uu": np.asarray(state.u.data[:, 1:, :].transpose(1, 0, 2)),
        "vv": np.asarray(state.v.data[1:, :, :].transpose(1, 0, 2)),
        "ssh": np.asarray(state.eta.data.T),
    }


def _c2_trace_fields(trace, card, kind: int) -> dict[str, object]:
    """Pure observation adapter: names/indexing only, no physical arithmetic."""
    if kind == 1:
        entry = trace.ssm_entry
        out = {
            name: int(getattr(entry, name))
            for name, _ in _KIND1_FLAGS
        }
        for name in (
            "temperature_now", "salinity_now", "u_before", "v_before",
            "ssh_now", "e3t_reference", "r3t_now", "surface_mask",
            "shortwave_fraction_top",
        ):
            out[name] = np.asarray(getattr(entry, name)).T
        out.update({
            f"{name}_prior": np.asarray(getattr(entry.prior, name)).T
            for name in NemoSurfaceMeans._fields
        })
        return out
    if kind in (2, 3):
        return {
            name: np.asarray(getattr(trace.means, name)).T
            for name in NemoSurfaceMeans._fields
        }
    if kind == 4:
        tail, flux = trace.surface, trace.ice_fluxes
        return {
            "utau": np.asarray(tail.utau).T,
            "vtau": np.asarray(tail.vtau).T,
            "taum": np.asarray(tail.taum).T,
            "utauU": np.asarray(tail.utau_u).T,
            "vtauV": np.asarray(tail.vtau_v).T,
            "qsr": np.asarray(tail.qsr).T,
            "qns": np.asarray(tail.qns).T,
            "emp": np.asarray(tail.emp).T,
            "sfx": np.asarray(tail.sfx).T,
            "fr_i": np.asarray(flux.fr_i),
            "fwfice": np.asarray(flux.fwfice),
            "tmask": np.asarray(card.surface_mask).T,
            "umask": np.asarray(card.umask).T,
            "vmask": np.asarray(card.vmask).T,
        }
    if kind == 5:
        return _c2_ocean_fields(trace.ocean_entry)
    if kind == 6:
        return _c2_ocean_fields(trace.ocean_exit)
    if trace.ice_core is None and kind in (15, 16):
        raise ValueError(f"kind {kind} is unavailable on an even ocean step")
    if kind == 15:
        from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
            _state_score_fields,
        )

        core = trace.ice_core
        h = int(card.ice.state.nn_hls)
        out = dict(_state_score_fields(core.final.state, interior_halo=h))
        out.update({name: getattr(core.final.fluxes, name) for name in core.final.fluxes._fields})
        out.update({name: getattr(core.update_forcing, name) for name in core.update_forcing._fields})
        out["tmask_k1"] = core.update_forcing.tmask
        out["snwice_mass"] = core.final.state.snwice_mass
        out["snwice_mass_b"] = core.final.state.snwice_mass_b
        return out
    if kind == 16:
        return {
            name: getattr(trace.ice_core.final.outputs, name)
            for name in trace.ice_core.final.outputs._fields
        }
    if kind == 19:
        return {
            name: getattr(trace.stress_inputs, name)
            for name in SI3StressInputs._fields
        }
    if kind == 20:
        return {
            name: getattr(trace.stress, name)
            for name in ("utau", "vtau", "taum")
        }
    if kind == 34:
        return {
            name: getattr(trace.stress, name)
            for name in ("tmod_io", "utau_oce", "vtau_oce")
        }
    raise ValueError(f"unsupported C2 observation kind {kind}")


def _c2_shard(value, shape: tuple[int, ...], rank: int) -> np.ndarray:
    """Index a global ORCA2 observation into one two-rank D225 shard."""
    value = np.asarray(value, dtype=np.float64)
    # NEMO stores jpk=31 including its all-zero bottom dummy level; legoESM's
    # prognostic state stores the 30 live levels.  This is record indexing, not
    # a physical transformation.
    if (
        len(shape) >= 3
        and value.ndim == len(shape)
        and value.shape[2] + 1 == shape[2]
    ):
        value = np.concatenate(
            (value, np.zeros(value.shape[:2] + (1,) + value.shape[3:], dtype=value.dtype)),
            axis=2,
        )
    if value.shape == shape:
        return value
    tail = shape[2:]
    if value.shape[:2] == (184, 152):
        owned = value[2:-2, 2:-2]
    elif value.shape[:2] == (180, 148):
        owned = value
    else:
        raise ValueError(f"cannot shard global C2 field {value.shape} as {shape}")
    owned = owned[rank * 90:(rank + 1) * 90]
    if shape[:2] == (90, 148) and owned.shape == shape:
        return owned
    if shape[:2] != (94, 152) or owned.shape != (90, 148, *tail):
        raise ValueError(f"C2 shard shape mismatch: global={value.shape}, target={shape}")
    result = np.zeros(shape, dtype=np.float64)
    result[2:-2, 2:-2] = owned
    return result


def run_c2_compiled(evidence: Path) -> dict:
    """Run the two-step cold C2 trajectory once under ``jax.jit`` and score it."""
    from legoesm.ocean import bulk_flux_omip as bulk

    expected, masks = load_c2_expected(evidence)
    carry, forcing, sponge, card, model = _c2_cold_start(evidence / "REC_C2")

    @jax.jit
    def compiled_pair(entry):
        odd, odd_trace = bulk.nemo_orca2_si3_coupled_step(
            entry, forcing[0], card, model=model, sponge=sponge[0]
        )
        even, even_trace = bulk.nemo_orca2_si3_coupled_step(
            odd, forcing[1], card, model=model, sponge=sponge[1]
        )
        return even, (odd_trace, even_trace)

    final, traces = compiled_pair(carry)
    jax.block_until_ready(final)
    candidate: dict[tuple[int, int, int], dict[str, object]] = {}
    for kt, trace in enumerate(traces, start=1):
        for kind in (*C2_REPORT_ONLY_KINDS, *C2_VERDICT_KINDS):
            if kind in (15, 16) and kt % 2 == 0:
                continue
            global_fields = _c2_trace_fields(trace, card, kind)
            for rank in (0, 1):
                want = expected[(kind, kt, rank)]
                fields = {}
                for name, reference in want.items():
                    if np.asarray(reference).dtype.kind != "f":
                        fields[name] = global_fields[name]
                    else:
                        fields[name] = _c2_shard(
                            global_fields[name], np.asarray(reference).shape, rank
                        )
                candidate[(kind, kt, rank)] = fields
    return score_c2_observations(candidate, expected, masks)


def _coupled_resolved_namelist(run: Path) -> dict:
    """Read SI3 parameters plus the coupled run's executed ice-drag arm."""
    ice_path = run / "output.namelist.ice"
    if not ice_path.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing resolved ice namelist: {ice_path}")
    values = parse_namelist(ice_path, "ice")
    required = (
        "jpl",
        "nlay_i",
        "nlay_s",
        "rn_simin",
        "rn_sinew",
        "nn_icesal",
        "ln_pnd",
        "ln_pnd_lev",
        "ln_pnd_topo",
        "ln_pnd_lids",
        "ln_pnd_alb",
        "nn_snwfra",
    )
    parsed = {}
    for name in required:
        matches = [tokens for (_, _, key), tokens in values.items() if key == name]
        if len(matches) != 1 or len(matches[0]) != 1:
            raise ValueError(
                f"STATUS PLANT-FIRED: coupled resolved ice namelist lacks one scalar {name}"
            )
        parsed[name if name != "ln_pnd_lev" else "ln_pnd_LEV"] = matches[0][0]
    parsed["ln_pnd_TOPO"] = parsed.pop("ln_pnd_topo")
    ocean_output = run / "ocean.output"
    if not ocean_output.is_file():
        raise ValueError(f"STATUS PLANT-FIRED: missing coupled run log: {ocean_output}")
    drag_values = re.findall(
        r"implicit ice-ocean drag\s+ln_drgice_imp\s*=\s*([TF])\b",
        ocean_output.read_text(errors="replace"),
        flags=re.I,
    )
    resolved = {value.upper() == "T" for value in drag_values}
    if len(resolved) != 1:
        raise ValueError(
            "STATUS PLANT-FIRED: coupled ocean.output does not resolve one "
            "executed ln_drgice_imp value"
        )
    parsed["ln_drgice_imp"] = resolved.pop()
    return parsed


def _admitted_resolved_namelist(evidence: Path) -> dict:
    """Read the executed C3 configuration and refuse non-ICE5 evidence."""
    namelist = _coupled_resolved_namelist(evidence / "REC_C3")
    dimensions = tuple(int(namelist[name]) for name in ("jpl", "nlay_i", "nlay_s"))
    if dimensions != ADMITTED_ICE_DIMENSIONS:
        raise ValueError(
            "STATUS PLANT-FIRED: resolved ice dimensions "
            f"{dimensions} are inadmissible; expected ICE5 {ADMITTED_ICE_DIMENSIONS}"
        )
    return namelist


def _interior_master_fields(record: dict) -> dict[str, np.ndarray]:
    """Crop full local-domain fields to the MPI shard's recorded interior."""
    meta = record["meta"]
    sl = (
        slice(meta["ntsi"] - 1, meta["ntei"]),
        slice(meta["ntsj"] - 1, meta["ntej"]),
    )
    full_shape = (meta["jpi"], meta["jpj"])
    return {
        name: value[sl] if value.ndim >= 2 and value.shape[:2] == full_shape else value
        for name, value in record["fields"].items()
    }


def _nemo_cadenced_tau_step(
    inputs: SI3StressInputs,
    *,
    config: SI3ORCA2StateConfig,
    ln_drgice_imp: bool,
    nn_fsbc: int,
    kt: int,
) -> SI3StressOutputs:
    """Use the production cadence arm already present on the main line."""
    return si3_ice_update_tau(
        inputs,
        config=config,
        ln_drgice_imp=ln_drgice_imp,
        nn_fsbc=nn_fsbc,
        kt=kt,
    )


def score_c0(records: Path, *, plant: str | None = None) -> dict:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    entries = sorted(records.glob("c_kind1_*.bin"))
    if not entries:
        raise ValueError("STATUS PLANT-FIRED: no C0 entry records")
    for entry_path in entries:
        entry = read_c_record(entry_path)
        exit_path = records / entry_path.name.replace("c_kind1_", "c_kind2_")
        exit_record = read_c_record(exit_path)
        f = entry["fields"]
        prior = NemoSurfaceMeans(
            *(
                jnp.asarray(f[f"{name}_prior"])
                for name in ("ssu", "ssv", "sst", "sss", "ssh", "e3t", "frq")
            )
        )
        got = nemo_sbc_ssm_step(
            prior,
            kt=entry["kt"],
            nit000=1,
            forcing_stride=2,
            restart_means_present=bool(f["restart_means_present"]),
            use_conservative_temperature=bool(f["use_conservative_temperature"]),
            remove_inverse_barometer=bool(f["remove_inverse_barometer"]),
            u_before=f["u_before"],
            v_before=f["v_before"],
            temperature_now=f["temperature_now"],
            salinity_now=f["salinity_now"],
            ssh_now=f["ssh_now"],
            # The production helper does not use this argument when the recorded
            # flag is false; reuse an existing same-shape field without inventing
            # data that the allocation-off NEMO run did not emit.
            ssh_inverse_barometer=f.get("ssh_inverse_barometer", f["ssh_now"]),
            e3t_reference=f["e3t_reference"],
            r3t_now=f["r3t_now"],
            surface_mask=f["surface_mask"],
            shortwave_fraction_top=f["shortwave_fraction_top"],
        )
        expected = dict(exit_record["fields"])
        if plant == "numeric" and entry_path == entries[0]:
            expected["ssu"] = expected["ssu"].copy()
            expected["ssu"].flat[0] = np.nextafter(expected["ssu"].flat[0], np.inf)
        for name, value in zip(NemoSurfaceMeans._fields, got, strict=True):
            _equal(f"C0 step={entry['kt']} rank={entry['rank']} {name}", value, expected[name])
    return {"status": "PASS_C0_0_ULP", "records": len(entries)}


def score_c1(evidence: Path, *, plant: str | None = None) -> dict:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    run = evidence / "REC_C3"
    records = run / "o8_o16_master_records"
    namelist = _admitted_resolved_namelist(evidence)
    ice_dimensions = tuple(int(namelist[name]) for name in ("jpl", "nlay_i", "nlay_s"))
    first = read_record_verified(
        records / record_name(15, 1, 0, 0),
        expected_ice_dimensions=ice_dimensions,
    )
    cfg = _late_config(
        namelist,
        first["scalars"],
        ice_dimensions=ice_dimensions,
        nn_hls=2,
    )._replace(nn_fsbc=2)
    flux_cfg = cfg._replace(nn_hls=0)
    surface_config = SI3SurfaceConfig.from_resolved_namelists(
        run / "output.namelist.dyn", run / "output.namelist.ice"
    )._replace(switches=SI3_HARD_SWITCHES)
    steps = sorted(
        {int(path.name.split("_step")[1][:4]) for path in records.glob("o8o16_kind15_*.bin")}
    )
    for kt in steps:
        for rank in (0, 1):
            r15 = read_record_verified(
                records / record_name(15, kt, 0, rank),
                expected_ice_dimensions=ice_dimensions,
            )
            r16 = read_record_verified(
                records / record_name(16, kt, 0, rank),
                expected_ice_dimensions=ice_dimensions,
            )
            state_fields = dict(r15["fields"])
            sample = state_fields["a_i"]
            template = SI3State.zeros(
                sample.shape[0],
                sample.shape[1],
                config=flux_cfg,
            )
            state_fields["hi_max"] = np.asarray(template.bounds.hi_max)
            state_fields["hi_mean"] = np.asarray(template.bounds.hi_mean)
            state = _state_from_record(state_fields, flux_cfg, full_orca2=False)
            flux = _fluxes_from_record(r15["fields"])
            _, out = si3_ice_update_flx(
                state,
                flux,
                _update_forcing(_interior_master_fields(r15)),
                config=flux_cfg,
                surface_config=surface_config,
            )
            expected = _interior_master_fields(r16)
            for name in out._fields:
                _equal(f"C1 flux step={kt} rank={rank} {name}", getattr(out, name), expected[name])
        # Flux records occur only on SI3 calls; stress records occur every ocean step.
    tau_steps = sorted(
        {int(path.name.split("_step")[1][:4]) for path in records.glob("o8o16_kind19_*.bin")}
    )
    for kt in tau_steps:
        for rank in (0, 1):
            r19 = read_record_verified(
                records / record_name(19, kt, 0, rank),
                expected_ice_dimensions=ice_dimensions,
            )
            r20 = read_record_verified(
                records / record_name(20, kt, 0, rank),
                expected_ice_dimensions=ice_dimensions,
            )
            private_record = read_o17_record(
                evidence
                / "REC_C3/o17_master_records"
                / f"o17_kind34_tau_private_step{kt:04d}_rank{rank:04d}.bin"
            )
            private_dimensions = tuple(
                int(private_record["meta"][name]) for name in ("jpl", "nlay_i", "nlay_s")
            )
            if private_dimensions != ice_dimensions:
                raise ValueError(
                    "STATUS PLANT-FIRED: O17 record ice dimensions "
                    f"{private_dimensions} disagree with resolved run {ice_dimensions}"
                )
            c4 = read_c_record(
                evidence
                / "REC_C3/c_ladder_records"
                / f"c_kind4_step{kt:04d}_rank{rank:04d}.bin"
            )["fields"]
            stress = SI3StressInputs(
                **{
                    name: jnp.asarray(r19["fields"][name])
                    for name in SI3StressInputs._fields
                    if name != "tmask_k1"
                },
                tmask_k1=jnp.asarray(c4["tmask"]),
            )
            scored_kt = kt
            if plant == "wrong-parity":
                scored_kt += 1
            got = _nemo_cadenced_tau_step(
                stress,
                config=cfg,
                ln_drgice_imp=bool(namelist["ln_drgice_imp"]),
                nn_fsbc=cfg.nn_fsbc,
                kt=scored_kt,
            )
            for name in ("utau", "vtau", "taum"):
                _equal(
                    f"C1 tau step={kt} rank={rank} {name}", getattr(got, name), r20["fields"][name]
                )
            private = private_record["fields"]
            for name in ("tmod_io", "utau_oce", "vtau_oce"):
                _equal(
                    f"C1 tau private step={kt} rank={rank} {name}",
                    getattr(got, name),
                    private[name],
                )
            meta = r20["meta"]
            i0, i1 = meta["ntsi"] - 1, meta["ntei"]
            j0, j1 = meta["ntsj"] - 1, meta["ntej"]
            got_u, got_v = nemo_t_stress_to_uv(
                c4["utau"][i0:i1, j0:j1],
                c4["vtau"][i0:i1, j0:j1],
                c4["utau"][i0 + 1 : i1 + 1, j0:j1],
                c4["vtau"][i0:i1, j0 + 1 : j1 + 1],
                c4["tmask"][i0:i1, j0:j1],
                c4["tmask"][i0 + 1 : i1 + 1, j0:j1],
                c4["tmask"][i0:i1, j0 + 1 : j1 + 1],
                c4["umask"][i0:i1, j0:j1],
                c4["vmask"][i0:i1, j0:j1],
            )
            _equal(
                f"C1 final native U stress step={kt} rank={rank}", got_u, c4["utauU"][i0:i1, j0:j1]
            )
            _equal(
                f"C1 final native V stress step={kt} rank={rank}", got_v, c4["vtauV"][i0:i1, j0:j1]
            )
    return {"status": "PASS_C1_0_ULP", "flux_steps": len(steps), "tau_steps": len(tau_steps)}


def status_report() -> dict:
    return {"C3": UNSCORABLE, "C4": UNSCORABLE}


def self_test(plant: str | None, scratch: Path) -> dict:
    if plant is None:
        valid = scratch / "c_kind1_step0001_rank0000.bin"
        legacy = scratch / "old_unconditional_kind1.bin"
        _write_stub_kind1_record(
            valid, remove_inverse_barometer=0, old_unconditional_layout=False
        )
        record = read_c_record(valid)
        if record["nfields"] != 19 or "ssh_inverse_barometer" in record["fields"]:
            raise ValueError("allocation-off kind-1 stub did not use the 19-field layout")
        _write_stub_kind1_record(
            legacy, remove_inverse_barometer=0, old_unconditional_layout=True
        )
        try:
            read_c_record(legacy)
        except ValueError as exc:
            refusal = str(exc)
            if "remove_inverse_barometer=0 requires nfields=19, got 20" not in refusal:
                raise
        else:
            raise ValueError("old unconditional kind-1 layout was accepted")
        return {
            "status": "PASS_C_SCORER_SELF_TEST",
            "kind1_ln_apr_dyn_off": "PASS_19_FIELDS_NO_SSH",
            "old_unconditional_kind1_plant": f"REFUSED: {refusal}",
            **status_report(),
        }
    if plant == "numeric":
        _equal("numeric plant", np.array([1.0]), np.array([np.nextafter(1.0, np.inf)]))
    if plant == "truncation":
        path = scratch / "truncated.bin"
        path.write_bytes(MAGIC)
        read_c_record(path)
    if plant == "wrong-parity":
        expected = tuple((kt - 1) % 2 == 0 for kt in range(1, 5))
        planted = tuple(not value for value in expected)
        if planted != expected:
            raise ValueError("STATUS PLANT-FIRED: SI3 cadence parity/time level")
    return {"status": "PASS_C_SCORER_SELF_TEST", **status_report()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--ppsrc", type=Path)
    parser.add_argument("--plant", choices=("numeric", "wrong-parity"))
    parser.add_argument(
        "--self-test", choices=("baseline", "numeric", "truncation", "wrong-parity")
    )
    parser.add_argument("--scratch", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.self_test:
        print(
            json.dumps(
                self_test(None if args.self_test == "baseline" else args.self_test, args.scratch),
                indent=2,
            )
        )
        return 0
    if args.evidence is None or args.ppsrc is None:
        parser.error("--evidence and --ppsrc are required outside --self-test")
    _admitted_resolved_namelist(args.evidence)
    report = {
        "C0": score_c0(
            args.evidence / "REC_C3/c_ladder_records",
            plant=args.plant if args.plant == "numeric" else None,
        ),
        "C1": score_c1(args.evidence, plant=args.plant),
        "C2": run_c2_compiled(args.evidence),
        **status_report(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
