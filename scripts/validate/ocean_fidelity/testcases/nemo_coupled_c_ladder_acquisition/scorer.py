#!/usr/bin/env python3
"""Fail-closed C0/C1 scorer and D225/D226 compiled C2/C3 comparisons."""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import struct
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.core.nemo_namelist import parse_namelist
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
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
COUPLED_MATH_CHOICES = ("libm-cpu", "fast-literal")
C2_VERDICT_KINDS = (1, 2, 3, 4, 5, 6, 15, 16, 19, 20, 34)
C2_REPORT_ONLY_KINDS = ()
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


def configure_coupled_math(math_mode: str) -> None:
    """Select exact libm or round-344's fp64/native fast-literal arm."""

    if math_mode == "libm-cpu":
        if jax.default_backend() != "cpu":
            raise RuntimeError("libm-cpu math requires the CPU backend")
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        return
    if math_mode == "fast-literal":
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        return
    raise ValueError(
        f"coupled math must be one of {COUPLED_MATH_CHOICES}, got {math_mode!r}"
    )


@contextlib.contextmanager
def coupled_math_scope(math_mode: str):
    """Hold the explicit coupled policy over the outermost JIT boundary."""

    previous = get_policy()
    configure_coupled_math(math_mode)
    try:
        yield
    finally:
        set_policy(previous)


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
    absolute = np.abs(got - want)
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
    wet_values = int(np.count_nonzero(wet))
    return {
        "values": int(got.size),
        "wet_values": wet_values,
        "dry_values": int(got.size) - wet_values,
        "wet_different": int(np.count_nonzero(wet_bad)),
        "dry_zero_sign_different": int(np.count_nonzero(dry_sign_zero)),
        "dry_other_different": int(np.count_nonzero(dry_bad)),
        "wet_max_abs_difference": float(np.max(absolute[wet_bad], initial=0.0)),
        "dry_max_abs_difference": float(np.max(absolute[dry_bad], initial=0.0)),
        "first": first,
    }


def _merge_d225_counts(total: dict, row: dict) -> None:
    for name in (
        "values", "wet_values", "dry_values", "wet_different",
        "dry_zero_sign_different", "dry_other_different",
    ):
        total[name] = total.get(name, 0) + row[name]


def _merge_d225_field_counts(total: dict, row: dict) -> None:
    _merge_d225_counts(total, row)
    for name in ("wet_max_abs_difference", "dry_max_abs_difference"):
        total[name] = max(total.get(name, 0.0), row[name])


def _score_c_observations(
    candidate: dict[tuple[int, int, int], dict[str, object]],
    expected: dict[tuple[int, int, int], dict[str, object]],
    masks: dict[tuple[int, int], dict[str, np.ndarray]],
    *,
    steps: tuple[int, ...],
    mode: str,
) -> dict:
    """Compare compiled recipe observations; perform no physical arithmetic.

    Keys are ``(kind, kt, rank)``.  The runner constructs ``candidate`` from
    the two jitted recipe traces; this function only compares and reports.
    """
    required = {
        (kind, kt, rank)
        for kind in (*C2_REPORT_ONLY_KINDS, *C2_VERDICT_KINDS)
        for kt in steps
        for rank in (0, 1)
        if not (kind in (15, 16) and kt % 2 == 0)
    }
    missing = sorted(required - set(candidate))
    if missing:
        raise ValueError(f"{mode} candidate observations incomplete; missing={missing[:8]}")
    missing_expected = sorted(required - set(expected))
    if missing_expected:
        raise ValueError(f"{mode} NEMO observations incomplete; missing={missing_expected[:8]}")

    counts: dict[int, dict] = {}
    report_only: dict[int, dict] = {}
    field_counts: dict[tuple[int, int, str], dict] = {}
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
                f"{mode} kind={kind} kt={kt} rank={rank} field set mismatch: "
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
                        f"{mode} integer metadata differs kind={kind} field={field} kt={kt} rank={rank}"
                    )
                continue
            if field in {
                "uu", "ssu", "ssu_prior", "utauU", "uu_kbb", "u_before",
                "u_ice", "u_oce",
            }:
                staggering = "umask"
            elif field in {
                "vv", "ssv", "ssv_prior", "vtauV", "vv_kbb", "v_before",
                "v_ice", "v_oce",
            }:
                staggering = "vmask"
            else:
                staggering = "tmask"
            mask_name = f"{staggering}_3d" if got.ndim >= 3 and field in {"ts", "uu", "vv"} else staggering
            row = _d225_field(got, want_fields[field], mask_names[mask_name] > 0.5)
            _merge_d225_counts(kind_total, row)
            _merge_d225_field_counts(
                field_counts.setdefault((kind, kt, field), {}), row
            )
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
        status = f"PASS_{mode}_0_ULP_WET"
    else:
        f = first_failure
        status = (
            f"FAIL_{mode} step={f['kt']} kind={f['kind']} field={f['field']} "
            f"cell={f['cell']} "
            f"rank={f['rank']} got={f['got_bits']} "
            f"expected={f['nemo_bits']} domain={f['domain']}"
        )
    return {
        "status": status,
        "compiled": True,
        "verdict_kinds": list(C2_VERDICT_KINDS),
        "report_only_kinds": list(C2_REPORT_ONLY_KINDS),
        "counts": counts,
        "report_only_counts": report_only,
        "field_counts": [
            {"kind": kind, "kt": kt, "field": field, **row}
            for (kind, kt, field), row in sorted(
                field_counts.items(),
                key=lambda item: (
                    item[0][1], order_index[item[0][0]], item[0][2]
                ),
            )
        ],
        "first_failure": first_failure,
    }


def score_c2_observations(
    candidate: dict[tuple[int, int, int], dict[str, object]],
    expected: dict[tuple[int, int, int], dict[str, object]],
    masks: dict[tuple[int, int], dict[str, np.ndarray]],
) -> dict:
    """Compare the compiled two-step recipe under D225/D226/D230."""

    return _score_c_observations(
        candidate, expected, masks, steps=(1, 2), mode="C2"
    )


def score_c3_pair_observations(
    candidate: dict[tuple[int, int, int], dict[str, object]],
    expected: dict[tuple[int, int, int], dict[str, object]],
    masks: dict[tuple[int, int], dict[str, np.ndarray]],
    *,
    steps: tuple[int, int],
) -> dict:
    """Public pair scorer shared by the compiled C3 route and its controls."""

    return _score_c_observations(
        candidate, expected, masks, steps=steps, mode="C3"
    )


def _load_expected_steps(
    run: Path, steps: tuple[int, ...], dimensions: tuple[int, int, int]
) -> tuple[dict, dict]:
    """Load selected NEMO boundaries and masks without model arithmetic."""

    c_root = run / "c_ladder_records"
    master_root = run / "o8_o16_master_records"
    o17_root = run / "o17_master_records"
    expected: dict[tuple[int, int, int], dict[str, object]] = {}
    masks: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    import netCDF4

    mask_3d_by_rank = {}
    for rank in (0, 1):
        with netCDF4.Dataset(run / f"mesh_mask_{rank:04d}.nc") as dataset:
            mask_3d_by_rank[rank] = {
                f"{name}_3d": np.asarray(
                    dataset.variables[name][0], dtype=np.float64
                ).transpose(2, 1, 0)
                for name in ("tmask", "umask", "vmask")
            }
    for kt in steps:
        for rank in (0, 1):
            for kind in (1, 2, 3, 4, 5, 6):
                record = read_c_record(
                    c_root / f"c_kind{kind}_step{kt:04d}_rank{rank:04d}.bin"
                )
                expected[(kind, kt, rank)] = record["fields"]
                if kind == 4:
                    masks[(kt, rank)] = {
                        name: record["fields"][name]
                        for name in ("tmask", "umask", "vmask")
                    }
                    masks[(kt, rank)].update(mask_3d_by_rank[rank])
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
    return _load_expected_steps(run, (1, 2), dimensions)


def score_c2(evidence: Path, candidate: dict[tuple[int, int, int], dict]) -> dict:
    """Score observations emitted by one externally jitted coupled-pair run."""
    expected, masks = load_c2_expected(evidence)
    return score_c2_observations(candidate, expected, masks)


def _c2_cold_start(run: Path, *, n_steps: int, math_mode: str):
    """Load records and bind them to the production coupled recipe."""
    with coupled_math_scope(math_mode):
        return _c2_cold_start_selected_math(run, n_steps=n_steps, math_mode=math_mode)


def _c2_cold_start_selected_math(run: Path, *, n_steps: int, math_mode: str):
    """Build the coupled recipe while the caller-selected policy is active."""
    from legoesm.ocean import bulk_flux_omip as bulk
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round340_score_composed_card as r340_score,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round340_standalone_gate as r340,
    )
    from scripts.validate.ocean_fidelity.testcases import (
        nemo_si3_o17_composition_scorer as o17_score,
    )

    k0 = bulk.build_nemo_orca2_k0_ocean_card(run)
    controls = r340.build_real_composed_inputs(
        k0, run, r340_score.DEFAULT_CORE_MANIFEST, r340_score.DEFAULT_ICE_COVER,
        n_steps=n_steps,
    )
    receipt = json.loads((run / "C_DECK_RECEIPT.json").read_text())
    master = Path(receipt["sas_ice_reference"]).parents[1]
    master_run = master / "record_100step"
    ln_teos10 = o17_score.si3_teos10_from_resolved_namelist(
        run / "output.namelist.dyn"
    )
    coupled_ocean = build_orca2_zps_card(run, initial_state="si3_iceistate")
    ice_card = o17_score.card_for(master_run)._replace(
        ln_drgice_imp=bool(_coupled_resolved_namelist(run)["ln_drgice_imp"]),
        ln_teos10=ln_teos10,
    )
    result = bulk.build_nemo_orca2_si3_coupled_recipe(
        run,
        coupled_ocean=coupled_ocean,
        k0=k0,
        controls=controls,
        ice_card=ice_card,
    )
    if math_mode == "fast-literal":
        carry, forcing, sponge, card, model = result
        # Round 344's GPU arm changes only this selector.  It deliberately does
        # not enter ``nemo_fast``, whose source-rounding removal is a different
        # numerical experiment.
        card = card._replace(
            ocean=card.ocean._replace(transcendentals="native")
        )
        result = carry, forcing, sponge, card, model
    return result


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
        out.update({
            name: getattr(core.correction.fluxes, name)
            for name in core.correction.fluxes._fields
        })
        out.update({name: getattr(core.update_forcing, name) for name in core.update_forcing._fields})
        out["tmask_k1"] = core.update_forcing.tmask
        out["snwice_mass"] = core.correction.state.snwice_mass
        out["snwice_mass_b"] = core.correction.state.snwice_mass_b
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
    if value.shape[:2] == (184, 152) and shape[:2] == (94, 152):
        start = rank * (shape[0] - 4)
        result = value[start:start + shape[0]]
        if result.shape != shape:
            raise ValueError(f"C2 full-halo shard shape mismatch: {result.shape} != {shape}")
        return result
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


def _trace_observations(traces, card, expected: dict, steps: tuple[int, ...]) -> dict:
    """Adapt one compiled pair trace to record names/indexing, then discard it."""

    candidate: dict[tuple[int, int, int], dict[str, object]] = {}
    for kt, trace in zip(steps, traces, strict=True):
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
    return candidate


def _validate_sponge_steps(sponge, first_kt: int) -> None:
    """Host-side stale-target guard before a carried counter becomes traced."""

    first_scalar = np.asarray(first_kt)
    if (first_scalar.shape != ()
            or first_scalar.dtype.kind not in "iu"
            or int(first_scalar) < 1
            or int(first_scalar) % 2 != 1):
        raise ValueError(
            "coupled pair requires an odd positive integer entry kt"
        )
    first = int(first_scalar)
    for offset, target in enumerate(sponge):
        stamped = None if target is None else target.nemo_kt
        stamped_scalar = None if stamped is None else np.asarray(stamped)
        expected = first + offset
        if (stamped_scalar is None
                or stamped_scalar.shape != ()
                or stamped_scalar.dtype.kind not in "iu"
                or int(stamped_scalar) < 1
                or int(stamped_scalar) != expected):
            raise ValueError(
                f"L6 damping target is stale: target kt={stamped!r}, "
                f"requested kt={expected!r}"
            )


def _run_c2_compiled_observations(evidence: Path, *, math_mode: str):
    """Run the two-step cold C2 trajectory and return scorer-ready observations."""
    from legoesm.ocean import bulk_flux_omip as bulk

    expected, masks = load_c2_expected(evidence)
    carry, forcing, sponge, card, model = _c2_cold_start(
        evidence / "REC_C2", n_steps=2, math_mode=math_mode
    )
    _validate_sponge_steps(sponge, 1)

    @jax.jit
    def compiled_pair(entry, odd_forcing, even_forcing, odd_sponge, even_sponge):
        return bulk.nemo_orca2_si3_coupled_pair(
            entry,
            odd_forcing,
            card,
            even_forcing=even_forcing,
            model=model,
            sponge=odd_sponge,
            even_sponge=even_sponge,
        )

    final, traces = compiled_pair(carry, forcing[0], forcing[1], sponge[0], sponge[1])
    jax.block_until_ready(final)
    candidate = _trace_observations(traces, card, expected, (1, 2))
    return candidate, expected, masks


def run_c2_compiled_observations(evidence: Path, *, math_mode: str):
    """Run C2 under explicit exact-CPU or round-344 fast-literal math."""

    with coupled_math_scope(math_mode):
        return _run_c2_compiled_observations(evidence, math_mode=math_mode)


def run_c2_compiled(evidence: Path, *, math_mode: str) -> dict:
    """Run the two-step cold C2 trajectory once under ``jax.jit`` and score it."""
    candidate, expected, masks = run_c2_compiled_observations(
        evidence, math_mode=math_mode
    )
    report = score_c2_observations(candidate, expected, masks)
    report["math"] = math_mode
    return report


def _run_c3_compiled(evidence: Path, *, pair_limit: int, math_mode: str) -> dict:
    """Compile one traced pair once, carry it forward, and score pairwise.

    ``pair_limit`` exists only for the required preflight timing measurement.
    Anything shorter than 50 pairs is explicitly non-verdict-bearing.
    """

    from legoesm.ocean import bulk_flux_omip as bulk

    run = evidence / "REC_C3"
    namelist = _admitted_resolved_namelist(evidence)
    dimensions = tuple(int(namelist[name]) for name in ("jpl", "nlay_i", "nlay_s"))
    if type(pair_limit) is not int or not 1 <= pair_limit <= 50:
        raise ValueError("C3 pair_limit must be an integer in [1, 50]")
    n_steps = 2 * pair_limit
    carry, forcing, sponge, card, model = _c2_cold_start(
        run, n_steps=n_steps, math_mode=math_mode
    )
    if len(forcing) != n_steps or len(sponge) != n_steps:
        raise ValueError(
            f"C3 deck readers produced {len(forcing)} forcing and {len(sponge)} "
            f"damping records; expected {n_steps} each"
        )
    _validate_sponge_steps(sponge, 1)

    compile_count = 0

    def pair_program(entry, odd_forcing, even_forcing, odd_sponge, even_sponge):
        nonlocal compile_count
        compile_count += 1
        return bulk.nemo_orca2_si3_coupled_pair(
            entry,
            odd_forcing,
            card,
            even_forcing=even_forcing,
            model=model,
            sponge=odd_sponge,
            even_sponge=even_sponge,
        )

    compiled_pair = jax.jit(pair_program)
    compile_start = time.perf_counter()
    executable = compiled_pair.lower(
        carry, forcing[0], forcing[1], sponge[0], sponge[1]
    ).compile()
    executable_calls = 0
    compile_seconds = time.perf_counter() - compile_start
    if compile_count != 1:
        raise ValueError(f"C3 pair compiled {compile_count} times; expected exactly 1")

    counts: dict[int, dict] = {}
    step_counts: dict[tuple[int, int], dict] = {}
    field_counts: list[dict] = []
    first_failure = None
    pair_wall_seconds = []
    for pair_index in range(pair_limit):
        odd_kt = 2 * pair_index + 1
        steps = (odd_kt, odd_kt + 1)
        started = time.perf_counter()
        next_carry, traces = executable(
            carry,
            forcing[odd_kt - 1],
            forcing[odd_kt],
            sponge[odd_kt - 1],
            sponge[odd_kt],
        )
        executable_calls += 1
        jax.block_until_ready(next_carry)
        pair_wall_seconds.append(time.perf_counter() - started)
        expected, masks = _load_expected_steps(run, steps, dimensions)
        candidate = _trace_observations(traces, card, expected, steps)
        pair_report = score_c3_pair_observations(
            candidate, expected, masks, steps=steps
        )
        for kind, row in pair_report["counts"].items():
            _merge_d225_counts(counts.setdefault(kind, {}), row)
        for row in pair_report["field_counts"]:
            field_counts.append(row)
            _merge_d225_field_counts(
                step_counts.setdefault((row["kind"], row["kt"]), {}), row
            )
        if first_failure is None and pair_report["first_failure"] is not None:
            first_failure = pair_report["first_failure"]
        carry = next_carry
        if first_failure is not None:
            # The first bad pair is already fully scored and reported.  Stop
            # before executing later physics so localization remains anchored
            # to the earliest NEMO-order boundary.
            break

    if compile_count != 1:
        raise ValueError(f"C3 pair retraced {compile_count} times; expected exactly 1")
    if first_failure is None and pair_limit == 50:
        status = "PASS_C3_0_ULP_WET"
    elif first_failure is None:
        status = f"PROFILE_C3_{pair_limit}_PAIRS_NO_VERDICT"
    else:
        f = first_failure
        status = (
            f"FAIL_C3 step={f['kt']} kind={f['kind']} field={f['field']} "
            f"cell={f['cell']} rank={f['rank']} got={f['got_bits']} "
            f"expected={f['nemo_bits']} domain={f['domain']}"
        )
    return {
        "math": math_mode,
        "status": status,
        "compiled": True,
        "compile_count": compile_count,
        "compiled_executable_calls": executable_calls,
        "pairs_scored": len(pair_wall_seconds),
        "compile_seconds": compile_seconds,
        "pair_wall_seconds": pair_wall_seconds,
        "mean_pair_wall_seconds": float(np.mean(pair_wall_seconds)),
        "counts": counts,
        "step_counts": [
            {"kind": kind, "kt": kt, **row}
            for (kind, kt), row in sorted(
                step_counts.items(), key=lambda item: (item[0][1], item[0][0])
            )
        ],
        "field_counts": field_counts,
        "first_failure": first_failure,
        "input_readers": {
            "atmosphere_CORE": "load_core_normal_year + nemo_fld_interp",
            "runoff_and_sss_target": "_monthly_slots + nemo_fld_time_interp",
            "chlorophyll": "chlorophyll_at_step",
            "geothermal_heat_flux": "read_geothermal_heatflow",
            "tracer_damping_target": "build_orca2_damping_for_step",
            "ocean_initial_state_and_static_grid": "build_orca2_zps_card",
            "SI3_initial_state_and_static_grid": "build_nemo_orca2_si3_static",
        },
    }


def run_c3_compiled(
    evidence: Path, *, pair_limit: int, math_mode: str
) -> dict:
    """Run C3 under explicit exact-CPU or round-344 fast-literal math."""

    with coupled_math_scope(math_mode):
        return _run_c3_compiled(
            evidence, pair_limit=pair_limit, math_mode=math_mode
        )


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


def score_c0(
    records: Path, *, math_mode: str, plant: str | None = None
) -> dict:
    configure_coupled_math(math_mode)
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


def score_c1(
    evidence: Path, *, math_mode: str, plant: str | None = None
) -> dict:
    configure_coupled_math(math_mode)
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
    return {"C4": UNSCORABLE}


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
    parser.add_argument("--mode", choices=("C0-C2", "C3"))
    parser.add_argument("--math", choices=COUPLED_MATH_CHOICES)
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
    if args.mode is None:
        parser.error("--mode C0-C2 or --mode C3 is required outside --self-test")
    if args.math is None:
        parser.error("--math libm-cpu or --math fast-literal is required outside --self-test")
    if args.mode == "C0-C2" and args.math == "fast-literal":
        parser.error(
            "--mode C0-C2 --math fast-literal is refused: C1 alb_ice is not "
            "0 ULP; the exactness bar is unchanged"
        )
    _admitted_resolved_namelist(args.evidence)
    if args.mode == "C3":
        report = {
            "math": args.math,
            "C3": run_c3_compiled(
                args.evidence, pair_limit=50, math_mode=args.math
            ),
            **status_report(),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2 if report["C3"]["status"].startswith("FAIL_C3") else 0
    report = {
        "math": args.math,
        "C0": score_c0(
            args.evidence / "REC_C3/c_ladder_records",
            math_mode=args.math,
            plant=args.plant if args.plant == "numeric" else None,
        ),
        "C1": score_c1(args.evidence, math_mode=args.math, plant=args.plant),
        "C2": run_c2_compiled(args.evidence, math_mode=args.math),
        **status_report(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["C2"]["status"].startswith("FAIL_C2") else 0


if __name__ == "__main__":
    raise SystemExit(main())
