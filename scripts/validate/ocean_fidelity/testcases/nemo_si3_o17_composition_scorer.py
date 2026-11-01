#!/usr/bin/env python3
"""Score unified O17 records by executing the production O17 entry."""

from __future__ import annotations

import argparse
import json
import resource
import struct
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.dynamics import SI3CGridAEVPForcing, SI3CGridAEVPState, SI3CGridMetrics
from legoesm.ice.si3_finalize import SI3StressInputs
from legoesm.ice.si3_o17 import (
    SI3O17LiveForcing,
    SI3O17StepState,
    SI3O17SurfaceForcing,
    build_orca2_sas_ice_o17_card,
    si3_o17_step,
)
from legoesm.ice.si3_orca2_state import (
    SI3AllPratherMoments,
    SI3CategoryBounds,
    SI3ORCA2AdvectionGrid,
    SI3PratherMoments,
    SI3State,
    si3_var_agg,
    si3_var_glo2eqv,
)
from legoesm.ice.si3_surface import SI3SurfaceConfig
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o8_o12_master_scorer import (
    DEFAULT_PPSRC,
    parse_resolved_ice_namelist,
)

MAGIC = b"O17S0001"
VERSION = 1
VALID_LABELS = ("INDEPENDENT", "GIVEN-NEMO-ENTRY")
KINDS = {
    22: "step_entry", 23: "after_rheology", 24: "after_advection",
    25: "after_rdgrft", 26: "after_dyn_cor", 27: "after_thermo",
    28: "step_exit", 29: "prather_entry", 30: "prather_exit",
    31: "after_flx_other", 32: "raw_blk1", 33: "raw_blk2", 34: "tau_private_exit",
}
STATE_FIELDS = (
    "v_i", "v_s", "sv_i", "a_i", "t_su", "u_ice", "v_ice", "oa_i",
    "a_ip", "v_ip", "v_il", "e_s", "e_i", "szv_i", "t_si",
    "snwice_mass", "snwice_mass_b", "snwice_fmass",
)
FLUX_FIELDS = (
    "wfx_snw", "wfx_snw_sni", "wfx_snw_sum", "wfx_pnd", "wfx_spr", "wfx_sub",
    "wfx_snw_sub", "wfx_ice_sub", "wfx_ice", "wfx_sni", "wfx_opw", "wfx_bog",
    "wfx_dyn", "wfx_bom", "wfx_sum", "wfx_lam", "wfx_res", "wfx_err_sub",
    "sfx_bog", "sfx_bom", "sfx_lam", "sfx_sum", "sfx_sni", "sfx_opw", "sfx_bri",
    "sfx_dyn", "sfx_res", "sfx_sub", "hfx_bog", "hfx_bom", "hfx_sum", "hfx_opw",
    "hfx_dif", "hfx_snw", "hfx_err_dif", "hfx_sub", "hfx_spr", "hfx_thd",
    "hfx_dyn", "hfx_res", "qlead", "qsb_ice_bot", "fhld",
)
PRATHER_PREFIX = {
    "ice": "ice", "sn": "sn", "a": "a", "age": "age", "sal": "sal",
    "ap": "ap", "vp": "vp", "vl": "vl", "c0": "c0", "e": "e", "si": "si",
}


def read_record(path: Path) -> dict[str, Any]:
    """Read a self-describing stream record; no halo extent is inferred."""

    with path.open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError(f"bad O17 magic: {path}")
        if struct.unpack("=i", stream.read(4))[0] != VERSION:
            raise ValueError(f"bad O17 version: {path}")
        meta = struct.unpack("=8i", stream.read(32))
        fields: dict[str, np.ndarray] = {}
        while True:
            raw_name = stream.read(32)
            if len(raw_name) != 32:
                raise ValueError(f"truncated field header: {path}")
            name = raw_name.decode("ascii").rstrip(" \x00")
            rank = struct.unpack("=i", stream.read(4))[0]
            dims = struct.unpack("=4i", stream.read(16))
            if name == "__END__" and rank == -1:
                break
            if name in fields or rank not in range(5):
                raise ValueError(f"bad/duplicate O17 field {name!r}: {path}")
            shape = tuple(dims[:rank])
            count = int(np.prod(shape, dtype=np.int64)) if rank else 1
            raw = stream.read(count * 8)
            if len(raw) != count * 8:
                raise ValueError(f"truncated O17 field {name}: {path}")
            value = np.frombuffer(raw, dtype="=f8").copy()
            fields[name] = value.reshape(shape, order="F") if rank else value.reshape(())
        if stream.read(1):
            raise ValueError(f"trailing bytes after O17 terminator: {path}")
    kind, kt, rank_id, jpi, jpj, jpl, nlay_i, nlay_s = meta
    if kind not in KINDS or min(kt, jpi, jpj, jpl, nlay_i, nlay_s) < 1 or rank_id < 0:
        raise ValueError(f"invalid O17 metadata: {path}")
    return {
        "meta": {"kind": kind, "kt": kt, "rank": rank_id, "jpi": jpi, "jpj": jpj,
                 "jpl": jpl, "nlay_i": nlay_i, "nlay_s": nlay_s},
        "fields": fields,
    }


def _record_index(root: Path) -> dict[tuple[int, int, int], dict[str, Any]]:
    index: dict[tuple[int, int, int], dict[str, Any]] = {}
    for path in sorted(root.glob("o17_kind*.bin")):
        record = read_record(path)
        meta = record["meta"]
        key = meta["kind"], meta["kt"], meta["rank"]
        if key in index:
            raise ValueError(f"duplicate O17 record {key}")
        index[key] = record
    if not index:
        raise ValueError(f"no O17 records in {root}")
    steps = sorted({key[1] for key in index})
    ranks = sorted({key[2] for key in index})
    expected = {(kind, kt, rank) for kind in KINDS for kt in steps for rank in ranks}
    if set(index) != expected:
        raise ValueError(f"O17 index incomplete; missing={sorted(expected-set(index))[:8]}")
    return index


def _full(value: np.ndarray, shape: tuple[int, int], h: int, fill: float = 0.0) -> jnp.ndarray:
    value = np.asarray(value)
    if value.shape[:2] == shape:
        return jnp.asarray(value)
    interior = shape[0] - 2 * h, shape[1] - 2 * h
    if value.shape[:2] != interior:
        raise ValueError(f"cannot place {value.shape} in halo shape {shape}")
    out = jnp.full(shape + value.shape[2:], fill, dtype=jnp.float64)
    return out.at[h:-h, h:-h, ...].set(value)


def _owned(value: np.ndarray, shape: tuple[int, int], h: int) -> jnp.ndarray:
    value = np.asarray(value)
    return jnp.asarray(value[h:-h, h:-h, ...] if value.shape[:2] == shape else value)


def _moments(fields: dict[str, np.ndarray]) -> SI3AllPratherMoments:
    def family(prefix: str) -> SI3PratherMoments:
        names = (f"sx{prefix}", f"sy{prefix}", f"sxx{prefix}", f"syy{prefix}", f"sxy{prefix}")
        return SI3PratherMoments(*(jnp.asarray(fields[name]) for name in names))
    return SI3AllPratherMoments(**{name: family(prefix) for name, prefix in PRATHER_PREFIX.items()})


def _entry_state(entry: dict[str, Any], prather: dict[str, Any], card) -> SI3O17StepState:
    f = entry["fields"]
    jpi, jpj = entry["meta"]["jpi"], entry["meta"]["jpj"]
    base = SI3State.zeros(jpi, jpj, card.state)
    eqv = si3_var_glo2eqv(
        v_i=jnp.asarray(f["v_i"]), v_s=jnp.asarray(f["v_s"]), sv_i=jnp.asarray(f["sv_i"]),
        a_i=jnp.asarray(f["a_i"]), oa_i=jnp.asarray(f["oa_i"]), a_ip=jnp.asarray(f["a_ip"]),
        v_ip=jnp.asarray(f["v_ip"]), v_il=jnp.asarray(f["v_il"]), e_s=jnp.asarray(f["e_s"]),
        e_i=jnp.asarray(f["e_i"]), szv_i=jnp.asarray(f["szv_i"]), kn=2,
        hi_max_top=float(np.asarray(f["hi_max"])[-1]), sss_m=jnp.asarray(f["sss_m"]), config=card.state,
    )
    agg = si3_var_agg(
        v_i=jnp.asarray(f["v_i"]), v_s=eqv.v_s, sv_i=jnp.asarray(f["sv_i"]), a_i=eqv.a_i,
        oa_i=jnp.asarray(f["oa_i"]), a_ip=jnp.asarray(f["a_ip"]), v_ip=jnp.asarray(f["v_ip"]),
        v_il=jnp.asarray(f["v_il"]), t_su=jnp.asarray(f["t_su"]), t_si=jnp.asarray(f["t_si"]),
        e_s=jnp.asarray(f["e_s"]), e_i=jnp.asarray(f["e_i"]), szv_i=jnp.asarray(f["szv_i"]),
        a_ip_eff=eqv.a_ip_eff, t_i=eqv.t_i, t_s=eqv.t_s, kn=2, config=card.state,
    )
    state = base._replace(
        **{name: jnp.asarray(f[name]) for name in STATE_FIELDS},
        prather=_moments(prather["fields"]), equivalent=eqv, aggregated=agg,
        bounds=SI3CategoryBounds(jnp.asarray(f["hi_max"]), jnp.asarray(f["hi_mean"])),
        cflice_delayed=(
            jnp.asarray(f["cflice_delayed"]) if "cflice_delayed" in f else None
        ),  # mpp_allreduce_generic.h90:147-152; unallocated at kt=1
    )
    aevp = SI3CGridAEVPState(
        state.u_ice, state.v_ice, jnp.asarray(f["stress1_i"]),
        jnp.asarray(f["stress2_i"]), jnp.asarray(f["stress12_i"]),
    )
    return SI3O17StepState(state, aevp)


def _forcing(
    entry: dict[str, Any], rheology: dict[str, Any], raw1: dict[str, Any], raw2: dict[str, Any], card
) -> SI3O17LiveForcing:
    f, masks, r1, r2 = entry["fields"], rheology["fields"], raw1["fields"], raw2["fields"]
    shape = entry["meta"]["jpi"], entry["meta"]["jpj"]
    h = int(card.state.nn_hls)
    metrics = SI3CGridMetrics(*(jnp.asarray(f[name]) for name in (
        "e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f",
        "e1e2t", "e1e2u", "e1e2v", "e1e2f",
    )))
    grid = SI3ORCA2AdvectionGrid(
        e1u=jnp.asarray(f["e1u"]), e2v=jnp.asarray(f["e2v"]),
        r1_e1u=jnp.asarray(f["r1_e1u"]), r1_e2v=jnp.asarray(f["r1_e2v"]),
        e2u=jnp.asarray(f["e2u"]), e1v=jnp.asarray(f["e1v"]), e1e2t=jnp.asarray(f["e1e2t"]),
        r1_e1e2t=jnp.asarray(f["r1_e1e2t"]), tmask=jnp.asarray(f["tmask"]),
        umask=jnp.asarray(f["umask"]), vmask=jnp.asarray(f["vmask"]),
    )
    zero = jnp.zeros(shape, jnp.float64)
    aevp_forcing = SI3CGridAEVPForcing(
        *(zero for _ in range(7)), jnp.asarray(f["drag_io"]), jnp.asarray(f["u_oce"]),
        jnp.asarray(f["v_oce"]), jnp.asarray(f["ssh_m"]), jnp.asarray(f["ff_t"]),
        jnp.asarray(f["tmask"]), jnp.asarray(f["umask"]), jnp.asarray(f["vmask"]),
        jnp.asarray(masks["fast_tmask"]), zero, zero, zero,
        jnp.asarray(masks["icb_tmask"]), jnp.asarray(masks["icb_umask"]),
        jnp.asarray(masks["icb_vmask"]),
    )
    surface = SI3O17SurfaceForcing(
        wndi=jnp.asarray(r1["wndi"]), wndj=jnp.asarray(r1["wndj"]),
        theta_air_zt=jnp.asarray(r1["tair"]), q_air_zt=jnp.asarray(r1["qair"]),
        slp=jnp.asarray(r1["slp"]), rhoa=jnp.asarray(r1["rhoa"]),
        cloud_fra=_owned(f["cloud_fra"], shape, h), smask0=jnp.asarray(r1["smask"]),
        qlw=jnp.asarray(r2["qlw"]), precip=jnp.asarray(r2["precip"]), snow=jnp.asarray(r2["snow"]),
        qsr=jnp.asarray(r2["qsr"]), emp=jnp.asarray(r2["emp"]),
        qns_oce=_owned(f["qns_oce"], shape, h), qsr_oce=_owned(f["qsr_oce"], shape, h),
        ssu_m=jnp.asarray(f["ssu_m"]), ssv_m=jnp.asarray(f["ssv_m"]),
        utau=_owned(f["utau"], shape, h), vtau=_owned(f["vtau"], shape, h),
        frq_m=_owned(f["frq_m"], shape, h), e3t_m=_owned(f["e3t_m"], shape, h),
    )
    stress = SI3StressInputs(
        uu_kbb=jnp.asarray(f["uu_kbb"]), vv_kbb=jnp.asarray(f["vv_kbb"]),
        umask_k1=jnp.asarray(f["umask"]), vmask_k1=jnp.asarray(f["vmask"]),
        u_ice=jnp.asarray(f["u_ice"]), v_ice=jnp.asarray(f["v_ice"]),
        u_oce=jnp.asarray(f["u_oce"]), v_oce=jnp.asarray(f["v_oce"]),
        drag_io=jnp.asarray(f["drag_io"]), at_i=zero,
        utau=_full(f["utau"], shape, h), vtau=_full(f["vtau"], shape, h), tmod_io=zero,
        taum=_owned(f["taum"], shape, h), utau_oce=_owned(f["utau"], shape, h),
        vtau_oce=_owned(f["vtau"], shape, h),
    )
    return SI3O17LiveForcing(
        kt=int(entry["meta"]["kt"]), sss_m=jnp.asarray(f["sss_m"]), sst_m=jnp.asarray(f["sst_m"]),
        rn_amax_2d=_full(f["rn_amax_2d"], shape, h, fill=1.0), advection_grid=grid,
        aevp_forcing=aevp_forcing, aevp_metrics=metrics, ssh_m=jnp.asarray(f["ssh_m"]),
        surface=surface, stress=stress,
    )


def _comparison(actual: Any, expected: Any, exact: bool) -> tuple[int, float]:
    a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if a.shape != b.shape:
        return max(a.size, b.size), float("inf")
    different = np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)) if exact else np.count_nonzero(a != b)
    scale = np.maximum(np.abs(b), np.finfo(np.float64).tiny)
    return int(different), float(np.max(np.abs(a-b)/scale, initial=0.0))


def _align_owned(actual: Any, expected: Any, shape: tuple[int, int], h: int = 2) -> Any:
    """Align a full-halo writer field with an A2D(0) model output."""
    actual_array, expected_array = np.asarray(actual), np.asarray(expected)
    if (
        expected_array.ndim >= 2
        and expected_array.shape[:2] == shape
        and actual_array.shape[:2] == (shape[0] - 2 * h, shape[1] - 2 * h)
    ):
        return expected_array[h:-h, h:-h, ...]
    return expected


def scorer_self_test(*, plant_one_ulp: bool = False) -> dict[str, Any]:
    """Full-ORCA2 synthetic owned/full-halo scorer control."""
    shape = (186, 153)
    actual = np.zeros((182, 149), dtype=np.float64)
    expected = np.full(shape, -7.0, dtype=np.float64)
    expected[2:-2, 2:-2] = actual
    if plant_one_ulp:
        expected[2, 2] = np.nextafter(0.0, 1.0)
    different, relative = _comparison(actual, _align_owned(actual, expected, shape), True)
    if plant_one_ulp:
        if different != 1:
            raise ValueError("one-ULP scorer plant did not produce exactly one differing value")
        return {"status": "FAIL", "different": different, "max_relative": relative}
    if different != 0 or relative != 0.0:
        raise ValueError("identical full-ORCA2 synthetic record did not pass")
    return {"status": "PASS", "shape": list(shape), "owned_shape": list(actual.shape)}


def _score_output(
    out: SI3O17StepState,
    trace,
    record: dict[str, Any],
    prather_exit: dict[str, Any],
    tau_private: dict[str, Any],
    exact: bool,
) -> dict[str, Any]:
    f = record["fields"]
    shape = record["meta"]["jpi"], record["meta"]["jpj"]
    h = 2
    rows: dict[str, dict[str, Any]] = {}
    def add(key: str, actual: Any, expected: Any) -> None:
        expected = _align_owned(actual, expected, shape, h)  # A2D(0) owned extent
        different, relative = _comparison(actual, expected, exact)
        rows[key] = {"different": different, "max_relative": relative}
    for name in STATE_FIELDS:
        add(f"state.{name}", getattr(out.ice, name), f[name])
    for family, prefix in PRATHER_PREFIX.items():
        actual_family = getattr(out.ice.prather, family)
        for member, stem in zip(actual_family._fields, ("sx", "sy", "sxx", "syy", "sxy"), strict=True):
            add(f"prather.{family}.{member}", getattr(actual_family, member), prather_exit["fields"][f"{stem}{prefix}"])
    for name in FLUX_FIELDS:
        add(f"fluxes.{name}", getattr(trace.finalize.final.fluxes, name), f[name])
    for name in ("qsr", "qns", "emp", "sfx", "snwice_mass", "snwice_mass_b", "snwice_fmass"):
        add(f"outputs.{name}", getattr(trace.finalize.final.outputs, name), f[name])
    for name in ("utau", "vtau", "taum"):
        add(f"stress.{name}", getattr(trace.finalize.stress, name), f[name])
    for name in ("tmod_io", "utau_oce", "vtau_oce"):
        add(f"stress.{name}", getattr(trace.finalize.stress, name), tau_private["fields"][name])
    return {"different": sum(row["different"] for row in rows.values()),
            "max_relative": max(row["max_relative"] for row in rows.values()), "fields": rows}


def _score_boundaries(trace, index, kt: int, rank: int, exact: bool) -> dict[str, Any]:
    """Use the recorded NEMO call boundaries to localise the first divergence."""

    rows: dict[str, dict[str, Any]] = {}
    def add(boundary: str, name: str, actual: Any) -> None:
        record = index[(boundary, kt, rank)]
        expected = record["fields"][name]
        shape = record["meta"]["jpi"], record["meta"]["jpj"]
        expected = _align_owned(actual, expected, shape)  # A2D(0) owned extent
        different, relative = _comparison(actual, expected, exact)
        rows[f"{KINDS[boundary]}.{name}"] = {"different": different, "max_relative": relative}

    for name, actual in (
        ("u_ice", trace.aevp.u_ice_u), ("v_ice", trace.aevp.v_ice_v),
        ("stress1_i", trace.aevp.stress1_t), ("stress2_i", trace.aevp.stress2_t),
        ("stress12_i", trace.aevp.stress12_f),
    ):
        add(23, name, actual)
    for name in STATE_FIELDS:
        add(24, name, getattr(trace.advection_state, name))
    for name in trace.redistribution.state._fields:
        if name in index[(25, kt, rank)]["fields"]:
            add(25, name, getattr(trace.redistribution.state, name))
    for name in trace.postdyn_correction._fields:
        if name in index[(26, kt, rank)]["fields"]:
            add(26, name, getattr(trace.postdyn_correction, name))
    for name in STATE_FIELDS:
        add(27, name, getattr(trace.finalize.correction.state, name))
    for name in ("qlead", "qsb_ice_bot", "fhld"):
        add(31, name, getattr(trace.surface_stress.flx_other, name))  # icestp.F90:182-189
    return {"different": sum(row["different"] for row in rows.values()),
            "max_relative": max(row["max_relative"] for row in rows.values()), "fields": rows}


def score(evidence: Path, labels: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    if labels != list(VALID_LABELS):
        raise ValueError(f"labels must be exactly {VALID_LABELS}")
    run = evidence / "record_100step"
    index = _record_index(run / "o17_master_records")
    ranks = sorted({key[2] for key in index})
    steps = sorted({key[1] for key in index})
    if steps != list(range(1, 101)):
        raise ValueError(f"O17 requires steps 1..100, got {steps[:3]}..{steps[-3:]}")
    nml = parse_resolved_ice_namelist(run/"output.namelist.ice", run/"output.namelist.dyn", DEFAULT_PPSRC)
    surface_cfg = SI3SurfaceConfig.from_resolved_namelists(run/"output.namelist.dyn", run/"output.namelist.ice")
    card = build_orca2_sas_ice_o17_card(surface=surface_cfg, thermo_namelist=nml)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    per_label: dict[str, Any] = {label: {"steps": {}} for label in labels}
    independent: dict[int, SI3O17StepState] = {}
    for kt in steps:
        for rank in ranks:
            nemo_entry = _entry_state(index[(22, kt, rank)], index[(29, kt, rank)], card)
            forcing = _forcing(
                index[(22, kt, rank)], index[(23, kt, rank)],
                index[(32, kt, rank)], index[(33, kt, rank)], card,
            )
            for label in labels:
                entry = nemo_entry if label == "GIVEN-NEMO-ENTRY" or kt == 1 else independent[rank]
                out, trace = si3_o17_step(entry, forcing, card)
                row = _score_output(
                    out, trace, index[(28, kt, rank)], index[(30, kt, rank)],
                    index[(34, kt, rank)],
                    label == "GIVEN-NEMO-ENTRY",
                )
                row["boundaries"] = _score_boundaries(
                    trace, index, kt, rank, label == "GIVEN-NEMO-ENTRY"
                )
                row["different"] += row["boundaries"]["different"]
                row["max_relative"] = max(row["max_relative"], row["boundaries"]["max_relative"])
                per_label[label]["steps"][f"{kt}:{rank}"] = row
                if label == "INDEPENDENT":
                    independent[rank] = out
    for label in labels:
        rows = list(per_label[label]["steps"].values())
        maximum = max(row["max_relative"] for row in rows)
        different = sum(row["different"] for row in rows)
        passed = different == 0 if label == "GIVEN-NEMO-ENTRY" else maximum <= 1.0e-15
        per_label[label].update(status="PASS" if passed else "FAIL", different=different,
                                max_relative=maximum, ulp_bar=0,
                                trajectory_relative_bar=1.0e-15)
    status = "PASS" if all(per_label[label]["status"] == "PASS" for label in labels) else "FAIL"
    return {"status": status, "labels": per_label, "steps": steps, "ranks": ranks,
            "production_entry": "legoesm.ice.si3_o17.si3_o17_step",
            "scorer_wall_seconds": time.monotonic() - started,
            "scorer_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def schema() -> dict[str, Any]:
    return {"status": "PASS", "magic": MAGIC.decode(), "version": VERSION, "kinds": KINDS,
            "state_fields": STATE_FIELDS, "flux_fields": FLUX_FIELDS, "labels": VALID_LABELS,
            "bars": {"local_ulp": 0, "trajectory_relative": 1.0e-15}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--labels", nargs=2, default=list(VALID_LABELS))
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--read-record", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--plant-one-ulp", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            result = scorer_self_test(plant_one_ulp=args.plant_one_ulp)
        elif args.schema:
            result = schema()
        elif args.read_record:
            record = read_record(args.read_record)
            result = {"status": "PASS", "meta": record["meta"], "fields": sorted(record["fields"])}
        elif args.evidence:
            result = score(args.evidence.resolve(), args.labels)
        else:
            raise ValueError("one of --schema, --read-record, or --evidence is required")
    except (OSError, ValueError, KeyError, struct.error) as exc:
        print(json.dumps({"status": "REFUSE", "reason": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
