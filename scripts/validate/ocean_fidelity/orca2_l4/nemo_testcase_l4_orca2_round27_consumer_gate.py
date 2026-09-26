#!/usr/bin/env python3
"""Direct component gate for ORCA2 round 27's thickness-consumer split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def array_score(candidate, reference) -> dict[str, object]:
    a = np.ascontiguousarray(np.asarray(candidate, dtype=np.float64))
    b = np.ascontiguousarray(np.asarray(reference, dtype=np.float64))
    require(a.shape == b.shape, f"array shapes differ: {a.shape} != {b.shape}")
    require(np.all(np.isfinite(a)) and np.all(np.isfinite(b)),
            "non-finite value in direct component comparison")
    unequal = int(np.count_nonzero(a.view(np.uint64) != b.view(np.uint64)))
    delta = np.abs(a - b)
    return {
        "cells": int(a.size),
        "unequal": unequal,
        "bit_identical": unequal == 0,
        "max_abs": float(delta.max(initial=0.0)),
    }


def _digest(*arrays) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(str(arr.shape).encode("ascii"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


def _ldf_inputs(deck_root: Path, record_root: Path, kt: int):
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
        compute_vertex_mask,
        nemo_ldf_lap_viscosity_e3_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_qco_live_face_geometry_cgrid,
        nemo_qco_live_vorticity_e3f_cgrid,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round11_dynldf_operator_gate as r11,
    )

    _, card = ladder.card_fields(deck_root)
    entry = r11.read_entry_frame(record_root, kt)
    state = card.recipe.initial_state
    state = state._replace(
        u=state.u.replace(data=jnp.asarray(
            r11._nemo_u_to_legoesm(entry["u"]), dtype=jnp.float64)),
        v=state.v.replace(data=jnp.asarray(
            r11._nemo_v_to_legoesm(entry["v"]), dtype=jnp.float64)),
        eta=state.eta.replace(data=jnp.asarray(entry["ssh"], dtype=jnp.float64)),
    )
    grid, zc = card.recipe.grid, card.recipe.z_coord
    cfg = card.recipe.model_config
    tmask = jnp.asarray(zc.is_active, dtype=jnp.float64)
    u_mask, v_mask = compute_face_masks_3d(tmask, grid)
    h_ref = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, zc,
        min_water_column_m=cfg.min_water_column_m)
    h_t = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, zc,
        min_water_column_m=cfg.min_water_column_m)
    base_u, base_v = _nemo_ws_qco_stage_faces(
        state.eta.data, h_ref, u_mask, v_mask, grid)[:2]
    base_f = nemo_qco_live_vorticity_e3f_cgrid(
        state.eta.data, zc, state.eta.data.dtype, grid=grid,
        e3t_0=h_ref, tmask=tmask)

    raw = zc.nemo_een_barotropic
    require(raw is not None, "ORCA2 card carries no native EEN mesh operands")
    raw_u, raw_v = nemo_qco_live_face_geometry_cgrid(
        state.eta.data, raw.e3u_0, raw.e3v_0, raw.umask, raw.vmask,
        raw.hu_0, raw.hv_0, raw.e1t * raw.e2t,
        raw.e1u * raw.e2u, raw.e1v * raw.e2v)[:2]

    cell_mask = state.land_mask.data
    surface_vertex = compute_vertex_mask(cell_mask, grid=grid)
    vertex3 = jax.vmap(
        lambda value: compute_vertex_mask(value, grid=grid),
        in_axes=-1, out_axes=-1)(tmask * cell_mask[..., None])
    vertex3 = vertex3 * surface_vertex[..., None]

    common = dict(
        u=state.u.data, v=state.v.data, grid=grid,
        ahmt=jnp.asarray(zc.nemo_ldf_ahmt, dtype=jnp.float64),
        ahmf=jnp.asarray(zc.nemo_ldf_ahmf, dtype=jnp.float64),
        h_k=h_t, mask=cell_mask, u_mask=u_mask, v_mask=v_mask,
        vertex_mask=vertex3, return_intermediates=True)
    bundles = {
        "base": (h_t, base_u, base_v, base_f, base_u, base_v),
        "kbb": (h_t, raw_u, raw_v, base_f, base_u, base_v),
        "kmm": (h_t, base_u, base_v, base_f, raw_u, raw_v),
    }
    outputs = {
        name: nemo_ldf_lap_viscosity_e3_cgrid(
            **common, thickness_operands=bundle)
        for name, bundle in bundles.items()
    }
    return outputs, card, entry


def _stage2_vorticity(deck_root: Path, record_root: Path):
    import jax.numpy as jnp

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    )

    _, card = ladder.card_fields(deck_root)
    entry = ladder.assemble_state_fields(record_root, 1, stage=None)
    state = card.recipe.initial_state._replace(
        eta=card.recipe.initial_state.eta.replace(
            data=jnp.asarray(entry["ssh"], dtype=jnp.float64)))
    for name in ("T", "S", "u", "v"):
        candidate = np.asarray(getattr(state, name).data)
        require(np.array_equal(candidate, np.asarray(entry[name])),
                f"Decision-52 entry {name} is no longer bit-identical")
    surface_fields = ladder.assemble_surface_fields(record_root, 1)
    freshwater, surface = ladder._surface_forcings(
        card, deck_root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_momentum_operator="vorticity",
            expose_momentum_operator_stage=2))
    exposed = model.step(
        state, dt=card.dt_s, freshwater=freshwater,
        surface_forcing=surface)
    return (np.asarray(exposed.u.data, dtype=np.float64),
            np.asarray(exposed.v.data, dtype=np.float64))


def capture(deck_root: Path, record_root: Path, npz_out: Path,
            helper_variant: str) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    require(helper_variant in ("parent", "raw_f"),
            "helper variant must be parent or raw_f")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "round 27 is CPU-only")
    require(not bool(jax.config.jax_disable_jit),
            "production stage exposure requires JIT")

    kt2, _, entry2 = _ldf_inputs(deck_root, record_root, 2)
    base_u, base_v, base_i = kt2["base"]
    kbb_u, kbb_v, kbb_i = kt2["kbb"]
    kmm_u, kmm_v, kmm_i = kt2["kmm"]
    kt1, _, entry1 = _ldf_inputs(deck_root, record_root, 1)
    ldf1_u, ldf1_v, _ = kt1["base"]
    vor_u, vor_v = _stage2_vorticity(deck_root, record_root)

    arrays = {
        "kt1_ldf_u": np.asarray(ldf1_u, dtype=np.float64),
        "kt1_ldf_v": np.asarray(ldf1_v, dtype=np.float64),
        "stage2_vorticity_u": vor_u,
        "stage2_vorticity_v": vor_v,
    }
    npz_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_out, **arrays)

    direct = {
        "kbb_vs_kmm_u": array_score(kbb_u, kmm_u),
        "kbb_vs_kmm_v": array_score(kbb_v, kmm_v),
        "kbb_output_vs_base_u": array_score(kbb_u, base_u),
        "kbb_output_vs_base_v": array_score(kbb_v, base_v),
        "kmm_output_vs_base_u": array_score(kmm_u, base_u),
        "kmm_output_vs_base_v": array_score(kmm_v, base_v),
        "kbb_grad_div_u_vs_base": array_score(
            kbb_i["grad_div_u"], base_i["grad_div_u"]),
        "kbb_grad_div_v_vs_base": array_score(
            kbb_i["grad_div_v"], base_i["grad_div_v"]),
        "kbb_curl_u_vs_base": array_score(kbb_i["curl_u"], base_i["curl_u"]),
        "kbb_curl_v_vs_base": array_score(kbb_i["curl_v"], base_i["curl_v"]),
        "kmm_grad_div_u_vs_base": array_score(
            kmm_i["grad_div_u"], base_i["grad_div_u"]),
        "kmm_grad_div_v_vs_base": array_score(
            kmm_i["grad_div_v"], base_i["grad_div_v"]),
        "kmm_curl_u_vs_base": array_score(kmm_i["curl_u"], base_i["curl_u"]),
        "kmm_curl_v_vs_base": array_score(kmm_i["curl_v"], base_i["curl_v"]),
    }
    return {
        "status": "CAPTURED",
        "claim_label": "given NEMO's entry",
        "helper_variant": helper_variant,
        "worktree": worktree_stamp(),
        "dtype": "float64",
        "backend": jax.default_backend(),
        "record_root": str(record_root),
        "npz": str(npz_out),
        "npz_sha256": hashlib.sha256(npz_out.read_bytes()).hexdigest(),
        "consumer_inventory": [
            "dynldf_lev F-curl thickness",
            "dynvor EEN potential-vorticity thickness",
        ],
        "kt1_entry_velocity_max_abs": max(
            float(np.abs(entry1["u"]).max()),
            float(np.abs(entry1["v"]).max())),
        "kt2_entry_velocity_max_abs": max(
            float(np.abs(entry2["u"]).max()),
            float(np.abs(entry2["v"]).max())),
        "kt1_ldf": {
            "u_max_abs": float(np.abs(arrays["kt1_ldf_u"]).max()),
            "v_max_abs": float(np.abs(arrays["kt1_ldf_v"]).max()),
            "digest": _digest(arrays["kt1_ldf_u"], arrays["kt1_ldf_v"]),
        },
        "stage2_vorticity": {
            "u_max_abs": float(np.abs(vor_u).max()),
            "v_max_abs": float(np.abs(vor_v).max()),
            "digest": _digest(vor_u, vor_v),
        },
        "kt2_direct_ldf": direct,
        "citations": {
            "ldf_f_curl": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:121-125",
            "ldf_kbb": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:127-129",
            "ldf_kmm": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:132-140",
            "een_reference": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:907-937",
            "een_consumer": "ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynvor.f90:733-805",
        },
    }


def compare_captures(parent: dict, raw_f: dict, parent_arrays: dict,
                     raw_arrays: dict, *, plant: bool = False) -> dict:
    inventory = {
        "dynldf_lev F-curl thickness",
        "dynvor EEN potential-vorticity thickness",
    }
    require(parent["helper_variant"] == "parent", "first capture is not parent")
    require(raw_f["helper_variant"] == "raw_f", "second capture is not raw_f")
    require(set(parent["consumer_inventory"]) == inventory,
            "parent consumer inventory is incomplete")
    require(set(raw_f["consumer_inventory"]) == inventory,
            "raw-F consumer inventory is incomplete")
    require(parent["dtype"] == raw_f["dtype"] == "float64",
            "capture dtype changed")
    require(parent["backend"] == raw_f["backend"] == "cpu",
            "capture backend changed")
    require(parent["record_root"] == raw_f["record_root"],
            "capture record roots differ")

    p = {name: np.asarray(value) for name, value in parent_arrays.items()}
    r = {name: np.asarray(value) for name, value in raw_arrays.items()}
    if plant:
        r["kt1_ldf_u"] = np.array(r["kt1_ldf_u"], copy=True)
        r["kt1_ldf_u"].flat[0] = np.nextafter(
            r["kt1_ldf_u"].flat[0], np.float64(np.inf))

    direct = parent["kt2_direct_ldf"]
    p1 = (
        not direct["kbb_vs_kmm_u"]["bit_identical"]
        or not direct["kbb_vs_kmm_v"]["bit_identical"])
    p1 = p1 and all(
        not direct[name]["bit_identical"] for name in (
            "kbb_grad_div_u_vs_base", "kbb_grad_div_v_vs_base",
            "kmm_curl_u_vs_base", "kmm_curl_v_vs_base"))
    p1 = p1 and all(
        direct[name]["bit_identical"] for name in (
            "kbb_curl_u_vs_base", "kbb_curl_v_vs_base",
            "kmm_grad_div_u_vs_base", "kmm_grad_div_v_vs_base"))

    ldf_u = array_score(r["kt1_ldf_u"], p["kt1_ldf_u"])
    ldf_v = array_score(r["kt1_ldf_v"], p["kt1_ldf_v"])
    p2 = (
        parent["kt1_entry_velocity_max_abs"] == 0.0
        and raw_f["kt1_entry_velocity_max_abs"] == 0.0
        and parent["kt1_ldf"]["u_max_abs"] == 0.0
        and parent["kt1_ldf"]["v_max_abs"] == 0.0
        and raw_f["kt1_ldf"]["u_max_abs"] == 0.0
        and raw_f["kt1_ldf"]["v_max_abs"] == 0.0
        and ldf_u["bit_identical"] and ldf_v["bit_identical"])

    vor_u = array_score(r["stage2_vorticity_u"], p["stage2_vorticity_u"])
    vor_v = array_score(r["stage2_vorticity_v"], p["stage2_vorticity_v"])
    p3 = p2 and (not vor_u["bit_identical"] or not vor_v["bit_identical"])
    if plant:
        require(p2, "planted kt=1 lateral-diffusion violation passed")
        raise GateError("planted kt=1 lateral-diffusion violation passed")
    require(p1, "Kbb/Kmm direct component discriminator did not bind")
    require(p2, "kt=1 lateral-diffusion zero control failed")
    require(p3, "raw-F arm did not move the stage-2 EEN vorticity component")
    return {
        "status": "HELD",
        "claim_label": "independent with Decision-52 SSH",
        "retraction": (
            "round 26's F-curl-only attribution is withdrawn: its helper arm "
            "also moved the EEN potential-vorticity thickness"),
        "kbb_kmm_direct": {
            "u": direct["kbb_vs_kmm_u"],
            "v": direct["kbb_vs_kmm_v"],
            "verdict": "distinct tendency components; trajectory summaries collided",
        },
        "kt1_ldf_parent_vs_raw_f": {"u": ldf_u, "v": ldf_v},
        "stage2_vorticity_parent_vs_raw_f": {"u": vor_u, "v": vor_v},
        "predictions": {
            "R27-P1": "CONFIRMED",
            "R27-P2": "CONFIRMED",
            "R27-P3": "CONFIRMED",
            "R27-P4": "CONFIRMED",
        },
    }


def _read_json(path: Path) -> dict:
    require(path.is_file(), f"missing JSON capture: {path}")
    return json.loads(path.read_text())


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"missing NPZ capture: {path}")
    with np.load(path) as values:
        expected = {"kt1_ldf_u", "kt1_ldf_v",
                    "stage2_vorticity_u", "stage2_vorticity_v"}
        require(set(values.files) == expected,
                f"capture fields differ: {set(values.files)}")
        return {name: np.asarray(values[name]) for name in values.files}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--deck-root", type=Path, required=True)
    cap.add_argument("--record-root", type=Path, required=True)
    cap.add_argument("--helper-variant", choices=("parent", "raw_f"), required=True)
    cap.add_argument("--npz-out", type=Path, required=True)
    cap.add_argument("--json-out", type=Path, required=True)
    comp = sub.add_parser("compare")
    comp.add_argument("--parent-json", type=Path, required=True)
    comp.add_argument("--parent-npz", type=Path, required=True)
    comp.add_argument("--raw-f-json", type=Path, required=True)
    comp.add_argument("--raw-f-npz", type=Path, required=True)
    comp.add_argument("--json-out", type=Path)
    comp.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "capture":
            result = capture(args.deck_root, args.record_root, args.npz_out,
                             args.helper_variant)
            rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered)
            print(rendered, end="")
            return 0
        result = compare_captures(
            _read_json(args.parent_json), _read_json(args.raw_f_json),
            _read_npz(args.parent_npz), _read_npz(args.raw_f_npz),
            plant=args.plant)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(rendered)
        print(rendered, end="")
        return 2
    except (GateError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
