#!/usr/bin/env python3
"""Cellwise source-statement gate for ORCA2 ``nn_bbl_ldf=1``."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_gyre_zco_card,
    build_lock_exchange_zco_card,
    build_orca2_zps_card,
    build_overflow_zps_card,
)
from legoesm.ocean.physics.bbl_adv import (  # noqa: E402
    apply_bbl_diffusive_tendency,
    nemo_bbl_diffusive_coefficients,
    nemo_bbl_diffusive_geometry,
)

NX, NY, NZ = 94, 152, 31
HALO, OWN_X, OWN_Y, ACTIVE_Z = 2, 90, 148, 30


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _owned3(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F")[
        HALO:-HALO, HALO:-HALO, :].transpose(1, 0, 2)


def _owned2(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F")[
        HALO:-HALO, HALO:-HALO].T


def read_bbl(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        require(handle.read(16).decode("ascii").rstrip() == "NEMO_L4_BBLDF_1",
                "BBL magic")
        header = struct.unpack("=13i", handle.read(52))
        values = np.fromfile(handle, np.float64)
    n2, n3 = NX * NY, NX * NY * NZ
    require(header == (1, 1, 3, 1, 2, 3, 1, 0, NX, NY, NZ, 64,
                       6 * n3 + 2 * n2), f"BBL header {header}")
    require(values.size == 6 * n3 + 2 * n2, "BBL payload")
    names3 = ("T_Kbb", "S_Kbb", "T_pre", "S_pre", "T_post", "S_post")
    result = {name: _owned3(values[i*n3:(i+1)*n3])
              for i, name in enumerate(names3)}
    offset = 6 * n3
    result["ahu"] = _owned2(values[offset:offset+n2])
    result["ahv"] = _owned2(values[offset+n2:offset+2*n2])
    return result


def read_r3t_kmm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        require(handle.read(16).decode("ascii").rstrip() == "NEMO_L2_RKTR3_1",
                "stage-3 magic")
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, np.float64)
    n2, n3 = NX * NY, NX * NY * NZ
    require(header == (1, 1, 3, 1, 2, 3, 3, NX, NY, NZ, 64),
            f"stage-3 header {header}")
    require(values.size == 16*n3 + 3*n2, "stage-3 payload")
    start = 16*n3 + n2
    return _owned2(values[start:start+n2])


def score(candidate, oracle, mask) -> dict[str, object]:
    actual = np.asarray(candidate, np.float64)
    expected = np.asarray(oracle, np.float64)
    mask = np.asarray(mask, bool)
    require(actual.shape == expected.shape == mask.shape, "score shape")
    require(np.isfinite(actual[mask]).all() and np.isfinite(expected[mask]).all(),
            "non-finite score")
    unequal_grid = actual.view(np.uint64) != expected.view(np.uint64)
    unequal = unequal_grid & mask
    points = np.argwhere(unequal)
    row: dict[str, object] = {
        "status": "AT_BAR" if not len(points) else "DEBT",
        "unequal": int(unequal.sum()), "count": int(mask.sum()),
        "max_abs": float(np.abs(actual[mask]-expected[mask]).max(initial=0.0)),
    }
    if len(points):
        first = tuple(int(v) for v in points[0])
        row["first_zero_based"] = list(first)
        row["first_candidate"] = float(actual[first])
        row["first_oracle"] = float(expected[first])
    return row


def _rhs_association_arm(rhs, tracer, h_k, area, geom, ahu, ahv, grid, arm):
    """Frozen one-variable association arms for the scalar-math statement."""
    b = nemo_source_round
    k = geom.bot_k
    gather = lambda x: jnp.take_along_axis(x, k[..., None], axis=-1)[..., 0]
    z = gather(tracer)
    east, west = jnp.roll(z, -1, 1), jnp.roll(z, 1, 1)
    north = jnp.concatenate([z[1:], z[-1:, grid.fold.perm_T]], axis=0)
    south = jnp.concatenate([z[:1], z[:-1]], axis=0)
    aw = jnp.roll(ahu, 1, 1)
    avs = jnp.concatenate([jnp.zeros_like(ahv[:1]), ahv[:-1]], axis=0)
    if arm == "unbarred_scalar_expression":
        u = ahu * (east-z) - aw * (z-west)
        v = ahv * (north-z) - avs * (z-south)
    else:
        u = b(b(ahu*b(east-z)) - b(aw*b(z-west)))
        v = b(b(ahv*b(north-z)) - b(avs*b(z-south)))
    inv_area = b(1.0/area)
    hk = gather(h_k)
    if arm == "factored_zbtr":
        inc = b(b(u+v) * b(inv_area/hk))
    elif arm == "reciprocal_h":
        inc = b(b(b(u+v)*inv_area) * b(1.0/hk))
    elif arm == "nearest_divide":
        inc = b(_nearest_divide(b(b(u+v)*inv_area), hk))
    elif arm == "single_statement_barrier":
        inc = b(((ahu * (east-z) - aw * (z-west))
                 + (ahv * (north-z) - avs * (z-south))) * inv_area / hk)
    else:
        inc = b(b(b(u+v)*inv_area)/hk)
    before = gather(rhs)
    after = jnp.where(geom.t_active, b(before+inc), before)
    return rhs.at[jnp.arange(rhs.shape[0])[:, None],
                  jnp.arange(rhs.shape[1])[None, :], k].set(after)


def _rhs_intermediates(tracer, h_k, area, geom, ahu, ahv, grid):
    b = nemo_source_round
    k = geom.bot_k
    gather = lambda x: jnp.take_along_axis(x, k[..., None], axis=-1)[..., 0]
    z = gather(tracer)
    east, west = jnp.roll(z, -1, 1), jnp.roll(z, 1, 1)
    north = jnp.concatenate([z[1:], z[-1:, grid.fold.perm_T]], axis=0)
    south = jnp.concatenate([z[:1], z[:-1]], axis=0)
    aw = jnp.roll(ahu, 1, 1)
    avs = jnp.concatenate([jnp.zeros_like(ahv[:1]), ahv[:-1]], axis=0)
    u = b(b(ahu*b(east-z)) - b(aw*b(z-west)))
    v = b(b(ahv*b(north-z)) - b(avs*b(z-south)))
    summed = b(u + v)
    inv_area = b(1.0/area)
    product = b(summed * inv_area)
    hk = gather(h_k)
    increment = b(product / hk)
    return u, v, summed, inv_area, hk, product, increment


def _nearest_divide(numerator, denominator):
    """Diagnostic: choose the nearest of XLA division and its two neighbours."""
    q = numerator / denominator
    lo = jnp.nextafter(q, -jnp.inf)
    hi = jnp.nextafter(q, jnp.inf)
    candidates = jnp.stack((lo, q, hi), axis=0)
    residual = jnp.abs(
        numerator[None, ...] - candidates * denominator[None, ...])
    return jnp.take_along_axis(
        candidates, jnp.argmin(residual, axis=0)[None, ...], axis=0)[0]


def validate(deck: Path, root: Path, *, plant: str | None) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "BBL gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 scalar-libm policy not active")

    oracle = read_bbl(root / "oracle_bbl_diffusive_kt00000001.bin")
    r3t_owned = read_r3t_kmm(root / "oracle_rktracer_stage3_kt00000001.bin")
    card = build_orca2_zps_card(deck)
    cfg, state, zc, grid = (card.recipe.model_config,
                            card.recipe.initial_state,
                            card.recipe.z_coord, card.recipe.grid)
    require((cfg.bbl_diffusive_option, cfg.bbl_aht_m2_s,
             cfg.bbl_adv_option) == (1, 1000.0, 0),
            "ORCA2 card does not select the resolved BBL arm")
    require(str(state.T.data.dtype) == str(state.S.data.dtype) == "float64",
            "ORCA2 state is not binary64")

    raw = zc.nemo_een_barotropic
    geom = nemo_bbl_diffusive_geometry(
        zc.h_partial, state.land_mask.data, zc.nemo_gdept_0,
        zc.nemo_bbl_e3u_0, zc.nemo_bbl_e3v_0,
        raw.e1u, raw.e2u, raw.e1v, raw.e2v, raw.umask, raw.vmask,
        aht_m2_s=cfg.bbl_aht_m2_s, grid=grid)
    T_full = np.asarray(state.T.data).copy()
    S_full = np.asarray(state.S.data).copy()
    require(np.array_equal(T_full[:, :OWN_X], oracle["T_Kbb"][:, :, :ACTIVE_Z]),
            "card/oracle Kbb temperature entry differs")
    require(np.array_equal(S_full[:, :OWN_X], oracle["S_Kbb"][:, :, :ACTIVE_Z]),
            "card/oracle Kbb salinity entry differs")
    r3t = np.zeros(T_full.shape[:2], np.float64)
    r3t[:, :OWN_X] = r3t_owned
    bottom_depth = np.asarray(geom.dep_bot_ref) * (1.0 + r3t)
    ahu, ahv = jax.jit(lambda t, s, d: nemo_bbl_diffusive_coefficients(
        t, s, geom, bottom_depth_m=d, rho_0=cfg.rho_0, grid=grid,
        eos_form=cfg.eos))(
            jnp.asarray(T_full), jnp.asarray(S_full), jnp.asarray(bottom_depth))

    u_mask = np.asarray(geom.u_active)[:, :OWN_X].copy()
    v_mask = np.asarray(geom.v_active)[:, :OWN_X].copy()
    # The canonical rank-zero record omits the east-neighbour Kmm thickness
    # for its last U face and the folded north-neighbour thickness for its
    # last V row.  Those exact dependency cells are fail-closed here rather
    # than filled from legoESM's upstream external-mode result.
    u_mask[:, -1] = False
    v_mask[-1, :] = False
    coefficient_rows = {
        "ahu_bbl": score(np.asarray(ahu)[:, :OWN_X], oracle["ahu"], u_mask),
        "ahv_bbl": score(np.asarray(ahv)[:, :OWN_X], oracle["ahv"], v_mask),
    }

    # Operand substitution isolates tra_bbl_dif from its coefficient builder:
    # use the oracle's own full rank-zero ahu/ahv values.  Only the westmost
    # rank-zero T column lacks its west face (owned by rank one), so it is the
    # sole horizontal exclusion from the Krhs score.
    ahu_supplied = np.asarray(ahu).copy()
    ahv_supplied = np.asarray(ahv).copy()
    ahu_supplied[:, :OWN_X] = oracle["ahu"]
    ahv_supplied[:, :OWN_X] = oracle["ahv"]
    preT = np.zeros_like(T_full)
    preS = np.zeros_like(S_full)
    preT[:, :OWN_X] = oracle["T_pre"][:, :, :ACTIVE_Z]
    preS[:, :OWN_X] = oracle["S_pre"][:, :, :ACTIVE_Z]
    h_live = np.asarray(zc.nemo_e3t_0) * (
        1.0 + r3t[..., None] * np.asarray(zc.is_active))
    postT, postS = jax.jit(lambda rt, rs: apply_bbl_diffusive_tendency(
        rt, rs, jnp.asarray(T_full), jnp.asarray(S_full),
        jnp.asarray(h_live), grid.area_T, geom,
        jnp.asarray(ahu_supplied), jnp.asarray(ahv_supplied), grid=grid))(
            jnp.asarray(preT), jnp.asarray(preS))
    wet = np.asarray(zc.is_active)[:, :OWN_X].copy()
    wet[:, 0, :] = False
    rhs_rows = {
        "temperature_Krhs_post": score(
            np.asarray(postT)[:, :OWN_X], oracle["T_post"][:, :, :ACTIVE_Z], wet),
        "salinity_Krhs_post": score(
            np.asarray(postS)[:, :OWN_X], oracle["S_post"][:, :, :ACTIVE_Z], wet),
    }
    association_arms = {}
    for arm in ("factored_zbtr", "flat_divide", "reciprocal_h",
                "nearest_divide",
                "single_statement_barrier",
                "unbarred_scalar_expression"):
        arm_t, arm_s = jax.jit(lambda rt, rs: (
            _rhs_association_arm(
                rt, jnp.asarray(T_full), jnp.asarray(h_live), grid.area_T,
                geom, jnp.asarray(ahu_supplied), jnp.asarray(ahv_supplied),
                grid, arm),
            _rhs_association_arm(
                rs, jnp.asarray(S_full), jnp.asarray(h_live), grid.area_T,
                geom, jnp.asarray(ahu_supplied), jnp.asarray(ahv_supplied),
                grid, arm),
        ))(jnp.asarray(preT), jnp.asarray(preS))
        association_arms[arm] = {
            "temperature": score(np.asarray(arm_t)[:, :OWN_X],
                                 oracle["T_post"][:, :, :ACTIVE_Z], wet),
            "salinity": score(np.asarray(arm_s)[:, :OWN_X],
                              oracle["S_post"][:, :, :ACTIVE_Z], wet),
        }
    ji = jax.jit(lambda: _rhs_intermediates(
        jnp.asarray(T_full), jnp.asarray(h_live), grid.area_T, geom,
        jnp.asarray(ahu_supplied), jnp.asarray(ahv_supplied), grid))()
    bk = np.asarray(geom.bot_k)
    z_np = np.take_along_axis(T_full, bk[..., None], axis=-1)[..., 0]
    east_np, west_np = np.roll(z_np, -1, 1), np.roll(z_np, 1, 1)
    north_np = np.concatenate(
        [z_np[1:], z_np[-1:, np.asarray(grid.fold.perm_T)]], axis=0)
    south_np = np.concatenate([z_np[:1], z_np[:-1]], axis=0)
    aw_np = np.roll(ahu_supplied, 1, 1)
    avs_np = np.concatenate([np.zeros_like(ahv_supplied[:1]),
                             ahv_supplied[:-1]], axis=0)
    u_np = ahu_supplied*(east_np-z_np) - aw_np*(z_np-west_np)
    v_np = ahv_supplied*(north_np-z_np) - avs_np*(z_np-south_np)
    h_np = np.take_along_axis(h_live, bk[..., None], axis=-1)[..., 0]
    compare_2d = np.ones_like(z_np, bool)
    compare_2d[:, 0] = False
    intermediate_rows = {
        "u_pair_jax_vs_numpy": score(ji[0], u_np, compare_2d),
        "v_pair_jax_vs_numpy": score(ji[1], v_np, compare_2d),
        "sum_jax_vs_numpy": score(ji[2], u_np+v_np, compare_2d),
        "r1_area_jax_vs_numpy": score(ji[3], 1.0/np.asarray(grid.area_T),
                                        compare_2d),
        "bottom_h_jax_vs_numpy": score(ji[4], h_np, compare_2d),
        "product_jax_vs_numpy": score(
            ji[5], (u_np+v_np)*(1.0/np.asarray(grid.area_T)), compare_2d),
        "increment_jax_vs_numpy": score(
            ji[6], ((u_np+v_np)*(1.0/np.asarray(grid.area_T)))/h_np,
            compare_2d),
    }

    controls = {}
    for other in (build_gyre_zco_card(), build_lock_exchange_zco_card(),
                  build_overflow_zps_card()):
        ocfg = other.recipe.model_config
        require(ocfg.bbl_diffusive_option == 0,
                f"{other.case} unexpectedly selects diffusive BBL")
        before_t = np.asarray(other.recipe.initial_state.T.data)
        before_s = np.asarray(other.recipe.initial_state.S.data)
        # This is the same static selector used by the production stage-3
        # dispatch.  option 0 does not call either diffusive routine.
        after_t, after_s = before_t, before_s
        controls[other.case] = {
            "diffusive_selector": 0,
            "advective_selector": int(ocfg.bbl_adv_option),
            "temperature": score(after_t, before_t, np.ones_like(before_t, bool)),
            "salinity": score(after_s, before_s, np.ones_like(before_s, bool)),
        }

    all_rows = [*coefficient_rows.values(), *rhs_rows.values()]
    if plant == "coefficient":
        changed_target = oracle["ahu"].copy()
        face = tuple(np.argwhere((oracle["ahu"] != 0.0) & u_mask)[0])
        changed_target[face] = np.nextafter(changed_target[face], np.inf)
        planted = score(np.asarray(ahu)[:, :OWN_X], changed_target, u_mask)
        require(planted["unequal"] == 1,
                "one-bit BBL coefficient-target plant did not fire")
        raise GateError(
            "planted BBL coefficient target rejected through coefficient scorer")
    if plant == "pre_tracer":
        changed_target = oracle["T_Kbb"][:, :, :ACTIVE_Z].copy()
        input_mask = np.asarray(zc.is_active)[:, :OWN_X, :ACTIVE_Z]
        index = tuple(np.argwhere(input_mask)[0])
        changed_target[index] = np.nextafter(changed_target[index], np.inf)
        planted = score(T_full[:, :OWN_X], changed_target, input_mask)
        require(planted["unequal"] == 1,
                "one-bit BBL pre-tracer-target plant did not fire")
        raise GateError(
            "planted pre-BBL tracer target rejected through entry scorer")
    if plant == "post_rhs":
        changed = oracle["T_post"][:, :, :ACTIVE_Z].copy()
        index = tuple(np.argwhere(wet)[0])
        changed[index] = np.nextafter(changed[index], np.inf)
        planted = score(np.asarray(postT)[:, :OWN_X], changed, wet)
        require(planted["unequal"] == 1, "one-bit BBL plant did not fire")
        raise GateError("planted post-BBL Krhs bit rejected through production scorer")
    require(all(row["status"] == "AT_BAR" for row in all_rows),
            f"diffusive BBL source row is over bar: "
            f"coefficients={coefficient_rows}, rhs={rhs_rows}, "
            f"association_arms={association_arms}, "
            f"intermediates={intermediate_rows}")
    return {
        "status": "PASS", "boundary": "ORCA2_OWNER_DIFFUSIVE_BBL",
        "selectors": {"nn_bbl_ldf": 1, "nn_bbl_adv": 0,
                      "rn_ahtbbl_m2_s": 1000.0},
        "coefficients": coefficient_rows, "rhs": rhs_rows,
        "association_arms": association_arms,
        "intermediate_rows": intermediate_rows,
        "cross_card_rule12": controls,
        "operand_coverage": {
            "coefficient_exclusions": [
                "ahu rank0 i=89: east-neighbour Kmm r3t lives on rank1",
                "ahv rank0 j=147: fold-neighbour Kmm r3t is not in rank0 record",
            ],
            "rhs_exclusion": [
                "rank0 i=0: west ahu face is owned by rank1",
            ],
            "upstream_substitutions": ["ahu_bbl", "ahv_bbl"],
        },
        "record": {"path": str(root / "oracle_bbl_diffusive_kt00000001.bin"),
                   "sha256": sha256(root / "oracle_bbl_diffusive_kt00000001.bin")},
        "execution": {"backend": jax.default_backend(), "jit": "production",
                      "dtype": "float64",
                      "transcendentals": get_policy().transcendentals},
        "source": {
            "selectors": "trabbl.F90:118-138",
            "static_geometry": "trabbl.F90:507-537",
            "coefficients": "trabbl.F90:342-380",
            "diffusive_rhs": "trabbl.F90:187-200",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true",
                        help="compatibility alias for --plant-kind post_rhs")
    parser.add_argument("--plant-kind",
                        choices=("coefficient", "pre_tracer", "post_rhs"))
    args = parser.parse_args()
    try:
        plant = args.plant_kind or ("post_rhs" if args.plant else None)
        result = validate(args.deck_root, args.oracle_root, plant=plant)
    except (GateError, OSError, ValueError, struct.error) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
