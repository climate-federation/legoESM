#!/usr/bin/env python3
"""Adjudicate ORCA2 stage-1 WZV on the compiled production step."""

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
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (  # noqa: E402
    nemo_qco_wzv_recurrence,
    nemo_transport_wzv_divergence_level,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_zps_card,
    validate_nemo_testcase_card,
)
from legoesm.ocean.freshwater import FreshwaterForcing  # noqa: E402

NX, NY, NZ = 94, 152, 31
OWNED_NX, OWNED_NY = NX - 4, NY - 4
NLEV = NZ - 1
WZV_RECORD = "oracle_stage1_wzv_operands_kt00000001.bin"
STAGE3_RECORD = "oracle_stage_kt00000001_s3.bin"
BT_RECORD = "oracle_bt_frames_kt00000001.bin"
TRANSPORT_RECORD = "oracle_rkstage1_transport_operands_kt00000001.bin"


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _xy(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F")[2:-2, 2:-2].T


def _xyz(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _stencil(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX - 3, NY - 3, NZ), order="F").transpose(1, 0, 2)


def read_wzv(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=14i", handle.read(56))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L4_WZVS1_1", f"bad WZV magic {magic!r}")
    require(
        header == (1, 1, 1, 1, 1, 3, NX, NY, NZ, 2, NX - 2, 2, NY - 2, 64),
        f"bad WZV header {header}",
    )
    n2, n3, ns = NX * NY, NX * NY * NZ, (NX - 3) * (NY - 3) * NZ
    require(values.size == 2 * ns + 5 * n3 + 4 * n2, "bad WZV payload")
    pfu = _stencil(values[:ns])
    pfv = _stencil(values[ns:2 * ns])
    offset = 2 * ns
    e3t = _xyz(values[offset:offset + n3]); offset += n3
    e3t0 = _xyz(values[offset:offset + n3]); offset += n3
    tmask = _xyz(values[offset:offset + n3]); offset += n3
    r3bb = _xy(values[offset:offset + n2]); offset += n2
    r3aa = _xy(values[offset:offset + n2]); offset += n2
    r1_area = _xy(values[offset:offset + n2]); offset += n2
    runoff = _xy(values[offset:offset + n2]); offset += n2
    ww = _xyz(values[offset:offset + n3]); offset += n3
    pfw = _xyz(values[offset:offset + n3]); offset += n3
    require(offset == values.size, "WZV walk did not reach EOF")
    require(np.isfinite(values).all(), "non-finite WZV payload")
    return {
        "pFu_stencil": pfu, "pFv_stencil": pfv,
        "e3t": e3t, "e3t0": e3t0, "tmask": tmask,
        "r3bb": r3bb, "r3aa": r3aa, "r1_area": r1_area,
        "runoff": runoff, "ww": ww, "pfw": pfw,
    }


def read_final_ssh(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=9i", handle.read(36))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L1_STAGE_1", f"bad stage magic {magic!r}")
    require(header == (1, 1, 3, 3, NX, NY, NZ, 2, 64), f"bad stage header {header}")
    n3, n2 = NX * NY * NZ, NX * NY
    require(values.size == 4 * n3 + n2, "bad stage payload")
    return _xy(values[4 * n3:])


def read_external_transports(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=6i", handle.read(24))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L1_BTFRM_1", f"bad BT magic {magic!r}")
    require(header == (1, 1, 3, NX, NY, 64), f"bad BT header {header}")
    n2 = NX * NY
    require(values.size == 4 * n2, "bad BT payload")
    return _xy(values[2 * n2:3 * n2]), _xy(values[3 * n2:])


def read_transport_metrics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the two native face metrics from the frozen transport schema."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L2_TRPOP_2", f"bad transport magic {magic!r}")
    require(header == (2, 1, 1, 1, NX, NY, NZ, 64), "bad transport header")
    n2, n3 = NX * NY, NX * NY * NZ
    require(values.size == 10 * n2 + 8 * n3, "bad transport payload")
    e2u = _xy(values[:n2])
    e1v_offset = 2 * n2 + 4 * n3
    e1v = _xy(values[e1v_offset:e1v_offset + n2])
    return e2u, e1v


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(actual.shape == expected.shape and actual.size, "empty production-W score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "non-finite production-W value")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    ia, ie = actual.view(np.int64), expected.view(np.int64)
    oa = ia ^ ((ia >> 63) & 0x7fffffffffffffff)
    oe = ie ^ ((ie >> 63) & 0x7fffffffffffffff)
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(unequal.size),
        "max_abs": float(np.abs(actual - expected).max(initial=0.0)),
        "max_ulp": int(np.abs(oa - oe).max(initial=0)),
    }


def validate(deck_root: Path, oracle_root: Path, *, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "production-W gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    paths = {name: oracle_root / name for name in (
        WZV_RECORD, STAGE3_RECORD, BT_RECORD, TRANSPORT_RECORD)}
    for path in paths.values():
        require(path.is_file(), f"missing {path}")
    wzv = read_wzv(paths[WZV_RECORD])
    final_ssh = read_final_ssh(paths[STAGE3_RECORD])
    un_adv, vn_adv = read_external_transports(paths[BT_RECORD])
    e2u_oracle, e1v_oracle = read_transport_metrics(paths[TRANSPORT_RECORD])

    card = build_orca2_zps_card(deck_root)
    validate_nemo_testcase_card(card)
    cfg = card.recipe.model_config
    require(cfg.momentum_advection == "vector_invariant",
            "ORCA2 card does not select vector momentum")
    require(cfg.ke_gradient_scheme == "c2",
            "ORCA2 card does not select nn_dynkeg=0/C2")
    metric_rows = {
        "e2u": score(
            np.asarray(card.recipe.grid.dy_u)[:, 1:OWNED_NX + 1],
            e2u_oracle, e2u_oracle != 0.0),
        "e1v": score(
            np.asarray(card.recipe.grid.dx_v)[1:OWNED_NY + 1, :OWNED_NX],
            e1v_oracle, e1v_oracle != 0.0),
    }

    # The card's selected EOS-80 TKE/EVD entry is itself a downstream
    # UNMEASURED shared boundary: the shared bn2 closure currently refuses
    # eos_form='eos80'.  Use the already constructible TEOS-10 bn2 only in
    # upstream tendencies whose external result is replaced below.  No proxy
    # value reaches _g0 or the exposed WZV value.
    vmix = cfg.physics.vertical_mixing
    tke = vmix.tke._replace(n2_eos_form="teos10")
    convection = cfg.physics.convection._replace(
        enhanced_diffusion=cfg.physics.convection.enhanced_diffusion._replace(
            n2_eos_form="teos10"))
    diagnostic_cfg = cfg._replace(
        # The production EEN momentum tendency has not yet acquired the
        # ORCA2 tripolar avg4 fold.  It is upstream of the substituted
        # external endpoint and cannot reach the exposed _g0 value.
        een_e3f_scheme="min",
        # The selected spatial nemo_div_curl viscosity is another unmeasured
        # upstream tripolar operator.  Zero it only in the discarded tendency.
        lateral_viscosity=cfg.lateral_viscosity._replace(A_h=0.0),
        physics=cfg.physics._replace(
            vertical_mixing=vmix._replace(tke=tke),
            convection=convection))

    # The external mode is a registered upstream shared debt.  Substitute its
    # NEMO endpoint over the measured rank-zero slab, while retaining a full
    # global production card.  The east partition support column and north
    # fold support row are excluded below because rank zero did not dump their
    # remote operands.
    eta_after = np.asarray(card.recipe.initial_state.eta.data).copy()
    hu_avg = np.zeros(card.recipe.initial_state.u.data.shape[:2], dtype=np.float64)
    hv_avg = np.zeros(card.recipe.initial_state.v.data.shape[:2], dtype=np.float64)
    eta_after[:, :OWNED_NX] = final_ssh
    # legoESM retains redundant west/south faces; NEMO's native east/north
    # faces map to indices 1: in those layouts.
    hu_avg[:, 1:OWNED_NX + 1] = un_adv
    hv_avg[1:OWNED_NY + 1, :OWNED_NX] = vn_adv
    zeros = np.zeros(eta_after.shape, dtype=np.float64)
    runoff = zeros.copy()
    runoff[:, :OWNED_NX] = wzv["runoff"]
    freshwater = FreshwaterForcing(
        precip=jnp.asarray(zeros), evap=jnp.asarray(zeros),
        runoff=jnp.asarray(runoff), ice_fw=jnp.asarray(zeros),
        restoring=jnp.asarray(zeros),
    )
    hooks = _NEMOWSRK3TestHooks(
        external_mode_result_override=(
            jnp.asarray(eta_after), jnp.asarray(hu_avg), jnp.asarray(hv_avg)),
        expose_stage1_wzv=True,
        expose_tracer_transport_stage=1,
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, diagnostic_cfg,
        _nemo_ws_test_hooks=hooks,
    )
    result = model.step(
        card.recipe.initial_state, card.dt_s, freshwater=freshwater)
    candidate = np.asarray(result.T.data)[:, :OWNED_NX, :NLEV]
    oracle = wzv["ww"][..., :NLEV]
    live = wzv["tmask"][..., :NLEV] != 0.0
    support = np.zeros((OWNED_NY, OWNED_NX), dtype=bool)
    support[:-1, 1:-1] = True
    mask = live & support[..., None]
    row = score(candidate, oracle, mask)

    pfu_oracle = wzv["pFu_stencil"][1:, 1:, :NLEV]
    pfv_oracle = wzv["pFv_stencil"][1:, 1:, :NLEV]
    pfu_candidate = np.asarray(result.u.data)[:, 1:OWNED_NX + 1, :NLEV]
    pfv_candidate = np.asarray(result.v.data)[1:OWNED_NY + 1, :OWNED_NX, :NLEV]
    # WZV's T-cell live mask is sufficient for the causal decomposition: score
    # the east/north face consumed by each live T cell on the same safe slab.
    transport_rows = {
        "zFu": score(pfu_candidate, pfu_oracle, mask),
        "zFv": score(pfv_candidate, pfv_oracle, mask),
    }

    # Clock-only discriminator.  Hold the recorded face transports, runoff,
    # mask and geometry fixed, but use the production pair: full external SSH
    # endpoint with the full 10,800 s denominator.  This is the composition the
    # old Phase-2k ablation failed to measure.
    fu = wzv["pFu_stencil"]
    fv = wzv["pFv_stencil"]
    fu_c, fu_w = fu[1:, 1:, :], fu[1:, :-1, :]
    fv_c, fv_s = fv[1:, 1:, :], fv[:-1, 1:, :]

    def clock_program(fu_, fuw_, fv_, fvs_, e3_, e30_, tmask_, area_, rnf_,
                      eta_b_, eta_a_):
        levels = []
        for jk in range(NLEV):
            levels.append(nemo_transport_wzv_divergence_level(
                fu_[..., jk], fuw_[..., jk], fv_[..., jk], fvs_[..., jk],
                area_, e3_[..., jk], tmask_[..., jk],
                runoff_mass_flux=(rnf_ if jk == 0 else None)))
        flux_div = jnp.stack(levels, axis=-1)
        h0 = jnp.zeros_like(eta_b_)
        for jk in range(NLEV):
            h0 = jax.lax.optimization_barrier(
                h0 + e30_[..., jk] * tmask_[..., jk])
        r1_h0 = jax.lax.optimization_barrier(
            1.0 / jnp.where(h0 > 0.0, h0, 1.0))
        r3_b = jax.lax.optimization_barrier(eta_b_ * r1_h0)
        r3_a = jax.lax.optimization_barrier(eta_a_ * r1_h0)
        return nemo_qco_wzv_recurrence(
            flux_div, e30_[..., :NLEV], r3_b, r3_a,
            tmask_[..., :NLEV], jnp.asarray(10800.0, e3_.dtype))

    eta_before_rank0 = np.asarray(card.recipe.initial_state.eta.data)[:, :OWNED_NX]
    full_clock_w = np.asarray(jax.jit(clock_program)(
        *map(jnp.asarray, (
            fu_c, fu_w, fv_c, fv_s, wzv["e3t"], wzv["e3t0"], wzv["tmask"],
            wzv["r1_area"], wzv["runoff"], eta_before_rank0, final_ssh))))
    clock_row = score(full_clock_w[..., :NLEV], oracle, mask)
    if plant:
        exact_control = oracle.copy()
        index = tuple(np.argwhere(mask)[0])
        exact_control[index] = np.nextafter(exact_control[index], np.inf)
        planted = score(exact_control, oracle, mask)
        require(planted["unequal"] == 1, "production-W plant did not fire once")
        raise GateError(
            "planted production stage-1 ww rejected through scorer "
            f"({planted['unequal']}/{planted['count']})")

    # Binding selector plant: validator must reject the alternative KEG arm.
    bad_cfg = cfg._replace(ke_gradient_scheme="hollingsworth")
    bad_recipe = card.recipe._replace(model_config=bad_cfg, physics_config=bad_cfg.physics)
    selector_plant_rejected = False
    try:
        validate_nemo_testcase_card(card._replace(recipe=bad_recipe))
    except ValueError:
        selector_plant_rejected = True
    require(selector_plant_rejected, "vector/C2 selector plant was accepted")

    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O5-B/compiled-production-stage1-tracer-ww",
        "result": row["status"],
        "row": row,
        "production_transport_rows": transport_rows,
        "card_metric_rows": metric_rows,
        "clock_only_full_endpoint_row": clock_row,
        "owner": (
            "CONFIRMED_PRODUCTION_WZV_AT_BAR"
            if row["status"] == "AT_BAR"
            else "UNASSIGNED_PRODUCTION_WZV_DEBT"
        ),
        "records": {name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in paths.items()},
        "operand_substitution": {
            "label": "ORACLE_SUPPLIED_EXTERNAL_MODE",
            "fields": ["final_ssh", "un_adv", "vn_adv"],
            "certifies_external_mode": False,
            "ignored_upstream_constructibility_proxy": (
                "TEOS-10 bn2 replaces unsupported EOS-80 bn2 only before the "
                "oracle external endpoint; EEN avg4 likewise uses the "
                "constructible min fold and tripolar div-curl viscosity is "
                "zeroed upstream; no proxy value reaches _g0"
            ),
        },
        "comparison_domain": {
            "description": (
                "rank0 wet T cells, 30 active levels, excluding east MPI "
                "support column and north-fold support row"
            ),
            "excluded_support_cells_2d": int(OWNED_NY * OWNED_NX - support.sum()),
        },
        "selectors": {
            "namelist": {"ln_dynadv_vec": True, "nn_dynkeg": 0},
            "card": {
                "momentum_advection": cfg.momentum_advection,
                "ke_gradient_scheme": cfg.ke_gradient_scheme,
            },
            "selector_plant_rejected": selector_plant_rejected,
        },
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
            "dt_s": card.dt_s,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, plant=args.plant)
    except (GateError, ValueError, IndexError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
