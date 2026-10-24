#!/usr/bin/env python3
"""Offline kt-1 stage-3 dyn_zdf replay from the passive round-328 record."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round328_dynzdf_gate as admission,
    nemo_testcase_l4_orca2_round328_zdf_terms_gate as zdf_admission,
)

DT = np.float64(10800.0)
RHO0 = np.float64(1026.0)
COMPONENTS = ("u", "v")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values).view(np.uint64)


def score(actual, expected, support=None) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}")
    selected = np.ones(actual.shape, dtype=bool) if support is None else np.asarray(support, bool)
    require(selected.shape == actual.shape, "score support shape differs")
    unequal = (bits(actual) != bits(expected)) & selected
    count = int(np.count_nonzero(unequal))
    if count:
        locations = np.argwhere(unequal)
        first = [int(value) for value in locations[0]]
        delta = np.abs(actual[unequal] - expected[unequal])
        maximum = float(np.max(delta))
        max_at = [int(value) for value in locations[int(np.argmax(delta))]]
    else:
        first = max_at = None
        maximum = 0.0
    return {
        "bit_identical": count == 0,
        "unequal": count,
        "compared": int(np.count_nonzero(selected)),
        "first_unequal_jik": first,
        "max_abs": maximum,
        "max_abs_jik": max_at,
    }


def native(values: np.ndarray) -> np.ndarray:
    """Fortran (i,j,k) record layout to legoESM/Numpy (j,i,k)."""
    return np.asarray(values).transpose(1, 0, 2)


def mesh_fields(root: Path, rank: int) -> dict[str, np.ndarray]:
    with Dataset(root / f"mesh_mask_{rank:04d}.nc") as dataset:
        return {
            name: np.asarray(dataset.variables[name][0]).transpose(1, 2, 0).astype(np.float64)
            for name in ("umask", "vmask", "e3u_0", "e3v_0")
        } | {"e3w_1d": np.asarray(dataset.variables["e3w_1d"][0], dtype=np.float64)}


def pair_sum(record_field: np.ndarray, component: str, nx: int, ny: int) -> np.ndarray:
    if component == "u":
        result = record_field[1:nx + 1, :ny, 0] + record_field[:nx, :ny, 0]
    else:
        result = record_field[:nx, 1:ny + 1, 0] + record_field[:nx, :ny, 0]
    return result.T


def avm_face(avm: np.ndarray, component: str, nx: int, ny: int, nz: int) -> np.ndarray:
    if component == "u":
        total = avm[1:nx + 1, :ny, 1:nz] + avm[:nx, :ny, 1:nz]
    else:
        total = avm[:nx, 1:ny + 1, 1:nz] + avm[:nx, :ny, 1:nz]
    return (np.float64(0.5) * total).transpose(1, 0, 2)


def level_masks(wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = wet.copy()
    top[..., 1:] *= (wet[..., :-1] == 0.0)
    bottom = wet.copy()
    bottom[..., :-1] *= (wet[..., 1:] == 0.0)
    return top, bottom


def replay_rank(record: dict, mesh: dict[str, np.ndarray]) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import rk3_stage_velocity_update
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_dynzdf_drag_rhs
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_nemo_momentum,
    )

    fields = record["fields"]
    header = record["header"]
    nx, ny, nz = (header[name] for name in ("nx", "ny", "nz"))
    rows: dict[str, object] = {}

    for component in COMPONENTS:
        wet = mesh[f"{component}mask"][..., :nz]
        before = native(fields[f"before_{component}"])
        rhs = native(fields[f"rhs_{component}"])
        explicit_expected = native(fields[f"explicit_{component}"])
        explicit = jax.device_get(jax.jit(lambda a, b, m: rk3_stage_velocity_update(
            a, b, DT, m, vector_form=True, written_order=True,
        ))(jnp.asarray(before), jnp.asarray(rhs), jnp.asarray(wet)))
        rows[f"{component}_explicit"] = score(explicit, explicit_expected, wet > 0.0)

        bt = native(fields[f"barotropic_{component}"])
        baro_expected = native(fields[f"baro_sub_{component}"])
        baro = jax.device_get(jax.jit(lambda a, b, m: (a - b) * m)(
            jnp.asarray(explicit), jnp.asarray(bt), jnp.asarray(wet)))
        rows[f"{component}_barotropic_subtraction"] = score(
            baro, baro_expected, wet > 0.0)

        e3_ref = mesh[f"e3{component}_0"][..., :nz]
        r3_kaa = native(fields[f"r3{component}_kaa"])[..., 0]
        r3_kmm = native(fields[f"r3{component}_kmm"])[..., 0]
        dz_after = e3_ref * (np.float64(1.0) + r3_kaa[..., None] * wet)
        e3w_now = mesh["e3w_1d"][1:nz][None, None, :] * (
            np.float64(1.0) + r3_kmm[..., None])
        top_mask, bottom_mask = level_masks(wet)
        zdt2 = jnp.asarray(DT * np.float64(0.5), dtype=jnp.float64)

        bot_pair = pair_sum(fields["rCdU_bot"], component, nx, ny)
        bot_delta = jax.device_get(nemo_dynzdf_drag_rhs(
            zdt2, jnp.asarray(bot_pair), jnp.asarray(bt), jnp.asarray(dz_after)))
        bottom = jax.device_get(nemo_source_round(
            jnp.asarray(baro) + jnp.asarray(bot_delta) * jnp.asarray(bottom_mask)))
        bottom_expected = native(fields[f"bottom_rhs_{component}"])
        rows[f"{component}_bottom_rhs"] = score(bottom, bottom_expected, wet > 0.0)

        top_pair = pair_sum(fields["rCdU_top"], component, nx, ny)
        top_delta = jax.device_get(nemo_dynzdf_drag_rhs(
            zdt2, jnp.asarray(top_pair), jnp.asarray(bt), jnp.asarray(dz_after)))
        top_rhs = jax.device_get(nemo_source_round(
            jnp.asarray(bottom) + jnp.asarray(top_delta) * jnp.asarray(top_mask)))
        top_expected = native(fields[f"top_rhs_{component}"])
        rows[f"{component}_top_rhs"] = score(top_rhs, top_expected, wet > 0.0)

        base = tuple(native(fields[f"{component}_base_zdf_{term}"])
                     for term in ("low", "diag", "up"))
        after_zad = tuple(native(fields[f"{component}_after_zad_{term}"])
                          for term in ("low", "diag", "up"))
        after_ldf = tuple(native(fields[f"{component}_after_ldf_{term}"])
                          for term in ("low", "diag", "up"))
        rows[f"{component}_zad_inactive"] = {
            "bit_identical": all(np.array_equal(bits(a), bits(b))
                                 for a, b in zip(base, after_zad)),
        }
        rows[f"{component}_ldf_inactive"] = {
            "bit_identical": all(np.array_equal(bits(a), bits(b))
                                 for a, b in zip(after_zad, after_ldf)),
        }

        avm = avm_face(fields["avm_input"], component, nx, ny, nz)
        bottom_diag = native(fields[f"{component}_bottom_diag_diag"])
        top_diag = native(fields[f"{component}_top_diag_diag"])
        extra_diag = top_diag - base[1]
        stress = native(fields["utauU" if component == "u" else "vtauV"])[..., 0]
        r = nemo_source_round
        numerator = r(jnp.asarray(DT) * jnp.asarray(stress))
        divisor = r(jnp.asarray(dz_after[..., 0]) * jnp.asarray(RHO0))
        divisor = jnp.where(divisor == 0.0, 1.0, divisor)
        wind = jax.device_get(r(r(numerator / divisor) * jnp.asarray(wet[..., 0])))
        solve_in = np.array(top_rhs, copy=True)
        solve_in[..., 0] += wind
        solve = jax.device_get(jax.jit(implicit_vertical_diffusion_nemo_momentum)(
            jnp.asarray(solve_in), jnp.asarray(avm), jnp.asarray(dz_after),
            jnp.asarray(e3w_now), DT, jnp.asarray(wet > 0.0),
            extra_diag=jnp.asarray(extra_diag)))
        solve_expected = native(fields[f"solve_{component}"])
        rows[f"{component}_base_plus_drag_ordered_solve"] = score(
            solve, solve_expected, wet > 0.0)
        rows[f"{component}_matrix_phase_changes"] = {
            "bottom_diag_unequal": int(np.count_nonzero(bits(bottom_diag) != bits(after_ldf[1]))),
            "top_diag_unequal": int(np.count_nonzero(bits(top_diag) != bits(bottom_diag))),
        }

    order = [
        *(f"{c}_explicit" for c in COMPONENTS),
        *(f"{c}_barotropic_subtraction" for c in COMPONENTS),
        *(f"{c}_bottom_rhs" for c in COMPONENTS),
        *(f"{c}_top_rhs" for c in COMPONENTS),
        *(f"{c}_base_plus_drag_ordered_solve" for c in COMPONENTS),
    ]
    first = next((name for name in order if not rows[name]["bit_identical"]), None)
    return {"rows": rows, "first_non_bit_boundary": first}


def assemble_record_avm(records: dict[int, dict]) -> np.ndarray:
    slabs = []
    for rank in (0, 1):
        record = records[rank]
        nx, ny, nz = (record["header"][name] for name in ("nx", "ny", "nz"))
        slabs.append(record["fields"]["avm_input"][:nx, :ny, 1:nz])
    return np.concatenate(slabs, axis=0).transpose(1, 0, 2)


def read_record_tracer_profiles(root: Path) -> dict[str, np.ndarray]:
    """Read the admitted rank-0 final avt/avs record written after zdf_phy."""
    path = root / "oracle_zdf_entry_kt00000001.bin"
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        payload = np.fromfile(handle, dtype=np.float64)
    version, kt, level, nx, ny, nz, real_bits = header
    require(magic == "NEMO_L4_ZDF___2", f"bad ZDF-entry magic {magic!r}")
    require((version, kt, level, real_bits) == (2, 1, 1, 64),
            f"bad ZDF-entry header {header!r}")
    halo = 2
    full = nx * ny * nz
    interior = (nx - 2 * halo) * (ny - 2 * halo) * nz
    require(payload.size == full + 3 * interior,
            f"ZDF-entry payload {payload.size} != {full + 3 * interior}")
    require(np.isfinite(payload).all(), "non-finite ZDF-entry payload")
    offset = full
    fields = {}
    for name in ("avt", "avs", "en"):
        native_field = payload[offset:offset + interior].reshape(
            (nx - 2 * halo, ny - 2 * halo, nz), order="F")
        fields[name] = native_field.transpose(1, 0, 2)
        offset += interior
    require(offset == payload.size, "ZDF-entry parser did not reach EOF")
    return fields


def vertical_counts(mask: np.ndarray) -> list[int]:
    return [int(value) for value in np.count_nonzero(mask, axis=(0, 1))]


def term_localization(candidate: np.ndarray, record: np.ndarray,
                      after_ddm: np.ndarray, closure: np.ndarray,
                      active: np.ndarray) -> dict[str, object]:
    """Classify final-avm residuals against the two source-order term boundaries."""
    unequal = (bits(candidate) != bits(record)) & active
    no_iwm_equal = (bits(after_ddm) == bits(record)) & unequal
    closure_equal = (bits(closure) == bits(record)) & unequal
    iwm_changed = (bits(candidate) != bits(after_ddm)) & active
    ddm_changed = (bits(after_ddm) != bits(closure)) & active
    greater = (candidate > record) & unequal
    return {
        "unequal_by_level": vertical_counts(unequal),
        "candidate_greater_than_record": int(np.count_nonzero(greater)),
        "candidate_greater_by_level": vertical_counts(greater),
        "record_equals_after_ddm_without_iwm": int(np.count_nonzero(no_iwm_equal)),
        "record_equals_closure_without_ddm_or_iwm": int(np.count_nonzero(closure_equal)),
        "iwm_changes_candidate_at_residual": int(np.count_nonzero(iwm_changed & unequal)),
        "ddm_changes_candidate_at_residual": int(np.count_nonzero(ddm_changed & unequal)),
        "iwm_changes_at_residual_by_level": vertical_counts(iwm_changed & unequal),
        "ddm_changes_at_residual_by_level": vertical_counts(ddm_changed & unequal),
        "iwm_unchanged_at_residual": int(np.count_nonzero(~iwm_changed & unequal)),
        "ddm_unchanged_at_residual": int(np.count_nonzero(~ddm_changed & unequal)),
    }


def assemble_zdf_term_field(records: dict[int, dict], name: str) -> np.ndarray:
    """Assemble rank-local NEMO (i,j,k) ZDF fields as global (j,i,k)."""
    return np.concatenate(
        [records[rank]["fields"][name] for rank in (0, 1)], axis=0
    ).transpose(1, 0, 2)


def zdf_term_localization(candidates: dict[str, np.ndarray],
                          records: dict[int, dict],
                          active: np.ndarray) -> dict[str, object]:
    """Name the first unequal source-order ZDF term boundary at each level."""
    ordered = (("post_tke", "b1_avm"), ("post_ddm", "b2_avm"),
               ("post_iwm", "b3_avm"))
    record_wet = assemble_zdf_term_field(records, "tke_wmask") != 0.0
    require(record_wet.shape == active.shape, "ZDF record wet-mask shape differs")
    require(np.array_equal(record_wet, active), "ZDF record/model wet masks differ")
    rows = {}
    for boundary, field in ordered:
        expected = assemble_zdf_term_field(records, field)
        require(candidates[boundary].shape == expected.shape,
                f"{boundary} candidate/record shapes differ")
        rows[boundary] = score(candidates[boundary], expected, active)
        unequal = (bits(candidates[boundary]) != bits(expected)) & active
        rows[boundary]["unequal_by_level"] = vertical_counts(unequal)
    first_by_level = []
    for level in range(active.shape[-1]):
        first = next((name for name, _ in ordered
                      if rows[name]["unequal_by_level"][level] != 0), None)
        first_by_level.append(first)
    return {"boundaries": rows, "first_unequal_boundary_by_level": first_by_level}


def score_by_level(actual: np.ndarray, expected: np.ndarray,
                   support: np.ndarray) -> dict[str, object]:
    result = score(actual, expected, support)
    unequal = (bits(np.asarray(actual)) != bits(np.asarray(expected))) & support
    result["unequal_by_level"] = vertical_counts(unequal)
    return result


def compose_literal_terms(model, card, closure_heat, closure_momentum, active):
    """Apply the model's certified DDM then IWM statements to its carried TKE K."""
    import jax
    import jax.numpy as jnp

    from legoesm.ocean.physics.vertical_mixing.double_diffusion import nemo_zdf_ddm
    from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import nemo_zdf_iwm

    bundle = model._tke_step_entry_n2_bundle(card.recipe.initial_state)
    require(bundle.ddm_operands is not None, "TKE bundle lacks DDM operands")
    require(bundle.iwm_stretch is not None, "TKE bundle lacks IWM stretch")
    zdt, zds = bundle.ddm_operands
    vmix = card.recipe.model_config.physics.vertical_mixing
    after_ddm = nemo_zdf_ddm(
        jnp.asarray(closure_heat), jnp.asarray(closure_momentum),
        zdt, zds, bundle.rn2, jnp.asarray(active), vmix.ddm)
    e3w = np.asarray(card.recipe.z_coord.nemo_e3w_0)
    e3w_cols = e3w.reshape(-1, e3w.shape[-1])
    require(np.array_equal(e3w_cols, np.broadcast_to(
        e3w_cols[:1], e3w_cols.shape)), "nemo_e3w_0 is not column-uniform")
    final = nemo_zdf_iwm(
        *after_ddm, card.recipe.iwm_forcing, bundle.rn2,
        bundle.iwm_stretch, card.recipe.initial_state.H_bathy.data,
        np.asarray(card.recipe.z_coord.t_depth_ref), e3w_cols[0],
        jnp.asarray(active), cfg=vmix.iwm,
        rho_0=card.recipe.model_config.physics.constants.rho_0)
    return tuple(np.asarray(value, dtype=np.float64)
                 for value in jax.device_get((*after_ddm, *final)))


def _with_vmix_terms(config, *, ddm: bool, iwm: bool):
    vmix = config.physics.vertical_mixing
    vmix = vmix._replace(
        ddm=vmix.ddm._replace(enabled=ddm),
        iwm=vmix.iwm._replace(enabled=iwm),
    )
    return config._replace(
        physics=config.physics._replace(vertical_mixing=vmix))


def _profile_setup(model, card, freshwater, surface, sponge):
    state = card.recipe.initial_state
    bundle = model._tke_step_entry_n2_bundle(state)
    state_expl, operands = model._step_impl(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface,
        sponge=sponge,
        _apply_implicit_vmix=False,
        _tke_n2_bundle_override=bundle)
    return state, bundle, state_expl, operands


def _diagnose_profiles(model, card, freshwater, surface, sponge):
    import jax

    state, bundle, state_expl, operands = _profile_setup(
        model, card, freshwater, surface, sponge)
    (K_v_phys, A_v_phys, _k33, _surface_tracer_forcing, tke_source,
     _diss_incr, _tracer_source) = operands
    tke_old = state.tke.data if state.tke is not None else None
    heat, momentum = jax.device_get(model._apply_implicit_vertical_mixing(
        state_expl, card.dt_s, surface,
        K_v_phys=K_v_phys, A_v_phys=A_v_phys,
        dt_mom=card.dt_s / model.config.dt_mom_ratio,
        tke_old=tke_old, tke_source=tke_source,
        n2_tracers=model._n2_before_advection_tracers(state),
        n2_tracers_before=model._n2_nemo_before_tracers(state),
        tke_n2_bundle=bundle, eta_now=state.eta.data,
        eta_step_entry=state.eta.data, u_now=state.u.data,
        v_now=state.v.data, return_K_profiles=True))
    return (np.asarray(heat, dtype=np.float64),
            np.asarray(momentum, dtype=np.float64))


def _diagnose_salt_profile(model, card, freshwater, surface, sponge):
    """Read avs before lateral K33 from the model's own literal solve setup."""
    import jax

    state, bundle, state_expl, operands = _profile_setup(
        model, card, freshwater, surface, sponge)
    (K_v_phys, A_v_phys, _k33, surface_tracer_forcing, tke_source,
     _diss_incr, tracer_source) = operands
    tke_old = state.tke.data if state.tke is not None else None
    state_out, trace = jax.device_get(model._apply_implicit_vertical_mixing(
        state_expl, card.dt_s, surface,
        K_v_phys=K_v_phys, A_v_phys=A_v_phys, K33_iso=None,
        dt_mom=card.dt_s / model.config.dt_mom_ratio,
        surface_tracer_forcing=surface_tracer_forcing,
        tracer_source=tracer_source, tke_old=tke_old, tke_source=tke_source,
        n2_tracers=model._n2_before_advection_tracers(state),
        n2_tracers_before=model._n2_nemo_before_tracers(state),
        tke_n2_bundle=bundle, eta_now=state.eta.data,
        eta_step_entry=state.eta.data, u_now=state.u.data,
        v_now=state.v.data, return_tracer_solve_trace=True))
    del state_out
    require(trace.salt_K is not None,
            "literal tracer solve did not expose its consumed avs")
    return np.asarray(trace.salt_K, dtype=np.float64)


def base_lower_from_avm(avm, dz_after, e3w_now, wet):
    """The production literal solver's first base-matrix coefficient."""
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round

    r = nemo_source_round
    wet_f = jnp.asarray(wet, dtype=jnp.float64)
    interface_wet = wet_f[..., 1:] * wet_f[..., :-1]
    zdt2 = jnp.asarray(0.5, dtype=jnp.float64) * jnp.asarray(DT)
    avm_sum = jnp.asarray(2.0, dtype=jnp.float64) * jnp.asarray(avm)
    product = r(-zdt2 * avm_sum)
    denominator = r(jnp.asarray(dz_after)[..., 1:] * jnp.asarray(e3w_now))
    coefficient = r(product / denominator) * interface_wet
    return jnp.concatenate([jnp.zeros_like(coefficient[..., :1]), coefficient], axis=-1)


def model_avm(root: Path, records: dict[int, dict],
              zdf_records: dict[int, dict] | None = None) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.grids.operators_latlon_cgrid import interp_cell_to_uface
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import interp_to_v_points

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    _, shipped = rung10.card_fields(root)
    card = rung10.comparison_card(shipped)
    surface_fields = rung10.assemble_surface_fields(root, 1)
    freshwater, surface = rung10._surface_forcings(card, root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing)
    step_kwargs = rung10.damping_step_kwargs(card, root, 1)
    state_after = jax.device_get(model.step(
        card.recipe.initial_state, card.dt_s, freshwater=freshwater,
        surface_forcing=surface, **step_kwargs))
    require(state_after.tke_avm is not None,
            "ordinary comparison step did not return its closure avm")
    require(state_after.tke_avt is not None,
            "ordinary comparison step did not return its closure avt")
    closure_state = np.asarray(state_after.tke_avm.data, dtype=np.float64)
    closure_heat_state = np.asarray(state_after.tke_avt.data, dtype=np.float64)
    record_avm = assemble_record_avm(records)
    active = np.array(card.recipe.z_coord.is_active[..., 1:], dtype=bool, copy=True)
    active &= np.asarray(card.recipe.z_coord.is_active[..., :-1], dtype=bool)
    (heat_after_ddm, salt_after_ddm, consumed_after_ddm,
     consumed_heat, salt, consumed) = compose_literal_terms(
        model, card, closure_heat_state, closure_state, active)
    require(consumed.shape == record_avm.shape,
            f"model avm {consumed.shape} != record {record_avm.shape}")
    require(active.shape == consumed.shape, "active-interface shape differs")
    row = score(consumed, record_avm, active)
    ice = np.asarray(surface.ice_concentration) > 0.0
    unequal = (bits(consumed) != bits(record_avm)) & active
    under_ice = unequal & ice[..., None]
    row["unequal_under_ice"] = int(np.count_nonzero(under_ice))
    row["unequal_open_water"] = int(np.count_nonzero(unequal & ~ice[..., None]))
    if np.any(under_ice):
        row["first_under_ice_jik"] = [int(v) for v in np.argwhere(under_ice)[0]]
    else:
        row["first_under_ice_jik"] = None
    row["backend"] = jax.default_backend()
    row["dtype"] = str(consumed.dtype)
    row["term_localization"] = term_localization(
        consumed, record_avm, consumed_after_ddm, closure_state, active)
    row["profile_phase"] = "ordinary step carried TKE closure plus literal DDM/IWM"

    tracer_record = read_record_tracer_profiles(root)
    rank0_active = active[:, :tracer_record["avt"].shape[1], :]
    record_avt = tracer_record["avt"][..., 1:-1]
    record_avs = tracer_record["avs"][..., 1:-1]
    record_en = tracer_record["en"][..., 1:-1]
    require(record_avt.shape == consumed_heat[:, :record_avt.shape[1]].shape,
            "rank-0 avt record differs from the consumed model profile")
    require(record_avs.shape == salt[:, :record_avs.shape[1]].shape,
            "rank-0 avs record differs from the consumed model profile")
    tracer_rows = {
        "avt_rank0": score(
            consumed_heat[:, :record_avt.shape[1]], record_avt, rank0_active),
        "avs_rank0": score(
            salt[:, :record_avs.shape[1]], record_avs, rank0_active),
    }
    if state_after.tke is not None:
        model_en = np.asarray(state_after.tke.data, dtype=np.float64)
        require(record_en.shape == model_en[:, :record_en.shape[1]].shape,
                "rank-0 en record differs from the model TKE shape")
        tracer_rows["en_rank0"] = score(
            model_en[:, :record_en.shape[1]], record_en, rank0_active)
    candidate_faces = {
        "u": np.asarray(interp_cell_to_uface(jnp.asarray(consumed)))[:, 1:],
        "v": np.asarray(interp_to_v_points(
            jnp.asarray(consumed), card.recipe.grid))[1:],
    }
    first_statement = {}
    for component in COMPONENTS:
        rank_rows = []
        for rank in (0, 1):
            record = records[rank]
            header, fields = record["header"], record["fields"]
            nx, ny, nz = (header[name] for name in ("nx", "ny", "nz"))
            mesh = mesh_fields(root, rank)
            wet = mesh[f"{component}mask"][..., :nz]
            r3_kaa = native(fields[f"r3{component}_kaa"])[..., 0]
            r3_kmm = native(fields[f"r3{component}_kmm"])[..., 0]
            dz_after = mesh[f"e3{component}_0"][..., :nz] * (
                np.float64(1.0) + r3_kaa[..., None] * wet)
            e3w_now = mesh["e3w_1d"][1:nz][None, None, :] * (
                np.float64(1.0) + r3_kmm[..., None])
            i0 = rank * nx
            face_avm = candidate_faces[component][:ny, i0:i0 + nx]
            actual = np.asarray(jax.jit(base_lower_from_avm)(
                jnp.asarray(face_avm), jnp.asarray(dz_after),
                jnp.asarray(e3w_now), jnp.asarray(wet)))
            expected = native(fields[f"{component}_base_zdf_low"])
            support = np.zeros_like(wet, dtype=bool)
            support[..., 1:] = (wet[..., 1:] * wet[..., :-1]) > 0.0
            scored = score(actual, expected, support)
            scored["rank"] = rank
            if scored["first_unequal_jik"] is not None:
                global_first = list(scored["first_unequal_jik"])
                global_first[1] += i0
                scored["first_unequal_global_jik"] = global_first
            else:
                scored["first_unequal_global_jik"] = None
            rank_rows.append(scored)
        first_statement[component] = rank_rows
    result = {
        "coefficient": row,
        "tracer_coefficients": tracer_rows,
        "first_base_lower_statement": first_statement,
    }
    if zdf_records is not None:
        result["zdf_term_boundaries"] = zdf_term_localization({
            "post_tke": closure_state,
            "post_ddm": consumed_after_ddm,
            "post_iwm": consumed,
        }, zdf_records, active)
        record_en = assemble_zdf_term_field(zdf_records, "tke_en")
        model_en = np.asarray(state_after.tke.data, dtype=np.float64)
        bundle = model._tke_step_entry_n2_bundle(card.recipe.initial_state)
        model_rn2 = np.asarray(bundle.rn2, dtype=np.float64)
        record_rn2 = assemble_zdf_term_field(zdf_records, "tke_rn2")
        require(model_en.shape == record_en.shape == active.shape,
                "post-TKE energy shape differs from ZDF record")
        require(model_rn2.shape == record_rn2.shape == active.shape,
                "TKE rn2 shape differs from ZDF record")
        result["zdf_tke_operands"] = {
            "post_solve_en": score_by_level(model_en, record_en, active),
            "entry_rn2": score_by_level(model_rn2, record_rn2, active),
        }
    return result


def measure(root: Path, expect_commit: str, expect_tree: str) -> dict[str, object]:
    stamp = worktree_stamp(repo=REPO_ROOT)
    require(stamp["commit"] == expect_tree, "worktree commit differs from --expect-tree")
    require(stamp["clean"], "replay requires a clean tracked tree")
    admitted = admission.run_gate(root, expect_commit)
    refusals = {}
    for plant in admission.PLANTS:
        try:
            admission.run_gate(root, expect_commit, plant)
        except admission.GateError:
            refusals[plant] = True
        else:
            refusals[plant] = False
    require(all(refusals.values()), "one or more admission plants stayed green")
    records = {rank: admission.read_record(root / admission.record_name(rank))
               for rank in (0, 1)}
    zdf_admitted = zdf_admission.run_gate(root, expect_commit)
    zdf_records = {
        rank: zdf_admission.read_record(root / zdf_admission.record_name(rank))
        for rank in (0, 1)
    }
    replay = {str(rank): replay_rank(records[rank], mesh_fields(root, rank))
              for rank in (0, 1)}
    return {
        "format": "nemo-testcase-l4-orca2-round328-dynzdf-replay-v1",
        "worktree": stamp,
        "record_admission": admitted,
        "admission_refusals": refusals,
        "rank_replays": replay,
        "zdf_term_admission": zdf_admitted,
        "model_avm_vs_record": model_avm(root, records, zdf_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-tree", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.record_dir, args.expect_commit, args.expect_tree)
    except (GateError, admission.GateError, OSError, ValueError) as error:
        print(f"REFUSE: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_R328_DYNZDF_OFFLINE_REPLAY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
