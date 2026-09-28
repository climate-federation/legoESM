#!/usr/bin/env python3
"""Year-long pointwise gate for the C1D_OMIP_L3 SI3 ice-side bulk fluxes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import struct
from pathlib import Path

import jax
import jaxlib
import numpy as np

BAR = 1.0e-15
EXPECTED_STEPS = 8760
MAGIC = b"NEMO_L3BULK_001 "
EXPECTED_BULK_SHA256 = "57868f3212646bdf6b0c4add0f48701c0082718331bc76a153050c6d1d44dfe9"
EXPECTED_EXCHANGE_SHA256 = "091395cf604e83d88fbf458c4ef76ac3df2d5a65dac9cdc502e224cc1d1af2e4"
ORACLE_VERSION = "V2_SCALAR_MATH"
ACCEPTED_RUNTIME = {
    "python": "3.13.0",
    "jax": "0.10.0",
    "jaxlib": "0.10.0",
    "numpy": "2.4.4",
}
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l3/"
    "c1d_omip_l3_sasice_scalarmath_v2_a"
)

STAGE_NAMES = {0: "POST_BLK_ICE_1", 1: "POST_BLK_ICE_2", 2: "POST_ICE_FLX_OTHER"}
STAGE_COUNTS = {0: 17, 1: 50, 2: 39}
STAGE_SOURCES = {
    0: "icesbc.F90:83-108; sbcblk.F90:1085-1168",
    1: "icesbc.F90:149-194; icealb.F90:124-185; sbcblk.F90:1218-1346",
    2: "icesbc.F90:194-199,310-437",
}

STAGE0_NAMES = (
    "u_air", "v_air", "theta_air", "q_air", "p_surface", "T_surface",
    "rho_air", "wind", "Cd", "Ch", "Ce", "theta_zu", "q_zu", "fr_i",
    "mask", "utau_ice", "vtau_ice",
)
STAGE1_NAMES = (
    "T_surface", "h_snow", "h_ice", "pond_fraction", "pond_depth",
    "pond_lid", "cloud_fraction", "theta_air", "q_air", "p_surface",
    "lw_down", "precip", "snow", "qsr", "rho_air", "wind", "Ch", "Ce",
    "ice_fraction_before", "category_fraction_before", "sst_celsius",
    "qns_ocean", "qsr_ocean", "emp_ocean_raw", "fr_i", "albedo",
    "qsr_ice", "qla_ice", "dqla_ice", "qns_ice", "dqns_ice", "evap_ice",
    "devap_ice", "tprecip", "sprecip", "emp_oce", "emp_ice", "emp_tot",
    "qemp_oce", "qemp_ice", "qns_tot", "qsr_tot", "qprec_ice",
    "qevap_ice", "qtr_ice_top", "qsat_ice", "theta_ice", "qlw_ice",
    "qsb_ice", "dqlw_ice",
)
STAGE2_NAMES = (
    "ice_fraction", "ice_fraction_before", "ice_volume", "u_ice",
    "u_ice_west", "v_ice", "v_ice_south", "u_ocean", "u_ocean_west",
    "v_ocean", "v_ocean_south", "drag_io", "utau", "vtau", "frq",
    "qsr_ocean", "qns_ocean", "qemp_ocean", "ocean_layer_thickness",
    "sst_celsius", "T_bottom", "dt", "inverse_dt", "rho_ocean", "c_ocean",
    "T0", "ice_epsilon", "max_ice_fraction", "mask", "ln_icedyn",
    "ln_form_drag", "nn_form_drag", "ln_leadhfx", "ln_icedO", "ln_icedH",
    "drag_coefficient", "qsb_ice_bot", "fhld", "qlead",
)

# End-of-step exchange schema from config-local icestp.F90.  Status here means
# this rung's disposition, not a claim that later thermodynamic mutations equal
# the pre-thermodynamic bulk boundary.
COVERAGE = {
    "qns_ice": ("VERIFIED", "POST_BLK_ICE_2 output; end stream later mutated by ZDF"),
    "qsr_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "qla_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "dqla_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "dqns_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "tn_ice": ("WAIVED", "thermodynamic surface-temperature state"),
    "alb_ice": ("VERIFIED", "ice_alb output at POST_BLK_ICE_2; recomputed after thermodynamics"),
    "qml_ice": ("WAIVED", "ice_thd output, not bulk flux"),
    "qcn_ice": ("WAIVED", "ice_thd/ZDF output, not bulk flux"),
    "qtr_ice_top": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "utau_ice": ("VERIFIED", "POST_BLK_ICE_1 output"),
    "vtau_ice": ("VERIFIED", "POST_BLK_ICE_1 output"),
    "emp_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "evap_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "devap_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "qns_oce": ("VERIFIED", "registered input to ice_flx_other"),
    "qsr_oce": ("VERIFIED", "registered input to ice_flx_other"),
    "qemp_oce": ("VERIFIED", "POST_BLK_ICE_2 output and ice_flx_other input"),
    "qemp_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "qevap_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "qprec_ice": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "emp_oce": ("VERIFIED", "POST_BLK_ICE_2 output"),
    "wndm_ice": ("VERIFIED", "POST_BLK_ICE_1 output"),
    "sstfrz": ("WAIVED", "eos_fzp output upstream of this rung"),
    "rCdU_ice": ("WAIVED", "inactive because ln_drgice_imp is false in SAS"),
    "snwice_mass": ("WAIVED", "thermodynamic snow/ice exchange output"),
    "snwice_mass_b": ("WAIVED", "thermodynamic snow/ice exchange history"),
    "snwice_fmass": ("WAIVED", "thermodynamic snow/ice exchange output"),
    "utau": ("WAIVED", "not used by active ln_icedyn branch; ocean NCAR producer and later ice_update_tau uncertified"),
    "vtau": ("WAIVED", "not used by active ln_icedyn branch; ocean NCAR producer and later ice_update_tau uncertified"),
    "taum": ("WAIVED", "ocean-side NCAR diagnostic"),
    "qsr": ("VERIFIED", "registered blk_ice_2 atmospheric input"),
    "qns": ("WAIVED", "ocean-side NCAR output"),
    "emp": ("VERIFIED", "registered blk_ice_2 ocean evaporation input"),
    "sfx": ("WAIVED", "ice thermodynamic salt flux"),
    "fr_i": ("VERIFIED", "registered POST_BLK_ICE_1/2 input"),
}
EXPECTED_EXCHANGE_FIELDS = tuple(COVERAGE)

BIT_OWNER = {
    (stage, name): (
        "jax_exp_ice_alb"
        if stage == 1 and name in {"albedo", "qsr_ice", "qsr_tot"}
        else "binary64_operation_order"
    )
    for stage, names in (
        (0, ("wndm_ice", "utau_ice", "vtau_ice")),
        (1, STAGE1_NAMES[25:45]),
        (2, ("qsb_ice_bot", "fhld", "qlead")),
    )
    for name in names
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def runtime_versions() -> dict[str, str]:
    """Return the exact numeric runtime used to certify the bitwise counts."""
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
    }


def validate_runtime(runtime: dict[str, str]) -> None:
    """Fail closed when asked to reproduce version-sensitive bitwise evidence."""
    require(
        runtime == ACCEPTED_RUNTIME,
        f"unregistered numeric runtime: {runtime}; expected {ACCEPTED_RUNTIME}",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bulk_stream(path: Path) -> dict[int, np.ndarray]:
    rows = {stage: [] for stage in STAGE_NAMES}
    with path.open("rb") as stream:
        for expected_step in range(1, EXPECTED_STEPS + 1):
            for expected_stage in STAGE_NAMES:
                require(stream.read(16) == MAGIC, f"bulk magic at step {expected_step}")
                header = struct.unpack("=5i", stream.read(20))
                version, step, stage, count, bits = header
                require(
                    (version, step, stage, count, bits) == (
                        1, expected_step, expected_stage,
                        STAGE_COUNTS[expected_stage], 64,
                    ),
                    f"bulk registry {header}",
                )
                values = np.fromfile(stream, np.float64, count)
                require(values.size == count and np.all(np.isfinite(values)), "bulk payload")
                rows[stage].append(values)
        require(stream.read(1) == b"", "unregistered trailing bulk frame")
    return {stage: np.asarray(values) for stage, values in rows.items()}


def read_exchange_active(path: Path) -> dict[str, np.ndarray]:
    """Read every active C1D member using the canonical exchange schema."""
    sibling = Path(__file__).with_name("nemo_si3_exchange_drift_gate.py")
    spec = importlib.util.spec_from_file_location("_si3_exchange_schema", sibling)
    require(spec is not None and spec.loader is not None, "exchange schema import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layout, record_bytes = module._layout(5, 5, 1)
    require(record_bytes == 2120, "exchange record size")
    raw = np.memmap(path, np.uint8, mode="r", shape=(EXPECTED_STEPS, record_bytes))
    values: dict[str, list[float]] = {item.name: [] for item, _, _ in layout}
    active_index = {"reduced": 0, "halo1": 4, "full": 12}
    for index in range(EXPECTED_STEPS):
        module._read_header(memoryview(raw[index]), index + 1)
        for item, start, count in layout:
            field = np.frombuffer(raw[index], np.float64, count=count, offset=start)
            values[item.name].append(float(field[active_index[item.size]]))
    return {name: np.asarray(field) for name, field in values.items()}


def validate_coverage(coverage=COVERAGE) -> None:
    require(set(coverage) == set(EXPECTED_EXCHANGE_FIELDS),
            "exchange coverage register is incomplete")
    require(all(row[0] in ("VERIFIED", "WAIVED") and row[1] for row in coverage.values()),
            "exchange coverage disposition")


def _as_dict(values: np.ndarray, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: values[:, index] for index, name in enumerate(names)}


def _scalar_glibc_albedo_replay(stage1: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gate-only scalar-libm replay of the active no-pond SI3 albedo path."""
    from legoesm import constants

    inv_log_interval = 1.0 / (
        math.log(constants.albedo_ice_pivot_orca1)
        - math.log(constants.albedo_ice_thin_break_nemo)
    )
    log_pivot = math.log(constants.albedo_ice_pivot_orca1)
    albedo = []
    for surface, h_ice, h_snow, cloud in zip(
        stage1["T_surface"], stage1["h_ice"], stage1["h_snow"],
        stage1["cloud_fraction"], strict=True,
    ):
        snow_fraction = h_snow / (h_snow + constants.snow_cover_depth_nemo)
        bare_thick = (
            constants.albedo_ice_melt_orca1
            if h_snow == 0.0 and surface >= constants.T_freeze
            else constants.albedo_ice_dry_orca1
        )
        bare_mid = bare_thick + (
            constants.albedo_ice_thin_nemo - bare_thick
        ) * inv_log_interval * (log_pivot - math.log(h_ice))
        bare_thin = constants.albedo_ocean_nemo + (
            constants.albedo_ice_thin_nemo - constants.albedo_ocean_nemo
        ) * (1.0 / constants.albedo_ice_thin_break_nemo) * h_ice
        bare = (
            bare_thin if h_ice <= constants.albedo_ice_thin_break_nemo
            else bare_mid if h_ice <= constants.albedo_ice_pivot_orca1
            else bare_thick
        )
        if surface < constants.T_freeze:
            snow = constants.albedo_snow_dry_orca1 - (
                constants.albedo_snow_dry_orca1 - bare
            ) * math.exp(
                -h_snow * (1.0 / constants.albedo_snow_decay_dry_nemo)
            )
        else:
            snow = constants.albedo_snow_melt_orca1 - (
                constants.albedo_snow_melt_orca1 - bare
            ) * math.exp(
                -h_snow * (1.0 / constants.albedo_snow_decay_melt_nemo)
            )
        overcast = snow_fraction * snow + (1.0 - snow_fraction) * bare
        clear = overcast - (
            constants.albedo_cloud_quad_nemo * overcast * overcast
            + constants.albedo_cloud_linear_nemo * overcast
            + constants.albedo_cloud_offset_nemo
        )
        albedo.append((1.0 - cloud) * clear + cloud * overcast)
    albedo = np.asarray(albedo, dtype=np.float64)
    qsr_ice = np.asarray([
        (1.0 / (1.0 - constants.albedo_ocean_nemo)) * (1.0 - alb) * qsr
        for alb, qsr in zip(albedo, stage1["qsr"], strict=True)
    ], dtype=np.float64)
    qsr_total = np.asarray([
        (1.0 - fraction) * ocean + category * ice
        for fraction, ocean, category, ice in zip(
            stage1["ice_fraction_before"], stage1["qsr_ocean"],
            stage1["category_fraction_before"], qsr_ice, strict=True,
        )
    ], dtype=np.float64)
    return {"albedo": albedo, "qsr_ice": qsr_ice, "qsr_tot": qsr_total}


def _bit_owner_groups(rows: list[dict[str, object]], *, plant: bool = False) -> dict[str, object]:
    owners = dict(BIT_OWNER)
    if plant:
        owners.pop((1, "albedo"))
    require(set(owners) == set(BIT_OWNER), "bit-owner register is incomplete")
    grouped: dict[str, dict[str, object]] = {}
    for row in rows:
        key = next(
            key for key in owners
            if STAGE_NAMES[key[0]] == row["stage"] and key[1] == row["variable"]
        )
        count = int(row["non_bit_identical_count"])
        if not count:
            continue
        owner = owners[key]
        group = grouped.setdefault(owner, {
            "non_bit_identical_rows": 0,
            "max_relative_error_nonzero_oracle": 0.0,
            "exact_zero_oracle_non_bit_rows": 0,
            "variables": [],
            "largest_relative_row": None,
            "status": (
                "AWAITING_LIBM_POLICY" if owner == "jax_exp_ice_alb"
                else "DISCLOSED_AT_BAR_REASSOCIATION"
            ),
        })
        group["non_bit_identical_rows"] += count
        row_relative = float(row["max_relative_error_nonzero_oracle"])
        if row_relative > float(group["max_relative_error_nonzero_oracle"]):
            group["max_relative_error_nonzero_oracle"] = row_relative
            group["largest_relative_row"] = {
                "stage": row["stage"], "variable": row["variable"],
                "step": row["max_relative_error_step"],
            }
        group["exact_zero_oracle_non_bit_rows"] += int(
            row["exact_zero_oracle_non_bit_count"]
        )
        group["variables"].append(f'{row["stage"]}.{row["variable"]}')
    return grouped


def validate_selector(stage0: dict[str, np.ndarray], stage2: dict[str, np.ndarray]) -> None:
    from legoesm import constants

    expected = constants.bulk_transfer_ice_orca1
    for name in ("Cd", "Ch", "Ce"):
        require(np.all(stage0[name] == expected), f"non-ORCA1 {name}")
    require(np.all(stage2["ln_icedyn"] == 1.0), "ln_icedyn branch drift")
    require(np.all(stage2["ln_form_drag"] == 0.0), "form-drag branch drift")
    require(np.all(stage2["nn_form_drag"] == 2.0), "nn_frm drift")
    for name in ("ln_leadhfx", "ln_icedO", "ln_icedH", "mask"):
        require(np.all(stage2[name] == 1.0), f"{name} branch drift")


def _score(predicted, oracle, stage: str, variable: str) -> dict[str, object]:
    predicted = np.asarray(predicted, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    error = np.abs(predicted - oracle)
    normalized = error / np.maximum(np.abs(oracle), 1.0)
    index = int(np.argmax(normalized))
    over = np.flatnonzero(normalized > BAR)
    predicted_bits = predicted.view(np.uint64)
    oracle_bits = oracle.view(np.uint64)
    nonbit = np.flatnonzero(predicted_bits != oracle_bits)
    nonzero_oracle = oracle != 0.0
    nonbit_nonzero = (predicted_bits != oracle_bits) & nonzero_oracle
    relative = np.zeros_like(error)
    relative[nonzero_oracle] = (
        error[nonzero_oracle] / np.abs(oracle[nonzero_oracle])
    )
    relative_nonbit = relative * (predicted_bits != oracle_bits)
    relative_index = int(np.argmax(relative_nonbit))
    return {
        "stage": stage,
        "variable": variable,
        "max_absolute": float(error[index]),
        "max_normalized": float(normalized[index]),
        "max_step": index + 1,
        "over_bar_count": int(over.size),
        "first_over_step": int(over[0]) + 1 if over.size else None,
        "bit_identical_count": int(predicted.size - nonbit.size),
        "non_bit_identical_count": int(nonbit.size),
        "first_non_bit_step": int(nonbit[0]) + 1 if nonbit.size else None,
        "max_relative_error_nonzero_oracle": (
            float(np.max(relative[nonbit_nonzero]))
            if np.any(nonbit_nonzero) else 0.0
        ),
        "max_relative_error_step": (
            relative_index + 1 if np.any(nonbit_nonzero) else None
        ),
        "exact_zero_oracle_non_bit_count": int(np.sum(
            (predicted_bits != oracle_bits) & ~nonzero_oracle
        )),
    }


def evaluate(root: Path = DEFAULT_ROOT, *, plant: str | None = None) -> dict[str, object]:
    from legoesm import constants
    from legoesm.core.bulk_flux import nemo_si3_constant_fluxes
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
    from legoesm.ice.sea_ice import _nemo_si3_blk_ice_2, _nemo_si3_ice_flx_other

    runtime = runtime_versions()
    if plant == "runtime":
        runtime = {**runtime, "jax": "0.0.0-planted"}
    validate_runtime(runtime)
    set_policy(PrecisionPolicy.fp64())
    require(jax.config.read("jax_enable_x64"), "JAX x64 disabled")
    require(jax.default_backend() == "cpu", "bulk gate is CPU-only")
    bulk_path = root / "oracle_si3_bulk_operands.bin"
    exchange_path = root / "oracle_si3_exchange_frames.bin"
    require(sha256(bulk_path) == EXPECTED_BULK_SHA256, "bulk stream hash drift")
    require(sha256(exchange_path) == EXPECTED_EXCHANGE_SHA256, "exchange stream hash drift")
    stages = read_bulk_stream(bulk_path)
    exchange = read_exchange_active(exchange_path)
    s0_np = _as_dict(stages[0], STAGE0_NAMES)
    s1_np = _as_dict(stages[1], STAGE1_NAMES)
    s2_np = _as_dict(stages[2], STAGE2_NAMES)
    validate_coverage()
    validate_selector(s0_np, s2_np)
    # Force the executing comparison through JAX, rather than allowing NumPy
    # arrays to evaluate a prefix of an expression before its first jnp call.
    s0 = jax.tree.map(jax.numpy.asarray, s0_np)
    s1 = jax.tree.map(jax.numpy.asarray, s1_np)
    s2 = jax.tree.map(jax.numpy.asarray, s2_np)

    raw = nemo_si3_constant_fluxes(
        s0["u_air"], s0["v_air"], s0["theta_air"], s0["q_air"],
        s0["T_surface"], s0["p_surface"], s0["rho_air"],
        constants.bulk_transfer_ice_orca1,
        constants.bulk_transfer_ice_orca1,
        constants.bulk_transfer_ice_orca1,
    )
    predictions = {
        (0, "wndm_ice"): raw[2],
        (0, "utau_ice"): raw[0],
        (0, "vtau_ice"): raw[1],
    }
    oracle = {
        (0, "wndm_ice"): s0["wind"],
        (0, "utau_ice"): s0["utau_ice"],
        (0, "vtau_ice"): s0["vtau_ice"],
    }
    flux2 = _nemo_si3_blk_ice_2(
        T_surface=s1["T_surface"], h_ice=s1["h_ice"], h_snow=s1["h_snow"],
        cloud_fraction=s1["cloud_fraction"], theta_air=s1["theta_air"],
        q_air=s1["q_air"], p_surface=s1["p_surface"], lw_down=s1["lw_down"],
        precip=s1["precip"], snow=s1["snow"], qsr=s1["qsr"],
        rho_air=s1["rho_air"], wind=s1["wind"], Ch=s1["Ch"], Ce=s1["Ce"],
        ice_fraction_before=s1["ice_fraction_before"],
        category_fraction_before=s1["category_fraction_before"],
        sst_celsius=s1["sst_celsius"], qns_ocean=s1["qns_ocean"],
        qsr_ocean=s1["qsr_ocean"], emp_ocean_raw=s1["emp_ocean_raw"],
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
    )
    for name in STAGE1_NAMES[25:45]:
        predictions[(1, name)] = flux2[name]
        oracle[(1, name)] = s1[name]
    other = _nemo_si3_ice_flx_other(
        ice_fraction=s2["ice_fraction"], ice_fraction_before=s2["ice_fraction_before"],
        ice_volume=s2["ice_volume"], u_ice=s2["u_ice"],
        u_ice_west=s2["u_ice_west"], v_ice=s2["v_ice"],
        v_ice_south=s2["v_ice_south"], u_ocean=s2["u_ocean"],
        u_ocean_west=s2["u_ocean_west"], v_ocean=s2["v_ocean"],
        v_ocean_south=s2["v_ocean_south"], drag_io=s2["drag_io"],
        frq=s2["frq"], qsr_ocean=s2["qsr_ocean"], qns_ocean=s2["qns_ocean"],
        qemp_ocean=s2["qemp_ocean"], ocean_layer_thickness=s2["ocean_layer_thickness"],
        sst_celsius=s2["sst_celsius"], T_bottom=s2["T_bottom"], dt=s2["dt"],
        rho_ocean=s2["rho_ocean"], c_ocean=s2["c_ocean"], T0=s2["T0"],
        ice_epsilon=s2["ice_epsilon"], max_ice_fraction=s2["max_ice_fraction"],
    )
    for name in ("qsb_ice_bot", "fhld", "qlead"):
        predictions[(2, name)] = other[name]
        oracle[(2, name)] = s2[name]

    # Fields not subsequently mutated by ice thermodynamics must retain their
    # registered bulk-boundary value in the stable end-of-step exchange stream.
    exchange_links = {
        "qsr_ice": s1["qsr_ice"], "qla_ice": s1["qla_ice"],
        "dqla_ice": s1["dqla_ice"], "dqns_ice": s1["dqns_ice"],
        "qtr_ice_top": s1["qtr_ice_top"],
        "utau_ice": s0["utau_ice"], "vtau_ice": s0["vtau_ice"],
        "emp_ice": s1["emp_ice"], "evap_ice": s1["evap_ice"],
        "devap_ice": s1["devap_ice"], "qns_oce": s1["qns_ocean"],
        "qsr_oce": s1["qsr_ocean"], "qemp_oce": s1["qemp_oce"],
        "qemp_ice": s1["qemp_ice"], "qevap_ice": s1["qevap_ice"],
        "qprec_ice": s1["qprec_ice"], "emp_oce": s1["emp_oce"],
        "wndm_ice": s0["wind"],
    }

    ablated_flux2 = _nemo_si3_blk_ice_2(
        T_surface=s1["T_surface"], h_ice=s1["h_ice"], h_snow=s1["h_snow"],
        cloud_fraction=s1["cloud_fraction"], theta_air=s1["theta_air"],
        q_air=s1["q_air"], p_surface=s1["p_surface"], lw_down=s1["lw_down"],
        precip=s1["precip"], snow=s1["snow"], qsr=s1["qsr"],
        rho_air=s1["rho_air"], wind=s1["wind"], Ch=s1["Ch"], Ce=s1["Ce"],
        ice_fraction_before=s1["ice_fraction_before"],
        category_fraction_before=s1["category_fraction_before"],
        sst_celsius=s1["sst_celsius"], qns_ocean=s1["qns_ocean"],
        qsr_ocean=s1["qsr_ocean"], emp_ocean_raw=s1["emp_ocean_raw"],
        ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
        _preserve_subnormal_snow=False,
    )
    ablation_rows = [
        _score(ablated_flux2[name], s1[name], STAGE_NAMES[1], name)
        for name in STAGE1_NAMES[25:45]
    ]
    ablation_over = sum(int(row["over_bar_count"]) for row in ablation_rows)

    plant_targets = {
        "blk_ice_1": (0, "utau_ice"),
        "ice_alb": (1, "albedo"),
        "blk_ice_2": (1, "qns_ice"),
        "ice_flx_other": (2, "qlead"),
    }
    if plant in plant_targets:
        key = plant_targets[plant]
        predictions[key] = np.asarray(predictions[key]) + 1.0
    elif plant == "stream_hash":
        raise GateError("planted stream hash drift")
    elif plant == "coverage":
        bad = dict(COVERAGE)
        bad.pop("fr_i")
        validate_coverage(bad)
    elif plant == "selector":
        bad = dict(s0_np)
        bad["Cd"] = np.asarray(bad["Cd"]) * 2.0
        validate_selector(bad, s2_np)
    elif plant == "bit_owner":
        pass
    elif plant is not None:
        raise GateError(f"unknown plant {plant}")

    rows = [
        _score(predictions[key], oracle[key], STAGE_NAMES[key[0]], key[1])
        for key in predictions
    ]
    scalar_outputs = _scalar_glibc_albedo_replay(s1_np)
    scalar_rows = [
        _score(values, s1_np[name], "SCALAR_GLIBC_REPLAY", name)
        for name, values in scalar_outputs.items()
    ]
    require(
        all(int(row["non_bit_identical_count"]) == 0 for row in scalar_rows),
        f"scalar-glibc albedo replay drift: {scalar_rows}",
    )
    bit_groups = _bit_owner_groups(rows, plant=plant == "bit_owner")
    exchange_rows = [
        _score(value, exchange[name], "EXCHANGE_END_STABLE", name)
        for name, value in exchange_links.items()
    ]
    exchange_debt = [row for row in exchange_rows if row["over_bar_count"]]
    require(not exchange_debt, f"bulk/exchange time-level linkage violated: {exchange_debt}")
    total_over = sum(int(row["over_bar_count"]) for row in rows)
    total_nonbit = sum(int(row["non_bit_identical_count"]) for row in rows)
    largest = max(rows, key=lambda row: float(row["max_normalized"]))
    debt_rows = [row for row in rows if int(row["over_bar_count"])]
    require(total_over == 0, f"bulk fidelity bar violated: {total_over} rows; {debt_rows}")
    return {
        "verdict": "AT-BAR",
        "oracle_version": ORACLE_VERSION,
        "bar": BAR,
        "steps": EXPECTED_STEPS,
        "comparisons": len(rows) * EXPECTED_STEPS,
        "over_bar_rows": total_over,
        "bit_comparisons": len(rows) * EXPECTED_STEPS,
        "non_bit_identical_rows": total_nonbit,
        "bit_identical_rows": len(rows) * EXPECTED_STEPS - total_nonbit,
        "numeric_runtime": runtime,
        "bit_owner_groups": bit_groups,
        "scalar_glibc_owner_probe": {
            "source": "icealb.F90:124-185; Python math.exp/log call scalar glibc libm",
            "rows": scalar_rows,
            "interpretation": (
                "The scalar-libm replay is bit-identical for the three exp-owned "
                "outputs; the corresponding JAX differences await the shared "
                "library-exact exp precision policy."
            ),
        },
        "largest_row": largest,
        "dtypes": {"numpy": str(stages[0].dtype), "jax_x64": True, "backend": "cpu"},
        "streams": {
            "bulk": {"path": str(bulk_path), "sha256": sha256(bulk_path),
                     "bytes": bulk_path.stat().st_size, "frames": 3 * EXPECTED_STEPS},
            "exchange": {"path": str(exchange_path), "sha256": sha256(exchange_path)},
        },
        "frame_registry": [
            {"stage": STAGE_NAMES[stage], "time_level": "current ice step",
             "source": STAGE_SOURCES[stage], "values": STAGE_COUNTS[stage]}
            for stage in STAGE_NAMES
        ],
        "coverage": {name: {"status": status, "reason": reason}
                     for name, (status, reason) in COVERAGE.items()},
        "rows": rows,
        "input_legitimacy_rows": exchange_rows,
        "scope": {
            "certified": "ORCA1 constant-coefficient SI3 ice-side bulk arm",
            "uncertified": ["shipped C1D ECMWF/1.4e-3 arm", "ocean-side NCAR producer"],
        },
        "private_arms": {
            "numeric_snow_zero_comparison": {
                "hypothesis": "XLA flushes positive-subnormal h_snow in floating comparison",
                "over_bar_rows": ablation_over,
                "first_affected_step": min(
                    row["first_over_step"] for row in ablation_rows if row["over_bar_count"]
                ),
                "status": "REFUTED identity arm",
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", choices=(
        "blk_ice_1", "ice_alb", "blk_ice_2", "ice_flx_other",
        "stream_hash", "coverage", "selector", "bit_owner",
        "runtime",
    ))
    args = parser.parse_args()
    result = evaluate(args.root, plant=args.plant)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
