#!/usr/bin/env python3
"""ORCA2 Round 343: standalone ORCA2 K0 speed & 8-day accuracy benchmark.

Measures execution speed (compile time, steady-state s/step over >= 20 steps
after warm-up, host/GPU memory, and projected 1-year / 2,920-step wall-clock)
and 8-day (64-step) SST/T accuracy cost against the exact-mode daily snapshots
from Round 342 across four arms on the standalone ORCA2 K0 composed card:

  (a) exact_cpu_fp64  : Exact NEMO mode on CPU (float64, transcendentals='libm')
  (b) normal_cpu_fp64 : Normal mode on CPU (float64, transcendentals='native')
  (c) normal_gpu_fp64 : Normal mode on 1 idle GPU (float64, transcendentals='native')
  (d) normal_gpu_fp32 : Normal mode on 1 idle GPU (float32, transcendentals='native')

Measurement only: no model code or default in ``packages/`` is modified.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import warnings
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _pkg in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(_pkg) not in sys.path:
        sys.path.insert(0, str(_pkg))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean import bulk_flux_omip as bulk  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_orca2_damping_for_step,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from legoesm.ocean.forcing.nemo_surface import NemoSurfaceMeans  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig  # noqa: E402
from legoesm.ocean.physics.lateral_mixing.mle import MLEConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.config import TKEConfig  # noqa: E402
from legoesm.ocean.physics.vertical_mixing.double_diffusion import (  # noqa: E402
    DoubleDiffusionConfig,
)
from legoesm.ocean.physics.vertical_mixing.internal_wave_mixing import (  # noqa: E402
    IWMConfig,
)
from legoesm.ocean.state import (  # noqa: E402
    BarotropicConfig,
    LatLonCGridOceanConfig,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_o1_numerical_gate as o1,
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
    nemo_testcase_l4_orca2_round339_common as common,
    nemo_testcase_l4_orca2_round340_standalone_gate as r340_s4,
)

GateError = common.GateError
require = common.require
sha256_file = common.sha256

DEFAULT_DECK_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/"
    "acquisition/orca1ice_surface_entry_every_step_a_np2"
)
DEFAULT_CORE_MANIFEST = DEFAULT_DECK_ROOT / "input_files.sha256"
DEFAULT_ICE_COVER_PATH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round339/"
    "ice_cover_producer/orca2_full_si3_year1_np2/ice_cover_clim.nc"
)
DEFAULT_EXACT_REF_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r342/"
    "legoesm_year1_2920step"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/orca2_r343"
)

YEAR_STEPS = 2920
STEPS_PER_DAY = 8
WARMUP_STEPS = 2
MIN_STEADY_STEPS = 20
EXACT_BENCHMARK_STEPS = WARMUP_STEPS + MIN_STEADY_STEPS  # 22 steps (kt=1..22)
NORMAL_BENCHMARK_STEPS = 8 * STEPS_PER_DAY               # 64 steps (8 days)
DAILY_CHECK_STEPS_8D = tuple(range(STEPS_PER_DAY, NORMAL_BENCHMARK_STEPS + 1, STEPS_PER_DAY))
DAILY_CHECK_STEPS_EXACT = (8, 16)

MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype=np.int64)
MONTH_END_SEC = (np.cumsum(MONTH_DAYS) * 86400).astype(np.int64)
MONTH_CENTRES_SEC = ((np.cumsum(MONTH_DAYS) - MONTH_DAYS / 2) * 86400).astype(np.int64)
EXTENDED_MONTH_CENTRES_SEC = np.concatenate((
    [MONTH_CENTRES_SEC[-1] - 365 * 86400],
    MONTH_CENTRES_SEC,
    [MONTH_CENTRES_SEC[0] + 365 * 86400],
))

ARM_IDS = (
    "exact_cpu_fp64",
    "normal_cpu_fp64",
    "normal_gpu_fp64",
    "normal_gpu_fp32",
)


class SwitchSpec(NamedTuple):
    """Inventory entry for a switch distinguishing exact-NEMO mode from normal mode."""

    name: str
    owner: str
    path_in_card: str
    category: str
    exact_value: Any
    normal_value: Any
    switched_in_normal_mode: bool
    exact_citation: str
    normal_citation: str
    notes: str


def build_orca2_exact_vs_normal_switch_inventory() -> tuple[SwitchSpec, ...]:
    """Return the complete code-sourced inventory of exact-mode switches on ORCA2 K0."""
    return (
        # --- Category A: Hard libm-guarded switches (6) ---
        SwitchSpec(
            name="PrecisionPolicy.transcendentals",
            owner="PrecisionPolicy",
            path_in_card="policy.transcendentals",
            category="A_libm_guarded",
            exact_value="libm",
            normal_value="native",
            switched_in_normal_mode=True,
            exact_citation="packages/core/legoesm/core/precision.py:117-126",
            normal_citation="packages/core/legoesm/core/precision.py:103-115",
            notes=(
                "libm dispatches tx.exp/log/log10/sin/cos/tanh/atan/pow through "
                "jax.pure_callback + Python scalar ctypes loop over libm.so.6 on CPU "
                "(packages/core/legoesm/core/transcendentals.py:32-85); native uses "
                "vectorized XLA jnp primitives on CPU/GPU."
            ),
        ),
        SwitchSpec(
            name="NEMOTestcaseCard.transcendentals",
            owner="NEMOTestcaseCard",
            path_in_card="ocean.transcendentals",
            category="A_libm_guarded",
            exact_value="libm",
            normal_value="native",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:66",
            normal_citation="packages/core/legoesm/core/precision.py:103",
            notes="Card metadata stamping the active transcendental backend.",
        ),
        SwitchSpec(
            name="tke_htau_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_htau_evaluation",
            category="A_libm_guarded",
            exact_value="nemo_scalar_libm",
            normal_value="jax_expression",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2152",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:424",
            notes=(
                "nemo_scalar_libm is guarded by _require_libm_policy "
                "(packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:2708-2717,2740)."
            ),
        ),
        SwitchSpec(
            name="tke_etau_exponential_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_etau_exponential_evaluation",
            category="A_libm_guarded",
            exact_value="nemo_scalar_libm",
            normal_value="jax_expression",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2153",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:432",
            notes=(
                "nemo_scalar_libm is guarded by _require_libm_policy "
                "(packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:2815-2817)."
            ),
        ),
        SwitchSpec(
            name="ddm.(enabled,form)",
            owner="DoubleDiffusionConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.ddm",
            category="A_libm_guarded",
            exact_value=(True, "nemo_literal"),
            normal_value=(False, "hybrid"),
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2174-2175",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/double_diffusion.py:245-251",
            notes=(
                "form='nemo_literal' raises RuntimeError when transcendentals!='libm' "
                "(double_diffusion.py:325-329); enabled=True with form='hybrid' is "
                "refused by zdf_implicit_solver_evaluation='nemo_literal' "
                "(ocean_model_latlon_cgrid.py:14828-14832)."
            ),
        ),
        SwitchSpec(
            name="iwm.(form,tsdiff)",
            owner="IWMConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.iwm",
            category="A_libm_guarded",
            exact_value=("nemo_literal", True),
            normal_value=("generic", False),
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2167-2173",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/internal_wave_mixing.py:187-203",
            notes=(
                "form='nemo_literal' raises RuntimeError when transcendentals!='libm' "
                "(internal_wave_mixing.py:411-415); k_profiles.py:580-590 requires "
                "tsdiff=False when form='generic'."
            ),
        ),
        # --- Category B: Source-rounding / operand-association / literal-order switches (36) ---
        SwitchSpec(
            name="nemo_bn2_source_rounded",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_bn2_source_rounded",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2137",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:356",
            notes="Statement-level source rounding in N^2 buoyancy frequency.",
        ),
        SwitchSpec(
            name="nemo_tke_solve_source_rounded",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_tke_solve_source_rounded",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2139",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:373",
            notes="Source rounding inside the TKE tridiagonal Thomas solve.",
        ),
        SwitchSpec(
            name="nemo_tke_matrix_rhs_source_rounded",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_tke_matrix_rhs_source_rounded",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2141",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:389",
            notes="Source rounding when assembling the TKE tridiagonal matrix and RHS.",
        ),
        SwitchSpec(
            name="nemo_sh2_r3_source_rounded",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_sh2_r3_source_rounded",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2142",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:397",
            notes="Source rounding in zdf_sh2 shear production.",
        ),
        SwitchSpec(
            name="nemo_sh2_r3_stored_reciprocal",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.nemo_sh2_r3_stored_reciprocal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2143",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:406",
            notes="Multiply by stored reciprocal vs direct division in zdf_sh2.",
        ),
        SwitchSpec(
            name="tke_mxl_raw_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_mxl_raw_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="factored",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2156",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:439",
            notes="Mixing-length raw formula operand grouping.",
        ),
        SwitchSpec(
            name="tke_solver_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_solver_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="shared_thomas",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:422",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:415",
            notes="NEMO literal tridiagonal sweep vs shared Thomas solver.",
        ),
        SwitchSpec(
            name="tke_langmuir_evaluation",
            owner="TKEConfig",
            path_in_card="ocean.recipe.model_config.physics.vertical_mixing.tke.tke_langmuir_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="vectorized",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1946",
            normal_citation="packages/ocean/legoesm/ocean/physics/vertical_mixing/config.py:446",
            notes="Langmuir mixing-length loop vs vectorized evaluation.",
        ),
        SwitchSpec(
            name="slope_uv_source_rounding",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.slope_uv_source_rounding",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2934",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:417",
            notes="Source rounding on U/V isoneutral slope components.",
        ),
        SwitchSpec(
            name="redi_a33_source_ordering",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.redi_a33_source_ordering",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2935",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:489",
            notes="NEMO statement ordering in Redi K33 diagonal.",
        ),
        SwitchSpec(
            name="redi_a33_evaluation",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.redi_a33_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="normalized_square",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2929",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:485",
            notes="Literal triad square vs normalized square in Redi K33.",
        ),
        SwitchSpec(
            name="redi_vertical_skew_evaluation",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.redi_vertical_skew_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_literal",
            normal_value="normalized_sums",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2928",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:480",
            notes="Literal triad sums vs normalized sums in Redi vertical skew flux.",
        ),
        SwitchSpec(
            name="slope_metric_evaluation",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.slope_metric_evaluation",
            category="B_source_rounding_or_order",
            exact_value="nemo_reciprocal",
            normal_value="division",
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2920",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:412",
            notes="Stored metric reciprocal vs direct division in GM/Redi slopes.",
        ),
        SwitchSpec(
            name="mle.nemo_association",
            owner="MLEConfig",
            path_in_card="ocean.recipe.model_config.physics.mle.nemo_association",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2990",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py:130",
            notes="NEMO parenthesis association in Fox-Kemper MLE streamfunction.",
        ),
        SwitchSpec(
            name="nemo_r3t_stored_product",
            owner="ZStarCoordinate",
            path_in_card="ocean.recipe.z_coord.nemo_r3t_stored_product",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2209",
            normal_citation="packages/ocean/legoesm/ocean/vertical.py:157",
            notes="Source-rounded stored r3t stretch product in z* thickness.",
        ),
        SwitchSpec(
            name="nemo_eos_gdept_rounded_product",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_eos_gdept_rounded_product",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1983",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2686",
            notes="Source-rounded gdept_0 * (1 + r3t) product in EOS depth.",
        ),
        SwitchSpec(
            name="zdf_tracer_nemo_rounded_thickness",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.zdf_tracer_nemo_rounded_thickness",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1976",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3171",
            notes="Source-rounded layer thickness in implicit tracer vertical diffusion.",
        ),
        SwitchSpec(
            name="barotropic_rhs_ascending_level_sum",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.barotropic_rhs_ascending_level_sum",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1977",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2739",
            notes="Sequential ascending k=1..jpkm1 loop with barriers vs jnp.sum for barotropic RHS.",
        ),
        SwitchSpec(
            name="rk3_stage1_rhs_nemo_literal",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage1_rhs_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1978",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2631",
            notes="Stage-1 momentum RHS accumulation in NEMO written order.",
        ),
        SwitchSpec(
            name="rk3_stage_velocity_written_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_velocity_written_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1979",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2641",
            notes="Source-rounded written order for WS-RK3 stage velocity update.",
        ),
        SwitchSpec(
            name="rk3_stage_tracer_written_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_tracer_written_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1991",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2651",
            notes="Source-rounded written order for WS-RK3 stage-1/2 tracer update.",
        ),
        SwitchSpec(
            name="rk3_stage_r3f_nemo_literal",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_r3f_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1981",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2670",
            notes="Interpolate stored r3fb/r3fa ratios vs ratio of interpolated SSH.",
        ),
        SwitchSpec(
            name="rk3_stage_r3t_nemo_literal",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_r3t_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1982",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2678",
            notes="Interpolate stored r3t ratios vs ratio of interpolated stage SSH.",
        ),
        SwitchSpec(
            name="rk3_stage_r3uv_nemo_literal",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage_r3uv_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1984",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2694",
            notes="Interpolate stored r3u/r3v ratios vs ratio of interpolated stage SSH.",
        ),
        SwitchSpec(
            name="nemo_bottom_drag_written_statements",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_bottom_drag_written_statements",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2002",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3270",
            notes="Statement-rounded bottom drag coefficient evaluation.",
        ),
        SwitchSpec(
            name="rk3_vector_rhs_nemo_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_vector_rhs_nemo_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2014",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2747",
            notes="NEMO written addition order for vector-form momentum RHS.",
        ),
        SwitchSpec(
            name="een_vorticity_written_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.een_vorticity_written_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2015",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2492",
            notes="NEMO written triad summation order in EEN vorticity Coriolis/vorticity.",
        ),
        SwitchSpec(
            name="nemo_wzv_stored_r3_ratios",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_wzv_stored_r3_ratios",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2017",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2756",
            notes="Use stored r3t/r3u/r3v ratios in wzv vertical velocity diagnostic.",
        ),
        SwitchSpec(
            name="nemo_zad_written_association",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_zad_written_association",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2018",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2764",
            notes="NEMO written parenthesis association in vertical momentum advection (dynzad).",
        ),
        SwitchSpec(
            name="nemo_stage2_momentum_wzv_half_step",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_stage2_momentum_wzv_half_step",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2019",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2772",
            notes="Stage-2 wzv half-step thickness ratio rounding.",
        ),
        SwitchSpec(
            name="nemo_stage_momentum_wzv_stored_kmm_ratios",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_stage_momentum_wzv_stored_kmm_ratios",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2020",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2781",
            notes="Stored Kmm ratios in stage wzv vertical velocity.",
        ),
        SwitchSpec(
            name="nemo_ldf_stored_f_area",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_ldf_stored_f_area",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2026",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2790",
            notes="Use stored F-cell area reciprocal in lateral friction.",
        ),
        SwitchSpec(
            name="nemo_ldf_stage_r3f",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.nemo_ldf_stage_r3f",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2027",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2798",
            notes="Stage r3f ratio in lateral momentum diffusion.",
        ),
        SwitchSpec(
            name="rk3_stage1_ldf_source_order",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage1_ldf_source_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2028",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2806",
            notes="Stage-1 lateral diffusion accumulation order in Krhs.",
        ),
        SwitchSpec(
            name="barotropic_substep_dt_nemo_literal",
            owner="BarotropicConfig",
            path_in_card="ocean.recipe.model_config.barotropic.barotropic_substep_dt_nemo_literal",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1957",
            normal_citation="packages/ocean/legoesm/ocean/state.py:1572",
            notes="NEMO literal substep rdt_e rounding in barotropic solver.",
        ),
        SwitchSpec(
            name="barotropic_een_ratio_written_order",
            owner="BarotropicConfig",
            path_in_card="ocean.recipe.model_config.barotropic.barotropic_een_ratio_written_order",
            category="B_source_rounding_or_order",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=True,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2053",
            normal_citation="packages/ocean/legoesm/ocean/state.py:1600",
            notes="NEMO written order for barotropic EEN thickness ratio.",
        ),
        # --- Category C: Unconditional source-rounding barriers (1 representative entry) ---
        SwitchSpec(
            name="nemo_source_round / nemo_source_barrier",
            owner="source_rounding",
            path_in_card="unconditional_helper",
            category="C_unconditional_barrier",
            exact_value="optimization_barrier + copysign_zero",
            normal_value="optimization_barrier + copysign_zero",
            switched_in_normal_mode=False,
            exact_citation="packages/core/legoesm/core/source_rounding.py:14-35",
            normal_citation="packages/core/legoesm/core/source_rounding.py:14-35",
            notes=(
                "Direct calls in bulk_flux_omip.py:1086-1096, eos.py, and "
                "shortwave_penetration.py have no config switch; retained as-is "
                "because packages/ is unmodified."
            ),
        ),
        # --- Category D: Structurally coupled WS-RK3 / partial-cell / tripolar selectors (3 entries) ---
        SwitchSpec(
            name="zdf_implicit_solver_evaluation (+ companion zdf flags)",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.zdf_implicit_solver_evaluation",
            category="D_structural_scheme_selector",
            exact_value="nemo_literal",
            normal_value="shared_thomas",
            switched_in_normal_mode=False,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:421",
            normal_citation="packages/ocean/legoesm/ocean/state.py:3162",
            notes=(
                "Retained at 'nemo_literal' on ORCA2 K0 because _zdf_literal is the "
                "only branch in _apply_implicit_vertical_mixing "
                "(ocean_model_latlon_cgrid.py:14827-14975) that consumes "
                "nemo_tracer_content_rhs, nemo_aimp_*, and 3-D partial-cell masks."
            ),
        ),
        SwitchSpec(
            name="orca2_l8_krhs_selection (+ stage-3 tracer/EIV/Treguier bundle)",
            owner="LatLonCGridOceanConfig",
            path_in_card="ocean.recipe.model_config.rk3_stage3_damping_after_geothermal_nemo_order",
            category="D_structural_scheme_selector",
            exact_value=True,
            normal_value=False,
            switched_in_normal_mode=False,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1926-1940",
            normal_citation="packages/ocean/legoesm/ocean/state.py:2822",
            notes=(
                "Required by LatLonCGridOceanModel constructor validation when "
                "rk3_ws + gm_bolus_advection='through_fct' + bbl_diffusive_option=1 "
                "are active (ocean_model_latlon_cgrid.py:2606-2622,4116-4293)."
            ),
        ),
        SwitchSpec(
            name="tripolar_and_ws_rk3_scheme_selectors (kappa_redi_horizontal_evaluation, tke_matrix_evaluation, barotropic_continuity_evaluation)",
            owner="GMRediConfig",
            path_in_card="ocean.recipe.model_config.gm_redi.kappa_redi_horizontal_evaluation",
            category="D_structural_scheme_selector",
            exact_value="nemo_metric_literal",
            normal_value="cosine_scaled",
            switched_in_normal_mode=False,
            exact_citation="packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2923",
            normal_citation="packages/ocean/legoesm/ocean/physics/lateral_mixing/config.py:466",
            notes=(
                "Required by constructor validation on tripolar / WS-RK3 cards "
                "(ocean_model_latlon_cgrid.py:3248-3256,4507-4523; tke.py:3283-3287)."
            ),
        ),
    )


def _extract_switch_value(card: Any, spec: SwitchSpec) -> Any:
    if spec.path_in_card == "policy.transcendentals":
        return get_policy().transcendentals
    if spec.path_in_card == "unconditional_helper":
        return spec.exact_value
    if spec.path_in_card == "ocean.recipe.model_config.physics.vertical_mixing.ddm":
        ddm = card.ocean.recipe.model_config.physics.vertical_mixing.ddm
        return (ddm.enabled, ddm.form)
    if spec.path_in_card == "ocean.recipe.model_config.physics.vertical_mixing.iwm":
        iwm = card.ocean.recipe.model_config.physics.vertical_mixing.iwm
        return (iwm.form, iwm.tsdiff)
    obj = card
    for attr in spec.path_in_card.split("."):
        obj = getattr(obj, attr)
    return obj


def _extract_class_default_value(spec: SwitchSpec) -> Any:
    attr = spec.path_in_card.split(".")[-1]
    if spec.owner == "PrecisionPolicy":
        return PrecisionPolicy.fp64().transcendentals
    if spec.owner == "NEMOTestcaseCard":
        return PrecisionPolicy.fp64().transcendentals
    if spec.owner == "TKEConfig":
        return getattr(TKEConfig(), attr)
    if spec.owner == "DoubleDiffusionConfig":
        ddm = DoubleDiffusionConfig()
        return (ddm.enabled, ddm.form)
    if spec.owner == "IWMConfig":
        iwm = IWMConfig()
        return (iwm.form, iwm.tsdiff)
    if spec.owner == "GMRediConfig":
        return getattr(GMRediConfig(), attr)
    if spec.owner == "MLEConfig":
        return getattr(MLEConfig(), attr)
    if spec.owner == "ZStarCoordinate":
        return False
    if spec.owner == "LatLonCGridOceanConfig":
        return getattr(LatLonCGridOceanConfig(), attr)
    if spec.owner == "BarotropicConfig":
        return getattr(BarotropicConfig(), attr)
    return spec.normal_value


def verify_switch_inventory_against_card(
    base_card: Any,
    normal_card: Any,
    *,
    inventory: Sequence[SwitchSpec] | None = None,
) -> dict[str, Any]:
    """Verify exact vs normal card switch settings and confirm packages/ defaults are untouched."""
    specs = (
        build_orca2_exact_vs_normal_switch_inventory()
        if inventory is None
        else tuple(inventory)
    )
    require(len(specs) > 0, "switch inventory must not be empty")
    switched_count = 0
    cat_a_count = 0
    cat_b_count = 0
    for spec in specs:
        class_default = _extract_class_default_value(spec)
        require(
            class_default == spec.normal_value,
            f"Switch {spec.name!r} normal_value {spec.normal_value!r} != "
            f"class default {class_default!r} (was a default in packages/ changed?)",
        )
        if spec.path_in_card in ("policy.transcendentals", "unconditional_helper"):
            if spec.switched_in_normal_mode:
                switched_count += 1
                if spec.category == "A_libm_guarded":
                    cat_a_count += 1
            continue
        base_val = _extract_switch_value(base_card, spec)
        norm_val = _extract_switch_value(normal_card, spec)
        require(
            base_val == spec.exact_value,
            f"base_card switch {spec.name!r} has {base_val!r}, expected exact_value {spec.exact_value!r}",
        )
        if spec.switched_in_normal_mode:
            switched_count += 1
            if spec.category == "A_libm_guarded":
                cat_a_count += 1
            elif spec.category == "B_source_rounding_or_order":
                cat_b_count += 1
            require(
                norm_val == spec.normal_value,
                f"normal_card switch {spec.name!r} has {norm_val!r}, expected normal_value {spec.normal_value!r}",
            )
            require(
                base_val != norm_val,
                f"switched entry {spec.name!r} did not change between exact and normal card",
            )
        else:
            require(
                norm_val == spec.exact_value,
                f"retained switch {spec.name!r} changed on normal_card: {norm_val!r} != {spec.exact_value!r}",
            )
    require(cat_a_count == 6, f"expected 6 Category A switches, got {cat_a_count}")
    require(cat_b_count == 36, f"expected 36 Category B switches, got {cat_b_count}")
    require(switched_count == 42, f"expected 42 switched normal-mode settings, got {switched_count}")
    return {
        "total_inventory_entries": len(specs),
        "switched_in_normal_mode": switched_count,
        "category_a_count": cat_a_count,
        "category_b_count": cat_b_count,
    }


def build_normal_mode_orca2_k0_card(
    base_card: Any,
    *,
    variant: str = "cat_a_plus_b",
    precision_policy_name: str = "fp64",
) -> Any:
    """Return a copy of ``base_card`` with exact-mode switches set to normal defaults."""
    require(
        variant in ("cat_a_plus_b", "cat_a_only"),
        f"unknown normal-mode variant {variant!r}; expected 'cat_a_plus_b' or 'cat_a_only'",
    )
    require(
        precision_policy_name in ("fp64", "fp32"),
        f"unknown precision_policy_name {precision_policy_name!r}; expected 'fp64' or 'fp32'",
    )
    mc = base_card.ocean.recipe.model_config
    zc = base_card.ocean.recipe.z_coord
    vm = mc.physics.vertical_mixing

    tke = vm.tke._replace(
        tke_htau_evaluation="jax_expression",
        tke_etau_exponential_evaluation="jax_expression",
    )
    ddm = vm.ddm._replace(enabled=False, form="hybrid")
    iwm = vm.iwm._replace(form="generic", tsdiff=False)

    if variant == "cat_a_plus_b":
        tke = tke._replace(
            nemo_bn2_source_rounded=False,
            nemo_tke_solve_source_rounded=False,
            nemo_tke_matrix_rhs_source_rounded=False,
            nemo_sh2_r3_source_rounded=False,
            nemo_sh2_r3_stored_reciprocal=False,
            tke_mxl_raw_evaluation="factored",
            tke_solver_evaluation="shared_thomas",
            tke_langmuir_evaluation="vectorized",
        )
        gm = mc.gm_redi._replace(
            slope_uv_source_rounding=False,
            redi_a33_source_ordering=False,
            redi_a33_evaluation="normalized_square",
            redi_vertical_skew_evaluation="normalized_sums",
            slope_metric_evaluation="division",
        )
        mle = mc.physics.mle._replace(nemo_association=False)
        bt = mc.barotropic._replace(
            barotropic_substep_dt_nemo_literal=False,
            barotropic_een_ratio_written_order=False,
        )
        zc = zc._replace(nemo_r3t_stored_product=False)
        mc = mc._replace(
            nemo_eos_gdept_rounded_product=False,
            zdf_tracer_nemo_rounded_thickness=False,
            barotropic_rhs_ascending_level_sum=False,
            rk3_stage1_rhs_nemo_literal=False,
            rk3_stage_velocity_written_order=False,
            rk3_stage_tracer_written_order=False,
            rk3_stage_r3f_nemo_literal=False,
            rk3_stage_r3t_nemo_literal=False,
            rk3_stage_r3uv_nemo_literal=False,
            nemo_bottom_drag_written_statements=False,
            rk3_vector_rhs_nemo_order=False,
            een_vorticity_written_order=False,
            nemo_wzv_stored_r3_ratios=False,
            nemo_zad_written_association=False,
            nemo_stage2_momentum_wzv_half_step=False,
            nemo_stage_momentum_wzv_stored_kmm_ratios=False,
            nemo_ldf_stored_f_area=False,
            nemo_ldf_stage_r3f=False,
            rk3_stage1_ldf_source_order=False,
            gm_redi=gm,
            barotropic=bt,
            physics=mc.physics._replace(
                vertical_mixing=vm._replace(tke=tke, ddm=ddm, iwm=iwm),
                mle=mle,
            ),
        )
    else:
        mc = mc._replace(
            physics=mc.physics._replace(
                vertical_mixing=vm._replace(tke=tke, ddm=ddm, iwm=iwm),
            ),
        )

    recipe = base_card.ocean.recipe._replace(
        model_config=mc,
        physics_config=mc.physics,
        z_coord=zc,
    )
    ocean = base_card.ocean._replace(
        recipe=recipe,
        transcendentals="native",
        precision_policy=precision_policy_name,
    )
    return base_card._replace(ocean=ocean)


def compute_step_timing_summary(
    step_times_s: Sequence[float],
    *,
    warmup_steps: int = WARMUP_STEPS,
    min_steady_steps: int = MIN_STEADY_STEPS,
    year_steps: int = YEAR_STEPS,
) -> dict[str, Any]:
    """Compute compile time, steady-state s/step stats, and 1-year projections."""
    times = np.asarray(step_times_s, dtype=np.float64)
    require(times.ndim == 1, "step_times_s must be 1-D")
    require(warmup_steps >= 1, f"warmup_steps must be >= 1, got {warmup_steps}")
    require(min_steady_steps >= 20, f"min_steady_steps must be >= 20, got {min_steady_steps}")
    require(
        times.size >= warmup_steps + min_steady_steps,
        f"need >= {warmup_steps + min_steady_steps} total steps "
        f"({warmup_steps} warmup + >={min_steady_steps} steady), got {times.size}",
    )
    require(bool(np.isfinite(times).all()) and bool((times > 0.0).all()),
            "all step times must be finite and positive")

    warmup = times[:warmup_steps]
    steady = times[warmup_steps:]
    steady_kts = np.arange(warmup_steps + 1, times.size + 1)
    odd_mask = (steady_kts % 2) == 1
    even_mask = (steady_kts % 2) == 0

    median_s = float(np.median(steady))
    mean_s = float(np.mean(steady))
    std_s = float(np.std(steady))
    q25, q75 = np.percentile(steady, [25.0, 75.0])
    iqr_s = float(q75 - q25)
    min_s = float(np.min(steady))
    max_s = float(np.max(steady))
    odd_median_s = float(np.median(steady[odd_mask])) if np.any(odd_mask) else median_s
    even_median_s = float(np.median(steady[even_mask])) if np.any(even_mask) else median_s

    compile_warmup_total_s = float(np.sum(warmup))
    compile_overhead_est_s = float(max(0.0, compile_warmup_total_s - warmup_steps * median_s))

    proj_steady_s = float(year_steps * median_s)
    proj_with_compile_s = float(compile_warmup_total_s + (year_steps - warmup_steps) * median_s)
    proj_mean_s = float(year_steps * mean_s)

    return {
        "total_steps": int(times.size),
        "warmup_steps": int(warmup_steps),
        "steady_state_steps": int(steady.size),
        "step1_s": float(times[0]),
        "step2_s": float(times[1]) if times.size >= 2 else 0.0,
        "compile_warmup_total_s": compile_warmup_total_s,
        "compile_overhead_est_s": compile_overhead_est_s,
        "steady_median_s": median_s,
        "steady_mean_s": mean_s,
        "steady_std_s": std_s,
        "steady_iqr_s": iqr_s,
        "steady_min_s": min_s,
        "steady_max_s": max_s,
        "steady_odd_kt_median_s": odd_median_s,
        "steady_even_kt_median_s": even_median_s,
        "projected_year_steps": int(year_steps),
        "projected_year_steady_s": proj_steady_s,
        "projected_year_steady_min": proj_steady_s / 60.0,
        "projected_year_steady_hr": proj_steady_s / 3600.0,
        "projected_year_with_compile_s": proj_with_compile_s,
        "projected_year_with_compile_min": proj_with_compile_s / 60.0,
        "projected_year_with_compile_hr": proj_with_compile_s / 3600.0,
        "projected_year_from_mean_s": proj_mean_s,
        "projected_year_from_mean_min": proj_mean_s / 60.0,
        "step_times_s": [float(x) for x in times],
    }


def score_daily_temperature_against_reference(
    candidate_t: np.ndarray,
    reference_t: np.ndarray,
    wet_mask_3d: np.ndarray,
    *,
    kt: int,
) -> dict[str, Any]:
    """Compute SST and 3-D T RMS and max absolute differences at step ``kt``."""
    cand = np.asarray(candidate_t, dtype=np.float64)
    ref = np.asarray(reference_t, dtype=np.float64)
    wet3 = np.asarray(wet_mask_3d, dtype=bool)
    require(cand.shape == (148, 180, 30), f"candidate T shape {cand.shape} != (148, 180, 30)")
    require(ref.shape == (148, 180, 30), f"reference T shape {ref.shape} != (148, 180, 30)")
    require(wet3.shape == (148, 180, 30), f"wet_mask_3d shape {wet3.shape} != (148, 180, 30)")
    require(bool(np.isfinite(cand).all()), f"candidate T at kt={kt} contains non-finite values")
    require(bool(np.isfinite(ref).all()), f"reference T at kt={kt} contains non-finite values")
    require(bool(np.any(wet3)), "wet_mask_3d must have at least one active cell")

    wet2 = wet3[..., 0]
    diff_t = cand - ref
    diff_sst = diff_t[..., 0]

    return {
        "kt": int(kt),
        "day": int(kt // STEPS_PER_DAY),
        "sst_rms_all_c": float(np.sqrt(np.mean(diff_sst ** 2))),
        "sst_rms_wet_c": float(np.sqrt(np.mean(diff_sst[wet2] ** 2))),
        "sst_max_abs_c": float(np.max(np.abs(diff_sst))),
        "t3d_rms_all_c": float(np.sqrt(np.mean(diff_t ** 2))),
        "t3d_rms_wet_c": float(np.sqrt(np.mean(diff_t[wet3] ** 2))),
        "t3d_max_abs_c": float(np.max(np.abs(diff_t))),
    }


def read_memory_telemetry() -> dict[str, Any]:
    """Read host RSS/VmHWM and optional GPU memory stats for the current process."""
    ru_maxrss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    vm_hwm_kib = ru_maxrss_kib
    vm_rss_kib = 0
    status_path = Path("/proc/self/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) >= 2:
                    vm_hwm_kib = max(vm_hwm_kib, int(parts[1]))
            elif line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    vm_rss_kib = int(parts[1])
    telemetry: dict[str, Any] = {
        "host_peak_rss_kib": vm_hwm_kib,
        "host_peak_rss_mib": vm_hwm_kib / 1024.0,
        "host_peak_rss_gib": vm_hwm_kib / (1024.0 * 1024.0),
        "host_current_rss_mib": vm_rss_kib / 1024.0,
        "gpu_peak_bytes_in_use": None,
        "gpu_peak_mib": None,
        "gpu_bytes_limit": None,
    }
    if jax.default_backend() == "gpu":
        dev = jax.devices()[0]
        if hasattr(dev, "memory_stats"):
            stats = dev.memory_stats() or {}
            if "peak_bytes_in_use" in stats:
                peak_b = int(stats["peak_bytes_in_use"])
                telemetry["gpu_peak_bytes_in_use"] = peak_b
                telemetry["gpu_peak_mib"] = peak_b / (1024.0 * 1024.0)
            if "bytes_limit" in stats:
                telemetry["gpu_bytes_limit"] = int(stats["bytes_limit"])
    return telemetry


def load_core_window_for_steps(
    deck_root: Path,
    manifest: Path,
    max_steps: int,
) -> dict[str, np.ndarray]:
    """Load and interpolate all CORE records reachable across steps 1..max_steps."""
    require(1 <= max_steps <= YEAR_STEPS, f"max_steps must be in 1..{YEAR_STEPS}, got {max_steps}")
    declared: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            declared[parts[1]] = parts[0]
    required = {filename for _, filename, _, _ in o1.FIELD_SPECS}
    required |= {
        "weights_core2_orca2_bicub.nc",
        "weights_core2_orca2_bilin.nc",
        "ORCA_R2_zps_domcfg.nc",
    }
    for name in required:
        require(name in declared, f"CORE input manifest omits {name}")
        require(sha256_file(deck_root / name) == declared[name],
                f"CORE input digest moved: {name}")

    n_6h = min(1460, max(5, (max_steps - 1) // 2 + 1))
    n_daily = min(365, max(2, (max_steps - 1) // 8 + 1))
    n_monthly = 1 if max_steps <= 10 else 12

    mapped: dict[str, np.ndarray] = {}
    weights = {bicubic: o1.load_weights(deck_root, bicubic) for bicubic in (False, True)}
    for field, filename, variable, bicubic in o1.FIELD_SPECS:
        with netCDF4.Dataset(deck_root / filename, "r") as ds:
            src_var = ds[variable]
            total_recs = int(src_var.shape[0])
            count = {1460: n_6h, 365: n_daily, 12: n_monthly}.get(total_recs)
            require(count is not None, f"CORE cadence moved for {variable}: {src_var.shape}")
            raw_slice = np.asarray(src_var[:count].data, dtype=np.float64)
        indices, coefficients = weights[bicubic]
        mapped[field] = np.stack([
            np.asarray(o1.nemo_fld_interp(raw_slice[idx], indices, coefficients, bicubic=bicubic))
            for idx in range(count)
        ])

    with netCDF4.Dataset(deck_root / "ORCA_R2_zps_domcfg.nc", "r") as ds:
        cosine, sine = o1.nemo_t_rotation_from_domain(*(
            np.asarray(ds[name][:].data, dtype=np.float64)
            for name in ("glamt", "gphit", "glamv", "gphiv")
        ))
    wind_shape = mapped["wndi"].shape
    mapped["wndi"], mapped["wndj"] = map(
        np.asarray,
        o1.rotate_en_to_ij(
            mapped["wndi"],
            mapped["wndj"],
            np.broadcast_to(cosine, wind_shape),
            np.broadcast_to(sine, wind_shape),
        ),
    )
    require(all(np.isfinite(val).all() for val in mapped.values()),
            "mapped CORE normal-year input contains non-finite values")
    return mapped


def _load_monthly_12(path: Path, variable: str) -> np.ndarray:
    require(path.is_file(), f"missing K0 monthly file: {path}")
    with netCDF4.Dataset(path, "r") as ds:
        require(variable in ds.variables, f"K0 variable {variable!r} absent from {path.name}")
        src = ds.variables[variable]
        src.set_auto_maskandscale(False)
        val = np.asarray(src[:], dtype=np.float64)
    if val.ndim == 4:
        val = val[:, 0]
    require(val.shape == (12, 148, 180), f"unexpected K0 monthly shape in {path.name}: {val.shape}")
    require(bool(np.isfinite(val).all()), f"non-finite monthly values in {path.name}")
    return val


def _monthly_pair_for_clock(
    records_12: np.ndarray,
    isecsbc: int,
) -> bulk.NemoStandaloneFieldPair:
    after_ext = int(np.searchsorted(EXTENDED_MONTH_CENTRES_SEC, isecsbc, side="right"))
    before_ext = after_ext - 1
    before_s = int(EXTENDED_MONTH_CENTRES_SEC[before_ext])
    after_s = int(EXTENDED_MONTH_CENTRES_SEC[after_ext])
    before_idx = (before_ext - 1) % 12
    after_idx = (after_ext - 1) % 12
    return bulk.NemoStandaloneFieldPair(
        jnp.asarray(records_12[before_idx], dtype=jnp.float64),
        jnp.asarray(records_12[after_idx], dtype=jnp.float64),
        isecsbc,
        before_s,
        after_s,
    )


def precompute_step_controls(
    card: Any,
    deck_root: Path,
    core_manifest: Path,
    ice_cover_path: Path,
    steps: int,
) -> tuple[dict[str, Any], ...]:
    """Precompute per-step physical forcing controls for steps 1..steps outside the step timer."""
    recipe = card.ocean.recipe
    state = recipe.initial_state
    mask_np = np.asarray(state.land_mask.data, dtype=np.float64)
    ny, nx = mask_np.shape
    require((ny, nx) == (148, 180), f"unexpected ORCA2 grid shape: {(ny, nx)}")
    active_np = np.asarray(recipe.z_coord.is_active) > 0.5
    wet_2d = mask_np > 0.5

    mapped_core = load_core_window_for_steps(deck_root, core_manifest, steps)
    cover = r340_s4.read_decision171_cover(ice_cover_path)
    runoff_12 = _load_monthly_12(deck_root / "runoff_core_monthly.nc", "sorunoff")
    sss_12 = _load_monthly_12(deck_root / "sss_data.nc", "sss")
    geothermal = jnp.asarray(
        ladder.read_geothermal_heatflow(deck_root, (148, 180)),
        dtype=jnp.float64,
    )
    with netCDF4.Dataset(deck_root / "ORCA_R2_zps_domcfg.nc", "r") as ds:
        ff_f = jnp.asarray(np.asarray(ds.variables["ff_f"][:], dtype=np.float64))
    with netCDF4.Dataset(deck_root / "runoff_core_monthly.nc", "r") as ds:
        mouth = jnp.asarray(np.asarray(ds.variables["socoefr0"][:], dtype=np.float64))

    stress_lbc, tmask_east, tmask_north, umask, vmask = r340_s4._build_stress_lbc_and_masks(
        mask_np,
        np.asarray(state.u_mask.data, dtype=np.float64),
        np.asarray(state.v_mask.data, dtype=np.float64),
        recipe.grid.fold,
    )
    zero = jnp.zeros((ny, nx), dtype=jnp.float64)
    one = jnp.ones((ny, nx), dtype=jnp.float64)
    mask_jnp = jnp.asarray(mask_np, dtype=jnp.float64)
    cover_jnp = jnp.asarray(cover, dtype=jnp.float64)

    means_init = NemoSurfaceMeans(*(zero for _ in range(7)))
    s0_base = bulk.NemoS0Inputs(
        prior=means_init,
        kt=1,
        nit000=1,
        forcing_stride=2,
        restart_means_present=False,
        use_conservative_temperature=False,
        remove_inverse_barometer=False,
        u_before=zero,
        v_before=zero,
        temperature_now=zero,
        salinity_now=zero,
        ssh_now=zero,
        ssh_inverse_barometer=zero,
        e3t_reference=one,
        r3t_now=zero,
        surface_mask=mask_jnp,
        shortwave_fraction_top=one,
    )
    s2_base = bulk.NemoS2Inputs(
        stress_lbc=stress_lbc,
        post_ice_sfx=zero,
        runoff_now=zero,
        prior_runoff_mass=zero,
        prior_runoff_temperature_content=zero,
        prior_runoff_salinity_content=zero,
        prior_restoring_volume_flux=zero,
        prior_restoring_heat_flux=zero,
        prior_restoring_coefficient=one,
        sss_target=zero,
        runoff_mouth_mask=mouth,
        surface_mask=mask_jnp,
        tmask_east=jnp.asarray(tmask_east, dtype=jnp.float64),
        tmask_north=jnp.asarray(tmask_north, dtype=jnp.float64),
        umask=jnp.asarray(umask, dtype=jnp.float64),
        vmask=jnp.asarray(vmask, dtype=jnp.float64),
        freshwater_budget_correction=jnp.asarray(0.0, dtype=jnp.float64),
        runoff_factor=jnp.asarray(1.0, dtype=jnp.float64),
        reference_density=jnp.asarray(recipe.model_config.physics.constants.rho_0, dtype=jnp.float64),
        restoring_mm_per_day=jnp.asarray(-166.67, dtype=jnp.float64),
        restoring_bound_mm_per_day=jnp.asarray(4.0, dtype=jnp.float64),
        seconds_per_day=jnp.asarray(86400.0, dtype=jnp.float64),
        salinity_floor=jnp.asarray(1.0e-20, dtype=jnp.float64),
        heat_capacity=jnp.asarray(recipe.model_config.physics.constants.c_sw, dtype=jnp.float64),
    )

    dt_s = int(card.ocean.dt_s)
    controls: list[dict[str, Any]] = []
    for kt in range(1, steps + 1):
        isecsbc = kt * dt_s
        retained_kt = kt if kt % 2 else kt - 1
        retained_clock = retained_kt * dt_s

        idx_6h = retained_clock // 21600
        idx_daily = retained_clock // 86400
        idx_monthly = int(np.searchsorted(MONTH_END_SEC, retained_clock, side="right"))
        if mapped_core["precip_raw"].shape[0] == 1:
            idx_monthly = 0

        atmosphere = bulk.NemoStandaloneAtmosphereFields(
            jnp.asarray(mapped_core["wndi"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["wndj"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["tair"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["humi"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["slp"][idx_6h], dtype=jnp.float64),
            jnp.asarray(mapped_core["qsr_down"][idx_daily], dtype=jnp.float64),
            jnp.asarray(mapped_core["qlw_down"][idx_daily], dtype=jnp.float64),
            jnp.asarray(mapped_core["precip_raw"][idx_monthly], dtype=jnp.float64),
            jnp.asarray(mapped_core["snow_raw"][idx_monthly], dtype=jnp.float64),
        )
        pair_runoff = _monthly_pair_for_clock(runoff_12, isecsbc)
        pair_sss = _monthly_pair_for_clock(sss_12, isecsbc)
        s0f = bulk.NemoStandaloneS0FInputs(
            atmosphere,
            atmosphere,
            isecsbc,
            isecsbc - 1,
            isecsbc + 1,
            pair_runoff,
            pair_sss,
        )
        ice_before_s, ice_after_s = (
            (-15768000, 15768000)
            if isecsbc <= 15768000
            else (15768000, 47304000)
        )
        ice_if = bulk.NemoSbcIceIfInputs(
            cover_jnp,
            cover_jnp,
            isecsbc,
            ice_before_s,
            ice_after_s,
            zero,
            zero,
            zero,
            ff_f,
            mask_jnp,
            mask_jnp,
        )
        chl = jnp.asarray(
            ladder.chlorophyll_at_step(deck_root, kt, wet_2d, dt_s=card.ocean.dt_s),
            dtype=jnp.float64,
        )
        sponge = build_orca2_damping_for_step(
            deck_root,
            active_np,
            kt=kt,
            dt_s=card.ocean.dt_s,
            mask_evaluation="nemo_tmask_multiply",
        )
        controls.append({
            "s0f": s0f,
            "ice_if": ice_if,
            "s0_base": s0_base,
            "s2_base": s2_base,
            "chl": chl,
            "geothermal_heat_flux_mwm2": geothermal,
            "sponge": sponge,
        })
    return tuple(controls)


def run_single_benchmark_arm(
    arm_id: str,
    *,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    exact_ref_dir: Path = DEFAULT_EXACT_REF_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    variant: str = "cat_a_plus_b",
) -> dict[str, Any]:
    """Execute one benchmark arm in the current process and save its JSON artifact."""
    require(arm_id in ARM_IDS, f"unknown benchmark arm {arm_id!r}; expected one of {ARM_IDS}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if arm_id == "exact_cpu_fp64":
        require(jax.config.jax_enable_x64, "exact_cpu_fp64 requires JAX_ENABLE_X64=1")
        require(jax.default_backend() == "cpu", f"exact_cpu_fp64 requires CPU backend, got {jax.default_backend()!r}")
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        steps = EXACT_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_EXACT
    elif arm_id == "normal_cpu_fp64":
        require(jax.config.jax_enable_x64, "normal_cpu_fp64 requires JAX_ENABLE_X64=1")
        require(jax.default_backend() == "cpu", f"normal_cpu_fp64 requires CPU backend, got {jax.default_backend()!r}")
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        steps = NORMAL_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_8D
    elif arm_id == "normal_gpu_fp64":
        require(jax.config.jax_enable_x64, "normal_gpu_fp64 requires JAX_ENABLE_X64=1")
        require(jax.default_backend() == "gpu", f"normal_gpu_fp64 requires GPU backend, got {jax.default_backend()!r}")
        set_policy(PrecisionPolicy.fp64(transcendentals="native"))
        steps = NORMAL_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_8D
    else:  # normal_gpu_fp32
        require(not jax.config.jax_enable_x64, "normal_gpu_fp32 requires JAX_ENABLE_X64=0")
        require(jax.default_backend() == "gpu", f"normal_gpu_fp32 requires GPU backend, got {jax.default_backend()!r}")
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype",
            category=UserWarning,
        )
        set_policy(PrecisionPolicy.fp32(transcendentals="native"))
        steps = NORMAL_BENCHMARK_STEPS
        daily_check_steps = DAILY_CHECK_STEPS_8D

    t_setup_start = time.perf_counter()
    base_card = bulk.build_nemo_orca2_k0_ocean_card(deck_root)
    normal_card_check = build_normal_mode_orca2_k0_card(base_card, variant="cat_a_plus_b")
    inventory_check = verify_switch_inventory_against_card(base_card, normal_card_check)

    if arm_id == "exact_cpu_fp64":
        card = base_card
    else:
        pol_name = "fp32" if arm_id == "normal_gpu_fp32" else "fp64"
        card = build_normal_mode_orca2_k0_card(
            base_card,
            variant=variant,
            precision_policy_name=pol_name,
        )

    wet_3d = np.asarray(base_card.ocean.recipe.z_coord.is_active, dtype=bool)
    controls = precompute_step_controls(card, deck_root, core_manifest, ice_cover_path, steps)
    recipe = card.ocean.recipe
    model = LatLonCGridOceanModel(
        recipe.grid,
        recipe.z_coord,
        recipe.model_config,
        iwm_forcing=recipe.iwm_forcing,
    )
    state = recipe.initial_state
    setup_wall_s = time.perf_counter() - t_setup_start

    expected_dtype = np.dtype("float32") if arm_id == "normal_gpu_fp32" else np.dtype("float64")
    require(
        np.dtype(state.T.data.dtype) == expected_dtype,
        f"arm {arm_id}: initial state.T.data.dtype={state.T.data.dtype} != {expected_dtype}",
    )

    carry_s0 = None
    carry_s2 = None
    prior_surface = None
    step_times_s: list[float] = []
    daily_accuracy: list[dict[str, Any]] = []

    for kt in range(1, steps + 1):
        ctrl = controls[kt - 1]
        s0_base = ctrl["s0_base"]
        s2_base = ctrl["s2_base"]
        s0 = (
            s0_base
            if carry_s0 is None
            else s0_base._replace(
                prior=carry_s0.prior,
                shortwave_fraction_top=carry_s0.shortwave_fraction_top,
            )
        )
        s2 = (
            s2_base
            if carry_s2 is None
            else s2_base._replace(
                prior_runoff_mass=carry_s2.prior_runoff_mass,
                prior_runoff_temperature_content=carry_s2.prior_runoff_temperature_content,
                prior_runoff_salinity_content=carry_s2.prior_runoff_salinity_content,
                prior_restoring_volume_flux=carry_s2.prior_restoring_volume_flux,
                prior_restoring_heat_flux=carry_s2.prior_restoring_heat_flux,
                prior_restoring_coefficient=carry_s2.prior_restoring_coefficient,
            )
        )

        t0 = time.perf_counter()
        res = bulk.nemo_orca2_k0_ocean_step(
            card,
            state,
            model=model,
            nemo_kt=kt,
            s0f=ctrl["s0f"],
            ice_if=ctrl["ice_if"],
            s0=s0,
            s2=s2,
            chl=ctrl["chl"],
            geothermal_heat_flux_mwm2=ctrl["geothermal_heat_flux_mwm2"],
            prior_surface=prior_surface,
            sponge=ctrl["sponge"],
        )
        state = res.ocean.state_after if hasattr(res.ocean, "state_after") else res.ocean
        jax.block_until_ready(state.T.data)
        dt_wall = time.perf_counter() - t0
        step_times_s.append(dt_wall)

        carry_s0 = res.next_s0
        carry_s2 = res.next_s2
        prior_surface = res.surface

        if kt in daily_check_steps:
            ref_path = exact_ref_dir / f"candidate_step{kt:08d}.npz"
            require(ref_path.is_file(), f"missing exact reference snapshot: {ref_path}")
            with np.load(ref_path) as ref_npz:
                t_ref = np.asarray(ref_npz["T"], dtype=np.float64)
            t_cand = np.asarray(state.T.data, dtype=np.float64)
            score = score_daily_temperature_against_reference(t_cand, t_ref, wet_3d, kt=kt)
            score["reference_snapshot"] = ref_path.name
            score["reference_sha256"] = sha256_file(ref_path)
            daily_accuracy.append(score)

    for field_name in ("T", "S", "u", "v", "eta"):
        arr = np.asarray(getattr(state, field_name).data)
        require(bool(np.isfinite(arr).all()), f"arm {arm_id}: non-finite terminal {field_name}")

    timing = compute_step_timing_summary(step_times_s)
    memory = read_memory_telemetry()
    dev = jax.devices()[0]

    arm_report = {
        "arm_id": arm_id,
        "variant": "exact" if arm_id == "exact_cpu_fp64" else variant,
        "backend": jax.default_backend(),
        "device_kind": getattr(dev, "device_kind", str(dev)),
        "dtype": str(expected_dtype),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "transcendentals": get_policy().transcendentals,
        "setup_wall_s": setup_wall_s,
        "timing": timing,
        "memory": memory,
        "daily_accuracy": daily_accuracy,
        "switch_inventory_check": inventory_check,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_path = output_dir / f"arm_{arm_id}.json"
    out_path.write_text(json.dumps(arm_report, indent=2) + "\n", encoding="utf-8")
    return arm_report


def validate_benchmark_report(
    report: Mapping[str, Any],
    *,
    require_exact_zero_on_arm_a: bool = True,
) -> None:
    """Validate a combined 4-arm benchmark report against preregistered gates."""
    require("arms" in report and isinstance(report["arms"], Mapping), "report missing 'arms' mapping")
    arms = report["arms"]
    for arm_id in ARM_IDS:
        require(arm_id in arms, f"report missing required arm {arm_id!r}")
        arm = arms[arm_id]
        timing = arm.get("timing", {})
        require(
            int(timing.get("steady_state_steps", 0)) >= MIN_STEADY_STEPS,
            f"arm {arm_id}: steady_state_steps < {MIN_STEADY_STEPS}",
        )
        require(
            float(timing.get("steady_median_s", 0.0)) > 0.0
            and np.isfinite(float(timing.get("steady_median_s", np.nan))),
            f"arm {arm_id}: invalid steady_median_s",
        )
        require(
            float(timing.get("compile_warmup_total_s", 0.0)) > 0.0,
            f"arm {arm_id}: invalid compile_warmup_total_s",
        )
        mem = arm.get("memory", {})
        require(
            float(mem.get("host_peak_rss_mib", 0.0)) > 0.0,
            f"arm {arm_id}: invalid host_peak_rss_mib",
        )
        daily = arm.get("daily_accuracy", [])
        if arm_id == "exact_cpu_fp64":
            require(
                tuple(d["kt"] for d in daily) == DAILY_CHECK_STEPS_EXACT,
                f"arm {arm_id}: expected daily_accuracy kts {DAILY_CHECK_STEPS_EXACT}",
            )
            if require_exact_zero_on_arm_a:
                for entry in daily:
                    require(
                        entry["sst_max_abs_c"] == 0.0 and entry["t3d_max_abs_c"] == 0.0,
                        f"arm {arm_id} kt={entry['kt']} drifted from exact reference: {entry}",
                    )
        else:
            require(
                tuple(d["kt"] for d in daily) == DAILY_CHECK_STEPS_8D,
                f"arm {arm_id}: expected 8 daily_accuracy kts {DAILY_CHECK_STEPS_8D}",
            )
            for entry in daily:
                for key in (
                    "sst_rms_all_c",
                    "sst_rms_wet_c",
                    "sst_max_abs_c",
                    "t3d_rms_all_c",
                    "t3d_rms_wet_c",
                    "t3d_max_abs_c",
                ):
                    val = float(entry[key])
                    require(
                        np.isfinite(val) and val >= 0.0,
                        f"arm {arm_id} kt={entry['kt']} invalid {key}={val}",
                    )
                require(
                    float(entry["sst_max_abs_c"]) >= float(entry["sst_rms_all_c"]),
                    f"arm {arm_id} kt={entry['kt']}: sst_max_abs_c < sst_rms_all_c",
                )
                require(
                    float(entry["t3d_max_abs_c"]) >= float(entry["t3d_rms_all_c"]),
                    f"arm {arm_id} kt={entry['kt']}: t3d_max_abs_c < t3d_rms_all_c",
                )


def assemble_benchmark_report_from_arm_files(
    *,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    exact_ref_dir: Path = DEFAULT_EXACT_REF_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Assemble and validate the combined 4-arm report from saved arm JSON artifacts."""
    arms_out: dict[str, Any] = {}
    for arm_id in ARM_IDS:
        arm_file = output_dir / f"arm_{arm_id}.json"
        require(arm_file.is_file(), f"missing arm artifact {arm_file}")
        arms_out[arm_id] = json.loads(arm_file.read_text(encoding="utf-8"))

    inventory = [spec._asdict() for spec in build_orca2_exact_vs_normal_switch_inventory()]
    combined = {
        "round": "orca2_r343",
        "author": "Argon",
        "description": "Standalone ORCA2 K0 speed and 8-day accuracy benchmark (exact vs normal mode)",
        "provenance": worktree_stamp(),
        "deck_root": str(deck_root),
        "core_manifest": str(core_manifest),
        "ice_cover_path": str(ice_cover_path),
        "exact_ref_dir": str(exact_ref_dir),
        "switch_inventory": inventory,
        "arms": arms_out,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    validate_benchmark_report(combined)
    combined_path = output_dir / "orca2_round343_speed_benchmark_report.json"
    combined_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    return combined


def run_all_arms_via_subprocesses(
    *,
    deck_root: Path = DEFAULT_DECK_ROOT,
    core_manifest: Path = DEFAULT_CORE_MANIFEST,
    ice_cover_path: Path = DEFAULT_ICE_COVER_PATH,
    exact_ref_dir: Path = DEFAULT_EXACT_REF_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    gpu_device: str = "0",
    variant: str = "cat_a_plus_b",
) -> dict[str, Any]:
    """Run all four benchmark arms in isolated subprocesses and assemble the combined report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = Path(__file__).resolve()

    arm_envs = {
        "exact_cpu_fp64": {"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        "normal_cpu_fp64": {"JAX_ENABLE_X64": "1", "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""},
        "normal_gpu_fp64": {"JAX_ENABLE_X64": "1", "CUDA_VISIBLE_DEVICES": str(gpu_device)},
        "normal_gpu_fp32": {"JAX_ENABLE_X64": "0", "CUDA_VISIBLE_DEVICES": str(gpu_device)},
    }

    for arm_id in ARM_IDS:
        env = dict(os.environ)
        env.pop("JAX_PLATFORMS", None)
        env.update(arm_envs[arm_id])
        cmd = [
            sys.executable,
            str(script_path),
            "--arm",
            arm_id,
            "--deck-root",
            str(deck_root),
            "--core-manifest",
            str(core_manifest),
            "--ice-cover-path",
            str(ice_cover_path),
            "--exact-ref-dir",
            str(exact_ref_dir),
            "--output-dir",
            str(output_dir),
            "--variant",
            variant,
        ]
        print(f"[round343] launching arm={arm_id} ...", flush=True)
        proc = subprocess.run(cmd, env=env, check=True, text=True, capture_output=True)
        if proc.stdout:
            print(proc.stdout, end="", flush=True)

    return assemble_benchmark_report_from_arm_files(
        deck_root=deck_root,
        core_manifest=core_manifest,
        ice_cover_path=ice_cover_path,
        exact_ref_dir=exact_ref_dir,
        output_dir=output_dir,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=(*ARM_IDS, "all", "assemble"),
        default="all",
        help="Benchmark arm to execute, 'all' to run all 4 arms, or 'assemble' from saved arm JSONs.",
    )
    parser.add_argument("--deck-root", type=Path, default=DEFAULT_DECK_ROOT)
    parser.add_argument("--core-manifest", type=Path, default=DEFAULT_CORE_MANIFEST)
    parser.add_argument("--ice-cover-path", type=Path, default=DEFAULT_ICE_COVER_PATH)
    parser.add_argument("--exact-ref-dir", type=Path, default=DEFAULT_EXACT_REF_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpu-device", type=str, default="0")
    parser.add_argument(
        "--variant",
        choices=("cat_a_plus_b", "cat_a_only"),
        default="cat_a_plus_b",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.arm in ("all", "assemble"):
        if args.arm == "all":
            report = run_all_arms_via_subprocesses(
                deck_root=args.deck_root,
                core_manifest=args.core_manifest,
                ice_cover_path=args.ice_cover_path,
                exact_ref_dir=args.exact_ref_dir,
                output_dir=args.output_dir,
                gpu_device=args.gpu_device,
                variant=args.variant,
            )
        else:
            report = assemble_benchmark_report_from_arm_files(
                deck_root=args.deck_root,
                core_manifest=args.core_manifest,
                ice_cover_path=args.ice_cover_path,
                exact_ref_dir=args.exact_ref_dir,
                output_dir=args.output_dir,
            )
        print(
            f"[round343] completed 4-arm report -> "
            f"{args.output_dir / 'orca2_round343_speed_benchmark_report.json'}"
        )
        for arm_id, arm in report["arms"].items():
            t = arm["timing"]
            print(
                f"  {arm_id}: compile_warmup={t['compile_warmup_total_s']:.2f}s "
                f"median={t['steady_median_s']:.4f}s/step "
                f"(odd={t['steady_odd_kt_median_s']:.4f}s, even={t['steady_even_kt_median_s']:.4f}s) "
                f"1yr_steady={t['projected_year_steady_min']:.2f}min"
            )
    else:
        arm_report = run_single_benchmark_arm(
            args.arm,
            deck_root=args.deck_root,
            core_manifest=args.core_manifest,
            ice_cover_path=args.ice_cover_path,
            exact_ref_dir=args.exact_ref_dir,
            output_dir=args.output_dir,
            variant=args.variant,
        )
        t = arm_report["timing"]
        print(
            f"[round343] {args.arm}: compile_warmup={t['compile_warmup_total_s']:.2f}s "
            f"median={t['steady_median_s']:.4f}s/step "
            f"(odd={t['steady_odd_kt_median_s']:.4f}s, even={t['steady_even_kt_median_s']:.4f}s) "
            f"1yr_steady={t['projected_year_steady_min']:.2f}min"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
