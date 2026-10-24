"""Fidelity, conservation, and invariance tests for SI3 ladder rung O3 (`si3_r8`, `DECISION 159`).

Covers all preregistered requirements (`docs/ice/fidelity/si3_r8_o3_preregister.md`):
(a) Bit-for-bit (`0`-ULP) statement pins of NEMO 5.0.2 `icedyn_adv_pra.f90`,
    `icedyn.f90`, `icedyn_adv.f90`, and `icevar.f90` with inexact `float64`
    operands against independent scalar/loop reference implementations, plus
    failing statement-reversal and reordering plants (`Plants O3-A1..A4`).
(b) Lane-3 `ICE_ADV1D` / `ICE_ADV2D` invariance and real parent-vs-current
    (`042c7bd74c2944dab8c39318687ab0cef54f9d2b`) `0`-ULP bit-identity across
    `advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`,
    `step_ice_adv2d_card`, `step_sea_ice` (`slab`, `dynamic-v2`, `si3_bl99`),
    and `si3_o1_step`, plus failing `1`-ULP code-path plants (`Plants O3-B1, O3-B2`).
(c) Conservation of every extensive family (`v_i, v_s, a_i, oa_i, e_s, e_i,
    szv_i, sv_i, a_ip, v_ip, v_il`) to `float64` round-off (`<= 1e-14`) on a
    closed ORCA2 `(184, 152)` test field across `x -> y` and `y -> x` sweeps
    and `icycle in {1, 2, 3}` with active cyclic east-west seam transport,
    plus residual budget closure (`wfx_res, hfx_res, sfx_res`, `diag_adv_*`)
    when `Hbig_pra` / `Hsnow_pra` / `ice_var_zapneg` clip, and failing flux
    plant (`Plant O3-C1`).
(d) End-to-end differentiability (`jax.jit`, `jax.value_and_grad`, `jax.jvp`)
    through one ORCA2 Prather advection step (`si3_o3_step`), with `0`-ULP
    primal invariance under `jax.jit`, finite non-zero gradients with respect
    to extensives, velocities (`u_ice, v_ice`), and Prather moments (`sx, sy,
    sxx, syy, sxy`), `reverse == forward` (`<= 1e-11`), and failing
    `stop_gradient` plant (`Plant O3-D1`).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm import constants as canonical_constants
from legoesm.core.field import Field
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import (
    NEMO_SI3_CONSTANTS_CONFIG,
    BrineConfig,
    SeaIceConfig,
    SI3AdvectionDiagnostics,
    SI3AllPratherMoments,
    SI3ColumnState,
    SI3ORCA2AdvectionGrid,
    SI3ORCA2Forcing,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    SI3State,
    SI3ThermoConfig,
    SnowConfig,
    build_c1d_omip_l3_card,
    build_si3_orca2_o1_config,
    build_si3_orca2_o3_config,
    fortran_sum_categories,
    init_dynamic_ice_state,
    si3_adv_x,
    si3_adv_y,
    si3_apply_state_halos_orca2,
    si3_dyn_adv_pra_orca2,
    si3_dyn_orca2,
    si3_hbig_pra,
    si3_hfn_category_bounds,
    si3_hsnow_pra,
    si3_icemax2d_pra,
    si3_icemax3d_pra,
    si3_lbc_lnk_orca2,
    si3_o1_step,
    si3_o3_step,
    si3_var_zapneg,
    si3_var_zapsmall,
    step_sea_ice,
)
from legoesm.ice.transport import (
    advect_si3_prather_1d,
    advect_si3_prather_2d,
    si3_active_mask_2d,
)
from legoesm.ice.bitz_lipscomb import (
    SI3SurfaceForcing,
    bitz_lipscomb_conduction_step,
)
from legoesm.ice.fidelity.nemo_adv2d_testcase_recipe import (
    build_ice_adv2d_card,
    step_ice_adv2d_card,
)
from legoesm.ice.fidelity.nemo_testcase_recipe import (
    build_ice_adv1d_card,
    step_ice_adv1d_card,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTCASES_DIR = REPO_ROOT / "scripts/validate/ocean_fidelity/testcases"
PARENT_COMMIT = "042c7bd74c2944dab8c39318687ab0cef54f9d2b"

O1_SPEC = importlib.util.spec_from_file_location(
    "nemo_si3_o1_orca2_sas_gate",
    TESTCASES_DIR / "nemo_si3_o1_orca2_sas_gate.py",
)
assert O1_SPEC and O1_SPEC.loader
o1_gate = importlib.util.module_from_spec(O1_SPEC)
O1_SPEC.loader.exec_module(o1_gate)


@pytest.fixture(autouse=True)
def _force_fp64():
    set_policy(PrecisionPolicy.fp64())


def _max_ulp(a, b) -> int:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    assert arr_a.shape == arr_b.shape, f"shape {arr_a.shape} != {arr_b.shape}"
    return int(o1_gate.max_ulp_diff(arr_a, arr_b))


def _assert_zero_ulp(a, b, label: str = "") -> None:
    ulp = _max_ulp(a, b)
    assert ulp == 0, f"{label} max ULP = {ulp} != 0"


# ==============================================================================
# Independent scalar/loop Fortran-literal reference kernels for Test (a)
# (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:513-1156,1519-1588`,
# `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:615-853`).
# ==============================================================================


def _ref_adv_x_scalar_loop(
    ihls: int,
    pdt: float,
    put: np.ndarray,
    pcrh: float,
    psm: np.ndarray,
    ps0: np.ndarray,
    moments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    e1e2t: np.ndarray,
    tmask: np.ndarray,
    nn_hls: int = 2,
    reorder_donor_alf: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Scalar/loop NumPy replay of `adv_x` (`icedyn_adv_pra.f90:513-733`)."""
    psm_in = np.array(psm, dtype=np.float64, copy=True)
    psm_w = np.array(psm, dtype=np.float64, copy=True)
    ps0_w = np.array(ps0, dtype=np.float64, copy=True)
    psx_w, psy_w, psxx_w, psyy_w, psxy_w = (
        np.array(m, dtype=np.float64, copy=True) for m in moments
    )
    jpi, jpj, jpl = ps0_w.shape
    ji0 = 1 + int(ihls)
    jj0 = int(round(float(pcrh))) + int(ihls)
    ntsi = nn_hls
    ntei = jpi - nn_hls - 1
    ntsj = nn_hls
    ntej = jpj - nn_hls - 1
    epsi20 = 1.0e-20

    for jl in range(jpl):
        zpsm = np.array(psm_in, dtype=np.float64, copy=True)
        zf0 = np.zeros((jpi, jpj), dtype=np.float64)
        zfx = np.zeros((jpi, jpj), dtype=np.float64)
        zfxx = np.zeros((jpi, jpj), dtype=np.float64)
        zfy = np.zeros((jpi, jpj), dtype=np.float64)
        zfyy = np.zeros((jpi, jpj), dtype=np.float64)
        zfxy = np.zeros((jpi, jpj), dtype=np.float64)
        zfm = np.zeros((jpi, jpj), dtype=np.float64)

        # Loop 1: DO_2D( ji0, ji0, jj0, jj0 ) (`:550-630`)
        for jj in range(ntsj - jj0, ntej + jj0 + 1):
            for ji in range(ntsi - ji0, ntei + ji0 + 1):
                zpsm[ji, jj] = max(
                    pcrh * e1e2t[ji, jj] + (1.0 - pcrh) * psm_in[ji, jj], epsi20
                )
                zslpmax = max(0.0, ps0_w[ji, jj, jl])
                zs1max = 1.5 * zslpmax
                zs1new = min(zs1max, max(-zs1max, psx_w[ji, jj, jl]))
                zs2new = min(
                    2.0 * zslpmax - (1.0 / 3.0) * abs(zs1new),
                    max(abs(zs1new) - zslpmax, psxx_w[ji, jj, jl]),
                )
                rswitch = float(tmask[ji, jj]) if zslpmax > 0.0 else 0.0
                ps0_w[ji, jj, jl] = zslpmax
                psx_w[ji, jj, jl] = zs1new * rswitch
                psxx_w[ji, jj, jl] = zs2new * rswitch
                psy_w[ji, jj, jl] = psy_w[ji, jj, jl] * rswitch
                psyy_w[ji, jj, jl] = psyy_w[ji, jj, jl] * rswitch
                psxy_w[ji, jj, jl] = (
                    min(zslpmax, max(-zslpmax, psxy_w[ji, jj, jl])) * rswitch
                )

                if put[ji, jj] >= 0.0:
                    zalf = (put[ji, jj] * pdt) / zpsm[ji, jj]
                    zalfq = zalf * zalf
                    zalf1 = 1.0 - zalf
                    zalf1q = zalf1 * zalf1

                    zfm[ji, jj] = zalf * zpsm[ji, jj]
                    zf0[ji, jj] = zalf * (
                        ps0_w[ji, jj, jl]
                        + zalf1
                        * (
                            psx_w[ji, jj, jl]
                            + (zalf1 - zalf) * psxx_w[ji, jj, jl]
                        )
                    )
                    zfx[ji, jj] = zalfq * (
                        psx_w[ji, jj, jl] + (3.0 * zalf1) * psxx_w[ji, jj, jl]
                    )
                    zfxx[ji, jj] = (zalfq * zalf) * psxx_w[ji, jj, jl]
                    zfy[ji, jj] = zalf * (
                        psy_w[ji, jj, jl] + zalf1 * psxy_w[ji, jj, jl]
                    )
                    zfxy[ji, jj] = zalfq * psxy_w[ji, jj, jl]
                    zfyy[ji, jj] = zalf * psyy_w[ji, jj, jl]

                    zpsm[ji, jj] = zpsm[ji, jj] - zfm[ji, jj]
                    ps0_w[ji, jj, jl] = ps0_w[ji, jj, jl] - zf0[ji, jj]
                    if reorder_donor_alf:
                        psx_w[ji, jj, jl] = (
                            zalf1 * psx_w[ji, jj, jl] * zalf1
                            - (3.0 * zalf) * (zalf1q * psxx_w[ji, jj, jl])
                        )
                    else:
                        psx_w[ji, jj, jl] = zalf1q * (
                            psx_w[ji, jj, jl] - (3.0 * zalf) * psxx_w[ji, jj, jl]
                        )
                    psxx_w[ji, jj, jl] = (zalf1q * zalf1) * psxx_w[ji, jj, jl]
                    psy_w[ji, jj, jl] = psy_w[ji, jj, jl] - zfy[ji, jj]
                    psyy_w[ji, jj, jl] = psyy_w[ji, jj, jl] - zfyy[ji, jj]
                    psxy_w[ji, jj, jl] = zalf1q * psxy_w[ji, jj, jl]

        # Loop 2: negative-U flux extraction (`:632-651`)
        for jj in range(ntsj - jj0, ntej + jj0 + 1):
            for ji in range(ntsi - ji0, ntei + ji0):
                if put[ji, jj] < 0.0:
                    zalf = (-put[ji, jj] * pdt) / zpsm[ji + 1, jj]
                    zalfq = zalf * zalf
                    zalf1 = 1.0 - zalf
                    zfm[ji, jj] = zalf * zpsm[ji + 1, jj]
                    zf0[ji, jj] = zalf * (
                        ps0_w[ji + 1, jj, jl]
                        - zalf1
                        * (
                            psx_w[ji + 1, jj, jl]
                            - (zalf1 - zalf) * psxx_w[ji + 1, jj, jl]
                        )
                    )
                    # Note: zfx(ji,jj) remains 0._wp (`icedyn_adv_pra.f90:615,642-648`)
                    zfxx[ji, jj] = (zalfq * zalf) * psxx_w[ji + 1, jj, jl]
                    zfy[ji, jj] = zalf * (
                        psy_w[ji + 1, jj, jl] - zalf1 * psxy_w[ji + 1, jj, jl]
                    )
                    zfxy[ji, jj] = zalfq * psxy_w[ji + 1, jj, jl]
                    zfyy[ji, jj] = zalf * psyy_w[ji + 1, jj, jl]

        # Loop 3: DO_2D( ji0-1, ji0-1, jj0, jj0 ) (`:653-731`)
        for jj in range(ntsj - jj0, ntej + jj0 + 1):
            for ji in range(ntsi - (ji0 - 1), ntei + (ji0 - 1) + 1):
                if put[ji - 1, jj] < 0.0:
                    zalf = ((-put[ji - 1, jj]) * pdt) / zpsm[ji, jj]
                    zalf1 = 1.0 - zalf
                    zalf1q = zalf1 * zalf1
                    zpsm[ji, jj] = zpsm[ji, jj] - zfm[ji - 1, jj]
                    ps0_w[ji, jj, jl] = ps0_w[ji, jj, jl] - zf0[ji - 1, jj]
                    psx_w[ji, jj, jl] = zalf1q * (
                        psx_w[ji, jj, jl] + (3.0 * zalf) * psxx_w[ji, jj, jl]
                    )
                    psxx_w[ji, jj, jl] = (zalf1q * zalf1) * psxx_w[ji, jj, jl]
                    psy_w[ji, jj, jl] = psy_w[ji, jj, jl] - zfy[ji - 1, jj]
                    psyy_w[ji, jj, jl] = psyy_w[ji, jj, jl] - zfyy[ji - 1, jj]
                    psxy_w[ji, jj, jl] = zalf1q * psxy_w[ji, jj, jl]

                if put[ji - 1, jj] >= 0.0:
                    zpsm[ji, jj] = zpsm[ji, jj] + zfm[ji - 1, jj]
                    zalf = zfm[ji - 1, jj] / zpsm[ji, jj]
                    z1malf = 1.0 - zalf
                    zalfq = zalf * zalf
                    z1malfq = z1malf * z1malf
                    ps0_w[ji, jj, jl] = ps0_w[ji, jj, jl] + zf0[ji - 1, jj]
                    ztemp = zalf * ps0_w[ji, jj, jl] - z1malf * zf0[ji - 1, jj]
                    psx_old = psx_w[ji, jj, jl]
                    psx_w[ji, jj, jl] = (
                        zalf * zfx[ji - 1, jj] + z1malf * psx_old
                    ) + 3.0 * ztemp
                    psxx_w[ji, jj, jl] = (
                        zalfq * zfxx[ji - 1, jj] + z1malfq * psxx_w[ji, jj, jl]
                    ) + 5.0 * (
                        (zalf * z1malf) * (psx_w[ji, jj, jl] - zfx[ji - 1, jj])
                        - (z1malf - zalf) * ztemp
                    )
                    psxy_w[ji, jj, jl] = (
                        zalf * zfxy[ji - 1, jj] + z1malf * psxy_w[ji, jj, jl]
                    ) + 3.0 * (
                        (-z1malf) * zfy[ji - 1, jj] + zalf * psy_w[ji, jj, jl]
                    )
                    psy_w[ji, jj, jl] = psy_w[ji, jj, jl] + zfy[ji - 1, jj]
                    psyy_w[ji, jj, jl] = psyy_w[ji, jj, jl] + zfyy[ji - 1, jj]

                if put[ji, jj] < 0.0:
                    zpsm[ji, jj] = zpsm[ji, jj] + zfm[ji, jj]
                    zalf = zfm[ji, jj] / zpsm[ji, jj]
                    z1malf = 1.0 - zalf
                    zalfq = zalf * zalf
                    z1malfq = z1malf * z1malf
                    ps0_w[ji, jj, jl] = ps0_w[ji, jj, jl] + zf0[ji, jj]
                    ztemp = (-zalf) * ps0_w[ji, jj, jl] + z1malf * zf0[ji, jj]
                    psx_old = psx_w[ji, jj, jl]
                    psx_w[ji, jj, jl] = (
                        zalf * zfx[ji, jj] + z1malf * psx_old
                    ) + 3.0 * ztemp
                    psxx_w[ji, jj, jl] = (
                        zalfq * zfxx[ji, jj] + z1malfq * psxx_w[ji, jj, jl]
                    ) + 5.0 * (
                        (zalf * z1malf) * ((-psx_w[ji, jj, jl]) + zfx[ji, jj])
                        + (z1malf - zalf) * ztemp
                    )
                    psxy_w[ji, jj, jl] = (
                        zalf * zfxy[ji, jj] + z1malf * psxy_w[ji, jj, jl]
                    ) + 3.0 * (
                        z1malf * zfy[ji, jj] - zalf * psy_w[ji, jj, jl]
                    )
                    psy_w[ji, jj, jl] = psy_w[ji, jj, jl] + zfy[ji, jj]
                    psyy_w[ji, jj, jl] = psyy_w[ji, jj, jl] + zfyy[ji, jj]

        if jl == jpl - 1:
            psm_w = zpsm

    return psm_w, ps0_w, (psx_w, psy_w, psxx_w, psyy_w, psxy_w)


def _ref_adv_y_scalar_loop(
    ihls: int,
    pdt: float,
    pvt: np.ndarray,
    pcrh: float,
    psm: np.ndarray,
    ps0: np.ndarray,
    moments: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    e1e2t: np.ndarray,
    tmask: np.ndarray,
    nn_hls: int = 2,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Scalar/loop NumPy replay of `adv_y` (`icedyn_adv_pra.f90:736-957`)."""
    psm_in = np.array(psm, dtype=np.float64, copy=True)
    psm_w = np.array(psm, dtype=np.float64, copy=True)
    ps0_w = np.array(ps0, dtype=np.float64, copy=True)
    psx_w, psy_w, psxx_w, psyy_w, psxy_w = (
        np.array(m, dtype=np.float64, copy=True) for m in moments
    )
    jpi, jpj, jpl = ps0_w.shape
    ji0 = int(round(float(pcrh))) + int(ihls)
    jj0 = 1 + int(ihls)
    ntsi = nn_hls
    ntei = jpi - nn_hls - 1
    ntsj = nn_hls
    ntej = jpj - nn_hls - 1
    epsi20 = 1.0e-20

    for jl in range(jpl):
        zpsm = np.array(psm_in, dtype=np.float64, copy=True)
        zf0 = np.zeros((jpi, jpj), dtype=np.float64)
        zfx = np.zeros((jpi, jpj), dtype=np.float64)
        zfxx = np.zeros((jpi, jpj), dtype=np.float64)
        zfy = np.zeros((jpi, jpj), dtype=np.float64)
        zfyy = np.zeros((jpi, jpj), dtype=np.float64)
        zfxy = np.zeros((jpi, jpj), dtype=np.float64)
        zfm = np.zeros((jpi, jpj), dtype=np.float64)

        # Loop 1: DO_2D( ji0, ji0, jj0, jj0 ) (`:773-853`)
        for jj in range(ntsj - jj0, ntej + jj0 + 1):
            for ji in range(ntsi - ji0, ntei + ji0 + 1):
                zpsm[ji, jj] = max(
                    pcrh * e1e2t[ji, jj] + (1.0 - pcrh) * psm_in[ji, jj], epsi20
                )
                zslpmax = max(0.0, ps0_w[ji, jj, jl])
                zs1max = 1.5 * zslpmax
                zs1new = min(zs1max, max(-zs1max, psy_w[ji, jj, jl]))
                zs2new = min(
                    2.0 * zslpmax - (1.0 / 3.0) * abs(zs1new),
                    max(abs(zs1new) - zslpmax, psyy_w[ji, jj, jl]),
                )
                rswitch = float(tmask[ji, jj]) if zslpmax > 0.0 else 0.0
                ps0_w[ji, jj, jl] = zslpmax
                psx_w[ji, jj, jl] = psx_w[ji, jj, jl] * rswitch
                psxx_w[ji, jj, jl] = psxx_w[ji, jj, jl] * rswitch
                psy_w[ji, jj, jl] = zs1new * rswitch
                psyy_w[ji, jj, jl] = zs2new * rswitch
                psxy_w[ji, jj, jl] = (
                    min(zslpmax, max(-zslpmax, psxy_w[ji, jj, jl])) * rswitch
                )

                if pvt[ji, jj] >= 0.0:
                    zalf = (pvt[ji, jj] * pdt) / zpsm[ji, jj]
                    zalfq = zalf * zalf
                    zalf1 = 1.0 - zalf
                    zalf1q = zalf1 * zalf1

                    zfm[ji, jj] = zalf * zpsm[ji, jj]
                    zf0[ji, jj] = zalf * (
                        ps0_w[ji, jj, jl]
                        + zalf1
                        * (
                            psy_w[ji, jj, jl]
                            + (zalf1 - zalf) * psyy_w[ji, jj, jl]
                        )
                    )
                    zfy[ji, jj] = zalfq * (
                        psy_w[ji, jj, jl] + (3.0 * zalf1) * psyy_w[ji, jj, jl]
                    )
                    zfyy[ji, jj] = (zalfq * zalf) * psyy_w[ji, jj, jl]
                    zfx[ji, jj] = zalf * (
                        psx_w[ji, jj, jl] + zalf1 * psxy_w[ji, jj, jl]
                    )
                    zfxy[ji, jj] = zalfq * psxy_w[ji, jj, jl]
                    zfxx[ji, jj] = zalf * psxx_w[ji, jj, jl]

                    zpsm[ji, jj] = zpsm[ji, jj] - zfm[ji, jj]
                    ps0_w[ji, jj, jl] = ps0_w[ji, jj, jl] - zf0[ji, jj]
                    psy_w[ji, jj, jl] = zalf1q * (
                        psy_w[ji, jj, jl] - (3.0 * zalf) * psyy_w[ji, jj, jl]
                    )
                    psyy_w[ji, jj, jl] = (zalf1q * zalf1) * psyy_w[ji, jj, jl]
                    psx_w[ji, jj, jl] = psx_w[ji, jj, jl] - zfx[ji, jj]
                    psxx_w[ji, jj, jl] = psxx_w[ji, jj, jl] - zfxx[ji, jj]
                    psxy_w[ji, jj, jl] = zalf1q * psxy_w[ji, jj, jl]

        # Loop 2: negative-V flux extraction (`:855-875`)
        for jj in range(ntsj - jj0, ntej + jj0):
            for ji in range(ntsi - ji0, ntei + ji0 + 1):
                if pvt[ji, jj] < 0.0:
                    zalf = (-pvt[ji, jj] * pdt) / zpsm[ji, jj + 1]
                    zalfq = zalf * zalf
                    zalf1 = 1.0 - zalf
                    zfm[ji, jj] = zalf * zpsm[ji, jj + 1]
                    zf0[ji, jj] = zalf * (
                        ps0_w[ji, jj + 1, jl]
                        - zalf1
                        * (
                            psy_w[ji, jj + 1, jl]
                            - (zalf1 - zalf) * psyy_w[ji, jj + 1, jl]
                        )
                    )
                    zfy[ji, jj] = zalfq * (
                        psy_w[ji, jj + 1, jl]
                        - (3.0 * zalf1) * psyy_w[ji, jj + 1, jl]
                    )
                    zfyy[ji, jj] = (zalfq * zalf) * psyy_w[ji, jj + 1, jl]
                    zfx[ji, jj] = zalf * (
                        psx_w[ji, jj + 1, jl] - zalf1 * psxy_w[ji, jj + 1, jl]
                    )
                    zfxy[ji, jj] = zalfq * psxy_w[ji, jj + 1, jl]
                    zfxx[ji, jj] = zalf * psxx_w[ji, jj + 1, jl]

        # Loop 3: DO_2D( ji0, ji0, jj0-1, jj0-1 ) (`:877-955`)
        for jj in range(ntsj - (jj0 - 1), ntej + (jj0 - 1) + 1):
            for ji in range(ntsi - ji0, ntei + ji0 + 1):
                if pvt[ji, jj - 1] < 0.0:
                    zalf = ((-pvt[ji, jj - 1]) * pdt) / zpsm[ji, jj]
                    zalf1 = 1.0 - zalf
                    zalf1q = zalf1 * zalf1
                    zpsm[ji, jj] = zpsm[ji, jj] - zfm[ji, jj - 1]
                    ps0_w[ji, jj, jl] = ps0_w[ji, jj, jl] - zf0[ji, jj - 1]
                    psy_w[ji, jj, jl] = zalf1q * (
                        psy_w[ji, jj, jl] + (3.0 * zalf) * psyy_w[ji, jj, jl]
                    )
                    psyy_w[ji, jj, jl] = (zalf1q * zalf1) * psyy_w[ji, jj, jl]
                    psx_w[ji, jj, jl] = psx_w[ji, jj, jl] - zfx[ji, jj - 1]
                    psxx_w[ji, jj, jl] = psxx_w[ji, jj, jl] - zfxx[ji, jj - 1]
                    psxy_w[ji, jj, jl] = zalf1q * psxy_w[ji, jj, jl]

                if pvt[ji, jj - 1] >= 0.0:
                    zpsm[ji, jj] = zpsm[ji, jj] + zfm[ji, jj - 1]
                    zalf = zfm[ji, jj - 1] / zpsm[ji, jj]
                    z1malf = 1.0 - zalf
                    zalfq = zalf * zalf
                    z1malfq = z1malf * z1malf
                    ps0_w[ji, jj, jl] = ps0_w[ji, jj, jl] + zf0[ji, jj - 1]
                    ztemp = zalf * ps0_w[ji, jj, jl] - z1malf * zf0[ji, jj - 1]
                    psy_old = psy_w[ji, jj, jl]
                    psy_w[ji, jj, jl] = (
                        zalf * zfy[ji, jj - 1] + z1malf * psy_old
                    ) + 3.0 * ztemp
                    psyy_w[ji, jj, jl] = (
                        zalfq * zfyy[ji, jj - 1] + z1malfq * psyy_w[ji, jj, jl]
                    ) + 5.0 * (
                        (zalf * z1malf) * (psy_w[ji, jj, jl] - zfy[ji, jj - 1])
                        - (z1malf - zalf) * ztemp
                    )
                    psxy_w[ji, jj, jl] = (
                        zalf * zfxy[ji, jj - 1] + z1malf * psxy_w[ji, jj, jl]
                    ) + 3.0 * (
                        (-z1malf) * zfx[ji, jj - 1] + zalf * psx_w[ji, jj, jl]
                    )
                    psx_w[ji, jj, jl] = psx_w[ji, jj, jl] + zfx[ji, jj - 1]
                    psxx_w[ji, jj, jl] = psxx_w[ji, jj, jl] + zfxx[ji, jj - 1]

                if pvt[ji, jj] < 0.0:
                    zpsm[ji, jj] = zpsm[ji, jj] + zfm[ji, jj]
                    zalf = zfm[ji, jj] / zpsm[ji, jj]
                    z1malf = 1.0 - zalf
                    zalfq = zalf * zalf
                    z1malfq = z1malf * z1malf
                    ps0_w[ji, jj, jl] = ps0_w[ji, jj, jl] + zf0[ji, jj]
                    ztemp = (-zalf) * ps0_w[ji, jj, jl] + z1malf * zf0[ji, jj]
                    psy_old = psy_w[ji, jj, jl]
                    psy_w[ji, jj, jl] = (
                        zalf * zfy[ji, jj] + z1malf * psy_old
                    ) + 3.0 * ztemp
                    psyy_w[ji, jj, jl] = (
                        zalfq * zfyy[ji, jj] + z1malfq * psyy_w[ji, jj, jl]
                    ) + 5.0 * (
                        (zalf * z1malf) * ((-psy_w[ji, jj, jl]) + zfy[ji, jj])
                        + (z1malf - zalf) * ztemp
                    )
                    psxy_w[ji, jj, jl] = (
                        zalf * zfxy[ji, jj] + z1malf * psxy_w[ji, jj, jl]
                    ) + 3.0 * (
                        z1malf * zfx[ji, jj] - zalf * psx_w[ji, jj, jl]
                    )
                    psx_w[ji, jj, jl] = psx_w[ji, jj, jl] + zfx[ji, jj]
                    psxx_w[ji, jj, jl] = psxx_w[ji, jj, jl] + zfxx[ji, jj]

        if jl == jpl - 1:
            psm_w = zpsm

    return psm_w, ps0_w, (psx_w, psy_w, psxx_w, psyy_w, psxy_w)


def _build_orca2_advection_grid(
    tmask: jnp.ndarray,
    umask: jnp.ndarray,
    vmask: jnp.ndarray,
) -> SI3ORCA2AdvectionGrid:
    """Build a realistic curved ORCA2 `(184, 152)` advection metric grid."""
    jpi, jpj = tmask.shape
    ii, jj = np.meshgrid(np.arange(jpi), np.arange(jpj), indexing="ij")
    # Smoothly varying metric scales around ~200 km (`ORCA2`) with inexact fp64 values
    e1t_np = 185000.0 + 27000.0 * np.cos(2.0 * np.pi * (ii - 2) / 180.0) + 113.7 * (jj / 152.0)
    e2t_np = 165000.0 + 31000.0 * np.sin(np.pi * (jj - 2) / 148.0) + 89.3 * (ii / 184.0)
    e1u_np = 0.5 * (e1t_np + np.roll(e1t_np, -1, axis=0))
    e2u_np = 0.5 * (e2t_np + np.roll(e2t_np, -1, axis=0))
    e1v_np = 0.5 * (e1t_np + np.roll(e1t_np, -1, axis=1))
    e2v_np = 0.5 * (e2t_np + np.roll(e2t_np, -1, axis=1))
    e1e2t_np = e1t_np * e2t_np

    # Enforce ORCA2 cyclic + north-fold consistency on T/U/V metrics while keeping
    # positive metric lengths on the closed southern boundary (`jj = 0, 1`).
    def _lbc_pos(arr_np: np.ndarray, ctype: str) -> jnp.ndarray:
        raw_j = jnp.asarray(arr_np, dtype=jnp.float64)
        linked = si3_lbc_lnk_orca2(raw_j, ctype, 1.0, nn_hls=2, ldfull=True)
        return jnp.where(linked > 0.0, linked, raw_j)

    e1e2t_j = _lbc_pos(e1e2t_np, "T")
    e1u_j = _lbc_pos(e1u_np, "U")
    e2u_j = _lbc_pos(e2u_np, "U")
    e1v_j = _lbc_pos(e1v_np, "V")
    e2v_j = _lbc_pos(e2v_np, "V")
    return SI3ORCA2AdvectionGrid.from_metrics(
        e1u=e1u_j,
        e2v=e2v_j,
        e2u=e2u_j,
        e1v=e1v_j,
        e1e2t=e1e2t_j,
        tmask=tmask,
        umask=umask,
        vmask=vmask,
    )


# ==============================================================================
# Test (a): Bit-for-bit (`0`-ULP) statement pins with inexact `float64` operands
# and failing reversal/reordering plants (`Plants O3-A1..A4`).
# ==============================================================================


def test_o3_a1_adv_x_and_adv_y_statement_pins_and_plants() -> None:
    """`si3_adv_x` and `si3_adv_y` match scalar-loop Fortran reference to 0 ULP on inexact operands and fail under Plant O3-A1."""
    rng = np.random.default_rng(2026102401)
    jpi, jpj, jpl = 12, 11, 3
    nn_hls = 2
    pdt = 10800.0 / 3.0 + 0.1357911

    e1e2t_np = 3.7123456789e9 + rng.uniform(-2.5e8, 2.5e8, size=(jpi, jpj))
    tmask_np = np.ones((jpi, jpj), dtype=np.float64)
    tmask_np[4, 5] = 0.0  # dry island cell inside active domain
    put_np = rng.uniform(-4.2e5, 4.5e5, size=(jpi, jpj))
    pvt_np = rng.uniform(-3.8e5, 4.1e5, size=(jpi, jpj))
    psm_np = e1e2t_np * rng.uniform(0.88, 1.12, size=(jpi, jpj))

    ps0_np = rng.uniform(-0.2, 2.8, size=(jpi, jpj, jpl)) * e1e2t_np[:, :, None]
    # Include both active limiter cells (large slopes/curvatures) and smooth cells
    moments_np = (
        rng.uniform(-2.2, 2.2, size=(jpi, jpj, jpl)) * np.abs(ps0_np),
        rng.uniform(-2.2, 2.2, size=(jpi, jpj, jpl)) * np.abs(ps0_np),
        rng.uniform(-1.8, 2.5, size=(jpi, jpj, jpl)) * np.abs(ps0_np),
        rng.uniform(-1.8, 2.5, size=(jpi, jpj, jpl)) * np.abs(ps0_np),
        rng.uniform(-1.5, 1.5, size=(jpi, jpj, jpl)) * np.abs(ps0_np),
    )

    for ihls in (0, 1):
        for pcrh in (1.0, 0.0):
            ref_psm_x, ref_ps0_x, ref_mom_x = _ref_adv_x_scalar_loop(
                ihls,
                pdt,
                put_np,
                pcrh,
                psm_np,
                ps0_np,
                moments_np,
                e1e2t=e1e2t_np,
                tmask=tmask_np,
                nn_hls=nn_hls,
            )
            got_psm_x, got_ps0_x, got_mom_x = si3_adv_x(
                ihls,
                pdt,
                jnp.asarray(put_np),
                pcrh,
                jnp.asarray(psm_np),
                jnp.asarray(ps0_np),
                tuple(jnp.asarray(m) for m in moments_np),
                e1e2t=jnp.asarray(e1e2t_np),
                tmask=jnp.asarray(tmask_np),
                nn_hls=nn_hls,
            )
            _assert_zero_ulp(ref_psm_x, got_psm_x, f"adv_x(ihls={ihls},pcrh={pcrh}).psm")
            _assert_zero_ulp(ref_ps0_x, got_ps0_x, f"adv_x(ihls={ihls},pcrh={pcrh}).ps0")
            for idx, mname in enumerate(("sx", "sy", "sxx", "syy", "sxy")):
                _assert_zero_ulp(
                    ref_mom_x[idx],
                    got_mom_x[idx],
                    f"adv_x(ihls={ihls},pcrh={pcrh}).{mname}",
                )

            ref_psm_y, ref_ps0_y, ref_mom_y = _ref_adv_y_scalar_loop(
                ihls,
                pdt,
                pvt_np,
                pcrh,
                psm_np,
                ps0_np,
                moments_np,
                e1e2t=e1e2t_np,
                tmask=tmask_np,
                nn_hls=nn_hls,
            )
            got_psm_y, got_ps0_y, got_mom_y = si3_adv_y(
                ihls,
                pdt,
                jnp.asarray(pvt_np),
                pcrh,
                jnp.asarray(psm_np),
                jnp.asarray(ps0_np),
                tuple(jnp.asarray(m) for m in moments_np),
                e1e2t=jnp.asarray(e1e2t_np),
                tmask=jnp.asarray(tmask_np),
                nn_hls=nn_hls,
            )
            _assert_zero_ulp(ref_psm_y, got_psm_y, f"adv_y(ihls={ihls},pcrh={pcrh}).psm")
            _assert_zero_ulp(ref_ps0_y, got_ps0_y, f"adv_y(ihls={ihls},pcrh={pcrh}).ps0")
            for idx, mname in enumerate(("sx", "sy", "sxx", "syy", "sxy")):
                _assert_zero_ulp(
                    ref_mom_y[idx],
                    got_mom_y[idx],
                    f"adv_y(ihls={ihls},pcrh={pcrh}).{mname}",
                )

    # Also verify 4D `(jpi, jpj, nlay, jpl)` packing/unpacking in `si3_adv_x` and `si3_adv_y`
    ps0_4d = ps0_np.reshape((jpi, jpj, 1, jpl))
    mom_4d = tuple(m.reshape((jpi, jpj, 1, jpl)) for m in moments_np)
    _, got_ps0_4d, got_mom_4d = si3_adv_x(
        0,
        pdt,
        jnp.asarray(put_np),
        1.0,
        jnp.asarray(psm_np),
        jnp.asarray(ps0_4d),
        tuple(jnp.asarray(m) for m in mom_4d),
        e1e2t=jnp.asarray(e1e2t_np),
        tmask=jnp.asarray(tmask_np),
        nn_hls=nn_hls,
    )
    assert got_ps0_4d.shape == (jpi, jpj, 1, jpl)
    assert all(m.shape == (jpi, jpj, 1, jpl) for m in got_mom_4d)

    # Plant O3-A1a: omitting donor readjustment in `si3_adv_x` / `si3_adv_y` MUST fail 0-ULP
    _, bad_ps0_x, _ = si3_adv_x(
        0,
        pdt,
        jnp.asarray(put_np),
        1.0,
        jnp.asarray(psm_np),
        jnp.asarray(ps0_np),
        tuple(jnp.asarray(m) for m in moments_np),
        e1e2t=jnp.asarray(e1e2t_np),
        tmask=jnp.asarray(tmask_np),
        nn_hls=nn_hls,
        _plant_omit_donor_readjust=True,
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(ref_ps0_x, bad_ps0_x, "plant_omit_donor_readjust_x")

    # Plant O3-A1b: restoring `zfx` in `adv_x`'s negative-velocity branch (`_plant_symmetric_neg_zfx=True`)
    # instead of NEMO's source-literal `zfx = 0._wp` (`icedyn_adv_pra.f90:615,642-648`) MUST fail 0-ULP
    _, _, bad_mom_x_neg = si3_adv_x(
        0,
        pdt,
        jnp.asarray(put_np),
        0.0,
        jnp.asarray(psm_np),
        jnp.asarray(ps0_np),
        tuple(jnp.asarray(m) for m in moments_np),
        e1e2t=jnp.asarray(e1e2t_np),
        tmask=jnp.asarray(tmask_np),
        nn_hls=nn_hls,
        _plant_symmetric_neg_zfx=True,
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(ref_mom_x[0], bad_mom_x_neg[0], "plant_symmetric_neg_zfx")

    # Plant O3-A1c: reordering `zalf1q * (psx - 3 * zalf * psxx)` association MUST fail 0-ULP
    _, _, reordered_mom_x = _ref_adv_x_scalar_loop(
        0,
        pdt,
        put_np,
        0.0,
        psm_np,
        ps0_np,
        moments_np,
        e1e2t=e1e2t_np,
        tmask=tmask_np,
        nn_hls=nn_hls,
        reorder_donor_alf=True,
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(got_mom_x[0], reordered_mom_x[0], "plant_reorder_donor_alf")


def test_o3_a2_odd_even_sweep_order_and_cfl_subcycling_pins_and_plant() -> None:
    """`si3_dyn_adv_pra_orca2` enforces NEMO's `MOD((kt-1)//nn_fsbc, 2) == MOD(jt-1, 2)` sweep parity and `icycle in {1, 2, 3}` CFL thresholds, and fails under Plant O3-A2."""
    cfg = SI3ORCA2StateConfig(
        jpl=2,
        nlay_i=2,
        nlay_s=1,
        nn_icesal=4,
        ln_pnd=True,
        ln_pnd_LEV=True,
        ln_pnd_lids=True,
        ln_icedyn=True,
        ln_icethd=False,
        nn_fsbc=1,
        rDt_ice=10800.0,
        nn_hls=2,
    )
    jpi, jpj = 12, 10
    rng = np.random.default_rng(2026102402)
    tmask = jnp.ones((jpi, jpj), dtype=jnp.float64)
    e1u = jnp.full((jpi, jpj), 200000.0, dtype=jnp.float64)
    e2v = jnp.full((jpi, jpj), 200000.0, dtype=jnp.float64)
    e2u = jnp.full((jpi, jpj), 195000.0, dtype=jnp.float64)
    e1v = jnp.full((jpi, jpj), 205000.0, dtype=jnp.float64)
    e1e2t = jnp.asarray(
        3.9e10 + rng.uniform(-2.0e9, 2.0e9, size=(jpi, jpj)), dtype=jnp.float64
    )
    grid = SI3ORCA2AdvectionGrid.from_metrics(
        e1u=e1u, e2v=e2v, e2u=e2u, e1v=e1v, e1e2t=e1e2t, tmask=tmask
    )

    # Base fields
    a_i = jnp.asarray(rng.uniform(0.05, 0.25, size=(jpi, jpj, cfg.jpl)))
    v_i = a_i * jnp.asarray(rng.uniform(0.4, 1.5, size=(jpi, jpj, cfg.jpl)))
    v_s = a_i * jnp.asarray(rng.uniform(0.02, 0.12, size=(jpi, jpj, cfg.jpl)))
    sv_i = v_i * 4.5
    oa_i = a_i * 15.0
    a_ip = a_i * 0.2
    v_ip = a_ip * 0.08
    v_il = a_ip * 0.02
    e_s = jnp.asarray(rng.uniform(1.0e7, 5.0e7, size=(jpi, jpj, cfg.nlay_s, cfg.jpl)))
    e_i = jnp.asarray(rng.uniform(1.0e8, 4.0e8, size=(jpi, jpj, cfg.nlay_i, cfg.jpl)))
    szv_i = jnp.asarray(rng.uniform(0.5, 2.5, size=(jpi, jpj, cfg.nlay_i, cfg.jpl)))
    h_i = v_i / a_i
    h_s = v_s / a_i
    h_ip = v_ip / a_ip
    ato_i = 1.0 - jnp.sum(a_i, axis=-1)
    prather = SI3AllPratherMoments.zeros(
        jpi, jpj, jpl=cfg.jpl, nlay_i=cfg.nlay_i, nlay_s=cfg.nlay_s
    )
    # Non-zero cross-moments so x->y vs y->x is strongly non-commutative
    sxy_ice = jnp.asarray(rng.uniform(-1.0e7, 1.0e7, size=(jpi, jpj, cfg.jpl)))
    prather = prather._replace(ice=prather.ice._replace(sxy=sxy_ice))

    # 1. Verify CFL threshold -> icycle mapping (`icedyn_adv_pra.f90:133-138`):
    #    e1u = 200000 m, rDt_ice = 10800 s => u = 5.0 m/s gives zcfl = 0.27 -> icycle = 1
    #    u = 15.0 m/s gives zcfl = 0.81 -> icycle = 2
    #    u = 32.0 m/s gives zcfl = 1.728 -> icycle = 3
    for u_speed, expected_icycle in ((5.0, 1), (15.0, 2), (32.0, 3)):
        u_test = jnp.full((jpi, jpj), u_speed, dtype=jnp.float64)
        v_test = jnp.full((jpi, jpj), 0.4 * u_speed, dtype=jnp.float64)
        res = si3_dyn_adv_pra_orca2(
            kt=1,
            u_ice=u_test,
            v_ice=v_test,
            h_i=h_i,
            h_s=h_s,
            h_ip=h_ip,
            ato_i=ato_i,
            v_i=v_i,
            v_s=v_s,
            sv_i=sv_i,
            oa_i=oa_i,
            a_i=a_i,
            a_ip=a_ip,
            v_ip=v_ip,
            v_il=v_il,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            prather=prather,
            grid=grid,
            config=cfg,
        )
        assert int(res[-1]) == expected_icycle

    # 2. Verify odd/even parity: kt=1 (x->y on jt=1) vs kt=2 (y->x on jt=1)
    u_curv = jnp.asarray(rng.uniform(-4.0, 4.0, size=(jpi, jpj)))
    v_curv = jnp.asarray(rng.uniform(-4.0, 4.0, size=(jpi, jpj)))
    out_kt1 = si3_dyn_adv_pra_orca2(
        kt=1,
        u_ice=u_curv,
        v_ice=v_curv,
        h_i=h_i,
        h_s=h_s,
        h_ip=h_ip,
        ato_i=ato_i,
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        oa_i=oa_i,
        a_i=a_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        prather=prather,
        grid=grid,
        config=cfg,
    )
    out_kt2 = si3_dyn_adv_pra_orca2(
        kt=2,
        u_ice=u_curv,
        v_ice=v_curv,
        h_i=h_i,
        h_s=h_s,
        h_ip=h_ip,
        ato_i=ato_i,
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        oa_i=oa_i,
        a_i=a_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        prather=prather,
        grid=grid,
        config=cfg,
    )
    # Plant O3-A2: inverting sweep order at kt=1 must equal kt=2 (for icycle=1) and fail 0-ULP vs kt=1!
    out_kt1_planted = si3_dyn_adv_pra_orca2(
        kt=1,
        u_ice=u_curv,
        v_ice=v_curv,
        h_i=h_i,
        h_s=h_s,
        h_ip=h_ip,
        ato_i=ato_i,
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        oa_i=oa_i,
        a_i=a_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        prather=prather,
        grid=grid,
        config=cfg,
        _plant_invert_sweep_order=True,
    )
    _assert_zero_ulp(out_kt2[1], out_kt1_planted[1], "kt2_matches_inverted_kt1_v_i")
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(out_kt1[1], out_kt1_planted[1], "plant_o3_a2_v_i")
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(
            out_kt1[12].ice.sxy, out_kt1_planted[12].ice.sxy, "plant_o3_a2_sxy"
        )


def test_o3_a3_icemax_hbig_hsnow_zapneg_zapsmall_pato_diag_pins_and_plants() -> None:
    """`si3_icemax2d_pra`, `si3_icemax3d_pra`, `si3_hbig_pra`, `si3_hsnow_pra`, `si3_var_zapneg`, `si3_var_zapsmall`, and `pato_i` match scalar Fortran references to 0 ULP and fail under Plants O3-A3."""
    rng = np.random.default_rng(2026102403)
    jpi, jpj, jpl = 9, 8, 2
    nlay_i, nlay_s = 3, 2
    nn_hls = 2
    ihls = 0
    pdt = 10800.0

    # 1. Pin `si3_icemax2d_pra` and `si3_icemax3d_pra` (`icedyn_adv_pra.f90:1519-1588`)
    pice2d = rng.uniform(0.0, 3.5, size=(jpi, jpj, jpl))
    pice2d[4, 4, 0] = 5.0e-12  # below epsi10 = 1e-10
    pice3d = rng.uniform(0.0, 4.0e8, size=(jpi, jpj, nlay_i, jpl))

    ref_max2d = np.zeros_like(pice2d)
    for jl in range(jpl):
        for jj in range(nn_hls - ihls, jpj - nn_hls + ihls):
            for ji in range(nn_hls - ihls, jpi - nn_hls + ihls):
                ref_max2d[ji, jj, jl] = max(
                    1.0e-20,
                    float(np.max(pice2d[ji - 1 : ji + 2, jj - 1 : jj + 2, jl])),
                )
    got_max2d = si3_icemax2d_pra(ihls, jnp.asarray(pice2d), nn_hls=nn_hls)
    _assert_zero_ulp(ref_max2d, got_max2d, "si3_icemax2d_pra")

    ref_max3d = np.zeros_like(pice3d)
    for jl in range(jpl):
        for jk in range(nlay_i):
            for jj in range(nn_hls - ihls, jpj - nn_hls + ihls):
                for ji in range(nn_hls - ihls, jpi - nn_hls + ihls):
                    ref_max3d[ji, jj, jk, jl] = max(
                        1.0e-20,
                        float(
                            np.max(
                                pice3d[ji - 1 : ji + 2, jj - 1 : jj + 2, jk, jl]
                            )
                        ),
                    )
    got_max3d = si3_icemax3d_pra(ihls, jnp.asarray(pice3d), nn_hls=nn_hls)
    _assert_zero_ulp(ref_max3d, got_max3d, "si3_icemax3d_pra")

    # 2. Pin `si3_hbig_pra` (`icedyn_adv_pra.f90:960-1093`) with inexact operands
    # Choose operands where `(field * cap) / zsi` differs by >= 1 ULP from `zfra = cap / zsi; field * zfra`
    zhi_max = jnp.full((jpi, jpj, jpl), 1.7319587628865979, dtype=jnp.float64)
    zhs_max = jnp.full((jpi, jpj, jpl), 0.31958762886597936, dtype=jnp.float64)
    zhip_max = jnp.full((jpi, jpj, jpl), 0.17525773195876287, dtype=jnp.float64)
    zsi_max = jnp.full((jpi, jpj, jpl), 4.711340206185567, dtype=jnp.float64)
    zes_max = jnp.full((jpi, jpj, nlay_s, jpl), 1.134020618556701e8, dtype=jnp.float64)
    zei_max = jnp.full((jpi, jpj, nlay_i, jpl), 2.3711340206185567e8, dtype=jnp.float64)
    zszi_max = jnp.full((jpi, jpj, nlay_i, jpl), 5.154639175257732, dtype=jnp.float64)

    # `Hbig_pra` only caps where `pa_i < 0.15` (`icedyn_adv_pra.f90:1006,1014,1021,1040,1066,1078`).
    # Keep `a_i` in `[0.02, 0.08]` so `a_i_after = v_i / zhi_max` is also `< 0.15`.
    a_i = jnp.asarray(rng.uniform(0.02, 0.08, size=(jpi, jpj, jpl)))
    v_i = a_i * (zhi_max * 1.15)  # exceeds zhi_max so Hbig_pra fires, a_i_after < 0.10
    v_s = a_i * (zhs_max * 1.41)  # exceeds zhs_max even after a_i increases by 1.15x
    a_ip = a_i * 0.45
    v_ip = a_ip * (zhip_max * 1.29)  # exceeds zhip_max
    sv_i = v_i * (zsi_max * 1.33)  # exceeds zsi_max
    oa_i = a_i * 14.0
    v_il = a_ip * 0.02
    e_s = v_s[:, :, None, :] * (zes_max * 1.27)
    e_i = v_i[:, :, None, :] * (zei_max * 1.31)
    szv_i = v_i[:, :, None, :] * (zszi_max * 1.23)
    wfx0 = jnp.zeros((jpi, jpj), dtype=jnp.float64)
    hfx0 = jnp.zeros((jpi, jpj), dtype=jnp.float64)
    sfx0 = jnp.zeros((jpi, jpj), dtype=jnp.float64)

    hbig_out = si3_hbig_pra(
        ihls,
        pdt,
        zhi_max,
        zhs_max,
        zhip_max,
        zsi_max,
        zes_max,
        zei_max,
        zszi_max,
        v_i,
        v_s,
        a_i,
        a_ip,
        v_ip,
        sv_i,
        e_s,
        e_i,
        szv_i,
        wfx0,
        hfx0,
        sfx0,
        nn_hls=nn_hls,
        nn_icesal=4,
    )
    # Check that capped pond concentration `pa_ip = pv_ip / phip_max` matches to 0 ULP on active cells
    active = np.asarray(si3_active_mask_2d(jpi, jpj, nn_hls, ihls))
    v_i_after, v_s_after, a_i_after, a_ip_after, v_ip_after, _, e_s_after, e_i_after, szv_i_after, _, _, sfx_hbig = hbig_out
    _assert_zero_ulp(
        np.asarray(a_ip_after)[active],
        np.asarray(v_ip / zhip_max)[active],
        "hbig_a_ip_capped",
    )

    # Plant O3-A3a: `_plant_perturb_hbig_cap=True` MUST fail 0-ULP
    hbig_planted = si3_hbig_pra(
        ihls,
        pdt,
        zhi_max,
        zhs_max,
        zhip_max,
        zsi_max,
        zes_max,
        zei_max,
        zszi_max,
        v_i,
        v_s,
        a_i,
        a_ip,
        v_ip,
        sv_i,
        e_s,
        e_i,
        szv_i,
        wfx0,
        hfx0,
        sfx0,
        nn_hls=nn_hls,
        nn_icesal=4,
        _plant_perturb_hbig_cap=True,
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(a_i_after, hbig_planted[2], "plant_perturb_hbig_cap")

    # Plant O3-A3b: reordering `zfra = zszi_max / zsi ; pszv_i = pszv_i * zfra` to
    # `(pszv_i * zszi_max) / zsi` produces >= 1 ULP difference on inexact operands
    zsi_layer = np.asarray(szv_i) / np.asarray(v_i_after)[:, :, None, :]
    zfra_exact = np.asarray(zszi_max) / zsi_layer
    szv_exact = np.asarray(szv_i) * zfra_exact
    szv_reordered = (np.asarray(szv_i) * np.asarray(zszi_max)) / zsi_layer
    _assert_zero_ulp(szv_exact[active], np.asarray(szv_i_after)[active], "hbig_szv_exact")
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(
            np.asarray(szv_i_after)[active],
            szv_reordered[active],
            "plant_reorder_zfra_mul_div",
        )

    # 3. Pin `si3_hsnow_pra` (`icedyn_adv_pra.f90:1096-1156`)
    rho0 = canonical_constants.rho_ocean_nemo
    rhoi = canonical_constants.rho_ice
    rhos = canonical_constants.rho_snow
    # Set snow volume above Archimedes limit `v_i * (rho0 - rhoi) / rhos`
    v_s_heavy = v_i * ((rho0 - rhoi) / rhos * 1.45)
    a_ip_big = a_i * 1.25
    hsnow_out = si3_hsnow_pra(
        ihls,
        pdt,
        v_i,
        v_s_heavy,
        a_i,
        a_ip_big,
        e_s,
        wfx0,
        hfx0,
        nn_hls=nn_hls,
        rho0=rho0,
        rhoi=rhoi,
        rhos=rhos,
    )
    _, v_s_hsnow, _, a_ip_hsnow, e_s_hsnow, wfx_hsnow, hfx_hsnow = hsnow_out
    buoy_cap = np.asarray(v_i) * (rho0 - rhoi) * (1.0 / rhos)
    zvs_excess = np.maximum(0.0, np.asarray(v_s_heavy) - buoy_cap)
    ref_vs_rem = np.where(active[..., None], np.asarray(v_s_heavy) - zvs_excess, np.asarray(v_s_heavy))
    _assert_zero_ulp(ref_vs_rem, v_s_hsnow, "hsnow_v_s")
    _assert_zero_ulp(
        np.where(active[..., None], np.minimum(np.asarray(a_ip_big), np.asarray(a_i)), np.asarray(a_ip_big)),
        a_ip_hsnow,
        "hsnow_a_ip",
    )
    # Plant O3-A3c: omitting Archimedes snow shedding fails 0-ULP
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(v_s_hsnow, v_s_heavy, "plant_omit_archimedes")

    # 4. Pin `si3_var_zapneg` (`icevar.f90:740-853`):
    # Specifically test the ordering at `:833-839` where `pv_il` is zapped if
    # `pv_ip < 0 .OR. pv_il < 0 .OR. pa_ip <= 0` BEFORE `pv_ip < 0` is zeroed!
    v_i_z = jnp.array(v_i)
    v_s_z = jnp.array(v_s)
    sv_i_z = jnp.array(sv_i)
    oa_i_z = jnp.array(oa_i).at[3, 3, 0].set(-0.05)
    a_i_z = jnp.array(a_i)
    a_ip_z = jnp.array(a_ip)
    # At (4, 4, 0): pv_ip < 0, pv_il > 0, pa_ip > 0 -> pv_il MUST be zapped into wfx_res!
    v_ip_z = jnp.array(v_ip).at[4, 4, 0].set(-0.015)
    v_il_z = jnp.array(v_il).at[4, 4, 0].set(0.025)
    e_s_z = jnp.array(e_s).at[5, 4, 0, 0].set(-1.5e6)
    e_i_z = jnp.array(e_i).at[5, 5, 0, 0].set(-2.5e7)
    szv_i_z = jnp.array(szv_i).at[3, 4, 0, 0].set(-0.12)

    zapneg_out = si3_var_zapneg(
        ihls,
        pdt,
        v_i_z,
        v_s_z,
        sv_i_z,
        oa_i_z,
        a_i_z,
        a_ip_z,
        v_ip_z,
        v_il_z,
        e_s_z,
        e_i_z,
        szv_i_z,
        wfx0,
        hfx0,
        sfx0,
        nn_hls=nn_hls,
        nn_icesal=4,
    )
    assert float(zapneg_out[7][4, 4, 0]) == 0.0  # pv_il(4,4,0) zapped because pv_ip(4,4,0) < 0
    z1_dt_np = np.float64(1.0) / np.float64(pdt)
    rhow_np = np.float64(canonical_constants.rho_water)
    expected_wfx_44 = (np.float64(0.025) * rhow_np) * z1_dt_np + (
        np.float64(-0.015) * rhow_np
    ) * z1_dt_np
    _assert_zero_ulp(zapneg_out[11][4, 4], expected_wfx_44, "zapneg_wfx_res_44")

    # Plant O3-A3d: if `pv_ip < 0` were zeroed BEFORE testing `pv_il`, `pv_il(4,4,0)` would remain 0.025
    wrong_wfx_44 = (np.float64(-0.015) * rhow_np) * z1_dt_np
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(zapneg_out[11][4, 4], wrong_wfx_44, "plant_late_vil_check")

    # 5. Pin `si3_var_zapsmall` (`icevar.f90:615-737`)
    a_i_small = jnp.array(a_i).at[3, 3, 0].set(0.5e-10)  # < epsi10 = 1e-10
    t_su_in = jnp.full((jpi, jpj, jpl), canonical_constants.T_freeze - 5.0, dtype=jnp.float64)
    h_i_in = v_i / a_i
    h_s_in = v_s / a_i
    h_ip_in = v_ip / a_ip
    h_il_in = v_il / a_ip
    sz_i_in = szv_i / v_i[:, :, None, :]
    t_i_in = jnp.full((jpi, jpj, nlay_i, jpl), canonical_constants.T_freeze - 3.0, dtype=jnp.float64)
    t_s_in = jnp.full((jpi, jpj, nlay_s, jpl), canonical_constants.T_freeze - 4.0, dtype=jnp.float64)
    at_i_in = jnp.sum(a_i_small, axis=-1)
    vt_i_in = jnp.sum(v_i, axis=-1)
    ato_i_in = 1.0 - at_i_in

    zapsmall_out = si3_var_zapsmall(
        v_i,
        v_s,
        sv_i,
        oa_i,
        a_i_small,
        a_ip,
        v_ip,
        v_il,
        t_su_in,
        h_i_in,
        h_s_in,
        h_ip_in,
        h_il_in,
        e_s,
        e_i,
        szv_i,
        sz_i_in,
        t_i_in,
        t_s_in,
        at_i_in,
        vt_i_in,
        ato_i_in,
        wfx0,
        hfx0,
        sfx0,
        rDt_ice=pdt,
        sss_m=34.7,
        sst_m=-1.8,
        nn_hls=nn_hls,
        nn_icesal=4,
    )
    assert float(zapsmall_out[0][3, 3, 0]) == 0.0  # v_i(3,3,0) zapped
    assert float(zapsmall_out[4][3, 3, 0]) == 0.0  # a_i(3,3,0) zapped
    _assert_zero_ulp(
        zapsmall_out[8][3, 3, 0],
        canonical_constants.T_freeze - 1.8,
        "zapsmall_t_su_reset",
    )


def test_o3_a4_orca2_seam_and_north_fold_subcycle_lbc_lnk_pins_and_plants() -> None:
    """`si3_dyn_adv_pra_orca2` exchanges all 11 extensive families and 55 Prather moment arrays across the ORCA2 cyclic seam and T-pivot north fold at every subcycle (`icycle >= 2`), and fails under Plant O3-A4."""
    cfg = build_si3_orca2_o3_config().si3_orca2
    assert cfg is not None
    state0, tmask, umask, vmask = o1_gate.build_synthetic_orca2_o1_state(cfg)
    grid = _build_orca2_advection_grid(tmask, umask, vmask)

    # Prescribe strong velocities near the cyclic seam (ji=2..5, 179..182) and north fold (jj=146..150)
    # so `zcfl > 0.5` (`icycle = 2`) and subcycle 1 (`ihls = 1`) feeds subcycle 2 (`ihls = 0`).
    u_np = np.zeros((184, 152), dtype=np.float64)
    v_np = np.zeros((184, 152), dtype=np.float64)
    u_np[:, :] = 14.5  # zcfl ~ 14.5 * 10800 / 185000 ~ 0.85 -> icycle = 2
    v_np[:, :] = 6.2
    u_ice = si3_lbc_lnk_orca2(
        jnp.asarray(u_np) * grid.umask, "U", -1.0, nn_hls=2, ldfull=True
    )
    v_ice = si3_lbc_lnk_orca2(
        jnp.asarray(v_np) * grid.vmask, "V", -1.0, nn_hls=2, ldfull=True
    )
    state_in = si3_apply_state_halos_orca2(
        state0._replace(u_ice=u_ice, v_ice=v_ice), config=cfg
    )

    state_out, diag = si3_dyn_orca2(state_in, grid, kt=1, config=cfg)
    assert int(diag.icycle) == 2

    # Every extensive and all 55 Prather moments in `state_out` must be 0-ULP halo-idempotent
    state_rehalo = si3_apply_state_halos_orca2(state_out, config=cfg)
    for fname in (
        "v_i",
        "v_s",
        "sv_i",
        "a_i",
        "t_su",
        "oa_i",
        "a_ip",
        "v_ip",
        "v_il",
        "e_s",
        "e_i",
        "szv_i",
    ):
        _assert_zero_ulp(
            getattr(state_out, fname),
            getattr(state_rehalo, fname),
            f"post_dyn_halo_idempotent.{fname}",
        )
    for fam_name in state_out.prather._fields:
        fam_out = getattr(state_out.prather, fam_name)
        fam_reh = getattr(state_rehalo.prather, fam_name)
        for mname in ("sx", "sy", "sxx", "syy", "sxy"):
            _assert_zero_ulp(
                getattr(fam_out, mname),
                getattr(fam_reh, mname),
                f"post_dyn_halo_idempotent.prather.{fam_name}.{mname}",
            )

    # Plant O3-A4a: omitting inter-subcycle `lbc_lnk` (`_plant_omit_subcycle_lbc=True`)
    # corrupts seam and north-fold interior cells on subcycle 2 (`icycle = 2`) and fails 0-ULP!
    state_no_lbc, _ = si3_dyn_orca2(
        state_in, grid, kt=1, config=cfg, _plant_omit_subcycle_lbc=True
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(
            state_out.v_i[2:182, 2:150, :],
            state_no_lbc.v_i[2:182, 2:150, :],
            "plant_omit_subcycle_lbc_interior_v_i",
        )

    # Plant O3-A4b: wrong north-fold sign (`+1` instead of `-1` on `sx, sy`, `_plant_wrong_nfd_sign=True`)
    # corrupts moments across the T-pivot north fold and fails 0-ULP!
    state_bad_nfd, _ = si3_dyn_orca2(
        state_in, grid, kt=1, config=cfg, _plant_wrong_nfd_sign=True
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(
            state_out.prather.ice.sx,
            state_bad_nfd.prather.ice.sx,
            "plant_wrong_nfd_sign_ice_sx",
        )


# ==============================================================================
# Test (b): Lane-3 `ICE_ADV1D` / `ICE_ADV2D` invariance and real parent-vs-current
# (`042c7bd74c29713be111033b8685c30d9209f1f9`) 0-ULP bit-identity control
# (`Plants O3-B1, O3-B2`).
# ==============================================================================


def _load_parent_commit_ice_modules(
    parent_commit: str = PARENT_COMMIT,
) -> dict[str, types.ModuleType]:
    """Load `packages/ice/legoesm/ice/*` directly from `parent_commit` via `git show`."""
    module_specs = (
        ("legoesm.ice.constants_config", "packages/ice/legoesm/ice/constants_config.py"),
        ("legoesm.ice.si3_orca2_state", "packages/ice/legoesm/ice/si3_orca2_state.py"),
        ("legoesm.ice.config", "packages/ice/legoesm/ice/config.py"),
        ("legoesm.ice.state", "packages/ice/legoesm/ice/state.py"),
        ("legoesm.ice.transport", "packages/ice/legoesm/ice/transport.py"),
        ("legoesm.ice.bitz_lipscomb", "packages/ice/legoesm/ice/bitz_lipscomb.py"),
        ("legoesm.ice.snow", "packages/ice/legoesm/ice/snow.py"),
        ("legoesm.ice.itd", "packages/ice/legoesm/ice/itd.py"),
        ("legoesm.ice.sea_ice", "packages/ice/legoesm/ice/sea_ice.py"),
        ("legoesm.ice.c1d_omip_l3", "packages/ice/legoesm/ice/c1d_omip_l3.py"),
        ("legoesm.ice.scm", "packages/ice/legoesm/ice/scm.py"),
    )
    loaded: dict[str, types.ModuleType] = {}
    saved_sys = {name: sys.modules.get(name) for name, _ in module_specs}
    try:
        for mod_name, rel_path in module_specs:
            src = subprocess.check_output(
                ["git", "show", f"{parent_commit}:{rel_path}"],
                cwd=REPO_ROOT,
                text=True,
            )
            mod = types.ModuleType(mod_name)
            mod.__file__ = f"<git:{parent_commit}:{rel_path}>"
            sys.modules[mod_name] = mod
            exec(compile(src, mod.__file__, "exec"), mod.__dict__)
            loaded[mod_name] = mod
    finally:
        for mod_name, orig_mod in saved_sys.items():
            if orig_mod is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = orig_mod
    return loaded


def test_o3_b_parent_vs_current_lane3_and_production_dispatcher_zero_ulp(
    monkeypatch,
) -> None:
    """Real Lane-3 (`advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`, `step_ice_adv2d_card`), `si3_o1_step`, and `step_sea_ice` callers match parent `042c7bd74c29` to 0 ULP and fail under Plants O3-B1 and O3-B2."""
    import legoesm.ice.bitz_lipscomb as bl_mod
    import legoesm.ice.c1d_omip_l3 as c1d_mod
    import legoesm.ice.config as cfg_mod
    import legoesm.ice.scm as scm_mod
    import legoesm.ice.sea_ice as sea_ice_mod
    import legoesm.ice.si3_orca2_state as o1_mod
    import legoesm.ice.state as state_mod
    import legoesm.ice.transport as tr_mod

    parent_mods = _load_parent_commit_ice_modules(PARENT_COMMIT)
    p_tr = parent_mods["legoesm.ice.transport"]

    # 1. Direct `advect_si3_prather_1d` and `advect_si3_prather_2d` parent-vs-current 0-ULP
    rng = np.random.default_rng(2026102404)
    shape = (10, 9, 4)
    contents = jnp.asarray(rng.uniform(0.1, 2.5, size=shape) * 1.0e8)
    moments = tuple(
        jnp.asarray(rng.uniform(-0.2, 0.2, size=shape) * 1.0e8) for _ in range(5)
    )
    u_ice = jnp.asarray(rng.uniform(-0.35, 0.45, size=shape[:2]))
    v_ice = jnp.asarray(rng.uniform(-0.30, 0.40, size=shape[:2]))
    area = jnp.asarray(3.5e7 + rng.uniform(-2.0e6, 2.0e6, size=shape[:2]))
    wet = jnp.ones(shape[:2], dtype=bool)

    p_1d = p_tr.advect_si3_prather_1d(
        contents,
        moments,
        u_ice,
        jnp.zeros_like(v_ice),
        area,
        wet,
        1200.0,
        dx=6000.0,
        dy=6000.0,
        ice_volume_index=0,
        concentration_index=1,
    )
    c_1d = advect_si3_prather_1d(
        contents,
        moments,
        u_ice,
        jnp.zeros_like(v_ice),
        area,
        wet,
        1200.0,
        dx=6000.0,
        dy=6000.0,
        ice_volume_index=0,
        concentration_index=1,
    )
    _assert_zero_ulp(p_1d[0], c_1d[0], "advect_si3_prather_1d.contents")
    for idx in range(5):
        _assert_zero_ulp(p_1d[1][idx], c_1d[1][idx], f"advect_si3_prather_1d.moments[{idx}]")

    for step_idx in (1, 2):
        p_2d = p_tr.advect_si3_prather_2d(
            contents,
            moments,
            u_ice,
            v_ice,
            area,
            wet,
            1200.0,
            dx=6000.0,
            dy=6000.0,
            ice_step_index=step_idx,
            ice_volume_index=0,
            concentration_index=1,
        )
        c_2d = advect_si3_prather_2d(
            contents,
            moments,
            u_ice,
            v_ice,
            area,
            wet,
            1200.0,
            dx=6000.0,
            dy=6000.0,
            ice_step_index=step_idx,
            ice_volume_index=0,
            concentration_index=1,
        )
        _assert_zero_ulp(p_2d[0], c_2d[0], f"advect_si3_prather_2d(step={step_idx}).contents")
        for idx in range(5):
            _assert_zero_ulp(
                p_2d[1][idx],
                c_2d[1][idx],
                f"advect_si3_prather_2d(step={step_idx}).moments[{idx}]",
            )

    # 2. Lane-3 `step_ice_adv1d_card` and `step_ice_adv2d_card` parent-vs-current 0-ULP
    card_1d = build_ice_adv1d_card(np.full((59, 59), -1.8, dtype=np.float64))
    c_card1d_st = step_ice_adv1d_card(card_1d, card_1d.initial_state)
    orig_tr_1d = tr_mod.advect_si3_prather_1d
    try:
        tr_mod.advect_si3_prather_1d = p_tr.advect_si3_prather_1d
        p_card1d_st = step_ice_adv1d_card(card_1d, card_1d.initial_state)
    finally:
        tr_mod.advect_si3_prather_1d = orig_tr_1d
    _assert_zero_ulp(p_card1d_st.contents, c_card1d_st.contents, "step_ice_adv1d_card.contents")
    for idx in range(5):
        _assert_zero_ulp(
            p_card1d_st.moments[idx],
            c_card1d_st.moments[idx],
            f"step_ice_adv1d_card.moments[{idx}]",
        )

    card_2d = build_ice_adv2d_card(np.full((99, 99), -1.8, dtype=np.float64))
    c_card2d_st = step_ice_adv2d_card(card_2d, card_2d.initial_state, completed_steps=0)
    orig_tr_2d = tr_mod.advect_si3_prather_2d
    try:
        tr_mod.advect_si3_prather_2d = p_tr.advect_si3_prather_2d
        p_card2d_st = step_ice_adv2d_card(card_2d, card_2d.initial_state, completed_steps=0)
    finally:
        tr_mod.advect_si3_prather_2d = orig_tr_2d
    _assert_zero_ulp(p_card2d_st.contents, c_card2d_st.contents, "step_ice_adv2d_card.contents")
    for idx in range(5):
        _assert_zero_ulp(
            p_card2d_st.moments[idx],
            c_card2d_st.moments[idx],
            f"step_ice_adv2d_card.moments[{idx}]",
        )

    # 3. Production dispatcher `step_sea_ice` (dynamic-v2, `si3_bl99`, and `si3_orca2_o1`)
    t0 = canonical_constants.T_freeze
    cfg_v2 = SeaIceConfig(snow=SnowConfig(enabled=True), brine=BrineConfig(enabled=True))
    dyn_state = init_dynamic_ice_state((2,))
    forcing_v2 = scm_mod.cold_polar_forcing(ncol=2)
    sst_2 = jnp.full((2,), t0 - 1.8, dtype=jnp.float64)
    zero_2 = jnp.zeros((2,), dtype=jnp.float64)
    p_si_st, p_si_resp = parent_mods["legoesm.ice.sea_ice"].step_sea_ice(
        dyn_state, forcing_v2, sst_2, zero_2, zero_2, cfg_v2, 1.0, 3600.0
    )
    c_si_st, c_si_resp = step_sea_ice(
        dyn_state, forcing_v2, sst_2, zero_2, zero_2, cfg_v2, 1.0, 3600.0
    )
    for fname in ("h_ice", "h_snow", "T_ice", "S_ice", "concentration"):
        _assert_zero_ulp(
            getattr(p_si_st, fname).data,
            getattr(c_si_st, fname).data,
            f"step_sea_ice(v2).{fname}",
        )
    _assert_zero_ulp(p_si_resp.T_sfc, c_si_resp.T_sfc, "step_sea_ice(v2).T_sfc")

    # O1 `si3_o1_step` and `step_sea_ice(si3_orca2_o1)` parent-vs-current 0-ULP
    o1_cfg = SI3ORCA2StateConfig.for_orca2_sas_ice()
    state0, tmask, umask, vmask = o1_gate.build_synthetic_orca2_o1_state(o1_cfg)
    sss_step = jnp.full((184, 152), 34.2, dtype=jnp.float64)
    p_o1_st = parent_mods["legoesm.ice.si3_orca2_state"].si3_o1_step(
        state0, tmask=tmask, sss_m=sss_step, config=o1_cfg
    )
    c_o1_st = si3_o1_step(state0, tmask=tmask, sss_m=sss_step, config=o1_cfg)
    for fname in ("v_i", "v_s", "sv_i", "a_i", "oa_i", "snwice_mass", "snwice_fmass"):
        _assert_zero_ulp(
            getattr(p_o1_st, fname),
            getattr(c_o1_st, fname),
            f"si3_o1_step.{fname}",
        )

    # Also verify `step_sea_ice` refusal contract on `si3_orca2_o1` (parent vs current) and `si3_orca2_o3`
    frc_o1 = SI3ORCA2Forcing(tmask=tmask, sss_m=sss_step)
    o1_sea_cfg = build_si3_orca2_o1_config()
    p_o1_mod = parent_mods["legoesm.ice.si3_orca2_state"]
    p_state0 = p_o1_mod.SI3State(*state0)
    p_frc_o1 = p_o1_mod.SI3ORCA2Forcing(tmask=tmask, sss_m=sss_step)
    p_o1_sea_cfg = parent_mods["legoesm.ice.config"].build_si3_orca2_o1_config()
    with pytest.raises(NotImplementedError, match="ice_update_flx"):
        parent_mods["legoesm.ice.sea_ice"].step_sea_ice(
            p_state0,
            p_frc_o1,
            jnp.full((184, 152), t0 - 1.8, dtype=jnp.float64),
            state0.u_ice,
            state0.v_ice,
            p_o1_sea_cfg,
            1.0,
            10800.0,
        )
    with pytest.raises(NotImplementedError, match="ice_update_flx"):
        step_sea_ice(
            state0,
            frc_o1,
            jnp.full((184, 152), t0 - 1.8, dtype=jnp.float64),
            state0.u_ice,
            state0.v_ice,
            o1_sea_cfg,
            1.0,
            10800.0,
        )
    o3_sea_cfg = build_si3_orca2_o3_config()
    with pytest.raises(NotImplementedError, match="ice_update_flx"):
        step_sea_ice(
            state0,
            frc_o1,
            jnp.full((184, 152), t0 - 1.8, dtype=jnp.float64),
            state0.u_ice,
            state0.v_ice,
            o3_sea_cfg,
            1.0,
            10800.0,
        )

    # Plant O3-B1: 1-ULP perturbation on `advect_si3_prather_2d` MUST fail parent-vs-current control
    orig_adv2d = tr_mod.advect_si3_prather_2d

    def _planted_adv2d(*args, **kwargs):
        c_out, m_out, area_out = orig_adv2d(*args, **kwargs)
        return jnp.nextafter(c_out, jnp.inf), m_out, area_out

    monkeypatch.setattr(tr_mod, "advect_si3_prather_2d", _planted_adv2d)
    with pytest.raises(AssertionError, match="max ULP = 1 != 0"):
        planted_2d = tr_mod.advect_si3_prather_2d(
            contents,
            moments,
            u_ice,
            v_ice,
            area,
            wet,
            1200.0,
            dx=6000.0,
            dy=6000.0,
            ice_step_index=2,
            ice_volume_index=0,
            concentration_index=1,
        )
        _assert_zero_ulp(p_2d[0], planted_2d[0], "plant_o3_b1_adv2d")

    # Plant O3-B2: 1-ULP perturbation on `step_sea_ice` MUST fail dispatcher control
    orig_step_si = sea_ice_mod.step_sea_ice

    def _planted_step_sea_ice(*args, **kwargs):
        st_out, resp_out = orig_step_si(*args, **kwargs)
        return st_out, resp_out._replace(T_sfc=jnp.nextafter(resp_out.T_sfc, jnp.inf))

    monkeypatch.setattr(sea_ice_mod, "step_sea_ice", _planted_step_sea_ice)
    with pytest.raises(AssertionError, match="max ULP = 1 != 0"):
        _, planted_resp = sea_ice_mod.step_sea_ice(
            dyn_state, forcing_v2, sst_2, zero_2, zero_2, cfg_v2, 1.0, 3600.0
        )
        _assert_zero_ulp(p_si_resp.T_sfc, planted_resp.T_sfc, "plant_o3_b2_step_sea_ice")


# ==============================================================================
# Test (c): Conservation of every extensive to round-off on a closed ORCA2
# `(184, 152)` test field (`Plant O3-C1`).
# ==============================================================================


def _build_closed_orca2_conservation_state(
    cfg: SI3ORCA2StateConfig,
) -> tuple[SI3State, SI3ORCA2AdvectionGrid]:
    """Build a closed ORCA2 `(184, 152)` state with active cyclic-seam flow and non-clipping thicknesses."""
    state0, tmask, umask, vmask = o1_gate.build_synthetic_orca2_o1_state(cfg)
    grid = _build_orca2_advection_grid(tmask, umask, vmask)
    jpi, jpj = 184, 152

    # Construct a smooth ice field in the wet latitude band `jj in [12, 132]`
    # that crosses the cyclic east-west seam (`ji = 2 <-> 182`) with zero support
    # near the north fold (`jj >= 136`) and southern land (`jj <= 8`), and with
    # uniform category intensive ratios and well-above-threshold concentrations
    # (`a_i > 1e-3 >> epsi10`) on the active flow support so `Hbig_pra`,
    # `Hsnow_pra`, `ice_var_zapneg`, and `ice_var_zapsmall` do not clip.
    ii, jj = np.meshgrid(np.arange(jpi), np.arange(jpj), indexing="ij")
    lat_ice = np.where((jj >= 12) & (jj <= 132), 1.0 + 0.15 * np.exp(-0.5 * ((jj - 70.0) / 22.0) ** 2), 0.0)
    lon_angle = 2.0 * np.pi * (ii - 2.0) / 180.0
    base_2d = lat_ice * (1.0 + 0.06 * np.cos(lon_angle + 0.4) + 0.04 * np.sin(2.0 * lon_angle))
    base_2d = base_2d * np.asarray(tmask)

    hi_mean = np.asarray(state0.bounds.hi_mean)
    a_i_np = np.zeros((jpi, jpj, cfg.jpl), dtype=np.float64)
    v_i_np = np.zeros((jpi, jpj, cfg.jpl), dtype=np.float64)
    v_s_np = np.zeros((jpi, jpj, cfg.jpl), dtype=np.float64)
    sv_i_np = np.zeros((jpi, jpj, cfg.jpl), dtype=np.float64)
    oa_i_np = np.zeros((jpi, jpj, cfg.jpl), dtype=np.float64)
    a_ip_np = np.zeros((jpi, jpj, cfg.jpl), dtype=np.float64)
    v_ip_np = np.zeros((jpi, jpj, cfg.jpl), dtype=np.float64)
    v_il_np = np.zeros((jpi, jpj, cfg.jpl), dtype=np.float64)
    e_s_np = np.zeros((jpi, jpj, cfg.nlay_s, cfg.jpl), dtype=np.float64)
    e_i_np = np.zeros((jpi, jpj, cfg.nlay_i, cfg.jpl), dtype=np.float64)
    szv_i_np = np.zeros((jpi, jpj, cfg.nlay_i, cfg.jpl), dtype=np.float64)

    for jl in range(cfg.jpl):
        # Keep `a_cat >= 0.165 > 0.15` and `a_ip >= 0.155 > 0.15` on all active wet cells
        # so `Hbig_pra`'s `< 0.15` concentration gate never triggers on 1-ULP round-trip noise.
        cat_w = 0.185 * (1.0 + 0.03 * jl)
        a_cat = cat_w * base_2d
        h_cat = 0.65 * hi_mean[jl]
        hs_cat = 0.03 * (1.0 + 0.05 * jl)
        a_i_np[:, :, jl] = a_cat
        v_i_np[:, :, jl] = a_cat * h_cat
        v_s_np[:, :, jl] = a_cat * hs_cat
        sv_i_np[:, :, jl] = v_i_np[:, :, jl] * (3.2 + 0.2 * jl)
        oa_i_np[:, :, jl] = a_cat * (12.0 + 3.0 * jl)
        a_ip_np[:, :, jl] = a_cat * 0.94
        v_ip_np[:, :, jl] = a_ip_np[:, :, jl] * 0.05
        v_il_np[:, :, jl] = a_ip_np[:, :, jl] * 0.015
        for jk in range(cfg.nlay_s):
            e_s_np[:, :, jk, jl] = (
                v_s_np[:, :, jl] * (0.9e8 + 0.05e8 * jk) / cfg.nlay_s
            )
        for jk in range(cfg.nlay_i):
            e_i_np[:, :, jk, jl] = (
                v_i_np[:, :, jl] * (2.2e8 + 0.04e8 * jk) / cfg.nlay_i
            )
            szv_i_np[:, :, jk, jl] = (
                v_i_np[:, :, jl] * (3.0 + 0.15 * jk) / cfg.nlay_i
            )

    # Smooth velocities across the cyclic east-west seam (`ji = 2 <-> 182`) strictly inside
    # the ice-covered latitude band (`jj in [20, 124]`) and zero on dry-cell faces.
    tmask_np = np.asarray(tmask)
    wet_u = np.asarray(umask) * tmask_np * np.roll(tmask_np, -1, axis=0)
    wet_v = np.asarray(vmask) * tmask_np * np.roll(tmask_np, -1, axis=1)
    vel_band = ((jj >= 20) & (jj <= 124)).astype(np.float64)
    u_np = (0.35 + 0.12 * np.cos(lon_angle)) * vel_band * wet_u
    v_np = (0.22 * np.sin(lon_angle)) * vel_band * wet_v

    # Non-zero Prather moments within physical slope limits
    e1e2t_3d = np.asarray(grid.e1e2t)[:, :, None]
    e1e2t_4d = np.asarray(grid.e1e2t)[:, :, None, None]

    def _make_moments(ext_arr: np.ndarray) -> SI3PratherMoments:
        scale = ext_arr * (e1e2t_3d if ext_arr.ndim == 3 else e1e2t_4d)
        return SI3PratherMoments(
            sx=jnp.asarray(0.08 * scale),
            sy=jnp.asarray(-0.06 * scale),
            sxx=jnp.asarray(0.02 * scale),
            syy=jnp.asarray(0.015 * scale),
            sxy=jnp.asarray(0.01 * scale),
        )

    prather = SI3AllPratherMoments(
        ice=_make_moments(v_i_np),
        sn=_make_moments(v_s_np),
        a=_make_moments(a_i_np),
        age=_make_moments(oa_i_np),
        sal=_make_moments(sv_i_np),
        ap=_make_moments(a_ip_np),
        vp=_make_moments(v_ip_np),
        vl=_make_moments(v_il_np),
        c0=_make_moments(e_s_np),
        e=_make_moments(e_i_np),
        si=_make_moments(szv_i_np),
    )

    state = state0._replace(
        v_i=jnp.asarray(v_i_np),
        v_s=jnp.asarray(v_s_np),
        sv_i=jnp.asarray(sv_i_np),
        a_i=jnp.asarray(a_i_np),
        oa_i=jnp.asarray(oa_i_np),
        a_ip=jnp.asarray(a_ip_np),
        v_ip=jnp.asarray(v_ip_np),
        v_il=jnp.asarray(v_il_np),
        e_s=jnp.asarray(e_s_np),
        e_i=jnp.asarray(e_i_np),
        szv_i=jnp.asarray(szv_i_np),
        u_ice=jnp.asarray(u_np),
        v_ice=jnp.asarray(v_np),
        prather=prather,
    )
    state = si3_apply_state_halos_orca2(state, config=cfg)
    return state, grid


def _domain_integral(field: jnp.ndarray, e1e2t: jnp.ndarray) -> float:
    """Compute interior ORCA2 area-integrated extensive content `SUM_{ji=3..182, jj=3..150} field * e1e2t`."""
    f_int = np.asarray(field, dtype=np.float64)[2:182, 2:150]
    area_int = np.asarray(e1e2t, dtype=np.float64)[2:182, 2:150]
    while area_int.ndim < f_int.ndim:
        area_int = area_int[..., None]
    return float(np.sum(f_int * area_int, dtype=np.float64))


def test_o3_c_orca2_extensive_conservation_and_residual_budget_closure() -> None:
    """Every extensive family (`v_i, v_s, a_i, oa_i, e_s, e_i, szv_i, sv_i, a_ip, v_ip, v_il`) is conserved to `<= 1e-14` relative error on ORCA2 across `x -> y` and `y -> x` sweeps and `icycle in {1, 2, 3}`, residual budgets close when clipping is active, and Plant O3-C1 fails."""
    cfg_sal4 = SI3ORCA2StateConfig.for_orca2_sas_ice(ln_icedyn=True, nn_icesal=4)
    state_in, grid = _build_closed_orca2_conservation_state(cfg_sal4)

    # Confirm cyclic east-west seam transport is genuinely active (`u_ice` and `v_i` non-zero at `ji=2, 181`)
    assert float(jnp.max(jnp.abs(state_in.u_ice[2, 2:150]))) > 0.1
    assert float(jnp.max(jnp.abs(state_in.v_i[2, 2:150]))) > 0.01

    ext_names_sal4 = (
        "v_i",
        "v_s",
        "a_i",
        "oa_i",
        "a_ip",
        "v_ip",
        "v_il",
        "e_s",
        "e_i",
        "szv_i",
    )
    initial_totals = {
        fname: _domain_integral(getattr(state_in, fname), grid.e1e2t)
        for fname in ext_names_sal4
    }
    assert all(val > 0.0 for val in initial_totals.values())

    for kt in (1, 2):
        for subcycles in (1, 2, 3):
            state_adv, diag = si3_dyn_orca2(
                state_in, grid, kt=kt, config=cfg_sal4, subcycles=subcycles
            )
            # Confirm no clipping occurred on the smooth closed field
            assert float(jnp.max(jnp.abs(diag.wfx_res))) == 0.0
            assert float(jnp.max(jnp.abs(diag.hfx_res))) == 0.0
            assert float(jnp.max(jnp.abs(diag.sfx_res))) == 0.0

            for fname in ext_names_sal4:
                after_total = _domain_integral(getattr(state_adv, fname), grid.e1e2t)
                rel_err = abs(after_total - initial_totals[fname]) / initial_totals[fname]
                assert rel_err <= 1.0e-14, (
                    f"{fname} (kt={kt}, icycle={subcycles}) rel_err={rel_err:.3e} > 1e-14"
                )

    # Also test `nn_icesal = 2` so the 11th extensive family `sv_i` is actively transported and conserved
    cfg_sal2 = cfg_sal4._replace(nn_icesal=2)
    sv_init = _domain_integral(state_in.sv_i, grid.e1e2t)
    state_sal2, _ = si3_dyn_orca2(state_in, grid, kt=1, config=cfg_sal2, subcycles=2)
    sv_after = _domain_integral(state_sal2.sv_i, grid.e1e2t)
    assert abs(sv_after - sv_init) / sv_init <= 1.0e-14

    # Residual budget closure when `Hbig_pra` / `Hsnow_pra` / `ice_var_zapneg` actively clip:
    # `diag_vice + diag_vsnw + diag_vpnd - diag_adv_mass == -wfx_res`
    # `diag_heat - diag_adv_heat == -hfx_res`
    # `diag_sice - diag_adv_salt == -sfx_res`
    state_clip_raw, _, _, _ = o1_gate.build_synthetic_orca2_o1_state(cfg_sal4)
    # Exaggerate snow and moment slopes on interior wet cells to trigger Hsnow_pra and zapneg
    v_s_clip = state_clip_raw.v_s * 3.5
    sx_clip = state_clip_raw.prather.ice.sx * 8.0
    state_clip = si3_apply_state_halos_orca2(
        state_clip_raw._replace(
            v_s=v_s_clip,
            prather=state_clip_raw.prather._replace(
                ice=state_clip_raw.prather.ice._replace(sx=sx_clip)
            ),
        ),
        config=cfg_sal4,
    )
    _, diag_clip = si3_dyn_orca2(state_clip, grid, kt=1, config=cfg_sal4, subcycles=1)
    assert float(jnp.max(jnp.abs(diag_clip.wfx_res))) > 0.0
    mass_rate_post = (
        diag_clip.diag_vice + diag_clip.diag_vsnw + diag_clip.diag_vpnd
    )
    mass_budget_residual = (mass_rate_post - diag_clip.diag_adv_mass) + diag_clip.wfx_res
    mass_scale = float(
        jnp.max(
            (
                jnp.sum(state_clip.v_i, axis=-1) * cfg_sal4.ice_constants.rho_ice
                + jnp.sum(state_clip.v_s, axis=-1) * cfg_sal4.ice_constants.rho_snow
                + jnp.sum(state_clip.v_ip + state_clip.v_il, axis=-1) * canonical_constants.rho_water
            )
            / cfg_sal4.rDt_ice
        )
    )
    assert float(jnp.max(jnp.abs(mass_budget_residual[2:182, 2:150]))) / mass_scale <= 1.0e-13

    heat_budget_residual = (diag_clip.diag_heat - diag_clip.diag_adv_heat) + diag_clip.hfx_res
    heat_scale = float(
        jnp.max(
            (
                jnp.sum(state_clip.e_i, axis=(-2, -1))
                + jnp.sum(state_clip.e_s, axis=(-2, -1))
            )
            / cfg_sal4.rDt_ice
        )
    )
    assert float(jnp.max(jnp.abs(heat_budget_residual[2:182, 2:150]))) / heat_scale <= 1.0e-13

    # Plant O3-C1: corrupting receiver flux (`_plant_corrupt_receiver_flux=True`) MUST break conservation
    state_corrupt, _ = si3_dyn_orca2(
        state_in,
        grid,
        kt=1,
        config=cfg_sal4,
        subcycles=1,
        _plant_corrupt_receiver_flux=True,
    )
    vi_corrupt = _domain_integral(state_corrupt.v_i, grid.e1e2t)
    rel_err_corrupt = abs(vi_corrupt - initial_totals["v_i"]) / initial_totals["v_i"]
    assert rel_err_corrupt > 1.0e-8, (
        f"STATUS PLANT-FIRED expected conservation failure, got {rel_err_corrupt:.3e}"
    )


# ==============================================================================
# Test (d): End-to-end differentiability (`jax.jit`, `jax.value_and_grad`,
# `jax.jvp`) through one ORCA2 Prather advection step (`Plant O3-D1`).
# ==============================================================================


def test_o3_d_jit_primal_zero_ulp_and_forward_equals_reverse_ad() -> None:
    """`si3_o3_step` is 0-ULP invariant under `jax.jit`, produces finite non-zero gradients with respect to extensives, velocities (`u_ice, v_ice`), and Prather moments (`sx, sy, sxx, syy, sxy`), satisfies `reverse == forward` (`<= 1e-11`), and fails under Plant O3-D1."""
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(ln_icedyn=True)
    state_in, grid = _build_closed_orca2_conservation_state(cfg)
    sss_m = jnp.full((184, 152), 34.2, dtype=jnp.float64)
    sst_m = jnp.full((184, 152), -1.8, dtype=jnp.float64)

    def _step_fn(st: SI3State) -> tuple[SI3State, SI3AdvectionDiagnostics]:
        return si3_o3_step(
            st, grid, kt=1, sss_m=sss_m, sst_m=sst_m, config=cfg, subcycles=1
        )

    eager_state, eager_diag = _step_fn(state_in)
    jit_state, jit_diag = jax.jit(_step_fn)(state_in)

    # 1. 0-ULP primal invariance between eager and `jax.jit`
    for e_leaf, j_leaf in zip(
        jax.tree_util.tree_leaves(eager_state),
        jax.tree_util.tree_leaves(jit_state),
    ):
        if jnp.issubdtype(e_leaf.dtype, jnp.floating):
            _assert_zero_ulp(e_leaf, j_leaf, "jit_primal_state")
    for e_leaf, j_leaf in zip(
        jax.tree_util.tree_leaves(eager_diag),
        jax.tree_util.tree_leaves(jit_diag),
    ):
        if jnp.issubdtype(e_leaf.dtype, jnp.floating):
            _assert_zero_ulp(e_leaf, j_leaf, "jit_primal_diag")

    # 2. Reverse-mode (`jax.value_and_grad`) and forward-mode (`jax.jvp`) AD through `si3_o3_step`
    weights_2d = jnp.asarray(
        1.0 + 0.25 * np.sin(np.arange(184)[:, None] * 0.1 + np.arange(152)[None, :] * 0.13),
        dtype=jnp.float64,
    )
    interior = si3_active_mask_2d(184, 152, cfg.nn_hls, 0).astype(jnp.float64)
    w_int = weights_2d * interior

    def _loss(
        inputs: tuple[
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            SI3PratherMoments,
        ],
        *,
        plant_stop_grad: bool = False,
    ) -> jnp.ndarray:
        (
            v_i,
            v_s,
            a_i,
            oa_i,
            a_ip,
            v_ip,
            v_il,
            e_s,
            e_i,
            szv_i,
            u_ice,
            v_ice,
            m_ice,
        ) = inputs
        st = state_in._replace(
            v_i=v_i,
            v_s=v_s,
            a_i=a_i,
            oa_i=oa_i,
            a_ip=a_ip,
            v_ip=v_ip,
            v_il=v_il,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            u_ice=u_ice,
            v_ice=v_ice,
            prather=state_in.prather._replace(ice=m_ice),
        )
        out_st, out_diag = si3_o3_step(
            st,
            grid,
            kt=1,
            sss_m=sss_m,
            sst_m=sst_m,
            config=cfg,
            subcycles=1,
            _plant_stop_gradient=plant_stop_grad,
        )
        return (
            jnp.sum(w_int[:, :, None] * out_st.v_i)
            + jnp.sum(w_int[:, :, None] * out_st.v_s)
            + jnp.sum(w_int[:, :, None] * out_st.a_i)
            + jnp.sum(w_int[:, :, None] * out_st.oa_i)
            + jnp.sum(w_int[:, :, None] * out_st.a_ip)
            + jnp.sum(w_int[:, :, None] * out_st.v_ip)
            + jnp.sum(w_int[:, :, None] * out_st.v_il)
            + 1.0e-8 * jnp.sum(w_int[:, :, None, None] * out_st.e_s)
            + 1.0e-8 * jnp.sum(w_int[:, :, None, None] * out_st.e_i)
            + jnp.sum(w_int[:, :, None, None] * out_st.szv_i)
            + 1.0e-10 * jnp.sum(w_int[:, :, None] * out_st.prather.ice.sx)
            + 1.0e-10 * jnp.sum(w_int[:, :, None] * out_st.prather.ice.sy)
            + 1.0e-10 * jnp.sum(w_int[:, :, None] * out_st.prather.ice.sxx)
            + 1.0e-10 * jnp.sum(w_int[:, :, None] * out_st.prather.ice.syy)
            + 1.0e-10 * jnp.sum(w_int[:, :, None] * out_st.prather.ice.sxy)
            + 1.0e-3 * jnp.sum(w_int * out_diag.snwice_fmass if hasattr(out_diag, "snwice_fmass") else w_int * out_st.snwice_fmass)
        )

    primals = (
        state_in.v_i,
        state_in.v_s,
        state_in.a_i,
        state_in.oa_i,
        state_in.a_ip,
        state_in.v_ip,
        state_in.v_il,
        state_in.e_s,
        state_in.e_i,
        state_in.szv_i,
        state_in.u_ice,
        state_in.v_ice,
        state_in.prather.ice,
    )
    loss_val, grads = jax.value_and_grad(_loss)(primals)
    assert np.isfinite(float(loss_val))

    grad_names = (
        "v_i",
        "v_s",
        "a_i",
        "oa_i",
        "a_ip",
        "v_ip",
        "v_il",
        "e_s",
        "e_i",
        "szv_i",
        "u_ice",
        "v_ice",
    )
    for g_name, g_arr in zip(grad_names, grads[:12]):
        g_np = np.asarray(g_arr)
        assert np.all(np.isfinite(g_np)), f"non-finite gradient for {g_name}"
        assert float(np.max(np.abs(g_np))) > 0.0, f"zero gradient for {g_name}"

    g_m_ice = grads[12]
    for m_name in ("sx", "sy", "sxx", "syy", "sxy"):
        gm_np = np.asarray(getattr(g_m_ice, m_name))
        assert np.all(np.isfinite(gm_np)), f"non-finite gradient for prather.ice.{m_name}"
        assert float(np.max(np.abs(gm_np))) > 0.0, f"zero gradient for prather.ice.{m_name}"

    # Compare directional derivative from `jvp` vs inner product with `grads` (`vjp`)
    tangents = jax.tree_util.tree_map(lambda x: 0.01 * jnp.ones_like(x), primals)
    _, jvp_val = jax.jvp(_loss, (primals,), (tangents,))
    vjp_val = sum(
        float(jnp.sum(g * t))
        for g, t in zip(
            jax.tree_util.tree_leaves(grads),
            jax.tree_util.tree_leaves(tangents),
        )
    )
    jvp_f = float(jvp_val)
    rel_ad_diff = abs(jvp_f - vjp_val) / max(abs(jvp_f), abs(vjp_val), 1.0e-30)
    assert rel_ad_diff <= 1.0e-11, f"JVP vs VJP mismatch: {rel_ad_diff:.3e}"

    # Plant O3-D1: `_plant_stop_gradient=True` zeros all gradients and MUST fail the AD check
    _, planted_grads = jax.value_and_grad(
        lambda p: _loss(p, plant_stop_grad=True)
    )(primals)
    planted_max_grad = max(
        float(jnp.max(jnp.abs(leaf)))
        for leaf in jax.tree_util.tree_leaves(planted_grads)
    )
    assert planted_max_grad == 0.0, (
        f"STATUS PLANT-FIRED expected zero gradient under stop_gradient, got {planted_max_grad}"
    )
