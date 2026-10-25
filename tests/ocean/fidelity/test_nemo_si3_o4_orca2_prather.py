"""Fidelity, conservation, and invariance tests for SI3 ladder rung O4 (`si3_r8`, `DECISION 159`).

Covers all preregistered requirements (`docs/ice/fidelity/si3_r8_o4_preregister.md`,
`docs/ice/fidelity/si3_ladder_design.md:224`):
(a) Bit-for-bit (`0`-ULP) statement pins of NEMO 5.0.2 `icedyn_adv_pra.f90`,
    `icedyn.f90`, `icedyn_adv.f90`, `icevar.f90`, and `lib_mpp.f90` (including
    delayed CFL `mpp_max(..., cdelay='cflice')` when `ln_mppdelay = .TRUE.`)
    with inexact `float64` operands against independent scalar/loop reference
    implementations, independent NEMO-literal `lbc_lnk` seam and T-pivot north-fold
    moment checks across all 11 families, missing-input refusal gates, and
    failing statement-reversal/reordering/sign plants (`Plants O4-A1..A4`).
(b) Lane-3 `ICE_ADV1D` / `ICE_ADV2D` invariance and real parent-vs-current
    (`042c7bd74c2944dab8c39318687ab0cef54f9d2b`) `0`-ULP bit-identity across
    `advect_si3_prather_1d`, `advect_si3_prather_2d`, `step_ice_adv1d_card`,
    `step_ice_adv2d_card`, `step_sea_ice` (`slab`, `dynamic-v2`, `si3_bl99`),
    and `si3_o1_step`, plus failing `1`-ULP code-path plants (`Plants O4-B1, O4-B2`).
(c) Conservation of every extensive family (`v_i, v_s, a_i, oa_i, e_s, e_i,
    szv_i, sv_i, a_ip, v_ip, v_il`) to `float64` round-off (`<= 1e-14`) on a
    closed ORCA2 `(184, 152)` test field across `x -> y` and `y -> x` sweeps
    and `icycle in {1, 2, 3}` with active cyclic east-west seam transport and
    ice at and next to the north fold (`jj = 148, 149`), plus residual budget
    closure (`wfx_res, hfx_res, sfx_res`, `diag_adv_*`) when `Hbig_pra` /
    `Hsnow_pra` / `ice_var_zapneg` clip, and failing flux plant (`Plant O4-C1`).
(d) End-to-end differentiability (`jax.jit`, `jax.value_and_grad`, `jax.jvp`)
    through one ORCA2 Prather advection step (`si3_o4_step`), with `icycle`
    selected inside `jax.jit` from the carried delayed `cflice` state, with `0`-ULP primal invariance under `jax.jit`, finite non-zero
    gradients with respect to extensives, velocities (`u_ice, v_ice`), and
    Prather moments (`sx, sy, sxx, syy, sxy`), `reverse == forward` (`<= 1e-11`),
    central finite-difference ratio `~ 1` at `float64`, and failing
    `stop_gradient` plant (`Plant O4-D1`).
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import (
    SI3AdvectionDiagnostics,
    SI3AllPratherMoments,
    SI3ORCA2AdvectionGrid,
    SI3ORCA2StateConfig,
    SI3PratherMoments,
    SI3State,
    build_si3_orca2_o4_config,
    si3_adv_x,
    si3_adv_y,
    si3_apply_state_halos_orca2,
    si3_dyn_adv_pra_orca2,
    si3_dyn_orca2,
    si3_hbig_pra,
    si3_hsnow_pra,
    si3_icemax2d_pra,
    si3_icemax3d_pra,
    si3_lbc_lnk_orca2,
    si3_o4_step,
    si3_rst_from_netcdf_dict,
    si3_rst_to_netcdf_dict,
    si3_var_glo2eqv,
    si3_var_zapneg,
    si3_var_zapsmall,
)
from legoesm.ice.transport import (
    si3_active_mask_2d,
)

from legoesm import constants as canonical_constants

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


# Delayed `zcfl` values selecting `icycle = 1, 2, 3` (`icedyn_adv_pra.f90:138-141`).
_CFL_FOR_ICYCLE = {1: 0.25, 2: 1.0, 3: 2.0}


def _dyn_inputs(jpi: int = 184, jpj: int = 152) -> dict:
    """Replayed state velocities plus `diag_set0` zero accumulators (`icestp.f90:478-523`)."""
    zero = jnp.zeros((jpi, jpj), dtype=jnp.float64)
    return dict(
        rn_uice=None,
        rn_vice=None,
        wfx_res=zero,
        hfx_res=zero,
        sfx_res=zero,
        diag_adv_mass=zero,
        diag_adv_salt=zero,
        diag_adv_heat=zero,
    )


# ==============================================================================
# Independent scalar/loop Fortran-literal reference kernels for Test (a)
# (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:450-1156,1519-1588`,
# `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icevar.f90:615-853`,
# `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbcnfd.f90:96-150`).
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


def _ref_nemo_lbc_lnk_orca2_t_point(
    field: np.ndarray,
    psgn: float,
    *,
    nn_hls: int = 2,
) -> np.ndarray:
    """Independent scalar-loop NEMO 5.0.2 replica of `lbc_lnk` on ORCA2 T-point arrays.

    Transcribes east-west cyclic periodicity (`jperio == 4`), `lbc_nfd` case `'T'`
    with `ldfull = .TRUE.` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/lbcnfd.f90:96-150`,
    `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:450-494`), and south
    closed boundary (`lbc_lnk_pt2pt_generic.h90:245-260`).
    """
    arr = np.array(field, dtype=np.float64, copy=True)
    orig_shape = arr.shape
    ipi, ipj = orig_shape[:2]
    ihls = int(nn_hls)
    ni0glo = ipi - 2 * ihls
    n_planes = int(np.prod(orig_shape[2:])) if arr.ndim > 2 else 1
    arr_3d = arr.reshape((ipi, ipj, n_planes))
    zsgn = np.float64(psgn)

    for jn in range(n_planes):
        plane = arr_3d[:, :, jn]
        # 1. East-west cyclic seam (`1..ihls <- ipi-2*ihls+1..ipi-ihls`,
        #    `ipi-ihls+1..ipi <- ihls+1..2*ihls` in 1-based Fortran indexing)
        for ji_0 in range(ihls):
            plane[ji_0, :] = plane[ipi - 2 * ihls + ji_0, :]
            plane[ipi - ihls + ji_0, :] = plane[ihls + ji_0, :]

        # 2. T-pivot north fold (`lbcnfd.f90:96-150`, 1-based Fortran indices)
        #    Last `ihls` lines (`from ipj to ipj-ihls+1`) : full (`lbcnfd.f90:99-129`)
        for jj in range(1, ihls + 1):
            ij1 = ipj - jj + 1
            ij2 = ipj - 2 * ihls + jj - 1
            # `DO ji = 1, ihls` (`lbcnfd.f90:104-108`)
            for ji in range(1, ihls + 1):
                ii1 = ji
                ii2 = 2 * ihls + 2 - ji
                plane[ii1 - 1, ij1 - 1] = zsgn * plane[ii2 - 1, ij2 - 1]
            # `DO ji = 1, 1` (`lbcnfd.f90:109-113`)
            for ji in range(1, 2):
                ii1 = ihls + ji
                ii2 = ii1
                plane[ii1 - 1, ij1 - 1] = zsgn * plane[ii2 - 1, ij2 - 1]
            # `DO ji = 1, Ni0glo - 1` (`lbcnfd.f90:114-118`)
            for ji in range(1, ni0glo):
                ii1 = 2 + ihls + ji - 1
                ii2 = ipi - ihls - ji + 1
                plane[ii1 - 1, ij1 - 1] = zsgn * plane[ii2 - 1, ij2 - 1]
            # `DO ji = 1, COUNT( (/ihls > 0/) )` (`lbcnfd.f90:119-123`)
            if ihls > 0:
                ii1 = ipi - ihls + 1
                ii2 = ihls + 1
                plane[ii1 - 1, ij1 - 1] = zsgn * plane[ii2 - 1, ij2 - 1]
            # `DO ji = 1, ihls-1` (`lbcnfd.f90:124-128`)
            for ji in range(1, ihls):
                ii1 = ipi - ihls + 1 + ji
                ii2 = ipi - ihls + 1 - ji
                plane[ii1 - 1, ij1 - 1] = zsgn * plane[ii2 - 1, ij2 - 1]

        # Line number `ipj-ihls` : right half (`lbcnfd.f90:131-148`, `ldfull = .TRUE.`)
        ij1 = ipj - ihls
        ij2 = ij1
        # `DO ji = 1, Ni0glo/2-1` (`lbcnfd.f90:136-140`)
        for ji in range(1, ni0glo // 2):
            ii1 = ipi // 2 + ji + 1
            ii2 = ipi // 2 - ji + 1
            plane[ii1 - 1, ij1 - 1] = zsgn * plane[ii2 - 1, ij2 - 1]
        # `DO ji = 1, ihls` (`lbcnfd.f90:141-146`)
        for ji in range(1, ihls + 1):
            ii1 = ji
            ii2 = 2 * ihls + 2 - ji
            plane[ii1 - 1, ij1 - 1] = zsgn * plane[ii2 - 1, ij2 - 1]

        # 3. South closed boundary (`rows 1..ihls` in 1-based set to `0.0`)
        plane[:, :ihls] = 0.0

    return arr_3d.reshape(orig_shape)


def _build_orca2_advection_grid(
    tmask: jnp.ndarray,
    umask: jnp.ndarray,
    vmask: jnp.ndarray,
) -> SI3ORCA2AdvectionGrid:
    """Build a realistic curved ORCA2 `(184, 152)` advection metric grid."""
    jpi, jpj = tmask.shape
    ii, jj = np.meshgrid(np.arange(jpi), np.arange(jpj), indexing="ij")
    e1t_np = 185000.0 + 27000.0 * np.cos(2.0 * np.pi * (ii - 2) / 180.0) + 113.7 * (jj / 152.0)
    e2t_np = 165000.0 + 31000.0 * np.sin(np.pi * (jj - 2) / 148.0) + 89.3 * (ii / 184.0)
    e1u_np = 0.5 * (e1t_np + np.roll(e1t_np, -1, axis=0))
    e2u_np = 0.5 * (e2t_np + np.roll(e2t_np, -1, axis=0))
    e1v_np = 0.5 * (e1t_np + np.roll(e1t_np, -1, axis=1))
    e2v_np = 0.5 * (e2t_np + np.roll(e2t_np, -1, axis=1))
    e1e2t_np = e1t_np * e2t_np

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
# and failing reversal/reordering/sign plants (`Plants O4-A1..A4`).
# ==============================================================================


def test_o4_a1_adv_x_and_adv_y_statement_pins_and_plants() -> None:
    """`si3_adv_x` and `si3_adv_y` match scalar-loop Fortran reference to 0 ULP on inexact operands and fail under Plant O4-A1."""
    rng = np.random.default_rng(2026102401)
    jpi, jpj, jpl = 12, 11, 3
    nn_hls = 2
    pdt = 10800.0 / 3.0 + 0.1357911

    e1e2t_np = 3.7123456789e9 + rng.uniform(-2.5e8, 2.5e8, size=(jpi, jpj))
    tmask_np = np.ones((jpi, jpj), dtype=np.float64)
    tmask_np[4, 5] = 0.0
    put_np = rng.uniform(-4.2e5, 4.5e5, size=(jpi, jpj))
    pvt_np = rng.uniform(-3.8e5, 4.1e5, size=(jpi, jpj))
    psm_np = e1e2t_np * rng.uniform(0.88, 1.12, size=(jpi, jpj))

    ps0_np = rng.uniform(-0.2, 2.8, size=(jpi, jpj, jpl)) * e1e2t_np[:, :, None]
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

    # Plant O4-A1a: omitting donor readjustment MUST fail 0-ULP
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

    # Plant O4-A1b: restoring `zfx` in `adv_x`'s negative-velocity branch MUST fail 0-ULP
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

    # Plant O4-A1c: reordering `zalf1q * (psx - 3 * zalf * psxx)` association MUST fail 0-ULP
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


def test_o4_a2_odd_even_sweep_order_and_cfl_subcycling_pins_and_plant() -> None:
    """`si3_dyn_adv_pra_orca2` enforces NEMO's `MOD((kt-1)//nn_fsbc, 2) == MOD(jt-1, 2)` sweep parity, `icycle in {1, 2, 3}` CFL thresholds, and `ln_mppdelay` one-call delayed `cflice` semantics (`icedyn_adv_pra.f90:131-141`, `lib_mpp.f90:789-846`), and fails under Plants O4-A2a and O4-A2b."""
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
        ln_mppdelay=True,
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
        e1u=e1u,
        e2v=e2v,
        e2u=e2u,
        e1v=e1v,
        e1e2t=e1e2t,
        tmask=tmask,
        umask=tmask,
        vmask=tmask,
    )

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
    sxy_ice = jnp.asarray(rng.uniform(-1.0e7, 1.0e7, size=(jpi, jpj, cfg.jpl)))
    prather = prather._replace(ice=prather.ice._replace(sxy=sxy_ice))

    z2 = jnp.zeros((jpi, jpj), dtype=jnp.float64)
    acc = dict(
        hi_max_top=99.0,
        wfx_res=z2,
        hfx_res=z2,
        sfx_res=z2,
        diag_adv_mass=z2,
        diag_adv_salt=z2,
        diag_adv_heat=z2,
    )

    # 1a. First call from rest (`cflice_delayed=None`, `lib_mpp.f90:801-828`):
    #     e1u = 200000 m, rDt_ice = 10800 s =>
    #     u = 5.0 m/s -> zcfl_curr = 0.27 -> icycle = 1, cflice_delayed_out = 0.27
    #     u = 15.0 m/s -> zcfl_curr = 0.81 -> icycle = 2, cflice_delayed_out = 0.81
    #     u = 32.0 m/s -> zcfl_curr = 1.728 -> icycle = 3, cflice_delayed_out = 1.728
    cfl_5 = (np.float64(5.0) * np.float64(10800.0)) * (np.float64(1.0) / np.float64(200000.0))
    cfl_15 = (np.float64(15.0) * np.float64(10800.0)) * (np.float64(1.0) / np.float64(200000.0))
    cfl_32 = (np.float64(32.0) * np.float64(10800.0)) * (np.float64(1.0) / np.float64(200000.0))
    for u_speed, expected_icycle, expected_cfl in (
        (5.0, 1, cfl_5),
        (15.0, 2, cfl_15),
        (32.0, 3, cfl_32),
    ):
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
        **acc,
            cflice_delayed=None,
        )
        assert int(res[-1]) == expected_icycle
        _assert_zero_ulp(res[-3], expected_cfl, f"first_call_zcfl_{u_speed}")
        _assert_zero_ulp(res[-2], expected_cfl, f"first_call_delayed_out_{u_speed}")

    # 1b. Subsequent calls with `ln_mppdelay = True` (`lib_mpp.f90:830-842`):
    #     Call 2 receives `cflice_delayed = cfl_5` (from Call 1 with u=5.0) while current
    #     velocity is `u = 32.0` (`zcfl_curr = cfl_32`): NEMO uses the delayed `cfl_5`
    #     so `icycle = 1` (NOT 3), and stores `cflice_delayed_out = cfl_32` for Call 3!
    u_fast = jnp.full((jpi, jpj), 32.0, dtype=jnp.float64)
    v_fast = jnp.full((jpi, jpj), 0.4 * 32.0, dtype=jnp.float64)
    res_call2 = si3_dyn_adv_pra_orca2(
        kt=2,
        u_ice=u_fast,
        v_ice=v_fast,
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
        **acc,
        cflice_delayed=jnp.asarray(cfl_5, dtype=jnp.float64),
    )
    assert int(res_call2[-1]) == 1  # delayed zcfl = 0.27 -> icycle = 1
    _assert_zero_ulp(res_call2[-3], cfl_5, "call2_used_delayed_zcfl")
    _assert_zero_ulp(res_call2[-2], cfl_32, "call2_stored_current_zcfl")

    # Call 3 receives `cflice_delayed = res_call2[-2] = cfl_32` while current velocity
    # drops back to `u = 5.0` (`zcfl_curr = cfl_5`): NEMO uses the delayed `cfl_32`
    # so `icycle = 3` (NOT 1), and stores `cflice_delayed_out = cfl_5`!
    u_slow = jnp.full((jpi, jpj), 5.0, dtype=jnp.float64)
    v_slow = jnp.full((jpi, jpj), 0.4 * 5.0, dtype=jnp.float64)
    res_call3 = si3_dyn_adv_pra_orca2(
        kt=3,
        u_ice=u_slow,
        v_ice=v_slow,
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
        **acc,
        cflice_delayed=res_call2[-2],
    )
    assert int(res_call3[-1]) == 3  # delayed zcfl = 1.728 -> icycle = 3
    _assert_zero_ulp(res_call3[-3], cfl_32, "call3_used_delayed_zcfl")
    _assert_zero_ulp(res_call3[-2], cfl_5, "call3_stored_current_zcfl")

    # Plant O4-A2b: ignoring `ln_mppdelay` (`_plant_ignore_mppdelay=True`) on Call 2
    # uses current-step `zcfl = 1.728` (`icycle = 3` instead of `1`) and MUST fail 0-ULP!
    res_call2_planted = si3_dyn_adv_pra_orca2(
        kt=2,
        u_ice=u_fast,
        v_ice=v_fast,
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
        **acc,
        cflice_delayed=jnp.asarray(0.27, dtype=jnp.float64),
        _plant_ignore_mppdelay=True,
    )
    assert int(res_call2_planted[-1]) == 3
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(res_call2[1], res_call2_planted[1], "plant_o4_a2b_ignore_mppdelay")

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
        **acc,
        cflice_delayed=None,
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
        **acc,
        cflice_delayed=None,
    )
    # Plant O4-A2a: inverting sweep order at kt=1 must equal kt=2 (for icycle=1) and fail 0-ULP vs kt=1!
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
        **acc,
        cflice_delayed=None,
        _plant_invert_sweep_order=True,
    )
    _assert_zero_ulp(out_kt2[1], out_kt1_planted[1], "kt2_matches_inverted_kt1_v_i")
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(out_kt1[1], out_kt1_planted[1], "plant_o4_a2a_v_i")
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(
            out_kt1[12].ice.sxy, out_kt1_planted[12].ice.sxy, "plant_o4_a2a_sxy"
        )


def test_o4_a3_icemax_hbig_hsnow_zapneg_zapsmall_pato_diag_pins_and_plants() -> None:
    """`si3_icemax2d_pra`, `si3_icemax3d_pra`, `si3_hbig_pra`, `si3_hsnow_pra`, `si3_var_zapneg`, `si3_var_zapsmall`, and `pato_i` match scalar Fortran references to 0 ULP, refuse missing inputs, and fail under Plants O4-A3."""
    rng = np.random.default_rng(2026102403)
    jpi, jpj, jpl = 9, 8, 2
    nlay_i, nlay_s = 3, 2
    nn_hls = 2
    ihls = 0
    pdt = 10800.0

    # 1. Pin `si3_icemax2d_pra` and `si3_icemax3d_pra` (`icedyn_adv_pra.f90:1519-1588`)
    pice2d = rng.uniform(0.0, 3.5, size=(jpi, jpj, jpl))
    pice2d[4, 4, 0] = 5.0e-12
    pice3d = rng.uniform(0.0, 4.0e8, size=(jpi, jpj, nlay_i, jpl))

    ref_max2d = np.zeros_like(pice2d)
    for jl in range(jpl):
        for jj in range(nn_hls - ihls, jpj - nn_hls + ihls):
            for ji in range(nn_hls - ihls, jpi - nn_hls + ihls):
                ref_max2d[ji, jj, jl] = max(
                    1.0e-20,
                    float(np.max(pice2d[ji - 1 : ji + 2, jj - 1 : jj + 2, jl])),
                )
    got_max2d = si3_icemax2d_pra(ihls, jnp.asarray(pice2d), nn_hls=nn_hls, mask_inactive=True)
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
    got_max3d = si3_icemax3d_pra(ihls, jnp.asarray(pice3d), nn_hls=nn_hls, mask_inactive=True)
    _assert_zero_ulp(ref_max3d, got_max3d, "si3_icemax3d_pra")

    # 2. Pin `si3_hbig_pra` (`icedyn_adv_pra.f90:960-1093`) with inexact operands
    zhi_max = jnp.full((jpi, jpj, jpl), 1.7319587628865979, dtype=jnp.float64)
    zhs_max = jnp.full((jpi, jpj, jpl), 0.31958762886597936, dtype=jnp.float64)
    zhip_max = jnp.full((jpi, jpj, jpl), 0.17525773195876287, dtype=jnp.float64)
    zsi_max = jnp.full((jpi, jpj, jpl), 4.711340206185567, dtype=jnp.float64)
    zes_max = jnp.full((jpi, jpj, nlay_s, jpl), 1.134020618556701e8, dtype=jnp.float64)
    zei_max = jnp.full((jpi, jpj, nlay_i, jpl), 2.3711340206185567e8, dtype=jnp.float64)
    zszi_max = jnp.full((jpi, jpj, nlay_i, jpl), 5.154639175257732, dtype=jnp.float64)

    a_i = jnp.asarray(rng.uniform(0.02, 0.08, size=(jpi, jpj, jpl)))
    v_i = a_i * (zhi_max * 1.15)
    v_s = a_i * (zhs_max * 1.41)
    a_ip = a_i * 0.45
    v_ip = a_ip * (zhip_max * 1.29)
    sv_i = v_i * (zsi_max * 1.33)
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
        ln_pnd_LEV=True,
        ln_pnd_TOPO=False,
        hi_max_top=99.0,
        rhoi=canonical_constants.rho_ice,
        rhos=canonical_constants.rho_snow,
        apply_hi_cap=True,
        multiply_before_divide=False,
        extensive_area=None,
    )
    active = np.asarray(si3_active_mask_2d(jpi, jpj, nn_hls, ihls))
    v_i_after, _, a_i_after, a_ip_after, _, _, _, _, szv_i_after, _, _, _ = hbig_out
    _assert_zero_ulp(
        np.asarray(a_ip_after)[active],
        np.asarray(v_ip / zhip_max)[active],
        "hbig_a_ip_capped",
    )

    # Plant O4-A3a: `_plant_perturb_hbig_cap=True` MUST fail 0-ULP
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
        ln_pnd_LEV=True,
        ln_pnd_TOPO=False,
        hi_max_top=99.0,
        rhoi=canonical_constants.rho_ice,
        rhos=canonical_constants.rho_snow,
        apply_hi_cap=True,
        multiply_before_divide=False,
        extensive_area=None,
        _plant_perturb_hbig_cap=True,
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(a_i_after, hbig_planted[2], "plant_perturb_hbig_cap")

    # Plant O4-A3b: reordering `zfra = zszi_max / zsi ; pszv_i = pszv_i * zfra` to
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
        use_reciprocal_rhos=True,
        extensive_area=None,
    )
    _, v_s_hsnow, _, a_ip_hsnow, _, _, _ = hsnow_out
    buoy_cap = np.asarray(v_i) * (rho0 - rhoi) * (1.0 / rhos)
    zvs_excess = np.maximum(0.0, np.asarray(v_s_heavy) - buoy_cap)
    ref_vs_rem = np.where(active[..., None], np.asarray(v_s_heavy) - zvs_excess, np.asarray(v_s_heavy))
    _assert_zero_ulp(ref_vs_rem, v_s_hsnow, "hsnow_v_s")
    _assert_zero_ulp(
        np.where(active[..., None], np.minimum(np.asarray(a_ip_big), np.asarray(a_i)), np.asarray(a_ip_big)),
        a_ip_hsnow,
        "hsnow_a_ip",
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(v_s_hsnow, v_s_heavy, "plant_omit_archimedes")

    # 4. Pin `si3_var_zapneg` (`icevar.f90:740-853`)
    v_i_z = jnp.array(v_i)
    v_s_z = jnp.array(v_s)
    sv_i_z = jnp.array(sv_i)
    oa_i_z = jnp.array(oa_i).at[3, 3, 0].set(-0.05)
    a_i_z = jnp.array(a_i)
    a_ip_z = jnp.array(a_ip)
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
        rhoi=canonical_constants.rho_ice,
        rhos=canonical_constants.rho_snow,
        rhow=canonical_constants.rho_water,
    )
    assert float(zapneg_out[7][4, 4, 0]) == 0.0
    z1_dt_np = np.float64(1.0) / np.float64(pdt)
    rhow_np = np.float64(canonical_constants.rho_water)
    expected_wfx_44 = (np.float64(0.025) * rhow_np) * z1_dt_np + (
        np.float64(-0.015) * rhow_np
    ) * z1_dt_np
    _assert_zero_ulp(zapneg_out[11][4, 4], expected_wfx_44, "zapneg_wfx_res_44")

    wrong_wfx_44 = (np.float64(-0.015) * rhow_np) * z1_dt_np
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(zapneg_out[11][4, 4], wrong_wfx_44, "plant_late_vil_check")

    # 5. Pin `si3_var_zapsmall` (`icevar.f90:615-737`)
    a_i_small = jnp.array(a_i).at[3, 3, 0].set(0.5e-10)
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
        rn_simin=0.1,
        rn_sinew=0.75,
        rt0=canonical_constants.T_freeze,
        rhoi=canonical_constants.rho_ice,
        rhos=canonical_constants.rho_snow,
        rhow=canonical_constants.rho_water,
    )
    assert float(zapsmall_out[0][3, 3, 0]) == 0.0
    assert float(zapsmall_out[4][3, 3, 0]) == 0.0
    _assert_zero_ulp(
        zapsmall_out[8][3, 3, 0],
        canonical_constants.T_freeze - 1.8,
        "zapsmall_t_su_reset",
    )

    zapsmall_namelist = dict(
        rn_simin=0.1,
        rn_sinew=0.75,
        rt0=canonical_constants.T_freeze,
        rhoi=canonical_constants.rho_ice,
        rhos=canonical_constants.rho_snow,
        rhow=canonical_constants.rho_water,
    )
    # 6. Verify explicit missing-input refusal gates (no hidden defaults for sss_m, sst_m, umask, vmask, si3_orca2)
    with pytest.raises(ValueError, match="sss_m"):
        si3_var_zapsmall(
            v_i, v_s, sv_i, oa_i, a_i_small, a_ip, v_ip, v_il,
            t_su_in, h_i_in, h_s_in, h_ip_in, h_il_in, e_s, e_i, szv_i, sz_i_in,
            t_i_in, t_s_in, at_i_in, vt_i_in, ato_i_in, wfx0, hfx0, sfx0,
            rDt_ice=pdt, sss_m=None, sst_m=-1.8, nn_hls=nn_hls, nn_icesal=4, **zapsmall_namelist,
        )
    with pytest.raises(ValueError, match="sst_m"):
        si3_var_zapsmall(
            v_i, v_s, sv_i, oa_i, a_i_small, a_ip, v_ip, v_il,
            t_su_in, h_i_in, h_s_in, h_ip_in, h_il_in, e_s, e_i, szv_i, sz_i_in,
            t_i_in, t_s_in, at_i_in, vt_i_in, ato_i_in, wfx0, hfx0, sfx0,
            rDt_ice=pdt, sss_m=34.7, sst_m=None, nn_hls=nn_hls, nn_icesal=4, **zapsmall_namelist,
        )
    with pytest.raises(TypeError):
        SI3ORCA2AdvectionGrid.from_metrics(
            e1u=wfx0 + 1.0, e2v=wfx0 + 1.0, e2u=wfx0 + 1.0, e1v=wfx0 + 1.0,
            e1e2t=wfx0 + 1.0, tmask=wfx0 + 1.0,
        )
    with pytest.raises(ValueError, match="umask"):
        SI3ORCA2AdvectionGrid.from_metrics(
            e1u=wfx0 + 1.0, e2v=wfx0 + 1.0, e2u=wfx0 + 1.0, e1v=wfx0 + 1.0,
            e1e2t=wfx0 + 1.0, tmask=wfx0 + 1.0, umask=None, vmask=wfx0 + 1.0,
        )
    with pytest.raises(ValueError, match="vmask"):
        SI3ORCA2AdvectionGrid.from_metrics(
            e1u=wfx0 + 1.0, e2v=wfx0 + 1.0, e2u=wfx0 + 1.0, e1v=wfx0 + 1.0,
            e1e2t=wfx0 + 1.0, tmask=wfx0 + 1.0, umask=wfx0 + 1.0, vmask=None,
        )
    with pytest.raises(TypeError):
        build_si3_orca2_o4_config()
    with pytest.raises(ValueError, match="SI3ORCA2StateConfig"):
        build_si3_orca2_o4_config(None)  # type: ignore[arg-type]


def test_o4_a4_orca2_seam_and_north_fold_subcycle_lbc_lnk_pins_and_plants() -> None:
    """`si3_dyn_adv_pra_orca2` exchanges all 11 extensive families and 55 Prather moment arrays across the ORCA2 cyclic seam and T-pivot north fold at every subcycle (`icycle >= 2`), matching an independent NEMO-literal `lbc_lnk` replica at every fold and seam cell, and failing when any of `sx, sy, sxx, syy, sxy` has a wrong north-fold sign (`Plants O4-A4a..f`)."""
    o4_state_cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(ln_icedyn=True)
    cfg = build_si3_orca2_o4_config(o4_state_cfg).si3_orca2
    assert cfg is not None
    state0, tmask, umask, vmask = o1_gate.build_synthetic_orca2_o1_state(cfg)

    # Ensure wet cells and non-zero ice + non-zero Prather moments across the
    # north fold (`jj = 146..151`) and cyclic east-west seam (`ji = 0..5, 178..183`)
    # so every moment of every family is non-vacuously tested at both the fold and seam.
    tmask_full = si3_lbc_lnk_orca2(
        tmask.at[:, 140:150].set(1.0), "T", 1.0, nn_hls=2, ldfull=True
    )
    umask_full = si3_lbc_lnk_orca2(
        umask.at[:, 140:150].set(1.0), "U", 1.0, nn_hls=2, ldfull=True
    )
    vmask_full = si3_lbc_lnk_orca2(
        vmask.at[:, 140:150].set(1.0), "V", 1.0, nn_hls=2, ldfull=True
    )
    grid = _build_orca2_advection_grid(tmask_full, umask_full, vmask_full)

    rng = np.random.default_rng(2026102405)
    e1e2t_np = np.asarray(grid.e1e2t)

    def _seed_moments(shape: tuple[int, ...], base_scale: float) -> SI3PratherMoments:
        area_b = e1e2t_np.reshape((184, 152) + (1,) * (len(shape) - 2))
        return SI3PratherMoments(
            sx=jnp.asarray(rng.uniform(-0.12, 0.14, size=shape) * base_scale * area_b),
            sy=jnp.asarray(rng.uniform(-0.11, 0.13, size=shape) * base_scale * area_b),
            sxx=jnp.asarray(rng.uniform(-0.04, 0.05, size=shape) * base_scale * area_b),
            syy=jnp.asarray(rng.uniform(-0.04, 0.05, size=shape) * base_scale * area_b),
            sxy=jnp.asarray(rng.uniform(-0.03, 0.04, size=shape) * base_scale * area_b),
        )

    a_i_seed = jnp.asarray(rng.uniform(0.08, 0.14, size=state0.a_i.shape))
    v_i_seed = a_i_seed * jnp.asarray(rng.uniform(0.5, 1.8, size=state0.v_i.shape))
    v_s_seed = a_i_seed * jnp.asarray(rng.uniform(0.03, 0.12, size=state0.v_s.shape))
    sv_i_seed = v_i_seed * 4.2
    oa_i_seed = a_i_seed * 18.0
    a_ip_seed = a_i_seed * 0.35
    v_ip_seed = a_ip_seed * 0.07
    v_il_seed = a_ip_seed * 0.02
    e_s_seed = jnp.asarray(rng.uniform(1.0e7, 4.0e7, size=state0.e_s.shape))
    e_i_seed = jnp.asarray(rng.uniform(1.0e8, 3.5e8, size=state0.e_i.shape))
    szv_i_seed = jnp.asarray(rng.uniform(0.4, 2.2, size=state0.szv_i.shape))

    prather_seed = SI3AllPratherMoments(
        ice=_seed_moments(state0.v_i.shape, 0.4),
        sn=_seed_moments(state0.v_s.shape, 0.03),
        a=_seed_moments(state0.a_i.shape, 0.08),
        age=_seed_moments(state0.oa_i.shape, 1.0),
        sal=_seed_moments(state0.sv_i.shape, 1.2),
        ap=_seed_moments(state0.a_ip.shape, 0.02),
        vp=_seed_moments(state0.v_ip.shape, 0.002),
        vl=_seed_moments(state0.v_il.shape, 0.0005),
        c0=_seed_moments(state0.e_s.shape, 1.0e6),
        e=_seed_moments(state0.e_i.shape, 1.0e7),
        si=_seed_moments(state0.szv_i.shape, 0.1),
    )

    u_np = np.full((184, 152), 14.5, dtype=np.float64)
    v_np = np.full((184, 152), 6.2, dtype=np.float64)
    u_ice = si3_lbc_lnk_orca2(
        jnp.asarray(u_np) * grid.umask, "U", -1.0, nn_hls=2, ldfull=True
    )
    v_ice = si3_lbc_lnk_orca2(
        jnp.asarray(v_np) * grid.vmask, "V", -1.0, nn_hls=2, ldfull=True
    )
    state_in = si3_apply_state_halos_orca2(
        state0._replace(
            a_i=a_i_seed,
            v_i=v_i_seed,
            v_s=v_s_seed,
            sv_i=sv_i_seed,
            oa_i=oa_i_seed,
            a_ip=a_ip_seed,
            v_ip=v_ip_seed,
            v_il=v_il_seed,
            e_s=e_s_seed,
            e_i=e_i_seed,
            szv_i=szv_i_seed,
            u_ice=u_ice,
            v_ice=v_ice,
            prather=prather_seed,
        ),
        config=cfg,
    )
    sss_m = jnp.full((184, 152), 34.2, dtype=jnp.float64)
    sst_m = jnp.full((184, 152), -1.8, dtype=jnp.float64)

    # Refuse missing sss_m or sst_m in `si3_dyn_orca2` and `si3_o4_step`
    with pytest.raises(ValueError, match="sss_m"):
        si3_dyn_orca2(state_in, grid, kt=1, sss_m=None, sst_m=sst_m, config=cfg, **_dyn_inputs())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sst_m"):
        si3_dyn_orca2(state_in, grid, kt=1, sss_m=sss_m, sst_m=None, config=cfg, **_dyn_inputs())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sss_m"):
        si3_o4_step(state_in, grid, kt=1, sss_m=None, sst_m=sst_m, config=cfg, rn_uice=None, rn_vice=None, apply_halos=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sst_m"):
        si3_o4_step(state_in, grid, kt=1, sss_m=sss_m, sst_m=None, config=cfg, rn_uice=None, rn_vice=None, apply_halos=False)  # type: ignore[arg-type]

    state_out, diag = si3_dyn_orca2(
        state_in, grid, kt=1, sss_m=sss_m, sst_m=sst_m, config=cfg, **_dyn_inputs()
    )
    assert int(diag.icycle) == 2

    # Compare every extensive and all 55 Prather moment arrays at every north-fold
    # cell (`jj = 149, 150, 151`) and every east-west cyclic seam cell
    # (`ji = 0, 1, 182, 183`) against the independent NEMO-literal replica
    # `_ref_nemo_lbc_lnk_orca2_t_point` (`icedyn_adv_pra.f90:450-494`, `lbcnfd.f90:96-150`).
    moment_signs = {
        "sx": -1.0,
        "sy": -1.0,
        "sxx": 1.0,
        "syy": 1.0,
        "sxy": 1.0,
    }
    fold_rows = slice(149, 152)
    seam_cols = [0, 1, 182, 183]

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
        arr_out = np.asarray(getattr(state_out, fname), dtype=np.float64)
        ref_linked = _ref_nemo_lbc_lnk_orca2_t_point(arr_out, psgn=1.0, nn_hls=2)
        _assert_zero_ulp(arr_out[:, fold_rows], ref_linked[:, fold_rows], f"fold.{fname}")
        _assert_zero_ulp(arr_out[seam_cols, :], ref_linked[seam_cols, :], f"seam.{fname}")

    for fam_name in state_out.prather._fields:
        fam_out = getattr(state_out.prather, fam_name)
        for mname, psgn in moment_signs.items():
            m_arr = np.asarray(getattr(fam_out, mname), dtype=np.float64)
            ref_m = _ref_nemo_lbc_lnk_orca2_t_point(m_arr, psgn=psgn, nn_hls=2)
            assert float(np.max(np.abs(m_arr[:, fold_rows]))) > 0.0, (
                f"vacuous zero fold moments for {fam_name}.{mname}"
            )
            assert float(np.max(np.abs(m_arr[seam_cols, :]))) > 0.0, (
                f"vacuous zero seam moments for {fam_name}.{mname}"
            )
            _assert_zero_ulp(
                m_arr[:, fold_rows],
                ref_m[:, fold_rows],
                f"independent_nfd_replica.prather.{fam_name}.{mname}",
            )
            _assert_zero_ulp(
                m_arr[seam_cols, :],
                ref_m[seam_cols, :],
                f"independent_seam_replica.prather.{fam_name}.{mname}",
            )

    # Plant O4-A4a: omitting inter-subcycle `lbc_lnk` (`_plant_omit_subcycle_lbc=True`)
    # corrupts seam and north-fold interior cells on subcycle 2 (`icycle = 2`) and fails 0-ULP!
    state_no_lbc, _ = si3_dyn_orca2(
        state_in,
        grid,
        kt=1,
        sss_m=sss_m,
        sst_m=sst_m,
        config=cfg,
        **_dyn_inputs(),
        _plant_omit_subcycle_lbc=True,
    )
    with pytest.raises(AssertionError, match="max ULP"):
        _assert_zero_ulp(
            state_out.v_i[2:182, 2:150, :],
            state_no_lbc.v_i[2:182, 2:150, :],
            "plant_omit_subcycle_lbc_interior_v_i",
        )

    # Plants O4-A4b..f: planting a wrong north-fold sign on EACH of `sx, sy, sxx, syy, sxy`
    # individually (`_plant_wrong_nfd_sign=mname`) MUST fail the independent NEMO-literal
    # north-fold replica check on that moment (`icedyn_adv_pra.f90:450-490`)!
    for mname, psgn in moment_signs.items():
        state_bad_m, _ = si3_dyn_orca2(
            state_in,
            grid,
            kt=1,
            sss_m=sss_m,
            sst_m=sst_m,
            config=cfg,
            **_dyn_inputs(),
            _plant_wrong_nfd_sign=mname,
        )
        bad_arr = np.asarray(getattr(state_bad_m.prather.ice, mname), dtype=np.float64)
        ref_arr = _ref_nemo_lbc_lnk_orca2_t_point(bad_arr, psgn=psgn, nn_hls=2)
        with pytest.raises(AssertionError, match="max ULP"):
            _assert_zero_ulp(
                bad_arr[:, fold_rows],
                ref_arr[:, fold_rows],
                f"plant_wrong_nfd_sign_{mname}",
            )


# ==============================================================================
# Test (b): Lane-3 `ICE_ADV1D` / `ICE_ADV2D` invariance and real parent-vs-current
# (`042c7bd74c2944dab8c39318687ab0cef54f9d2b`) 0-ULP bit-identity control
# (`Plants O4-B1, O4-B2`).
# ==============================================================================


_LANE3_DUMP = TESTCASES_DIR / "nemo_si3_lane3_prather_dump.py"
_PACKAGE_ROOTS = (
    "packages/core",
    "packages/ocean",
    "packages/atmosphere",
    "packages/coupler",
    "packages/ice",
    "packages/land",
    "packages/ml",
    "packages/tools",
    "src",
)


def _run_lane3_dump(tree: Path, out: Path, *extra: str) -> dict[str, np.ndarray]:
    """Run the lane-3 dump in a fresh interpreter whose `legoesm` comes only from `tree`."""
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(str(tree / root) for root in _PACKAGE_ROOTS),
        JAX_ENABLE_X64="1",
        JAX_PLATFORMS="cpu",
    )
    subprocess.run(
        [sys.executable, str(_LANE3_DUMP), str(out), *extra],
        env=env,
        cwd=out.parent,
        check=True,
    )
    with np.load(out) as dump:
        return {key: dump[key] for key in dump.files}


def test_o4_b_lane3_parent_vs_current_zero_ulp(tmp_path: Path) -> None:
    """Lane-3 cards, the shared correction ledger and `step_sea_ice` match parent `042c7bd74c29` at 0 ULP; a real arithmetic-order plant fails."""
    parent_tree = tmp_path / "parent"
    parent_tree.mkdir()
    archive = subprocess.run(
        ["git", "archive", PARENT_COMMIT, *_PACKAGE_ROOTS],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(parent_tree)], input=archive, check=True)

    parent = _run_lane3_dump(parent_tree, tmp_path / "parent.npz")
    current = _run_lane3_dump(REPO_ROOT, tmp_path / "current.npz")
    assert str(parent.pop("_module_file")).startswith(str(parent_tree))
    assert str(current.pop("_module_file")).startswith(str(REPO_ROOT / "packages"))
    assert parent.keys() == current.keys()
    for key, value in parent.items():
        if value.dtype == np.float64:
            _assert_zero_ulp(value, current[key], f"lane3.{key}")
        else:
            np.testing.assert_array_equal(value, current[key], err_msg=key)

    # Non-vacuity: Hsnow/Hbig/zapneg change most tracers in both calling conventions.
    for tag in ("int", "ext"):
        assert int(np.count_nonzero(current[f"corr_{tag}.changed_per_tracer"])) >= 10

    # Plant O4-B1: flipping Hbig's divide/multiply order in the lane-3 ledger must fail.
    planted = _run_lane3_dump(REPO_ROOT, tmp_path / "planted.npz", "--plant-multiply-order")
    planted.pop("_module_file")
    differing = {key for key, value in parent.items() if not np.array_equal(value, planted[key])}
    assert {"corr_int", "corr_ext"} <= differing


# ==============================================================================
# Test (c): Conservation of every extensive to round-off on a closed ORCA2
# `(184, 152)` test field including ice at and next to the north fold (`Plant O4-C1`).
# ==============================================================================


def _build_closed_orca2_conservation_state(
    cfg: SI3ORCA2StateConfig,
) -> tuple[SI3State, SI3ORCA2AdvectionGrid]:
    """Build a closed ORCA2 `(184, 152)` state with active cyclic-seam flow and ice at and next to the north fold (`jj = 148, 149`)."""
    state0, tmask, umask, vmask = o1_gate.build_synthetic_orca2_o1_state(cfg)
    jpi, jpj = 184, 152

    # Make northern band (`jj = 130..149`) wet and fold-symmetric so ice is
    # present and actively advected AT (`jj = 149`) and NEXT TO (`jj = 148`)
    # the ORCA2 T-pivot north fold (`Nje0 = 150` in 1-based indexing).
    tmask_wet_fold = si3_lbc_lnk_orca2(
        tmask.at[:, 130:150].set(1.0), "T", 1.0, nn_hls=2, ldfull=True
    )
    umask_wet_fold = si3_lbc_lnk_orca2(
        umask.at[:, 130:150].set(1.0), "U", 1.0, nn_hls=2, ldfull=True
    )
    vmask_wet_fold = si3_lbc_lnk_orca2(
        vmask.at[:, 130:150].set(1.0), "V", 1.0, nn_hls=2, ldfull=True
    )
    grid = _build_orca2_advection_grid(tmask_wet_fold, umask_wet_fold, vmask_wet_fold)

    # Construct a smooth ice field in `jj in [12, 149]` (including `jj = 148` next to
    # the fold and `jj = 149` at the fold) that is symmetric under the T-pivot
    # reflection `ji <-> 184 - ji` (`cos(2*pi*(ii - 2)/180)`) and crosses the
    # cyclic east-west seam (`ji = 2 <-> 181`), with `a_cat >= 0.165 > 0.15` on
    # all active wet cells so `Hbig_pra`, `Hsnow_pra`, `ice_var_zapneg`, and
    # `ice_var_zapsmall` do not clip.
    ii, jj = np.meshgrid(np.arange(jpi), np.arange(jpj), indexing="ij")
    lat_ice = np.where(
        (jj >= 12) & (jj <= 149),
        1.0 + 0.12 * np.exp(-0.5 * ((jj - 90.0) / 35.0) ** 2),
        0.0,
    )
    lon_angle = 2.0 * np.pi * (ii - 2.0) / 180.0
    base_2d = lat_ice * (1.0 + 0.06 * np.cos(lon_angle) + 0.03 * np.cos(2.0 * lon_angle))
    base_2d = np.asarray(
        si3_lbc_lnk_orca2(
            jnp.asarray(base_2d) * tmask_wet_fold, "T", 1.0, nn_hls=2, ldfull=True
        )
    )

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

    # Velocities across the cyclic east-west seam (`ji = 2 <-> 181`) and next to / at
    # the north fold (`jj = 147, 148, 149`):
    # - On `jj = 20..148` (including `jj = 148`, next to the fold), `u_ice` crosses
    #   the cyclic seam (`ji = 2 <-> 181`) and `v_ice` on `jj = 20..147` transports
    #   directly into/out of `jj = 148` (next to the fold).
    # - On `jj = 149` (the T-pivot fold row), `u_ice` is T-pivot anti-symmetric
    #   (`u_ice[183 - ji, 149] == -u_ice[ji, 149]`, via `sin((ii - 1.5)*2*pi/180)`)
    #   so zonal advection is active right on the fold row `jj = 149` while preserving
    #   the T-pivot fold symmetry `field[184 - ji, 149] == field[ji, 149]`.
    tmask_np = np.asarray(tmask_wet_fold)
    wet_u = np.asarray(umask_wet_fold) * tmask_np * np.roll(tmask_np, -1, axis=0)
    wet_v = np.asarray(vmask_wet_fold) * tmask_np * np.roll(tmask_np, -1, axis=1)
    u_band_subfold = ((jj >= 20) & (jj <= 148)).astype(np.float64)
    u_band_fold = (jj == 149).astype(np.float64)
    v_band = ((jj >= 20) & (jj <= 147)).astype(np.float64)
    u_fold_sym = 0.28 * np.sin(2.0 * np.pi * (ii - 1.5) / 180.0)
    u_np = ((0.35 + 0.12 * np.cos(lon_angle)) * u_band_subfold + u_fold_sym * u_band_fold) * wet_u
    v_np = (0.22 * np.cos(lon_angle)) * v_band * wet_v
    u_ice_j = si3_lbc_lnk_orca2(jnp.asarray(u_np), "U", -1.0, nn_hls=2, ldfull=True)
    v_ice_j = si3_lbc_lnk_orca2(jnp.asarray(v_np), "V", -1.0, nn_hls=2, ldfull=True)

    e1e2t_3d = np.asarray(grid.e1e2t)[:, :, None]
    e1e2t_4d = np.asarray(grid.e1e2t)[:, :, None, None]
    sin_3d = np.sin(lon_angle)[:, :, None]
    sin_4d = np.sin(lon_angle)[:, :, None, None]

    def _make_moments(ext_arr: np.ndarray) -> SI3PratherMoments:
        scale = ext_arr * (e1e2t_3d if ext_arr.ndim == 3 else e1e2t_4d)
        s_ang = sin_3d if ext_arr.ndim == 3 else sin_4d
        # Zero initial moments on `jj >= 148` so the negative-velocity `zfx = 0`
        # asymmetry in `adv_x` (`icedyn_adv_pra.f90:615,642-648`) does not break
        # `sx` mirror symmetry on the T-pivot row `jj = 149`, while keeping
        # non-zero `sx, sy, sxx, syy, sxy` across `jj < 148`.
        m_mask = ((jj < 148).astype(np.float64))[:, :, None]
        if ext_arr.ndim == 4:
            m_mask = m_mask[:, :, None, :]
        return SI3PratherMoments(
            sx=jnp.asarray(0.08 * s_ang * scale * m_mask),
            sy=jnp.asarray(-0.06 * s_ang * scale * m_mask),
            sxx=jnp.asarray(0.02 * scale * m_mask),
            syy=jnp.asarray(0.015 * scale * m_mask),
            sxy=jnp.asarray(0.01 * scale * m_mask),
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
        u_ice=u_ice_j,
        v_ice=v_ice_j,
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


def test_o4_c_orca2_extensive_conservation_and_residual_budget_closure() -> None:
    """Every extensive family (`v_i, v_s, a_i, oa_i, e_s, e_i, szv_i, sv_i, a_ip, v_ip, v_il`) is conserved to `<= 1e-14` relative error on ORCA2 across `x -> y` and `y -> x` sweeps and `icycle in {1, 2, 3}` with active cyclic-seam flow and ice at and next to the north fold (`jj = 148, 149`), residual budgets close when clipping is active, and Plant O4-C1 fails."""
    cfg_sal4 = SI3ORCA2StateConfig.for_orca2_sas_ice(ln_icedyn=True, nn_icesal=4)
    state_in, grid = _build_closed_orca2_conservation_state(cfg_sal4)
    sss_m = jnp.full((184, 152), 34.2, dtype=jnp.float64)
    sst_m = jnp.full((184, 152), -1.8, dtype=jnp.float64)

    # Confirm both cyclic east-west seam transport (`ji = 2, 181`) AND ice + velocity
    # AT (`jj = 149`) and NEXT TO (`jj = 148`) the north fold are genuinely active!
    assert float(jnp.max(jnp.abs(state_in.u_ice[2, 2:150]))) > 0.1
    assert float(jnp.max(jnp.abs(state_in.v_i[2, 2:150]))) > 0.01
    assert float(jnp.max(jnp.abs(state_in.v_i[2:182, 148, :]))) > 0.05  # next to fold
    assert float(jnp.max(jnp.abs(state_in.v_i[2:182, 149, :]))) > 0.05  # at fold
    assert float(jnp.max(jnp.abs(state_in.u_ice[2:182, 148]))) > 0.1  # u next to fold
    assert float(jnp.max(jnp.abs(state_in.u_ice[2:182, 149]))) > 0.1  # u at fold
    assert float(jnp.max(jnp.abs(state_in.v_ice[2:182, 147]))) > 0.1  # v into row 148

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
                state_in._replace(
                    cflice_delayed=jnp.asarray(_CFL_FOR_ICYCLE[subcycles], dtype=jnp.float64)
                ),
                grid,
                kt=kt,
                sss_m=sss_m,
                sst_m=sst_m,
                config=cfg_sal4,
                **_dyn_inputs(),
            )
            assert int(diag.icycle) == subcycles
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
    state_sal2, _ = si3_dyn_orca2(
        state_in._replace(cflice_delayed=jnp.asarray(_CFL_FOR_ICYCLE[2], dtype=jnp.float64)),
        grid,
        kt=1,
        sss_m=sss_m,
        sst_m=sst_m,
        config=cfg_sal2,
        **_dyn_inputs(),
    )
    sv_after = _domain_integral(state_sal2.sv_i, grid.e1e2t)
    assert abs(sv_after - sv_init) / sv_init <= 1.0e-14

    # Residual budget closure when `Hbig_pra` / `Hsnow_pra` / `ice_var_zapneg` actively clip:
    state_clip_raw, _, _, _ = o1_gate.build_synthetic_orca2_o1_state(cfg_sal4)
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
    _, diag_clip = si3_dyn_orca2(
        state_clip._replace(cflice_delayed=jnp.asarray(_CFL_FOR_ICYCLE[1], dtype=jnp.float64)),
        grid,
        kt=1,
        sss_m=sss_m,
        sst_m=sst_m,
        config=cfg_sal4,
        **_dyn_inputs(),
    )
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

    # Plant O4-C1: corrupting receiver flux (`_plant_corrupt_receiver_flux=True`) MUST break conservation
    state_corrupt, _ = si3_dyn_orca2(
        state_in._replace(cflice_delayed=jnp.asarray(_CFL_FOR_ICYCLE[1], dtype=jnp.float64)),
        grid,
        kt=1,
        sss_m=sss_m,
        sst_m=sst_m,
        config=cfg_sal4,
        **_dyn_inputs(),
        _plant_corrupt_receiver_flux=True,
    )
    vi_corrupt = _domain_integral(state_corrupt.v_i, grid.e1e2t)
    rel_err_corrupt = abs(vi_corrupt - initial_totals["v_i"]) / initial_totals["v_i"]
    assert rel_err_corrupt > 1.0e-8, (
        f"STATUS PLANT-FIRED expected conservation failure, got {rel_err_corrupt:.3e}"
    )


# ==============================================================================
# Test (d): End-to-end differentiability (`jax.jit`, `jax.value_and_grad`,
# `jax.jvp`, and central finite differences) through one ORCA2 Prather advection
# step with `icycle` taken from carried delayed-CFL state (`Plant O4-D1`).
# ==============================================================================


def test_o4_d_jit_primal_zero_ulp_and_forward_equals_reverse_ad() -> None:
    """`si3_o4_step` jits and differentiates with `icycle` from carried delayed-CFL state, is 0-ULP invariant under `jax.jit`, produces finite non-zero gradients with respect to extensives, velocities (`u_ice, v_ice`), and Prather moments (`sx, sy, sxx, syy, sxy`), satisfies `reverse == forward` (`<= 1e-11`) and central finite-difference ratio `~ 1` at `float64`, and fails under Plant O4-D1."""
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(ln_icedyn=True)
    state_in, grid = _build_closed_orca2_conservation_state(cfg)
    sss_m = jnp.full((184, 152), 34.2, dtype=jnp.float64)
    sst_m = jnp.full((184, 152), -1.8, dtype=jnp.float64)

    # `icycle in {1, 2, 3}` is selected from traced carried state (`cflice_delayed`).
    def _step_fn(
        st: SI3State,
        g: SI3ORCA2AdvectionGrid,
        sss: jnp.ndarray,
        sst: jnp.ndarray,
    ) -> tuple[SI3State, SI3AdvectionDiagnostics]:
        return si3_o4_step(
            st, g, kt=1, sss_m=sss, sst_m=sst, config=cfg,
            rn_uice=None, rn_vice=None, apply_halos=False,
        )

    eager_state, eager_diag = _step_fn(state_in, grid, sss_m, sst_m)
    jit_state, jit_diag = jax.jit(_step_fn)(state_in, grid, sss_m, sst_m)

    # Verify `t_si` halo contract (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/ice.f90:465`, `icestp.f90:526-540`):
    # `t_si` is allocated on `A2D(0)` (`Nis0:Nie0, Njs0:Nje0`) in NEMO with no halo cells,
    # so padded `(184, 152, jpl)` storage is zero outside `2:182, 2:150`.
    assert float(jnp.max(jnp.abs(eager_state.t_si[:2, :, :]))) == 0.0
    assert float(jnp.max(jnp.abs(eager_state.t_si[182:, :, :]))) == 0.0
    assert float(jnp.max(jnp.abs(eager_state.t_si[:, :2, :]))) == 0.0
    assert float(jnp.max(jnp.abs(eager_state.t_si[:, 150:, :]))) == 0.0
    assert float(jnp.min(eager_state.t_si[2:182, 2:150, :])) > 200.0

    # `ice_stp` repeats `ice_var_glo2eqv(2)` (`icestp.f90:226,254`); with no thermodynamics
    # in between, a second call on the step output must change nothing.
    eqv_again = si3_var_glo2eqv(
        v_i=eager_state.v_i,
        v_s=eager_state.v_s,
        sv_i=eager_state.sv_i,
        a_i=eager_state.a_i,
        oa_i=eager_state.oa_i,
        a_ip=eager_state.a_ip,
        v_ip=eager_state.v_ip,
        v_il=eager_state.v_il,
        e_s=eager_state.e_s,
        e_i=eager_state.e_i,
        szv_i=eager_state.szv_i,
        kn=2,
        hi_max_top=eager_state.bounds.hi_max[-1],
        sss_m=sss_m,
        config=cfg,
    )
    for e_leaf, again_leaf in zip(
        jax.tree_util.tree_leaves(eager_state.equivalent),
        jax.tree_util.tree_leaves(eqv_again),
    ):
        _assert_zero_ulp(e_leaf, again_leaf, "glo2eqv_idempotent")

    # Verify `cflice_delayed` is carried on `SI3State` and serialized as `DELAY__r8_cflice`
    assert eager_state.cflice_delayed is not None
    _assert_zero_ulp(eager_state.cflice_delayed, eager_diag.cflice_delayed, "cflice_delayed_state_diag")
    rst_dict = si3_rst_to_netcdf_dict(eager_state, config=cfg)
    assert "DELAY__r8_cflice" in rst_dict
    reloaded_state = si3_rst_from_netcdf_dict(rst_dict, grid.tmask, config=cfg)
    _assert_zero_ulp(reloaded_state.cflice_delayed, eager_state.cflice_delayed, "cflice_delayed_rst_roundtrip")

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

    # 1b. A traced delayed `cflice` selects each `icycle` inside one compiled step.
    step_jit = jax.jit(_step_fn)
    for want in (1, 2, 3):
        st_d = state_in._replace(
            cflice_delayed=jnp.asarray(_CFL_FOR_ICYCLE[want], dtype=jnp.float64)
        )
        e_st, e_dg = _step_fn(st_d, grid, sss_m, sst_m)
        j_st, j_dg = step_jit(st_d, grid, sss_m, sst_m)
        assert int(e_dg.icycle) == want and int(j_dg.icycle) == want
        _assert_zero_ulp(e_st.v_i, j_st.v_i, f"jit_delayed_icycle{want}.v_i")
        _assert_zero_ulp(e_st.prather.ice.sx, j_st.prather.ice.sx, f"jit_delayed_icycle{want}.sx")

    # 2. Reverse-mode (`jax.value_and_grad`) and forward-mode (`jax.jvp`) AD through `si3_o4_step`
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
        out_st, _ = si3_o4_step(
            st,
            grid,
            kt=1,
            sss_m=sss_m,
            sst_m=sst_m,
            config=cfg,
            rn_uice=None,
            rn_vice=None,
            apply_halos=False,
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
            + 1.0e-3 * jnp.sum(w_int * out_st.snwice_fmass)
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

    # 3. Central finite-difference check at `float64` (`ratio ~ 1`):
    #    Perturb `u_ice` and `v_i` along smooth interior directions with step `eps = 1e-6`
    #    and verify `fd_deriv / ad_deriv` matches `1.0` to `<= 1e-7`.
    eps_fd = 1.0e-6
    dir_u = interior * jnp.asarray(np.cos(np.arange(184)[:, None] * 0.07), dtype=jnp.float64)
    ad_u_deriv = float(jnp.sum(grads[10] * dir_u))
    assert abs(ad_u_deriv) > 1.0e-3
    primals_u_plus = primals[:10] + (primals[10] + eps_fd * dir_u,) + primals[11:]
    primals_u_minus = primals[:10] + (primals[10] - eps_fd * dir_u,) + primals[11:]
    fd_u_deriv = float(_loss(primals_u_plus) - _loss(primals_u_minus)) / (2.0 * eps_fd)
    fd_u_ratio = fd_u_deriv / ad_u_deriv
    assert abs(fd_u_ratio - 1.0) <= 1.0e-7, (
        f"u_ice central FD ratio = {fd_u_ratio:.12f} (ad={ad_u_deriv}, fd={fd_u_deriv})"
    )

    dir_vi = interior[:, :, None] * jnp.ones_like(primals[0])
    ad_vi_deriv = float(jnp.sum(grads[0] * dir_vi))
    assert abs(ad_vi_deriv) > 1.0
    primals_vi_plus = (primals[0] + eps_fd * dir_vi,) + primals[1:]
    primals_vi_minus = (primals[0] - eps_fd * dir_vi,) + primals[1:]
    fd_vi_deriv = float(_loss(primals_vi_plus) - _loss(primals_vi_minus)) / (2.0 * eps_fd)
    fd_vi_ratio = fd_vi_deriv / ad_vi_deriv
    assert abs(fd_vi_ratio - 1.0) <= 1.0e-7, (
        f"v_i central FD ratio = {fd_vi_ratio:.12f} (ad={ad_vi_deriv}, fd={fd_vi_deriv})"
    )

    # Plant O4-D1: `_plant_stop_gradient=True` zeros all gradients and MUST fail the AD check
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
