#!/usr/bin/env python3
"""Committed probe for the S-19 ``wzv`` arm on the certified L1 NEMO cards.

Two sections, both fp64, both run per invocation:

SCALE -- the bound the two arms can possibly differ by.  NEMO's qco ``wzv``
(``sshwzv.F90:331-336``) subtracts ``r1_Dt*e3t_0(k)*(r3t(Kaa)-r3t(Kbb))``;
legoESM's generic ``diagnose_w_from_flux_div`` subtracts ``sigma(k)*deta_dt``
with ``deta_dt`` from the COLUMN INTEGRAL OF THE SAME STAGE TRANSPORT.  The
per-level weights are the same reference-thickness fraction on these cards, so

    max |w_NEMO - w_generic|  <=  max | (ssh(Kaa)-ssh(Kbb))/dt + column_div |

NONVACUITY -- an arm that moves nothing is evidence only if it RAN.  Counts
the branch entries, prints the stage-by-stage ``w`` difference, and plants a
+1e-6 scaling inside the literal branch which MUST move the stepped state.

Usage: nemo_testcase_wzv_arm_probe.py [LOCK_EXCHANGE-zco|OVERFLOW-zps]
"""
from __future__ import annotations

import sys

import numpy as np

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

set_policy(PrecisionPolicy.fp64())
assert get_policy() == PrecisionPolicy.fp64()

from legoesm.ocean.dynamics import ocean_model_latlon_cgrid as omlc  # noqa: E402
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (  # noqa: E402
    LatLonCGridOceanModel,
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (  # noqa: E402
    build_nemo_testcase_card,
)

CASE = sys.argv[1] if len(sys.argv) > 1 else "OVERFLOW-zps"
card = build_nemo_testcase_card(CASE)
state = card.recipe.initial_state
dt = float(card.dt_s)


def build(**hooks):
    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks))


def section_scale() -> None:
    captured = []
    real_diag = omlc.diagnose_w_from_flux_div

    def spy(flux_div_k, z_coord=None, thickness_weighted=False):
        out = real_diag(flux_div_k, z_coord, thickness_weighted)
        captured.append((np.asarray(flux_div_k), np.asarray(out)))
        return out

    omlc.diagnose_w_from_flux_div = spy
    try:
        new = build()._step_impl(state, dt)
    finally:
        omlc.diagnose_w_from_flux_div = real_diag

    eta_b = np.asarray(state.eta.data)
    eta_a = np.asarray(new.eta.data)
    wet = np.asarray(state.land_mask.data) > 0.5
    print(f"[SCALE] case={CASE} dt={dt} policy=fp64")
    print(f"  dtypes: eta={eta_b.dtype} h_partial="
          f"{np.asarray(card.recipe.z_coord.h_partial).dtype} "
          f"stage_div={captured[0][0].dtype}")
    deta_baro = (eta_a - eta_b) / dt
    for i, (fd, w) in enumerate(captured):
        if fd.ndim != 3:
            continue
        deta_div = -fd.sum(axis=-1)
        resid = float(np.max(np.abs(deta_baro - deta_div)[wet]))
        wmax = float(np.max(np.abs(w)))
        print(f"  capture[{i}] max|w|={wmax:.6e} m/s  "
              f"BOUND max|dw| <= {resid:.6e} m/s  "
              f"ratio={resid / max(wmax, 1e-300):.3e}")


def section_nonvacuity() -> None:
    calls = {"literal": 0, "generic": 0}
    w_literal: list = []
    w_generic: list = []
    real_stage = omlc._nemo_ws_stage_transport

    def spy(*a, **kw):
        out = real_stage(*a, **kw)
        tag = "literal" if kw.get("literal_wzv") else "generic"
        calls[tag] += 1
        (w_literal if tag == "literal" else w_generic).append(
            np.asarray(out[2]))
        return out

    omlc._nemo_ws_stage_transport = spy
    try:
        s_gen = build()._step_impl(state, dt)
        s_lit = build(literal_stage_wzv=True)._step_impl(state, dt)
    finally:
        omlc._nemo_ws_stage_transport = real_stage

    print(f"[NONVACUITY] branch entries {calls}")
    assert calls["literal"] > 0, "literal branch NEVER entered -- arm vacuous"
    for i, (a, b) in enumerate(zip(w_generic, w_literal, strict=True)):
        d = float(np.max(np.abs(a - b)))
        m = float(np.max(np.abs(a)))
        print(f"  stage[{i}] max|w_gen|={m:.6e}  max|w_lit-w_gen|={d:.6e}"
              f"  rel={d / max(m, 1e-300):.3e}"
              f"  bit_identical={np.array_equal(a, b)}")
    for name in ("T", "u", "eta"):
        a = np.asarray(getattr(s_gen, name).data)
        b = np.asarray(getattr(s_lit, name).data)
        print(f"  kt2 {name}: max|delta|={np.max(np.abs(a - b)):.6e} "
              f"bit_identical={np.array_equal(a, b)}")

    real_wzv = omlc.nemo_qco_wzv_operands

    def planted(*a, **kw):
        ww, hu, hv = real_wzv(*a, **kw)
        return ww * 1.000001, hu, hv

    omlc.nemo_qco_wzv_operands = planted
    try:
        s_bad = build(literal_stage_wzv=True)._step_impl(state, dt)
    finally:
        omlc.nemo_qco_wzv_operands = real_wzv
    assert omlc.nemo_qco_wzv_operands is real_wzv
    moved = False
    for name in ("T", "u", "eta"):
        a = np.asarray(getattr(s_lit, name).data)
        b = np.asarray(getattr(s_bad, name).data)
        delta = float(np.max(np.abs(a - b)))
        moved = moved or delta > 0.0
        print(f"  PLANTED(+1e-6 on literal w) kt2 {name}: "
              f"max|delta|={delta:.6e}")
    assert moved, "planted violation did not move the state -- gate is blind"


if __name__ == "__main__":
    section_scale()
    section_nonvacuity()
