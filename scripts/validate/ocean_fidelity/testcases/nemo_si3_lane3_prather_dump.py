"""Dump lane-3 SI3 Prather outputs for a parent-vs-current bit-identity control.

Run once with a parent tree first on PYTHONPATH and once with the current tree;
the caller compares every array at 0 ULP.  Covers the card step functions
(ICE_ADV1D, ICE_ADV2D both correction orders), the shared Hbig/Hsnow/zapneg
correction ledger on inputs that make every clip fire (extensive and intensive
calling conventions), the two sweep kernels, and the production `step_sea_ice`.

Usage: nemo_si3_lane3_prather_dump.py OUT.npz [--plant-multiply-order]
"""

from __future__ import annotations

import sys

import jax.numpy as jnp
import legoesm.ice.fidelity.nemo_adv2d_testcase_recipe as adv2d
import legoesm.ice.scm as scm
import numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice import BrineConfig, SeaIceConfig, SnowConfig, init_dynamic_ice_state, step_sea_ice
from legoesm.ice.fidelity.nemo_testcase_recipe import build_ice_adv1d_card, step_ice_adv1d_card
from legoesm.ice.transport import advect_si3_prather_1d, advect_si3_prather_2d

from legoesm import constants

_ADV2D_STEPS = 4
_ADV1D_STEPS = 3


def _corrections_inputs(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Intensive ICE_ADV2D entry/transported fields where every correction branch fires."""
    names = adv2d.ICE_ADV2D_TRACERS
    shape = (14, 13)
    entry = {n: rng.uniform(0.5, 1.5, size=shape) for n in names}
    entry["a_i"] = rng.uniform(0.02, 0.12, size=shape)
    entry["v_i"] = entry["a_i"] * rng.uniform(0.5, 2.0, size=shape)
    entry["v_s"] = entry["a_i"] * rng.uniform(0.05, 0.2, size=shape)
    entry["a_ip"] = entry["a_i"] * 0.3
    entry["v_ip"] = entry["a_ip"] * 0.1
    entry["v_il"] = entry["a_ip"] * 0.01
    for n in names:
        if n.startswith("e_") or n.startswith("szv_"):
            base = entry["v_s"] if n.startswith("e_s") else entry["v_i"]
            entry[n] = base * rng.uniform(1.0e8, 3.0e8, size=shape)
    moved = {n: v.copy() for n, v in entry.items()}
    for n in names:
        moved[n] = moved[n] * rng.uniform(1.05, 1.9, size=shape)
    moved["a_i"] = entry["a_i"] * rng.uniform(0.3, 0.8, size=shape)
    moved["a_ip"] = np.minimum(moved["a_ip"], 0.1)
    moved["v_s"] = moved["v_i"] * rng.uniform(0.2, 0.9, size=shape)
    moved["oa_i"][3, 4] = -0.01
    moved["e_i_l01"][5, 6] = -1.0e6
    moved["v_ip"][7, 2] = -1.0e-4
    stack = lambda d: np.stack([d[n] for n in names], axis=-1)  # noqa: E731
    return stack(entry), stack(moved)


def main(argv: list[str]) -> None:
    out_path = argv[1]
    plant = "--plant-multiply-order" in argv[2:]
    set_policy(PrecisionPolicy.fp64())
    if plant:
        real_hbig = adv2d.si3_hbig_pra

        def _planted_hbig(*args, **kwargs):
            kwargs["multiply_before_divide"] = not kwargs["multiply_before_divide"]
            return real_hbig(*args, **kwargs)

        adv2d.si3_hbig_pra = _planted_hbig

    out: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(2026102404)

    shape = (10, 9, 4)
    contents = jnp.asarray(rng.uniform(0.1, 2.5, size=shape) * 1.0e8)
    moments = tuple(jnp.asarray(rng.uniform(-0.2, 0.2, size=shape) * 1.0e8) for _ in range(5))
    u_ice = jnp.asarray(rng.uniform(-0.35, 0.45, size=shape[:2]))
    v_ice = jnp.asarray(rng.uniform(-0.30, 0.40, size=shape[:2]))
    area = jnp.asarray(3.5e7 + rng.uniform(-2.0e6, 2.0e6, size=shape[:2]))
    wet = jnp.ones(shape[:2], dtype=bool)
    c1, m1, _ = advect_si3_prather_1d(
        contents, moments, u_ice, jnp.zeros_like(v_ice), area, wet, 1200.0,
        dx=6000.0, dy=6000.0, ice_volume_index=0, concentration_index=1,
    )
    out["adv1d.contents"] = np.asarray(c1)
    for k, m in enumerate(m1):
        out[f"adv1d.m{k}"] = np.asarray(m)
    for step in (1, 2):
        c2, m2, _ = advect_si3_prather_2d(
            contents, moments, u_ice, v_ice, area, wet, 1200.0,
            dx=6000.0, dy=6000.0, ice_step_index=step,
            ice_volume_index=0, concentration_index=1,
        )
        out[f"adv2d_k{step}.contents"] = np.asarray(c2)
        for k, m in enumerate(m2):
            out[f"adv2d_k{step}.m{k}"] = np.asarray(m)

    card1 = build_ice_adv1d_card(np.full((59, 59), -1.8, dtype=np.float64))
    st1 = card1.initial_state
    for step in range(_ADV1D_STEPS):
        st1 = step_ice_adv1d_card(card1, st1)
        out[f"card1d_s{step}.contents"] = np.asarray(st1.contents)
        for k, m in enumerate(st1.moments):
            out[f"card1d_s{step}.m{k}"] = np.asarray(m)

    card2 = adv2d.build_ice_adv2d_card(np.full((99, 99), -1.8, dtype=np.float64))
    for legacy in (False, True):
        st2 = card2.initial_state
        for step in range(_ADV2D_STEPS):
            st2 = adv2d.step_ice_adv2d_card(
                card2, st2, completed_steps=step, _use_legacy_extensive_correction_order=legacy
            )
            out[f"card2d_l{int(legacy)}_s{step}.contents"] = np.asarray(st2.contents)
            for k, m in enumerate(st2.moments):
                out[f"card2d_l{int(legacy)}_s{step}.m{k}"] = np.asarray(m)

    entry, moved = _corrections_inputs(rng)
    cell_area = 4.0e6
    for intensive in (True, False):
        scale = 1.0 if intensive else cell_area
        corrected = adv2d.apply_si3_prather_source_corrections(
            jnp.asarray(entry * scale),
            jnp.asarray(moved * scale),
            tracer_names=adv2d.ICE_ADV2D_TRACERS,
            nlay_i=3,
            nlay_s=3,
            cell_area_m2=cell_area,
            halo_width=2,
            entry_intensive_contents=jnp.asarray(entry) if intensive else None,
            contents_are_intensive=intensive,
        )
        tag = "int" if intensive else "ext"
        out[f"corr_{tag}"] = np.asarray(corrected)
        changed = np.asarray(corrected)[2:-2, 2:-2] != (moved * scale)[2:-2, 2:-2]
        out[f"corr_{tag}.changed_per_tracer"] = changed.sum(axis=(0, 1))

    cfg_v2 = SeaIceConfig(snow=SnowConfig(enabled=True), brine=BrineConfig(enabled=True))
    sst = jnp.full((2,), constants.T_freeze - 1.8, dtype=jnp.float64)
    zero = jnp.zeros((2,), dtype=jnp.float64)
    st_si, resp = step_sea_ice(
        init_dynamic_ice_state((2,)), scm.cold_polar_forcing(ncol=2), sst, zero, zero,
        cfg_v2, 1.0, 3600.0,
    )
    for name in ("h_ice", "h_snow", "T_ice", "S_ice", "concentration"):
        out[f"step_sea_ice.{name}"] = np.asarray(getattr(st_si, name).data)
    out["step_sea_ice.T_sfc"] = np.asarray(resp.T_sfc)

    out["_module_file"] = np.asarray(adv2d.__file__)
    np.savez(out_path, **out)


if __name__ == "__main__":
    main(sys.argv)
