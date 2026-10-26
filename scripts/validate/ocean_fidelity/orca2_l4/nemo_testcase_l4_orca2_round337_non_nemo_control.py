#!/usr/bin/env python3
"""Produce or compare a parent/current census of pre-existing shared callers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.field import Field
from legoesm.ocean.coupler.omip2_applicator import compute_omip2_surface_forcing_jax
from legoesm.ocean.coupler.sss_apply import apply_sss_restoring_step
from legoesm.ocean.forcing.sss_restoring import (
    SSSRestoringConfig,
    compute_sss_restoring_flux,
)
from legoesm.ocean.freshwater import (
    FreshwaterForcing,
    freshwater_eta_tendency,
    net_freshwater_flux,
    normalize_freshwater_net,
    real_freshwater_dilution_tendencies,
    virtual_closure_temperature_twin,
    virtual_salt_flux,
)


class _State(NamedTuple):
    T: Field
    S: Field
    land_mask: Field


def _field(data, name):
    return Field(jnp.asarray(data), name=name, dims=(), units="")


def produce(path: Path) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    x = jnp.arange(12, dtype=jnp.float64).reshape(3, 4)
    salinity = 33.0 + x / 10.0
    target = 34.0 + x / 30.0
    latitude = jnp.broadcast_to(jnp.asarray([-70.0, 0.0, 75.0])[:, None], x.shape)
    longitude = jnp.broadcast_to(jnp.asarray([0.0, 90.0, 180.0, 270.0])[None, :], x.shape)
    ice = jnp.clip(x / 14.0, 0.0, 1.0)
    runoff = jnp.where(x > 5.0, 2.0e-6, 0.0)
    configs = (
        SSSRestoringConfig(
            enabled=True, regions=(), tau_restore_days_default=365.0,
            ice_gate=True, ice_gate_mode="tanh", normalization="s_target",
        ),
        SSSRestoringConfig(
            enabled=True, regions=(), tau_restore_days_default=45.5,
            ice_gate=True, ice_gate_mode="nemo_linear", normalization="live_s",
        ),
    )
    arrays: dict[str, np.ndarray] = {}
    for index, config in enumerate(configs):
        output = jax.jit(
            lambda s, r: compute_sss_restoring_flux(
                s, target, latitude, longitude, ice, config,
                river_runoff=r, sst_C=x - 2.0,
            )
        )(salinity, runoff)
        for name, value in output.items():
            if value is not None:
                arrays[f"sss/{index}/{name}"] = np.asarray(value)

    fw = FreshwaterForcing(
        precip=1.0e-5 + x * 1.0e-7,
        evap=7.0e-6 + x * 2.0e-7,
        runoff=runoff,
        ice_fw=(ice - 0.5) * 1.0e-6,
        restoring=(target - salinity) * 1.0e-7,
    )
    h = 5.0 + jnp.stack((x, x + 1.0, x + 2.0), axis=-1)
    tracer_s = jnp.stack((salinity, salinity + 0.2, salinity + 0.4), axis=-1)
    tracer_t = jnp.stack((x, x - 0.5, x - 1.0), axis=-1)
    mask = jnp.where(x == 0.0, 0.0, 1.0)
    fw_outputs = {
        "net_all": net_freshwater_flux(fw),
        "net_no_runoff": net_freshwater_flux(fw, include_runoff=False),
        "eta": freshwater_eta_tendency(fw, constants.rho_ocean),
        "virtual": virtual_salt_flux(fw, salinity, h[..., 0], constants.rho_ocean),
        "temperature_twin": virtual_closure_temperature_twin(
            fw, tracer_t[..., 0], h[..., 0], constants.rho_ocean, mask
        ),
        "normalized": normalize_freshwater_net(
            net_freshwater_flux(fw), 1.0 + x, mask
        ),
    }
    dilution = real_freshwater_dilution_tendencies(
        net_freshwater_flux(fw) / constants.rho_ocean,
        tracer_s,
        tracer_t,
        h,
        mask,
    )
    fw_outputs["dilution_s"], fw_outputs["dilution_t"] = dilution
    arrays.update({f"freshwater/{name}": np.asarray(value) for name, value in fw_outputs.items()})

    state = _State(
        T=_field(jnp.stack((x, x + 1.0), axis=-1), "T"),
        S=_field(jnp.stack((salinity, salinity + 0.1), axis=-1), "S"),
        land_mask=_field(mask, "land_mask"),
    )
    restored = apply_sss_restoring_step(
        state,
        S_target=target,
        ice_concentration=ice,
        config=configs[0],
        grid=SimpleNamespace(
            lat=np.radians(np.asarray([-70.0, 0.0, 75.0])),
            lon=np.radians(np.asarray([0.0, 90.0, 180.0, 270.0])),
        ),
        z_coord=SimpleNamespace(),
        dt=1800.0,
        lat2d_deg=np.asarray(latitude),
        lon2d_deg=np.asarray(longitude),
        river_runoff=np.asarray(runoff),
    )
    arrays["coupler/sss"] = np.asarray(restored.S.data)

    grid_shape = (2, 2)
    stack = {}
    for offset, name in enumerate(
        ("u10", "v10", "T_air", "q_air", "sw_down", "lw_down", "precip", "snow", "slp")
    ):
        base = jnp.arange(12, dtype=jnp.float64).reshape(1, 3, 4)
        if name == "T_air":
            base = base + 275.0
        elif name == "q_air":
            base = base * 1.0e-4 + 0.005
        elif name == "slp":
            base = base + constants.p_atm_std
        else:
            base = base + float(offset + 1)
        stack[name] = base
    scan_state = SimpleNamespace(T=_field(jnp.zeros(grid_shape + (1,)), "T"))
    surface = jax.jit(
        lambda: compute_omip2_surface_forcing_jax(
            scan_state,
            forcing_stack=stack,
            nn_i=jnp.asarray([0, 0, 1, 2]),
            nn_j=jnp.asarray([0, 2, 1, 3]),
            grid_shape=grid_shape,
            idx_t=jnp.asarray(0),
        )
    )()
    for name in ("tau_x", "tau_y", "q_net", "sw_down"):
        arrays[f"omip_scan/{name}"] = np.asarray(getattr(surface, name))

    np.savez(path, **arrays)
    return {"arrays": len(arrays), "words": sum(value.size for value in arrays.values())}


def compare(parent_path: Path, current_path: Path, plant: bool) -> dict[str, object]:
    rows = {}
    with np.load(parent_path, allow_pickle=False) as parent, np.load(
        current_path, allow_pickle=False
    ) as current:
        if set(parent.files) != set(current.files):
            raise ValueError("parent/current inventories differ")
        words = unequal = 0
        for key in sorted(parent.files):
            left = np.asarray(parent[key])
            right = np.asarray(current[key]).copy()
            if plant and key == "omip_scan/q_net":
                right.reshape(-1)[0] = np.nextafter(right.reshape(-1)[0], np.inf)
            count = int(np.count_nonzero(left.view(np.uint64) != right.view(np.uint64)))
            rows[key] = {"words": int(left.size), "unequal": count}
            words += left.size
            unequal += count
    if plant:
        if unequal != 1:
            raise ValueError(f"one-ULP plant changed {unequal} words")
        status = "PASS_PLANT_REJECTED"
    else:
        status = "PASS" if unequal == 0 else "FAIL"
    return {"status": status, "arrays": len(rows), "words": words, "unequal": unequal, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--produce", type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    if args.produce:
        report = {"status": "PRODUCED", **produce(args.produce)}
    elif args.parent and args.current:
        report = compare(args.parent, args.current, args.plant)
    else:
        parser.error("use --produce or --parent with --current")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.plant:
        return 1 if report["status"] == "PASS_PLANT_REJECTED" else 2
    return 0 if report["status"] in {"PRODUCED", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
