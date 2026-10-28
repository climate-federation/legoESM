#!/usr/bin/env python3
"""Write a small schema-faithful O4 fixture derived from admitted NEMO bytes."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ice.si3_orca2_state import (
    SI3AllPratherMoments, SI3ORCA2AdvectionGrid, SI3ORCA2StateConfig,
    si3_dyn_adv_pra_orca2,
)
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import (
    COMPONENTS, FAMILIES, GRID, KINDS, MAGIC, STATE, _flatten_moments,
)


def _write(path: Path, kind: int, step: int, fields: dict[str, np.ndarray], *,
           scalars: tuple[float, float, float, float], icycle: int) -> None:
    jpi, jpj, jpl, nlay_i, nlay_s, h = 12, 10, 2, 2, 1, 2
    meta = np.asarray((kind, step, 0, jpi, jpj, jpl, nlay_i, nlay_s, h,
                       1 + h, jpi - h, 1 + h, jpj - h, len(KINDS[kind]),
                       icycle if kind == 4 else 0, (step - 1) % 2), dtype="<i4")
    with path.open("wb") as stream:
        stream.write(MAGIC); stream.write(np.asarray([1], dtype="<i4").tobytes())
        stream.write(meta.tobytes()); stream.write(np.asarray(scalars, dtype="<f8").tobytes())
        for name in KINDS[kind]:
            stream.write(np.asfortranarray(fields[name], dtype="<f8").tobytes(order="F"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-real", required=True, type=Path)
    args = parser.parse_args()
    source = args.source_real.read_bytes()
    seed = int.from_bytes(hashlib.sha256(source).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    set_policy(PrecisionPolicy.fp64()); jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    jpi, jpj, jpl, nlay_i, nlay_s, h = 12, 10, 2, 2, 1, 2
    one = np.ones((jpi, jpj)); e1u = np.full_like(one, 10000.0); e2v = np.full_like(one, 10000.0)
    e2u = np.full_like(one, 9500.0); e1v = np.full_like(one, 10500.0)
    area = 1.0e8 + rng.uniform(-1.0e6, 1.0e6, one.shape)
    u = np.full_like(one, 0.5); v = np.full_like(one, 0.5)
    grid_fields = dict(zip(GRID, (e1u, e2v, 1/e1u, 1/e2v, e2u, e1v, area, 1/area,
                                  one, one, one, u, v), strict=True))
    grid = SI3ORCA2AdvectionGrid(*(jnp.asarray(grid_fields[name]) for name in GRID[:11]))
    a_i = rng.uniform(0.05, 0.2, (jpi, jpj, jpl)); v_i = a_i * rng.uniform(0.5, 1.2, a_i.shape)
    v_s = a_i * rng.uniform(0.02, 0.08, a_i.shape); a_ip = a_i * 0.1
    state = {
        "ato_i": 1.0 - np.sum(a_i, axis=-1), "h_i": v_i/a_i, "h_s": v_s/a_i,
        "h_ip": np.full_like(a_i, 0.05), "v_i": v_i, "v_s": v_s, "sv_i": v_i*4.0,
        "oa_i": a_i*2.0, "a_i": a_i, "a_ip": a_ip, "v_ip": a_ip*0.05,
        "v_il": a_ip*0.01, "e_s": rng.uniform(1e6, 2e6, (jpi,jpj,nlay_s,jpl)),
        "e_i": rng.uniform(1e7, 2e7, (jpi,jpj,nlay_i,jpl)),
        "szv_i": rng.uniform(0.1, 0.2, (jpi,jpj,nlay_i,jpl)),
    }
    for name in STATE[15:]: state[name] = np.zeros((jpi-2*h, jpj-2*h))
    moments = SI3AllPratherMoments.zeros(jpi, jpj, jpl, nlay_i, nlay_s)
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(jpl=jpl,nlay_i=nlay_i,nlay_s=nlay_s,
        nn_hls=h,nn_icesal=4,ln_icedyn=True,ln_icethd=False,ln_dynadv_pra=True,rDt_ice=10800.0)
    delayed = np.float64(0.54)
    for step in (1, 2):
        entry = {name: np.asarray(state[name]) for name in STATE}
        moment_entry = _flatten_moments(moments)
        zfull = {name: jnp.pad(jnp.asarray(entry[name]), ((h,h),(h,h))) for name in STATE[15:]}
        out = si3_dyn_adv_pra_orca2(step,jnp.asarray(u),jnp.asarray(v),jnp.asarray(entry["h_i"]),
            jnp.asarray(entry["h_s"]),jnp.asarray(entry["h_ip"]),jnp.asarray(entry["ato_i"]),
            jnp.asarray(entry["v_i"]),jnp.asarray(entry["v_s"]),jnp.asarray(entry["sv_i"]),
            jnp.asarray(entry["oa_i"]),jnp.asarray(entry["a_i"]),jnp.asarray(entry["a_ip"]),
            jnp.asarray(entry["v_ip"]),jnp.asarray(entry["v_il"]),jnp.asarray(entry["e_s"]),
            jnp.asarray(entry["e_i"]),jnp.asarray(entry["szv_i"]),moments,grid,config=cfg,
            hi_max_top=99.0,cflice_delayed=jnp.asarray(delayed),**zfull)
        state_names = STATE[:1] + STATE[4:15]
        for name, value in zip(state_names, out[:12], strict=True): state[name] = np.asarray(value)
        for name, value in zip(STATE[15:], out[13:19], strict=True): state[name] = np.asarray(value)[h:-h,h:-h]
        moments = out[12]; current = float(np.asarray(out[20])); used = float(np.asarray(out[19])); cycle = int(np.asarray(out[21]))
        scalars = (10800.0, 99.0, current, used)
        _write(args.output/f"o4_kind01_step{step:04d}_rank0000.bin",1,step,grid_fields,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind02_step{step:04d}_rank0000.bin",2,step,entry,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind03_step{step:04d}_rank0000.bin",3,step,moment_entry,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind04_step{step:04d}_rank0000.bin",4,step,{},scalars=scalars,icycle=cycle)
        _write(args.output/f"o4_kind05_step{step:04d}_rank0000.bin",5,step,state,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind06_step{step:04d}_rank0000.bin",6,step,_flatten_moments(moments),scalars=scalars,icycle=0)
        delayed = np.float64(current)


if __name__ == "__main__": main()
