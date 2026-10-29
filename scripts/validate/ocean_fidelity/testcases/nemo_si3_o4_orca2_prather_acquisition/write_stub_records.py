#!/usr/bin/env python3
"""Write an independent, schema-faithful O4 zero-transport fixture.

This fixture deliberately does not import or call the production Prather path.
Its only oracle-derived input is the digest of a retained real NEMO log, used
to make the positive grid metrics non-constant.  Zero velocity and zero ice
inventories make the exact exit analytically equal to the entry for either
sweep order, while nonzero wet seam/fold masks keep the regional score live.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import (
    GRID, KINDS, MAGIC, MOMENTS, RESIDUALS, STATE, TRANSPORT_DIAGNOSTICS,
)


def _lbc_t_point(field: np.ndarray, sign: float, halo: int = 2) -> np.ndarray:
    """Independent scalar transcription of ORCA2 T-point seam/fold linking."""
    out = np.array(field, dtype=np.float64, copy=True)
    ni, nj = out.shape[:2]
    interior_i = ni - 2 * halo
    planes = out.reshape(ni, nj, -1)
    s = np.float64(sign)
    for plane in np.moveaxis(planes, 2, 0):
        for i in range(halo):
            plane[i, :] = plane[ni - 2 * halo + i, :]
            plane[ni - halo + i, :] = plane[halo + i, :]
        for j in range(1, halo + 1):
            j1, j2 = nj - j, nj - 2 * halo + j - 2
            for i in range(1, halo + 1):
                plane[i - 1, j1] = s * plane[2 * halo + 1 - i, j2]
            plane[halo, j1] = s * plane[halo, j2]
            for i in range(1, interior_i):
                plane[halo + i, j1] = s * plane[ni - halo - i, j2]
            plane[ni - halo, j1] = s * plane[halo, j2]
            for i in range(1, halo):
                plane[ni - halo + i, j1] = s * plane[ni - halo - i, j2]
        j1 = nj - halo - 1
        for i in range(1, interior_i // 2):
            plane[ni // 2 + i, j1] = s * plane[ni // 2 - i, j1]
        for i in range(1, halo + 1):
            plane[i - 1, j1] = s * plane[2 * halo + 1 - i, j1]
        plane[:, :halo] = 0.0
    return planes.reshape(out.shape)


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


def _write_nontrivial_score_fixture(output: Path, source: bytes) -> None:
    """Write a nontrivial hook-plant fixture; never used by stub NEMO."""
    import jax
    import jax.numpy as jnp
    from legoesm import constants as canonical_constants
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ice.si3_orca2_state import (
        SI3AllPratherMoments, SI3ORCA2AdvectionGrid, SI3ORCA2StateConfig,
        si3_dyn_adv_pra_orca2, si3_lbc_lnk_orca2, si3_o4_step, si3_var_zapsmall,
    )
    from scripts.validate.ocean_fidelity.testcases.nemo_si3_o4_orca2_prather_gate import (
        _build_entry_state, _flatten_moments, _state_and_diagnostics,
    )
    seed = int.from_bytes(hashlib.sha256(source).digest()[:8], "little")
    rng = np.random.default_rng(seed); set_policy(PrecisionPolicy.fp64()); jax.config.update("jax_enable_x64", True)
    output.mkdir(parents=True, exist_ok=True)
    jpi,jpj,jpl,nlay_i,nlay_s,h=12,10,2,2,1,2
    one=np.ones((jpi,jpj)); e1u=np.full_like(one,10000.0); e2v=np.full_like(one,10000.0)
    e2u=np.full_like(one,9500.0); e1v=np.full_like(one,10500.0)
    area=1.0e8+rng.uniform(-1.0e6,1.0e6,one.shape); u=np.full_like(one,0.5); v=np.full_like(one,0.5)
    sss=np.full_like(one,34.0); sst=np.full_like(one,-1.8)
    grid_fields=dict(zip(GRID,(e1u,e2v,1/e1u,1/e2v,e2u,e1v,area,1/area,one,one,one,u,v,sss,sst),strict=True))
    grid=SI3ORCA2AdvectionGrid(*(jnp.asarray(grid_fields[n]) for n in GRID[:11]))
    a_i=rng.uniform(0.05,0.2,(jpi,jpj,jpl)); v_i=a_i*rng.uniform(0.5,1.2,a_i.shape)
    v_s=a_i*rng.uniform(0.02,0.08,a_i.shape); a_ip=a_i*0.1
    state={"ato_i":1.0-np.sum(a_i,axis=-1),"h_i":v_i/a_i,"h_s":v_s/a_i,"h_ip":np.full_like(a_i,0.05),
           "h_il":np.full_like(a_i,0.01),
           "v_i":v_i,"v_s":v_s,"sv_i":v_i*4.0,"oa_i":a_i*2.0,"a_i":a_i,"a_ip":a_ip,
           "v_ip":a_ip*0.05,"v_il":a_ip*0.01,"t_su":np.full_like(a_i,260.0),
           "e_s":rng.uniform(1e6,2e6,(jpi,jpj,nlay_s,jpl)),
           "e_i":rng.uniform(1e7,2e7,(jpi,jpj,nlay_i,jpl)),"szv_i":rng.uniform(0.1,0.2,(jpi,jpj,nlay_i,jpl)),
           "sz_i":rng.uniform(2.0,4.0,(jpi,jpj,nlay_i,jpl)),
           "t_i":rng.uniform(250.0,265.0,(jpi,jpj,nlay_i,jpl)),
           "t_s":rng.uniform(250.0,265.0,(jpi,jpj,nlay_s,jpl)),
           "at_i":np.sum(a_i,axis=-1),"vt_i":np.sum(v_i,axis=-1),"snwice_mass":np.zeros((jpi,jpj))}
    for name in RESIDUALS+TRANSPORT_DIAGNOSTICS: state[name]=np.zeros((jpi-2*h,jpj-2*h))
    moments=SI3AllPratherMoments.zeros(jpi,jpj,jpl,nlay_i,nlay_s)
    cfg=SI3ORCA2StateConfig.for_orca2_sas_ice(jpl=jpl,nlay_i=nlay_i,nlay_s=nlay_s,nn_hls=h,nn_icesal=4,
        ln_icedyn=True,ln_icethd=False,ln_dynadv_pra=True,rDt_ice=10800.0)
    delayed=np.float64(0.54)
    for step in (1,2):
        entry={n:np.asarray(state[n]) for n in STATE}; moment_entry=_flatten_moments(moments)
        zfull={n:jnp.pad(jnp.asarray(entry[n]),((h,h),(h,h))) for n in RESIDUALS}
        out=si3_dyn_adv_pra_orca2(step,jnp.asarray(u),jnp.asarray(v),jnp.asarray(entry['h_i']),jnp.asarray(entry['h_s']),
            jnp.asarray(entry['h_ip']),jnp.asarray(entry['ato_i']),jnp.asarray(entry['v_i']),jnp.asarray(entry['v_s']),
            jnp.asarray(entry['sv_i']),jnp.asarray(entry['oa_i']),jnp.asarray(entry['a_i']),jnp.asarray(entry['a_ip']),
            jnp.asarray(entry['v_ip']),jnp.asarray(entry['v_il']),jnp.asarray(entry['e_s']),jnp.asarray(entry['e_i']),
            jnp.asarray(entry['szv_i']),moments,grid,config=cfg,hi_max_top=99.0,cflice_delayed=jnp.asarray(delayed),**zfull)
        adv={name:np.asarray(state[name]).copy() for name in STATE}
        for name,value in zip(("ato_i","v_i","v_s","sv_i","oa_i","a_i","a_ip","v_ip","v_il","e_s","e_i","szv_i"),out[:12],strict=True): adv[name]=np.asarray(value)
        for name,value in zip(RESIDUALS,out[13:19],strict=True): adv[name]=np.asarray(value)[h:-h,h:-h]
        moments=out[12]; current=float(np.asarray(out[20])); used=float(np.asarray(out[19])); cycle=int(np.asarray(out[21])); scalars=(10800.0,99.0,current,used)
        entry_state=_build_entry_state(entry,moment_entry,grid_fields,cfg,used)
        final_state,final_diag=si3_o4_step(entry_state,grid,kt=step,sss_m=jnp.asarray(sss),sst_m=jnp.asarray(sst),
            rn_uice=0.5,rn_vice=0.5,config=cfg,apply_halos=False)
        final=_state_and_diagnostics(final_state,final_diag)
        for name in RESIDUALS+TRANSPORT_DIAGNOSTICS: final[name]=np.asarray(final[name])[h:-h,h:-h]
        zap=si3_var_zapsmall(
            v_i=out[1],v_s=out[2],sv_i=out[3],oa_i=out[4],a_i=out[5],a_ip=out[6],
            v_ip=out[7],v_il=out[8],t_su=jnp.asarray(entry['t_su']),h_i=jnp.asarray(entry['h_i']),
            h_s=jnp.asarray(entry['h_s']),h_ip=jnp.asarray(entry['h_ip']),h_il=jnp.asarray(entry['h_il']),
            e_s=out[9],e_i=out[10],szv_i=out[11],sz_i=jnp.asarray(entry['sz_i']),
            t_i=jnp.asarray(entry['t_i']),t_s=jnp.asarray(entry['t_s']),at_i=jnp.asarray(entry['at_i']),
            vt_i=jnp.asarray(entry['vt_i']),ato_i=out[0],wfx_res=out[13],hfx_res=out[14],sfx_res=out[15],
            rDt_ice=10800.0,sss_m=jnp.asarray(sss),sst_m=jnp.asarray(sst),nn_hls=h,nn_icesal=4,
            rn_simin=cfg.rn_simin,rn_sinew=cfg.rn_sinew,rt0=cfg.ice_constants.T0,
            rhoi=cfg.ice_constants.rho_ice,rhos=cfg.ice_constants.rho_snow,rhow=canonical_constants.rho_water)
        stage7={name:np.asarray(adv[name]).copy() for name in STATE}
        for name,value in zip(("v_i","v_s","sv_i","oa_i","a_i","a_ip","v_ip","v_il","t_su",
                               "h_i","h_s","h_ip","h_il","e_s","e_i","szv_i","sz_i","t_i","t_s",
                               "at_i","vt_i","ato_i","wfx_res","hfx_res","sfx_res"),zap,strict=True):
            stage7[name]=np.asarray(value)[h:-h,h:-h] if name in RESIDUALS else np.asarray(value)
        for name in TRANSPORT_DIAGNOSTICS: stage7[name]=np.asarray(getattr(final_diag,name))[h:-h,h:-h]
        stage8={name:np.asarray(stage7[name]).copy() for name in STATE}
        for name in ("a_i","v_i","v_s","sv_i","oa_i","t_su","e_i","e_s","szv_i"):
            stage8[name]=np.asarray(si3_lbc_lnk_orca2(jnp.asarray(stage7[name]),"T",1.0,nn_hls=h,ldfull=True))
        _write(output/f"o4_kind01_step{step:04d}_rank0000.bin",1,step,grid_fields,scalars=scalars,icycle=0)
        _write(output/f"o4_kind02_step{step:04d}_rank0000.bin",2,step,entry,scalars=scalars,icycle=0)
        _write(output/f"o4_kind03_step{step:04d}_rank0000.bin",3,step,moment_entry,scalars=scalars,icycle=0)
        _write(output/f"o4_kind04_step{step:04d}_rank0000.bin",4,step,{},scalars=scalars,icycle=cycle)
        _write(output/f"o4_kind05_step{step:04d}_rank0000.bin",5,step,adv,scalars=scalars,icycle=0)
        _write(output/f"o4_kind06_step{step:04d}_rank0000.bin",6,step,_flatten_moments(moments),scalars=scalars,icycle=0)
        _write(output/f"o4_kind07_step{step:04d}_rank0000.bin",7,step,stage7,scalars=scalars,icycle=0)
        _write(output/f"o4_kind08_step{step:04d}_rank0000.bin",8,step,stage8,scalars=scalars,icycle=0)
        _write(output/f"o4_kind09_step{step:04d}_rank0000.bin",9,step,final,scalars=scalars,icycle=0)
        state=final
        delayed=np.float64(current)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-real", required=True, type=Path)
    parser.add_argument("--nontrivial-score-fixture", action="store_true")
    args = parser.parse_args()
    source = args.source_real.read_bytes()
    if args.nontrivial_score_fixture:
        _write_nontrivial_score_fixture(args.output, source)
        return
    seed = int.from_bytes(hashlib.sha256(source).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    args.output.mkdir(parents=True, exist_ok=True)
    jpi, jpj, jpl, nlay_i, nlay_s, h = 12, 10, 2, 2, 1, 2
    one = np.ones((jpi, jpj)); e1u = np.full_like(one, 10000.0); e2v = np.full_like(one, 10000.0)
    e2u = np.full_like(one, 9500.0); e1v = np.full_like(one, 10500.0)
    area = 1.0e8 + rng.uniform(-1.0e6, 1.0e6, one.shape)
    u = np.zeros_like(one); v = np.zeros_like(one)
    sss=np.zeros_like(one); sst=np.full_like(one,-273.15)
    grid_fields = dict(zip(GRID, (e1u, e2v, 1/e1u, 1/e2v, e2u, e1v, area, 1/area,
                                  one, one, one, u, v, sss, sst), strict=True))
    zero3 = np.zeros((jpi, jpj, jpl))
    state = {
        "ato_i": one.copy(), "h_i": one[..., None].repeat(jpl, axis=2),
        "h_s": zero3.copy(), "h_ip": zero3.copy(), "h_il":zero3.copy(), "v_i": zero3.copy(),
        "v_s": zero3.copy(), "sv_i": zero3.copy(), "oa_i": zero3.copy(),
        "a_i": zero3.copy(), "a_ip": zero3.copy(), "v_ip": zero3.copy(),
        "v_il": zero3.copy(), "t_su":zero3.copy(), "e_s": np.zeros((jpi,jpj,nlay_s,jpl)),
        "e_i": np.zeros((jpi,jpj,nlay_i,jpl)),
        "szv_i": np.zeros((jpi,jpj,nlay_i,jpl)),
        "sz_i":np.zeros((jpi,jpj,nlay_i,jpl)),"t_i":np.zeros((jpi,jpj,nlay_i,jpl)),
        "t_s":np.zeros((jpi,jpj,nlay_s,jpl)),"at_i":np.zeros((jpi,jpj)),
        "vt_i":np.zeros((jpi,jpj)),"snwice_mass":np.zeros((jpi,jpj)),
    }
    for name in RESIDUALS+TRANSPORT_DIAGNOSTICS: state[name] = np.zeros((jpi-2*h, jpj-2*h))
    moment_fields = {}
    for name in MOMENTS:
        family = name.split(".", 1)[0]
        shape = ((jpi,jpj,nlay_s,jpl) if family == "c0" else
                 (jpi,jpj,nlay_i,jpl) if family in {"e", "si"} else
                 (jpi,jpj,jpl))
        moment_fields[name] = np.zeros(shape)
    expected_state = {name: np.asarray(value).copy() for name, value in state.items()}
    expected_state["ato_i"] = _lbc_t_point(expected_state["ato_i"], 1.0, h)
    post_zap={name:np.asarray(expected_state[name]).copy() for name in STATE}
    post_zap["h_i"][h:-h,h:-h,:]=0.0
    post_zap["t_i"][h:-h,h:-h,:,:]=273.15
    post_zap["t_s"][h:-h,h:-h,:,:]=273.15
    post_lbc={name:np.asarray(post_zap[name]).copy() for name in STATE}
    final_state={name:np.asarray(post_lbc[name]).copy() for name in STATE}
    final_state["ato_i"][:]=1.0
    final_state["h_i"][:]=0.0
    final_state["t_i"][:]=273.15
    final_state["t_s"][:]=273.15
    expected_moments = {
        name: _lbc_t_point(value, -1.0 if name.rsplit(".", 1)[1] in {"sx", "sy"} else 1.0, h)
        for name, value in moment_fields.items()
    }
    for step in (1, 2):
        entry = {name: np.asarray(state[name]) for name in STATE}
        scalars = (10800.0, 99.0, 0.0, 0.0)
        _write(args.output/f"o4_kind01_step{step:04d}_rank0000.bin",1,step,grid_fields,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind02_step{step:04d}_rank0000.bin",2,step,entry,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind03_step{step:04d}_rank0000.bin",3,step,moment_fields,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind04_step{step:04d}_rank0000.bin",4,step,{},scalars=scalars,icycle=1)
        _write(args.output/f"o4_kind05_step{step:04d}_rank0000.bin",5,step,expected_state,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind06_step{step:04d}_rank0000.bin",6,step,expected_moments,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind07_step{step:04d}_rank0000.bin",7,step,post_zap,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind08_step{step:04d}_rank0000.bin",8,step,post_lbc,scalars=scalars,icycle=0)
        _write(args.output/f"o4_kind09_step{step:04d}_rank0000.bin",9,step,final_state,scalars=scalars,icycle=0)


if __name__ == "__main__": main()
