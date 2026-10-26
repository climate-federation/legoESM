#!/usr/bin/env python3
"""Existing-dump row-1.3 PGF acceptance after the literal-metric fix."""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, os, subprocess
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
import jax, jax.numpy as jnp, numpy as np
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _nemo_literal_barotropic_pressure_gradient
from legoesm.ocean.experiments.dino import dino_config_for_recipe, dino_lat_lon_model_config
from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask, read_nemo_restart
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
import spg_substep_chain as inherited

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def metric(a,b,m,n):
    x,y=np.asarray(a)[m],np.asarray(b)[m]
    if x.size != n: raise SystemExit(f"population {x.size} != {n}")
    d=x-y; r=float(np.sqrt(np.mean(y*y)))
    e=float(np.sqrt(np.mean(d*d))/r); mx=float(np.max(np.abs(d))/r)
    return {"n":n,"normalized_rms_error":e,"max_error_over_nemo_rms":mx,
            "bit_mismatch_count":int(np.count_nonzero(x.view(np.uint64)!=y.view(np.uint64))),
            "gate_status":"AT BAR" if e<=1e-15 and mx<=1e-15 else "DEBT"}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--qco-run',type=Path,required=True); p.add_argument('--recurrence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    set_policy(PrecisionPolicy.fp64())
    root=Path(__file__).resolve().parents[4]; commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(): raise SystemExit('clean tree required')
    if jax.default_backend()!='cpu' or not jax.config.jax_enable_x64: raise SystemExit('cpu/fp64 required')
    rec=json.loads(a.recurrence.read_text()); primary=[x for x in rec['measurements'] if x['arm']=='forcing_only']
    seeds=[x for x in primary if x['subrow']=='1.2']
    if any(x['normalized_rms_error']!=0 for x in seeds): raise SystemExit('row 1.2 not exact')
    g=read_nemo_mesh_mask(str(a.run/'mesh_mask.nc'),nn_hls=0); s=read_nemo_restart(str(a.run/inherited.RESTART_FILE),nn_hls=0)
    cfg=dataclasses.replace(dino_config_for_recipe('nemo_dino_kamm_mlf'),lon_west_deg=1.,lon_east_deg=49.,sill_lon_m_deg=1.)
    br=bridge_nemo_to_legoesm_topo(
        g,s,periodic_i=True,full_step=True,omega=cfg.omega,
        coriolis_placement=cfg.coriolis_placement,carry_native_lat_deg=True)
    mc,_=dino_lat_lon_model_config(br.geometry,cfg)
    z=inherited._load_full(str(a.qco_run/'qco_dump_zsshp2_substep1.bin'),56,203,2)
    um=np.asarray(g.umask)[...,0]>0; vm=np.asarray(g.vmask)[...,0]>0
    pu,pv=_nemo_literal_barotropic_pressure_gradient(jnp.asarray(z),br.geometry,jnp.asarray(mc.g),br.state.u_mask.data,br.state.v_mask.data)
    # Independent executed-source transcription, native east/north faces.
    du=np.roll(z,-1,axis=1)-z; north=np.concatenate([z[1:],np.zeros_like(z[:1])],axis=0)
    nu=((-float(mc.g)*du)*(1.0/np.asarray(g.e1u)))*um
    nv=((-float(mc.g)*(north-z))*(1.0/np.asarray(g.e2v)))*vm
    rows={"u":metric(np.asarray(pu)[:,1:],nu,um,9758),"v":metric(np.asarray(pv)[1:],nv,vm,9868)}
    plant=nu.copy(); ij=tuple(np.argwhere(um)[0]); plant[ij] += 1.0e-6 * float(np.sqrt(np.mean(nu[um]**2)))
    control=metric(plant,nu,um,9758)['gate_status']=='DEBT'
    if not control: raise SystemExit('plant failed')
    out={"schema":"dino-split-explicit-momentum-chain-round21-v1","session_id":os.environ.get('CODEX_SESSION_ID'),
         "git_commit":commit,"backend":"cpu","jax_enable_x64":True,"rows":rows,
         "disposition":"PGF_AT_BAR_ADVANCE_TO_CORIOLIS" if all(x['gate_status']=='AT BAR' for x in rows.values()) else "PGF_DEBT",
         "controls":{"one_ulp_debt":control},"bindings":{"recurrence_sha256":sha(a.recurrence),"mesh_sha256":sha(a.run/'mesh_mask.nc'),
         "restart_sha256":sha(a.run/inherited.RESTART_FILE),"zsshp_sha256":sha(a.qco_run/'qco_dump_zsshp2_substep1.bin'),"script_sha256":sha(__file__)}}
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(out['disposition'],sha(a.output),rows)
    return 0
if __name__=='__main__': raise SystemExit(main())
