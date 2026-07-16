"""Clean 1D column BGC — legoESM NPZD, matching OceanBioME column example.

Key design decisions:
- 150m column, 30 uniform 5m layers
- No bottom restoring (closed N cycle via instant bottom remineralisation)  
- Implicit diffusion (scipy banded) — unconditionally stable
- No hard clipping of BGC state (let the model conserve mass naturally)
- dt = 20 min (well below growth timescale)

Usage
-----
cd /work/bd1083/b309178/diffESM/legoesm_ggn
JAX_ENABLE_X64=1 .venv/bin/python scripts/run_column_bgc.py --years 2
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from scipy.linalg import solve_banded
from legoesm.ocean.biogeochemistry import (
    BiogeoConfig, OceanBiogeoState, init_biogeo_state,
    step_ocean_biogeochemistry,
)

# ── Grid: 30 uniform 5m layers, 0–150m ───────────────────────────────────────
def make_grid(H=150.0, nlev=30):
    dz = np.full(nlev, H / nlev)
    zi = np.concatenate([[0.0], -np.cumsum(dz)])
    zf = 0.5 * (zi[:-1] + zi[1:])          # negative mid-points
    return zf.astype(np.float64), dz.astype(np.float64)

# ── Forcing: day 0 = Jan 1, Northern Hemisphere mid-latitude ─────────────────
def par_surf(t):
    doy = t % 365.25
    ang = 2*np.pi*(doy - 355)/365.25        # 0 at winter solstice
    return float(max(0.0, 200.0*0.5*(1 - np.cos(ang))))

def mld(t):
    doy = t % 365.25
    ang = 2*np.pi*(doy - 40)/365.25         # deep Feb, shallow Aug
    return float(10.0 + 110.0*0.5*(1 + np.cos(ang)))

def sst(t):
    doy = t % 365.25
    ang = 2*np.pi*(doy - 40)/365.25
    return float(8.0 + 12.0*0.5*(1 - np.cos(ang)))

def T_S(zf, sst_val, mld_val):
    T = np.full(len(zf), sst_val)
    S = np.full(len(zf), 35.0)
    for k in range(len(zf)):
        z = -zf[k]
        if z > mld_val:
            f = min((z - mld_val)/100.0, 1.0)
            T[k] = sst_val - (sst_val - 4.0)*f
    return T, S

# ── Implicit vertical diffusion ───────────────────────────────────────────────
def build_ab(dz, Kv, dt):
    n = len(dz)
    dzh = np.zeros(n+1)
    dzh[1:-1] = 0.5*(dz[:-1]+dz[1:])
    dzh[0] = dz[0]; dzh[-1] = dz[-1]
    a = np.zeros(n); b = np.ones(n); c = np.zeros(n)
    for k in range(n):
        if k > 0:
            r = dt*Kv[k]/(dz[k]*dzh[k]); a[k]=-r; b[k]+=r
        if k < n-1:
            r = dt*Kv[k+1]/(dz[k]*dzh[k+1]); c[k]=-r; b[k]+=r
    ab = np.zeros((3,n))
    ab[0,1:]=c[:-1]; ab[1,:]=b; ab[2,:-1]=a[1:]
    return ab

def diffuse(tr, ab):
    arr = np.where(np.isfinite(np.asarray(tr)), np.asarray(tr), 0.0)
    r = solve_banded((1,1), ab, arr.astype(np.float64), check_finite=False)
    return np.where(np.isfinite(r), r, arr)

def Kv_prof(zf, mld_val, Km=1e-2, Kd=1e-5):
    n = len(zf)
    Km_ = np.where(-zf < mld_val, Km, Kd)
    Ki = np.zeros(n+1)
    Ki[0]=Km_[0]; Ki[-1]=Kd
    Ki[1:-1]=0.5*(Km_[:-1]+Km_[1:])
    return Ki

# ── Main ──────────────────────────────────────────────────────────────────────
def run(n_years=2, dt_min=20.0, out="column_bgc.nc", save_h=24.0):
    print("="*55+"\nlegoESM 1D column BGC\n"+"="*55)

    nlev = 30
    zf, dz = make_grid(H=150.0, nlev=nlev)
    zj = jnp.array(zf); dzj = jnp.array(dz)

    print(f"Grid: {nlev} × {dz[0]:.1f} m layers, H=150 m")
    print("Forcing (day 0 = Jan 1):")
    for d,lab in [(0,"Jan"),(91,"Apr"),(172,"Jul"),(264,"Oct")]:
        print(f"  {lab}: PAR={par_surf(d):.0f} W/m²  "
              f"MLD={mld(d):.0f} m  SST={sst(d):.1f} °C")

    # BGC config — conservative, well-tested NPZD parameters
    cfg = BiogeoConfig(
        scheme="npzd",
        pCO2_atm=400.0, wind_speed=7.0,
        mu_max=1.5,   k_N=0.5e-3,  alpha_P=0.025,
        g_max=0.2,    k_P=0.8e-3,  gamma_Z=0.7,
        m_P=0.05,     m_Pq=0.05,
        m_Zl=0.2,     m_Z=0.6,
        remin_rate=0.05, w_sink=5.0,
        k_w_atten=0.05, k_chl_atten=20.0,
        # Initial conditions — nutrient-rich winter start
        NO3_init_surf=8e-3,  NO3_init_deep=15e-3,
        Phyto_init=0.01e-3,  Zoo_init=0.005e-3,
        Det_init=0.001e-3,
        DIC_init=2.10, ALK_init=2.35,
    )

    state = init_biogeo_state((nlev,), zj, cfg)
    mask  = jnp.ones((), dtype=jnp.float64)

    dt = dt_min*60.0
    n_steps = int(n_years*365.25*86400/dt)
    save_ev = max(1, int(save_h*3600/dt))
    print(f"\n{n_years} yr, dt={dt_min} min, {n_steps} steps")

    tdays=[]; NO3o=[]; Po=[]; Zo=[]; Do=[]; DICo=[]; ALKo=[]
    pco2o=[]; fco2o=[]; PARo=[]; MLDo=[]

    print("Compiling...")
    t0 = time.time()
    step_jit = jax.jit(step_ocean_biogeochemistry, static_argnames=("cfg",))
    T0,S0 = T_S(zf, sst(0), mld(0))
    _ = step_jit(state, jnp.array(T0), jnp.array(S0), dzj, zj,
                 mask, dt, cfg, PAR_surf=jnp.array(par_surf(0.0)))
    print(f"  Done {time.time()-t0:.1f}s\nRunning...")

    t0r = time.time(); prog = max(1, n_steps//20)

    for step in range(n_steps):
        td = step*dt/86400.0
        sv=sst(td); mv=mld(td); pv=par_surf(td)
        Tc,Sc = T_S(zf, sv, mv)

        state, diag = step_jit(
            state, jnp.array(Tc), jnp.array(Sc), dzj, zj,
            mask, dt, cfg, PAR_surf=jnp.array(pv),
        )

        # Vertical diffusion
        Kv = Kv_prof(zf, mv)
        ab = build_ab(dz, Kv, dt)
        NO3_np  = np.maximum(diffuse(state.NO3,   ab), 0.0)
        P_np    = np.maximum(diffuse(state.Phyto, ab), 0.0)
        Z_np    = np.maximum(diffuse(state.Zoo,   ab), 0.0)
        D_np    = np.maximum(diffuse(state.Det,   ab), 0.0)
        DIC_np  = diffuse(state.DIC, ab)
        ALK_np  = diffuse(state.ALK, ab)

        state = OceanBiogeoState(
            DIC=jnp.array(DIC_np), ALK=jnp.array(ALK_np),
            NO3=jnp.array(NO3_np), Phyto=jnp.array(P_np),
            Zoo=jnp.array(Z_np),   Det=jnp.array(D_np),
        )

        if step % save_ev == 0:
            tdays.append(td)
            NO3o.append(NO3_np.copy()); Po.append(P_np.copy())
            Zo.append(Z_np.copy());     Do.append(D_np.copy())
            DICo.append(DIC_np.copy()); ALKo.append(ALK_np.copy())
            pco2o.append(float(diag.pCO2_ocean))
            fco2o.append(float(diag.flux_co2))
            PARo.append(pv); MLDo.append(mv)

        if step % prog == 0 and step > 0:
            el=time.time()-t0r; eta=el/(step/n_steps)*(1-step/n_steps)
            print(f"  Yr {td/365.25:.2f}  "
                  f"P={float(state.Phyto[0])*1e3:.3f} mmol/m³  "
                  f"NO3={float(state.NO3[0])*1e3:.2f} mmol/m³  "
                  f"pCO2={float(diag.pCO2_ocean):.1f} uatm  ETA {eta:.0f}s")

    print(f"\nDone: {n_steps} steps in {time.time()-t0r:.1f}s")

    try:
        import netCDF4 as nc4
        ds = nc4.Dataset(out,"w")
        ds.createDimension("time",len(tdays))
        ds.createDimension("depth",nlev)
        tv=ds.createVariable("time","f8",("time",)); tv[:]=np.array(tdays); tv.units="days"
        zv=ds.createVariable("depth","f8",("depth",)); zv[:]=zf; zv.units="m"
        dv=ds.createVariable("dz","f8",("depth",)); dv[:]=dz
        for nm,dat,un in [("NO3",NO3o,"mol N m-3"),("Phyto",Po,"mol N m-3"),
                           ("Zoo",Zo,"mol N m-3"),("Det",Do,"mol N m-3"),
                           ("DIC",DICo,"mol C m-3"),("ALK",ALKo,"mol eq m-3")]:
            v=ds.createVariable(nm,"f4",("time","depth"),zlib=True)
            v[:]=np.array(dat); v.units=un
        for nm,dat,un in [("pCO2",pco2o,"uatm"),("flux_co2",fco2o,"mol C m-2 s-1"),
                           ("PAR_surf",PARo,"W m-2"),("MLD",MLDo,"m")]:
            v=ds.createVariable(nm,"f4",("time",)); v[:]=np.array(dat); v.units=un
        ds.description="legoESM 1D column NPZD — clean"
        ds.close()
        print(f"Written: {Path(out).resolve()}")
    except ImportError:
        np.savez(out.replace(".nc",".npz"),time=np.array(tdays),depth=zf,
                 NO3=np.array(NO3o),Phyto=np.array(Po),pCO2=np.array(pco2o))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--years",type=int,default=2)
    p.add_argument("--dt",type=float,default=20.0)
    p.add_argument("--out",type=str,default="column_bgc.nc")
    p.add_argument("--save-every",type=float,default=24.0)
    a=p.parse_args()
    run(a.years,a.dt,a.out,a.save_every)
