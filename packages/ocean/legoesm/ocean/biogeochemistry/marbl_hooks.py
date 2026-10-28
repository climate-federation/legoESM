"""MARBL on the forced OMIP drivers: the ``--biogeo marbl`` runner.

Same four entry points as :class:`omip_hooks.BGCRunner` (``before_step``,
``after_step``, ``save``, ``load``), so ``run_omip_core2.py`` needs no further
wiring; transport reuses :class:`transport.TracerTransporter` on the stacked
32-tracer state, and the restart sidecar is the same ``<leg>.bgc.npz``.

MARBL (``biogeochemistry/marbl``, CESM2.1 PFT set) works in **mmol/m^3**
(Chl mg/m^3, ALK meq/m^3); the inorganic initial fields in ``bgc_init.npz``
are mol/m^3, so they are scaled by 1e3 on load.  Everything inside this
runner — state, inventories, sidecar — stays in MARBL's units.

Forcing for the column step, from what the physics step already has in hand:
    sw_net   = sf.sw_down (shortwave entering the ocean, before ice)
    u10_sqr  = |tau| / (rho_a C_D)
    ifrac    = total sea-ice concentration from the prognostic ice state
    xco2     = --pco2-atm
    deposition (dust / Fe / NOx / NHy) = 0 unless --marbl-forcing is given.
Sea-level pressure is taken as 1 atm (the CORE-II slp is on the forcing
grid, not on the ocean grid in the driver's surface-forcing object).

Settling is inside MARBL's column step (its sinking-particle module resolves
the flux through the levels itself), so the transporter moves only the
dissolved and suspended tracers.
"""
from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.biogeochemistry.omip_hooks import BGCRunner, _to_centre, _RHO_A, _C_D
from legoesm.ocean.biogeochemistry.transport import TracerTransporter
from legoesm.ocean.biogeochemistry.marbl.config import TRACER_NAMES, MarblConfig, MarblTracers
from legoesm.ocean.biogeochemistry.marbl.driver3d import Marbl2DForcing, make_marbl_stepper

MOL_TO_MMOL = 1.0e3
SEC_PER_YEAR = 365.0 * 86400.0
FORCING_VARS = ("dust_flux", "iron_flux", "nox_flux", "nhy_flux", "ifrac")

# Uniform seeds [mmol/m^3] for pools that no climatology provides (CESM-like
# spin-up starting values); the inorganic pools come from GLODAP/WOA.
_SEEDS = {
    "Lig": 1.0e-3, "DIC_ALT_CO2": None, "ALK_ALT_CO2": None,      # None -> copy of DIC / ALK
    # semi-labile DOM: small (turns over in ~years; a 40 mmol/m^3 seed = ~640 PgC draining into DIC
    # at ~100 PgC/yr, seen in C2_marbl_6mo); refractory DOM carries the ~40 uM deep background
    "DOC": 1.0, "DOCr": 40.0, "DON": 0.1, "DONr": 2.0, "DOP": 0.01, "DOPr": 0.03,
    "zooC": 0.1,
    "spC": 0.1, "spChl": 0.1, "spP": 0.1 / 117.0, "spFe": 0.1 * 3.0e-6, "spCaCO3": 0.01,
    "diatC": 0.1, "diatChl": 0.1, "diatP": 0.1 / 117.0, "diatFe": 0.1 * 3.0e-6, "diatSi": 0.1 * 0.137,
    "diazC": 0.01, "diazChl": 0.01, "diazP": 0.01 / 117.0, "diazFe": 0.01 * 3.0e-6,
}
_FE_SURF, _FE_DEEP = 0.1e-3, 0.6e-3       # mmol/m^3 (0.1 -> 0.6 nM), Parekh-like profile
_FROM_INIT = {"DIC": "DIC", "ALK": "ALK", "NO3": "NO3", "PO4": "PO4", "SiO3": "Si", "O2": "O2"}


class MARBLRunner(BGCRunner):
    @classmethod
    def from_args(cls, args, state, grid, z_coord, model, area2d):
        shape = tuple(state.T.data.shape)
        ocean = np.asarray(state.land_mask.data) > 0.5     # legoESM land_mask: 1 = ocean
        dtype = jnp.float64 if jax.config.jax_enable_x64 else state.T.data.dtype
        if not jax.config.jax_enable_x64:
            print("  WARNING: JAX_ENABLE_X64 is off; MARBL runs in float32", flush=True)

        # ---- initial state -------------------------------------------------
        if args.marbl_ic:
            bgc = _load_ic_netcdf(args.marbl_ic, shape, ocean, dtype)
            print(f"[marbl] 32 tracers from {args.marbl_ic}", flush=True)
        else:
            bgc = _seed_state(args, shape, ocean, np.asarray(z_coord.z_full_ref), dtype)
        # ---- MARBL column stepper -------------------------------------------
        cfg = MarblConfig()
        every = max(1, int(args.bgc_every))
        dt_bgc = float(args.dt) * every
        zt = -np.asarray(z_coord.z_full_ref)
        zw = -np.asarray(z_coord.z_half_ref)[1:]
        dz = np.asarray(z_coord.dz_ref)
        step = make_marbl_stepper(cfg, dt_bgc, zt, zw, dz,
                                  n_iter=int(args.marbl_n_iter),
                                  batch_size=int(args.marbl_batch_size))
        tr = TracerTransporter(grid, z_coord, model.config, state.land_mask.data,
                               k_bg=float(args.bgc_k_bg))
        r = cls(cfg, bgc, tr, z_coord, area2d, state.land_mask.data,
                every=every, diag_every_days=args.bgc_diag_every_days)
        r.step_fn = step
        r.xco2 = float(args.pco2_atm)
        r.forcing = _load_forcing(args.marbl_forcing, shape[:2])
        r.dt_bgc = dt_bgc
        r._acc = None; r._n_acc = 0
        wet = ocean
        print(f"[marbl] CESM2.1 PFT set, {len(TRACER_NAMES)} tracers, source/sink every {every} step(s) "
              f"(dt_bgc = {dt_bgc:.0f} s), xCO2 = {r.xco2} ppmv; surface means PO4 "
              f"{float(np.asarray(bgc.PO4)[..., 0][wet].mean()):.3f}, NO3 "
              f"{float(np.asarray(bgc.NO3)[..., 0][wet].mean()):.2f}, DIC "
              f"{float(np.asarray(bgc.DIC)[..., 0][wet].mean()):.0f} mmol/m3; "
              f"advection={tr.tracer_advection}/{tr.tracer_time_integrator} "
              f"adv={'on' if tr.do_adv else 'OFF'} vmix={'on' if tr.do_vmix else 'OFF'}", flush=True)
        return r

    # ------------------------------------------------------------- per step
    def after_step(self, h_old, state_new, sf, ice_state, dt, step: int):
        self.bgc = self._unstack(self.tr.step(self._stack(), h_old, state_new, dt))
        if step % self.every == 0:
            sst = state_new.T.data[..., 0]
            ones = jnp.ones_like(sst)
            A = jnp.zeros_like(sst)
            if ice_state is not None and getattr(ice_state, "concentration", None) is not None:
                A = jnp.clip(jnp.asarray(getattr(ice_state.concentration, "data",
                                                 ice_state.concentration)), 0.0, 1.0)
                if A.ndim == 3:
                    A = jnp.clip(jnp.sum(A, axis=-1), 0.0, 1.0)
            sw = jnp.asarray(getattr(sf.sw_down, "data", sf.sw_down))
            tx = _to_centre(jnp.asarray(getattr(sf.tau_x, "data", sf.tau_x)), sst.shape)
            ty = _to_centre(jnp.asarray(getattr(sf.tau_y, "data", sf.tau_y)), sst.shape)
            u10_sqr = jnp.hypot(tx, ty) / (_RHO_A * _C_D)
            day = step * dt / 86400.0
            if self.forcing is None:
                zero = jnp.zeros_like(sst)
                dep = dict(dust_flux=zero, iron_flux=zero, nox_flux=zero, nhy_flux=zero)
            else:
                m = int((day % 365.0) / 365.0 * 12) % 12
                dep = {k: self.forcing[k][m] * self.ocean_mask for k in
                       ("dust_flux", "iron_flux", "nox_flux", "nhy_flux")}
                if "ifrac" in self.forcing:
                    A = self.forcing["ifrac"][m]
            forcing = Marbl2DForcing(sw_net=sw, u10_sqr=u10_sqr, atm_press=ones,
                                     ifrac=A, xco2=self.xco2 * ones, **dep)
            h_new = self.tr.thickness(state_new)
            bgc_new, diag = self.step_fn(self.bgc, state_new.T.data, state_new.S.data, h_new, forcing)
            wet = self.ocean_mask
            self.bgc = jax.tree_util.tree_map(lambda x: jnp.where(wet[..., None], x, 0.0), bgc_new)
            diag = jax.tree_util.tree_map(lambda x: jnp.where(wet, x, 0.0), diag)
            self.last_diag = diag
            self._acc = diag if self._acc is None else jax.tree_util.tree_map(jnp.add, self._acc, diag)
            self._n_acc += 1
        if self.diag_every_days > 0:
            stride = max(1, int(round(self.diag_every_days * 86400.0 / dt)))
            if step % stride == 0 or step == 1:
                self.print_diag(state_new, step, dt)

    # ------------------------------------------------------------- diagnostics
    def print_diag(self, state, step, dt):
        b = self.bgc
        pools = {
            "P": b.PO4 + b.DOP + b.DOPr + b.spP + b.diatP + b.diazP + b.zooC / 117.0,
            "N": b.NO3 + b.NH4 + b.DON + b.DONr + (b.spC + b.diatC + b.diazC + b.zooC) * (16.0 / 117.0),
            "Si": b.SiO3 + b.diatSi, "Fe": b.Fe, "DIC": b.DIC, "ALK": b.ALK,
            # total carbon: DIC + semi-labile/refractory DOM + living + CaCO3 (closed apart from air-sea)
            "C": b.DIC + b.DOC + b.DOCr + b.spC + b.diatC + b.diazC + b.zooC + b.spCaCO3,
        }
        stack = jnp.stack(list(pools.values()))
        inv = np.asarray(self.tr.inventory(stack, state, self.area2d))
        vol = float(self.tr.wet_volume(state, self.area2d))
        if self._inv0 is None or len(self._inv0) != len(pools):
            self._inv0 = inv.copy(); self._vol0 = vol
        day = step * dt / 86400.0
        parts = " ".join(f"{k}={inv[i]:.6e} ({(inv[i] - self._inv0[i]) / max(abs(self._inv0[i]), 1e-30):+.2e})"
                         for i, k in enumerate(pools))
        print(f"[marbl] day={day:.2f} wet volume={vol:.6e} m^3 ({(vol - self._vol0) / self._vol0:+.2e}); "
              f"element inventories (mmol, rel. change since ref): {parts}", flush=True)
        if self._acc is not None and self._n_acc:
            area = np.asarray(self.area2d) * np.asarray(self.ocean_mask)
            mean = {k: np.asarray(v) / self._n_acc for k, v in self._acc._asdict().items()}
            to_pg = 12.011e-3 * SEC_PER_YEAR * 1e-15                      # mmol/s -> PgC/yr
            npp = float(np.nansum(mean["NPP"].sum(0) * area)) * to_pg
            exp = float(np.nansum(mean["POC_flux_100m"] * area)) * to_pg
            fco2 = float(np.nansum(mean["flux_co2"] * area)) * to_pg
            chl = float(np.nanmean(np.where(self.ocean_mask, mean["Chl_surf"], np.nan)))
            pco2 = float(np.nanmean(np.where(self.ocean_mask, mean["pCO2surf"], np.nan)))
            print(f"[marbl]   NPP {npp:.2f} PgC/yr, export(100 m) {exp:.2f} PgC/yr, air-sea CO2 "
                  f"{fco2:+.3f} PgC/yr (+ into ocean), surface Chl {chl:.3f} mg/m3, pCO2 {pco2:.1f} ppm", flush=True)
            self._acc = None; self._n_acc = 0
        neg = {k: int(jnp.sum((getattr(b, k) < 0) & self.tr.active_3d)) for k in TRACER_NAMES}
        neg = {k: v for k, v in neg.items() if v}
        print(f"[marbl]   negative wet cells: {neg if neg else 'none'}", flush=True)


# ----------------------------------------------------------------------------- helpers
def _seed_state(args, shape, ocean, z_full_ref, dtype):
    nlev = shape[-1]
    wet = ocean[..., None]
    out = {}
    init = None
    if args.bgc_init_dir:
        init = np.load(Path(args.bgc_init_dir) / "bgc_init.npz")
        for k, src in _FROM_INIT.items():
            if src in init.files:
                a = np.asarray(init[src], np.float64) * MOL_TO_MMOL
                if a.shape != shape:
                    raise ValueError(f"bgc_init.npz:{src} has shape {a.shape}, model grid is {shape}")
                out[k] = a
        print(f"[marbl] inorganic initial fields from {args.bgc_init_dir}/bgc_init.npz: "
              f"{sorted(k for k in out)} (mol/m^3 -> mmol/m^3)", flush=True)
    else:
        print("[marbl] WARNING: no --bgc-init-dir and no --marbl-ic: inorganic pools start "
              "from crude uniform values", flush=True)
        out.update(DIC=np.full(shape, 2100.0), ALK=np.full(shape, 2300.0), NO3=np.full(shape, 15.0),
                   PO4=np.full(shape, 1.0), SiO3=np.full(shape, 30.0), O2=np.full(shape, 200.0))
    z_norm = np.clip(-z_full_ref / 1000.0, 0.0, 1.0)
    out["Fe"] = np.broadcast_to((_FE_SURF + (_FE_DEEP - _FE_SURF) * z_norm)
                                .reshape((1,) * (len(shape) - 1) + (nlev,)), shape)
    out["NH4"] = np.full(shape, 0.01)
    for k, v in _SEEDS.items():
        if v is None:
            out[k] = out["DIC"] if k == "DIC_ALT_CO2" else out["ALK"]
        else:
            out[k] = np.full(shape, v)
    missing = [k for k in TRACER_NAMES if k not in out]
    if missing:
        raise KeyError(f"marbl seed: no value for tracers {missing}")
    return MarblTracers(**{k: jnp.asarray(np.where(wet, out[k], 0.0), dtype) for k in TRACER_NAMES})


def _load_ic_netcdf(path, shape, ocean, dtype):
    import xarray as xr
    ds = xr.open_dataset(path)
    out = {}
    for k in TRACER_NAMES:
        if k not in ds:
            raise KeyError(f"{path}: missing tracer {k!r}")
        a = np.asarray(ds[k].values, np.float64)
        if a.shape != shape:
            raise ValueError(f"{path}: {k} has shape {a.shape}, model grid is {shape}")
        out[k] = jnp.asarray(np.where(np.isfinite(a) & ocean[..., None], a, 0.0), dtype)
    return MarblTracers(**out)


def _load_forcing(path, shape2d):
    if not path:
        print("[marbl] no --marbl-forcing: dust / Fe / N deposition = 0; ice fraction from the sea-ice model",
              flush=True)
        return None
    import xarray as xr
    ds = xr.open_dataset(path)
    out = {}
    for k in FORCING_VARS:
        if k not in ds:
            if k == "ifrac":
                continue
            raise KeyError(f"{path}: missing {k!r}")
        a = np.nan_to_num(np.asarray(ds[k].values, np.float64))
        if a.shape != (12,) + tuple(shape2d):
            raise ValueError(f"{path}: {k} has shape {a.shape}, expected (12,) + {tuple(shape2d)}")
        out[k] = jnp.asarray(a)
    print(f"[marbl] monthly deposition {', '.join(out)} from {path}", flush=True)
    return out
