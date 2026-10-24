"""BGC hooks for the forced OMIP drivers (``scripts/run/run_omip_core2.py``).

Four entry points, so the driver edit stays a few lines:

    bgc = BGCRunner.from_args(args, state, grid, z_coord, model, area2d)   # or None
    ...
    h_old = bgc.before_step(state)                 # per step, before model.step
    state_new = model.step(...)
    bgc.after_step(h_old, state_new, sf, ice_state, dt, step)   # transport + source/sink
    ...
    bgc.save(restart_path, step)                   # beside --restart-save
    bgc.load(restart_path, step)                   # beside --restart-from

The BGC tracers live OUTSIDE the ocean state (``OceanBiogeoState``, stacked
for transport), so the physics step, its restart slot policy and its
configuration fingerprint are untouched.  Operator splitting: physics step,
then transport with the step's own fluxes (``transport.TracerTransporter``),
then the scheme's source/sink (``step_ocean_biogeochemistry``).

Forcing for the source/sink, from the surface-forcing object the driver built
for the physics step:
    PAR_surf = 0.43 * max(sw_down, 0) * (1 - A_ice)
    U10      = sqrt(|tau| / (rho_a C_D)) * sqrt(1 - A_ice)      (k_w ~ U10^2 -> flux ~ (1-A))
with rho_a = 1.225, C_D = 1.3e-3.  Ice gating is the open-water fraction only
(no gas exchange through ice; light reaches the ocean only through leads).

Restart: ``<restart>.bgc.npz`` next to the ocean restart, atomic write, with
the absolute step recorded; ``load`` refuses a step mismatch.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.ocean.biogeochemistry import (
    BiogeoConfig, init_biogeo_state, step_ocean_biogeochemistry)
from legoesm.ocean.biogeochemistry.config import OceanBiogeoState
from legoesm.ocean.biogeochemistry.transport import TracerTransporter

_RHO_A = 1.225
_C_D = 1.3e-3

SCHEMES = ("none", "abiotic", "npzd", "npzd_v2", "marbl")


def add_arguments(p) -> None:
    g = p.add_argument_group("ocean biogeochemistry (operator-split, outside the physics step)")
    g.add_argument("--biogeo", choices=SCHEMES, default="none",
                   help="BGC scheme: abiotic = DIC+ALK with air-sea CO2; npzd adds NO3/P/Z/D; "
                        "npzd_v2 = 18 tracers (2 PFTs, NH4, PO4/Si/Fe, DON/DOP, POC/DOC, O2); "
                        "marbl = the 32-tracer MARBL port (biogeochemistry/marbl, CESM2.1 PFTs).")
    g.add_argument("--pco2-atm", type=float, default=284.0,
                   help="Atmospheric pCO2 [uatm] (default 284 = pre-industrial).")
    g.add_argument("--bgc-init-dir", type=str, default=None,
                   help="Directory with DIC/ALK/NO3 initial fields regridded onto the model grid "
                        "(bgc_init.npz with any of DIC, ALK, NO3, PO4, Si, O2, Fe; shape (n_lat, n_lon, nlev), "
                        "mol/m^3). Default: BiogeoConfig uniform initial values.")
    g.add_argument("--bgc-k-bg", type=float, default=1.0e-5,
                   help="Background vertical diffusivity floor for BGC tracers [m^2/s].")
    g.add_argument("--bgc-diag-every-days", type=float, default=5.0,
                   help="Print global inventories / air-sea flux every N days (0 = off).")
    g.add_argument("--bgc-every", type=int, default=1,
                   help="Call the BGC source/sink every N physics steps (transport is every step).")
    g.add_argument("--marbl-ic", type=str, default=None,
                   help="MARBL initial state: NetCDF with all 32 tracers on the model grid [mmol/m^3] "
                        "(scripts/build_marbl_ic.py). Default: DIC/ALK/NO3/PO4/SiO3/O2 from --bgc-init-dir "
                        "and small uniform seeds for the organic / plankton pools.")
    g.add_argument("--marbl-forcing", type=str, default=None,
                   help="Monthly deposition forcing (dust_flux, iron_flux, nox_flux, nhy_flux[, ifrac]) "
                        "on the model grid, (12, n_lat, n_lon). Default: no deposition.")
    g.add_argument("--marbl-batch-size", type=int, default=4096)
    g.add_argument("--marbl-n-iter", type=int, default=50,
                   help="Carbonate-system solver iterations in the MARBL column step.")


class BGCRunner:
    def __init__(self, cfg: BiogeoConfig, bgc: OceanBiogeoState, transporter: TracerTransporter,
                 z_coord, area2d, land_mask_2d, *, every: int, diag_every_days: float):
        self.cfg = cfg
        self.scheme = getattr(cfg, "scheme", "marbl")
        self.bgc = bgc
        self.tr = transporter
        self.z_coord = z_coord
        self.area2d = jnp.asarray(area2d)
        self.ocean_mask = jnp.asarray(land_mask_2d) > 0.5
        self.every = max(1, int(every))
        self.diag_every_days = float(diag_every_days)
        self.fields = [f for f in bgc._fields if getattr(bgc, f) is not None]
        self.last_diag = None
        self._inv0 = None          # reference inventories (persisted in the sidecar)
        self._vol0 = None
        self._flux_acc = 0.0   # mol C accumulated since last print
        self._flux_n = 0

    # ------------------------------------------------------------- build
    @classmethod
    def from_args(cls, args, state, grid, z_coord, model, area2d):
        if getattr(args, "biogeo", "none") == "none":
            return None
        if args.biogeo == "marbl":
            from legoesm.ocean.biogeochemistry.marbl_hooks import MARBLRunner
            return MARBLRunner.from_args(args, state, grid, z_coord, model, area2d)
        cfg = BiogeoConfig(scheme=args.biogeo, pCO2_atm=float(args.pco2_atm))
        shape = tuple(state.T.data.shape)
        bgc = init_biogeo_state(shape, jnp.asarray(z_coord.z_full_ref), cfg)
        land = np.asarray(state.land_mask.data) > 0.5
        if args.bgc_init_dir:
            p = Path(args.bgc_init_dir) / "bgc_init.npz"
            d = np.load(p)
            upd = {}
            for k in ("DIC", "ALK", "NO3", "PO4", "Si", "O2", "Fe"):
                if k in d.files and getattr(bgc, k, None) is not None:
                    a = np.asarray(d[k], dtype=np.float64)
                    if a.shape != shape:
                        raise ValueError(f"{p}:{k} has shape {a.shape}, model grid is {shape}")
                    upd[k] = jnp.asarray(np.where(land[..., None], a, 0.0),
                                         dtype=bgc.DIC.dtype)
            bgc = bgc._replace(**upd)
            print(f"[bgc] initial fields from {p}: {sorted(upd)}", flush=True)
        else:
            print(f"[bgc] uniform initial fields (DIC={cfg.DIC_init}, ALK={cfg.ALK_init})", flush=True)
        tr = TracerTransporter(grid, z_coord, model.config, state.land_mask.data,
                               k_bg=float(args.bgc_k_bg))
        r = cls(cfg, bgc, tr, z_coord, area2d, state.land_mask.data,
                every=args.bgc_every, diag_every_days=args.bgc_diag_every_days)
        print(f"[bgc] scheme={cfg.scheme} pCO2_atm={cfg.pCO2_atm} uatm tracers={r.fields} "
              f"advection={tr.tracer_advection}/{tr.tracer_time_integrator} "
              f"adv={'on' if tr.do_adv else 'OFF'} vmix={'on' if tr.do_vmix else 'OFF'} "
              f"k_bg={tr.k_bg:g}", flush=True)
        return r

    # ------------------------------------------------------------- stack helpers
    def _stack(self):
        return jnp.stack([getattr(self.bgc, f) for f in self.fields])

    def _unstack(self, arr):
        return self.bgc._replace(**{f: arr[i] for i, f in enumerate(self.fields)})

    # ------------------------------------------------------------- per step
    def before_step(self, state):
        return self.tr.thickness(state)

    def after_step(self, h_old, state_new, sf, ice_state, dt, step: int):
        # 1. transport with the physics' own fluxes and K_H
        self.bgc = self._unstack(self.tr.step(self._stack(), h_old, state_new, dt))
        # 2. source/sink
        if step % self.every == 0:
            dt_bgc = dt * self.every
            A = 0.0
            if ice_state is not None and getattr(ice_state, "concentration", None) is not None:
                A = jnp.clip(jnp.asarray(getattr(ice_state.concentration, "data",
                                                 ice_state.concentration)), 0.0, 1.0)
                if A.ndim == 3:            # multi-category: total concentration
                    A = jnp.clip(jnp.sum(A, axis=-1), 0.0, 1.0)
            open_w = 1.0 - A
            sw = jnp.asarray(getattr(sf.sw_down, "data", sf.sw_down))
            par = 0.43 * jnp.maximum(sw, 0.0) * open_w
            tx = _to_centre(jnp.asarray(getattr(sf.tau_x, "data", sf.tau_x)), par.shape)
            ty = _to_centre(jnp.asarray(getattr(sf.tau_y, "data", sf.tau_y)), par.shape)
            tau = jnp.hypot(tx, ty)
            u10 = jnp.sqrt(tau / (_RHO_A * _C_D)) * jnp.sqrt(open_w)
            self.bgc, diag = step_ocean_biogeochemistry(
                self.bgc, state_new.T.data, state_new.S.data,
                jnp.asarray(self.z_coord.dz_ref), jnp.asarray(self.z_coord.z_full_ref),
                self.ocean_mask, dt_bgc, self.cfg, U10=u10, PAR_surf=par)
            self.last_diag = diag
            # global air-sea CO2 flux [mol C/s], positive into ocean, accumulated to a mean
            self._flux_acc += float(jnp.sum(jnp.where(self.ocean_mask, diag.flux_co2, 0.0)
                                            * self.area2d))
            self._flux_n += 1
        # 3. diagnostics
        if self.diag_every_days > 0:
            stride = max(1, int(round(self.diag_every_days * 86400.0 / dt)))
            if step % stride == 0 or step == 1:
                self.print_diag(state_new, step, dt)

    # ------------------------------------------------------------- diagnostics
    def print_diag(self, state, step, dt):
        inv = np.asarray(self.tr.inventory(self._stack(), state, self.area2d))
        vol = float(self.tr.wet_volume(state, self.area2d))
        if self._inv0 is None:
            self._inv0 = inv.copy()
            self._vol0 = vol
        day = step * dt / 86400.0
        parts = []
        for i, f in enumerate(self.fields):
            rel = (inv[i] - self._inv0[i]) / max(abs(self._inv0[i]), 1e-30)
            parts.append(f"{f}={inv[i]:.6e} ({rel:+.2e})")
        print(f"[bgc] day={day:.2f} wet volume={vol:.6e} m^3 ({(vol - self._vol0) / self._vol0:+.2e} since ref); "
              f"inventory = sum(h*c*A), mass-conserving under volume change", flush=True)
        mean_flux_pg_yr = (self._flux_acc / max(self._flux_n, 1)) * 12.011e-15 * 365.0 * 86400.0 \
            if self._flux_n else float("nan")
        surf = {f: float(jnp.nanmean(jnp.where(self.ocean_mask, getattr(self.bgc, f)[..., 0], jnp.nan)))
                for f in self.fields}
        neg = {f: int(jnp.sum((getattr(self.bgc, f) < 0) & self.tr.active_3d)) for f in self.fields}
        print(f"[bgc] day={day:.2f} inventories(mol, rel. change since ref): " + " ".join(parts),
              flush=True)
        print(f"[bgc]   air-sea CO2 flux (mean since last print) = {mean_flux_pg_yr:+.4f} PgC/yr into ocean | "
              f"surface means {', '.join(f'{k}={v:.4g}' for k, v in surf.items())} | "
              f"negative wet cells {neg}", flush=True)
        self._flux_acc = 0.0
        self._flux_n = 0

    # ------------------------------------------------------------- restart
    @staticmethod
    def sidecar(restart_path) -> Path:
        p = Path(restart_path)
        return p.with_name(p.stem + ".bgc.npz")   # restart_leg_X.npz -> restart_leg_X.bgc.npz

    def save(self, restart_path, step: int) -> Path:
        out = self.sidecar(restart_path)
        payload = {f: np.asarray(getattr(self.bgc, f)) for f in self.fields}
        payload["_step"] = np.asarray(int(step))
        payload["_scheme"] = np.asarray(self.scheme)
        payload["_pco2_atm"] = np.asarray(float(self.cfg.pCO2_atm))
        if self._inv0 is not None:
            payload["_inv0"] = np.asarray(self._inv0)
            payload["_vol0"] = np.asarray(float(self._vol0))
            payload["_inv0_fields"] = np.asarray(self.fields)
        tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **payload)
        os.replace(tmp, out)
        print(f"[bgc] saved {len(self.fields)} tracers at step {step} -> {out}", flush=True)
        return out

    def load(self, restart_path, expect_step: int) -> None:
        src = self.sidecar(restart_path)
        _legacy = Path(restart_path).with_name(Path(restart_path).name + ".bgc.npz")
        if not src.exists() and _legacy.exists():
            src = _legacy
        if not src.exists():
            raise FileNotFoundError(
                f"--biogeo {self.scheme}: resuming from {restart_path} but its BGC sidecar "
                f"{src} does not exist. Resume from a leg that ran with BGC, or branch with a "
                f"fresh BGC state (omit --restart-from for the BGC by deleting the sidecar expectation).")
        d = np.load(src, allow_pickle=False)
        st = int(d["_step"])
        if st != int(expect_step):
            raise ValueError(f"{src}: BGC step {st} != ocean restart step {expect_step}")
        sch = str(d["_scheme"])
        if sch != self.scheme:
            raise ValueError(f"{src}: scheme {sch!r} != requested {self.scheme!r}")
        upd = {f: jnp.asarray(d[f], dtype=self.bgc.DIC.dtype) for f in self.fields if f in d.files}
        missing = [f for f in self.fields if f not in d.files]
        if missing:
            raise ValueError(f"{src}: tracers {missing} missing from the sidecar")
        self.bgc = self.bgc._replace(**upd)
        if "_inv0" in d.files and list(d["_inv0_fields"]) == self.fields:
            self._inv0 = np.asarray(d["_inv0"]); self._vol0 = float(d["_vol0"])
        print(f"[bgc] restored {len(upd)} tracers from {src} (step {st})", flush=True)


def _to_centre(f, shape):
    """Average a face-valued 2-D field onto cell centres (u-faces: n_lon+1; v-faces: n_lat+1);
    pass a centre-valued field through."""
    f = jnp.asarray(f)
    if f.shape == tuple(shape):
        return f
    if f.shape == (shape[0], shape[1] + 1):
        return 0.5 * (f[:, :-1] + f[:, 1:])
    if f.shape == (shape[0] + 1, shape[1]):
        return 0.5 * (f[:-1, :] + f[1:, :])
    return jnp.broadcast_to(f, shape)
