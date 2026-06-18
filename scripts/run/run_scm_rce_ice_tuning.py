"""Reference-free single-column rrtmgp + morrison RCE for ICE-CLOUD WATER BUDGET
tuning.

The coupled CMIP TOA cold-drift is driven by WARM-RAIN-only kessler leaving
SUPERCOOLED-LIQUID high cloud (70% of column cloud liquid at T<0C, peak -20C;
no freeze->snow->precip sink) -> high cloud optical depth -> OLR collapse.  The
fix is an ice-capable scheme (morrison) whose ice-budget knobs convert that
supercooled liquid to ice.  This driver equilibrates a SINGLE COLUMN under
rrtmgp + morrison + convection (ncol=1 -> the rrtmgp graph is tiny, so it
compiles cheaply, unlike the C18 global run that times out in XLA compile) and
SWEEPS the morrison ice-budget levers, ranking them by the supercooled-liquid
fraction (the cold-drift diagnostic).

Reference-free: uses ``SingleColumnModel.create`` (atmosphere/scm.py) with a
moist tropical sounding directly — NO CRM ``vol_*.npz`` reference files (the
``run_scm_rce_campaign`` harness is gated on those; this driver bypasses that
gate by stepping the column itself and keeping the FINAL state so q_c / q_i can
be split, which ``run_scm_rce`` discards).

Production *Config defaults are NEVER mutated: every trial builds a NEW
``MorrisonConfig`` via ``_replace`` with the override dict (the traced-override
contract).  Each override is bounds-checked against the morrison
``__param_spec__`` so no knob escapes its spec range.

Login-node: NOTHING here runs on the Ginsburg login node — drive via sbatch.

Usage (sbatch / compute node):
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python scripts/run/run_scm_rce_ice_tuning.py \
        --mode sweep --days 40 --dt 600 --nlev 40 \
        --out results/scm_rce_ice_tuning
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from legoesm import constants
from legoesm.atmosphere.physics import (
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
    MicrophysicsConfig,
    ConvectionConfig,
    GravityWaveDragConfig,
)
from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.config import (
    __param_spec__ as _MICRO_PARAM_SPEC,
)
from legoesm.atmosphere.scm import SingleColumnModel

# --- Ice-cloud water-budget tuning levers (morrison) ----------------------
# Each: (default, [sweep values]).  Direction (all INCREASE) converts MORE
# supercooled liquid to ice aloft, removing the high supercooled-liquid cloud
# that drives the cold drift.  Bounds enforced from the morrison __param_spec__.
ICE_LEVERS = {
    # PRIMARY: raise the all-glaciation threshold so supercooled liquid freezes
    # at WARMER altitudes (attacks the -20C liquid peak directly).
    "homogeneous_freeze_T": (233.15, [236.0, 238.0, 240.0]),
    # Steeper Cooper(1986) ice-nucleation curve -> more crystals/K undercooling
    # -> faster emergent WBF stripping liquid in the mixed-phase zone.
    "cooper_a": (0.304, [0.5, 0.7, 0.9]),
    # More base ice crystals -> more depositional surface -> faster WBF.
    "N_i0": (5.0, [20.0, 50.0]),
    # Faster riming collection of supercooled cloud water onto ice/snow.
    "rime_coeff": (1.0, [1.5, 2.0]),
}


def _morrison_spec() -> dict:
    """The morrison field spec dict {field -> {bounds, ...}}.

    ``config.__param_spec__`` maps a config name -> a block
    ``{"scheme_key": ..., "excluded": [...], "params": {field: {...}}}``; the
    per-field specs (with ``bounds``) live under the block's ``params`` key."""
    for name, block in _MICRO_PARAM_SPEC.items():
        if not isinstance(block, dict):
            continue
        if name == "MorrisonConfig" or str(
                block.get("scheme_key", "")).endswith("MorrisonConfig"):
            params = block.get("params", {})
            if "homogeneous_freeze_T" in params:
                return params
    raise RuntimeError("morrison __param_spec__ params block not found")


def _assert_in_bounds(overrides: dict) -> None:
    spec = _morrison_spec()
    for name, val in overrides.items():
        if name not in spec:
            raise ValueError(
                f"ice-budget override {name!r} is not a specced morrison field; "
                f"specced: {sorted(spec)}")
        lo, hi = spec[name]["bounds"]
        if not (lo <= float(val) <= hi):
            raise ValueError(
                f"override {name}={val} out of spec bounds [{lo}, {hi}]")


def build_initial_profiles(nlev: int, T_sfc: float):
    """Moist tropical sounding (reused from scripts/matrix/scm/rce.py): 6.5 K/km
    lapse capped at 200 K, near-surface humidity decaying upward."""
    sigma = jnp.linspace(0.01, 1.0, nlev)
    H = 8.0e3
    z = -H * jnp.log(jnp.maximum(sigma, 1e-3))
    T = jnp.maximum(T_sfc - 6.5e-3 * z, 200.0)
    q_v = 1.8e-2 * jnp.exp(-z / 3.0e3)
    return T, q_v


def make_cfg(overrides: dict, *, rad_interval: int) -> PhysicsConfig:
    """rrtmgp (resolved clouds) + morrison + mass-flux convection + Louis BL.

    ``overrides`` splices traced leaves into a NEW MorrisonConfig via ``_replace``
    (production MorrisonConfig() defaults are untouched)."""
    _assert_in_bounds(overrides)
    morrison = MorrisonConfig()._replace(**{k: float(v) for k, v in overrides.items()})
    return PhysicsConfig(
        radiation=RadiationConfig(
            scheme="rrtmgp", cloud_scheme="resolved",
            # include_clouds=True so the resolved q_c/q_i cloud optics are
            # honoured by the RRTMGP solver (the cloud-radiation gate rejects
            # cloud_scheme='resolved' with the default include_clouds=False).
            rrtmgp=RRTMGPConfig(include_clouds=True),
            update_interval_steps=rad_interval, diurnal_cycle=False,
        ),
        convection=ConvectionConfig(scheme="mass_flux"),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="morrison", morrison=morrison),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def run_column(overrides: dict, *, nlev: int, dt: float, days: float,
               T_sfc: float, lat: float, rad_interval: int) -> dict:
    """Equilibrate one column; return per-level T/q_c/q_i + the supercooled-
    liquid fraction (mass-weighted q_c at T<T_freeze / total q_c) and column
    LWP/IWP."""
    T0, qv0 = build_initial_profiles(nlev, T_sfc)
    cfg = make_cfg(overrides, rad_interval=rad_interval)
    nsteps = max(1, int(days * 86400.0 / dt))
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=nlev, dt=dt,
        T_profile=T0, q_v_profile=qv0, latitude_deg=lat,
        time_integrator="forward_euler",
    )
    final, _hist = scm.run(nsteps=nsteps, save_every=max(1, nsteps // 10))

    T = np.asarray(final.T.data[0, 0, 0])                       # (nlev,)
    tr = final.tracers
    q_c = np.asarray(tr["q_c"].data[0, 0, 0]) if "q_c" in tr else np.zeros(nlev)
    q_i = np.asarray(tr["q_i"].data[0, 0, 0]) if "q_i" in tr else np.zeros(nlev)
    p_s = float(final.p_s.data[0, 0, 0])
    dsig = np.asarray(scm.sigma_coord.dsigma)                   # (nlev,)
    dp = dsig * p_s                                             # [Pa]

    cold = T < constants.T_freeze
    qc_mass = q_c * dp
    qi_mass = q_i * dp
    tot_qc = float(qc_mass.sum())
    supercooled_frac = float(qc_mass[cold].sum() / tot_qc) if tot_qc > 1e-20 else 0.0
    lwp_col = float((q_c * dp / constants.g).sum())            # [kg/m^2]
    iwp_col = float((q_i * dp / constants.g).sum())
    ice_frac = iwp_col / max(lwp_col + iwp_col, 1e-20)         # frozen fraction
    return {
        "overrides": {k: float(v) for k, v in overrides.items()},
        "supercooled_liquid_frac": supercooled_frac,
        "frozen_condensate_frac": ice_frac,
        "lwp_col_kg_m2": lwp_col,
        "iwp_col_kg_m2": iwp_col,
        "T_sfc_K": float(T[-1]),
        "T_top_K": float(T[0]),
        "T_profile_K": T.tolist(),
        "q_c_profile_kg_kg": q_c.tolist(),
        "q_i_profile_kg_kg": q_i.tolist(),
        "finite": bool(np.all(np.isfinite(T)) and np.all(np.isfinite(q_c))
                       and np.all(np.isfinite(q_i))),
    }


def sweep(*, nlev: int, dt: float, days: float, T_sfc: float, lat: float,
          rad_interval: int) -> list[dict]:
    """Baseline (morrison defaults) + one-at-a-time per-lever sweep + a combined
    'max-ice' trial.  Ranked by supercooled-liquid fraction (lower = better)."""
    trials = [{}]  # baseline = all defaults
    for name, (_default, values) in ICE_LEVERS.items():
        for v in values:
            trials.append({name: v})
    # Combined 'max-ice' trial: push every primary lever to its strongest value.
    trials.append({
        "homogeneous_freeze_T": 240.0, "cooper_a": 0.9,
        "N_i0": 50.0, "rime_coeff": 2.0,
    })
    out = []
    for i, ov in enumerate(trials):
        rec = run_column(ov, nlev=nlev, dt=dt, days=days, T_sfc=T_sfc,
                         lat=lat, rad_interval=rad_interval)
        label = "baseline" if not ov else ",".join(f"{k}={v}" for k, v in ov.items())
        rec["label"] = label
        print(f"[{i+1}/{len(trials)}] {label:55s} "
              f"supercooled_frac={rec['supercooled_liquid_frac']:.3f} "
              f"frozen_frac={rec['frozen_condensate_frac']:.3f} "
              f"T_sfc={rec['T_sfc_K']:.1f}K finite={rec['finite']}")
        out.append(rec)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "sweep"], default="sweep")
    p.add_argument("--days", type=float, default=40.0)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--nlev", type=int, default=40)
    p.add_argument("--T-sfc", type=float, default=300.0)
    p.add_argument("--latitude-deg", type=float, default=0.0)
    p.add_argument("--rad-interval", type=int, default=12,
                   help="radiation update interval [steps] (amortise rrtmgp)")
    p.add_argument("--out", type=str, default="results/scm_rce_ice_tuning")
    args = p.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        recs = [run_column({}, nlev=args.nlev, dt=args.dt, days=args.days,
                           T_sfc=args.T_sfc, lat=args.latitude_deg,
                           rad_interval=args.rad_interval)]
        recs[0]["label"] = "baseline"
        print(f"[single] supercooled_frac="
              f"{recs[0]['supercooled_liquid_frac']:.3f} "
              f"frozen_frac={recs[0]['frozen_condensate_frac']:.3f}")
    else:
        recs = sweep(nlev=args.nlev, dt=args.dt, days=args.days,
                     T_sfc=args.T_sfc, lat=args.latitude_deg,
                     rad_interval=args.rad_interval)
        ranked = sorted(recs, key=lambda r: r["supercooled_liquid_frac"])
        print("\n=== ranked by supercooled-liquid fraction (lower = better) ===")
        for r in ranked:
            print(f"  {r['supercooled_liquid_frac']:.3f}  "
                  f"frozen={r['frozen_condensate_frac']:.3f}  {r['label']}")
        base = next(r for r in recs if r["label"] == "baseline")
        best = ranked[0]
        print(f"\nbaseline supercooled_frac={base['supercooled_liquid_frac']:.3f}"
              f" -> best={best['supercooled_liquid_frac']:.3f} ({best['label']})")

    (outdir / "ice_tuning.json").write_text(json.dumps(recs, indent=1))
    print(f"wrote {outdir/'ice_tuning.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
