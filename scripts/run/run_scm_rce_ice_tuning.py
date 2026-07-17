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

Reference-free: uses ``SingleColumnModel.create`` (atmosphere/forcing/scm/scm.py) with a
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
_SCRIPTS_RUN = str(Path(__file__).resolve().parent)  # for run_scm_rce_campaign
if _SCRIPTS_RUN not in sys.path:
    sys.path.insert(0, _SCRIPTS_RUN)

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
from legoesm.atmosphere.physics.microphysics.config import (
    MorrisonConfig,
    __param_spec__ as _MICRO_PARAM_SPEC,
)
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel

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


def _radiation_cfg(radiation: str, rad_interval: int) -> RadiationConfig:
    """Radiation deck for the sweep.

    'rrtmgp' = the LEAN rrtmgp deck (resolved clouds + include_clouds) — the
    physically faithful cloud-radiative feedback, but its CPU LLVM codegen is
    memory-fragile (fails ~dylib_9) and on GPU each trial RECOMPILES the heavy
    graph (~tens of min x N trials).  'gray' = a cheap deck that compiles in
    seconds: NO cloud-radiative feedback, but the ICE-BUDGET RANKING (relative
    supercooled-liquid fraction across levers) is MICROPHYSICS-driven, so gray
    gives the valid lever ranking quickly; use rrtmgp for a few anchor points."""
    if radiation == "gray":
        return RadiationConfig(scheme="gray", diurnal_cycle=False,
                               update_interval_steps=rad_interval)
    if radiation == "rrtmgp":
        return RadiationConfig(
            scheme="rrtmgp", cloud_scheme="resolved",
            # FORWARD rrtmgp deck for CPU/GPU/TPU. gpoint_batch_size=16 = the
            # vmap-over-blocks g-point path (no prevent_cse checkpoint, ~15
            # batched kernels). Combined with the optics-table
            # optimization_barriers (gas_optics.py + cloud_optics.py) that stop
            # XLA constant-folding the per-band/g-point table slices into the
            # kernels, this keeps the rrtmgp executable under the XLA-CPU
            # LLVM-JIT code-region limit and speeds GPU/TPU compile.
            rrtmgp=RRTMGPConfig(include_clouds=True, gpoint_batch_size=16),
            update_interval_steps=rad_interval, diurnal_cycle=False)
    raise ValueError(f"radiation must be 'rrtmgp' or 'gray', got {radiation!r}")


def make_cfg(overrides: dict, *, rad_interval: int,
             radiation: str = "rrtmgp",
             convection: str = "mass_flux") -> PhysicsConfig:
    """rrtmgp (resolved clouds, include_clouds) or gray + morrison + convection +
    Louis BL, with the ice-budget ``overrides`` spliced into a NEW MorrisonConfig
    via ``_replace`` (production defaults untouched).

    DELIBERATELY a LEAN rrtmgp deck, NOT the campaign's full RCEMIP deck
    (``make_physics_config``: MLS ozone + SAM ocean albedo + RCEMIP insolation).
    That fuller deck reliably FAILS CPU LLVM codegen — ``JaxRuntimeError:
    Failed to materialize symbols`` in ``two_stream.solve_sw`` at ~the 4th JIT
    dylib (the SW solver's lax.cond fusions exceed the host codegen budget) —
    whereas this lean deck compiles + runs (smoke-verified).  The ICE-BUDGET
    RANKING is the RELATIVE supercooled-liquid fraction across levers that ALL
    share this SAME deck, so it is robust to the absolute radiation climate; the
    lean deck is the right tool for the ranking objective (the full RCEMIP deck
    is for absolute-realism scoring, out of scope here)."""
    _assert_in_bounds(overrides)
    morrison = MorrisonConfig()._replace(
        **{k: float(v) for k, v in overrides.items()})
    return PhysicsConfig(
        radiation=_radiation_cfg(radiation, rad_interval),
        convection=ConvectionConfig(scheme=convection),
        turbulence=TurbulenceConfig(scheme="louis"),
        microphysics=MicrophysicsConfig(scheme="morrison", morrison=morrison),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )


def _column_diag(state, dsigma, nlev: int) -> dict:
    """Per-level T/q_c/q_i + the supercooled-liquid fraction (mass-weighted q_c
    at T<T_freeze / total q_c) + column LWP/IWP, from a column state."""
    T = np.asarray(state.T.data[0, 0, 0])
    tr = state.tracers or {}
    q_c = np.asarray(tr["q_c"].data[0, 0, 0]) if "q_c" in tr else np.zeros(nlev)
    q_i = np.asarray(tr["q_i"].data[0, 0, 0]) if "q_i" in tr else np.zeros(nlev)
    p_s = float(state.p_s.data[0, 0, 0])
    dp = np.asarray(dsigma) * p_s
    cold = T < constants.T_freeze
    qc_mass = q_c * dp
    tot_qc = float(qc_mass.sum())
    sc_frac = float(qc_mass[cold].sum() / tot_qc) if tot_qc > 1e-20 else 0.0
    lwp = float((q_c * dp / constants.g).sum())
    iwp = float((q_i * dp / constants.g).sum())
    return {
        "supercooled_liquid_frac": sc_frac,
        "frozen_condensate_frac": iwp / max(lwp + iwp, 1e-20),
        "lwp_col_kg_m2": lwp, "iwp_col_kg_m2": iwp,
        "T_sfc_K": float(T[-1]), "T_top_K": float(T[0]),
        "T_profile_K": T.tolist(),
        "q_c_profile_kg_kg": q_c.tolist(),
        "q_i_profile_kg_kg": q_i.tolist(),
        "finite": bool(np.all(np.isfinite(T)) and np.all(np.isfinite(q_c))
                       and np.all(np.isfinite(q_i))),
    }


def run_column(overrides: dict, *, nlev: int, dt: float, days: float,
               T_sfc: float, lat: float, rad_interval: int,
               radiation: str = "rrtmgp", tail_days: float = 10.0) -> dict:
    """Equilibrate one column under FIXED SST (prescribed T_s, so each ice-lever
    trial sees the SAME surface boundary — not a free-running, lever-dependent
    drift), then report per-level q_c/q_i + the supercooled-liquid fraction.  A
    convergence check compares the fraction before and after a final ``tail_days``
    window (``converged`` if it moved < 0.05) so a transient does not mis-rank a
    lever; the reported fraction is the tail (more-equilibrated) value."""
    import os
    from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing
    T0, qv0 = build_initial_profiles(nlev, T_sfc)
    cfg = make_cfg(overrides, rad_interval=rad_interval, radiation=radiation)
    _Ts = float(T_sfc)
    # Debug isolation toggle: skip the prescribed-SST forcing (free-running) to
    # test whether the forcing graph is what tips the CPU JIT mmap over.
    forcing = (None if os.environ.get("LEGOESM_SCM_NO_FORCING")
               else SCMForcing(prescribe="T_s", T_s=lambda _t: _Ts))
    scm = SingleColumnModel.create(
        physics_config=cfg, nlev=nlev, dt=dt,
        T_profile=T0, q_v_profile=qv0, latitude_deg=lat,
        time_integrator="forward_euler", forcing=forcing,
    )
    n_tail = max(1, int(tail_days * 86400.0 / dt))
    n_main = max(1, int(max(days - tail_days, 0.0) * 86400.0 / dt))
    dsig = np.asarray(scm.sigma_coord.dsigma)
    if n_main > 0:
        spinup, _ = scm.run(nsteps=n_main, save_every=n_main)
        frac_pre = _column_diag(spinup, dsig, nlev)["supercooled_liquid_frac"]
    else:
        frac_pre = float("nan")
    final, _ = scm.run(nsteps=n_tail, save_every=n_tail)
    diag = _column_diag(final, dsig, nlev)
    diag["overrides"] = {k: float(v) for k, v in overrides.items()}
    diag["supercooled_frac_pre_tail"] = frac_pre
    diag["converged"] = bool(
        np.isfinite(frac_pre)
        and abs(diag["supercooled_liquid_frac"] - frac_pre) < 0.05)
    return diag


def trial_list() -> list[dict]:
    """The sweep trials: baseline (morrison defaults) + one-at-a-time per-lever +
    a combined 'max-ice' trial pushing every primary lever to its strongest."""
    trials = [{}]
    for name, (_default, values) in ICE_LEVERS.items():
        for v in values:
            trials.append({name: v})
    trials.append({
        "homogeneous_freeze_T": 240.0, "cooper_a": 0.9,
        "N_i0": 50.0, "rime_coeff": 2.0,
    })
    return trials


def _override_str(ov: dict) -> str:
    return "baseline" if not ov else ",".join(f"{k}={v}" for k, v in ov.items())


def _parse_override(s: str) -> dict:
    s = (s or "").strip()
    if not s or s == "baseline":
        return {}
    out = {}
    for tok in s.split(","):
        k, _, v = tok.partition("=")
        out[k.strip()] = float(v)
    return out


def _safe(label: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in label)


# NOTE: each distinct MorrisonConfig is baked STATIC into the SCM graph, so each
# trial RECOMPILES the full rrtmgp+morrison graph.  Running many trials in ONE
# process exhausts the XLA-CPU JIT dylib space ("Failed to materialize symbols"
# / OOM at trial ~10), so the sweep is driven as ONE PROCESS PER TRIAL (the
# sbatch loops emit-trials -> single -> aggregate).  A single trial compiles
# cleanly (smoke-verified).


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "emit-trials", "aggregate"],
                   default="single")
    p.add_argument("--override", type=str, default="",
                   help="single-trial knob string, e.g. 'homogeneous_freeze_T=240,N_i0=50'")
    p.add_argument("--days", type=float, default=40.0)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--nlev", type=int, default=40)
    p.add_argument("--T-sfc", type=float, default=300.0)
    p.add_argument("--latitude-deg", type=float, default=0.0)
    p.add_argument("--rad-interval", type=int, default=12,
                   help="radiation update interval [steps] (amortise rrtmgp)")
    p.add_argument("--radiation", choices=["rrtmgp", "gray"], default="rrtmgp",
                   help="gray = fast valid lever ranking; rrtmgp = faithful but compile-heavy")
    p.add_argument("--out", type=str, default="results/scm_rce_ice_tuning")
    args = p.parse_args(argv)

    if args.mode == "emit-trials":
        for ov in trial_list():
            print(_override_str(ov))
        return 0

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.mode == "aggregate":
        recs = [json.loads(f.read_text()) for f in sorted(outdir.glob("trial_*.json"))]
        if not recs:
            print(f"[aggregate] no trial_*.json under {outdir}")
            return 1
        ranked = sorted(recs, key=lambda r: r["supercooled_liquid_frac"])
        print("\n=== ranked by supercooled-liquid fraction (lower = better) ===")
        for r in ranked:
            print(f"  {r['supercooled_liquid_frac']:.3f}  "
                  f"frozen={r['frozen_condensate_frac']:.3f}  "
                  f"conv={r.get('converged')}  {r['label']}")
        base = next((r for r in recs if r["label"] == "baseline"), None)
        best = ranked[0]
        if base is not None:
            print(f"\nbaseline supercooled_frac="
                  f"{base['supercooled_liquid_frac']:.3f} -> "
                  f"best={best['supercooled_liquid_frac']:.3f} ({best['label']})")
        (outdir / "ice_tuning_summary.json").write_text(json.dumps(ranked, indent=1))
        print(f"wrote {outdir/'ice_tuning_summary.json'}")
        return 0

    # mode == single
    ov = _parse_override(args.override)
    rec = run_column(ov, nlev=args.nlev, dt=args.dt, days=args.days,
                     T_sfc=args.T_sfc, lat=args.latitude_deg,
                     rad_interval=args.rad_interval, radiation=args.radiation)
    rec["label"] = _override_str(ov)
    print(f"[trial {rec['label']}] supercooled_frac="
          f"{rec['supercooled_liquid_frac']:.3f} "
          f"frozen_frac={rec['frozen_condensate_frac']:.3f} "
          f"T_sfc={rec['T_sfc_K']:.1f}K converged={rec['converged']} "
          f"finite={rec['finite']}")
    (outdir / f"trial_{_safe(rec['label'])}.json").write_text(json.dumps(rec, indent=1))
    print(f"wrote {outdir/('trial_'+_safe(rec['label'])+'.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
