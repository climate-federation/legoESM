"""RCE single-column scheme sweep: stability + moist-adiabat realism.

Runs radiative-convective equilibrium (gray radiation + Kessler
microphysics) in **SCM mode** (:class:`legoesm.atmosphere.forcing.scm.scm.SingleColumnModel`,
no dynamical core) while swapping, in turn,

  * every moist-convection scheme (turbulence held at ``louis``), and
  * every turbulence / PBL scheme (convection held at ``mass_flux``).

For each run it integrates a tropical column toward equilibrium and then
assesses:

  1. **Stability** — final state finite, temperature in a plausible band,
     water-vapour non-negative, and the surface-temperature trajectory
     no longer drifting (equilibrated).
  2. **Physical realism** — in the free troposphere (above the mixed
     layer, below the cold-point tropopause) the temperature profile must
     sit close to the surface-parcel **moist adiabat**, computed with the
     model's own :func:`compute_moist_adiabat` (no re-derived thermo).
     We also flag super-adiabatic layers and unphysical inversions aloft.

The moist-adiabat reference, the saturation thermodynamics, the sigma→
pressure map, and all physical constants come from the production model
modules — nothing is re-derived here (CLAUDE.md shared-utility rule).

Usage
-----
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/validate/rce_scm_scheme_sweep.py --mode both --days 80

Exit code is non-zero if any swept scheme fails stability or realism.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.atmosphere.physics import (
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
    MicrophysicsConfig,
    ConvectionConfig,
    GravityWaveDragConfig,
)
from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.thermodynamics import compute_moist_adiabat
from legoesm.atmosphere.forcing.scm.scm import SingleColumnModel

# Schemes to sweep (excluding "none").  Mirrors the factory dispatch in
# convection/integration.py and turbulence/integration.py.
CONVECTION_SCHEMES = (
    "sbm", "dca", "kuo", "mass_flux", "edmf",
    "zhang_mcfarlane", "kain_fritsch", "emanuel", "tiedtke", "bechtold",
)
TURBULENCE_SCHEMES = (
    "smagorinsky", "louis", "tke", "mynn25", "clubb_lite",
    "holtslag_boville", "ysu", "edmf",
)


def build_initial_profiles(nlev: int, T_sfc: float):
    """Moist tropical sounding (same construction as scripts/matrix/scm/rce.py):
    tropospheric lapse 6.5 K/km capped at 200 K, near-saturated near-
    surface humidity decaying upward."""
    sigma = jnp.linspace(0.01, 1.0, nlev)
    H = 8.0e3
    z = -H * jnp.log(jnp.maximum(sigma, 1e-3))
    T = jnp.maximum(T_sfc - 6.5e-3 * z, 200.0)
    q_v = 1.8e-2 * jnp.exp(-z / 3.0e3)
    return T, q_v


@dataclass
class RceResult:
    label: str
    convection: str
    turbulence: str
    ok: bool
    crashed: bool
    error: str
    T_sfc: float
    T_top: float
    T_coldpoint: float
    p_coldpoint_hPa: float
    q_sfc_gkg: float
    q_min: float
    drift_K: float            # |T_sfc(end) - T_sfc(80% mark)|
    madiab_mean_abs_K: float  # mean |T - T_moist| in free troposphere
    madiab_max_abs_K: float
    madiab_bias_K: float      # mean (T - T_moist); + = warmer than parcel
    superadiab_max_K_per_km: float  # worst super-adiabatic lapse aloft
    n_freetrop_levels: int
    fail_reasons: str


# ---- assessment thresholds -------------------------------------------------
T_SFC_MIN, T_SFC_MAX = 240.0, 320.0
T_TOP_MIN, T_TOP_MAX = 150.0, 270.0
DRIFT_TOL_K = 3.0                 # surface T still moving => not equilibrated
MADIAB_MEAN_TOL_K = 6.0          # "not far from moist adiabat"
MADIAB_MAX_TOL_K = 12.0
SUPERADIAB_TOL_K_PER_KM = 11.0   # dry-adiabatic ~9.8; allow margin for grid


def assess(label, conv, turb, Tmean_traj, final, sigma_coord) -> RceResult:
    fails: list[str] = []

    T = np.asarray(final.T.data[0, 0, 0])              # (nlev,) top->bottom
    q = np.asarray(final.tracers["q_v"].data[0, 0, 0])
    p_s = float(np.asarray(final.p_s.data[0, 0, 0]))
    p_full = np.asarray(sigma_coord.pressure_at_full(final.p_s.data))[0, 0, 0]

    finite = bool(np.all(np.isfinite(T)) and np.all(np.isfinite(q)))
    if not finite:
        fails.append("non-finite final state")

    T_sfc = float(T[-1]); T_top = float(T[0])
    q_sfc = float(q[-1]); q_min = float(np.min(q))
    sigma = p_full / p_s

    # equilibration: compare final column-mean T with the value ~80%
    # through the run (surface is clamped under fixed-SST, so use the
    # whole-column mean as the settling indicator).
    i80 = int(0.8 * (len(Tmean_traj) - 1))
    drift = float(abs(Tmean_traj[-1] - Tmean_traj[i80]))

    # ---- lapse-rate tropopause (WMO-style) ----
    # Hydrostatic height z ~ -H ln(sigma); arrays are top->bottom so z
    # decreases with index (z[-1]=surface). Lapse rate Γ = -dT/dz [K/km]:
    # ~6.5 in the troposphere, →0/negative in the near-isothermal gray
    # stratosphere. The tropopause is the LOWEST level (scanning upward
    # from the surface, restricted to p < 500 hPa so the mixed layer /
    # any near-surface inversion cannot be mistaken for it) where Γ first
    # falls below 2 K/km. Using argmin(T) instead would pick the model
    # top whenever T decreases monotonically to the lid, which then leaks
    # the stratosphere into the moist-adiabat comparison.
    H = 8.0e3
    z = -H * np.log(np.maximum(sigma, 1e-3))   # top->bottom, decreasing
    lapse = np.full(len(T), np.nan)             # K/km at interface above k
    for k in range(1, len(T)):
        dz = z[k - 1] - z[k]                     # >0 (k-1 is higher)
        if dz > 0:
            lapse[k] = -(T[k - 1] - T[k]) / dz * 1e3
    i_trop = 0
    for k in range(len(T) - 1, 0, -1):          # surface -> top
        if p_full[k] < 5.0e4 and np.isfinite(lapse[k]) and lapse[k] < 2.0:
            i_trop = k
            break
    T_cold = float(T[i_trop]); p_cold = float(p_full[i_trop])

    # ---- moist-adiabat realism in the free troposphere ----
    # Parcel launched from the surface with the column's surface humidity.
    T_parcel = np.asarray(
        compute_moist_adiabat(
            jnp.asarray([T_sfc]),
            jnp.asarray(p_full)[None, :],
            q_v_base=jnp.asarray([q_sfc]),
        )
    )[0]
    # Free troposphere: above mixed layer (sigma < 0.85) and at/below the
    # tropopause (index >= i_trop, i.e. p >= p_cold). Moist-adiabat
    # tracking is only physically expected in this convecting layer.
    idx = np.arange(len(T))
    ft = (sigma < 0.85) & (idx >= i_trop)
    n_ft = int(np.sum(ft))
    if n_ft >= 3:
        dev = T[ft] - T_parcel[ft]
        madiab_mean_abs = float(np.mean(np.abs(dev)))
        madiab_max_abs = float(np.max(np.abs(dev)))
        madiab_bias = float(np.mean(dev))
    else:
        madiab_mean_abs = madiab_max_abs = madiab_bias = float("nan")
        fails.append(f"too few free-trop levels ({n_ft})")

    # super-adiabatic check aloft (within the free troposphere): flag
    # layers more unstable than dry-adiabatic by a margin (z from above).
    superadiab_max = 0.0
    for k in range(len(T) - 1):
        if not (ft[k] or ft[k + 1]):
            continue
        dz = z[k] - z[k + 1]  # z decreases with index; positive upward step
        if dz <= 0:
            continue
        lapse = (T[k + 1] - T[k]) / dz * 1e3  # K/km, +ve = T decreasing up
        superadiab_max = max(superadiab_max, lapse)

    # ---- verdict ----
    if not (T_SFC_MIN < T_sfc < T_SFC_MAX):
        fails.append(f"T_sfc={T_sfc:.1f} out of band")
    if not (T_TOP_MIN < T_top < T_TOP_MAX):
        fails.append(f"T_top={T_top:.1f} out of band")
    if q_min < -1e-8:
        fails.append(f"q_v<0 (min={q_min:.2e})")
    if finite and drift > DRIFT_TOL_K:
        fails.append(f"not equilibrated (drift={drift:.2f}K)")
    if finite and n_ft >= 3 and madiab_mean_abs > MADIAB_MEAN_TOL_K:
        fails.append(f"far from moist adiabat (mean|dev|={madiab_mean_abs:.1f}K)")
    if finite and n_ft >= 3 and madiab_max_abs > MADIAB_MAX_TOL_K:
        fails.append(f"local moist-adiabat dev={madiab_max_abs:.1f}K")
    if finite and superadiab_max > SUPERADIAB_TOL_K_PER_KM:
        fails.append(f"super-adiabatic aloft ({superadiab_max:.1f} K/km)")

    return RceResult(
        label=label, convection=conv, turbulence=turb,
        ok=(len(fails) == 0), crashed=False, error="",
        T_sfc=T_sfc, T_top=T_top, T_coldpoint=T_cold,
        p_coldpoint_hPa=p_cold / 100.0, q_sfc_gkg=q_sfc * 1e3, q_min=q_min,
        drift_K=drift, madiab_mean_abs_K=madiab_mean_abs,
        madiab_max_abs_K=madiab_max_abs, madiab_bias_K=madiab_bias,
        superadiab_max_K_per_km=superadiab_max, n_freetrop_levels=n_ft,
        fail_reasons="; ".join(fails),
    )


def scan_integrate(scm, nsteps: int, sst=None, surface_rh: float = 0.8):
    """Integrate the whole RCE trajectory in a single ``lax.scan``.

    The SCM's own ``run()`` is a Python loop that re-dispatches the JIT'd
    physics every step (~10 min for an entraining-plume scheme at
    nlev=30 / 2160 steps on CPU x64). For the no-forcing, non-diurnal
    gray RCE the forward-Euler ``_step_fn`` and ``_tend_fn`` are pure
    functions of ``(state, phys_state)`` — stage time only feeds the
    (absent) forcing and the (disabled) diurnal cycle — so the entire
    trajectory fuses into one compiled scan, cutting wall-clock ~50x.

    Fixed-SST lower boundary (``sst`` not None): after each step the
    lowest model level is reset to ``T = sst`` and ``q_v = surface_rh *
    q_sat(sst, p_s)``. This is a Dirichlet ocean-surface BC — the pinned
    warm, near-saturated bottom drives an upward turbulent heat/moisture
    flux through the interior vertical diffusion (and supplies the
    convective parcel its launch properties), so the column equilibrates
    to the SST-anchored moist adiabat instead of drifting to an
    unconstrained surface temperature. It is jittable (pure array set),
    unlike the ``SCMForcing(prescribe='T_s')`` path whose radiation hook
    mutates closure state and cannot run inside ``lax.scan``. The
    surface-emission temperature seen by the gray scheme is the lowest
    air level, which tracks ``sst`` once the BC is enforced.

    Returns ``(final_state, final_phys, Tsfc_trajectory)``. The full T/qv
    columns are read from ``final_state``; only the surface-temperature
    time series is stacked (for the equilibration/drift check).
    """
    from legoesm.thermo import saturation_mixing_ratio
    # lax.scan requires a constant carry pytree. The SCM's Python-loop
    # ``_apply_tendencies`` lazily materialises hydrometeor species the
    # first time physics emits them (e.g. microphysics adds ``q_g`` on
    # step 1), which would change the tracer-dict structure mid-scan.
    # Pre-seed the full hydrometeor set with zeros so the structure is
    # fixed from step 0. The pipeline emits at most q_v/q_c/q_r/q_i/q_s/q_g.
    from legoesm.core.field import Field
    _DIMS_3D = ("face", "x", "y", "level")
    tr = dict(scm.state.tracers or {})
    template = tr["q_v"].data
    for sp in ("q_c", "q_r", "q_i", "q_s", "q_g"):
        if sp not in tr:
            tr[sp] = Field(data=jnp.zeros_like(template), name=sp,
                           dims=_DIMS_3D, units="kg/kg")
    scm.state = scm.state._replace(tracers=tr)

    step_fn = scm._step_fn
    tend_fn = scm._tend_fn
    dt = scm.dt
    _dtype = template.dtype
    sst_arr = None if sst is None else jnp.asarray(sst, dtype=_dtype)

    def pin_surface(state):
        if sst_arr is None:
            return state
        p_s = state.p_s.data                              # (1,1,1)
        qsat_s = saturation_mixing_ratio(sst_arr, p_s[0, 0, 0])
        Tdat = state.T.data.at[..., -1].set(sst_arr)
        new_T = state.T.replace(data=Tdat)
        qv = state.tracers["q_v"]
        qvd = qv.data.at[..., -1].set(surface_rh * qsat_s)
        new_tr = dict(state.tracers)
        new_tr["q_v"] = qv.replace(data=qvd)
        return state._replace(T=new_T, tracers=new_tr)

    def body(carry, k):
        state, phys = carry
        t = k.astype(jnp.float64 if jax.config.read("jax_enable_x64")
                     else jnp.float32) * dt
        new_state, new_phys = step_fn(state, phys, tend_fn, dt, t)
        new_state = pin_surface(new_state)
        # Track column-mean T for the equilibration check: under a fixed-
        # SST BC the surface level is clamped, so its trajectory is flat
        # and useless for detecting whether the column has settled.
        return (new_state, new_phys), jnp.mean(new_state.T.data[0, 0, 0])

    @jax.jit
    def driver(state0, phys0):
        (sf, pf), tsfc = lax.scan(
            body, (state0, phys0), jnp.arange(nsteps))
        return sf, pf, tsfc

    return driver(scm.state, scm.phys_state)


def run_one(conv: str, turb: str, args) -> tuple[RceResult, dict]:
    label = f"conv={conv},turb={turb}"
    T0, qv0 = build_initial_profiles(args.nlev, T_sfc=args.T_sfc)
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="gray", diurnal_cycle=False),
        convection=ConvectionConfig(scheme=conv, zhang_mcfarlane=ZhangMcFarlaneConfig(land_fraction="none")),  # RCE: aquaplanet
        turbulence=TurbulenceConfig(scheme=turb),
        microphysics=MicrophysicsConfig(scheme="kessler"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"),
    )
    nsteps = int((args.days * 86400.0) / args.dt)
    try:
        scm = SingleColumnModel.create(
            physics_config=cfg, nlev=args.nlev, dt=args.dt,
            T_profile=T0, q_v_profile=qv0,
            latitude_deg=args.latitude_deg,
            time_integrator="forward_euler",
        )
        sst = args.T_sfc if args.surface == "fixed" else None
        final, _final_phys, Tmean_traj = scan_integrate(
            scm, nsteps, sst=sst, surface_rh=args.surface_rh)
        Tmean_traj = np.asarray(Tmean_traj)
        res = assess(label, conv, turb, Tmean_traj, final, scm.sigma_coord)
        p_full = np.asarray(
            scm.sigma_coord.pressure_at_full(final.p_s.data))[0, 0, 0]
        diag = {
            "p_full": p_full,
            "T": np.asarray(final.T.data[0, 0, 0]),
            "q_v": np.asarray(final.tracers["q_v"].data[0, 0, 0]),
            "Tmean_traj": Tmean_traj,
            "time": np.arange(nsteps) * args.dt,
        }
        return res, diag
    except Exception as exc:  # noqa: BLE001 — record per-scheme crash
        import traceback
        err = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        res = RceResult(
            label=label, convection=conv, turbulence=turb,
            ok=False, crashed=True, error=err,
            T_sfc=float("nan"), T_top=float("nan"), T_coldpoint=float("nan"),
            p_coldpoint_hPa=float("nan"), q_sfc_gkg=float("nan"),
            q_min=float("nan"), drift_K=float("nan"),
            madiab_mean_abs_K=float("nan"), madiab_max_abs_K=float("nan"),
            madiab_bias_K=float("nan"), superadiab_max_K_per_km=float("nan"),
            n_freetrop_levels=0, fail_reasons=f"CRASHED: {err}",
        )
        return res, {}


def print_table(results: list[RceResult]) -> None:
    hdr = (f"{'scheme':<34}{'ok':<4}{'T_sfc':>7}{'T_cold':>8}{'p_ct':>7}"
           f"{'drift':>7}{'mad|':>7}{'madX':>7}{'bias':>7}{'sup':>7}  reasons")
    print(hdr); print("-" * len(hdr))
    for r in results:
        flag = "OK" if r.ok else ("XX" if r.crashed else "!!")
        print(f"{r.label:<34}{flag:<4}{r.T_sfc:>7.1f}{r.T_coldpoint:>8.1f}"
              f"{r.p_coldpoint_hPa:>7.0f}{r.drift_K:>7.2f}"
              f"{r.madiab_mean_abs_K:>7.1f}{r.madiab_max_abs_K:>7.1f}"
              f"{r.madiab_bias_K:>7.1f}{r.superadiab_max_K_per_km:>7.1f}"
              f"  {r.fail_reasons}")


def maybe_plot(results, diags, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[plot] skipped ({exc})")
        return
    items = [(r, diags[r.label]) for r in results if diags.get(r.label)]
    if not items:
        return
    n = len(items)
    ncol = 5
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (r, d) in enumerate(items):
        ax = axes[i // ncol][i % ncol]; ax.set_visible(True)
        p = d["p_full"] / 100.0
        ax.plot(d["T"], p, "-", color="C0", lw=1.6, label="model")
        T_parcel = np.asarray(compute_moist_adiabat(
            jnp.asarray([float(d["T"][-1])]),
            jnp.asarray(d["p_full"])[None, :],
            q_v_base=jnp.asarray([float(d["q_v"][-1])]),
        ))[0]
        ax.plot(T_parcel, p, "--", color="C3", lw=1.0, label="moist adiabat")
        ax.invert_yaxis()
        ax.set_title(r.label.replace("conv=", "").replace("turb=", "t:"),
                     fontsize=7)
        col = "green" if r.ok else "red"
        ax.tick_params(labelsize=6)
        for s in ax.spines.values():
            s.set_color(col); s.set_linewidth(1.5)
        if i == 0:
            ax.legend(fontsize=6)
    fig.suptitle("RCE SCM scheme sweep: T(p) vs surface moist adiabat",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    import os
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "rce_scm_scheme_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"[plot] wrote {path}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["convection", "turbulence", "both"],
                   default="both")
    p.add_argument("--days", type=float, default=80.0)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--nlev", type=int, default=40)
    p.add_argument("--T-sfc", dest="T_sfc", type=float, default=300.0)
    p.add_argument("--latitude-deg", type=float, default=0.0)
    p.add_argument("--surface", choices=["fixed", "free"], default="fixed",
                   help="fixed = pin lowest level to SST (fixed-SST RCE); "
                        "free = isolated column, surface T evolves freely")
    p.add_argument("--surface-rh", type=float, default=0.8,
                   help="surface relative humidity for the fixed-SST BC")
    p.add_argument("--base-turbulence", default="louis",
                   help="turbulence held fixed during the convection sweep")
    p.add_argument("--base-convection", default="mass_flux",
                   help="convection held fixed during the turbulence sweep")
    p.add_argument("--only", default=None,
                   help="comma-separated subset of scheme names to run")
    p.add_argument("--outdir", default="results/rce_scm_sweep")
    args = p.parse_args(argv)

    only = set(args.only.split(",")) if args.only else None
    combos: list[tuple[str, str]] = []
    if args.mode in ("convection", "both"):
        for c in CONVECTION_SCHEMES:
            if only is None or c in only:
                combos.append((c, args.base_turbulence))
    if args.mode in ("turbulence", "both"):
        for t in TURBULENCE_SCHEMES:
            if only is None or t in only:
                combos.append((args.base_convection, t))

    results: list[RceResult] = []
    diags: dict[str, dict] = {}
    for conv, turb in combos:
        print(f"\n=== RCE  conv={conv}  turb={turb}  "
              f"({args.days} d, nlev={args.nlev}, dt={args.dt}) ===")
        res, diag = run_one(conv, turb, args)
        results.append(res)
        if diag:
            diags[res.label] = diag
        print(f"  -> {'OK' if res.ok else 'FAIL'}  {res.fail_reasons}")

    print("\n" + "=" * 80)
    print_table(results)

    import os
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    maybe_plot(results, diags, args.outdir)

    n_fail = sum(1 for r in results if not r.ok)
    print(f"\n{len(results) - n_fail}/{len(results)} schemes passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
