"""Tune an SCM turbulence closure to LES truth (derivative-free) — LES_SUITE D4.

Reads a cached LES reference artifact (never re-integrates the LES) and calibrates
one closure's ``__param_spec__`` coefficients by minimising the SCM-vs-LES
prognostic profile loss (``scm_runner.scm_les_final_loss``). Uses the derivative-free
path — the coordinate-probe + random search that treats every closure identically,
giving the *fair* ranking (D4); the AD path is a follow-up.

The search space and bounds come from ``training.param_collector`` (the same registry
the trainer uses — no forked bounds); candidates are the default, each param probed
at 25%/75% of its bound, and ``n_random`` uniform draws. Overrides are spliced into
the caller-provided base ``TurbulenceConfig`` via ``apply_param_overrides`` (which
raises on an unknown field — never a silent no-op), so production ``*Config`` defaults
are never mutated.

Usage (dry CBL, mynn25)::

    JAX_ENABLE_X64=1 .venv/bin/python scripts/run/tune_scm_to_les.py \\
        --artifact results/les_suite/artifacts/cbl_nieuwstadt__lasd.npz \\
        --scheme mynn25 --tiers 1 --n-random 12 --dt 10 --nlev 32
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from legoesm.atmosphere.forcing.scm.scm_forcing import SCMForcing  # noqa: F401  (doc)
from legoesm.atmosphere.les_suite.bridge import load_artifact
from legoesm.atmosphere.les_suite.scm_runner import scm_les_final_loss
from legoesm.atmosphere.physics.turbulence.tunable_subconfig import (
    rewrap_tunable_subconfig,
    tunable_subconfig,
)
from legoesm.training.param_collector import apply_param_overrides, build_registry


@dataclass(frozen=True)
class TuneResult:
    """Best overrides + loss and the full evaluated history for one closure."""

    scheme: str
    best_overrides: dict
    best_loss: float
    default_loss: float
    n_evaluated: int
    history: list  # list of (loss, overrides) — for the AD-vs-DF comparison later


def _scheme_tunable_metas(scheme_key: str, tiers: tuple[int, ...]) -> list:
    """Scalar tunable ParamMeta for ``scheme_key`` in the requested tiers."""
    metas = [
        m for m in build_registry()
        if m.scheme_key == scheme_key and m.shape_key is None
        and m.tunable_tier in tiers
    ]
    if not metas:
        raise SystemExit(
            f"no scalar tunable params for scheme_key={scheme_key!r} in tiers {tiers}; "
            "check --scheme / --tiers")
    return metas


def _candidate_overrides(metas: list, n_random: int, seed: int) -> list[dict]:
    """Default + per-param 25%/75% coordinate probes + ``n_random`` uniform draws.

    Deterministic given ``seed`` (numpy Generator). Every candidate is a dict of
    ``{field: value}`` for the scheme's tunable fields; the default candidate is the
    empty dict (no override).
    """
    defaults = {m.field: float(m.default) for m in metas}
    cands: list[dict] = [{}]  # the defaults (no override)
    # coordinate probes: one param moved to 25% / 75% of its bound, others default
    for m in metas:
        lo, hi = float(m.bounds[0]), float(m.bounds[1])
        for frac in (0.25, 0.75):
            c = dict(defaults)
            c[m.field] = lo + frac * (hi - lo)
            cands.append(c)
    # random fill
    rng = np.random.default_rng(seed)
    for _ in range(max(0, n_random)):
        c = {}
        for m in metas:
            lo, hi = float(m.bounds[0]), float(m.bounds[1])
            c[m.field] = float(rng.uniform(lo, hi))
        cands.append(c)
    return cands


def apply_overrides_to_base(base_turbulence, overrides: dict):
    """Splice tuned ``overrides`` into ``base_turbulence``'s ACTIVE sub-config.

    Descends to the tunable leaf first, so full CLUBB (whose coefficients nest in
    ``CLUBBConfig.params``) is handled exactly like every flat scheme — for which the
    shared ``tunable_subconfig``/``rewrap_tunable_subconfig`` helpers are identities.
    Empty ``overrides`` returns ``base_turbulence`` unchanged (the default candidate).
    This is the tuner's single apply-site, factored out so it is directly unit-tested.
    """
    if not overrides:
        return base_turbulence
    scheme = base_turbulence.scheme
    sub = getattr(base_turbulence, scheme)
    tuned_leaf = apply_param_overrides(tunable_subconfig(sub), overrides)
    tuned_sub = rewrap_tunable_subconfig(sub, tuned_leaf)
    return base_turbulence._replace(**{scheme: tuned_sub})


def tune_closure_derivative_free(
    artifact,
    base_turbulence,
    scheme_key: str,
    *,
    tiers: tuple[int, ...] = (1,),
    n_random: int = 8,
    nlev: int = 32,
    dt: float = 10.0,
    seed: int = 0,
) -> TuneResult:
    """Derivative-free calibration of one closure against an LES artifact.

    ``base_turbulence`` is a fully-configured :class:`TurbulenceConfig` (surface
    layer etc. set for the case); the tuner only overrides its active sub-config's
    tunable fields. Returns the best overrides + loss and the full history.
    """
    scheme = base_turbulence.scheme
    metas = _scheme_tunable_metas(scheme_key, tiers)

    def _loss(overrides: dict) -> float:
        cfg = apply_overrides_to_base(base_turbulence, overrides)
        return float(scm_les_final_loss(artifact, cfg, nlev=nlev, dt=dt))

    history: list = []
    best_overrides: dict = {}
    best_loss = np.inf
    default_loss = np.nan
    for i, cand in enumerate(_candidate_overrides(metas, n_random, seed)):
        loss = _loss(cand)
        history.append((loss, cand))
        if i == 0:
            default_loss = loss
        if np.isfinite(loss) and loss < best_loss:
            best_loss = loss
            best_overrides = cand
    if not np.isfinite(best_loss):
        # every candidate diverged (non-finite SCM/loss) — a search failure, NOT a
        # tuned result. Raise rather than return inf/NaN (which would also serialise
        # to invalid JSON and silently poison the scorecard).
        n_finite = sum(1 for loss, _ in history if np.isfinite(loss))
        raise RuntimeError(
            f"{scheme}: no candidate produced a finite loss "
            f"({n_finite}/{len(history)} finite); SCM likely diverged — check "
            "nlev/dt/sigma_top or the artifact.")
    return TuneResult(
        scheme=scheme,
        best_overrides=best_overrides,
        best_loss=float(best_loss),
        default_loss=float(default_loss),
        n_evaluated=len(history),
        history=history,
    )


# --- scheme → (sub-config class, registry scheme_key) --------------------------
# The closures wired for the dry-CBL (prescribed-flux) tuner, spanning the
# closure-order ladder for Q1/Q2 (local first-order → nonlocal → 1.5-order).
def _cbl_scheme_table():
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.config import (
        CLUBBLiteConfig,
        HoltslagBovilleConfig,
        LouisConfig,
        MYNN25Config,
        SmagorinskyConfig,
        TKEConfig,
        TurbulentEDMFConfig,
        YSUConfig,
    )

    # The Q2 closure-order ladder (LES_SUITE.md D3): local → nonlocal → 1.5-order →
    # higher-order/mass-flux → full higher-order. Each entry is (config_cls, registry
    # scheme_key), constructed as ``config_cls(surface=surf)``; the scheme_key selects
    # the tunable ParamMeta. Full ``clubb`` nests its tier-1 coefficients in
    # ``CLUBBConfig.params`` (scheme_key atm.turb.CLUBBParams, NOT the CLUBBConfig
    # wrapper), so the apply-site descends via the shared ``tunable_subconfig`` helper;
    # ``CLUBBConfig(surface=surf)`` still constructs the base like the flat schemes.
    return {
        "smagorinsky": (SmagorinskyConfig, "atm.turb.SmagorinskyConfig"),   # local
        "louis": (LouisConfig, "atm.turb.LouisConfig"),                     # local
        "holtslag_boville": (HoltslagBovilleConfig, "atm.turb.HoltslagBovilleConfig"),  # nonlocal
        "ysu": (YSUConfig, "atm.turb.YSUConfig"),                          # nonlocal
        "tke": (TKEConfig, "atm.turb.TKEConfig"),                          # 1.5-order (k-l)
        "mynn25": (MYNN25Config, "atm.turb.MYNN25Config"),                 # 1.5-order (MYNN)
        "clubb_lite": (CLUBBLiteConfig, "atm.turb.CLUBBLiteConfig"),       # higher-order
        "edmf": (TurbulentEDMFConfig, "atm.turb.TurbulentEDMFConfig"),     # mass-flux
        "clubb": (CLUBBConfig, "atm.turb.CLUBBParams"),                    # full higher-order
    }


def _base_turbulence(scheme: str):
    """A CBL-appropriate base TurbulenceConfig + its registry scheme_key."""
    from legoesm.atmosphere.physics import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig

    table = _cbl_scheme_table()
    if scheme not in table:
        raise SystemExit(
            f"--scheme {scheme!r} not wired for the CBL tuner; "
            f"available: {sorted(table)}")
    config_cls, scheme_key = table[scheme]
    surf = SurfaceLayerConfig(z0=0.1, Cd_neutral=1.5e-3, Ch_neutral=0.0)
    turb = TurbulenceConfig(scheme=scheme, **{scheme: config_cls(surface=surf)})
    return turb, scheme_key


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", type=Path, required=True,
                   help="LES reference artifact (.npz) from run_les_suite.py")
    p.add_argument("--scheme", default="mynn25", help="turbulence closure to tune")
    p.add_argument("--tiers", type=int, nargs="+", default=[1],
                   help="tunable tiers to include (1=core, 2=extended, 3=aggressive)")
    p.add_argument("--n-random", type=int, default=12,
                   help="number of random candidates after the coordinate probe")
    p.add_argument("--nlev", type=int, default=32, help="SCM vertical levels")
    p.add_argument("--dt", type=float, default=10.0, help="SCM timestep [s]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None,
                   help="write the tuned result JSON here (default: results/les_suite/tuned)")
    args = p.parse_args(argv)

    if not args.artifact.exists():
        print(f"error: {args.artifact} does not exist", file=sys.stderr)
        return 2
    artifact = load_artifact(args.artifact)
    base, scheme_key = _base_turbulence(args.scheme)

    result = tune_closure_derivative_free(
        artifact, base, scheme_key,
        tiers=tuple(args.tiers), n_random=args.n_random,
        nlev=args.nlev, dt=args.dt, seed=args.seed,
    )
    default_finite = bool(np.isfinite(result.default_loss))
    print(f"[tune] case={artifact.case_name} scheme={result.scheme} "
          f"evaluated={result.n_evaluated}")
    print(f"  default loss = "
          f"{result.default_loss:.4f}" if default_finite else "  default loss = diverged")
    print(f"  best    loss = {result.best_loss:.4f}")
    if default_finite:
        improvement = result.default_loss - result.best_loss
        print(f"  improvement  = {improvement:+.4f} "
              f"({100 * improvement / result.default_loss:+.1f}%)")
    else:
        print("  improvement  = n/a (default config diverged; tuning recovered it)")
    rounded = {k: round(v, 4) for k, v in result.best_overrides.items()}
    print(f"  best overrides: {json.dumps(rounded)}")

    # Default filename keys off the ARTIFACT stem, not case_name: several flux
    # artifacts share one case_name (e.g. every CBL Q0), so a case-name default would
    # silently overwrite the same JSON across fluxes and lose the per-flux records.
    out = args.output or Path("results/les_suite/tuned") / \
        f"{args.artifact.stem}__{result.scheme}__df.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Provenance: which artifact/flux this record tuned against, so the scorecard can
    # group per surface-flux (Q1/Q3 axis) and never double-count the same flux point.
    # ``w_theta_s`` is the applied surface kinematic heat flux [K m/s]; for the CBL
    # anchor it is constant = Q0. ``None`` for cases with no prescribed surface flux.
    q0 = (float(np.asarray(artifact.w_theta_s).reshape(-1)[0])
          if artifact.w_theta_s is not None else None)
    with open(out, "w") as f:
        # JSON has no Inf/NaN literal — write null for a non-finite default loss so
        # the scorecard reader (which falls back to best_loss) stays valid.
        json.dump({
            "case": artifact.case_name,
            "artifact": args.artifact.stem,
            "sgs": artifact.sgs,
            "q0": q0,
            "scheme": result.scheme,
            "method": "derivative_free",
            "tiers": list(args.tiers),
            "default_loss": result.default_loss if default_finite else None,
            "best_loss": result.best_loss,
            "best_overrides": result.best_overrides,
            "n_evaluated": result.n_evaluated,
        }, f, indent=2)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
