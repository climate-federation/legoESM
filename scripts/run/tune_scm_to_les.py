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


# A tuned loss at/above this is the AD divergence penalty (scm_les_loss_jax returns a
# large finite barrier for a non-finite SCM), NOT a real fit — real losses are O(1).
_AD_DIVERGED_LOSS = 1.0e5


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


def tune_closure_ad(
    artifact,
    base_turbulence,
    scheme_key: str,
    *,
    tiers: tuple[int, ...] = (1,),
    nlev: int = 24,
    dt: float = 10.0,
    steps: int = 20,
    lr: float = 0.08,
    seed: int = 0,  # noqa: ARG001 — deterministic AD; kept for a uniform signature
) -> TuneResult:
    """AD calibration (D4): gradient descent on the tunable params via the DIFFERENTIABLE
    ``scm_runner.scm_les_loss_jax`` (same final-snapshot objective as the DF path), and
    the gradient is the D4 AD-vs-DF signal. The params enter as TRACED leaves, so the
    grad has ONE jaxpr shape across every param value (XLA reuses the compiled executable)
    — vs the DF path building a NEW static-leaf config per candidate (a fresh trace +
    compile each). CAVEATS: (1) the SCM's ``run`` is a Python step-loop that ``jax.grad``
    UNROLLS, so that backward graph grows with ``nsteps`` (practical only at modest
    resolution until a ``lax.scan`` rollout lands); (2) a diverged trajectory can still
    yield a NaN gradient (0×NaN backprop through the diverged states) — caught here by the
    ``isfinite(grad)`` step guard, which stops the descent (the DF path is the fallback).

    Optimises in NORMALISED space (each param mapped to [0,1] over its bounds) with Adam,
    so disparate scales (C_s~0.2 vs l_mix_max~275) + tiny θ-loss gradients are handled;
    clips back into bounds each step. A non-finite loss/grad (diverged SCM) stops the
    descent (AD cannot use a concrete +inf under trace) — the DF path is the fallback.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from legoesm.atmosphere.les_suite.scm_runner import scm_les_loss_jax

    metas = _scheme_tunable_metas(scheme_key, tiers)
    fields = [m.field for m in metas]
    lo = jnp.asarray([float(m.bounds[0]) for m in metas])
    hi = jnp.asarray([float(m.bounds[1]) for m in metas])
    span = hi - lo
    if not bool(jnp.all(span > 0.0)):  # zero-span bound ⇒ normalisation divide-by-zero
        bad = [m.field for m in metas if float(m.bounds[1]) <= float(m.bounds[0])]
        raise SystemExit(f"{scheme_key}: params with non-positive bound span: {bad}")
    n0 = jnp.clip((jnp.asarray([float(m.default) for m in metas]) - lo) / span, 0.0, 1.0)

    def _overrides(norm):
        vals = lo + jnp.clip(norm, 0.0, 1.0) * span
        return {f: vals[i] for i, f in enumerate(fields)}

    def _to_py(norm) -> dict:
        vals = lo + jnp.clip(norm, 0.0, 1.0) * span
        return {f: float(vals[i]) for i, f in enumerate(fields)}

    def loss_of(norm):
        cfg = apply_overrides_to_base(base_turbulence, _overrides(norm))
        return scm_les_loss_jax(artifact, cfg, nlev=nlev, dt=dt)

    # NOT jax.jit'd: build_cbl_scm_from_artifact has concrete float() grid-setup that is
    # jit-incompatible with a traced config. The compile-reuse win is STRUCTURAL anyway —
    # the params enter as TRACED leaves (one jaxpr across all param values, XLA-cached),
    # whereas the DF path builds a NEW static-leaf config per candidate ⇒ a fresh compile
    # each time. So AD re-uses the compiled grad while DF recompiles per candidate.
    value_and_grad = jax.value_and_grad(loss_of)
    opt = optax.adam(lr)
    opt_state = opt.init(n0)
    norm = n0
    default_loss = float(loss_of(n0))
    best_loss = default_loss if np.isfinite(default_loss) else np.inf
    best_norm = n0
    history: list = [(default_loss, _to_py(n0))]
    for _ in range(max(1, steps)):
        loss, grad = value_and_grad(norm)
        if not (np.isfinite(float(loss)) and bool(jnp.all(jnp.isfinite(grad)))):
            break  # diverged → stop; the derivative-free path is the fallback
        updates, opt_state = opt.update(grad, opt_state)
        norm = jnp.clip(optax.apply_updates(norm, updates), 0.0, 1.0)
        cur = float(loss_of(norm))
        history.append((cur, _to_py(norm)))
        if np.isfinite(cur) and cur < best_loss:
            best_loss, best_norm = cur, norm
    # A diverged trajectory returns the large finite _AD_DIVERGE_PENALTY (not +inf), so a
    # finite best_loss is not enough — a best still in penalty territory means EVERY
    # evaluation diverged (no real fit). Threshold well above any real loss (O(1)).
    if not np.isfinite(best_loss) or best_loss >= _AD_DIVERGED_LOSS:
        raise RuntimeError(
            f"{base_turbulence.scheme}: AD tuning found no non-diverged loss "
            f"(best={best_loss:g}); try the derivative-free path or a smaller dt/lr.")
    return TuneResult(
        scheme=base_turbulence.scheme,
        best_overrides=_to_py(best_norm),
        best_loss=best_loss,
        default_loss=default_loss,
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
    p.add_argument("--method", choices=("df", "ad", "both"), default="df",
                   help="tuning method: df=derivative-free (default), ad=AD gradient "
                        "descent (D4; traced-leaf params reuse one compiled grad, vs the "
                        "DF per-candidate recompile), both=run+compare")
    p.add_argument("--n-random", type=int, default=12,
                   help="number of random candidates after the coordinate probe (df)")
    p.add_argument("--ad-steps", type=int, default=20, help="AD gradient steps")
    p.add_argument("--ad-lr", type=float, default=0.08, help="AD Adam learning rate")
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

    results: dict = {}
    if args.method in ("df", "both"):
        results["df"] = tune_closure_derivative_free(
            artifact, base, scheme_key, tiers=tuple(args.tiers),
            n_random=args.n_random, nlev=args.nlev, dt=args.dt, seed=args.seed)
    if args.method in ("ad", "both"):
        results["ad"] = tune_closure_ad(
            artifact, base, scheme_key, tiers=tuple(args.tiers), nlev=args.nlev,
            dt=args.dt, steps=args.ad_steps, lr=args.ad_lr, seed=args.seed)

    for tag, r in results.items():
        fin = bool(np.isfinite(r.default_loss))
        imp = (f" ({100 * (r.default_loss - r.best_loss) / r.default_loss:+.1f}%)"
               if fin and r.default_loss > 0 else "")
        print(f"[tune {tag}] case={artifact.case_name} scheme={r.scheme} "
              f"evaluated={r.n_evaluated}: default={r.default_loss:.4f} "
              f"best={r.best_loss:.4f}{imp}")
        rounded = {k: round(v, 4) for k, v in r.best_overrides.items()}
        print(f"  best overrides: {json.dumps(rounded)}")

    # D4 AD-vs-DF comparison (the tuning-method deliverable): do the two optima agree?
    comparison = None
    if args.method == "both":
        df, ad = results["df"], results["ad"]
        comparison = {
            "df_best_loss": df.best_loss, "ad_best_loss": ad.best_loss,
            "loss_gap": abs(df.best_loss - ad.best_loss),
            "better": "ad" if ad.best_loss < df.best_loss else "df",
        }
        print(f"  [D4 AD-vs-DF] df={df.best_loss:.4f} ad={ad.best_loss:.4f} "
              f"gap={comparison['loss_gap']:.4f} better={comparison['better']}")

    primary = results["ad" if args.method == "ad" else "df"]
    default_finite = bool(np.isfinite(primary.default_loss))
    method_name = {"df": "derivative_free", "ad": "ad", "both": "both"}[args.method]
    # Filename suffix by method so df/ad/both records don't overwrite one another; the
    # default 'df' keeps the campaign's existing ``..._df.json`` name unchanged.
    out = args.output or Path("results/les_suite/tuned") / \
        f"{args.artifact.stem}__{args.scheme}__{args.method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    q0 = (float(np.asarray(artifact.w_theta_s).reshape(-1)[0])
          if artifact.w_theta_s is not None else None)
    record = {
        "case": artifact.case_name,
        "artifact": args.artifact.stem,
        "sgs": artifact.sgs,
        "q0": q0,
        "scheme": args.scheme,
        "method": method_name,
        "tiers": list(args.tiers),
        "nlev": args.nlev,
        "dt": args.dt,
        "default_loss": primary.default_loss if default_finite else None,
        "best_loss": primary.best_loss,
        "best_overrides": primary.best_overrides,
        "n_evaluated": primary.n_evaluated,
    }
    if comparison is not None:
        record["comparison"] = comparison
        record["ad_best_overrides"] = results["ad"].best_overrides
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
