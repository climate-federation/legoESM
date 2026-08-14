#!/usr/bin/env python
"""Is every declared tunable parameter REACHABLE and DIFFERENTIABLE?

Two properties are audited together because one measurement settles both:

* **tuning reachability** — moving the parameter changes the scheme's output.
  A parameter that cannot change the output is search budget spent on nothing,
  and the tuner reports it as "applied" the whole time.
* **differentiability** — ``jax.grad`` of the output with respect to the
  parameter is finite and non-zero, i.e. a gradient-based trainer can move it.

The repo already gates two OTHER kinds of reachability and neither catches
this one: ``tests/test_param_specs.py`` checks that a parameter is DECLARED,
and ``tests/unit/test_params_reachability_audit.py`` checks that a calibration
file can SET it from a production driver.  Both pass for a parameter that no
numerical code ever reads.  In the 2026-08 SCM-RCE sub-cloud work, six of
fourteen curated parameters were exactly that.

WHAT IS MEASURED

For each scheme, the physics is called on several idealized columns and
reduced to a scalar by a FIXED random projection of the whole tendency pytree,

    L(theta) = sum_leaves sum_i w_i * leaf_i(theta)

with ``w`` drawn from a fixed seed.  A projection is used rather than a norm
because both ``sum|x|`` and ``sum x^2`` have zero derivative wherever the
tendency itself is zero, which is precisely the regime a dormant scheme sits
in — a norm would report a live parameter as dead.  Two independent
projections are taken and the larger gradient magnitude is kept, so a single
accidental cancellation cannot manufacture a DEAD verdict.

``jax.grad`` is taken through the PRODUCTION training path — raw values ->
``TrainablePhysicsParams.to_overrides`` -> ``apply_param_overrides`` INSIDE the
differentiated function, so the config leaves are traced exactly as they are
in ``scripts/run/train_scm_rce_params.py``.  One reverse pass yields the
gradient for EVERY parameter of that scheme at once, which is what makes a
whole-registry audit affordable.

THE FOUR OUTCOMES, and why they are not one outcome

``live``      |dL/dtheta| > 0 and finite.  Tunable and trainable.
``blocked``   gradient is exactly 0 but a finite-difference sweep DOES move the
              output: the parameter is read, and the derivative is severed —
              ``lax.stop_gradient``, an integer cast, or use only inside a
              comparison.  Tunable by a derivative-free search, NOT trainable
              by a gradient.  Different defect, different fix.
``dead``      gradient is 0 and the finite-difference sweep moves nothing: no
              path from the parameter to the output in these states.  Either
              the field is read by no code at all, or it sits behind a
              default-off flag.
``nondiff``   the gradient is non-finite, or tracing raised — typically a
              Python ``if`` on the traced leaf inside a factory.  A real bug
              for any gradient-based trainer.

WHAT A ``dead`` VERDICT DOES AND DOES NOT MEAN.  Inherited honestly from the
convection probe this generalizes, and every one of these limits is still
real:

* the columns carry NO horizontal grid, so a parameter whose only consumer
  needs a grid-derived input (Kuo's moisture convergence) reads dead without
  having been tested;
* ``phys_state`` is a COLD START unless ``--spinup-steps`` is given, so a cap
  on a mass flux that has not spun up reads dead;
* the verdict is over the union of the sampled columns only.

So ``dead`` means "no path to the output on these states, at this spin-up" —
which is exactly the condition a tuner would also fail to exploit, and is
therefore the right gate for a tuning campaign.  It is NOT proof the field is
unused in a full model run, and the report says so next to every row.

Usage::

    python scripts/validate/audit_param_gradients.py --categories all
    python scripts/validate/audit_param_gradients.py \\
        --categories turbulence,convection --json-out audit.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics import (  # noqa: E402
    ConvectionConfig,
    GravityWaveDragConfig,
    MicrophysicsConfig,
    PhysicsConfig,
    RadiationConfig,
    TurbulenceConfig,
    make_physics,
)
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.param_overrides import apply_param_overrides  # noqa: E402
from legoesm.core.state import HydrostaticState  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402
from legoesm.training.param_collector import (  # noqa: E402
    build_registry,
    build_trainable_params,
)
from legoesm.training.trainable_params import TrainablePhysicsParams  # noqa: E402

#: Physics categories this harness can activate one at a time.  A category
#: absent here is NOT audited, and the report says so rather than implying
#: coverage it does not have.
ATM_CATEGORIES = ("convection", "turbulence", "microphysics", "radiation",
                  "gravity_wave_drag")

#: Config field name on ``PhysicsConfig`` for each category, and the wrapper
#: NamedTuple that carries ``scheme=``.
_CATEGORY_WRAPPER = {
    "convection": ConvectionConfig,
    "turbulence": TurbulenceConfig,
    "microphysics": MicrophysicsConfig,
    "radiation": RadiationConfig,
    "gravity_wave_drag": GravityWaveDragConfig,
}

P_SFC = 101_480.0

#: Finite-difference sweep, only ever run for a parameter whose gradient came
#: back exactly zero — it is what separates ``dead`` from ``blocked``.  Same
#: sampling as the tuner's own (fractions of the declared range) PLUS
#: multiples of the default, because a threshold's realistic value often sits
#: outside its declared bounds window.
RANGE_FRACTIONS = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
DEFAULT_MULTIPLES = (0.1, 0.5, 2.0, 10.0)

#: Fixed seed for the projection weights.  Reproducibility is the point: a
#: verdict that changed between runs would be unciteable.
PROJECTION_SEED = 20260814
N_PROJECTIONS = 2


@dataclass(frozen=True)
class ParamVerdict:
    category: str
    scheme: str
    scheme_key: str
    field: str
    qualified_name: str
    tunable_tier: int
    default: float
    lower: float
    upper: float
    #: max |dL/dtheta| over projections and states.
    grad_abs: float
    #: max |dL| over the finite-difference sweep; NaN when not run (gradient
    #: was already non-zero, so the distinction was not needed).
    fd_abs: float
    verdict: str
    note: str = ""


def _column(nlev: int, *, T_sfc: float, rh: float, lapse_K_km: float,
            name: str):
    """One idealized tropical column: linear lapse to a 200 K cap, uniform RH.

    Built with the SHARED sigma factory, never positionally: ``SigmaCoordinate``
    carries derived fields the physics reads.
    """
    from legoesm.thermo import saturation_mixing_ratio

    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full, dtype=float)
    p_full = sigma_full * P_SFC
    z = -(constants.R_d * 290.0 / constants.g) * np.log(
        np.maximum(sigma_full, 1e-6))
    T = np.maximum(T_sfc - lapse_K_km * 1e-3 * z, 200.0)
    q_sat = np.asarray(saturation_mixing_ratio(jnp.asarray(T),
                                               jnp.asarray(p_full)))
    q_v = np.maximum(rh * q_sat, 1e-11)
    dims4 = ("face", "x", "y", "level")
    dims3 = ("face", "x", "y")
    shape4 = (1, 1, 1, nlev)
    zeros4 = jnp.zeros(shape4)
    # Every condensate slot the ice-capable schemes expect is allocated, so a
    # microphysics parameter that only touches snow or graupel is not reported
    # dead for want of a tracer.
    tracers = {
        "q_v": Field(data=jnp.asarray(q_v).reshape(shape4), name="q_v",
                     dims=dims4),
    }
    for extra, seed_value in (("q_c", 1.0e-5), ("q_r", 1.0e-6),
                              ("q_i", 1.0e-6), ("q_s", 1.0e-6),
                              ("q_g", 1.0e-7)):
        # Seeded NON-ZERO: a process rate proportional to a species that is
        # exactly zero has an exactly zero derivative, which would be reported
        # as a dead parameter when it is only an empty column.
        tracers[extra] = Field(data=zeros4 + seed_value, name=extra,
                               dims=dims4)
    for number in ("N_c", "N_r", "N_i"):
        tracers[number] = Field(data=zeros4 + 1.0e6, name=number, dims=dims4)
    state = HydrostaticState(
        u=Field(data=jnp.full(shape4, 5.0), name="u", dims=dims4),
        v=Field(data=jnp.full(shape4, 1.0), name="v", dims=dims4),
        T=Field(data=jnp.asarray(T).reshape(shape4), name="T", dims=dims4),
        p_s=Field(data=jnp.full((1, 1, 1), P_SFC), name="p_s", dims=dims3),
        phis=Field(data=jnp.zeros((1, 1, 1)), name="phis", dims=dims3),
        tracers=tracers,
    )
    return name, state, sigma


def build_states(nlev: int):
    """Columns spanning the branches a scheme can take.

    A parameter dead in ALL of these is dead for a tuner's purposes; dead in
    one is merely untested there.  A single column cannot support a dead
    verdict — schemes are branch forests behind trigger predicates.
    """
    return [
        _column(nlev, T_sfc=300.0, rh=0.95, lapse_K_km=6.5,
                name="rce_saturated"),
        _column(nlev, T_sfc=300.0, rh=0.80, lapse_K_km=6.5,
                name="rce_subsaturated"),
        _column(nlev, T_sfc=302.0, rh=0.85, lapse_K_km=8.0,
                name="deep_unstable"),
        _column(nlev, T_sfc=298.0, rh=0.60, lapse_K_km=5.0,
                name="suppressed_stable"),
        _column(nlev, T_sfc=300.0, rh=0.70, lapse_K_km=7.0,
                name="trade_shallow"),
    ]


def _single_scheme_config(category: str, scheme: str, subcfg=None) -> PhysicsConfig:
    """A PhysicsConfig with exactly ONE category active.

    Isolation is the point: with two schemes live, a gradient could arrive
    through the other one and a dead parameter would read live.
    """
    if category not in _CATEGORY_WRAPPER:
        raise ValueError(
            f"_single_scheme_config: unknown category {category!r}; "
            f"expected one of {sorted(_CATEGORY_WRAPPER)}")
    kwargs = {}
    for cat, wrapper in _CATEGORY_WRAPPER.items():
        if cat == category:
            fields = {"scheme": scheme}
            if subcfg is not None:
                fields[scheme] = subcfg
            kwargs[cat] = wrapper(**fields)
        else:
            kwargs[cat] = wrapper(scheme="none")
    return PhysicsConfig(**kwargs)


def _subconfig_of(cfg: PhysicsConfig, category: str):
    component = getattr(cfg, category)
    return getattr(component, component.scheme, None)


def _tunable_object(subcfg):
    """The object whose DIRECT fields carry the spec'd params.

    Flat for every scheme except full CLUBB, which nests them in ``.params``.
    Mirrors the campaign's own accessor rather than guessing.
    """
    if subcfg is None:
        return None
    try:
        from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    except ImportError:  # pragma: no cover - component not installed
        return subcfg
    return subcfg.params if isinstance(subcfg, CLUBBConfig) else subcfg


def _rewrap(subcfg, tuned):
    try:
        from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig
    except ImportError:  # pragma: no cover
        return tuned
    return subcfg._replace(params=tuned) if isinstance(subcfg, CLUBBConfig) else tuned


def scheme_key_map(categories) -> dict[str, tuple[str, str]]:
    """``{scheme_key: (category, scheme_name)}`` for the auditable schemes.

    Built by ACTIVATING each scheme and reading back the resulting sub-config's
    registry key, so the map cannot drift from the factory.  A scheme whose
    sub-config carries no spec'd parameters simply does not appear.
    """
    by_class = {m.config_class: m.scheme_key for m in build_registry()}
    out: dict[str, tuple[str, str]] = {}
    from scripts.run.run_scm_rce_campaign import SCHEME_SWEEPS

    for category in categories:
        for scheme in SCHEME_SWEEPS[category]:
            if scheme == "none":
                continue
            try:
                cfg = _single_scheme_config(category, scheme)
                sub = _subconfig_of(cfg, category)
            except (TypeError, ValueError):
                continue
            tunable = _tunable_object(sub)
            if tunable is None:
                continue
            key = by_class.get(type(tunable).__name__)
            if key is not None:
                out[key] = (category, scheme)
    return out


def _projection_weights(tree, key):
    """Fixed pseudo-random weights shaped like ``tree``.

    A norm cannot be used: ``sum|x|`` and ``sum x**2`` both have zero
    derivative wherever the tendency is zero, which is exactly where a dormant
    scheme sits, so a norm would call a live parameter dead.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, max(1, len(leaves)))
    weights = [jax.random.normal(k, jnp.shape(leaf), dtype=jnp.float64)
               for k, leaf in zip(keys, leaves)]
    return jax.tree_util.tree_unflatten(treedef, weights)


def _numeric_leaves(tree):
    """Inexact leaves only — an int/bool leaf is not differentiable and would
    make the projection dtype-invalid."""
    return [leaf for leaf in jax.tree_util.tree_leaves(tree)
            if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)]


def scheme_gradients(
    category: str,
    scheme: str,
    scheme_key: str,
    *,
    nlev: int,
    dt: float,
    states,
) -> tuple[dict[str, float], str]:
    """max |dL/dtheta| per parameter for one scheme, over states+projections.

    Returns ``({qualified_name: grad_abs}, note)``.  ``note`` is non-empty when
    tracing failed, which is itself a differentiability verdict.
    """
    base_cfg = _single_scheme_config(category, scheme)
    base_sub = _subconfig_of(base_cfg, category)
    params = build_trainable_params(
        active_scheme_keys={scheme_key}, tier="aggressive", dtype=jnp.float64)
    constraints = [c for c in params.constraints
                   if c.name.rsplit(".", 1)[0] == scheme_key]
    if not constraints:
        return {}, "no spec'd parameters"
    raw0 = {c.name: params.raw_values[c.name] for c in constraints}

    def loss_for(state, sigma, weights_holder):
        def loss(raw):
            trial = TrainablePhysicsParams(raw_values=raw,
                                           constraints=constraints)
            field_values = trial.to_overrides().get(scheme_key, {})
            tuned = apply_param_overrides(_tunable_object(base_sub),
                                          field_values)
            cfg = _single_scheme_config(category, scheme,
                                        _rewrap(base_sub, tuned))
            fn = make_physics(cfg, model_type="hydrostatic", dt=dt)
            tend, carry = fn(state, None, sigma)
            leaves = _numeric_leaves((tend, carry))
            if not leaves:
                return jnp.asarray(0.0, dtype=jnp.float64)
            if weights_holder["w"] is None:
                weights_holder["w"] = _projection_weights(
                    leaves, weights_holder["key"])
            return sum(jnp.sum(w * jnp.asarray(leaf, dtype=jnp.float64))
                       for w, leaf in zip(weights_holder["w"], leaves))
        return loss

    best = {c.name: 0.0 for c in constraints}
    for p in range(N_PROJECTIONS):
        key = jax.random.PRNGKey(PROJECTION_SEED + p)
        for _name, state, sigma in states:
            holder = {"w": None, "key": key}
            try:
                grads = jax.grad(loss_for(state, sigma, holder))(raw0)
            except Exception as exc:  # noqa: BLE001 - the failure IS a verdict
                return ({c.name: float("nan") for c in constraints},
                        f"{type(exc).__name__}: {exc}"[:200])
            for c in constraints:
                g = np.asarray(grads[c.name], dtype=float)
                value = float(np.max(np.abs(g))) if g.size else 0.0
                if not np.isfinite(value):
                    best[c.name] = float("nan")
                elif not np.isnan(best[c.name]):
                    best[c.name] = max(best[c.name], value)
    return best, ""


def finite_difference_response(
    category: str,
    scheme: str,
    constraint,
    default: float,
    *,
    dt: float,
    states,
) -> float:
    """max |change in the tendency| over a sweep of ONE parameter.

    Only called for a parameter whose gradient was exactly zero — it is what
    separates "read but the derivative is severed" from "not read at all".
    """
    base_cfg = _single_scheme_config(category, scheme)
    base_sub = _subconfig_of(base_cfg, category)
    lo, hi = float(constraint.min_val), float(constraint.max_val)
    trials = [lo + f * (hi - lo) for f in RANGE_FRACTIONS]
    trials += [min(max(default * k, lo), hi) for k in DEFAULT_MULTIPLES]

    def tendency(value):
        tuned = apply_param_overrides(_tunable_object(base_sub),
                                      {constraint.field: value})
        cfg = _single_scheme_config(category, scheme, _rewrap(base_sub, tuned))
        fn = make_physics(cfg, model_type="hydrostatic", dt=dt)
        out = []
        for _name, state, sigma in states:
            tend, carry = fn(state, None, sigma)
            out.append([np.asarray(leaf, dtype=float)
                        for leaf in _numeric_leaves((tend, carry))])
        return out

    base = tendency(default)
    worst = 0.0
    for value in trials:
        if value == default:
            continue
        trial = tendency(value)
        for b_state, t_state in zip(base, trial):
            for b, t in zip(b_state, t_state):
                if b.shape != t.shape:
                    return float("inf")
                d = np.abs(t - b)
                # NaN must never read as "no response": max(0.0, nan) keeps
                # 0.0, so a scheme that blew up on a trial value would be
                # reported DEAD.
                if not np.all(np.isfinite(d)):
                    raise FloatingPointError(
                        f"{constraint.field}={value!r} produced a non-finite "
                        "tendency difference; a DEAD verdict here would be an "
                        "artifact of NaN handling")
                worst = max(worst, float(np.max(d)) if d.size else 0.0)
    return worst


def audit(
    categories,
    *,
    nlev: int = 30,
    dt: float = 600.0,
    run_finite_difference: bool = True,
) -> list[ParamVerdict]:
    states = build_states(nlev)
    key_map = scheme_key_map(categories)
    meta_by_name = {m.qualified_name: m for m in build_registry()}
    verdicts: list[ParamVerdict] = []
    for scheme_key, (category, scheme) in sorted(key_map.items()):
        grads, note = scheme_gradients(
            category, scheme, scheme_key, nlev=nlev, dt=dt, states=states)
        if not grads:
            continue
        params = build_trainable_params(
            active_scheme_keys={scheme_key}, tier="aggressive",
            dtype=jnp.float64)
        constraints = {c.name: c for c in params.constraints
                       if c.name.rsplit(".", 1)[0] == scheme_key}
        defaults = {n: float(params.as_dict()[n]) for n in constraints}
        print(f"[{category}/{scheme}] {len(constraints)} parameters"
              + (f" — TRACE FAILED: {note}" if note else ""), flush=True)
        for name, c in sorted(constraints.items()):
            meta = meta_by_name[name]
            g = grads.get(name, 0.0)
            fd = float("nan")
            if note:
                verdict = "nondiff"
            elif np.isnan(g):
                verdict = "nondiff"
            elif g > 0.0:
                verdict = "live"
            else:
                if run_finite_difference:
                    fd = finite_difference_response(
                        category, scheme, c, defaults[name], dt=dt,
                        states=states)
                    verdict = "blocked" if fd > 0.0 else "dead"
                else:
                    verdict = "zero_grad_unclassified"
            verdicts.append(ParamVerdict(
                category=category, scheme=scheme, scheme_key=scheme_key,
                field=c.field, qualified_name=name,
                tunable_tier=int(meta.tunable_tier),
                default=defaults[name], lower=float(c.min_val),
                upper=float(c.max_val), grad_abs=float(g), fd_abs=fd,
                verdict=verdict, note=note))
    return verdicts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--categories", default="all",
                   help=f"'all' or comma-separated from {ATM_CATEGORIES}")
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--no-finite-difference", action="store_true",
                   help="skip the dead-vs-blocked discrimination (faster, "
                        "leaves zero-gradient parameters unclassified)")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args(argv)

    if args.categories == "all":
        categories = ATM_CATEGORIES
    else:
        categories = tuple(c.strip() for c in args.categories.split(",")
                           if c.strip())
        unknown = [c for c in categories if c not in ATM_CATEGORIES]
        if unknown:
            raise SystemExit(
                f"unknown categories {unknown}; expected from {ATM_CATEGORIES}")

    verdicts = audit(categories, nlev=args.nlev, dt=args.dt,
                     run_finite_difference=not args.no_finite_difference)
    if not verdicts:
        raise SystemExit("no parameters audited — the registry or the scheme "
                         "map resolved empty, which is itself a defect")

    order = {"nondiff": 0, "dead": 1, "blocked": 2,
             "zero_grad_unclassified": 3, "live": 4}
    print()
    print(f"{'parameter':58s} {'tier':>4s} {'|dL/dp|':>11s} {'fd':>11s}  verdict")
    print("-" * 100)
    for v in sorted(verdicts, key=lambda v: (order[v.verdict], v.qualified_name)):
        if v.verdict == "live":
            continue
        print(f"{v.qualified_name:58s} {v.tunable_tier:4d} "
              f"{v.grad_abs:11.3e} {v.fd_abs:11.3e}  {v.verdict}"
              + (f"  [{v.note}]" if v.note else ""))
    counts = {k: sum(1 for v in verdicts if v.verdict == k) for k in order}
    print()
    print("SUMMARY  " + "  ".join(f"{k}={counts[k]}" for k in order)
          + f"  total={len(verdicts)}")
    print("Audited categories: " + ", ".join(categories))
    print("NOT audited by this harness (no driver yet): land, ocean, ice, "
          "coupler, and the atmospheric clouds/aerosol configs that are not "
          "reached through make_physics on a bare column.")
    print("A `dead` verdict means: no path to the output on these five "
          "columns at cold start. It is the condition a tuner would also fail "
          "to exploit; it is not proof the field is unused in a full run.")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"categories": list(categories), "nlev": args.nlev, "dt": args.dt,
             "verdicts": [asdict(v) for v in verdicts]}, indent=2))
        print(f"wrote {args.json_out}")
    # A non-differentiable parameter is a bug for every gradient trainer, so it
    # fails the run; dead/blocked are ratcheted by the test, not here.
    return 1 if counts["nondiff"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
