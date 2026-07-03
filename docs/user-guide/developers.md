# Developers

How legoESM is organized and the invariants every change must preserve. The full
contributor guide is [`CONTRIBUTING.md`](https://github.com/climate-federation/legoESM/blob/main/CONTRIBUTING.md)
(the same working rules live in `CLAUDE.md` for AI-assisted work).

## Setup

```bash
git clone https://github.com/climate-federation/legoESM
cd legoESM
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
.venv/bin/python -m pytest tests/
```

Scientific tests must enable 64-bit floats (spectral cores + conservation checks
rely on it); on Apple Silicon run on CPU:

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest <target>
JAX_PLATFORMS=cpu .venv/bin/python -m pytest <target>   # Apple Silicon
```

## The non-negotiables

Every change must preserve these invariants (see `CONTRIBUTING.md` for the long form):

1. **Differentiability.** End-to-end `jax.grad` through the coupled model is the
   defining feature — never break autodiff, JIT, or pytree structure. Use
   `jnp.where`/`lax.cond`/`lax.scan`/`fori_loop` on traced values, not Python
   control flow. Implicit solves are differentiated via the implicit-function
   theorem, never by backprop through an unbounded loop.
2. **Conservation.** Mass is a hard constraint; energy and momentum are conserved
   wherever the scheme permits. Flux-form coupling conserves by construction. New
   exchanged quantities get a global-budget test.
3. **A direct unit test per new `.py`, same PR.** For a physics scheme, test the
   leaf tendency function — not just an integration smoke test through a factory.
4. **Shared utilities, never re-derived.** Constants live in `legoesm.constants`,
   saturation thermodynamics in `legoesm.thermo`, losses in `ml/loss.py`, etc.
   Re-deriving them is how silent divergences creep in.

## Layered architecture

Each Earth-system component depends only on `core` (plus truly-shared bricks),
never on another component or on the coupler/driver. That one-way dependency is
exactly what lets `pip install legoesm-ocean` run standalone:

```
core (substrate)  <  {atmosphere, ocean, land, ice}  <  coupler  <  driver
```

The import-boundary contracts in `pyproject.toml` enforce this.

## Repository layout

| Path | Contents |
|------|----------|
| `packages/<pkg>/legoesm/` | Source, federation namespace — `atmosphere`, `core`, `coupler`, `ice`, `land`, `ml`, `ocean`, `tools`. |
| `tests/` | Mirrors the package tree (`tests/<component>/<tier>/…`); curated dycore regressions in `tests/atmosphere/dycore/regression/`. |
| `scripts/` | Bucketed: `run/`, `matrix/`, `bench/`, `plot/`, `validate/`, `data/`, `experiment/`, `cluster/`; throwaway probes in `scripts/tmp/`. |
| `config/` | YAML experiment configs + templates. |
| `docs/` | This site (curated pages) plus [`docs/dev-notes/`](../dev-notes/README.md) for internal notes/logs/audits. |

## Testing & validation

- **Tiered strategy** — research → operational rungs with conservation gates:
  see [Testing strategy](../validation/TESTING.md).
- **Source-guardrail harness** — fast static tripwires (ratchet audits for
  constants/saturation, dispatch hardening, spec-first physics contracts,
  federation boundaries) plus LIVE editor hooks; the cheapest "verify the existing
  code" pass. See [TESTING.md §5](../validation/TESTING.md) and
  [`ai_guardrails/domain_architect_vs_syntax_engine.md`](../architecture/ai_guardrails/domain_architect_vs_syntax_engine.md).
  Run: `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu pytest tests/test_no_hardcoded_constants.py tests/test_no_saturation_reimpl.py tests/test_dispatch_hardening.py tests/test_physics_contracts.py tests/test_import_boundaries.py tests/test_federation_plan.py -q`
- **Adversarial-review agents** — Codex (`/codex:adversarial-review --wait` → fix →
  `/codex:review --wait`, iterate to clean) and specialized subagents
  (`physics-validator`, `lego-modularity-tester`, …). See [TESTING.md §6](../validation/TESTING.md).
- **Dycore benchmarks** — Williamson, Galewsky, Jablonowski-Williamson, DCMIP,
  Held-Suarez: see [Dycore validation catalog](../validation/dycore_validation_catalog.md).
- **Physics parameterizations** — [Physics parameterization tests](../validation/PHYSICS_PARAMETERIZATION_TESTS.md).
- **Distributed/MPI invariants** — [Distributed architecture](../architecture/DISTRIBUTED_ARCHITECTURE.md).
- **Performance** — [Real-hardware scaling](../performance/REAL_HARDWARE_SCALING.md).

**Visual verification is mandatory for spatial/grid artifacts.** Passing norms are
necessary but not sufficient for cubed-sphere ops, halo exchange, and diffusion —
inspect the W2 v-wind / W5 wind-speed PNGs against baselines.

**Gradient work requires the fp64 compute policy.** `JAX_ENABLE_X64=1` does NOT
make the finite-volume models compute in double precision — they cast state to
their compute dtype (float32 by default) via `legoesm.core.precision`. Any
AD-vs-FD check, adjoint study, or calibration script that skips

```python
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy.fp64())
```

silently measures float32 noise floors (AD-vs-FD plateaus near 1e-2,
reverse-vs-forward near 1e-7) instead of gradient correctness — the 2026-06-11
ocean differentiability audit initially mismeasured an entire probe matrix this
way. Production *training* may still run float32 deliberately; the float32
gradient-NaN hazards found by that audit (limiter/FCT eps-ratio underflow on
near-uniform tracers) are guarded in-code with a dedicated underflow pin test
(PR #415, `grad_safe_ratio`). The step-level gradient
health of the lat-lon ocean config space is gated by
`tests/ocean/validation/test_step_gradient_matrix.py` (fp64,
reverse-vs-forward agreement — the kink-immune transpose check; finite-
difference agreement saturates on C⁰ limiter landscapes and is only a
secondary gate there).

## Internal development notes

Working notes, plans, audits, and review trackers live in
[`docs/dev-notes/`](../dev-notes/README.md). They are referenced from code by **basename**
(e.g. `see fv3_faithful.md`) so comments survive the move out of the published nav.
