# Domain Architect vs Syntax Engine

*How legoESM uses AI without letting it silently overwrite domain expertise.*

## The problem

Generative AI is unmatched at producing **syntactically valid, compiling code**.
It does **not** optimize for factual truth or domain correctness. On a complex,
domain-specific problem an unguided model will confidently invent a rule, silently
drop an established principle, or pick a plausible-but-wrong convention — whatever
makes the code run inside its context window. In a differentiable Earth-system
model the result is an **invisible failure**: the output looks convincing, the
tests are green, the error norms even improve — yet a sign is flipped, a unit is
wrong, mass conservation has drifted, a saturation curve was re-derived and now
diverges from the model, or a misspelled `scheme=` silently selects a *different
physics scheme*. Nobody notices until a slow physical bias surfaces months later.

## The division of labour

We do not "trust the AI less." We **redefine who decides what**:

| Role | Who | Owns |
|------|-----|------|
| **Domain Architect** | the human / domain expert | the *logic*: units, sign conventions, conserved quantities, valid scheme sets, literature references, acceptance criteria |
| **Syntax Engine** | the AI | the *body*: filling in the implementation that satisfies the architect's declared contract |

The architect dictates the requirements; the AI is restricted to syntax
generation; and **every invariant the architect declared is checked
mechanically** so a violation FAILS LOUDLY instead of passing silently.

> **Honest framing — a tripwire, not a proof.** None of this makes code
> "perfect." The achievable goal is to turn every *silent* invariant-violation
> into a *loud, mechanical* failure, and to make every gate **provably
> non-vacuous** (it must reject a known-bad input — every gate ships a synthetic
> self-test that proves it goes red on a deliberate violation). This is
> defense-in-depth, not a correctness proof. Wrong equations, wrong-but-correctly-
> named constants, and values pulled from a config/JSON layer are *outside* what a
> textual ratchet can see — those are caught by the truth-tier tests
> (conservation / equivariance / analytic), not by the tripwires below.

## Three lines of defense

### 1. Specify before you synthesize (the architect's contract)

Every physics scheme module declares a machine-checked **`__physics_contract__`**
— a module-level dict literal pinning the units of every input/output, the sign
convention, what it conserves, whether it is differentiable, the literature
reference, and an idealized acceptance criterion. The contract and at least one
acceptance test are authored **before** the AI writes the body; reviewers and the
`physics-validator` agent then check the body against the *declared* contract
instead of re-inferring intent.

- Mechanism: `tests/test_physics_contracts.py` partitions **every**
  `*/physics/*.py` file into `EXCLUDED` (plumbing), `CONTRACT_TODO`
  (shrink-only), or annotated — so a new physics file cannot dodge: it must ship
  a contract or be explicitly classified.
- Example: `gravity_wave_drag/rayleigh.py`, `ocean/physics/bottom_drag/linear.py`.
- *This is the line that caught a real error during development:* an early
  Rayleigh-drag contract claimed `conserves: ["none"]`; the adversarial review
  noted the body returns frictional heating (`dT_dt = -(u·du+v·dv)/c_pd`), so
  **energy is conserved** (momentum is not). The contract system exists precisely
  to surface that kind of plausible-but-wrong claim.

### 2. Mechanical CI ratchets (prose rules → hard failures)

The repo's strongest guardrails are mechanical and **ratcheting** (they can only
get stricter — e.g. import-linter with `alerting="error"`, the inline-import
budgets). These extend that pattern; each is AST-based and ships a non-vacuity
self-test:

| Gate | Enforces |
|------|----------|
| `tests/test_no_hardcoded_constants.py` | no bare physical-constant literal (`273.15`, `9.80616`, …) outside `constants.py` — value-folded, fingerprinted per source line |
| `tests/test_no_saturation_reimpl.py` | no re-derived saturation curve (`611.2·exp(17.67·T_c/(T_c+243.5))`) outside `thermo.py` |
| `tests/test_no_formula_reimpl.py` | extensible registry — no canonical formula re-derived inline (buoyancy/Brunt-Väisälä `g/θ` debt-ratchet; Monin-Obukhov stability-fn guard); add a formula = one registry entry + a self-test |
| `tests/test_dispatch_hardening.py` | an existing `scheme=` factory's unknown-value `raise` is never silently deleted (76-entry grow-only baseline) |
| `tests/test_validate_strict_coverage.py` | every scheme-like config field is membership-validated in `validate_strict` (fail-early, not at JIT) |
| `tests/_ratchet_audit.py` | shared discovery/fold/exemption machinery for the above (no duplication) |

Escape valve for genuine non-physical literals: a real `# const-ok: <reason>` /
`# satcurve-ok: <reason>` comment on the offending line.

### 3. Real-time local hooks (block before it lands)

`.claude/hooks/` (wired in `.claude/settings.json`):

- **PreToolUse** `check_banned_literals.py` — blocks an `Edit/Write/MultiEdit/
  NotebookEdit` that introduces a banned constant / saturation prefactor into a
  `.py` file, *before* it lands. Fail-open; honors the `# const-ok` escape.
- **Stop** `require_review_artifact.py` — a once-per-session reminder to run the
  mandatory Codex adversarial review when an uncommitted numerics/physics change
  has no recent review patch. Loop-safe, fail-open, advisory.

These are a convenience layer; **CI (line 2) is authoritative** — a raw `Bash`
heredoc or a non-canonical spelling can slip past the hook but not past the AST
ratchets.

### (Visual) Cube-artifact regression

`scripts/validate/visual_regression.py` turns CLAUDE.md's mandatory *visual* cube
check into numbers (SSIM + perceptual hash + edge-artifact ratio on the W2
v-wind). The deterministic metric math is unit-tested (CI-blocking); the full
cube shallow-water `--check` runs as a **nightly, non-blocking** diagnostic until
its tolerances are calibrated across CI hardware.

## Best coding practices — one system, not a parallel one

The guardrails above sit on top of the existing engineering machinery; "good
practice" here means using these, not inventing new ones:

- **Lint / types:** `ruff` and `mypy` (CI `lint` / `type-check` jobs).
- **Architecture boundaries:** import-linter contracts (`alerting="error"`,
  ratchet toward zero) + the inline-import budgets.
- **Reuse over duplication:** pre-impl grep is mandatory; constants from
  `legoesm.constants`, saturation from `thermo`, EOS from `ocean.eos`, losses
  from `ml/loss.py`, column integrals from `diagnostics/` — never re-derived.
- **Tests with teeth:** every new `.py` gets a direct unit test; every *gate*
  ships a synthetic self-test proving it is non-vacuous; behavioural teeth
  (actually invoke the factory with a bogus value) outrank AST presence checks.
- **Hygiene sweeps:** the `slopbuster` agent (dead code, untested modules,
  duplicate impls, dead config branches).
- **Adversarial review:** the iterate-with-codex loop is mandatory for any
  substantial numerics/physics/parallel change (CLAUDE.md).

## How to extend the harness

- **Add a gate:** copy the `tests/_ratchet_audit.py` idiom — AST detector +
  shrink-only/grow-only pinned baseline + a synthetic-violation self-test. Never
  ship a gate that can pass on empty input (add a discovery-sanity test).
- **Annotate a physics module:** add `__physics_contract__` (see
  `test_physics_contracts.py` schema), then delete the path from `CONTRACT_TODO`.
- **Burn down debt:** the `*_TODO` / budget lists only shrink; lowering a count
  as you replace a literal or annotate a module is always welcome.
- **A new physics file** must be classified (contract, `CONTRACT_TODO`, or
  `EXCLUDED`) or the contract gate fails — by design.
