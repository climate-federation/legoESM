# Agent orientation — legoESM

This file orients any coding agent (Codex, Claude, or other) working in this
repo. `CLAUDE.md` at the repo root is the full house rulebook; everything there
binds you too. This file is the short version plus the map.

## What this repo is

A differentiable Earth-system model in JAX (atmosphere, ocean, land, sea ice,
coupler). End-to-end `jax.grad` compatibility is a hard requirement: never break
autodiff, JIT, or pytree purity. Mass conservation is hard; energy/momentum
where the scheme permits. All physical constants come from `legoesm.constants`
— never literal `273.15`/`9.80616`/`7.292e-05` (CI ratchets enforce this).

## The active campaign: DINO/NEMO ocean fidelity

Goal: make legoESM's DINO ocean statistically indistinguishable from NEMO 5.0.2.

**Read these, in order, before touching anything in this area:**

1. `docs/ocean/fidelity/dino_campaign_synthesis.md` — the whole campaign in one
   document: the result, the open problem, the ranked next-actions register,
   the instrument canon, the ledger. Every number in it is sourced.
2. `docs/ocean/fidelity/dino_outstanding_fidelity_debt.md` — the ranked queue.
3. GitHub issue **#1455** — the live evidence trail. Every finding, retraction,
   and verdict is posted there. Post yours there too.

**Current state (2026-08-27):** the circumpolar channel is statistically
indistinguishable from NEMO at one year as a band statistic — but the regional
audit showed every "indistinguishable" band verdict rests on 96–98 % per-row
cancellation, so per-row rescores are promoted work. The southern basin carries
a −0.95 Sv compounding deficit (fully mapped, majority owner unknown). The
equatorial undercurrent has a ~30 % shear error invisible to depth integrals.
The two models' horizontal/vertical geometry and constants are now identical to
machine precision.

## Hard-earned working rules (each one paid for by a measured failure)

- **Read the oracle's source before hypothesizing.** NEMO lives at
  `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2` (DINO config in `cfgs/DINO`,
  instrumented sources in `MY_SRC/`). Quote `file:line`. Verify the branch you
  quote actually RUNS under DINO's namelist/preprocessor state — dead-arm
  quotes have caused multiple retractions.
- **Pre-register before running.** Any measurement lane writes down, in a
  committed file, the number it will produce and the values that CONFIRM vs
  REFUTE — before the run. Post-hoc statistics must be labeled post-hoc.
- **The bar is exact and lives in gates, not judgment.**
  `scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py` classifies
  AT-BAR / DEBT / UNMEASURED. Never relax bar constants; never call a 0.99
  "matched" or "faithful".
- **The twin harness polices itself.** `kamm_twin_90d.py` refuses a dirty
  tracked tree, forces fp64 (env `FP64=0` is the loud escape), runs NEMO's
  seasonal clock (offset from the restart's own `adatrj`), bridges the
  before-level by default, and stamps clock/ladder/dtype/start-mode into every
  artifact. The acceptance gate (`acceptance_gate_90d.py`) refuses out-of-season
  or legacy-clock candidates and withholds verdicts off-claim. Do not work
  around the gates; they exist because each miss cost real GPU-days.
- **Controls must be able to fail.** Every check ships a planted violation that
  proves it fires. A control that compares a file against itself, perturbs a
  zero, or asserts `|x| ≤ 10|x|` has happened here and been caught; do not add
  another.
- **A retraction lives in the tool, not the commit message.** If a claim is
  withdrawn, the code that printed it must stop printing it (this failure has
  recurred four times).
- **Known instrument traps** (full canon in the synthesis): injection ≠
  accumulation; a tendency error ≠ a transport error; a bar belongs to a
  statistic, not a quantity; layer-averaging over-weights the surface ~54× —
  use thickness weighting; 10-day restart strides sample one leapfrog parity;
  band aggregates mask heterogeneous sub-bands; two-step projections annihilate
  fixed fields; the equator row is a structural zero for vorticity statistics;
  fp32 snapshot storage bounds what any spread can resolve.
- **Cancelling pairs:** four of five measured "fix one half and things get
  worse" instances were real. Before fixing any term that shares a coefficient
  or channel with another known error, run the pair analysis (see the vertex-
  area lane for the template — pairs can also be *refuted* by measurement).

## Mechanics

- Env: `pip install -e ".[dev]"`; tests `.venv/bin/python -m pytest tests/`;
  science tests need `JAX_ENABLE_X64=1` (note: that alone does NOT set the
  precision policy — oracle comparisons must go through the harness or
  `legoesm.core.precision.set_policy(PrecisionPolicy.fp64())`).
- Two GPUs; pin with `CUDA_VISIBLE_DEVICES`. Run test suites on CPU while long
  integrations hold the GPUs. The big unit suites can hit a per-process
  compiler limit — split them rather than raising timeouts.
- Committed probes live in `scripts/validate/ocean_fidelity/dino_1226/`; every
  number cited anywhere must come from a committed probe with provenance
  stamps. Throwaway probes are not citable.
- Stage with explicit pathspecs, never `git add .`/`-A`.
- Bare `git stash push` on a clean file matches nothing and its `pop` will grab
  an unrelated stash — this has bitten twice. Verify every revert with
  `git status --porcelain`.

## Coordination

- **One PR = one finished unit.** Maintainers merge fast, at whatever state the
  branch is in — twice a PR branch was merged early and stranded its tail.
  Mark PR bodies COMPLETE and never push more commits to an open PR's branch;
  new work starts a new branch.
- **Read the review comments on your PRs** (including agent-authored reviews)
  and disposition every finding explicitly — a HOLD that gets merged-by-
  subsumption without answers is a process failure that has happened here.
- Adversarial review before citing numbers or shipping physics: two independent
  reviewers, author never reviews own work. Reviewer disagreement is signal —
  name the discriminating measurement, never average.
- Multiple agents may be active. Do not work in a checkout another agent
  occupies (use `git worktree`); check `git worktree list` and #1455's recent
  comments before starting; post findings to #1455 so the ledger stays single.
